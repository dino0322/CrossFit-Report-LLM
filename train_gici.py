import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# =========================
# ℹ️ 딥러닝 가속기 설정
# =========================

def get_device(prefer='auto', verbose=True):
    if prefer == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif prefer == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif prefer == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    if verbose:
        print(f"[Current Device]: {device}")

    return device

DEVICE = get_device()


class TextDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer, max_length: int = 512):
        self.samples: List[str] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                self.samples.append(text)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


class MeanPoolingHead(nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


@dataclass
class TrainConfig:
    korean_dir: str
    english_dir: str
    output_dir: str = "./saved_models_gici"
    korean_model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    english_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    iterations: int = 10
    epochs_per_iteration: int = 1
    batch_size: int = 8
    lr: float = 1e-5
    max_length: int = 512
    device: str = "auto"
    seed: int = 42


class GICITrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device, verbose=True)
        self.pooler = MeanPoolingHead().to(self.device)

        self.output_dir = Path(cfg.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.plot_dir = self.output_dir / "plots"
        self.log_dir = self.output_dir / "logs"
        for d in [self.output_dir, self.ckpt_dir, self.plot_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.kor_files = sorted(str(p) for p in Path(cfg.korean_dir).glob("*.txt"))
        self.eng_files = sorted(str(p) for p in Path(cfg.english_dir).glob("*.txt"))

        if not self.kor_files:
            raise FileNotFoundError(f"No Korean txt files found in {cfg.korean_dir}")
        if not self.eng_files:
            raise FileNotFoundError(f"No English txt files found in {cfg.english_dir}")

        self.kor_tokenizer = AutoTokenizer.from_pretrained(cfg.korean_model_name)
        self.eng_tokenizer = AutoTokenizer.from_pretrained(cfg.english_model_name)

        self.mkor = AutoModel.from_pretrained(cfg.korean_model_name).to(self.device)
        self.meng = AutoModel.from_pretrained(cfg.english_model_name).to(self.device)
        self.mmerged = AutoModel.from_pretrained(cfg.korean_model_name).to(self.device)

        self.history = {
            "kor_train_loss": [],
            "eng_train_loss": [],
            "merge_step": [],
        }

    def split_into_chunks(self, files: List[str]) -> List[List[str]]:
        chunks = np.array_split(np.array(files, dtype=object), self.cfg.iterations)
        return [list(chunk.tolist()) for chunk in chunks if len(chunk) > 0]

    def build_loader(self, file_paths: List[str], tokenizer) -> DataLoader:
        dataset = TextDataset(file_paths, tokenizer, max_length=self.cfg.max_length)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after loading text files.")
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

    def unsupervised_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        # 논문 코드 계열에서 사용하던 단순 embedding regularization 형태를 유지
        return torch.mean(embeddings ** 2)

    def train_one_model(self, model: AutoModel, loader: DataLoader, optimizer: AdamW) -> float:
        model.train()
        total_loss = 0.0
        steps = 0

        for _ in range(self.cfg.epochs_per_iteration):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = self.pooler(outputs.last_hidden_state, attention_mask)
                loss = self.unsupervised_loss(embeddings)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                steps += 1

        return total_loss / max(steps, 1)

    def average_models(self, model_a: AutoModel, model_b: AutoModel) -> AutoModel:
        state_a = model_a.state_dict()
        state_b = model_b.state_dict()
        merged = {}

        for key in state_a.keys():
            if key not in state_b:
                raise KeyError(f"Missing key in second model state_dict: {key}")
            merged[key] = 0.5 * state_a[key] + 0.5 * state_b[key]

        self.mmerged.load_state_dict(merged)
        return self.mmerged

    def synchronize_models_from_merged(self):
        merged_state = self.mmerged.state_dict()
        self.mkor.load_state_dict(copy.deepcopy(merged_state), strict=False)
        self.meng.load_state_dict(copy.deepcopy(merged_state), strict=False)

    def save_hf_model(self, model: AutoModel, tokenizer, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    def save_state_dict(self, model: AutoModel, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)

    def save_history(self):
        history_path = self.log_dir / "train_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        plt.figure(figsize=(10, 6))
        plt.plot(self.history["merge_step"], self.history["kor_train_loss"], label="Korean model loss")
        plt.plot(self.history["merge_step"], self.history["eng_train_loss"], label="English model loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("GICI Training Loss by Iteration")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plot_dir / "gici_training_loss.png")
        plt.close()

    def train(self):
        kor_chunks = self.split_into_chunks(self.kor_files)
        eng_chunks = self.split_into_chunks(self.eng_files)
        n_steps = min(len(kor_chunks), len(eng_chunks), self.cfg.iterations)

        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Korean files: {len(self.kor_files)}")
        print(f"[INFO] English files: {len(self.eng_files)}")
        print(f"[INFO] Iterations: {n_steps}")

        for iteration in range(n_steps):
            print(f"\n=== GICI Iteration {iteration + 1}/{n_steps} ===")
            kor_loader = self.build_loader(kor_chunks[iteration], self.kor_tokenizer)
            eng_loader = self.build_loader(eng_chunks[iteration], self.eng_tokenizer)

            kor_optimizer = AdamW(self.mkor.parameters(), lr=self.cfg.lr)
            eng_optimizer = AdamW(self.meng.parameters(), lr=self.cfg.lr)

            kor_loss = self.train_one_model(self.mkor, kor_loader, kor_optimizer)
            eng_loss = self.train_one_model(self.meng, eng_loader, eng_optimizer)

            print(f"[KOR] train_loss={kor_loss:.6f}")
            print(f"[ENG] train_loss={eng_loss:.6f}")

            self.average_models(self.mkor, self.meng)
            self.synchronize_models_from_merged()

            self.history["merge_step"].append(iteration + 1)
            self.history["kor_train_loss"].append(kor_loss)
            self.history["eng_train_loss"].append(eng_loss)

            self.save_state_dict(self.mkor, self.ckpt_dir / f"iter_{iteration + 1}_mkor.pt")
            self.save_state_dict(self.meng, self.ckpt_dir / f"iter_{iteration + 1}_meng.pt")
            self.save_state_dict(self.mmerged, self.ckpt_dir / f"iter_{iteration + 1}_mmerged.pt")

        # 최종 저장: HF 형식 + pt 형식 둘 다 저장
        self.save_hf_model(self.mkor, self.kor_tokenizer, self.output_dir / "final_mkor")
        self.save_hf_model(self.meng, self.eng_tokenizer, self.output_dir / "final_meng")
        self.save_hf_model(self.mmerged, self.kor_tokenizer, self.output_dir / "final_mmerged")

        self.save_state_dict(self.mkor, self.output_dir / "final_mkor.pt")
        self.save_state_dict(self.meng, self.output_dir / "final_meng.pt")
        self.save_state_dict(self.mmerged, self.output_dir / "final_mmerged.pt")

        self.save_history()
        print("\n[INFO] GICI training complete.")
        print(f"[INFO] Saved to: {self.output_dir}")


def parse_args() -> TrainConfig:
    config_dict = {
        "korean_dir": "./data/korean",             
        "english_dir": "./data/english",           
        "output_dir": "./saved_models_gici",       
        "korean_model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "english_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "iterations": 6,                           
        "epochs_per_iteration": 1,
        "batch_size": 2,                           
        "lr": 1e-5,                                
        "max_length": 512,
        "device": "auto",                          
        "seed": 42
    }
    #parser = argparse.ArgumentParser(description="Train GICI exactly in the paper-style iterative manner.")
    # parser.add_argument("--korean_dir", type=str, required=True)
    # parser.add_argument("--english_dir", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, default="./saved_models_gici")
    # parser.add_argument("--korean_model_name", type=str, default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    # parser.add_argument("--english_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    # parser.add_argument("--iterations", type=int, default=10)
    # parser.add_argument("--epochs_per_iteration", type=int, default=1)
    # parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--lr", type=float, default=1e-5)
    # parser.add_argument("--max_length", type=int, default=512)
    # parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    #parser.add_argument("--seed", type=int, default=42)
    #args = parser.parse_args()

    #torch.manual_seed(config_dict.seed)
    #np.random.seed(config_dict.seed)

    return TrainConfig(**config_dict)


if __name__ == "__main__":
    config = parse_args()
    trainer = GICITrainer(config)
    trainer.train()
