from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class MeanPooling:
    def __call__(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


@dataclass
class GICIRetrieverConfig:
    mkor_dir: str
    meng_dir: str
    max_length: int = 512


class GICIRetriever:
    def __init__(self, cfg: GICIRetrieverConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.pooler = MeanPooling()

        self.kor_tokenizer = AutoTokenizer.from_pretrained(cfg.mkor_dir)
        self.eng_tokenizer = AutoTokenizer.from_pretrained(cfg.meng_dir)
        self.mkor = AutoModel.from_pretrained(cfg.mkor_dir).to(device).eval()
        self.meng = AutoModel.from_pretrained(cfg.meng_dir).to(device).eval()

    def encode_single(self, text: str, model: AutoModel, tokenizer) -> np.ndarray:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.cfg.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = self.pooler(outputs.last_hidden_state, inputs["attention_mask"])
        return embedding.squeeze(0).detach().cpu().numpy()

    def encode_gici(self, text: str) -> np.ndarray:
        emb_kor = self.encode_single(text, self.mkor, self.kor_tokenizer)
        emb_eng = self.encode_single(text, self.meng, self.eng_tokenizer)
        return (emb_kor + emb_eng) / 2.0

    def build_page_embeddings(self, pages: List[Dict]) -> np.ndarray:
        embeddings = []
        for page in pages:
            text = page["text"] if page["text"].strip() else " "
            embeddings.append(self.encode_gici(text))
        return np.vstack(embeddings)

    def search(self, query: str, pages: List[Dict], page_embeddings: np.ndarray, top_k: int = 3) -> List[Dict]:
        query_embedding = self.encode_gici(query).reshape(1, -1)
        sims = cosine_similarity(query_embedding, page_embeddings)[0]
        top_indices = sims.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            page = pages[idx]
            results.append(
                {
                    "page_num": page["page_num"],
                    "score": float(sims[idx]),
                    "text": page["text"],
                    "image": page.get("image"),
                    "vision_extraction": page.get("vision_extraction", ""),
                }
            )
        return results
