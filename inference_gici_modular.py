from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

import torch

from model_backends.factory import build_text_generator, build_vision_extractor
from pdf_utils.page_loader import extract_pdf_pages
from retrievers.gici_retriever import GICIRetriever, GICIRetrieverConfig
from report_context_utils import classify_page, select_template_pages, merge_selected_and_retrieved, build_context_with_visuals
from report_prompt_utils import build_eval_report_prompt


def get_device(prefer: str = "auto", verbose: bool = True):
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif prefer == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif prefer == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"[Current Device]: {device}")
    return device


def find_report_pdfs(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"폴더 없음: {folder_path}")
    pdfs = list(folder.glob("*_Model_Evaluation_Report.pdf"))
    return sorted(pdfs)


def ensure_output_dir(path: str = "./outputs") -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@dataclass
class GICIInferenceConfig:
    mkor_dir: str
    meng_dir: str
    device: str = "auto"
    max_length: int = 512

    llm_provider: str = "hf_qwen"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    ollama_model: str = "llama3.2:3b"

    use_vlm: bool = True
    vlm_provider: str = "qwen_vl"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    max_new_tokens: int = 2000
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    max_context_tokens: int = 3000
    vlm_max_new_tokens: int = 512

    pdf_zoom: float = 2.0
    save_page_extractions: bool = True


class GICIInferenceEngine:
    def __init__(self, cfg: GICIInferenceConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device, verbose=True)

        self.retriever = GICIRetriever(
            GICIRetrieverConfig(
                mkor_dir=cfg.mkor_dir,
                meng_dir=cfg.meng_dir,
                max_length=cfg.max_length,
            ),
            device=self.device,
        )
        self.text_generator = build_text_generator(cfg, self.device)
        self.vision_extractor = build_vision_extractor(cfg, self.device)

    def extract_pdf_pages(self, pdf_path: str) -> List[Dict]:
        return extract_pdf_pages(pdf_path, zoom=self.cfg.pdf_zoom)

    def build_page_embeddings(self, pages: List[Dict]):
        return self.retriever.build_page_embeddings(pages)

    def search(self, query: str, pages: List[Dict], page_embeddings, top_k: int = 3) -> List[Dict]:
        return self.retriever.search(query, pages, page_embeddings, top_k=top_k)

    def enrich_pages_with_vision(self, pages: List[Dict]) -> List[Dict]:
        if self.vision_extractor is None:
            return pages

        enriched = []
        for page in pages:
            category = classify_page(page)
            vision_text = self.vision_extractor.extract(
                image=page["image"],
                page_type=category,
                page_text=page.get("text", ""),
            )
            new_page = dict(page)
            new_page["vision_extraction"] = vision_text
            enriched.append(new_page)
        return enriched

    def generate_report(self, request: str, context: str) -> str:
        context = self.text_generator.truncate_context(request, context, build_eval_report_prompt)
        prompt = build_eval_report_prompt(request, context)
        return self.text_generator.generate(prompt)


def save_page_level_extractions(output_dir: Path, pdf_stem: str, pages: List[Dict]) -> None:
    page_dir = output_dir / "page_level_extractions" / pdf_stem
    page_dir.mkdir(parents=True, exist_ok=True)
    for page in pages:
        save_path = page_dir / f"page_{page['page_num']:03d}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(
                f"[Page {page['page_num']}]\n"
                f"[Category]\n{classify_page(page)}\n\n"
                f"[Extracted Text]\n{page.get('text', '')}\n\n"
                f"[Vision Extraction]\n{page.get('vision_extraction', '')}\n"
            )


if __name__ == "__main__":
    engine = GICIInferenceEngine(
        GICIInferenceConfig(
            mkor_dir="./saved_models_gici/final_mkor",
            meng_dir="./saved_models_gici/final_meng",
            llm_provider="hf_qwen",
            llm_model="Qwen/Qwen2.5-3B-Instruct",
            use_vlm=True,
            vlm_provider="qwen_vl",
            vlm_model="Qwen/Qwen2.5-VL-7B-Instruct",
            max_new_tokens=2000,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
            max_context_tokens=3000,
            vlm_max_new_tokens=512,
        )
    )

    folder_path = "./reports"
    pdf_files = find_report_pdfs(folder_path)
    if not pdf_files:
        raise ValueError("reports 폴더에 *_Model_Evaluation_Report.pdf 파일이 없습니다.")

    output_dir = ensure_output_dir("./outputs")

    for pdf_path in pdf_files:
        print(f"\n=== Processing: {pdf_path.name} ===")

        pages = engine.extract_pdf_pages(str(pdf_path))
        pages = engine.enrich_pages_with_vision(pages)

        if engine.cfg.save_page_extractions:
            save_page_level_extractions(output_dir, pdf_path.stem, pages)

        page_embeddings = engine.build_page_embeddings(pages)
        request = (
            "이 모델 평가 리포트를 분석해줘. 성능표가 여러 개이면 모델 간 비교를 먼저 하고, "
            "시각화는 해당 페이지에 실제로 있는 모델만 해석해줘."
        )

        retrieved = engine.search(request, pages, page_embeddings, top_k=3)
        template_pages = select_template_pages(pages)
        selected_pages = merge_selected_and_retrieved(template_pages, retrieved)

        context = build_context_with_visuals(selected_pages)
        report = engine.generate_report(request, context)

        print("\n=== RESULT ===")
        print(report)

        save_path = output_dir / f"{pdf_path.stem}_analysis.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved analysis to {save_path}")
