from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .base import BaseVisionExtractor


class QwenVLExtractor(BaseVisionExtractor):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_new_tokens: int = 512,
    ):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN is not set.")

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)

        if device.type == "cuda":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            ).eval()
        elif device.type == "mps":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
            ).to(device).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                token=hf_token,
            ).to(device).eval()

    def _build_prompt(self, page_type: str, page_text: str) -> str:
        common = (
            "보이는 정보만 사용하라. 추측하지 마라. 숫자, 모델명, 축 라벨, feature 이름이 보이면 그대로 적어라. "
            "문맥상 확인 불가한 내용은 그렇게 적어라. 한국어로 간결하게 답하라."
        )
        prompts = {
            "metrics": (
                "이 페이지의 성능표를 읽고 모델명과 Accuracy, Precision, Recall, F1, AUC, AP 같은 핵심 지표를 정리하라. "
                "표에 여러 모델이 있으면 비교 가능하도록 줄바꿈으로 정리하라. " + common
            ),
            "confusion": (
                "이 페이지의 confusion matrix를 읽고 가능한 경우 TN, FP, FN, TP를 추출하라. "
                "행/열 라벨이나 모델명이 보이면 같이 적고, 오류 유형을 한두 문장으로 요약하라. " + common
            ),
            "roc": (
                "이 페이지의 ROC curve를 읽고 어떤 모델의 곡선인지, AUC 값이 보이는지, 곡선이 좌상단에 가까운지 요약하라. " + common
            ),
            "pr": (
                "이 페이지의 Precision-Recall curve를 읽고 어떤 모델의 곡선인지, recall이 증가할 때 precision 유지 정도를 요약하라. " + common
            ),
            "calibration": (
                "이 페이지의 calibration curve를 읽고 대각선 근접 여부와 확률 보정 상태를 요약하라. " + common
            ),
            "shap": (
                "이 페이지의 SHAP summary plot을 읽고 상위 feature 이름과 중요도를 요약하라. 방향성이 보이면 함께 적어라. " + common
            ),
            "lime": (
                "이 페이지의 LIME explanation을 읽고 양의 기여와 음의 기여 feature를 구분해 요약하라. " + common
            ),
            "other": (
                "이 페이지는 모델 평가 리포트의 일부다. 보이는 시각 정보와 핵심 텍스트를 요약하라. " + common
            ),
        }
        header = "[페이지 텍스트 힌트]\n" + (page_text[:1200] if page_text else "없음") + "\n\n"
        return header + prompts.get(page_type, prompts["other"])

    def extract(self, image: Image.Image, page_type: Optional[str] = None, page_text: str = "") -> str:
        prompt = self._build_prompt(page_type or "other", page_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        if self.device.type in {"cpu", "mps"}:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[:, input_len:]
        result = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result or "문맥상 확인 불가"
