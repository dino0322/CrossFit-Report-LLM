from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseTextGenerator


class HFQwenTextGenerator(BaseTextGenerator):
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_new_tokens: int = 2000,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
        max_context_tokens: int = 3000,
        system_prompt: str = "너는 머신러닝 모델 평가 전문가다.",
    ):
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN is not set.")

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.max_context_tokens = max_context_tokens
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device.type == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            ).eval()
        elif device.type == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
            ).to(device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
            ).to(device).eval()

    def truncate_context(self, request: str, context: str, prompt_builder: Callable[[str, str], str]) -> str:
        fixed_prompt = prompt_builder(request, "")
        fixed_tokens = len(self.tokenizer.encode(fixed_prompt, add_special_tokens=False))
        allowed = self.max_context_tokens - fixed_tokens
        if allowed <= 0:
            return ""

        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(context_tokens) <= allowed:
            return context
        return self.tokenizer.decode(context_tokens[:allowed], skip_special_tokens=True)

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(chat_text, return_tensors="pt", truncation=True)
        if self.device.type in {"cpu", "mps"}:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                top_p=self.top_p if self.do_sample else None,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return answer or "응답을 생성하지 못했습니다."


class OpenAITextGenerator(BaseTextGenerator):
    def __init__(
        self,
        model_name: str,
        max_context_tokens: int = 3000,
        temperature: float = 0.1,
        system_prompt: str = "당신은 문서 기반으로만 답변하는 정확한 연구 보조 도우미입니다.",
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

    def truncate_context(self, request: str, context: str, prompt_builder: Callable[[str, str], str]) -> str:
        return context

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or "응답을 생성하지 못했습니다."


class OllamaTextGenerator(BaseTextGenerator):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_context_tokens: int = 3000,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_tokens = max_context_tokens

    def truncate_context(self, request: str, context: str, prompt_builder: Callable[[str, str], str]) -> str:
        return context

    def generate(self, prompt: str) -> str:
        import requests

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip() or "응답을 생성하지 못했습니다."
