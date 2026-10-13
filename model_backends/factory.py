from __future__ import annotations

from .text_generators import HFQwenTextGenerator, OpenAITextGenerator, OllamaTextGenerator
from .vision_generators import QwenVLExtractor


def build_text_generator(cfg, device):
    if cfg.llm_provider == "hf_qwen":
        return HFQwenTextGenerator(
            model_name=cfg.llm_model,
            device=device,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            max_context_tokens=cfg.max_context_tokens,
        )
    if cfg.llm_provider == "openai":
        return OpenAITextGenerator(
            model_name=cfg.llm_model,
            max_context_tokens=cfg.max_context_tokens,
            temperature=cfg.temperature,
        )
    if cfg.llm_provider == "ollama":
        return OllamaTextGenerator(
            model_name=cfg.ollama_model,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_context_tokens=cfg.max_context_tokens,
        )
    raise ValueError(f"Unsupported llm_provider: {cfg.llm_provider}")


def build_vision_extractor(cfg, device):
    if not getattr(cfg, "use_vlm", True):
        return None
    if cfg.vlm_provider == "qwen_vl":
        return QwenVLExtractor(
            model_name=cfg.vlm_model,
            device=device,
            max_new_tokens=cfg.vlm_max_new_tokens,
        )
    raise ValueError(f"Unsupported vlm_provider: {cfg.vlm_provider}")
