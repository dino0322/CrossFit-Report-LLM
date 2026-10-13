from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseTextGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def truncate_context(self, request: str, context: str, prompt_builder) -> str:
        raise NotImplementedError


class BaseVisionExtractor(ABC):
    @abstractmethod
    def extract(self, image: Any, page_type: Optional[str] = None, page_text: str = "") -> str:
        raise NotImplementedError
