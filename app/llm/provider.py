from __future__ import annotations

from typing import Protocol

from app.config import get_settings

CALC_MARKERS = ("how much", "owe", "calculate", "estimate", "take-home", "take home", "$")
SCOPE_MARKERS = ("state tax", "corporate", "vat", "capital of", "weather")


class LLMProvider(Protocol):
    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str: ...

    def classify(self, prompt: str, labels: list[str], *, system: str | None = None) -> str: ...


class DummyLLM:
    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        return "[dummy] " + prompt.strip()[:200]

    def classify(self, prompt: str, labels: list[str], *, system: str | None = None) -> str:
        low = prompt.lower()
        for label in labels:
            if label.replace("_", " ") in low or label in low:
                return label
        if "out_of_scope" in labels and any(m in low for m in SCOPE_MARKERS):
            return "out_of_scope"
        has_calc = any(m in low for m in CALC_MARKERS)
        wants_rule = any(w in low for w in ("what is", "explain", "rule", "deduction", "who"))
        if has_calc and wants_rule and "rag_plus_calc" in labels:
            return "rag_plus_calc"
        if has_calc and "needs_calc" in labels:
            return "needs_calc"
        return labels[0]


class OllamaLLM:
    def __init__(self) -> None:
        import ollama

        s = get_settings()
        self._client = ollama.Client(host=s.ollama_base_url)
        self._model = s.llm_model

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        resp = self._client.generate(
            model=self._model,
            prompt=prompt,
            system=system or "",
            options={"num_predict": max_tokens, "temperature": 0.1},
        )
        return resp["response"].strip()

    def classify(self, prompt: str, labels: list[str], *, system: str | None = None) -> str:
        instruction = (
            f"{prompt}\n\nAnswer with exactly one of these labels and nothing else: "
            f"{', '.join(labels)}."
        )
        raw = self.complete(instruction, system=system, max_tokens=8).lower()
        for label in labels:
            if label in raw:
                return label
        return labels[0]


def get_llm() -> LLMProvider:
    if get_settings().llm_mode == "dummy":
        return DummyLLM()
    return OllamaLLM()
