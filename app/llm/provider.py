from __future__ import annotations

import logging
from typing import Protocol

from app.config import get_settings
from app.observability.metrics import (
    LLM_FALLBACK,
    record_llm_tokens,
    time_llm_call,
)

_log = logging.getLogger(__name__)

CALC_MARKERS = ("how much", "owe", "calculate", "estimate", "take-home", "take home", "$")
SCOPE_MARKERS = ("state tax", "corporate", "vat", "capital of", "weather")

# Cached result of the one-time "is the configured Ollama model reachable?" probe.
_ollama_ready: bool | None = None


class LLMProvider(Protocol):
    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str: ...

    def classify(self, prompt: str, labels: list[str], *, system: str | None = None) -> str: ...


class DummyLLM:
    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        with time_llm_call("complete", "dummy"):
            return "[dummy] " + prompt.strip()[:200]

    def classify(self, prompt: str, labels: list[str], *, system: str | None = None) -> str:
        with time_llm_call("classify", "dummy"):
            return self._classify(prompt, labels)

    def _classify(self, prompt: str, labels: list[str]) -> str:
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

    def _generate(self, prompt: str, system: str | None, max_tokens: int, op: str) -> str:
        with time_llm_call(op, "ollama"):
            resp = self._client.generate(
                model=self._model,
                prompt=prompt,
                system=system or "",
                options={"num_predict": max_tokens, "temperature": 0.1},
            )
        record_llm_tokens(resp.get("prompt_eval_count"), resp.get("eval_count"))
        return resp["response"].strip()

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        return self._generate(prompt, system, max_tokens, "complete")

    def classify(self, prompt: str, labels: list[str], *, system: str | None = None) -> str:
        instruction = (
            f"{prompt}\n\nAnswer with exactly one of these labels and nothing else: "
            f"{', '.join(labels)}."
        )
        raw = self._generate(instruction, system, 8, "classify").lower()
        for label in labels:
            if label in raw:
                return label
        return labels[0]


def _ollama_model_reachable(base_url: str, model: str) -> bool:
    """Probe the Ollama server once; cache the answer for the process."""
    global _ollama_ready
    if _ollama_ready is None:
        try:
            import ollama

            models = ollama.Client(host=base_url, timeout=3.0).list().get("models", [])
            have = {(m.get("model") or m.get("name") or "").split(":")[0] for m in models}
            _ollama_ready = model.split(":")[0] in have
            if not _ollama_ready:
                _log.warning("Ollama at %s has no model matching %r (available: %s)",
                             base_url, model, sorted(have) or "none")
        except Exception as e:  # noqa: BLE001 - any failure means "not usable"
            _log.warning("Ollama not reachable at %s: %s", base_url, e)
            _ollama_ready = False
    return _ollama_ready


def get_llm() -> LLMProvider:
    s = get_settings()
    if s.llm_mode == "dummy":
        return DummyLLM()
    if s.llm_fallback_dummy and not _ollama_model_reachable(s.ollama_base_url, s.llm_model):
        _log.warning("Falling back to the dummy LLM (set LLM_FALLBACK_DUMMY=false to fail hard)")
        LLM_FALLBACK.inc()
        return DummyLLM()
    return OllamaLLM()
