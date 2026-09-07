from __future__ import annotations

from app.config import get_settings


def get_langfuse_callbacks() -> list:
    s = get_settings()
    if not s.langfuse_enabled:
        return []
    try:
        from langfuse.callback import CallbackHandler

        return [CallbackHandler(public_key=s.langfuse_public_key,
                                secret_key=s.langfuse_secret_key, host=s.langfuse_host)]
    except Exception:  # noqa: BLE001 - tracing must never break request handling
        return []
