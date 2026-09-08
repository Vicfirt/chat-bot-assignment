import os

import pytest

os.environ.setdefault("LLM_MODE", "dummy")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.config import get_settings
    from app.rag import cache

    get_settings.cache_clear()
    cache.clear_all()
    yield
    get_settings.cache_clear()
    cache.clear_all()
