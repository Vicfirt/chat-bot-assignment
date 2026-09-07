import os

import pytest

os.environ.setdefault("LLM_MODE", "dummy")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
