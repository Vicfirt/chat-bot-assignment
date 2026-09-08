from app.config import get_settings


def test_defaults_present():
    s = get_settings()
    assert s.tax_year == 2025
    assert s.chroma_collection == "irs_pubs"
    assert s.max_retries == 1


def test_env_override(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("TAX_YEAR", "2023")
    s = get_settings()
    assert s.tax_year == 2023
    get_settings.cache_clear()
