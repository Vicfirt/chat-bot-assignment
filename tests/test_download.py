import hashlib
from pathlib import Path

from app.ingest.download import load_sources, verify_sha256


def test_load_sources_reads_manifest():
    src = load_sources(Path("app/ingest/sources.yaml"))
    names = {s["name"] for s in src}
    assert {"pub17", "pub501", "pub505"} <= names
    assert all(s["url"].startswith("https://") for s in src)


def test_verify_sha256(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    good = hashlib.sha256(b"hello").hexdigest()
    assert verify_sha256(f, good) is True
    assert verify_sha256(f, "deadbeef") is False
