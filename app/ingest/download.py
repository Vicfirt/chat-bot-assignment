from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import yaml

from app.config import get_settings


def load_sources(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("sources.yaml must be a list")
    return data


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h == expected


def download_all(
    sources_path: str | Path | None = None,
    dest_dir: str | Path | None = None,
) -> list[Path]:
    sources_path = Path(sources_path or "app/ingest/sources.yaml")
    dest = Path(dest_dir or get_settings().raw_pdf_dir)
    dest.mkdir(parents=True, exist_ok=True)

    out: list[Path] = []
    for s in load_sources(sources_path):
        target = dest / f"{s['name']}.pdf"
        if target.exists() and s["sha256"] != "PENDING" and verify_sha256(target, s["sha256"]):
            out.append(target)
            continue
        print(f"downloading {s['url']} -> {target}")
        urllib.request.urlretrieve(s["url"], target)  # noqa: S310  (trusted IRS URL)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if s["sha256"] == "PENDING":
            print(f"  computed sha256 for {s['name']}: {digest}")
        elif digest != s["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {s['name']}: got {digest}")
        out.append(target)
    return out


if __name__ == "__main__":
    for p in download_all():
        print(p)
