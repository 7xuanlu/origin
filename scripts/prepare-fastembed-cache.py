#!/usr/bin/env python3
"""Prepare Wenlan's pinned FastEmbed snapshot once for all CI platforms."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Callable


MODEL_REVISION = "738cad1c108e2f23649db9e44b2eab988626493b"
REPO_CACHE_NAME = "models--Qdrant--bge-base-en-v1.5-onnx-Q"
MODEL_FILES = {
    "model_optimized.onnx": "4e556722bc4f65716c544c8a931f1e90fb3f866e5741fd93a96f051d673339c7",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "config.json": "86f84a5285de7f1ee673f712387219ef1e261ec27dcd870e793a80f9da1aaa3b",
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "tokenizer_config.json": "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
}
MODEL_BASE_URL = (
    "https://huggingface.co/Qdrant/bge-base-en-v1.5-onnx-Q"
    f"/resolve/{MODEL_REVISION}"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(path: Path) -> None:
    """Replace a valid legacy symlink with a portable regular file."""
    if not path.is_symlink():
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        with path.open("rb") as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, path)


def download(
    url: str,
    target: Path,
    expected_sha256: str,
    *,
    retries: int,
    sleep: Callable[[float], None],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary: Path | None = None
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "wenlan-ci-fastembed-cache/1"},
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    delete=False,
                ) as destination:
                    temporary = Path(destination.name)
                    shutil.copyfileobj(response, destination, length=1024 * 1024)
            actual_sha256 = sha256(temporary)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"{target.name} sha256 {actual_sha256}, expected {expected_sha256}"
                )
            os.replace(temporary, target)
            return
        except Exception as error:
            last_error = error
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt < retries:
                sleep(min(2**attempt, 30))
    raise RuntimeError(f"failed to fetch {target.name} after {retries} attempts") from last_error


def ensure_model_cache(
    cache_dir: Path,
    base_url: str = MODEL_BASE_URL,
    *,
    retries: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    repo = cache_dir / REPO_CACHE_NAME
    snapshot = repo / "snapshots" / MODEL_REVISION
    snapshot.mkdir(parents=True, exist_ok=True)

    for name, expected_sha256 in MODEL_FILES.items():
        target = snapshot / name
        if target.exists() and sha256(target) == expected_sha256:
            materialize(target)
            continue
        target.unlink(missing_ok=True)
        download(
            f"{base_url.rstrip('/')}/{name}",
            target,
            expected_sha256,
            retries=retries,
            sleep=sleep,
        )

    refs = repo / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(MODEL_REVISION)

    # The portable archive needs only the materialized snapshot. Removing the
    # legacy blob store avoids archiving the 218 MB model twice.
    shutil.rmtree(repo / "blobs", ignore_errors=True)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("FASTEMBED_CACHE_DIR", ".fastembed_cache")),
    )
    parser.add_argument("--base-url", default=MODEL_BASE_URL)
    args = parser.parse_args()

    snapshot = ensure_model_cache(args.cache_dir, args.base_url)
    print(f"FastEmbed snapshot ready: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
