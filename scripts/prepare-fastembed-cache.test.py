#!/usr/bin/env python3
"""Contract tests for the cold FastEmbed cache preparation helper."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError


SCRIPT = Path(__file__).with_name("prepare-fastembed-cache.py")


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_fastembed_cache", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareFastEmbedCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.files = {
            "model_optimized.onnx": b"model fixture",
            "tokenizer.json": b'{"tokenizer":true}',
            "config.json": b'{"config":true}',
        }
        for name, content in self.files.items():
            (self.source / name).write_bytes(content)
        self.module.MODEL_FILES = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in self.files.items()
        }
        self.module.MODEL_REVISION = "fixture-revision"
        self.module.REPO_CACHE_NAME = "models--fixture"

    def test_cold_cache_is_materialized_and_corruption_is_repaired(self) -> None:
        cache = self.root / "cache"
        self.module.ensure_model_cache(
            cache,
            self.source.as_uri(),
            retries=1,
            sleep=lambda _seconds: None,
        )

        snapshot = cache / "models--fixture" / "snapshots" / "fixture-revision"
        self.assertEqual(
            (cache / "models--fixture" / "refs" / "main").read_text(),
            "fixture-revision",
        )
        for name, expected in self.files.items():
            path = snapshot / name
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.read_bytes(), expected)

        (snapshot / "model_optimized.onnx").write_bytes(b"corrupt")
        self.module.ensure_model_cache(
            cache,
            self.source.as_uri(),
            retries=1,
            sleep=lambda _seconds: None,
        )
        self.assertEqual(
            (snapshot / "model_optimized.onnx").read_bytes(),
            self.files["model_optimized.onnx"],
        )

    def test_transient_download_failure_retries_before_succeeding(self) -> None:
        cache = self.root / "cache"
        original_urlopen = self.module.urllib.request.urlopen
        attempts = 0

        def flaky_urlopen(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise URLError("transient")
            return original_urlopen(*args, **kwargs)

        self.module.urllib.request.urlopen = flaky_urlopen
        self.addCleanup(
            setattr,
            self.module.urllib.request,
            "urlopen",
            original_urlopen,
        )

        self.module.ensure_model_cache(
            cache,
            self.source.as_uri(),
            retries=2,
            sleep=lambda _seconds: None,
        )
        self.assertGreaterEqual(attempts, len(self.files) + 1)


if __name__ == "__main__":
    unittest.main()
