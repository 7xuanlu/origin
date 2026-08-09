#!/usr/bin/env python3
"""Tests that archive smoke uses exact payloads and fails closed."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import release_archive
from release_targets import release_assets


PACKAGE_PATH = Path(__file__).with_name("package-release-artifacts.py")
PACKAGE_SPEC = importlib.util.spec_from_file_location("package_release", PACKAGE_PATH)
assert PACKAGE_SPEC and PACKAGE_SPEC.loader
PACKAGE = importlib.util.module_from_spec(PACKAGE_SPEC)
PACKAGE_SPEC.loader.exec_module(PACKAGE)

SMOKE_PATH = Path(__file__).with_name("smoke-release-archive.py")
SMOKE_SPEC = importlib.util.spec_from_file_location("smoke_release", SMOKE_PATH)
assert SMOKE_SPEC and SMOKE_SPEC.loader
SMOKE = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(SMOKE)


class SmokeReleaseArchiveTests(unittest.TestCase):
    def package(self, root: Path, target: str) -> Path:
        binary_dir = root / "bin"
        binary_dir.mkdir()
        for member in {
            member
            for asset in release_assets(target=target)
            for member in asset["members"]
        }:
            (binary_dir / member).write_bytes(b"payload")
        dist = root / "dist"
        PACKAGE.package_target(target, binary_dir, dist, 1_700_000_000)
        return dist

    @mock.patch.object(SMOKE.subprocess, "run")
    def test_linux_archive_runs_only_three_canonical_binaries(self, run: mock.Mock) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dist = self.package(root, "x86_64-unknown-linux-gnu")
            SMOKE.smoke_target(
                "x86_64-unknown-linux-gnu", dist, root / "unused.ps1"
            )
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(call.args[0][1] == "--help" for call in run.call_args_list))

    @mock.patch.object(SMOKE.subprocess, "run")
    def test_windows_archive_runs_binaries_and_semantic_smoke(self, run: mock.Mock) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dist = self.package(root, "x86_64-pc-windows-msvc")
            smoke_script = root / "smoke.ps1"
            smoke_script.write_text("exit 0", encoding="utf-8")
            SMOKE.smoke_target("x86_64-pc-windows-msvc", dist, smoke_script)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(run.call_args_list[-1].args[0][:3], ["pwsh", "-NoProfile", "-File"])

    @mock.patch.object(SMOKE.subprocess, "run")
    def test_extra_asset_and_failed_binary_are_fatal(self, run: mock.Mock) -> None:
        run.return_value.returncode = 1
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dist = self.package(root, "x86_64-unknown-linux-gnu")
            (dist / "extra").write_bytes(b"no")
            with self.assertRaisesRegex(SMOKE.SmokeError, "inventory mismatch"):
                SMOKE.smoke_target(
                    "x86_64-unknown-linux-gnu", dist, root / "unused.ps1"
                )
            (dist / "extra").unlink()
            with self.assertRaisesRegex(SMOKE.SmokeError, "--help failed"):
                SMOKE.smoke_target(
                    "x86_64-unknown-linux-gnu", dist, root / "unused.ps1"
                )

    @mock.patch.object(SMOKE.subprocess, "run")
    def test_darwin_standalone_bytes_must_equal_multi_archive_member(
        self, run: mock.Mock
    ) -> None:
        run.return_value.returncode = 0
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dist = self.package(root, "aarch64-apple-darwin")
            hostile = root / "hostile-wenlan"
            hostile.write_bytes(b"different bytes")
            release_archive.create_tar_gz(
                dist / "wenlan-cli-darwin-arm64.tar.gz",
                [(hostile, "wenlan")],
                1_700_000_000,
            )
            with self.assertRaisesRegex(SMOKE.SmokeError, "standalone archive bytes"):
                SMOKE.smoke_target(
                    "aarch64-apple-darwin", dist, root / "unused.ps1"
                )
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
