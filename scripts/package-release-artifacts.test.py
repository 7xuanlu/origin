#!/usr/bin/env python3
"""Tests for canonical final release packaging."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from release_archive import ArchiveError, safe_extract_archive
from release_targets import release_assets, release_matrix


MODULE_PATH = Path(__file__).with_name("package-release-artifacts.py")
SPEC = importlib.util.spec_from_file_location("package_release_artifacts", MODULE_PATH)
assert SPEC and SPEC.loader
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class PackageReleaseArtifactsTests(unittest.TestCase):
    def make_binary_dir(self, root: Path, target: str) -> Path:
        binary_dir = root / "bin"
        binary_dir.mkdir()
        members = {
            member
            for asset in release_assets(target=target)
            for member in asset["members"]
        }
        for member in members:
            payload = binary_dir / member
            payload.write_bytes(f"payload:{target}:{member}\n".encode())
            if not member.lower().endswith((".dll", ".txt")):
                payload.chmod(0o755)
        return binary_dir

    def test_every_target_produces_only_exact_assets_and_members(self) -> None:
        for target_entry in release_matrix()["include"]:
            target = target_entry["target"]
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                binary_dir = self.make_binary_dir(root, target)
                output_dir = root / "dist"
                outputs = PACKAGE.package_target(target, binary_dir, output_dir, 1_700_000_000)
                expected_assets = release_assets(target=target)
                self.assertEqual(
                    [path.name for path in outputs],
                    [asset["name"] for asset in expected_assets],
                )
                self.assertEqual(
                    sorted(path.name for path in output_dir.iterdir()),
                    sorted(asset["name"] for asset in expected_assets),
                )
                for index, asset in enumerate(expected_assets):
                    extracted = root / f"extract-{index}"
                    safe_extract_archive(
                        output_dir / asset["name"],
                        extracted,
                        asset["archive"],
                        asset["members"],
                    )
                    self.assertEqual(
                        sorted(path.name for path in extracted.iterdir()),
                        sorted(asset["members"]),
                    )

    def test_missing_runtime_or_binary_fails_closed(self) -> None:
        for target, missing in [
            ("x86_64-unknown-linux-gnu", "wenlan-mcp"),
            ("x86_64-pc-windows-msvc", "onnxruntime.dll"),
            ("x86_64-pc-windows-msvc", "VulkanRT-License.txt"),
        ]:
            with self.subTest(target=target, missing=missing), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                binary_dir = self.make_binary_dir(root, target)
                (binary_dir / missing).unlink()
                with self.assertRaisesRegex(ArchiveError, "regular file"):
                    PACKAGE.package_target(
                        target, binary_dir, root / "dist", 1_700_000_000
                    )

    def test_nonempty_output_and_unknown_target_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            binary_dir = self.make_binary_dir(root, "aarch64-apple-darwin")
            output_dir = root / "dist"
            output_dir.mkdir()
            (output_dir / "unexpected").write_text("no", encoding="utf-8")
            with self.assertRaisesRegex(ArchiveError, "absent or empty"):
                PACKAGE.package_target(
                    "aarch64-apple-darwin",
                    binary_dir,
                    output_dir,
                    1_700_000_000,
                )
            with self.assertRaisesRegex(ValueError, "not a shipped release target"):
                PACKAGE.package_target(
                    "x86_64-apple-darwin",
                    binary_dir,
                    root / "other",
                    1_700_000_000,
                )

    def test_same_inputs_and_epoch_are_byte_deterministic(self) -> None:
        for target in ("aarch64-apple-darwin", "x86_64-pc-windows-msvc"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                binary_dir = self.make_binary_dir(root, target)
                first = PACKAGE.package_target(
                    target, binary_dir, root / "first", 1_700_000_000
                )
                second = PACKAGE.package_target(
                    target, binary_dir, root / "second", 1_700_000_000
                )
                self.assertEqual(
                    {path.name: path.read_bytes() for path in first},
                    {path.name: path.read_bytes() for path in second},
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
