#!/usr/bin/env python3
"""Contract tests for exact-byte release runtime image preparation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import release_archive


SCRIPT = Path(__file__).with_name("verify-release-runtime-image.py")
ROOT = SCRIPT.parent.parent
DOCKERFILE = ROOT / "docker" / "Dockerfile.release-runtime"


def load_script():
    spec = importlib.util.spec_from_file_location("verify_release_runtime_image", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RuntimeImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()

    def make_linux_asset(self, root: Path) -> tuple[Path, dict]:
        inputs = root / "inputs"
        inputs.mkdir()
        members = []
        for name in self.module.ARCHIVE_MEMBERS:
            path = inputs / name
            path.write_bytes(f"exact-{name}-bytes".encode())
            path.chmod(0o755)
            members.append((path, name))
        assets = root / "assets"
        assets.mkdir()
        archive = assets / "wenlan-linux-x64.tar.gz"
        release_archive.create_tar_gz(archive, members, 1_700_000_000)
        receipt = {
            "assets": [
                {
                    "name": archive.name,
                    "sha256": sha256(archive),
                    "size": archive.stat().st_size,
                    "target": "x86_64-unknown-linux-gnu",
                }
            ]
        }
        return assets, receipt

    def test_prepare_context_binds_archive_and_server_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            assets, receipt = self.make_linux_asset(root)
            work = root / "work"
            work.mkdir()

            evidence = self.module.prepare_context(
                receipt, assets, "x86_64-unknown-linux-gnu", work
            )

            context = Path(str(evidence["build_context"]))
            self.assertEqual(evidence["archive_sha256"], receipt["assets"][0]["sha256"])
            self.assertEqual(
                evidence["server_sha256"], sha256(context / "wenlan-server")
            )
            self.assertTrue((context / "data" / ".volume-seed").is_file())
            self.assertTrue(os.access(context / "wenlan-server", os.X_OK))

    def test_prepare_context_rejects_archive_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            assets, receipt = self.make_linux_asset(root)
            receipt["assets"][0]["sha256"] = "0" * 64
            work = root / "work"
            work.mkdir()

            with self.assertRaisesRegex(
                self.module.RuntimeImageError, "closed receipt"
            ):
                self.module.prepare_context(
                    receipt, assets, "x86_64-unknown-linux-gnu", work
                )

    def test_image_inspection_is_closed_world_for_runtime_contract(self) -> None:
        image = {
            "Id": "sha256:" + "a" * 64,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {
                "User": "65532:65532",
                "Entrypoint": ["/usr/local/bin/wenlan-server"],
                "Volumes": {"/data": {}},
                "Env": sorted(self.module.REQUIRED_ENV),
            },
        }
        result = subprocess.CompletedProcess(
            ["docker"], 0, stdout=json.dumps([image]), stderr=None
        )
        with mock.patch.object(self.module, "_run", return_value=result):
            self.assertEqual(
                self.module._inspect_image("wenlan:test", "amd64")["Id"], image["Id"]
            )

        for field, value in [
            ("Architecture", "arm64"),
            ("Config.User", "root"),
            ("Config.Entrypoint", ["/bin/sh"]),
            ("Config.Volumes", {}),
        ]:
            candidate = json.loads(json.dumps(image))
            if field.startswith("Config."):
                candidate["Config"][field.split(".", 1)[1]] = value
            else:
                candidate[field] = value
            failed = subprocess.CompletedProcess(
                ["docker"], 0, stdout=json.dumps([candidate]), stderr=None
            )
            with self.subTest(field=field), mock.patch.object(
                self.module, "_run", return_value=failed
            ), self.assertRaises(self.module.RuntimeImageError):
                self.module._inspect_image("wenlan:test", "amd64")

    def test_runtime_dockerfile_and_prelogin_script_are_minimal(self) -> None:
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        active = [
            line.strip()
            for line in dockerfile.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            active[0],
            "FROM gcr.io/distroless/cc-debian13:nonroot@sha256:"
            "d97bc0a941b8d4be647dc0ee75b264ddbb772f1ac5ba690a4309c00723b23775",
        )
        self.assertEqual(sum(line.startswith("FROM ") for line in active), 1)
        self.assertIn(
            "COPY --chown=0:0 --chmod=0755 wenlan-server /usr/local/bin/wenlan-server",
            active,
        )
        self.assertIn("COPY --chown=65532:65532 data/ /data/", active)
        self.assertIn("USER 65532:65532", active)
        self.assertIn('VOLUME ["/data"]', active)
        forbidden = ("RUN ", "ADD ", "cargo", "strip", "libonnxruntime")
        for marker in forbidden:
            self.assertFalse(any(marker in line for line in active), marker)

        script = SCRIPT.read_text(encoding="utf-8")
        for marker in (
            '"buildx",\n                "build"',
            '"--load"',
            '"docker", "cp"',
            "/api/health",
            "/api/memory/store",
            "/api/memory/search",
            "/api/status",
        ):
            self.assertIn(marker, script)
        self.assertNotIn('"docker", "login"', script)
        self.assertNotIn('"docker", "push"', script)


if __name__ == "__main__":
    unittest.main()
