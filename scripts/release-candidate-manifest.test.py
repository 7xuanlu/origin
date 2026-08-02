#!/usr/bin/env python3
"""Mutation tests for release-candidate manifest creation and parsing."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from release_targets import release_assets


MODULE_PATH = Path(__file__).with_name("release-candidate-manifest.py")
SPEC = importlib.util.spec_from_file_location("release_candidate_manifest", MODULE_PATH)
assert SPEC and SPEC.loader
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


HEAD_SHA = "a" * 40
TREE_SHA = "b" * 40


class ReleaseCandidateManifestTests(unittest.TestCase):
    def make_manifest(self, root: Path) -> dict:
        target = "aarch64-apple-darwin"
        for asset in release_assets(target=target):
            (root / asset["name"]).write_bytes(asset["name"].encode())
        return MANIFEST.build_manifest(
            repository="7xuanlu/wenlan",
            repository_id=123,
            run_id=456,
            run_attempt=2,
            pr_number=419,
            pr_author="7xuanlu",
            head_sha=HEAD_SHA,
            tree_sha=TREE_SHA,
            version="0.15.4",
            target=target,
            dist_dir=root,
        )

    def test_create_and_read_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = self.make_manifest(root)
            path = root / MANIFEST.MANIFEST_NAME
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(MANIFEST.read_manifest(path), manifest)

    def test_schema_identity_and_inventory_mutations_fail(self) -> None:
        mutations = {
            "schema": lambda value: value.__setitem__("schema_version", 2),
            "extra-key": lambda value: value.__setitem__("unexpected", True),
            "run": lambda value: value["workflow"].__setitem__("run_id", 999),
            "head": lambda value: value["source"].__setitem__("commit_sha", "c" * 40),
            "tag": lambda value: value["release"].__setitem__("tag", "v9.9.9"),
            "asset-name": lambda value: value["assets"][0].__setitem__("name", "other"),
            "asset-digest": lambda value: value["assets"][0].__setitem__("sha256", "no"),
            "missing-asset": lambda value: value["assets"].pop(),
        }
        with tempfile.TemporaryDirectory() as raw:
            original = self.make_manifest(Path(raw))
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    candidate = copy.deepcopy(original)
                    mutate(candidate)
                    with self.assertRaises(MANIFEST.ManifestError):
                        MANIFEST.validate_manifest(candidate)

    def test_dist_extra_missing_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = "x86_64-unknown-linux-gnu"
            asset = release_assets(target=target)[0]
            path = root / asset["name"]
            path.write_bytes(b"archive")
            (root / "extra").write_bytes(b"no")
            with self.assertRaisesRegex(MANIFEST.ManifestError, "inventory mismatch"):
                MANIFEST.build_manifest(
                    repository="7xuanlu/wenlan",
                    repository_id=1,
                    run_id=1,
                    run_attempt=1,
                    pr_number=1,
                    pr_author="7xuanlu",
                    head_sha=HEAD_SHA,
                    tree_sha=TREE_SHA,
                    version="0.15.4",
                    target=target,
                    dist_dir=root,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
