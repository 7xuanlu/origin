#!/usr/bin/env python3
"""Contract tests for the shared shipped release-target inventory."""

from __future__ import annotations

import unittest

from release_targets import TargetError, release_matrix, require_target


EXPECTED = [
    {
        "target": "aarch64-apple-darwin",
        "os": "macos-14",
        "archive": "tar.gz",
        "artifact_name": "wenlan-darwin-arm64",
    },
    {
        "target": "aarch64-unknown-linux-gnu",
        "os": "ubuntu-24.04-arm",
        "archive": "tar.gz",
        "artifact_name": "wenlan-linux-arm64",
    },
    {
        "target": "x86_64-unknown-linux-gnu",
        "os": "ubuntu-24.04",
        "archive": "tar.gz",
        "artifact_name": "wenlan-linux-x64",
    },
    {
        "target": "x86_64-pc-windows-msvc",
        "os": "windows-2022",
        "archive": "zip",
        "artifact_name": "wenlan-windows-x64",
    },
]


class ReleaseTargetTests(unittest.TestCase):
    def test_matrix_is_exactly_the_four_shipped_targets(self) -> None:
        self.assertEqual(release_matrix(), {"include": EXPECTED})

    def test_every_shipped_target_is_accepted(self) -> None:
        for entry in EXPECTED:
            with self.subTest(target=entry["target"]):
                self.assertEqual(require_target(entry["target"]), entry)

    def test_unknown_target_fails_closed(self) -> None:
        with self.assertRaisesRegex(TargetError, "not a shipped release target"):
            require_target("x86_64-apple-darwin")

    def test_callers_cannot_mutate_the_canonical_inventory(self) -> None:
        first = release_matrix()
        first["include"].clear()

        self.assertEqual(release_matrix(), {"include": EXPECTED})


if __name__ == "__main__":
    unittest.main(verbosity=2)
