#!/usr/bin/env python3
"""Contract tests for the shared shipped release-target inventory."""

from __future__ import annotations

import unittest

from release_targets import (
    TargetError,
    release_assets,
    release_matrix,
    release_preflight_matrix,
    require_target,
)


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

EXPECTED_ASSET_NAMES = [
    "wenlan-darwin-arm64.tar.gz",
    "wenlan-cli-darwin-arm64.tar.gz",
    "wenlan-mcp-darwin-arm64.tar.gz",
    "wenlan-linux-arm64.tar.gz",
    "wenlan-linux-x64.tar.gz",
    "wenlan-windows-x64.zip",
]


class ReleaseTargetTests(unittest.TestCase):
    def test_matrix_is_exactly_the_four_shipped_targets(self) -> None:
        self.assertEqual(release_matrix(), {"include": EXPECTED})

    def test_every_shipped_target_is_accepted(self) -> None:
        for entry in EXPECTED:
            with self.subTest(target=entry["target"]):
                self.assertEqual(require_target(entry["target"]), entry)

    def test_pr_matrix_can_exclude_only_the_slow_windows_release_profile(self) -> None:
        self.assertEqual(
            release_matrix(exclude_targets=["x86_64-pc-windows-msvc"]),
            {"include": EXPECTED[:-1]},
        )

    def test_ordinary_pr_preflight_excludes_windows(self) -> None:
        self.assertEqual(
            release_preflight_matrix(
                event_name="pull_request",
                ref="refs/pull/431/merge",
                head_ref="codex/promote-release-artifacts",
            ),
            {"include": EXPECTED[:-1]},
        )

    def test_ordinary_main_preflight_only_warms_windows(self) -> None:
        self.assertEqual(
            release_preflight_matrix(
                event_name="push",
                ref="refs/heads/main",
                head_ref="",
            ),
            {"include": EXPECTED[-1:]},
        )

    def test_release_please_and_manual_preflight_keep_every_target(self) -> None:
        release_please = release_preflight_matrix(
            event_name="pull_request",
            ref="refs/pull/430/merge",
            head_ref="release-please--branches--main",
        )
        manual = release_preflight_matrix(
            event_name="workflow_dispatch",
            ref="refs/heads/main",
            head_ref="",
        )

        self.assertEqual(release_please, {"include": EXPECTED})
        self.assertEqual(manual, {"include": EXPECTED})

    def test_preflight_matrix_rejects_unowned_events_and_push_refs(self) -> None:
        with self.assertRaisesRegex(TargetError, "does not accept event"):
            release_preflight_matrix(
                event_name="schedule",
                ref="refs/heads/main",
                head_ref="",
            )
        with self.assertRaisesRegex(TargetError, "does not accept push ref"):
            release_preflight_matrix(
                event_name="push",
                ref="refs/heads/not-main",
                head_ref="",
            )

    def test_matrix_rejects_unknown_exclusions(self) -> None:
        with self.assertRaisesRegex(TargetError, "unknown shipped targets"):
            release_matrix(exclude_targets=["x86_64-apple-darwin"])

    def test_unknown_target_fails_closed(self) -> None:
        with self.assertRaisesRegex(TargetError, "not a shipped release target"):
            require_target("x86_64-apple-darwin")

    def test_callers_cannot_mutate_the_canonical_inventory(self) -> None:
        first = release_matrix()
        first["include"].clear()

        assets = release_assets()
        assets[0]["members"].clear()

        self.assertEqual(release_matrix(), {"include": EXPECTED})
        self.assertEqual(
            [asset["name"] for asset in release_assets()], EXPECTED_ASSET_NAMES
        )

    def test_asset_inventory_is_exact_unique_and_target_owned(self) -> None:
        assets = release_assets()
        self.assertEqual([asset["name"] for asset in assets], EXPECTED_ASSET_NAMES)
        self.assertEqual(len({asset["name"] for asset in assets}), 6)
        self.assertEqual(
            [asset["name"] for asset in release_assets(target="aarch64-apple-darwin")],
            EXPECTED_ASSET_NAMES[:3],
        )
        for target in [entry["target"] for entry in EXPECTED[1:]]:
            with self.subTest(target=target):
                owned = release_assets(target=target)
                self.assertEqual(len(owned), 1)
                self.assertEqual(owned[0]["target"], target)

    def test_asset_inventory_rejects_unknown_target(self) -> None:
        with self.assertRaisesRegex(TargetError, "not a shipped release target"):
            release_assets(target="x86_64-apple-darwin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
