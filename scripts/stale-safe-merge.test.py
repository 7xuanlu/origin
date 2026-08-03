#!/usr/bin/env python3
"""Contract tests for the stale-but-safe merge decision."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("stale-safe-merge.py")


def load_script():
    spec = importlib.util.spec_from_file_location("stale_safe_merge", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaleSafeMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()

    def pull(self, **overrides) -> dict:
        base = {
            "number": 123,
            "mergeable": "MERGEABLE",
            "mergeStateStatus": "BEHIND",
            "headRefOid": "cb32417a55c13991b226fca6e5d4a4f1dadb40cb",
            "isDraft": False,
            "labels": [{"name": self.module.OPT_IN_LABEL}],
            "files": ["scripts/only_mine.py"],
        }
        return base | overrides

    def test_green_and_non_overlapping_is_safe(self) -> None:
        safe, reason = self.module.decide(
            self.pull(), {"docs/unrelated.md"}, "success"
        )
        self.assertTrue(safe, reason)

    def test_overlap_is_never_safe(self) -> None:
        safe, reason = self.module.decide(
            self.pull(), {"scripts/only_mine.py"}, "success"
        )
        self.assertFalse(safe)
        self.assertIn("overlaps", reason)

    def test_opt_in_label_is_required(self) -> None:
        safe, reason = self.module.decide(
            self.pull(labels=[{"name": "other"}]), {"docs/x.md"}, "success"
        )
        self.assertFalse(safe)
        self.assertIn("not opted in", reason)

    def test_only_a_verified_green_head_may_be_bypassed(self) -> None:
        # The merge bypasses required checks, so a non-success conclusion must
        # stop it here — nothing downstream will.
        for conclusion in ["failure", "cancelled", "pending", "", "neutral"]:
            with self.subTest(conclusion=conclusion):
                safe, reason = self.module.decide(
                    self.pull(), {"docs/x.md"}, conclusion
                )
                self.assertFalse(safe)
                self.assertIn("not success", reason)

    def test_only_behind_and_mergeable_qualifies(self) -> None:
        # Every other state means something is genuinely wrong or unknown.
        for status in ["BLOCKED", "DIRTY", "UNSTABLE", "UNKNOWN", "CLEAN"]:
            with self.subTest(status=status):
                safe, _ = self.module.decide(
                    self.pull(mergeStateStatus=status), {"docs/x.md"}, "success"
                )
                self.assertFalse(safe)
        for mergeable in ["CONFLICTING", "UNKNOWN"]:
            with self.subTest(mergeable=mergeable):
                safe, _ = self.module.decide(
                    self.pull(mergeable=mergeable), {"docs/x.md"}, "success"
                )
                self.assertFalse(safe)

    def test_draft_is_never_safe(self) -> None:
        safe, _ = self.module.decide(
            self.pull(isDraft=True), {"docs/x.md"}, "success"
        )
        self.assertFalse(safe)

    def test_empty_file_list_is_not_treated_as_no_overlap(self) -> None:
        # An empty changed-file list would otherwise intersect to empty and
        # read as "safe" on missing data rather than on evidence.
        safe, reason = self.module.decide(
            self.pull(files=[]), {"docs/x.md"}, "success"
        )
        self.assertFalse(safe)
        self.assertIn("no changed files", reason)

    def test_malformed_payload_raises_rather_than_defaulting(self) -> None:
        for missing in ["number", "mergeable", "mergeStateStatus", "headRefOid"]:
            payload = self.pull()
            del payload[missing]
            with self.subTest(missing=missing), self.assertRaises(
                self.module.StaleMergeError
            ):
                self.module.decide(payload, {"docs/x.md"}, "success")


if __name__ == "__main__":
    unittest.main()
