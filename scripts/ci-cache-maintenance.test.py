#!/usr/bin/env python3
"""Unit tests for deterministic GitHub Actions cache pruning."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("ci-cache-maintenance.py")
SPEC = importlib.util.spec_from_file_location("ci_cache_maintenance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def cache(cache_id: int, size: int, *, key: str = "rust", accessed: int = 1) -> dict:
    timestamp = f"2026-08-{accessed:02d}T00:00:00Z"
    return {
        "id": cache_id,
        "key": key,
        "ref": "refs/heads/main",
        "size_in_bytes": size,
        "last_accessed_at": timestamp,
        "created_at": timestamp,
    }


class CacheMaintenanceTests(unittest.TestCase):
    def test_empty_repository_is_within_target(self) -> None:
        plan = MODULE.build_prune_plan([], usage_bytes=0, target_bytes=20)
        self.assertEqual(plan["status"], "within_target")

    def test_within_target_deletes_nothing(self) -> None:
        plan = MODULE.build_prune_plan(
            [cache(1, 10)], usage_bytes=10, target_bytes=20
        )
        self.assertEqual(plan["status"], "within_target")
        self.assertEqual(plan["deletions"], [])

    def test_oldest_entries_are_selected_until_target(self) -> None:
        plan = MODULE.build_prune_plan(
            [cache(3, 30, accessed=3), cache(1, 40, accessed=1), cache(2, 20, accessed=2)],
            usage_bytes=90,
            target_bytes=45,
        )
        self.assertEqual([row["id"] for row in plan["deletions"]], [1, 2])
        self.assertEqual(plan["projected_bytes"], 30)
        self.assertEqual(plan["status"], "planned")

    def test_fastembed_snapshot_is_protected(self) -> None:
        plan = MODULE.build_prune_plan(
            [
                cache(1, 40, key="fastembed-bge-base-en-v1.5-q-v3-portable"),
                cache(2, 70, accessed=2),
            ],
            usage_bytes=110,
            target_bytes=60,
        )
        self.assertEqual([row["id"] for row in plan["deletions"]], [2])
        self.assertEqual(plan["protected_bytes"], 40)

    def test_protected_inventory_can_fail_closed(self) -> None:
        plan = MODULE.build_prune_plan(
            [cache(1, 80, key="fastembed-bge-base-en-v1.5-q-v3-portable")],
            usage_bytes=80,
            target_bytes=50,
        )
        self.assertEqual(plan["status"], "insufficient_candidates")

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(MODULE.MaintenanceError, "duplicate ids"):
            MODULE.build_prune_plan(
                [cache(1, 20), cache(1, 20, accessed=2)],
                usage_bytes=40,
                target_bytes=30,
            )


if __name__ == "__main__":
    unittest.main()
