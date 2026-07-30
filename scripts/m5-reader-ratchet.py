#!/usr/bin/env python3
"""R6 semantic ratchet around M5 reader-inventory regeneration.

Line addresses may change while post_write items move.  The semantic summary
and the exact exposure identity set may not.  ``--capture`` records the live
pre-move receipt; ``--check`` first runs the existing generator check and then
compares both the pinned R6 boundary and an optional captured receipt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "scripts/m5-reader-sweep.py"
PINNED_SUMMARY = {"rows": 191, "depths": {"0": 55, "1": 50, "2": 86}, "exposure": 22}
PINNED_EXPOSURES = {
    ("crates/wenlan-core/src/db.rs", "reconcile_entity_page_parity", 0),
    ("crates/wenlan-core/src/db.rs", "list_recent_retrievals_scoped", 0),
    ("crates/wenlan-core/src/db.rs", "load_page_source_index", 0),
    ("crates/wenlan-core/src/db.rs", "list_stale_pages_scoped", 0),
    ("crates/wenlan-core/src/db.rs", "find_stale_archived_pages", 0),
    ("crates/wenlan-core/src/db/scoped_entities.rs", "list_entities_scoped", 0),
    ("crates/wenlan-core/src/db/scoped_entities.rs", "get_entity_detail_scoped", 0),
    ("crates/wenlan-core/src/db/scoped_entities.rs", "list_recent_relations_scoped", 0),
    ("crates/wenlan-core/src/db/scoped_entities.rs", "search_entities_by_vector_scoped", 0),
    ("crates/wenlan-core/src/db/scoped_pages.rs", "list_recent_changes_scoped", 0),
    ("crates/wenlan-core/src/db.rs", "get_page", 1),
    ("crates/wenlan-core/src/db.rs", "list_pages", 1),
    ("crates/wenlan-core/src/db.rs", "resolve_orphan_page_links", 1),
    (
        "crates/wenlan-core/src/db/scoped_pages.rs",
        "list_recent_pages_with_badges_scoped",
        1,
    ),
    ("crates/wenlan-core/src/maintenance.rs", "run_maintenance_stage_slice", 1),
    ("crates/wenlan-core/src/post_ingest.rs", "run_page_growth_slice", 1),
    (
        "crates/wenlan-core/src/repair.rs",
        "prepare_memory_reclassification_with_pages",
        1,
    ),
    ("crates/wenlan-core/src/db.rs", "rebind_source_id_with_source_page", 2),
    ("crates/wenlan-core/src/page_map_improve.rs", "improve_page_map", 2),
    (
        "crates/wenlan-core/src/refinery/mod.rs",
        "run_periodic_steep_phase_with_api",
        2,
    ),
    ("crates/wenlan-core/src/repair.rs", "apply_repair_with_pages", 2),
    (
        "crates/wenlan-core/src/synthesis/refinement_queue.rs",
        "apply_refinement_with_decision",
        2,
    ),
}


def load_sweep():
    spec = importlib.util.spec_from_file_location("m5_reader_sweep", SWEEP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    depths = Counter(row["depth"] for row in rows)
    exposures = sorted(
        [row["file"], row["fn"], row["depth"]]
        for row in rows
        if row["exposure"]
    )
    return {
        "summary": {
            "rows": len(rows),
            "depths": {str(depth): depths[depth] for depth in (0, 1, 2)},
            "exposure": len(exposures),
        },
        "exposures": exposures,
    }


def expected_receipt() -> dict[str, Any]:
    return {
        "summary": PINNED_SUMMARY,
        "exposures": sorted([path, name, depth] for path, name, depth in PINNED_EXPOSURES),
    }


def receipt_violations(
    actual: dict[str, Any], expected: dict[str, Any], label: str
) -> list[str]:
    violations = []
    if actual["summary"] != expected["summary"]:
        violations.append(
            f"{label} summary changed: {actual['summary']!r} != {expected['summary']!r}"
        )
    actual_set = {tuple(item) for item in actual["exposures"]}
    expected_set = {tuple(item) for item in expected["exposures"]}
    if actual_set != expected_set:
        violations.append(
            f"{label} exposure set changed: "
            f"missing={sorted(expected_set - actual_set)!r} "
            f"added={sorted(actual_set - expected_set)!r}"
        )
    return violations


def live_receipt(check_inventory: bool) -> dict[str, Any]:
    sweep = load_sweep()
    rows = sweep.sweep()
    if sweep.TRUNCATED:
        raise RuntimeError(
            "M5 reader brace scan lost sync: " + ", ".join(sweep.TRUNCATED)
        )
    if check_inventory and not sweep.check_inventory(rows):
        raise RuntimeError("existing M5 generated inventory check failed")
    return receipt(rows)


def selftest() -> None:
    expected = expected_receipt()
    assert not receipt_violations(expected, expected, "control")

    summary_mutation = json.loads(json.dumps(expected))
    summary_mutation["summary"]["rows"] -= 1
    assert any(
        "summary changed" in item
        for item in receipt_violations(summary_mutation, expected, "synthetic")
    ), "a synthetic changed summary must fail"

    exposure_mutation = json.loads(json.dumps(expected))
    exposure_mutation["exposures"][0][1] = "different_reader"
    assert any(
        "exposure set changed" in item
        for item in receipt_violations(exposure_mutation, expected, "synthetic")
    ), "a synthetic changed exposure identity must fail"
    print("M5 reader ratchet selftest: ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--capture", type=Path)
    action.add_argument("--check", action="store_true")
    action.add_argument("--self-test", action="store_true")
    parser.add_argument("--against", type=Path)
    args = parser.parse_args()

    if args.self_test:
        selftest()
        return 0

    try:
        actual = live_receipt(check_inventory=True)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    violations = receipt_violations(actual, expected_receipt(), "pinned R6")
    if args.against:
        captured = json.loads(args.against.read_text())
        violations.extend(receipt_violations(actual, captured, "captured pre-move"))
    if violations:
        print("M5 R6 ratchet failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    if args.capture:
        args.capture.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"captured M5 R6 ratchet receipt: {args.capture}")
        return 0
    print("M5 R6 ratchet: ok (191 rows; depth 55/50/86; exposure 22)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
