#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "ci-observer.py"
SHA = "a" * 40


def event():
    return {
        "workflow_run": {
            "id": 9001,
            "run_attempt": 2,
            "name": "CI",
            "status": "completed",
            "conclusion": "success",
            "head_sha": SHA,
            "run_started_at": "2026-07-24T01:00:00Z",
            "updated_at": "2026-07-24T01:21:00Z",
        }
    }


def job(job_id=101, name="test ($(whoami))"):
    return {
        "id": job_id,
        "run_id": 9001,
        "run_attempt": 2,
        "name": name,
        "started_at": "2026-07-24T01:01:00Z",
        "completed_at": "2026-07-24T01:20:00Z",
        "conclusion": "success",
        "runner_name": "GitHub Actions 7",
        "runner_group_name": "GitHub Actions",
        "labels": ["ubuntu-24.04"],
        "steps": [
            {
                "number": 1,
                "name": "checkout `malicious`",
                "started_at": "2026-07-24T01:01:00Z",
                "completed_at": "2026-07-24T01:01:03Z",
                "conclusion": "success",
            }
        ],
    }


class CiObserverTests(unittest.TestCase):
    def run_observer(self, event_payload, pages, usage, limit):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        inputs = {
            "event": event_payload,
            "jobs-pages": pages,
            "cache-usage": usage,
            "cache-limit": limit,
        }
        paths = {}
        for name, payload in inputs.items():
            path = root / f"{name}.json"
            path.write_text(json.dumps(payload))
            paths[name] = path
        output = root / "receipt.json"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--event",
                str(paths["event"]),
                "--jobs-pages",
                str(paths["jobs-pages"]),
                "--cache-usage",
                str(paths["cache-usage"]),
                "--cache-limit",
                str(paths["cache-limit"]),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        payload = json.loads(output.read_text()) if output.exists() else None
        return result, payload

    def assert_error_receipt(self, receipt, message):
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["schema_version"], 1)
        self.assertEqual(receipt["status"], "invalid")
        self.assertIn(message, receipt["error"]["message"])
        self.assertEqual(
            set(receipt["raw_refs"]),
            {"event", "jobs_pages", "cache_usage", "cache_limit"},
        )
        for raw_ref in receipt["raw_refs"].values():
            self.assertTrue(raw_ref["exists"])
            self.assertGreater(raw_ref["size_bytes"], 0)

    def test_stable_receipt_preserves_untrusted_names_as_json_data(self):
        conclusion = job(102, "conclusion")
        conclusion["completed_at"] = "2026-07-24T01:20:30Z"
        result, receipt = self.run_observer(
            event(),
            [
                {"total_count": 2, "jobs": [conclusion]},
                {"total_count": 2, "jobs": [job()]},
            ],
            {"active_caches_size_in_bytes": 9_000_000_000, "active_caches_count": 20},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["observed_run"]["head_sha"], SHA)
        self.assertEqual(receipt["jobs"][1]["name"], "test ($(whoami))")
        self.assertEqual(receipt["jobs"][1]["duration_ms"], 1_140_000)
        self.assertEqual(receipt["required_gate_elapsed_ms"], 1_230_000)
        self.assertEqual(receipt["cache"]["status"], "under_budget")

    def test_required_gate_target_is_strictly_under_twenty_minutes(self):
        conclusion = job(102, "conclusion")
        conclusion["completed_at"] = "2026-07-24T01:19:59Z"
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [conclusion]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["required_gate_target_ms"], 1_200_000)
        self.assertTrue(receipt["required_gate_target_met"])

        conclusion["completed_at"] = "2026-07-24T01:20:00Z"
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [conclusion]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["required_gate_target_ms"], 1_200_000)
        self.assertFalse(receipt["required_gate_target_met"])

    def test_normal_eviction_pressure_is_recorded_without_failing(self):
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {
                "active_caches_size_in_bytes": 10_000_000_001,
                "active_caches_count": 21,
            },
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["cache"]["over_by_bytes"], 1)
        self.assertEqual(receipt["cache"]["status"], "over_budget")
        self.assertFalse(receipt["cache"]["alert"])

    def test_sustained_cache_overshoot_writes_receipt_then_exits_two(self):
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {
                "active_caches_size_in_bytes": 11_500_000_001,
                "active_caches_count": 21,
            },
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(receipt["cache"]["status"], "over_budget")
        self.assertTrue(receipt["cache"]["alert"])

    def test_rejects_short_sha_and_attempt_mismatch(self):
        bad_event = event()
        bad_event["workflow_run"]["head_sha"] = "abc123"
        result, receipt = self.run_observer(
            bad_event,
            [{"total_count": 1, "jobs": [job()]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 1)
        self.assert_error_receipt(receipt, "head SHA")

        mismatched = job()
        mismatched["run_attempt"] = 3
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [mismatched]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 1)
        self.assert_error_receipt(receipt, "different run or attempt")

    def test_rejects_duplicate_jobs_and_inverted_timestamps(self):
        duplicate = job()
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 2, "jobs": [duplicate, duplicate]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 1)
        self.assert_error_receipt(receipt, "unique integers")

        inverted = job()
        inverted["completed_at"] = "2026-07-24T00:59:00Z"
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [inverted]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 1)
        self.assert_error_receipt(receipt, "completion timestamp precedes")

    def test_invalid_json_shapes_write_error_receipts(self):
        result, receipt = self.run_observer(
            [],
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 1)
        self.assert_error_receipt(receipt, "event payload must be a JSON object")

        result, receipt = self.run_observer(
            event(),
            [42],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 1)
        self.assert_error_receipt(receipt, "jobs page must be a JSON object")

    def test_clamps_github_clock_skew_on_skipped_jobs(self):
        skipped = job()
        skipped["name"] = "skipped"
        skipped["conclusion"] = "skipped"
        skipped["started_at"] = "2026-07-24T01:01:18Z"
        skipped["completed_at"] = "2026-07-24T01:00:17Z"
        conclusion = job(102, "conclusion")
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 2, "jobs": [skipped, conclusion]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            {"max_cache_size_gb": 10},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        skipped_receipt = next(job for job in receipt["jobs"] if job["name"] == "skipped")
        self.assertEqual(skipped_receipt["duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()
