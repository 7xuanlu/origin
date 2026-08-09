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
            "event": "pull_request",
            "head_branch": "fix/ordinary-pr",
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
    def run_observer(
        self, event_payload, pages, usage, budget="10", gate_target="20", missing=()
    ):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        inputs = {
            "event": event_payload,
            "jobs-pages": pages,
            "cache-usage": usage,
        }
        paths = {}
        for name, payload in inputs.items():
            path = root / f"{name}.json"
            paths[name] = path
            if name not in missing:
                path.write_text(json.dumps(payload))
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
                "--cache-budget-gb",
                str(budget),
                "--required-gate-target-minutes",
                str(gate_target),
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
        self.assertEqual(receipt["schema_version"], 2)
        self.assertEqual(receipt["status"], "invalid")
        self.assertIn(message, receipt["error"]["message"])
        self.assertEqual(
            set(receipt["raw_refs"]),
            {"event", "jobs_pages", "cache_usage"},
        )
        self.assertEqual(receipt["cache_budget"]["source"], "repository_policy")
        self.assertIn("configured_minutes", receipt["required_gate_target"])

    def test_stable_receipt_preserves_untrusted_names_as_json_data(self):
        conclusion = job(102, "conclusion")
        conclusion["completed_at"] = "2026-07-24T01:19:30Z"
        result, receipt = self.run_observer(
            event(),
            [
                {"total_count": 2, "jobs": [conclusion]},
                {"total_count": 2, "jobs": [job()]},
            ],
            {"active_caches_size_in_bytes": 9_000_000_000, "active_caches_count": 20},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["schema_version"], 2)
        self.assertEqual(receipt["observed_run"]["head_sha"], SHA)
        self.assertEqual(receipt["jobs"][1]["name"], "test ($(whoami))")
        self.assertEqual(receipt["jobs"][1]["duration_ms"], 1_140_000)
        self.assertEqual(receipt["required_gate_elapsed_ms"], 1_170_000)
        self.assertEqual(receipt["required_gate_target"]["status"], "within_target")
        self.assertEqual(receipt["required_gate_target"]["minutes"], 20)
        self.assertTrue(receipt["required_gate_target"]["applicable"])
        self.assertEqual(
            receipt["required_gate_target"]["scope"], "ordinary_pull_request"
        )
        self.assertEqual(receipt["required_gate_target"]["over_by_ms"], 0)
        self.assertEqual(receipt["cache"]["status"], "under_budget")
        self.assertEqual(receipt["cache"]["budget_gb"], 10)
        self.assertEqual(receipt["cache"]["budget_source"], "repository_policy")

    def test_normal_eviction_pressure_is_recorded_without_failing(self):
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {
                "active_caches_size_in_bytes": 10_000_000_001,
                "active_caches_count": 21,
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["cache"]["over_by_bytes"], 1)
        self.assertEqual(receipt["cache"]["status"], "over_budget")
        self.assertFalse(receipt["cache"]["alert"])
        self.assertIn("::warning title=CI cache budget exceeded::", result.stdout)

    def test_required_gate_over_target_warns_without_gating(self):
        conclusion = job(102, "conclusion")
        conclusion["completed_at"] = "2026-07-24T01:20:30Z"
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [conclusion]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["required_gate_elapsed_ms"], 1_230_000)
        self.assertEqual(receipt["required_gate_target"]["status"], "over_target")
        self.assertEqual(receipt["required_gate_target"]["over_by_ms"], 30_000)
        self.assertIn(
            "::warning title=CI required gate target exceeded::", result.stdout
        )

    def test_release_preflight_pr_is_not_an_ordinary_pr_slo_sample(self):
        conclusion = job(102, "conclusion")
        conclusion["completed_at"] = "2026-07-24T01:20:30Z"
        result, receipt = self.run_observer(
            event(),
            [
                {
                    "total_count": 2,
                    "jobs": [
                        conclusion,
                        job(103, "release-preflight (x86_64-pc-windows-msvc)"),
                    ],
                }
            ],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(receipt["required_gate_target"]["applicable"])
        self.assertEqual(receipt["required_gate_target"]["status"], "not_applicable")
        self.assertEqual(
            receipt["required_gate_target"]["exempt_jobs"],
            ["release-preflight (x86_64-pc-windows-msvc)"],
        )
        self.assertNotIn("CI required gate target exceeded", result.stdout)

    def test_skipped_release_preflight_keeps_ordinary_pr_in_slo_sample(self):
        conclusion = job(102, "conclusion")
        conclusion["completed_at"] = "2026-07-24T01:19:30Z"
        skipped_preflight = job(
            103, "release-preflight (x86_64-pc-windows-msvc)"
        )
        skipped_preflight["conclusion"] = "skipped"
        result, receipt = self.run_observer(
            event(),
            [
                {
                    "total_count": 2,
                    "jobs": [conclusion, skipped_preflight],
                }
            ],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(receipt["required_gate_target"]["applicable"])
        self.assertEqual(receipt["required_gate_target"]["status"], "within_target")
        self.assertEqual(receipt["required_gate_target"]["exempt_jobs"], [])

    def test_main_and_release_please_runs_are_not_ordinary_pr_samples(self):
        for run_event, head_branch in (
            ("push", "main"),
            ("pull_request", "release-please--branches--main"),
        ):
            with self.subTest(run_event=run_event, head_branch=head_branch):
                payload = event()
                payload["workflow_run"]["event"] = run_event
                payload["workflow_run"]["head_branch"] = head_branch
                result, receipt = self.run_observer(
                    payload,
                    [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
                    {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(receipt["required_gate_target"]["applicable"])
                self.assertEqual(
                    receipt["required_gate_target"]["status"], "not_applicable"
                )

    def test_large_cache_overshoot_warns_without_failing(self):
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {
                "active_caches_size_in_bytes": 11_500_000_001,
                "active_caches_count": 21,
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["cache"]["status"], "over_budget")
        self.assertTrue(receipt["cache"]["alert"])
        self.assertIn("::warning title=CI cache budget exceeded::", result.stdout)

        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {
                "active_caches_size_in_bytes": 11_500_000_000,
                "active_caches_count": 21,
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(receipt["cache"]["alert"])

    def test_cancelled_run_may_have_no_conclusion_job(self):
        cancelled = event()
        cancelled["workflow_run"]["conclusion"] = "cancelled"
        result, receipt = self.run_observer(
            cancelled,
            [{"total_count": 0, "jobs": []}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["observed_run"]["conclusion"], "cancelled")
        self.assertIsNone(receipt["required_gate_elapsed_ms"])
        self.assertEqual(receipt["required_gate_target"]["status"], "unavailable")

    def test_non_cancelled_run_without_conclusion_fails_closed(self):
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [job()]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_error_receipt(receipt, "exactly one required conclusion")
        self.assertIn("::warning title=CI observer invalid input::", result.stdout)

    def test_rejects_short_sha_and_attempt_mismatch(self):
        bad_event = event()
        bad_event["workflow_run"]["head_sha"] = "abc123"
        result, receipt = self.run_observer(
            bad_event,
            [{"total_count": 1, "jobs": [job()]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_error_receipt(receipt, "head SHA")

        mismatched = job()
        mismatched["run_attempt"] = 3
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [mismatched]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_error_receipt(receipt, "different run or attempt")

    def test_rejects_duplicate_jobs_and_inverted_timestamps(self):
        duplicate = job()
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 2, "jobs": [duplicate, duplicate]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_error_receipt(receipt, "unique integers")

        inverted = job()
        inverted["completed_at"] = "2026-07-24T00:59:00Z"
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [inverted]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_error_receipt(receipt, "completion timestamp precedes")

    def test_invalid_json_shapes_write_error_receipts(self):
        result, receipt = self.run_observer(
            [],
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_error_receipt(receipt, "event payload must be a JSON object")

        result, receipt = self.run_observer(
            event(),
            [42],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_error_receipt(receipt, "jobs page must be a JSON object")

    def test_missing_metadata_writes_an_error_receipt_and_exits_zero(self):
        result, receipt = self.run_observer(
            event(),
            [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
            {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
            missing={"cache-usage"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(receipt["error"]["type"], "FileNotFoundError")
        self.assertFalse(receipt["raw_refs"]["cache_usage"]["exists"])
        self.assertIn("::warning title=CI observer invalid input::", result.stdout)

    def test_rejects_malformed_cache_usage(self):
        for usage, message in (
            ({"active_caches_count": 1}, "active cache size"),
            (
                {"active_caches_size_in_bytes": "1", "active_caches_count": 1},
                "active cache size",
            ),
            (
                {"active_caches_size_in_bytes": 1, "active_caches_count": True},
                "active cache count",
            ),
        ):
            with self.subTest(usage=usage):
                result, receipt = self.run_observer(
                    event(),
                    [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
                    usage,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_error_receipt(receipt, message)

    def test_rejects_non_positive_or_non_integer_budget(self):
        for budget in ("0", "-1", "1.5", "ten"):
            with self.subTest(budget=budget):
                result, receipt = self.run_observer(
                    event(),
                    [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
                    {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
                    budget=budget,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_error_receipt(receipt, "cache budget in GB must be a positive integer")

    def test_rejects_non_positive_or_non_integer_gate_target(self):
        for gate_target in ("0", "-1", "1.5", "twenty"):
            with self.subTest(gate_target=gate_target):
                result, receipt = self.run_observer(
                    event(),
                    [{"total_count": 1, "jobs": [job(102, "conclusion")]}],
                    {"active_caches_size_in_bytes": 1, "active_caches_count": 1},
                    gate_target=gate_target,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_error_receipt(
                    receipt, "required gate target minutes must be a positive integer"
                )

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
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        skipped_receipt = next(job for job in receipt["jobs"] if job["name"] == "skipped")
        self.assertEqual(skipped_receipt["duration_ms"], 0)


if __name__ == "__main__":
    unittest.main()
