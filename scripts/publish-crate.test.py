#!/usr/bin/env python3
"""Unit and mutation contracts for bounded crates.io publication."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("publish-crate.py")
SPEC = importlib.util.spec_from_file_location("publish_crate", SCRIPT)
assert SPEC and SPEC.loader
PUBLISH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PUBLISH
SPEC.loader.exec_module(PUBLISH)


class Result:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


class Probe:
    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, crate_name: str, version: str) -> bool:
        self.calls.append((crate_name, version))
        if not self.answers:
            raise AssertionError("probe called more times than expected")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return bool(answer)


def helper_contract_violations(source: str) -> list[str]:
    violations: list[str] = []
    for marker in (
        'command = [cargo_bin, "publish", "-p", crate_name]',
        "if publish_result.returncode != 0:",
        "return publish_result.returncode",
        "record[\"name\"] == crate_name and record[\"vers\"] == version",
    ):
        if marker not in source:
            violations.append(f"publish helper omits fail-closed contract {marker!r}")
    source_lines = set(source.splitlines())
    for marker in (
        "FAILURE_POLL_ATTEMPTS = 12",
        "SUCCESS_POLL_ATTEMPTS = 6",
        "POLL_INTERVAL_SECONDS = 5.0",
    ):
        if marker not in source_lines:
            violations.append(f"publish helper omits fail-closed contract {marker!r}")
    for forbidden in ("--no-verify", "--dry-run", "shell=True"):
        if forbidden in source:
            violations.append(f"publish helper bypasses safe Cargo behavior: {forbidden}")
    return violations


class SparseIndexTests(unittest.TestCase):
    def test_sparse_paths_follow_cargo_registry_layout(self) -> None:
        self.assertEqual(PUBLISH.sparse_index_path("a"), "1/a")
        self.assertEqual(PUBLISH.sparse_index_path("ab"), "2/ab")
        self.assertEqual(PUBLISH.sparse_index_path("AbC"), "3/a/abc")
        self.assertEqual(
            PUBLISH.sparse_index_path("wenlan-types"), "we/nl/wenlan-types"
        )

    def test_index_parser_requires_exact_name_and_version_fields(self) -> None:
        body = b"\n".join(
            json.dumps(record).encode()
            for record in (
                {"name": "wenlan-types", "vers": "1.2.30"},
                {"name": "other", "vers": "1.2.3"},
                {"name": "wenlan-types", "vers": "1.2.3"},
            )
        )
        self.assertTrue(PUBLISH.sparse_index_contains(body, "wenlan-types", "1.2.3"))
        self.assertFalse(
            PUBLISH.sparse_index_contains(body, "wenlan-types", "1.2.4")
        )

    def test_malformed_index_fails_closed(self) -> None:
        with self.assertRaisesRegex(PUBLISH.PublishError, "not valid JSON"):
            PUBLISH.sparse_index_contains(b"not-json\n", "wenlan-types", "1.2.3")


class PublishReconciliationTests(unittest.TestCase):
    def run_publish(self, probe: Probe, returncode: int = 0, **kwargs: object) -> tuple[int, list[list[str]]]:
        commands: list[list[str]] = []

        def runner(command: object) -> Result:
            commands.append(list(command))
            return Result(returncode)

        status = PUBLISH.publish_crate(
            "wenlan-types",
            "1.2.3",
            registry_token="secret",
            probe=probe,
            run_command=runner,
            sleeper=lambda _seconds: None,
            poll_interval=0,
            **kwargs,
        )
        return status, commands

    def test_exact_precheck_skips_publish(self) -> None:
        status, commands = self.run_publish(Probe(True))
        self.assertEqual(status, 0)
        self.assertEqual(commands, [])

    def test_missing_token_fails_only_when_exact_version_is_absent(self) -> None:
        with self.assertRaisesRegex(PUBLISH.PublishError, "CARGO_REGISTRY_TOKEN"):
            PUBLISH.publish_crate(
                "wenlan-types",
                "1.2.3",
                registry_token=None,
                probe=Probe(False),
            )

    def test_publish_keeps_cargo_packaged_crate_verification(self) -> None:
        status, commands = self.run_publish(Probe(False, True))
        self.assertEqual(status, 0)
        self.assertEqual(commands, [["cargo", "publish", "-p", "wenlan-types"]])
        self.assertNotIn("--no-verify", commands[0])

    def test_nonzero_publish_is_success_if_exact_version_appears(self) -> None:
        status, commands = self.run_publish(Probe(False, False, True), returncode=101)
        self.assertEqual(status, 0)
        self.assertEqual(len(commands), 1)

    def test_nonzero_publish_preserves_original_failure_after_bounded_poll(self) -> None:
        probe = Probe(False, False, False, False)
        status, _commands = self.run_publish(
            probe,
            returncode=73,
            failure_poll_attempts=3,
        )
        self.assertEqual(status, 73)
        self.assertEqual(len(probe.calls), 4)

    def test_success_uses_short_bounded_visibility_poll(self) -> None:
        probe = Probe(False, False, False)
        with self.assertRaisesRegex(PUBLISH.PublishError, "2 short sparse-index checks"):
            self.run_publish(probe, success_poll_attempts=2)
        self.assertEqual(len(probe.calls), 3)

    def test_precheck_transport_error_refuses_to_publish(self) -> None:
        commands: list[object] = []
        with self.assertRaisesRegex(PUBLISH.PublishError, "refusing to upload"):
            PUBLISH.publish_crate(
                "wenlan-types",
                "1.2.3",
                registry_token="secret",
                probe=Probe(PUBLISH.PublishError("offline")),
                run_command=lambda command: commands.append(command),
            )
        self.assertEqual(commands, [])


class PublishHelperMutationTests(unittest.TestCase):
    def test_helper_contract_rejects_safety_mutations(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(helper_contract_violations(source), [])
        mutations = (
            (
                'command = [cargo_bin, "publish", "-p", crate_name]',
                'command = [cargo_bin, "publish", "-p", crate_name, "--no-verify"]',
                "--no-verify",
            ),
            (
                "return publish_result.returncode",
                "return 0",
                "return publish_result.returncode",
            ),
            (
                'record["name"] == crate_name and record["vers"] == version',
                'record["name"] == crate_name and version in record["vers"]',
                "record[\"name\"] == crate_name and record[\"vers\"] == version",
            ),
            (
                "FAILURE_POLL_ATTEMPTS = 12",
                "FAILURE_POLL_ATTEMPTS = 120",
                "FAILURE_POLL_ATTEMPTS = 12",
            ),
        )
        for old, new, expected in mutations:
            with self.subTest(old=old):
                self.assertIn(old, source)
                violations = helper_contract_violations(source.replace(old, new, 1))
                self.assertTrue(any(expected in item for item in violations), violations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
