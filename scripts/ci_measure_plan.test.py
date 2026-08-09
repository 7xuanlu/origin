#!/usr/bin/env python3
"""Hermetic tests for the main measurement planner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("ci_measure_plan.py")
SPEC = importlib.util.spec_from_file_location("ci_measure_plan", MODULE_PATH)
assert SPEC and SPEC.loader
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)

VALIDATOR_MODULE_PATH = Path(__file__).with_name("validate-release-candidate.py")
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_release_candidate", VALIDATOR_MODULE_PATH
)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def release_files(version: str) -> dict[str, str]:
    values = {
        path: f"release version {version}\n" for path in PLANNER.RELEASE_MANAGED_PATHS
    }
    values["version.txt"] = f"{version}\n"
    values["CHANGELOG.md"] = "# Changelog\n"
    values[".release-please-manifest.json"] = json.dumps({".": version}) + "\n"
    values["Cargo.toml"] = f'[workspace.package]\nversion = "{version}"\n'
    values["Cargo.lock"] = f'version = "{version}"\n'
    values["crates/wenlan-cli/npm/package.json"] = (
        json.dumps({"name": "wenlan", "version": version}) + "\n"
    )
    values["crates/wenlan-mcp/npm/package.json"] = (
        json.dumps({"name": "wenlan-mcp", "version": version}) + "\n"
    )
    values["plugin/.claude-plugin/plugin.json"] = (
        json.dumps({"name": "wenlan", "version": version}) + "\n"
    )
    values["plugin-codex/.codex-plugin/plugin.json"] = (
        json.dumps({"name": "wenlan", "version": f"{version}+codex"}) + "\n"
    )
    return values


class Repository:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        run_git(self.root, "init", "-q")
        run_git(self.root, "config", "user.name", "CI Planner Test")
        run_git(self.root, "config", "user.email", "planner@example.invalid")
        self.write_many(release_files("0.15.3"))
        self.base = self.commit("base")

    def close(self) -> None:
        self.temporary.cleanup()

    def write(self, path: str, content: str) -> None:
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    def write_many(self, files: dict[str, str]) -> None:
        for path, content in files.items():
            self.write(path, content)

    def commit(self, message: str) -> str:
        run_git(self.root, "add", "-A")
        run_git(self.root, "commit", "-q", "-m", message)
        return run_git(self.root, "rev-parse", "HEAD")

    def change(self, path: str, content: str = "changed\n") -> str:
        self.write(path, content)
        return self.commit(f"change {path}")

    def plan(self, head: str, base: str | None = None) -> dict[str, object]:
        return PLANNER.build_plan(
            event_name="push",
            repository=self.root,
            base_sha=base or self.base,
            head_sha=head,
        )


class MeasurementPlanTests(unittest.TestCase):
    def make_repository(self) -> Repository:
        repository = Repository()
        self.addCleanup(repository.close)
        return repository

    def assert_routes(
        self, plan: dict[str, object], canary: bool, coverage: bool
    ) -> None:
        self.assertEqual(plan["main_canary_required"], canary)
        self.assertEqual(plan["coverage_required"], coverage)

    def test_manual_dispatch_always_runs_both_without_git_inputs(self) -> None:
        plan = PLANNER.build_plan(event_name="workflow_dispatch")
        self.assert_routes(plan, True, True)

    def test_exact_release_managed_next_patch_transform_skips_both(self) -> None:
        repository = self.make_repository()
        updated = release_files("0.15.4")
        updated["CHANGELOG.md"] = "# Changelog\n\n## [0.15.4] - current\n"
        repository.write_many(updated)
        head = repository.commit("release 0.15.4")
        plan = repository.plan(head)
        self.assert_routes(plan, False, False)
        self.assertIn("exact release-managed", plan["reason"])

    def test_partial_or_camouflaged_release_change_is_not_skipped(self) -> None:
        repository = self.make_repository()
        head = repository.change("version.txt", "0.15.4\n")
        self.assert_routes(repository.plan(head), True, True)

        repository = self.make_repository()
        updated = release_files("0.15.4")
        updated["CHANGELOG.md"] = "# rewritten\n\n## [0.15.4] - current\n"
        repository.write_many(updated)
        head = repository.commit("camouflaged release")
        self.assert_routes(repository.plan(head), True, True)

    def test_dependency_cargo_change_runs_both(self) -> None:
        repository = self.make_repository()
        head = repository.change(
            "Cargo.toml", '[workspace.package]\nversion = "0.15.3"\nresolver = "2"\n'
        )
        self.assert_routes(repository.plan(head), True, True)

    def test_planner_contract_change_runs_both(self) -> None:
        for path in ("scripts/ci_measure_plan.py", "scripts/ci_measure_plan.test.py"):
            with self.subTest(path=path):
                repository = self.make_repository()
                head = repository.change(path)
                self.assert_routes(repository.plan(head), True, True)

    def test_planner_change_runs_both_despite_scripts_nonowner_prefix(self) -> None:
        repository = self.make_repository()
        head = repository.change("scripts/ci_measure_plan.py")
        self.assert_routes(repository.plan(head), True, True)

    def test_core_types_and_eval_changes_run_both(self) -> None:
        for path in (
            "crates/wenlan-core/src/lib.rs",
            "crates/wenlan-types/src/lib.rs",
            "app/eval/fixtures/case.json",
        ):
            with self.subTest(path=path):
                repository = self.make_repository()
                head = repository.change(path)
                self.assert_routes(repository.plan(head), True, True)

    def test_server_change_runs_coverage_only(self) -> None:
        repository = self.make_repository()
        head = repository.change("crates/wenlan-server/src/main.rs")
        self.assert_routes(repository.plan(head), False, True)

    def test_each_workflow_owns_only_its_measurement(self) -> None:
        cases = {
            ".github/workflows/main-canary.yml": (True, False),
            ".github/workflows/coverage.yml": (False, True),
            ".config/nextest.toml": (True, False),
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                repository = self.make_repository()
                head = repository.change(path)
                self.assert_routes(repository.plan(head), *expected)

    def test_docs_and_plugin_are_explicit_nonowners(self) -> None:
        for path in (
            "docs/ci-routing.md",
            "README.md",
            "plugin/skills/recall/SKILL.md",
            "plugin-codex/README.txt",
            "crates/wenlan-mcp/src/main.rs",
            ".github/workflows/release.yml",
            "crates/wenlan-core/AGENTS.md",
            "crates/wenlan-core/src/drift_guard.rs",
        ):
            with self.subTest(path=path):
                repository = self.make_repository()
                head = repository.change(path)
                self.assert_routes(repository.plan(head), False, False)

    def test_unknown_path_fails_closed_to_both(self) -> None:
        repository = self.make_repository()
        head = repository.change("new-surface/config.bin")
        plan = repository.plan(head)
        self.assert_routes(plan, True, True)
        self.assertIn("unknown path fails closed", plan["reason"])

    def test_two_dot_diff_compares_exact_trees_not_merge_base(self) -> None:
        repository = self.make_repository()
        run_git(repository.root, "checkout", "-q", "-b", "side")
        side = repository.change("crates/wenlan-core/src/side.rs")
        run_git(repository.root, "checkout", "-q", "-b", "main-test", repository.base)
        main = repository.change("docs/main.md")

        # merge-base..main would contain docs only.  Exact side..main also
        # observes removal of side's core file, so the canary must run.
        self.assert_routes(repository.plan(main, base=side), True, True)

    def test_invalid_refs_and_malformed_diff_raise_from_helpers(self) -> None:
        repository = self.make_repository()
        with self.assertRaisesRegex(PLANNER.PlanError, "full lowercase commit SHA"):
            PLANNER.build_plan(
                event_name="push",
                repository=repository.root,
                base_sha="deadbeef",
                head_sha=repository.base,
            )
        with self.assertRaises(PLANNER.PlanError):
            PLANNER.build_plan(
                event_name="push",
                repository=repository.root,
                base_sha="0" * 40,
                head_sha=repository.base,
            )
        with self.assertRaisesRegex(PLANNER.PlanError, "NUL terminated"):
            PLANNER.parse_name_status_z(b"M\0Cargo.toml")
        with self.assertRaisesRegex(PLANNER.PlanError, "invalid or duplicate path"):
            PLANNER.parse_name_status_z(b"M\0bad\npath\0")
        with self.assertRaisesRegex(PLANNER.PlanError, "unsupported event"):
            PLANNER.build_plan(event_name="pull_request")

    def test_cli_turns_invalid_input_into_true_outputs(self) -> None:
        repository = self.make_repository()
        output = repository.root / "github-output.txt"
        result = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--event-name",
                "push",
                "--repository",
                str(repository.root),
                "--before",
                "deadbeef",
                "--head",
                repository.base,
                "--github-output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("planner failure fails closed", result.stdout)
        lines = output.read_text(encoding="utf-8").splitlines()
        self.assertIn("main-canary-required=true", lines)
        self.assertIn("coverage-required=true", lines)
        self.assertTrue(any(line.startswith("reason=") for line in lines))


class ReleaseManagedPathsConsistencyTests(unittest.TestCase):
    """A stale copy of RELEASE_MANAGED_PATHS must not silently drift.

    Unlike verify-release-merge.py (which only rejects unexpected paths, so a
    superset is safe), _is_exact_release_transform above requires
    ``set(records) == {("M", path) for path in RELEASE_MANAGED_PATHS}``: this
    copy must name the *exact* file set a genuine release-please PR touches, so
    the contract with the canonical validate-release-candidate.py list is
    equality, not superset.
    """

    def test_release_managed_paths_equals_validator_required_paths(self) -> None:
        self.assertEqual(
            PLANNER.RELEASE_MANAGED_PATHS,
            VALIDATOR.REQUIRED_RELEASE_PATHS,
            "ci_measure_plan.py RELEASE_MANAGED_PATHS has drifted from "
            "validate-release-candidate.py REQUIRED_RELEASE_PATHS: "
            f"missing={sorted(VALIDATOR.REQUIRED_RELEASE_PATHS - PLANNER.RELEASE_MANAGED_PATHS)} "
            f"extra={sorted(PLANNER.RELEASE_MANAGED_PATHS - VALIDATOR.REQUIRED_RELEASE_PATHS)}",
        )


if __name__ == "__main__":
    unittest.main()
