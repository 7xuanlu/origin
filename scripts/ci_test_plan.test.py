#!/usr/bin/env python3
"""Contract tests for fail-closed differential Rust test planning."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ci_test_plan import PlanError, build_plan, command_groups_for


WORKSPACE_ROOT = "/repo"


def package(
    name: str,
    directory: str,
    dependencies: tuple[str, ...],
    integration_targets: tuple[str, ...] = (),
) -> dict:
    targets = [
        {
            "name": name.replace("-", "_"),
            "kind": ["lib"],
            "src_path": f"{WORKSPACE_ROOT}/{directory}/src/lib.rs",
        }
    ]
    targets.extend(
        {
            "name": target,
            "kind": ["test"],
            "src_path": f"{WORKSPACE_ROOT}/{directory}/tests/{target}.rs",
        }
        for target in integration_targets
    )
    return {
        "name": name,
        "manifest_path": f"{WORKSPACE_ROOT}/{directory}/Cargo.toml",
        "dependencies": [
            {
                "name": dependency,
                "path": f"{WORKSPACE_ROOT}/crates/{dependency}",
            }
            for dependency in dependencies
        ],
        "targets": targets,
    }


def cargo_metadata() -> dict:
    return {
        "workspace_root": WORKSPACE_ROOT,
        "workspace_members": [
            "wenlan",
            "wenlan-core",
            "wenlan-types",
            "wenlan-server",
            "wenlan-mcp",
        ],
        "packages": [
            package(
                "wenlan",
                "crates/wenlan-cli",
                ("wenlan-core", "wenlan-types"),
                ("cli_integration", "distribution"),
            ),
            package(
                "wenlan-core",
                "crates/wenlan-core",
                ("wenlan-types",),
                ("eval_harness", "folder_ingest_e2e", "read_scope"),
            ),
            package("wenlan-types", "crates/wenlan-types", ()),
            package(
                "wenlan-server",
                "crates/wenlan-server",
                ("wenlan-core", "wenlan-types"),
                ("graceful_shutdown", "space_scoping_e2e"),
            ),
            package(
                "wenlan-mcp",
                "crates/wenlan-mcp",
                ("wenlan-core", "wenlan-server", "wenlan-types"),
                ("real_router",),
            ),
        ],
    }


def plan_for(*paths: str, existing_paths: set[str] | None = None) -> dict:
    return build_plan(
        list(paths),
        cargo_metadata(),
        event_name="pull_request",
        existing_paths=set(paths) if existing_paths is None else existing_paths,
    )


class FiltersetExecutionTests(unittest.TestCase):
    def run_filterset_with_listing(
        self,
        listing: dict,
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        plan = plan_for(
            "crates/wenlan-core/src/lint/pages/security_test.rs"
        )
        with tempfile.TemporaryDirectory() as directory:
            fake_cargo = Path(directory) / "fake_cargo.py"
            run_marker = Path(directory) / "nextest-ran"
            fake_cargo.write_text(
                """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

if sys.argv[1] == "metadata":
    print(os.environ["FAKE_CARGO_METADATA"])
elif sys.argv[1:3] == ["nextest", "list"]:
    print(os.environ["FAKE_NEXTEST_LISTING"])
elif sys.argv[1:3] == ["nextest", "run"]:
    Path(os.environ["NEXTEST_RUN_MARKER"]).write_text("ran")
else:
    raise SystemExit(f"unexpected cargo arguments: {sys.argv[1:]}")
""",
                encoding="utf-8",
            )
            if os.name == "nt":
                fake_cargo_launcher = Path(directory) / "cargo.cmd"
                fake_cargo_launcher.write_text(
                    f'@echo off\n"{sys.executable}" "{fake_cargo}" %*\n',
                    encoding="utf-8",
                )
            else:
                fake_cargo_launcher = fake_cargo
                fake_cargo_launcher.chmod(0o755)
            environment = os.environ.copy()
            environment["CARGO"] = str(fake_cargo_launcher)
            environment["FAKE_CARGO_METADATA"] = json.dumps(cargo_metadata())
            environment["FAKE_NEXTEST_LISTING"] = json.dumps(listing)
            environment["NEXTEST_RUN_MARKER"] = str(run_marker)

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("ci_test_plan.py")),
                    "run",
                    "--suite",
                    "workspace-lib",
                    "--plan-json",
                    json.dumps(plan),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            did_run = run_marker.exists()
        return result, did_run

    def test_zero_match_filterset_fails_before_running_tests(self) -> None:
        result, did_run = self.run_filterset_with_listing(
            {
                "rust-suites": {
                    "wenlan-core": {
                        "testcases": {
                            "wrong::test": {
                                "ignored": False,
                                "filter-match": {"status": "mismatch"},
                            }
                        }
                    }
                }
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selected zero tests", result.stderr)
        self.assertFalse(did_run)

    def test_ignored_only_match_fails_before_running_tests(self) -> None:
        result, did_run = self.run_filterset_with_listing(
            {
                "rust-suites": {
                    "wenlan-core": {
                        "testcases": {
                            "owned::ignored": {
                                "ignored": True,
                                "filter-match": {"status": "matches"},
                            }
                        }
                    }
                }
            }
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selected zero tests", result.stderr)
        self.assertFalse(did_run)

    def test_runnable_match_executes_nextest(self) -> None:
        result, did_run = self.run_filterset_with_listing(
            {
                "rust-suites": {
                    "wenlan-core": {
                        "testcases": {
                            "owned::active": {
                                "ignored": False,
                                "filter-match": {"status": "matches"},
                            }
                        }
                    }
                }
            }
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(did_run)


class PackageClosureTests(unittest.TestCase):
    def test_server_source_runs_server_and_reverse_dependent_mcp_libs(self) -> None:
        plan = plan_for("crates/wenlan-server/src/routes.rs")

        self.assertEqual(plan["workspace_lib"]["mode"], "packages")
        self.assertEqual(
            plan["workspace_lib"]["packages"],
            ["wenlan-mcp", "wenlan-server"],
        )
        self.assertEqual(
            plan["cli_server_integration"],
            {"mode": "packages", "packages": ["wenlan-server"]},
        )
        self.assertEqual(plan["core_integration"], {"mode": "skip"})

    def test_types_source_runs_every_reverse_dependent_package_and_integration(self) -> None:
        plan = plan_for("crates/wenlan-types/src/responses.rs")

        self.assertEqual(plan["workspace_lib"]["mode"], "packages")
        self.assertEqual(
            plan["workspace_lib"]["packages"],
            [
                "wenlan",
                "wenlan-core",
                "wenlan-mcp",
                "wenlan-server",
                "wenlan-types",
            ],
        )
        self.assertEqual(
            plan["cli_server_integration"],
            {"mode": "packages", "packages": ["wenlan", "wenlan-server"]},
        )
        self.assertEqual(plan["core_integration"], {"mode": "full"})

    def test_unknown_source_inside_known_crate_uses_package_closure_not_module_guess(
        self,
    ) -> None:
        plan = plan_for("crates/wenlan-core/src/new_algorithm.rs")

        self.assertEqual(plan["workspace_lib"]["mode"], "packages")
        self.assertEqual(
            plan["workspace_lib"]["packages"],
            ["wenlan", "wenlan-core", "wenlan-mcp", "wenlan-server"],
        )
        self.assertNotIn("filterset", plan["workspace_lib"])


class NarrowOwnerTests(unittest.TestCase):
    def test_extant_core_integration_file_selects_only_its_test_target(self) -> None:
        path = "crates/wenlan-core/tests/folder_ingest_e2e.rs"
        plan = plan_for(path)

        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
        self.assertEqual(
            plan["core_integration"],
            {"mode": "targets", "targets": ["folder_ingest_e2e"]},
        )

    def test_extant_cli_integration_file_selects_only_its_test_target(self) -> None:
        path = "crates/wenlan-cli/tests/distribution.rs"
        plan = plan_for(path)

        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertEqual(
            plan["cli_server_integration"],
            {
                "mode": "targets",
                "targets": {"wenlan": ["distribution"]},
            },
        )
        self.assertEqual(plan["core_integration"], {"mode": "skip"})

    def test_explicit_isolated_test_module_selects_its_real_prefix(self) -> None:
        path = "crates/wenlan-core/src/lint/pages/security_test.rs"
        plan = plan_for(path)

        self.assertEqual(
            plan["workspace_lib"],
            {
                "mode": "filterset",
                "packages": ["wenlan-core"],
                "filterset": (
                    "package(wenlan-core) "
                    "& test(/^lint::pages::fs::tests::security_cases::/)"
                ),
            },
        )
        self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
        self.assertEqual(plan["core_integration"], {"mode": "skip"})


class FailClosedTests(unittest.TestCase):
    def test_deleted_integration_target_falls_back_to_owning_package_suite(self) -> None:
        path = "crates/wenlan-server/tests/graceful_shutdown.rs"
        plan = plan_for(path, existing_paths=set())

        self.assertEqual(
            plan["cli_server_integration"],
            {"mode": "packages", "packages": ["wenlan-server"]},
        )

    def test_shared_test_helper_falls_back_to_all_suites(self) -> None:
        plan = plan_for("crates/wenlan-core/src/lint/test_support.rs")

        self.assertEqual(plan["mode"], "full")
        self.assertEqual(plan["workspace_lib"], {"mode": "full"})
        self.assertEqual(plan["cli_server_integration"], {"mode": "full"})
        self.assertEqual(plan["core_integration"], {"mode": "full"})

    def test_unknown_workspace_rust_path_falls_back_to_all_suites(self) -> None:
        plan = plan_for("crates/new-crate/src/lib.rs")

        self.assertEqual(plan["mode"], "full")

    def test_cargo_build_native_and_ci_inputs_fall_back_to_all_suites(self) -> None:
        for path in [
            "Cargo.lock",
            "crates/wenlan-core/Cargo.toml",
            "crates/wenlan-core/build.rs",
            "crates/wenlan-core/native/bridge.cpp",
            ".config/nextest.toml",
            ".github/workflows/ci.yml",
            "scripts/ci_test_plan.py",
        ]:
            with self.subTest(path=path):
                self.assertEqual(plan_for(path)["mode"], "full")

    def test_non_pr_event_always_runs_full_backstop(self) -> None:
        plan = build_plan(
            ["crates/wenlan-server/src/routes.rs"],
            cargo_metadata(),
            event_name="push",
            existing_paths={"crates/wenlan-server/src/routes.rs"},
        )

        self.assertEqual(plan["mode"], "full")

    def test_non_pr_event_without_diff_inventory_runs_full_backstop(self) -> None:
        plan = build_plan(
            [],
            cargo_metadata(),
            event_name="workflow_dispatch",
            existing_paths=set(),
        )

        self.assertEqual(plan["mode"], "full")

    def test_pr_without_diff_inventory_fails_closed(self) -> None:
        with self.assertRaisesRegex(PlanError, "changed path inventory is empty"):
            build_plan(
                [],
                cargo_metadata(),
                event_name="pull_request",
                existing_paths=set(),
            )

    def test_malformed_metadata_fails_instead_of_emitting_empty_plan(self) -> None:
        with self.assertRaises(PlanError):
            build_plan(
                ["crates/wenlan-server/src/routes.rs"],
                {"packages": []},
                event_name="pull_request",
                existing_paths={"crates/wenlan-server/src/routes.rs"},
            )


class CommandGenerationTests(unittest.TestCase):
    def test_workspace_package_plan_is_an_argv_vector_without_shell_text(self) -> None:
        plan = plan_for("crates/wenlan-server/src/routes.rs")

        self.assertEqual(
            command_groups_for("workspace-lib", plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "wenlan-mcp",
                    "-p",
                    "wenlan-server",
                    "--lib",
                ]
            ],
        )

    def test_isolated_module_uses_literal_nextest_filterset_argument(self) -> None:
        plan = plan_for(
            "crates/wenlan-core/src/lint/pages/security_test.rs"
        )

        self.assertEqual(
            command_groups_for("workspace-lib", plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "wenlan-core",
                    "--lib",
                    "-E",
                    (
                        "package(wenlan-core) "
                        "& test(/^lint::pages::fs::tests::security_cases::/)"
                    ),
                ]
            ],
        )

    def test_owned_cli_target_is_executed_without_other_integration_targets(
        self,
    ) -> None:
        plan = plan_for("crates/wenlan-cli/tests/distribution.rs")

        self.assertEqual(
            command_groups_for(
                "cli-server-integration", plan, cargo_metadata()
            ),
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "wenlan",
                    "--test",
                    "distribution",
                ]
            ],
        )

    def test_full_core_command_excludes_only_explicit_manual_targets(self) -> None:
        plan = plan_for("Cargo.lock")

        self.assertEqual(
            command_groups_for("core-integration", plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "wenlan-core",
                    "--features",
                    "eval-harness",
                    "--test",
                    "folder_ingest_e2e",
                    "--test",
                    "read_scope",
                ]
            ],
        )

    def test_new_core_target_is_automatically_owned_by_full_plan(self) -> None:
        metadata = cargo_metadata()
        core = next(
            package
            for package in metadata["packages"]
            if package["name"] == "wenlan-core"
        )
        core["targets"].append(
            {
                "name": "new_required_target",
                "kind": ["test"],
                "src_path": (
                    f"{WORKSPACE_ROOT}/crates/wenlan-core/tests/"
                    "new_required_target.rs"
                ),
            }
        )
        plan = build_plan(
            ["Cargo.lock"],
            metadata,
            event_name="pull_request",
            existing_paths={"Cargo.lock"},
        )

        commands = command_groups_for("core-integration", plan, metadata)

        self.assertIn("new_required_target", commands[0])

    def test_skipped_suite_executes_no_command(self) -> None:
        plan = plan_for("crates/wenlan-core/tests/folder_ingest_e2e.rs")

        self.assertEqual(
            command_groups_for(
                "cli-server-integration", plan, cargo_metadata()
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
