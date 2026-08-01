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

from ci_test_plan import (
    PlanError,
    affected_package_names,
    archive_command_for,
    build_plan,
    clippy_command_for,
    command_groups_for,
    local_test_commands_for,
    required_suite_outputs,
)


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
            package(
                "wenlan-types",
                "crates/wenlan-types",
                (),
                ("page_draft_wire", "plugin_distribution"),
            ),
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
        *,
        partition: str | None = None,
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
    if "--partition" in sys.argv:
        raise SystemExit("partitioned filterset validation can reject a valid empty shard")
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

            arguments = [
                sys.executable,
                str(Path(__file__).with_name("ci_test_plan.py")),
                "run",
                "--suite",
                "workspace-lib",
                "--plan-json",
                json.dumps(plan),
            ]
            if partition is not None:
                arguments.extend(["--partition", partition])
            result = subprocess.run(
                arguments,
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

    def test_partition_does_not_narrow_filterset_validation(self) -> None:
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
            },
            partition="slice:2/2",
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
        self.assertEqual(
            plan["contract_integration"],
            {"mode": "packages", "packages": ["wenlan-mcp"]},
        )

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
        self.assertEqual(
            plan["contract_integration"],
            {
                "mode": "packages",
                "packages": ["wenlan-mcp", "wenlan-types"],
            },
        )

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

    def test_extant_contract_integration_selects_only_its_target(self) -> None:
        path = "crates/wenlan-mcp/tests/real_router.rs"
        plan = plan_for(path)

        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertEqual(
            plan["contract_integration"],
            {
                "mode": "targets",
                "targets": {"wenlan-mcp": ["real_router"]},
            },
        )


class PluginOwnerTests(unittest.TestCase):
    PR_415_PATHS = (
        "plugin-codex/skills/setup/SKILL.md",
        "plugin/skills/setup/SKILL.md",
    )

    def test_pr_415_exact_paths_are_owned_without_full_rust_fallback(self) -> None:
        plan = plan_for(*self.PR_415_PATHS)

        self.assertEqual(plan["mode"], "differential")
        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
        self.assertEqual(plan["core_integration"], {"mode": "skip"})
        self.assertEqual(plan["contract_integration"], {"mode": "skip"})
        self.assertEqual(plan["canonical_smokes"], {"mode": "skip"})

    def test_mixed_rust_and_pr_415_diff_keeps_the_rust_only_plan(self) -> None:
        rust_path = "crates/wenlan-server/src/routes.rs"
        rust_only = plan_for(rust_path)
        mixed = plan_for(rust_path, *self.PR_415_PATHS)

        for suite in (
            "workspace_lib",
            "cli_server_integration",
            "core_integration",
            "contract_integration",
            "canonical_smokes",
        ):
            self.assertEqual(mixed[suite], rust_only[suite])
        self.assertEqual(mixed["mode"], "differential")

    def test_every_plugin_job_owned_path_is_ignored_by_cargo_ownership(self) -> None:
        paths = (
            "plugin/skills/setup/SKILL.md",
            "plugin-codex/skills/setup/SKILL.md",
            ".agents/plugins/wenlan/skills/setup/SKILL.md",
            ".claude-plugin/marketplace.json",
            "plugin-contract.json",
            "scripts/validate-codex-plugin-slice.py",
            "scripts/validate-plugin-contract.py",
            "scripts/validate-plugin-contract.test.sh",
        )

        for path in paths:
            with self.subTest(path=path):
                plan = plan_for(path)
                self.assertEqual(plan["mode"], "differential")
                self.assertFalse(any(required_suite_outputs(plan).values()))


class NonRustOwnerTests(unittest.TestCase):
    def assert_non_rust_owner(self, *paths: str) -> None:
        plan = plan_for(*paths)
        self.assertEqual(plan["mode"], "differential")
        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
        self.assertEqual(plan["core_integration"], {"mode": "skip"})
        self.assertEqual(plan["contract_integration"], {"mode": "skip"})
        self.assertEqual(plan["canonical_smokes"], {"mode": "skip"})
        self.assertFalse(any(required_suite_outputs(plan).values()))

    def test_exact_docs_only_paths_have_no_rust_suites(self) -> None:
        paths = (
            "README.md",
            "docs/windows-vulkan.md",
            "crates/wenlan-core/AGENTS.md",
            "LICENSE",
            "scripts/check-readme-translations.py",
            "scripts/check-readme-translations.test.sh",
            "scripts/update-readme-eval.py",
            "scripts/update-readme-eval.test.py",
            "scripts/validate-versions.sh",
            "scripts/validate-versions.test.sh",
        )

        for path in paths:
            with self.subTest(path=path):
                self.assert_non_rust_owner(path)

    def test_exact_npm_only_paths_have_no_rust_suites(self) -> None:
        for path in (
            "crates/wenlan-cli/npm/package.json",
            "crates/wenlan-mcp/npm/install.js",
        ):
            with self.subTest(path=path):
                self.assert_non_rust_owner(path)

    def test_docs_npm_and_plugin_only_diff_stays_on_fast_lanes(self) -> None:
        self.assert_non_rust_owner(
            "docs/windows-vulkan.md",
            "crates/wenlan-mcp/npm/install.js",
            "plugin/skills/setup/SKILL.md",
        )

    def test_non_rust_owners_do_not_widen_a_mixed_rust_plan(self) -> None:
        rust_path = "crates/wenlan-server/src/routes.rs"
        rust_only = plan_for(rust_path)
        mixed = plan_for(
            rust_path,
            "docs/windows-vulkan.md",
            "crates/wenlan-mcp/npm/install.js",
            "plugin/skills/setup/SKILL.md",
        )

        for suite in (
            "workspace_lib",
            "cli_server_integration",
            "core_integration",
            "contract_integration",
            "canonical_smokes",
        ):
            self.assertEqual(mixed[suite], rust_only[suite])

    def test_markdown_test_fixture_is_not_misclassified_as_docs_only(self) -> None:
        plan = plan_for("crates/wenlan-core/tests/fixtures/example.md")

        self.assertEqual(plan["mode"], "full")
        self.assertTrue(required_suite_outputs(plan)["rust-ci-required"])


class SuiteOutputTests(unittest.TestCase):
    def test_behavioral_source_requires_lib_integration_and_canonical_smokes(
        self,
    ) -> None:
        outputs = required_suite_outputs(
            plan_for("crates/wenlan-server/src/routes.rs")
        )

        self.assertEqual(
            outputs,
            {
                "workspace-lib-required": True,
                "cli-server-integration-required": True,
                "core-integration-required": False,
                "contract-integration-required": True,
                "canonical-smokes-required": True,
                "canonical-acceptance-required": True,
                "rust-ci-required": True,
            },
        )

    def test_direct_integration_requires_only_its_canonical_suite(self) -> None:
        outputs = required_suite_outputs(
            plan_for("crates/wenlan-cli/tests/distribution.rs")
        )

        self.assertFalse(outputs["workspace-lib-required"])
        self.assertTrue(outputs["cli-server-integration-required"])
        self.assertFalse(outputs["core-integration-required"])
        self.assertFalse(outputs["contract-integration-required"])
        self.assertFalse(outputs["canonical-smokes-required"])
        self.assertTrue(outputs["canonical-acceptance-required"])
        self.assertTrue(outputs["rust-ci-required"])

    def test_isolated_and_plugin_paths_do_not_require_canonical_acceptance(
        self,
    ) -> None:
        isolated = required_suite_outputs(
            plan_for("crates/wenlan-core/src/lint/pages/security_test.rs")
        )
        plugin = required_suite_outputs(
            plan_for("plugin-codex/skills/setup/SKILL.md")
        )

        self.assertTrue(isolated["workspace-lib-required"])
        self.assertFalse(isolated["canonical-acceptance-required"])
        self.assertFalse(any(plugin.values()))

    def test_full_plan_requires_every_suite(self) -> None:
        self.assertTrue(
            all(required_suite_outputs(plan_for("Cargo.lock")).values())
        )

    def test_plan_command_writes_each_required_suite_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "metadata.json"
            output_path = Path(directory) / "github-output.txt"
            metadata_path.write_text(json.dumps(cargo_metadata()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("ci_test_plan.py")),
                    "plan",
                    "--changed-files-json",
                    json.dumps(["Cargo.lock"]),
                    "--metadata-file",
                    str(metadata_path),
                    "--event-name",
                    "pull_request",
                    "--github-output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            output = output_path.read_text(encoding="utf-8")
            for name in (
                "workspace-lib-required",
                "cli-server-integration-required",
                "core-integration-required",
                "contract-integration-required",
                "canonical-smokes-required",
                "canonical-acceptance-required",
                "rust-ci-required",
            ):
                self.assertIn(f"{name}=true\n", output)

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
        self.assertEqual(plan["contract_integration"], {"mode": "full"})

    def test_unknown_workspace_rust_path_falls_back_to_all_suites(self) -> None:
        plan = plan_for("crates/new-crate/src/lib.rs")

        self.assertEqual(plan["mode"], "full")

    def test_unknown_non_plugin_repository_path_stays_fail_closed(self) -> None:
        plan = plan_for("config/new-ci-input.json")

        self.assertEqual(plan["mode"], "full")
        self.assertEqual(plan["canonical_smokes"], {"mode": "full"})
        self.assertTrue(all(required_suite_outputs(plan).values()))

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
    def test_pr_clippy_uses_affected_dependency_closure(self) -> None:
        plan = plan_for("crates/wenlan-server/src/routes.rs")

        self.assertEqual(
            affected_package_names(plan, cargo_metadata()),
            ["wenlan-mcp", "wenlan-server"],
        )
        self.assertEqual(
            clippy_command_for(plan, cargo_metadata()),
            [
                "cargo",
                "clippy",
                "-p",
                "wenlan-mcp",
                "-p",
                "wenlan-server",
                "--lib",
                "--bins",
                "--",
                "-D",
                "warnings",
            ],
        )

    def test_direct_integration_change_keeps_test_target_clippy(self) -> None:
        plan = plan_for("crates/wenlan-mcp/tests/real_router.rs")

        self.assertEqual(
            clippy_command_for(plan, cargo_metadata()),
            [
                "cargo",
                "clippy",
                "-p",
                "wenlan-mcp",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        )

    def test_main_plan_keeps_full_workspace_clippy_backstop(self) -> None:
        plan = plan_for("Cargo.lock")

        self.assertEqual(
            clippy_command_for(plan, cargo_metadata()),
            [
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        )

    def test_local_push_runs_affected_libs_and_only_direct_integration(self) -> None:
        source_plan = plan_for("crates/wenlan-server/src/routes.rs")
        target_plan = plan_for("crates/wenlan-mcp/tests/real_router.rs")

        self.assertEqual(
            local_test_commands_for(source_plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "test",
                    "-p",
                    "wenlan-mcp",
                    "-p",
                    "wenlan-server",
                    "--lib",
                ],
                [
                    "cargo",
                    "test",
                    "-p",
                    "wenlan-server",
                    "--bin",
                    "wenlan-server",
                ],
            ],
        )
        self.assertEqual(
            local_test_commands_for(target_plan, cargo_metadata()),
            [["cargo", "test", "-p", "wenlan-mcp", "--test", "real_router"]],
        )

    def test_contract_target_generates_required_target_only_command(self) -> None:
        plan = plan_for("crates/wenlan-types/tests/page_draft_wire.rs")

        self.assertEqual(
            command_groups_for("contract-integration", plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "wenlan-types",
                    "--test",
                    "page_draft_wire",
                ]
            ],
        )

    def test_partition_rejects_shell_text_and_non_workspace_suites(self) -> None:
        plan = plan_for("Cargo.lock")
        for partition in ["slice:0/2", "slice:3/2", "slice:1/2;echo unsafe"]:
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(PlanError, "invalid nextest partition"):
                    command_groups_for(
                        "workspace-lib",
                        plan,
                        cargo_metadata(),
                        partition=partition,
                    )
        with self.assertRaisesRegex(PlanError, "only supported for workspace-lib"):
            command_groups_for(
                "core-integration",
                plan,
                cargo_metadata(),
                partition="slice:1/2",
            )

    def test_workspace_partition_is_appended_without_changing_the_plan(self) -> None:
        plan = plan_for("Cargo.lock")

        self.assertEqual(
            command_groups_for(
                "workspace-lib",
                plan,
                cargo_metadata(),
                partition="slice:1/2",
            ),
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "--workspace",
                    "--lib",
                    "--bin",
                    "wenlan-server",
                    "--partition",
                    "slice:1/2",
                ]
            ],
        )

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
                    "--bin",
                    "wenlan-server",
                ]
            ],
        )

    def test_full_workspace_archive_compiles_the_complete_lib_inventory(self) -> None:
        plan = plan_for("Cargo.lock")

        self.assertEqual(
            archive_command_for(
                plan,
                cargo_metadata(),
                archive_file="/tmp/workspace-lib.tar.zst",
            ),
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                "/tmp/workspace-lib.tar.zst",
                "--workspace",
                "--lib",
                "--bin",
                "wenlan-server",
            ],
        )

    def test_package_archive_compiles_only_selected_package_libs(self) -> None:
        plan = plan_for("crates/wenlan-server/src/routes.rs")

        self.assertEqual(
            archive_command_for(
                plan,
                cargo_metadata(),
                archive_file="/tmp/workspace-lib.tar.zst",
            ),
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                "/tmp/workspace-lib.tar.zst",
                "-p",
                "wenlan-mcp",
                "-p",
                "wenlan-server",
                "--lib",
                "--bin",
                "wenlan-server",
            ],
        )

    def test_filterset_archive_defers_test_selector_until_partition_run(self) -> None:
        plan = plan_for(
            "crates/wenlan-core/src/lint/pages/security_test.rs"
        )
        filterset = (
            "package(wenlan-core) "
            "& test(/^lint::pages::fs::tests::security_cases::/)"
        )

        archive = archive_command_for(
            plan,
            cargo_metadata(),
            archive_file="/tmp/workspace-lib.tar.zst",
        )
        run = command_groups_for(
            "workspace-lib",
            plan,
            cargo_metadata(),
            partition="slice:2/2",
            archive_file="/tmp/workspace-lib.tar.zst",
            workspace_remap="/repo",
        )

        self.assertEqual(
            archive,
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                "/tmp/workspace-lib.tar.zst",
                "-p",
                "wenlan-core",
                "--lib",
            ],
        )
        self.assertNotIn("-E", archive)
        self.assertEqual(
            run,
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "--archive-file",
                    "/tmp/workspace-lib.tar.zst",
                    "--workspace-remap",
                    "/repo",
                    "-E",
                    filterset,
                    "--no-tests=pass",
                    "--partition",
                    "slice:2/2",
                ]
            ],
        )

    def test_archive_run_requires_a_safe_workspace_remap(self) -> None:
        plan = plan_for("Cargo.lock")

        with self.assertRaisesRegex(PlanError, "requires workspace-remap"):
            command_groups_for(
                "workspace-lib",
                plan,
                cargo_metadata(),
                archive_file="/tmp/workspace-lib.tar.zst",
            )
        with self.assertRaisesRegex(PlanError, "non-empty path argument"):
            archive_command_for(
                plan,
                cargo_metadata(),
                archive_file="--unexpected-option",
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
