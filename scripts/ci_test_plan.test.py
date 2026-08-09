#!/usr/bin/env python3
"""Contract tests for fail-closed differential Rust test planning."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from fnmatch import fnmatchcase
from pathlib import Path

from ci_test_plan import (
    PlanError,
    affected_package_names,
    archive_command_for,
    build_plan,
    build_platform_plan,
    clippy_command_for,
    command_groups_for,
    local_test_commands_for,
    macos_archive_commands_for,
    macos_archive_run_commands_for,
    required_suite_outputs,
)


WORKSPACE_ROOT = "/repo"
M5_READER_PATHS = (
    "scripts/m5-reader-sweep.py",
    "crates/wenlan-core/AGENTS.md",
    "docs/plans/2026-07-27-m5-reader-manifest-inventory.md",
)
M5_READER_FILTERSET = (
    "package(wenlan-core) & "
    "test(/^drift_guard::m5_reader_inventory_matches_current_tree$/)"
)
RUST_SUITE_OUTPUTS = (
    "workspace-lib-required",
    "cli-server-integration-required",
    "core-integration-required",
    "contract-integration-required",
    "canonical-smokes-required",
    "canonical-acceptance-required",
    "rust-ci-required",
)
MACOS_ARCHIVE_CONTRACT_PATHS = (
    "scripts/ci_test_plan.py",
    "scripts/ci_test_plan.test.py",
)


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
            "wenlan-app",
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
            package(
                "wenlan-app",
                "app",
                ("wenlan-types",),
                ("sources_integration",),
            ),
        ],
    }


def macos_metadata() -> dict:
    metadata = cargo_metadata()
    core = next(
        package for package in metadata["packages"] if package["name"] == "wenlan-core"
    )
    core["targets"].append(
        {
            "name": "m4_community_gates",
            "kind": ["test"],
            "src_path": f"{WORKSPACE_ROOT}/crates/wenlan-core/tests/m4_community_gates.rs",
        }
    )
    return metadata


def plan_for(
    *paths: str,
    existing_paths: set[str] | None = None,
    event_name: str = "pull_request",
) -> dict:
    return build_plan(
        list(paths),
        cargo_metadata(),
        event_name=event_name,
        existing_paths=set(paths) if existing_paths is None else existing_paths,
    )


class FiltersetExecutionTests(unittest.TestCase):
    def run_filterset_with_listing(
        self,
        listing: dict,
        *,
        partition: str | None = None,
        plan_from_env: bool = False,
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
            environment["FAKE_CI_TEST_PLAN"] = json.dumps(plan)

            arguments = [
                sys.executable,
                str(Path(__file__).with_name("ci_test_plan.py")),
                "run",
                "--suite",
                "workspace-lib",
            ]
            if plan_from_env:
                arguments.extend(["--plan-env", "FAKE_CI_TEST_PLAN"])
            else:
                arguments.extend(["--plan-json", json.dumps(plan)])
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

    def test_plan_environment_preserves_compact_json_on_windows(self) -> None:
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
            plan_from_env=True,
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


class PlatformPlanTests(unittest.TestCase):
    INFRASTRUCTURE_PATHS = (
        ".config/nextest.toml",
        ".github/workflows/ci.yml",
        ".githooks/pre-push",
        "scripts/ci_test_plan.py",
        "scripts/ci_test_plan.test.py",
        "scripts/release_targets.py",
        "scripts/release_targets.test.py",
        "scripts/build-release-binaries.sh",
        "scripts/release-workflow-contract.test.py",
        "crates/wenlan-cli/tests/distribution.rs",
        "crates/wenlan-core/src/drift_guard.rs",
    )

    def platform_plan_for(self, *paths: str) -> dict:
        return build_platform_plan(
            paths,
            cargo_metadata(),
            event_name="pull_request",
            existing_paths=set(paths),
        )

    def test_infrastructure_only_diff_skips_every_platform_behavior_suite(self) -> None:
        plan = self.platform_plan_for(*self.INFRASTRUCTURE_PATHS)

        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
        self.assertEqual(plan["core_integration"], {"mode": "skip"})
        self.assertEqual(plan["contract_integration"], {"mode": "skip"})
        self.assertFalse(any(required_suite_outputs(plan).values()))

    def test_archive_contract_uses_canonical_fail_closed_plan_only_when_routed(self) -> None:
        canonical = plan_for(*MACOS_ARCHIVE_CONTRACT_PATHS)
        platform = self.platform_plan_for(*MACOS_ARCHIVE_CONTRACT_PATHS)

        self.assertEqual(canonical["mode"], "full")
        self.assertEqual(canonical["workspace_lib"], {"mode": "full"})
        self.assertEqual(canonical["cli_server_integration"], {"mode": "full"})
        self.assertEqual(platform["workspace_lib"], {"mode": "skip"})
        self.assertEqual(platform["cli_server_integration"], {"mode": "skip"})
        self.assertFalse(any(required_suite_outputs(platform).values()))

    def test_m5_reader_inventory_diff_skips_platform_and_release_suites(self) -> None:
        plan = self.platform_plan_for(*M5_READER_PATHS)

        self.assertEqual(plan["mode"], "differential")
        self.assertFalse(any(required_suite_outputs(plan).values()))

        workflow_path = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
        workflow = workflow_path.read_text(encoding="utf-8")
        release_block = workflow.split("            release-preflight:\n", 1)[1]
        release_block = release_block.split("            mcp-platform:\n", 1)[0]
        patterns = re.findall(r"^\s+- '([^']+)'\s*$", release_block, re.MULTILINE)
        self.assertTrue(patterns)
        self.assertFalse(
            any(fnmatchcase(M5_READER_PATHS[0], pattern) for pattern in patterns),
            patterns,
        )

    def test_infrastructure_does_not_widen_mixed_server_change(self) -> None:
        product = "crates/wenlan-server/src/bind_addr_tests.rs"
        focused = self.platform_plan_for(product)
        mixed = self.platform_plan_for(product, *self.INFRASTRUCTURE_PATHS)

        for suite in (
            "workspace_lib",
            "cli_server_integration",
            "core_integration",
            "contract_integration",
            "canonical_smokes",
        ):
            self.assertEqual(mixed[suite], focused[suite])
        self.assertEqual(
            mixed["workspace_lib"]["packages"],
            ["wenlan-server"],
        )
        self.assertEqual(
            mixed["cli_server_integration"],
            {"mode": "packages", "packages": ["wenlan-server"]},
        )

    def test_platform_workspace_lib_excludes_compile_only_reverse_dependents(self) -> None:
        plan = self.platform_plan_for("crates/wenlan-types/src/lib.rs")

        self.assertEqual(
            plan["workspace_lib"]["packages"],
            ["wenlan", "wenlan-core", "wenlan-server"],
        )
        self.assertNotIn("wenlan-mcp", plan["workspace_lib"]["packages"])
        self.assertNotIn("wenlan-types", plan["workspace_lib"]["packages"])

    def test_platform_filterset_excludes_reverse_dependents_from_expression(
        self,
    ) -> None:
        plan = self.platform_plan_for(
            "crates/wenlan-server/src/bind_addr_tests.rs",
            "crates/wenlan-core/src/lint/pages/security_test.rs",
        )

        self.assertEqual(plan["workspace_lib"]["mode"], "filterset")
        self.assertEqual(
            plan["workspace_lib"]["packages"],
            ["wenlan-core", "wenlan-server"],
        )
        self.assertNotIn("package(wenlan-mcp)", plan["workspace_lib"]["filterset"])

    def test_manifest_native_and_toolchain_inputs_still_fail_closed_full(self) -> None:
        for path in (
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "crates/wenlan-core/Cargo.toml",
            "crates/wenlan-core/build.rs",
            "crates/wenlan-core/native/bridge.cpp",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.platform_plan_for(path)["mode"], "full")

    def test_nextest_config_uses_platform_smokes_not_full_behavior_suites(self) -> None:
        plan = self.platform_plan_for(".config/nextest.toml")

        self.assertEqual(plan["mode"], "differential")
        self.assertFalse(any(required_suite_outputs(plan).values()))

    def test_push_platform_plan_uses_differential_routing(self) -> None:
        plan = build_platform_plan(
            self.INFRASTRUCTURE_PATHS,
            cargo_metadata(),
            event_name="push",
            existing_paths=set(self.INFRASTRUCTURE_PATHS),
        )

        self.assertEqual(plan["mode"], "differential")
        self.assertFalse(any(required_suite_outputs(plan).values()))

    def test_filtered_m4_only_pr_inventory_skips_generic_platform_suites(
        self,
    ) -> None:
        plan = build_platform_plan(
            [],
            cargo_metadata(),
            event_name="pull_request",
            existing_paths=set(),
        )

        self.assertEqual(plan["mode"], "differential")
        self.assertEqual(
            plan["reasons"], ["no generic platform behavioral inputs changed"]
        )
        self.assertFalse(any(required_suite_outputs(plan).values()))

    def test_filtered_empty_push_inventory_skips_platform_suites(self) -> None:
        plan = build_platform_plan(
            [],
            cargo_metadata(),
            event_name="push",
            existing_paths=set(),
        )

        self.assertEqual(plan["mode"], "differential")
        self.assertEqual(
            plan["reasons"], ["no generic platform behavioral inputs changed"]
        )
        self.assertFalse(any(required_suite_outputs(plan).values()))

    def test_canonical_plan_guards_the_filtered_platform_projection(self) -> None:
        workflow_path = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
        workflow = workflow_path.read_text(encoding="utf-8")
        canonical_marker = "      - name: Plan affected Rust tests\n"
        platform_marker = "      - name: Plan affected platform behavior tests\n"
        next_marker = "      - name: Test release target inventory\n"

        canonical_index = workflow.index(canonical_marker)
        platform_index = workflow.index(platform_marker)
        next_index = workflow.index(next_marker, platform_index)
        canonical_body = workflow[canonical_index:platform_index]
        platform_body = workflow[platform_index:next_index]
        release_guard = (
            "if: steps.release-proof.outputs.verified-release-merge != 'true'"
        )

        self.assertLess(canonical_index, platform_index)
        self.assertIn(
            "CHANGED_FILES_JSON: ${{ steps.filter.outputs.impact_files }}",
            canonical_body,
        )
        self.assertIn(release_guard, canonical_body)
        self.assertIn(release_guard, platform_body)
        self.assertNotIn("continue-on-error:", canonical_body)
        for status_override in ("always()", "failure()", "!cancelled()"):
            self.assertNotIn(status_override, platform_body)
        for marker in (
            "MACOS_FILES_JSON: ${{ steps.filter.outputs.macos_files }}",
            "MACOS_M4_FILES_JSON: ${{ steps.filter.outputs['macos-m4_files'] }}",
            "M5_PLATFORM_FILES_JSON: ${{ steps.filter.outputs['m5-platform_files'] }}",
            "WINDOWS_FILES_JSON: ${{ steps.filter.outputs.windows_files }}",
            "'(($macos - $macos_m4 - $m5_platform) "
            "+ ($windows - $m5_platform)) | unique'",
        ):
            self.assertIn(marker, platform_body)

    def test_release_manual_and_unknown_platform_events_keep_full_backstop(
        self,
    ) -> None:
        for event_name in ("release-please", "workflow_dispatch", "schedule"):
            with self.subTest(event_name=event_name):
                plan = build_platform_plan(
                    [],
                    cargo_metadata(),
                    event_name=event_name,
                    existing_paths=set(),
                )

                self.assertEqual(plan["mode"], "full")
                self.assertTrue(all(required_suite_outputs(plan).values()))


class NarrowOwnerTests(unittest.TestCase):
    R4_MANIFESTS = (
        "crates/wenlan-core/src/drift_guard/r4_test_support_api_manifest.txt",
        "crates/wenlan-core/src/drift_guard/r4_test_support_raw_manifest.txt",
    )

    def test_m5_reader_diff_selects_only_the_exact_inventory_contract(self) -> None:
        plan = plan_for(*M5_READER_PATHS)

        self.assertEqual(plan["mode"], "differential")
        self.assertEqual(
            plan["workspace_lib"],
            {
                "mode": "filterset",
                "packages": ["wenlan-core"],
                "filterset": M5_READER_FILTERSET,
            },
        )
        self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
        self.assertEqual(plan["core_integration"], {"mode": "skip"})
        self.assertEqual(plan["contract_integration"], {"mode": "skip"})
        self.assertEqual(plan["canonical_smokes"], {"mode": "skip"})
        outputs = required_suite_outputs(plan)
        self.assertTrue(outputs["workspace-lib-required"])
        self.assertTrue(outputs["rust-ci-required"])
        self.assertFalse(outputs["canonical-acceptance-required"])

    def test_removed_m5_reader_sweep_fails_closed_to_every_suite(self) -> None:
        plan = plan_for(M5_READER_PATHS[0], existing_paths=set())

        self.assertEqual(plan["mode"], "full")
        outputs = required_suite_outputs(plan)
        self.assertTrue(all(outputs[name] for name in RUST_SUITE_OUTPUTS))
        self.assertFalse(outputs["fmt-required"])
        self.assertTrue(outputs["clippy-required"])

    def test_broad_core_change_overrides_exact_m5_reader_filter(self) -> None:
        core_path = "crates/wenlan-core/src/search.rs"
        core_only = plan_for(core_path)
        mixed = plan_for(core_path, M5_READER_PATHS[0])

        for suite in (
            "workspace_lib",
            "cli_server_integration",
            "core_integration",
            "contract_integration",
            "canonical_smokes",
        ):
            self.assertEqual(mixed[suite], core_only[suite])

    def test_r4_manifests_select_only_their_canonical_contract_module(self) -> None:
        for path in self.R4_MANIFESTS:
            with self.subTest(path=path):
                plan = plan_for(path)

                self.assertEqual(plan["mode"], "differential")
                self.assertEqual(
                    plan["workspace_lib"],
                    {
                        "mode": "filterset",
                        "packages": ["wenlan-core"],
                        "filterset": (
                            "package(wenlan-core) & "
                            "test(/^drift_guard::r4_test_support_test::/)"
                        ),
                    },
                )
                self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
                self.assertEqual(plan["core_integration"], {"mode": "skip"})
                self.assertEqual(plan["contract_integration"], {"mode": "skip"})
                self.assertEqual(plan["canonical_smokes"], {"mode": "skip"})
                outputs = required_suite_outputs(plan)
                self.assertTrue(outputs["rust-ci-required"])
                self.assertFalse(outputs["canonical-acceptance-required"])

    def test_removed_r4_manifest_fails_closed_to_every_suite(self) -> None:
        for path in self.R4_MANIFESTS:
            with self.subTest(path=path):
                plan = plan_for(path, existing_paths=set())

                self.assertEqual(plan["mode"], "full")
                outputs = required_suite_outputs(plan)
                self.assertTrue(all(outputs[name] for name in RUST_SUITE_OUTPUTS))
                self.assertFalse(outputs["fmt-required"])
                self.assertTrue(outputs["clippy-required"])

    def test_unknown_r4_manifest_sibling_stays_fail_closed(self) -> None:
        path = "crates/wenlan-core/src/drift_guard/new_manifest.txt"
        plan = plan_for(path)

        self.assertEqual(plan["mode"], "full")
        outputs = required_suite_outputs(plan)
        self.assertTrue(all(outputs[name] for name in RUST_SUITE_OUTPUTS))
        self.assertFalse(outputs["fmt-required"])
        self.assertTrue(outputs["clippy-required"])

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

    def test_git_hook_contract_paths_have_no_rust_suites(self) -> None:
        for path in (
            ".githooks/pre-commit",
            ".githooks/pre-push",
            "scripts/git-hooks.test.sh",
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


class AppOwnerTests(unittest.TestCase):
    def test_types_source_change_never_selects_wenlan_app(self) -> None:
        # wenlan-app path-depends on wenlan-types in real Cargo metadata, but
        # the app crate carries its own CI lane and must stay out of the
        # reverse-dependency closure the Rust planner selects.
        plan = plan_for("crates/wenlan-types/src/responses.rs")

        self.assertEqual(plan["workspace_lib"]["mode"], "packages")
        self.assertNotIn("wenlan-app", plan["workspace_lib"]["packages"])

    def test_app_only_diff_has_no_rust_suites(self) -> None:
        plan = plan_for("app/src/main.rs", "src/App.tsx", "package.json")

        self.assertEqual(plan["mode"], "differential")
        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertEqual(plan["cli_server_integration"], {"mode": "skip"})
        self.assertEqual(plan["core_integration"], {"mode": "skip"})
        self.assertEqual(plan["contract_integration"], {"mode": "skip"})
        self.assertEqual(plan["canonical_smokes"], {"mode": "skip"})
        # App paths keep the fast fmt/clippy gate (rustfmt/clippy don't need
        # GTK to lint text), but select zero packages, so the workspace lint
        # step still runs with no wenlan-app compile in scope.
        outputs = required_suite_outputs(plan)
        self.assertFalse(outputs["workspace-lib-required"])
        self.assertFalse(outputs["cli-server-integration-required"])
        self.assertFalse(outputs["core-integration-required"])
        self.assertFalse(outputs["contract-integration-required"])
        self.assertFalse(outputs["canonical-smokes-required"])
        self.assertFalse(outputs["rust-ci-required"])
        self.assertEqual(clippy_command_for(plan, cargo_metadata()), [])

    def test_app_eval_fixture_falls_through_to_full_plan(self) -> None:
        # app/eval/** is consumed by wenlan-core tests, not app-only, so it
        # must keep the full-plan fallback rather than being swallowed by the
        # app-job classifier.
        plan = plan_for("app/eval/somefixture.json")

        self.assertEqual(plan["mode"], "full")
        self.assertIn("unowned changed path: app/eval/somefixture.json", plan["reasons"])

    def test_app_cargo_toml_is_app_owned_not_a_full_plan_trigger(self) -> None:
        plan = plan_for("app/Cargo.toml")

        self.assertEqual(plan["mode"], "differential")
        self.assertEqual(plan["workspace_lib"], {"mode": "skip"})
        self.assertFalse(required_suite_outputs(plan)["rust-ci-required"])
        self.assertEqual(clippy_command_for(plan, cargo_metadata()), [])


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
                "fmt-required": True,
                "clippy-required": True,
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

    def test_full_cargo_plan_requires_every_suite_and_only_relevant_daily_gate(
        self,
    ) -> None:
        outputs = required_suite_outputs(plan_for("Cargo.lock"))

        self.assertTrue(all(outputs[name] for name in RUST_SUITE_OUTPUTS))
        self.assertFalse(outputs["fmt-required"])
        self.assertTrue(outputs["clippy-required"])

    def test_daily_gate_routes_exact_input_classes(self) -> None:
        cases = {
            "crates/wenlan-server/src/routes.rs": (True, True),
            "Cargo.toml": (False, True),
            "Cargo.lock": (False, True),
            "rust-toolchain.toml": (False, True),
            "clippy.toml": (False, True),
            "crates/wenlan-core/Cargo.toml": (False, True),
            "crates/wenlan-core/build.rs": (True, True),
            "crates/wenlan-core/native/bridge.cpp": (False, True),
            M5_READER_PATHS[0]: (False, False),
            ".config/nextest.toml": (False, False),
            ".github/workflows/ci.yml": (False, False),
            "scripts/ci_test_plan.py": (False, False),
            ".githooks/pre-commit": (False, False),
            ".githooks/pre-push": (False, False),
            "scripts/git-hooks.test.sh": (False, False),
            "docs/windows-vulkan.md": (False, False),
            "plugin-codex/skills/setup/SKILL.md": (False, False),
            "crates/wenlan-mcp/npm/install.js": (False, False),
        }

        for event_name in ("pull_request", "push"):
            for path, (fmt_required, clippy_required) in cases.items():
                with self.subTest(event_name=event_name, path=path):
                    outputs = required_suite_outputs(
                        plan_for(path, event_name=event_name)
                    )
                    self.assertEqual(outputs["fmt-required"], fmt_required)
                    self.assertEqual(outputs["clippy-required"], clippy_required)

    def test_mixed_contract_and_rust_source_requires_both_daily_gates(self) -> None:
        outputs = required_suite_outputs(
            plan_for(M5_READER_PATHS[0], "crates/wenlan-core/src/search.rs")
        )

        self.assertTrue(outputs["fmt-required"])
        self.assertTrue(outputs["clippy-required"])

    def test_deleted_rust_and_non_rust_fallbacks_keep_clippy_fail_closed(
        self,
    ) -> None:
        cases = (
            (M5_READER_PATHS[0], set(), False),
            ("config/new-ci-input.json", {"config/new-ci-input.json"}, False),
            ("crates/wenlan-server/tests/graceful_shutdown.rs", set(), True),
        )

        for path, existing_paths, fmt_required in cases:
            with self.subTest(path=path):
                outputs = required_suite_outputs(
                    plan_for(path, existing_paths=existing_paths)
                )
                self.assertEqual(outputs["fmt-required"], fmt_required)
                self.assertTrue(outputs["clippy-required"])

    def test_daily_gate_output_fields_reject_missing_or_non_boolean_mutations(
        self,
    ) -> None:
        for field, mutation in (
            ("fmt_required", None),
            ("fmt_required", "true"),
            ("clippy_required", None),
            ("clippy_required", 1),
        ):
            with self.subTest(field=field, mutation=mutation):
                plan = plan_for("crates/wenlan-core/src/search.rs")
                if mutation is None:
                    plan.pop(field)
                else:
                    plan[field] = mutation
                with self.assertRaisesRegex(PlanError, "has no boolean"):
                    required_suite_outputs(plan)

    def test_non_pr_empty_inventory_keeps_both_daily_gate_backstops(self) -> None:
        plan = build_plan(
            [],
            cargo_metadata(),
            event_name="workflow_dispatch",
            existing_paths=set(),
        )

        outputs = required_suite_outputs(plan)
        self.assertTrue(outputs["fmt-required"])
        self.assertTrue(outputs["clippy-required"])

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
                *RUST_SUITE_OUTPUTS,
                "clippy-required",
            ):
                self.assertIn(f"{name}=true\n", output)
            self.assertIn("fmt-required=false\n", output)

    def test_platform_plan_command_accepts_empty_filtered_push_inventory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "metadata.json"
            output_path = Path(directory) / "github-output.txt"
            metadata_path.write_text(json.dumps(cargo_metadata()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("ci_test_plan.py")),
                    "plan",
                    "--scope",
                    "platform",
                    "--changed-files-json",
                    "[]",
                    "--metadata-file",
                    str(metadata_path),
                    "--event-name",
                    "push",
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
            for name in (*RUST_SUITE_OUTPUTS, "fmt-required", "clippy-required"):
                self.assertIn(f"{name}=false\n", output)

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
        outputs = required_suite_outputs(plan)
        self.assertTrue(all(outputs[name] for name in RUST_SUITE_OUTPUTS))
        self.assertFalse(outputs["fmt-required"])
        self.assertTrue(outputs["clippy-required"])

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

    def test_push_uses_the_same_differential_plan_as_pull_request(self) -> None:
        path = "crates/wenlan-server/src/routes.rs"
        pull_request = plan_for(path)
        push = plan_for(path, event_name="push")

        self.assertEqual(push, pull_request)
        self.assertEqual(push["mode"], "differential")

    def test_release_manual_and_unknown_events_keep_full_backstop(self) -> None:
        for event_name in ("release-please", "workflow_dispatch", "schedule"):
            with self.subTest(event_name=event_name):
                plan = build_plan(
                    [],
                    cargo_metadata(),
                    event_name=event_name,
                    existing_paths=set(),
                )

                self.assertEqual(plan["mode"], "full")
                self.assertTrue(all(required_suite_outputs(plan).values()))

    def test_differential_event_without_diff_inventory_fails_closed(self) -> None:
        for event_name in ("pull_request", "push"):
            with self.subTest(event_name=event_name):
                with self.assertRaisesRegex(
                    PlanError, "changed path inventory is empty"
                ):
                    build_plan(
                        [],
                        cargo_metadata(),
                        event_name=event_name,
                        existing_paths=set(),
                    )

    def test_push_deletions_and_unknown_paths_stay_fail_closed_full(self) -> None:
        for path, existing_paths in (
            (M5_READER_PATHS[0], set()),
            ("config/new-ci-input.json", {"config/new-ci-input.json"}),
        ):
            with self.subTest(path=path):
                plan = plan_for(
                    path,
                    existing_paths=existing_paths,
                    event_name="push",
                )
                self.assertEqual(plan["mode"], "full")

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
                "--exclude",
                "wenlan-app",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        )

    def test_main_plan_local_tests_exclude_wenlan_app(self) -> None:
        plan = plan_for("Cargo.lock")

        self.assertEqual(
            local_test_commands_for(plan, cargo_metadata()),
            [
                ["cargo", "test", "--workspace", "--exclude", "wenlan-app", "--lib"],
                ["cargo", "test", "-p", "wenlan-server", "--bin", "wenlan-server"],
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

    def test_local_push_runs_isolated_modules_by_exact_cargo_prefix(self) -> None:
        plan = plan_for(
            "crates/wenlan-core/src/drift_guard/r4_test_support_raw_manifest.txt"
        )

        self.assertEqual(
            local_test_commands_for(plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "test",
                    "-p",
                    "wenlan-core",
                    "--lib",
                    "drift_guard::r4_test_support_test::",
                ]
            ],
        )

    def test_m5_reader_owner_uses_exact_local_and_nextest_commands(self) -> None:
        plan = plan_for(M5_READER_PATHS[0])

        self.assertEqual(
            local_test_commands_for(plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "test",
                    "-p",
                    "wenlan-core",
                    "--lib",
                    "drift_guard::m5_reader_inventory_matches_current_tree",
                    "--",
                    "--exact",
                ]
            ],
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
                    M5_READER_FILTERSET,
                ]
            ],
        )

    def test_m5_reader_archive_defers_exact_selector_to_run(self) -> None:
        plan = plan_for(M5_READER_PATHS[0])

        archive = archive_command_for(
            plan,
            cargo_metadata(),
            archive_file="/tmp/workspace-lib.tar.zst",
        )
        run = command_groups_for(
            "workspace-lib",
            plan,
            cargo_metadata(),
            partition="slice:1/2",
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
                    M5_READER_FILTERSET,
                    "--no-tests=pass",
                    "--partition",
                    "slice:1/2",
                ]
            ],
        )

    def test_local_push_preserves_mixed_broad_and_isolated_owners(self) -> None:
        plan = plan_for(
            "crates/wenlan-mcp/src/tools.rs",
            "crates/wenlan-core/src/drift_guard/r4_test_support_api_manifest.txt",
        )

        self.assertEqual(plan["workspace_lib"]["mode"], "filterset")
        self.assertEqual(
            local_test_commands_for(plan, cargo_metadata()),
            [
                ["cargo", "test", "-p", "wenlan-mcp", "--lib"],
                [
                    "cargo",
                    "test",
                    "-p",
                    "wenlan-core",
                    "--lib",
                    "drift_guard::r4_test_support_test::",
                ],
            ],
        )

    def test_local_push_rejects_unrecognized_filterset_text(self) -> None:
        plan = plan_for(
            "crates/wenlan-core/src/drift_guard/r4_test_support_raw_manifest.txt"
        )
        plan["workspace_lib"]["filterset"] = "all()"

        with self.assertRaisesRegex(PlanError, "not locally executable"):
            local_test_commands_for(plan, cargo_metadata())

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
                    "--exclude",
                    "wenlan-app",
                    "--lib",
                    "--bin",
                    "wenlan-server",
                    "--partition",
                    "slice:1/2",
                ]
            ],
        )

    def test_live_workspace_full_run_excludes_wenlan_app_without_an_archive(
        self,
    ) -> None:
        # A direct `run --suite workspace-lib` without --archive-file hits the
        # unarchived nextest-run branch of a full plan. A full plan must never
        # compile wenlan-app through this surface either, not just the
        # archive-file path exercised above.
        plan = plan_for("Cargo.lock")

        self.assertEqual(
            command_groups_for("workspace-lib", plan, cargo_metadata()),
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "--workspace",
                    "--exclude",
                    "wenlan-app",
                    "--lib",
                    "--bin",
                    "wenlan-server",
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
                "--exclude",
                "wenlan-app",
                "--lib",
                "--bin",
                "wenlan-server",
            ],
        )

    def test_macos_archive_builds_each_no_feature_owner_once(self) -> None:
        plan = plan_for("Cargo.lock")

        commands = macos_archive_commands_for(
            plan,
            macos_metadata(),
            archive_dir="/tmp/macos-nextest",
            include_m4=True,
        )

        self.assertEqual(len(commands), 3)
        self.assertEqual(
            commands[0],
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                str(Path("/tmp/macos-nextest/workspace-lib.tar.zst")),
                "--workspace",
                "--exclude",
                "wenlan-app",
                "--lib",
                "--bin",
                "wenlan-server",
            ],
        )
        self.assertEqual(
            commands[1],
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                str(Path("/tmp/macos-nextest/cli-server-integration-1.tar.zst")),
                "-p",
                "wenlan",
                "-p",
                "wenlan-server",
                "-E",
                "kind(test)",
            ],
        )
        self.assertEqual(
            commands[2],
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                str(Path("/tmp/macos-nextest/m4-community-gates.tar.zst")),
                "-p",
                "wenlan-core",
                "--test",
                "m4_community_gates",
            ],
        )
        self.assertNotIn("--features", [argument for command in commands for argument in command])

    def test_macos_archive_runs_every_owner_over_the_same_three_slices(self) -> None:
        plan = plan_for("Cargo.lock")

        commands = macos_archive_run_commands_for(
            plan,
            macos_metadata(),
            archive_dir="/tmp/macos-nextest",
            workspace_remap="/repo",
            partition="slice:2/3",
            include_m4=True,
        )

        self.assertEqual(len(commands), 3)
        self.assertTrue(
            all(command[-2:] == ["--partition", "slice:2/3"] for command in commands)
        )
        self.assertEqual(
            [
                Path(command[command.index("--archive-file") + 1]).name
                for command in commands
            ],
            [
                "workspace-lib.tar.zst",
                "cli-server-integration-1.tar.zst",
                "m4-community-gates.tar.zst",
            ],
        )
        self.assertTrue(all("--workspace-remap" in command for command in commands))

    def test_macos_archive_rejects_a_missing_m4_target(self) -> None:
        metadata = macos_metadata()
        core = next(
            package for package in metadata["packages"] if package["name"] == "wenlan-core"
        )
        core["targets"] = [
            target for target in core["targets"] if target["name"] != "m4_community_gates"
        ]

        with self.assertRaisesRegex(PlanError, "no m4_community_gates target"):
            macos_archive_commands_for(
                plan_for("Cargo.lock"),
                metadata,
                archive_dir="/tmp/macos-nextest",
                include_m4=True,
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
