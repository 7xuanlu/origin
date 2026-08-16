#!/usr/bin/env python3
"""Build fail-closed Rust test plans from a Git diff and Cargo metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Iterable


class PlanError(ValueError):
    """Planner input is malformed and CI must stop."""


DIFFERENTIAL_EVENTS = frozenset({"pull_request", "push"})


ISOLATED_UNIT_MODULES = {
    "crates/wenlan-core/src/drift_guard/r4_test_support_api_manifest.txt": (
        "wenlan-core",
        "drift_guard::r4_test_support_test",
    ),
    "crates/wenlan-core/src/drift_guard/r4_test_support_raw_manifest.txt": (
        "wenlan-core",
        "drift_guard::r4_test_support_test",
    ),
    "crates/wenlan-core/src/lint/pages/security_test.rs": (
        "wenlan-core",
        "lint::pages::fs::tests::security_cases",
    ),
}

ISOLATED_UNIT_TESTS = {
    "scripts/m5-reader-sweep.py": (
        "wenlan-core",
        "drift_guard::m5_reader_inventory_matches_current_tree",
    ),
}

SHARED_TEST_HELPERS = {
    "crates/wenlan-core/src/lint/test_support.rs",
    "crates/wenlan-core/src/lint/test_support_db.rs",
    "crates/wenlan-core/src/lint/test_support_fs.rs",
    "crates/wenlan-core/src/lint/test_support_privacy.rs",
}

FULL_INPUTS = {
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".config/nextest.toml",
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    "scripts/ci_test_plan.py",
    "scripts/ci_test_plan.test.py",
}

CLIPPY_INPUTS = {
    "Cargo.lock",
    "clippy.toml",
    "rust-toolchain.toml",
}

NON_CLIPPY_CONTRACT_INPUTS = {
    ".config/nextest.toml",
    "scripts/ci_test_plan.py",
    "scripts/ci_test_plan.test.py",
}

DETECT_CONTRACT_INPUTS = {
    ".githooks/pre-commit",
    ".githooks/pre-push",
    "scripts/git-hooks.test.sh",
}

# Repository contracts are executable Rust tests, but they do not exercise
# platform behavior. Linux owns their canonical execution; routing them into
# macOS and Windows would duplicate product suites without adding OS evidence.
PLATFORM_CONTRACT_ONLY_PATHS = {
    "crates/wenlan-cli/tests/distribution.rs",
    "crates/wenlan-core/src/drift_guard.rs",
}

NATIVE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".m",
    ".mm",
}

CLI_SERVER_PACKAGES = {"wenlan", "wenlan-server"}
CONTRACT_PACKAGES = {"wenlan-mcp", "wenlan-types"}
MANUAL_CORE_INTEGRATION = {"cached_scenario_db_check", "eval_harness"}

PLUGIN_JOB_PREFIXES = (
    "plugin",
    "plugin-codex",
    ".agents/plugins",
    ".claude-plugin",
)
PLUGIN_JOB_FILES = {
    "plugin-contract.json",
    "scripts/validate-codex-plugin-slice.py",
    "scripts/validate-plugin-contract.py",
    "scripts/validate-plugin-contract.test.sh",
}

DOCS_JOB_PREFIXES = ("docs",)
DOCS_JOB_FILES = {
    "LICENSE",
    "scripts/check-readme-translations.py",
    "scripts/check-readme-translations.test.sh",
    "scripts/update-readme-eval.py",
    "scripts/update-readme-eval.test.py",
    "scripts/validate-versions.sh",
    "scripts/validate-versions.test.sh",
}

# The Tauri desktop app crate plus the root-level frontend dirs it shares a
# release pipeline with. All of it runs under the separate app-check job
# (macos-14), never the Rust workspace plan.
APP_JOB_PREFIXES = ("app", "src", "e2e", "preview", "review", "public")
APP_JOB_FILES = {
    "package.json",
    "pnpm-lock.yaml",
    "pnpm-workspace.yaml",
    "index.html",
    "tsconfig.json",
    "vite.config.ts",
    "vite.preview.config.ts",
    "vite.review.config.ts",
    "vitest.config.ts",
    "playwright.config.ts",
    "playwright.review.config.ts",
    "scripts/prepare-sidecars.sh",
    "scripts/prepare-tauri-build-sidecars.sh",
    "scripts/open-review-app.mjs",
    "scripts/open-review-app.test.ts",
    "scripts/prepare-sidecars.test.ts",
    "scripts/refactor/api-route-diff.mjs",
    "scripts/refactor/api-route-diff.test.ts",
    "scripts/refactor/inventory.sh",
    "scripts/refactor/inventory.test.ts",
    "scripts/release-version-sync.test.ts",
    "scripts/resolve-backend-dir.sh",
    "scripts/verify-review-bundle.mjs",
}

SUITE_OUTPUT_KEYS = {
    "workspace-lib-required": "workspace_lib",
    "cli-server-integration-required": "cli_server_integration",
    "core-integration-required": "core_integration",
    "contract-integration-required": "contract_integration",
    "canonical-smokes-required": "canonical_smokes",
}


def _cargo_executable() -> str:
    """Return Cargo's executable without parsing a shell command string."""

    cargo = os.environ.get("CARGO", "cargo")
    if not cargo:
        raise PlanError("CARGO must name a Cargo executable")
    return cargo


def _executable_command(command: list[str]) -> list[str]:
    if not command or command[0] != "cargo":
        raise PlanError(f"cannot execute non-Cargo command: {command!r}")
    return [_cargo_executable(), *command[1:]]


def _full_plan(reason: str) -> dict:
    return {
        "version": 1,
        "mode": "full",
        "reasons": [reason],
        "fmt_required": True,
        "clippy_required": True,
        "workspace_lib": {"mode": "full"},
        "cli_server_integration": {"mode": "full"},
        "core_integration": {"mode": "full"},
        "contract_integration": {"mode": "full"},
        "canonical_smokes": {"mode": "full"},
    }


def _skip_plan(reason: str) -> dict:
    return {
        "version": 1,
        "mode": "differential",
        "reasons": [reason],
        "fmt_required": False,
        "clippy_required": False,
        "workspace_lib": {"mode": "skip"},
        "cli_server_integration": {"mode": "skip"},
        "core_integration": {"mode": "skip"},
        "contract_integration": {"mode": "skip"},
        "canonical_smokes": {"mode": "skip"},
    }


def _normalize_path(raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise PlanError("changed paths must be non-empty strings")
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise PlanError(f"changed path is not repository-relative: {raw_path!r}")
    return path.as_posix()


def _workspace(metadata: object) -> tuple[dict[str, dict], dict[str, str]]:
    if not isinstance(metadata, dict):
        raise PlanError("Cargo metadata must be a JSON object")
    workspace_root = metadata.get("workspace_root")
    packages = metadata.get("packages")
    if not isinstance(workspace_root, str) or not workspace_root:
        raise PlanError("Cargo metadata is missing workspace_root")
    if not isinstance(packages, list) or not packages:
        raise PlanError("Cargo metadata has no workspace packages")

    root = Path(workspace_root)
    by_name: dict[str, dict] = {}
    directories: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise PlanError("Cargo metadata package is not an object")
        name = package.get("name")
        manifest_path = package.get("manifest_path")
        dependencies = package.get("dependencies")
        targets = package.get("targets")
        if (
            not isinstance(name, str)
            or not name
            or name in by_name
            or not isinstance(manifest_path, str)
            or not isinstance(dependencies, list)
            or not isinstance(targets, list)
        ):
            raise PlanError(f"malformed Cargo package metadata: {package!r}")
        try:
            directory = Path(manifest_path).parent.relative_to(root).as_posix()
        except ValueError as error:
            raise PlanError(
                f"workspace manifest is outside workspace_root: {manifest_path}"
            ) from error
        if name == "wenlan-app":
            # wenlan-app is a workspace member but ships its own CI lane
            # (app-check, macos-14). Ubuntu runners lack GTK, so the planner
            # must never select it, and app/ must never be treated as
            # rust-plan-owned territory.
            continue
        by_name[name] = package
        directories[directory] = name

    return by_name, directories


def _reverse_dependencies(packages: dict[str, dict]) -> dict[str, set[str]]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for package_name, package in packages.items():
        for dependency in package["dependencies"]:
            if not isinstance(dependency, dict):
                raise PlanError(
                    f"malformed dependency metadata in package {package_name}"
                )
            dependency_name = dependency.get("name")
            dependency_path = dependency.get("path")
            if dependency_path is None:
                continue
            if not isinstance(dependency_name, str):
                raise PlanError(
                    f"path dependency without a name in package {package_name}"
                )
            if dependency_name not in packages:
                raise PlanError(
                    f"workspace path dependency {dependency_name!r} is absent"
                )
            reverse[dependency_name].add(package_name)
    return reverse


def _closure(seed: str, reverse: dict[str, set[str]]) -> set[str]:
    selected: set[str] = set()
    pending = deque([seed])
    while pending:
        package = pending.popleft()
        if package in selected:
            continue
        selected.add(package)
        pending.extend(sorted(reverse.get(package, ())))
    return selected


def _owner(path: str, directories: dict[str, str]) -> tuple[str, str] | None:
    matches = [
        (directory, package)
        for directory, package in directories.items()
        if path == directory or path.startswith(f"{directory}/")
    ]
    if not matches:
        return None
    directory, package = max(matches, key=lambda item: len(item[0]))
    return package, path[len(directory) + 1 :]


def _plugin_job_owns(path: str) -> bool:
    if path in PLUGIN_JOB_FILES:
        return True
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in PLUGIN_JOB_PREFIXES
    )


def _npm_job_owns(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return len(parts) >= 4 and parts[0] == "crates" and parts[2] == "npm"


def _rust_fixture_owns(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if len(parts) < 4 or parts[0] != "crates":
        return False
    return "tests" in parts[2:-1] or parts[2] == "resources"


def _docs_job_owns(path: str) -> bool:
    if path in DOCS_JOB_FILES:
        return True
    if any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in DOCS_JOB_PREFIXES
    ):
        return True
    return path.endswith(".md") and not _rust_fixture_owns(path)


def _app_job_owns(path: str) -> bool:
    # app/eval/** holds eval fixtures consumed by wenlan-core tests, so those
    # changes must keep falling through to the full-plan path rather than
    # being classified as app-only. Root Cargo.toml / Cargo.lock and
    # crates/wenlan-types/** deliberately stay OUT of this classifier too —
    # they keep their rust-plan meaning; ci.yml's own `app:` path filter
    # triggers app-check for them independently.
    if path == "app/eval" or path.startswith("app/eval/"):
        return False
    if path in APP_JOB_FILES:
        return True
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in APP_JOB_PREFIXES
    )


def _is_clippy_input(path: str) -> bool:
    if _plugin_job_owns(path) or _npm_job_owns(path) or _docs_job_owns(path):
        return False
    suffix = PurePosixPath(path).suffix.lower()
    return (
        path in CLIPPY_INPUTS
        or path.endswith("/Cargo.toml")
        or path.endswith("/build.rs")
        or suffix == ".rs"
        or suffix in NATIVE_SUFFIXES
    )


def _is_non_clippy_contract_input(path: str, existing: set[str]) -> bool:
    if _plugin_job_owns(path) or _npm_job_owns(path) or _docs_job_owns(path):
        return True
    if (
        path.startswith(".github/workflows/")
        or path in NON_CLIPPY_CONTRACT_INPUTS
        or path in DETECT_CONTRACT_INPUTS
    ):
        return True
    return path in existing and (
        path in ISOLATED_UNIT_TESTS or path in ISOLATED_UNIT_MODULES
    )


def _daily_gate_requirements(
    plan: dict,
    paths: list[str],
    *,
    event_name: str,
    existing: set[str],
) -> tuple[bool, bool]:
    # PRs and ordinary main pushes pay only for gates that can observe their
    # inputs. Manual, release, and unknown events retain the full backstop. A
    # full differential fallback remains conservative unless every path has an
    # explicit non-Clippy contract owner.
    if event_name not in DIFFERENTIAL_EVENTS:
        return True, True
    fmt_required = any(PurePosixPath(path).suffix.lower() == ".rs" for path in paths)
    clippy_required = any(_is_clippy_input(path) for path in paths)
    if (
        not clippy_required
        and plan.get("mode") == "full"
        and any(not _is_non_clippy_contract_input(path, existing) for path in paths)
    ):
        clippy_required = True
    return fmt_required, clippy_required


def _integration_targets(package: dict) -> set[str]:
    targets = set()
    for target in package["targets"]:
        if not isinstance(target, dict):
            raise PlanError("Cargo target metadata is not an object")
        name = target.get("name")
        kinds = target.get("kind")
        if not isinstance(name, str) or not isinstance(kinds, list):
            raise PlanError(f"malformed Cargo target metadata: {target!r}")
        if "test" in kinds:
            targets.add(name)
    return targets


def _workspace_lib_plan(
    broad_packages: set[str],
    isolated_filters: dict[str, set[str]],
    isolated_tests: dict[str, set[str]],
) -> dict:
    for package in broad_packages:
        isolated_filters.pop(package, None)
        isolated_tests.pop(package, None)
    expressions = [f"package({package})" for package in sorted(broad_packages)]
    for package in sorted(isolated_filters):
        for prefix in sorted(isolated_filters[package]):
            expressions.append(f"package({package}) & test(/^{prefix}::/)")
    for package in sorted(isolated_tests):
        for test_name in sorted(isolated_tests[package]):
            expressions.append(f"package({package}) & test(/^{test_name}$/)")
    if not expressions:
        return {"mode": "skip"}
    if isolated_filters or isolated_tests:
        return {
            "mode": "filterset",
            "packages": sorted(
                broad_packages | set(isolated_filters) | set(isolated_tests)
            ),
            "filterset": " | ".join(expressions),
        }
    return {"mode": "packages", "packages": sorted(broad_packages)}


def _build_test_plan(
    changed_paths: Iterable[object],
    cargo_metadata: object,
    *,
    event_name: str,
    existing_paths: set[str] | None = None,
) -> dict:
    """Return the suite portion of a deterministic test plan."""

    packages, directories = _workspace(cargo_metadata)
    reverse = _reverse_dependencies(packages)
    paths = [_normalize_path(path) for path in changed_paths]
    if event_name not in DIFFERENTIAL_EVENTS:
        return _full_plan(f"{event_name} keeps the full backstop")
    if not paths:
        raise PlanError("changed path inventory is empty")
    if existing_paths is None:
        existing = {path for path in paths if Path(path).exists()}
    else:
        existing = {_normalize_path(path) for path in existing_paths}

    broad_packages: set[str] = set()
    isolated_filters: dict[str, set[str]] = defaultdict(set)
    isolated_tests: dict[str, set[str]] = defaultdict(set)
    cli_server_packages: set[str] = set()
    cli_server_targets: dict[str, set[str]] = defaultdict(set)
    contract_packages: set[str] = set()
    contract_targets: dict[str, set[str]] = defaultdict(set)
    core_full = False
    core_targets: set[str] = set()
    canonical_smokes = False
    reasons: list[str] = []

    for path in paths:
        # Plugin distribution files have an explicit required CI owner. Check
        # that ownership before Cargo/build classification so a mixed
        # Rust-plus-plugin diff keeps the Rust portion's narrow plan.
        if _plugin_job_owns(path):
            reasons.append(f"plugin contract job owns changed path: {path}")
            continue

        if _npm_job_owns(path):
            reasons.append(f"npm contract job owns changed path: {path}")
            continue

        if _docs_job_owns(path):
            reasons.append(f"docs contract job owns changed path: {path}")
            continue

        if path in DETECT_CONTRACT_INPUTS:
            reasons.append(f"detect contract job owns changed path: {path}")
            continue

        if _app_job_owns(path):
            reasons.append(f"app contract job owns changed path: {path}")
            continue

        if (
            path in FULL_INPUTS
            or path.endswith("/Cargo.toml")
            or path.endswith("/build.rs")
            or PurePosixPath(path).suffix.lower() in NATIVE_SUFFIXES
        ):
            return _full_plan(f"shared build or native input changed: {path}")
        if path in SHARED_TEST_HELPERS:
            return _full_plan(f"shared test helper changed: {path}")

        isolated_test = ISOLATED_UNIT_TESTS.get(path)
        if isolated_test is not None:
            if path not in existing:
                return _full_plan(f"isolated unit test owner was removed: {path}")
            package, test_name = isolated_test
            if package not in packages:
                raise PlanError(
                    f"isolated unit test maps to unknown package {package!r}"
                )
            isolated_tests[package].add(test_name)
            reasons.append(f"isolated unit test owner changed: {path}")
            continue

        isolated = ISOLATED_UNIT_MODULES.get(path)
        if isolated is not None:
            if path not in existing:
                return _full_plan(f"isolated test module was removed or renamed: {path}")
            package, prefix = isolated
            if package not in packages:
                raise PlanError(
                    f"isolated test module maps to unknown package {package!r}"
                )
            isolated_filters[package].add(prefix)
            reasons.append(f"isolated unit module changed: {path}")
            continue

        owner = _owner(path, directories)
        if owner is None:
            return _full_plan(f"unowned changed path: {path}")
        package_name, package_path = owner
        package = packages[package_name]

        parts = PurePosixPath(package_path).parts
        if len(parts) == 2 and parts[0] == "tests" and path.endswith(".rs"):
            target = PurePosixPath(package_path).stem
            known_targets = _integration_targets(package)
            if path not in existing or target not in known_targets:
                if package_name in CLI_SERVER_PACKAGES:
                    cli_server_packages.add(package_name)
                    reasons.append(
                        f"removed or unknown integration target uses package suite: {path}"
                    )
                    continue
                if package_name in CONTRACT_PACKAGES:
                    contract_packages.add(package_name)
                    reasons.append(
                        f"removed or unknown contract target uses package suite: {path}"
                    )
                    continue
                if package_name == "wenlan-core":
                    core_full = True
                    reasons.append(
                        f"removed or unknown core target uses full core suite: {path}"
                    )
                    continue
                return _full_plan(
                    f"unowned integration target changed in {package_name}: {path}"
                )
            if package_name in CLI_SERVER_PACKAGES:
                cli_server_targets[package_name].add(target)
                reasons.append(f"owned integration target changed: {path}")
                continue
            if package_name in CONTRACT_PACKAGES:
                contract_targets[package_name].add(target)
                reasons.append(f"owned contract target changed: {path}")
                continue
            if package_name == "wenlan-core":
                if target not in MANUAL_CORE_INTEGRATION:
                    core_targets.add(target)
                reasons.append(f"owned core integration target changed: {path}")
                continue
            return _full_plan(
                f"integration target has no required-suite planner: {path}"
            )

        if parts and parts[0] == "src" and path.endswith(".rs"):
            closure = _closure(package_name, reverse)
            broad_packages.update(closure)
            cli_server_packages.update(closure & CLI_SERVER_PACKAGES)
            contract_packages.update(closure & CONTRACT_PACKAGES)
            core_full = core_full or "wenlan-core" in closure
            canonical_smokes = True
            reasons.append(f"source change selects reverse dependency closure: {path}")
            continue

        return _full_plan(f"unclassified crate input changed: {path}")

    for package_name in cli_server_packages:
        cli_server_targets.pop(package_name, None)
    for package_name in contract_packages:
        contract_targets.pop(package_name, None)

    if cli_server_packages:
        cli_server_plan = {
            "mode": "packages",
            "packages": sorted(cli_server_packages),
        }
    elif cli_server_targets:
        cli_server_plan = {
            "mode": "targets",
            "targets": {
                package: sorted(targets)
                for package, targets in sorted(cli_server_targets.items())
            },
        }
    else:
        cli_server_plan = {"mode": "skip"}

    if core_full:
        core_plan = {"mode": "full"}
    elif core_targets:
        core_plan = {"mode": "targets", "targets": sorted(core_targets)}
    else:
        core_plan = {"mode": "skip"}

    if contract_packages:
        contract_plan = {
            "mode": "packages",
            "packages": sorted(contract_packages),
        }
    elif contract_targets:
        contract_plan = {
            "mode": "targets",
            "targets": {
                package: sorted(targets)
                for package, targets in sorted(contract_targets.items())
            },
        }
    else:
        contract_plan = {"mode": "skip"}

    return {
        "version": 1,
        "mode": "differential",
        "reasons": sorted(set(reasons)),
        "workspace_lib": _workspace_lib_plan(
            broad_packages, isolated_filters, isolated_tests
        ),
        "cli_server_integration": cli_server_plan,
        "core_integration": core_plan,
        "contract_integration": contract_plan,
        "canonical_smokes": {"mode": "full" if canonical_smokes else "skip"},
    }


def build_plan(
    changed_paths: Iterable[object],
    cargo_metadata: object,
    *,
    event_name: str,
    existing_paths: set[str] | None = None,
) -> dict:
    """Return a deterministic test and daily-gate plan or raise PlanError."""

    raw_paths = list(changed_paths)
    plan = _build_test_plan(
        raw_paths,
        cargo_metadata,
        event_name=event_name,
        existing_paths=existing_paths,
    )
    paths = [_normalize_path(path) for path in raw_paths]
    if event_name not in DIFFERENTIAL_EVENTS:
        existing = set()
    elif existing_paths is None:
        existing = {path for path in paths if Path(path).exists()}
    else:
        existing = {_normalize_path(path) for path in existing_paths}
    fmt_required, clippy_required = _daily_gate_requirements(
        plan,
        paths,
        event_name=event_name,
        existing=existing,
    )
    plan["fmt_required"] = fmt_required
    plan["clippy_required"] = clippy_required
    return plan


def build_platform_plan(
    changed_paths: Iterable[object],
    cargo_metadata: object,
    *,
    event_name: str,
    existing_paths: set[str] | None = None,
) -> dict:
    """Plan platform behavior without letting CI infrastructure widen it."""

    paths = [_normalize_path(path) for path in changed_paths]
    if event_name not in DIFFERENTIAL_EVENTS:
        return build_plan(
            paths,
            cargo_metadata,
            event_name=event_name,
            existing_paths=existing_paths,
        )
    if not paths:
        # This inventory is already restricted to platform-owned paths by the
        # workflow. An empty filtered inventory is therefore a proven skip,
        # not a missing canonical diff. build_plan keeps the fail-closed guard
        # for an empty repository-wide inventory on both PRs and pushes.
        return _skip_plan("no generic platform behavioral inputs changed")

    # Linux owns repository/workflow/planner contracts. Platform behavior is
    # widened only by product crate inputs or the shared Cargo/toolchain graph.
    # Repository-contract tests run canonically on Linux and must not turn an
    # infrastructure-only PR into two full native suites. nextest repository
    # config can contain host/target overrides, so CI executes a small real
    # nextest test on each platform; Linux remains the owner of the complete
    # suite. See https://nexte.st/docs/configuration/specifying-platforms/.
    shared_platform_inputs = {
        "Cargo.toml",
        "Cargo.lock",
        "rust-toolchain.toml",
    }
    relevant = [
        path
        for path in paths
        if path in shared_platform_inputs
        or (
            path.startswith("crates/")
            and path not in PLATFORM_CONTRACT_ONLY_PATHS
        )
    ]
    if not relevant:
        return _skip_plan("no platform behavioral inputs changed")

    relevant_existing = None
    if existing_paths is not None:
        normalized_existing = {_normalize_path(path) for path in existing_paths}
        relevant_existing = set(relevant) & normalized_existing
    plan = build_plan(
        relevant,
        cargo_metadata,
        event_name=event_name,
        existing_paths=relevant_existing,
    )

    # Cross-platform MCP/type compile contracts have dedicated Linux and
    # mcp-platform owners. Do not pull their reverse-dev-dependency tests into
    # the expensive macOS behavior lane merely because server/core changed.
    workspace = plan["workspace_lib"]
    if plan["mode"] == "differential" and workspace["mode"] in {
        "packages",
        "filterset",
    }:
        behavioral_packages = {"wenlan-core", "wenlan-server", "wenlan"}
        packages = [
            package
            for package in workspace["packages"]
            if package in behavioral_packages
        ]
        if packages:
            workspace["packages"] = packages
            if workspace["mode"] == "filterset":
                expressions = [
                    expression
                    for expression in workspace["filterset"].split(" | ")
                    if any(
                        expression.startswith(f"package({package})")
                        for package in packages
                    )
                ]
                if not expressions:
                    raise PlanError(
                        "platform workspace filterset lost every behavioral owner"
                    )
                workspace["filterset"] = " | ".join(expressions)
        else:
            plan["workspace_lib"] = {"mode": "skip"}
    return plan


def required_suite_outputs(plan: object) -> dict[str, bool]:
    """Return validated booleans used by workflow job and step gates."""

    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise PlanError("unsupported or malformed test plan")
    required: dict[str, bool] = {}
    for output_name, field_name in (
        ("fmt-required", "fmt_required"),
        ("clippy-required", "clippy_required"),
    ):
        value = plan.get(field_name)
        if not isinstance(value, bool):
            raise PlanError(f"test plan has no boolean {field_name!r}")
        required[output_name] = value
    for output_name, suite_name in SUITE_OUTPUT_KEYS.items():
        suite = plan.get(suite_name)
        if not isinstance(suite, dict):
            raise PlanError(f"test plan has no suite {suite_name!r}")
        mode = suite.get("mode")
        if not isinstance(mode, str) or not mode:
            raise PlanError(f"test plan suite {suite_name!r} has no mode")
        required[output_name] = mode != "skip"
    required["canonical-acceptance-required"] = any(
        required[name]
        for name in (
            "cli-server-integration-required",
            "core-integration-required",
            "canonical-smokes-required",
        )
    )
    required["rust-ci-required"] = (
        required["workspace-lib-required"]
        or required["canonical-acceptance-required"]
        or required["contract-integration-required"]
    )
    return required


def _validated_package_names(
    raw_names: object,
    packages: dict[str, dict],
    *,
    allowed: set[str] | None = None,
) -> list[str]:
    if (
        not isinstance(raw_names, list)
        or not raw_names
        or not all(isinstance(name, str) and name for name in raw_names)
    ):
        raise PlanError("suite package inventory must be a non-empty string list")
    names = sorted(set(raw_names))
    unknown = set(names) - set(packages)
    if unknown:
        raise PlanError(f"suite plan contains unknown packages: {sorted(unknown)}")
    if allowed is not None and not set(names) <= allowed:
        raise PlanError(f"suite plan contains packages outside its owner set: {names}")
    return names


def _package_args(package_names: Iterable[str]) -> list[str]:
    arguments: list[str] = []
    for package in package_names:
        arguments.extend(["-p", package])
    return arguments


def _validated_path_argument(raw_path: object, *, name: str) -> str:
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or "\n" in raw_path
        or "\r" in raw_path
        or raw_path.startswith("-")
    ):
        raise PlanError(f"{name} must be a non-empty path argument")
    return raw_path


def affected_package_names(plan: object, cargo_metadata: object) -> list[str]:
    """Return the package closure that owns local/CI compilation checks."""

    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise PlanError("unsupported or malformed test plan")
    packages, _directories = _workspace(cargo_metadata)
    selected: set[str] = set()
    for suite_name, owners in (
        ("workspace_lib", None),
        ("cli_server_integration", CLI_SERVER_PACKAGES),
        ("contract_integration", CONTRACT_PACKAGES),
    ):
        suite = plan.get(suite_name)
        if not isinstance(suite, dict):
            raise PlanError(f"test plan has no suite {suite_name!r}")
        mode = suite.get("mode")
        if mode == "full":
            selected.update(packages if owners is None else owners)
        elif mode in {"packages", "filterset"}:
            selected.update(
                _validated_package_names(
                    suite.get("packages"), packages, allowed=owners
                )
            )
        elif mode == "targets":
            targets = suite.get("targets")
            if not isinstance(targets, dict) or not targets:
                raise PlanError(f"test plan target suite {suite_name!r} is empty")
            unknown = set(targets) - set(packages)
            if unknown or (owners is not None and not set(targets) <= owners):
                raise PlanError(
                    f"test plan target suite {suite_name!r} has unknown owners"
                )
            selected.update(targets)
        elif mode != "skip":
            raise PlanError(f"unknown {suite_name} mode: {mode!r}")

    core = plan.get("core_integration")
    if not isinstance(core, dict):
        raise PlanError("test plan has no suite 'core_integration'")
    if core.get("mode") != "skip":
        if "wenlan-core" not in packages:
            raise PlanError("Cargo metadata has no wenlan-core package")
        selected.add("wenlan-core")
    return sorted(selected)


def clippy_command_for(plan: object, cargo_metadata: object) -> list[str]:
    """Build one affected-package Clippy command, preserving main's full backstop."""

    names = affected_package_names(plan, cargo_metadata)
    if not names:
        return []
    if isinstance(plan, dict) and plan.get("mode") == "full":
        return [
            "cargo",
            "clippy",
            "--workspace",
            "--exclude",
            "wenlan-app",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]
    direct_test_change = any(
        isinstance(plan.get(suite_name), dict)
        and plan[suite_name].get("mode") == "targets"
        for suite_name in (
            "cli_server_integration",
            "core_integration",
            "contract_integration",
        )
    )
    # Ordinary source edits already compile their integration owners in the
    # selected test lanes. Lint lib/bin targets here so local push and PR lint
    # do not compile every integration target a second time. A directly edited
    # integration target keeps all-targets lint for its owning package.
    target_scope = ["--all-targets"] if direct_test_change else ["--lib", "--bins"]
    return [
        "cargo",
        "clippy",
        *_package_args(names),
        *target_scope,
        "--",
        "-D",
        "warnings",
    ]


def _workspace_suite_mode(plan: object) -> object:
    workspace = plan.get("workspace_lib") if isinstance(plan, dict) else None
    return workspace.get("mode") if isinstance(workspace, dict) else None


def local_test_commands_for(plan: object, cargo_metadata: object) -> list[list[str]]:
    """Return bounded pre-push tests: affected libs plus directly changed targets."""

    packages, _directories = _workspace(cargo_metadata)
    commands: list[list[str]] = []
    workspace = plan.get("workspace_lib") if isinstance(plan, dict) else None
    if not isinstance(workspace, dict):
        raise PlanError("test plan has no suite 'workspace_lib'")
    workspace_mode = workspace.get("mode")
    if workspace_mode == "full":
        # Every fail-closed widening lands here: a workflow file, an unowned
        # path, a shared build input. Those are the changes least likely to
        # break a specific package and the most expensive to prove locally --
        # the whole workspace lib suite, which is over an hour on Windows. CI
        # owns that run through `workspace-lib-required`, so pre-push stops at
        # Clippy rather than making every broad change unpushable. Narrow plans
        # below still run their own tests, which is where the local signal is
        # actually worth its cost.
        return []
    if workspace_mode == "packages":
        names = _validated_package_names(workspace.get("packages"), packages)
        commands.append(["cargo", "test", *_package_args(names), "--lib"])
        if "wenlan-server" in names:
            commands.append(
                ["cargo", "test", "-p", "wenlan-server", "--bin", "wenlan-server"]
            )
    elif workspace_mode == "filterset":
        names = set(_validated_package_names(workspace.get("packages"), packages))
        filterset = workspace.get("filterset")
        if not isinstance(filterset, str) or not filterset:
            raise PlanError("workspace filterset is empty")
        broad: set[str] = set()
        isolated: list[tuple[str, str]] = []
        exact: list[tuple[str, str]] = []
        for expression in filterset.split(" | "):
            broad_match = re.fullmatch(r"package\(([A-Za-z0-9_-]+)\)", expression)
            isolated_match = re.fullmatch(
                r"package\(([A-Za-z0-9_-]+)\) & test\(/\^([A-Za-z0-9_:]+)::/\)",
                expression,
            )
            exact_match = re.fullmatch(
                r"package\(([A-Za-z0-9_-]+)\) & test\(/\^([A-Za-z0-9_:]+)\$/\)",
                expression,
            )
            if broad_match is not None:
                broad.add(broad_match.group(1))
            elif isolated_match is not None:
                isolated.append((isolated_match.group(1), isolated_match.group(2)))
            elif exact_match is not None:
                exact.append((exact_match.group(1), exact_match.group(2)))
            else:
                raise PlanError("workspace filterset is not locally executable")
        selected = (
            broad
            | {package for package, _prefix in isolated}
            | {package for package, _test_name in exact}
        )
        if selected != names:
            raise PlanError("workspace filterset package inventory is inconsistent")
        if broad:
            commands.append(
                ["cargo", "test", *_package_args(sorted(broad)), "--lib"]
            )
            if "wenlan-server" in broad:
                commands.append(
                    ["cargo", "test", "-p", "wenlan-server", "--bin", "wenlan-server"]
                )
        for package, prefix in isolated:
            commands.append(
                ["cargo", "test", "-p", package, "--lib", f"{prefix}::"]
            )
        for package, test_name in exact:
            commands.append(
                [
                    "cargo",
                    "test",
                    "-p",
                    package,
                    "--lib",
                    test_name,
                    "--",
                    "--exact",
                ]
            )
    elif workspace_mode != "skip":
        raise PlanError(f"unknown workspace-lib mode: {workspace_mode!r}")

    for suite_name in (
        "cli-server-integration",
        "core-integration",
        "contract-integration",
    ):
        suite = plan.get(suite_name.replace("-", "_")) if isinstance(plan, dict) else None
        if not isinstance(suite, dict):
            raise PlanError(f"test plan has no suite {suite_name!r}")
        # Source changes are covered locally by affected lib tests and Clippy's
        # --all-targets compile. CI owns the broader integration/smoke proof.
        if suite.get("mode") != "targets":
            continue
        for command in command_groups_for(suite_name, plan, cargo_metadata):
            commands.append(["cargo", "test", *command[3:]])
    return commands


def archive_command_for(
    plan: object,
    cargo_metadata: object,
    *,
    archive_file: str,
) -> list[str]:
    """Build one affected unit-test archive without applying test predicates."""

    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise PlanError("unsupported or malformed test plan")
    packages, _directories = _workspace(cargo_metadata)
    suite = plan.get("workspace_lib")
    if not isinstance(suite, dict):
        raise PlanError("test plan has no suite 'workspace-lib'")
    mode = suite.get("mode")
    if mode == "skip":
        return []

    cargo = [
        "cargo",
        "nextest",
        "archive",
        "--archive-file",
        _validated_path_argument(archive_file, name="archive-file"),
    ]
    if mode == "full":
        return [
            *cargo,
            "--workspace",
            "--exclude",
            "wenlan-app",
            "--lib",
            "--bin",
            "wenlan-server",
        ]
    if mode in {"packages", "filterset"}:
        if mode == "filterset":
            filterset = suite.get("filterset")
            if not isinstance(filterset, str) or not filterset:
                raise PlanError("workspace filterset is empty")
        names = _validated_package_names(suite.get("packages"), packages)
        # Filterset plans archive each selected package's complete lib test
        # binary. Test-name predicates are valid only when the archive runs.
        command = [*cargo, *_package_args(names), "--lib"]
        if "wenlan-server" in names:
            command.extend(["--bin", "wenlan-server"])
        return command
    raise PlanError(f"unknown workspace-lib mode: {mode!r}")


def macos_archive_commands_for(
    plan: object,
    cargo_metadata: object,
    *,
    archive_dir: str,
    include_m4: bool,
) -> list[list[str]]:
    """Build the exact no-feature macOS inventory into reusable archives."""

    packages, _directories = _workspace(cargo_metadata)
    directory = Path(
        _validated_path_argument(archive_dir, name="archive-dir")
    )
    commands: list[list[str]] = []

    workspace_archive = str(directory / "workspace-lib.tar.zst")
    workspace = archive_command_for(
        plan,
        cargo_metadata,
        archive_file=workspace_archive,
    )
    if workspace:
        commands.append(workspace)

    integrations = command_groups_for(
        "cli-server-integration",
        plan,
        cargo_metadata,
    )
    for index, run_command in enumerate(integrations, start=1):
        if run_command[:3] != ["cargo", "nextest", "run"]:
            raise PlanError("CLI/server integration command is not a nextest run")
        commands.append(
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                str(directory / f"cli-server-integration-{index}.tar.zst"),
                *run_command[3:],
            ]
        )

    if include_m4:
        if "wenlan-core" not in packages or "m4_community_gates" not in _integration_targets(
            packages["wenlan-core"]
        ):
            raise PlanError("Cargo metadata has no m4_community_gates target")
        commands.append(
            [
                "cargo",
                "nextest",
                "archive",
                "--archive-file",
                str(directory / "m4-community-gates.tar.zst"),
                "-p",
                "wenlan-core",
                "--test",
                "m4_community_gates",
            ]
        )
    return commands


def macos_archive_run_commands_for(
    plan: object,
    cargo_metadata: object,
    *,
    archive_dir: str,
    workspace_remap: str,
    partition: str,
    include_m4: bool,
) -> list[list[str]]:
    """Run each archived macOS suite once across the shared partition set."""

    packages, _directories = _workspace(cargo_metadata)
    match = re.fullmatch(r"slice:([1-9][0-9]*)/([1-9][0-9]*)", partition)
    if match is None or int(match.group(1)) > int(match.group(2)):
        raise PlanError(f"invalid nextest partition: {partition!r}")
    directory = Path(
        _validated_path_argument(archive_dir, name="archive-dir")
    )
    remap = _validated_path_argument(workspace_remap, name="workspace-remap")
    commands: list[list[str]] = []

    workspace_archive = str(directory / "workspace-lib.tar.zst")
    commands.extend(
        command_groups_for(
            "workspace-lib",
            plan,
            cargo_metadata,
            partition=partition,
            archive_file=workspace_archive,
            workspace_remap=remap,
        )
    )

    integrations = command_groups_for(
        "cli-server-integration",
        plan,
        cargo_metadata,
    )
    for index, _run_command in enumerate(integrations, start=1):
        commands.append(
            [
                "cargo",
                "nextest",
                "run",
                "--archive-file",
                str(directory / f"cli-server-integration-{index}.tar.zst"),
                "--workspace-remap",
                remap,
                "--no-tests=pass",
                "--partition",
                partition,
            ]
        )

    if include_m4:
        if "wenlan-core" not in packages or "m4_community_gates" not in _integration_targets(
            packages["wenlan-core"]
        ):
            raise PlanError("Cargo metadata has no m4_community_gates target")
        commands.append(
            [
                "cargo",
                "nextest",
                "run",
                "--archive-file",
                str(directory / "m4-community-gates.tar.zst"),
                "--workspace-remap",
                remap,
                "--no-tests=pass",
                "--partition",
                partition,
            ]
        )
    return commands


def command_groups_for(
    suite_name: str,
    plan: object,
    cargo_metadata: object,
    *,
    partition: str | None = None,
    archive_file: str | None = None,
    workspace_remap: str | None = None,
) -> list[list[str]]:
    """Translate a plan suite into validated argv vectors."""

    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise PlanError("unsupported or malformed test plan")
    packages, _directories = _workspace(cargo_metadata)
    plan_key = suite_name.replace("-", "_")
    suite = plan.get(plan_key)
    if not isinstance(suite, dict):
        raise PlanError(f"test plan has no suite {suite_name!r}")
    mode = suite.get("mode")
    if mode == "skip":
        return []

    cargo = ["cargo", "nextest", "run"]
    if partition is not None:
        match = re.fullmatch(r"slice:([1-9][0-9]*)/([1-9][0-9]*)", partition)
        if match is None or int(match.group(1)) > int(match.group(2)):
            raise PlanError(f"invalid nextest partition: {partition!r}")
        if suite_name != "workspace-lib":
            raise PlanError("nextest partition is only supported for workspace-lib")

    if archive_file is None and workspace_remap is not None:
        raise PlanError("workspace-remap requires an archive-file")
    if archive_file is not None:
        if suite_name != "workspace-lib":
            raise PlanError("nextest archives are only supported for workspace-lib")
        if workspace_remap is None:
            raise PlanError("archive-file requires workspace-remap")
        archive_file = _validated_path_argument(
            archive_file,
            name="archive-file",
        )
        workspace_remap = _validated_path_argument(
            workspace_remap,
            name="workspace-remap",
        )

    def workspace_command(arguments: list[str]) -> list[str]:
        if partition is None:
            return arguments
        return [*arguments, "--partition", partition]

    if suite_name == "workspace-lib":
        if archive_file is not None:
            archived = [
                *cargo,
                "--archive-file",
                archive_file,
                "--workspace-remap",
                workspace_remap,
            ]
            if mode == "full":
                return [workspace_command(archived)]
            if mode == "packages":
                _validated_package_names(suite.get("packages"), packages)
                return [workspace_command(archived)]
            if mode == "filterset":
                filterset = suite.get("filterset")
                if not isinstance(filterset, str) or not filterset:
                    raise PlanError("workspace filterset is empty")
                _validated_package_names(suite.get("packages"), packages)
                return [
                    workspace_command(
                        [*archived, "-E", filterset, "--no-tests=pass"]
                    )
                ]
            raise PlanError(f"unknown workspace-lib mode: {mode!r}")
        if mode == "full":
            return [
                workspace_command(
                    [
                        *cargo,
                        "--workspace",
                        "--exclude",
                        "wenlan-app",
                        "--lib",
                        "--bin",
                        "wenlan-server",
                    ]
                )
            ]
        if mode == "packages":
            names = _validated_package_names(suite.get("packages"), packages)
            command = [*cargo, *_package_args(names), "--lib"]
            if "wenlan-server" in names:
                command.extend(["--bin", "wenlan-server"])
            return [workspace_command(command)]
        if mode == "filterset":
            filterset = suite.get("filterset")
            if not isinstance(filterset, str) or not filterset:
                raise PlanError("workspace filterset is empty")
            names = _validated_package_names(suite.get("packages"), packages)
            return [
                workspace_command(
                    [*cargo, *_package_args(names), "--lib", "-E", filterset]
                )
            ]
        raise PlanError(f"unknown workspace-lib mode: {mode!r}")

    if suite_name == "cli-server-integration":
        if mode == "full":
            names = sorted(CLI_SERVER_PACKAGES)
            return [[*cargo, *_package_args(names), "-E", "kind(test)"]]
        if mode == "packages":
            names = _validated_package_names(
                suite.get("packages"),
                packages,
                allowed=CLI_SERVER_PACKAGES,
            )
            return [[*cargo, *_package_args(names), "-E", "kind(test)"]]
        if mode == "targets":
            raw_targets = suite.get("targets")
            if not isinstance(raw_targets, dict) or not raw_targets:
                raise PlanError("CLI/server target plan is empty")
            commands = []
            for package_name in sorted(raw_targets):
                if package_name not in CLI_SERVER_PACKAGES:
                    raise PlanError(
                        f"integration target package has no owner: {package_name}"
                    )
                names = raw_targets[package_name]
                if (
                    not isinstance(names, list)
                    or not names
                    or not all(isinstance(name, str) and name for name in names)
                ):
                    raise PlanError(
                        f"integration targets for {package_name} are malformed"
                    )
                known = _integration_targets(packages[package_name])
                unknown = set(names) - known
                if unknown:
                    raise PlanError(
                        f"unknown integration targets for {package_name}: "
                        f"{sorted(unknown)}"
                    )
                target_args = [
                    argument
                    for target in sorted(set(names))
                    for argument in ("--test", target)
                ]
                commands.append(
                    [*cargo, "-p", package_name, *target_args]
                )
            return commands
        raise PlanError(f"unknown cli-server-integration mode: {mode!r}")

    if suite_name == "core-integration":
        if "wenlan-core" not in packages:
            raise PlanError("Cargo metadata has no wenlan-core package")
        known = _integration_targets(packages["wenlan-core"])
        if mode == "full":
            targets = sorted(known - MANUAL_CORE_INTEGRATION)
        elif mode == "targets":
            raw_targets = suite.get("targets")
            if (
                not isinstance(raw_targets, list)
                or not raw_targets
                or not all(
                    isinstance(target, str) and target for target in raw_targets
                )
            ):
                raise PlanError("core integration target plan is empty")
            targets = sorted(set(raw_targets))
            unknown = set(targets) - known
            if unknown:
                raise PlanError(
                    f"unknown core integration targets: {sorted(unknown)}"
                )
            forbidden = set(targets) & MANUAL_CORE_INTEGRATION
            if forbidden:
                raise PlanError(
                    f"manual-only core targets entered required CI: "
                    f"{sorted(forbidden)}"
                )
        else:
            raise PlanError(f"unknown core-integration mode: {mode!r}")
        if not targets:
            raise PlanError("required core integration inventory is empty")
        target_args = [
            argument
            for target in targets
            for argument in ("--test", target)
        ]
        return [
            [
                *cargo,
                "-p",
                "wenlan-core",
                "--features",
                "eval-harness",
                *target_args,
            ]
        ]

    if suite_name == "contract-integration":
        if mode == "full":
            names = sorted(CONTRACT_PACKAGES)
            return [[*cargo, *_package_args(names), "-E", "kind(test)"]]
        if mode == "packages":
            names = _validated_package_names(
                suite.get("packages"),
                packages,
                allowed=CONTRACT_PACKAGES,
            )
            return [[*cargo, *_package_args(names), "-E", "kind(test)"]]
        if mode == "targets":
            raw_targets = suite.get("targets")
            if not isinstance(raw_targets, dict) or not raw_targets:
                raise PlanError("contract integration target plan is empty")
            commands = []
            for package_name in sorted(raw_targets):
                if package_name not in CONTRACT_PACKAGES:
                    raise PlanError(
                        f"contract target package has no owner: {package_name}"
                    )
                names = raw_targets[package_name]
                if (
                    not isinstance(names, list)
                    or not names
                    or not all(isinstance(name, str) and name for name in names)
                ):
                    raise PlanError(
                        f"contract targets for {package_name} are malformed"
                    )
                known = _integration_targets(packages[package_name])
                unknown = set(names) - known
                if unknown:
                    raise PlanError(
                        f"unknown contract targets for {package_name}: {sorted(unknown)}"
                    )
                target_args = [
                    argument
                    for target in sorted(set(names))
                    for argument in ("--test", target)
                ]
                commands.append([*cargo, "-p", package_name, *target_args])
            return commands
        raise PlanError(f"unknown contract-integration mode: {mode!r}")

    raise PlanError(f"unknown suite name: {suite_name!r}")


def _load_json_file(path: str) -> object:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot read JSON from {path}: {error}") from error


def _cargo_metadata() -> object:
    result = subprocess.run(
        [
            _cargo_executable(),
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--no-deps",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PlanError("cargo metadata emitted invalid JSON") from error


def _require_nextest_match(command: list[str], *, label: str) -> None:
    if command[:3] != ["cargo", "nextest", "run"]:
        raise PlanError(f"cannot validate non-nextest command: {command!r}")
    unpartitioned = command
    if "--partition" in command:
        index = command.index("--partition")
        if index + 1 >= len(command):
            raise PlanError("nextest partition has no value")
        unpartitioned = [*command[:index], *command[index + 2 :]]
    # The run may legitimately have an empty shard after the unpartitioned
    # selector was proven non-empty. `nextest list` does not accept this flag.
    unpartitioned = [
        argument for argument in unpartitioned if argument != "--no-tests=pass"
    ]
    list_command = [
        _cargo_executable(),
        "nextest",
        "list",
        *unpartitioned[3:],
        "--message-format",
        "json",
    ]
    print("+", " ".join(list_command), flush=True)
    result = subprocess.run(
        list_command,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        listing = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PlanError("cargo nextest list emitted invalid JSON") from error
    suites = listing.get("rust-suites") if isinstance(listing, dict) else None
    matched = 0
    if isinstance(suites, dict):
        for suite in suites.values():
            testcases = suite.get("testcases") if isinstance(suite, dict) else None
            if not isinstance(testcases, dict):
                continue
            matched += sum(
                isinstance(testcase, dict)
                and testcase.get("ignored") is False
                and isinstance(testcase.get("filter-match"), dict)
                and testcase["filter-match"].get("status") == "matches"
                for testcase in testcases.values()
            )
    if matched == 0:
        raise PlanError(f"{label} selected zero tests; archived ownership has drifted")
    print(f"{label} matched {matched} tests")


def _require_filterset_match(command: list[str]) -> None:
    _require_nextest_match(command, label="workspace-lib filterset")


def _write_github_output(path: str, plan: object, plan_json: str) -> None:
    if "\n" in plan_json or "\r" in plan_json:
        raise PlanError("compact plan JSON unexpectedly contains a newline")
    required = required_suite_outputs(plan)
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(f"test-plan={plan_json}\n")
        for name, value in required.items():
            output.write(f"{name}={'true' if value else 'false'}\n")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--changed-files-json", required=True)
    plan_parser.add_argument("--metadata-file", required=True)
    plan_parser.add_argument("--event-name", required=True)
    plan_parser.add_argument(
        "--scope", choices=("canonical", "platform"), default="canonical"
    )
    plan_parser.add_argument("--github-output")

    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--plan-json", required=True)
    archive_parser.add_argument("--archive-file", required=True)

    macos_archive_parser = subparsers.add_parser("macos-archive")
    macos_archive_parser.add_argument("--plan-json", required=True)
    macos_archive_parser.add_argument("--archive-dir", required=True)
    macos_archive_parser.add_argument(
        "--include-m4", choices=("true", "false"), required=True
    )

    macos_run_parser = subparsers.add_parser("macos-run")
    macos_run_parser.add_argument("--plan-json", required=True)
    macos_run_parser.add_argument("--archive-dir", required=True)
    macos_run_parser.add_argument("--workspace-remap", required=True)
    macos_run_parser.add_argument("--partition", required=True)
    macos_run_parser.add_argument(
        "--include-m4", choices=("true", "false"), required=True
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--suite",
        required=True,
        choices=(
            "workspace-lib",
            "cli-server-integration",
            "core-integration",
            "contract-integration",
        ),
    )
    run_plan_source = run_parser.add_mutually_exclusive_group(required=True)
    run_plan_source.add_argument("--plan-json")
    run_plan_source.add_argument("--plan-env")
    run_parser.add_argument("--partition")
    run_parser.add_argument("--archive-file")
    run_parser.add_argument("--workspace-remap")

    clippy_parser = subparsers.add_parser("clippy")
    clippy_parser.add_argument("--plan-json", required=True)

    local_parser = subparsers.add_parser("local")
    local_parser.add_argument("--changed-files-json", required=True)
    local_parser.add_argument("--level", choices=("commit", "push"), required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "plan":
        try:
            changed_paths = json.loads(arguments.changed_files_json)
        except json.JSONDecodeError as error:
            raise PlanError("changed-files-json is invalid") from error
        if not isinstance(changed_paths, list):
            raise PlanError("changed-files-json must be an array")
        planner = build_platform_plan if arguments.scope == "platform" else build_plan
        plan = planner(
            changed_paths,
            _load_json_file(arguments.metadata_file),
            event_name=arguments.event_name,
        )
        plan_json = json.dumps(plan, separators=(",", ":"), sort_keys=True)
        print(plan_json)
        if arguments.github_output:
            _write_github_output(arguments.github_output, plan, plan_json)
        return 0

    metadata = _cargo_metadata()
    if arguments.command == "local":
        try:
            changed_paths = json.loads(arguments.changed_files_json)
        except json.JSONDecodeError as error:
            raise PlanError("changed-files-json is invalid") from error
        if not isinstance(changed_paths, list) or not changed_paths:
            raise PlanError("changed-files-json must be a non-empty array")
        plan = build_plan(changed_paths, metadata, event_name="pull_request")
        commands = [clippy_command_for(plan, metadata)]
        if arguments.level == "push":
            if _workspace_suite_mode(plan) == "full":
                print(
                    "local checks: plan widened to the full workspace; "
                    "CI owns that suite",
                    flush=True,
                )
            commands.extend(local_test_commands_for(plan, metadata))
        commands = [command for command in commands if command]
        if not commands:
            print("local checks: no affected Rust packages")
            return 0
        for command in commands:
            executable_command = _executable_command(command)
            print("+", " ".join(executable_command), flush=True)
            subprocess.run(executable_command, check=True)
        return 0

    plan_json = arguments.plan_json
    if arguments.command == "run" and arguments.plan_env is not None:
        plan_json = os.environ.get(arguments.plan_env)
        if plan_json is None:
            raise PlanError(f"plan environment variable is unset: {arguments.plan_env}")
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as error:
        raise PlanError("plan-json is invalid") from error
    if arguments.command == "clippy":
        command = clippy_command_for(plan, metadata)
        if not command:
            print("clippy: no affected Rust packages")
            return 0
        executable_command = _executable_command(command)
        print("+", " ".join(executable_command), flush=True)
        subprocess.run(executable_command, check=True)
        return 0
    if arguments.command == "archive":
        command = archive_command_for(
            plan,
            metadata,
            archive_file=arguments.archive_file,
        )
        if not command:
            print("workspace-lib: no affected tests to archive")
            return 0
        executable_command = _executable_command(command)
        print("+", " ".join(executable_command), flush=True)
        subprocess.run(executable_command, check=True)
        return 0
    if arguments.command == "macos-archive":
        archive_dir = _validated_path_argument(
            arguments.archive_dir,
            name="archive-dir",
        )
        Path(archive_dir).mkdir(parents=True, exist_ok=True)
        commands = macos_archive_commands_for(
            plan,
            metadata,
            archive_dir=archive_dir,
            include_m4=arguments.include_m4 == "true",
        )
        if not commands:
            raise PlanError("macOS archive route selected no tests")
        for command in commands:
            executable_command = _executable_command(command)
            print("+", " ".join(executable_command), flush=True)
            subprocess.run(executable_command, check=True)
        return 0
    if arguments.command == "macos-run":
        commands = macos_archive_run_commands_for(
            plan,
            metadata,
            archive_dir=arguments.archive_dir,
            workspace_remap=arguments.workspace_remap,
            partition=arguments.partition,
            include_m4=arguments.include_m4 == "true",
        )
        if not commands:
            raise PlanError("macOS archive route selected no tests")
        for index, command in enumerate(commands, start=1):
            _require_nextest_match(command, label=f"macOS archive {index}")
            executable_command = _executable_command(command)
            print("+", " ".join(executable_command), flush=True)
            subprocess.run(executable_command, check=True)
        return 0

    commands = command_groups_for(
        arguments.suite,
        plan,
        metadata,
        partition=arguments.partition,
        archive_file=arguments.archive_file,
        workspace_remap=arguments.workspace_remap,
    )
    if not commands:
        print(f"{arguments.suite}: no affected tests")
        return 0
    for command in commands:
        suite_key = arguments.suite.replace("-", "_")
        suite = plan.get(suite_key) if isinstance(plan, dict) else None
        if (
            arguments.suite == "workspace-lib"
            and isinstance(suite, dict)
            and suite.get("mode") == "filterset"
        ):
            _require_filterset_match(command)
        executable_command = _executable_command(command)
        print("+", " ".join(executable_command), flush=True)
        subprocess.run(executable_command, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (PlanError, subprocess.CalledProcessError) as error:
        print(f"ci_test_plan: {error}", file=sys.stderr)
        raise SystemExit(1)
