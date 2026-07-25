#!/usr/bin/env python3
"""Build fail-closed Rust test plans from a Git diff and Cargo metadata."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from typing import Iterable


class PlanError(ValueError):
    """Planner input is malformed and CI must stop."""


ISOLATED_UNIT_MODULES = {
    "crates/wenlan-core/src/lint/pages/security_test.rs": (
        "wenlan-core",
        "lint::pages::fs::tests::security_cases",
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
MANUAL_CORE_INTEGRATION = {"cached_scenario_db_check", "eval_harness"}


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
        "workspace_lib": {"mode": "full"},
        "cli_server_integration": {"mode": "full"},
        "core_integration": {"mode": "full"},
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
) -> dict:
    for package in broad_packages:
        isolated_filters.pop(package, None)
    expressions = [f"package({package})" for package in sorted(broad_packages)]
    for package in sorted(isolated_filters):
        for prefix in sorted(isolated_filters[package]):
            expressions.append(f"package({package}) & test(/^{prefix}::/)")
    if not expressions:
        return {"mode": "skip"}
    if isolated_filters:
        return {
            "mode": "filterset",
            "packages": sorted(broad_packages | set(isolated_filters)),
            "filterset": " | ".join(expressions),
        }
    return {"mode": "packages", "packages": sorted(broad_packages)}


def build_plan(
    changed_paths: Iterable[object],
    cargo_metadata: object,
    *,
    event_name: str,
    existing_paths: set[str] | None = None,
) -> dict:
    """Return a deterministic test plan or raise PlanError for malformed input."""

    packages, directories = _workspace(cargo_metadata)
    reverse = _reverse_dependencies(packages)
    paths = [_normalize_path(path) for path in changed_paths]
    if event_name != "pull_request":
        return _full_plan(f"{event_name} keeps the full backstop")
    if not paths:
        raise PlanError("changed path inventory is empty")
    if existing_paths is None:
        existing = {path for path in paths if Path(path).exists()}
    else:
        existing = {_normalize_path(path) for path in existing_paths}

    broad_packages: set[str] = set()
    isolated_filters: dict[str, set[str]] = defaultdict(set)
    cli_server_packages: set[str] = set()
    cli_server_targets: dict[str, set[str]] = defaultdict(set)
    core_full = False
    core_targets: set[str] = set()
    reasons: list[str] = []

    for path in paths:
        if (
            path in FULL_INPUTS
            or path.endswith("/Cargo.toml")
            or path.endswith("/build.rs")
            or PurePosixPath(path).suffix.lower() in NATIVE_SUFFIXES
        ):
            return _full_plan(f"shared build or native input changed: {path}")
        if path in SHARED_TEST_HELPERS:
            return _full_plan(f"shared test helper changed: {path}")

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
            core_full = core_full or "wenlan-core" in closure
            reasons.append(f"source change selects reverse dependency closure: {path}")
            continue

        return _full_plan(f"unclassified crate input changed: {path}")

    for package_name in cli_server_packages:
        cli_server_targets.pop(package_name, None)

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

    return {
        "version": 1,
        "mode": "differential",
        "reasons": sorted(set(reasons)),
        "workspace_lib": _workspace_lib_plan(broad_packages, isolated_filters),
        "cli_server_integration": cli_server_plan,
        "core_integration": core_plan,
    }


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


def command_groups_for(
    suite_name: str,
    plan: object,
    cargo_metadata: object,
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
    if suite_name == "workspace-lib":
        if mode == "full":
            return [[*cargo, "--workspace", "--lib"]]
        if mode == "packages":
            names = _validated_package_names(suite.get("packages"), packages)
            return [[*cargo, *_package_args(names), "--lib"]]
        if mode == "filterset":
            filterset = suite.get("filterset")
            if not isinstance(filterset, str) or not filterset:
                raise PlanError("workspace filterset is empty")
            names = _validated_package_names(suite.get("packages"), packages)
            return [[*cargo, *_package_args(names), "--lib", "-E", filterset]]
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


def _require_filterset_match(command: list[str]) -> None:
    if command[:3] != ["cargo", "nextest", "run"]:
        raise PlanError(f"cannot validate non-nextest command: {command!r}")
    list_command = [
        _cargo_executable(),
        "nextest",
        "list",
        *command[3:],
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
        raise PlanError(
            "workspace-lib filterset selected zero tests; "
            "isolated module ownership has drifted"
        )
    print(f"workspace-lib filterset matched {matched} tests")


def _write_github_output(path: str, plan_json: str) -> None:
    if "\n" in plan_json or "\r" in plan_json:
        raise PlanError("compact plan JSON unexpectedly contains a newline")
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(f"test-plan={plan_json}\n")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--changed-files-json", required=True)
    plan_parser.add_argument("--metadata-file", required=True)
    plan_parser.add_argument("--event-name", required=True)
    plan_parser.add_argument("--github-output")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--suite",
        required=True,
        choices=(
            "workspace-lib",
            "cli-server-integration",
            "core-integration",
        ),
    )
    run_parser.add_argument("--plan-json", required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "plan":
        try:
            changed_paths = json.loads(arguments.changed_files_json)
        except json.JSONDecodeError as error:
            raise PlanError("changed-files-json is invalid") from error
        if not isinstance(changed_paths, list):
            raise PlanError("changed-files-json must be an array")
        plan = build_plan(
            changed_paths,
            _load_json_file(arguments.metadata_file),
            event_name=arguments.event_name,
        )
        plan_json = json.dumps(plan, separators=(",", ":"), sort_keys=True)
        print(plan_json)
        if arguments.github_output:
            _write_github_output(arguments.github_output, plan_json)
        return 0

    try:
        plan = json.loads(arguments.plan_json)
    except json.JSONDecodeError as error:
        raise PlanError("plan-json is invalid") from error
    commands = command_groups_for(arguments.suite, plan, _cargo_metadata())
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
