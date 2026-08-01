#!/usr/bin/env python3
"""Fail-loud static contracts for release promotion and public install paths."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
RELEASE_PLEASE_PATH = REPO_ROOT / ".github" / "workflows" / "release-please.yml"

EXPECTED_NODE24_ACTIONS = {
    "actions/checkout": "d23441a48e516b6c34aea4fa41551a30e30af803",
    "actions/upload-artifact": "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    "actions/download-artifact": "3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    "actions/setup-node": "249970729cb0ef3589644e2896645e5dc5ba9c38",
    "docker/setup-buildx-action": "bb05f3f5519dd87d3ba754cc423b652a5edd6d2c",
    "docker/login-action": "dbcb813823bdd20940b903addbd779551569679f",
    "docker/build-push-action": "53b7df96c91f9c12dcc8a07bcb9ccacbed38856a",
    "softprops/action-gh-release": "3d0d9888cb7fd7b750713d6e236d1fcb99157228",
    "googleapis/release-please-action": "0dfd8538845b8e92600d271a895a5372865d4062",
}


def job_body(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        re.MULTILINE | re.DOTALL,
    )
    return match.group("body") if match else ""


def contract_violations(release: str, release_please: str) -> list[str]:
    violations: list[str] = []
    docker = job_body(release, "docker")
    manifest = job_body(release, "docker-manifest")
    finalize = job_body(release, "finalize-release")
    prepare = job_body(release, "prepare-release")
    homebrew = job_body(release, "update-homebrew")

    if "    needs: prepare-release" not in docker:
        violations.append("Docker architecture builds no longer start after prepare-release")
    required_manifest_needs = (
        "    needs: [docker, release, publish-crates, publish-npm, update-homebrew]"
    )
    if required_manifest_needs not in manifest:
        violations.append(
            "GHCR promotion dependencies do not include every required publish channel"
        )
    if "    needs: docker-manifest" not in finalize:
        violations.append("GitHub release finalization bypasses the GHCR promotion barrier")

    for marker in [
        'docker buildx imagetools create -t "$IMG:$TAG"',
        'if [[ "$TAG" == *-* ]]',
        'docker buildx imagetools create -t "$IMG:latest"',
    ]:
        if marker not in manifest:
            violations.append(
                "GHCR promotion does not preserve the version alias while protecting latest from prerelease tags"
            )
            break

    gate_index = prepare.find("- name: Gate the release behind a prerelease")
    tap_index = prepare.find("- name: Require a public Homebrew tap")
    tap_markers = [
        '[[ -z "$HOMEBREW_TAP_TOKEN" ]]',
        "GIT_TERMINAL_PROMPT: \"0\"",
        "git ls-remote https://github.com/7xuanlu/homebrew-tap.git HEAD",
        "not anonymously readable",
    ]
    if (
        gate_index < 0
        or tap_index <= gate_index
        or any(marker not in prepare for marker in tap_markers)
    ):
        violations.append(
            "public Homebrew tap preflight is missing or runs before the release is safely demoted"
        )

    homebrew_markers = [
        "    runs-on: macos-14",
        "git clone https://github.com/7xuanlu/homebrew-tap.git public-tap",
        "cmp tap/Formula/wenlan.rb public-tap/Formula/wenlan.rb",
        "cmp tap/Formula/wenlan-mcp.rb public-tap/Formula/wenlan-mcp.rb",
        "brew install 7xuanlu/tap/wenlan 7xuanlu/tap/wenlan-mcp",
        "brew test 7xuanlu/tap/wenlan",
        "brew test 7xuanlu/tap/wenlan-mcp",
    ]
    if any(marker not in homebrew for marker in homebrew_markers):
        violations.append(
            "Homebrew installation proof does not anonymously install and test both formulas on macOS arm64"
        )

    action_documents = release + "\n" + release_please
    seen: set[str] = set()
    for action, reference in re.findall(r"uses:\s*([^@\s]+)@([^\s#]+)", action_documents):
        expected = EXPECTED_NODE24_ACTIONS.get(action)
        if expected is None:
            continue
        seen.add(action)
        if reference != expected:
            violations.append(
                f"Node 24 action {action} uses mutable or unexpected reference {reference}"
            )
    missing = sorted(EXPECTED_NODE24_ACTIONS.keys() - seen)
    if missing:
        violations.append(f"required Node 24 action pins are absent: {', '.join(missing)}")

    return violations


def release_cache_contract_violations(ci: str, release: str) -> list[str]:
    """Keep the main-owned release cache usable by the tag consumer."""

    violations: list[str] = []
    producer = job_body(ci, "release-preflight")
    consumer = job_body(release, "release")
    shared_markers = [
        "uses: Swatinem/rust-cache@e18b497796c12c097a38f9edb9d0641fb99eee32",
        "shared-key: release-v2-${{ matrix.target }}",
        "workspaces: . -> target/${{ matrix.target }}",
        'cache-workspace-crates: "false"',
    ]
    for owner, body in (("CI producer", producer), ("tag consumer", consumer)):
        missing = [marker for marker in shared_markers if marker not in body]
        if missing:
            violations.append(
                f"{owner} Windows release cache contract is incomplete: {', '.join(missing)}"
            )

    producer_markers = [
        'cache-all-crates: "true"',
        "cache-targets: ${{ matrix.target == 'x86_64-pc-windows-msvc' }}",
        "save-if: ${{ github.ref == 'refs/heads/main' }}",
    ]
    if any(marker not in producer for marker in producer_markers):
        violations.append("CI no longer owns the bounded Windows release target cache")

    consumer_markers = [
        "cache-all-crates: ${{ matrix.target == 'x86_64-pc-windows-msvc' }}",
        "cache-targets: ${{ matrix.target == 'x86_64-pc-windows-msvc' }}",
        'save-if: "false"',
    ]
    if any(marker not in consumer for marker in consumer_markers):
        violations.append("tag release no longer restores the Windows cache read-only")

    return violations


def assert_mutation_detected(
    release: str,
    release_please: str,
    old: str,
    new: str,
    expected: str,
    *,
    mutate_release_please: bool = False,
) -> None:
    source = release_please if mutate_release_please else release
    if old not in source:
        raise AssertionError(f"mutation fixture is stale; missing {old!r}")
    mutated = source.replace(old, new, 1)
    candidate_release = release if mutate_release_please else mutated
    candidate_release_please = mutated if mutate_release_please else release_please
    violations = contract_violations(candidate_release, candidate_release_please)
    if not any(expected in violation for violation in violations):
        raise AssertionError(
            f"mutation did not exercise {expected!r}: {violations!r}"
        )


def main() -> None:
    ci = CI_PATH.read_text(encoding="utf-8")
    release = RELEASE_PATH.read_text(encoding="utf-8")
    release_please = RELEASE_PLEASE_PATH.read_text(encoding="utf-8")
    violations = contract_violations(release, release_please)
    violations.extend(release_cache_contract_violations(ci, release))
    if violations:
        raise AssertionError("release workflow contract drift:\n" + "\n".join(violations))

    assert_mutation_detected(
        release,
        release_please,
        "needs: [docker, release, publish-crates, publish-npm, update-homebrew]",
        "needs: [docker, release]",
        "promotion dependencies",
    )
    assert_mutation_detected(
        release,
        release_please,
        'if [[ "$TAG" == *-* ]]',
        "if false",
        "protecting latest",
    )
    assert_mutation_detected(
        release,
        release_please,
        "git ls-remote https://github.com/7xuanlu/homebrew-tap.git HEAD",
        "git ls-remote authenticated-only",
        "public Homebrew tap preflight",
    )
    assert_mutation_detected(
        release,
        release_please,
        "brew test 7xuanlu/tap/wenlan-mcp",
        "echo skipped-wenlan-mcp-test",
        "Homebrew installation proof",
    )
    assert_mutation_detected(
        release,
        release_please,
        "googleapis/release-please-action@0dfd8538845b8e92600d271a895a5372865d4062",
        "googleapis/release-please-action@v5",
        "mutable or unexpected",
        mutate_release_please=True,
    )
    mutated_ci = ci.replace(
        "workspaces: . -> target/${{ matrix.target }}",
        "workspaces: . -> target",
        1,
    )
    cache_violations = release_cache_contract_violations(mutated_ci, release)
    if not any("CI producer" in violation for violation in cache_violations):
        raise AssertionError(
            "mutation did not exercise explicit-target cache mapping: "
            f"{cache_violations!r}"
        )
    print("PASS: release promotion, Homebrew, and Node 24 action contracts")


if __name__ == "__main__":
    main()
