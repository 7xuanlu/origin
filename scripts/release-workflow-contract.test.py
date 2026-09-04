#!/usr/bin/env python3
"""Fail-loud static contracts for release promotion and public install paths."""

from __future__ import annotations

import atexit
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parent.parent
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
RELEASE_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
RELEASE_PLEASE_PATH = REPO_ROOT / ".github" / "workflows" / "release-please.yml"
RELEASE_PLEASE_CONFIG_PATH = REPO_ROOT / "release-please-config.json"
FAST_MAINTENANCE_PATH = (
    REPO_ROOT / ".github" / "workflows" / "release-pr-maintenance.yml"
)
OBSERVER_PATH = REPO_ROOT / ".github" / "workflows" / "release-candidate-observer.yml"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate-release-candidate.py"
CLASSIFIER_PATH = REPO_ROOT / "scripts" / "classify-release-candidate.py"
ARCHIVE_PATH = REPO_ROOT / "scripts" / "release_archive.py"
PROMOTION_PATH = REPO_ROOT / "scripts" / "release-promotion.py"
SYNC_RELEASE_PR_PATH = REPO_ROOT / "scripts" / "sync-release-pr.py"
SIGNPATH_STATUS_PATH = REPO_ROOT / ".github" / "workflows" / "signpath-status.yml"
RUNTIME_IMAGE_PATH = REPO_ROOT / "scripts" / "verify-release-runtime-image.py"
PUBLISH_CRATE_TEST_PATH = REPO_ROOT / "scripts" / "publish-crate.test.py"

# The Windows installer's Authenticode signer. It is in this allowlist because
# the loop that reads the allowlist SKIPS any action missing from it, so an
# action absent here is an action nobody checks the pin of.
SIGNPATH_ACTION = "SignPath/github-action-submit-signing-request"
SIGNPATH_ACTION_SHA = "c92b958760219087e01f8d67a1669ed57afe2627"

EXPECTED_NODE24_ACTIONS = {
    SIGNPATH_ACTION: SIGNPATH_ACTION_SHA,
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


def named_step_body(job: str, step_name: str) -> str:
    match = re.search(
        rf"^      - name: {re.escape(step_name)}\n(?P<body>.*?)(?=^      - (?:name:|uses:)|^      #|^  #|\Z)",
        job,
        re.MULTILINE | re.DOTALL,
    )
    return match.group("body").strip() if match else ""


def heredoc_body(job: str, path: str) -> str:
    """The body of `cat > <path> << 'TAG'` inside a job, without the delimiters."""
    match = re.search(
        rf"^\s*cat > {re.escape(path)} << '(?P<tag>[A-Z_]+)'\n(?P<body>.*?)^\s*(?P=tag)\s*$",
        job,
        re.MULTILINE | re.DOTALL,
    )
    return match.group("body") if match else ""


def contract_violations(
    ci: str,
    release: str,
    release_please: str,
    release_please_config: str,
    fast_maintenance: str,
    promotion: str,
    sync_release_pr: str,
) -> list[str]:
    """Keep release publication bound to the PR-built immutable archives."""

    violations: list[str] = []
    try:
        config = json.loads(release_please_config)
    except json.JSONDecodeError:
        config = {}
    if config.get("packages", {}).get(".", {}).get("always-update") is not True:
        violations.append("release-please package always-update is not exact true")
    if "- '.github/workflows/release-pr-maintenance.yml'" not in ci:
        violations.append("fast release maintenance cannot bootstrap its Rust contract")
    if "associated_pulls=associated" not in promotion:
        violations.append(
            "main release gate does not reuse one commit association snapshot"
        )
    ci_gate = named_step_body(job_body(ci, "detect-changes"), "Verify reusable release merge")
    for marker in [
        "python3 scripts/release-promotion.py gate-main",
        '--wait-seconds 720',
        '--main-run-id "$GITHUB_RUN_ID"',
        '--main-run-attempt "$GITHUB_RUN_ATTEMPT"',
        '--plan-output "$RUNNER_TEMP/main-release-promotion-receipt.json"',
    ]:
        if marker not in ci_gate:
            violations.append(f"main release gate omits thin receipt contract {marker!r}")
    ci_receipt = named_step_body(
        job_body(ci, "detect-changes"), "Upload main release promotion receipt"
    )
    for marker in [
        "if: steps.release-proof.outputs.release-gate-state == 'validated'",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "name: main-release-promotion-receipt-${{ github.run_id }}-${{ github.run_attempt }}",
        "path: ${{ runner.temp }}/main-release-promotion-receipt.json",
        "compression-level: 0",
        "retention-days: 30",
        "if-no-files-found: error",
        "overwrite: false",
    ]:
        if marker not in ci_receipt:
            violations.append(f"main promotion receipt upload omits {marker!r}")

    if re.search(r"\n\s+(workflow_dispatch|push|pull_request):", release_please):
        violations.append("release-please has a privileged trigger outside completed main CI")
    for marker in [
        "group: release-please-main",
        "cancel-in-progress: false",
    ]:
        if marker not in release_please:
            violations.append(f"release-please concurrency omits {marker!r}")
    for job in ["route-main", "maintain-release-pr", "create-validated-tag"]:
        if not job_body(release_please, job):
            violations.append(f"release-please omits hybrid route job {job!r}")
    route = job_body(release_please, "route-main")
    for marker in [
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.conclusion == 'success'",
        "Verify observed main CI identity",
        ".path == \".github/workflows/ci.yml\"",
        "scripts/release-promotion.py consume-main-receipt",
    ]:
        if marker not in route:
            violations.append(f"release-please main route omits {marker!r}")
    maintain = job_body(release_please, "maintain-release-pr")
    for marker in [
        "needs.route-main.outputs.state == 'ordinary'",
        "group: release-pr-maintenance-main",
        "cancel-in-progress: false",
        "queue: max",
        "skip-github-release: true",
        "contents: write",
        "pull-requests: write",
    ]:
        if marker not in maintain:
            violations.append(f"ordinary release-please path omits PR-only contract {marker!r}")
    for workflow, label, trusted_ref in [
        (fast_maintenance, "fast", "${{ github.sha }}"),
        (release_please, "fallback", "${{ github.sha }}"),
    ]:
        maintenance_job = job_body(workflow, "maintain-release-pr")
        trusted_checkout = named_step_body(
            maintenance_job, "Checkout trusted release PR synchronizer"
        )
        resolver = named_step_body(maintenance_job, "Resolve exact pending Release PR")
        exact_checkout = named_step_body(
            maintenance_job, "Checkout exact release PR head"
        )
        synchronizer = named_step_body(
            maintenance_job, "Merge main and sync release PR branch"
        )
        for marker in [
            "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
            f"ref: {trusted_ref}",
            "persist-credentials: false",
        ]:
            if marker not in trusted_checkout:
                violations.append(f"{label} maintenance trusted checkout omits {marker!r}")
        for marker in [
            "id: release_pr",
            "GITHUB_TOKEN: ${{ github.token }}",
            'cp scripts/sync-release-pr.py "$RUNNER_TEMP/sync-release-pr.py"',
            'cp scripts/bump-version.sh "$RUNNER_TEMP/bump-version.sh"',
            'python3 "$RUNNER_TEMP/sync-release-pr.py" resolve',
            '--repository "$GITHUB_REPOSITORY"',
            '--github-output "$GITHUB_OUTPUT"',
        ]:
            if marker not in resolver:
                violations.append(f"{label} maintenance resolver omits {marker!r}")
        for marker in [
            "if: steps.release_pr.outputs.release_pr_state == 'present'",
            "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
            "ref: ${{ steps.release_pr.outputs.release_pr_head_sha }}",
            "token: ${{ secrets.RELEASE_TOKEN }}",
            "fetch-depth: 0",
            "persist-credentials: true",
        ]:
            if marker not in exact_checkout:
                violations.append(f"{label} maintenance exact checkout omits {marker!r}")
        for marker in [
            "if: steps.release_pr.outputs.release_pr_state == 'present'",
            'python3 "$RUNNER_TEMP/sync-release-pr.py" sync',
            '--expected-head-sha "${{ steps.release_pr.outputs.release_pr_head_sha }}"',
            '--bump-script "$RUNNER_TEMP/bump-version.sh"',
        ]:
            if marker not in synchronizer:
                violations.append(f"{label} maintenance synchronizer omits {marker!r}")
        if "steps.release.outputs.pr" in maintenance_job:
            violations.append(f"{label} maintenance still trusts optional action PR output")
    create_tag = job_body(release_please, "create-validated-tag")
    for marker in [
        "needs.route-main.outputs.state == 'validated'",
        "GH_TOKEN: ${{ secrets.RELEASE_TOKEN }}",
        "MAIN_SHA: ${{ github.event.workflow_run.head_sha }}",
        "tag_lookup_status=$?",
        "'.status | tostring'",
        'if [[ "$tag_api_status" != 404 ]]',
        'if [[ "$pending" != true || "$tagged" != false ]]',
        'if [[ "$existing_sha" != "$MAIN_SHA" ]]',
        '-f ref="refs/tags/$RELEASE_TAG"',
        '-f sha="$MAIN_SHA"',
    ]:
        if marker not in create_tag:
            violations.append(f"validated tag creation omits {marker!r}")
    if re.search(r"git/ref/tags/\$RELEASE_TAG[\s\S]{0,240}\|\| true", create_tag):
        violations.append("validated tag lookup swallows an API failure")

    trigger = re.search(
        r"^on:\n(?P<body>.*?)(?=^concurrency:)",
        fast_maintenance,
        re.MULTILINE | re.DOTALL,
    )
    trigger_lines = (
        []
        if trigger is None
        else [
            line
            for line in trigger.group("body").splitlines()
            if line and not line.startswith("#")
        ]
    )
    if trigger_lines != ["  push:", "    branches: [main]"]:
        violations.append("fast release maintenance trigger is not exact main push")
    fast_job = job_body(fast_maintenance, "maintain-release-pr")
    fast_jobs = fast_maintenance.partition("\njobs:\n")[2]
    if not fast_job or len(re.findall(r"^  [A-Za-z0-9_-]+:\n", fast_jobs, re.MULTILINE)) != 1:
        violations.append("fast release maintenance job inventory is not exactly PR-only")
    for marker in [
        "group: release-pr-maintenance-main",
        "cancel-in-progress: false",
        "queue: max",
        "github.event_name == 'push' && github.ref == 'refs/heads/main'",
        "timeout-minutes: 5",
        "contents: read",
        "pull-requests: read",
        "contents: write",
        "pull-requests: write",
        "googleapis/release-please-action@0dfd8538845b8e92600d271a895a5372865d4062",
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "skip-github-release: true",
        "manifest-file: .release-please-manifest.json",
        "config-file: release-please-config.json",
        "target-branch: main",
        "scripts/sync-release-pr.py",
    ]:
        if marker not in fast_maintenance:
            violations.append(f"fast release maintenance omits {marker!r}")
    for forbidden in [
        "workflow_run:",
        "workflow_dispatch:",
        "pull_request:",
        "create-validated-tag",
        "release-promotion.py",
        "refs/tags/",
        "/git/refs",
        "git tag",
        "gh release",
        "autorelease:",
        "/issues/",
        "--method POST",
        "git push --force",
        "git push -f",
    ]:
        if forbidden in fast_maintenance:
            violations.append(
                f"fast release maintenance contains publishing or unsafe path {forbidden!r}"
            )
    for uses in re.findall(r"\buses:\s+([^\s#]+)", fast_maintenance):
        if re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", uses) is None:
            violations.append(f"fast release maintenance action is not SHA-pinned: {uses!r}")

    for marker in [
        '"X-GitHub-Api-Version": "2026-03-10"',
        '"state": "open"',
        '"base": BASE_BRANCH',
        '"head": f"{RELEASE_AUTHOR}:{RELEASE_BRANCH}"',
        "if owner != RELEASE_AUTHOR:",
        "if not payload:",
        "if len(payload) != 1:",
        'user.get("login") != RELEASE_AUTHOR',
        'head["repo"].get("full_name") != repository',
        "current != expected_head_sha",
        '["git", "fetch", "--no-tags", "origin", BASE_BRANCH]',
        '["git", "merge", "--no-edit", "origin/main"]',
        '["bash", str(bump_script)]',
        '["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{RELEASE_BRANCH}"]',
        "remote_shas != [expected_head_sha]",
        '"HEAD:refs/heads/release-please--branches--main"',
        "except subprocess.CalledProcessError as error:",
    ]:
        if marker not in sync_release_pr:
            violations.append(f"release PR synchronizer omits {marker!r}")
    for forbidden in [
        "git rebase",
        "--force",
        '"-f"',
        "git tag",
        "gh release",
        "refs/tags/",
    ]:
        if forbidden in sync_release_pr:
            violations.append(f"release PR synchronizer contains unsafe mutation {forbidden!r}")

    for marker in [
        "workflow_dispatch:",
        "release_sha:",
        "release_tag:",
        "source_run_id:",
        "source_run_attempt:",
        "release-pr-number: ${{ steps.release-gate.outputs.release-pr-number }}",
        "required: true",
        "group: release-${{ github.event_name == 'workflow_dispatch' && inputs.release_tag || github.ref_name }}",
    ]:
        if marker not in release:
            violations.append(f"release recovery dispatch omits {marker!r}")
    checkout_refs = re.findall(
        r"uses: actions/checkout@[0-9a-f]{40}[^\n]*\n\s+with:\n(?:\s+[^\n]+\n)*?\s+ref: ([^\n]+)",
        release,
    )
    if not checkout_refs or any(
        ref.strip() not in {"${{ github.sha }}", "${{ env.RELEASE_SHA }}"}
        for ref in checkout_refs
    ) or "ref: ${{ env.RELEASE_SHA }}" not in release:
        violations.append("release checkout is not pinned to the immutable control or resolved release SHA")
    resolver_checkout = job_body(release, "resolve-promotion")
    if (
        "ref: ${{ github.sha }}" not in resolver_checkout
        or "git rev-parse HEAD)\" == \"$GITHUB_SHA" not in resolver_checkout
        or "ref: ${{ env.RELEASE_SHA }}" in resolver_checkout
    ):
        violations.append("release resolver is not pinned to its immutable main control SHA")
    # Promotion runs the same resolver as resolve-promotion, so it is control
    # plane too. Pinning it to the release commit would run the release's own
    # copy of the promotion tooling, so a resolver fix could never reach a
    # recovery of that release.
    # Docker packaging is the same shape: its checkout supplies only the
    # runtime Dockerfile and the image verifier, while the bytes placed in the
    # image come from the receipt-verified artifact.
    for job_name in ["promote-assets", "docker"]:
        packaging = job_body(release, job_name)
        if (
            "ref: ${{ github.sha }}" not in packaging
            or "ref: ${{ env.RELEASE_SHA }}" in packaging
        ):
            violations.append(
                f"release packaging job {job_name!r} is not pinned to its main control SHA"
            )
    for job_name in [
        "prepare-release",
        "publish-crates",
        "publish-npm",
    ]:
        job = job_body(release, job_name)
        if "ref: ${{ env.RELEASE_SHA }}" not in job or "ref: ${{ github.sha }}" in job:
            violations.append(f"release source job {job_name!r} is not pinned to RELEASE_SHA")
    bind = job_body(release, "bind-release-tag")
    if (
        "actions: write" not in bind
        or "contents: read" not in bind
        or "issues: read" not in bind
        or "actions/checkout@" in bind
        or "/git/refs" not in bind
        or "GH_TAG_TOKEN: ${{ secrets.RELEASE_TOKEN }}" not in bind
        or "event=push&head_sha=$RELEASE_SHA" not in bind
        or '.head_branch == \\"$RELEASE_TAG\\"' not in bind
        or "/actions/runs/$legacy_run_id/cancel" not in bind
        or "$'completed\\tcancelled'" not in bind
        or "GATE_STATE" not in bind
        or "RELEASE_PR_NUMBER" not in bind
        or 'index("autorelease: pending") != null' not in bind
        or 'index("autorelease: tagged") == null' not in bind
    ):
        violations.append("receipt-derived tag binding lacks isolated write authority")
    # The PAT is deliberately confined to two REST-only sites: the tag bind
    # above, and the release-as cleanup PR in finalize-release (a
    # GITHUB_TOKEN-created branch would not trigger the cleanup PR's required
    # CI). Any third occurrence is drift.
    if release.count("secrets.RELEASE_TOKEN") != 2:
        violations.append(
            "release token is not confined to the exact tag bind and the"
            " release-as cleanup step"
        )
    cleanup = named_step_body(
        job_body(release, "finalize-release"),
        "Open the release-as override cleanup PR",
    )
    if (
        cleanup.count("secrets.RELEASE_TOKEN") != 1
        or "actions/checkout@" in cleanup
        or 'if [[ "$RELEASE_TAG" == *-* ]]' not in cleanup
        or "contents/release-please-config.json?ref=main" not in cleanup
        or "/git/ref/heads/$branch" not in cleanup
    ):
        violations.append(
            "release-as cleanup step is missing, checks out code, or lost its"
            " prerelease/idempotency guards"
        )
    if any(
        marker not in resolver_checkout
        for marker in ["actions: read", "contents: read", "pull-requests: read"]
    ) or "contents: write" in resolver_checkout:
        violations.append("release resolver does not retain read-only token authority")
    if "ref: refs/tags/${{ env.RELEASE_TAG }}" in release:
        violations.append("tag release can checkout a mutable tag ref")
    if job_body(release, "release"):
        violations.append("tag release retains the duplicate release build matrix")
    if "cargo build" in release or "build-release-binaries" in release:
        violations.append("tag release can recompile the PR-validated release binaries")
    if "--dry-run" in job_body(release, "publish-crates"):
        violations.append("tag release duplicates Cargo publish verification")
    crates = job_body(release, "publish-crates")
    for marker in [
        "python3 scripts/publish-crate.py",
        "--package wenlan-types",
        "--package wenlan-mcp",
        '--version "$VERSION"',
    ]:
        if marker not in crates:
            violations.append(f"crates.io publication bypasses publish helper {marker!r}")
    if crates.count("python3 scripts/publish-crate.py") != 2:
        violations.append("crates.io publication does not call the helper exactly twice")
    for forbidden in ["seq 1 60", "sleep 10", "--no-verify"]:
        if forbidden in crates:
            violations.append(
                f"crates.io publication retains unsafe or serial polling {forbidden!r}"
            )
    if "if: env.CARGO_REGISTRY_TOKEN != ''" in crates:
        violations.append("crates.io publication can silently skip a missing credential")
    for job, timeout in {
        "prepare-release": 10,
        "publish-crates": 15,
        "publish-npm": 10,
        "update-homebrew": 20,
        "docker-manifest": 10,
        "finalize-release": 10,
        # Raised to 210 when the notary wait went to 90 minutes. The job
        # compiles the workspace from scratch before it waits, and staples,
        # repacks and verifies after, so 120 could let a slow compile push the
        # notarize step past the job's own cap -- a hard kill that never prints
        # the submission id the operator needs.
        "app-bundle": 210,
        # Raised to 165 when SignPath signing was wired in. A job timeout starts
        # when the JOB starts, so this budgets the whole job, not the tail:
        # ~40 minutes of measured pre-signing work (runs 33043452863,
        # 33121699208, 33284603345 and 33291555184 took 38m21s to 39m00s end to
        # end), 10 for moving a 62 MB installer to SignPath and back, the
        # 5400-second wait the action is given for a human approver, 5 for
        # replace/re-sign/stage/verify, and 20 of margin. Raise the action's
        # wait and this moves with it.
        "app-bundle-windows": 165,
    }.items():
        if not re.search(
            rf"^    timeout-minutes: {timeout}\s*$",
            job_body(release, job),
            re.MULTILINE,
        ):
            violations.append(
                f"release job {job!r} does not keep its {timeout}-minute bound"
            )
    for job in [
        "resolve-promotion",
        "bind-release-tag",
        "prepare-release",
        "app-bundle",
        "app-bundle-windows",
        "promote-assets",
        "promote-app-assets",
        "docker",
        "docker-manifest",
        "finalize-release",
    ]:
        if not job_body(release, job):
            violations.append(f"tag release omits artifact-promotion job {job!r}")
    if "    needs: [resolve-promotion, bind-release-tag]" not in job_body(release, "prepare-release"):
        violations.append("release preparation can start before receipt-derived tag binding")
    if "    needs: [resolve-promotion, bind-release-tag]" not in job_body(release, "app-bundle-windows"):
        violations.append("Windows app bundling can start before receipt-derived tag binding")
    # The CLI wrappers and the desktop app are promoted by two jobs on
    # purpose. Only the app job waits on the desktop builds, so a failed
    # Windows bundle no longer skips Docker, npm, crates and Homebrew with it.
    # The pair of assertions below is what keeps that split honest: the CLI job
    # must NOT wait on an app build, and the app job must wait on both.
    if "    needs: [resolve-promotion, bind-release-tag, prepare-release]" not in job_body(
        release, "promote-assets"
    ):
        violations.append("asset publication bypasses receipt resolution, tag binding, or the prerelease gate")
    if re.search(r"app-bundle", job_body(release, "promote-assets").split("steps:")[0]):
        violations.append(
            "CLI asset publication waits on a desktop app build again; a failed "
            "Windows bundle would take the whole release down with it"
        )
    if "    needs: [resolve-promotion, bind-release-tag, prepare-release, app-bundle, app-bundle-windows]" not in job_body(
        release, "promote-app-assets"
    ):
        violations.append(
            "desktop app promotion bypasses receipt resolution, tag binding, "
            "the prerelease gate, or one of the two app bundles"
        )
    resolve = job_body(release, "resolve-promotion")
    for marker in [
        "scripts/release-promotion.py gate-main",
        "scripts/release-promotion.py consume-main-receipt",
        '--sha "$RELEASE_SHA"',
        ".main_sha == $sha and .main_run == null",
        ".receipt.run_id == $source_run_id",
        ".receipt.run_attempt == $source_run_attempt",
        '"$GITHUB_REF" == "refs/heads/main"',
        "name: release-promotion-plan-${{ github.run_id }}",
        "retention-days: 30",
        "overwrite: true",
    ]:
        if marker not in resolve:
            violations.append(f"tag promotion resolver omits {marker!r}")
    for job_name in [
        "prepare-release",
        "resolve-promotion",
        "bind-release-tag",
        "app-bundle",
        "app-bundle-windows",
        "promote-assets",
        "promote-app-assets",
        "docker",
        "docker-manifest",
        "publish-crates",
        "publish-npm",
        "update-homebrew",
        "finalize-release",
    ]:
        job = job_body(release, job_name)
        for marker in ["/git/ref/tags/$RELEASE_TAG", "RELEASE_SHA"]:
            if marker not in job:
                violations.append(
                    f"publication job {job_name!r} omits immutable tag check {marker!r}"
                )
    prepare = job_body(release, "prepare-release")
    if (
        "Existing stable release cannot enter an incremental publication rerun."
        not in prepare
        or 'isPrerelease' not in prepare
    ):
        violations.append("existing stable release can enter incremental publication")
    promote = job_body(release, "promote-assets")
    for marker in [
        "scripts/release-promotion.py download-assets",
        "Download exact validated wrapper once",
        "Existing release asset $name differs; refusing to clobber.",
        "name: release-promotion-plan-${{ github.run_id }}",
        "name: homebrew-artifacts",
        "promoted-assets/wenlan-darwin-arm64.tar.gz",
        "name: docker-runtime-inputs",
    ]:
        if marker not in promote:
            violations.append(f"validated asset promotion omits {marker!r}")
    # `wenlan background on` needs wenlan-server next to the brewed CLI, so the
    # `wenlan` formula must install from the full darwin archive and ship both.
    # The assertions read the formula heredoc's active lines, not the job
    # text, so a marker moved into a comment cannot satisfy them.
    homebrew = job_body(release, "update-homebrew")
    formula = heredoc_body(homebrew, "tap/Formula/wenlan.rb")
    if not formula:
        violations.append("Homebrew wenlan formula heredoc is missing")
    active_formula = [
        line.strip()
        for line in formula.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for marker in [
        'url "https://github.com/7xuanlu/wenlan/releases/download/vVERSION_PLACEHOLDER/wenlan-darwin-arm64.tar.gz"',
        'bin.install "wenlan", "wenlan-server"',
        'assert_match "wenlan-server", shell_output("#{bin}/wenlan-server --help")',
    ]:
        if marker not in active_formula:
            violations.append(f"Homebrew wenlan formula omits {marker!r}")
    if "wenlan-cli-darwin-arm64.tar.gz" in formula:
        violations.append("Homebrew wenlan formula still installs the CLI-only archive")
    # The checksum the formula is stamped with must come from the same archive.
    if "SHA_DARWIN_ARM64=$(shasum -a 256 wenlan-darwin-arm64.tar.gz" not in homebrew:
        violations.append("Homebrew wenlan formula checksum is not taken from the full darwin archive")
    after_formula = homebrew.split("cat > tap/Formula/wenlan.rb", 1)[-1]
    if 'SHA="$SHA_DARWIN_ARM64" perl' not in after_formula:
        violations.append("Homebrew wenlan formula sha256 is not stamped from SHA_DARWIN_ARM64")

    app_bundle = job_body(release, "app-bundle")
    app_bundle_windows = job_body(release, "app-bundle-windows")
    if "tauri-action" in release:
        violations.append("app bundling must build directly, never via tauri-action")
    for job_name, job in [
        ("app-bundle", app_bundle),
        ("app-bundle-windows", app_bundle_windows),
    ]:
        if "contents: read" not in job:
            violations.append(
                f"{job_name} job does not scope permissions to contents: read"
            )
        if "TAURI_SIGNING_PRIVATE_KEY" not in job:
            violations.append(f"{job_name} job omits the Tauri updater signing key")
    # The signing key belongs to the two jobs that build a signed bundle and
    # nowhere else. Counting occurrences catches a leak into a job that has no
    # business holding it, which a containment check on one job would miss.
    if release.count("TAURI_SIGNING_PRIVATE_KEY") != app_bundle.count(
        "TAURI_SIGNING_PRIVATE_KEY"
    ) + app_bundle_windows.count("TAURI_SIGNING_PRIVATE_KEY"):
        violations.append("Tauri signing key leaks outside the app bundling jobs")
    # An installer that omits what the daemon dynamically loads installs fine
    # and then fails on first use, so the release path proves the payload
    # itself rather than trusting a CI run against a different commit.
    for marker in [
        "onnxruntime.dll",
        "vulkan-1.dll",
        "the Windows installer is missing runtime files the daemon needs",
    ]:
        if marker not in app_bundle_windows:
            violations.append(
                f"Windows app bundling does not prove its installer payload: {marker!r}"
            )
    promote_app = job_body(release, "promote-app-assets")
    for marker in ["latest.json", "darwin-aarch64-app", "windows-x86_64"]:
        if marker not in promote_app:
            violations.append(f"desktop app promotion omits updater manifest {marker!r}")
    for marker in [
        "needs.app-bundle.outputs.dmg_sha256",
        "needs.app-bundle-windows.outputs.setup_sha256",
        "needs.app-bundle-windows.outputs.sig_sha256",
    ]:
        if marker not in promote_app:
            violations.append(
                f"app bundle SHA-256 re-verification is not wired to {marker!r}"
            )
    verify_idx = promote_app.find("Verify app bundle bytes before promotion")
    upload_idx = promote_app.find(
        "Upload desktop app assets and updater metadata without clobbering"
    )
    if verify_idx == -1 or upload_idx == -1:
        violations.append(
            "desktop app promotion omits app bundle SHA-256 re-verification before upload"
        )
    elif verify_idx > upload_idx:
        violations.append("app bundle assets are uploaded before their SHA-256 re-verification")

    docker = job_body(release, "docker")
    if "    needs: [promote-assets, bind-release-tag]" not in docker:
        violations.append("runtime images can start before exact validated asset promotion")
    for marker in [
        "docker/Dockerfile.release-runtime",
        "scripts/verify-release-runtime-image.py",
    ]:
        if marker not in docker:
            violations.append(f"runtime image lane omits binary-reuse proof {marker!r}")
    if "docker/Dockerfile.daemon" in docker or "cargo build" in docker:
        violations.append("runtime image lane can compile a different server binary")
    # The verifier also accepts a published-release-asset digest so CI can smoke
    # the image on a PR. That source is immutable but it is not a receipt, and a
    # release must bind its bytes to the receipt that validated them.
    if "--receipt" not in docker or "--published-digest" in docker:
        violations.append("release runtime image lane does not bind bytes to the closed receipt")
    for job_name, artifact_name in [
        ("promote-assets", "homebrew-artifacts"),
        ("promote-assets", "docker-runtime-inputs"),
        ("docker", "docker-image-digest-${{ matrix.tag-suffix }}"),
    ]:
        job = job_body(release, job_name)
        match = re.search(
            rf"name: {re.escape(artifact_name)}[\s\S]{{0,700}}?overwrite: true",
            job,
        )
        if match is None:
            violations.append(
                f"retryable internal artifact {artifact_name!r} is not overwrite-safe"
            )
    runtime_image = RUNTIME_IMAGE_PATH.read_text(encoding="utf-8")
    for marker in [
        '"--load"',
        "_verify_copied_binary(",
        "_semantic_smoke(",
        "Linux archive bytes differ from the closed receipt",
    ]:
        if marker not in runtime_image:
            violations.append(f"runtime image verifier omits {marker!r}")
    manifest = job_body(release, "docker-manifest")
    if (
        "    needs: [docker, promote-assets, publish-crates, publish-npm, update-homebrew, bind-release-tag]"
        not in manifest
    ):
        violations.append("GHCR promotion dependencies omit a required publish channel")
    npm = job_body(release, "publish-npm")
    if "    needs: [promote-assets, bind-release-tag]" not in npm or "needs: publish-crates" in npm:
        violations.append("npm publishing is serialized behind crates.io propagation")
    finalize = job_body(release, "finalize-release")
    # promote-app-assets is listed here and nowhere downstream. That is the
    # whole fail-closed half of the promote-assets split: the desktop uploads
    # no longer block the CLI channels, so this entry is the only thing left
    # keeping a release whose installers never landed out of releases/latest.
    if "    needs: [docker-manifest, promote-app-assets, bind-release-tag]" not in finalize:
        violations.append(
            "GitHub release finalization bypasses the GHCR promotion barrier or "
            "would promote a release whose desktop app assets never landed"
        )
    # The lifecycle step resolves the merged release PR through
    # GET /commits/{sha}/pulls, then POSTs and DELETEs on /issues/{pr}/labels to
    # move it from pending to tagged. The /issues/ path is only the REST
    # spelling: a pull request routes label WRITES through pull-requests, so
    # `issues: write` alone cannot move them. An explicit permissions block sets
    # every unlisted scope to none, so an under-grant here returns 403 only
    # AFTER `gh release edit` has already promoted the release — stable but
    # still labelled pending, with no way to retry the half that ran. v0.15.4
    # under-granted the read and v0.15.5 the write, each failing one call later
    # than the last. Assert every scope the step actually exercises, at the
    # exact level it exercises it.
    finalize_permissions = re.search(
        r"    permissions:\n(?P<body>(?:      [^\n]+\n)+)", finalize
    )
    granted = {
        line.strip()
        for line in (
            finalize_permissions.group("body") if finalize_permissions else ""
        ).splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for scope in ["contents: write", "issues: write", "pull-requests: write"]:
        if scope not in granted:
            violations.append(
                f"release finalization does not grant {scope!r} for the lifecycle step"
            )

    tagged = finalize.find('"labels":["autorelease: tagged"]')
    pending = finalize.find("labels/autorelease%3A%20pending")
    if tagged < 0 or pending <= tagged:
        violations.append("release lifecycle does not add tagged before removing pending")
    for marker in [
        'index("autorelease: tagged") != null',
        'index("autorelease: pending") == null',
    ]:
        if marker not in finalize:
            violations.append("release lifecycle omits the final closed-state assertion")
    if re.search(
        r"labels/autorelease%3A%20pending[\s\S]{0,120}\|\| true", finalize
    ):
        violations.append("release lifecycle swallows a pending-label deletion failure")

    # A swallowed tag listing yields an empty highest-tag, which silently drops
    # the latest promotion while still reporting success.
    for label, body in (("GHCR", manifest), ("GitHub release", finalize)):
        if "tag_list_status=$?" not in body:
            violations.append(
                f"{label} latest decision does not branch on the tag listing exit status"
            )
        if re.search(r"matching-refs/tags/v[\s\S]{0,200}\|\| true", body):
            violations.append(f"{label} latest decision swallows a tag listing failure")
        if 'if [[ -z "$highest" ]]' not in body:
            violations.append(f"{label} latest decision accepts an empty tag list")

    for marker in [
        'expected_name = f"validated-release-receipt-{run_id}-{run_attempt}"',
        "MAX_RECEIPT_CANDIDATES = 20",
        "observer reruns produced conflicting release semantics",
        "latest trusted observer attempt",
        "main-release-promotion-receipt-{run_id}-",
        "/actions/runs/{run_id}/attempts/{run_attempt}/jobs",
        "main promotion receipt claims a future run attempt",
        "main promotion receipt reruns produced conflicting semantics",
        'subparsers.add_parser("consume-main-receipt")',
        'subparsers.add_parser("download-assets")',
        "validated assets wrapper size or digest mismatch",
        "safe_extract_zip(wrapper, output_dir, expected)",
        "safe_extract_archive(",
    ]:
        if marker not in promotion:
            violations.append(f"release promotion resolver omits fail-closed evidence {marker!r}")
    if "output_dir.mkdir(parents=True, exist_ok=False)" in promotion:
        violations.append("release promotion pre-creates the safe extraction destination")

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
    violations.extend(release_cache_retry_contract_violations(ci))
    violations.extend(signpath_signing_contract_violations(ci, release))
    violations.extend(authenticode_lane_violations(ci))
    return violations


#: The job that runs THIS suite on a host that can answer the Authenticode
#: half of it, and the variable that makes an unanswerable row fatal there.
AUTHENTICODE_LANE_JOB = "windows-release-contract"
AUTHENTICODE_LANE_FLAG = 'WENLAN_REQUIRE_AUTHENTICODE: "1"'


def authenticode_lane_violations(ci: str) -> list[str]:
    """A lane that CAN measure Authenticode must exist, and must be required.

    Everything else in this file checks the release workflow. This checks the
    only thing that decides whether the checks are ever run against a host that
    can perform them.

    The hole it closes: this suite reaches CI through
    scripts/validate-versions.test.sh, which runs in ci.yml's `docs` job on
    ubuntu-24.04. There is no Get-AuthenticodeSignature there, so the truth
    table's signature rows and the mutations only those rows can catch print
    UNCHECKED and the suite exits 0 -- a failed measurement wearing the same
    exit status as a clean one. `WENLAN_REQUIRE_AUTHENTICODE=1` already turns
    those lines fatal; until now nothing set it anywhere, so it was an opt-in
    that nothing opted into.
    """

    violations: list[str] = []
    job = job_body(ci, AUTHENTICODE_LANE_JOB)
    if not job:
        return [
            f"ci.yml has no {AUTHENTICODE_LANE_JOB!r} job, so nothing runs this "
            "suite on a host that can call Get-AuthenticodeSignature. Every "
            "signature row would be UNCHECKED on the Ubuntu docs lane, and "
            "UNCHECKED there exits 0"
        ]
    if "runs-on: windows-" not in job:
        violations.append(
            f"{AUTHENTICODE_LANE_JOB} does not run on Windows; the rows it exists "
            "to make fatal cannot be measured anywhere else"
        )
    if AUTHENTICODE_LANE_FLAG not in job:
        violations.append(
            f"{AUTHENTICODE_LANE_JOB} does not set {AUTHENTICODE_LANE_FLAG}, so an "
            "Authenticode row that could not be built stays a printed line and the "
            "job exits 0 -- the lane exists and measures nothing"
        )
    if "scripts/release-workflow-contract.test.py" not in job:
        violations.append(
            f"{AUTHENTICODE_LANE_JOB} does not run this suite"
        )
    # Required, not merely present. `conclusion` is the aggregating check, and a
    # job missing from its `needs` is a job whose failure nothing reads.
    # Comments stripped here for a second reason: commenting the line out is
    # how a required job stops being required, and a check that reads commented
    # text is a check that cannot see that happen.
    conclusion = "\n".join(
        line
        for line in job_body(ci, "conclusion").splitlines()
        if not line.lstrip().startswith("#")
    )
    if AUTHENTICODE_LANE_JOB not in conclusion.split("steps:")[0]:
        violations.append(
            f"{AUTHENTICODE_LANE_JOB} is not in conclusion's needs, so the lane can "
            "fail without failing CI"
        )
    if f"expect_job {AUTHENTICODE_LANE_JOB} " not in conclusion:
        violations.append(
            f"conclusion has no expect_job line for {AUTHENTICODE_LANE_JOB}; a job "
            "in `needs` with `if: always()` above it is still not required until "
            "its result is asserted"
        )
    # And NOT on a lane that cannot answer. Telling the Ubuntu docs job to
    # require Authenticode would make it red for a capability it has never had,
    # which ends with the flag being removed again.
    # Comments stripped first: a job's body runs to the next job key, so the
    # prose introducing the NEXT job -- which names this variable, because it
    # explains why the variable is not set here -- would otherwise read as the
    # docs lane setting it.
    docs = "\n".join(
        line
        for line in job_body(ci, "docs").splitlines()
        if not line.lstrip().startswith("#")
    )
    if "WENLAN_REQUIRE_AUTHENTICODE" in docs:
        violations.append(
            "the ubuntu docs lane sets WENLAN_REQUIRE_AUTHENTICODE; it cannot run "
            "Get-AuthenticodeSignature at all, so this makes it permanently red "
            "rather than making anything measured"
        )
    return violations


# Steps that may only run when SignPath is fully configured. The
# presence-consistency check is deliberately NOT in this list: guarding it would
# reopen the exact hole it exists to close.
SIGNPATH_GUARDED_STEPS = [
    "Upload the unsigned installer for SignPath",
    "Submit the installer to SignPath for signing",
    "Replace the unsigned installer with the signed one",
    "Regenerate the updater signature over the signed installer",
    "Verify the installer carries a valid SignPath signature",
]

# The order these have to happen in, and the reason each edge exists:
#   build -> upload           nothing to submit before the installer is built
#   upload -> submit          the action takes a github-artifact-id
#   submit -> replace         the bundle path still holds unsigned bytes
#   replace -> re-sign        the .sig covers the bytes as they are now
#   re-sign -> stage          staging copies and hashes both files
#   stage -> Authenticode     the assertion runs on the bytes that ship
#   Authenticode -> DLL/upload nothing publishes before both gates pass
SIGNPATH_STEP_ORDER = [
    "Build Windows desktop app bundle",
    "Upload the unsigned installer for SignPath",
    "Submit the installer to SignPath for signing",
    "Replace the unsigned installer with the signed one",
    "Regenerate the updater signature over the signed installer",
    "Stage app bundle assets and checksums",
    "Verify the installer carries a valid SignPath signature",
    "Verify the runtime DLLs ship inside the installer",
    "Upload Windows app bundle artifact",
]


def signpath_signing_contract_violations(ci: str, release: str) -> list[str]:
    """Keep Windows Authenticode signing guarded, pinned, ordered, and asserted.

    None of this is exercised by a run today: the SignPath Foundation
    application is pending, no SIGNPATH_* secret exists, so every guarded step
    below is skipped on every real release. That is precisely why the shape is
    pinned statically -- the first time these steps run for real will be a tag
    release, and the failure modes are all silent ones.
    """

    violations: list[str] = []
    job = job_body(release, "app-bundle-windows")
    if not job:
        return ["release.yml no longer defines app-bundle-windows"]

    # ---- the two sentinels and the permission the action needs ----
    #
    # SIGNPATH_CONFIGURED was pinned to the exact single-secret string
    # `secrets.SIGNPATH_API_TOKEN != ''`, which was never a bypass -- the
    # unconditional all-or-nothing step rejects every 1-to-3-of-4 combination --
    # but it named one fifth of what it claimed to measure, and this contract
    # actively REJECTED the clearer expression. The rule is now the property:
    # all four required secrets have to appear in the sentinel, so no subset of
    # them can be present while it reads true.
    sentinel = re.search(r"^      SIGNPATH_CONFIGURED: (?P<expr>.+)$", job, re.MULTILINE)
    if sentinel is None:
        violations.append("app-bundle-windows no longer defines SIGNPATH_CONFIGURED")
    else:
        expression = sentinel.group("expr")
        for secret in (
            "SIGNPATH_API_TOKEN",
            "SIGNPATH_ORGANIZATION_ID",
            "SIGNPATH_PROJECT_SLUG",
            "SIGNPATH_SIGNING_POLICY_SLUG",
        ):
            if f"secrets.{secret} != ''" not in expression:
                violations.append(
                    f"SIGNPATH_CONFIGURED does not require secrets.{secret}; the "
                    "sentinel every signing step keys on would read true with that "
                    "secret absent"
                )
    # SIGNPATH_REQUIRED is the other half, and the one that did not exist. With
    # only SIGNPATH_CONFIGURED, a repository whose secrets were never installed
    # is indistinguishable from a repository that is not supposed to sign: both
    # skip all five signing steps, stage the unsigned installer, and finish
    # green. That is correct while the SignPath Foundation application is
    # pending, and it becomes the critical hole on the day it is accepted.
    #
    # Two properties, because either alone is satisfiable by the wrong thing: it
    # must be scoped to the upstream repository (so no fork can be forced to
    # sign) and it must key on an Actions VARIABLE (so the activation is a
    # visible, auditable, single flip rather than a value nobody can read back).
    required = re.search(r"^      SIGNPATH_REQUIRED: (?P<expr>.+)$", job, re.MULTILINE)
    if required is None:
        violations.append(
            "app-bundle-windows defines no SIGNPATH_REQUIRED sentinel; without one "
            "an accepted-but-unwired repository publishes an unsigned installer and "
            "reports success, exactly as an intentionally unsigned one does"
        )
    else:
        expression = required.group("expr")
        if "github.repository == '7xuanlu/wenlan'" not in expression:
            violations.append(
                "SIGNPATH_REQUIRED is not scoped to the upstream repository; a fork "
                "has no SignPath secrets and must still build green"
            )
        if "vars.SIGNPATH_ACTIVE" not in expression:
            violations.append(
                "SIGNPATH_REQUIRED does not key on vars.SIGNPATH_ACTIVE; the "
                "activation must be an Actions variable a human can read back, not "
                "a secret whose value nothing can audit"
            )
        if "secrets." in expression:
            violations.append(
                "SIGNPATH_REQUIRED reads a secret; then 'must this build be signed' "
                "and 'are the secrets here' would answer the same question again "
                "and the missing-secret case could never be detected"
            )
    permissions_block = re.search(
        r"^    permissions:\n(?P<body>(?:      \S.*\n|      #.*\n)+)", job, re.MULTILINE
    )
    if permissions_block is None or "      actions: read\n" not in permissions_block.group("body"):
        violations.append(
            "app-bundle-windows does not grant actions: read; the SignPath action "
            "cannot read job details or download the artifact it just uploaded"
        )

    # ---- the one step that must NOT be guarded ----
    consistency = named_step_body(job, SIGNPATH_GUARD_STEP)
    if not consistency:
        violations.append(
            "app-bundle-windows has no unconditional SignPath presence-consistency check"
        )
    else:
        if "SIGNPATH_CONFIGURED" in consistency.split("run:")[0]:
            violations.append(
                "the SignPath presence-consistency check is guarded by the very "
                "sentinel it exists to police; a missing API token would skip it "
                "along with every signing step and publish an unsigned installer"
            )
        for marker in [
            "must be all set or all unset",
            "SECRET_SIGNPATH_API_TOKEN",
            "SECRET_SIGNPATH_ORGANIZATION_ID",
            "SECRET_SIGNPATH_PROJECT_SLUG",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG",
            # The required-but-absent latch lives in this same step, because it
            # needs the same property: it must run in the configuration it
            # rejects. A step whose `if:` encoded the rule would skip exactly
            # when the rule was about to fire.
            "SIGNPATH_REQUIRED",
        ]:
            if marker not in consistency:
                violations.append(
                    f"SignPath presence-consistency check omits {marker!r}"
                )
        if "SIGNPATH_REQUIRED" in consistency.split("run:")[0].split("env:")[0]:
            violations.append(
                "the SignPath presence-consistency check is guarded by "
                "SIGNPATH_REQUIRED; the latch that fails a required-but-unconfigured "
                "build must not skip when signing is not required, or it can never "
                "run in the state it exists to reject"
            )

    # ---- every signing step must be guarded ----
    for step_name in SIGNPATH_GUARDED_STEPS:
        body = named_step_body(job, step_name)
        if not body:
            violations.append(f"Windows release job omits SignPath step {step_name!r}")
        elif "if: env.SIGNPATH_CONFIGURED == 'true'" not in body:
            violations.append(
                f"SignPath step {step_name!r} is not guarded; a fork with no "
                "SignPath secret would fail instead of building"
            )

    # ---- the pin, and the reason the allowlist alone is not enough ----
    signpath_references = [
        (action, reference)
        for action, reference in re.findall(r"uses:\s*([^@\s]+)@([^\s#]+)", job)
        if action.lower().endswith("github-action-submit-signing-request")
    ]
    if not signpath_references:
        violations.append("Windows release job no longer submits the installer to SignPath")
    for action, reference in signpath_references:
        if action != SIGNPATH_ACTION:
            violations.append(
                f"SignPath action is spelled {action!r}, not {SIGNPATH_ACTION!r}; the "
                "Node 24 pin allowlist skips any action it does not recognise, so "
                "this reference would be unpinned and nothing would say so"
            )
        if reference != SIGNPATH_ACTION_SHA:
            violations.append(
                f"SignPath action reference {reference!r} is not the pinned "
                f"{SIGNPATH_ACTION_SHA}"
            )

    # ---- the submission itself ----
    submit = named_step_body(job, "Submit the installer to SignPath for signing")
    if submit:
        if "output-artifact-directory:" not in submit:
            violations.append(
                "SignPath submission omits output-artifact-directory; the action's "
                "own manifest says it then does not download the signed artifact, "
                "the job stays green, and the release ships the unsigned installer"
            )
        if 'skip-decompress: "false"' not in submit:
            violations.append(
                'SignPath submission does not keep skip-decompress: "false"; the '
                "upload is an archived artifact, so the returned ZIP has to be "
                "extracted for the replacement step to find a bare installer"
            )
        wait = re.search(
            r'wait-for-completion-timeout-in-seconds:\s*"?(\d+)"?', submit
        )
        if wait is None:
            violations.append(
                "SignPath submission does not set wait-for-completion-timeout-in-seconds"
            )
        elif int(wait.group(1)) <= 600:
            violations.append(
                f"SignPath wait timeout {wait.group(1)}s is not raised past the "
                "600-second default; the Foundation requires a human to approve "
                "each signing request"
            )
        if "github-artifact-id: ${{ steps.unsigned-installer.outputs.artifact-id }}" not in submit:
            violations.append(
                "SignPath submission does not take its artifact id from the upload step"
            )

    # ---- the upload that feeds it ----
    upload = named_step_body(job, "Upload the unsigned installer for SignPath")
    if upload:
        # Line-anchored, because the step explains both of these decisions in
        # prose and a substring test would be satisfied by the comment.
        if re.search(r"^\s*archive:\s*false\s*$", upload, re.MULTILINE):
            violations.append(
                "the unsigned installer upload uses archive: false, which makes the "
                "artifact name the file's basename and breaks retry safety"
            )
        if not re.search(r"^\s*overwrite:\s*true\s*$", upload, re.MULTILINE):
            violations.append(
                "the unsigned installer upload is not retry-safe; a re-run collides "
                "with the previous attempt's artifact of the same name"
            )
        if not re.search(r"^\s*if-no-files-found:\s*error\s*$", upload, re.MULTILINE):
            violations.append(
                "the unsigned installer upload does not fail on an empty match"
            )

    # ---- replacement is a content assertion, never a timestamp one ----
    replace = named_step_body(job, "Replace the unsigned installer with the signed one")
    if replace:
        for marker in [
            "Get-FileHash",
            "if ($before -eq $after)",
            "nothing was signed",
        ]:
            if marker not in replace:
                violations.append(
                    "the signed-installer replacement does not compare content: "
                    f"{marker!r} is missing. ZIP extraction restores the archive "
                    "entry's timestamp, so an mtime comparison passes and fails for "
                    "the wrong reasons"
                )
        for marker in ["$unsigned.Count -ne 1", "$signed.Count -ne 1", ".Name -ne "]:
            if marker not in replace:
                violations.append(
                    f"the signed-installer replacement omits a cardinality or "
                    f"filename assertion: {marker!r}"
                )
        if "LastWriteTime" in replace:
            violations.append(
                "the signed-installer replacement gates on a file timestamp; "
                "extraction restores the archive entry's mtime, so this is neither "
                "necessary nor sufficient"
            )

    # ---- the updater signature covers the signed bytes ----
    resign = named_step_body(job, "Regenerate the updater signature over the signed installer")
    if resign:
        for marker in [
            "TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}",
            "TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}",
            "pnpm tauri signer sign",
        ]:
            if marker not in resign:
                violations.append(
                    "the updater re-signing step needs its own copy of the signing "
                    f"key: {marker!r} is missing. Those secrets are env: on the "
                    "bundle build step alone"
                )
        if 'before_sig' not in resign:
            violations.append(
                "the updater re-signing step does not prove the .sig changed; it "
                "was computed over the unsigned bytes"
            )

    # ---- the publisher assertion ----
    authenticode = named_step_body(job, "Verify the installer carries a valid SignPath signature")
    if authenticode:
        for marker in [
            "Get-AuthenticodeSignature",
            "$signature.Status -ne 'Valid'",
            "GetNameInfo(",
            # -cne, not -ne. PowerShell's -ne is case-INSENSITIVE, so the check
            # this marker guards accepted 'signpath foundation' and every other
            # casing of it. The case-sensitive operator is the assertion; the
            # case-insensitive one only looks like it.
            "$publisher -cne 'SignPath Foundation'",
            "steps.stage.outputs.setup_filename",
        ]:
            if marker not in authenticode:
                violations.append(
                    "the Authenticode assertion is incomplete: "
                    f"{marker!r} is missing. Status alone lets any publisher pass, "
                    "and a skipped download must not read as a signed build"
                )
        if re.search(r"\$\w*[Cc]ertificate\.Subject\s+-(?:eq|like|match)", authenticode):
            violations.append(
                "the Authenticode assertion compares the raw certificate Subject; "
                "that is a full distinguished name, so use GetNameInfo(SimpleName)"
            )
        # The marker above proves the case-SENSITIVE comparison is present; this
        # proves the case-insensitive one has not been added back beside it. A
        # second `-ne`/`-ine` against the publisher would make the -cne line
        # decorative.
        if re.search(r"\$publisher\s+-(?:ne|ine|eq|ieq|like|ilike)\b", authenticode):
            violations.append(
                "the Authenticode assertion compares $publisher with a "
                "case-INSENSITIVE operator; PowerShell's -ne, -eq and -like all "
                "ignore case, so 'signpath foundation' passes a check written to "
                "demand one exact name. Use -cne"
            )
        # Nobody in this repository has seen SignPath Foundation's certificate --
        # the application is pending -- so the first signed build's log is the
        # only place the real Subject and Thumbprint will ever appear. Printed
        # unconditionally, BEFORE the comparison: a run that fails the publisher
        # check is exactly the run whose values someone needs, and a Write-Host
        # placed after the throw prints on the passing runs only.
        for marker in ["$($certificate.Subject)", "$($certificate.Thumbprint)"]:
            if marker not in authenticode:
                violations.append(
                    f"the Authenticode assertion never logs {marker!r}; the "
                    "publisher check compares a name nobody has verified against "
                    "a certificate nobody has seen, and without this the only way "
                    "to learn what it really says is to cut another release"
                )
        subject_log = authenticode.find("$($certificate.Subject)")
        comparison = authenticode.find("$publisher -cne")
        if -1 not in (subject_log, comparison) and subject_log > comparison:
            violations.append(
                "the certificate Subject is logged after the publisher "
                "comparison, so the run that most needs the value -- the one "
                "that failed -- is the one that does not print it"
            )

    # ---- order, which nothing else in this repository checks ----
    positions: dict[str, int] = {}
    for step_name in SIGNPATH_STEP_ORDER:
        index = job.find(f"      - name: {step_name}\n")
        if index < 0:
            violations.append(f"Windows release job omits ordered step {step_name!r}")
        else:
            positions[step_name] = index
    ordered = [name for name in SIGNPATH_STEP_ORDER if name in positions]
    for earlier, later in zip(ordered, ordered[1:]):
        if positions[earlier] >= positions[later]:
            violations.append(
                f"Windows release step order is wrong: {later!r} must come after "
                f"{earlier!r}. Signing that lands after staging publishes unsigned "
                "bytes or a digest that no longer matches the file"
            )

    # ---- the release job is the only Windows job allowed to talk to SignPath ----
    ci_job = job_body(ci, "app-windows-bundle")
    if ci_job and "signpath" in ci_job.lower():
        violations.append(
            "ci.yml app-windows-bundle references SignPath; that job publishes "
            "nothing and must not consume the signing credentials, and its markers "
            "are asserted in BOTH Windows recipes"
        )
    return violations


def release_cache_retry_contract_violations(ci: str) -> list[str]:
    """Keep the Windows release-cache retry bounded, coherent, and pre-build."""

    violations: list[str] = []
    job = job_body(ci, "release-preflight")
    primary = named_step_body(job, "Cache release artifacts (main-owned)")
    initial_probe = named_step_body(job, "Probe initial Windows release cache restore")
    backoff = named_step_body(job, "Back off before one Windows cache retry")
    retry = named_step_body(job, "Retry Windows release cache restore once")
    final_probe = named_step_body(job, "Finalize Windows release cache state")
    pin = "Swatinem/rust-cache@e18b497796c12c097a38f9edb9d0641fb99eee32"
    cold_gate = (
        "if: matrix.target == 'x86_64-pc-windows-msvc' && "
        "steps.windows-cache-probe.outputs.state == 'cold-miss'"
    )

    primary_inputs = [
        "shared-key: release-v4-${{ matrix.target }}",
        "workspaces: . -> target",
        'cache-all-crates: "true"',
        'cache-workspace-crates: "false"',
        "cache-targets: ${{ matrix.target == 'x86_64-pc-windows-msvc' }}",
    ]
    input_pattern = re.compile(r"^ {10}([a-z][a-z-]+):\s*(.+)$", re.MULTILINE)
    primary_input_map = dict(input_pattern.findall(primary))
    retry_input_map = dict(input_pattern.findall(retry))
    expected_primary_inputs = {
        "shared-key": "release-v4-${{ matrix.target }}",
        "workspaces": ". -> target",
        "cache-all-crates": '"true"',
        "cache-workspace-crates": '"false"',
        "cache-targets": "${{ matrix.target == 'x86_64-pc-windows-msvc' }}",
        "save-if": "${{ github.ref == 'refs/heads/main' }}",
    }
    expected_retry_inputs = expected_primary_inputs | {"save-if": '"false"'}
    if (
        "id: windows-release-cache" not in primary
        or f"uses: {pin}" not in primary
        or "save-if: ${{ github.ref == 'refs/heads/main' }}" not in primary
        or any(marker not in primary for marker in primary_inputs)
        or primary_input_map != expected_primary_inputs
    ):
        violations.append("primary Windows release cache restore contract has drifted")
    if (
        "id: windows-release-cache-retry" not in retry
        or f"uses: {pin}" not in retry
        or 'save-if: "false"' not in retry
        or any(marker not in retry for marker in primary_inputs)
        or retry_input_map != expected_retry_inputs
        or "continue-on-error" in retry
        or job.count(f"uses: {pin}") != 2
    ):
        violations.append(
            "Windows release cache retry is not a single pinned restore-only attempt"
        )
    if cold_gate not in backoff or cold_gate not in retry:
        violations.append("Windows release cache retry is not gated by the measured cold miss")
    if "Start-Sleep -Seconds 25" not in backoff:
        violations.append("Windows release cache retry backoff is not exactly 25 seconds")

    initial_contract = [
        "id: windows-cache-probe",
        "$exactHit = '${{ steps.windows-release-cache.outputs.cache-hit }}'",
        '"exact=$exactHit" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"state=$state" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"host-count=$hostCount" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"target-count=$targetCount" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        "partial Windows cache restore:",
        'if ($exactHit -eq "true" -and $state -ne "exact-restore")',
        "Initial Windows cache receipt:",
    ]
    if any(marker not in initial_probe for marker in initial_contract):
        violations.append("initial Windows cache probe omits observable coherent outputs")

    final_contract = [
        "id: windows-cache-final",
        "$initialState = '${{ steps.windows-cache-probe.outputs.state }}'",
        "$retryOutcome = '${{ steps.windows-release-cache-retry.outcome }}'",
        "$retryExact = '${{ steps.windows-release-cache-retry.outputs.cache-hit }}'",
        'throw "partial Windows cache restore after retry:',
        'if ($exactHit -eq "true" -and $state -ne "exact-restore")',
        '$state = "cold-miss"\n            $jobs = 2',
        '$state = "exact-restore"\n              $jobs = 4',
        '$state = "fallback-restore"\n              $jobs = 3',
        '"state=$state" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"jobs=$jobs" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"source=$source" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"exact=$exactHit" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"host-count=$hostCount" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"target-count=$targetCount" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
        '"CARGO_BUILD_JOBS=$jobs" | Out-File -FilePath $env:GITHUB_ENV -Append',
        "Final Windows cache receipt:",
    ]
    if any(marker not in final_probe for marker in final_contract):
        violations.append("final Windows cache receipt omits fail-closed state or bounded jobs")

    ordered_steps = [
        "Cache release artifacts (main-owned)",
        "Probe initial Windows release cache restore",
        "Back off before one Windows cache retry",
        "Retry Windows release cache restore once",
        "Finalize Windows release cache state",
        "Build and smoke shipped release binaries",
    ]
    positions = [job.find(f"- name: {name}") for name in ordered_steps]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        violations.append("Windows release cache retry ordering can overlay live Cargo artifacts")

    return violations


def candidate_observer_contract_violations(
    ci: str, observer: str, validator: str, archive: str
) -> list[str]:
    """Keep the shadow producer unprivileged and its observer read-only."""

    violations: list[str] = []
    producer = job_body(ci, "release-preflight")
    exact_gate = (
        "github.event_name == 'pull_request' && "
        "github.event.pull_request.base.ref == 'main' && "
        "github.event.pull_request.head.ref == 'release-please--branches--main' && "
        "github.event.pull_request.head.repo.full_name == github.repository && "
        "github.event.pull_request.head.repo.fork == false && "
        "github.event.pull_request.draft == false && "
        "github.event.pull_request.user.login == '7xuanlu'"
    )
    producer_steps = [
        "Package canonical release candidate archives",
        "Smoke exact release candidate archives",
        "Write untrusted release candidate claim manifest",
        "Upload immutable release candidate artifact",
    ]
    bodies = [named_step_body(producer, name) for name in producer_steps]
    if any(not body or exact_gate not in body for body in bodies):
        violations.append("candidate producer is not isolated by the exact same-repository release PR gate")
    indices = [producer.find(f"- name: {name}") for name in producer_steps]
    if any(index < 0 for index in indices) or indices != sorted(indices):
        violations.append("candidate producer does not package, smoke, manifest, then upload in order")
    checkout = re.search(
        r"uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6\n"
        r"\s+with:\n(?P<body>.*?)(?=\n\s+- uses:|\n\s+- name:)",
        producer,
        re.DOTALL,
    )
    checkout_body = checkout.group("body") if checkout else ""
    if (
        exact_gate not in checkout_body
        or "github.event.pull_request.head.sha || github.sha" not in checkout_body
        or "persist-credentials: false" not in checkout_body
    ):
        violations.append("candidate producer does not build the exact head SHA without credentials")
    upload = bodies[-1]
    for marker in [
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "name: release-candidate-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.target }}",
        "path: dist/*",
        "compression-level: 0",
        "retention-days: 14",
        "if-no-files-found: error",
        "overwrite: false",
    ]:
        if marker not in upload:
            violations.append(f"candidate producer upload omits immutable contract {marker!r}")
    if any(marker in ci for marker in ["id-token: write", "attestations: write", "actions/attest@"]):
        violations.append("PR CI candidate producer gained signing or OIDC privilege")

    if not re.search(
        r"on:\n  workflow_run:\n    workflows: \[CI\]\n"
        r"    types: \[completed\]\n"
        r"    branches: \[release-please--branches--main\]\n",
        observer,
    ) or re.search(r"\n\s+(workflow_dispatch|pull_request|push):", observer):
        violations.append("candidate observer trigger is not exact release-branch completed CI workflow_run")
    permission_block = re.search(
        r"permissions:\n(?P<body>(?:  [^\n]+\n)+)", observer
    )
    permissions = permission_block.group("body") if permission_block else ""
    if permissions != "  actions: read\n  contents: read\n  pull-requests: read\n":
        violations.append("candidate observer permissions are not the exact read-only set")
    for forbidden in [
        "id-token:",
        "attestations:",
        "packages:",
        "contents: write",
        "pull-requests: write",
        "actions: write",
        "actions/cache",
        "rust-cache",
        "sccache",
        "download-artifact",
        "actions/attest",
        "${{ secrets.",
    ]:
        if forbidden in observer:
            violations.append(f"candidate observer contains forbidden capability {forbidden!r}")
    observer_job = job_body(observer, "validate")
    step_count = len(
        re.findall(r"^      - (?:uses:|name:)", observer_job, re.MULTILINE)
    )
    uses = re.findall(
        r"^\s+(?:- )?uses: ([^\s#]+)", observer_job, re.MULTILINE
    )
    if step_count != 5 or uses != [
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    ]:
        violations.append("candidate observer must have exactly five closed-receipt steps")
    checkout = re.search(
        r"^      - uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6\n"
        r"        with:\n"
        r"          ref: \$\{\{ github\.sha \}\}\n"
        r"          fetch-depth: 1\n"
        r"          persist-credentials: false\n",
        observer_job,
        re.MULTILINE,
    )
    if checkout is None:
        violations.append("candidate observer checkout is not the exact trusted default-branch step")
    expected_validator_step = """env:
          GITHUB_TOKEN: ${{ github.token }}
        run: |
          python3 scripts/validate-release-candidate.py \\
            --event "$GITHUB_EVENT_PATH" \\
            --repository "$GITHUB_REPOSITORY" \\
            --temp-root "$RUNNER_TEMP" \\
            --summary "$GITHUB_STEP_SUMMARY" \\
            --validated-assets-dir "$RUNNER_TEMP/validated-release-assets" \\
            --receipt "$RUNNER_TEMP/validated-release-receipt.json"
""".strip()
    validator_step = named_step_body(
        observer_job, "Validate release candidate as untrusted data"
    )
    if validator_step != expected_validator_step:
        violations.append("candidate observer validator env or command is not exact")
    observer_step_names = [
        "Validate release candidate as untrusted data",
        "Upload exact validated release assets",
        "Close receipt over validated assets artifact",
        "Upload closed validation receipt",
    ]
    indices = [observer_job.find(f"- name: {name}") for name in observer_step_names]
    if any(index < 0 for index in indices) or indices != sorted(indices):
        violations.append("candidate observer does not validate, upload assets, close, then upload receipt")
    validated_upload = named_step_body(
        observer_job, "Upload exact validated release assets"
    )
    closed_upload = named_step_body(observer_job, "Upload closed validation receipt")
    close_step = named_step_body(
        observer_job, "Close receipt over validated assets artifact"
    )
    for marker in [
        "id: validated-assets",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "name: validated-release-assets-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}-${{ github.run_id }}-${{ github.run_attempt }}",
        "path: ${{ runner.temp }}/validated-release-assets/*",
        "compression-level: 0",
        "retention-days: 30",
        "if-no-files-found: error",
        "overwrite: false",
    ]:
        if marker not in validated_upload:
            violations.append(f"validated assets upload omits closed contract {marker!r}")
    for marker in [
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "name: validated-release-receipt-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}",
        "path: ${{ runner.temp }}/validated-release-receipt.json",
        "compression-level: 0",
        "retention-days: 30",
        "if-no-files-found: error",
        "overwrite: true",
    ]:
        if marker not in closed_upload:
            violations.append(f"closed receipt upload omits retry-safe locator contract {marker!r}")
    for marker in [
        "close-receipt",
        "GH_TOKEN: ${{ github.token }}",
        '"/repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"',
        "steps.validated-assets.outputs.artifact-id",
        "steps.validated-assets.outputs.artifact-digest",
        '--observer-run-id "$GITHUB_RUN_ID"',
        '--observer-run-attempt "$GITHUB_RUN_ATTEMPT"',
        '--observer-workflow-id "$observer_workflow_id"',
        '--observer-code-sha "$GITHUB_WORKFLOW_SHA"',
    ]:
        if marker not in close_step:
            violations.append(f"receipt close step omits artifact binding {marker!r}")
    action_pin = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
    for action, reference in re.findall(r"uses:\s*([^@\s]+)@([^\s#]+)", observer):
        if not action_pin.fullmatch(f"{action}@{reference}"):
            violations.append("candidate observer uses a third-party action without a full SHA pin")

    required_validator_markers = [
        'CI_WORKFLOW_PATH = ".github/workflows/ci.yml"',
        'RELEASE_BRANCH = "release-please--branches--main"',
        'RELEASE_AUTHOR = "7xuanlu"',
        "class _CredentialStrippingRedirect",
        'redirected.remove_header("Authorization")',
        "/actions/runs/{run_id}",
        "/actions/workflows/{workflow_id}",
        "/commits/{head_sha}/pulls",
        "/actions/runs/{run_id}/artifacts",
        "/actions/runs/{run_id}/attempts/{run_attempt}/jobs",
        "/actions/artifacts/{artifact_id}/zip",
        'payload.get("total_count")',
        "safe_extract_zip(wrapper, extracted, outer_names)",
        "safe_extract_archive(",
        "validate_release_pr_content",
        "base_records, head_records = _validate_release_tree_modes(",
        "new != old.replace(old_version, new_version)",
        "_release_version_policy(",
        "Darwin standalone archive bytes differ",
        "merged candidate tree differs",
        "/issues/{pr_number}/events",
        "REQUIRED_RELEASE_PATHS = RELEASE_MANAGED_PATHS",
        "artifact[\"size_in_bytes\"]",
        "artifact[\"digest\"]",
        "This receipt is observe-only",
        "| Canonical inner asset |",
        "def write_validated_receipt(",
        "def close_receipt(",
        'document["receipt_state"] = "closed"',
        '"validated_assets_artifact"',
        "def _latest_release_target_attempts(",
        'job.get("run_attempt") != attempt',
        "def _latest_candidate_artifact_attempts(",
        "release-preflight job for {target!r} in artifact attempt {attempt}",
        "def _candidate_artifacts_for_attempts(",
    ]
    for marker in required_validator_markers:
        if marker not in validator:
            violations.append(f"candidate validator omits fail-closed evidence {marker!r}")
    artifact_selection = validator.find("target_attempts = _latest_candidate_artifact_attempts(")
    job_validation = validator.find("    _latest_release_target_attempts(", artifact_selection)
    artifact_binding = validator.find("    selected_artifacts = _candidate_artifacts_for_attempts(", job_validation)
    if artifact_selection < 0 or job_validation < 0 or artifact_binding < 0:
        violations.append("candidate validator does not select artifacts before validating attempt jobs")
    for forbidden in [
        "subprocess.",
        ".chmod(",
        "os.system(",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_OUTPUT",
    ]:
        if forbidden in validator:
            violations.append(f"candidate validator may execute or route artifact bytes via {forbidden!r}")
    for marker in [
        "def _validate_zip_structure(",
        "compression_method not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}",
        "def _decompress_canonical_gzip(",
        "zlib.decompressobj(wbits=31)",
        "expanded_size = _decompress_canonical_gzip(path, raw_path)",
        "decompressor.unused_data",
        "def _validate_tar_termination(",
        "def _validate_raw_tar(",
        "raw_records = _validate_raw_tar(raw_path)",
        "tar extensions and special entries are forbidden",
        "tar.gz has hidden data after its last member",
    ]:
        if marker not in archive:
            violations.append(f"release archive parser omits hostile-input bound {marker!r}")
    return violations


def trusted_candidate_gate_violations(
    ci: str, classifier: str, validator: str
) -> list[str]:
    """Only a semantically closed Release PR with green base CI may omit duplicates."""

    violations: list[str] = []
    detect = job_body(ci, "detect-changes")
    trust_step = named_step_body(detect, "Classify trusted release candidate")
    trusted_checkout = named_step_body(
        detect, "Checkout trusted release candidate classifier"
    )
    for marker in [
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "ref: ${{ github.event.pull_request.base.sha || github.sha }}",
        "path: trusted-release-gate",
        "fetch-depth: 1",
        "persist-credentials: false",
    ]:
        if marker not in trusted_checkout:
            violations.append(f"trusted candidate base checkout omits {marker!r}")
    for marker in [
        "trusted-release-candidate: ${{ steps.release-candidate-trust.outputs.trusted-release-candidate }}",
        "id: release-candidate-trust",
        "classifier=trusted-release-gate/scripts/classify-release-candidate.py",
        'python3 "$classifier"',
        'echo "trusted-release-candidate=false" >> "$GITHUB_OUTPUT"',
        '--event "$GITHUB_EVENT_PATH"',
        '--event-name "$GITHUB_EVENT_NAME"',
        '--repository "$GITHUB_REPOSITORY"',
        '--github-output "$GITHUB_OUTPUT"',
    ]:
        owner = ci if marker.startswith("trusted-release-candidate:") else trust_step
        if marker not in owner:
            violations.append(f"trusted candidate detect gate omits {marker!r}")

    matrix = re.search(
        r"^      - id: matrix\n(?P<body>.*?)(?=^      - )",
        detect,
        re.MULTILINE | re.DOTALL,
    )
    matrix_body = matrix.group("body") if matrix else ""
    trusted_branch = (
        'elif [ "${{ steps.release-candidate-trust.outputs.trusted-release-candidate }}" = "true" ]; then'
    )
    lookalike_branch = (
        'elif [ "${{ startsWith(github.head_ref, \'release-please--branches--\') }}" = "true" ]; then'
    )
    if (
        trusted_branch not in matrix_body
        or lookalike_branch not in matrix_body
        or matrix_body.find(trusted_branch) >= matrix_body.find(lookalike_branch)
        or "include=\"\""
        not in matrix_body[
            matrix_body.find(trusted_branch) : matrix_body.find(lookalike_branch)
        ]
        or "include=\"${macos},${windows}\""
        not in matrix_body[matrix_body.find(lookalike_branch) :]
    ):
        violations.append("trusted candidate matrix is not fail-closed against lookalike branches")

    trust_guard = "needs.detect-changes.outputs.trusted-release-candidate != 'true'"
    for job in [
        "fmt",
        "lint",
        "linux-nextest-build",
        "linux-nextest",
        "test",
        "mcp-platform",
        "canonical-acceptance",
        "contract-integration",
    ]:
        body = job_body(ci, job)
        condition = re.search(r"^    if: (?P<value>.+)$", body, re.MULTILINE)
        if condition is None or trust_guard not in condition.group("value"):
            violations.append(
                f"trusted candidate duplicate base test skip omits {job}"
            )

    base_proof = job_body(ci, "release-base-proof")
    proof_checkout = named_step_body(base_proof, "Checkout trusted main CI verifier")
    proof_run = named_step_body(base_proof, "Verify exact base main CI succeeded")
    for marker in [
        "needs: [detect-changes]",
        "trusted-release-candidate == 'true'",
        "timeout-minutes: 45",
        "actions: read",
        "contents: read",
    ]:
        if marker not in base_proof:
            violations.append(f"trusted candidate base CI proof omits {marker!r}")
    for marker in [
        "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803",
        "ref: ${{ github.event.pull_request.base.sha }}",
        "path: trusted-main-ci-proof",
        "fetch-depth: 1",
        "persist-credentials: false",
    ]:
        if marker not in proof_checkout:
            violations.append(f"trusted candidate base CI checkout omits {marker!r}")
    for marker in [
        "GITHUB_TOKEN: ${{ github.token }}",
        "trusted-main-ci-proof/scripts/release-promotion.py verify-main-ci",
        '--repository "$GITHUB_REPOSITORY"',
        '--sha "${{ github.event.pull_request.base.sha }}"',
        "--wait-seconds 2400",
    ]:
        if marker not in proof_run:
            violations.append(f"trusted candidate base CI proof omits {marker!r}")
    conclusion = job_body(ci, "conclusion")
    if (
        "run_platform='${{ needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.trusted-release-candidate != 'true'"
        not in conclusion
        or "expect_job release-base-proof '${{ needs.detect-changes.outputs.verified-release-merge != 'true' && needs.detect-changes.outputs.trusted-release-candidate == 'true' }}'"
        not in conclusion
        or "release-base-proof" not in conclusion
    ):
        violations.append("conclusion does not require the semantic candidate base CI proof")

    for marker in [
        "VALIDATOR.validate_trusted_release_candidate(",
        "trusted-release-candidate={'true' if trusted else 'false'}",
        "except Exception as error:",
    ]:
        if marker not in classifier:
            violations.append(f"trusted candidate classifier omits fail-closed marker {marker!r}")
    for marker in [
        "def validate_trusted_release_candidate(",
        "validate_release_pr_content(api, repository, pr)",
        'api.get_json(f"/repos/{repository}/git/ref/heads/main")',
        'event_head_repo.get("fork") is not False',
        'event_user.get("login") != RELEASE_AUTHOR',
    ]:
        if marker not in validator:
            violations.append(f"shared semantic release validator omits {marker!r}")
    return violations


SIGNPATH_GUARD_STEP = (
    "Check SignPath configuration is all-or-nothing, and present when required"
)


def posix_bash() -> str:
    """A POSIX bash, never WSL's.

    On any Windows machine with WSL installed the first `bash` on PATH is
    C:\\Windows\\System32\\bash.exe, which is the Linux distro. Running a
    workflow fragment there fails with an opaque RPC error that has nothing to
    do with the fragment. Same resolution order as scripts/run-bash.mjs, which
    exists for exactly this. Missing bash raises: a guard that cannot be run is
    a guard that was not checked, not one that passed.
    """
    if sys.platform != "win32":
        return "bash"
    candidates = []
    if os.environ.get("WENLAN_BASH"):
        candidates.append(os.environ["WENLAN_BASH"])
    roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("ProgramW6432"),
    ]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(os.path.join(local, "Programs"))
    for root in roots:
        if root:
            candidates.append(os.path.join(root, "Git", "bin", "bash.exe"))
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if re.search(r"[\\/]git[\\/](cmd|bin)$", entry, re.IGNORECASE):
            candidates.append(os.path.join(entry, "..", "bin", "bash.exe"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise AssertionError(
        "no Git Bash found; set WENLAN_BASH. The `bash` on PATH is WSL and "
        "cannot run a Windows workflow fragment."
    )


def signpath_guard_script(release: str) -> str:
    """The SHIPPED body of the all-or-nothing guard, ready to run under bash.

    Keyed on the job and the step name, and required to appear exactly once. An
    extractor that takes "the first `run: |` in the file" tests whichever block
    happens to come first, so a decoy step added above it would leave this one
    changing unwatched.
    """
    job = job_body(release, "app-bundle-windows")
    if not job:
        raise AssertionError("release.yml has no app-bundle-windows job")
    occurrences = job.count(f"- name: {SIGNPATH_GUARD_STEP}\n")
    if occurrences != 1:
        raise AssertionError(
            f"expected exactly one {SIGNPATH_GUARD_STEP!r} step, found {occurrences}"
        )
    step = named_step_body(job, SIGNPATH_GUARD_STEP)
    match = re.search(r"^[ ]*run: \|\n(?P<body>.*)\Z", step, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError(f"{SIGNPATH_GUARD_STEP!r} has no `run: |` block")
    body = match.group("body").split("\n")
    indent = len(body[0]) - len(body[0].lstrip())
    script = "\n".join(line[indent:] if len(line) >= indent else line for line in body)
    if "present=0" not in script:
        raise AssertionError("the extracted guard is not the guard; shape changed")
    return script


# Every configuration this step can meet, and the status it must exit with.
# The static contract below checks that the diagnostics and the secret names are
# present; it cannot see the arithmetic. `present=$((present + 1))` mutated to
# `+ 0` leaves every marker in place, passes the static contract, and lets a
# release with three of four secrets ship an UNSIGNED installer. Only running
# the step catches that.
#
# Columns: a description, the environment (the four secrets, the optional
# artifact slug, SIGNPATH_CONFIGURED and SIGNPATH_REQUIRED), the exit status the
# step must produce, and every string its output must contain.
#
# The output column exists because a status is not a diagnosis. The
# required-but-unconfigured latch is the only thing standing between an accepted
# SignPath application and a green unsigned release, and it will fire in front
# of somebody who has never seen it before; "exit 1" with nothing naming the
# missing secret sends them to read the workflow. Asserting the text is also the
# only way to tell the latch apart from the all-or-nothing check, which exits 1
# in some of the same rows for an entirely different reason.
SIGNPATH_GUARD_TRUTH_TABLE: tuple[
    tuple[str, dict[str, str], int, tuple[str, ...]], ...
] = (
    (
        "nothing configured is the pending state, and every fork's state",
        {"SIGNPATH_CONFIGURED": "false"},
        0,
        ("publishes an UNSIGNED Windows installer",),
    ),
    (
        "all four present with the sentinel agreeing",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SIGNPATH_CONFIGURED": "true",
        },
        0,
        ("The installer will be Authenticode-signed",),
    ),
    (
        "all four present plus the optional artifact slug",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SECRET_SIGNPATH_ARTIFACT_CONFIGURATION_SLUG": "artifacts",
            "SIGNPATH_CONFIGURED": "true",
        },
        0,
        ("The installer will be Authenticode-signed",),
    ),
    (
        "three of four: the token is missing and the guarded steps would skip",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SIGNPATH_CONFIGURED": "false",
        },
        1,
        ("SIGNPATH_API_TOKEN is missing",),
    ),
    (
        "the organization id alone is missing",
        {
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SIGNPATH_CONFIGURED": "true",
        },
        1,
        ("SIGNPATH_ORGANIZATION_ID is missing",),
    ),
    (
        "the project slug alone is missing",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SIGNPATH_CONFIGURED": "true",
        },
        1,
        ("SIGNPATH_PROJECT_SLUG is missing",),
    ),
    (
        "the signing policy slug alone is missing",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SIGNPATH_CONFIGURED": "true",
        },
        1,
        ("SIGNPATH_SIGNING_POLICY_SLUG is missing",),
    ),
    (
        "the optional artifact slug set on its own, which signs nothing",
        {
            "SECRET_SIGNPATH_ARTIFACT_CONFIGURATION_SLUG": "artifacts",
            "SIGNPATH_CONFIGURED": "false",
        },
        1,
        ("ARTIFACT_CONFIGURATION_SLUG is set while none",),
    ),
    (
        "all four present but the sentinel says false",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SIGNPATH_CONFIGURED": "false",
        },
        1,
        ("SIGNPATH_CONFIGURED is 'false'",),
    ),
    (
        "none present but the sentinel says true",
        {"SIGNPATH_CONFIGURED": "true"},
        1,
        ("SIGNPATH_CONFIGURED is true but no SignPath secret is set",),
    ),
    # ---- the latch: the four combinations of REQUIRED x CONFIGURED ----
    #
    # Rows 1 and 2 above already cover REQUIRED=false: unset behaves as false,
    # and the guard cannot tell an unset variable from an explicit one. These
    # four make each combination explicit, because a matrix that is only implied
    # is a matrix nobody checked.
    (
        "not required, not configured: today's upstream state and every fork's",
        {"SIGNPATH_CONFIGURED": "false", "SIGNPATH_REQUIRED": "false"},
        0,
        ("publishes an UNSIGNED Windows installer", "SIGNPATH_REQUIRED is 'false'"),
    ),
    (
        "not required but configured anyway: signing happens, nothing fails",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SIGNPATH_CONFIGURED": "true",
            "SIGNPATH_REQUIRED": "false",
        },
        0,
        ("The installer will be Authenticode-signed", "SIGNPATH_REQUIRED is 'false'"),
    ),
    (
        "required AND configured: the state this whole design is aiming at",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SECRET_SIGNPATH_API_TOKEN": "token",
            "SIGNPATH_CONFIGURED": "true",
            "SIGNPATH_REQUIRED": "true",
        },
        0,
        ("The installer will be Authenticode-signed", "SIGNPATH_REQUIRED is 'true'"),
    ),
    (
        "required and NOT configured: the hole. Zero of four is green today and "
        "must be red the day signing is switched on",
        {"SIGNPATH_CONFIGURED": "false", "SIGNPATH_REQUIRED": "true"},
        1,
        (
            "has switched SignPath signing ON",
            "MISSING: SIGNPATH_ORGANIZATION_ID",
            "MISSING: SIGNPATH_PROJECT_SLUG",
            "MISSING: SIGNPATH_SIGNING_POLICY_SLUG",
            "MISSING: SIGNPATH_API_TOKEN",
        ),
    ),
    (
        "required and half configured: the all-or-nothing message wins, because "
        "it names the one thing that is wrong",
        {
            "SECRET_SIGNPATH_ORGANIZATION_ID": "org",
            "SECRET_SIGNPATH_PROJECT_SLUG": "proj",
            "SECRET_SIGNPATH_SIGNING_POLICY_SLUG": "policy",
            "SIGNPATH_CONFIGURED": "false",
            "SIGNPATH_REQUIRED": "true",
        },
        1,
        ("SIGNPATH_API_TOKEN is missing",),
    ),
    (
        "only the literal 'true' arms the latch; a variable set to 1, yes or "
        "TRUE is not the switch being thrown",
        {"SIGNPATH_CONFIGURED": "false", "SIGNPATH_REQUIRED": "TRUE"},
        0,
        ("publishes an UNSIGNED Windows installer",),
    ),
)


def signpath_guard_behaviour_violations(release: str) -> list[str]:
    """Run the shipped guard against every configuration it can meet."""
    script = signpath_guard_script(release)
    violations: list[str] = []
    for description, env, expected_status, required_text in SIGNPATH_GUARD_TRUTH_TABLE:
        result = subprocess.run(
            [posix_bash(), "-c", script],
            env={"PATH": os.environ.get("PATH", ""), **env},
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        if result.returncode != expected_status:
            violations.append(
                f"SignPath guard: {description}: expected exit {expected_status}, "
                f"got {result.returncode}; stdout={result.stdout.strip()!r} "
                f"stderr={result.stderr.strip()!r}"
            )
            continue
        # Right status, wrong reason is still wrong. Several of these rows exit
        # 1 through different checks, and a guard that reaches the right answer
        # by the wrong route stops being the guard the next edit thinks it is.
        for needle in required_text:
            if needle not in output:
                violations.append(
                    f"SignPath guard: {description}: exited {result.returncode} as "
                    f"expected but never said {needle!r}; output="
                    f"{output.strip()[:600]!r}"
                )
    return violations


# --------------------------------------------------------------------------
# finalize-release -- the desktop links in the published notes
# --------------------------------------------------------------------------
#
# The Install section prepare-release writes is entirely command line, so the
# one artifact built for Windows users is reachable only by expanding a
# collapsed list of fifteen assets. finalize-release links it into the notes.
#
# Everything that step decides is arithmetic and whole-line matching that no
# reading of the YAML can check: whether both assets are really on the release,
# whether the Install section already carries the links, whether one link is a
# finished re-run or a half-written section. Both defects found while writing it
# were found by running it, not by reading it -- `grep -Fxq` read a marker
# beginning "- " as a bundle of short options and never matched anything, and
# matching the whole body instead of the Install section declared a release
# already linked when the only copy of the links was in a changelog line.

DESKTOP_LINK_STEP = "Link the desktop app from the release notes"

# Deliberately not a version that exists. The step derives both filenames from
# RELEASE_TAG, and a fixture reusing a shipped version would pass just as well
# against a step that had them hardcoded.
DESKTOP_LINK_TAG = "v9.9.9"
DESKTOP_LINK_REPO = "7xuanlu/wenlan"
DESKTOP_LINK_WIN = "Wenlan_9.9.9_x64-setup.exe"
DESKTOP_LINK_DMG = "Wenlan_9.9.9_aarch64.dmg"
DESKTOP_LINK_BASE = (
    f"https://github.com/{DESKTOP_LINK_REPO}/releases/download/{DESKTOP_LINK_TAG}"
)

# `gh` as a shell FUNCTION, not an executable on a doctored PATH. An
# extensionless stub in a scratch directory does not shadow gh.exe on Windows:
# PATHEXT resolution walks straight past it to the real binary, which answers
# the asset query with "401 Bad credentials" and turns rows green for a reason
# that has nothing to do with the step. A function is resolved before PATH is
# consulted at all, on every platform.
DESKTOP_LINK_GH_STUB = """
gh() {
  case "$*" in
    *"--json assets"*) cat "$WORK/assets.txt"; return 0 ;;
    *"--json body"*)   cat "$WORK/body.txt";   return 0 ;;
  esac
  if [[ "$1" == "release" && "$2" == "edit" ]]; then
    local arg
    for arg in "$@"; do [[ -f "$arg" ]] && cp "$arg" "$WORK/edited.md"; done
    echo GH-RELEASE-EDIT-CALLED
    return 0
  fi
  echo "gh stub: unexpected $*" >&2
  return 9
}
"""

DESKTOP_LINK_EDITED = "GH-RELEASE-EDIT-CALLED"

DESKTOP_LINK_NOTES = """## What's Changed

### Bug Fixes

* **app:** something

## Install

**Wenlan setup (local runtime + daemon):**

```
npx -y wenlan setup
```
"""

DESKTOP_LINK_NO_INSTALL = "## What's Changed\n\nNothing here.\n"

DESKTOP_LINK_SECTION = (
    "## Install\n\n**Desktop app:**\n\n"
    f"- Windows: [{DESKTOP_LINK_WIN}]({DESKTOP_LINK_BASE}/{DESKTOP_LINK_WIN})\n"
    f"- macOS (Apple silicon): [{DESKTOP_LINK_DMG}]({DESKTOP_LINK_BASE}/{DESKTOP_LINK_DMG})\n"
)
# A release whose notes the step has already edited.
DESKTOP_LINK_DONE = DESKTOP_LINK_NOTES.replace("## Install", DESKTOP_LINK_SECTION, 1)
# One of the two rendered items, which is damage and not a completed run.
DESKTOP_LINK_HALF = DESKTOP_LINK_NOTES.replace(
    "## Install",
    "## Install\n\n**Desktop app:**\n\n"
    f"- Windows: [{DESKTOP_LINK_WIN}]({DESKTOP_LINK_BASE}/{DESKTOP_LINK_WIN})\n",
    1,
)


class DesktopLinkCase(NamedTuple):
    """One release-note shape, and what the step must do with it.

    `expect` and `forbid` are matched against the step's own output AND the
    notes it wrote back; `ordered` only against the notes, and in sequence, so
    a block inserted in the right file but the wrong place still fails.
    """

    description: str
    assets: tuple[str, ...]
    body: str
    signpath: str
    status: int
    expect: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    ordered: tuple[str, ...] = ()


# Not in the table, and deliberately: a body carrying TWO `## Install` headings.
# awk's `inside` toggles on each, so the extracted section is both of them
# concatenated, and links under the second would report the first as done. That
# is a real wrong answer, but prepare-release composes exactly one Install
# heading and nothing else writes to these notes before this step runs, so there
# is no shape to assert against without first inventing the release that
# produces it. Named here rather than left to be discovered as a silent gap.
DESKTOP_LINK_TRUTH_TABLE: tuple[DesktopLinkCase, ...] = (
    DesktopLinkCase(
        "links both installers, under the Install heading and above the CLI methods",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG, "SHA256SUMS"),
        DESKTOP_LINK_NOTES,
        "false",
        0,
        expect=(
            "**Desktop app:**",
            f"- Windows: [{DESKTOP_LINK_WIN}]({DESKTOP_LINK_BASE}/{DESKTOP_LINK_WIN})",
            f"- macOS (Apple silicon): [{DESKTOP_LINK_DMG}]"
            f"({DESKTOP_LINK_BASE}/{DESKTOP_LINK_DMG})",
            "Windows protected your PC",
        ),
        ordered=("## Install", "**Desktop app:**", "npx -y wenlan setup"),
    ),
    # The whole point of checking rather than assuming. promote-app-assets is a
    # `needs`, so the assets should be there -- but "should" is what published
    # the dead link this step exists to avoid.
    DesktopLinkCase(
        "refuses when the Windows installer is not on the release",
        (DESKTOP_LINK_DMG, "SHA256SUMS"),
        DESKTOP_LINK_NOTES,
        "false",
        1,
        expect=(f"ERROR: {DESKTOP_LINK_WIN} is not on {DESKTOP_LINK_TAG}",),
        forbid=(DESKTOP_LINK_EDITED,),
    ),
    DesktopLinkCase(
        "refuses when the macOS dmg is not on the release",
        (DESKTOP_LINK_WIN, "SHA256SUMS"),
        DESKTOP_LINK_NOTES,
        "false",
        1,
        expect=(f"ERROR: {DESKTOP_LINK_DMG} is not on {DESKTOP_LINK_TAG}",),
        forbid=(DESKTOP_LINK_EDITED,),
    ),
    DesktopLinkCase(
        "refuses notes with no Install heading instead of editing them blindly",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_NO_INSTALL,
        "false",
        1,
        expect=("no '## Install' heading",),
        forbid=(DESKTOP_LINK_EDITED,),
    ),
    # `--jq .body` on a release with no notes returns an empty string. Refusing
    # is the only safe answer: the alternative is a release whose entire body is
    # the desktop block, written over whatever prepare-release failed to write.
    DesktopLinkCase(
        "refuses a release with no notes at all",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        "",
        "false",
        1,
        expect=("no '## Install' heading",),
        forbid=(DESKTOP_LINK_EDITED,),
    ),
    # Install last, with no following `## ` to switch awk's `inside` back off,
    # so the extracted section runs to end of file. The insert still has to land
    # directly under the heading and not at the end of it.
    DesktopLinkCase(
        "links notes whose Install section is the last one",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        "## What's Changed\n\n* **app:** something\n\n"
        "## Install\n\n```\nnpx -y wenlan setup\n```\n",
        "false",
        0,
        expect=(DESKTOP_LINK_EDITED,),
        ordered=("## Install", "**Desktop app:**", "npx -y wenlan setup"),
    ),
    # Signing changes the reason, never the click path: the user still meets
    # SmartScreen while the hash or the publisher accumulates reputation.
    DesktopLinkCase(
        "keeps the SmartScreen instructions once signed, and changes only the reason",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_NOTES,
        "true",
        0,
        expect=(
            "**Desktop app:**",
            "hash or its publisher has built up enough reputation",
            "More info → Run anyway",
        ),
        forbid=("is not code-signed yet",),
    ),
    DesktopLinkCase(
        "is a no-op on a re-run rather than appending a second block",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_DONE,
        "false",
        0,
        expect=("already present",),
        forbid=(DESKTOP_LINK_EDITED,),
    ),
    DesktopLinkCase(
        "stops on a half-written section instead of calling it finished",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_HALF,
        "false",
        1,
        expect=("1 of the 2 desktop links",),
        forbid=(DESKTOP_LINK_EDITED, "already present"),
    ),
    # Why the markers are the rendered list items and not bare URLs: a changelog
    # line that merely mentions the installer must not stop the release.
    DesktopLinkCase(
        "inserts normally when a changelog entry mentions the installer URL",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_NOTES.replace(
            "* **app:** something",
            f"* **app:** stop shipping a broken {DESKTOP_LINK_BASE}/{DESKTOP_LINK_WIN}",
            1,
        ),
        "false",
        0,
        expect=(DESKTOP_LINK_EDITED, "- macOS (Apple silicon): ["),
        forbid=("refusing to guess", "already present"),
    ),
    # And why they are not the heading: release-please renders a
    # `fix(Desktop app):` scope as exactly the bold string this step writes.
    DesktopLinkCase(
        "is not fooled by a changelog entry scoped 'Desktop app'",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_NOTES.replace(
            "* **app:** something",
            "* **Desktop app:** stop the window flashing on cold start",
            1,
        ),
        "false",
        0,
        expect=(DESKTOP_LINK_EDITED, f"- Windows: [{DESKTOP_LINK_WIN}]"),
        forbid=("refusing to guess", "already present"),
    ),
    # GitHub hands back whatever line endings the body was stored with, and
    # anything edited through the web UI comes back CRLF.
    DesktopLinkCase(
        "links a CRLF release body",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_NOTES.replace("\n", "\r\n"),
        "false",
        0,
        expect=(DESKTOP_LINK_EDITED, f"- Windows: [{DESKTOP_LINK_WIN}]"),
        forbid=("no '## Install' heading",),
    ),
    DesktopLinkCase(
        "treats an already-linked CRLF body as done",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_DONE.replace("\n", "\r\n"),
        "false",
        0,
        expect=("already present",),
        forbid=(DESKTOP_LINK_EDITED,),
    ),
    DesktopLinkCase(
        "ignores trailing whitespace on links that are already there",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_DONE.replace(
            f"]({DESKTOP_LINK_BASE}/{DESKTOP_LINK_WIN})",
            f"]({DESKTOP_LINK_BASE}/{DESKTOP_LINK_WIN})  ",
            1,
        ),
        "false",
        0,
        expect=("already present",),
        forbid=(DESKTOP_LINK_EDITED,),
    ),
    # The row that made whole-body matching wrong. Both bullets are present, but
    # quoted in a later section; the Install section still has nothing in it, and
    # a step that reported "already present" here would leave it that way.
    DesktopLinkCase(
        "does not count links quoted outside the Install section",
        (DESKTOP_LINK_WIN, DESKTOP_LINK_DMG),
        DESKTOP_LINK_NOTES
        + "\n## Notes for maintainers\n\nThe release should end up with:\n\n"
        f"- Windows: [{DESKTOP_LINK_WIN}]({DESKTOP_LINK_BASE}/{DESKTOP_LINK_WIN})\n"
        f"- macOS (Apple silicon): [{DESKTOP_LINK_DMG}]"
        f"({DESKTOP_LINK_BASE}/{DESKTOP_LINK_DMG})\n",
        "false",
        0,
        expect=(DESKTOP_LINK_EDITED,),
        forbid=("already present", "refusing to guess"),
    ),
)


def desktop_link_script(release: str) -> str:
    """The SHIPPED body of the release-note link step, ready to run under bash.

    Keyed on the job and the step name, and required to appear exactly once, for
    the same reason as `signpath_guard_script`: an extractor that takes the
    first `run: |` it finds tests whichever block happens to come first.
    """
    job = job_body(release, "finalize-release")
    if not job:
        raise AssertionError("release.yml has no finalize-release job")
    occurrences = job.count(f"- name: {DESKTOP_LINK_STEP}\n")
    if occurrences != 1:
        raise AssertionError(
            f"expected exactly one {DESKTOP_LINK_STEP!r} step, found {occurrences}"
        )
    step = named_step_body(job, DESKTOP_LINK_STEP)
    match = re.search(r"^[ ]*run: \|\n(?P<body>.*)\Z", step, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError(f"{DESKTOP_LINK_STEP!r} has no `run: |` block")
    body = match.group("body").split("\n")
    indent = len(body[0]) - len(body[0].lstrip())
    script = "\n".join(line[indent:] if len(line) >= indent else line for line in body)
    for shape in ('gh release edit "$RELEASE_TAG"', "$RUNNER_TEMP/desktop.md"):
        if shape not in script:
            raise AssertionError(
                f"the extracted step is not the link step; {shape!r} is gone"
            )
    return script


def desktop_link_order_violations(release: str) -> list[str]:
    """Where the step sits in finalize-release is part of what it promises.

    It edits prose and nothing else in the release depends on it. `Publish
    SHA256SUMS` and `Promote` are load-bearing, so an asset filename this step
    cannot match must never be what leaves a release unchecksummed or
    unpromoted -- which is exactly what happens if it runs ahead of them and
    exits 1. It must equally stay AHEAD of the cleanup PR, which can fail after
    creating its branch and then exit 0 on the re-run: behind it, a release that
    met that failure would keep the unlinked notes it shipped with for good.
    """
    job = job_body(release, "finalize-release")
    if not job:
        return ["release.yml has no finalize-release job"]
    order = re.findall(r"^      - name: (.+)$", job, re.MULTILINE)
    violations: list[str] = []
    if DESKTOP_LINK_STEP not in order:
        return [f"finalize-release has no {DESKTOP_LINK_STEP!r} step"]
    here = order.index(DESKTOP_LINK_STEP)
    for earlier in ("Publish SHA256SUMS", "Promote"):
        if earlier not in order:
            violations.append(f"finalize-release no longer has a {earlier!r} step")
        elif order.index(earlier) > here:
            violations.append(
                f"{DESKTOP_LINK_STEP!r} runs before {earlier!r}; a filename it "
                "cannot match would leave the release unpromoted or unchecksummed"
            )
    later = "Open the release-as override cleanup PR"
    if later in order and order.index(later) < here:
        violations.append(
            f"{DESKTOP_LINK_STEP!r} runs after {later!r}, which can fail after "
            "creating its branch and then exit 0 on the re-run, so the links "
            "would never be written"
        )
    return violations


def desktop_link_behaviour_violations(release: str) -> list[str]:
    """Run the shipped step against every note shape it can be handed."""
    script = desktop_link_script(release)
    violations: list[str] = []
    for case in DESKTOP_LINK_TRUTH_TABLE:
        with tempfile.TemporaryDirectory() as work:
            runner_temp = os.path.join(work, "runner_temp")
            os.makedirs(runner_temp)
            # Bytes, so the CRLF rows carry the line endings they say they do
            # and nothing on the way in normalises them back.
            Path(work, "body.txt").write_bytes(case.body.encode("utf-8"))
            Path(work, "assets.txt").write_bytes(
                "".join(f"{name}\n" for name in case.assets).encode("utf-8")
            )
            result = subprocess.run(
                [posix_bash(), "-c", DESKTOP_LINK_GH_STUB + script],
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "WORK": work.replace("\\", "/"),
                    "RUNNER_TEMP": runner_temp.replace("\\", "/"),
                    "RELEASE_TAG": DESKTOP_LINK_TAG,
                    "GITHUB_REPOSITORY": DESKTOP_LINK_REPO,
                    "SIGNPATH_CONFIGURED": case.signpath,
                    "GH_TOKEN": "stub-token-the-real-gh-never-sees",
                },
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            edited = Path(work, "edited.md")
            written = edited.read_text(encoding="utf-8") if edited.exists() else ""

        label = f"desktop links: {case.description}"
        if result.returncode != case.status:
            violations.append(
                f"{label}: expected exit {case.status}, got {result.returncode}; "
                f"output={output.strip()[:600]!r}"
            )
            continue
        # Right status, wrong reason is still wrong: three rows exit 1 and three
        # exit 0, each through a different branch, and a step that reaches the
        # right answer by the wrong route is not the step the next edit reads.
        haystack = f"{written}\n{output}"
        for needle in case.expect:
            if needle not in haystack:
                violations.append(
                    f"{label}: exited {result.returncode} as expected but never "
                    f"said {needle!r}; output={haystack.strip()[:600]!r}"
                )
        for needle in case.forbid:
            if needle in haystack:
                violations.append(f"{label}: should not have said {needle!r}")
        cursor = -1
        for needle in case.ordered:
            found = written.find(needle, cursor + 1)
            if found < 0:
                violations.append(
                    f"{label}: the notes it wrote back have no {needle!r} after "
                    "the text that must precede it"
                )
                break
            cursor = found
    return violations


def awk_compares_crlf_lines_intact() -> bool:
    """Whether this machine's awk sees the CR at the end of a CRLF line.

    GNU Awk under MSYS reads in text mode and strips it, so `$0 == "## Install"`
    matches a CRLF heading there and the CR-stripping `sed` in the step looks
    like dead code. On GitHub's Ubuntu runners it does not, and dropping that
    `sed` makes the step report a missing heading on a release that has one.
    This is the difference between the two, asked directly rather than assumed,
    so the mutation below is enforced where it is real and reported as
    unchecked where it cannot be.
    """
    probe = subprocess.run(
        [posix_bash(), "-c", "printf '## Install\\r\\n' | awk '$0 == \"## Install\"'"],
        capture_output=True,
        text=True,
        check=False,
    )
    # awk echoes the line only if the comparison held, i.e. only if the CR was
    # already gone. No output means the CR was still there to be compared.
    return not probe.stdout.strip()


# --------------------------------------------------------------------------
# signpath-status.yml -- the activation monitor
# --------------------------------------------------------------------------
#
# release.yml can only ever answer "are the secrets here". Whether SignPath
# ACCEPTED the application, and whether this repository has been SWITCHED ON,
# are questions only this workflow asks -- and the gap between those two is the
# state that publishes a green unsigned installer. So it is contracted here,
# statically and by running it.


def signpath_status_script(text: str | None = None) -> str:
    """The SHIPPED body of the status step, ready to run under bash.

    `text` exists for the mutation controls below: they need to run a MODIFIED
    workflow without writing it to disk, because a harness that mutates a
    tracked file in place leaves it mutated the first time something crashes
    between the write and the restore.
    """
    if text is None:
        text = SIGNPATH_STATUS_PATH.read_text(encoding="utf-8")
    match = re.search(r"^\s*run: \|\n(?P<body>.*)\Z", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError("signpath-status.yml has no `run: |` block")
    body = match.group("body").split("\n")
    indent = len(body[0]) - len(body[0].lstrip())
    script = "\n".join(line[indent:] if len(line) >= indent else line for line in body)
    if "== Configuration ==" not in script:
        raise AssertionError("the extracted status script is not the status script")
    return script


def signpath_status_violations() -> list[str]:
    """Static shape: the schedule, the switch, and the two-sentinel agreement."""
    text = SIGNPATH_STATUS_PATH.read_text(encoding="utf-8")
    violations: list[str] = []
    # A monitor that only answers when asked cannot notice a state that arrives
    # while nobody is asking, and "SignPath accepted us three weeks ago" is
    # exactly such a state.
    if not re.search(r"^on:\n(?:.*\n)*?  schedule:\n    - cron: ", text, re.MULTILINE):
        violations.append(
            "signpath-status.yml has no schedule; manual dispatch cannot notice "
            "an application that was accepted while nobody was looking, which is "
            "the one thing this workflow exists to notice"
        )
    if "workflow_dispatch:" not in text:
        violations.append("signpath-status.yml can no longer be dispatched by hand")
    # The switch has to be the SAME switch release.yml reads, and it has to be a
    # variable: a secret cannot be read back, so a switch stored as one is a
    # switch nobody can audit.
    if "vars.SIGNPATH_ACTIVE" not in text:
        violations.append(
            "signpath-status.yml never reads vars.SIGNPATH_ACTIVE, so it cannot "
            "report the accepted-but-not-activated state at all"
        )
    if "secrets.SIGNPATH_ACTIVE" in text:
        violations.append(
            "signpath-status.yml reads SIGNPATH_ACTIVE as a secret; release.yml "
            "reads it as a variable, and the two would disagree about the switch"
        )
    return violations


#: Columns: description, environment, the fake server's reply table, the exit
#: status, and every string the output must contain.
#:
#: The reply table maps a URL substring to `<http status>|<body>`; the stub curl
#: below walks it in order and takes the first match. A needle ending in `$`
#: must match the END of the URL -- needed because the script's control probe
#: derives its impossible slug from a fixed prefix, so a plain substring test
#: for the configured slug would also match the control and the differential
#: would silently be comparing a URL with itself. `""` as a status means curl
#: itself produced nothing, which is not an HTTP outcome.
_ALL_FOUR = {
    "SIGNPATH_ORGANIZATION_ID": "org-guid",
    "SIGNPATH_PROJECT_SLUG": "wenlan",
    "SIGNPATH_SIGNING_POLICY_SLUG": "release-signing",
    "SIGNPATH_API_TOKEN": "token",
}
#: A server that applies the filters: the configured slugs answer 200, anything
#: else is 404. Ordered longest-first, since the stub takes the first match.
_FILTERING_SERVER = (
    ("signingPolicySlug=release-signing$", "200|[]"),
    ("signingPolicySlug=", "404|{}"),
    ("projectSlug=wenlan&", "200|[]"),
    ("projectSlug=wenlan$", "200|[]"),
    ("projectSlug=", "404|{}"),
    ("SigningRequests", "200|[]"),
)
#: A server that ignores unknown query parameters, which is the shape that makes
#: a filtered read meaningless.
_IGNORING_SERVER = (("SigningRequests", "200|[]"),)

#: A server that rejects the CONTROL for its SPELLING and applies no filter at
#: all. A fixed-prefix control such as `wenlan-no-such-project-<epoch><pid>`
#: differs from the configured slug in its length, its content and its number of
#: separators as well as in the one property the probe is about, so a length
#: cap, a charset rule or a reserved-word rule rejects it on sight: `real=1,
#: control=0`, printed as `resolved`, for a project this server never looked up.
#: The rotation control has the same length, the same separator positions and
#: the same character class at every position, so no rule of that kind can
#: separate the two, both probes come back 200, and the pair reads UNMEASURED.
#:
#: `no-such-project` and not a length rule, because the stub matches on URL
#: substrings: it is the same discrimination, expressed in the one vocabulary
#: this stub has.
_SPELLING_RULE_SERVER = (
    ("no-such-project", "404|{}"),
    ("SigningRequests", "200|[]"),
)

#: A server that rejects every slug-filtered read and answers the unfiltered
#: credential route. Paired with slugs that contain no alphanumeric at all, so
#: the rotation lands back on the configured slug and the two probes become the
#: SAME request: both are rejected, and without `resolve_slug`'s refusal the
#: pair reads `absent` -- a measured NEGATIVE about a slug, off a differential
#: against itself.
_SLUG_REJECTING_SERVER = (
    ("projectSlug=", "404|{}"),
    ("SigningRequests", "200|[]"),
)


def _filtering_server_but(project_reply: str) -> tuple[tuple[str, str], ...]:
    """`_FILTERING_SERVER` with one different answer: the CONFIGURED project probe.

    Ordered longest-first, since the stub takes the first match. Everything
    except that single request still behaves like a server that applies its
    filters -- the control is rejected, the policy pair resolves -- so a row
    built on this is about the one reply and not about the route. Without that,
    a row could go red because nothing resolved rather than because of the
    reply it was written to be about.
    """
    return (
        ("projectSlug=wenlan$", project_reply),
        ("signingPolicySlug=release-signing$", "200|[]"),
        ("signingPolicySlug=", "404|{}"),
        ("projectSlug=wenlan&", "200|[]"),
        ("projectSlug=", "404|{}"),
        ("SigningRequests", "200|[]"),
    )

#: Columns: description, env, stub replies, expected exit, text the output
#: MUST contain, text it must NOT contain.
#:
#: The exit column carries three states, not two, and that is what this table
#: exists to hold on to:
#:
#:   0  everything this run looked at answered affirmatively
#:   1  something answered NEGATIVELY
#:   2  something COULD NOT BE MEASURED
#:
#: The monitor can recreate, at its own outermost boundary, the defect it was
#: written to detect: with the credential good and both slugs coming back "NOT
#: validated -- this run could not measure it", exiting 0 is the same result as
#: both slugs resolving. The `_IGNORING_SERVER` row below is where that is
#: pinned. A truth table that encodes the defect is worse than no truth table,
#: because it certifies it.
#:
#: The last column exists for the same reason. Several of these rows are about
#: what the run must NOT say -- a recommendation to switch signing on is a
#: claim about slugs that resolved, and a row where they did not resolve must
#: not carry it. Requiring text can only catch a message that went missing; it
#: cannot catch one that should never have been printed.
SIGNPATH_STATUS_TRUTH_TABLE: tuple[
    tuple[
        str,
        dict[str, str],
        tuple[tuple[str, str], ...],
        int,
        tuple[str, ...],
        tuple[str, ...],
    ],
    ...,
] = (
    (
        "nothing wired and nothing switched on: today, and every fork",
        {},
        (),
        0,
        ("nothing is wired", "UNMEASURED here, not 'pending'"),
        (),
    ),
    (
        "switched on with nothing wired: the release would fail, so say so here",
        {"SIGNPATH_ACTIVE": "true"},
        (),
        1,
        ("SWITCHED ON WITH NOTHING WIRED",),
        (),
    ),
    (
        "half configured",
        {"SIGNPATH_ORGANIZATION_ID": "org-guid", "SIGNPATH_API_TOKEN": "token"},
        (),
        1,
        ("half-configured", "missing: SIGNPATH_PROJECT_SLUG"),
        (),
    ),
    (
        "accepted but not activated: everything resolves and the switch is off",
        _ALL_FOUR,
        _FILTERING_SERVER,
        1,
        (
            "ACCEPTED BUT NOT ACTIVATED",
            "SIGNPATH_PROJECT_SLUG: validated against the SignPath API",
            "SIGNPATH_SIGNING_POLICY_SLUG: validated against the SignPath API",
            "Set the repository variable SIGNPATH_ACTIVE",
        ),
        ("NOT FULLY VERIFIED",),
    ),
    (
        # The ONLY row that may exit 0. Everything measured, and measured
        # affirmatively.
        "accepted and activated: the state this is all aiming at",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        _FILTERING_SERVER,
        0,
        ("the organization id and the API token are both good",
         "SIGNPATH_ACTIVE is 'true'",
         "VERDICT: everything this run looked at answered"),
        ("COULD NOT MEASURE",),
    ),
    (
        "a 200 that does not parse is COULD NOT MEASURE, not a working credential",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", "200|<html>bad gateway</html>"),),
        2,
        (
            "COULD NOT MEASURE the credential",
            # A NAMED field, not the bare phrase: the slug lines say "NOT
            # validated" whatever the credential did, so the bare phrase
            # survives a mutation that credits the credential.
            "SIGNPATH_API_TOKEN: NOT validated",
            "VERDICT: something COULD NOT BE MEASURED",
        ),
        (),
    ),
    # ---- malformed but VALID JSON: the half an HTML control cannot reach ----
    #
    # An expression such as
    #   if type == "object" then (.values // .items // []) else . end | length
    # accepts anything countable. Measured on this host with jq 1.8.2:
    # `{}` yielded 0, `"proxy error"` yielded 11 (a string's length is its
    # characters), `42` yielded 42 (a number's length is its absolute value).
    # Each one set credential=ok and exited 0 while printing "the organization
    # id and the API token are both good". The HTML row above is the easy half:
    # it fails at the parser. These three get past the parser and are still not
    # an answer, which is the whole distinction the verdict rests on.
    (
        "a 200 whose body is {}: valid JSON, countable, and not a list",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", "200|{}"),),
        2,
        ("COULD NOT MEASURE the credential",
         "SIGNPATH_API_TOKEN: NOT validated"),
        ("the organization id and the API token are both good",),
    ),
    (
        "a 200 whose body is a bare JSON string, whose length is its characters",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", '200|"proxy error"'),),
        2,
        ("COULD NOT MEASURE the credential",),
        ("the organization id and the API token are both good",),
    ),
    (
        "a 200 whose body is a bare number, whose length is its own value",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", "200|42"),),
        2,
        ("COULD NOT MEASURE the credential",),
        ("Signing requests visible to this token: 42",),
    ),
    (
        # THE ROW THAT ENCODED THE DEFECT. It expected 0.
        "the server ignores the slug filters, so the slugs stay unmeasured",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        _IGNORING_SERVER,
        2,
        (
            "SIGNPATH_PROJECT_SLUG: NOT validated -- this run could not measure it",
            "SIGNPATH_SIGNING_POLICY_SLUG: NOT validated -- this run could not measure it",
            "COULD NOT MEASURE a configured slug",
            "VERDICT: something COULD NOT BE MEASURED",
        ),
        ("VERDICT: everything this run looked at answered",),
    ),
    (
        # Accepted, switched off, and the slugs unmeasured: the run must not
        # tell anyone to throw the switch on the strength of what it could not
        # resolve, because release.yml then REQUIRES signing and every release
        # fails at the signing request, after the ~40-minute bundle build.
        "accepted, not activated, and the slugs could not be measured",
        _ALL_FOUR,
        _IGNORING_SERVER,
        1,
        ("ACCEPTED, NOT ACTIVATED, AND NOT FULLY VERIFIED",
         "Do NOT switch signing on yet",
         "COULD NOT MEASURE a configured slug"),
        ("Set the repository variable SIGNPATH_ACTIVE",),
    ),
    (
        "a project slug that does not resolve, on a server that filters",
        {**_ALL_FOUR, "SIGNPATH_PROJECT_SLUG": "typo", "SIGNPATH_ACTIVE": "true"},
        _FILTERING_SERVER,
        1,
        (
            "a configured slug does not resolve",
            "SIGNPATH_PROJECT_SLUG does not name a project",
            "VERDICT: something answered NEGATIVELY",
        ),
        (),
    ),
    (
        "401 is a measured negative about the token",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", "401|{}"),),
        1,
        ("401 Unauthorized", "REJECTED by the SignPath API"),
        ("COULD NOT MEASURE the credential",),
    ),
    (
        "curl produced no status at all: nothing was learned",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", "|"),),
        2,
        ("COULD NOT MEASURE -- curl produced no HTTP status",),
        (),
    ),
    (
        "an undocumented HTTP status says nothing either way",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", "500|{}"),),
        2,
        ("COULD NOT MEASURE -- unexpected HTTP 500",),
        (),
    ),
    # A request that received its status line and then did not complete. curl
    # prints `200` through -w and exits 28 (timeout), 18 (body shorter than
    # Content-Length) or 56 (reset), and the body that DID land here is a
    # syntactically perfect `[]` -- so with curl's status discarded by
    # `|| true`, jq accepts it, the credential reads `ok`, both slug probes
    # answer off the same swallowed failures, and with SIGNPATH_ACTIVE=true the
    # workflow exits 0 over a measurement that failed.
    #
    # The forbidden column is the half that matters: exiting 2 while still
    # printing "the organization id and the API token are both good" would be
    # the same claim in the channel a human reads.
    (
        "a 200 whose transfer never completed is not a measurement",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (("SigningRequests", "200|[]|28"),),
        2,
        ("COULD NOT MEASURE the credential -- curl exited 28",
         "VERDICT: something COULD NOT BE MEASURED"),
        ("the organization id and the API token are both good",),
    ),
    # And the same failure on a SLUG probe, with the credential request whole.
    # The differential is only a measurement if both of its halves completed:
    # here the configured probe times out after printing 200 and the control is
    # rejected for real, which is the exact pair that reads `resolved` without
    # the transfer status.
    (
        "a slug probe whose transfer never completed cannot resolve the slug",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        (
            # The filtering server, with the CONFIGURED project probe timing out
            # after its status line. Longest-first, since the stub takes the
            # first match: the policy pair still resolves, so this row is about
            # the one probe that did not complete and not about the route.
            ("projectSlug=wenlan$", "200|[]|28"),
            ("signingPolicySlug=release-signing$", "200|[]"),
            ("signingPolicySlug=", "404|{}"),
            ("projectSlug=wenlan&", "200|[]"),
            ("projectSlug=", "404|{}"),
            ("SigningRequests", "200|[]"),
        ),
        2,
        ("COULD NOT MEASURE a configured slug",),
        ("SIGNPATH_PROJECT_SLUG: validated against the SignPath API",),
    ),
    # ---- one row per repair in the slug arm ----
    #
    # Each row below is refused by exactly ONE rule in the step, which is what
    # makes it a control for that rule rather than for the arm in general. The
    # mutation list at the bottom of main() names the revert each one catches.
    (
        # 1. THE CONTROL'S SPELLING. A fixed-prefix control -- say
        # `wenlan-no-such-project-` plus an epoch and a pid -- is rejected by
        # this server for its spelling while the filters are ignored entirely,
        # so the pair answers `real=1, control=0` and the step prints
        # `resolved`, exit 0, for a project it never looked up: SYNTAX measured,
        # EXISTENCE reported. As shipped the control is the configured slug with
        # every alphanumeric rotated one place -- same length, same separator
        # positions, same character class throughout -- so nothing here can tell
        # them apart, both probes answer 200, and the pair is UNMEASURED.
        "a server that rejects the control for its spelling and ignores the filters",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        _SPELLING_RULE_SERVER,
        2,
        (
            "SIGNPATH_PROJECT_SLUG: NOT validated -- this run could not measure it",
            "SIGNPATH_SIGNING_POLICY_SLUG: NOT validated -- this run could not measure it",
            "COULD NOT MEASURE a configured slug",
        ),
        (
            "SIGNPATH_PROJECT_SLUG: validated against the SignPath API",
            "VERDICT: everything this run looked at answered",
        ),
    ),
    (
        # 1b. THE OTHER HALF OF THE SAME REMEDY, failing in the OTHER
        # direction. A slug with no alphanumeric in it rotates to itself, so
        # the two probes are the same URL and the "difference" between them is
        # zero by construction. Both are rejected here, and without
        # `resolve_slug`'s refusal that pair reads `absent` -- a measured
        # NEGATIVE, "this slug does not name a project", exit 1, off a
        # comparison of a request with itself. A self-differential cannot
        # produce a false `resolved` (real and control are always equal), which
        # is why this needs its own row: the forbidden column, not the exit
        # code, is the half the affirmative rows above would never have seen.
        "a slug that rotates to itself is not its own control",
        {
            **_ALL_FOUR,
            "SIGNPATH_PROJECT_SLUG": "---",
            "SIGNPATH_SIGNING_POLICY_SLUG": "___",
            "SIGNPATH_ACTIVE": "true",
        },
        _SLUG_REJECTING_SERVER,
        2,
        (
            "SIGNPATH_PROJECT_SLUG: NOT validated -- this run could not measure it",
            "SIGNPATH_SIGNING_POLICY_SLUG: NOT validated -- this run could not measure it",
            "COULD NOT MEASURE a configured slug",
        ),
        (
            "a configured slug does not resolve",
            "SIGNPATH_PROJECT_SLUG does not name a project",
        ),
    ),
    (
        # 2. A SLUG 200 THAT IS NOT A SIGNING-REQUEST LIST. The credential arm
        # refuses this body; a slug probe that reduces the reply to its status
        # line and throws the payload away reads the very same proxy page that
        # is COULD-NOT-MEASURE three screens above as `resolved` down here. The
        # policy pair still resolves and is REQUIRED to below, which is what
        # keeps this row about the one reply rather
        # than about the route: a row that went red because nothing resolved
        # would pass while measuring something else entirely.
        "a slug 200 that is a proxy page is not a signing-request list",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        _filtering_server_but("200|<html>bad gateway</html>"),
        2,
        (
            "SIGNPATH_PROJECT_SLUG: NOT validated -- this run could not measure it",
            "SIGNPATH_SIGNING_POLICY_SLUG: validated against the SignPath API",
            "COULD NOT MEASURE a configured slug",
        ),
        (
            "SIGNPATH_PROJECT_SLUG: validated against the SignPath API",
            "VERDICT: everything this run looked at answered",
        ),
    ),
    (
        # 3. THE ITEMS. A 200 that IS a list
        # -- of signing requests for `someone-else`, returned to a request that
        # filtered on projectSlug=wenlan. Every other rule in the step says
        # this resolved: the transfer completed, the status is 200, the
        # envelope is an array, and the control was rejected. The server told
        # us, in the field this request filtered on, that it did not apply the
        # filter. The item read is the only thing that can refute that, and
        # this row is the only place it is measured.
        "a slug 200 whose items name a different slug refutes the filter",
        {**_ALL_FOUR, "SIGNPATH_ACTIVE": "true"},
        _filtering_server_but('200|[{"projectSlug":"someone-else"}]'),
        2,
        (
            "SIGNPATH_PROJECT_SLUG: NOT validated -- this run could not measure it",
            "SIGNPATH_SIGNING_POLICY_SLUG: validated against the SignPath API",
            "COULD NOT MEASURE a configured slug",
        ),
        (
            "SIGNPATH_PROJECT_SLUG: validated against the SignPath API",
            "VERDICT: everything this run looked at answered",
        ),
    ),
)


def _stub_curl(directory: str, replies: tuple[tuple[str, str], ...]) -> None:
    """A `curl` that answers from a table instead of from the network.

    Only the flags the shipped script uses: -o for the body and -w for the
    status. Anything the table does not name is a hard failure rather than a
    default, so a script that starts calling a new URL is noticed here instead
    of being answered with a plausible 200.

    A reply may carry a third field, `<status>|<body>|<curl exit>`, which is the
    shape a TRANSFER THAT FAILED has: curl prints the status line it had already
    received through `-w` and then exits non-zero (28 timeout, 18 short body, 56
    reset). Without it this stub could only ever produce completed requests, and
    the rows that matter here are the ones that do not complete.
    """
    lines = [
        "#!/usr/bin/env bash",
        "url=\"${@: -1}\"",
        "out=\"\"",
        "prev=\"\"",
        "for a in \"$@\"; do",
        "  if [ \"$prev\" = -o ]; then out=\"$a\"; fi",
        "  prev=\"$a\"",
        "done",
    ]
    for needle, reply in replies:
        code, _, rest = reply.partition("|")
        body, _, curl_exit = rest.partition("|")
        curl_exit = curl_exit or "0"
        # The literal is double-quoted INSIDE the glob so that `&` and `=` stay
        # ordinary characters; the `*`s outside the quotes are still wildcards.
        if needle.endswith("$"):
            pattern = '*"' + needle[:-1] + '"'
        else:
            pattern = '*"' + needle + '"*'
        lines += [
            f'case "$url" in {pattern})',
            f'  [ -n "$out" ] && printf %s {shlex.quote(body)} > "$out"',
            f'  printf %s {shlex.quote(code)}',
            f"  exit {curl_exit} ;;",
            "esac",
        ]
    lines += [
        'echo "stub curl: no table entry for $url" >&2',
        "exit 7",
    ]
    path = os.path.join(directory, "curl")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o755)


class SignPathStatusRun(NamedTuple):
    """What one pass over the SignPath status truth table measured.

    Per ROW, like `AuthenticodeRun`, so a mutation control can demand that the
    rows it names as catchers both ran and failed instead of settling for "some
    violation appeared somewhere in the table".
    """

    violations: list[str]
    ran: set[str]
    failed: set[str]


def signpath_status_behaviour_violations(
    text: str | None = None, only: frozenset[str] | None = None
) -> SignPathStatusRun:
    """Run the shipped status script against a fake SignPath.

    `only` restricts the pass to the named rows. Each row starts a bash and a
    stub curl -- 1.7s on this host -- so a mutation control that already knows
    which rows can catch its mutation should not pay for the other twenty.
    """
    script = signpath_status_script(text)
    violations: list[str] = []
    ran: set[str] = set()
    failed: set[str] = set()
    bash = posix_bash()
    # The script parses the API response with jq, so a host without jq can only
    # ever reach the could-not-parse arm. Those rows are reported UNCHECKED by
    # name rather than run against a stub jq: a stubbed parser would turn the
    # "a 200 that does not parse" row into a test of the stub.
    have_jq = shutil.which("jq") is not None
    for description, env, replies, expected_status, required, forbidden in (
        SIGNPATH_STATUS_TRUTH_TABLE
    ):
        if only is not None and description not in only:
            continue
        if not have_jq and any(reply.startswith("200|") for _, reply in replies):
            print(
                f"UNCHECKED: SignPath status: {description}: this host has no jq, "
                "so the script cannot reach any verdict but could-not-parse"
            )
            continue
        with tempfile.TemporaryDirectory() as work:
            stub_dir = os.path.join(work, "bin")
            os.makedirs(stub_dir)
            _stub_curl(stub_dir, replies)
            child = {
                "PATH": os.environ.get("PATH", ""),
                "RUNNER_TEMP": work,
                "WENLAN_STUB_DIR": stub_dir,
                **env,
            }
            # PATH is prepended INSIDE the shell, not in the child environment.
            # The MSYS runtime puts its own /usr/bin and /mingw64/bin ahead of
            # whatever PATH it inherits, so a stub prepended from Python loses
            # to the real curl -- silently, by making a real network call that
            # then fails for its own reasons. Measured: `type curl` reported
            # /mingw64/bin/curl with the stub directory first in the inherited
            # PATH.
            preamble = (
                'if command -v cygpath >/dev/null 2>&1; then\n'
                '  _wenlan_stub="$(cygpath -u "$WENLAN_STUB_DIR")"\n'
                'else\n'
                '  _wenlan_stub="$WENLAN_STUB_DIR"\n'
                'fi\n'
                'PATH="$_wenlan_stub:$PATH"\n'
                'export PATH\n'
                # And prove it took, rather than assuming: a stub that did not
                # win the lookup would otherwise let every row below make real
                # network calls and report their failures as the script's
                # verdicts.
                'case "$(command -v curl)" in\n'
                '  "$_wenlan_stub"/*) : ;;\n'
                '  *) echo "STUB CURL NOT IN EFFECT: $(command -v curl)" >&2; exit 97 ;;\n'
                'esac\n'
            )
            # Written to a FILE, not passed to `bash -c`. Measured on this
            # Windows host: `bash -c` with this ~14 KB script ran the first
            # ~7.3 KB of it and then exited 0 as though it had reached the end,
            # which is the same failure shape everything else here is about --
            # a run that stopped early wearing the exit status of one that
            # finished. A file has no argument-length limit to hit.
            step_file = os.path.join(work, "status.sh")
            with open(step_file, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(preamble + script)
            result = subprocess.run(
                [bash, step_file],
                env=child,
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            ran.add(description)
            before = len(violations)
            if result.returncode != expected_status:
                violations.append(
                    f"SignPath status: {description}: expected exit "
                    f"{expected_status}, got {result.returncode}; output="
                    f"{output.strip()[:600]!r}"
                )
            else:
                for needle in required:
                    if needle not in output:
                        violations.append(
                            f"SignPath status: {description}: exited "
                            f"{result.returncode} as expected but never said "
                            f"{needle!r}; output={output.strip()[:600]!r}"
                        )
                for needle in forbidden:
                    if needle in output:
                        violations.append(
                            f"SignPath status: {description}: exited "
                            f"{result.returncode} as expected but SAID {needle!r}, "
                            "which is a claim this run did not earn; output="
                            f"{output.strip()[:600]!r}"
                        )
            if len(violations) > before:
                failed.add(description)
    return SignPathStatusRun(violations, ran, failed)


#: Every row whose verdict is COULD NOT MEASURE, derived rather than listed. The
#: tri-state IS the exit status, so anything that collapses status 2 back into
#: status 0 is visible from all of them -- and a row of that kind added later
#: joins the catcher set instead of quietly narrowing it.
_UNMEASURED_ROWS: frozenset[str] = frozenset(
    description
    for description, _, _, expected_status, _, _ in SIGNPATH_STATUS_TRUTH_TABLE
    if expected_status == 2
)

#: The rows that reach the slug arm with something to say about it: the two that
#: refute a slug 200 on its body, the two that cannot resolve a slug at all, and
#: the two whose control probe decides the read was not a measurement.
_SLUG_ROWS: frozenset[str] = frozenset({
    "a server that rejects the control for its spelling and ignores the filters",
    "a slug 200 that is a proxy page is not a signing-request list",
    "a slug 200 whose items name a different slug refutes the filter",
    "a slug probe whose transfer never completed cannot resolve the slug",
    "a slug that rotates to itself is not its own control",
    "the server ignores the slug filters, so the slugs stay unmeasured",
})


# Columns: the shipped text, what it is reverted to, the truth-table rows that
# CAN catch that revert, and what is being reverted.
#
# The catcher column is measured, not guessed: each mutation was run against the
# whole table once and the rows that failed are the rows named here. It buys two
# things. A mutant now runs `catchers & baseline.ran` instead of all 21 rows --
# 43 rows across the fourteen mutations rather than 294, each row a bash and a
# stub curl -- and, because every named row must FAIL, the set is a claim about
# which rows can see the revert rather than a bare "something went red".
#
# A mutation whose catchers all went unrun is a FAILURE here, not an UNCHECKED.
# Unlike the Authenticode table these rows need no host capability beyond jq, so
# "no live catcher" means the table changed under the mutation list, which is
# the stale-fixture case this file exists to catch.
SIGNPATH_STATUS_MUTATIONS: tuple[tuple[str, str, frozenset[str], str], ...] = (
    (
        'if [[ "$credential" == "ok" && "$active" != "true" ]]; then',
        "if false; then",
        frozenset({"accepted but not activated: everything resolves and the switch is off"}),
        "the accepted-but-not-activated verdict",
    ),
    (
        "                credential=unmeasured\n                note_unmeasured\n",
        "                credential=ok\n",
        frozenset({
            "a 200 that does not parse is COULD NOT MEASURE, not a working credential",
            "a 200 whose body is {}: valid JSON, countable, and not a list",
        }),
        "a 200 that does not parse being its own state, not a success",
    ),
    # THE SHAPE ASSERTION. A control that feeds HTML cannot catch this revert --
    # HTML fails at the parser either way. The rows that catch it feed valid JSON
    # that is not a list -- {}, a string, a number -- which is exactly what the
    # reverted expression waves through.
    (
        '                    if type == "array" then length\n'
        '                    elif type == "object" then\n'
        '                      if (.values | type) == "array" then (.values | length)\n'
        '                      elif (.items | type) == "array" then (.items | length)\n'
        '                      else empty end\n'
        '                    else empty end',
        '                    if type == "object" then (.values // .items // [])\n'
        '                    else . end | length',
        frozenset({
            "a 200 whose body is {}: valid JSON, countable, and not a list",
            "a 200 whose body is a bare JSON string, whose length is its characters",
            "a 200 whose body is a bare number, whose length is its own value",
        }),
        "the positive shape assertion on the credential response",
    ),
    (
        '              if [[ "$real" == "1" && "$control" == "0" ]]; then',
        '              if [[ "$real" == "1" ]]; then',
        frozenset({
            "a server that rejects the control for its spelling and ignores the filters",
            "accepted, not activated, and the slugs could not be measured",
            "the server ignores the slug filters, so the slugs stay unmeasured",
        }),
        "the control probe that makes a filtered read a measurement",
    ),
    # The two halves of the transfer status, mutated back to `|| true`: a curl
    # whose status nobody reads makes an incomplete response indistinguishable
    # from a complete one, at the credential and at each slug probe.
    (
        "          if (( curl_rc != 0 )); then\n",
        "          if false; then\n",
        frozenset({"a 200 whose transfer never completed is not a measurement"}),
        "a failed credential transfer being distinguishable from a completed one",
    ),
    (
        "              if (( rc != 0 )); then\n                echo 2\n",
        "              if false; then\n                echo 2\n",
        frozenset({
            "a slug probe whose transfer never completed cannot resolve the slug",
        }),
        "a failed slug transfer being distinguishable from a completed one",
    ),
    # THE OUTERMOST BOUNDARY. A tri-state that collapses back to two at the exit
    # is a tri-state nobody downstream can see: on a scheduled run the workflow
    # result is the entire signal.
    (
        '            && [[ "$project_state" == "unmeasured" || "$policy_state" == "unmeasured" ]]; then\n            note_unmeasured\n',
        '            && [[ "$project_state" == "unmeasured" || "$policy_state" == "unmeasured" ]]; then\n',
        _SLUG_ROWS,
        "an unmeasured slug having any consequence at all",
    ),
    (
        "              status=2\n",
        "              status=0\n",
        _UNMEASURED_ROWS,
        "could-not-measure exiting differently from measured-affirmative",
    ),
    (
        '            && [[ "$project_state" != "resolved" || "$policy_state" != "resolved" ]]; then',
        "            && false; then",
        frozenset({"accepted, not activated, and the slugs could not be measured"}),
        "refusing to recommend the switch on slugs that never resolved",
    ),
    (
        '          describe SIGNPATH_PROJECT_SLUG "$project_state"',
        '          echo "  SIGNPATH_PROJECT_SLUG: validated against the SignPath API"',
        _SLUG_ROWS,
        "the summary being derived from what the run measured",
    ),
    # ---- the three slug-arm remedies, each reverted on its own ----
    #
    # SEPARATELY and not as a block: a single mutation that undoes all three is
    # caught by the first rule it happens to trip and says nothing about the
    # other two.
    #
    # The rotation, back to the fixed impossible prefix. The control then differs
    # from the configured slug in length, in content and in separator count as
    # well as in whether it names anything, and `_SPELLING_RULE_SERVER` is a
    # server that separates the two on the first of those and never applies a
    # filter at all.
    (
        "rotate_slug() { printf '%s' \"$1\" | tr 'a-zA-Z0-9' 'b-zaB-ZA1-90'; }",
        "rotate_slug() { printf '%s' \"wenlan-no-such-project-$$\"; }",
        frozenset({
            "a server that rejects the control for its spelling and ignores the filters",
            "a slug that rotates to itself is not its own control",
        }),
        "a control slug that differs from the configured one in nothing but "
        "whether it names something",
    ),
    (
        '              if [[ "$4" == "$5" ]]; then\n',
        "              if false; then\n",
        frozenset({"a slug that rotates to itself is not its own control"}),
        "refusing a differential whose two halves are the same request",
    ),
    # The slug body, back to the status line alone. `200) ;;` falls through to
    # the envelope assertion and the item read below it; `200) echo 1` is the arm
    # as it stood, where a 200 was affirmative whatever arrived with it. Narrower
    # is not available here: an unparseable body cannot be admitted by any jq
    # program, so the only revert that lets a proxy page through is the one that
    # stops reading the body.
    (
        "                200)     ;;\n",
        "                200)     echo 1; return 0 ;;\n",
        frozenset({
            "a slug 200 that is a proxy page is not a signing-request list",
            "a slug 200 whose items name a different slug refutes the filter",
        }),
        "a slug 200 having to BE a signing-request list",
    ),
    # And the item read alone, with the envelope assertion left standing -- so
    # the row that catches this one is refused by NOTHING else in the step: valid
    # JSON, a real array, a completed transfer, a rejected control.
    (
        "              if (( wrong > 0 )); then\n",
        "              if false; then\n",
        frozenset({"a slug 200 whose items name a different slug refutes the filter"}),
        "the returned items being read, and contradicting the filter",
    ),
)


RESIGN_STEP = "Regenerate the updater signature over the signed installer"


def resign_script(release: str) -> str:
    """The SHIPPED body of the updater re-signing step, ready to run under bash.

    Keyed the same way as the guard above: job, step name, exactly one match.
    """
    job = job_body(release, "app-bundle-windows")
    if not job:
        raise AssertionError("release.yml has no app-bundle-windows job")
    occurrences = job.count(f"- name: {RESIGN_STEP}\n")
    if occurrences != 1:
        raise AssertionError(
            f"expected exactly one {RESIGN_STEP!r} step, found {occurrences}"
        )
    step = named_step_body(job, RESIGN_STEP)
    match = re.search(r"^[ ]*run: \|\n(?P<body>.*)\Z", step, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError(f"{RESIGN_STEP!r} has no `run: |` block")
    body = match.group("body").split("\n")
    indent = len(body[0]) - len(body[0].lstrip())
    script = "\n".join(line[indent:] if len(line) >= indent else line for line in body)
    if "pnpm tauri signer sign" not in script:
        raise AssertionError("the extracted re-sign step is not the re-sign step")
    return script


# Stands in for `pnpm tauri signer sign <installer>`, so the step can be run
# without a Tauri toolchain or a signing key. $4 is the installer path.
#
# `fresh` DERIVES the signature from the installer it was handed, rather than
# writing a fixed string. That is what lets the caller assert the sidecar on
# disk at the end of the step is the one produced over these bytes: a fixed
# string can only be checked for "non-empty and different", which an
# `printf x >> "$sig"` appended after the step's own assertion satisfies.
#
# What this still cannot catch, said plainly: a signer that signs the WRONG
# file. The step has no key to verify against, so a real cryptographic check
# belongs to the updater client, not here.
RESIGN_SIGNER_STUB = """#!/bin/sh
case "${STUB_MODE}" in
  fresh)   printf 'sig:%s' "$(sha256sum < "$4" | cut -d' ' -f1)" > "$4.sig" ;;
  same)    printf '%s' "${STUB_BEFORE}" > "$4.sig" ;;
  missing) : ;;
  empty)   : > "$4.sig" ;;
  broken)  echo "signer exploded" >&2; exit 1 ;;
  *)       echo "unknown STUB_MODE ${STUB_MODE}" >&2; exit 2 ;;
esac
"""

# The .sig the bundle build produced covers the UNSIGNED installer. Every way
# the re-signing can fail to replace it has to be non-zero, because the step
# after this one uploads whatever is on disk: an Authenticode-signed installer
# beside a signature over the bytes it no longer has is an update the client
# rejects, and nothing downstream looks at the .sig again.
#
# The static contract cannot see any of this. `exit 0` inserted after
# before_sig= leaves every marker in the file, passes the static contract, and
# ships exactly that pair. Only running the step catches it.
#
# Columns: description, stub mode, installer present, .sig content before, exit.
RESIGN_TRUTH_TABLE: tuple[tuple[str, str, bool, str | None, int], ...] = (
    (
        "the signer replaces the signature",
        "fresh",
        True,
        "sig-over-unsigned-bytes",
        0,
    ),
    (
        "the signer leaves the signature over the unsigned bytes in place",
        "same",
        True,
        "sig-over-unsigned-bytes",
        1,
    ),
    (
        "the signer writes no signature at all",
        "missing",
        True,
        "sig-over-unsigned-bytes",
        1,
    ),
    (
        "the signer writes an empty signature",
        "empty",
        True,
        "sig-over-unsigned-bytes",
        1,
    ),
    (
        "the signer itself fails",
        "broken",
        True,
        "sig-over-unsigned-bytes",
        1,
    ),
    (
        "the bundle build left no .sig to replace",
        "fresh",
        True,
        None,
        1,
    ),
    (
        "there is no installer under the nsis directory",
        "fresh",
        False,
        None,
        1,
    ),
)


def resign_behaviour_violations(release: str) -> list[str]:
    """Run the shipped re-signing step against a stub signer."""
    script = resign_script(release)
    violations: list[str] = []
    for description, mode, installer, before, expected_status in RESIGN_TRUTH_TABLE:
        with tempfile.TemporaryDirectory() as work:
            nsis = os.path.join(
                work, "target", "x86_64-pc-windows-msvc", "release", "bundle", "nsis"
            )
            os.makedirs(nsis)
            setup = os.path.join(nsis, "Wenlan_9.9.9_x64-setup.exe")
            installer_bytes = b"MZ authenticode-signed installer bytes"
            expected_sig = "sig:" + hashlib.sha256(installer_bytes).hexdigest()
            if installer:
                Path(setup).write_bytes(installer_bytes)
            if before is not None:
                Path(setup + ".sig").write_text(before, encoding="utf-8")
            stub_dir = os.path.join(work, "stub")
            os.makedirs(stub_dir)
            stub = os.path.join(stub_dir, "pnpm")
            with open(stub, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(RESIGN_SIGNER_STUB)
            os.chmod(stub, 0o755)
            result = subprocess.run(
                [posix_bash(), "-c", script],
                cwd=work,
                env={
                    "PATH": stub_dir + os.pathsep + os.environ.get("PATH", ""),
                    "STUB_MODE": mode,
                    "STUB_BEFORE": before or "",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != expected_status:
                violations.append(
                    f"re-sign step: {description}: expected exit {expected_status}, "
                    f"got {result.returncode}; stdout={result.stdout.strip()!r} "
                    f"stderr={result.stderr.strip()!r}"
                )
                continue
            if expected_status != 0:
                continue
            # Exit 0 is not the claim. The claim is that the file on disk now
            # covers the signed bytes, so the successful row asserts the
            # replacement happened rather than trusting the status.
            #
            # "Non-empty and different from before" is too weak: a step that
            # asserted correctly and THEN corrupted the file (`printf x >>
            # "$sig"` after the check) satisfies both and still uploads a
            # signature no client will accept. The sidecar must be byte-exact
            # the one the signer produced over this installer, checked after
            # the whole step has run.
            after = Path(setup + ".sig")
            if not after.exists() or not after.read_text(encoding="utf-8").strip():
                violations.append(
                    f"re-sign step: {description}: exited 0 with no signature on disk"
                )
            elif after.read_text(encoding="utf-8") == before:
                violations.append(
                    f"re-sign step: {description}: exited 0 with the signature over "
                    "the unsigned bytes still on disk"
                )
            elif after.read_text(encoding="utf-8") != expected_sig:
                violations.append(
                    f"re-sign step: {description}: exited 0 but the sidecar on disk is "
                    f"{after.read_text(encoding='utf-8')!r}, not the signature the "
                    f"signer produced over the installer ({expected_sig!r}); the step "
                    "changed it after checking it"
                )
    return violations


AUTHENTICODE_STEP = "Verify the installer carries a valid SignPath signature"


def authenticode_script(release: str) -> str:
    """The SHIPPED body of the Authenticode verification step, as PowerShell."""
    job = job_body(release, "app-bundle-windows")
    if not job:
        raise AssertionError("release.yml has no app-bundle-windows job")
    occurrences = job.count(f"- name: {AUTHENTICODE_STEP}\n")
    if occurrences != 1:
        raise AssertionError(
            f"expected exactly one {AUTHENTICODE_STEP!r} step, found {occurrences}"
        )
    step = named_step_body(job, AUTHENTICODE_STEP)
    match = re.search(r"^[ ]*run: \|\n(?P<body>.*)\Z", step, re.MULTILINE | re.DOTALL)
    if not match:
        raise AssertionError(f"{AUTHENTICODE_STEP!r} has no `run: |` block")
    body = match.group("body").split("\n")
    indent = len(body[0]) - len(body[0].lstrip())
    script = "\n".join(line[indent:] if len(line) >= indent else line for line in body)
    if "Get-AuthenticodeSignature" not in script:
        raise AssertionError("the extracted verification step is not the verification step")
    return script


SIGNING_STEPS = (
    "Submit the installer to SignPath for signing",
    "Regenerate the updater signature over the signed installer",
    AUTHENTICODE_STEP,
    "Verify the runtime DLLs ship inside the installer",
)


#: The only environment variable the verification body may read. Everything
#: else it needs is derived from the file it is pointed at.
AUTHENTICODE_BODY_ENV = frozenset({"STAGED_SETUP"})


def authenticode_body_env_violations(release: str) -> list[str]:
    """The body may read STAGED_SETUP and nothing else.

    The truth table runs the body in the harness's own environment, so a
    body-level early-out keyed on a variable the harness does not set is
    invisible to all five rows:

        if ($env:SIGNPATH_CONFIGURED -eq 'true') { exit 0 }

    Green here, where SIGNPATH_CONFIGURED is unset; a no-op in the release,
    where it is exactly 'true'. Running the body with the release job's
    environment (see `AUTHENTICODE_STEP_ENV`) catches that one
    variable -- but only that one, and the next such guard would key on CI,
    RUNNER_OS, GITHUB_REF or anything else the runner defines. Enumerating what
    the environment holds is unwinnable; enumerating what the body is allowed
    to read is one line. A gate that consults the environment to decide whether
    to gate is not a gate.
    """
    script = authenticode_script(release)
    # A case-sensitive regex for ONE spelling, `$env:NAME`, is not the rule:
    # PowerShell has at least five more --
    # `$Env:NAME`, `${env:NAME}`, `Get-Item Env:NAME`, `dir env:`, and
    # `[System.Environment]::GetEnvironmentVariable(...)`. Enumerating spellings
    # is the same losing game as enumerating variables.
    #
    # So the rule is not a list of spellings. Every one of those forms contains
    # the token `env`, case-insensitively, because that token IS how PowerShell
    # names the environment -- the `Env:` drive and the `Environment` class both
    # carry it. Blank out the single permitted read and any surviving `env` in
    # the body is an unauthorised one, whatever it is spelled like. The cost is
    # that a comment cannot say "environment"; that is a thirty-line body, and a
    # loud false positive is the right side to be wrong on.
    permitted = re.compile(r"\$env:(?:" + "|".join(sorted(AUTHENTICODE_BODY_ENV)) + r")\b")
    masked = permitted.sub(lambda m: "_" * len(m.group(0)), script)
    lines = script.splitlines()
    stray = []
    for match in re.finditer(r"(?i)env", masked):
        number = masked[: match.start()].count("\n")
        stray.append(f"line {number + 1}: {lines[number].strip()!r}")
    if not stray:
        return []
    return [
        f"{AUTHENTICODE_STEP!r} names the environment somewhere other than "
        f"{sorted('$env:' + name for name in AUTHENTICODE_BODY_ENV)}: "
        + "; ".join(stray)
        + " -- a body-level branch on a variable the truth table does not set passes "
        "here and skips the gate in the release, and it does not have to be spelled "
        "`$env:` to do it"
    ]


#: A YAML alias used as a value, including the merge key. The only shapes that
#: can put a property on a step without the property's name appearing in the
#: step's own text.
_ALIAS_VALUE = re.compile(r"^\s*(?:<<\s*:|[\w.\-\"']+\s*:)\s*\*[\w.\-]+\s*$", re.M)
# The other presentation detail the token scan is blind to, and the one that
# actually got past it: a double-quoted key carrying a backslash escape.
# `"continue-on-error": true` decodes to the forbidden key and contains no
# matching source token. YAML 1.2.2 SS5.7 makes escapes a presentation detail of
# the double-quoted style, so only a parser can see through one. Listing both
# classes here is what keeps the no-parser fallback from reporting "clean" about
# an input it cannot read -- the same failed-measurement-as-negative rule the
# rest of this file is about, applied to this file's own dependency.
_ESCAPED_KEY = re.compile(r'^[ \t]*"[^"\n]*\\[^"\n]*"[ \t]*:', re.M)


def continue_on_error_violations(release: str, job: str) -> list[str]:
    """`continue-on-error`, read as a DECODED property, not as a token.

    `re.finditer("continue-on-error", job)` is a lexical net over the job's
    text. It is strictly conservative for what it can
    see -- the token has no other business in this job -- but a property does
    not have to appear in a step's text to be on that step. A merge key pulls
    one in from an anchor defined anywhere else in the file:

        x-swallow: &swallow
          continue-on-error: true
        ...
          - name: Verify the installer carries a valid SignPath signature
            <<: *swallow

    The token is in release.yml and not in `job_body(release, ...)`, the scan
    sees nothing, and the gate's failure is discarded. Reading the key off the
    decoded document is the only way to ask the question GitHub Actions will
    answer. The token scan stays, as a second and independent net, and because
    it is the one that still works when there is no parser here.
    """
    out: list[str] = []
    try:
        import yaml  # noqa: PLC0415  -- optional; the fallback below is measured
    except ImportError:
        # No parser. Rather than claim the coverage or fail blind, measure what
        # is actually lost: with no alias and no escaped key anywhere in the job,
        # every property a step carries IS written in the job's text as the token
        # scan spells it, so that scan is complete for this input and nothing is
        # unchecked. Both classes are named, because reporting only the one that
        # happens to be checked is how a partial measurement becomes a clean bill
        # of health.
        blind: list[str] = []
        if _ALIAS_VALUE.search(job):
            blind.append("a YAML alias (a merge key can carry the property in)")
        if _ESCAPED_KEY.search(job):
            blind.append(
                'a double-quoted key carrying a backslash escape '
                '(`"continue-on-\\u0065rror"` decodes to the forbidden key)'
            )
        if blind:
            out.append(
                "app-bundle-windows contains " + " and ".join(blind) + ", and "
                "PyYAML is not installed here, so `continue-on-error` could not be "
                "read as a decoded property -- it can be carried onto a signing "
                "step without the token appearing in the job. Install PyYAML or "
                "remove the construct; this is unmeasured, not clean"
            )
        return out
    try:
        doc = yaml.safe_load(release)
    except yaml.YAMLError as exc:
        return [f"release.yml does not parse as YAML: {exc}"]
    jobs = doc.get("jobs") if isinstance(doc, dict) else None
    node = jobs.get("app-bundle-windows") if isinstance(jobs, dict) else None
    if not isinstance(node, dict):
        return ["release.yml has no app-bundle-windows job once decoded"]
    swallow = (
        "may not have its failure discarded; the installer it signs and verifies "
        "is what a user runs"
    )
    if "continue-on-error" in node:
        out.append(
            "the app-bundle-windows job itself decodes with continue-on-error: "
            f"{node['continue-on-error']!r}; a job that signs and verifies {swallow}"
        )
    steps = node.get("steps")
    for index, step in enumerate(steps if isinstance(steps, list) else []):
        if isinstance(step, dict) and "continue-on-error" in step:
            owner = step.get("name") or f"the unnamed step #{index + 1}"
            out.append(
                f"{owner!r} decodes with continue-on-error: "
                f"{step['continue-on-error']!r}; a step in this job {swallow}"
            )
    return out


def authenticode_step_metadata_violations(release: str) -> list[str]:
    """The step's YAML around the body, which no body-level test can see.

    A five-row truth table over the extracted `run:` block proves what the
    script does when it runs. `continue-on-error: true` on the step leaves every
    row and every static marker passing while the release ignores the failure
    and publishes anyway -- a gate that runs, fails, and is discarded. The body
    is not the whole step.

    The rule covers the JOB, not a hand-listed set of steps: a four-step list
    omitted "Check SignPath configuration is all-or-nothing", the guard that
    stops a half-configured repo from publishing an unsigned installer under a
    configured-looking run. A list of load-bearing steps is a list someone has
    to keep right; no step in this job may swallow a failure.
    """
    job = job_body(release, "app-bundle-windows")
    if not job:
        return ["release.yml has no app-bundle-windows job"]
    violations = []
    for name in SIGNING_STEPS:
        if job.count(f"- name: {name}\n") != 1:
            violations.append(f"expected exactly one {name!r} step in app-bundle-windows")
    # EVERY step, and the job itself. Attribution is by the nearest preceding
    # step name so the message still says which one, without the enumeration
    # being what decides whether the rule applies.
    # `^\s*continue-on-error:` is one YAML spelling of the key;
    # `"continue-on-error": true` and `{continue-on-error: true}` are the same
    # key and invisible to it. There is no other reason for that token to appear
    # anywhere in this job, so the TOKEN is the rule -- no anchor, no quoting to
    # enumerate, no place left to hide it.
    for match in re.finditer(r"continue-on-error", job):
        seen = re.findall(r"^      - name: (.+)$", job[: match.start()], re.MULTILINE)
        owner = repr(seen[-1]) if seen else "the app-bundle-windows job itself"
        line = job[: match.start()].count("\n")
        violations.append(
            f"{owner} carries {job.splitlines()[line].strip()!r}; a step in the job "
            "that signs and verifies the installer may not have its failure discarded"
        )
    # And the same question asked of the decoded document, which is the one
    # GitHub Actions answers. The scan above cannot follow a merge key.
    violations.extend(continue_on_error_violations(release, job))
    violations.extend(authenticode_body_env_violations(release))
    verify = named_step_body(job, AUTHENTICODE_STEP)
    if not re.search(r"^\s*shell: pwsh\s*$", verify, re.MULTILINE):
        violations.append(f"{AUTHENTICODE_STEP!r} must declare `shell: pwsh`")
    if not re.search(r"^\s*if: env\.SIGNPATH_CONFIGURED == 'true'\s*$", verify, re.MULTILINE):
        violations.append(
            f"{AUTHENTICODE_STEP!r} must be guarded on SIGNPATH_CONFIGURED, like the "
            "macOS steps, or a fork without secrets fails the build"
        )
    return violations


#: The environment the step runs in during a real release, as far as it can
#: matter to the body. Supplying STAGED_SETUP alone runs the body with
#: SIGNPATH_CONFIGURED unset while the shipping job runs it with
#: SIGNPATH_CONFIGURED == 'true', and a branch on that difference is a step that
#: passes every row here and does nothing there.
#: `authenticode_body_env_violations` is the load-bearing half of that fix,
#: since no list of variables can be complete; this one makes the variables the
#: step is MOST likely to be branched on hold their release values anyway.
#:
#: The list is GitHub's own documented set of default environment variables, in
#: full, rather than a hand-picked subset: an eleven-name selection omitted
#: `GITHUB_WORKFLOW_REF`, which GitHub documents as a default and which
#: identifies the workflow file uniquely.
#: It still cannot be complete in principle: a step may branch on anything the
#: job's `env:` supplies, or on a variable an action exported. That is why
#: `authenticode_body_env_violations` -- which rejects ANY read of the
#: environment in the step's body -- is the load-bearing half and this is the
#: belt, and why that rule is a LEXICAL net over the body's text rather than a
#: semantic one: it matches the spellings of an environment read that are known
#: today, and a sufficiently indirect read (a variable built from string
#: fragments, a call through a helper defined elsewhere) is not caught by it.
AUTHENTICODE_STEP_ENV = {
    "SIGNPATH_CONFIGURED": "true",
    "CI": "true",
    # https://docs.github.com/actions/reference/workflows-and-actions/variables
    # #default-environment-variables, in the order GitHub lists them.
    "GITHUB_ACTION": "__run",
    "GITHUB_ACTION_PATH": "",
    "GITHUB_ACTION_REPOSITORY": "",
    "GITHUB_ACTIONS": "true",
    "GITHUB_ACTOR": "github-actions[bot]",
    "GITHUB_ACTOR_ID": "41898282",
    "GITHUB_API_URL": "https://api.github.com",
    "GITHUB_BASE_REF": "",
    "GITHUB_ENV": "",
    "GITHUB_EVENT_NAME": "push",
    "GITHUB_EVENT_PATH": "",
    "GITHUB_GRAPHQL_URL": "https://api.github.com/graphql",
    "GITHUB_HEAD_REF": "",
    # A branch that exits only when GITHUB_JOB is app-bundle-windows runs
    # everywhere else and skips the gate where it counts.
    "GITHUB_JOB": "app-bundle-windows",
    "GITHUB_OUTPUT": "",
    "GITHUB_PATH": "",
    "GITHUB_REF": "refs/tags/v9.9.9",
    "GITHUB_REF_NAME": "v9.9.9",
    "GITHUB_REF_PROTECTED": "false",
    "GITHUB_REF_TYPE": "tag",
    "GITHUB_REPOSITORY": "7xuanlu/wenlan",
    "GITHUB_REPOSITORY_ID": "0",
    "GITHUB_REPOSITORY_OWNER": "7xuanlu",
    "GITHUB_REPOSITORY_OWNER_ID": "0",
    "GITHUB_RETENTION_DAYS": "90",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_RUN_ID": "0",
    "GITHUB_RUN_NUMBER": "0",
    "GITHUB_SERVER_URL": "https://github.com",
    "GITHUB_SHA": "0" * 40,
    "GITHUB_STEP_SUMMARY": "",
    "GITHUB_TRIGGERING_ACTOR": "github-actions[bot]",
    "GITHUB_WORKFLOW": "release",
    "GITHUB_WORKFLOW_REF": "7xuanlu/wenlan/.github/workflows/release.yml@refs/tags/v9.9.9",
    "GITHUB_WORKFLOW_SHA": "0" * 40,
    "GITHUB_WORKSPACE": "",
    "RUNNER_ARCH": "X64",
    "RUNNER_DEBUG": "",
    "RUNNER_ENVIRONMENT": "github-hosted",
    "RUNNER_NAME": "GitHub Actions 1",
    "RUNNER_OS": "Windows",
    "RUNNER_TEMP": "",
    "RUNNER_TOOL_CACHE": "",
}


def _powershell() -> str | None:
    """A PowerShell that can run the step. pwsh first, Windows PowerShell after."""
    import shutil

    return shutil.which("pwsh") or shutil.which("powershell")


#: What `_signed_fixture` found, which is also the host-capability answer.
#: "fixture"      a signed binary was found; every row can be built.
#: "no-support"   Get-AuthenticodeSignature is not usable here. A real
#:                incapacity: PowerShell on Linux and macOS has no Authenticode
#:                at all, and the probe SAYS so rather than merely exiting.
#: "none-found"   the cmdlet works and a SEARCH of the system binary
#:                directories turned up no validly signed file. That is not a
#:                host that cannot measure; it is a probe or a trust store that
#:                is broken, and it is reported as a failure wherever it happens.
#: "probe-failed" the probe did not answer. PowerShell could not be started, or
#:                it died on its own, or it printed something this cannot parse.
#:
#: Read off the EXIT STATUS alone, every way the probe can blow up with status
#: 2 -- a PowerShell that refused the command line, a host policy that killed
#: it, a syntax error in the probe text below -- arrives here spelled
#: "no-support", the one answer allowed to excuse an unmeasured row. So the
#: probe PRINTS which arm it took, the status and the token have to agree, and
#: anything else is `probe-failed`, which is a failure and never an excuse.
#:
#: The third element is the publisher when the capability is "fixture", and the
#: reason it could not answer when it is "probe-failed".
FixtureResult = tuple[str, str | None, str | None]


def _authenticode_status(shell: str, path: str) -> str | None:
    """What this host's Authenticode makes of a file, or None if it could not
    be asked. The path travels in the environment rather than inside the
    command string, so a temp directory containing a quote cannot rewrite it.
    """
    try:
        result = subprocess.run(
            [
                shell,
                "-NoProfile",
                "-Command",
                "(Get-AuthenticodeSignature -LiteralPath "
                "$env:AUTHENTICODE_PROBE_PATH).Status.ToString()",
            ],
            env={**os.environ, "AUTHENTICODE_PROBE_PATH": path},
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    return lines[0] if result.returncode == 0 and lines else None


def _authenticode_hashed_byte(data: bytes) -> tuple[int | None, str]:
    """A file offset Authenticode covers, or why this fixture has none.

    The first cut of the tampered-installer row wrote `data[4096:4100]`, which
    is two assumptions wearing one line. Python slice assignment past the end
    of a bytearray APPENDS, so on a host whose smallest signed binary is under
    4 KiB -- and the probe sorts by size ascending, so it goes looking for one
    -- the tampered-in-the-middle fixture silently became a second copy of the
    byte-appended fixture: same bytes, same NotSigned, and the row that exists
    to prove the HashMismatch branch is reachable was re-measuring the row
    above it. A catalog-signed pick fails the same way for a different reason:
    it carries no embedded certificate table, so nothing done to it can
    produce a hash mismatch.

    Both are answered from the file's own geometry instead of a constant.
    Parse the PE header, find the certificate table, and return an offset that
    is inside the hashed region and past the spans Authenticode excludes.
    """
    if len(data) < 0x40:
        return None, f"the fixture is {len(data)} bytes, too short for a PE header"
    pe = int.from_bytes(data[0x3C:0x40], "little")
    if pe + 0x78 > len(data) or data[pe : pe + 4] != b"PE\0\0":
        return None, "the fixture has no PE signature where its DOS header points"
    magic = int.from_bytes(data[pe + 24 : pe + 26], "little")
    if magic == 0x20B:  # PE32+
        directories = pe + 24 + 112
    elif magic == 0x10B:  # PE32
        directories = pe + 24 + 96
    else:
        return None, f"unrecognised PE optional-header magic 0x{magic:x}"
    # IMAGE_DIRECTORY_ENTRY_SECURITY is index 4, and it is the one directory
    # whose first field is a FILE OFFSET rather than an RVA.
    entry = directories + 4 * 8
    if entry + 8 > len(data):
        return None, "the PE data directory is truncated before the security entry"
    table = int.from_bytes(data[entry : entry + 4], "little")
    size = int.from_bytes(data[entry + 4 : entry + 8], "little")
    if table == 0 or size == 0:
        return None, (
            "the fixture is catalog-signed -- it carries no embedded certificate "
            "table -- so a modified copy of it reports NotSigned, and a hash "
            "mismatch is not reachable through this file at all"
        )
    # Everything ahead of the certificate table is hashed except the checksum
    # field and this directory entry, both of which live in the headers. The
    # last byte before the table is past both.
    #
    # A table that starts past the end of the file is one whose headers and
    # contents disagree. The first cut clamped that case with
    # `min(table, len(data)) - 1` and answered anyway -- and the last byte of
    # an embedded-signed file is NOT hashed (measured: flipping it leaves the
    # signature Valid), so the clamp hands back an offset that produces a
    # still-valid fixture, and the row then reports the shipped step accepting
    # a tampered installer. A parser error, wearing a product regression's
    # clothes. A parser that cannot trust its input refuses instead.
    if table > len(data):
        return None, (
            f"the certificate table claims to start at {table} in a "
            f"{len(data)}-byte file, so its headers and its contents disagree "
            "and no offset derived from them can be trusted"
        )
    offset = table - 1
    if offset <= entry + 8:
        return None, (
            f"the certificate table starts at {table}, which leaves no hashed "
            "byte after the header fields Authenticode excludes"
        )
    return offset, ""


_FIXTURE_DIR: list[str] = []


def _fixture_session_dir() -> str:
    """One directory per process, holding the copied signature fixture."""
    if not _FIXTURE_DIR:
        path = tempfile.mkdtemp(prefix="wenlan-authenticode-")
        atexit.register(shutil.rmtree, path, True)
        _FIXTURE_DIR.append(path)
    return _FIXTURE_DIR[0]


def _signed_fixture(shell: str, work: str) -> FixtureResult:
    """This host's Authenticode capability, measured once per shell and cached.

    Eight callers ask for it in one run and nothing it measures can change
    inside one, so it is measured once and the answer reused. The saving is
    bounded by how long the search takes, which is a property of the host: it
    stops at the first EMBEDDED signature it sees and costs 0.7s here, but its
    ceiling is 60 `Get-AuthenticodeSignature` calls per system binary directory
    on a host whose small binaries are all catalog-signed. `work` is the caller's
    scratch directory and is no longer where the probe runs; it stays in the
    signature because
    scripts/negative-controls/authenticode-step-receipt.py calls this predicate
    directly and rewrites its first body line to substitute a stub shell.

    The binary it names is COPIED once into a session directory, because every
    fixture row reads those bytes again and a file the suite owns cannot be
    serviced out from under it mid-run.

    The copy cannot change the CLASSIFICATION, only which path carries it. This
    predicate answers "what can this host do", and
    scripts/negative-controls/authenticode-step-receipt.py holds it to exactly
    that by feeding it a stub shell that names a path which does not exist and
    requiring `fixture` back. So a copy that fails hands back the path the probe
    found -- what this returned before there was a copy -- and never downgrades a
    measured capability to `probe-failed` on the strength of a file operation.
    """
    return _cached_signed_fixture(shell)


@functools.lru_cache(maxsize=None)
def _cached_signed_fixture(shell: str) -> FixtureResult:
    with tempfile.TemporaryDirectory() as work:
        capability, path, extra = _probe_signed_fixture(shell, work)
    if capability != "fixture" or path is None:
        return (capability, path, extra)
    local = os.path.join(_fixture_session_dir(), "signed-" + os.path.basename(path))
    try:
        shutil.copyfile(path, local)
    except OSError:
        return (capability, path, extra)
    return (capability, local, extra)


def _probe_signed_fixture(shell: str, work: str) -> FixtureResult:
    """A real Authenticode-signed binary on this host, its publisher, and why not.

    `os.name == "nt"` is an OS proxy wearing a capability's name: it says where
    the code is running, not what the host can do. A probe that looks at three
    hard-coded paths is the matching half, where "no signed binary on this host"
    means "not one of these three files".

    So the probe SEARCHES the system binary
    directories instead of naming files, and it distinguishes its two negative
    answers, which is the whole tri-state discipline applied to the capability
    question itself: a cmdlet that cannot run on this platform is an incapacity;
    a cmdlet that runs and finds nothing signed anywhere in System32 is a
    failure, on any OS. The caller no longer has to ask what OS this is.
    """
    probe = os.path.join(work, "probe.ps1")
    Path(probe).write_text(
        # Support first, and by calling it rather than by asking what OS this is:
        # on Linux and macOS the cmdlet exists and throws PlatformNotSupported.
        "if (-not (Get-Command Get-AuthenticodeSignature "
        "-ErrorAction SilentlyContinue)) { Write-Output 'NO-SUPPORT cmdlet-absent'; "
        "exit 2 }\n"
        "try { $null = Get-AuthenticodeSignature -LiteralPath "
        "$PSCommandPath -ErrorAction Stop } catch { "
        "Write-Output 'NO-SUPPORT cmdlet-unusable'; exit 2 }\n"
        "$roots = @([System.Environment]::SystemDirectory, $PSHOME) |\n"
        "  Where-Object { $_ } | Select-Object -Unique\n"
        # An embedded signature is PREFERRED, not required, so this is the
        # first valid file seen rather than the answer. Measured on one
        # Windows 11 host: 613 of the 622 validly signed .exe files under
        # these roots are CATALOG-signed, which carries no certificate
        # table of its own -- a modified copy of one reports NotSigned and
        # can never report HashMismatch. Taking the first valid file made
        # which kind the caller got a matter of luck, and the tampered
        # -installer row then read that luck as a regression in the
        # shipped step. Requiring an embedded one instead would answer
        # NONE-FOUND on a host that has none in range, and this suite
        # treats NONE-FOUND as a broken probe. Prefer, fall back, and let
        # the one row that needs the certificate table say so itself.
        "$fallback = $null\n"
        "foreach ($root in $roots) {\n"
        "  if (-not (Test-Path -LiteralPath $root)) { continue }\n"
        # Smallest first: the fixture gets read, patched and re-hashed five times.
        "  $files = @(Get-ChildItem -LiteralPath $root -Filter *.exe -File "
        "-ErrorAction SilentlyContinue |\n"
        "    Sort-Object Length | Select-Object -First 60)\n"
        "  foreach ($f in $files) {\n"
        "    $s = Get-AuthenticodeSignature -LiteralPath $f.FullName\n"
        "    if ($s.Status -ne 'Valid' -or $null -eq $s.SignerCertificate) { continue }\n"
        "    $n = $s.SignerCertificate.GetNameInfo(\n"
        "      [System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName, "
        "$false)\n"
        "    if (-not $n) { continue }\n"
        "    $line = \"FIXTURE`t$($f.FullName)`t$n\"\n"
        "    if ($s.SignatureType -eq 'Authenticode') { Write-Output $line; exit 0 }\n"
        "    if (-not $fallback) { $fallback = $line }\n"
        "  }\n"
        "}\n"
        "if ($fallback) { Write-Output $fallback; exit 0 }\n"
        "Write-Output 'NONE-FOUND'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", probe],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return ("probe-failed", None, f"{shell} could not be started: {exc}")
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    token = lines[0].strip() if lines else ""
    # Status AND token, together. Either one alone is a single witness, and the
    # arm that matters -- "this host genuinely cannot do Authenticode" -- is the
    # one an unrelated crash is most likely to be mistaken for.
    if result.returncode == 2 and token.startswith("NO-SUPPORT"):
        return ("no-support", None, None)
    if result.returncode == 1 and token == "NONE-FOUND":
        return ("none-found", None, None)
    if result.returncode == 0 and token.startswith("FIXTURE\t"):
        parts = token.split("\t")
        if len(parts) == 3 and parts[1] and parts[2]:
            return ("fixture", parts[1], parts[2])
    return (
        "probe-failed",
        None,
        f"exit {result.returncode} with first line {token[:120]!r}"
        + (f"; stderr={result.stderr.strip()[:200]!r}" if result.stderr.strip() else ""),
    )


# Columns: description, fixture, expected exit, text the output must contain.
#
# "fixture" is how the staged installer is prepared:
#   missing   - no file at all
#   notsigned - a signed binary with a byte APPENDED, which breaks the
#               certificate table and reports NotSigned
#   mismatch  - a signed binary with a byte patched in the MIDDLE, which keeps
#               the table and reports HashMismatch. Both are reachable and they
#               are different statuses; a gate written against only one of them
#               would ship the other.
#   signed    - the untouched signed binary (publisher is not SignPath)
#   accepted  - the untouched signed binary, with the step's expected publisher
#               rewritten to this host's. The ONLY row that reaches the PASS.
#   casefolded- the same substitution with the case swapped, which the
#               case-sensitive comparison must reject. A publisher name differing
#               only in case is a DIFFERENT publisher; PowerShell's -ne, -eq and
#               -like are all case-insensitive, so the shipped check said -ne and
#               meant nothing of the sort. This row is the only one that can see
#               the difference between -ne and -cne.
AUTHENTICODE_TRUTH_TABLE: tuple[tuple[str, str, int, str], ...] = (
    ("no staged installer", "missing", 1, "staged installer missing"),
    ("the download was skipped and the installer is unsigned", "notsigned", 1,
     "Authenticode status is 'NotSigned'"),
    ("the installer was modified after signing", "mismatch", 1,
     "Authenticode status is 'HashMismatch'"),
    ("someone else signed it", "signed", 1, "installer publisher is"),
    ("SignPath signed it", "accepted", 0, "PASS"),
    ("a publisher whose name differs from the expected one only in case",
     "casefolded", 1, "installer publisher is"),
)

#: The fixtures above that DO NOT run the shipped step body verbatim: they
#: rewrite the expected publisher name before running it. Exported because the
#: receipt in scripts/negative-controls/authenticode-step-receipt.py reports how
#: many rows ran verbatim and how many ran with a substitution, and a count
#: restated by hand goes stale the moment a substituted row is added -- which is
#: exactly the "a claim that no longer matches what was measured" defect this
#: file exists to catch. Derive it from here instead.
AUTHENTICODE_SUBSTITUTED_KINDS: frozenset[str] = frozenset({"accepted", "casefolded"})


class AuthenticodeRun(NamedTuple):
    """What one pass over the truth table actually measured.

    `ran` and `failed` exist so a mutation control can say something stronger
    than "some violation appeared": it can require that the violation came from
    a row capable of catching THAT mutation, and that the row ran at all. A
    mutation that merely makes a row unbuildable produces a violation too, and
    without these two sets it is indistinguishable from a caught mutation.
    """

    violations: list[str]
    unchecked: list[str]
    ran: set[str]
    failed: set[str]
    #: Why the unchecked rows went unchecked, as a HOST capability rather than
    #: an operating system: "fixture" (all rows buildable), "no-shell",
    #: "no-support" (a real incapacity), or "none-found" (a broken probe or
    #: trust store, which is a failure anywhere).
    capability: str = "fixture"


def authenticode_behaviour_violations(
    release: str, only: frozenset[str] | None = None
) -> AuthenticodeRun:
    """Run the shipped Authenticode step against real signed and broken files.

    Windows itself answers, over files whose signature state was produced rather
    than asserted. One qualification: the `accepted` row substitutes the expected
    publisher, since no host running this has a SignPath-signed file. It measures
    that a valid signature from the expected publisher reaches PASS; the literal
    `SignPath Foundation` is pinned by a separate static contract, and the
    publisher mutation below is what shows the comparison is load-bearing. That
    static contract is all substring markers, so an `exit 0` inserted at the top
    of the body satisfies every one of them and ships an unsigned installer --
    the exact defect class this workstream exists to catch, in the gate that is
    supposed to catch it.

    Coverage degrades by ROW, not all-or-nothing. The `missing` row needs no
    Authenticode at all -- the step throws on Test-Path before it reaches the
    cmdlet -- so it runs anywhere pwsh exists, including the Ubuntu lane that
    actually runs this suite in CI, and it is the row that catches an early
    `exit 0`. The four signature-state rows need Windows. The caller is told
    which kinds ran so it can require exactly the mutations those rows can catch,
    and name the rest UNCHECKED rather than passing over them.

    `only` restricts the pass to the named kinds. A mutation control already
    knows which rows can catch its mutation, and the rows that cannot are pure
    cost: each one builds a fixture and starts a shell to re-measure something
    the baseline established. Restricting the pass cannot weaken the control,
    because both of its demands -- that every catcher RAN, and that every catcher
    FAILED -- are subsets of `only`.
    """
    script = authenticode_script(release)
    shell = _powershell()
    if shell is None:
        return AuthenticodeRun(
            [], ["no pwsh or powershell on this host"], set(), set(), "no-shell"
        )
    violations: list[str] = []
    unchecked: list[str] = []
    ran: set[str] = set()
    failed: set[str] = set()
    with tempfile.TemporaryDirectory() as work:
        capability, signed_path, publisher = _signed_fixture(shell, work)
        why_not = {
            "no-support": "Get-AuthenticodeSignature is not usable on this host",
            "none-found": "the search of this host's system binary directories found "
            "no validly signed file, which is a broken probe or trust store, not an "
            "incapacity",
            # The reason travels in the publisher slot for this one arm; see
            # FixtureResult. Naming it here is what keeps a crashed probe from
            # reading, in the log a reviewer skims, like a platform that cannot
            # do Authenticode.
            "probe-failed": "the capability probe did not answer at all "
            f"({publisher}), which is a failed measurement, not an incapacity",
        }.get(capability, "")
        for description, kind, expected_status, required in AUTHENTICODE_TRUTH_TABLE:
            if only is not None and kind not in only:
                continue
            body = script
            staged = os.path.join(work, f"Wenlan_9.9.9_x64-setup-{kind}.exe")
            if kind == "missing":
                pass
            elif capability != "fixture":
                unchecked.append(f"{description}: {why_not}")
                continue
            else:
                assert signed_path is not None and publisher is not None
                data = bytearray(Path(signed_path).read_bytes())
                if kind == "notsigned":
                    data += b"X"
                elif kind == "mismatch":
                    offset, why = _authenticode_hashed_byte(bytes(data))
                    if offset is None:
                        unchecked.append(f"{description}: {why}")
                        continue
                    data[offset] ^= 0xFF
                    # Ask Authenticode about the fixture BEFORE the step does.
                    #
                    # Not `data[offset] != was` after `data[offset] ^= 0xFF`:
                    # that is true by arithmetic, and item assignment cannot
                    # change a bytearray's length either. A control that cannot
                    # fail, sitting inside the fixture builder for the row whose
                    # whole purpose is to enforce that controls can fail.
                    #
                    # What can actually go wrong is that the chosen byte turns
                    # out not to be covered by the signature. Then the step
                    # accepts the file, the row sees exit 0, and it reports a
                    # shipped gate that waves through tampered installers --
                    # when the truth is a fixture that could not pose the
                    # question. Asking here attributes it correctly, and this
                    # check CAN fail: it did, against a deliberately clamped
                    # offset at the end of the file, which comes back 'Valid'.
                    Path(staged).write_bytes(bytes(data))
                    built = _authenticode_status(shell, staged)
                    if built != "HashMismatch":
                        unchecked.append(
                            f"{description}: byte {offset} of this host's "
                            f"fixture is not covered by its signature -- the "
                            f"tampered copy reports {built!r} rather than "
                            "'HashMismatch', so this row cannot ask its "
                            "question here"
                        )
                        continue
                elif kind in AUTHENTICODE_SUBSTITUTED_KINDS:
                    # The OPERATOR is matched, not assumed. A mutation that
                    # swaps -cne for -ne must leave both rows buildable, or the
                    # row that exists to catch that swap reports "could not be
                    # built" instead of "caught it" -- and the runner below can
                    # tell those apart only because the row still runs.
                    found = re.findall(
                        r"-c?ne '(?:SignPath Foundation)'", body
                    )
                    if len(found) != 1:
                        violations.append(
                            "the expected-publisher comparison is no longer a "
                            "single -cne/-ne against 'SignPath Foundation'; the "
                            f"{kind} row cannot be built"
                        )
                        continue
                    needle = found[0]
                    operator = needle.split(" ", 1)[0]
                    # The one documented substitution, made by exactly the rows
                    # named in AUTHENTICODE_SUBSTITUTED_KINDS: this host has no
                    # SignPath-signed file, so these rows swap the publisher the
                    # step demands for the publisher this host can actually
                    # produce (the casefolded row then re-cases it). Everything
                    # else -- the Valid check, the null-certificate check, the
                    # SimpleName parse, the PASS -- is the shipped code running
                    # against a real signature.
                    wanted = publisher
                    if kind == "casefolded":
                        # The same certificate, the same name, one different
                        # casing. -cne rejects it; -ne, -eq, -like and -ine all
                        # accept it, and this is the only row in the table that
                        # can tell those two comparisons apart.
                        wanted = publisher.swapcase()
                        if wanted == publisher:
                            unchecked.append(
                                f"{description}: this host's fixture publisher "
                                f"{publisher!r} has no cased letters, so a "
                                "case-folded variant of it does not exist"
                            )
                            continue
                    body = body.replace(needle, f"{operator} '{wanted}'")
                Path(staged).write_bytes(bytes(data))
            step_file = os.path.join(work, f"step-{kind}.ps1")
            Path(step_file).write_text(body, encoding="utf-8")
            result = subprocess.run(
                [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", step_file],
                env={**os.environ, **AUTHENTICODE_STEP_ENV, "STAGED_SETUP": staged},
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            ran.add(kind)
            if result.returncode != expected_status:
                failed.add(kind)
                violations.append(
                    f"Authenticode step: {description}: expected exit "
                    f"{expected_status}, got {result.returncode}; output="
                    f"{output.strip()[:400]!r}"
                )
            elif required not in output:
                # A row can exit for the right reason and still not prove its
                # own claim. The tampered-installer row asks for HashMismatch;
                # if this host's signed binary refuses with some other status
                # the gate did hold -- the installer was rejected -- but the
                # HashMismatch branch went unexercised. That is the third
                # state, and it is reported with the status that actually came
                # back, so a reader can tell it from both a pass and a
                # regression. The row also leaves `ran`, which is what stops
                # the mutation loop below from crediting it with catching
                # anything.
                status = re.search(r"^Status: (\S+)", output, re.MULTILINE)
                if (
                    kind == "mismatch"
                    and status is not None
                    and status.group(1) != "HashMismatch"
                    and "Authenticode status is" in output
                ):
                    unchecked.append(
                        f"{description}: this host's fixture reports "
                        f"{status.group(1)!r} when tampered rather than "
                        "'HashMismatch'; the step refused it either way, but "
                        "nothing here exercised the hash-mismatch branch"
                    )
                    ran.discard(kind)
                    continue
                failed.add(kind)
                violations.append(
                    f"Authenticode step: {description}: exited {result.returncode} "
                    f"without saying why; expected {required!r} in {output.strip()[:400]!r}"
                )
    return AuthenticodeRun(violations, unchecked, ran, failed, capability)


class InvarianceRun(NamedTuple):
    violations: list[str]
    unchecked: list[str]
    ran: list[str]


def _contrary_step_env() -> dict[str, str]:
    """Every variable the release supplies, set to something else entirely."""
    contrary = {
        "SIGNPATH_CONFIGURED": "false",
        "CI": "false",
        "GITHUB_ACTIONS": "false",
        "RUNNER_OS": "Linux",
        "RUNNER_ARCH": "ARM64",
        "RUNNER_ENVIRONMENT": "self-hosted",
        "GITHUB_JOB": "docs",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_REF_TYPE": "branch",
        "GITHUB_REF_NAME": "main",
        "GITHUB_REF_PROTECTED": "true",
        "GITHUB_WORKFLOW": "ci",
        "GITHUB_WORKFLOW_REF": "someone/else/.github/workflows/ci.yml@refs/heads/main",
        "GITHUB_REPOSITORY": "someone/else",
        "GITHUB_REPOSITORY_OWNER": "someone",
        "GITHUB_SERVER_URL": "https://example.invalid",
        "GITHUB_API_URL": "https://example.invalid/api",
    }
    return {
        name: contrary.get(name, "wrong-" + name.lower())
        for name in AUTHENTICODE_STEP_ENV
    }


def authenticode_environment_invariance(release: str) -> InvarianceRun:
    """The step's outcome may not depend on the environment. A property, run.

    Every other guard on this is LEXICAL: `authenticode_body_env_violations`
    blanks the one permitted read and rejects the surviving token `env`,
    case-insensitively,
    which is a good net and still only a net. PowerShell can assemble the drive
    name at runtime --

        if ((Get-Item ('E' + 'nv:GITHUB_WORKFLOW_REF')).Value) { exit 0 }

    -- and the token never appears. Widening `AUTHENTICODE_STEP_ENV` to
    GitHub's full documented set closes the other half of that particular
    example, but "enumerate the variables" and "enumerate the spellings" are the
    same losing game, one level apart.

    So this asks the question directly, and by running the step rather than by
    reading it: the same row, three times, under the release's own environment,
    under none of it, and under every value replaced by a wrong one. A gate
    whose answer depends on any of that is a gate that can be switched off from
    the workflow, however it is spelled. Nothing here reads the body's text.

    The `missing` row needs no fixture, so this property is measured even on a
    lane with no Authenticode at all -- which is where a bypass would be least
    likely to be noticed.
    """
    script = authenticode_script(release)
    shell = _powershell()
    if shell is None:
        return InvarianceRun([], ["no pwsh or powershell on this host"], [])
    violations: list[str] = []
    unchecked: list[str] = []
    ran: list[str] = []
    with tempfile.TemporaryDirectory() as work:
        capability, signed_path, reason = _signed_fixture(shell, work)
        step_file = os.path.join(work, "invariance.ps1")
        Path(step_file).write_text(script, encoding="utf-8")
        rows = [("missing", os.path.join(work, "no-such-installer.exe"))]
        if capability == "fixture" and signed_path is not None:
            staged = os.path.join(work, "Wenlan_9.9.9_x64-setup-signed.exe")
            Path(staged).write_bytes(Path(signed_path).read_bytes())
            rows.append(("signed", staged))
        else:
            unchecked.append(
                "the `signed` row needs a signed binary; the capability probe "
                f"answered {capability!r}"
                + (f" ({reason})" if capability == "probe-failed" else "")
            )
        # The base environment with every name the release supplies REMOVED, so
        # "nothing at all" really is nothing rather than whatever this shell
        # happens to be carrying.
        base = {k: v for k, v in os.environ.items() if k not in AUTHENTICODE_STEP_ENV}
        contrary = _contrary_step_env()
        for kind, staged in rows:
            outcomes: dict[str, tuple[int, str]] = {}
            for label, extra in (
                ("the release's own values", AUTHENTICODE_STEP_ENV),
                ("none of them set", {}),
                ("every one of them wrong", contrary),
            ):
                result = subprocess.run(
                    [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", step_file],
                    env={**base, **extra, "STAGED_SETUP": staged},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                outcomes[label] = (
                    result.returncode,
                    (result.stdout + result.stderr).strip()[:300],
                )
            ran.append(kind)
            if len(set(outcomes.values())) != 1:
                violations.append(
                    f"the {kind!r} row's outcome depends on the environment, and "
                    "the step is allowed to read STAGED_SETUP and nothing else: "
                    + "; ".join(
                        f"{label} -> exit {rc} {out[:80]!r}"
                        for label, (rc, out) in outcomes.items()
                    )
                )
    return InvarianceRun(violations, unchecked, ran)


# Columns: the shipped text, its replacement, the truth-table rows that can
# catch the replacement, and why it matters.
#
# The catcher sets exist because coverage here is host-dependent: the four
# signature-state rows need an Authenticode-signed file, which Ubuntu does not
# have, and skipping ALL of the mutations whenever ANY row is unavailable prints
# PASS over nothing. Naming the catchers per mutation means the rows that DID
# run still have to earn their keep, and the rest are reported UNCHECKED by name
# rather than passed over.
#
# The first mutation is the one that matters on a fixture-free host: `exit 0`
# above `$ErrorActionPreference` reaches nothing at all, so even the `missing`
# row -- which needs no signature, only a Test-Path throw -- catches it. The
# second sits below the Test-Path guard, so `missing` still exits 1 and only the
# fixture rows can see it. Two different mutations because one of them is
# checkable in the Ubuntu `docs` job that actually runs this suite in CI.
AUTHENTICODE_MUTATIONS: tuple[tuple[str, str, frozenset[str], str], ...] = (
    (
        '          $ErrorActionPreference = "Stop"\n'
        "          if (-not (Test-Path -LiteralPath $env:STAGED_SETUP)) {\n",
        '          $ErrorActionPreference = "Stop"\n'
        "          exit 0\n"
        "          if (-not (Test-Path -LiteralPath $env:STAGED_SETUP)) {\n",
        frozenset({"missing", "notsigned", "mismatch", "signed", "accepted"}),
        "an early return above every one of the step's own assertions",
    ),
    (
        "          $signature = Get-AuthenticodeSignature -LiteralPath $env:STAGED_SETUP\n",
        "          exit 0\n"
        "          $signature = Get-AuthenticodeSignature -LiteralPath $env:STAGED_SETUP\n",
        frozenset({"notsigned", "mismatch", "signed", "accepted"}),
        "an early return that reaches none of the signature assertions",
    ),
    (
        "          if ($signature.Status -ne 'Valid') {\n"
        "            throw \"Authenticode status is '$($signature.Status)',"
        " expected 'Valid'. $($signature.StatusMessage)\"\n"
        "          }\n",
        "",
        frozenset({"notsigned", "mismatch"}),
        "the Authenticode status check",
    ),
    (
        "          if ($publisher -cne 'SignPath Foundation') {\n"
        "            throw \"installer publisher is '$publisher',"
        " expected 'SignPath Foundation'\"\n"
        "          }\n",
        "",
        # `signed` alone, deliberately. Deleting the comparison also deletes the
        # line the `accepted` and `casefolded` rows rewrite, so those two rows
        # become unbuildable rather than failing -- bookkeeping, not detection,
        # exactly as the comment on the runner below says.
        frozenset({"signed"}),
        "the publisher check, which is what makes it SignPath's signature",
    ),
    (
        # The one mutation `mismatch` alone can see, and the reason that row is
        # worth the PE parsing it costs. Widening the status check to admit
        # HashMismatch ships a gate that accepts an installer modified after
        # signing, which is the whole threat this step exists for -- and every
        # other row still passes: `notsigned` still throws (NotSigned is
        # neither Valid nor HashMismatch), `missing` never reaches the check,
        # and the three Valid rows are untouched. Without this entry the
        # catcher sets only ever list `mismatch` alongside `notsigned`, so the
        # mutation loop could never demand that the row catch anything by
        # itself, and the row's unique claim was stated rather than enforced.
        "          if ($signature.Status -ne 'Valid') {\n",
        "          if ($signature.Status -ne 'Valid' -and "
        "$signature.Status -ne 'HashMismatch') {\n",
        frozenset({"mismatch"}),
        "the status check's refusal of a file modified after signing",
    ),
    (
        # One operator, and the defect it hides is invisible to every other row
        # in the table: -ne compares case-insensitively, so the check keeps
        # rejecting a different name and quietly stops rejecting a different
        # CASING of the right one. `casefolded` is the only catcher, which is
        # exactly why that row exists.
        "          if ($publisher -cne 'SignPath Foundation') {\n",
        "          if ($publisher -ne 'SignPath Foundation') {\n",
        frozenset({"casefolded"}),
        "the case-SENSITIVE publisher comparison",
    ),
)


def assert_mutation_detected(
    ci: str,
    release: str,
    release_please: str,
    release_please_config: str,
    fast_maintenance: str,
    promotion: str,
    sync_release_pr: str,
    old: str,
    new: str,
    expected: str,
    *,
    owner: str,
) -> None:
    documents = {
        "ci": ci,
        "release": release,
        "release_please": release_please,
        "release_please_config": release_please_config,
        "fast_maintenance": fast_maintenance,
        "promotion": promotion,
        "sync_release_pr": sync_release_pr,
    }
    source = documents[owner]
    if old not in source:
        raise AssertionError(f"mutation fixture is stale; missing {old!r}")
    documents[owner] = source.replace(old, new, 1)
    violations = contract_violations(
        documents["ci"],
        documents["release"],
        documents["release_please"],
        documents["release_please_config"],
        documents["fast_maintenance"],
        documents["promotion"],
        documents["sync_release_pr"],
    )
    if not any(expected in violation for violation in violations):
        raise AssertionError(
            f"mutation did not exercise {expected!r}: {violations!r}"
        )


# The Windows desktop bundle is built by two hand-copied recipes:
# ci.yml's `app-windows-bundle`, which anyone can dispatch and which has
# actually run, and release.yml's `app-bundle-windows`, which nothing exercises
# until a real release is cut. Whatever the proven one needs to produce a
# working installer, the unproven one needs too, and a change to either alone
# is drift nobody would notice until release day.
#
# Each marker is asserted present in BOTH files, so editing one side is a
# failure that names the other. That also keeps this list honest: a marker
# deleted from CI cannot silently stop being checked.
WINDOWS_BUILD_RECIPE_MARKERS = [
    "runs-on: windows-2022",
    "targets: x86_64-pc-windows-msvc",
    "bash scripts/stabilize-rust-cache-toolchains.sh",
    # libsql does not bundle SQLite on Windows.
    "vcpkg install sqlite3:x64-windows-static-md",
    "& scripts/setup-vulkan-sdk-windows.ps1",
    "& scripts/setup-msvc-ninja-windows.ps1",
    # The two libraries the daemon dynamically loads, staged where
    # app/tauri.windows.conf.json bundles them from.
    "& scripts/stage-onnxruntime-windows.ps1 -DestinationDirectory $dllDir",
    "& scripts/stage-vulkan-loader-windows.ps1 -DestinationDirectory $dllDir",
    "pnpm tauri build --target x86_64-pc-windows-msvc",
    # Both jobs unpack the installer they produce and look inside it, and both
    # refuse a hit in NSIS's scratch directory, which is deleted when the
    # installer exits and so is not a file the app ships.
    "onnxruntime.dll",
    "vulkan-1.dll",
    "VulkanRT-License.txt",
    "$PLUGINSDIR",
]


def windows_recipe_drift_violations(ci: str, release: str) -> list[str]:
    violations: list[str] = []
    ci_job = job_body(ci, "app-windows-bundle")
    release_job = job_body(release, "app-bundle-windows")
    if not ci_job:
        return ["ci.yml no longer defines app-windows-bundle, the proven Windows recipe"]
    if not release_job:
        return ["release.yml no longer defines app-bundle-windows"]
    for marker in WINDOWS_BUILD_RECIPE_MARKERS:
        in_ci = marker in ci_job
        in_release = marker in release_job
        if in_ci and in_release:
            continue
        missing, present = (
            ("ci.yml app-windows-bundle", "release.yml app-bundle-windows")
            if not in_ci
            else ("release.yml app-bundle-windows", "ci.yml app-windows-bundle")
        )
        violations.append(
            f"Windows build recipes have drifted: {marker!r} is in {present} "
            f"but not in {missing}"
        )
    # The one difference that is deliberate. CI publishes nothing and nobody
    # installs its output, so it mints a throwaway updater keypair per run
    # instead of borrowing the real release secret; the release job is the only
    # Windows job that may touch that secret.
    if "tauri signer generate" not in ci_job:
        violations.append(
            "ci.yml app-windows-bundle no longer mints a throwaway updater key"
        )
    if "secrets.TAURI_SIGNING_PRIVATE_KEY" in ci_job:
        violations.append(
            "ci.yml app-windows-bundle borrows the real updater signing secret"
        )
    if "secrets.TAURI_SIGNING_PRIVATE_KEY" not in release_job:
        violations.append(
            "release.yml app-bundle-windows no longer signs with the release key"
        )
    if "tauri signer generate" in release_job:
        violations.append(
            "release.yml app-bundle-windows signs installers with a throwaway key"
        )
    return violations


def main() -> None:
    publish_helper_tests = subprocess.run(
        [sys.executable, str(PUBLISH_CRATE_TEST_PATH)],
        check=False,
    )
    if publish_helper_tests.returncode != 0:
        raise AssertionError("crates.io publish helper contracts failed")
    ci = CI_PATH.read_text(encoding="utf-8")
    release = RELEASE_PATH.read_text(encoding="utf-8")
    release_please = RELEASE_PLEASE_PATH.read_text(encoding="utf-8")
    release_please_config = RELEASE_PLEASE_CONFIG_PATH.read_text(encoding="utf-8")
    fast_maintenance = FAST_MAINTENANCE_PATH.read_text(encoding="utf-8")
    observer = OBSERVER_PATH.read_text(encoding="utf-8")
    validator = VALIDATOR_PATH.read_text(encoding="utf-8")
    classifier = CLASSIFIER_PATH.read_text(encoding="utf-8")
    archive = ARCHIVE_PATH.read_text(encoding="utf-8")
    promotion = PROMOTION_PATH.read_text(encoding="utf-8")
    sync_release_pr = SYNC_RELEASE_PR_PATH.read_text(encoding="utf-8")
    violations = contract_violations(
        ci,
        release,
        release_please,
        release_please_config,
        fast_maintenance,
        promotion,
        sync_release_pr,
    )
    violations.extend(windows_recipe_drift_violations(ci, release))
    violations.extend(desktop_link_order_violations(release))
    violations.extend(candidate_observer_contract_violations(ci, observer, validator, archive))
    violations.extend(trusted_candidate_gate_violations(ci, classifier, validator))
    violations.extend(signpath_status_violations())
    if violations:
        raise AssertionError("release workflow contract drift:\n" + "\n".join(violations))

    release_mutations = [
        (
            "associated_pulls=associated",
            "associated_pulls=None",
            "one commit association snapshot",
            "promotion",
        ),
        (
            "Start-Sleep -Seconds 25",
            "Start-Sleep -Seconds 5",
            "backoff is not exactly 25 seconds",
            "ci",
        ),
        (
            "      - name: Retry Windows release cache restore once\n"
            "        id: windows-release-cache-retry\n"
            "        if: matrix.target == 'x86_64-pc-windows-msvc' && steps.windows-cache-probe.outputs.state == 'cold-miss'\n"
            "        uses: Swatinem/rust-cache@e18b497796c12c097a38f9edb9d0641fb99eee32 # v2",
            "      - name: Retry Windows release cache restore once\n"
            "        id: windows-release-cache-retry\n"
            "        if: matrix.target == 'x86_64-pc-windows-msvc' && steps.windows-cache-probe.outputs.state == 'cold-miss'\n"
            "        uses: Swatinem/rust-cache@v2",
            "single pinned restore-only attempt",
            "ci",
        ),
        (
            "if: matrix.target == 'x86_64-pc-windows-msvc' && steps.windows-cache-probe.outputs.state == 'cold-miss'",
            "if: matrix.target == 'x86_64-pc-windows-msvc' && true",
            "gated by the measured cold miss",
            "ci",
        ),
        (
            '$state = "cold-miss"\n            $jobs = 2',
            '$state = "cold-miss"\n            $jobs = 1',
            "final Windows cache receipt",
            "ci",
        ),
        (
            '"host-count=$hostCount" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
            '"host-total=$hostCount" | Out-File -FilePath $env:GITHUB_OUTPUT -Append',
            "initial Windows cache probe",
            "ci",
        ),
        (
            "--wait-seconds 720",
            "--wait-seconds 0",
            "thin receipt contract",
            "ci",
        ),
        (
            "- '.github/workflows/release-pr-maintenance.yml'",
            "- '.github/workflows/release-pr-maintenance-disabled.yml'",
            "bootstrap its Rust contract",
            "ci",
        ),
        (
            "skip-github-release: true",
            "skip-github-release: false",
            "PR-only contract",
            "release_please",
        ),
        (
            '"always-update": true',
            '"always-update": false',
            "always-update is not exact true",
            "release_please_config",
        ),
        (
            '      "always-update": true,\n',
            "",
            "always-update is not exact true",
            "release_please_config",
        ),
        (
            "  push:\n    branches: [main]",
            "  workflow_dispatch:\n    branches: [main]",
            "exact main push",
            "fast_maintenance",
        ),
        (
            "skip-github-release: true",
            "skip-github-release: false",
            "fast release maintenance omits",
            "fast_maintenance",
        ),
        (
            "ref: ${{ steps.release_pr.outputs.release_pr_head_sha }}",
            "ref: release-please--branches--main",
            "exact checkout omits",
            "fast_maintenance",
        ),
        (
            "if not payload:",
            "if False:",
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            "if len(payload) != 1:",
            "if False:",
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            "if owner != RELEASE_AUTHOR:",
            "if False:",
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            'user.get("login") != RELEASE_AUTHOR',
            'user.get("login") == RELEASE_AUTHOR',
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            "current != expected_head_sha",
            "False",
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            "remote_shas != [expected_head_sha]",
            "False",
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            '["git", "merge", "--no-edit", "origin/main"]',
            '["git", "rebase", "origin/main"]',
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            '"HEAD:refs/heads/release-please--branches--main"',
            '"+HEAD:refs/heads/release-please--branches--main"',
            "synchronizer omits",
            "sync_release_pr",
        ),
        (
            "group: release-pr-maintenance-main",
            "group: release-please-main",
            "fast release maintenance omits",
            "fast_maintenance",
        ),
        (
            "queue: max",
            "queue: single",
            "fast release maintenance omits",
            "fast_maintenance",
        ),
        (
            "github.event.workflow_run.conclusion == 'success'",
            "github.event.workflow_run.conclusion != 'success'",
            "release-please main route omits",
            "release_please",
        ),
        (
            "      - name: Checkout trusted release PR synchronizer\n"
            "        uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6\n"
            "        with:\n"
            "          # workflow_run's github.sha is the immutable default-branch commit\n"
            "          # that owns this privileged workflow. The observed CI head may be an\n"
            "          # older main commit that predates the synchronizer itself.\n"
            "          ref: ${{ github.sha }}",
            "      - name: Checkout trusted release PR synchronizer\n"
            "        uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6\n"
            "        with:\n"
            "          ref: ${{ github.event.workflow_run.head_sha }}",
            "fallback maintenance trusted checkout omits",
            "release_please",
        ),
        (
            "MAIN_SHA: ${{ github.event.workflow_run.head_sha }}",
            "MAIN_SHA: ${{ github.sha }}",
            "validated tag creation omits",
            "release_please",
        ),
        (
            "scripts/release-promotion.py consume-main-receipt",
            "scripts/release-promotion.py gate-main",
            "consume-main-receipt",
            "release_please",
        ),
        (
            "scripts/release-promotion.py download-assets",
            "scripts/build-release-binaries.sh",
            "recompile",
            "release",
        ),
        (
            "GH_TAG_TOKEN: ${{ secrets.RELEASE_TOKEN }}",
            "GH_TAG_TOKEN: ${{ github.token }}",
            "isolated write authority",
            "release",
        ),
        (
            "event=push&head_sha=$RELEASE_SHA",
            "event=push",
            "isolated write authority",
            "release",
        ),
        (
            "/actions/runs/$legacy_run_id/cancel",
            "/actions/runs/$GITHUB_RUN_ID/cancel",
            "isolated write authority",
            "release",
        ),
        (
            "--package wenlan-types",
            "--package unverified-types",
            "bypasses publish helper",
            "release",
        ),
        (
            "--package wenlan-mcp",
            "--package unverified-mcp",
            "bypasses publish helper",
            "release",
        ),
        (
            "  publish-crates:\n"
            "    name: Publish crates.io (wenlan-types + wenlan-mcp)\n"
            "    needs: [promote-assets, bind-release-tag]\n"
            "    runs-on: ubuntu-latest\n"
            "    timeout-minutes: 15",
            "  publish-crates:\n"
            "    name: Publish crates.io (wenlan-types + wenlan-mcp)\n"
            "    needs: [promote-assets, bind-release-tag]\n"
            "    runs-on: ubuntu-latest\n"
            "    timeout-minutes: 150",
            "15-minute bound",
            "release",
        ),
        (
            "docker/Dockerfile.release-runtime",
            "docker/Dockerfile.daemon",
            "compile a different",
            "release",
        ),
        (
            "publish-npm:\n    name: Publish to npm\n    needs: [promote-assets, bind-release-tag]",
            "publish-npm:\n    name: Publish to npm\n    needs: publish-crates",
            "serialized behind crates.io",
            "release",
        ),
        (
            'index("autorelease: pending") == null',
            'index("autorelease: pending") != null',
            "closed-state assertion",
            "release",
        ),
        (
            "MAX_RECEIPT_CANDIDATES = 20",
            "MAX_RECEIPT_CANDIDATES = 1000",
            "MAX_RECEIPT_CANDIDATES",
            "promotion",
        ),
        (
            "observer reruns produced conflicting release semantics",
            "observer reruns are accepted",
            "conflicting release semantics",
            "promotion",
        ),
        # The Homebrew formula must ship the daemon: an install line moved
        # into a comment, or only the URL pointing back at the CLI-only
        # archive, must each be caught on its own.
        (
            '              bin.install "wenlan", "wenlan-server"',
            '              # bin.install "wenlan", "wenlan-server"\n              bin.install "wenlan"',
            "Homebrew wenlan formula omits 'bin.install",
            "release",
        ),
        (
            'vVERSION_PLACEHOLDER/wenlan-darwin-arm64.tar.gz"',
            'vVERSION_PLACEHOLDER/wenlan-cli-darwin-arm64.tar.gz"',
            "still installs the CLI-only archive",
            "release",
        ),
        # ---- Windows Authenticode signing ----------------------------------
        # None of these steps runs today, and none of them will run until
        # SignPath Foundation accepts the application. Every failure they guard
        # against is silent, so each one gets a mutation that proves the check
        # bites rather than merely existing.
        (
            "output-artifact-directory: ${{ runner.temp }}/signpath-signed",
            "# output-artifact-directory removed",
            "omits output-artifact-directory",
            "release",
        ),
        (
            "uses: SignPath/github-action-submit-signing-request@c92b958760219087e01f8d67a1669ed57afe2627",
            "uses: SignPath/github-action-submit-signing-request@v2",
            "mutable or unexpected reference",
            "release",
        ),
        # Owner casing matters: the Node 24 pin allowlist looks the action up by
        # its exact path and silently skips anything it does not recognise, so a
        # lowercase spelling would leave the pin unenforced.
        (
            "uses: SignPath/github-action-submit-signing-request@c92b958760219087e01f8d67a1669ed57afe2627",
            "uses: signpath/github-action-submit-signing-request@c92b958760219087e01f8d67a1669ed57afe2627",
            "the Node 24 pin allowlist skips any action it does not recognise",
            "release",
        ),
        (
            f"      - name: {SIGNPATH_GUARD_STEP}\n        shell: bash\n",
            f"      - name: {SIGNPATH_GUARD_STEP}\n"
            "        if: env.SIGNPATH_CONFIGURED == 'true'\n        shell: bash\n",
            "guarded by the very sentinel it exists to police",
            "release",
        ),
        # The same defect through the other sentinel. A latch that skips
        # whenever signing is not required can never run in the state it exists
        # to reject: SIGNPATH_REQUIRED true and the secrets absent is precisely
        # the configuration this `if:` would evaluate as... true, and then the
        # step runs -- but the reverse spelling, or a maintainer "optimising"
        # the step away when signing is off, is one character from here.
        (
            f"      - name: {SIGNPATH_GUARD_STEP}\n        shell: bash\n",
            f"      - name: {SIGNPATH_GUARD_STEP}\n"
            "        if: env.SIGNPATH_REQUIRED == 'true'\n        shell: bash\n",
            "must not skip when signing is not required",
            "release",
        ),
        (
            "      - name: Replace the unsigned installer with the signed one\n"
            "        if: env.SIGNPATH_CONFIGURED == 'true'\n",
            "      - name: Replace the unsigned installer with the signed one\n",
            "is not guarded",
            "release",
        ),
        (
            "      actions: read\n      contents: read\n    env:\n"
            "      # TWO questions",
            "      contents: read\n    env:\n      # TWO questions",
            "does not grant actions: read",
            "release",
        ),
        (
            'wait-for-completion-timeout-in-seconds: "5400"',
            'wait-for-completion-timeout-in-seconds: "600"',
            "is not raised past the",
            "release",
        ),
        (
            'skip-decompress: "false"',
            'skip-decompress: "true"',
            "skip-decompress",
            "release",
        ),
        (
            "    timeout-minutes: 165",
            "    timeout-minutes: 120",
            "165-minute bound",
            "release",
        ),
        # Staging that lands before signing publishes unsigned bytes or a digest
        # that no longer matches the file. Nothing else in this repository
        # checks the order of these steps.
        (
            "      - name: Upload the unsigned installer for SignPath\n",
            "      - name: Stage app bundle assets and checksums\n"
            "      - name: Upload the unsigned installer for SignPath\n",
            "must come after",
            "release",
        ),
        (
            "if ($before -eq $after) {",
            "if ($false) {",
            "does not compare content",
            "release",
        ),
        (
            "          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}\n"
            "        run: |\n          set -euo pipefail\n          nsis_dir=",
            "        run: |\n          set -euo pipefail\n          nsis_dir=",
            "own copy of the signing",
            "release",
        ),
        (
            "$publisher -cne 'SignPath Foundation'",
            "$false",
            "Authenticode assertion is incomplete",
            "release",
        ),
        # And the operator on its own. Losing the `c` leaves every marker in
        # place, reads as a publisher check, and compares case-insensitively --
        # which is no identity check at all.
        (
            "$publisher -cne 'SignPath Foundation'",
            "$publisher -ne 'SignPath Foundation'",
            "case-INSENSITIVE operator",
            "release",
        ),
        # The certificate values are the whole record of what SignPath
        # Foundation's certificate says, and nobody has seen it yet.
        (
            '          Write-Host "Subject:    $($certificate.Subject)"\n',
            "",
            "never logs",
            "release",
        ),
        (
            '          Write-Host "Thumbprint: $($certificate.Thumbprint)"\n',
            "",
            "never logs",
            "release",
        ),
        # Logged, but after the comparison that throws: the failing run, which
        # is the run whose values someone needs, prints nothing.
        (
            '          Write-Host "Publisher:  $publisher"\n',
            "          if ($publisher -cne 'SignPath Foundation') { }\n"
            '          Write-Host "Publisher:  $publisher"\n',
            "logged after the publisher comparison",
            "release",
        ),
        # SignerCertificate.Subject is a distinguished name, so bare-string
        # equality against it is either always false or a substring match.
        (
            "          $publisher = $certificate.GetNameInfo(",
            "          $publisher = ($certificate.Subject -eq 'SignPath Foundation') # (",
            "compares the raw certificate Subject",
            "release",
        ),
        # ci.yml's Windows bundle job publishes nothing and nobody installs its
        # output, so it must not consume the signing credentials -- the same
        # asymmetry that keeps it off the real updater key.
        (
            "  app-windows-bundle:\n    name: app-windows-bundle\n",
            "  app-windows-bundle:\n    name: app-windows-bundle\n"
            "    # borrow the SignPath credentials here too\n",
            "ci.yml app-windows-bundle references SignPath",
            "ci",
        ),
        # ---- the lane that makes the Authenticode rows fatal ----
        # Each of these leaves a job that looks like coverage. The first deletes
        # it outright; the second leaves it running on a host that can measure
        # everything and requiring none of it; the third and fourth leave it
        # green-or-red with nothing reading the answer.
        (
            "  windows-release-contract:\n    name: windows release contract\n",
            "  windows-release-contract-disabled:\n    name: windows release contract\n",
            "has no 'windows-release-contract' job",
            "ci",
        ),
        (
            '          WENLAN_REQUIRE_AUTHENTICODE: "1"\n',
            "",
            "does not set WENLAN_REQUIRE_AUTHENTICODE",
            "ci",
        ),
        (
            "docs, windows-release-contract, plugin,",
            "docs, plugin,",
            "is not in conclusion's needs",
            "ci",
        ),
        (
            "          expect_job windows-release-contract ",
            "          # expect_job windows-release-contract ",
            "no expect_job line for windows-release-contract",
            "ci",
        ),
        # And the flag on a lane that cannot answer, which is the other way to
        # end up with nothing measured: a permanently red Ubuntu job gets the
        # flag removed again within a week.
        (
            "      - name: Validate README translations and eval provenance\n        run: |\n",
            "      - name: Validate README translations and eval provenance\n"
            "        env:\n"
            '          WENLAN_REQUIRE_AUTHENTICODE: "1"\n'
            "        run: |\n",
            "ubuntu docs lane sets WENLAN_REQUIRE_AUTHENTICODE",
            "ci",
        ),
    ]
    for old, new, expected, owner in release_mutations:
        assert_mutation_detected(
            ci,
            release,
            release_please,
            release_please_config,
            fast_maintenance,
            promotion,
            sync_release_pr,
            old,
            new,
            expected,
            owner=owner,
        )
    candidate_mutations = [
        (
            "needs.detect-changes.outputs.trusted-release-candidate != 'true'",
            "true",
            "duplicate base test skip",
            "ci",
        ),
        (
            "trusted-main-ci-proof/scripts/release-promotion.py verify-main-ci",
            "trusted-main-ci-proof/scripts/release-promotion.py skipped-main-ci-proof",
            "base CI proof omits",
            "ci",
        ),
        (
            "github.event.pull_request.head.repo.full_name == github.repository",
            "true",
            "exact head SHA",
            "ci",
        ),
        (
            "name: release-candidate-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.target }}",
            "name: release-candidate-latest",
            "immutable contract",
            "ci",
        ),
        (
            "path: dist/*",
            "path: .",
            "immutable contract",
            "ci",
        ),
        (
            "  actions: read\n  contents: read\n  pull-requests: read",
            "  actions: write\n  contents: write\n  pull-requests: write",
            "exact read-only",
            "observer",
        ),
        (
            "  workflow_run:\n",
            "  workflow_dispatch:\n  workflow_run:\n",
            "trigger is not exact",
            "observer",
        ),
        (
            "    branches: [release-please--branches--main]\n",
            "",
            "trigger is not exact",
            "observer",
        ),
        (
            "ref: ${{ github.sha }}",
            "ref: ${{ github.event.workflow_run.head_sha }}",
            "checkout is not the exact trusted",
            "observer",
        ),
        (
            "retention-days: 30",
            "retention-days: 14",
            "validated assets upload omits",
            "observer",
        ),
        (
            "name: validated-release-receipt-${{ github.event.workflow_run.id }}-${{ github.event.workflow_run.run_attempt }}",
            "name: validated-release-receipt-latest",
            "closed receipt upload omits",
            "observer",
        ),
        (
            "          overwrite: true",
            "          overwrite: false",
            "retry-safe locator",
            "observer",
        ),
        (
            '--observer-workflow-id "$observer_workflow_id"',
            "# observer workflow identity removed",
            "artifact binding",
            "observer",
        ),
        (
            "      - name: Validate release candidate as untrusted data\n",
            "      - name: Unexpected third step\n        run: echo no\n"
            "      - name: Validate release candidate as untrusted data\n",
            "exactly five closed-receipt steps",
            "observer",
        ),
        (
            "          GITHUB_TOKEN: ${{ github.token }}",
            "          EXTRA: value\n          GITHUB_TOKEN: ${{ github.token }}",
            "env or command is not exact",
            "observer",
        ),
        (
            "--summary \"$GITHUB_STEP_SUMMARY\"",
            "--summary \"$RUNNER_TEMP/summary\"",
            "env or command is not exact",
            "observer",
        ),
        (
            "safe_extract_zip(wrapper, extracted, outer_names)",
            "# outer ZIP validation removed",
            "fail-closed evidence",
            "validator",
        ),
        (
            "base_records, head_records = _validate_release_tree_modes(",
            "base_records, head_records = _release_tree_modes_removed(",
            "fail-closed evidence",
            "validator",
        ),
        (
            "new != old.replace(old_version, new_version)",
            "new == old",
            "fail-closed evidence",
            "validator",
        ),
        (
            "target_attempts = _latest_candidate_artifact_attempts(",
            "target_attempts = _jobs_first_attempt_guess(",
            "does not select artifacts",
            "validator",
        ),
        (
            "| Canonical inner asset |",
            "| Missing inner asset receipt |",
            "fail-closed evidence",
            "validator",
        ),
        (
            "expanded_size = _decompress_canonical_gzip(path, raw_path)",
            "expanded_size = 0 # gzip preflight removed",
            "hostile-input bound",
            "archive",
        ),
        (
            "raw_records = _validate_raw_tar(raw_path)",
            "raw_records = [] # hostile preflight removed",
            "hostile-input bound",
            "archive",
        ),
    ]
    for old, new, expected, owner in candidate_mutations:
        source = {
            "ci": ci,
            "observer": observer,
            "validator": validator,
            "archive": archive,
        }[owner]
        if old not in source:
            raise AssertionError(f"candidate mutation fixture is stale: {old!r}")
        mutated = source.replace(old, new, 1)
        candidate_violations = candidate_observer_contract_violations(
            mutated if owner == "ci" else ci,
            mutated if owner == "observer" else observer,
            mutated if owner == "validator" else validator,
            mutated if owner == "archive" else archive,
        )
        candidate_violations.extend(
            trusted_candidate_gate_violations(
                mutated if owner == "ci" else ci,
                classifier,
                mutated if owner == "validator" else validator,
            )
        )
        if not any(expected in violation for violation in candidate_violations):
            raise AssertionError(
                f"candidate mutation did not exercise {expected!r}: {candidate_violations!r}"
            )
    # The static contract above reads markers; this runs the guard. Both are
    # needed: the markers survive a mutation of the arithmetic that decides
    # whether a release ships signed.
    guard_violations = signpath_guard_behaviour_violations(release)
    if guard_violations:
        raise AssertionError(
            "SignPath guard behaviour contract failed:\n  "
            + "\n  ".join(guard_violations)
        )
    # And the truth table has to be able to fail. A mutation the static contract
    # cannot see must change at least one row's exit status, or the table is
    # decoration.
    for old, new, why in (
        # Anchored on the line above it as well: finalize-release's link step
        # counts with an identically indented `present=$((present + 1))`, and a
        # bare fixture would depend on which job happens to come first in the
        # file for the mutation to land in this one.
        (
            '            if [[ -n "${!var:-}" ]]; then\n'
            "              present=$((present + 1))",
            '            if [[ -n "${!var:-}" ]]; then\n'
            "              present=$((present + 0))",
            "the count that decides whether the guard fires",
        ),
        (
            '          if [[ "$present" -eq 0 && -n "${SECRET_SIGNPATH_ARTIFACT_CONFIGURATION_SLUG:-}" ]]; then',
            '          if [[ "$present" -eq -1 ]]; then',
            "the optional-slug-alone check",
        ),
        # THE LATCH. Zero of four is a green release today and must be a red one
        # the day signing is switched on; this condition is the entire
        # difference between those two, and nothing else in the job can see it.
        (
            '          if [[ "${SIGNPATH_REQUIRED:-}" == "true" && "$present" -ne "${#names[@]}" ]]; then',
            "          if false; then",
            "the required-but-unconfigured latch",
        ),
        # And the diagnosis it prints. An operator meeting this failure has
        # never seen it before -- it can only fire once, on the first release
        # after the switch is thrown -- so a latch that fires without naming the
        # missing secrets sends them to read the workflow instead.
        (
            '                echo "       MISSING: $name" >&2',
            '                echo "       (something is missing)" >&2',
            "the per-secret diagnosis the latch prints",
        ),
    ):
        if old not in release:
            raise AssertionError(f"SignPath guard mutation fixture is stale: {old!r}")
        if not signpath_guard_behaviour_violations(release.replace(old, new, 1)):
            raise AssertionError(
                f"the SignPath guard truth table did not notice a mutation of {why}"
            )
    # The other step in this file whose whole job is a decision no reading of
    # the YAML can check: the desktop links in the published release notes.
    link_violations = desktop_link_behaviour_violations(release)
    if link_violations:
        raise AssertionError(
            "desktop release-note link behaviour contract failed:\n  "
            + "\n  ".join(link_violations)
        )
    # Same rule as above: a truth table that cannot fail is decoration. Each
    # mutation below is a way the step has actually been wrong, or a way it
    # would silently go wrong; every one must turn at least one row red.
    desktop_link_unchecked: list[str] = []
    for old, new, why, needs_strict_awk in (
        (
            'for asset in "$win" "$dmg"; do',
            'for asset in "$win"; do',
            "the check that BOTH installers are on the release before linking",
            False,
        ),
        # The `--` was missing once. Every marker starts with "- ", so grep read
        # it as a bundle of short options, matched nothing, and the step
        # re-inserted the section on a release that already had it.
        (
            'grep -Fxq -- "$marker" "$RUNNER_TEMP/install.md"',
            'grep -Fxq "$marker" "$RUNNER_TEMP/install.md"',
            "the -- that stops a marker being read as grep options",
            False,
        ),
        # And matching the whole body was the first version. Both links quoted
        # in a maintainers' note made it report success having linked nothing.
        (
            'grep -Fxq -- "$marker" "$RUNNER_TEMP/install.md"',
            'grep -Fxq -- "$marker" "$RUNNER_TEMP/body.md"',
            "scoping the already-linked check to the Install section",
            False,
        ),
        (
            '          if [[ "$present" -eq 2 ]]; then',
            '          if [[ "$present" -ge 1 ]]; then',
            "the refusal to call a half-written section finished",
            False,
        ),
        (
            "          sed -i 's/\\r$//' \"$RUNNER_TEMP/body.md\"\n",
            "",
            "the CR strip that lets a CRLF body find its own Install heading",
            True,
        ),
    ):
        if release.count(old) != 1:
            raise AssertionError(
                f"desktop link mutation fixture is stale or ambiguous: {old!r}"
            )
        if needs_strict_awk and not awk_compares_crlf_lines_intact():
            # MSYS awk strips the CR before comparing, so removing the `sed`
            # changes nothing here. Saying so beats a row that quietly passes:
            # on Linux, where the release actually runs, this mutation bites.
            desktop_link_unchecked.append(
                f"{why}: this machine's awk strips CR from a CRLF line before "
                "comparing it, so the mutation is invisible here; it is checked "
                "on the Linux runners this workflow ships from"
            )
            continue
        if not desktop_link_behaviour_violations(release.replace(old, new, 1)):
            raise AssertionError(
                f"the desktop link truth table did not notice a mutation of {why}"
            )
    for line in desktop_link_unchecked:
        print(f"UNCHECKED: desktop release-note links: {line}")
    # The activation monitor, run against a fake SignPath. Its whole subject is
    # states nobody can produce here -- an accepted application, a resolving
    # project slug -- so a stub server is the only way any of it is measured at
    # all before the day it matters.
    status_baseline = signpath_status_behaviour_violations()
    if status_baseline.violations:
        raise AssertionError(
            "SignPath status behaviour contract failed:\n  "
            + "\n  ".join(status_baseline.violations)
        )
    status_text = SIGNPATH_STATUS_PATH.read_text(encoding="utf-8")
    status_rows = {row[0] for row in SIGNPATH_STATUS_TRUTH_TABLE}
    for old, new, catchers, why in SIGNPATH_STATUS_MUTATIONS:
        if status_text.count(old) != 1:
            raise AssertionError(
                f"signpath-status mutation fixture is stale "
                f"({status_text.count(old)} matches): {old!r}"
            )
        # The catcher set is checked before it is used, so a mutation cannot
        # quietly stop being measured. An empty set, or one naming a row that no
        # longer exists, is a stale table -- and a stale table that merely
        # skipped would leave the mutation unmeasured while the suite still said
        # PASS.
        if not catchers:
            raise AssertionError(f"the catcher set for {why} is empty")
        unknown = catchers - status_rows
        if unknown:
            raise AssertionError(
                f"the catcher set for {why} names {sorted(unknown)}, which is "
                "not a row of the SignPath status truth table"
            )
        live = catchers & status_baseline.ran
        if not live:
            raise AssertionError(
                f"no row that can catch a mutation of {why} ran here: catching "
                f"it needs one of {sorted(catchers)} and this host ran "
                f"{sorted(status_baseline.ran) or 'nothing'}"
            )
        mutant = signpath_status_behaviour_violations(
            status_text.replace(old, new, 1), only=live
        )
        # Two demands, not one. "A violation appeared" is satisfied by a
        # mutation that merely breaks a row; what is being claimed is that the
        # rows named as catchers SAW this revert, so every one of them must have
        # run and every one of them must have failed.
        if not live <= mutant.ran:
            raise AssertionError(
                f"mutating {why} stopped {sorted(live - mutant.ran)} from "
                "running; a row that never ran cannot be the row that caught it"
            )
        if not live <= mutant.failed:
            raise AssertionError(
                f"the SignPath status truth table did not notice a mutation of "
                f"{why} in {sorted(live - mutant.failed)}, which the catcher set "
                f"claims can see it (rows that did fail: "
                f"{sorted(mutant.failed) or 'none'})"
            )

    # The re-signing step has the same shape of hole and none of the guard's
    # coverage: its assertions are all reachable-only, so anything that returns
    # early satisfies every one of them by never reaching them.
    resign_violations = resign_behaviour_violations(release)
    if resign_violations:
        raise AssertionError(
            "updater re-signing behaviour contract failed:\n  "
            + "\n  ".join(resign_violations)
        )
    for old, new, why in (
        (
            '          before_sig="$(cat "$sig")"\n',
            '          before_sig="$(cat "$sig")"\n          exit 0\n',
            "an early return that reaches none of the step's own assertions",
        ),
        (
            '          [[ "$(cat "$sig")" != "$before_sig" ]] || '
            '{ echo "ERROR: $sig is unchanged; it still covers the unsigned bytes." >&2; exit 1; }\n',
            "",
            "the check that the signature actually changed",
        ),
        (
            '          [[ -s "$sig" ]] || { echo "ERROR: re-signing produced no $sig." >&2; exit 1; }',
            '          [[ -e "$sig" ]] || { echo "ERROR: re-signing produced no $sig." >&2; exit 1; }',
            "the emptiness check on the regenerated signature",
        ),
        (
            # Everything the step asserts is true at the moment it asserts it,
            # and false by the time the next step uploads the file. A table that
            # only asks "non-empty and changed" cannot see this; the sidecar has
            # to be compared against what the signer actually produced.
            '          echo "Regenerated $sig over the Authenticode-signed installer."',
            '          printf x >> "$sig"\n'
            '          echo "Regenerated $sig over the Authenticode-signed installer."',
            "corruption of the sidecar after the step's own assertions passed",
        ),
    ):
        if release.count(old) != 1:
            raise AssertionError(
                f"re-sign mutation fixture is stale ({release.count(old)} matches): {old!r}"
            )
        if not resign_behaviour_violations(release.replace(old, new, 1)):
            raise AssertionError(
                f"the re-sign truth table did not notice a mutation of {why}"
            )
    # The Authenticode gate is the last thing standing between a silently
    # skipped SignPath download and a published unsigned installer, and until
    # now it was held by substring markers alone.
    #
    # Two separate gates, because they fail in different places. The metadata
    # gate reads the workflow: `continue-on-error: true` on a signing step is
    # invisible to every body-level assertion above, and turns the gate into a
    # log line. The behaviour gate runs the step.
    meta_violations = authenticode_step_metadata_violations(release)
    if meta_violations:
        raise AssertionError(
            "signing step metadata contract failed:\n  " + "\n  ".join(meta_violations)
        )
    for name in SIGNING_STEPS:
        marker = f"- name: {name}\n"
        if release.count(marker) != 1:
            raise AssertionError(
                f"metadata mutation fixture is stale for {name!r} "
                f"({release.count(marker)} matches)"
            )
        mutated = release.replace(marker, marker + "        continue-on-error: true\n", 1)
        if not authenticode_step_metadata_violations(mutated):
            raise AssertionError(
                f"the metadata contract did not notice continue-on-error on {name!r}"
            )
    # named_step_body strips, so the body it returns has lost the indentation of
    # its first line; splice against the ORIGINAL slice of release.yml instead,
    # or a whole-line removal silently becomes a partial one.
    auth_header = f"      - name: {AUTHENTICODE_STEP}\n"
    if release.count(auth_header) != 1:
        raise AssertionError(
            f"metadata mutation fixture is stale ({release.count(auth_header)} "
            f"matches): {auth_header!r}"
        )
    auth_start = release.index(auth_header) + len(auth_header)
    auth_body = named_step_body(job_body(release, "app-bundle-windows"), AUTHENTICODE_STEP)
    auth_end = release.index(auth_body, auth_start) + len(auth_body)
    auth_region = release[auth_start:auth_end]
    for old, why in (
        ("        shell: pwsh\n", "the `shell: pwsh` declaration"),
        (
            "        if: env.SIGNPATH_CONFIGURED == 'true'\n",
            "the SIGNPATH_CONFIGURED guard",
        ),
    ):
        if auth_region.count(old) != 1:
            raise AssertionError(
                f"metadata mutation fixture is stale "
                f"({auth_region.count(old)} matches): {old!r}"
            )
        mutated = (
            release[:auth_start] + auth_region.replace(old, "", 1) + release[auth_end:]
        )
        if not authenticode_step_metadata_violations(mutated):
            raise AssertionError(f"the metadata contract did not notice removing {why}")
    # And the insertions. An environment-keyed early-out is the one mutation the
    # truth table structurally cannot see, because the table supplies the
    # environment: `exit 0` under a branch on SIGNPATH_CONFIGURED is skipped
    # here and taken in the release, so all five rows stay green while the gate
    # does nothing. It is caught statically, on every host, fixture or not.
    for anchor, inserted, why in (
        (
            '          $ErrorActionPreference = "Stop"\n',
            "          if ($env:SIGNPATH_CONFIGURED -eq 'true') { exit 0 }\n",
            "an early return taken only when the release's own guard variable is set",
        ),
        (
            '          $ErrorActionPreference = "Stop"\n',
            "          if ($env:CI -eq 'true') { exit 0 }\n",
            "an early return taken only on a CI runner",
        ),
    ):
        if auth_region.count(anchor) != 1:
            raise AssertionError(
                f"metadata insertion fixture is stale "
                f"({auth_region.count(anchor)} matches): {anchor!r}"
            )
        mutated = (
            release[:auth_start]
            + auth_region.replace(anchor, anchor + inserted, 1)
            + release[auth_end:]
        )
        if not authenticode_step_metadata_violations(mutated):
            raise AssertionError(f"the metadata contract did not notice {why}")

    baseline = authenticode_behaviour_violations(release)
    auth_unchecked = list(baseline.unchecked)
    if baseline.violations:
        raise AssertionError(
            "Authenticode verification behaviour contract failed:\n  "
            + "\n  ".join(baseline.violations)
        )
    # Per-mutation, not all-or-nothing: whatever rows this host CAN run must
    # still catch every mutation they are capable of catching. On a fixture-free
    # host that is exactly one mutation -- and it is the one that guts the whole
    # step -- so the Ubuntu lane stops being a free pass.
    for old, new, catchers, why in AUTHENTICODE_MUTATIONS:
        if release.count(old) != 1:
            raise AssertionError(
                "Authenticode mutation fixture is stale "
                f"({release.count(old)} matches): {old!r}"
            )
        live = catchers & baseline.ran
        if not live:
            auth_unchecked.append(
                f"mutation of {why} was not exercised; catching it needs one of "
                f"{sorted(catchers)} and this host ran {sorted(baseline.ran) or 'nothing'}"
            )
            continue
        mutant = authenticode_behaviour_violations(
            release.replace(old, new, 1), only=live
        )
        # Two separate demands, because "a violation appeared" is not the same
        # claim as "the mutation was caught". Removing the publisher check, for
        # instance, also makes the `accepted` row unbuildable -- that violation
        # is bookkeeping, not detection. So: every row that is supposed to catch
        # this must still have RUN, and at least one of them must have FAILED.
        if not live <= mutant.ran:
            raise AssertionError(
                f"mutating {why} stopped {sorted(live - mutant.ran)} from running; "
                "a row that never ran cannot be the row that caught it"
            )
        # EVERY listed catcher must fail, not merely one of them. `catchers` is
        # a claim about which rows can see this mutation, and it is the claim
        # the UNCHECKED accounting rests on: an over-generous set makes a
        # fixture-free host report "enforced" for a mutation nothing there could
        # actually catch. Requiring the set to be exact is what keeps the claim
        # honest; a set that is too small is safe and merely reports UNCHECKED.
        if not live <= mutant.failed:
            raise AssertionError(
                f"the Authenticode truth table did not notice a mutation of {why} "
                f"in {sorted(live - mutant.failed)}, which the catcher set claims "
                f"can see it (rows that did fail: {sorted(mutant.failed) or 'none'})"
            )
    if auth_unchecked:
        # Never silent. A gate that could not run is unchecked, not passed, and
        # the line says so in the log the reviewer reads.
        for line in auth_unchecked:
            print(f"UNCHECKED: Authenticode step: {line}")
        # `os.name == "nt"` is an OS proxy, not a capability predicate. The
        # question is not which OS this is; it is whether the host COULD have
        # measured what it did not measure, which the probe answers directly:
        # "no-support" means Get-AuthenticodeSignature does not work here at
        # all, which is a real incapacity on Linux and macOS;
        # "none-found" means it works and a search of the system binary
        # directories still produced nothing signed, which is a broken probe or
        # a broken trust store on ANY operating system, and never a reason to
        # pass over the rows it prevented.
        if baseline.capability in ("none-found", "probe-failed"):
            why = (
                "Get-AuthenticodeSignature runs on this host, and a search of its "
                "system binary directories still found no validly signed file"
                if baseline.capability == "none-found"
                else "the capability probe did not answer at all -- its exit status "
                "and what it printed do not agree on any arm, so nothing here knows "
                "whether this host can measure Authenticode or not"
            )
            raise AssertionError(
                f"the Authenticode probe is broken. {why} -- so the rows below did "
                "not go unmeasured for want of a capability, they failed to be "
                "measured:\n  " + "\n  ".join(auth_unchecked)
            )
        if os.environ.get("WENLAN_REQUIRE_AUTHENTICODE") == "1":
            raise AssertionError(
                "the Authenticode contract is only partly measured on a lane that "
                "declared it would measure all of it (WENLAN_REQUIRE_AUTHENTICODE=1):"
                "\n  " + "\n  ".join(auth_unchecked)
            )

    # And the property no reading of the body can establish: run the step, and
    # require the same answer under the release's environment, under none of it,
    # and under all of it wrong. See `authenticode_environment_invariance`.
    # Last, so that a lane which is only partly measured says which rows of the
    # truth table it could not build BEFORE it says the same about this.
    invariance = authenticode_environment_invariance(release)
    if invariance.violations:
        raise AssertionError(
            "the Authenticode step's outcome depends on the environment:\n  "
            + "\n  ".join(invariance.violations)
        )
    for line in invariance.unchecked:
        print(f"UNCHECKED: Authenticode environment invariance: {line}")
    if invariance.unchecked and os.environ.get("WENLAN_REQUIRE_AUTHENTICODE") == "1":
        raise AssertionError(
            "environment invariance is only partly measured on a lane that "
            "declared it would measure all of it (WENLAN_REQUIRE_AUTHENTICODE=1):"
            "\n  " + "\n  ".join(invariance.unchecked)
        )
    if not invariance.ran and not invariance.unchecked:
        raise AssertionError(
            "environment invariance measured nothing at all and said nothing "
            "about why; the `missing` row needs no fixture and must always run"
        )

    print("PASS: release promotion, Homebrew, and Node 24 action contracts")


if __name__ == "__main__":
    main()
