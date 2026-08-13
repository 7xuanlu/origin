#!/usr/bin/env python3
"""Fail-loud static contracts for release promotion and public install paths."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


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
RUNTIME_IMAGE_PATH = REPO_ROOT / "scripts" / "verify-release-runtime-image.py"
PUBLISH_CRATE_TEST_PATH = REPO_ROOT / "scripts" / "publish-crate.test.py"

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


def named_step_body(job: str, step_name: str) -> str:
    match = re.search(
        rf"^      - name: {re.escape(step_name)}\n(?P<body>.*?)(?=^      - (?:name:|uses:)|^      #|^  #|\Z)",
        job,
        re.MULTILINE | re.DOTALL,
    )
    return match.group("body").strip() if match else ""


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
    if release.count("secrets.RELEASE_TOKEN") != 1:
        violations.append("release recovery token is not confined to the exact tag bind")
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
        "app-bundle": 90,
        "app-bundle-windows": 120,
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
        "name: docker-runtime-inputs",
    ]:
        if marker not in promote:
            violations.append(f"validated asset promotion omits {marker!r}")

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
        "timeout-minutes: 20",
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
        "--wait-seconds 1080",
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
    violations.extend(candidate_observer_contract_violations(ci, observer, validator, archive))
    violations.extend(trusted_candidate_gate_violations(ci, classifier, validator))
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
    print("PASS: release promotion, Homebrew, and Node 24 action contracts")


if __name__ == "__main__":
    main()
