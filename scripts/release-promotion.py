#!/usr/bin/env python3
"""Resolve and consume one validated release candidate without rebuilding it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from release_archive import MAX_ARCHIVE_BYTES, safe_extract_archive, safe_extract_zip
from release_targets import release_assets


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load("validate_release_candidate", "validate-release-candidate.py")
MERGE_PROOF = _load("verify_release_merge", "verify-release-merge.py")

API_VERSION = "2026-03-10"
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
OBSERVER_WORKFLOW_PATH = ".github/workflows/release-candidate-observer.yml"
CI_WORKFLOW_FILE = "ci.yml"
OBSERVER_WORKFLOW_FILE = "release-candidate-observer.yml"
RELEASE_BRANCH = "release-please--branches--main"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
MAX_RECEIPT_WRAPPER_BYTES = 2 * 1024 * 1024
MAX_MAIN_PLAN_WRAPPER_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_PAGES = 10
MAX_RECEIPT_CANDIDATES = 20
MAIN_CI_PENDING_STATUSES = frozenset(
    {"pending", "queued", "in_progress", "requested", "waiting"}
)


class PromotionError(RuntimeError):
    """Terminal release evidence failure."""


class PendingEvidence(RuntimeError):
    """The trusted observer has not published its closed receipt yet."""


class PromotionApi(Protocol):
    def get_json(
        self, path: str, *, params: dict[str, object] | None = None
    ) -> object: ...

    def download(self, path: str, destination: Path, maximum_bytes: int) -> tuple[int, str]: ...


@dataclass
class GateResult:
    state: str
    reason: str
    receipt: dict | None = None
    receipt_artifact: dict | None = None
    main_sha: str | None = None
    main_tree_sha: str | None = None
    main_run_id: int | None = None
    main_run_attempt: int | None = None


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise PromotionError(f"{label} is not an object")
    return value


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise PromotionError(f"{label} is not an array")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PromotionError(f"{label} is not a positive integer")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise PromotionError(f"{label} is not a canonical SHA")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise PromotionError(f"{label} is not a canonical SHA-256 digest")
    return value


def _repo_identity(value: object, repository: str, repository_id: int, label: str) -> None:
    repo = _mapping(value, label)
    if repo.get("id") != repository_id or repo.get("full_name") != repository:
        raise PromotionError(f"{label} differs from the trusted repository")


def _tree_sha(api: PromotionApi, repository: str, commit_sha: str) -> str:
    commit = _mapping(
        api.get_json(f"/repos/{repository}/git/commits/{commit_sha}"),
        f"Git commit {commit_sha}",
    )
    tree = _mapping(commit.get("tree"), f"Git commit {commit_sha} tree")
    return _sha(tree.get("sha"), f"Git commit {commit_sha} tree SHA")


def _release_context(
    api: PromotionApi, repository: str, main_sha: str
) -> tuple[object | None, str | None]:
    associated = _array(
        api.get_json(
            f"/repos/{repository}/commits/{main_sha}/pulls",
            params={"per_page": 100},
        ),
        "main commit pull requests",
    )
    release_like: list[dict] = []
    malformed_release_like = False
    for raw in associated:
        pr = _mapping(raw, "associated pull request")
        head = _mapping(pr.get("head"), "associated pull request head")
        base = _mapping(pr.get("base"), "associated pull request base")
        if head.get("ref") == RELEASE_BRANCH:
            if base.get("ref") != "main":
                malformed_release_like = True
            else:
                release_like.append(pr)
    if not release_like:
        if malformed_release_like:
            raise PromotionError("release-please association has the wrong base branch")
        return None, None
    if len(release_like) != 1:
        raise PromotionError(
            f"expected one associated release-please PR, found {len(release_like)}"
        )

    proof = MERGE_PROOF.verify_release_merge(
        api,
        repository,
        event_name="push",
        ref="refs/heads/main",
        sha=main_sha,
    )
    if not proof.verified:
        raise PromotionError(f"release merge proof failed: {proof.reason}")
    return proof, proof.pr_head_sha


def _conclusion_check_suite(
    api: PromotionApi, repository: str, head_sha: str
) -> int:
    payload = _mapping(
        api.get_json(
            f"/repos/{repository}/commits/{head_sha}/check-runs",
            params={"filter": "latest", "per_page": 100},
        ),
        "release head check runs",
    )
    matches: list[int] = []
    for raw in _array(payload.get("check_runs"), "release head check runs"):
        check = _mapping(raw, "release head check run")
        app = _mapping(check.get("app"), "release head check app")
        if (
            check.get("name") == "conclusion"
            and check.get("status") == "completed"
            and check.get("conclusion") == "success"
            and check.get("head_sha") == head_sha
            and app.get("slug") == "github-actions"
        ):
            suite = _mapping(check.get("check_suite"), "conclusion check suite")
            matches.append(_positive_int(suite.get("id"), "conclusion check suite id"))
    if len(matches) != 1:
        raise PromotionError(
            f"expected one successful latest conclusion check, found {len(matches)}"
        )
    return matches[0]


def _source_run(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    head_sha: str,
    check_suite_id: int,
) -> dict:
    payload = _mapping(
        api.get_json(
            f"/repos/{repository}/actions/workflows/{CI_WORKFLOW_FILE}/runs",
            params={
                "event": "pull_request",
                "head_sha": head_sha,
                "status": "success",
                "per_page": 100,
            },
        ),
        "candidate CI workflow runs",
    )
    matches: list[dict] = []
    for raw in _array(payload.get("workflow_runs"), "candidate CI workflow runs"):
        run = _mapping(raw, "candidate CI workflow run")
        if run.get("check_suite_id") != check_suite_id:
            continue
        run_repo = _mapping(run.get("repository"), "candidate CI run repository")
        head_repo = _mapping(
            run.get("head_repository"), "candidate CI run head repository"
        )
        if (
            run.get("path") != CI_WORKFLOW_PATH
            or run.get("event") != "pull_request"
            or run.get("status") != "completed"
            or run.get("conclusion") != "success"
            or run.get("head_branch") != RELEASE_BRANCH
            or run.get("head_sha") != head_sha
            or run_repo.get("id") != repository_id
            or run_repo.get("full_name") != repository
            or head_repo.get("id") != repository_id
            or head_repo.get("full_name") != repository
            or head_repo.get("fork") is not False
        ):
            raise PromotionError("matched candidate CI run has invalid identity")
        _positive_int(run.get("id"), "candidate CI run id")
        _positive_int(run.get("run_attempt"), "candidate CI run attempt")
        _positive_int(run.get("workflow_id"), "candidate CI workflow id")
        matches.append(run)
    if len(matches) != 1:
        raise PromotionError(
            f"conclusion check suite maps to {len(matches)} candidate CI runs"
        )
    return matches[0]


def _artifact_pages(api: PromotionApi, repository: str, name: str) -> list[dict]:
    artifacts: list[dict] = []
    expected_total: int | None = None
    for page in range(1, MAX_ARTIFACT_PAGES + 1):
        payload = _mapping(
            api.get_json(
                f"/repos/{repository}/actions/artifacts",
                params={"name": name, "per_page": 100, "page": page},
            ),
            f"repository artifacts page {page}",
        )
        values = _array(payload.get("artifacts"), f"repository artifacts page {page}")
        total = payload.get("total_count")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise PromotionError("repository artifact total_count is invalid")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise PromotionError("repository artifact total_count changed during pagination")
        if total > MAX_RECEIPT_CANDIDATES:
            raise PromotionError("receipt candidate count exceeds the fail-closed bound")
        artifacts.extend(_mapping(value, "repository artifact") for value in values)
        if len(values) < 100:
            if len(artifacts) != expected_total:
                raise PromotionError("repository artifact total_count does not match pages")
            return artifacts
    raise PromotionError("repository artifact search exceeded the bounded page limit")


def _latest_observer_run(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    source_run_id: int,
    source_run_attempt: int,
) -> dict:
    """Return only the newest terminal observer attempt for one source run."""
    display_title = (
        f"release-candidate-observer source {source_run_id}/{source_run_attempt}"
    )
    payload = _mapping(
        api.get_json(
            f"/repos/{repository}/actions/workflows/{OBSERVER_WORKFLOW_FILE}/runs",
            params={"event": "workflow_run", "per_page": 100},
        ),
        "observer workflow runs",
    )
    matches = [
        _mapping(raw, "observer workflow run")
        for raw in _array(payload.get("workflow_runs"), "observer workflow runs")
        if isinstance(raw, dict) and raw.get("display_title") == display_title
    ]
    if not matches:
        raise PendingEvidence("trusted observer run is not scheduled yet")
    run_ids = {run.get("id") for run in matches}
    if len(run_ids) != 1:
        raise PromotionError("source CI run maps to multiple observer workflow runs")
    latest = max(matches, key=lambda run: int(run.get("run_attempt", 0)))
    _repo_identity(
        latest.get("repository"), repository, repository_id, "observer repository"
    )
    if (
        latest.get("path") != OBSERVER_WORKFLOW_PATH
        or latest.get("event") != "workflow_run"
    ):
        raise PromotionError("latest observer run has invalid workflow identity")
    if latest.get("status") != "completed":
        raise PendingEvidence("latest trusted observer attempt is still pending")
    if latest.get("conclusion") != "success":
        raise PromotionError(
            f"latest trusted observer attempt concluded {latest.get('conclusion')!r}"
        )
    _positive_int(latest.get("id"), "latest observer run id")
    _positive_int(latest.get("run_attempt"), "latest observer run attempt")
    return latest


def _download_receipt(
    api: PromotionApi,
    repository: str,
    artifact: dict,
    temp_root: Path,
) -> tuple[dict, str]:
    artifact_id = _positive_int(artifact.get("id"), "receipt artifact id")
    size = _positive_int(artifact.get("size_in_bytes"), "receipt artifact size")
    if size > MAX_RECEIPT_WRAPPER_BYTES:
        raise PromotionError("receipt artifact exceeds the small-receipt bound")
    expected_digest = _digest(artifact.get("digest"), "receipt artifact digest")
    work = Path(tempfile.mkdtemp(prefix="promotion-receipt-", dir=temp_root))
    wrapper = work / f"receipt-{artifact_id}.zip"
    downloaded_size, downloaded_digest = api.download(
        f"/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        wrapper,
        size,
    )
    if (
        downloaded_size != size
        or f"sha256:{downloaded_digest}" != expected_digest
    ):
        raise PromotionError("receipt wrapper size or digest differs from REST metadata")
    extracted = work / "receipt"
    safe_extract_zip(wrapper, extracted, ["validated-release-receipt.json"])
    receipt_path = extracted / "validated-release-receipt.json"
    try:
        receipt = VALIDATOR.read_closed_receipt(receipt_path)
    except Exception as error:
        raise PromotionError(f"closed receipt validation failed: {error}") from error
    return receipt, expected_digest


def _artifact_metadata(
    api: PromotionApi,
    repository: str,
    artifact_id: int,
) -> dict:
    return _mapping(
        api.get_json(f"/repos/{repository}/actions/artifacts/{artifact_id}"),
        f"artifact {artifact_id}",
    )


def _validate_observer_and_assets(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    receipt_artifact: dict,
    receipt: dict,
) -> dict:
    artifact_run = _mapping(
        receipt_artifact.get("workflow_run"), "receipt artifact workflow run"
    )
    observer = _mapping(receipt.get("observer"), "receipt observer")
    observer_run_id = _positive_int(observer.get("run_id"), "observer run id")
    observer_attempt = _positive_int(
        observer.get("run_attempt"), "observer run attempt"
    )
    if artifact_run.get("id") != observer_run_id:
        raise PromotionError("receipt artifact is not owned by its observer run")
    run = _mapping(
        api.get_json(
            f"/repos/{repository}/actions/runs/{observer_run_id}/attempts/{observer_attempt}"
        ),
        "observer workflow run attempt",
    )
    workflow_id = _positive_int(run.get("workflow_id"), "observer workflow id")
    workflow = _mapping(
        api.get_json(f"/repos/{repository}/actions/workflows/{workflow_id}"),
        "observer workflow metadata",
    )
    _repo_identity(run.get("repository"), repository, repository_id, "observer repository")
    if (
        run.get("id") != observer_run_id
        or run.get("run_attempt") != observer_attempt
        or run.get("workflow_id") != observer.get("workflow_id")
        or run.get("path") != OBSERVER_WORKFLOW_PATH
        or run.get("event") != "workflow_run"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("head_sha") != observer.get("code_sha")
        or workflow.get("id") != workflow_id
        or workflow.get("path") != OBSERVER_WORKFLOW_PATH
        or workflow.get("state") != "active"
    ):
        raise PromotionError("observer workflow control-plane identity mismatch")

    validated_claim = _mapping(
        receipt.get("validated_assets_artifact"), "validated assets claim"
    )
    validated_id = _positive_int(validated_claim.get("id"), "validated assets id")
    validated = _artifact_metadata(api, repository, validated_id)
    validated_run = _mapping(
        validated.get("workflow_run"), "validated assets workflow run"
    )
    if (
        validated.get("id") != validated_id
        or validated.get("name") != validated_claim.get("name")
        or validated.get("digest") != validated_claim.get("digest")
        or validated.get("expired") is not False
        or not isinstance(validated.get("size_in_bytes"), int)
        or validated.get("size_in_bytes", 0) < 1
        or validated.get("size_in_bytes", 0) > MAX_ARCHIVE_BYTES
        or validated_run.get("id") != observer_run_id
        or validated_run.get("repository_id") != repository_id
        or validated_run.get("head_repository_id") != repository_id
    ):
        raise PromotionError("validated assets REST metadata differs from the closed receipt")
    _digest(validated.get("digest"), "validated assets digest")
    return validated


def _receipt_semantics(receipt: dict) -> str:
    semantic = {
        key: receipt[key]
        for key in (
            "assets",
            "check_suite_id",
            "head_sha",
            "paths",
            "pr_number",
            "repository",
            "repository_id",
            "run_attempt",
            "run_id",
            "tree_sha",
            "version",
            "workflow_id",
        )
    }
    return json.dumps(semantic, separators=(",", ":"), sort_keys=True)


def _find_receipt(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    source_run: dict,
    pr_number: int,
    head_sha: str,
    tree_sha: str,
    temp_root: Path,
) -> tuple[dict, dict]:
    run_id = source_run["id"]
    run_attempt = source_run["run_attempt"]
    latest_observer = _latest_observer_run(
        api, repository, repository_id, run_id, run_attempt
    )
    expected_name = f"validated-release-receipt-{run_id}-{run_attempt}"
    candidates = [
        artifact
        for artifact in _artifact_pages(api, repository, expected_name)
        if artifact.get("name") == expected_name
    ]
    if not candidates:
        raise PendingEvidence("trusted observer receipt is not available yet")
    valid: list[tuple[dict, dict]] = []
    latest_valid: list[tuple[dict, dict]] = []
    for artifact in sorted(candidates, key=lambda item: int(item.get("id", 0)), reverse=True):
        if artifact.get("expired") is not False:
            raise PromotionError("observer receipt artifact is expired")
        artifact_run = _mapping(
            artifact.get("workflow_run"), "receipt artifact workflow run"
        )
        if artifact_run.get("id") != latest_observer.get("id"):
            continue
        receipt, _wrapper_digest = _download_receipt(
            api, repository, artifact, temp_root
        )
        if (
            receipt.get("repository") != repository
            or receipt.get("repository_id") != repository_id
            or receipt.get("run_id") != run_id
            or receipt.get("run_attempt") != run_attempt
            or receipt.get("workflow_id") != source_run.get("workflow_id")
            or receipt.get("check_suite_id") != source_run.get("check_suite_id")
            or receipt.get("pr_number") != pr_number
            or receipt.get("head_sha") != head_sha
            or receipt.get("tree_sha") != tree_sha
            or receipt.get("observer", {}).get("run_id") != latest_observer.get("id")
        ):
            raise PromotionError("closed receipt differs from the merged release context")
        _validate_observer_and_assets(
            api, repository, repository_id, artifact, receipt
        )
        valid.append((receipt, artifact))
        if (
            receipt.get("observer", {}).get("run_attempt")
            == latest_observer.get("run_attempt")
        ):
            latest_valid.append((receipt, artifact))
    if not latest_valid:
        raise PromotionError("latest successful observer attempt has no closed receipt")
    semantics = {_receipt_semantics(receipt) for receipt, _artifact in valid}
    if len(semantics) != 1:
        raise PromotionError("observer reruns produced conflicting release semantics")
    return latest_valid[0]


def verify_main(
    api: PromotionApi,
    repository: str,
    main_sha: str,
    temp_root: Path,
    *,
    wait_seconds: int = 0,
    sleep=time.sleep,
) -> GateResult:
    _sha(main_sha, "main SHA")
    if re.fullmatch(r"[^/\s]+/[^/\s]+", repository) is None:
        return GateResult("blocked", "repository must be OWNER/REPO", main_sha=main_sha)
    try:
        repository_info = _mapping(
            api.get_json(f"/repos/{repository}"), "repository metadata"
        )
        repository_id = _positive_int(repository_info.get("id"), "repository id")
        if repository_info.get("full_name") != repository:
            raise PromotionError("repository metadata differs from workflow scope")
        proof, head_sha = _release_context(api, repository, main_sha)
        if proof is None or head_sha is None:
            return GateResult(
                "ordinary",
                "main commit is not a release-please merge",
                main_sha=main_sha,
            )
        tree_sha = _tree_sha(api, repository, main_sha)
        check_suite_id = _conclusion_check_suite(api, repository, head_sha)
        source_run = _source_run(
            api, repository, repository_id, head_sha, check_suite_id
        )
        deadline = time.monotonic() + max(wait_seconds, 0)
        while True:
            try:
                receipt, receipt_artifact = _find_receipt(
                    api,
                    repository,
                    repository_id,
                    source_run,
                    proof.pr_number,
                    head_sha,
                    tree_sha,
                    temp_root,
                )
                return GateResult(
                    "validated",
                    "merged release tree and trusted observer receipt verified",
                    receipt=receipt,
                    receipt_artifact=receipt_artifact,
                    main_sha=main_sha,
                    main_tree_sha=tree_sha,
                )
            except PendingEvidence as pending:
                if time.monotonic() >= deadline:
                    return GateResult(
                        "blocked",
                        str(pending),
                        main_sha=main_sha,
                        main_tree_sha=tree_sha,
                    )
                sleep(min(15, max(1, wait_seconds)))
    except (PromotionError, VALIDATOR.CandidateError, MERGE_PROOF.ApiError) as error:
        return GateResult("blocked", str(error), main_sha=main_sha)
    except Exception as error:
        return GateResult(
            "blocked",
            f"release gate error {type(error).__name__}: {error}",
            main_sha=main_sha,
        )


def write_plan(path: Path, result: GateResult) -> None:
    if result.state != "validated" or result.receipt is None or result.receipt_artifact is None:
        raise PromotionError("only a validated release can write a promotion plan")
    document = {
        "schema_version": 1,
        "main_sha": result.main_sha,
        "main_tree_sha": result.main_tree_sha,
        "main_run": (
            {
                "run_attempt": result.main_run_attempt,
                "run_id": result.main_run_id,
                "workflow_path": CI_WORKFLOW_PATH,
            }
            if result.main_run_id is not None and result.main_run_attempt is not None
            else None
        ),
        "receipt": result.receipt,
        "receipt_artifact": {
            key: result.receipt_artifact[key]
            for key in ("digest", "id", "name", "size_in_bytes")
        },
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_plan(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > VALIDATOR.MAX_SOURCE_BYTES:
        raise PromotionError("promotion plan is missing or oversized")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "main_sha",
        "main_tree_sha",
        "main_run",
        "receipt",
        "receipt_artifact",
        "schema_version",
    } or document.get("schema_version") != 1:
        raise PromotionError("promotion plan schema is invalid")
    _sha(document.get("main_sha"), "plan main SHA")
    _sha(document.get("main_tree_sha"), "plan main tree SHA")
    if document.get("main_run") is not None:
        main_run = _mapping(document.get("main_run"), "plan main run")
        if set(main_run) != {"run_attempt", "run_id", "workflow_path"}:
            raise PromotionError("promotion plan main run schema is invalid")
        _positive_int(main_run.get("run_id"), "plan main run id")
        _positive_int(main_run.get("run_attempt"), "plan main run attempt")
        if main_run.get("workflow_path") != CI_WORKFLOW_PATH:
            raise PromotionError("promotion plan main workflow path is invalid")
    return document


def _main_ci_run(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    main_sha: str,
    run_id: int | None,
    run_attempt: int | None,
) -> dict:
    if run_id is None:
        payload = _mapping(
            api.get_json(
                f"/repos/{repository}/actions/workflows/{CI_WORKFLOW_FILE}/runs",
                params={"event": "push", "head_sha": main_sha, "per_page": 100},
            ),
            "main CI workflow runs",
        )
        candidates = [
            _mapping(raw, "main CI workflow run")
            for raw in _array(payload.get("workflow_runs"), "main CI workflow runs")
            if isinstance(raw, dict)
            and raw.get("event") == "push"
            and raw.get("head_sha") == main_sha
            and raw.get("head_branch") == "main"
        ]
        run_ids = {candidate.get("id") for candidate in candidates}
        if len(run_ids) != 1:
            raise PromotionError(
                f"expected one main CI run for the release merge, found {len(run_ids)}"
            )
        run_id = _positive_int(candidates[0].get("id"), "main CI run id")
    else:
        run_id = _positive_int(run_id, "main CI run id")

    run = _mapping(
        api.get_json(f"/repos/{repository}/actions/runs/{run_id}"),
        "main CI workflow run",
    )
    actual_attempt = _positive_int(run.get("run_attempt"), "main CI run attempt")
    if run_attempt is not None and actual_attempt != _positive_int(
        run_attempt, "expected main CI run attempt"
    ):
        raise PromotionError("main CI run attempt differs from the triggering event")
    workflow_id = _positive_int(run.get("workflow_id"), "main CI workflow id")
    workflow = _mapping(
        api.get_json(f"/repos/{repository}/actions/workflows/{workflow_id}"),
        "main CI workflow metadata",
    )
    _repo_identity(run.get("repository"), repository, repository_id, "main CI repository")
    _repo_identity(
        run.get("head_repository"), repository, repository_id, "main CI head repository"
    )
    if (
        run.get("id") != run_id
        or run.get("path") != CI_WORKFLOW_PATH
        or run.get("event") != "push"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("head_branch") != "main"
        or run.get("head_sha") != main_sha
        or workflow.get("id") != workflow_id
        or workflow.get("path") != CI_WORKFLOW_PATH
        or workflow.get("state") != "active"
    ):
        raise PromotionError("main CI control-plane identity mismatch")
    return run


def _verify_main_ci_once(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    main_sha: str,
) -> dict:
    path = f"/repos/{repository}/actions/workflows/{CI_WORKFLOW_FILE}/runs"
    params = {"event": "push", "head_sha": main_sha, "per_page": 100}
    payload = _mapping(
        api.get_json(path, params=params),
        "main CI workflow runs",
    )
    runs = _array(payload.get("workflow_runs"), "main CI workflow runs")
    total_count = payload.get("total_count")
    if (
        not isinstance(total_count, int)
        or isinstance(total_count, bool)
        or total_count < 0
        or total_count != len(runs)
    ):
        raise PromotionError("main CI workflow run inventory is incomplete")
    if not runs:
        raise PendingEvidence("main CI workflow run is not visible yet")
    if len(runs) != 1:
        raise PromotionError(
            f"expected exactly one main CI run at {main_sha}, found {len(runs)}"
        )

    listed = _mapping(runs[0], "main CI workflow run")
    run_id = _positive_int(listed.get("id"), "main CI run id")
    run = _mapping(
        api.get_json(f"/repos/{repository}/actions/runs/{run_id}"),
        "main CI workflow run",
    )
    workflow_id = _positive_int(run.get("workflow_id"), "main CI workflow id")
    workflow = _mapping(
        api.get_json(f"/repos/{repository}/actions/workflows/{workflow_id}"),
        "main CI workflow metadata",
    )
    _repo_identity(run.get("repository"), repository, repository_id, "main CI repository")
    _repo_identity(
        run.get("head_repository"), repository, repository_id, "main CI head repository"
    )
    head_repository = _mapping(run.get("head_repository"), "main CI head repository")
    if (
        run.get("id") != run_id
        or run.get("path") != CI_WORKFLOW_PATH
        or run.get("event") != "push"
        or run.get("head_branch") != "main"
        or run.get("head_sha") != main_sha
        or head_repository.get("fork") is not False
        or workflow.get("id") != workflow_id
        or workflow.get("path") != CI_WORKFLOW_PATH
        or workflow.get("state") != "active"
    ):
        raise PromotionError("main CI control-plane identity mismatch")
    _positive_int(run.get("run_attempt"), "main CI run attempt")

    status = run.get("status")
    conclusion = run.get("conclusion")
    if status in MAIN_CI_PENDING_STATUSES:
        if conclusion is not None:
            raise PromotionError("pending main CI run has a terminal conclusion")
        raise PendingEvidence(f"main CI run {run_id} is {status}")
    if status != "completed":
        raise PromotionError(f"main CI run has unknown status {status!r}")
    if conclusion != "success":
        raise PromotionError(
            f"main CI run {run_id} completed with conclusion {conclusion!r}"
        )
    return run


def verify_main_ci(
    api: PromotionApi,
    repository: str,
    main_sha: str,
    *,
    wait_seconds: int = 0,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> dict:
    if re.fullmatch(r"[^/\s]+/[^/\s]+", repository) is None:
        raise PromotionError("repository must be OWNER/REPO")
    _sha(main_sha, "main SHA")
    repository_info = _mapping(
        api.get_json(f"/repos/{repository}"),
        "repository metadata",
    )
    repository_id = _positive_int(repository_info.get("id"), "repository id")
    if repository_info.get("full_name") != repository:
        raise PromotionError("repository metadata differs from workflow scope")

    deadline = monotonic() + max(wait_seconds, 0)
    while True:
        try:
            return _verify_main_ci_once(
                api,
                repository,
                repository_id,
                main_sha,
            )
        except PendingEvidence as error:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise PromotionError(f"timed out waiting for main CI: {error}") from error
            sleep(min(15, remaining))


def _main_receipt_artifacts(
    api: PromotionApi,
    repository: str,
    run_id: int,
) -> list[dict]:
    prefix = f"main-release-promotion-receipt-{run_id}-"
    candidates: list[dict] = []
    seen_records = 0
    expected_total: int | None = None
    for page in range(1, MAX_ARTIFACT_PAGES + 1):
        payload = _mapping(
            api.get_json(
                f"/repos/{repository}/actions/runs/{run_id}/artifacts",
                params={"per_page": 100, "page": page},
            ),
            f"main CI artifacts page {page}",
        )
        values = _array(payload.get("artifacts"), f"main CI artifacts page {page}")
        total = payload.get("total_count")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise PromotionError("main CI artifact total_count is invalid")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise PromotionError("main CI artifact total_count changed during pagination")
        seen_records += len(values)
        for raw in values:
            artifact = _mapping(raw, "main CI artifact")
            name = artifact.get("name")
            if isinstance(name, str) and name.startswith(prefix):
                if re.fullmatch(re.escape(prefix) + r"[1-9][0-9]*", name) is None:
                    raise PromotionError("main promotion receipt name is malformed")
                candidates.append(artifact)
                if len(candidates) > MAX_RECEIPT_CANDIDATES:
                    raise PromotionError(
                        "main promotion receipt count exceeds the fail-closed bound"
                    )
        if len(values) < 100:
            if seen_records != expected_total:
                raise PromotionError("main CI artifact total_count does not match pages")
            return candidates
    raise PromotionError("main CI artifact search exceeded the bounded page limit")


def _detect_changes_job(
    api: PromotionApi,
    repository: str,
    run_id: int,
    run_attempt: int,
    main_sha: str,
) -> dict:
    matches: list[dict] = []
    seen_records = 0
    expected_total: int | None = None
    for page in range(1, MAX_ARTIFACT_PAGES + 1):
        payload = _mapping(
            api.get_json(
                f"/repos/{repository}/actions/runs/{run_id}/attempts/{run_attempt}/jobs",
                params={"per_page": 100, "page": page},
            ),
            f"main CI attempt {run_attempt} jobs page {page}",
        )
        values = _array(
            payload.get("jobs"), f"main CI attempt {run_attempt} jobs page {page}"
        )
        total = payload.get("total_count")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise PromotionError("main CI job total_count is invalid")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise PromotionError("main CI job total_count changed during pagination")
        seen_records += len(values)
        matches.extend(
            _mapping(raw, "main CI job")
            for raw in values
            if isinstance(raw, dict) and raw.get("name") == "detect-changes"
        )
        if len(values) < 100:
            if seen_records != expected_total:
                raise PromotionError("main CI job total_count does not match pages")
            break
    else:
        raise PromotionError("main CI job search exceeded the bounded page limit")
    if len(matches) != 1:
        raise PromotionError(
            f"expected one detect-changes job in main attempt {run_attempt}, "
            f"found {len(matches)}"
        )
    job = matches[0]
    if (
        job.get("run_id") != run_id
        or job.get("run_attempt") != run_attempt
        or job.get("head_sha") != main_sha
        or job.get("status") != "completed"
        or job.get("conclusion") != "success"
    ):
        raise PromotionError(
            f"detect-changes job in main attempt {run_attempt} is not an exact success"
        )
    _positive_int(job.get("id"), "detect-changes job id")
    return job


def _download_main_plan_artifact(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    artifact: dict,
    temp_root: Path,
) -> dict:
    artifact_run = _mapping(
        artifact.get("workflow_run"), "main receipt workflow run"
    )
    size = _positive_int(artifact.get("size_in_bytes"), "main receipt artifact size")
    if (
        artifact.get("expired") is not False
        or artifact_run.get("repository_id") != repository_id
        or artifact_run.get("head_repository_id") != repository_id
        or size > MAX_MAIN_PLAN_WRAPPER_BYTES
    ):
        raise PromotionError("main promotion receipt metadata is invalid")
    expected_digest = _digest(
        artifact.get("digest"), "main receipt artifact digest"
    )
    artifact_id = _positive_int(artifact.get("id"), "main receipt artifact id")
    work = Path(tempfile.mkdtemp(prefix="main-promotion-receipt-", dir=temp_root))
    wrapper = work / "main-promotion-receipt.zip"
    downloaded_size, downloaded_digest = api.download(
        f"/repos/{repository}/actions/artifacts/{artifact_id}/zip", wrapper, size
    )
    if downloaded_size != size or f"sha256:{downloaded_digest}" != expected_digest:
        raise PromotionError("main promotion receipt wrapper metadata mismatch")
    extracted = work / "plan"
    safe_extract_zip(wrapper, extracted, ["main-release-promotion-receipt.json"])
    return read_plan(extracted / "main-release-promotion-receipt.json")


def _main_plan_semantics(plan: dict) -> str:
    main_run = _mapping(plan.get("main_run"), "plan main run")
    receipt = _mapping(plan.get("receipt"), "plan receipt")
    semantic = {
        "schema_version": plan.get("schema_version"),
        "main_sha": plan.get("main_sha"),
        "main_tree_sha": plan.get("main_tree_sha"),
        "main_run": {
            "run_id": main_run.get("run_id"),
            "workflow_path": main_run.get("workflow_path"),
        },
        # Rerunning the trusted observer creates a new immutable wrapper ID even
        # when the source run and all six archive hashes are unchanged. Wrapper
        # identity is revalidated after selecting the newest plan; it is not a
        # release-semantic difference by itself.
        "receipt": json.loads(_receipt_semantics(receipt)),
    }
    return json.dumps(semantic, separators=(",", ":"), sort_keys=True)


def _download_main_plan(
    api: PromotionApi,
    repository: str,
    repository_id: int,
    main_run: dict,
    temp_root: Path,
) -> dict:
    run_id = _positive_int(main_run.get("id"), "main CI run id")
    terminal_attempt = _positive_int(
        main_run.get("run_attempt"), "main CI run attempt"
    )
    main_sha = _sha(main_run.get("head_sha"), "main CI head SHA")
    candidates = _main_receipt_artifacts(api, repository, run_id)
    if not candidates:
        raise PromotionError("main promotion receipt is not available")

    validated: list[tuple[int, int, dict]] = []
    validated_attempts: set[int] = set()
    for artifact in sorted(candidates, key=lambda item: int(item.get("id", 0))):
        artifact_run = _mapping(
            artifact.get("workflow_run"), "main receipt workflow run"
        )
        if artifact_run.get("id") != run_id:
            raise PromotionError("main promotion receipt belongs to a different run")
        plan = _download_main_plan_artifact(
            api, repository, repository_id, artifact, temp_root
        )
        plan_run = _mapping(plan.get("main_run"), "plan main run")
        origin_attempt = _positive_int(
            plan_run.get("run_attempt"), "plan main run attempt"
        )
        if origin_attempt > terminal_attempt:
            raise PromotionError("main promotion receipt claims a future run attempt")
        if (
            plan_run.get("run_id") != run_id
            or plan_run.get("workflow_path") != CI_WORKFLOW_PATH
            or artifact.get("name")
            != f"main-release-promotion-receipt-{run_id}-{origin_attempt}"
        ):
            raise PromotionError("main promotion receipt origin is inconsistent")
        if origin_attempt not in validated_attempts:
            _detect_changes_job(
                api, repository, run_id, origin_attempt, main_sha
            )
            validated_attempts.add(origin_attempt)
        validated.append(
            (
                origin_attempt,
                _positive_int(artifact.get("id"), "main receipt artifact id"),
                plan,
            )
        )

    semantics = {_main_plan_semantics(plan) for _attempt, _id, plan in validated}
    if len(semantics) != 1:
        raise PromotionError("main promotion receipt reruns produced conflicting semantics")
    return max(validated, key=lambda value: (value[0], value[1]))[2]


def consume_main_receipt(
    api: PromotionApi,
    repository: str,
    main_sha: str,
    temp_root: Path,
    *,
    main_run_id: int | None = None,
    main_run_attempt: int | None = None,
) -> tuple[GateResult, dict | None]:
    result = verify_main(api, repository, main_sha, temp_root)
    if result.state != "validated":
        return result, None
    try:
        repository_info = _mapping(
            api.get_json(f"/repos/{repository}"), "repository metadata"
        )
        repository_id = _positive_int(repository_info.get("id"), "repository id")
        main_run = _main_ci_run(
            api,
            repository,
            repository_id,
            main_sha,
            main_run_id,
            main_run_attempt,
        )
        plan = _download_main_plan(
            api, repository, repository_id, main_run, temp_root
        )
        expected_artifact = {
            key: result.receipt_artifact[key]
            for key in ("digest", "id", "name", "size_in_bytes")
        }
        plan_run = _mapping(plan.get("main_run"), "plan main run")
        if (
            plan.get("main_sha") != main_sha
            or plan.get("main_tree_sha") != result.main_tree_sha
            or plan_run.get("run_id") != main_run.get("id")
            or plan.get("receipt_artifact") != expected_artifact
            or _receipt_semantics(_mapping(plan.get("receipt"), "plan receipt"))
            != _receipt_semantics(_mapping(result.receipt, "validated receipt"))
        ):
            raise PromotionError("main promotion receipt differs from fresh validation")
        result.main_run_id = _positive_int(main_run.get("id"), "main CI run id")
        result.main_run_attempt = _positive_int(
            main_run.get("run_attempt"), "main CI run attempt"
        )
        return result, plan
    except (PromotionError, VALIDATOR.CandidateError, MERGE_PROOF.ApiError) as error:
        return GateResult("blocked", str(error), main_sha=main_sha), None
    except Exception as error:
        return (
            GateResult(
                "blocked",
                f"main receipt error {type(error).__name__}: {error}",
                main_sha=main_sha,
            ),
            None,
        )


def download_validated_assets(
    api: PromotionApi,
    repository: str,
    plan: dict,
    output_dir: Path,
) -> list[Path]:
    receipt = _mapping(plan.get("receipt"), "plan receipt")
    claim = _mapping(
        receipt.get("validated_assets_artifact"), "validated assets claim"
    )
    artifact_id = _positive_int(claim.get("id"), "validated assets id")
    metadata = _artifact_metadata(api, repository, artifact_id)
    if (
        metadata.get("name") != claim.get("name")
        or metadata.get("digest") != claim.get("digest")
        or metadata.get("expired") is not False
    ):
        raise PromotionError("validated assets metadata changed after main validation")
    size = _positive_int(metadata.get("size_in_bytes"), "validated assets size")
    if size > MAX_ARCHIVE_BYTES:
        raise PromotionError("validated assets wrapper exceeds the archive bound")
    output_dir.mkdir(parents=True, exist_ok=False)
    wrapper = output_dir.parent / f"validated-assets-{artifact_id}.zip"
    downloaded_size, downloaded_digest = api.download(
        f"/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        wrapper,
        size,
    )
    if (
        downloaded_size != size
        or f"sha256:{downloaded_digest}" != metadata.get("digest")
    ):
        raise PromotionError("validated assets wrapper size or digest mismatch")
    expected = [asset["name"] for asset in release_assets()]
    paths = safe_extract_zip(wrapper, output_dir, expected)
    claims = {asset["name"]: asset for asset in receipt.get("assets", [])}
    definitions = {asset["name"]: asset for asset in release_assets()}
    for path in paths:
        claim_asset = _mapping(claims.get(path.name), f"asset claim {path.name}")
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if path.stat().st_size != claim_asset.get("size") or digest != claim_asset.get("sha256"):
            raise PromotionError(f"validated asset {path.name} differs from its receipt")
        definition = definitions[path.name]
        safe_extract_archive(
            path,
            output_dir.parent / f"inspect-{path.name}",
            definition["archive"],
            definition["members"],
        )
    return sorted(paths)


def _single_line(value: object) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ")[:500]


def _write_outputs(path: Path, result: GateResult) -> None:
    receipt = result.receipt or {}
    validated = receipt.get("validated_assets_artifact") or {}
    receipt_artifact = result.receipt_artifact or {}
    values = {
        "release-gate-state": result.state,
        "verified-release-merge": str(result.state == "validated").lower(),
        "release-proof-reason": result.reason,
        "release-pr-number": receipt.get("pr_number", ""),
        "release-pr-head-sha": receipt.get("head_sha", ""),
        "release-tree-sha": result.main_tree_sha or "",
        "release-version": receipt.get("version", ""),
        "release-tag": f"v{receipt['version']}" if receipt.get("version") else "",
        "release-receipt-artifact-id": receipt_artifact.get("id", ""),
        "release-receipt-artifact-digest": receipt_artifact.get("digest", ""),
        "validated-assets-artifact-id": validated.get("id", ""),
        "validated-assets-artifact-digest": validated.get("digest", ""),
    }
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={_single_line(value)}\n")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate = subparsers.add_parser("gate-main")
    gate.add_argument("--repository", required=True)
    gate.add_argument("--sha", required=True)
    gate.add_argument("--temp-root", required=True, type=Path)
    gate.add_argument("--wait-seconds", type=int, default=0)
    gate.add_argument("--github-output", required=True, type=Path)
    gate.add_argument("--plan-output", type=Path)
    gate.add_argument("--main-run-id", type=int)
    gate.add_argument("--main-run-attempt", type=int)

    verify_ci = subparsers.add_parser("verify-main-ci")
    verify_ci.add_argument("--repository", required=True)
    verify_ci.add_argument("--sha", required=True)
    verify_ci.add_argument("--wait-seconds", type=int, default=0)

    consume = subparsers.add_parser("consume-main-receipt")
    consume.add_argument("--repository", required=True)
    consume.add_argument("--sha", required=True)
    consume.add_argument("--temp-root", required=True, type=Path)
    consume.add_argument("--github-output", required=True, type=Path)
    consume.add_argument("--plan-output", type=Path)
    consume.add_argument("--main-run-id", type=int)
    consume.add_argument("--main-run-attempt", type=int)

    download = subparsers.add_parser("download-assets")
    download.add_argument("--repository", required=True)
    download.add_argument("--plan", required=True, type=Path)
    download.add_argument("--output-dir", required=True, type=Path)

    args = parser.parse_args(argv)
    api = VALIDATOR.GitHubApi(
        os.environ.get("GITHUB_TOKEN", ""),
        os.environ.get("GITHUB_API_URL", "https://api.github.com"),
    )
    if args.command == "verify-main-ci":
        try:
            run = verify_main_ci(
                api,
                args.repository,
                args.sha,
                wait_seconds=max(args.wait_seconds, 0),
            )
        except Exception as error:
            print(
                f"main CI verification failed: {type(error).__name__}: {_single_line(error)}",
                file=sys.stderr,
            )
            return 2
        print(
            json.dumps(
                {
                    "head_sha": run["head_sha"],
                    "run_attempt": run["run_attempt"],
                    "run_id": run["id"],
                    "status": "validated",
                    "workflow_path": CI_WORKFLOW_PATH,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "gate-main":
        result = verify_main(
            api,
            args.repository,
            args.sha,
            args.temp_root,
            wait_seconds=max(args.wait_seconds, 0),
        )
        if result.state == "validated" and args.main_run_id is not None:
            result.main_run_id = _positive_int(args.main_run_id, "main run id")
            result.main_run_attempt = _positive_int(
                args.main_run_attempt, "main run attempt"
            )
        _write_outputs(args.github_output, result)
        if result.state == "validated" and args.plan_output is not None:
            write_plan(args.plan_output, result)
        print(f"release-gate-state={result.state} reason={_single_line(result.reason)}")
        return 0 if result.state != "blocked" else 2

    if args.command == "consume-main-receipt":
        result, plan = consume_main_receipt(
            api,
            args.repository,
            args.sha,
            args.temp_root,
            main_run_id=args.main_run_id,
            main_run_attempt=args.main_run_attempt,
        )
        _write_outputs(args.github_output, result)
        if result.state == "validated" and args.plan_output is not None:
            if plan is None:
                raise PromotionError("validated main receipt has no promotion plan")
            args.plan_output.write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(f"release-gate-state={result.state} reason={_single_line(result.reason)}")
        return 0 if result.state != "blocked" else 2

    plan = read_plan(args.plan)
    paths = download_validated_assets(api, args.repository, plan, args.output_dir)
    print(json.dumps({"assets": [path.name for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
