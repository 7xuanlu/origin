#!/usr/bin/env python3
"""Observe and fail-closed validate untrusted release-candidate artifacts."""

from __future__ import annotations

import argparse
import base64
import datetime
import difflib
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

from release_archive import MAX_ARCHIVE_BYTES, safe_extract_archive, safe_extract_zip
from release_targets import release_assets, release_matrix

import importlib.util


_MANIFEST_PATH = Path(__file__).with_name("release-candidate-manifest.py")
_MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "release_candidate_manifest", _MANIFEST_PATH
)
assert _MANIFEST_SPEC and _MANIFEST_SPEC.loader
MANIFEST = importlib.util.module_from_spec(_MANIFEST_SPEC)
_MANIFEST_SPEC.loader.exec_module(MANIFEST)


API_VERSION = "2026-03-10"
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
CI_WORKFLOW_NAME = "CI"
RELEASE_BRANCH = "release-please--branches--main"
RELEASE_AUTHOR = "7xuanlu"
OBSERVER_WORKFLOW_PATH = ".github/workflows/release-candidate-observer.yml"
RECEIPT_SCHEMA_VERSION = 1
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WRAPPER_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
ACTION_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_API_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_PR_FILES = 32
MAX_TREE_ENTRIES = 50_000
RELEASE_MANAGED_PATHS = frozenset(
    {
        ".release-please-manifest.json",
        "CHANGELOG.md",
        "Cargo.lock",
        "Cargo.toml",
        "crates/wenlan-cli/npm/package.json",
        "crates/wenlan-mcp/npm/package.json",
        "plugin-codex/.codex-plugin/plugin.json",
        "plugin-codex/README.md",
        "plugin-codex/bin/wenlan-mcp-runner.sh",
        "plugin-codex/skills/setup/SKILL.md",
        "plugin/.claude-plugin/plugin.json",
        "plugin/skills/setup/SKILL.md",
        "version.txt",
    }
)
REQUIRED_RELEASE_PATHS = RELEASE_MANAGED_PATHS


class CandidateError(RuntimeError):
    """Trusted control-plane or untrusted payload evidence is invalid."""


class _CredentialStrippingRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward the GitHub bearer token to an artifact storage host."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        redirected = super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )
        if redirected is not None:
            old_host = urllib.parse.urlsplit(request.full_url).netloc
            new_host = urllib.parse.urlsplit(new_url).netloc
            if old_host.casefold() != new_host.casefold():
                redirected.remove_header("Authorization")
        return redirected


class JsonApi(Protocol):
    def get_json(
        self, path: str, *, params: dict[str, object] | None = None
    ) -> object: ...

    def download(self, path: str, destination: Path, maximum: int) -> tuple[int, str]: ...


class GitHubApi:
    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        if not token:
            raise CandidateError("GITHUB_TOKEN is missing")
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._opener = urllib.request.build_opener(_CredentialStrippingRedirect())

    def _request(self, path: str, params: dict[str, object] | None = None):
        query = urllib.parse.urlencode(params or {})
        url = f"{self._api_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "wenlan-release-candidate-observer",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        try:
            return self._opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            raise CandidateError(f"GitHub API returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as error:
            raise CandidateError(
                f"GitHub API request failed: {type(error).__name__}: {error}"
            ) from error

    def get_json(
        self, path: str, *, params: dict[str, object] | None = None
    ) -> object:
        with self._request(path, params) as response:
            payload = response.read(MAX_API_BYTES + 1)
        if len(payload) > MAX_API_BYTES:
            raise CandidateError("GitHub API JSON response exceeds the size bound")
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CandidateError("GitHub API returned invalid JSON") from error

    def download(self, path: str, destination: Path, maximum: int) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with self._request(path) as response, destination.open("xb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > maximum:
                    raise CandidateError("artifact download exceeds the declared size bound")
                digest.update(chunk)
                output.write(chunk)
        return size, digest.hexdigest()


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise CandidateError(f"{label} is not an object")
    return value


def _array(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise CandidateError(f"{label} is not an array")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CandidateError(f"{label} is not a positive integer")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise CandidateError(f"{label} is not a full commit SHA")
    return value


def _api_pages(api: JsonApi, path: str, label: str) -> list:
    values: list = []
    declared_total: int | None = None
    for page in range(1, 33):
        payload = api.get_json(path, params={"per_page": 100, "page": page})
        if isinstance(payload, dict):
            current_total = payload.get("total_count")
            if (
                not isinstance(current_total, int)
                or isinstance(current_total, bool)
                or current_total < 0
                or (declared_total is not None and current_total != declared_total)
            ):
                raise CandidateError(f"{label} has an invalid or unstable total_count")
            declared_total = current_total
            page_values = _array(payload.get("artifacts"), label)
        else:
            page_values = _array(payload, label)
        values.extend(page_values)
        if len(page_values) < 100:
            if declared_total is not None and len(values) != declared_total:
                raise CandidateError(f"{label} count differs from total_count")
            return values
    raise CandidateError(f"{label} pagination exceeds the bound")


def _content(
    api: JsonApi,
    repository: str,
    path: str,
    ref: str,
    expected_blob_sha: str | None = None,
) -> str:
    encoded_path = urllib.parse.quote(path, safe="")
    payload = _mapping(
        api.get_json(
            f"/repos/{repository}/contents/{encoded_path}", params={"ref": ref}
        ),
        f"contents {path}@{ref}",
    )
    if payload.get("type") != "file" or payload.get("encoding") != "base64":
        raise CandidateError(f"contents {path}@{ref} is not a base64 regular file")
    blob_sha = _sha(payload.get("sha"), f"contents {path}@{ref} blob SHA")
    if expected_blob_sha is not None and blob_sha != expected_blob_sha:
        raise CandidateError(f"contents {path}@{ref} differs from its Git tree blob SHA")
    encoded = payload.get("content")
    size = payload.get("size")
    if not isinstance(encoded, str) or not isinstance(size, int) or size < 0:
        raise CandidateError(f"contents {path}@{ref} has invalid metadata")
    if size > MAX_SOURCE_BYTES:
        raise CandidateError(f"contents {path}@{ref} exceeds the source size bound")
    try:
        compact = "".join(encoded.split())
        raw = base64.b64decode(compact, validate=True)
        text = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise CandidateError(f"contents {path}@{ref} is not canonical UTF-8/base64") from error
    if len(raw) != size or "\x00" in text:
        raise CandidateError(f"contents {path}@{ref} size or text encoding mismatches")
    return text


def _release_tree_records(
    api: JsonApi, repository: str, commit_sha: str, paths: set[str]
) -> dict[str, dict]:
    tree_sha = _tree_sha(api, repository, commit_sha)
    payload = _mapping(
        api.get_json(
            f"/repos/{repository}/git/trees/{tree_sha}", params={"recursive": "1"}
        ),
        f"recursive Git tree {tree_sha}",
    )
    if payload.get("sha") != tree_sha or payload.get("truncated") is not False:
        raise CandidateError("recursive Git tree is truncated or has the wrong root SHA")
    entries = _array(payload.get("tree"), "recursive Git tree entries")
    if len(entries) > MAX_TREE_ENTRIES:
        raise CandidateError("recursive Git tree exceeds the entry bound")
    records: dict[str, dict] = {}
    for raw_entry in entries:
        entry = _mapping(raw_entry, "Git tree entry")
        path = entry.get("path")
        if path not in paths:
            continue
        if not isinstance(path, str) or path in records:
            raise CandidateError("release-managed Git tree path is invalid or duplicated")
        if entry.get("type") != "blob" or entry.get("mode") not in {"100644", "100755"}:
            raise CandidateError(f"release-managed Git tree path {path!r} is not an expected blob")
        _sha(entry.get("sha"), f"Git tree blob {path} SHA")
        size = entry.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or size > MAX_SOURCE_BYTES:
            raise CandidateError(f"Git tree blob {path!r} has an invalid size")
        records[path] = entry
    if set(records) != paths:
        raise CandidateError("recursive Git tree omits a release-managed path")
    return records


def _validate_release_tree_modes(
    api: JsonApi,
    repository: str,
    base_sha: str,
    head_sha: str,
    paths: set[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    base = _release_tree_records(api, repository, base_sha, paths)
    head = _release_tree_records(api, repository, head_sha, paths)
    for path in paths:
        if base[path]["mode"] != head[path]["mode"] or base[path]["type"] != head[path]["type"]:
            raise CandidateError(f"release-managed Git mode/type changed for {path!r}")
    return base, head


def _version(value: str, label: str) -> tuple[int, int, int]:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value) is None:
        raise CandidateError(f"{label} is not a stable semantic version")
    return tuple(int(part) for part in value.split("."))


def _validate_content_delta(path: str, old: str, new: str, old_version: str, new_version: str) -> None:
    removed: list[str] = []
    added: list[str] = []
    for line in difflib.ndiff(old.splitlines(), new.splitlines()):
        if line.startswith("- "):
            removed.append(line[2:])
        elif line.startswith("+ "):
            added.append(line[2:])
    if not removed and not added:
        raise CandidateError(f"release-managed file {path!r} did not change")
    if len(removed) + len(added) > (1000 if path == "CHANGELOG.md" else 64):
        raise CandidateError(f"release-managed file {path!r} changes too many lines")
    if path == "CHANGELOG.md":
        if removed or not any(line.startswith(f"## [{new_version}]") for line in added):
            raise CandidateError("CHANGELOG.md is not an additive current-version entry")
        return
    if old_version not in old:
        raise CandidateError(f"release-managed file {path!r} has no base version to replace")
    if new != old.replace(old_version, new_version):
        raise CandidateError(
            f"release-managed file {path!r} is not the exact version-only transform"
        )


def _release_version_policy(config_text: str, old_version: str, new_version: str) -> None:
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as error:
        raise CandidateError("release-please config is not valid JSON") from error
    if not isinstance(config, dict) or set(config) != {"$schema", "packages"}:
        raise CandidateError("release-please config top-level schema is not closed")
    if not isinstance(config["$schema"], str) or len(config["$schema"]) > 512:
        raise CandidateError("release-please config $schema is invalid")
    packages = config["packages"]
    if not isinstance(packages, dict) or set(packages) != {"."}:
        raise CandidateError("release-please config package scope is not exactly root")
    package = packages["."]
    allowed = {"release-type", "versioning", "release-as"}
    if (
        not isinstance(package, dict)
        or not {"release-type", "versioning"}.issubset(package)
        or not set(package).issubset(allowed)
        or package.get("release-type") != "simple"
        or package.get("versioning") != "always-bump-patch"
    ):
        raise CandidateError("release-please root package policy is not closed always-bump-patch")
    old_parts = _version(old_version, "base version")
    new_parts = _version(new_version, "candidate version")
    release_as = package.get("release-as")
    if release_as is not None:
        if not isinstance(release_as, str):
            raise CandidateError("release-as is not a semantic version string")
        _version(release_as, "release-as")
        if new_version != release_as:
            raise CandidateError("candidate version does not exactly match release-as")
    elif new_parts != (old_parts[0], old_parts[1], old_parts[2] + 1):
        raise CandidateError("candidate version is not the required next patch")


def validate_release_pr_content(
    api: JsonApi, repository: str, pr: dict
) -> tuple[str, str, set[str]]:
    number = _positive_int(pr.get("number"), "pull request number")
    changed_files = _positive_int(pr.get("changed_files"), "pull request changed_files")
    if changed_files > MAX_PR_FILES:
        raise CandidateError("release PR changed_files exceeds the bounded allowlist")
    raw_files = _api_pages(api, f"/repos/{repository}/pulls/{number}/files", "PR files")
    if len(raw_files) != changed_files:
        raise CandidateError("release PR file inventory is incomplete or truncated")
    files: dict[str, dict] = {}
    for raw_file in raw_files:
        file = _mapping(raw_file, "pull request file")
        path = file.get("filename")
        if not isinstance(path, str) or not path or path in files:
            raise CandidateError("release PR file path is invalid or duplicated")
        if file.get("status") != "modified":
            raise CandidateError(f"release-managed path {path!r} is not a modification")
        files[path] = file
    paths = set(files)
    if paths != REQUIRED_RELEASE_PATHS:
        raise CandidateError(
            f"release PR paths mismatch: missing={sorted(REQUIRED_RELEASE_PATHS - paths)} "
            f"extra={sorted(paths - RELEASE_MANAGED_PATHS)}"
        )
    base = _mapping(pr.get("base"), "pull request base")
    head = _mapping(pr.get("head"), "pull request head")
    base_sha = _sha(base.get("sha"), "pull request base SHA")
    head_sha = _sha(head.get("sha"), "pull request head SHA")
    base_records, head_records = _validate_release_tree_modes(
        api, repository, base_sha, head_sha, paths
    )
    old_version = _content(
        api,
        repository,
        "version.txt",
        base_sha,
        base_records["version.txt"]["sha"],
    ).strip()
    new_version = _content(
        api,
        repository,
        "version.txt",
        head_sha,
        head_records["version.txt"]["sha"],
    ).strip()
    _release_version_policy(
        _content(api, repository, "release-please-config.json", base_sha),
        old_version,
        new_version,
    )
    for path in sorted(paths):
        old = _content(api, repository, path, base_sha, base_records[path]["sha"])
        new = _content(api, repository, path, head_sha, head_records[path]["sha"])
        _validate_content_delta(path, old, new, old_version, new_version)

    manifest = json.loads(_content(api, repository, ".release-please-manifest.json", head_sha))
    if manifest != {".": new_version}:
        raise CandidateError("release-please manifest has noncanonical content")
    for path in (
        "crates/wenlan-cli/npm/package.json",
        "crates/wenlan-mcp/npm/package.json",
        "plugin/.claude-plugin/plugin.json",
    ):
        value = json.loads(_content(api, repository, path, head_sha))
        if not isinstance(value, dict) or value.get("version") != new_version:
            raise CandidateError(f"{path} does not carry the candidate version")
    codex = json.loads(
        _content(api, repository, "plugin-codex/.codex-plugin/plugin.json", head_sha)
    )
    if not isinstance(codex, dict) or codex.get("version") != f"{new_version}+codex":
        raise CandidateError("Codex plugin manifest does not carry the candidate version")
    cargo = _content(api, repository, "Cargo.toml", head_sha)
    if f'version = "{new_version}"   # x-release-please-version' not in cargo:
        raise CandidateError("Cargo.toml lacks the marked candidate workspace version")
    return new_version, base_sha, paths


def _tree_sha(api: JsonApi, repository: str, commit_sha: str) -> str:
    commit = _mapping(
        api.get_json(f"/repos/{repository}/git/commits/{commit_sha}"), "Git commit"
    )
    tree = _mapping(commit.get("tree"), "Git commit tree")
    return _sha(tree.get("sha"), "Git tree SHA")


def _candidate_pr_state(
    api: JsonApi,
    repository: str,
    pr: dict,
    *,
    pr_number: int,
    repository_id: int,
    head_sha: str,
) -> tuple[str, str | None]:
    """Accept an open PR or a just-merged PR with an identical head tree."""

    base = _mapping(pr.get("base"), "candidate pull request base")
    head = _mapping(pr.get("head"), "candidate pull request head")
    user = _mapping(pr.get("user"), "candidate pull request author")
    base_repo = _mapping(base.get("repo"), "candidate base repository")
    head_repo = _mapping(head.get("repo"), "candidate head repository")
    if (
        pr.get("number") != pr_number
        or pr.get("draft") is not False
        or base.get("ref") != "main"
        or base_repo.get("id") != repository_id
        or base_repo.get("full_name") != repository
        or head.get("ref") != RELEASE_BRANCH
        or head.get("sha") != head_sha
        or head_repo.get("id") != repository_id
        or head_repo.get("full_name") != repository
        or head_repo.get("fork") is not False
        or user.get("login") != RELEASE_AUTHOR
    ):
        raise CandidateError("candidate pull request identity mismatch")

    if pr.get("state") == "open":
        if pr.get("merged") is not False or pr.get("merged_at") is not None:
            raise CandidateError("open candidate pull request has inconsistent merge state")
        return "open", None

    if pr.get("state") != "closed" or pr.get("merged") is not True:
        raise CandidateError("candidate pull request is closed without being merged")
    merged_at = pr.get("merged_at")
    if (
        not isinstance(merged_at, str)
        or len(merged_at) != 20
        or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", merged_at)
        is None
    ):
        raise CandidateError("merged candidate pull request has an invalid merged_at")
    try:
        datetime.datetime.strptime(merged_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise CandidateError("merged candidate pull request has an invalid merged_at") from error
    merge_events = [
        _mapping(event, "pull request issue event")
        for event in _api_pages(
            api,
            f"/repos/{repository}/issues/{pr_number}/events",
            "pull request issue events",
        )
        if isinstance(event, dict) and event.get("event") == "merged"
    ]
    if len(merge_events) != 1 or merge_events[0].get("created_at") != merged_at:
        raise CandidateError("merged pull request has no unique matching merge event")
    merge_commit_sha = _sha(
        merge_events[0].get("commit_id"), "pull request merge event commit SHA"
    )
    response_merge_sha = pr.get("merge_commit_sha")
    # API version 2026-03-10 currently returns null for this field, while the
    # trusted merged issue event retains the full commit_id. If GitHub returns
    # both, they must agree exactly.
    if response_merge_sha is not None and _sha(
        response_merge_sha, "pull request merge commit SHA"
    ) != merge_commit_sha:
        raise CandidateError("pull request and merge event commit SHAs differ")
    if _tree_sha(api, repository, merge_commit_sha) != _tree_sha(
        api, repository, head_sha
    ):
        raise CandidateError("merged candidate tree differs from the candidate head tree")
    return "merged", merge_commit_sha


def _expected_artifact_names(run_id: int, run_attempt: int) -> dict[str, str]:
    return {
        entry["target"]: f"release-candidate-{run_id}-{run_attempt}-{entry['target']}"
        for entry in release_matrix()["include"]
    }


def _validate_artifact_record(
    raw: object,
    *,
    expected_name: str,
    run_id: int,
    repository_id: int,
    head_sha: str,
) -> dict:
    artifact = _mapping(raw, "artifact")
    _positive_int(artifact.get("id"), "artifact id")
    if artifact.get("name") != expected_name:
        raise CandidateError("artifact name differs from the exact expected set")
    size = _positive_int(artifact.get("size_in_bytes"), "artifact size")
    if size > MAX_ARCHIVE_BYTES:
        raise CandidateError("artifact wrapper exceeds the size bound")
    if artifact.get("expired") is not False:
        raise CandidateError("artifact is expired or has no explicit expiry state")
    digest = artifact.get("digest")
    if not isinstance(digest, str) or WRAPPER_DIGEST_RE.fullmatch(digest) is None:
        raise CandidateError("artifact has no canonical SHA-256 digest")
    workflow_run = _mapping(artifact.get("workflow_run"), "artifact workflow_run")
    if (
        workflow_run.get("id") != run_id
        or workflow_run.get("repository_id") != repository_id
        or workflow_run.get("head_repository_id") != repository_id
        or workflow_run.get("head_branch") != RELEASE_BRANCH
        or workflow_run.get("head_sha") != head_sha
    ):
        raise CandidateError("artifact control-plane workflow identity mismatch")
    return artifact


def _reconcile_manifest(
    manifest: dict,
    *,
    repository: str,
    repository_id: int,
    run_id: int,
    run_attempt: int,
    pr_number: int,
    pr_author: str,
    head_sha: str,
    tree_sha: str,
    version: str,
    target: str,
    expected_artifact_name: str,
) -> None:
    expected = {
        "repository": {"full_name": repository, "id": repository_id},
        "workflow": {
            "event": "pull_request",
            "path": CI_WORKFLOW_PATH,
            "run_attempt": run_attempt,
            "run_id": run_id,
        },
        "pull_request": {
            "author": pr_author,
            "base_ref": "main",
            "head_ref": RELEASE_BRANCH,
            "head_sha": head_sha,
            "number": pr_number,
        },
        "source": {"commit_sha": head_sha, "tree_sha": tree_sha},
        "release": {"tag": f"v{version}", "version": version},
        "target": target,
        "artifact": {"name": expected_artifact_name},
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CandidateError(f"manifest {key} does not reconcile with trusted metadata")


def _download_and_validate(
    api: JsonApi,
    artifact: dict,
    *,
    target: str,
    work_dir: Path,
    trusted: dict,
    validated_assets_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    artifact_id = artifact["id"]
    wrapper = work_dir / f"artifact-{artifact_id}.zip"
    size, digest = api.download(
        f"/repos/{trusted['repository']}/actions/artifacts/{artifact_id}/zip",
        wrapper,
        artifact["size_in_bytes"],
    )
    expected_digest = WRAPPER_DIGEST_RE.fullmatch(artifact["digest"]).group(1)
    if size != artifact["size_in_bytes"] or digest != expected_digest:
        raise CandidateError("downloaded artifact wrapper size or digest mismatch")
    expected_assets = release_assets(target=target)
    outer_names = [asset["name"] for asset in expected_assets] + [MANIFEST.MANIFEST_NAME]
    extracted = work_dir / f"artifact-{artifact_id}"
    safe_extract_zip(wrapper, extracted, outer_names)
    manifest = MANIFEST.read_manifest(extracted / MANIFEST.MANIFEST_NAME)
    _reconcile_manifest(manifest, target=target, **trusted)
    assets: list[dict] = []
    payload_hashes: dict[tuple[str, str], str] = {}
    for index, (claim, expected) in enumerate(
        zip(manifest["assets"], expected_assets, strict=True)
    ):
        path = extracted / expected["name"]
        if path.stat().st_size != claim["size"] or MANIFEST.sha256_file(path) != claim["sha256"]:
            raise CandidateError(f"asset {expected['name']} size or SHA-256 mismatch")
        # Inspect and safely extract to prove there are no hidden paths, links,
        # special files, bombs, or schema-surprising members. Never execute or
        # chmod anything obtained from this untrusted workflow artifact.
        payload_paths = safe_extract_archive(
            path,
            work_dir / f"payload-{artifact_id}-{index}",
            expected["archive"],
            expected["members"],
        )
        for payload_path in payload_paths:
            payload_hashes[(expected["name"], payload_path.name)] = MANIFEST.sha256_file(
                payload_path
            )
        if validated_assets_dir is not None:
            destination = validated_assets_dir / expected["name"]
            if destination.exists():
                raise CandidateError("validated asset output name is duplicated")
            shutil.copyfile(path, destination)
            if (
                destination.stat().st_size != claim["size"]
                or MANIFEST.sha256_file(destination) != claim["sha256"]
            ):
                raise CandidateError("validated asset copy differs from inspected bytes")
        assets.append(
            {
                "name": expected["name"],
                "sha256": claim["sha256"],
                "size": claim["size"],
                "target": target,
            }
        )
    if target == "aarch64-apple-darwin":
        parity = [
            (
                ("wenlan-darwin-arm64.tar.gz", "wenlan"),
                ("wenlan-cli-darwin-arm64.tar.gz", "wenlan"),
            ),
            (
                ("wenlan-darwin-arm64.tar.gz", "wenlan-mcp"),
                ("wenlan-mcp-darwin-arm64.tar.gz", "wenlan-mcp"),
            ),
        ]
        if any(payload_hashes[main] != payload_hashes[standalone] for main, standalone in parity):
            raise CandidateError(
                "Darwin standalone archive bytes differ from the multi archive"
            )
    return assets, manifest


def validate_candidate(
    event: object,
    api: JsonApi,
    repository: str,
    temp_root: Path,
    *,
    validated_assets_dir: Path | None = None,
) -> dict:
    payload = _mapping(event, "workflow_run event")
    event_run = _mapping(payload.get("workflow_run"), "workflow_run")
    if event_run.get("head_branch") != RELEASE_BRANCH:
        return {
            "status": "skipped",
            "reason": "completed CI run is not the anchored release-please branch",
        }
    if payload.get("action") != "completed":
        raise CandidateError("candidate workflow_run action is not completed")
    run_id = _positive_int(event_run.get("id"), "event workflow run id")
    run_attempt = _positive_int(
        event_run.get("run_attempt"), "event workflow run attempt"
    )
    workflow_id = _positive_int(event_run.get("workflow_id"), "event workflow id")
    head_sha = _sha(event_run.get("head_sha"), "event workflow head SHA")

    repository_info = _mapping(
        api.get_json(f"/repos/{repository}"), "repository metadata"
    )
    repository_id = _positive_int(repository_info.get("id"), "repository id")
    if repository_info.get("full_name") != repository:
        raise CandidateError("repository full_name differs from validator scope")
    owner = _mapping(repository_info.get("owner"), "repository owner")
    if owner.get("login") != RELEASE_AUTHOR:
        raise CandidateError("repository owner differs from release PR author policy")

    run = _mapping(
        api.get_json(f"/repos/{repository}/actions/runs/{run_id}"), "workflow run"
    )
    workflow = _mapping(
        api.get_json(f"/repos/{repository}/actions/workflows/{workflow_id}"),
        "workflow metadata",
    )
    event_head_repository = _mapping(
        event_run.get("head_repository"), "event head repository"
    )
    run_head_repository = _mapping(run.get("head_repository"), "run head repository")
    event_repository = _mapping(event_run.get("repository"), "event run repository")
    run_repository = _mapping(run.get("repository"), "run repository")
    if (
        run.get("id") != run_id
        or run.get("run_attempt") != run_attempt
        or run.get("workflow_id") != workflow_id
        or workflow.get("id") != workflow_id
        or run.get("path") != CI_WORKFLOW_PATH
        or workflow.get("path") != CI_WORKFLOW_PATH
        or workflow.get("name") != CI_WORKFLOW_NAME
        or workflow.get("state") != "active"
        or event_run.get("name") != CI_WORKFLOW_NAME
        or run.get("event") != "pull_request"
        or event_run.get("event") != "pull_request"
        or run.get("status") != "completed"
        or event_run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or event_run.get("conclusion") != "success"
        or run.get("head_branch") != RELEASE_BRANCH
        or run.get("head_sha") != head_sha
        or event_head_repository.get("id") != repository_id
        or run_head_repository.get("id") != repository_id
        or event_head_repository.get("full_name") != repository
        or run_head_repository.get("full_name") != repository
        or event_repository.get("id") != repository_id
        or run_repository.get("id") != repository_id
        or event_repository.get("full_name") != repository
        or run_repository.get("full_name") != repository
        or event_head_repository.get("fork") is not False
        or run_head_repository.get("fork") is not False
    ):
        raise CandidateError("candidate workflow control-plane identity mismatch")

    associated = _array(
        api.get_json(
            f"/repos/{repository}/commits/{head_sha}/pulls", params={"per_page": 100}
        ),
        "head-SHA pull requests",
    )
    candidates = []
    for raw_pr in associated:
        pr = _mapping(raw_pr, "associated pull request")
        head = _mapping(pr.get("head"), "associated pull request head")
        if (
            pr.get("state") in {"open", "closed"}
            and pr.get("draft") is False
            and head.get("sha") == head_sha
        ):
            candidates.append(pr)
    if len(candidates) != 1:
        raise CandidateError(
            f"expected one open-or-merged nondraft PR for candidate head SHA, found {len(candidates)}"
        )
    pr_number = _positive_int(candidates[0].get("number"), "candidate PR number")
    pr = _mapping(
        api.get_json(f"/repos/{repository}/pulls/{pr_number}"), "candidate pull request"
    )
    pr_state, merge_commit_sha = _candidate_pr_state(
        api,
        repository,
        pr,
        pr_number=pr_number,
        repository_id=repository_id,
        head_sha=head_sha,
    )
    version, _base_sha, paths = validate_release_pr_content(api, repository, pr)
    tree_sha = _tree_sha(api, repository, head_sha)

    all_artifacts = _api_pages(
        api, f"/repos/{repository}/actions/runs/{run_id}/artifacts", "run artifacts"
    )
    expected_names = _expected_artifact_names(run_id, run_attempt)
    current_prefix = f"release-candidate-{run_id}-{run_attempt}-"
    candidate_artifacts = [
        artifact
        for artifact in all_artifacts
        if isinstance(artifact, dict)
        and isinstance(artifact.get("name"), str)
        and artifact["name"].startswith(current_prefix)
    ]
    by_name = {
        artifact.get("name"): artifact for artifact in candidate_artifacts
    }
    if len(candidate_artifacts) != 4 or set(by_name) != set(expected_names.values()):
        raise CandidateError("current run-attempt artifact set is not the exact four targets")
    ids = [artifact.get("id") for artifact in candidate_artifacts]
    if len(set(ids)) != 4:
        raise CandidateError("candidate artifact IDs are duplicated")

    work_dir = Path(tempfile.mkdtemp(prefix="release-candidate-", dir=temp_root))
    if validated_assets_dir is not None:
        validated_assets_dir.mkdir(parents=True, exist_ok=False)
    trusted_common = {
        "repository": repository,
        "repository_id": repository_id,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "pr_number": pr_number,
        "pr_author": RELEASE_AUTHOR,
        "head_sha": head_sha,
        "tree_sha": tree_sha,
        "version": version,
    }
    assets: list[dict] = []
    artifact_receipts: list[dict] = []
    for target, expected_name in expected_names.items():
        artifact = _validate_artifact_record(
            by_name[expected_name],
            expected_name=expected_name,
            run_id=run_id,
            repository_id=repository_id,
            head_sha=head_sha,
        )
        target_assets, _manifest = _download_and_validate(
            api,
            artifact,
            target=target,
            work_dir=work_dir,
            trusted={**trusted_common, "expected_artifact_name": expected_name},
            validated_assets_dir=validated_assets_dir,
        )
        assets.extend(target_assets)
        artifact_receipts.append(
            {
                "digest": artifact["digest"],
                "id": artifact["id"],
                "name": artifact["name"],
                "size": artifact["size_in_bytes"],
            }
        )
    expected_asset_names = {asset["name"] for asset in release_assets()}
    if len(assets) != 6 or {asset["name"] for asset in assets} != expected_asset_names:
        raise CandidateError("validated payload inventory is not the exact six release assets")
    return {
        "status": "passed",
        "repository": repository,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "workflow_id": workflow_id,
        "pr_number": pr_number,
        "pr_state": pr_state,
        "merge_commit_sha": merge_commit_sha,
        "head_sha": head_sha,
        "tree_sha": tree_sha,
        "version": version,
        "paths": sorted(paths),
        "artifacts": artifact_receipts,
        "assets": sorted(assets, key=lambda asset: asset["name"]),
    }


_VALIDATED_RECEIPT_KEYS = {
    "artifacts",
    "assets",
    "head_sha",
    "merge_commit_sha",
    "paths",
    "pr_number",
    "pr_state",
    "receipt_state",
    "repository",
    "run_attempt",
    "run_id",
    "schema_version",
    "status",
    "tree_sha",
    "version",
    "workflow_id",
}


def write_validated_receipt(path: Path, receipt: dict) -> None:
    if receipt.get("status") != "passed" or path.exists():
        raise CandidateError("only a new passed receipt can be written")
    document = {
        **receipt,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_state": "validated",
    }
    if set(document) != _VALIDATED_RECEIPT_KEYS:
        raise CandidateError("validated receipt schema is not closed")
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def close_receipt(
    path: Path,
    *,
    artifact_name: str,
    artifact_id: int,
    artifact_digest: str,
    observer_run_id: int,
    observer_run_attempt: int,
) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise CandidateError("validated receipt file is missing or oversized")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateError("validated receipt is not UTF-8 JSON") from error
    artifacts = document.get("artifacts") if isinstance(document, dict) else None
    assets = document.get("assets") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or set(document) != _VALIDATED_RECEIPT_KEYS
        or document.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or document.get("receipt_state") != "validated"
        or document.get("status") != "passed"
        or not isinstance(artifacts, list)
        or len(artifacts) != 4
        or not isinstance(assets, list)
        or len(assets) != 6
        or document.get("paths") != sorted(REQUIRED_RELEASE_PATHS)
    ):
        raise CandidateError("validated receipt cannot be closed")
    run_id = _positive_int(document.get("run_id"), "receipt source run id")
    run_attempt = _positive_int(
        document.get("run_attempt"), "receipt source run attempt"
    )
    _positive_int(document.get("workflow_id"), "receipt source workflow id")
    _positive_int(document.get("pr_number"), "receipt pull request number")
    _sha(document.get("head_sha"), "receipt head SHA")
    _sha(document.get("tree_sha"), "receipt tree SHA")
    receipt_version = document.get("version")
    if not isinstance(receipt_version, str):
        raise CandidateError("receipt version is not a string")
    _version(receipt_version, "receipt version")
    if (
        not isinstance(document.get("repository"), str)
        or re.fullmatch(r"[^/\s]+/[^/\s]+", document["repository"]) is None
    ):
        raise CandidateError("receipt repository is not OWNER/REPO")
    if document.get("pr_state") == "open":
        if document.get("merge_commit_sha") is not None:
            raise CandidateError("open receipt has a merge commit SHA")
    elif document.get("pr_state") == "merged":
        _sha(document.get("merge_commit_sha"), "receipt merge commit SHA")
    else:
        raise CandidateError("receipt PR state is invalid")
    expected_artifacts = _expected_artifact_names(run_id, run_attempt)
    artifact_names: set[str] = set()
    artifact_ids: set[int] = set()
    for raw_artifact in artifacts:
        artifact = _mapping(raw_artifact, "receipt source artifact")
        if set(artifact) != {"digest", "id", "name", "size"}:
            raise CandidateError("receipt source artifact schema is not closed")
        artifact_id_value = _positive_int(artifact.get("id"), "receipt artifact id")
        artifact_size = _positive_int(artifact.get("size"), "receipt artifact size")
        if artifact_size > MAX_ARCHIVE_BYTES:
            raise CandidateError("receipt source artifact exceeds the size bound")
        if WRAPPER_DIGEST_RE.fullmatch(str(artifact.get("digest"))) is None:
            raise CandidateError("receipt source artifact digest is invalid")
        artifact_names.add(str(artifact.get("name")))
        artifact_ids.add(artifact_id_value)
    if artifact_names != set(expected_artifacts.values()) or len(artifact_ids) != 4:
        raise CandidateError("receipt source artifact inventory is not exact")
    expected_assets = {asset["name"]: asset for asset in release_assets()}
    asset_names: set[str] = set()
    for raw_asset in assets:
        asset = _mapping(raw_asset, "receipt validated asset")
        if set(asset) != {"name", "sha256", "size", "target"}:
            raise CandidateError("receipt validated asset schema is not closed")
        name = asset.get("name")
        expected = expected_assets.get(name)
        if (
            expected is None
            or asset.get("target") != expected["target"]
            or not isinstance(asset.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None
        ):
            raise CandidateError("receipt validated asset identity is invalid")
        _positive_int(asset.get("size"), "receipt validated asset size")
        asset_names.add(name)
    if asset_names != set(expected_assets):
        raise CandidateError("receipt validated asset inventory is not exact")
    expected_name = (
        f"validated-release-assets-{run_id}-{run_attempt}-"
        f"{observer_run_id}-{observer_run_attempt}"
    )
    if artifact_name != expected_name:
        raise CandidateError("validated assets artifact name is not source-run scoped")
    _positive_int(artifact_id, "validated assets artifact id")
    if ACTION_DIGEST_RE.fullmatch(artifact_digest) is None:
        raise CandidateError("validated assets artifact digest is not canonical SHA-256")
    _positive_int(observer_run_id, "observer run id")
    _positive_int(observer_run_attempt, "observer run attempt")
    document["receipt_state"] = "closed"
    document["observer"] = {
        "run_attempt": observer_run_attempt,
        "run_id": observer_run_id,
        "workflow_path": OBSERVER_WORKFLOW_PATH,
    }
    document["validated_assets_artifact"] = {
        "digest": f"sha256:{artifact_digest}",
        "id": artifact_id,
        "name": artifact_name,
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def write_summary(path: Path, receipt: dict) -> None:
    def safe_text(value: object) -> str:
        text = str(value).replace("\r", " ").replace("\n", " ").replace("`", "'")
        return "".join(character if character.isprintable() else "?" for character in text)[:500]

    status = receipt.get("status", "failed")
    lines = ["## Release candidate observer", "", f"Status: **{status}**"]
    if status == "skipped":
        lines.extend(["", safe_text(receipt.get("reason", "non-candidate run"))])
    elif status == "failed":
        lines.extend(
            ["", f"Reason: `{safe_text(receipt.get('reason', 'validation failed'))}`"]
        )
    else:
        lines.extend(
            [
                "",
                f"- CI run/attempt: `{receipt['run_id']}/{receipt['run_attempt']}`",
                f"- PR: `#{receipt['pr_number']}`",
                f"- PR state / merge commit: `{receipt['pr_state']}` / `{receipt['merge_commit_sha'] or 'n/a'}`",
                f"- Head/tree: `{receipt['head_sha']}` / `{receipt['tree_sha']}`",
                f"- Candidate version: `{receipt['version']}`",
                f"- Exact artifacts/assets: `{len(receipt['artifacts'])}` / `{len(receipt['assets'])}`",
                "",
                "| Artifact ID | Name | Wrapper SHA-256 | Bytes |",
                "|---:|---|---|---:|",
            ]
        )
        for artifact in receipt["artifacts"]:
            lines.append(
                f"| {artifact['id']} | `{artifact['name']}` | `{artifact['digest']}` | {artifact['size']} |"
            )
        lines.extend(
            [
                "",
                "| Canonical inner asset | Target | SHA-256 | Bytes |",
                "|---|---|---|---:|",
            ]
        )
        for asset in sorted(receipt["assets"], key=lambda item: item["name"]):
            lines.append(
                f"| `{safe_text(asset['name'])}` | `{safe_text(asset['target'])}` | "
                f"`{safe_text(asset['sha256'])}` | {asset['size']} |"
            )
        lines.extend(
            ["", "This receipt is observe-only. It is not provenance and is not consumed by release.yml."]
        )
    with path.open("a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def _main(argv: list[str]) -> int:
    if argv[:1] == ["close-receipt"]:
        close_parser = argparse.ArgumentParser(
            description="Close a validated receipt over the uploaded six-asset wrapper."
        )
        close_parser.add_argument("--receipt", required=True, type=Path)
        close_parser.add_argument("--artifact-name", required=True)
        close_parser.add_argument("--artifact-id", required=True, type=int)
        close_parser.add_argument("--artifact-digest", required=True)
        close_parser.add_argument("--observer-run-id", required=True, type=int)
        close_parser.add_argument("--observer-run-attempt", required=True, type=int)
        arguments = close_parser.parse_args(argv[1:])
        try:
            closed = close_receipt(
                arguments.receipt,
                artifact_name=arguments.artifact_name,
                artifact_id=arguments.artifact_id,
                artifact_digest=arguments.artifact_digest,
                observer_run_id=arguments.observer_run_id,
                observer_run_attempt=arguments.observer_run_attempt,
            )
        except Exception as error:
            print(
                f"release candidate receipt close failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return 1
        print(
            json.dumps(
                {
                    "receipt_state": closed["receipt_state"],
                    "run_id": closed["run_id"],
                }
            )
        )
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--temp-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--validated-assets-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    args = parser.parse_args(argv)
    try:
        event = json.loads(args.event.read_text(encoding="utf-8"))
        api = GitHubApi(os.environ.get("GITHUB_TOKEN", ""), args.api_url)
        receipt = validate_candidate(
            event,
            api,
            args.repository,
            args.temp_root,
            validated_assets_dir=args.validated_assets_dir,
        )
        write_validated_receipt(args.receipt, receipt)
    except Exception as error:
        receipt = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        write_summary(args.summary, receipt)
        print(f"release candidate validation failed: {receipt['reason']}", file=sys.stderr)
        return 1
    write_summary(args.summary, receipt)
    print(json.dumps({"status": receipt["status"], "run_id": receipt.get("run_id")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
