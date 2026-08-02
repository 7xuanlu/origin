#!/usr/bin/env python3
"""Prove that a main push is an already-tested release-please merge.

A false result is not a CI failure: it means the caller must run the ordinary
full main backstop.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol


API_VERSION = "2022-11-28"
MAIN_REF = "refs/heads/main"
RELEASE_BRANCH = "release-please--branches--main"
MAX_PR_FILES = 3000
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
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
        "plugin/bin/wenlan-mcp-runner.sh",
        "plugin/skills/setup/SKILL.md",
        "version.txt",
    }
)


class ApiError(RuntimeError):
    """A bounded GitHub API or response-shape failure."""


class JsonApi(Protocol):
    def get_json(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> object: ...


class ReleaseProof:
    def __init__(
        self,
        verified: bool,
        reason: str,
        *,
        pr_number: int | None = None,
        pr_head_sha: str | None = None,
        tree_sha: str | None = None,
    ) -> None:
        self.verified = verified
        self.reason = reason
        self.pr_number = pr_number
        self.pr_head_sha = pr_head_sha
        self.tree_sha = tree_sha


class GitHubApi:
    def __init__(
        self,
        token: str,
        *,
        api_url: str = "https://api.github.com",
        timeout_seconds: float = 20.0,
    ) -> None:
        if not token:
            raise ApiError("GITHUB_TOKEN is missing")
        self._token = token
        self._api_url = api_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> object:
        query = urllib.parse.urlencode(params or {})
        url = f"{self._api_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "wenlan-release-merge-proof",
                "X-GitHub-Api-Version": API_VERSION,
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            raise ApiError(f"GitHub API returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            reason = getattr(error, "reason", error)
            raise ApiError(f"GitHub API request failed: {reason}") from error
        except (http.client.HTTPException, OSError) as error:
            raise ApiError(
                f"GitHub API response failed: {type(error).__name__}: {error}"
            ) from error
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiError("GitHub API returned invalid JSON") from error


def _mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ApiError(f"{label} is not an object")
    return value


def _list(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ApiError(f"{label} is not an array")
    return value


def _release_pr_candidates(pulls: object, sha: str) -> list[dict]:
    candidates = []
    for raw_pr in _list(pulls, "associated pull requests"):
        pr = _mapping(raw_pr, "associated pull request")
        base = _mapping(pr.get("base"), "pull request base")
        head = _mapping(pr.get("head"), "pull request head")
        if (
            pr.get("state") == "closed"
            and isinstance(pr.get("merged_at"), str)
            and pr.get("merge_commit_sha") == sha
            and base.get("ref") == "main"
            and head.get("ref") == RELEASE_BRANCH
        ):
            candidates.append(pr)
    return candidates


def _tree_sha(api: JsonApi, repository: str, sha: str) -> str:
    commit = _mapping(
        api.get_json(f"/repos/{repository}/git/commits/{sha}"),
        f"Git commit {sha}",
    )
    tree = _mapping(commit.get("tree"), f"Git commit {sha} tree")
    tree_sha = tree.get("sha")
    if not isinstance(tree_sha, str) or not SHA_RE.fullmatch(tree_sha):
        raise ApiError(f"Git commit {sha} has no valid tree SHA")
    return tree_sha


def _release_files(
    api: JsonApi,
    repository: str,
    pr_number: int,
    changed_files: int,
) -> set[str]:
    if changed_files < 1 or changed_files > MAX_PR_FILES:
        raise ApiError(
            f"release PR changed_files must be 1..{MAX_PR_FILES}, got {changed_files}"
        )
    filenames: list[str] = []
    page = 1
    while len(filenames) < changed_files:
        raw_files = _list(
            api.get_json(
                f"/repos/{repository}/pulls/{pr_number}/files",
                params={"per_page": 100, "page": page},
            ),
            f"pull request {pr_number} files page {page}",
        )
        if not raw_files:
            break
        for raw_file in raw_files:
            file = _mapping(raw_file, "pull request file")
            filename = file.get("filename")
            status = file.get("status")
            if not isinstance(filename, str) or not filename:
                raise ApiError("pull request file has no filename")
            if status not in {"added", "modified"}:
                raise ApiError(
                    f"release-managed file {filename!r} has unsupported status {status!r}"
                )
            filenames.append(filename)
        if len(raw_files) < 100:
            break
        page += 1
    if len(filenames) != changed_files or len(set(filenames)) != changed_files:
        raise ApiError(
            "release PR file inventory is incomplete, duplicated, or truncated"
        )
    unexpected = sorted(set(filenames) - RELEASE_MANAGED_PATHS)
    if unexpected:
        raise ApiError(f"release PR changes non-release-managed path {unexpected[0]!r}")
    return set(filenames)


def _has_successful_conclusion(
    api: JsonApi,
    repository: str,
    head_sha: str,
) -> bool:
    payload = _mapping(
        api.get_json(
            f"/repos/{repository}/commits/{head_sha}/check-runs",
            params={"filter": "latest", "per_page": 100},
        ),
        "check runs",
    )
    runs = _list(payload.get("check_runs"), "check runs")
    for raw_run in runs:
        run = _mapping(raw_run, "check run")
        app = _mapping(run.get("app"), "check run app")
        if (
            run.get("name") == "conclusion"
            and run.get("status") == "completed"
            and run.get("conclusion") == "success"
            and run.get("head_sha") == head_sha
            and app.get("slug") == "github-actions"
        ):
            return True
    return False


def verify_release_merge(
    api: JsonApi,
    repository: str,
    *,
    event_name: str,
    ref: str,
    sha: str,
    associated_pulls: object | None = None,
) -> ReleaseProof:
    if event_name != "push" or ref != MAIN_REF:
        return ReleaseProof(False, "event is not a push to refs/heads/main")
    if not SHA_RE.fullmatch(sha):
        return ReleaseProof(False, "current commit SHA is invalid")
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repository):
        return ReleaseProof(False, "repository must be OWNER/REPO")

    try:
        pulls = associated_pulls
        if pulls is None:
            pulls = api.get_json(
                f"/repos/{repository}/commits/{sha}/pulls",
                params={"per_page": 100},
            )
        candidates = _release_pr_candidates(pulls, sha)
        if len(candidates) != 1:
            return ReleaseProof(
                False,
                f"expected one associated merged release PR, found {len(candidates)}",
            )
        summary = candidates[0]
        pr_number = summary.get("number")
        if not isinstance(pr_number, int) or pr_number < 1:
            raise ApiError("release PR number is invalid")
        pr = _mapping(
            api.get_json(f"/repos/{repository}/pulls/{pr_number}"),
            f"pull request {pr_number}",
        )
        base = _mapping(pr.get("base"), "pull request base")
        head = _mapping(pr.get("head"), "pull request head")
        if (
            pr.get("number") != pr_number
            or pr.get("state") != "closed"
            or not isinstance(pr.get("merged_at"), str)
            or pr.get("merge_commit_sha") != sha
            or base.get("ref") != "main"
            or head.get("ref") != RELEASE_BRANCH
        ):
            raise ApiError("full release PR no longer matches its commit association")
        changed_files = pr.get("changed_files")
        head_sha = head.get("sha")
        if not isinstance(changed_files, int):
            raise ApiError("release PR changed_files is invalid")
        if not isinstance(head_sha, str) or not SHA_RE.fullmatch(head_sha):
            raise ApiError("release PR head SHA is invalid")

        _release_files(api, repository, pr_number, changed_files)
        if not _has_successful_conclusion(api, repository, head_sha):
            return ReleaseProof(
                False,
                "release PR head has no successful latest conclusion check",
                pr_number=pr_number,
                pr_head_sha=head_sha,
            )
        current_tree = _tree_sha(api, repository, sha)
        head_tree = _tree_sha(api, repository, head_sha)
        if current_tree != head_tree:
            return ReleaseProof(
                False,
                "release PR head tree does not match the main merge tree",
                pr_number=pr_number,
                pr_head_sha=head_sha,
            )
        return ReleaseProof(
            True,
            "associated release PR conclusion and identical tree verified",
            pr_number=pr_number,
            pr_head_sha=head_sha,
            tree_sha=current_tree,
        )
    except ApiError as error:
        return ReleaseProof(False, str(error))


def _single_line(value: object) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ")[:500]


def _write_github_output(path: str, proof: ReleaseProof) -> None:
    values = {
        "verified-release-merge": str(proof.verified).lower(),
        "release-proof-reason": proof.reason,
        "release-pr-number": proof.pr_number or "",
        "release-pr-head-sha": proof.pr_head_sha or "",
        "release-tree-sha": proof.tree_sha or "",
    }
    with Path(path).open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={_single_line(value)}\n")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--github-output", required=True)
    arguments = parser.parse_args(argv)

    if arguments.event_name != "push" or arguments.ref != MAIN_REF:
        proof = ReleaseProof(False, "event is not a push to refs/heads/main")
    else:
        try:
            api = GitHubApi(os.environ.get("GITHUB_TOKEN", ""))
            proof = verify_release_merge(
                api,
                arguments.repository,
                event_name=arguments.event_name,
                ref=arguments.ref,
                sha=arguments.sha,
            )
        except Exception as error:
            proof = ReleaseProof(
                False,
                f"release proof error {type(error).__name__}: {error}",
            )
    _write_github_output(arguments.github_output, proof)
    print(
        f"verified-release-merge={str(proof.verified).lower()} "
        f"reason={_single_line(proof.reason)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
