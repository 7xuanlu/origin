#!/usr/bin/env python3
"""Resolve and safely advance the single pending Release Please PR."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


BASE_BRANCH = "main"
RELEASE_BRANCH = "release-please--branches--main"
RELEASE_AUTHOR = "7xuanlu"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class SyncError(RuntimeError):
    """The release branch could not be identified or advanced safely."""


@dataclass(frozen=True)
class ReleasePullRequest:
    number: int
    head_sha: str


class GitHubApi:
    def __init__(self, token: str, api_url: str) -> None:
        if not token:
            raise SyncError("GITHUB_TOKEN is required")
        self._token = token
        self._api_url = api_url.rstrip("/")

    def get_json(self, path: str, *, params: dict[str, object]) -> object:
        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(
            f"{self._api_url}{path}?{query}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)


def resolve_release_pr(api: GitHubApi, repository: str) -> ReleasePullRequest | None:
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise SyncError("repository must be owner/name")
    owner = parts[0]
    if owner != RELEASE_AUTHOR:
        raise SyncError("repository owner does not match the pinned Release PR author")
    payload = api.get_json(
        f"/repos/{repository}/pulls",
        params={
            "state": "open",
            "base": BASE_BRANCH,
            "head": f"{RELEASE_AUTHOR}:{RELEASE_BRANCH}",
            "per_page": 100,
        },
    )
    if not isinstance(payload, list):
        raise SyncError("GitHub pull request response is not a list")
    if not payload:
        return None
    if len(payload) != 1:
        raise SyncError(f"expected at most one pending Release PR, found {len(payload)}")

    pr = payload[0]
    if not isinstance(pr, dict):
        raise SyncError("pending Release PR record is not an object")
    base = pr.get("base")
    head = pr.get("head")
    user = pr.get("user")
    if (
        pr.get("state") != "open"
        or pr.get("draft") is not False
        or not isinstance(base, dict)
        or base.get("ref") != BASE_BRANCH
        or not isinstance(base.get("repo"), dict)
        or base["repo"].get("full_name") != repository
        or not isinstance(head, dict)
        or head.get("ref") != RELEASE_BRANCH
        or not isinstance(head.get("repo"), dict)
        or head["repo"].get("full_name") != repository
        or head["repo"].get("fork") is not False
        or not isinstance(user, dict)
        or user.get("login") != RELEASE_AUTHOR
    ):
        raise SyncError("pending Release PR identity is not exact owner same-repository main")
    number = pr.get("number")
    head_sha = head.get("sha")
    if (
        isinstance(number, bool)
        or not isinstance(number, int)
        or number <= 0
        or not isinstance(head_sha, str)
        or SHA_RE.fullmatch(head_sha) is None
    ):
        raise SyncError("pending Release PR number or immutable head SHA is invalid")
    return ReleasePullRequest(number=number, head_sha=head_sha)


def write_resolution(path: Path, release_pr: ReleasePullRequest | None) -> None:
    with path.open("a", encoding="utf-8") as output:
        if release_pr is None:
            output.write("release_pr_state=absent\n")
            return
        output.write("release_pr_state=present\n")
        output.write(f"release_pr_number={release_pr.number}\n")
        output.write(f"release_pr_head_sha={release_pr.head_sha}\n")


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _default_run(command: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, **kwargs)


def synchronize_release_branch(
    expected_head_sha: str,
    bump_script: Path,
    *,
    run_command: RunCommand = _default_run,
) -> None:
    if SHA_RE.fullmatch(expected_head_sha) is None:
        raise SyncError("expected release PR head SHA is invalid")
    current = run_command(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True
    ).stdout.strip()
    if current != expected_head_sha:
        raise SyncError(f"checkout HEAD {current!r} does not match resolved Release PR head")

    run_command(
        ["git", "config", "user.name", "github-actions[bot]"], check=True
    )
    run_command(
        [
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ],
        check=True,
    )
    run_command(["git", "fetch", "--no-tags", "origin", BASE_BRANCH], check=True)
    run_command(["git", "merge", "--no-edit", "origin/main"], check=True)
    run_command(["bash", str(bump_script)], check=True)
    run_command(["git", "add", "-u"], check=True)
    staged = run_command(["git", "diff", "--cached", "--quiet"], check=False)
    if staged.returncode not in {0, 1}:
        raise SyncError("git diff could not inspect synchronized version changes")
    if staged.returncode == 1:
        version = Path("version.txt").read_text(encoding="utf-8").strip()
        run_command(
            ["git", "commit", "-m", f"chore: sync versions to {version}"], check=True
        )

    remote = run_command(
        ["git", "ls-remote", "--exit-code", "origin", f"refs/heads/{RELEASE_BRANCH}"],
        check=False,
        capture_output=True,
    )
    remote_lines = remote.stdout.splitlines() if remote.returncode == 0 else []
    remote_shas = [line.split()[0] for line in remote_lines if len(line.split()) == 2]
    if remote_shas != [expected_head_sha]:
        raise SyncError("release branch moved or disappeared after immutable checkout")

    push = [
        "git",
        "push",
        "origin",
        "HEAD:refs/heads/release-please--branches--main",
    ]
    try:
        run_command(push, check=True)
    except subprocess.CalledProcessError as error:
        raise SyncError(
            "non-force release branch push failed; the remote branch may have moved"
        ) from error


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--repository", required=True)
    resolve.add_argument("--github-output", type=Path, required=True)
    sync = subparsers.add_parser("sync")
    sync.add_argument("--expected-head-sha", required=True)
    sync.add_argument("--bump-script", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "resolve":
            api = GitHubApi(
                os.environ.get("GITHUB_TOKEN", ""),
                os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            )
            release_pr = resolve_release_pr(api, args.repository)
            write_resolution(args.github_output, release_pr)
            if release_pr is None:
                print("No exact pending Release PR; maintenance safely skipped.")
            else:
                print(
                    f"Resolved Release PR #{release_pr.number} at {release_pr.head_sha}."
                )
        else:
            synchronize_release_branch(args.expected_head_sha, args.bump_script)
    except (OSError, subprocess.CalledProcessError, SyncError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
