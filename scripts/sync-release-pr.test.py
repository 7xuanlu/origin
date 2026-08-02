#!/usr/bin/env python3
"""Focused tests for fail-closed Release PR branch synchronization."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("sync-release-pr.py")
SPEC = importlib.util.spec_from_file_location("sync_release_pr", SCRIPT)
assert SPEC and SPEC.loader
SYNC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SYNC
SPEC.loader.exec_module(SYNC)

REPOSITORY = "7xuanlu/wenlan"
HEAD_SHA = "a" * 40


def release_pr(**updates: object) -> dict[str, object]:
    fixture: dict[str, object] = {
        "number": 430,
        "state": "open",
        "draft": False,
        "user": {"login": "7xuanlu"},
        "base": {"ref": "main", "repo": {"full_name": REPOSITORY}},
        "head": {
            "ref": "release-please--branches--main",
            "sha": HEAD_SHA,
            "repo": {"full_name": REPOSITORY, "fork": False},
        },
    }
    fixture.update(updates)
    return fixture


class FakeApi:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_json(self, path: str, *, params: dict[str, object]) -> object:
        self.calls.append((path, params))
        return self.payload


class FakeRunner:
    def __init__(
        self, *, push_fails: bool = False, remote_sha: str = HEAD_SHA
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.push_fails = push_fails
        self.remote_sha = remote_sha

    def __call__(
        self,
        command: list[str],
        *,
        check: bool,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output
        recorded = tuple(command)
        self.commands.append(recorded)
        if command == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout=HEAD_SHA + "\n")
        if command == ["git", "diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(command, 0, stdout="")
        if command[:3] == ["git", "ls-remote", "--exit-code"]:
            ref = "refs/heads/release-please--branches--main"
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{self.remote_sha}\t{ref}\n"
            )
        if command[:3] == ["git", "push", "origin"] and self.push_fails:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="")


class SyncReleasePrTests(unittest.TestCase):
    def test_zero_one_and_multiple_release_prs_are_closed(self) -> None:
        self.assertIsNone(SYNC.resolve_release_pr(FakeApi([]), REPOSITORY))
        resolved = SYNC.resolve_release_pr(FakeApi([release_pr()]), REPOSITORY)
        self.assertEqual(resolved.number, 430)
        self.assertEqual(resolved.head_sha, HEAD_SHA)
        with self.assertRaisesRegex(SYNC.SyncError, "at most one"):
            SYNC.resolve_release_pr(
                FakeApi([release_pr(), release_pr(number=431)]), REPOSITORY
            )

    def test_wrong_release_pr_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(SYNC.SyncError, "pinned Release PR author"):
            SYNC.resolve_release_pr(FakeApi([release_pr()]), "mallory/wenlan")
        mutations = {
            "draft": {"draft": True},
            "owner": {"user": {"login": "mallory"}},
            "head repository": {
                "head": {
                    "ref": "release-please--branches--main",
                    "sha": HEAD_SHA,
                    "repo": {"full_name": "mallory/wenlan", "fork": True},
                }
            },
            "base": {
                "base": {"ref": "other", "repo": {"full_name": REPOSITORY}}
            },
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                SYNC.SyncError, "identity"
            ):
                SYNC.resolve_release_pr(FakeApi([release_pr(**mutation)]), REPOSITORY)

    def test_noop_still_pushes_exact_nonforce_refspec(self) -> None:
        runner = FakeRunner()
        SYNC.synchronize_release_branch(
            HEAD_SHA, Path("/trusted/bump-version.sh"), run_command=runner
        )
        self.assertIn(("git", "fetch", "--no-tags", "origin", "main"), runner.commands)
        self.assertIn(("git", "merge", "--no-edit", "origin/main"), runner.commands)
        self.assertEqual(
            runner.commands[-1],
            (
                "git",
                "push",
                "origin",
                "HEAD:refs/heads/release-please--branches--main",
            ),
        )
        self.assertNotIn("--force", runner.commands[-1])

    def test_branch_move_is_rejected_before_nonforce_push(self) -> None:
        runner = FakeRunner(remote_sha="b" * 40)
        with self.assertRaisesRegex(SYNC.SyncError, "moved or disappeared"):
            SYNC.synchronize_release_branch(
                HEAD_SHA, Path("/trusted/bump-version.sh"), run_command=runner
            )
        self.assertFalse(
            any(command[:3] == ("git", "push", "origin") for command in runner.commands)
        )

    def test_racing_nonforce_push_rejection_fails_closed(self) -> None:
        runner = FakeRunner(push_fails=True)
        with self.assertRaisesRegex(SYNC.SyncError, "branch may have moved"):
            SYNC.synchronize_release_branch(
                HEAD_SHA, Path("/trusted/bump-version.sh"), run_command=runner
            )


if __name__ == "__main__":
    unittest.main()
