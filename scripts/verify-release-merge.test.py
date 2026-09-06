#!/usr/bin/env python3
"""Contract tests for fail-closed release-merge reuse proof."""

from __future__ import annotations

import http.client
import importlib.util
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).with_name("verify-release-merge.py")
CURRENT_SHA = "61386c967734c3fe819586098f8236b2b9a6d961"
HEAD_SHA = "4e4efe0f772e9d3d0b29c1b668f91c2f93d0e5bb"
TREE_SHA = "98fa6fb36b6f0cfdd9a58b9a308ddc66be35d294"
REPOSITORY = "7xuanlu/wenlan"
RELEASE_FILES = (
    ".release-please-manifest.json",
    "CHANGELOG.md",
    "Cargo.lock",
    "Cargo.toml",
    "crates/wenlan-cli/npm/package.json",
    "crates/wenlan-mcp/npm/package.json",
    "plugin-codex/.codex-plugin/plugin.json",
    "plugin-codex/skills/setup/SKILL.md",
    "plugin/.claude-plugin/plugin.json",
    "plugin/skills/setup/SKILL.md",
    "version.txt",
)


def load_script():
    spec = importlib.util.spec_from_file_location("verify_release_merge", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_validator():
    validator_path = Path(__file__).with_name("validate-release-candidate.py")
    spec = importlib.util.spec_from_file_location(
        "validate_release_candidate", validator_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeApi:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple[tuple[str, object], ...]]] = []

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, object] | None = None,
    ) -> object:
        query = tuple(sorted((params or {}).items()))
        self.calls.append((path, query))
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        return response


class VerifyReleaseMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_script()
        prefix = f"/repos/{REPOSITORY}"
        self.paths = {
            "pulls": f"{prefix}/commits/{CURRENT_SHA}/pulls",
            "current_commit": f"{prefix}/git/commits/{CURRENT_SHA}",
            "head_commit": f"{prefix}/git/commits/{HEAD_SHA}",
            "checks": f"{prefix}/commits/{HEAD_SHA}/check-runs",
            "pr": f"{prefix}/pulls/376",
            "files": f"{prefix}/pulls/376/files",
        }
        self.responses: dict[str, object] = {
            self.paths["pulls"]: [
                {
                    "number": 376,
                    "state": "closed",
                    "merged_at": "2026-07-26T04:40:00Z",
                    "merge_commit_sha": CURRENT_SHA,
                    "base": {"ref": "main"},
                    "head": {
                        "ref": "release-please--branches--main",
                        "sha": HEAD_SHA,
                    },
                }
            ],
            self.paths["pr"]: {
                "number": 376,
                "state": "closed",
                "merged_at": "2026-07-26T04:40:00Z",
                "merge_commit_sha": CURRENT_SHA,
                "changed_files": len(RELEASE_FILES),
                "base": {"ref": "main"},
                "head": {
                    "ref": "release-please--branches--main",
                    "sha": HEAD_SHA,
                },
            },
            self.paths["current_commit"]: {"tree": {"sha": TREE_SHA}},
            self.paths["head_commit"]: {"tree": {"sha": TREE_SHA}},
            self.paths["checks"]: {
                "total_count": 1,
                "check_runs": [
                    {
                        "name": "conclusion",
                        "status": "completed",
                        "conclusion": "success",
                        "head_sha": HEAD_SHA,
                        "app": {"slug": "github-actions"},
                    }
                ],
            },
            self.paths["files"]: [
                {"filename": filename, "status": "modified"}
                for filename in RELEASE_FILES
            ],
        }

    def verify(self, api: FakeApi | None = None):
        return self.module.verify_release_merge(
            api or FakeApi(self.responses),
            REPOSITORY,
            event_name="push",
            ref="refs/heads/main",
            sha=CURRENT_SHA,
        )

    def test_accepts_the_live_v015_release_proof_shape(self) -> None:
        proof = self.verify()

        self.assertTrue(proof.verified, proof.reason)
        self.assertEqual(proof.pr_number, 376)
        self.assertEqual(proof.pr_head_sha, HEAD_SHA)
        self.assertEqual(proof.tree_sha, TREE_SHA)

    def test_compatibility_pr_api_recovers_null_modern_merge_sha(self) -> None:
        modern_responses = deepcopy(self.responses)
        modern_responses[self.paths["pulls"]][0]["merge_commit_sha"] = None
        modern_responses[self.paths["pr"]]["merge_commit_sha"] = None
        modern = FakeApi(modern_responses)
        compatibility = FakeApi(
            {
                self.paths["pulls"]: deepcopy(self.responses[self.paths["pulls"]]),
                self.paths["pr"]: deepcopy(self.responses[self.paths["pr"]]),
                self.paths["files"]: deepcopy(self.responses[self.paths["files"]]),
            }
        )

        proof = self.module.verify_release_merge(
            modern,
            REPOSITORY,
            event_name="push",
            ref="refs/heads/main",
            sha=CURRENT_SHA,
            pr_api=compatibility,
        )

        self.assertTrue(proof.verified, proof.reason)
        self.assertEqual(
            [path for path, _query in compatibility.calls],
            [self.paths["pulls"], self.paths["pr"], self.paths["files"]],
        )
        self.assertNotIn(
            self.paths["pulls"], [path for path, _query in modern.calls]
        )

    def test_compatibility_pr_api_still_rejects_mismatch_and_ambiguity(self) -> None:
        for label, mutate in {
            "mismatch": lambda responses: responses[self.paths["pr"]].update(
                merge_commit_sha="0" * 40
            ),
            "ambiguous": lambda responses: responses[self.paths["pulls"]].append(
                {**responses[self.paths["pulls"]][0], "number": 377}
            ),
        }.items():
            with self.subTest(label=label):
                self.setUp()
                modern = FakeApi(deepcopy(self.responses))
                compatibility_responses = {
                    self.paths["pulls"]: deepcopy(self.responses[self.paths["pulls"]]),
                    self.paths["pr"]: deepcopy(self.responses[self.paths["pr"]]),
                    self.paths["files"]: deepcopy(self.responses[self.paths["files"]]),
                }
                mutate(compatibility_responses)

                proof = self.module.verify_release_merge(
                    modern,
                    REPOSITORY,
                    event_name="push",
                    ref="refs/heads/main",
                    sha=CURRENT_SHA,
                    pr_api=FakeApi(compatibility_responses),
                )

                self.assertFalse(proof.verified, proof.reason)

    def test_reuses_caller_provided_association_snapshot(self) -> None:
        api = FakeApi(
            {
                path: response
                for path, response in self.responses.items()
                if path != self.paths["pulls"]
            }
        )

        proof = self.module.verify_release_merge(
            api,
            REPOSITORY,
            event_name="push",
            ref="refs/heads/main",
            sha=CURRENT_SHA,
            associated_pulls=self.responses[self.paths["pulls"]],
        )

        self.assertTrue(proof.verified, proof.reason)
        self.assertNotIn(self.paths["pulls"], [path for path, _query in api.calls])

    def test_direct_main_push_falls_back_without_an_associated_pr(self) -> None:
        self.responses[self.paths["pulls"]] = []

        proof = self.verify()

        self.assertFalse(proof.verified)
        self.assertIn("associated", proof.reason)

    def test_non_main_or_non_push_event_never_calls_the_api(self) -> None:
        for event_name, ref in [
            ("workflow_dispatch", "refs/heads/main"),
            ("push", "refs/heads/topic"),
            ("pull_request", "refs/pull/376/merge"),
        ]:
            with self.subTest(event_name=event_name, ref=ref):
                api = FakeApi(self.responses)
                proof = self.module.verify_release_merge(
                    api,
                    REPOSITORY,
                    event_name=event_name,
                    ref=ref,
                    sha=CURRENT_SHA,
                )
                self.assertFalse(proof.verified)
                self.assertEqual(api.calls, [])

    def test_wrong_branch_failed_check_tree_mismatch_or_extra_file_rejects(self) -> None:
        mutations = {
            "wrong branch": lambda: self.responses[self.paths["pulls"]][0][
                "head"
            ].update(ref="topic"),
            "wrong merge SHA": lambda: self.responses[self.paths["pr"]].update(
                merge_commit_sha="0" * 40
            ),
            "failed check": lambda: self.responses[self.paths["checks"]][
                "check_runs"
            ][0].update(conclusion="failure"),
            "tree mismatch": lambda: self.responses[
                self.paths["head_commit"]
            ].update(tree={"sha": "different"}),
            "extra file": lambda: (
                self.responses[self.paths["pr"]].update(
                    changed_files=len(RELEASE_FILES) + 1
                ),
                self.responses[self.paths["files"]].append(
                    {"filename": "crates/wenlan-core/src/lib.rs", "status": "modified"}
                ),
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                self.setUp()
                mutate()
                proof = self.verify()
                self.assertFalse(proof.verified, label)

    def test_ambiguous_association_and_api_error_fail_closed(self) -> None:
        self.responses[self.paths["pulls"]] = [
            *self.responses[self.paths["pulls"]],
            {
                **self.responses[self.paths["pulls"]][0],
                "number": 377,
            },
        ]
        ambiguous = self.verify()
        self.assertFalse(ambiguous.verified)

        api = FakeApi(
            {
                **self.responses,
                self.paths["pulls"]: self.module.ApiError("rate limited"),
            }
        )
        errored = self.verify(api)
        self.assertFalse(errored.verified)
        self.assertIn("rate limited", errored.reason)

    def test_incomplete_http_read_writes_false_instead_of_aborting_ci(self) -> None:
        class IncompleteResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                raise http.client.IncompleteRead(b"partial", 100)

        original_urlopen = self.module.urllib.request.urlopen
        original_token = os.environ.get("GITHUB_TOKEN")
        self.module.urllib.request.urlopen = lambda *_args, **_kwargs: IncompleteResponse()
        os.environ["GITHUB_TOKEN"] = "fixture-token"
        self.addCleanup(
            setattr,
            self.module.urllib.request,
            "urlopen",
            original_urlopen,
        )
        if original_token is None:
            self.addCleanup(os.environ.pop, "GITHUB_TOKEN", None)
        else:
            self.addCleanup(os.environ.__setitem__, "GITHUB_TOKEN", original_token)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            result = self.module._main(
                [
                    "--repository",
                    REPOSITORY,
                    "--event-name",
                    "push",
                    "--ref",
                    "refs/heads/main",
                    "--sha",
                    CURRENT_SHA,
                    "--github-output",
                    str(output),
                ]
            )
            emitted = output.read_text()

        self.assertEqual(result, 0)
        self.assertIn("verified-release-merge=false\n", emitted)
        self.assertIn("release-proof-reason=", emitted)


class ReleaseManagedPathsConsistencyTests(unittest.TestCase):
    """A stale copy of RELEASE_MANAGED_PATHS must not silently drift.

    validate-release-candidate.py is the canonical source of REQUIRED_RELEASE_PATHS.
    This copy may carry extra entries, so the contract is superset, not
    equality.
    """

    def test_release_managed_paths_is_superset_of_validator_required_paths(self) -> None:
        merge_module = load_script()
        validator_module = load_validator()
        missing = (
            validator_module.REQUIRED_RELEASE_PATHS - merge_module.RELEASE_MANAGED_PATHS
        )
        self.assertEqual(
            missing,
            set(),
            "verify-release-merge.py RELEASE_MANAGED_PATHS is missing paths tracked "
            f"by validate-release-candidate.py REQUIRED_RELEASE_PATHS: {sorted(missing)}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
