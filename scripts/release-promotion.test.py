#!/usr/bin/env python3
"""Focused contracts for the trusted release-promotion consumer."""

from __future__ import annotations

import io
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("release-promotion.py")
SPEC = importlib.util.spec_from_file_location("release_promotion", SCRIPT)
assert SPEC and SPEC.loader
PROMOTION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROMOTION
SPEC.loader.exec_module(PROMOTION)

REPOSITORY = "7xuanlu/wenlan"
HEAD_SHA = "1" * 40


class FakeApi:
    def __init__(self, responses: dict[tuple[str, tuple], object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple]] = []

    def get_json(self, path: str, *, params: dict[str, object] | None = None):
        key = (path, tuple(sorted((params or {}).items())))
        self.calls.append(key)
        return self.responses[key]


class ReleasePromotionTests(unittest.TestCase):
    def test_release_context_scopes_compatibility_client_to_pr_proof(self) -> None:
        main_sha = "2" * 40
        head_sha = "3" * 40
        association_path = f"/repos/{REPOSITORY}/commits/{main_sha}/pulls"
        compatibility = FakeApi(
            {
                (
                    association_path,
                    (("per_page", 100),),
                ): [
                    {
                        "number": 430,
                        "base": {"ref": "main"},
                        "head": {"ref": PROMOTION.RELEASE_BRANCH, "sha": head_sha},
                        "merge_commit_sha": main_sha,
                    }
                ]
            }
        )
        modern = FakeApi({})
        proof = mock.Mock(verified=True, pr_head_sha=head_sha)

        with mock.patch.object(
            PROMOTION.MERGE_PROOF,
            "verify_release_merge",
            return_value=proof,
        ) as verify:
            resolved, resolved_head = PROMOTION._release_context(
                modern, REPOSITORY, main_sha, pr_api=compatibility
            )

        self.assertIs(resolved, proof)
        self.assertEqual(resolved_head, head_sha)
        self.assertEqual(modern.calls, [])
        self.assertEqual(
            compatibility.calls,
            [(association_path, (("per_page", 100),))],
        )
        verify.assert_called_once_with(
            modern,
            REPOSITORY,
            event_name="push",
            ref="refs/heads/main",
            sha=main_sha,
            associated_pulls=compatibility.responses[
                (association_path, (("per_page", 100),))
            ],
            pr_api=compatibility,
        )

    def test_release_context_reuses_the_association_snapshot(self) -> None:
        main_sha = "2" * 40
        head_sha = "3" * 40
        association = {
            "number": 430,
            "base": {"ref": "main"},
            "head": {"ref": PROMOTION.RELEASE_BRANCH, "sha": head_sha},
        }
        path = f"/repos/{REPOSITORY}/commits/{main_sha}/pulls"
        api = FakeApi(
            {
                (path, (("per_page", 100),)): [association],
            }
        )
        proof = mock.Mock(verified=True, pr_head_sha=head_sha)

        with mock.patch.object(
            PROMOTION.MERGE_PROOF,
            "verify_release_merge",
            return_value=proof,
        ) as verify:
            resolved, resolved_head = PROMOTION._release_context(
                api, REPOSITORY, main_sha
            )

        self.assertIs(resolved, proof)
        self.assertEqual(resolved_head, head_sha)
        verify.assert_called_once_with(
            api,
            REPOSITORY,
            event_name="push",
            ref="refs/heads/main",
            sha=main_sha,
            associated_pulls=[association],
            pr_api=api,
        )
        self.assertEqual(api.calls, [(path, (("per_page", 100),))])

    def test_release_context_recovers_an_initially_missing_association(self) -> None:
        main_sha = "2" * 40
        head_sha = "3" * 40
        association_path = f"/repos/{REPOSITORY}/commits/{main_sha}/pulls"
        commit_path = f"/repos/{REPOSITORY}/git/commits/{main_sha}"
        pr_path = f"/repos/{REPOSITORY}/pulls/430"
        release_pr = {
            "number": 430,
            "state": "closed",
            "merged_at": "2026-08-02T19:57:49Z",
            "merge_commit_sha": main_sha,
            "base": {"ref": "main"},
            "head": {"ref": PROMOTION.RELEASE_BRANCH, "sha": head_sha},
        }
        api = FakeApi(
            {
                (association_path, (("per_page", 100),)): [],
                (commit_path, ()): {
                    "message": "chore(main): release 0.15.4 (#430)\n\nrelease body"
                },
                (pr_path, ()): release_pr,
            }
        )
        proof = mock.Mock(verified=True, pr_head_sha=head_sha)

        with mock.patch.object(
            PROMOTION.MERGE_PROOF,
            "verify_release_merge",
            return_value=proof,
        ) as verify:
            resolved, resolved_head = PROMOTION._release_context(
                api, REPOSITORY, main_sha
            )

        self.assertIs(resolved, proof)
        self.assertEqual(resolved_head, head_sha)
        verify.assert_called_once_with(
            api,
            REPOSITORY,
            event_name="push",
            ref="refs/heads/main",
            sha=main_sha,
            associated_pulls=[release_pr],
            pr_api=api,
        )

    def test_release_context_does_not_delay_an_ordinary_main_commit(self) -> None:
        main_sha = "2" * 40
        association_path = f"/repos/{REPOSITORY}/commits/{main_sha}/pulls"
        commit_path = f"/repos/{REPOSITORY}/git/commits/{main_sha}"
        api = FakeApi(
            {
                (association_path, (("per_page", 100),)): [],
                (commit_path, ()): {"message": "ci: ordinary main change"},
            }
        )

        resolved, resolved_head = PROMOTION._release_context(api, REPOSITORY, main_sha)

        self.assertIsNone(resolved)
        self.assertIsNone(resolved_head)
        self.assertEqual(
            api.calls,
            [
                (association_path, (("per_page", 100),)),
                (commit_path, ()),
            ],
        )

    def test_release_context_fails_closed_on_a_spoofed_release_title(self) -> None:
        main_sha = "2" * 40
        association_path = f"/repos/{REPOSITORY}/commits/{main_sha}/pulls"
        commit_path = f"/repos/{REPOSITORY}/git/commits/{main_sha}"
        pr_path = f"/repos/{REPOSITORY}/pulls/430"
        api = FakeApi(
            {
                (association_path, (("per_page", 100),)): [],
                (commit_path, ()): {
                    "message": "chore(main): release 0.15.4 (#430)"
                },
                (pr_path, ()): {
                    "number": 430,
                    "base": {"ref": "other"},
                    "head": {"ref": PROMOTION.RELEASE_BRANCH},
                },
            }
        )

        with self.assertRaisesRegex(
            PROMOTION.PromotionError, "does not resolve to the release-please PR"
        ):
            PROMOTION._release_context(api, REPOSITORY, main_sha)

    def test_release_context_rejects_an_old_release_pr_from_the_title(self) -> None:
        main_sha = "2" * 40
        association_path = f"/repos/{REPOSITORY}/commits/{main_sha}/pulls"
        commit_path = f"/repos/{REPOSITORY}/git/commits/{main_sha}"
        pr_path = f"/repos/{REPOSITORY}/pulls/430"
        api = FakeApi(
            {
                (association_path, (("per_page", 100),)): [],
                (commit_path, ()): {
                    "message": "chore(main): release 0.15.4 (#430)"
                },
                (pr_path, ()): {
                    "number": 430,
                    "state": "closed",
                    "merged_at": "2026-08-01T00:00:00Z",
                    "merge_commit_sha": "4" * 40,
                    "base": {"ref": "main"},
                    "head": {"ref": PROMOTION.RELEASE_BRANCH, "sha": "3" * 40},
                },
            }
        )

        with self.assertRaisesRegex(
            PROMOTION.PromotionError,
            "release merge proof failed",
        ):
            PROMOTION._release_context(api, REPOSITORY, main_sha)

    def test_conclusion_check_suite_maps_to_exact_ci_run(self) -> None:
        path = f"/repos/{REPOSITORY}/actions/workflows/{PROMOTION.CI_WORKFLOW_FILE}/runs"
        params = {
            "event": "pull_request",
            "head_sha": HEAD_SHA,
            "status": "success",
            "per_page": 100,
        }
        exact = {
            "id": 7,
            "run_attempt": 2,
            "workflow_id": 9,
            "check_suite_id": 11,
            "path": PROMOTION.CI_WORKFLOW_PATH,
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "head_branch": PROMOTION.RELEASE_BRANCH,
            "head_sha": HEAD_SHA,
            "repository": {"id": 2, "full_name": REPOSITORY},
            "head_repository": {
                "id": 2,
                "full_name": REPOSITORY,
                "fork": False,
            },
        }
        wrong_suite = {**exact, "id": 8, "check_suite_id": 12}
        api = FakeApi(
            {
                (path, tuple(sorted(params.items()))): {
                    "workflow_runs": [wrong_suite, exact]
                }
            }
        )

        resolved = PROMOTION._source_run(
            api, REPOSITORY, 2, HEAD_SHA, check_suite_id=11
        )

        self.assertEqual(resolved["id"], 7)

    def test_source_run_never_falls_back_to_workflow_name(self) -> None:
        path = f"/repos/{REPOSITORY}/actions/workflows/{PROMOTION.CI_WORKFLOW_FILE}/runs"
        params = {
            "event": "pull_request",
            "head_sha": HEAD_SHA,
            "status": "success",
            "per_page": 100,
        }
        api = FakeApi(
            {
                (path, tuple(sorted(params.items()))): {
                    "workflow_runs": [
                        {
                            "id": 7,
                            "run_attempt": 1,
                            "workflow_id": 9,
                            "check_suite_id": 99,
                            "name": "CI",
                        }
                    ]
                }
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "maps to 0"):
            PROMOTION._source_run(api, REPOSITORY, 2, HEAD_SHA, check_suite_id=11)

    def test_main_plan_binds_exact_run_and_receipt_artifact(self) -> None:
        result = PROMOTION.GateResult(
            "validated",
            "ok",
            receipt={"validated_assets_artifact": {"id": 50}},
            receipt_artifact={
                "id": 40,
                "name": "validated-release-receipt-7-2",
                "digest": "sha256:" + "a" * 64,
                "size_in_bytes": 123,
            },
            main_sha="2" * 40,
            main_tree_sha="3" * 40,
            main_run_id=100,
            main_run_attempt=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            PROMOTION.write_plan(path, result)
            plan = PROMOTION.read_plan(path)
        self.assertEqual(plan["main_run"]["run_id"], 100)
        self.assertEqual(plan["receipt_artifact"]["id"], 40)

    def test_semantic_reruns_ignore_wrapper_identity_but_not_asset_hashes(self) -> None:
        receipt = {
            "assets": [{"name": "a", "sha256": "1" * 64}],
            "check_suite_id": 1,
            "head_sha": "1" * 40,
            "paths": ["version.txt"],
            "pr_number": 4,
            "repository": REPOSITORY,
            "repository_id": 2,
            "run_attempt": 1,
            "run_id": 3,
            "tree_sha": "2" * 40,
            "version": "0.15.4",
            "workflow_id": 5,
        }
        rerun = {**receipt, "observer": {"run_id": 9}}
        self.assertEqual(
            PROMOTION._receipt_semantics(receipt),
            PROMOTION._receipt_semantics(rerun),
        )
        conflicting = {**receipt, "assets": [{"name": "a", "sha256": "2" * 64}]}
        self.assertNotEqual(
            PROMOTION._receipt_semantics(receipt),
            PROMOTION._receipt_semantics(conflicting),
        )

    def test_newest_terminal_observer_attempt_cannot_fall_back_to_old_success(self) -> None:
        path = (
            f"/repos/{REPOSITORY}/actions/workflows/"
            f"{PROMOTION.OBSERVER_WORKFLOW_FILE}/runs"
        )
        params = {"event": "workflow_run", "per_page": 100}
        base = {
            "id": 70,
            "display_title": "release-candidate-observer source 7/2",
            "path": PROMOTION.OBSERVER_WORKFLOW_PATH,
            "event": "workflow_run",
            "status": "completed",
            "repository": {"id": 2, "full_name": REPOSITORY},
        }
        api = FakeApi(
            {
                (path, tuple(sorted(params.items()))): {
                    "workflow_runs": [
                        {**base, "run_attempt": 1, "conclusion": "success"},
                        {**base, "run_attempt": 2, "conclusion": "failure"},
                    ]
                }
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "concluded 'failure'"):
            PROMOTION._latest_observer_run(api, REPOSITORY, 2, 7, 2)

    def test_receipt_locator_has_a_hard_candidate_cap(self) -> None:
        name = "validated-release-receipt-7-2"
        path = f"/repos/{REPOSITORY}/actions/artifacts"
        params = {"name": name, "per_page": 100, "page": 1}
        api = FakeApi(
            {
                (path, tuple(sorted(params.items()))): {
                    "total_count": PROMOTION.MAX_RECEIPT_CANDIDATES + 1,
                    "artifacts": [],
                }
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "candidate count"):
            PROMOTION._artifact_pages(api, REPOSITORY, name)

    def test_receipt_locator_rejects_incomplete_or_unstable_pagination(self) -> None:
        name = "validated-release-receipt-7-2"
        path = f"/repos/{REPOSITORY}/actions/artifacts"
        first = {"name": name, "id": 1}
        incomplete = FakeApi(
            {
                (
                    path,
                    (("name", name), ("page", 1), ("per_page", 100)),
                ): {"total_count": 2, "artifacts": [first]},
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "does not match pages"):
            PROMOTION._artifact_pages(incomplete, REPOSITORY, name)

        full_page = [dict(first, id=index + 1) for index in range(100)]
        unstable = FakeApi(
            {
                (
                    path,
                    (("name", name), ("page", 1), ("per_page", 100)),
                ): {"total_count": 20, "artifacts": full_page},
                (
                    path,
                    (("name", name), ("page", 2), ("per_page", 100)),
                ): {"total_count": 19, "artifacts": []},
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "changed during pagination"):
            PROMOTION._artifact_pages(unstable, REPOSITORY, name)

    def test_consistent_old_attempt_is_ignored_for_latest_successful_rerun(self) -> None:
        source = {
            "id": 7,
            "run_attempt": 2,
            "workflow_id": 9,
            "check_suite_id": 11,
        }
        base_receipt = {
            "assets": [{"name": "a", "sha256": "1" * 64}],
            "check_suite_id": 11,
            "head_sha": HEAD_SHA,
            "paths": ["version.txt"],
            "pr_number": 4,
            "repository": REPOSITORY,
            "repository_id": 2,
            "run_attempt": 2,
            "run_id": 7,
            "tree_sha": "2" * 40,
            "version": "0.15.4",
            "workflow_id": 9,
        }
        old = {**base_receipt, "observer": {"run_id": 70, "run_attempt": 1}}
        newest = {**base_receipt, "observer": {"run_id": 70, "run_attempt": 2}}
        artifacts = [
            {
                "id": 1,
                "name": "validated-release-receipt-7-2",
                "expired": False,
                "workflow_run": {"id": 70},
            },
            {
                "id": 2,
                "name": "validated-release-receipt-7-2",
                "expired": False,
                "workflow_run": {"id": 70},
            },
        ]
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION,
            "_latest_observer_run",
            return_value={"id": 70, "run_attempt": 2},
        ), mock.patch.object(
            PROMOTION, "_artifact_pages", return_value=artifacts
        ), mock.patch.object(
            PROMOTION,
            "_download_receipt",
            side_effect=[(newest, "sha256:" + "b" * 64), (old, "sha256:" + "a" * 64)],
        ), mock.patch.object(PROMOTION, "_validate_observer_and_assets"):
            receipt, artifact = PROMOTION._find_receipt(
                mock.Mock(),
                REPOSITORY,
                2,
                source,
                4,
                HEAD_SHA,
                "2" * 40,
                Path(raw),
            )
        self.assertEqual(receipt["observer"]["run_attempt"], 2)
        self.assertEqual(artifact["id"], 2)

    def test_consume_main_receipt_binds_exact_successful_main_run(self) -> None:
        receipt = {
            "assets": [{"name": "a", "sha256": "1" * 64}],
            "check_suite_id": 11,
            "head_sha": HEAD_SHA,
            "paths": ["version.txt"],
            "pr_number": 4,
            "repository": REPOSITORY,
            "repository_id": 2,
            "run_attempt": 2,
            "run_id": 7,
            "tree_sha": "2" * 40,
            "version": "0.15.4",
            "workflow_id": 9,
        }
        artifact = {
            "digest": "sha256:" + "a" * 64,
            "id": 40,
            "name": "validated-release-receipt-7-2",
            "size_in_bytes": 123,
        }
        result = PROMOTION.GateResult(
            "validated",
            "ok",
            receipt=receipt,
            receipt_artifact=artifact,
            main_sha="3" * 40,
            main_tree_sha="4" * 40,
        )
        main_run = {"id": 100, "run_attempt": 3}
        plan = {
            "schema_version": 1,
            "main_sha": "3" * 40,
            "main_tree_sha": "4" * 40,
            "main_run": {
                "run_id": 100,
                "run_attempt": 3,
                "workflow_path": PROMOTION.CI_WORKFLOW_PATH,
            },
            "receipt": receipt,
            "receipt_artifact": artifact,
        }
        api = FakeApi(
            {
                (f"/repos/{REPOSITORY}", ()): {
                    "id": 2,
                    "full_name": REPOSITORY,
                }
            }
        )
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION, "verify_main", return_value=result
        ), mock.patch.object(
            PROMOTION, "_main_ci_run", return_value=main_run
        ), mock.patch.object(PROMOTION, "_download_main_plan", return_value=plan):
            consumed, closed_plan = PROMOTION.consume_main_receipt(
                api,
                REPOSITORY,
                "3" * 40,
                Path(raw),
                main_run_id=100,
                main_run_attempt=3,
            )
        self.assertEqual(consumed.state, "validated")
        self.assertEqual(consumed.main_run_id, 100)
        self.assertEqual(closed_plan, plan)

    def test_consume_main_receipt_rejects_plan_drift(self) -> None:
        result = PROMOTION.GateResult(
            "validated",
            "ok",
            receipt={
                "assets": [],
                "check_suite_id": 1,
                "head_sha": HEAD_SHA,
                "paths": ["version.txt"],
                "pr_number": 4,
                "repository": REPOSITORY,
                "repository_id": 2,
                "run_attempt": 1,
                "run_id": 7,
                "tree_sha": "2" * 40,
                "version": "0.15.4",
                "workflow_id": 9,
            },
            receipt_artifact={
                "digest": "sha256:" + "a" * 64,
                "id": 40,
                "name": "receipt",
                "size_in_bytes": 123,
            },
            main_sha="3" * 40,
            main_tree_sha="4" * 40,
        )
        drifted = {
            "main_sha": "5" * 40,
            "main_tree_sha": "4" * 40,
            "main_run": {"run_id": 100, "run_attempt": 3},
            "receipt": result.receipt,
            "receipt_artifact": result.receipt_artifact,
        }
        api = FakeApi(
            {
                (f"/repos/{REPOSITORY}", ()): {
                    "id": 2,
                    "full_name": REPOSITORY,
                }
            }
        )
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION, "verify_main", return_value=result
        ), mock.patch.object(
            PROMOTION, "_main_ci_run", return_value={"id": 100, "run_attempt": 3}
        ), mock.patch.object(
            PROMOTION, "_download_main_plan", return_value=drifted
        ):
            consumed, closed_plan = PROMOTION.consume_main_receipt(
                api, REPOSITORY, "3" * 40, Path(raw)
            )
        self.assertEqual(consumed.state, "blocked")
        self.assertIsNone(closed_plan)

    def test_partial_rerun_accepts_attempt_one_receipt_after_attempt_two_success(self) -> None:
        main_sha = "3" * 40
        receipt = {
            "assets": [{"name": "a", "sha256": "1" * 64}],
            "check_suite_id": 11,
            "head_sha": HEAD_SHA,
            "paths": ["version.txt"],
            "pr_number": 4,
            "repository": REPOSITORY,
            "repository_id": 2,
            "run_attempt": 2,
            "run_id": 7,
            "tree_sha": "2" * 40,
            "version": "0.15.4",
            "workflow_id": 9,
        }
        receipt_artifact = {
            "digest": "sha256:" + "a" * 64,
            "id": 40,
            "name": "validated-release-receipt-7-2",
            "size_in_bytes": 123,
        }
        result = PROMOTION.GateResult(
            "validated",
            "ok",
            receipt=receipt,
            receipt_artifact=receipt_artifact,
            main_sha=main_sha,
            main_tree_sha="4" * 40,
        )
        plan = {
            "schema_version": 1,
            "main_sha": main_sha,
            "main_tree_sha": "4" * 40,
            "main_run": {
                "run_id": 100,
                "run_attempt": 1,
                "workflow_path": PROMOTION.CI_WORKFLOW_PATH,
            },
            "receipt": receipt,
            "receipt_artifact": receipt_artifact,
        }
        main_receipt = {
            "id": 60,
            "name": "main-release-promotion-receipt-100-1",
            "workflow_run": {"id": 100},
        }
        artifacts_path = f"/repos/{REPOSITORY}/actions/runs/100/artifacts"
        artifacts_params = {"per_page": 100, "page": 1}
        jobs_path = f"/repos/{REPOSITORY}/actions/runs/100/attempts/1/jobs"
        jobs_params = {"per_page": 100, "page": 1}
        api = FakeApi(
            {
                (f"/repos/{REPOSITORY}", ()): {
                    "id": 2,
                    "full_name": REPOSITORY,
                },
                (artifacts_path, tuple(sorted(artifacts_params.items()))): {
                    "total_count": 1,
                    "artifacts": [main_receipt],
                },
                (jobs_path, tuple(sorted(jobs_params.items()))): {
                    "total_count": 1,
                    "jobs": [
                        {
                            "id": 70,
                            "name": "detect-changes",
                            "run_id": 100,
                            "run_attempt": 1,
                            "head_sha": main_sha,
                            "status": "completed",
                            "conclusion": "success",
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION, "verify_main", return_value=result
        ), mock.patch.object(
            PROMOTION,
            "_main_ci_run",
            return_value={"id": 100, "run_attempt": 2, "head_sha": main_sha},
        ), mock.patch.object(
            PROMOTION, "_download_main_plan_artifact", return_value=plan
        ):
            consumed, closed_plan = PROMOTION.consume_main_receipt(
                api, REPOSITORY, main_sha, Path(raw)
            )
        self.assertEqual(consumed.state, "validated")
        self.assertEqual(consumed.main_run_attempt, 2)
        self.assertEqual(closed_plan, plan)

    def test_main_receipt_rejects_failed_detect_changes_origin(self) -> None:
        main_sha = "3" * 40
        artifact = {
            "id": 60,
            "name": "main-release-promotion-receipt-100-1",
            "workflow_run": {"id": 100},
        }
        plan = {
            "schema_version": 1,
            "main_sha": main_sha,
            "main_tree_sha": "4" * 40,
            "main_run": {
                "run_id": 100,
                "run_attempt": 1,
                "workflow_path": PROMOTION.CI_WORKFLOW_PATH,
            },
            "receipt": {},
            "receipt_artifact": {},
        }
        api = FakeApi(
            {
                (
                    f"/repos/{REPOSITORY}/actions/runs/100/attempts/1/jobs",
                    (("page", 1), ("per_page", 100)),
                ): {
                    "total_count": 1,
                    "jobs": [
                        {
                            "id": 70,
                            "name": "detect-changes",
                            "run_id": 100,
                            "run_attempt": 1,
                            "head_sha": main_sha,
                            "status": "completed",
                            "conclusion": "failure",
                        }
                    ],
                }
            }
        )
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION, "_main_receipt_artifacts", return_value=[artifact]
        ), mock.patch.object(
            PROMOTION, "_download_main_plan_artifact", return_value=plan
        ):
            with self.assertRaisesRegex(PROMOTION.PromotionError, "not an exact success"):
                PROMOTION._download_main_plan(
                    api,
                    REPOSITORY,
                    2,
                    {"id": 100, "run_attempt": 2, "head_sha": main_sha},
                    Path(raw),
                )

    def test_main_receipt_reruns_must_have_consistent_semantics(self) -> None:
        main_sha = "3" * 40
        receipt = {
            "assets": [{"name": "a", "sha256": "1" * 64}],
            "check_suite_id": 11,
            "head_sha": HEAD_SHA,
            "paths": ["version.txt"],
            "pr_number": 4,
            "repository": REPOSITORY,
            "repository_id": 2,
            "run_attempt": 1,
            "run_id": 7,
            "tree_sha": "2" * 40,
            "version": "0.15.4",
            "workflow_id": 9,
        }
        artifacts = [
            {
                "id": attempt,
                "name": f"main-release-promotion-receipt-100-{attempt}",
                "workflow_run": {"id": 100},
            }
            for attempt in (1, 2)
        ]
        plans = [
            {
                "schema_version": 1,
                "main_sha": main_sha,
                "main_tree_sha": digit * 40,
                "main_run": {
                    "run_id": 100,
                    "run_attempt": attempt,
                    "workflow_path": PROMOTION.CI_WORKFLOW_PATH,
                },
                "receipt": {**receipt, "observer": {"run_id": attempt}},
                "receipt_artifact": {"id": attempt},
            }
            for attempt, digit in ((1, "4"), (2, "5"))
        ]
        responses: dict[tuple[str, tuple], object] = {}
        for attempt in (1, 2):
            responses[
                (
                    f"/repos/{REPOSITORY}/actions/runs/100/attempts/{attempt}/jobs",
                    (("page", 1), ("per_page", 100)),
                )
            ] = {
                "total_count": 1,
                "jobs": [
                    {
                        "id": 70 + attempt,
                        "name": "detect-changes",
                        "run_id": 100,
                        "run_attempt": attempt,
                        "head_sha": main_sha,
                        "status": "completed",
                        "conclusion": "success",
                    }
                ],
            }
        api = FakeApi(responses)
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION, "_main_receipt_artifacts", return_value=artifacts
        ), mock.patch.object(
            PROMOTION, "_download_main_plan_artifact", side_effect=plans
        ):
            with self.assertRaisesRegex(PROMOTION.PromotionError, "conflicting semantics"):
                PROMOTION._download_main_plan(
                    api,
                    REPOSITORY,
                    2,
                    {"id": 100, "run_attempt": 2, "head_sha": main_sha},
                    Path(raw),
                )

    def test_main_receipt_selects_newest_consistent_origin(self) -> None:
        main_sha = "3" * 40
        receipt = {
            "assets": [{"name": "a", "sha256": "1" * 64}],
            "check_suite_id": 11,
            "head_sha": HEAD_SHA,
            "paths": ["version.txt"],
            "pr_number": 4,
            "repository": REPOSITORY,
            "repository_id": 2,
            "run_attempt": 1,
            "run_id": 7,
            "tree_sha": "2" * 40,
            "version": "0.15.4",
            "workflow_id": 9,
        }
        artifacts = [
            {
                "id": attempt,
                "name": f"main-release-promotion-receipt-100-{attempt}",
                "workflow_run": {"id": 100},
            }
            for attempt in (1, 2)
        ]
        plans = [
            {
                "schema_version": 1,
                "main_sha": main_sha,
                "main_tree_sha": "4" * 40,
                "main_run": {
                    "run_id": 100,
                    "run_attempt": attempt,
                    "workflow_path": PROMOTION.CI_WORKFLOW_PATH,
                },
                "receipt": {**receipt, "observer": {"run_id": attempt}},
                "receipt_artifact": {"id": attempt},
            }
            for attempt in (1, 2)
        ]
        responses: dict[tuple[str, tuple], object] = {}
        for attempt in (1, 2):
            responses[
                (
                    f"/repos/{REPOSITORY}/actions/runs/100/attempts/{attempt}/jobs",
                    (("page", 1), ("per_page", 100)),
                )
            ] = {
                "total_count": 1,
                "jobs": [
                    {
                        "id": 70 + attempt,
                        "name": "detect-changes",
                        "run_id": 100,
                        "run_attempt": attempt,
                        "head_sha": main_sha,
                        "status": "completed",
                        "conclusion": "success",
                    }
                ],
            }
        api = FakeApi(responses)
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION, "_main_receipt_artifacts", return_value=artifacts
        ), mock.patch.object(
            PROMOTION, "_download_main_plan_artifact", side_effect=plans
        ):
            selected = PROMOTION._download_main_plan(
                api,
                REPOSITORY,
                2,
                {"id": 100, "run_attempt": 2, "head_sha": main_sha},
                Path(raw),
            )
        self.assertEqual(selected["main_run"]["run_attempt"], 2)

    def test_main_receipt_future_attempt_and_malformed_name_fail_closed(self) -> None:
        main_sha = "3" * 40
        future_artifact = {
            "id": 60,
            "name": "main-release-promotion-receipt-100-3",
            "workflow_run": {"id": 100},
        }
        future_plan = {
            "schema_version": 1,
            "main_sha": main_sha,
            "main_tree_sha": "4" * 40,
            "main_run": {
                "run_id": 100,
                "run_attempt": 3,
                "workflow_path": PROMOTION.CI_WORKFLOW_PATH,
            },
            "receipt": {},
            "receipt_artifact": {},
        }
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(
            PROMOTION, "_main_receipt_artifacts", return_value=[future_artifact]
        ), mock.patch.object(
            PROMOTION, "_download_main_plan_artifact", return_value=future_plan
        ):
            with self.assertRaisesRegex(PROMOTION.PromotionError, "future run attempt"):
                PROMOTION._download_main_plan(
                    mock.Mock(),
                    REPOSITORY,
                    2,
                    {"id": 100, "run_attempt": 2, "head_sha": main_sha},
                    Path(raw),
                )

        artifacts_path = f"/repos/{REPOSITORY}/actions/runs/100/artifacts"
        malformed_api = FakeApi(
            {
                (
                    artifacts_path,
                    (("page", 1), ("per_page", 100)),
                ): {
                    "total_count": 1,
                    "artifacts": [
                        {
                            "id": 61,
                            "name": "main-release-promotion-receipt-100-bogus",
                        }
                    ],
                }
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "name is malformed"):
            PROMOTION._main_receipt_artifacts(malformed_api, REPOSITORY, 100)

    def test_main_receipt_and_job_pagination_fail_closed_on_short_pages(self) -> None:
        artifacts_path = f"/repos/{REPOSITORY}/actions/runs/100/artifacts"
        jobs_path = f"/repos/{REPOSITORY}/actions/runs/100/attempts/1/jobs"
        params = (("page", 1), ("per_page", 100))
        api = FakeApi(
            {
                (artifacts_path, params): {
                    "total_count": 2,
                    "artifacts": [
                        {
                            "id": 1,
                            "name": "main-release-promotion-receipt-100-1",
                        }
                    ],
                },
                (jobs_path, params): {
                    "total_count": 2,
                    "jobs": [
                        {
                            "id": 2,
                            "name": "detect-changes",
                            "run_id": 100,
                            "run_attempt": 1,
                            "head_sha": "3" * 40,
                            "status": "completed",
                            "conclusion": "success",
                        }
                    ],
                },
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "does not match pages"):
            PROMOTION._main_receipt_artifacts(api, REPOSITORY, 100)
        with self.assertRaisesRegex(PROMOTION.PromotionError, "does not match pages"):
            PROMOTION._detect_changes_job(api, REPOSITORY, 100, 1, "3" * 40)

    def test_main_ci_run_requires_exact_attempt_and_control_plane(self) -> None:
        run = {
            "id": 100,
            "run_attempt": 3,
            "workflow_id": 9,
            "path": PROMOTION.CI_WORKFLOW_PATH,
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "head_branch": "main",
            "head_sha": "3" * 40,
            "repository": {"id": 2, "full_name": REPOSITORY},
            "head_repository": {"id": 2, "full_name": REPOSITORY},
        }
        api = FakeApi(
            {
                (f"/repos/{REPOSITORY}/actions/runs/100", ()): run,
                (f"/repos/{REPOSITORY}/actions/workflows/9", ()): {
                    "id": 9,
                    "path": PROMOTION.CI_WORKFLOW_PATH,
                    "state": "active",
                },
            }
        )
        resolved = PROMOTION._main_ci_run(
            api, REPOSITORY, 2, "3" * 40, 100, 3
        )
        self.assertEqual(resolved["id"], 100)
        with self.assertRaisesRegex(PROMOTION.PromotionError, "attempt differs"):
            PROMOTION._main_ci_run(api, REPOSITORY, 2, "3" * 40, 100, 2)

        list_path = (
            f"/repos/{REPOSITORY}/actions/workflows/"
            f"{PROMOTION.CI_WORKFLOW_FILE}/runs"
        )
        list_params = {"event": "push", "head_sha": "3" * 40, "per_page": 100}
        discovery_api = FakeApi(
            {
                (list_path, tuple(sorted(list_params.items()))): {
                    "workflow_runs": [run]
                },
                (f"/repos/{REPOSITORY}/actions/runs/100", ()): run,
                (f"/repos/{REPOSITORY}/actions/workflows/9", ()): {
                    "id": 9,
                    "path": PROMOTION.CI_WORKFLOW_PATH,
                    "state": "active",
                },
            }
        )
        discovered = PROMOTION._main_ci_run(
            discovery_api, REPOSITORY, 2, "3" * 40, None, None
        )
        self.assertEqual(discovered["run_attempt"], 3)

    def test_verify_main_ci_waits_for_the_exact_run_then_succeeds(self) -> None:
        main_sha = "3" * 40
        list_path = (
            f"/repos/{REPOSITORY}/actions/workflows/"
            f"{PROMOTION.CI_WORKFLOW_FILE}/runs"
        )
        run_path = f"/repos/{REPOSITORY}/actions/runs/100"
        workflow_path = f"/repos/{REPOSITORY}/actions/workflows/9"
        run = {
            "id": 100,
            "run_attempt": 2,
            "workflow_id": 9,
            "path": PROMOTION.CI_WORKFLOW_PATH,
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "head_branch": "main",
            "head_sha": main_sha,
            "repository": {"id": 2, "full_name": REPOSITORY},
            "head_repository": {
                "id": 2,
                "full_name": REPOSITORY,
                "fork": False,
            },
        }
        pending = {**run, "status": "in_progress", "conclusion": None}
        states = [pending, run]
        inventories = [
            {"total_count": 0, "workflow_runs": []},
            {"total_count": 1, "workflow_runs": [{"id": 100}]},
            {"total_count": 1, "workflow_runs": [{"id": 100}]},
        ]

        def get_json(path, *, params=None):
            if path == f"/repos/{REPOSITORY}":
                return {"id": 2, "full_name": REPOSITORY}
            if path == list_path:
                self.assertEqual(
                    params,
                    {
                        "event": "push",
                        "head_sha": main_sha,
                        "per_page": 100,
                    },
                )
                return inventories.pop(0)
            if path == run_path:
                return states.pop(0)
            if path == workflow_path:
                return {
                    "id": 9,
                    "path": PROMOTION.CI_WORKFLOW_PATH,
                    "state": "active",
                }
            self.fail(f"unexpected API path: {path}")

        api = mock.Mock()
        api.get_json.side_effect = get_json
        clock = [0.0]
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        resolved = PROMOTION.verify_main_ci(
            api,
            REPOSITORY,
            main_sha,
            wait_seconds=45,
            sleep=sleep,
            monotonic=lambda: clock[0],
        )

        self.assertEqual(resolved["id"], 100)
        self.assertEqual(resolved["run_attempt"], 2)
        self.assertEqual(sleeps, [15, 15])

    def test_verify_main_ci_rejects_ambiguity_and_terminal_failure(self) -> None:
        main_sha = "3" * 40
        list_path = (
            f"/repos/{REPOSITORY}/actions/workflows/"
            f"{PROMOTION.CI_WORKFLOW_FILE}/runs"
        )
        params = {"event": "push", "head_sha": main_sha, "per_page": 100}
        ambiguous = FakeApi(
            {
                (f"/repos/{REPOSITORY}", ()): {
                    "id": 2,
                    "full_name": REPOSITORY,
                },
                (list_path, tuple(sorted(params.items()))): {
                    "total_count": 2,
                    "workflow_runs": [{"id": 100}, {"id": 101}],
                },
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "exactly one"):
            PROMOTION.verify_main_ci(ambiguous, REPOSITORY, main_sha)

        failed_run = {
            "id": 100,
            "run_attempt": 1,
            "workflow_id": 9,
            "path": PROMOTION.CI_WORKFLOW_PATH,
            "event": "push",
            "status": "completed",
            "conclusion": "failure",
            "head_branch": "main",
            "head_sha": main_sha,
            "repository": {"id": 2, "full_name": REPOSITORY},
            "head_repository": {
                "id": 2,
                "full_name": REPOSITORY,
                "fork": False,
            },
        }
        terminal = FakeApi(
            {
                (f"/repos/{REPOSITORY}", ()): {
                    "id": 2,
                    "full_name": REPOSITORY,
                },
                (list_path, tuple(sorted(params.items()))): {
                    "total_count": 1,
                    "workflow_runs": [{"id": 100}],
                },
                (f"/repos/{REPOSITORY}/actions/runs/100", ()): failed_run,
                (f"/repos/{REPOSITORY}/actions/workflows/9", ()): {
                    "id": 9,
                    "path": PROMOTION.CI_WORKFLOW_PATH,
                    "state": "active",
                },
            }
        )
        with self.assertRaisesRegex(PROMOTION.PromotionError, "conclusion 'failure'"):
            PROMOTION.verify_main_ci(terminal, REPOSITORY, main_sha, wait_seconds=30)

    def test_verify_main_ci_rejects_repository_head_and_workflow_drift(self) -> None:
        main_sha = "3" * 40
        list_path = (
            f"/repos/{REPOSITORY}/actions/workflows/"
            f"{PROMOTION.CI_WORKFLOW_FILE}/runs"
        )
        params = {"event": "push", "head_sha": main_sha, "per_page": 100}
        run = {
            "id": 100,
            "run_attempt": 1,
            "workflow_id": 9,
            "path": PROMOTION.CI_WORKFLOW_PATH,
            "event": "push",
            "status": "completed",
            "conclusion": "success",
            "head_branch": "main",
            "head_sha": main_sha,
            "repository": {"id": 2, "full_name": REPOSITORY},
            "head_repository": {
                "id": 2,
                "full_name": REPOSITORY,
                "fork": False,
            },
        }
        cases = [
            (
                "repository",
                {**run, "repository": {"id": 3, "full_name": REPOSITORY}},
                {"id": 9, "path": PROMOTION.CI_WORKFLOW_PATH, "state": "active"},
                "main CI repository differs",
            ),
            (
                "head repository",
                {**run, "head_repository": {"id": 3, "full_name": REPOSITORY, "fork": False}},
                {"id": 9, "path": PROMOTION.CI_WORKFLOW_PATH, "state": "active"},
                "main CI head repository differs",
            ),
            (
                "workflow",
                run,
                {"id": 9, "path": PROMOTION.CI_WORKFLOW_PATH, "state": "disabled_manually"},
                "control-plane identity mismatch",
            ),
        ]
        for label, candidate, workflow, expected in cases:
            with self.subTest(label=label):
                api = FakeApi(
                    {
                        (f"/repos/{REPOSITORY}", ()): {
                            "id": 2,
                            "full_name": REPOSITORY,
                        },
                        (list_path, tuple(sorted(params.items()))): {
                            "total_count": 1,
                            "workflow_runs": [{"id": 100}],
                        },
                        (f"/repos/{REPOSITORY}/actions/runs/100", ()): candidate,
                        (f"/repos/{REPOSITORY}/actions/workflows/9", ()): workflow,
                    }
                )
                with self.assertRaisesRegex(PROMOTION.PromotionError, expected):
                    PROMOTION.verify_main_ci(api, REPOSITORY, main_sha)

    def test_verify_main_ci_cli_prints_validated_run_identity(self) -> None:
        run = {
            "id": 100,
            "run_attempt": 3,
            "head_sha": "3" * 40,
        }
        with mock.patch.object(PROMOTION.VALIDATOR, "GitHubApi"), mock.patch.object(
            PROMOTION, "verify_main_ci", return_value=run
        ), mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = PROMOTION._main(
                [
                    "verify-main-ci",
                    "--repository",
                    REPOSITORY,
                    "--sha",
                    "3" * 40,
                    "--wait-seconds",
                    "1200",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "head_sha": "3" * 40,
                "run_attempt": 3,
                "run_id": 100,
                "status": "validated",
                "workflow_path": PROMOTION.CI_WORKFLOW_PATH,
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
