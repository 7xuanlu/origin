#!/usr/bin/env python3
"""Adversarial tests for the privileged-boundary candidate observer."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import io
import json
import shutil
import tarfile
import tempfile
import unittest
import urllib.request
import zipfile
from pathlib import Path

from release_targets import release_assets
import release_archive


def load_script(name: str, module_name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_script("validate-release-candidate.py", "validate_release_candidate")
PACKAGE = load_script("package-release-artifacts.py", "package_release_for_validator")
MANIFEST = load_script("release-candidate-manifest.py", "manifest_for_validator")


HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
TREE_SHA = "c" * 40
BASE_TREE_SHA = "d" * 40
HEAD_TREE_SHA = "e" * 40


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def canonical_gzip(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    with source.open("rb") as payload, destination.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as archive:
            archive.write(payload.read())


class FakeContentApi:
    def __init__(
        self,
        old: dict[str, str],
        new: dict[str, str],
        *,
        head_mode_overrides: dict[str, str] | None = None,
    ) -> None:
        self.old = old
        self.new = new
        self.head_mode_overrides = head_mode_overrides or {}

    def tree(self, contents: dict[str, str], *, head: bool) -> list[dict]:
        records = []
        for name in sorted(VALIDATOR.REQUIRED_RELEASE_PATHS):
            raw = contents[name].encode()
            default_mode = (
                "100755"
                if name == "plugin-codex/bin/wenlan-mcp-runner.sh"
                else "100644"
            )
            records.append(
                {
                    "path": name,
                    "mode": self.head_mode_overrides.get(name, default_mode)
                    if head
                    else default_mode,
                    "type": "blob",
                    "sha": git_blob_sha(raw),
                    "size": len(raw),
                }
            )
        return records

    def get_json(self, path: str, *, params=None):
        if path.endswith("/files"):
            if params["page"] > 1:
                return []
            return [
                {"filename": name, "status": "modified"}
                for name in sorted(VALIDATOR.REQUIRED_RELEASE_PATHS)
            ]
        if path.endswith(f"/git/commits/{BASE_SHA}"):
            return {"tree": {"sha": BASE_TREE_SHA}}
        if path.endswith(f"/git/commits/{HEAD_SHA}"):
            return {"tree": {"sha": HEAD_TREE_SHA}}
        if path.endswith(f"/git/trees/{BASE_TREE_SHA}"):
            self.assert_recursive(params)
            return {
                "sha": BASE_TREE_SHA,
                "truncated": False,
                "tree": self.tree(self.old, head=False),
            }
        if path.endswith(f"/git/trees/{HEAD_TREE_SHA}"):
            self.assert_recursive(params)
            return {
                "sha": HEAD_TREE_SHA,
                "truncated": False,
                "tree": self.tree(self.new, head=True),
            }
        marker = "/contents/"
        if marker in path:
            encoded = path.split(marker, 1)[1]
            name = __import__("urllib.parse").parse.unquote(encoded)
            content = self.old[name] if params["ref"] == BASE_SHA else self.new[name]
            raw = content.encode()
            return {
                "type": "file",
                "encoding": "base64",
                "size": len(raw),
                "sha": git_blob_sha(raw),
                "content": base64.encodebytes(raw).decode(),
            }
        raise AssertionError(f"unexpected API read {path} {params}")

    @staticmethod
    def assert_recursive(params) -> None:
        if params != {"recursive": "1"}:
            raise AssertionError(f"tree read was not exact recursive lookup: {params}")

    def download(self, path: str, destination: Path, maximum: int):
        raise AssertionError("content test must not download")


class DownloadApi:
    def __init__(self, source: Path) -> None:
        self.source = source

    def get_json(self, path: str, *, params=None):
        raise AssertionError("download test must not query JSON")

    def download(self, path: str, destination: Path, maximum: int):
        shutil.copyfile(self.source, destination)
        raw = destination.read_bytes()
        if len(raw) > maximum:
            raise AssertionError("test wrapper unexpectedly exceeds maximum")
        return len(raw), hashlib.sha256(raw).hexdigest()


def release_contents() -> tuple[dict[str, str], dict[str, str]]:
    old_version = "0.15.3"
    new_version = "0.15.4"
    old = {
        path: f"value={old_version}\n" for path in VALIDATOR.REQUIRED_RELEASE_PATHS
    }
    new = {
        path: f"value={new_version}\n" for path in VALIDATOR.REQUIRED_RELEASE_PATHS
    }
    old["version.txt"] = old_version + "\n"
    new["version.txt"] = new_version + "\n"
    old["CHANGELOG.md"] = "# Changelog\n"
    new["CHANGELOG.md"] = f"# Changelog\n\n## [{new_version}] release\n"
    old[".release-please-manifest.json"] = json.dumps({".": old_version})
    new[".release-please-manifest.json"] = json.dumps({".": new_version})
    for path in (
        "crates/wenlan-cli/npm/package.json",
        "crates/wenlan-mcp/npm/package.json",
        "plugin/.claude-plugin/plugin.json",
    ):
        old[path] = json.dumps({"version": old_version})
        new[path] = json.dumps({"version": new_version})
    codex = "plugin-codex/.codex-plugin/plugin.json"
    old[codex] = json.dumps({"version": f"{old_version}+codex"})
    new[codex] = json.dumps({"version": f"{new_version}+codex"})
    old["Cargo.toml"] = f'version = "{old_version}"   # x-release-please-version\n'
    new["Cargo.toml"] = f'version = "{new_version}"   # x-release-please-version\n'
    old["release-please-config.json"] = json.dumps(
        {
            "packages": {
                ".": {"release-type": "simple", "versioning": "always-bump-patch"}
            },
            "$schema": "https://example.invalid/release-please.schema.json",
        }
    )
    return old, new


def candidate_pr() -> dict:
    return {
        "number": 42,
        "changed_files": len(VALIDATOR.REQUIRED_RELEASE_PATHS),
        "base": {"sha": BASE_SHA},
        "head": {"sha": HEAD_SHA},
    }


def candidate_pr_detail(*, state: str, merged: bool) -> dict:
    return {
        "number": 42,
        "state": state,
        "draft": False,
        "merged": merged,
        "merged_at": "2026-08-01T12:34:56Z" if merged else None,
        "merge_commit_sha": "f" * 40 if merged else None,
        "base": {
            "ref": "main",
            "repo": {"id": 2, "full_name": "7xuanlu/wenlan"},
        },
        "head": {
            "ref": VALIDATOR.RELEASE_BRANCH,
            "sha": HEAD_SHA,
            "repo": {
                "id": 2,
                "full_name": "7xuanlu/wenlan",
                "fork": False,
            },
        },
        "user": {"login": "7xuanlu"},
    }


class PrStateApi:
    def __init__(self, merge_tree_sha: str = TREE_SHA) -> None:
        self.merge_tree_sha = merge_tree_sha

    def get_json(self, path: str, *, params=None):
        if path.endswith("/issues/42/events"):
            return [
                {
                    "event": "merged",
                    "created_at": "2026-08-01T12:34:56Z",
                    "commit_id": "f" * 40,
                }
            ]
        if path.endswith(f"/git/commits/{HEAD_SHA}"):
            return {"tree": {"sha": TREE_SHA}}
        if path.endswith(f"/git/commits/{'f' * 40}"):
            return {"tree": {"sha": self.merge_tree_sha}}
        raise AssertionError(f"unexpected PR state API read {path}")

    def download(self, path: str, destination: Path, maximum: int):
        raise AssertionError("PR state test must not download")


class HappyPathApi(FakeContentApi):
    def __init__(
        self,
        old: dict[str, str],
        new: dict[str, str],
        wrappers: dict[int, Path],
        artifacts: list[dict],
        pr: dict,
        event_run: dict,
        jobs_by_attempt: dict[int, list[dict]] | None = None,
    ) -> None:
        super().__init__(old, new)
        self.wrappers = wrappers
        self.artifacts = artifacts
        self.pr = pr
        self.event_run = event_run
        self.jobs_by_attempt = jobs_by_attempt or {
            event_run["run_attempt"]: [
                {
                    "id": 100 + index,
                    "run_id": event_run["id"],
                    "head_sha": event_run["head_sha"],
                    "name": f"release-preflight ({entry['target']})",
                    "status": "completed",
                    "conclusion": "success",
                }
                for index, entry in enumerate(VALIDATOR.release_matrix()["include"])
            ]
        }

    def get_json(self, path: str, *, params=None):
        prefix = "/repos/7xuanlu/wenlan"
        repository = {
            "id": 2,
            "full_name": "7xuanlu/wenlan",
            "owner": {"login": "7xuanlu"},
        }
        if path == prefix:
            return repository
        if path == f"{prefix}/actions/runs/1":
            return {
                **self.event_run,
                "path": VALIDATOR.CI_WORKFLOW_PATH,
                "head_repository": {
                    "id": 2,
                    "full_name": "7xuanlu/wenlan",
                    "fork": False,
                },
                "repository": repository,
            }
        if path == f"{prefix}/actions/workflows/3":
            return {
                "id": 3,
                "name": VALIDATOR.CI_WORKFLOW_NAME,
                "path": VALIDATOR.CI_WORKFLOW_PATH,
                "state": "active",
            }
        if path == f"{prefix}/commits/{HEAD_SHA}/pulls":
            return [self.pr]
        if path == f"{prefix}/pulls/42":
            return self.pr
        if path == f"{prefix}/actions/runs/1/artifacts":
            return {"total_count": 4, "artifacts": self.artifacts}
        attempt_prefix = f"{prefix}/actions/runs/1/attempts/"
        if path.startswith(attempt_prefix) and path.endswith("/jobs"):
            attempt = int(path.removeprefix(attempt_prefix).removesuffix("/jobs"))
            jobs = [
                {**job, "run_attempt": job.get("run_attempt", attempt)}
                for job in self.jobs_by_attempt.get(attempt, [])
            ]
            page = params["page"]
            start = (page - 1) * 100
            return {"total_count": len(jobs), "jobs": jobs[start : start + 100]}
        return super().get_json(path, params=params)

    def download(self, path: str, destination: Path, maximum: int):
        artifact_id = int(path.rsplit("/", 2)[1])
        source = self.wrappers[artifact_id]
        shutil.copyfile(source, destination)
        raw = destination.read_bytes()
        if len(raw) > maximum:
            raise AssertionError("happy-path artifact exceeds declared maximum")
        return len(raw), hashlib.sha256(raw).hexdigest()


def release_job(
    target: str,
    *,
    job_id: int,
    conclusion: str = "success",
    run_id: int = 1,
    head_sha: str = HEAD_SHA,
) -> dict:
    return {
        "id": job_id,
        "run_id": run_id,
        "head_sha": head_sha,
        "name": f"release-preflight ({target})",
        "status": "completed",
        "conclusion": conclusion,
    }


class AttemptJobsApi:
    def __init__(self, jobs_by_attempt: dict[int, list[dict]]) -> None:
        self.jobs_by_attempt = jobs_by_attempt

    def get_json(self, path: str, *, params=None):
        attempt = int(path.split("/attempts/", 1)[1].split("/", 1)[0])
        jobs = [
            {**job, "run_attempt": job.get("run_attempt", attempt)}
            for job in self.jobs_by_attempt.get(attempt, [])
        ]
        page = params["page"]
        start = (page - 1) * 100
        return {"total_count": len(jobs), "jobs": jobs[start : start + 100]}


class ValidateReleaseCandidateTests(unittest.TestCase):
    def test_full_validate_candidate_happy_path_reconciles_manifests_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            old, new = release_contents()
            pr = candidate_pr_detail(state="open", merged=False)
            pr["changed_files"] = len(VALIDATOR.REQUIRED_RELEASE_PATHS)
            pr["base"]["sha"] = BASE_SHA
            event_run = {
                "id": 1,
                "run_attempt": 2,
                "workflow_id": 3,
                "check_suite_id": 4,
                "name": VALIDATOR.CI_WORKFLOW_NAME,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "head_branch": VALIDATOR.RELEASE_BRANCH,
                "head_sha": HEAD_SHA,
                "head_repository": {
                    "id": 2,
                    "full_name": "7xuanlu/wenlan",
                    "fork": False,
                },
                "repository": {
                    "id": 2,
                    "full_name": "7xuanlu/wenlan",
                },
            }
            wrappers: dict[int, Path] = {}
            artifacts: list[dict] = []
            matrix_entries = VALIDATOR.release_matrix()["include"]
            target_attempts = {
                entry["target"]: 1 if index < 2 else 2
                for index, entry in enumerate(matrix_entries)
            }
            for artifact_id, matrix_entry in enumerate(matrix_entries, start=10):
                target = matrix_entry["target"]
                artifact_attempt = target_attempts[target]
                target_root = root / target
                binary_dir = target_root / "bin"
                binary_dir.mkdir(parents=True)
                members = {
                    member
                    for asset in release_assets(target=target)
                    for member in asset["members"]
                }
                for member in members:
                    (binary_dir / member).write_bytes(f"payload:{member}".encode())
                dist = target_root / "dist"
                PACKAGE.package_target(target, binary_dir, dist, 1_700_000_000)
                manifest = MANIFEST.build_manifest(
                    repository="7xuanlu/wenlan",
                    repository_id=2,
                    run_id=1,
                    run_attempt=artifact_attempt,
                    pr_number=42,
                    pr_author="7xuanlu",
                    head_sha=HEAD_SHA,
                    tree_sha=HEAD_TREE_SHA,
                    version="0.15.4",
                    target=target,
                    dist_dir=dist,
                )
                (dist / MANIFEST.MANIFEST_NAME).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                wrapper = target_root / "artifact.zip"
                with zipfile.ZipFile(
                    wrapper, "w", compression=zipfile.ZIP_STORED
                ) as archive:
                    for path in sorted(dist.iterdir()):
                        archive.write(path, path.name)
                wrapper_raw = wrapper.read_bytes()
                wrappers[artifact_id] = wrapper
                artifacts.append(
                    {
                        "id": artifact_id,
                        "name": f"release-candidate-1-{artifact_attempt}-{target}",
                        "size_in_bytes": len(wrapper_raw),
                        "expired": False,
                        "digest": "sha256:"
                        + hashlib.sha256(wrapper_raw).hexdigest(),
                        "workflow_run": {
                            "id": 1,
                            "repository_id": 2,
                            "head_repository_id": 2,
                            "head_branch": VALIDATOR.RELEASE_BRANCH,
                            "head_sha": HEAD_SHA,
                        },
                    }
                )
            jobs_by_attempt = {
                1: [
                    {
                        "id": 100 + index,
                        "run_id": 1,
                        "head_sha": HEAD_SHA,
                        "name": f"release-preflight ({entry['target']})",
                        "status": "completed",
                        "conclusion": "success" if index < 2 else "failure",
                    }
                    for index, entry in enumerate(matrix_entries)
                ],
                2: [
                    {
                        "id": 200 + index,
                        "run_id": 1,
                        "head_sha": HEAD_SHA,
                        "name": f"release-preflight ({entry['target']})",
                        "status": "completed",
                        "conclusion": "success",
                    }
                    for index, entry in enumerate(matrix_entries)
                    if index >= 2
                ],
            }
            api = HappyPathApi(
                old,
                new,
                wrappers,
                artifacts,
                pr,
                event_run,
                jobs_by_attempt,
            )
            event = {"action": "completed", "workflow_run": event_run}
            validated_assets = root / "validated-assets"
            receipt = VALIDATOR.validate_candidate(
                event,
                api,
                "7xuanlu/wenlan",
                root,
                validated_assets_dir=validated_assets,
            )
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["pr_state"], "open")
            self.assertIsNone(receipt["merge_commit_sha"])
            self.assertEqual(receipt["run_attempt"], 2)
            self.assertEqual(
                {artifact["name"] for artifact in receipt["artifacts"]},
                {
                    f"release-candidate-1-{target_attempts[target]}-{target}"
                    for target in target_attempts
                },
            )
            self.assertEqual(len(receipt["artifacts"]), 4)
            self.assertEqual(len(receipt["assets"]), 6)
            self.assertEqual(
                {path.name for path in validated_assets.iterdir()},
                {asset["name"] for asset in release_assets()},
            )
            receipt_path = root / "receipt.json"
            VALIDATOR.write_validated_receipt(receipt_path, receipt)
            closed = VALIDATOR.close_receipt(
                receipt_path,
                artifact_name="validated-release-assets-1-2-100-2",
                artifact_id=99,
                artifact_digest="9" * 64,
                observer_run_id=100,
                observer_run_attempt=2,
                observer_workflow_id=200,
                observer_code_sha="a" * 40,
            )
            self.assertEqual(closed["receipt_state"], "closed")
            self.assertEqual(closed["validated_assets_artifact"]["id"], 99)
            self.assertEqual(
                closed["validated_assets_artifact"]["digest"],
                "sha256:" + "9" * 64,
            )
            self.assertEqual(
                closed["observer"]["workflow_path"],
                VALIDATOR.OBSERVER_WORKFLOW_PATH,
            )
            with self.assertRaisesRegex(
                VALIDATOR.CandidateError, "cannot be closed"
            ):
                VALIDATOR.close_receipt(
                    receipt_path,
                    artifact_name="validated-release-assets-1-2-100-2",
                    artifact_id=99,
                    artifact_digest="9" * 64,
                    observer_run_id=100,
                    observer_run_attempt=2,
                    observer_workflow_id=200,
                    observer_code_sha="a" * 40,
                )
            summary = root / "summary.md"
            VALIDATOR.write_summary(summary, receipt)
            rendered = summary.read_text(encoding="utf-8")
            self.assertIn("Exact artifacts/assets: `4` / `6`", rendered)
            self.assertIn("PR state / merge commit: `open` / `n/a`", rendered)

    def test_candidate_pr_state_accepts_open_and_merged_equal_tree(self) -> None:
        self.assertEqual(
            VALIDATOR._candidate_pr_state(
                PrStateApi(),
                "7xuanlu/wenlan",
                candidate_pr_detail(state="open", merged=False),
                pr_number=42,
                repository_id=2,
                head_sha=HEAD_SHA,
            ),
            ("open", None),
        )
        self.assertEqual(
            VALIDATOR._candidate_pr_state(
                PrStateApi(),
                "7xuanlu/wenlan",
                candidate_pr_detail(state="closed", merged=True),
                pr_number=42,
                repository_id=2,
                head_sha=HEAD_SHA,
            ),
            ("merged", "f" * 40),
        )
        merged_without_response_sha = candidate_pr_detail(
            state="closed", merged=True
        )
        merged_without_response_sha["merge_commit_sha"] = None
        self.assertEqual(
            VALIDATOR._candidate_pr_state(
                PrStateApi(),
                "7xuanlu/wenlan",
                merged_without_response_sha,
                pr_number=42,
                repository_id=2,
                head_sha=HEAD_SHA,
            ),
            ("merged", "f" * 40),
        )

    def test_candidate_pr_state_rejects_closed_unmerged_and_merge_tree_mismatch(self) -> None:
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "closed without being merged"):
            VALIDATOR._candidate_pr_state(
                PrStateApi(),
                "7xuanlu/wenlan",
                candidate_pr_detail(state="closed", merged=False),
                pr_number=42,
                repository_id=2,
                head_sha=HEAD_SHA,
            )
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "tree differs"):
            VALIDATOR._candidate_pr_state(
                PrStateApi("1" * 40),
                "7xuanlu/wenlan",
                candidate_pr_detail(state="closed", merged=True),
                pr_number=42,
                repository_id=2,
                head_sha=HEAD_SHA,
            )

    def test_artifact_pagination_requires_exact_total_count(self) -> None:
        class CountApi:
            def get_json(self, path: str, *, params=None):
                return {"total_count": 2, "artifacts": [{"id": 1}]}

        with self.assertRaisesRegex(VALIDATOR.CandidateError, "total_count"):
            VALIDATOR._api_pages(CountApi(), "/artifacts", "artifacts")

    def test_target_attempts_follow_only_jobs_reexecuted_by_partial_rerun(self) -> None:
        targets = [entry["target"] for entry in VALIDATOR.release_matrix()["include"]]
        api = AttemptJobsApi(
            {
                1: [
                    release_job(
                        target,
                        job_id=100 + index,
                        conclusion="success" if index < 2 else "failure",
                    )
                    for index, target in enumerate(targets)
                ],
                2: [
                    release_job(target, job_id=200 + index)
                    for index, target in enumerate(targets)
                    if index >= 2
                ],
            }
        )
        selected = VALIDATOR._latest_release_target_attempts(
            api,
            "7xuanlu/wenlan",
            run_id=1,
            current_attempt=2,
            head_sha=HEAD_SHA,
        )
        self.assertEqual(
            selected,
            {
                target: 1 if index < 2 else 2
                for index, target in enumerate(targets)
            },
        )

    def test_latest_reexecuted_target_failure_cannot_fallback(self) -> None:
        targets = [entry["target"] for entry in VALIDATOR.release_matrix()["include"]]
        api = AttemptJobsApi(
            {
                1: [
                    release_job(target, job_id=100 + index)
                    for index, target in enumerate(targets)
                ],
                2: [release_job(targets[0], job_id=200, conclusion="failure")],
            }
        )
        with self.assertRaisesRegex(
            VALIDATOR.CandidateError, "latest release-preflight job.*failure"
        ):
            VALIDATOR._latest_release_target_attempts(
                api,
                "7xuanlu/wenlan",
                run_id=1,
                current_attempt=2,
                head_sha=HEAD_SHA,
            )

    def test_duplicate_release_target_job_in_one_attempt_is_rejected(self) -> None:
        targets = [entry["target"] for entry in VALIDATOR.release_matrix()["include"]]
        jobs = [
            release_job(target, job_id=100 + index)
            for index, target in enumerate(targets)
        ]
        jobs.append(release_job(targets[0], job_id=999))
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "more than once"):
            VALIDATOR._latest_release_target_attempts(
                AttemptJobsApi({1: jobs}),
                "7xuanlu/wenlan",
                run_id=1,
                current_attempt=1,
                head_sha=HEAD_SHA,
            )

    def test_release_target_job_attempt_must_match_attempt_endpoint(self) -> None:
        target = VALIDATOR.release_matrix()["include"][0]["target"]
        mismatched = release_job(target, job_id=100)
        mismatched["run_attempt"] = 2
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "control-plane identity"):
            VALIDATOR._latest_release_target_attempts(
                AttemptJobsApi({1: [mismatched]}),
                "7xuanlu/wenlan",
                run_id=1,
                current_attempt=1,
                head_sha=HEAD_SHA,
            )

    def test_candidate_artifact_attempt_index_is_closed_and_conflict_safe(self) -> None:
        targets = [entry["target"] for entry in VALIDATOR.release_matrix()["include"]]
        attempts = {target: 1 if index < 2 else 2 for index, target in enumerate(targets)}
        artifacts = [
            {
                "id": 100 + index,
                "name": f"release-candidate-1-{attempts[target]}-{target}",
            }
            for index, target in enumerate(targets)
        ]
        selected = VALIDATOR._candidate_artifacts_for_attempts(
            artifacts,
            run_id=1,
            current_attempt=2,
            target_attempts=attempts,
        )
        self.assertEqual(set(selected), set(targets))

        duplicate = artifacts + [dict(artifacts[0], id=999)]
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "duplicated"):
            VALIDATOR._candidate_artifacts_for_attempts(
                duplicate,
                run_id=1,
                current_attempt=2,
                target_attempts=attempts,
            )

        future = list(artifacts)
        future[0] = dict(
            future[0], name=f"release-candidate-1-3-{targets[0]}"
        )
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "attempt or target"):
            VALIDATOR._candidate_artifacts_for_attempts(
                future,
                run_id=1,
                current_attempt=2,
                target_attempts=attempts,
            )

    def test_cross_host_artifact_redirect_strips_bearer_token(self) -> None:
        request = urllib.request.Request(
            "https://api.github.com/repos/7xuanlu/wenlan/actions/artifacts/1/zip",
            headers={"Authorization": "Bearer secret"},
        )
        redirected = VALIDATOR._CredentialStrippingRedirect().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://objects.example.invalid/signed-artifact",
        )
        self.assertIsNotNone(redirected)
        self.assertIsNone(redirected.get_header("Authorization"))

    def test_non_candidate_branch_skips_without_api_access(self) -> None:
        class NoApi:
            def get_json(self, path: str, *, params=None):
                raise AssertionError("ordinary CI run must not query candidate metadata")

            def download(self, path: str, destination: Path, maximum: int):
                raise AssertionError("ordinary CI run must not download artifacts")

        event = {"workflow_run": {"head_branch": "feature/not-a-release"}}
        receipt = VALIDATOR.validate_candidate(event, NoApi(), "7xuanlu/wenlan", Path("."))
        self.assertEqual(receipt["status"], "skipped")

    def test_candidate_trigger_name_cannot_replace_api_workflow_path_proof(self) -> None:
        repository = {"id": 2, "full_name": "7xuanlu/wenlan", "owner": {"login": "7xuanlu"}}
        head_repository = {
            "id": 2,
            "full_name": "7xuanlu/wenlan",
            "fork": False,
        }
        event = {
            "action": "completed",
            "workflow_run": {
                "id": 1,
                "run_attempt": 1,
                "workflow_id": 3,
                "check_suite_id": 4,
                "name": "CI",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "head_branch": VALIDATOR.RELEASE_BRANCH,
                "head_sha": HEAD_SHA,
                "head_repository": head_repository,
                "repository": repository,
            },
        }

        class WrongPathApi:
            def get_json(self, path: str, *, params=None):
                if path == "/repos/7xuanlu/wenlan":
                    return repository
                if path.endswith("/actions/runs/1"):
                    return {
                        **event["workflow_run"],
                        "path": ".github/workflows/not-ci.yml",
                    }
                if path.endswith("/actions/workflows/3"):
                    return {
                        "id": 3,
                        "name": "CI",
                        "path": ".github/workflows/not-ci.yml",
                        "state": "active",
                    }
                raise AssertionError(path)

        with self.assertRaisesRegex(VALIDATOR.CandidateError, "control-plane identity"):
            VALIDATOR.validate_candidate(
                event, WrongPathApi(), "7xuanlu/wenlan", Path(".")
            )

    def test_release_pr_content_requires_exact_paths_and_version_only_deltas(self) -> None:
        old, new = release_contents()
        version, base_sha, paths = VALIDATOR.validate_release_pr_content(
            FakeContentApi(old, new), "7xuanlu/wenlan", candidate_pr()
        )
        self.assertEqual((version, base_sha), ("0.15.4", BASE_SHA))
        self.assertEqual(paths, VALIDATOR.REQUIRED_RELEASE_PATHS)

        new["plugin-codex/README.md"] += "unrelated executable instruction\n"
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "exact version-only"):
            VALIDATOR.validate_release_pr_content(
                FakeContentApi(old, new), "7xuanlu/wenlan", candidate_pr()
            )

    def test_exact_transform_rejects_new_version_camouflage(self) -> None:
        hostile_mutations = {
            "Cargo.toml": 'panic = "abort" # 0.15.4\n',
            "crates/wenlan-cli/npm/package.json": json.dumps(
                {"version": "0.15.4", "scripts": {"postinstall": "echo 0.15.4"}}
            ),
        }
        for path, hostile in hostile_mutations.items():
            with self.subTest(path=path):
                old, new = release_contents()
                if path == "Cargo.toml":
                    new[path] += hostile
                else:
                    new[path] = hostile
                with self.assertRaisesRegex(VALIDATOR.CandidateError, "exact version-only"):
                    VALIDATOR.validate_release_pr_content(
                        FakeContentApi(old, new), "7xuanlu/wenlan", candidate_pr()
                    )

    def test_release_runner_mode_change_is_rejected(self) -> None:
        old, new = release_contents()
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "mode/type changed"):
            VALIDATOR.validate_release_pr_content(
                FakeContentApi(
                    old,
                    new,
                    head_mode_overrides={
                        "plugin-codex/bin/wenlan-mcp-runner.sh": "100644"
                    },
                ),
                "7xuanlu/wenlan",
                candidate_pr(),
            )

    def test_always_bump_patch_and_release_as_are_fail_closed(self) -> None:
        base = {
            "packages": {
                ".": {"release-type": "simple", "versioning": "always-bump-patch"}
            },
            "$schema": "https://example.invalid/schema.json",
        }
        VALIDATOR._release_version_policy(json.dumps(base), "0.15.3", "0.15.4")
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "next patch"):
            VALIDATOR._release_version_policy(json.dumps(base), "0.15.3", "9.9.9")
        base["packages"]["."]["release-as"] = "1.0.0"
        VALIDATOR._release_version_policy(json.dumps(base), "0.15.3", "1.0.0")
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "exactly match release-as"):
            VALIDATOR._release_version_policy(json.dumps(base), "0.15.3", "0.15.4")

    def test_artifact_record_rejects_expiry_digest_and_fork_identity(self) -> None:
        name = "release-candidate-1-1-x86_64-unknown-linux-gnu"
        artifact = {
            "id": 9,
            "name": name,
            "size_in_bytes": 100,
            "expired": False,
            "digest": "sha256:" + "d" * 64,
            "workflow_run": {
                "id": 1,
                "repository_id": 2,
                "head_repository_id": 2,
                "head_branch": VALIDATOR.RELEASE_BRANCH,
                "head_sha": HEAD_SHA,
            },
        }
        VALIDATOR._validate_artifact_record(
            artifact,
            expected_name=name,
            run_id=1,
            repository_id=2,
            head_sha=HEAD_SHA,
        )
        for field, value in (("expired", True), ("digest", "sha256:no")):
            mutated = json.loads(json.dumps(artifact))
            mutated[field] = value
            with self.subTest(field=field), self.assertRaises(VALIDATOR.CandidateError):
                VALIDATOR._validate_artifact_record(
                    mutated,
                    expected_name=name,
                    run_id=1,
                    repository_id=2,
                    head_sha=HEAD_SHA,
                )
        mutated = json.loads(json.dumps(artifact))
        mutated["workflow_run"]["head_repository_id"] = 99
        with self.assertRaisesRegex(VALIDATOR.CandidateError, "identity mismatch"):
            VALIDATOR._validate_artifact_record(
                mutated,
                expected_name=name,
                run_id=1,
                repository_id=2,
                head_sha=HEAD_SHA,
            )

    def make_outer_artifact(
        self,
        root: Path,
        hostile: bool = False,
        hostile_inner: bool = False,
    ) -> tuple[Path, dict, dict]:
        target = "x86_64-unknown-linux-gnu"
        binary_dir = root / "bin"
        binary_dir.mkdir()
        for member in release_assets(target=target)[0]["members"]:
            (binary_dir / member).write_bytes(f"payload:{member}".encode())
        dist = root / "dist"
        PACKAGE.package_target(target, binary_dir, dist, 1_700_000_000)
        if hostile_inner:
            hostile_tar = root / "hostile-inner.tar"
            with tarfile.open(hostile_tar, "w") as archive:
                member = tarfile.TarInfo("../wenlan")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            canonical_gzip(hostile_tar, dist / "wenlan-linux-x64.tar.gz")
        manifest = MANIFEST.build_manifest(
            repository="7xuanlu/wenlan",
            repository_id=2,
            run_id=1,
            run_attempt=1,
            pr_number=42,
            pr_author="7xuanlu",
            head_sha=HEAD_SHA,
            tree_sha=TREE_SHA,
            version="0.15.4",
            target=target,
            dist_dir=dist,
        )
        (dist / MANIFEST.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        wrapper = root / "artifact.zip"
        with zipfile.ZipFile(wrapper, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in dist.iterdir():
                archive.write(path, path.name)
            if hostile:
                archive.writestr("../escape", b"no")
        raw = wrapper.read_bytes()
        artifact = {
            "id": 9,
            "size_in_bytes": len(raw),
            "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
        }
        trusted = {
            "repository": "7xuanlu/wenlan",
            "repository_id": 2,
            "run_id": 1,
            "run_attempt": 1,
            "pr_number": 42,
            "pr_author": "7xuanlu",
            "head_sha": HEAD_SHA,
            "tree_sha": TREE_SHA,
            "version": "0.15.4",
            "expected_artifact_name": f"release-candidate-1-1-{target}",
        }
        return wrapper, artifact, trusted

    def test_downloaded_payload_is_hashed_and_inspected_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrapper, artifact, trusted = self.make_outer_artifact(root)
            work = root / "work"
            work.mkdir()
            assets, _manifest = VALIDATOR._download_and_validate(
                DownloadApi(wrapper),
                artifact,
                target="x86_64-unknown-linux-gnu",
                work_dir=work,
                trusted=trusted,
            )
            self.assertEqual(len(assets), 1)
            self.assertEqual(assets[0]["name"], "wenlan-linux-x64.tar.gz")

    def test_downloaded_outer_zip_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrapper, artifact, trusted = self.make_outer_artifact(root, hostile=True)
            work = root / "work"
            work.mkdir()
            with self.assertRaisesRegex(Exception, "unsafe archive entry"):
                VALIDATOR._download_and_validate(
                    DownloadApi(wrapper),
                    artifact,
                    target="x86_64-unknown-linux-gnu",
                    work_dir=work,
                    trusted=trusted,
                )

    def test_downloaded_inner_archive_rejects_hostile_member_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrapper, artifact, trusted = self.make_outer_artifact(
                root, hostile_inner=True
            )
            work = root / "work"
            work.mkdir()
            with self.assertRaisesRegex(Exception, "unsafe archive entry"):
                VALIDATOR._download_and_validate(
                    DownloadApi(wrapper),
                    artifact,
                    target="x86_64-unknown-linux-gnu",
                    work_dir=work,
                    trusted=trusted,
                )

    def test_observer_rejects_darwin_standalone_member_byte_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = "aarch64-apple-darwin"
            binary_dir = root / "bin"
            binary_dir.mkdir()
            for member in {
                member
                for asset in release_assets(target=target)
                for member in asset["members"]
            }:
                (binary_dir / member).write_bytes(f"payload:{member}".encode())
            dist = root / "dist"
            PACKAGE.package_target(target, binary_dir, dist, 1_700_000_000)
            hostile = root / "hostile-wenlan"
            hostile.write_bytes(b"different bytes")
            release_archive.create_tar_gz(
                dist / "wenlan-cli-darwin-arm64.tar.gz",
                [(hostile, "wenlan")],
                1_700_000_000,
            )
            manifest = MANIFEST.build_manifest(
                repository="7xuanlu/wenlan",
                repository_id=2,
                run_id=1,
                run_attempt=1,
                pr_number=42,
                pr_author="7xuanlu",
                head_sha=HEAD_SHA,
                tree_sha=TREE_SHA,
                version="0.15.4",
                target=target,
                dist_dir=dist,
            )
            (dist / MANIFEST.MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            wrapper = root / "artifact.zip"
            with zipfile.ZipFile(wrapper, "w", compression=zipfile.ZIP_STORED) as archive:
                for path in dist.iterdir():
                    archive.write(path, path.name)
            raw_wrapper = wrapper.read_bytes()
            artifact = {
                "id": 9,
                "size_in_bytes": len(raw_wrapper),
                "digest": "sha256:" + hashlib.sha256(raw_wrapper).hexdigest(),
            }
            work = root / "work"
            work.mkdir()
            trusted = {
                "repository": "7xuanlu/wenlan",
                "repository_id": 2,
                "run_id": 1,
                "run_attempt": 1,
                "pr_number": 42,
                "pr_author": "7xuanlu",
                "head_sha": HEAD_SHA,
                "tree_sha": TREE_SHA,
                "version": "0.15.4",
                "expected_artifact_name": f"release-candidate-1-1-{target}",
            }
            with self.assertRaisesRegex(VALIDATOR.CandidateError, "standalone archive bytes"):
                VALIDATOR._download_and_validate(
                    DownloadApi(wrapper),
                    artifact,
                    target=target,
                    work_dir=work,
                    trusted=trusted,
                )

    def test_failure_summary_flattens_untrusted_control_characters(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "summary.md"
            VALIDATOR.write_summary(
                path,
                {"status": "failed", "reason": "bad`\n## injected\x00"},
            )
            summary = path.read_text(encoding="utf-8")
            self.assertNotIn("\n## injected", summary)
            self.assertNotIn("\x00", summary)
            self.assertIn("bad' ## injected?", summary)

    def test_passed_summary_has_exact_four_artifacts_and_six_inner_assets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "summary.md"
            assets = [
                {
                    "name": asset["name"],
                    "target": asset["target"],
                    "sha256": str(index) * 64,
                    "size": index,
                }
                for index, asset in enumerate(release_assets(), start=1)
            ]
            artifacts = [
                {
                    "id": index,
                    "name": f"release-candidate-1-1-target-{index}",
                    "digest": "sha256:" + str(index) * 64,
                    "size": index,
                }
                for index in range(1, 5)
            ]
            VALIDATOR.write_summary(
                path,
                {
                    "status": "passed",
                    "run_id": 1,
                    "run_attempt": 1,
                    "pr_number": 42,
                    "pr_state": "merged",
                    "merge_commit_sha": "f" * 40,
                    "head_sha": HEAD_SHA,
                    "tree_sha": TREE_SHA,
                    "version": "0.15.4",
                    "artifacts": artifacts,
                    "assets": list(reversed(assets)),
                },
            )
            summary = path.read_text(encoding="utf-8")
            self.assertIn("Exact artifacts/assets: `4` / `6`", summary)
            self.assertIn("PR state / merge commit: `merged`", summary)
            self.assertEqual(summary.count("| Canonical inner asset |"), 1)
            for asset in assets:
                row = (
                    f"| `{asset['name']}` | `{asset['target']}` | "
                    f"`{asset['sha256']}` | {asset['size']} |"
                )
                self.assertEqual(summary.count(row), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
