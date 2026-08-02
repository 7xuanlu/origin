#!/usr/bin/env python3
"""Create and strictly parse an untrusted release-candidate claim manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from release_targets import TargetError, release_assets, require_target


SCHEMA_VERSION = 1
MANIFEST_NAME = "release-candidate-manifest.json"
RELEASE_BRANCH = "release-please--branches--main"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
MAX_MANIFEST_BYTES = 64 * 1024


class ManifestError(ValueError):
    """A candidate manifest is malformed or does not match its files."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as payload:
        while chunk := payload.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _string(value: object, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ManifestError(f"{label} must be a bounded nonempty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ManifestError(f"{label} has an invalid value")
    return value


def _exact_keys(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ManifestError(f"{label} keys mismatch: {actual}")
    return value


def build_manifest(
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
    dist_dir: Path,
) -> dict:
    require_target(target)
    expected = release_assets(target=target)
    expected_names = {asset["name"] for asset in expected}
    actual_names = {
        path.name for path in dist_dir.iterdir() if path.name != MANIFEST_NAME
    }
    if actual_names != expected_names:
        raise ManifestError(
            f"dist inventory mismatch: missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    assets = []
    for expected_asset in expected:
        path = dist_dir / expected_asset["name"]
        if path.is_symlink() or not path.is_file():
            raise ManifestError(f"asset is not a regular file: {path.name}")
        assets.append(
            {
                "archive": expected_asset["archive"],
                "members": expected_asset["members"],
                "name": expected_asset["name"],
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "repository": {"full_name": repository, "id": repository_id},
        "workflow": {
            "event": "pull_request",
            "path": ".github/workflows/ci.yml",
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
        "artifact": {
            "name": f"release-candidate-{run_id}-{run_attempt}-{target}"
        },
        "assets": assets,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(value: object) -> dict:
    manifest = _exact_keys(
        value,
        {
            "schema_version",
            "repository",
            "workflow",
            "pull_request",
            "source",
            "release",
            "target",
            "artifact",
            "assets",
        },
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError("unsupported manifest schema_version")
    repository = _exact_keys(manifest["repository"], {"full_name", "id"}, "repository")
    repository_name = _string(repository["full_name"], "repository.full_name")
    if re.fullmatch(r"[^/\s]+/[^/\s]+", repository_name) is None:
        raise ManifestError("repository.full_name must be OWNER/REPO")
    _positive_int(repository["id"], "repository.id")
    workflow = _exact_keys(
        manifest["workflow"], {"event", "path", "run_attempt", "run_id"}, "workflow"
    )
    if workflow["event"] != "pull_request" or workflow["path"] != ".github/workflows/ci.yml":
        raise ManifestError("workflow identity is not the canonical pull_request CI")
    run_id = _positive_int(workflow["run_id"], "workflow.run_id")
    run_attempt = _positive_int(workflow["run_attempt"], "workflow.run_attempt")
    pull_request = _exact_keys(
        manifest["pull_request"],
        {"author", "base_ref", "head_ref", "head_sha", "number"},
        "pull_request",
    )
    _positive_int(pull_request["number"], "pull_request.number")
    _string(pull_request["author"], "pull_request.author")
    if pull_request["base_ref"] != "main" or pull_request["head_ref"] != RELEASE_BRANCH:
        raise ManifestError("pull request refs are not the release-please contract")
    head_sha = _string(pull_request["head_sha"], "pull_request.head_sha", SHA_RE)
    source = _exact_keys(manifest["source"], {"commit_sha", "tree_sha"}, "source")
    if _string(source["commit_sha"], "source.commit_sha", SHA_RE) != head_sha:
        raise ManifestError("source.commit_sha does not equal pull_request.head_sha")
    _string(source["tree_sha"], "source.tree_sha", SHA_RE)
    release = _exact_keys(manifest["release"], {"tag", "version"}, "release")
    version = _string(release["version"], "release.version", VERSION_RE)
    if release["tag"] != f"v{version}":
        raise ManifestError("release.tag does not match release.version")
    target = _string(manifest["target"], "target")
    require_target(target)
    artifact = _exact_keys(manifest["artifact"], {"name"}, "artifact")
    if artifact["name"] != f"release-candidate-{run_id}-{run_attempt}-{target}":
        raise ManifestError("artifact.name does not match run/attempt/target")
    if not isinstance(manifest["assets"], list):
        raise ManifestError("assets must be an array")
    expected_assets = release_assets(target=target)
    if len(manifest["assets"]) != len(expected_assets):
        raise ManifestError("asset count does not match canonical target inventory")
    for value_asset, expected in zip(manifest["assets"], expected_assets, strict=True):
        asset = _exact_keys(
            value_asset,
            {"archive", "members", "name", "sha256", "size"},
            "asset",
        )
        if (
            asset["archive"] != expected["archive"]
            or asset["members"] != expected["members"]
            or asset["name"] != expected["name"]
        ):
            raise ManifestError("asset identity differs from canonical inventory")
        _string(asset["sha256"], "asset.sha256", DIGEST_RE)
        _positive_int(asset["size"], "asset.size")
    return manifest


def read_manifest(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ManifestError("manifest is not a regular file")
    size = path.stat().st_size
    if size < 1 or size > MAX_MANIFEST_BYTES:
        raise ManifestError("manifest size is outside the allowed range")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError("manifest is not valid UTF-8 JSON") from error
    return validate_manifest(value)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repository-id", required=True, type=int)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-attempt", required=True, type=int)
    parser.add_argument("--pr-number", required=True, type=int)
    parser.add_argument("--pr-author", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--tree-sha", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dist-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_manifest(
        repository=args.repository,
        repository_id=args.repository_id,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        pr_number=args.pr_number,
        pr_author=args.pr_author,
        head_sha=args.head_sha,
        tree_sha=args.tree_sha,
        version=args.version,
        target=args.target,
        dist_dir=args.dist_dir,
    )
    output = args.dist_dir / MANIFEST_NAME
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (ManifestError, TargetError, OSError) as error:
        print(f"release-candidate-manifest: {error}", file=sys.stderr)
        raise SystemExit(1)
