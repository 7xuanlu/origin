#!/usr/bin/env python3
"""Prune the oldest GitHub Actions caches to a bounded repository target."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


API_VERSION = "2026-03-10"
DEFAULT_PROTECTED_PREFIXES = ("fastembed-bge-base-en-v1.5-q-v3-portable",)


class MaintenanceError(RuntimeError):
    """The cache inventory or maintenance request was invalid."""


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MaintenanceError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MaintenanceError(f"{label} must be a non-negative integer")
    return value


def _cache_row(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise MaintenanceError("cache inventory row must be an object")
    cache_id = _positive_int(raw.get("id"), "cache id")
    size = _positive_int(raw.get("size_in_bytes"), "cache size")
    key = raw.get("key")
    ref = raw.get("ref")
    last_accessed = raw.get("last_accessed_at")
    created = raw.get("created_at")
    if not all(isinstance(value, str) and value for value in (key, ref, last_accessed, created)):
        raise MaintenanceError(f"cache {cache_id} has incomplete string metadata")
    return {
        "id": cache_id,
        "key": key,
        "ref": ref,
        "size_in_bytes": size,
        "last_accessed_at": last_accessed,
        "created_at": created,
    }


def build_prune_plan(
    raw_caches: Iterable[object],
    *,
    usage_bytes: int,
    target_bytes: int,
    protected_prefixes: tuple[str, ...] = DEFAULT_PROTECTED_PREFIXES,
) -> dict:
    """Return a deterministic oldest-first eviction plan."""

    usage = _nonnegative_int(usage_bytes, "cache usage")
    target = _positive_int(target_bytes, "cache target")
    if not all(isinstance(prefix, str) and prefix for prefix in protected_prefixes):
        raise MaintenanceError("protected cache prefixes must be non-empty strings")

    caches = [_cache_row(raw) for raw in raw_caches]
    ids = [cache["id"] for cache in caches]
    if len(ids) != len(set(ids)):
        raise MaintenanceError("cache inventory contains duplicate ids")

    projected = usage
    selected: list[dict] = []
    protected_bytes = 0
    candidates = []
    for cache in caches:
        if any(cache["key"].startswith(prefix) for prefix in protected_prefixes):
            protected_bytes += cache["size_in_bytes"]
        else:
            candidates.append(cache)

    candidates.sort(
        key=lambda cache: (
            cache["last_accessed_at"],
            cache["created_at"],
            cache["id"],
        )
    )
    for cache in candidates:
        if projected <= target:
            break
        selected.append(cache)
        projected = max(0, projected - cache["size_in_bytes"])

    return {
        "usage_bytes": usage,
        "target_bytes": target,
        "projected_bytes": projected,
        "delete_bytes": usage - projected,
        "protected_bytes": protected_bytes,
        "protected_prefixes": list(protected_prefixes),
        "inventory_count": len(caches),
        "delete_count": len(selected),
        "status": "within_target" if usage <= target else (
            "planned" if projected <= target else "insufficient_candidates"
        ),
        "deletions": selected,
    }


@dataclass
class GitHubCacheClient:
    api_url: str
    repository: str
    token: str

    def _request(self, path: str, *, method: str = "GET") -> object | None:
        url = f"{self.api_url.rstrip('/')}{path}"
        request = urllib.request.Request(
            url,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "wenlan-ci-cache-maintenance/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise MaintenanceError(
                f"GitHub cache API {method} {path} failed with {error.code}: {detail}"
            ) from error
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError as error:
            raise MaintenanceError(f"GitHub cache API returned invalid JSON for {path}") from error

    def usage(self) -> tuple[int, int]:
        payload = self._request(f"/repos/{self.repository}/actions/cache/usage")
        if not isinstance(payload, dict):
            raise MaintenanceError("cache usage response must be an object")
        return (
            _nonnegative_int(payload.get("active_caches_size_in_bytes"), "cache usage"),
            _nonnegative_int(payload.get("active_caches_count"), "cache count"),
        )

    def caches(self) -> list[dict]:
        caches: list[dict] = []
        page = 1
        while True:
            query = urllib.parse.urlencode(
                {
                    "per_page": 100,
                    "page": page,
                    "sort": "last_accessed_at",
                    "direction": "asc",
                }
            )
            payload = self._request(
                f"/repos/{self.repository}/actions/caches?{query}"
            )
            if not isinstance(payload, dict) or not isinstance(
                payload.get("actions_caches"), list
            ):
                raise MaintenanceError("cache list response must contain actions_caches")
            rows = payload["actions_caches"]
            caches.extend(rows)
            if len(rows) < 100:
                break
            page += 1
        return caches

    def delete(self, cache_id: int) -> None:
        self._request(
            f"/repos/{self.repository}/actions/caches/{cache_id}",
            method="DELETE",
        )


def _write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    parser.add_argument("--target-gb", type=int, default=9)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not arguments.repository or "/" not in arguments.repository:
        raise MaintenanceError("--repository must be OWNER/REPO")
    if not token:
        raise MaintenanceError("GH_TOKEN or GITHUB_TOKEN is required")
    target_bytes = _positive_int(arguments.target_gb, "target GB") * 1_000_000_000
    client = GitHubCacheClient(arguments.api_url, arguments.repository, token)
    usage_bytes, active_count = client.usage()
    plan = build_prune_plan(
        client.caches(),
        usage_bytes=usage_bytes,
        target_bytes=target_bytes,
    )
    receipt = {
        "schema_version": 1,
        "repository": arguments.repository,
        "dry_run": arguments.dry_run,
        "active_count": active_count,
        **plan,
    }

    if plan["status"] == "insufficient_candidates":
        _write_receipt(arguments.receipt, receipt)
        raise MaintenanceError("protected caches prevent reaching the configured target")

    deleted_ids: list[int] = []
    if not arguments.dry_run:
        for cache in plan["deletions"]:
            client.delete(cache["id"])
            deleted_ids.append(cache["id"])
    receipt["deleted_ids"] = deleted_ids
    receipt["status"] = (
        "dry_run" if arguments.dry_run and plan["delete_count"] else plan["status"]
    )
    _write_receipt(arguments.receipt, receipt)
    print(
        f"cache maintenance: {receipt['status']}; "
        f"{plan['delete_count']} planned, {len(deleted_ids)} deleted; "
        f"{usage_bytes} -> {plan['projected_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except MaintenanceError as error:
        print(f"ci-cache-maintenance: {error}", file=sys.stderr)
        raise SystemExit(1)
