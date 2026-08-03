#!/usr/bin/env python3
"""Decide whether a green pull request went stale for a reason worth re-testing.

`main` requires branches to be up to date, so every merge invalidates the
green of every other open pull request. When the merge that landed and the
pull request left behind touch no file in common, re-running the whole matrix
proves nothing the previous run did not already prove.

This command decides that case and never acts on its own judgement alone: a
pull request must carry an explicit opt-in label, and its own aggregate check
must be verified green on its exact head commit. Merging with administrator
privileges bypasses branch protection, so the green is re-checked here rather
than assumed from the protection that is being bypassed.
"""

from __future__ import annotations

import argparse
import json
import sys

OPT_IN_LABEL = "admin-merge-when-safe"


class StaleMergeError(RuntimeError):
    """Inputs were shaped in a way no decision may be made from."""


def decide(pull: dict, pushed_files: set[str], check_conclusion: str) -> tuple[bool, str]:
    """Return (safe_to_merge, reason). Any doubt resolves to not-safe."""

    for field in ("number", "mergeable", "mergeStateStatus", "headRefOid"):
        if field not in pull:
            raise StaleMergeError(f"pull request payload omits {field!r}")

    labels = {label["name"] for label in pull.get("labels", []) if "name" in label}
    if OPT_IN_LABEL not in labels:
        return False, f"not opted in ({OPT_IN_LABEL!r} absent)"
    if pull.get("isDraft"):
        return False, "draft"

    # BEHIND is precisely "green but the base moved". BLOCKED, DIRTY, and
    # UNSTABLE each mean something is actually wrong, and UNKNOWN means GitHub
    # has not finished computing the merge — none of those may be bypassed.
    if pull["mergeable"] != "MERGEABLE":
        return False, f"mergeable={pull['mergeable']}"
    if pull["mergeStateStatus"] != "BEHIND":
        return False, f"mergeStateStatus={pull['mergeStateStatus']}"

    # The bypass skips required checks, so their result is established here
    # instead. This conclusion must come from the pull request's exact head
    # commit; a conclusion for any other commit says nothing about this one.
    if check_conclusion != "success":
        return False, f"aggregate check is {check_conclusion!r}, not success"

    changed = {path for path in pull.get("files", []) if path}
    if not changed:
        return False, "no changed files reported"
    overlap = sorted(changed & pushed_files)
    if overlap:
        return False, f"overlaps the merge that landed: {', '.join(overlap[:5])}"

    return True, f"green on {pull['headRefOid'][:8]}, no overlap with the landed merge"


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pull", required=True, help="pull request JSON")
    parser.add_argument("--pushed-files", required=True, help="JSON array of paths")
    parser.add_argument("--check-conclusion", required=True)
    args = parser.parse_args(argv)

    pull = json.loads(args.pull)
    pushed = json.loads(args.pushed_files)
    if not isinstance(pushed, list):
        raise StaleMergeError("pushed files must be a JSON array")

    safe, reason = decide(pull, {str(path) for path in pushed}, args.check_conclusion)
    print(json.dumps({"safe": safe, "reason": reason, "number": pull["number"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (StaleMergeError, json.JSONDecodeError, KeyError) as error:
        print(f"stale-safe-merge: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
