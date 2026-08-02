#!/usr/bin/env python3
"""Fail closed when deciding whether PR CI may omit duplicate platform tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


VALIDATOR_PATH = Path(__file__).with_name("validate-release-candidate.py")
SPEC = importlib.util.spec_from_file_location("release_candidate_validator", VALIDATOR_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def classify(
    *, event_name: str, event: object, api: object, repository: str
) -> tuple[bool, str]:
    if event_name != "pull_request":
        return False, "event is not pull_request"
    if not isinstance(event, dict):
        return False, "event payload is not an object"
    try:
        version, head_sha = VALIDATOR.validate_trusted_release_candidate(
            event, api, repository
        )
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    return True, f"trusted release candidate {version} at {head_sha}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, type=Path)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--github-output", required=True, type=Path)
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    args = parser.parse_args(argv)

    try:
        event = json.loads(args.event.read_text(encoding="utf-8"))
        api = VALIDATOR.GitHubApi(os.environ.get("GITHUB_TOKEN", ""), args.api_url)
        trusted, reason = classify(
            event_name=args.event_name,
            event=event,
            api=api,
            repository=args.repository,
        )
    except Exception as error:
        trusted = False
        reason = f"{type(error).__name__}: {error}"
    with args.github_output.open("a", encoding="utf-8") as output:
        output.write(f"trusted-release-candidate={'true' if trusted else 'false'}\n")
    print(f"trusted-release-candidate={str(trusted).lower()}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
