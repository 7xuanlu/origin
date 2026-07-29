#!/usr/bin/env python3
"""Check translated README files are synced to README.md.

Each translated README must include:

    <!-- README_SYNC: source=README.md sha256=<64 hex chars> -->

The hash is over README.md with the generated EVAL_SNAPSHOT block normalized
away. Updating generated benchmark tables in all languages does not require
restamping translations; changing English prose does.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
from pathlib import Path


TARGETS = ("README.zh-Hans.md", "README.zh-Hant.md", "README.es-ES.md")
START = "<!-- EVAL_SNAPSHOT_START -->"
END = "<!-- EVAL_SNAPSHOT_END -->"
MARKER_RE = re.compile(
    r"<!--\s*README_SYNC:\s*source=README\.md\s+sha256=([a-f0-9]{64})\s*-->"
)


def normalize_generated_snapshot(text: str) -> str:
    start = text.find(START)
    end = text.find(END)
    if start == -1 or end == -1:
        raise SystemExit("README markers not found: EVAL_SNAPSHOT_START / EVAL_SNAPSHOT_END")
    end += len(END)
    return text[:start] + START + "\n" + END + text[end:]


def readme_hash(root: Path) -> str:
    text = (root / "README.md").read_text(encoding="utf-8")
    normalized = normalize_generated_snapshot(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check(root: Path) -> list[str]:
    expected = readme_hash(root)
    errors: list[str] = []
    for rel in TARGETS:
        path = root / rel
        if not path.exists():
            errors.append(f"{rel}: missing translated README")
            continue
        text = path.read_text(encoding="utf-8")
        match = MARKER_RE.search(text)
        if not match:
            errors.append(f"{rel}: missing README_SYNC marker")
            continue
        actual = match.group(1)
        if actual != expected:
            errors.append(f"{rel}: stale sync hash {actual}; expected {expected}")
    return errors


def write_markers(root: Path) -> None:
    digest = readme_hash(root)
    marker = f"<!-- README_SYNC: source=README.md sha256={digest} -->"
    for rel in TARGETS:
        path = root / rel
        if not path.exists():
            raise SystemExit(f"{rel}: missing translated README")
        text = path.read_text(encoding="utf-8")
        updated, count = MARKER_RE.subn(marker, text, count=1)
        if count != 1:
            raise SystemExit(f"{rel}: missing README_SYNC marker")
        path.write_text(updated, encoding="utf-8")


def run_selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        table = START + "\n| Benchmark |\n|---|\n| Old |\n" + END
        (root / "README.md").write_text("English README\n\n" + table + "\n", encoding="utf-8")
        digest = readme_hash(root)
        marker = f"<!-- README_SYNC: source=README.md sha256={digest} -->\n"
        for rel in TARGETS:
            (root / rel).write_text(marker + "translated\n", encoding="utf-8")
        assert check(root) == []

        updated_table = START + "\n| Benchmark |\n|---|\n| New |\n" + END
        (root / "README.md").write_text("English README\n\n" + updated_table + "\n", encoding="utf-8")
        assert check(root) == []

        (root / "README.md").write_text("English README changed\n\n" + updated_table + "\n", encoding="utf-8")
        stale = check(root)
        assert len(stale) == len(TARGETS)
        assert all("stale sync hash" in err for err in stale)

        write_markers(root)
        assert check(root) == []

        digest = readme_hash(root)
        (root / TARGETS[0]).write_text(
            f"<!-- README_SYNC: source=README.md sha256={digest} -->\n",
            encoding="utf-8",
        )
        (root / TARGETS[1]).write_text("translated without marker\n", encoding="utf-8")
        errors = check(root)
        assert errors == [f"{TARGETS[1]}: missing README_SYNC marker"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--print-hash", action="store_true")
    parser.add_argument("--write-markers", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        print("selftest ok")
        return 0

    root = args.root.resolve()
    if args.print_hash:
        print(readme_hash(root))
        return 0
    if args.write_markers:
        write_markers(root)
        print("README translation sync markers updated")
        return 0

    errors = check(root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("README translations are in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
