#!/usr/bin/env python3
"""check-behavior-trace — enforce the plan-behaviors <-> tests mapping.

The intent anchor for "is this the right test": the plan/spec carries a
`## Behaviors` section (the contract IS the plan — no separate artifact):

    ## Behaviors
    - B1: given a stored memory, `wenlan memories` lists it
    - B2: given a dirty tree, mutprove refuses to mutate

Tests cite the behavior they protect with a `covers: B<n>` tag in a comment
(any language, any test framework):

    #[test]  // covers: B2
    fn refuses_dirty_tree() { ... }

This checker fails when the mapping breaks in either direction:
  - a behavior no test covers  (untested intent)
  - a covers-tag citing a behavior id the plan doesn't define (dangling tag —
    usually a renamed/deleted behavior, i.e. the test now mirrors nothing)

Usage:
    check-behavior-trace.py <plan.md> <test-file-or-dir>...
    check-behavior-trace.py --self-test

Exit: 0 mapping intact · 1 gaps · 2 usage/parse error.
"""

import os
import re
import sys

BEHAVIOR_RE = re.compile(r"^\s*[-*]\s+\**\b(B\d+)\**\s*:", re.MULTILINE)
COVERS_RE = re.compile(r"covers:\s*(B\d+(?:\s*,\s*B\d+)*)")
TEST_EXTS = (".rs", ".py", ".sh", ".ts", ".js", ".tsx", ".go")


def parse_behaviors(plan_path):
    text = open(plan_path).read()
    section = re.search(r"^##+\s+Behaviors\s*$(.*?)(?=^##+\s|\Z)",
                        text, re.MULTILINE | re.DOTALL)
    if not section:
        return None
    return BEHAVIOR_RE.findall(section.group(1))


def collect_covers(paths):
    tags = {}  # id -> [file:line]
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names
                          if n.endswith(TEST_EXTS)]
        else:
            files.append(p)
    for f in files:
        try:
            lines = open(f, errors="replace").read().splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            for m in COVERS_RE.finditer(line):
                for bid in re.findall(r"B\d+", m.group(1)):
                    tags.setdefault(bid, []).append(f"{f}:{i}")
    return tags


def run(plan_path, test_paths):
    behaviors = parse_behaviors(plan_path)
    if behaviors is None:
        print(f"error: no '## Behaviors' section in {plan_path}", file=sys.stderr)
        return 2
    if not behaviors:
        print(f"error: '## Behaviors' section in {plan_path} defines no "
              f"'- B<n>:' items", file=sys.stderr)
        return 2

    covers = collect_covers(test_paths)
    uncovered = [b for b in behaviors if b not in covers]
    dangling = {bid: locs for bid, locs in covers.items()
                if bid not in behaviors}

    for b in behaviors:
        locs = covers.get(b, [])
        mark = "OK  " if locs else "GAP "
        print(f"{mark}{b}: {', '.join(locs) if locs else 'NO covering test'}")
    for bid, locs in sorted(dangling.items()):
        print(f"DANGLING {bid}: cited by {', '.join(locs)} but not defined "
              f"in the plan")

    if uncovered or dangling:
        print(f"\nFAIL: {len(uncovered)} uncovered behavior(s), "
              f"{len(dangling)} dangling tag(s)")
        return 1
    print(f"\nPASS: {len(behaviors)} behaviors all covered, no dangling tags")
    return 0


def self_test():
    """Red proof: the checker must fail on gaps and pass on intact mappings."""
    import tempfile

    failures = []
    with tempfile.TemporaryDirectory(
            dir=os.environ.get("TMPDIR") or None) as td:
        plan = os.path.join(td, "plan.md")
        test = os.path.join(td, "t.rs")

        def write(plan_body, test_body):
            open(plan, "w").write(plan_body)
            open(test, "w").write(test_body)

        # 1. intact mapping -> 0
        write("## Behaviors\n- B1: lists memories\n- B2: refuses dirty\n",
              "// covers: B1\nfn a(){}\n// covers: B2\nfn b(){}\n")
        if run(plan, [td]) != 0:
            failures.append("intact mapping should pass")

        # 2. uncovered behavior -> 1
        write("## Behaviors\n- B1: lists\n- B2: refuses\n",
              "// covers: B1\nfn a(){}\n")
        if run(plan, [td]) != 1:
            failures.append("uncovered behavior should fail")

        # 3. dangling tag -> 1
        write("## Behaviors\n- B1: lists\n",
              "// covers: B1\nfn a(){}\n// covers: B9\nfn z(){}\n")
        if run(plan, [td]) != 1:
            failures.append("dangling tag should fail")

        # 4. no Behaviors section -> 2
        write("## Plan\nno behaviors here\n", "fn a(){}\n")
        if run(plan, [td]) != 2:
            failures.append("missing section should be a usage error")

        # 5. multi-id tag 'covers: B1, B2' -> both covered
        write("## Behaviors\n- B1: x\n- B2: y\n",
              "// covers: B1, B2\nfn a(){}\n")
        if run(plan, [td]) != 0:
            failures.append("multi-id covers tag should cover both")

    if failures:
        print("SELF-TEST FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("self-test: 5/5 cases correct")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(self_test())
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(run(sys.argv[1], sys.argv[2:]))
