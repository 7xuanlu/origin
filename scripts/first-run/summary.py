#!/usr/bin/env python3
"""Merge first-run gauntlet findings into GitHub Markdown.

    python3 summary.py <dir>

Reads every findings.tsv under <dir> (recursively). Each artifact is merged
into its own subfolder (findings-<job>-<os> or similar), so the folder that
holds the TSV is the run label. Prints one table (Channel | Label | PASS |
FAIL | INFO | seconds-to-health | Worst) and a "### Findings" section that
lists every FAIL row, then every INFO row. Always exits 0; no third-party
dependencies.
"""

import sys
from collections import defaultdict
from pathlib import Path

DETAIL_CAP = 300


def load(root):
    """Yield (label, channel, name, status, rc, detail) for every TSV row."""
    for tsv in sorted(root.rglob("findings.tsv")):
        label = tsv.parent.name if tsv.parent != root else root.name
        for line in tsv.read_text(encoding="utf-8", errors="replace").splitlines():
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            channel, name, status, rc = cols[0], cols[1], cols[2], cols[3]
            detail = cols[4] if len(cols) > 4 else ""
            yield label, channel, name, status, rc, detail


def md_escape(text):
    return text.replace("`", "\\`")


def main():
    if len(sys.argv) != 2:
        print("usage: summary.py <dir>", file=sys.stderr)
        return 0
    root = Path(sys.argv[1])
    rows = list(load(root))
    if not rows:
        print(f"_No findings.tsv found under `{md_escape(str(root))}`._")
        return 0

    counts = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "INFO": 0})
    health = {}
    for label, channel, name, status, rc, detail in rows:
        key = (channel, label)
        if status in counts[key]:
            counts[key][status] += 1
        if status == "INFO" and name == "seconds-to-health":
            health[key] = detail

    print("| Channel | Label | PASS | FAIL | INFO | seconds-to-health | Worst |")
    print("|---|---|---:|---:|---:|---:|---|")
    for channel, label in sorted(counts):
        c = counts[(channel, label)]
        worst = "FAIL" if c["FAIL"] else ("INFO" if c["INFO"] else "PASS")
        print(f"| {channel} | {label} | {c['PASS']} | {c['FAIL']} | {c['INFO']} | "
              f"{health.get((channel, label), '-')} | {worst} |")

    print()
    print("### Findings")
    listed = 0
    for want in ("FAIL", "INFO"):
        for label, channel, name, status, rc, detail in rows:
            if status != want:
                continue
            if status == "INFO" and name == "seconds-to-health":
                continue  # already a table column
            detail = detail[:DETAIL_CAP] + ("…" if len(detail) > DETAIL_CAP else "")
            print(f"- **{channel} / {name}** ({label}, rc={rc}): {md_escape(detail)}")
            listed += 1
    if not listed:
        print("- none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
