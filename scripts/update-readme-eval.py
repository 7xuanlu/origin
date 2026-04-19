#!/usr/bin/env python3
"""Update README benchmark snapshot from local gitignored JSON.

Source (gitignored): app/eval/baselines/readme_metrics.json
Target: README.md block between EVAL_SNAPSHOT_START / EVAL_SNAPSHOT_END
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
METRICS = ROOT / "app" / "eval" / "baselines" / "readme_metrics.json"
START = "<!-- EVAL_SNAPSHOT_START -->"
END = "<!-- EVAL_SNAPSHOT_END -->"


def pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def score(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}"


def build_table(data: dict) -> str:
    rows = data.get("benchmarks", {})
    longmem = rows.get("longmemeval", {})
    locomo = rows.get("locomo_plus", {})
    life = rows.get("lifebench", {})

    lines = [
        START,
        "| Benchmark | Recall@5 | MRR | NDCG@10 | Notes |",
        "|---|---:|---:|---:|---|",
        (
            f"| LongMemEval | **{pct(longmem.get('recall_at_5'))}** | "
            f"{score(longmem.get('mrr'))} | {score(longmem.get('ndcg_at_10'))} | "
            f"{longmem.get('notes', '-')}"
            " |"
        ),
        (
            f"| LoCoMo-Plus | {pct(locomo.get('recall_at_5'))} | "
            f"{score(locomo.get('mrr'))} | {score(locomo.get('ndcg_at_10'))} | "
            f"{locomo.get('notes', '-')}"
            " |"
        ),
        (
            f"| LifeBench | {pct(life.get('recall_at_5'))} | "
            f"{score(life.get('mrr'))} | {score(life.get('ndcg_at_10'))} | "
            f"{life.get('notes', '-')}"
            " |"
        ),
        END,
    ]
    return "\n".join(lines)


def main() -> None:
    if not METRICS.exists():
        raise SystemExit(
            f"Missing local metrics file: {METRICS}\n"
            "Create it from docs/eval/readme_metrics.example.json first."
        )

    data = json.loads(METRICS.read_text(encoding="utf-8"))
    readme = README.read_text(encoding="utf-8")
    start = readme.find(START)
    end = readme.find(END)
    if start == -1 or end == -1:
        raise SystemExit("README markers not found: EVAL_SNAPSHOT_START / EVAL_SNAPSHOT_END")

    end += len(END)
    table = build_table(data)
    updated = readme[:start] + table + readme[end:]
    README.write_text(updated, encoding="utf-8")
    print(f"Updated {README} from {METRICS}")


if __name__ == "__main__":
    main()
