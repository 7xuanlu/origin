#!/usr/bin/env python3
"""Update README benchmark snapshots from local gitignored JSON.

Source (gitignored): ${EVAL_BASELINES_DIR:-~/.cache/origin-eval}/readme_metrics.json
Targets: README files with blocks between EVAL_SNAPSHOT_START / EVAL_SNAPSHOT_END
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Final, TypeAlias


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
TRANSLATED_READMES = ("README.zh-Hans.md", "README.zh-Hant.md", "README.es-ES.md")
BASELINES_DIR = Path(
    os.environ.get("EVAL_BASELINES_DIR", str(Path.home() / ".cache" / "origin-eval"))
).expanduser()
METRICS = BASELINES_DIR / "readme_metrics.json"
START = "<!-- EVAL_SNAPSHOT_START -->"
END = "<!-- EVAL_SNAPSHOT_END -->"
METRIC_FIELDS: Final = ("recall_at_5", "mrr", "ndcg_at_10")
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


def score(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}"


def load_metrics(path: Path) -> JsonObject:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def object_at(value: JsonValue | None, context: str) -> JsonObject:
    if isinstance(value, dict):
        return value
    raise ValueError(f"{context}: expected object")


def string_at(value: JsonValue | None, context: str) -> str:
    if isinstance(value, str):
        return value
    raise ValueError(f"{context}: expected string")


def number_at(value: JsonValue | None, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context}: expected number")
    return float(value)


def validate_source_summaries(data: JsonObject, root: Path) -> list[str]:
    errors: list[str] = []
    try:
        benchmarks = object_at(data.get("benchmarks"), "benchmarks")
    except ValueError as exc:
        return [str(exc)]

    for benchmark_key, row_value in benchmarks.items():
        try:
            row = object_at(row_value, benchmark_key)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        source_summary_value = row.get("source_summary")
        if source_summary_value is None:
            continue

        try:
            source_summary = string_at(source_summary_value, f"{benchmark_key}.source_summary")
            source_metrics = string_at(row.get("source_metrics", "retrieval"), f"{benchmark_key}.source_metrics")
            summary_path = root / source_summary
            summary = load_metrics(summary_path)
            source_values = object_at(summary.get(source_metrics), f"{source_summary} {source_metrics}")

            for field in METRIC_FIELDS:
                actual = number_at(row.get(field), f"{benchmark_key}.{field}")
                expected = number_at(source_values.get(field), f"{source_summary} {source_metrics}.{field}")
                if actual != expected:
                    errors.append(
                        f"{benchmark_key}.{field}: {actual} does not match "
                        f"{source_summary} {source_metrics}.{field} {expected}"
                    )
        except (OSError, json.JSONDecodeError, SystemExit, ValueError) as exc:
            errors.append(f"{benchmark_key}: {exc}")

    return errors


def benchmark_rows(data: dict) -> list[dict]:
    benchmarks = data.get("benchmarks", {})
    known = [
        (("longmemeval_oracle", "longmemeval"), "LME_Oracle (500 Q)"),
        (("longmemeval_s",), "LME_S (deep, 90 Q)"),
    ]

    rows = []
    for keys, fallback_label in known:
        key = next((candidate for candidate in keys if candidate in benchmarks), None)
        if key is None:
            continue
        row = dict(benchmarks[key])
        row.setdefault("label", fallback_label)
        rows.append(row)
    return rows


def build_table(data: dict) -> str:
    lines = [
        START,
        "| Benchmark | Recall@5 | MRR | NDCG@10 |",
        "|---|---:|---:|---:|",
    ]
    for row in benchmark_rows(data):
        lines.append(
            f"| {row['label']} | {pct(row.get('recall_at_5'))} | {score(row.get('mrr'))} | "
            f"{score(row.get('ndcg_at_10'))} |"
        )
    lines.append(END)
    return "\n".join(lines)


def replace_snapshot(path: Path, table: str) -> bool:
    text = path.read_text(encoding="utf-8")
    start = text.find(START)
    end = text.find(END)
    if start == -1 or end == -1:
        raise SystemExit(f"{path}: markers not found: EVAL_SNAPSHOT_START / EVAL_SNAPSHOT_END")

    end += len(END)
    updated = text[:start] + table + text[end:]
    if updated == text:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def normalize_generated_snapshot(text: str) -> str:
    start = text.find(START)
    end = text.find(END)
    if start == -1 or end == -1:
        raise SystemExit("README markers not found: EVAL_SNAPSHOT_START / EVAL_SNAPSHOT_END")
    end += len(END)
    return text[:start] + START + "\n" + END + text[end:]


def readme_sync_hash(root: Path) -> str:
    text = (root / "README.md").read_text(encoding="utf-8")
    normalized = normalize_generated_snapshot(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def update_tree(root: Path, table: str) -> int:
    changed = int(replace_snapshot(root / "README.md", table))

    for rel in TRANSLATED_READMES:
        path = root / rel
        if not path.exists():
            continue
        changed += int(replace_snapshot(path, table))

    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="validate benchmark rows against tracked source summaries")
    args = parser.parse_args()

    if args.check:
        data = load_metrics(args.check)
        errors = validate_source_summaries(data, ROOT)
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise SystemExit(1)
        print(f"Metrics source check passed: {args.check}")
        return

    if not METRICS.exists():
        raise SystemExit(
            f"Missing local metrics file: {METRICS}\n"
            "Create it from docs/eval/readme_metrics.example.json first."
        )

    data = load_metrics(METRICS)
    errors = validate_source_summaries(data, ROOT)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)

    table = build_table(data)
    changed = update_tree(ROOT, table)
    print(f"Updated {changed} README file(s) from {METRICS}")


if __name__ == "__main__":
    main()
