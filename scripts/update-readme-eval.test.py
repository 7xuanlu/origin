#!/usr/bin/env python3
"""Unit tests for scripts/update-readme-eval.py."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("update-readme-eval.py")
SPEC = importlib.util.spec_from_file_location("update_readme_eval", SCRIPT)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class UpdateReadmeEvalTests(unittest.TestCase):
    def test_lme_oracle_and_lme_s_rows_are_rendered_without_locomo(self) -> None:
        table = module.build_table(
            {
                "benchmarks": {
                    "longmemeval_oracle": {
                        "label": "LME_Oracle (500 Q)",
                        "recall_at_5": 0.936,
                        "mrr": 0.857,
                        "ndcg_at_10": 0.883,
                    },
                    "longmemeval_s": {
                        "label": "LME_S (deep, 90 Q)",
                        "recall_at_5": 0.8767857142857144,
                        "mrr": 0.8145975056689342,
                        "ndcg_at_10": 0.8223431120728476,
                    },
                    "locomo": {
                        "label": "LoCoMo (locomo10)",
                        "recall_at_5": 0.7,
                        "mrr": 0.647,
                        "ndcg_at_10": 0.684,
                    },
                }
            }
        )

        oracle = "| LME_Oracle (500 Q) | 93.6% | 0.857 | 0.883 |"
        lme_s = "| LME_S (deep, 90 Q) | 87.7% | 0.815 | 0.822 |"
        self.assertIn(oracle, table)
        self.assertIn(lme_s, table)
        self.assertNotIn("LoCoMo", table)
        self.assertLess(table.index(oracle), table.index(lme_s))

    def test_update_tree_refreshes_all_snapshots_without_locomo(self) -> None:
        table = module.build_table(
            {
                "benchmarks": {
                    "longmemeval_oracle": {"recall_at_5": 0.936, "mrr": 0.857, "ndcg_at_10": 0.883},
                    "longmemeval_s": {
                        "recall_at_5": 0.8767857142857144,
                        "mrr": 0.8145975056689342,
                        "ndcg_at_10": 0.8223431120728476,
                    },
                    "locomo": {"recall_at_5": 0.7, "mrr": 0.647, "ndcg_at_10": 0.684},
                }
            }
        )
        old_block = (
            module.START
            + "\n| Benchmark | Recall@5 | MRR | NDCG@10 |\n|---|---:|---:|---:|\n"
            + "| Old | 0.0% | 0.000 | 0.000 |\n"
            + module.END
        )

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "README.md").write_text(f"English\n\n{old_block}\n", encoding="utf-8")
            digest = module.readme_sync_hash(root)
            marker = f"<!-- README_SYNC: source=README.md sha256={digest} -->"
            for rel in module.TRANSLATED_READMES:
                (root / rel).write_text(f"{marker}\n\nTranslated\n\n{old_block}\n", encoding="utf-8")

            changed = module.update_tree(root, table)

            self.assertEqual(changed, 1 + len(module.TRANSLATED_READMES))
            readme_hash = module.readme_sync_hash(root)
            for rel in ("README.md", *module.TRANSLATED_READMES):
                text = (root / rel).read_text(encoding="utf-8")
                self.assertIn(table, text)
                self.assertNotIn("LoCoMo", text)
            for rel in module.TRANSLATED_READMES:
                self.assertIn(
                    f"<!-- README_SYNC: source=README.md sha256={readme_hash} -->",
                    (root / rel).read_text(encoding="utf-8"),
                )

    def test_metrics_source_check_detects_lme_s_summary_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            summary = root / "docs/eval/results/lme_s.summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                """
{
  "retrieval": {
    "recall_at_5": 0.8767857142857144,
    "mrr": 0.8145975056689342,
    "ndcg_at_10": 0.8223431120728476
  }
}
""",
                encoding="utf-8",
            )

            errors = module.validate_source_summaries(
                {
                    "benchmarks": {
                        "longmemeval_s": {
                            "source_summary": "docs/eval/results/lme_s.summary.json",
                            "source_metrics": "retrieval",
                            "recall_at_5": 0.1,
                            "mrr": 0.8145975056689342,
                            "ndcg_at_10": 0.8223431120728476,
                        }
                    }
                },
                root,
            )

            self.assertEqual(
                errors,
                ["longmemeval_s.recall_at_5: 0.1 does not match docs/eval/results/lme_s.summary.json retrieval.recall_at_5 0.8767857142857144"],
            )

    def test_tracked_example_metrics_match_tracked_summaries(self) -> None:
        data = module.load_metrics(module.ROOT / "docs/eval/readme_metrics.example.json")

        self.assertEqual(module.validate_source_summaries(data, module.ROOT), [])


if __name__ == "__main__":
    unittest.main()
