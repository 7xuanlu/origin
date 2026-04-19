// SPDX-License-Identifier: AGPL-3.0-only
//! Eval report formatting — terminal table and JSON output.

use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct EvalReport {
    pub fixture_count: usize,
    pub file_count: usize,
    pub search_mode: String,
    // Primary (BEIR/MTEB headline)
    pub ndcg_at_10: f64,
    // Ranking quality
    pub ndcg_at_5: f64,
    pub map_at_5: f64,
    pub map_at_10: f64,
    pub mrr: f64,
    // Recall gradient
    pub recall_at_1: f64,
    pub recall_at_3: f64,
    pub recall_at_5: f64,
    // Hit rate
    pub hit_rate_at_1: f64,
    pub hit_rate_at_3: f64,
    // Precision
    pub precision_at_3: f64,
    pub precision_at_5: f64,
    // Negative analysis
    pub neg_above_relevant: usize,
    pub total_negatives: usize,
    pub negative_leakage: usize,
    // Quality gate
    #[serde(default)]
    pub gate_content_filtered: usize,
    #[serde(default)]
    pub gate_novelty_filtered: usize,
    // Empty-set precision
    #[serde(default)]
    pub empty_set_count: usize,
    #[serde(default)]
    pub empty_set_false_confidence: Option<f64>,
    #[serde(default)]
    pub score_gap: Option<f64>,
    // Temporal ordering
    #[serde(default)]
    pub temporal_ordering_total: usize,
    #[serde(default)]
    pub temporal_ordering_correct: usize,
    #[serde(default)]
    pub temporal_ordering_rate: Option<f64>,
    // Comparison
    pub baseline: Option<BaselineComparison>,
    pub per_case: Vec<CaseResult>,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct BaselineComparison {
    pub ndcg_at_10: f64,
    pub map_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub neg_above_relevant: usize,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct CaseResult {
    pub query: String,
    pub ndcg_at_10: f64,
    pub ndcg_at_5: f64,
    pub map_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
    pub precision_at_3: f64,
    pub negative_leakage: usize,
    pub neg_above_relevant: usize,
}

impl EvalReport {
    /// Format report as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("Origin Memory Eval\n");
        out.push_str("==================\n");
        out.push_str(&format!(
            "Fixtures: {} cases from {} files\n",
            self.fixture_count, self.file_count
        ));
        out.push_str(&format!("Search mode: {}\n\n", self.search_mode));

        out.push_str(&format!(
            "  NDCG@10:       {:.3}        <- primary\n",
            self.ndcg_at_10
        ));
        out.push_str(&format!("  NDCG@5:        {:.3}\n", self.ndcg_at_5));
        out.push_str(&format!("  MAP@10:        {:.3}\n", self.map_at_10));
        out.push_str(&format!("  MAP@5:         {:.3}\n", self.map_at_5));
        out.push_str(&format!("  MRR:           {:.3}\n", self.mrr));
        out.push_str(&format!("  Recall@1:      {:.3}\n", self.recall_at_1));
        out.push_str(&format!("  Recall@3:      {:.3}\n", self.recall_at_3));
        out.push_str(&format!("  Recall@5:      {:.3}\n", self.recall_at_5));
        out.push_str(&format!("  Hit Rate@1:    {:.3}\n", self.hit_rate_at_1));
        out.push_str(&format!("  Hit Rate@3:    {:.3}\n", self.hit_rate_at_3));
        out.push_str(&format!("  P@3:           {:.3}\n", self.precision_at_3));
        out.push_str(&format!("  P@5:           {:.3}\n", self.precision_at_5));
        out.push_str(&format!(
            "  Neg>relevant:  {}/{}\n",
            self.neg_above_relevant, self.total_negatives
        ));
        if self.gate_content_filtered > 0 || self.gate_novelty_filtered > 0 {
            out.push_str(&format!(
                "  Gate (content): {}\n",
                self.gate_content_filtered
            ));
            out.push_str(&format!(
                "  Gate (novelty): {}\n",
                self.gate_novelty_filtered
            ));
        }

        if self.empty_set_count > 0 {
            out.push_str(&format!(
                "\nEmpty-set precision ({} cases):\n",
                self.empty_set_count
            ));
            if let Some(fc) = self.empty_set_false_confidence {
                out.push_str(&format!("  False confidence: {:.3} (lower = better)\n", fc));
            }
            if let Some(sg) = self.score_gap {
                out.push_str(&format!(
                    "  Score gap:        {:.3} (higher = better)\n",
                    sg
                ));
            }
        }

        if self.temporal_ordering_total > 0 {
            out.push_str(&format!(
                "\nTemporal ordering: {}/{} correct",
                self.temporal_ordering_correct, self.temporal_ordering_total
            ));
            if let Some(rate) = self.temporal_ordering_rate {
                out.push_str(&format!(" (rate={:.3})", rate));
            }
            out.push('\n');
        }

        if let Some(ref b) = self.baseline {
            out.push_str("\nBaseline comparison:\n");
            let delta = |name: &str, old: f64, new: f64| -> String {
                let pct = ((new - old) / old.max(0.001)) * 100.0;
                format!("  {:<12} {:.3} -> {:.3} ({:+.1}%)\n", name, old, new, pct)
            };
            out.push_str(&delta("NDCG@10:", b.ndcg_at_10, self.ndcg_at_10));
            out.push_str(&delta("MAP@10:", b.map_at_10, self.map_at_10));
            out.push_str(&delta("MRR:", b.mrr, self.mrr));
            out.push_str(&delta("Recall@5:", b.recall_at_5, self.recall_at_5));
            out.push_str(&format!(
                "  Neg>rel:     {} -> {}\n",
                b.neg_above_relevant, self.neg_above_relevant
            ));
        }

        // Per-case breakdown sorted by NDCG@10 ascending (worst first)
        if !self.per_case.is_empty() {
            out.push_str("\nPer-case (worst first):\n");
            let mut sorted: Vec<_> = self.per_case.iter().collect();
            sorted.sort_by(|a, b| {
                a.ndcg_at_10
                    .partial_cmp(&b.ndcg_at_10)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for c in sorted {
                let query_short: String = c.query.chars().take(55).collect();
                out.push_str(&format!(
                    "  NDCG={:.2} MAP={:.2} MRR={:.2} R@5={:.2} Nab={} | {}\n",
                    c.ndcg_at_10,
                    c.map_at_10,
                    c.mrr,
                    c.recall_at_5,
                    c.neg_above_relevant,
                    query_short
                ));
            }
        }

        out
    }

    /// Save current metrics as baseline for future comparison.
    pub fn save_baseline(&self, path: &Path) -> Result<(), std::io::Error> {
        let baseline = BaselineComparison {
            ndcg_at_10: self.ndcg_at_10,
            map_at_10: self.map_at_10,
            mrr: self.mrr,
            recall_at_5: self.recall_at_5,
            neg_above_relevant: self.neg_above_relevant,
        };
        let json = serde_json::to_string_pretty(&baseline).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load baseline from a previous run for comparison.
    pub fn load_baseline(path: &Path) -> Option<BaselineComparison> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

// ---------------------------------------------------------------------------
// Shared JSONL history append
// ---------------------------------------------------------------------------

/// A single benchmark run entry for the JSONL history file.
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct BenchmarkHistoryEntry {
    pub timestamp: String,
    pub git_sha: String,
    pub benchmark: String,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
}

/// Append a benchmark run to a JSONL history file.
///
/// Each call writes one JSON line. The file is created if it doesn't exist.
pub fn append_history(path: &Path, entry: &BenchmarkHistoryEntry) -> Result<(), std::io::Error> {
    use std::io::Write;
    let json = serde_json::to_string(entry).map_err(std::io::Error::other)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", json)
}

// ---------------------------------------------------------------------------
// Shared baseline category struct
// ---------------------------------------------------------------------------

/// Per-category baseline metrics shared across benchmark types.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct CategoryBaseline {
    pub name: String,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("baseline.json");

        let report = EvalReport {
            fixture_count: 10,
            file_count: 3,
            search_mode: "search_memory".to_string(),
            ndcg_at_10: 0.95,
            ndcg_at_5: 0.93,
            map_at_5: 0.88,
            map_at_10: 0.90,
            mrr: 0.97,
            recall_at_1: 0.85,
            recall_at_3: 0.92,
            recall_at_5: 0.96,
            hit_rate_at_1: 0.90,
            hit_rate_at_3: 0.95,
            precision_at_3: 0.80,
            precision_at_5: 0.75,
            neg_above_relevant: 5,
            total_negatives: 20,
            negative_leakage: 15,
            gate_content_filtered: 0,
            gate_novelty_filtered: 0,
            empty_set_count: 0,
            empty_set_false_confidence: None,
            score_gap: None,
            temporal_ordering_total: 0,
            temporal_ordering_correct: 0,
            temporal_ordering_rate: None,
            baseline: None,
            per_case: vec![],
        };

        report.save_baseline(&path).unwrap();
        let loaded = EvalReport::load_baseline(&path).unwrap();

        assert!((loaded.ndcg_at_10 - 0.95).abs() < 0.001);
        assert!((loaded.mrr - 0.97).abs() < 0.001);
        assert_eq!(loaded.neg_above_relevant, 5);
    }

    #[test]
    fn test_append_history() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("history.jsonl");

        let entry1 = BenchmarkHistoryEntry {
            timestamp: "2026-04-05T12:00:00Z".to_string(),
            git_sha: "abc1234".to_string(),
            benchmark: "fixtures".to_string(),
            ndcg_at_10: 0.95,
            mrr: 0.90,
            recall_at_5: 0.85,
            hit_rate_at_1: 0.80,
        };
        let entry2 = BenchmarkHistoryEntry {
            timestamp: "2026-04-05T13:00:00Z".to_string(),
            git_sha: "def5678".to_string(),
            benchmark: "locomo".to_string(),
            ndcg_at_10: 0.65,
            mrr: 0.60,
            recall_at_5: 0.55,
            hit_rate_at_1: 0.50,
        };

        append_history(&path, &entry1).unwrap();
        append_history(&path, &entry2).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "expected 2 JSONL lines");

        // Parse each line as valid JSON
        let parsed1: BenchmarkHistoryEntry = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(parsed1.benchmark, "fixtures");
        assert!((parsed1.ndcg_at_10 - 0.95).abs() < 0.001);

        let parsed2: BenchmarkHistoryEntry = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(parsed2.benchmark, "locomo");
        assert_eq!(parsed2.git_sha, "def5678");
    }
}
