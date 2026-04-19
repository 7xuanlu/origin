// SPDX-License-Identifier: AGPL-3.0-only
//! LoCoMo-Plus benchmark adapter — cognitive memory evaluation.
//!
//! LoCoMo-Plus (arXiv:2602.10715) tests "Level-2 Cognitive Memory" —
//! implicit constraints (goals, values, causal context, state) that are
//! not explicitly queried. Four constraint types map to Origin memory types:
//!
//!   causal → decision, state → identity, goal → goal, value → preference
//!
//! Dataset: <https://github.com/xjtuleeyf/Locomo-Plus>
//!
//! Schema: 401 flat cue-trigger pairs. Each sample has a `cue_dialogue` (the
//! memory to seed) and a `trigger_query` (what the user says later — the search
//! query). The benchmark seeds ALL 401 cue_dialogues into one ephemeral DB, then
//! for each sample searches with trigger_query and checks where the matching cue
//! ranks. Tests semantic disconnect — the trigger doesn't explicitly reference the cue.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::metrics;
use crate::sources::RawDocument;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const CONSTRAINT_ORDER: &[&str] = &["causal", "state", "goal", "value", "factual"];

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LocomoPlusSample {
    pub relation_type: String,
    pub cue_dialogue: String,
    pub trigger_query: String,
    #[serde(default)]
    pub time_gap: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub scores: Option<HashMap<String, f64>>,
    #[serde(default)]
    pub ranks: Option<HashMap<String, u64>>,
    #[serde(default)]
    pub final_similarity_score: Option<f64>,
}

// ---------------------------------------------------------------------------
// Constraint mapping
// ---------------------------------------------------------------------------

/// Map a LoCoMo-Plus constraint type to an Origin memory type.
pub fn constraint_to_memory_type(constraint: &str) -> &'static str {
    match constraint {
        "causal" => "decision",
        "state" => "identity",
        "goal" => "goal",
        "value" => "preference",
        _ => "fact",
    }
}

/// Short code for a constraint type used in reporting.
pub fn constraint_code(constraint: &str) -> &'static str {
    match constraint {
        "causal" => "CAU",
        "state" => "STA",
        "goal" => "GOA",
        "value" => "VAL",
        "factual" => "FAC",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load LoCoMo-Plus dataset from a local JSON file.
pub fn load_locomo_plus(path: &Path) -> Result<Vec<LocomoPlusSample>, OriginError> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| OriginError::Generic(format!("Failed to read LoCoMo-Plus file: {e}")))?;
    let samples: Vec<LocomoPlusSample> = serde_json::from_str(&data)
        .map_err(|e| OriginError::Generic(format!("Failed to parse LoCoMo-Plus JSON: {e}")))?;
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Report structs
// ---------------------------------------------------------------------------

/// Baseline metrics for LoCoMo-Plus benchmark comparison across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocomoPlusBaseline {
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
    pub per_category: Vec<crate::eval::report::CategoryBaseline>,
}

/// Per-constraint-type results for LoCoMo-Plus.
#[derive(Debug, Clone, Serialize)]
pub struct LocomoPlusConstraintResult {
    pub constraint_type: String,    // causal, state, goal, value
    pub origin_memory_type: String, // decision, identity, goal, preference
    pub code: String,               // CAU, STA, GOA, VAL
    pub count: usize,
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
    /// Mean BGE rank from the dataset (lower is better; 1 = top result)
    pub baseline_bge_mean_rank: Option<f64>,
}

/// Full LoCoMo-Plus benchmark report.
#[derive(Debug, Clone, Serialize)]
pub struct LocomoPlusReport {
    pub aggregate_ndcg_at_10: f64,
    pub aggregate_mrr: f64,
    pub aggregate_recall_at_5: f64,
    pub aggregate_hit_rate_at_1: f64,
    pub total_questions: usize,
    pub total_memories: usize,
    pub per_constraint: Vec<LocomoPlusConstraintResult>,
    /// Hook for LLM-as-judge (None for now).
    pub constraint_consistency: Option<f64>,
    /// Mean BGE rank from the dataset across all samples (lower is better)
    pub baseline_bge_mean_rank: Option<f64>,
    /// Baseline comparison from a previous run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<LocomoPlusBaseline>,
}

impl LocomoPlusReport {
    /// Format as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("LoCoMo-Plus Benchmark (Cognitive Memory)\n");
        out.push_str("=========================================\n");
        out.push_str(&format!(
            "Total cue-trigger pairs: {}\n",
            self.total_questions
        ));
        out.push_str(&format!(
            "Memories seeded:         {}\n\n",
            self.total_memories
        ));

        out.push_str("Origin retrieval scores:\n");
        out.push_str(&format!(
            "  NDCG@10:     {:.4}  <- primary\n",
            self.aggregate_ndcg_at_10
        ));
        out.push_str(&format!("  MRR:         {:.4}\n", self.aggregate_mrr));
        out.push_str(&format!(
            "  Recall@5:    {:.4}\n",
            self.aggregate_recall_at_5
        ));
        out.push_str(&format!(
            "  Hit Rate@1:  {:.4}\n",
            self.aggregate_hit_rate_at_1
        ));

        if let Some(bge) = self.baseline_bge_mean_rank {
            out.push_str(&format!(
                "\nBaseline BGE mean rank (dataset): {:.1} (out of 401, lower=better)\n",
                bge
            ));
            out.push_str("(Note: dataset rank is retrieval position among ALL documents in the dataset's original index)\n");
        }

        if let Some(ref b) = self.baseline {
            out.push_str("\nBaseline comparison:\n");
            let delta = |name: &str, old: f64, new: f64| -> String {
                let pct = ((new - old) / old.max(0.001)) * 100.0;
                format!("  {:<12} {:.3} -> {:.3} ({:+.1}%)\n", name, old, new, pct)
            };
            out.push_str(&delta("NDCG@10:", b.ndcg_at_10, self.aggregate_ndcg_at_10));
            out.push_str(&delta("MRR:", b.mrr, self.aggregate_mrr));
            out.push_str(&delta(
                "Recall@5:",
                b.recall_at_5,
                self.aggregate_recall_at_5,
            ));
            out.push_str(&delta(
                "HR@1:",
                b.hit_rate_at_1,
                self.aggregate_hit_rate_at_1,
            ));

            if !b.per_category.is_empty() {
                out.push_str("  Per-category:\n");
                for cat_bl in &b.per_category {
                    if let Some(cat_new) = self
                        .per_constraint
                        .iter()
                        .find(|c| c.constraint_type == cat_bl.name)
                    {
                        let pct = ((cat_new.ndcg_at_10 - cat_bl.ndcg_at_10)
                            / cat_bl.ndcg_at_10.max(0.001))
                            * 100.0;
                        out.push_str(&format!(
                            "    {}: {:.3} -> {:.3} ({:+.1}%)\n",
                            cat_bl.name, cat_bl.ndcg_at_10, cat_new.ndcg_at_10, pct
                        ));
                    }
                }
            }
        }

        if let Some(cc) = self.constraint_consistency {
            out.push_str(&format!("  Constraint Consistency: {:.4}\n", cc));
        }

        out.push('\n');
        out.push_str("Per constraint:\n");
        out.push_str(&format!(
            "  {:3} {:8} ({:10}) (n={:>3}): NDCG@10  MRR     R@5     HR@1    BGE-rank\n",
            "cod", "type", "origin-type", ""
        ));
        out.push_str(&format!(
            "  {:-<3} {:-<8} ({:-<10}) ({:-<3}): {:-<8} {:-<8} {:-<8} {:-<8} {:-<9}\n",
            "", "", "", "", "", "", "", "", ""
        ));
        for c in &self.per_constraint {
            let bge_str = c
                .baseline_bge_mean_rank
                .map(|r| format!("{:.1}", r))
                .unwrap_or_else(|| "n/a".to_string());
            out.push_str(&format!(
                "  {} {:8} ({:10}) (n={:>3}): {:.4}   {:.4}   {:.4}   {:.4}   {}\n",
                c.code,
                c.constraint_type,
                c.origin_memory_type,
                c.count,
                c.ndcg_at_10,
                c.mrr,
                c.recall_at_5,
                c.hit_rate_at_1,
                bge_str,
            ));
        }
        out
    }

    /// Save current metrics as baseline for future comparison.
    pub fn save_baseline(&self, path: &Path) -> Result<(), std::io::Error> {
        let per_category: Vec<crate::eval::report::CategoryBaseline> = self
            .per_constraint
            .iter()
            .map(|c| crate::eval::report::CategoryBaseline {
                name: c.constraint_type.clone(),
                ndcg_at_10: c.ndcg_at_10,
                mrr: c.mrr,
                recall_at_5: c.recall_at_5,
            })
            .collect();
        let baseline = LocomoPlusBaseline {
            ndcg_at_10: self.aggregate_ndcg_at_10,
            mrr: self.aggregate_mrr,
            recall_at_5: self.aggregate_recall_at_5,
            hit_rate_at_1: self.aggregate_hit_rate_at_1,
            per_category,
        };
        let json = serde_json::to_string_pretty(&baseline).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load baseline from a previous run for comparison.
    pub fn load_baseline(path: &Path) -> Option<LocomoPlusBaseline> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Average a field across a score slice.
fn avg_field(
    scores: &[(String, f64, f64, f64, f64)],
    f: impl Fn(&(String, f64, f64, f64, f64)) -> f64,
) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let sum: f64 = scores.iter().map(&f).sum();
    sum / scores.len() as f64
}

/// Aggregate scores by constraint_type.
fn aggregate_by_constraint(
    scores: &[(String, f64, f64, f64, f64)],
    samples: &[LocomoPlusSample],
) -> Vec<LocomoPlusConstraintResult> {
    let mut results = Vec::new();
    for &constraint in CONSTRAINT_ORDER {
        let cat_scores: Vec<_> = scores
            .iter()
            .filter(|s| s.0 == constraint)
            .cloned()
            .collect();
        if cat_scores.is_empty() {
            continue;
        }

        // Compute mean BGE rank for this constraint from the dataset
        let bge_ranks: Vec<f64> = samples
            .iter()
            .filter(|s| s.relation_type == constraint)
            .filter_map(|s| s.ranks.as_ref()?.get("bge").map(|&r| r as f64))
            .collect();
        let baseline_bge_mean_rank = if bge_ranks.is_empty() {
            None
        } else {
            Some(bge_ranks.iter().sum::<f64>() / bge_ranks.len() as f64)
        };

        // ndcg_10=s.1, mrr=s.2, recall_5=s.3, hr_1=s.4
        results.push(LocomoPlusConstraintResult {
            constraint_type: constraint.to_string(),
            origin_memory_type: constraint_to_memory_type(constraint).to_string(),
            code: constraint_code(constraint).to_string(),
            count: cat_scores.len(),
            ndcg_at_5: avg_field(&cat_scores, |s| s.1), // using ndcg_10 as proxy for ndcg_5 slot
            ndcg_at_10: avg_field(&cat_scores, |s| s.1),
            mrr: avg_field(&cat_scores, |s| s.2),
            recall_at_5: avg_field(&cat_scores, |s| s.3),
            hit_rate_at_1: avg_field(&cat_scores, |s| s.4),
            baseline_bge_mean_rank,
        });
    }
    results
}

// ---------------------------------------------------------------------------
// End-to-end benchmark runner
// ---------------------------------------------------------------------------

/// Run LoCoMo-Plus benchmark.
///
/// Seeds ALL 401 cue_dialogues into one ephemeral DB, then for each sample
/// searches with trigger_query and checks where the matching cue ranks.
/// This tests semantic disconnect — the trigger doesn't explicitly reference the cue.
pub async fn run_locomo_plus_eval(
    path: &Path,
    scoring: Option<&crate::tuning::SearchScoringConfig>,
) -> Result<LocomoPlusReport, OriginError> {
    let samples = load_locomo_plus(path)?;
    let total_memories = samples.len();

    // Create ONE ephemeral DB and seed ALL cue_dialogues
    let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
    let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

    let docs: Vec<RawDocument> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| RawDocument {
            content: s.cue_dialogue.clone(),
            source_id: format!("lcp_{}", i),
            memory_type: Some(constraint_to_memory_type(&s.relation_type).to_string()),
            domain: Some("conversation".to_string()),
            source: "memory".to_string(),
            title: format!("{} cue {}", s.relation_type, i),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        })
        .collect();
    db.upsert_documents(docs).await?;

    // (relation_type, ndcg_10, mrr, recall_5, hr_1)
    let mut all_scores: Vec<(String, f64, f64, f64, f64)> = Vec::new();

    for (i, sample) in samples.iter().enumerate() {
        let target_id = format!("lcp_{}", i);

        let results = db
            .search_memory(
                &sample.trigger_query,
                10,
                None,
                None,
                None,
                None,
                None,
                scoring,
            )
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        let relevant: HashSet<&str> = [target_id.as_str()].into_iter().collect();
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| (*id, if *id == target_id.as_str() { 1u8 } else { 0u8 }))
            .collect();

        let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
        let mrr_val = metrics::mrr(&result_ids, &relevant);
        let recall_5 = metrics::recall_at_k(&result_ids, &relevant, 5);
        let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant, 1);

        all_scores.push((
            sample.relation_type.clone(),
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    // Compute overall baseline mean BGE rank from dataset
    let bge_ranks: Vec<f64> = samples
        .iter()
        .filter_map(|s| s.ranks.as_ref()?.get("bge").map(|&r| r as f64))
        .collect();
    let baseline_bge_mean_rank = if bge_ranks.is_empty() {
        None
    } else {
        Some(bge_ranks.iter().sum::<f64>() / bge_ranks.len() as f64)
    };

    // Aggregate per-constraint
    let per_constraint = aggregate_by_constraint(&all_scores, &samples);

    Ok(LocomoPlusReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.1),
        aggregate_mrr: avg_field(&all_scores, |s| s.2),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.3),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.4),
        total_questions: all_scores.len(),
        total_memories,
        per_constraint,
        constraint_consistency: None,
        baseline_bge_mean_rank,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_locomo_plus_sample_real_schema() {
        let json = r#"[{
            "relation_type": "causal",
            "cue_dialogue": "A: After learning to say 'no', I've felt a lot less stressed overall.\nB: That's a great skill to develop; protecting your time is important.",
            "trigger_query": "A: I ended up volunteering for that project, and now I'm totally overwhelmed.",
            "time_gap": "two weeks later",
            "model_name": "gpt-4o-mini",
            "scores": {"mpnet": 0.4, "bge": 0.571, "bm25": 1.521, "combined": 0.509},
            "ranks": {"mpnet": 262, "bge": 235, "bm25": 33, "combined": 20},
            "final_similarity_score": 0.509
        }]"#;

        let samples: Vec<LocomoPlusSample> = serde_json::from_str(json).unwrap();
        assert_eq!(samples.len(), 1);

        let s = &samples[0];
        assert_eq!(s.relation_type, "causal");
        assert!(s.cue_dialogue.contains("learning to say 'no'"));
        assert!(s.trigger_query.contains("volunteering for that project"));
        assert_eq!(s.time_gap.as_deref(), Some("two weeks later"));
        assert_eq!(s.model_name.as_deref(), Some("gpt-4o-mini"));

        let ranks = s.ranks.as_ref().unwrap();
        assert_eq!(ranks["bge"], 235);

        let scores = s.scores.as_ref().unwrap();
        assert!((scores["combined"] - 0.509).abs() < 0.001);

        assert!((s.final_similarity_score.unwrap() - 0.509).abs() < 0.001);
    }

    #[test]
    fn test_parse_locomo_plus_sample_minimal() {
        // Minimal schema — optional fields absent
        let json = r#"[{
            "relation_type": "goal",
            "cue_dialogue": "A: I really want to learn Spanish this year.\nB: That's a great goal!",
            "trigger_query": "A: I bought some Spanish textbooks today."
        }]"#;

        let samples: Vec<LocomoPlusSample> = serde_json::from_str(json).unwrap();
        assert_eq!(samples.len(), 1);

        let s = &samples[0];
        assert_eq!(s.relation_type, "goal");
        assert!(s.time_gap.is_none());
        assert!(s.model_name.is_none());
        assert!(s.scores.is_none());
        assert!(s.ranks.is_none());
        assert!(s.final_similarity_score.is_none());
    }

    #[test]
    fn test_constraint_to_memory_type() {
        assert_eq!(constraint_to_memory_type("causal"), "decision");
        assert_eq!(constraint_to_memory_type("state"), "identity");
        assert_eq!(constraint_to_memory_type("goal"), "goal");
        assert_eq!(constraint_to_memory_type("value"), "preference");
        // Fallback for unknown/factual
        assert_eq!(constraint_to_memory_type("factual"), "fact");
        assert_eq!(constraint_to_memory_type("unknown"), "fact");
    }

    #[test]
    fn test_constraint_code() {
        assert_eq!(constraint_code("causal"), "CAU");
        assert_eq!(constraint_code("state"), "STA");
        assert_eq!(constraint_code("goal"), "GOA");
        assert_eq!(constraint_code("value"), "VAL");
        assert_eq!(constraint_code("factual"), "FAC");
        // Fallback
        assert_eq!(constraint_code("unknown"), "?");
    }

    #[test]
    fn test_report_to_terminal_new_schema() {
        let report = LocomoPlusReport {
            aggregate_ndcg_at_10: 0.712,
            aggregate_mrr: 0.634,
            aggregate_recall_at_5: 0.580,
            aggregate_hit_rate_at_1: 0.501,
            total_questions: 401,
            total_memories: 401,
            per_constraint: vec![
                LocomoPlusConstraintResult {
                    constraint_type: "causal".to_string(),
                    origin_memory_type: "decision".to_string(),
                    code: "CAU".to_string(),
                    count: 101,
                    ndcg_at_5: 0.700,
                    ndcg_at_10: 0.720,
                    mrr: 0.750,
                    recall_at_5: 0.590,
                    hit_rate_at_1: 0.510,
                    baseline_bge_mean_rank: Some(235.5),
                },
                LocomoPlusConstraintResult {
                    constraint_type: "goal".to_string(),
                    origin_memory_type: "goal".to_string(),
                    code: "GOA".to_string(),
                    count: 100,
                    ndcg_at_5: 0.680,
                    ndcg_at_10: 0.700,
                    mrr: 0.620,
                    recall_at_5: 0.560,
                    hit_rate_at_1: 0.490,
                    baseline_bge_mean_rank: Some(190.2),
                },
            ],
            constraint_consistency: None,
            baseline_bge_mean_rank: Some(212.8),
            baseline: None,
        };

        let output = report.to_terminal();

        assert!(output.contains("LoCoMo-Plus Benchmark"), "missing header");
        assert!(output.contains("NDCG@10"), "missing NDCG@10 label");
        assert!(output.contains("CAU"), "missing CAU code");
        assert!(output.contains("causal"), "missing causal constraint type");
        assert!(output.contains("decision"), "missing decision memory type");
        assert!(output.contains("GOA"), "missing GOA code");
        assert!(output.contains("401"), "missing question count");
        assert!(output.contains("212.8"), "missing baseline BGE mean rank");
        assert!(output.contains("235.5"), "missing per-constraint BGE rank");
    }

    #[test]
    fn test_source_id_format() {
        // Verify source_id generation for a few indexes
        for i in 0..5usize {
            let id = format!("lcp_{}", i);
            assert!(id.starts_with("lcp_"), "id should start with lcp_");
        }
        assert_eq!(format!("lcp_{}", 0), "lcp_0");
        assert_eq!(format!("lcp_{}", 400), "lcp_400");
    }

    #[test]
    fn test_baseline_bge_mean_rank_computed() {
        // Simulate computing baseline mean rank from samples
        let samples = [
            LocomoPlusSample {
                relation_type: "causal".to_string(),
                cue_dialogue: "A: X\nB: Y".to_string(),
                trigger_query: "A: Z".to_string(),
                time_gap: None,
                model_name: None,
                scores: None,
                ranks: Some({
                    let mut m = HashMap::new();
                    m.insert("bge".to_string(), 100u64);
                    m
                }),
                final_similarity_score: None,
            },
            LocomoPlusSample {
                relation_type: "causal".to_string(),
                cue_dialogue: "A: A\nB: B".to_string(),
                trigger_query: "A: C".to_string(),
                time_gap: None,
                model_name: None,
                scores: None,
                ranks: Some({
                    let mut m = HashMap::new();
                    m.insert("bge".to_string(), 200u64);
                    m
                }),
                final_similarity_score: None,
            },
        ];

        let bge_ranks: Vec<f64> = samples
            .iter()
            .filter_map(|s| s.ranks.as_ref()?.get("bge").map(|&r| r as f64))
            .collect();
        let mean = bge_ranks.iter().sum::<f64>() / bge_ranks.len() as f64;
        assert!(
            (mean - 150.0).abs() < 0.01,
            "expected mean rank 150.0, got {}",
            mean
        );
    }

    #[test]
    fn test_baseline_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locomo_plus_baseline.json");

        let report = LocomoPlusReport {
            aggregate_ndcg_at_10: 0.712,
            aggregate_mrr: 0.634,
            aggregate_recall_at_5: 0.580,
            aggregate_hit_rate_at_1: 0.501,
            total_questions: 401,
            total_memories: 401,
            per_constraint: vec![
                LocomoPlusConstraintResult {
                    constraint_type: "causal".to_string(),
                    origin_memory_type: "decision".to_string(),
                    code: "CAU".to_string(),
                    count: 101,
                    ndcg_at_5: 0.700,
                    ndcg_at_10: 0.720,
                    mrr: 0.750,
                    recall_at_5: 0.590,
                    hit_rate_at_1: 0.510,
                    baseline_bge_mean_rank: Some(235.5),
                },
                LocomoPlusConstraintResult {
                    constraint_type: "goal".to_string(),
                    origin_memory_type: "goal".to_string(),
                    code: "GOA".to_string(),
                    count: 100,
                    ndcg_at_5: 0.680,
                    ndcg_at_10: 0.700,
                    mrr: 0.620,
                    recall_at_5: 0.560,
                    hit_rate_at_1: 0.490,
                    baseline_bge_mean_rank: Some(190.2),
                },
            ],
            constraint_consistency: None,
            baseline_bge_mean_rank: Some(212.8),
            baseline: None,
        };

        report.save_baseline(&path).unwrap();
        let loaded = LocomoPlusReport::load_baseline(&path).unwrap();

        assert!((loaded.ndcg_at_10 - 0.712).abs() < 0.001);
        assert!((loaded.mrr - 0.634).abs() < 0.001);
        assert!((loaded.recall_at_5 - 0.580).abs() < 0.001);
        assert!((loaded.hit_rate_at_1 - 0.501).abs() < 0.001);

        // Per-category baselines
        assert_eq!(loaded.per_category.len(), 2);
        assert_eq!(loaded.per_category[0].name, "causal");
        assert!((loaded.per_category[0].ndcg_at_10 - 0.720).abs() < 0.001);
        assert_eq!(loaded.per_category[1].name, "goal");
        assert!((loaded.per_category[1].mrr - 0.620).abs() < 0.001);
    }

    #[test]
    fn test_to_terminal_with_baseline() {
        let report = LocomoPlusReport {
            aggregate_ndcg_at_10: 0.750,
            aggregate_mrr: 0.670,
            aggregate_recall_at_5: 0.610,
            aggregate_hit_rate_at_1: 0.530,
            total_questions: 401,
            total_memories: 401,
            per_constraint: vec![LocomoPlusConstraintResult {
                constraint_type: "causal".to_string(),
                origin_memory_type: "decision".to_string(),
                code: "CAU".to_string(),
                count: 101,
                ndcg_at_5: 0.740,
                ndcg_at_10: 0.760,
                mrr: 0.680,
                recall_at_5: 0.620,
                hit_rate_at_1: 0.540,
                baseline_bge_mean_rank: None,
            }],
            constraint_consistency: None,
            baseline_bge_mean_rank: None,
            baseline: Some(LocomoPlusBaseline {
                ndcg_at_10: 0.712,
                mrr: 0.634,
                recall_at_5: 0.580,
                hit_rate_at_1: 0.501,
                per_category: vec![crate::eval::report::CategoryBaseline {
                    name: "causal".to_string(),
                    ndcg_at_10: 0.720,
                    mrr: 0.750,
                    recall_at_5: 0.590,
                }],
            }),
        };

        let text = report.to_terminal();
        assert!(text.contains("LoCoMo-Plus Benchmark"));
        assert!(text.contains("Baseline comparison:"));
        assert!(text.contains("->"));
        assert!(text.contains("causal"));
    }

    /// Full integration test against the real dataset.
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_run_real -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_run_real() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo_plus.json");
        if !path.exists() {
            println!("SKIP: dataset not found at {}", path.display());
            return;
        }
        let report = run_locomo_plus_eval(&path, None).await.unwrap();
        println!("\n{}", report.to_terminal());
    }

    /// Autoresearch: scan retrieval cue prefix formats on LoCoMo-Plus.
    /// Tests whether the [type | domain] prefix helps or hurts cognitive memory retrieval.
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_prefix_ablation -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_prefix_ablation() {
        use crate::db::MemoryDB;
        use crate::sources::RawDocument;

        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo_plus.json");
        if !path.exists() {
            println!("SKIP: dataset not found");
            return;
        }
        let samples = super::load_locomo_plus(&path).unwrap();

        // Define prefix strategies by varying what metadata is set on the RawDocument
        struct PrefixStrategy {
            name: &'static str,
            set_memory_type: bool,
            set_domain: bool,
            domain_value: Option<&'static str>,
        }

        let strategies = [
            PrefixStrategy {
                name: "no prefix (raw content)",
                set_memory_type: false,
                set_domain: false,
                domain_value: None,
            },
            PrefixStrategy {
                name: "[type] only",
                set_memory_type: true,
                set_domain: false,
                domain_value: None,
            },
            PrefixStrategy {
                name: "[domain] only",
                set_memory_type: false,
                set_domain: true,
                domain_value: Some("conversation"),
            },
            PrefixStrategy {
                name: "[type | conversation]",
                set_memory_type: true,
                set_domain: true,
                domain_value: Some("conversation"),
            },
            PrefixStrategy {
                name: "[type | personal]",
                set_memory_type: true,
                set_domain: true,
                domain_value: Some("personal"),
            },
            PrefixStrategy {
                name: "[type | memory]",
                set_memory_type: true,
                set_domain: true,
                domain_value: Some("memory"),
            },
        ];

        println!("\n{}", "=".repeat(80));
        println!("  AUTORESEARCH: Retrieval Cue Prefix Ablation (LoCoMo-Plus, 401 pairs)");
        println!("{}\n", "=".repeat(80));
        println!(
            "{:<30} {:>10} {:>8} {:>8} {:>8}",
            "Prefix Format", "NDCG@10", "MRR", "R@5", "HR@1"
        );
        println!("{}", "-".repeat(70));

        let mut best_ndcg = 0.0f64;
        let mut best_name = "";

        for strategy in &strategies {
            let tmp = tempfile::tempdir().unwrap();
            let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter))
                .await
                .unwrap();

            let docs: Vec<RawDocument> = samples
                .iter()
                .enumerate()
                .map(|(i, s)| RawDocument {
                    content: s.cue_dialogue.clone(),
                    source_id: format!("lcp_{}", i),
                    memory_type: if strategy.set_memory_type {
                        Some(super::constraint_to_memory_type(&s.relation_type).to_string())
                    } else {
                        None
                    },
                    domain: if strategy.set_domain {
                        strategy.domain_value.map(|v| v.to_string())
                    } else {
                        None
                    },
                    source: "memory".to_string(),
                    title: format!("{} cue {}", s.relation_type, i),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                })
                .collect();
            db.upsert_documents(docs).await.unwrap();

            // Run search for each sample
            let mut all_ndcg = Vec::new();
            let mut all_mrr = Vec::new();
            let mut all_recall5 = Vec::new();
            let mut all_hr1 = Vec::new();

            for (i, sample) in samples.iter().enumerate() {
                let target_id = format!("lcp_{}", i);
                let results = db
                    .search_memory(
                        &sample.trigger_query,
                        10,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .await
                    .unwrap();
                let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

                let relevant: std::collections::HashSet<&str> =
                    [target_id.as_str()].into_iter().collect();
                let grades: std::collections::HashMap<&str, u8> = result_ids
                    .iter()
                    .map(|id| (*id, if *id == target_id.as_str() { 1u8 } else { 0u8 }))
                    .collect();

                all_ndcg.push(crate::eval::metrics::ndcg_at_k(&result_ids, &grades, 10));
                all_mrr.push(crate::eval::metrics::mrr(&result_ids, &relevant));
                all_recall5.push(crate::eval::metrics::recall_at_k(&result_ids, &relevant, 5));
                all_hr1.push(crate::eval::metrics::hit_rate_at_k(
                    &result_ids,
                    &relevant,
                    1,
                ));
            }

            let n = samples.len() as f64;
            let ndcg = all_ndcg.iter().sum::<f64>() / n;
            let mrr = all_mrr.iter().sum::<f64>() / n;
            let r5 = all_recall5.iter().sum::<f64>() / n;
            let hr1 = all_hr1.iter().sum::<f64>() / n;

            let marker = if ndcg > best_ndcg + 0.001 {
                " <-- NEW BEST"
            } else {
                ""
            };
            println!(
                "{:<30} {:>10.4} {:>8.4} {:>8.4} {:>8.4}{}",
                strategy.name, ndcg, mrr, r5, hr1, marker
            );

            if ndcg > best_ndcg {
                best_ndcg = ndcg;
                best_name = strategy.name;
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("  BEST: {} (NDCG@10 = {:.4})", best_name, best_ndcg);
        println!("{}", "=".repeat(80));
    }

    /// Autoresearch: scan scoring parameters across ALL benchmarks.
    /// Runs fixture evals (confirmation_boost, recap_penalty, rrf_k) + LoCoMo-Plus (rrf_k).
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_autoresearch_scan -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_autoresearch_scan() {
        use crate::eval::runner::{run_eval, GateMode};
        use crate::tuning::SearchScoringConfig;

        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture_dir = manifest.join("eval/fixtures");
        let locomo_path = manifest.join("eval/data/locomo_plus.json");
        let has_locomo = locomo_path.exists();

        println!("\n{}", "=".repeat(70));
        println!("  AUTORESEARCH: Multi-Benchmark Parameter Sweep");
        println!(
            "  Fixture evals: {} | LoCoMo-Plus: {}",
            fixture_dir.exists(),
            has_locomo
        );
        println!("{}\n", "=".repeat(70));

        // --- Baseline ---
        let baseline_scoring = SearchScoringConfig::default();
        let baseline_fixture = run_eval(
            &fixture_dir,
            manifest,
            Some(&baseline_scoring),
            None,
            GateMode::Off,
        )
        .await
        .unwrap();
        let baseline_locomo = if has_locomo {
            Some(
                run_locomo_plus_eval(&locomo_path, Some(&baseline_scoring))
                    .await
                    .unwrap(),
            )
        } else {
            None
        };

        println!("BASELINE (current defaults: rrf_k=60, confirm=1.5, recap=0.3, domain=1.5)");
        println!(
            "  Fixtures:    NDCG@10={:.4}  MRR={:.4}  R@5={:.4}  neg_leak={}",
            baseline_fixture.ndcg_at_10,
            baseline_fixture.mrr,
            baseline_fixture.recall_at_5,
            baseline_fixture.negative_leakage
        );
        if let Some(ref lp) = baseline_locomo {
            println!(
                "  LoCoMo-Plus: NDCG@10={:.4}  MRR={:.4}  R@5={:.4}  HR@1={:.4}",
                lp.aggregate_ndcg_at_10,
                lp.aggregate_mrr,
                lp.aggregate_recall_at_5,
                lp.aggregate_hit_rate_at_1
            );
        }

        // --- Parameter grid ---
        struct Trial {
            name: String,
            scoring: SearchScoringConfig,
        }

        let mut trials: Vec<Trial> = Vec::new();

        // RRF_K scan
        for k in [10.0, 20.0, 40.0, 80.0, 120.0] {
            trials.push(Trial {
                name: format!("rrf_k={:.0}", k),
                scoring: SearchScoringConfig {
                    rrf_k: k,
                    ..Default::default()
                },
            });
        }

        // Confirmation boost scan
        for cb in [1.0, 1.2, 1.8, 2.0, 2.5] {
            trials.push(Trial {
                name: format!("confirm={:.1}", cb),
                scoring: SearchScoringConfig {
                    confirmation_boost: cb,
                    ..Default::default()
                },
            });
        }

        // Recap penalty scan
        for rp in [0.1, 0.2, 0.5, 0.7, 1.0] {
            trials.push(Trial {
                name: format!("recap={:.1}", rp),
                scoring: SearchScoringConfig {
                    recap_penalty: rp,
                    ..Default::default()
                },
            });
        }

        // Domain boost scan
        for db_val in [1.0, 1.2, 2.0, 3.0] {
            trials.push(Trial {
                name: format!("domain={:.1}", db_val),
                scoring: SearchScoringConfig {
                    domain_boost: db_val,
                    ..Default::default()
                },
            });
        }

        // FTS weight scan
        for fw in [0.2, 0.35, 0.5, 0.65, 0.8, 1.0] {
            trials.push(Trial {
                name: format!("fts_w={:.2}", fw),
                scoring: SearchScoringConfig {
                    fts_weight: fw,
                    ..Default::default()
                },
            });
        }

        // Print header
        println!(
            "\n{:<18} {:>10} {:>8} {:>8} {:>6}   {:>10} {:>8}",
            "Trial", "Fix NDCG", "Fix MRR", "Fix R@5", "Leak", "LP NDCG", "LP MRR"
        );
        println!("{}", "-".repeat(80));

        // Baseline row
        println!(
            "{:<18} {:>10.4} {:>8.4} {:>8.4} {:>6}   {:>10} {:>8}",
            "** BASELINE **",
            baseline_fixture.ndcg_at_10,
            baseline_fixture.mrr,
            baseline_fixture.recall_at_5,
            baseline_fixture.negative_leakage,
            baseline_locomo
                .as_ref()
                .map(|l| format!("{:.4}", l.aggregate_ndcg_at_10))
                .unwrap_or("-".into()),
            baseline_locomo
                .as_ref()
                .map(|l| format!("{:.4}", l.aggregate_mrr))
                .unwrap_or("-".into()),
        );

        let baseline_combined = baseline_fixture.ndcg_at_10
            + baseline_locomo
                .as_ref()
                .map(|l| l.aggregate_ndcg_at_10)
                .unwrap_or(0.0);
        let mut best_combined = baseline_combined;
        let mut best_trial = String::from("BASELINE");

        for trial in &trials {
            let fix_report = run_eval(
                &fixture_dir,
                manifest,
                Some(&trial.scoring),
                None,
                GateMode::Off,
            )
            .await
            .unwrap();
            let lp_report = if has_locomo {
                Some(
                    run_locomo_plus_eval(&locomo_path, Some(&trial.scoring))
                        .await
                        .unwrap(),
                )
            } else {
                None
            };

            let combined = fix_report.ndcg_at_10
                + lp_report
                    .as_ref()
                    .map(|l| l.aggregate_ndcg_at_10)
                    .unwrap_or(0.0);

            let fix_delta = fix_report.ndcg_at_10 - baseline_fixture.ndcg_at_10;
            let marker = if fix_delta > 0.001 {
                "+"
            } else if fix_delta < -0.001 {
                "-"
            } else {
                " "
            };

            println!(
                "{:<18} {:>10.4} {:>8.4} {:>8.4} {:>6}   {:>10} {:>8}  {}",
                trial.name,
                fix_report.ndcg_at_10,
                fix_report.mrr,
                fix_report.recall_at_5,
                fix_report.negative_leakage,
                lp_report
                    .as_ref()
                    .map(|l| format!("{:.4}", l.aggregate_ndcg_at_10))
                    .unwrap_or("-".into()),
                lp_report
                    .as_ref()
                    .map(|l| format!("{:.4}", l.aggregate_mrr))
                    .unwrap_or("-".into()),
                marker,
            );

            if combined > best_combined {
                best_combined = combined;
                best_trial = trial.name.clone();
            }
        }

        println!("\n{}", "=".repeat(70));
        println!(
            "  BEST: {} (combined NDCG@10 = {:.4})",
            best_trial, best_combined
        );
        if best_trial != "BASELINE" {
            println!(
                "  Delta vs baseline: +{:.4}",
                best_combined - baseline_combined
            );
        } else {
            println!("  Current defaults are optimal across benchmarks.");
        }
        println!("{}", "=".repeat(70));
    }

    /// Model scan: compare 768d candidates against BGE-Small (384d) baseline on LoCoMo.
    ///
    /// All models use [domain] prefix (proven +1.9% in 2x2). Multi-metric comparison.
    ///
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_model_scan -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_model_scan() {
        use crate::db::{EmbedConfig, MemoryDB};
        use crate::eval::locomo::{extract_observations, load_locomo};
        use crate::eval::metrics;
        use crate::sources::RawDocument;
        use std::collections::{HashMap, HashSet};

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
        if !path.exists() {
            println!("SKIP: locomo10.json not found");
            return;
        }
        let samples = load_locomo(&path).unwrap();

        struct ModelCandidate {
            name: &'static str,
            config: EmbedConfig,
        }

        let candidates = [
            ModelCandidate {
                name: "BGE-Small (384d) *baseline*",
                config: EmbedConfig::bge_small(),
            },
            ModelCandidate {
                name: "BGE-Base (768d)",
                config: EmbedConfig::bge_base(),
            },
            ModelCandidate {
                name: "BGE-Base-Q (768d)",
                config: EmbedConfig::bge_base_q(),
            },
            ModelCandidate {
                name: "GTE-Base-Q (768d)",
                config: EmbedConfig::gte_base_q(),
            },
            ModelCandidate {
                name: "Nomic-v1.5 (768d)",
                config: EmbedConfig::nomic_v15(),
            },
            ModelCandidate {
                name: "Nomic-v1.5-Q (768d)",
                config: EmbedConfig::nomic_v15_q(),
            },
            ModelCandidate {
                name: "Snowflake-M (768d)",
                config: EmbedConfig::snowflake_m(),
            },
            ModelCandidate {
                name: "all-mpnet-base (768d)",
                config: EmbedConfig::mpnet_base(),
            },
        ];

        println!("\n{}", "=".repeat(100));
        println!("  MODEL SWEEP: 768d candidates vs BGE-Small baseline (LoCoMo, {} convos, [domain] prefix)", samples.len());
        println!("{}\n", "=".repeat(100));
        println!(
            "{:<32} {:>10} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Model", "NDCG@10", "NDCG@5", "MRR", "R@5", "HR@1", "Δ NDCG@10"
        );
        println!("{}", "-".repeat(95));

        let mut baseline_ndcg: Option<f64> = None;

        for candidate in &candidates {
            let mut all_ndcg10 = Vec::new();
            let mut all_ndcg5 = Vec::new();
            let mut all_mrr = Vec::new();
            let mut all_r5 = Vec::new();
            let mut all_hr1 = Vec::new();

            for sample in &samples {
                let memories = extract_observations(sample);

                let tmp = tempfile::tempdir().unwrap();
                let db = MemoryDB::new_with_embed_config(tmp.path(), candidate.config.clone())
                    .await
                    .unwrap();

                let docs: Vec<RawDocument> = memories
                    .iter()
                    .enumerate()
                    .map(|(i, mem)| RawDocument {
                        content: mem.content.clone(),
                        source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
                        source: "memory".to_string(),
                        title: format!("{} session {}", mem.speaker, mem.session_num),
                        memory_type: Some("fact".to_string()),
                        domain: Some("conversation".to_string()),
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    })
                    .collect();
                db.upsert_documents(docs).await.unwrap();

                let dia_to_source: HashMap<String, String> = memories
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        (
                            m.dia_id.clone(),
                            format!("locomo_{}_obs_{}", sample.sample_id, i),
                        )
                    })
                    .collect();

                for qa in &sample.qa {
                    if qa.category == 5 {
                        continue;
                    }

                    let qresults = db
                        .search_memory(&qa.question, 10, None, None, None, None, None, None)
                        .await
                        .unwrap();
                    let relevant_ids: HashSet<String> = qa
                        .evidence
                        .iter()
                        .filter_map(|did| dia_to_source.get(did).cloned())
                        .collect();
                    if relevant_ids.is_empty() {
                        continue;
                    }

                    let result_ids: Vec<&str> =
                        qresults.iter().map(|r| r.source_id.as_str()).collect();
                    let relevant_refs: HashSet<&str> =
                        relevant_ids.iter().map(|s| s.as_str()).collect();
                    let grades: HashMap<&str, u8> = result_ids
                        .iter()
                        .map(|id| (*id, if relevant_refs.contains(id) { 2u8 } else { 0u8 }))
                        .collect();

                    all_ndcg10.push(metrics::ndcg_at_k(&result_ids, &grades, 10));
                    all_ndcg5.push(metrics::ndcg_at_k(&result_ids, &grades, 5));
                    all_mrr.push(metrics::mrr(&result_ids, &relevant_refs));
                    all_r5.push(metrics::recall_at_k(&result_ids, &relevant_refs, 5));
                    all_hr1.push(metrics::hit_rate_at_k(&result_ids, &relevant_refs, 1));
                }
            }

            let n = all_ndcg10.len().max(1) as f64;
            let ndcg10 = all_ndcg10.iter().sum::<f64>() / n;
            let ndcg5 = all_ndcg5.iter().sum::<f64>() / n;
            let mrr = all_mrr.iter().sum::<f64>() / n;
            let r5 = all_r5.iter().sum::<f64>() / n;
            let hr1 = all_hr1.iter().sum::<f64>() / n;

            if baseline_ndcg.is_none() {
                baseline_ndcg = Some(ndcg10);
            }
            let delta = ndcg10 - baseline_ndcg.unwrap();
            let marker = if delta > 0.005 {
                " <<<"
            } else if delta.abs() < 0.005 {
                " ~"
            } else {
                ""
            };

            println!(
                "{:<32} {:>10.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>+10.4}{}",
                candidate.name, ndcg10, ndcg5, mrr, r5, hr1, delta, marker
            );
        }

        println!("{}", "=".repeat(100));
    }

    /// Model scan on LoCoMo-Plus (401 trigger→cue pairs, all in one DB).
    /// Cross-validates model ranking from LoCoMo-10 on a different dataset/paradigm.
    ///
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_model_scan_plus -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_model_scan_plus() {
        use crate::db::{EmbedConfig, MemoryDB};
        use crate::eval::metrics;
        use crate::sources::RawDocument;
        use std::collections::{HashMap, HashSet};

        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo_plus.json");
        if !path.exists() {
            println!("SKIP: locomo_plus.json not found");
            return;
        }
        let samples = super::load_locomo_plus(&path).unwrap();

        struct ModelCandidate {
            name: &'static str,
            config: EmbedConfig,
        }

        let candidates = [
            ModelCandidate {
                name: "BGE-Small (384d) *baseline*",
                config: EmbedConfig::bge_small(),
            },
            ModelCandidate {
                name: "BGE-Base (768d)",
                config: EmbedConfig::bge_base(),
            },
            ModelCandidate {
                name: "BGE-Base-Q (768d)",
                config: EmbedConfig::bge_base_q(),
            },
            ModelCandidate {
                name: "GTE-Base-Q (768d)",
                config: EmbedConfig::gte_base_q(),
            },
            ModelCandidate {
                name: "Nomic-v1.5 (768d)",
                config: EmbedConfig::nomic_v15(),
            },
            ModelCandidate {
                name: "Nomic-v1.5-Q (768d)",
                config: EmbedConfig::nomic_v15_q(),
            },
            ModelCandidate {
                name: "Snowflake-M (768d)",
                config: EmbedConfig::snowflake_m(),
            },
            ModelCandidate {
                name: "all-mpnet-base (768d)",
                config: EmbedConfig::mpnet_base(),
            },
        ];

        println!("\n{}", "=".repeat(100));
        println!(
            "  MODEL SWEEP: LoCoMo-Plus ({} trigger→cue pairs, single DB, [domain] prefix)",
            samples.len()
        );
        println!("{}\n", "=".repeat(100));
        println!(
            "{:<32} {:>10} {:>8} {:>8} {:>8} {:>10}",
            "Model", "NDCG@10", "MRR", "R@5", "HR@1", "Δ NDCG@10"
        );
        println!("{}", "-".repeat(80));

        let mut baseline_ndcg: Option<f64> = None;

        for candidate in &candidates {
            let tmp = tempfile::tempdir().unwrap();
            let db = MemoryDB::new_with_embed_config(tmp.path(), candidate.config.clone())
                .await
                .unwrap();

            let docs: Vec<RawDocument> = samples
                .iter()
                .enumerate()
                .map(|(i, s)| RawDocument {
                    content: s.cue_dialogue.clone(),
                    source_id: format!("lcp_{}", i),
                    memory_type: Some(
                        super::constraint_to_memory_type(&s.relation_type).to_string(),
                    ),
                    domain: Some("conversation".to_string()),
                    source: "memory".to_string(),
                    title: format!("{} cue {}", s.relation_type, i),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                })
                .collect();
            db.upsert_documents(docs).await.unwrap();

            let mut all_ndcg = Vec::new();
            let mut all_mrr = Vec::new();
            let mut all_r5 = Vec::new();
            let mut all_hr1 = Vec::new();

            for (i, sample) in samples.iter().enumerate() {
                let target_id = format!("lcp_{}", i);
                let results = db
                    .search_memory(
                        &sample.trigger_query,
                        10,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
                    .await
                    .unwrap();
                let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
                let relevant: HashSet<&str> = [target_id.as_str()].into_iter().collect();
                let grades: HashMap<&str, u8> = [(target_id.as_str(), 2u8)].into_iter().collect();

                all_ndcg.push(metrics::ndcg_at_k(&result_ids, &grades, 10));
                all_mrr.push(metrics::mrr(&result_ids, &relevant));
                all_r5.push(metrics::recall_at_k(&result_ids, &relevant, 5));
                all_hr1.push(metrics::hit_rate_at_k(&result_ids, &relevant, 1));
            }

            let n = all_ndcg.len().max(1) as f64;
            let ndcg = all_ndcg.iter().sum::<f64>() / n;
            let mrr = all_mrr.iter().sum::<f64>() / n;
            let r5 = all_r5.iter().sum::<f64>() / n;
            let hr1 = all_hr1.iter().sum::<f64>() / n;

            if baseline_ndcg.is_none() {
                baseline_ndcg = Some(ndcg);
            }
            let delta = ndcg - baseline_ndcg.unwrap();
            let marker = if delta > 0.005 {
                " <<<"
            } else if delta.abs() < 0.005 {
                " ~"
            } else {
                ""
            };

            println!(
                "{:<32} {:>10.4} {:>8.4} {:>8.4} {:>8.4} {:>+10.4}{}",
                candidate.name, ndcg, mrr, r5, hr1, delta, marker
            );
        }

        println!("{}", "=".repeat(100));
    }

    /// Full model scan: 768d winners + 1024d candidates on BOTH benchmarks.
    ///
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_model_scan_1024 -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_model_scan_1024() {
        use crate::db::{EmbedConfig, MemoryDB};
        use crate::eval::locomo::{extract_observations, load_locomo};
        use crate::eval::metrics;
        use crate::sources::RawDocument;
        use std::collections::{HashMap, HashSet};

        let locomo_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
        let plus_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo_plus.json");
        if !locomo_path.exists() {
            println!("SKIP: locomo10.json not found");
            return;
        }
        let samples = load_locomo(&locomo_path).unwrap();
        let plus_samples = if plus_path.exists() {
            Some(super::load_locomo_plus(&plus_path).unwrap())
        } else {
            None
        };

        struct ModelCandidate {
            name: &'static str,
            config: EmbedConfig,
        }

        let candidates = [
            // Baseline
            ModelCandidate {
                name: "BGE-Small (384d)",
                config: EmbedConfig::bge_small(),
            },
            // 768d winners
            ModelCandidate {
                name: "BGE-Base (768d)",
                config: EmbedConfig::bge_base(),
            },
            ModelCandidate {
                name: "Nomic-v1.5 (768d)",
                config: EmbedConfig::nomic_v15(),
            },
            // 1024d candidates
            ModelCandidate {
                name: "BGE-Large (1024d)",
                config: EmbedConfig::bge_large(),
            },
            ModelCandidate {
                name: "BGE-Large-Q (1024d)",
                config: EmbedConfig::bge_large_q(),
            },
            ModelCandidate {
                name: "GTE-Large (1024d)",
                config: EmbedConfig::gte_large(),
            },
            ModelCandidate {
                name: "GTE-Large-Q (1024d)",
                config: EmbedConfig::gte_large_q(),
            },
            ModelCandidate {
                name: "Mxbai-Large (1024d)",
                config: EmbedConfig::mxbai_large(),
            },
            ModelCandidate {
                name: "Mxbai-Large-Q (1024d)",
                config: EmbedConfig::mxbai_large_q(),
            },
            ModelCandidate {
                name: "ModernBERT-Large (1024d)",
                config: EmbedConfig::modernbert_large(),
            },
            ModelCandidate {
                name: "Snowflake-L (1024d)",
                config: EmbedConfig::snowflake_l(),
            },
        ];

        // Collect all results: (name, locomo_ndcg10, locomo_mrr, locomo_r5, locomo_hr1, plus_ndcg10, plus_mrr, plus_hr1)
        #[allow(clippy::type_complexity)]
        let mut all_results: Vec<(String, f64, f64, f64, f64, f64, f64, f64)> = Vec::new();

        println!("\n{}", "=".repeat(115));
        println!("  MODEL SWEEP: 384d/768d/1024d — LoCoMo-10 + LoCoMo-Plus (all metrics, [domain] prefix)");
        println!("{}\n", "=".repeat(115));
        println!(
            "{:<28} {:>8} {:>8} {:>7} {:>7}  |  {:>8} {:>7} {:>7}  |  {:>8}",
            "Model", "L NDCG", "L MRR", "L R@5", "L HR1", "P NDCG", "P MRR", "P HR1", "Δ NDCG"
        );
        println!("{}", "-".repeat(115));

        let mut baseline_locomo: Option<f64> = None;

        for candidate in &candidates {
            // --- LoCoMo-10 ---
            let mut lm_ndcg = Vec::new();
            let mut lm_mrr = Vec::new();
            let mut lm_r5 = Vec::new();
            let mut lm_hr1 = Vec::new();

            for sample in &samples {
                let memories = extract_observations(sample);
                let tmp = tempfile::tempdir().unwrap();
                let db = MemoryDB::new_with_embed_config(tmp.path(), candidate.config.clone())
                    .await
                    .unwrap();

                let docs: Vec<RawDocument> = memories
                    .iter()
                    .enumerate()
                    .map(|(i, mem)| RawDocument {
                        content: mem.content.clone(),
                        source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
                        source: "memory".to_string(),
                        title: format!("{} session {}", mem.speaker, mem.session_num),
                        memory_type: Some("fact".to_string()),
                        domain: Some("conversation".to_string()),
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    })
                    .collect();
                db.upsert_documents(docs).await.unwrap();

                let dia_to_source: HashMap<String, String> = memories
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        (
                            m.dia_id.clone(),
                            format!("locomo_{}_obs_{}", sample.sample_id, i),
                        )
                    })
                    .collect();

                for qa in &sample.qa {
                    if qa.category == 5 {
                        continue;
                    }
                    let qr = db
                        .search_memory(&qa.question, 10, None, None, None, None, None, None)
                        .await
                        .unwrap();
                    let rel: HashSet<String> = qa
                        .evidence
                        .iter()
                        .filter_map(|d| dia_to_source.get(d).cloned())
                        .collect();
                    if rel.is_empty() {
                        continue;
                    }
                    let ids: Vec<&str> = qr.iter().map(|r| r.source_id.as_str()).collect();
                    let rr: HashSet<&str> = rel.iter().map(|s| s.as_str()).collect();
                    let gr: HashMap<&str, u8> = ids
                        .iter()
                        .map(|id| (*id, if rr.contains(id) { 2u8 } else { 0u8 }))
                        .collect();
                    lm_ndcg.push(metrics::ndcg_at_k(&ids, &gr, 10));
                    lm_mrr.push(metrics::mrr(&ids, &rr));
                    lm_r5.push(metrics::recall_at_k(&ids, &rr, 5));
                    lm_hr1.push(metrics::hit_rate_at_k(&ids, &rr, 1));
                }
            }

            let ln = lm_ndcg.len().max(1) as f64;
            let l_ndcg = lm_ndcg.iter().sum::<f64>() / ln;
            let l_mrr = lm_mrr.iter().sum::<f64>() / ln;
            let l_r5 = lm_r5.iter().sum::<f64>() / ln;
            let l_hr1 = lm_hr1.iter().sum::<f64>() / ln;

            // --- LoCoMo-Plus ---
            let (p_ndcg, p_mrr, p_hr1) = if let Some(ref ps) = plus_samples {
                let tmp = tempfile::tempdir().unwrap();
                let db = MemoryDB::new_with_embed_config(tmp.path(), candidate.config.clone())
                    .await
                    .unwrap();
                let docs: Vec<RawDocument> = ps
                    .iter()
                    .enumerate()
                    .map(|(i, s)| RawDocument {
                        content: s.cue_dialogue.clone(),
                        source_id: format!("lcp_{}", i),
                        memory_type: Some(
                            super::constraint_to_memory_type(&s.relation_type).to_string(),
                        ),
                        domain: Some("conversation".to_string()),
                        source: "memory".to_string(),
                        title: format!("{} cue {}", s.relation_type, i),
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    })
                    .collect();
                db.upsert_documents(docs).await.unwrap();

                let mut pn = Vec::new();
                let mut pm = Vec::new();
                let mut ph = Vec::new();
                for (i, s) in ps.iter().enumerate() {
                    let tid = format!("lcp_{}", i);
                    let r = db
                        .search_memory(&s.trigger_query, 10, None, None, None, None, None, None)
                        .await
                        .unwrap();
                    let ids: Vec<&str> = r.iter().map(|r| r.source_id.as_str()).collect();
                    let rel: HashSet<&str> = [tid.as_str()].into_iter().collect();
                    let gr: HashMap<&str, u8> = [(tid.as_str(), 2u8)].into_iter().collect();
                    pn.push(metrics::ndcg_at_k(&ids, &gr, 10));
                    pm.push(metrics::mrr(&ids, &rel));
                    ph.push(metrics::hit_rate_at_k(&ids, &rel, 1));
                }
                let n = pn.len().max(1) as f64;
                (
                    pn.iter().sum::<f64>() / n,
                    pm.iter().sum::<f64>() / n,
                    ph.iter().sum::<f64>() / n,
                )
            } else {
                (0.0, 0.0, 0.0)
            };

            if baseline_locomo.is_none() {
                baseline_locomo = Some(l_ndcg);
            }
            let delta = l_ndcg - baseline_locomo.unwrap();
            let marker = if delta > 0.005 {
                " <<<"
            } else if delta.abs() < 0.005 {
                " ~"
            } else {
                ""
            };

            println!(
                "{:<28} {:>8.4} {:>8.4} {:>7.4} {:>7.4}  |  {:>8.4} {:>7.4} {:>7.4}  |  {:>+8.4}{}",
                candidate.name, l_ndcg, l_mrr, l_r5, l_hr1, p_ndcg, p_mrr, p_hr1, delta, marker
            );

            all_results.push((
                candidate.name.to_string(),
                l_ndcg,
                l_mrr,
                l_r5,
                l_hr1,
                p_ndcg,
                p_mrr,
                p_hr1,
            ));
        }

        // Combined ranking: normalize each metric to [0,1] range across models, average
        println!("\n{}", "=".repeat(115));
        println!("  COMBINED RANKING (equal-weighted normalized score across all 7 metrics)");
        println!("{}", "-".repeat(115));

        let metric_count = 7;
        let model_count = all_results.len();
        // Extract metric columns
        let cols: Vec<Vec<f64>> = (0..metric_count)
            .map(|m| {
                all_results
                    .iter()
                    .map(|r| match m {
                        0 => r.1,
                        1 => r.2,
                        2 => r.3,
                        3 => r.4,
                        4 => r.5,
                        5 => r.6,
                        6 => r.7,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();

        // Min-max normalize each column
        let normed: Vec<Vec<f64>> = cols
            .iter()
            .map(|col| {
                let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = (max - min).max(1e-9);
                col.iter().map(|v| (v - min) / range).collect()
            })
            .collect();

        // Average normalized scores per model
        let mut scored: Vec<(String, f64)> = (0..model_count)
            .map(|i| {
                let avg: f64 =
                    (0..metric_count).map(|m| normed[m][i]).sum::<f64>() / metric_count as f64;
                (all_results[i].0.clone(), avg)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (rank, (name, score)) in scored.iter().enumerate() {
            let medal = match rank {
                0 => " *** BEST",
                1 => " ** 2nd",
                2 => " * 3rd",
                _ => "",
            };
            println!(
                "  #{:<2} {:<28} score={:.4}{}",
                rank + 1,
                name,
                score,
                medal
            );
        }
        println!("{}", "=".repeat(115));
    }

    /// Model scan on Origin fixtures — real-world memory patterns (agent-written facts,
    /// cross-domain retrieval, semantic vs keyword, contradiction detection, etc.)
    ///
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_model_scan_fixtures -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_model_scan_fixtures() {
        use crate::db::{EmbedConfig, MemoryDB};
        use crate::eval::fixtures::load_fixtures;
        use crate::eval::metrics;
        use crate::sources::RawDocument;
        use crate::tuning::ConfidenceConfig;
        use std::collections::{HashMap, HashSet};

        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/fixtures");

        struct ModelCandidate {
            name: &'static str,
            config: EmbedConfig,
        }

        let candidates = [
            ModelCandidate {
                name: "BGE-Small (384d)",
                config: EmbedConfig::bge_small(),
            },
            ModelCandidate {
                name: "BGE-Base-Q (768d)",
                config: EmbedConfig::bge_base_q(),
            },
            ModelCandidate {
                name: "Nomic-v1.5 (768d)",
                config: EmbedConfig::nomic_v15(),
            },
            ModelCandidate {
                name: "BGE-Large-Q (1024d)",
                config: EmbedConfig::bge_large_q(),
            },
        ];

        let cases = load_fixtures(&fixture_dir).unwrap();

        println!("\n{}", "=".repeat(100));
        println!(
            "  MODEL SWEEP: Origin Fixtures — real memory patterns ({} cases, [domain] prefix)",
            cases.len()
        );
        println!("{}\n", "=".repeat(100));
        println!(
            "{:<24} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6} {:>10}",
            "Model", "NDCG@10", "NDCG@5", "MRR", "R@5", "HR@1", "P@3", "Leak", "Δ NDCG"
        );
        println!("{}", "-".repeat(100));

        let confidence_cfg = ConfidenceConfig::default();
        let mut baseline_ndcg: Option<f64> = None;

        for candidate in &candidates {
            let mut all_ndcg10 = Vec::new();
            let mut all_ndcg5 = Vec::new();
            let mut all_mrr = Vec::new();
            let mut all_r5 = Vec::new();
            let mut all_hr1 = Vec::new();
            let mut all_p3 = Vec::new();
            let mut total_neg_leak = 0usize;

            for case in &cases {
                let tmp = tempfile::tempdir().unwrap();
                let db = MemoryDB::new_with_embed_config(tmp.path(), candidate.config.clone())
                    .await
                    .unwrap();

                // Seed positives + negatives
                let mut docs: Vec<RawDocument> = case
                    .seeds
                    .iter()
                    .map(|s| crate::eval::runner::seed_to_doc(s, &confidence_cfg))
                    .collect();
                for neg in &case.negative_seeds {
                    docs.push(crate::eval::runner::seed_to_doc(neg, &confidence_cfg));
                }
                db.upsert_documents(docs).await.unwrap();

                // Seed entities
                for entity in &case.entities {
                    let eid = db
                        .store_entity(
                            &entity.name,
                            &entity.entity_type,
                            entity.domain.as_deref(),
                            Some("eval"),
                            None,
                        )
                        .await
                        .unwrap();
                    for obs in &entity.observations {
                        db.add_observation(&eid, obs.content(), Some("eval"), None)
                            .await
                            .unwrap();
                    }
                }

                let results = db
                    .search_memory(
                        &case.query,
                        10,
                        None,
                        case.domain.as_deref(),
                        None,
                        None,
                        None,
                        None,
                    )
                    .await
                    .unwrap();
                let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

                let relevant: HashSet<&str> = case
                    .seeds
                    .iter()
                    .filter(|s| s.relevance >= 2)
                    .map(|s| s.id.as_str())
                    .collect();
                let grades: HashMap<&str, u8> = case
                    .seeds
                    .iter()
                    .map(|s| (s.id.as_str(), s.relevance))
                    .collect();
                let negatives: HashSet<&str> =
                    case.negative_seeds.iter().map(|s| s.id.as_str()).collect();

                all_ndcg10.push(metrics::ndcg_at_k(&ranked, &grades, 10));
                all_ndcg5.push(metrics::ndcg_at_k(&ranked, &grades, 5));
                all_mrr.push(metrics::mrr(&ranked, &relevant));
                all_r5.push(metrics::recall_at_k(&ranked, &relevant, 5));
                all_hr1.push(metrics::hit_rate_at_k(&ranked, &relevant, 1));
                all_p3.push(metrics::precision_at_k(&ranked, &relevant, 3));
                total_neg_leak += metrics::negative_leakage(&ranked, &negatives, 5);
            }

            let n = all_ndcg10.len().max(1) as f64;
            let ndcg10 = all_ndcg10.iter().sum::<f64>() / n;
            let ndcg5 = all_ndcg5.iter().sum::<f64>() / n;
            let mrr = all_mrr.iter().sum::<f64>() / n;
            let r5 = all_r5.iter().sum::<f64>() / n;
            let hr1 = all_hr1.iter().sum::<f64>() / n;
            let p3 = all_p3.iter().sum::<f64>() / n;

            if baseline_ndcg.is_none() {
                baseline_ndcg = Some(ndcg10);
            }
            let delta = ndcg10 - baseline_ndcg.unwrap();
            let marker = if delta > 0.005 {
                " <<<"
            } else if delta.abs() < 0.005 {
                " ~"
            } else {
                ""
            };

            println!(
                "{:<24} {:>8.4} {:>8.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>6} {:>+10.4}{}",
                candidate.name, ndcg10, ndcg5, mrr, r5, hr1, p3, total_neg_leak, delta, marker
            );
        }

        println!("{}", "=".repeat(100));

        // Per-case breakdown for the candidates
        println!("\n  PER-CASE BREAKDOWN:");
        println!(
            "{:<40} {:>12} {:>12} {:>12} {:>12}",
            "Case", "BGE-Sm", "BGE-B-Q", "Nomic", "BGE-L-Q"
        );
        println!("{}", "-".repeat(95));

        let configs = [
            EmbedConfig::bge_small(),
            EmbedConfig::bge_base_q(),
            EmbedConfig::nomic_v15(),
            EmbedConfig::bge_large_q(),
        ];

        for case in &cases {
            let mut case_scores = Vec::new();
            for config in &configs {
                let tmp = tempfile::tempdir().unwrap();
                let db = MemoryDB::new_with_embed_config(tmp.path(), config.clone())
                    .await
                    .unwrap();
                let mut docs: Vec<RawDocument> = case
                    .seeds
                    .iter()
                    .map(|s| crate::eval::runner::seed_to_doc(s, &confidence_cfg))
                    .collect();
                for neg in &case.negative_seeds {
                    docs.push(crate::eval::runner::seed_to_doc(neg, &confidence_cfg));
                }
                db.upsert_documents(docs).await.unwrap();
                let results = db
                    .search_memory(
                        &case.query,
                        10,
                        None,
                        case.domain.as_deref(),
                        None,
                        None,
                        None,
                        None,
                    )
                    .await
                    .unwrap();
                let ranked: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
                let grades: HashMap<&str, u8> = case
                    .seeds
                    .iter()
                    .map(|s| (s.id.as_str(), s.relevance))
                    .collect();
                case_scores.push(metrics::ndcg_at_k(&ranked, &grades, 10));
            }
            let query_short: String = case.query.chars().take(37).collect();
            println!(
                "{:<40} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                query_short, case_scores[0], case_scores[1], case_scores[2], case_scores[3]
            );
        }
        println!("{}", "=".repeat(95));
    }

    /// 2x2 ablation: Embedding Model × Domain Prefix on LoCoMo standard benchmark.
    ///
    /// Isolates the GTE-Base regression: is it the model, the prefix, or both?
    ///
    ///   Cells:
    ///     BGE-Small (384)  + no prefix
    ///     BGE-Small (384)  + [domain] prefix
    ///     GTE-Base  (768)  + no prefix
    ///     GTE-Base  (768)  + [domain] prefix
    ///
    /// Run with: cargo test --lib eval::locomo_plus::tests::test_model_prefix_2x2 -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_model_prefix_2x2() {
        use crate::db::{EmbedConfig, MemoryDB};
        use crate::eval::locomo::{extract_observations, load_locomo};
        use crate::eval::metrics;
        use crate::sources::RawDocument;
        use std::collections::{HashMap, HashSet};

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("eval/data/locomo10.json");
        if !path.exists() {
            println!("SKIP: locomo10.json not found");
            return;
        }
        let samples = load_locomo(&path).unwrap();

        struct CellConfig {
            name: &'static str,
            embed: EmbedConfig,
            use_domain: bool,
        }

        let cells = [
            CellConfig {
                name: "BGE-Small + no prefix",
                embed: EmbedConfig::bge_small(),
                use_domain: false,
            },
            CellConfig {
                name: "BGE-Small + [domain]",
                embed: EmbedConfig::bge_small(),
                use_domain: true,
            },
            CellConfig {
                name: "GTE-Base  + no prefix",
                embed: EmbedConfig::gte_base(),
                use_domain: false,
            },
            CellConfig {
                name: "GTE-Base  + [domain]",
                embed: EmbedConfig::gte_base(),
                use_domain: true,
            },
        ];

        println!("\n{}", "=".repeat(85));
        println!(
            "  2x2 ABLATION: Embedding Model x Domain Prefix (LoCoMo, {} conversations)",
            samples.len()
        );
        println!("{}\n", "=".repeat(85));
        println!(
            "{:<28} {:>10} {:>8} {:>8} {:>8} {:>8}",
            "Cell", "NDCG@10", "MRR", "R@5", "HR@1", "QAs"
        );
        println!("{}", "-".repeat(80));

        let mut results: Vec<(String, f64, f64, f64, f64, usize)> = Vec::new();

        for cell in &cells {
            let mut all_ndcg = Vec::new();
            let mut all_mrr = Vec::new();
            let mut all_r5 = Vec::new();
            let mut all_hr1 = Vec::new();

            for sample in &samples {
                let memories = extract_observations(sample);

                let tmp = tempfile::tempdir().unwrap();
                let db = MemoryDB::new_with_embed_config(tmp.path(), cell.embed.clone())
                    .await
                    .unwrap();

                let docs: Vec<RawDocument> = memories
                    .iter()
                    .enumerate()
                    .map(|(i, mem)| RawDocument {
                        content: mem.content.clone(),
                        source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
                        source: "memory".to_string(),
                        title: format!("{} session {}", mem.speaker, mem.session_num),
                        memory_type: Some("fact".to_string()),
                        domain: if cell.use_domain {
                            Some("conversation".to_string())
                        } else {
                            None
                        },
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    })
                    .collect();
                db.upsert_documents(docs).await.unwrap();

                let dia_to_source: HashMap<String, String> = memories
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        (
                            m.dia_id.clone(),
                            format!("locomo_{}_obs_{}", sample.sample_id, i),
                        )
                    })
                    .collect();

                for qa in &sample.qa {
                    if qa.category == 5 {
                        continue;
                    }

                    let qresults = db
                        .search_memory(&qa.question, 10, None, None, None, None, None, None)
                        .await
                        .unwrap();
                    let relevant_ids: HashSet<String> = qa
                        .evidence
                        .iter()
                        .filter_map(|did| dia_to_source.get(did).cloned())
                        .collect();
                    if relevant_ids.is_empty() {
                        continue;
                    }

                    let result_ids: Vec<&str> =
                        qresults.iter().map(|r| r.source_id.as_str()).collect();
                    let relevant_refs: HashSet<&str> =
                        relevant_ids.iter().map(|s| s.as_str()).collect();
                    let grades: HashMap<&str, u8> = result_ids
                        .iter()
                        .map(|id| (*id, if relevant_refs.contains(id) { 2u8 } else { 0u8 }))
                        .collect();

                    all_ndcg.push(metrics::ndcg_at_k(&result_ids, &grades, 10));
                    all_mrr.push(metrics::mrr(&result_ids, &relevant_refs));
                    all_r5.push(metrics::recall_at_k(&result_ids, &relevant_refs, 5));
                    all_hr1.push(metrics::hit_rate_at_k(&result_ids, &relevant_refs, 1));
                }
            }

            let n = all_ndcg.len().max(1) as f64;
            let ndcg = all_ndcg.iter().sum::<f64>() / n;
            let mrr = all_mrr.iter().sum::<f64>() / n;
            let r5 = all_r5.iter().sum::<f64>() / n;
            let hr1 = all_hr1.iter().sum::<f64>() / n;
            let count = all_ndcg.len();

            println!(
                "{:<28} {:>10.4} {:>8.4} {:>8.4} {:>8.4} {:>8}",
                cell.name, ndcg, mrr, r5, hr1, count
            );
            results.push((cell.name.to_string(), ndcg, mrr, r5, hr1, count));
        }

        // Print 2x2 summary
        println!("\n{}", "=".repeat(85));
        println!("  FACTOR ANALYSIS");
        println!("{}", "-".repeat(85));

        // Model effect (avg across prefix conditions)
        let bge_avg = (results[0].1 + results[1].1) / 2.0;
        let gte_avg = (results[2].1 + results[3].1) / 2.0;
        println!(
            "  Model effect (NDCG@10):     BGE-Small avg={:.4}  GTE-Base avg={:.4}  Δ={:+.4}",
            bge_avg,
            gte_avg,
            gte_avg - bge_avg
        );

        // Prefix effect (avg across model conditions)
        let no_prefix_avg = (results[0].1 + results[2].1) / 2.0;
        let prefix_avg = (results[1].1 + results[3].1) / 2.0;
        println!(
            "  Prefix effect (NDCG@10):    no prefix avg={:.4}  [domain] avg={:.4}  Δ={:+.4}",
            no_prefix_avg,
            prefix_avg,
            prefix_avg - no_prefix_avg
        );

        // Interaction: does prefix help more for one model than another?
        let bge_prefix_effect = results[1].1 - results[0].1;
        let gte_prefix_effect = results[3].1 - results[2].1;
        println!(
            "  Interaction:                BGE prefix Δ={:+.4}  GTE prefix Δ={:+.4}",
            bge_prefix_effect, gte_prefix_effect
        );

        println!("{}", "=".repeat(85));
    }
}
