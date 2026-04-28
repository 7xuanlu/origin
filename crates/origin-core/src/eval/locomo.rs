// SPDX-License-Identifier: AGPL-3.0-only
//! LoCoMo benchmark adapter — converts locomo10.json into Origin eval cases.
//!
//! LoCoMo (Long Conversational Memory) contains 10 conversations with
//! pre-extracted observations and ~1,986 QA pairs in 5 categories:
//!   1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop, 5 = adversarial
//!
//! Dataset: <https://github.com/snap-research/locomo>

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::eval::fixtures::{EvalCase, SeedMemory};
use crate::eval::metrics;
use crate::quality_gate::QualityGate;
use crate::sources::RawDocument;
use crate::tuning::GateConfig;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// Bring date helpers into scope for use within this module.
use crate::eval::dates::seed_event_date;
// Re-export so external callers using `crate::eval::locomo::parse_locomo_date` still compile.
pub use crate::eval::dates::parse_locomo_date;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LocomoSample {
    pub sample_id: String,
    /// We only read speakers from conversation; the full dialogue is unused.
    pub conversation: serde_json::Value,
    pub qa: Vec<LocomoQA>,
    /// Nested: { "session_N_observation": { "Speaker": [["fact", "dia_id"], ...] } }
    pub observation: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct LocomoQA {
    pub question: String,
    /// Present for non-adversarial questions (categories 1-4).
    /// Can be a string or integer in the dataset, so we use Value.
    pub answer: Option<serde_json::Value>,
    /// Present for adversarial questions (category 5).
    pub adversarial_answer: Option<serde_json::Value>,
    pub evidence: Vec<String>,
    /// 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial
    pub category: u8,
}

/// A single observation extracted from the LoCoMo dataset.
#[derive(Debug, Clone)]
pub struct LocomoMemory {
    pub content: String,
    pub speaker: String,
    pub session_num: usize,
    pub dia_id: String,
    pub sample_id: String,
    /// Raw "h:mm am/pm on D Month, YYYY" string for this session, when present.
    pub session_date: Option<String>,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load LoCoMo dataset from a local JSON file.
pub fn load_locomo(path: &Path) -> Result<Vec<LocomoSample>, OriginError> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| OriginError::Generic(format!("Failed to read LoCoMo file: {e}")))?;
    let samples: Vec<LocomoSample> = serde_json::from_str(&data)
        .map_err(|e| OriginError::Generic(format!("Failed to parse LoCoMo JSON: {e}")))?;
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Observation extraction
// ---------------------------------------------------------------------------

/// Extract all observation facts from a LoCoMo sample.
///
/// Iterates `session_N_observation` keys, then speaker keys, then `[fact, dia_id]` pairs.
pub fn extract_observations(sample: &LocomoSample) -> Vec<LocomoMemory> {
    let mut memories = Vec::new();
    let obs = match sample.observation.as_object() {
        Some(o) => o,
        None => return memories,
    };

    for (session_key, speakers_val) in obs {
        // Parse session number from keys like "session_1_observation"
        let session_num = parse_session_num(session_key).unwrap_or(0);

        let speakers = match speakers_val.as_object() {
            Some(s) => s,
            None => continue,
        };

        for (speaker, facts_val) in speakers {
            let facts = match facts_val.as_array() {
                Some(f) => f,
                None => continue,
            };

            for fact_pair in facts {
                let pair = match fact_pair.as_array() {
                    Some(p) if p.len() >= 2 => p,
                    _ => continue,
                };

                let content = match pair[0].as_str() {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                let dia_id = match pair[1].as_str() {
                    Some(s) => s.to_string(),
                    None => continue,
                };

                let session_date_key = session_key.replace("_observation", "_date_time");
                let session_date = sample
                    .conversation
                    .get(&session_date_key)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                memories.push(LocomoMemory {
                    content,
                    speaker: speaker.clone(),
                    session_num,
                    dia_id,
                    sample_id: sample.sample_id.clone(),
                    session_date,
                });
            }
        }
    }

    memories
}

/// Parse session number from a key like "session_3_observation" -> 3.
fn parse_session_num(key: &str) -> Option<usize> {
    // Expected format: session_N_observation
    let stripped = key.strip_prefix("session_")?;
    let num_str = stripped.split('_').next()?;
    num_str.parse().ok()
}

// ---------------------------------------------------------------------------
// Conversion to eval cases
// ---------------------------------------------------------------------------

/// Convert a LoCoMo sample into eval cases for Origin's runner.
///
/// For each non-adversarial QA pair (category != 5):
/// - Observations whose `dia_id` matches the QA evidence become seeds with `relevance=3`
/// - All other observations from the same conversation become seeds with `relevance=1`
///
/// Category 5 (adversarial) is skipped — we don't test abstention yet.
pub fn sample_to_eval_cases(sample: &LocomoSample, memories: &[LocomoMemory]) -> Vec<EvalCase> {
    let mut cases = Vec::new();

    for qa in &sample.qa {
        // Skip adversarial questions
        if qa.category == 5 {
            continue;
        }

        let evidence_set: std::collections::HashSet<&str> =
            qa.evidence.iter().map(|s| s.as_str()).collect();

        let mut seeds = Vec::new();
        for (i, mem) in memories.iter().enumerate() {
            let relevance = if evidence_set.contains(mem.dia_id.as_str()) {
                3
            } else {
                1
            };

            seeds.push(SeedMemory {
                id: format!("locomo_{}_{}", sample.sample_id, i),
                content: mem.content.clone(),
                memory_type: "fact".to_string(),
                domain: Some("conversation".to_string()),
                relevance,
                structured_fields: None,
                confidence: None,
                confirmed: None,
                quality: None,
                is_recap: None,
                source_agent: None,
                age_days: None,
                supersedes: None,
            });
        }

        cases.push(EvalCase {
            query: qa.question.clone(),
            domain: Some("conversation".to_string()),
            seeds,
            negative_seeds: vec![],
            entities: vec![],
            empty_set: false,
        });
    }

    cases
}

/// Category name for display/reporting.
pub fn category_name(cat: u8) -> &'static str {
    match cat {
        1 => "multi-hop",
        2 => "temporal",
        3 => "open-domain",
        4 => "single-hop",
        5 => "adversarial",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// Benchmark result structs
// ---------------------------------------------------------------------------

/// Baseline metrics for LoCoMo benchmark comparison across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocomoBaseline {
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
    pub per_category: Vec<crate::eval::report::CategoryBaseline>,
}

/// Per-category results
#[derive(Debug, Clone, Serialize)]
pub struct LocomoCategoryResult {
    pub category: u8,
    pub name: String,
    pub count: usize,
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
}

/// Results for one LoCoMo conversation
#[derive(Debug, Clone, Serialize)]
pub struct LocomoConversationResult {
    pub sample_id: String,
    pub memories_seeded: usize,
    pub questions_evaluated: usize,
    pub overall_ndcg_at_10: f64,
    pub overall_mrr: f64,
    pub overall_recall_at_5: f64,
    pub per_category: Vec<LocomoCategoryResult>,
}

/// Full LoCoMo benchmark report
#[derive(Debug, Clone, Serialize)]
pub struct LocomoReport {
    pub conversations: Vec<LocomoConversationResult>,
    pub aggregate_ndcg_at_10: f64,
    pub aggregate_mrr: f64,
    pub aggregate_recall_at_5: f64,
    pub aggregate_hit_rate_at_1: f64,
    pub total_questions: usize,
    pub total_memories: usize,
    pub per_category_aggregate: Vec<LocomoCategoryResult>,
    /// Placeholder for future LLM-as-judge QA accuracy (J-score).
    /// Currently None — requires an LLM to generate answers from retrieved context.
    pub qa_accuracy: Option<f64>,
    /// Baseline comparison from a previous run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<LocomoBaseline>,
}

impl LocomoReport {
    /// Format as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("LoCoMo Benchmark\n");
        out.push_str("================\n");
        out.push_str(&format!("Conversations: {}\n", self.conversations.len()));
        out.push_str(&format!("Total questions: {}\n", self.total_questions));
        out.push_str(&format!("Total memories: {}\n\n", self.total_memories));

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
                        .per_category_aggregate
                        .iter()
                        .find(|c| c.name == cat_bl.name)
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

        out.push_str("\nPer category:\n");
        for cat in &self.per_category_aggregate {
            out.push_str(&format!(
                "  {} (n={:>3}): NDCG@10={:.3} MRR={:.3} R@5={:.3} HR@1={:.3}\n",
                cat.name, cat.count, cat.ndcg_at_10, cat.mrr, cat.recall_at_5, cat.hit_rate_at_1,
            ));
        }

        // Per-conversation summary
        if !self.conversations.is_empty() {
            out.push_str("\nPer conversation:\n");
            for conv in &self.conversations {
                out.push_str(&format!(
                    "  {} (n={}, mem={}): NDCG@10={:.3} MRR={:.3} R@5={:.3}\n",
                    conv.sample_id,
                    conv.questions_evaluated,
                    conv.memories_seeded,
                    conv.overall_ndcg_at_10,
                    conv.overall_mrr,
                    conv.overall_recall_at_5,
                ));
            }
        }
        out
    }

    /// Save current metrics as baseline for future comparison.
    pub fn save_baseline(&self, path: &Path) -> Result<(), std::io::Error> {
        let per_category: Vec<crate::eval::report::CategoryBaseline> = self
            .per_category_aggregate
            .iter()
            .map(|c| crate::eval::report::CategoryBaseline {
                name: c.name.clone(),
                ndcg_at_10: c.ndcg_at_10,
                mrr: c.mrr,
                recall_at_5: c.recall_at_5,
            })
            .collect();
        let baseline = LocomoBaseline {
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
    pub fn load_baseline(path: &Path) -> Option<LocomoBaseline> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

// ---------------------------------------------------------------------------
// End-to-end benchmark runner
// ---------------------------------------------------------------------------

/// Run LoCoMo benchmark. For each conversation:
/// 1. Create fresh ephemeral DB
/// 2. Seed ALL observations as memories
/// 3. For each non-adversarial QA pair, search and score
/// 4. Aggregate per-category and overall metrics
pub async fn run_locomo_eval(path: &Path) -> Result<LocomoReport, OriginError> {
    let samples = load_locomo(path)?;
    let mut conversations = Vec::new();
    // (category, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();

    for sample in &samples {
        let memories = extract_observations(sample);

        // Create ephemeral DB for this conversation
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all observations as memories
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
                event_date: seed_event_date(mem.session_date.as_deref(), parse_locomo_date),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Map dia_id to source_id for relevance judgments
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

        let mut conv_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();

        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }

            let results = db
                .search_memory(&qa.question, 10, None, None, None, None, None, None)
                .await?;

            // Build relevance judgments: evidence dia_ids -> source_ids = relevant
            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();

            if relevant_ids.is_empty() {
                continue; // Skip if no mappable evidence
            }

            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

            // Binary relevance: 1 if in evidence set, 0 otherwise
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();

            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
            let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
            let mrr_val = metrics::mrr(&result_ids, &relevant_set);
            let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
            let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

            conv_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
            all_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
        }

        // Per-category for this conversation
        let per_cat = aggregate_by_category(&conv_scores);

        let n = conv_scores.len();
        conversations.push(LocomoConversationResult {
            sample_id: sample.sample_id.clone(),
            memories_seeded: memories.len(),
            questions_evaluated: n,
            overall_ndcg_at_10: avg_field(&conv_scores, |s| s.2),
            overall_mrr: avg_field(&conv_scores, |s| s.3),
            overall_recall_at_5: avg_field(&conv_scores, |s| s.4),
            per_category: per_cat,
        });
    }

    // Global aggregates
    let per_cat_agg = aggregate_by_category(&all_scores);

    // TODO: LLM-as-judge QA accuracy requires:
    // 1. Generate answer from retrieved context using an LLM
    // 2. Judge answer correctness against ground truth using LLM-as-judge
    // 3. This is what competitors report as "J-score" or "accuracy"
    // Currently we report retrieval metrics (NDCG, MRR, Recall) which measure
    // whether the right memories are found, not whether the final answer is correct.
    Ok(LocomoReport {
        conversations,
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories: samples.iter().map(|s| extract_observations(s).len()).sum(),
        per_category_aggregate: per_cat_agg,
        qa_accuracy: None,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Reranked benchmark runner — same as run_locomo_eval but uses search_memory_reranked
// ---------------------------------------------------------------------------

/// Same seeding/scoring logic as `run_locomo_eval`, but retrieval uses
/// `search_memory_reranked` with the supplied LLM for per-query reranking.
pub async fn run_locomo_eval_reranked(
    path: &Path,
    llm: std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<LocomoReport, OriginError> {
    let samples = load_locomo(path)?;
    let mut conversations = Vec::new();
    // (category, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();

    for sample in &samples {
        let memories = extract_observations(sample);

        // Create ephemeral DB for this conversation
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all observations as memories
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
                event_date: seed_event_date(mem.session_date.as_deref(), parse_locomo_date),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Map dia_id to source_id for relevance judgments
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

        let mut conv_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();

        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }

            let results = db
                .search_memory_reranked(&qa.question, 10, None, None, None, Some(llm.clone()))
                .await?;

            // Build relevance judgments: evidence dia_ids -> source_ids = relevant
            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();

            if relevant_ids.is_empty() {
                continue; // Skip if no mappable evidence
            }

            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

            // Binary relevance: 1 if in evidence set, 0 otherwise
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();

            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
            let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
            let mrr_val = metrics::mrr(&result_ids, &relevant_set);
            let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
            let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

            conv_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
            all_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
        }

        // Per-category for this conversation
        let per_cat = aggregate_by_category(&conv_scores);

        let n = conv_scores.len();
        conversations.push(LocomoConversationResult {
            sample_id: sample.sample_id.clone(),
            memories_seeded: memories.len(),
            questions_evaluated: n,
            overall_ndcg_at_10: avg_field(&conv_scores, |s| s.2),
            overall_mrr: avg_field(&conv_scores, |s| s.3),
            overall_recall_at_5: avg_field(&conv_scores, |s| s.4),
            per_category: per_cat,
        });
    }

    // Global aggregates
    let per_cat_agg = aggregate_by_category(&all_scores);

    Ok(LocomoReport {
        conversations,
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories: samples.iter().map(|s| extract_observations(s).len()).sum(),
        per_category_aggregate: per_cat_agg,
        qa_accuracy: None,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Expanded benchmark runner -- same as run_locomo_eval but uses search_memory_expanded
// ---------------------------------------------------------------------------

/// Same seeding/scoring logic as `run_locomo_eval`, but retrieval uses
/// `search_memory_expanded` with the supplied LLM for query expansion before search.
pub async fn run_locomo_eval_expanded(
    path: &Path,
    llm: std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<LocomoReport, OriginError> {
    let samples = load_locomo(path)?;
    let mut conversations = Vec::new();
    // (category, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();

    for sample in &samples {
        let memories = extract_observations(sample);

        // Create ephemeral DB for this conversation
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all observations as memories
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
                event_date: seed_event_date(mem.session_date.as_deref(), parse_locomo_date),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

        // Map dia_id to source_id for relevance judgments
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

        let mut conv_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();

        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }

            let results = db
                .search_memory_expanded(&qa.question, 10, None, None, None, Some(llm.clone()))
                .await?;

            // Build relevance judgments: evidence dia_ids -> source_ids = relevant
            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();

            if relevant_ids.is_empty() {
                continue; // Skip if no mappable evidence
            }

            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

            // Binary relevance: 1 if in evidence set, 0 otherwise
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();

            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
            let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
            let mrr_val = metrics::mrr(&result_ids, &relevant_set);
            let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
            let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

            conv_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
            all_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
        }

        // Per-category for this conversation
        let per_cat = aggregate_by_category(&conv_scores);

        let n = conv_scores.len();
        conversations.push(LocomoConversationResult {
            sample_id: sample.sample_id.clone(),
            memories_seeded: memories.len(),
            questions_evaluated: n,
            overall_ndcg_at_10: avg_field(&conv_scores, |s| s.2),
            overall_mrr: avg_field(&conv_scores, |s| s.3),
            overall_recall_at_5: avg_field(&conv_scores, |s| s.4),
            per_category: per_cat,
        });
    }

    // Global aggregates
    let per_cat_agg = aggregate_by_category(&all_scores);

    Ok(LocomoReport {
        conversations,
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories: samples.iter().map(|s| extract_observations(s).len()).sum(),
        per_category_aggregate: per_cat_agg,
        qa_accuracy: None,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Gated benchmark runner — clean / noisy / gated comparison
// ---------------------------------------------------------------------------

/// Controls how noise is handled in the LoCoMo benchmark.
#[derive(Debug, Clone, Copy)]
pub enum LocomoGateMode {
    /// No noise — only observations (baseline).
    Clean,
    /// Noise added alongside observations, no gate filtering.
    Noisy,
    /// Noise added, but each noise doc passes through the quality gate before insertion.
    Gated,
}

/// Generate domain-relevant noise documents proportional to observation count.
///
/// For every 3 observations, 1 noise memory is generated (33% ratio).
/// Noise is designed to compete with real personal-conversation memories:
///
/// - **Category 1**: System prompt fragments about conversation memory
///   (should be caught by the content gate's preamble detection)
/// - **Category 2**: Vague restates of common conversation topics
///   (competes semantically with real memories)
/// - **Category 3**: Hallucinated personal facts using actual speaker names
///   (maximally confusing — plausible but wrong)
/// - **Category 4**: Meta-commentary about conversation processing
///   (should be caught by content gate patterns or novelty)
/// - **Category 5**: Transient status about conversation processing
fn generate_noise(sample: &LocomoSample, observation_count: usize) -> Vec<RawDocument> {
    let noise_count = observation_count / 3; // 33% noise ratio

    // Extract speaker names from the conversation metadata
    let speaker_a = sample
        .conversation
        .get("speaker_a")
        .and_then(|v| v.as_str())
        .unwrap_or("Speaker A");
    let speaker_b = sample
        .conversation
        .get("speaker_b")
        .and_then(|v| v.as_str())
        .unwrap_or("Speaker B");

    // Category 1: System prompt fragments (should be caught by content gate)
    let sys_prompt_templates: Vec<&str> = vec![
        "You are a personal conversation memory assistant. Your role is to remember details about people's lives relationships and activities.",
        "As an AI you must accurately recall personal facts about family members friends and their daily activities.",
        "Your role is to track relationships between people including their family members friends and colleagues mentioned in conversations.",
    ];

    // Category 2: Vague restates of common conversation topics
    let vague_templates: Vec<&str> = vec![
        "Someone mentioned something about a family gathering or holiday celebration recently.",
        "There was a discussion about travel plans and visiting family members sometime soon.",
        "The speakers talked about their children and school activities at some point.",
        "Something about cooking or trying a new restaurant was mentioned in the conversation.",
        "There were some health-related concerns discussed between the speakers.",
        "Plans were made regarding an upcoming birthday party or celebration event.",
        "The conversation included discussion about work stress and career changes.",
        "Someone mentioned moving to a new house or apartment in the near future.",
        "Weekend plans involving outdoor activities or sports were discussed briefly.",
        "A discussion about pets or animals came up during the conversation session.",
    ];

    // Category 3: Hallucinated personal facts using actual speaker names
    // {a} and {b} are replaced with real names below
    let hallucinated_templates: Vec<String> = vec![
        "{a} enjoys playing tennis on weekends with neighborhood friends regularly.",
        "{b} mentioned planning to visit parents in Florida next month for holidays.",
        "{a} and {b} discussed their shared interest in cooking Italian food together.",
        "{a} recently started learning guitar as a new creative hobby this year.",
        "{b} is training for a local charity run happening in the spring season.",
        "{a} talked about attending a concert with friends downtown last weekend.",
        "{b} mentioned getting a new puppy from the local animal shelter recently.",
        "{a} said the family is planning a camping trip to the national park.",
        "{b} discussed redecorating the living room with new furniture and paint.",
        "{a} mentioned starting a book club with coworkers at the office monthly.",
    ]
    .into_iter()
    .map(|t| t.replace("{a}", speaker_a).replace("{b}", speaker_b))
    .collect();

    // Category 4: Meta-commentary about the conversation
    let meta_templates: Vec<&str> = vec![
        "I just stored several personal facts from this conversation about family and hobbies.",
        "The conversation contained interesting details about upcoming travel and social plans.",
        "Updated memory with new observations about the speakers' relationships and activities.",
        "Currently processing this dialogue to extract key personal facts and preferences.",
        "Analyzing the conversation context to identify family relationships and important events.",
    ];

    // Category 5: Transient status about conversation processing
    let transient_templates: Vec<&str> = vec![
        "Working on extracting personal details from the latest conversation session data.",
        "Reviewing the dialogue for mentions of names dates and relationship information.",
        "Processing conversation turns to identify facts worth storing in long-term memory.",
    ];

    // Build a combined cycle: interleave categories for variety
    let mut all_noise: Vec<(String, &str)> = Vec::new();
    for t in &sys_prompt_templates {
        all_noise.push((t.to_string(), "sys_prompt"));
    }
    for t in &vague_templates {
        all_noise.push((t.to_string(), "vague"));
    }
    for t in &hallucinated_templates {
        all_noise.push((t.clone(), "hallucinated"));
    }
    for t in &meta_templates {
        all_noise.push((t.to_string(), "meta"));
    }
    for t in &transient_templates {
        all_noise.push((t.to_string(), "transient"));
    }

    let mut docs = Vec::new();
    for i in 0..noise_count {
        let (content, _noise_type) = &all_noise[i % all_noise.len()];
        docs.push(RawDocument {
            content: content.clone(),
            source_id: format!("noise_{}_{}", sample.sample_id, i),
            source: "memory".to_string(),
            title: format!("noise_{}", i),
            memory_type: Some("fact".to_string()),
            domain: Some("conversation".to_string()),
            last_modified: chrono::Utc::now().timestamp(),
            ..Default::default()
        });
    }
    docs
}

/// Run LoCoMo benchmark with noise + quality gate comparison.
///
/// Three modes:
/// - **Clean**: Only observations seeded (baseline).
/// - **Noisy**: Observations + synthetic noise, all inserted without filtering.
/// - **Gated**: Observations inserted first, then each noise doc is run through
///   `QualityGate::evaluate()` (content patterns + novelty check) and only
///   inserted if admitted.
pub async fn run_locomo_eval_with_gate(
    path: &Path,
    mode: LocomoGateMode,
) -> Result<LocomoReport, OriginError> {
    let samples = load_locomo(path)?;
    let mut conversations = Vec::new();
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();
    let mut total_memories_inserted: usize = 0;

    let gate = match mode {
        LocomoGateMode::Gated => Some(QualityGate::new(GateConfig::default())),
        _ => None,
    };

    for sample in &samples {
        let memories = extract_observations(sample);

        // Create ephemeral DB for this conversation
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all observations as memories (ground truth — always inserted)
        let obs_docs: Vec<RawDocument> = memories
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
                event_date: seed_event_date(mem.session_date.as_deref(), parse_locomo_date),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(obs_docs).await?;

        let mut memories_in_db = memories.len();

        // For Noisy/Gated modes, generate and process noise
        match mode {
            LocomoGateMode::Clean => { /* no noise */ }
            LocomoGateMode::Noisy => {
                let noise = generate_noise(sample, memories.len());
                let noise_count = noise.len();
                db.upsert_documents(noise).await?;
                memories_in_db += noise_count;
            }
            LocomoGateMode::Gated => {
                let noise = generate_noise(sample, memories.len());
                let gate = gate.as_ref().unwrap();
                let mut admitted_docs = Vec::new();
                for doc in &noise {
                    let (result, _similar_id) = gate.evaluate(&doc.content, &db).await?;
                    if result.admitted {
                        admitted_docs.push(doc.clone());
                    }
                }
                let admitted_count = admitted_docs.len();
                if !admitted_docs.is_empty() {
                    db.upsert_documents(admitted_docs).await?;
                }
                memories_in_db += admitted_count;
            }
        }

        total_memories_inserted += memories_in_db;

        // Map dia_id to source_id for relevance judgments
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

        let mut conv_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();

        for qa in &sample.qa {
            if qa.category == 5 {
                continue;
            }

            let results = db
                .search_memory(&qa.question, 10, None, None, None, None, None, None)
                .await?;

            // Build relevance judgments: evidence dia_ids -> source_ids = relevant
            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();

            if relevant_ids.is_empty() {
                continue;
            }

            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

            // Binary relevance: 1 if in evidence set, 0 otherwise
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();

            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
            let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
            let mrr_val = metrics::mrr(&result_ids, &relevant_set);
            let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
            let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

            conv_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
            all_scores.push((qa.category, ndcg_5, ndcg_10, mrr_val, recall_5, hr_1));
        }

        let per_cat = aggregate_by_category(&conv_scores);

        conversations.push(LocomoConversationResult {
            sample_id: sample.sample_id.clone(),
            memories_seeded: memories_in_db,
            questions_evaluated: conv_scores.len(),
            overall_ndcg_at_10: avg_field(&conv_scores, |s| s.2),
            overall_mrr: avg_field(&conv_scores, |s| s.3),
            overall_recall_at_5: avg_field(&conv_scores, |s| s.4),
            per_category: per_cat,
        });
    }

    let per_cat_agg = aggregate_by_category(&all_scores);

    // TODO: LLM-as-judge QA accuracy requires:
    // 1. Generate answer from retrieved context using an LLM
    // 2. Judge answer correctness against ground truth using LLM-as-judge
    // 3. This is what competitors report as "J-score" or "accuracy"
    // Currently we report retrieval metrics (NDCG, MRR, Recall) which measure
    // whether the right memories are found, not whether the final answer is correct.
    Ok(LocomoReport {
        conversations,
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories: total_memories_inserted,
        per_category_aggregate: per_cat_agg,
        qa_accuracy: None,
        baseline: None,
    })
}

/// Average a field across a score slice.
fn avg_field(
    scores: &[(u8, f64, f64, f64, f64, f64)],
    f: impl Fn(&(u8, f64, f64, f64, f64, f64)) -> f64,
) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let sum: f64 = scores.iter().map(&f).sum();
    sum / scores.len() as f64
}

/// Aggregate scores by category.
fn aggregate_by_category(scores: &[(u8, f64, f64, f64, f64, f64)]) -> Vec<LocomoCategoryResult> {
    let mut results = Vec::new();
    for cat in [1u8, 2, 3, 4] {
        let cat_scores: Vec<_> = scores.iter().filter(|s| s.0 == cat).cloned().collect();
        if cat_scores.is_empty() {
            continue;
        }
        results.push(LocomoCategoryResult {
            category: cat,
            name: category_name(cat).to_string(),
            count: cat_scores.len(),
            ndcg_at_5: avg_field(&cat_scores, |s| s.1),
            ndcg_at_10: avg_field(&cat_scores, |s| s.2),
            mrr: avg_field(&cat_scores, |s| s.3),
            recall_at_5: avg_field(&cat_scores, |s| s.4),
            hit_rate_at_1: avg_field(&cat_scores, |s| s.5),
        });
    }
    results
}

// ---------------------------------------------------------------------------
// Backward-compat re-exports (prompt functions now live in judge.rs)
// ---------------------------------------------------------------------------

pub use crate::eval::judge::locomo_judge_prompt;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_locomo_sample() {
        let json = r#"[{
            "sample_id": "test-conv",
            "conversation": {"speaker_a": "Alice", "speaker_b": "Bob"},
            "observation": {
                "session_1_observation": {
                    "Alice": [["Alice likes hiking in the mountains.", "D1:3"]],
                    "Bob": [["Bob is training for a marathon.", "D1:5"]]
                }
            },
            "qa": [
                {"question": "What does Alice like?", "answer": "hiking in the mountains", "evidence": ["D1:3"], "category": 4},
                {"question": "Unanswerable", "adversarial_answer": "not mentioned", "evidence": ["D1:1"], "category": 5}
            ]
        }]"#;

        let samples: Vec<LocomoSample> = serde_json::from_str(json).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].sample_id, "test-conv");
        assert_eq!(samples[0].qa.len(), 2);
        assert_eq!(samples[0].qa[0].category, 4);
        assert!(samples[0].qa[1].answer.is_none());
    }

    #[test]
    fn test_extract_observations() {
        let json = r#"{
            "sample_id": "test-conv",
            "conversation": {"speaker_a": "Alice", "speaker_b": "Bob"},
            "observation": {
                "session_1_observation": {
                    "Alice": [["Alice likes hiking in the mountains.", "D1:3"]],
                    "Bob": [["Bob is training for a marathon.", "D1:5"]]
                }
            },
            "qa": []
        }"#;

        let sample: LocomoSample = serde_json::from_str(json).unwrap();
        let memories = extract_observations(&sample);
        assert_eq!(memories.len(), 2);

        // Check that both speakers are represented
        let speakers: Vec<&str> = memories.iter().map(|m| m.speaker.as_str()).collect();
        assert!(speakers.contains(&"Alice"));
        assert!(speakers.contains(&"Bob"));

        // Check dia_ids
        let dia_ids: Vec<&str> = memories.iter().map(|m| m.dia_id.as_str()).collect();
        assert!(dia_ids.contains(&"D1:3"));
        assert!(dia_ids.contains(&"D1:5"));

        // Check session number
        for mem in &memories {
            assert_eq!(mem.session_num, 1);
            assert_eq!(mem.sample_id, "test-conv");
        }
    }

    #[test]
    fn test_extract_multi_session_observations() {
        let json = r#"{
            "sample_id": "conv-multi",
            "conversation": {},
            "observation": {
                "session_1_observation": {
                    "Alice": [["Fact from session 1.", "D1:2"]]
                },
                "session_3_observation": {
                    "Alice": [["Fact from session 3.", "D3:4"]],
                    "Bob": [["Bob fact session 3.", "D3:7"]]
                }
            },
            "qa": []
        }"#;

        let sample: LocomoSample = serde_json::from_str(json).unwrap();
        let memories = extract_observations(&sample);
        assert_eq!(memories.len(), 3);

        let session_nums: Vec<usize> = memories.iter().map(|m| m.session_num).collect();
        assert!(session_nums.contains(&1));
        assert!(session_nums.contains(&3));
    }

    #[test]
    fn test_sample_to_eval_cases_skips_adversarial() {
        let json = r#"{
            "sample_id": "test-conv",
            "conversation": {},
            "observation": {
                "session_1_observation": {
                    "Alice": [["Alice likes hiking in the mountains.", "D1:3"]],
                    "Bob": [["Bob is training for a marathon.", "D1:5"]]
                }
            },
            "qa": [
                {"question": "What does Alice like?", "answer": "hiking in the mountains", "evidence": ["D1:3"], "category": 4},
                {"question": "Unanswerable", "adversarial_answer": "not mentioned", "evidence": ["D1:1"], "category": 5}
            ]
        }"#;

        let sample: LocomoSample = serde_json::from_str(json).unwrap();
        let memories = extract_observations(&sample);
        let cases = sample_to_eval_cases(&sample, &memories);

        // Only the non-adversarial question should produce an eval case
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].query, "What does Alice like?");
    }

    #[test]
    fn test_sample_to_eval_cases_relevance_grading() {
        let json = r#"{
            "sample_id": "test-conv",
            "conversation": {},
            "observation": {
                "session_1_observation": {
                    "Alice": [["Alice likes hiking in the mountains.", "D1:3"]],
                    "Bob": [["Bob is training for a marathon.", "D1:5"]]
                }
            },
            "qa": [
                {"question": "What does Alice like?", "answer": "hiking in the mountains", "evidence": ["D1:3"], "category": 4}
            ]
        }"#;

        let sample: LocomoSample = serde_json::from_str(json).unwrap();
        let memories = extract_observations(&sample);
        let cases = sample_to_eval_cases(&sample, &memories);

        assert_eq!(cases.len(), 1);
        // All observations are seeds (evidence + non-evidence)
        assert_eq!(cases[0].seeds.len(), 2);

        // Find the matching evidence seed (D1:3) and the non-matching one (D1:5)
        let evidence_seed = cases[0]
            .seeds
            .iter()
            .find(|s| s.content.contains("hiking"))
            .unwrap();
        let other_seed = cases[0]
            .seeds
            .iter()
            .find(|s| s.content.contains("marathon"))
            .unwrap();

        assert_eq!(evidence_seed.relevance, 3);
        assert_eq!(other_seed.relevance, 1);
    }

    #[test]
    fn test_parse_session_num() {
        assert_eq!(parse_session_num("session_1_observation"), Some(1));
        assert_eq!(parse_session_num("session_12_observation"), Some(12));
        assert_eq!(parse_session_num("not_a_session"), None);
    }

    #[test]
    fn test_category_name() {
        assert_eq!(category_name(1), "multi-hop");
        assert_eq!(category_name(4), "single-hop");
        assert_eq!(category_name(5), "adversarial");
        assert_eq!(category_name(99), "unknown");
    }

    #[test]
    fn test_multi_evidence_qa() {
        let json = r#"{
            "sample_id": "test-multi",
            "conversation": {},
            "observation": {
                "session_1_observation": {
                    "Alice": [
                        ["Alice works at Google.", "D1:2"],
                        ["Alice moved to NYC.", "D1:5"]
                    ],
                    "Bob": [["Bob likes coffee.", "D1:3"]]
                }
            },
            "qa": [
                {"question": "Where does Alice work and live?", "answer": "Google in NYC", "evidence": ["D1:2", "D1:5"], "category": 1}
            ]
        }"#;

        let sample: LocomoSample = serde_json::from_str(json).unwrap();
        let memories = extract_observations(&sample);
        assert_eq!(memories.len(), 3);

        let cases = sample_to_eval_cases(&sample, &memories);
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].seeds.len(), 3);

        // Two seeds should have relevance=3 (both evidence dia_ids match)
        let high_rel: Vec<_> = cases[0].seeds.iter().filter(|s| s.relevance == 3).collect();
        assert_eq!(high_rel.len(), 2);

        // One seed should have relevance=1 (Bob's non-evidence observation)
        let low_rel: Vec<_> = cases[0].seeds.iter().filter(|s| s.relevance == 1).collect();
        assert_eq!(low_rel.len(), 1);
        assert!(low_rel[0].content.contains("coffee"));
    }

    #[test]
    fn test_baseline_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locomo_baseline.json");

        let report = LocomoReport {
            conversations: vec![],
            aggregate_ndcg_at_10: 0.750,
            aggregate_mrr: 0.700,
            aggregate_recall_at_5: 0.520,
            aggregate_hit_rate_at_1: 0.450,
            total_questions: 100,
            total_memories: 500,
            per_category_aggregate: vec![
                LocomoCategoryResult {
                    category: 1,
                    name: "multi-hop".to_string(),
                    count: 25,
                    ndcg_at_5: 0.550,
                    ndcg_at_10: 0.560,
                    mrr: 0.500,
                    recall_at_5: 0.480,
                    hit_rate_at_1: 0.400,
                },
                LocomoCategoryResult {
                    category: 3,
                    name: "open-domain".to_string(),
                    count: 30,
                    ndcg_at_5: 0.430,
                    ndcg_at_10: 0.441,
                    mrr: 0.410,
                    recall_at_5: 0.380,
                    hit_rate_at_1: 0.350,
                },
            ],
            qa_accuracy: None,
            baseline: None,
        };

        report.save_baseline(&path).unwrap();
        let loaded = LocomoReport::load_baseline(&path).unwrap();

        assert!((loaded.ndcg_at_10 - 0.750).abs() < 0.001);
        assert!((loaded.mrr - 0.700).abs() < 0.001);
        assert!((loaded.recall_at_5 - 0.520).abs() < 0.001);
        assert!((loaded.hit_rate_at_1 - 0.450).abs() < 0.001);

        // Per-category baselines
        assert_eq!(loaded.per_category.len(), 2);
        assert_eq!(loaded.per_category[0].name, "multi-hop");
        assert!((loaded.per_category[0].ndcg_at_10 - 0.560).abs() < 0.001);
        assert_eq!(loaded.per_category[1].name, "open-domain");
        assert!((loaded.per_category[1].mrr - 0.410).abs() < 0.001);
    }

    #[test]
    fn test_to_terminal_with_baseline() {
        let report = LocomoReport {
            conversations: vec![],
            aggregate_ndcg_at_10: 0.660,
            aggregate_mrr: 0.610,
            aggregate_recall_at_5: 0.540,
            aggregate_hit_rate_at_1: 0.470,
            total_questions: 100,
            total_memories: 500,
            per_category_aggregate: vec![LocomoCategoryResult {
                category: 3,
                name: "open-domain".to_string(),
                count: 30,
                ndcg_at_5: 0.470,
                ndcg_at_10: 0.480,
                mrr: 0.440,
                recall_at_5: 0.400,
                hit_rate_at_1: 0.370,
            }],
            qa_accuracy: None,
            baseline: Some(LocomoBaseline {
                ndcg_at_10: 0.750,
                mrr: 0.700,
                recall_at_5: 0.520,
                hit_rate_at_1: 0.450,
                per_category: vec![crate::eval::report::CategoryBaseline {
                    name: "open-domain".to_string(),
                    ndcg_at_10: 0.441,
                    mrr: 0.410,
                    recall_at_5: 0.380,
                }],
            }),
        };

        let text = report.to_terminal();
        assert!(text.contains("LoCoMo Benchmark"));
        assert!(text.contains("NDCG@10"));
        assert!(text.contains("Baseline comparison:"));
        assert!(text.contains("open-domain"));
        // Verify delta printing is present
        assert!(text.contains("->"));
    }

    #[tokio::test]
    async fn test_locomo_seed_propagates_session_date() {
        use crate::sources::RawDocument;

        let last_modified = super::parse_locomo_date("1:56 pm on 8 May, 2023").unwrap();

        let tmp = tempfile::tempdir().unwrap();
        let db =
            crate::db::MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter))
                .await
                .unwrap();

        db.upsert_documents(vec![RawDocument {
            content: "Alice told Bob she moved to Tokyo".to_string(),
            source_id: "locomo/sample1/dia1".to_string(),
            source: "memory".to_string(),
            title: "test".to_string(),
            last_modified,
            memory_type: Some("fact".to_string()),
            domain: Some("conversation".to_string()),
            ..Default::default()
        }])
        .await
        .unwrap();

        let results = db
            .search_memory(
                "Tokyo",
                5,
                None,
                Some("conversation"),
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].last_modified, 1_683_554_160);
        assert_eq!(results[0].created_at, 1_683_554_160);
    }
}
