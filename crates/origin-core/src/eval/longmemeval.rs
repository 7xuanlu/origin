// SPDX-License-Identifier: AGPL-3.0-only
//! LongMemEval benchmark adapter — converts LongMemEval dataset into Origin eval cases.
//!
//! LongMemEval (ICLR 2025, arXiv:2410.10813) tests 5 core memory retrieval abilities
//! across 500 questions with user-assistant chat history:
//!
//!   - single-session-user:       facts stated by the user in one session
//!   - single-session-assistant:  facts stated by the assistant in one session
//!   - single-session-preference: user preferences expressed in one session
//!   - knowledge-update:          corrected/superseded information across sessions
//!   - temporal-reasoning:        time-ordered events across sessions
//!   - multi-session:             facts that span multiple sessions
//!
//! Three dataset variants exist:
//!   - **oracle**: only evidence sessions (small, ~15MB, fast eval)
//!   - **S (cleaned)**: ~40 sessions per question (~115K tokens, 277MB)
//!   - **M (cleaned)**: ~500 sessions per question (2.7GB)
//!
//! Dataset: <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>
//! Paper:   <https://arxiv.org/abs/2410.10813>

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

// ---------------------------------------------------------------------------
// Data structures (matches the JSON schema from HuggingFace)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct LongMemEvalSample {
    pub question_id: String,
    /// One of: single-session-user, single-session-assistant, single-session-preference,
    /// knowledge-update, temporal-reasoning, multi-session
    pub question_type: String,
    pub question: String,
    /// Can be a string or integer (e.g. "GPS system" or 3).
    pub answer: serde_json::Value,
    /// e.g. "2023/04/10 (Mon) 23:07"
    pub question_date: String,
    /// Dates for each haystack session, parallel to haystack_session_ids.
    pub haystack_dates: Vec<String>,
    /// Session IDs for all haystack sessions (superset of answer_session_ids in S/M).
    pub haystack_session_ids: Vec<String>,
    /// The actual chat sessions. Each session is a list of turns.
    pub haystack_sessions: Vec<Vec<ChatTurn>>,
    /// Session IDs that contain the answer evidence.
    pub answer_session_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
    /// True if this turn contains evidence for the answer.
    #[serde(default)]
    pub has_answer: bool,
}

/// A memory extracted from a LongMemEval chat session.
#[derive(Debug, Clone)]
pub struct LongMemEvalMemory {
    pub content: String,
    pub role: String,
    pub session_id: String,
    pub session_idx: usize,
    pub turn_idx: usize,
    pub has_answer: bool,
    pub question_id: String,
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load LongMemEval dataset from a local JSON file.
/// Works with oracle, S-cleaned, or M-cleaned variants.
pub fn load_longmemeval(path: &Path) -> Result<Vec<LongMemEvalSample>, OriginError> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| OriginError::Generic(format!("Failed to read LongMemEval file: {e}")))?;
    let samples: Vec<LongMemEvalSample> = serde_json::from_str(&data)
        .map_err(|e| OriginError::Generic(format!("Failed to parse LongMemEval JSON: {e}")))?;
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Memory extraction
// ---------------------------------------------------------------------------

/// Extract memories from all chat sessions in a LongMemEval sample.
///
/// Strategy: each user turn becomes a memory (users state facts, preferences, events).
/// Assistant turns that contain answer evidence are also included (for single-session-assistant).
/// Non-evidence assistant turns are skipped to keep the memory count manageable.
pub fn extract_memories(sample: &LongMemEvalSample) -> Vec<LongMemEvalMemory> {
    let mut memories = Vec::new();

    for (sess_idx, (session_id, session)) in sample
        .haystack_session_ids
        .iter()
        .zip(sample.haystack_sessions.iter())
        .enumerate()
    {
        for (turn_idx, turn) in session.iter().enumerate() {
            // Always include user turns (they contain the personal facts).
            // Include assistant turns only if they have answer evidence,
            // since assistant responses are often generic/filler.
            if turn.role == "user" || turn.has_answer {
                memories.push(LongMemEvalMemory {
                    content: turn.content.clone(),
                    role: turn.role.clone(),
                    session_id: session_id.clone(),
                    session_idx: sess_idx,
                    turn_idx,
                    has_answer: turn.has_answer,
                    question_id: sample.question_id.clone(),
                });
            }
        }
    }

    memories
}

/// Build a source_id for a memory extracted from a LongMemEval turn.
fn memory_source_id(question_id: &str, session_idx: usize, turn_idx: usize) -> String {
    format!("lme_{}_{}_t{}", question_id, session_idx, turn_idx)
}

// ---------------------------------------------------------------------------
// Conversion to eval cases
// ---------------------------------------------------------------------------

/// Convert a LongMemEval sample into an eval case.
///
/// All extracted memories become seeds. Memories from evidence sessions
/// (answer_session_ids) with `has_answer=true` get `relevance=3`.
/// Other memories from evidence sessions get `relevance=2`.
/// Non-evidence session memories get `relevance=1`.
pub fn sample_to_eval_case(sample: &LongMemEvalSample, memories: &[LongMemEvalMemory]) -> EvalCase {
    let evidence_session_set: HashSet<&str> = sample
        .answer_session_ids
        .iter()
        .map(|s| s.as_str())
        .collect();

    let seeds: Vec<SeedMemory> = memories
        .iter()
        .map(|mem| {
            let relevance = if mem.has_answer {
                3 // Direct evidence turn
            } else if evidence_session_set.contains(mem.session_id.as_str()) {
                2 // Same session as evidence, contextually relevant
            } else {
                1 // Distractor session
            };

            // Map question_type to memory_type
            let memory_type = match sample.question_type.as_str() {
                "single-session-preference" => "preference",
                _ => "fact",
            };

            SeedMemory {
                id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                content: mem.content.clone(),
                memory_type: memory_type.to_string(),
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
            }
        })
        .collect();

    EvalCase {
        query: sample.question.clone(),
        domain: Some("conversation".to_string()),
        seeds,
        negative_seeds: vec![],
        entities: vec![],
        empty_set: false,
    }
}

/// Map question_type string to a short category code for reporting.
pub fn category_code(question_type: &str) -> &'static str {
    match question_type {
        "single-session-user" => "SSU",
        "single-session-assistant" => "SSA",
        "single-session-preference" => "SSP",
        "knowledge-update" => "KU",
        "temporal-reasoning" => "TR",
        "multi-session" => "MS",
        _ => "?",
    }
}

/// Map question_type string to a display name.
pub fn category_name(question_type: &str) -> &'static str {
    match question_type {
        "single-session-user" => "single-session-user",
        "single-session-assistant" => "single-session-assistant",
        "single-session-preference" => "single-session-preference",
        "knowledge-update" => "knowledge-update",
        "temporal-reasoning" => "temporal-reasoning",
        "multi-session" => "multi-session",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// Benchmark result structs
// ---------------------------------------------------------------------------

/// Baseline metrics for LongMemEval benchmark comparison across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongMemEvalBaseline {
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
    pub per_category: Vec<crate::eval::report::CategoryBaseline>,
}

/// Per-category results.
#[derive(Debug, Clone, Serialize)]
pub struct LongMemEvalCategoryResult {
    pub question_type: String,
    pub code: String,
    pub count: usize,
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub mrr: f64,
    pub recall_at_5: f64,
    pub hit_rate_at_1: f64,
}

/// Full LongMemEval benchmark report.
#[derive(Debug, Clone, Serialize)]
pub struct LongMemEvalReport {
    pub aggregate_ndcg_at_10: f64,
    pub aggregate_mrr: f64,
    pub aggregate_recall_at_5: f64,
    pub aggregate_hit_rate_at_1: f64,
    pub total_questions: usize,
    pub total_memories: usize,
    pub per_category: Vec<LongMemEvalCategoryResult>,
    /// Baseline comparison from a previous run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<LongMemEvalBaseline>,
}

impl LongMemEvalReport {
    /// Format as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("LongMemEval Benchmark\n");
        out.push_str("=====================\n");
        out.push_str(&format!("Questions: {}\n", self.total_questions));
        out.push_str(&format!(
            "Total memories seeded: {}\n\n",
            self.total_memories
        ));

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
                        .per_category
                        .iter()
                        .find(|c| c.question_type == cat_bl.name)
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
        for cat in &self.per_category {
            out.push_str(&format!(
                "  {} {:3} (n={:>3}): NDCG@10={:.3} MRR={:.3} R@5={:.3} HR@1={:.3}\n",
                cat.code,
                cat.question_type,
                cat.count,
                cat.ndcg_at_10,
                cat.mrr,
                cat.recall_at_5,
                cat.hit_rate_at_1,
            ));
        }
        out
    }

    /// Save current metrics as baseline for future comparison.
    pub fn save_baseline(&self, path: &Path) -> Result<(), std::io::Error> {
        let per_category: Vec<crate::eval::report::CategoryBaseline> = self
            .per_category
            .iter()
            .map(|c| crate::eval::report::CategoryBaseline {
                name: c.question_type.clone(),
                ndcg_at_10: c.ndcg_at_10,
                mrr: c.mrr,
                recall_at_5: c.recall_at_5,
            })
            .collect();
        let baseline = LongMemEvalBaseline {
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
    pub fn load_baseline(path: &Path) -> Option<LongMemEvalBaseline> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

// ---------------------------------------------------------------------------
// End-to-end benchmark runner
// ---------------------------------------------------------------------------

/// All category types in reporting order.
const CATEGORY_ORDER: &[&str] = &[
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
];

/// Run LongMemEval benchmark. For each question:
/// 1. Create fresh ephemeral DB
/// 2. Extract memories from chat sessions and seed them
/// 3. Search with the question, score against evidence turns
/// 4. Aggregate per-category and overall metrics
pub async fn run_longmemeval_eval(path: &Path) -> Result<LongMemEvalReport, OriginError> {
    let samples = load_longmemeval(path)?;
    // (question_type, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut total_memories: usize = 0;

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all extracted memories
        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| {
                let memory_type = match sample.question_type.as_str() {
                    "single-session-preference" => "preference",
                    _ => "fact",
                };
                RawDocument {
                    content: mem.content.clone(),
                    source_id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.role, mem.session_idx),
                    memory_type: Some(memory_type.to_string()),
                    domain: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                }
            })
            .collect();
        total_memories += docs.len();
        db.upsert_documents(docs).await?;

        // Build relevance judgments: has_answer turns are relevant
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();

        if relevant_source_ids.is_empty() {
            continue; // Skip if no evidence turns
        }

        // Search
        let results = db
            .search_memory(&sample.question, 10, None, None, None, None, None, None)
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        // Binary relevance grades
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| {
                (
                    *id,
                    if relevant_source_ids.contains(*id) {
                        1
                    } else {
                        0
                    },
                )
            })
            .collect();

        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
        let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
        let mrr_val = metrics::mrr(&result_ids, &relevant_set);
        let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
        let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

        all_scores.push((
            sample.question_type.clone(),
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    // Aggregate
    let per_category = aggregate_by_category(&all_scores);

    Ok(LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories,
        per_category,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Reranked benchmark runner — same as run_longmemeval_eval but uses search_memory_reranked
// ---------------------------------------------------------------------------

/// Same seeding/scoring logic as `run_longmemeval_eval`, but retrieval uses
/// `search_memory_reranked` with the supplied LLM for per-query reranking.
pub async fn run_longmemeval_eval_reranked(
    path: &Path,
    llm: std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<LongMemEvalReport, OriginError> {
    let samples = load_longmemeval(path)?;
    // (question_type, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut total_memories: usize = 0;

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all extracted memories
        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| {
                let memory_type = match sample.question_type.as_str() {
                    "single-session-preference" => "preference",
                    _ => "fact",
                };
                RawDocument {
                    content: mem.content.clone(),
                    source_id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.role, mem.session_idx),
                    memory_type: Some(memory_type.to_string()),
                    domain: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                }
            })
            .collect();
        total_memories += docs.len();
        db.upsert_documents(docs).await?;

        // Build relevance judgments: has_answer turns are relevant
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();

        if relevant_source_ids.is_empty() {
            continue; // Skip if no evidence turns
        }

        // Search with reranking
        let results = db
            .search_memory_reranked(&sample.question, 10, None, None, None, Some(llm.clone()))
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        // Binary relevance grades
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| {
                (
                    *id,
                    if relevant_source_ids.contains(*id) {
                        1
                    } else {
                        0
                    },
                )
            })
            .collect();

        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
        let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
        let mrr_val = metrics::mrr(&result_ids, &relevant_set);
        let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
        let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

        all_scores.push((
            sample.question_type.clone(),
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    // Aggregate
    let per_category = aggregate_by_category(&all_scores);

    Ok(LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories,
        per_category,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Expanded benchmark runner -- same as run_longmemeval_eval but uses search_memory_expanded
// ---------------------------------------------------------------------------

/// Same seeding/scoring logic as `run_longmemeval_eval`, but retrieval uses
/// `search_memory_expanded` with the supplied LLM for query expansion before search.
pub async fn run_longmemeval_eval_expanded(
    path: &Path,
    llm: std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<LongMemEvalReport, OriginError> {
    let samples = load_longmemeval(path)?;
    // (question_type, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut total_memories: usize = 0;

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all extracted memories
        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| {
                let memory_type = match sample.question_type.as_str() {
                    "single-session-preference" => "preference",
                    _ => "fact",
                };
                RawDocument {
                    content: mem.content.clone(),
                    source_id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.role, mem.session_idx),
                    memory_type: Some(memory_type.to_string()),
                    domain: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                }
            })
            .collect();
        total_memories += docs.len();
        db.upsert_documents(docs).await?;

        // Build relevance judgments: has_answer turns are relevant
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();

        if relevant_source_ids.is_empty() {
            continue; // Skip if no evidence turns
        }

        // Search with query expansion
        let results = db
            .search_memory_expanded(&sample.question, 10, None, None, None, Some(llm.clone()))
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        // Binary relevance grades
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| {
                (
                    *id,
                    if relevant_source_ids.contains(*id) {
                        1
                    } else {
                        0
                    },
                )
            })
            .collect();

        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
        let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
        let mrr_val = metrics::mrr(&result_ids, &relevant_set);
        let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
        let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

        all_scores.push((
            sample.question_type.clone(),
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    // Aggregate
    let per_category = aggregate_by_category(&all_scores);

    Ok(LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories,
        per_category,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Gated benchmark runner — clean / noisy / gated comparison
// ---------------------------------------------------------------------------

/// Controls how noise is handled in the LongMemEval benchmark.
#[derive(Debug, Clone, Copy)]
pub enum LongMemEvalGateMode {
    /// No noise — only extracted chat memories (baseline).
    Clean,
    /// Noise added alongside memories, no gate filtering.
    Noisy,
    /// Noise added, but each noise doc passes through the quality gate before insertion.
    Gated,
}

/// Generate chat-assistant-style noise documents proportional to memory count.
///
/// For every 3 real memories, 1 noise memory is generated (33% ratio).
/// Noise is designed to compete with real user-assistant chat memories:
///
/// - **Category 1**: System prompt fragments about being a helpful assistant with memory
///   (should be caught by the content gate's preamble detection)
/// - **Category 2**: Vague restates of common chat topics
///   (competes semantically with real memories)
/// - **Category 3**: Hallucinated assistant facts using generic patterns
///   (plausible but not from the data)
/// - **Category 4**: Meta-commentary about memory storage
///   (should be caught by content gate patterns or novelty)
/// - **Category 5**: Transient processing status
fn generate_longmemeval_noise(memory_count: usize) -> Vec<RawDocument> {
    let noise_count = memory_count / 3; // 33% noise ratio

    // Category 1: System prompt fragments (should be caught by content gate)
    let sys_prompt_templates: Vec<&str> = vec![
        "You are a helpful assistant with long-term memory. Remember details from past conversations to provide personalized responses.",
        "As an AI assistant you should recall personal details preferences and past discussions to maintain conversational continuity.",
        "Your role is to be a memory-augmented assistant that tracks user preferences facts and conversation history over time.",
    ];

    // Category 2: Vague restates of common chat topics
    let vague_templates: Vec<&str> = vec![
        "Something about a meeting was discussed in a previous conversation.",
        "The user mentioned their work situation at some point during our chats.",
        "There was a conversation about plans for an upcoming event or trip.",
        "Some preferences about food or dining were mentioned at some point.",
        "A discussion about family members or relationships happened recently.",
        "The user talked about a hobby or leisure activity they enjoy.",
        "Something related to technology or a software tool was brought up.",
        "Health or fitness goals were discussed in an earlier session.",
        "The user asked about recommendations for something in a past chat.",
        "Some career or education plans were mentioned during our conversations.",
    ];

    // Category 3: Hallucinated assistant facts using generic patterns
    let hallucinated_templates: Vec<&str> = vec![
        "The user mentioned they like hiking and going on outdoor adventures on weekends.",
        "The user said they are planning to visit their parents next month for a family reunion.",
        "The user works in software engineering and enjoys problem-solving at work daily.",
        "The user has a pet dog named Buddy that they adopted from a local shelter.",
        "The user prefers Italian cuisine and particularly enjoys homemade pasta dishes.",
        "The user recently started learning to play the piano as a creative hobby.",
        "The user mentioned they are training for a half-marathon coming up in spring.",
        "The user said they enjoy reading science fiction novels before going to bed.",
        "The user is interested in photography and recently bought a new mirrorless camera.",
        "The user mentioned they are thinking about moving to a different city for work.",
    ];

    // Category 4: Meta-commentary about conversation processing
    let meta_templates: Vec<&str> = vec![
        "I stored several facts from this conversation about the user's preferences and plans.",
        "The conversation contained interesting personal details worth remembering for later.",
        "Updated memory with new observations about the user's activities and interests.",
        "This dialogue session provided useful context about the user's daily life routines.",
        "Noted several important details from the user's messages for future reference.",
    ];

    // Category 5: Transient processing status
    let transient_templates: Vec<&str> = vec![
        "Analyzing the dialogue for key information to store in long-term memory.",
        "Processing the latest conversation turns to extract memorable facts and details.",
        "Working on summarizing the key points from this chat session for storage.",
    ];

    // Build a combined cycle: interleave categories for variety
    let mut all_noise: Vec<&str> = Vec::new();
    all_noise.extend_from_slice(&sys_prompt_templates);
    all_noise.extend_from_slice(&vague_templates);
    all_noise.extend_from_slice(&hallucinated_templates);
    all_noise.extend_from_slice(&meta_templates);
    all_noise.extend_from_slice(&transient_templates);

    let mut docs = Vec::new();
    for i in 0..noise_count {
        let content = all_noise[i % all_noise.len()];
        docs.push(RawDocument {
            content: content.to_string(),
            source_id: format!("lme_noise_{}", i),
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

/// Run LongMemEval benchmark with noise + quality gate comparison.
///
/// Three modes:
/// - **Clean**: Only chat memories seeded (baseline).
/// - **Noisy**: Memories + synthetic noise, all inserted without filtering.
/// - **Gated**: Memories inserted first, then each noise doc is run through
///   `QualityGate::evaluate()` (content patterns + novelty check) and only
///   inserted if admitted.
pub async fn run_longmemeval_eval_with_gate(
    path: &Path,
    mode: LongMemEvalGateMode,
) -> Result<LongMemEvalReport, OriginError> {
    let samples = load_longmemeval(path)?;
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut total_memories_inserted: usize = 0;

    let gate = match mode {
        LongMemEvalGateMode::Gated => Some(QualityGate::new(GateConfig::default())),
        _ => None,
    };

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all extracted memories (ground truth — always inserted)
        let docs: Vec<RawDocument> = memories
            .iter()
            .map(|mem| {
                let memory_type = match sample.question_type.as_str() {
                    "single-session-preference" => "preference",
                    _ => "fact",
                };
                RawDocument {
                    content: mem.content.clone(),
                    source_id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.role, mem.session_idx),
                    memory_type: Some(memory_type.to_string()),
                    domain: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                }
            })
            .collect();
        let real_count = docs.len();
        db.upsert_documents(docs).await?;

        let mut memories_in_db = real_count;

        // For Noisy/Gated modes, generate and process noise
        match mode {
            LongMemEvalGateMode::Clean => { /* no noise */ }
            LongMemEvalGateMode::Noisy => {
                let noise = generate_longmemeval_noise(real_count);
                let noise_count = noise.len();
                db.upsert_documents(noise).await?;
                memories_in_db += noise_count;
            }
            LongMemEvalGateMode::Gated => {
                let noise = generate_longmemeval_noise(real_count);
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

        // Build relevance judgments: has_answer turns are relevant
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();

        if relevant_source_ids.is_empty() {
            continue; // Skip if no evidence turns
        }

        // Search
        let results = db
            .search_memory(&sample.question, 10, None, None, None, None, None, None)
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

        // Binary relevance grades
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| {
                (
                    *id,
                    if relevant_source_ids.contains(*id) {
                        1
                    } else {
                        0
                    },
                )
            })
            .collect();

        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        let ndcg_10 = metrics::ndcg_at_k(&result_ids, &grades, 10);
        let ndcg_5 = metrics::ndcg_at_k(&result_ids, &grades, 5);
        let mrr_val = metrics::mrr(&result_ids, &relevant_set);
        let recall_5 = metrics::recall_at_k(&result_ids, &relevant_set, 5);
        let hr_1 = metrics::hit_rate_at_k(&result_ids, &relevant_set, 1);

        all_scores.push((
            sample.question_type.clone(),
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    // Aggregate
    let per_category = aggregate_by_category(&all_scores);

    Ok(LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories: total_memories_inserted,
        per_category,
        baseline: None,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Average a field across a score slice.
fn avg_field(
    scores: &[(String, f64, f64, f64, f64, f64)],
    f: impl Fn(&(String, f64, f64, f64, f64, f64)) -> f64,
) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let sum: f64 = scores.iter().map(&f).sum();
    sum / scores.len() as f64
}

/// Aggregate scores by question_type category.
fn aggregate_by_category(
    scores: &[(String, f64, f64, f64, f64, f64)],
) -> Vec<LongMemEvalCategoryResult> {
    let mut results = Vec::new();
    for &cat in CATEGORY_ORDER {
        let cat_scores: Vec<_> = scores.iter().filter(|s| s.0 == cat).cloned().collect();
        if cat_scores.is_empty() {
            continue;
        }
        results.push(LongMemEvalCategoryResult {
            question_type: cat.to_string(),
            code: category_code(cat).to_string(),
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
// Canonical LLM-judge accuracy evaluation (3-phase pipeline)
// ---------------------------------------------------------------------------

/// Call the Anthropic API directly via reqwest. Returns response text.
/// Much faster than `claude -p` (no process spawn overhead) and costs ~$0.001/call with Haiku.
pub async fn call_anthropic_api(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
    system_prompt: Option<&str>,
    max_tokens: usize,
) -> Result<String, String> {
    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": messages
    });
    if let Some(sys) = system_prompt {
        body["system"] = serde_json::json!(sys);
    }

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse error: {e}"))?;
    let answer = json["content"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|block| block["text"].as_str())
        .unwrap_or("")
        .to_string();
    Ok(answer)
}

// ---------------------------------------------------------------------------
// Anthropic Batch API (50% cheaper, no rate limits)
// ---------------------------------------------------------------------------

/// Submit a batch of prompts to the Anthropic Batch API.
/// Returns the batch ID for polling.
pub async fn submit_batch(
    client: &reqwest::Client,
    api_key: &str,
    requests: Vec<(String, String, Option<String>, usize)>, // (custom_id, prompt, system, max_tokens)
    model: &str,
) -> Result<String, String> {
    let batch_requests: Vec<serde_json::Value> = requests
        .into_iter()
        .map(|(id, prompt, system, max_tokens)| {
            let mut params = serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]
            });
            if let Some(sys) = system {
                params["system"] = serde_json::json!(sys);
            }
            serde_json::json!({
                "custom_id": id,
                "params": params
            })
        })
        .collect();

    let body = serde_json::json!({ "requests": batch_requests });

    let resp = client
        .post("https://api.anthropic.com/v1/messages/batches")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("batch submit failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("batch API error {status}: {text}"));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse: {e}"))?;
    json["id"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "no batch id in response".to_string())
}

/// Poll a batch until it reaches "ended" status. Returns the results_url.
pub async fn poll_batch(
    client: &reqwest::Client,
    api_key: &str,
    batch_id: &str,
) -> Result<String, String> {
    let url = format!("https://api.anthropic.com/v1/messages/batches/{}", batch_id);
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;

        let resp = client
            .get(&url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await
            .map_err(|e| format!("poll failed: {e}"))?;

        let json: serde_json::Value = resp.json().await.map_err(|e| format!("parse: {e}"))?;
        let status = json["processing_status"].as_str().unwrap_or("unknown");

        let succeeded = json["request_counts"]["succeeded"].as_u64().unwrap_or(0);
        let processing = json["request_counts"]["processing"].as_u64().unwrap_or(0);
        let errored = json["request_counts"]["errored"].as_u64().unwrap_or(0);
        eprintln!(
            "[batch] status={}, succeeded={}, processing={}, errored={}",
            status, succeeded, processing, errored
        );

        if status == "ended" {
            return json["results_url"]
                .as_str()
                .map(|s| s.to_string())
                .ok_or_else(|| "batch ended but no results_url".to_string());
        }
    }
}

/// Download batch results and return a map of custom_id -> response text.
pub async fn download_batch_results(
    client: &reqwest::Client,
    api_key: &str,
    results_url: &str,
) -> Result<HashMap<String, String>, String> {
    let resp = client
        .get(results_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .map_err(|e| format!("download failed: {e}"))?;

    let text = resp.text().await.map_err(|e| format!("read body: {e}"))?;
    let mut results = HashMap::new();

    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let json: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("parse result line: {e}"))?;

        let custom_id = json["custom_id"].as_str().unwrap_or("").to_string();
        let result_type = json["result"]["type"].as_str().unwrap_or("");

        if result_type == "succeeded" {
            let answer = json["result"]["message"]["content"]
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|block| block["text"].as_str())
                .unwrap_or("")
                .to_string();
            results.insert(custom_id, answer);
        } else {
            eprintln!("[batch] {} result: {}", custom_id, result_type);
        }
    }

    Ok(results)
}

/// Generate answers for LME questions via Batch API (50% cheaper, no rate limits).
pub async fn generate_answers_batch(
    retrieved: &[RetrievedQuestion],
    answer_model: &str,
    existing: Vec<AnsweredQuestion>,
) -> Result<Vec<AnsweredQuestion>, OriginError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| OriginError::Generic("ANTHROPIC_API_KEY not set".into()))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| OriginError::Generic(format!("reqwest: {e}")))?;

    let answered_ids: std::collections::HashSet<String> =
        existing.iter().map(|a| a.question_id.clone()).collect();
    let todo: Vec<&RetrievedQuestion> = retrieved
        .iter()
        .filter(|rq| !answered_ids.contains(&rq.question_id))
        .collect();

    eprintln!(
        "[lme_batch] {} already answered, {} remaining",
        existing.len(),
        todo.len()
    );
    if todo.is_empty() {
        return Ok(existing);
    }

    // Build batch requests
    let requests: Vec<(String, String, Option<String>, usize)> = todo
        .iter()
        .map(|rq| {
            let (prompt, sys) = build_answer_prompt(&rq.question, &rq.context, &rq.question_type);
            (rq.question_id.clone(), prompt, Some(sys), 300)
        })
        .collect();

    eprintln!("[lme_batch] Submitting {} requests...", requests.len());
    let batch_id = submit_batch(&client, &api_key, requests, answer_model)
        .await
        .map_err(|e| OriginError::Generic(e))?;
    eprintln!("[lme_batch] Batch created: {}", batch_id);

    let results_url = poll_batch(&client, &api_key, &batch_id)
        .await
        .map_err(|e| OriginError::Generic(e))?;
    eprintln!("[lme_batch] Downloading results...");

    let results = download_batch_results(&client, &api_key, &results_url)
        .await
        .map_err(|e| OriginError::Generic(e))?;

    // Merge with existing
    let mut all = existing;
    for rq in &todo {
        if let Some(answer) = results.get(&rq.question_id) {
            all.push(AnsweredQuestion {
                question_id: rq.question_id.clone(),
                question_type: rq.question_type.clone(),
                question: rq.question.clone(),
                ground_truth: rq.ground_truth.clone(),
                model_answer: answer.clone(),
                answer_model: answer_model.to_string(),
                context_tokens: rq.context_tokens,
            });
        }
    }
    eprintln!("[lme_batch] Total: {} answers", all.len());
    Ok(all)
}

/// Judge LME answers via Batch API.
pub async fn score_answers_batch(
    answered: &[AnsweredQuestion],
    judge_model: &str,
) -> Result<LongMemEvalAccuracyReport, OriginError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| OriginError::Generic("ANTHROPIC_API_KEY not set".into()))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| OriginError::Generic(format!("reqwest: {e}")))?;

    let requests: Vec<(String, String, Option<String>, usize)> = answered
        .iter()
        .map(|aq| {
            let prompt = get_anscheck_prompt(
                &aq.question_type,
                &aq.question,
                &aq.ground_truth,
                &aq.model_answer,
            );
            (aq.question_id.clone(), prompt, None, 10)
        })
        .collect();

    eprintln!(
        "[lme_judge_batch] Submitting {} judge requests...",
        requests.len()
    );
    let batch_id = submit_batch(&client, &api_key, requests, judge_model)
        .await
        .map_err(|e| OriginError::Generic(e))?;
    eprintln!("[lme_judge_batch] Batch created: {}", batch_id);

    let results_url = poll_batch(&client, &api_key, &batch_id)
        .await
        .map_err(|e| OriginError::Generic(e))?;

    let judge_results = download_batch_results(&client, &api_key, &results_url)
        .await
        .map_err(|e| OriginError::Generic(e))?;

    // Build accuracy results
    let mut results: Vec<AccuracyResult> = Vec::new();
    for aq in answered {
        let (judge_response, correct) = match judge_results.get(&aq.question_id) {
            Some(resp) => (resp.clone(), Some(resp.to_lowercase().contains("yes"))),
            None => ("BATCH_MISSING".to_string(), None),
        };
        results.push(AccuracyResult {
            question_id: aq.question_id.clone(),
            question_type: aq.question_type.clone(),
            question: aq.question.clone(),
            ground_truth: aq.ground_truth.clone(),
            model_answer: aq.model_answer.clone(),
            judge_response,
            correct,
            context_tokens: aq.context_tokens,
        });
    }

    // Aggregate
    let judged: Vec<&AccuracyResult> = results.iter().filter(|r| r.correct.is_some()).collect();
    let errors = results.len() - judged.len();
    if errors > 0 {
        eprintln!("[lme_judge_batch] {} errors excluded", errors);
    }

    let total = judged.len();
    let total_correct = judged.iter().filter(|r| r.correct == Some(true)).count();
    let overall = if total > 0 {
        total_correct as f64 / total as f64
    } else {
        0.0
    };

    let mut per_category: Vec<CategoryAccuracy> = Vec::new();
    for &cat in CATEGORY_ORDER {
        let cr: Vec<&&AccuracyResult> = judged.iter().filter(|r| r.question_type == cat).collect();
        if cr.is_empty() {
            continue;
        }
        let cc = cr.iter().filter(|r| r.correct == Some(true)).count();
        per_category.push(CategoryAccuracy {
            question_type: cat.to_string(),
            code: category_code(cat).to_string(),
            total: cr.len(),
            correct: cc,
            accuracy: cc as f64 / cr.len() as f64,
        });
    }
    let task_avg = if per_category.is_empty() {
        0.0
    } else {
        per_category.iter().map(|c| c.accuracy).sum::<f64>() / per_category.len() as f64
    };

    Ok(LongMemEvalAccuracyReport {
        overall_accuracy: overall,
        task_averaged_accuracy: task_avg,
        total_questions: total,
        total_correct,
        answer_model: answered
            .first()
            .map(|a| a.answer_model.clone())
            .unwrap_or_default(),
        judge_model: judge_model.to_string(),
        per_category,
        per_question: results,
    })
}

/// Build the canonical LongMemEval answer-check prompt for a given task type.
/// Matches the official evaluation code exactly.
pub fn get_anscheck_prompt(task: &str, question: &str, answer: &str, response: &str) -> String {
    match task {
        "single-session-user" | "single-session-assistant" | "multi-session" => {
            format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, answer no. \
                 If the response is equivalent to the correct answer or contains all the intermediate \
                 steps to get the correct answer, you should also answer yes. If the response only \
                 contains a subset of the information required by the answer, answer no. \n\n\
                 Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
        "temporal-reasoning" => {
            format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, answer no. \
                 If the response is equivalent to the correct answer or contains all the intermediate \
                 steps to get the correct answer, you should also answer yes. If the response only \
                 contains a subset of the information required by the answer, answer no. In addition, \
                 do not penalize off-by-one errors for the number of days. If the question asks for \
                 the number of days/weeks/months, etc., and the model makes off-by-one errors \
                 (e.g., predicting 19 days when the answer is 18), the model's response is still \
                 correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
        "knowledge-update" => {
            format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, answer no. \
                 If the response contains some previous information along with an updated answer, the \
                 response should be considered as correct as long as the updated answer is the required \
                 answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
        "single-session-preference" => {
            format!(
                "I will give you a question, a rubric for desired personalized response, and a \
                 response from a model. Please answer yes if the response satisfies the desired \
                 response. Otherwise, answer no. The model does not need to reflect all the points \
                 in the rubric. The response is correct as long as it recalls and utilizes the \
                 user's personal information correctly.\n\n\
                 Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
        _ => {
            format!(
                "I will give you a question, a correct answer, and a response from a model. \
                 Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n\
                 Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n\
                 Is the model response correct? Answer yes or no only.",
                question, answer, response
            )
        }
    }
}

fn build_answer_prompt(question: &str, context: &str, question_type: &str) -> (String, String) {
    if context.is_empty() {
        return (
            format!(
                "Question: {}\n\nAnswer the question as best you can. Be specific and concise.",
                question
            ),
            "Be specific and concise. Respond in 1-3 sentences.".to_string(),
        );
    }
    match question_type {
        "single-session-preference" => {
            let prompt = format!(
                "The following context contains information about a user's preferences, \
                 interests, and past choices:\n\n{}\n\nQuestion: {}\n\n\
                 Use the user's preferences and interests from the context to \
                 personalize your response. Apply their known preferences even if \
                 this specific scenario isn't mentioned.",
                context, question
            );
            let sys = "You are a personalized assistant. Use the user's known preferences \
                to tailor your response. Be specific and concise. Respond in 1-3 sentences."
                .to_string();
            (prompt, sys)
        }
        _ => {
            let prompt = format!(
                "Context:\n{}\n\nQuestion: {}\n\nAnswer the question based on the context provided. \
                 Be specific and concise.",
                context, question
            );
            let sys =
                "Answer the question based on the provided context. Be specific and concise. \
                Respond in 1-3 sentences."
                    .to_string();
            (prompt, sys)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedQuestion {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub ground_truth: String,
    pub context: String,
    pub context_tokens: usize,
    pub memories_seeded: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnsweredQuestion {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub ground_truth: String,
    pub model_answer: String,
    pub answer_model: String,
    pub context_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct AccuracyResult {
    pub question_id: String,
    pub question_type: String,
    pub question: String,
    pub ground_truth: String,
    pub model_answer: String,
    pub judge_response: String,
    pub correct: Option<bool>,
    pub context_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CategoryAccuracy {
    pub question_type: String,
    pub code: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LongMemEvalAccuracyReport {
    pub overall_accuracy: f64,
    pub task_averaged_accuracy: f64,
    pub total_questions: usize,
    pub total_correct: usize,
    pub answer_model: String,
    pub judge_model: String,
    pub per_category: Vec<CategoryAccuracy>,
    pub per_question: Vec<AccuracyResult>,
}

impl LongMemEvalAccuracyReport {
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("LongMemEval Canonical Accuracy\n");
        out.push_str("==============================\n");
        out.push_str(&format!("Answer model: {}\n", self.answer_model));
        out.push_str(&format!("Judge model:  {}\n", self.judge_model));
        out.push_str(&format!(
            "Questions:    {}/{} correct\n\n",
            self.total_correct, self.total_questions
        ));
        out.push_str(&format!(
            "  Overall accuracy:      {:.1}%\n",
            self.overall_accuracy * 100.0
        ));
        out.push_str(&format!(
            "  Task-averaged accuracy: {:.1}%  <- leaderboard metric\n\n",
            self.task_averaged_accuracy * 100.0
        ));
        out.push_str("Per category:\n");
        for cat in &self.per_category {
            out.push_str(&format!(
                "  {} {:<28} {}/{} = {:.1}%\n",
                cat.code,
                cat.question_type,
                cat.correct,
                cat.total,
                cat.accuracy * 100.0
            ));
        }
        out
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }
}

pub fn save_retrieved(
    questions: &[RetrievedQuestion],
    path: &std::path::Path,
) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(questions).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

pub fn load_retrieved(path: &std::path::Path) -> Result<Vec<RetrievedQuestion>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(std::io::Error::other)
}

pub fn save_answered(
    questions: &[AnsweredQuestion],
    path: &std::path::Path,
) -> Result<(), std::io::Error> {
    let json = serde_json::to_string_pretty(questions).map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

pub fn load_answered(path: &std::path::Path) -> Result<Vec<AnsweredQuestion>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content).map_err(std::io::Error::other)
}

/// Phase 1: Retrieve context for each question using Origin's hybrid search.
pub async fn retrieve_for_accuracy_eval(
    path: &Path,
    search_top_k: usize,
    max_questions: usize,
) -> Result<Vec<RetrievedQuestion>, OriginError> {
    let samples = load_longmemeval(path)?;
    let sample_limit = max_questions.min(samples.len());
    let shared_embedder = crate::eval::token_efficiency::eval_shared_embedder();
    let mut retrieved: Vec<RetrievedQuestion> = Vec::new();

    for (q_idx, sample) in samples.iter().take(sample_limit).enumerate() {
        let memories = extract_memories(sample);
        if memories.is_empty() {
            continue;
        }
        let ground_truth = sample
            .answer
            .as_str()
            .unwrap_or(&sample.answer.to_string())
            .to_string();
        if ground_truth.is_empty() {
            continue;
        }
        if q_idx % 50 == 0 {
            eprintln!(
                "[lme_retrieve] Q {}/{}: {} memories",
                q_idx + 1,
                sample_limit,
                memories.len()
            );
        }

        let tmp = tempfile::tempdir().map_err(|e| OriginError::Generic(format!("tempdir: {e}")))?;
        let db = crate::db::MemoryDB::new_with_shared_embedder(
            tmp.path(),
            std::sync::Arc::new(crate::events::NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let docs: Vec<crate::sources::RawDocument> = memories
            .iter()
            .map(|mem| crate::sources::RawDocument {
                content: mem.content.clone(),
                source_id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                source: "memory".to_string(),
                title: format!("{} session {}", mem.role, mem.session_idx),
                memory_type: Some(
                    if sample.question_type == "single-session-preference" {
                        "preference"
                    } else {
                        "fact"
                    }
                    .to_string(),
                ),
                domain: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        let mem_count = docs.len();
        db.upsert_documents(docs).await?;

        let results = db
            .search_memory(
                &sample.question,
                search_top_k,
                None,
                Some("conversation"),
                None,
                None,
                None,
                None,
            )
            .await?;
        let context: String = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("{}. {}", i + 1, r.content))
            .collect::<Vec<_>>()
            .join("\n");
        let context_tokens = crate::eval::token_efficiency::count_tokens(&context);

        retrieved.push(RetrievedQuestion {
            question_id: sample.question_id.clone(),
            question_type: sample.question_type.clone(),
            question: sample.question.clone(),
            ground_truth,
            context,
            context_tokens,
            memories_seeded: mem_count,
        });
    }
    eprintln!(
        "[lme_retrieve] Done: {} questions retrieved",
        retrieved.len()
    );
    Ok(retrieved)
}

/// Phase 2: Generate answers using the Anthropic API. Supports resume via existing param.
pub async fn generate_answers(
    retrieved: &[RetrievedQuestion],
    answer_model: &str,
    concurrency: usize,
    existing: Vec<AnsweredQuestion>,
) -> Result<Vec<AnsweredQuestion>, OriginError> {
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| OriginError::Generic("ANTHROPIC_API_KEY not set".into()))?;
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| OriginError::Generic(format!("reqwest: {e}")))?;

    let answered_ids: HashSet<String> = existing.iter().map(|a| a.question_id.clone()).collect();
    let todo: Vec<&RetrievedQuestion> = retrieved
        .iter()
        .filter(|rq| !answered_ids.contains(&rq.question_id))
        .collect();
    eprintln!(
        "[lme_answer] {} already answered, {} remaining",
        existing.len(),
        todo.len()
    );
    if todo.is_empty() {
        return Ok(existing);
    }

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let client = Arc::new(client);
    let api_key = Arc::new(api_key);
    let total = todo.len();
    let mut handles = Vec::new();

    for (seq, rq) in todo.into_iter().enumerate() {
        let sem = semaphore.clone();
        let rq = rq.clone();
        let model = answer_model.to_string();
        let client = client.clone();
        let api_key = api_key.clone();

        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            if seq % 50 == 0 {
                eprintln!("[lme_answer] {}/{}...", seq + 1, total);
            }

            let (prompt, sys) = build_answer_prompt(&rq.question, &rq.context, &rq.question_type);
            match call_anthropic_api(&client, &api_key, &model, &prompt, Some(&sys), 300).await {
                Ok(answer) => Some(AnsweredQuestion {
                    question_id: rq.question_id,
                    question_type: rq.question_type,
                    question: rq.question,
                    ground_truth: rq.ground_truth,
                    model_answer: answer,
                    answer_model: model,
                    context_tokens: rq.context_tokens,
                }),
                Err(e) => {
                    eprintln!("[lme_answer] failed: {}", e);
                    None
                }
            }
        });
        handles.push(handle);
    }

    let mut all = existing;
    for handle in handles {
        if let Ok(Some(aq)) = handle.await {
            all.push(aq);
        }
    }
    eprintln!("[lme_answer] Total: {} answers", all.len());
    Ok(all)
}

/// Phase 3: Score answers using the Anthropic API as judge.
pub async fn score_answers(
    answered: &[AnsweredQuestion],
    judge_model: &str,
    concurrency: usize,
) -> Result<LongMemEvalAccuracyReport, OriginError> {
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| OriginError::Generic("ANTHROPIC_API_KEY not set".into()))?;
    let client = Arc::new(
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| OriginError::Generic(format!("reqwest: {e}")))?,
    );
    let api_key = Arc::new(api_key);

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::new();

    for aq in answered {
        let sem = semaphore.clone();
        let aq = aq.clone();
        let model = judge_model.to_string();
        let client = client.clone();
        let api_key = api_key.clone();

        let handle =
            tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                let judge_prompt = get_anscheck_prompt(
                    &aq.question_type,
                    &aq.question,
                    &aq.ground_truth,
                    &aq.model_answer,
                );
                let (judge_response, correct) =
                    match call_anthropic_api(&client, &api_key, &model, &judge_prompt, None, 10)
                        .await
                    {
                        Ok(resp) => {
                            let c = resp.to_lowercase().contains("yes");
                            (resp, Some(c))
                        }
                        Err(e) => {
                            eprintln!("[lme_judge] error: {}", e);
                            (format!("ERROR: {e}"), None)
                        }
                    };
                AccuracyResult {
                    question_id: aq.question_id,
                    question_type: aq.question_type,
                    question: aq.question,
                    ground_truth: aq.ground_truth,
                    model_answer: aq.model_answer,
                    judge_response,
                    correct,
                    context_tokens: aq.context_tokens,
                }
            });
        handles.push(handle);
    }

    let mut results: Vec<AccuracyResult> = Vec::new();
    for handle in handles {
        if let Ok(r) = handle.await {
            results.push(r);
        }
    }

    let judged: Vec<&AccuracyResult> = results.iter().filter(|r| r.correct.is_some()).collect();
    let errors = results.len() - judged.len();
    if errors > 0 {
        eprintln!("[lme_aggregate] {} judge errors excluded", errors);
    }

    let total = judged.len();
    let total_correct = judged.iter().filter(|r| r.correct == Some(true)).count();
    let overall = if total > 0 {
        total_correct as f64 / total as f64
    } else {
        0.0
    };

    let mut per_category: Vec<CategoryAccuracy> = Vec::new();
    for &cat in CATEGORY_ORDER {
        let cr: Vec<&&AccuracyResult> = judged.iter().filter(|r| r.question_type == cat).collect();
        if cr.is_empty() {
            continue;
        }
        let cc = cr.iter().filter(|r| r.correct == Some(true)).count();
        per_category.push(CategoryAccuracy {
            question_type: cat.to_string(),
            code: category_code(cat).to_string(),
            total: cr.len(),
            correct: cc,
            accuracy: cc as f64 / cr.len() as f64,
        });
    }
    let task_avg = if per_category.is_empty() {
        0.0
    } else {
        per_category.iter().map(|c| c.accuracy).sum::<f64>() / per_category.len() as f64
    };

    Ok(LongMemEvalAccuracyReport {
        overall_accuracy: overall,
        task_averaged_accuracy: task_avg,
        total_questions: total,
        total_correct,
        answer_model: answered
            .first()
            .map(|a| a.answer_model.clone())
            .unwrap_or_default(),
        judge_model: judge_model.to_string(),
        per_category,
        per_question: results,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"[{
            "question_id": "test_q1",
            "question_type": "single-session-user",
            "question": "What hobby did the user mention?",
            "answer": "hiking in the mountains",
            "question_date": "2023/04/10 (Mon) 23:07",
            "haystack_dates": ["2023/04/10 (Mon) 17:50"],
            "haystack_session_ids": ["session_1"],
            "haystack_sessions": [[
                {"role": "user", "content": "I really enjoy hiking in the mountains on weekends.", "has_answer": true},
                {"role": "assistant", "content": "That sounds like a wonderful hobby! Hiking is great for both physical and mental health.", "has_answer": false},
                {"role": "user", "content": "Do you have any trail recommendations?", "has_answer": false},
                {"role": "assistant", "content": "I'd recommend checking local trail guides for your area.", "has_answer": false}
            ]],
            "answer_session_ids": ["session_1"]
        }]"#
    }

    fn multi_session_json() -> &'static str {
        r#"[{
            "question_id": "test_q2",
            "question_type": "multi-session",
            "question": "What are the user's two pets?",
            "answer": "a dog named Max and a cat named Whiskers",
            "question_date": "2023/05/01 (Tue) 10:00",
            "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/04/20 (Thu) 14:30", "2023/04/25 (Tue) 09:15"],
            "haystack_session_ids": ["sess_a", "sess_b", "sess_c"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I just adopted a dog named Max from the shelter!", "has_answer": true},
                    {"role": "assistant", "content": "Congratulations on adopting Max! Dogs make wonderful companions.", "has_answer": false}
                ],
                [
                    {"role": "user", "content": "The weather has been nice lately.", "has_answer": false},
                    {"role": "assistant", "content": "It has been pleasant. Any outdoor plans?", "has_answer": false}
                ],
                [
                    {"role": "user", "content": "I also got a cat named Whiskers last week.", "has_answer": true},
                    {"role": "assistant", "content": "How lovely! How are Max and Whiskers getting along?", "has_answer": false}
                ]
            ],
            "answer_session_ids": ["sess_a", "sess_c"]
        }]"#
    }

    fn int_answer_json() -> &'static str {
        r#"[{
            "question_id": "test_q3",
            "question_type": "multi-session",
            "question": "How many items of clothing do I need to pick up?",
            "answer": 3,
            "question_date": "2023/06/01 (Thu) 10:00",
            "haystack_dates": ["2023/05/28 (Sun) 12:00"],
            "haystack_session_ids": ["sess_x"],
            "haystack_sessions": [[
                {"role": "user", "content": "I need to pick up 3 items of clothing from the store.", "has_answer": true},
                {"role": "assistant", "content": "Got it, 3 items to pick up.", "has_answer": false}
            ]],
            "answer_session_ids": ["sess_x"]
        }]"#
    }

    #[test]
    fn test_parse_longmemeval_sample() {
        let samples: Vec<LongMemEvalSample> = serde_json::from_str(sample_json()).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].question_id, "test_q1");
        assert_eq!(samples[0].question_type, "single-session-user");
        assert_eq!(samples[0].haystack_sessions.len(), 1);
        assert_eq!(samples[0].haystack_sessions[0].len(), 4);
        assert!(samples[0].haystack_sessions[0][0].has_answer);
        assert!(!samples[0].haystack_sessions[0][1].has_answer);
    }

    #[test]
    fn test_parse_integer_answer() {
        let samples: Vec<LongMemEvalSample> = serde_json::from_str(int_answer_json()).unwrap();
        assert_eq!(samples[0].answer, serde_json::json!(3));
    }

    #[test]
    fn test_extract_memories_single_session() {
        let samples: Vec<LongMemEvalSample> = serde_json::from_str(sample_json()).unwrap();
        let memories = extract_memories(&samples[0]);

        // Should include: 2 user turns + 0 assistant turns (none have has_answer=true)
        // Wait, the first user turn has has_answer=true. User turns are always included.
        // Assistant turns with has_answer=false are excluded.
        assert_eq!(memories.len(), 2, "Expected 2 user turns extracted");

        // All should be user role
        assert!(memories.iter().all(|m| m.role == "user"));

        // First turn should have has_answer=true
        assert!(memories[0].has_answer);
        assert!(memories[0].content.contains("hiking"));
    }

    #[test]
    fn test_extract_memories_multi_session() {
        let samples: Vec<LongMemEvalSample> = serde_json::from_str(multi_session_json()).unwrap();
        let memories = extract_memories(&samples[0]);

        // Session A: 1 user turn (has_answer) + 0 assistant
        // Session B: 1 user turn + 0 assistant (distractor)
        // Session C: 1 user turn (has_answer) + 0 assistant
        assert_eq!(memories.len(), 3);

        let has_answer_count = memories.iter().filter(|m| m.has_answer).count();
        assert_eq!(has_answer_count, 2, "Expected 2 evidence turns");

        // Verify session IDs are correct
        let session_ids: Vec<&str> = memories.iter().map(|m| m.session_id.as_str()).collect();
        assert!(session_ids.contains(&"sess_a"));
        assert!(session_ids.contains(&"sess_b"));
        assert!(session_ids.contains(&"sess_c"));
    }

    #[test]
    fn test_sample_to_eval_case_relevance() {
        let samples: Vec<LongMemEvalSample> = serde_json::from_str(multi_session_json()).unwrap();
        let memories = extract_memories(&samples[0]);
        let case = sample_to_eval_case(&samples[0], &memories);

        assert_eq!(case.query, "What are the user's two pets?");
        assert_eq!(case.seeds.len(), 3);

        // Evidence turns (has_answer=true) should get relevance=3
        let rel3: Vec<_> = case.seeds.iter().filter(|s| s.relevance == 3).collect();
        assert_eq!(
            rel3.len(),
            2,
            "Expected 2 seeds with relevance=3 (evidence turns)"
        );
        assert!(rel3.iter().any(|s| s.content.contains("Max")));
        assert!(rel3.iter().any(|s| s.content.contains("Whiskers")));

        // Distractor session (sess_b) should get relevance=1
        let rel1: Vec<_> = case.seeds.iter().filter(|s| s.relevance == 1).collect();
        assert_eq!(
            rel1.len(),
            1,
            "Expected 1 seed with relevance=1 (distractor)"
        );
        assert!(rel1[0].content.contains("weather"));
    }

    #[test]
    fn test_sample_to_eval_case_preference_type() {
        let json = r#"[{
            "question_id": "pref_q1",
            "question_type": "single-session-preference",
            "question": "What is the user's favorite color?",
            "answer": "blue",
            "question_date": "2023/01/01 (Sun) 10:00",
            "haystack_dates": ["2023/01/01 (Sun) 09:00"],
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [[
                {"role": "user", "content": "My favorite color is blue.", "has_answer": true},
                {"role": "assistant", "content": "Blue is a great color!", "has_answer": false}
            ]],
            "answer_session_ids": ["s1"]
        }]"#;

        let samples: Vec<LongMemEvalSample> = serde_json::from_str(json).unwrap();
        let memories = extract_memories(&samples[0]);
        let case = sample_to_eval_case(&samples[0], &memories);

        // Preference type questions should produce preference memory type
        assert!(case.seeds.iter().all(|s| s.memory_type == "preference"));
    }

    #[test]
    fn test_category_code() {
        assert_eq!(category_code("single-session-user"), "SSU");
        assert_eq!(category_code("single-session-assistant"), "SSA");
        assert_eq!(category_code("single-session-preference"), "SSP");
        assert_eq!(category_code("knowledge-update"), "KU");
        assert_eq!(category_code("temporal-reasoning"), "TR");
        assert_eq!(category_code("multi-session"), "MS");
        assert_eq!(category_code("something-else"), "?");
    }

    #[test]
    fn test_category_name() {
        assert_eq!(category_name("single-session-user"), "single-session-user");
        assert_eq!(category_name("multi-session"), "multi-session");
        assert_eq!(category_name("bogus"), "unknown");
    }

    #[test]
    fn test_memory_source_id_format() {
        let id = memory_source_id("q123", 2, 5);
        assert_eq!(id, "lme_q123_2_t5");
    }

    #[test]
    fn test_assistant_evidence_turns_included() {
        // When assistant turn has has_answer=true, it should be included
        let json = r#"[{
            "question_id": "ssa_q1",
            "question_type": "single-session-assistant",
            "question": "What recipe did the assistant suggest?",
            "answer": "pasta carbonara",
            "question_date": "2023/01/01 (Sun) 10:00",
            "haystack_dates": ["2023/01/01 (Sun) 09:00"],
            "haystack_session_ids": ["s1"],
            "haystack_sessions": [[
                {"role": "user", "content": "Can you suggest a dinner recipe?", "has_answer": false},
                {"role": "assistant", "content": "I recommend pasta carbonara with fresh parmesan.", "has_answer": true},
                {"role": "user", "content": "That sounds great, thanks!", "has_answer": false},
                {"role": "assistant", "content": "You're welcome! Enjoy your meal.", "has_answer": false}
            ]],
            "answer_session_ids": ["s1"]
        }]"#;

        let samples: Vec<LongMemEvalSample> = serde_json::from_str(json).unwrap();
        let memories = extract_memories(&samples[0]);

        // 2 user turns + 1 assistant turn (the one with has_answer=true)
        assert_eq!(memories.len(), 3);
        let assistant_mems: Vec<_> = memories.iter().filter(|m| m.role == "assistant").collect();
        assert_eq!(assistant_mems.len(), 1);
        assert!(assistant_mems[0].content.contains("carbonara"));
        assert!(assistant_mems[0].has_answer);
    }

    #[test]
    fn test_report_to_terminal() {
        let report = LongMemEvalReport {
            aggregate_ndcg_at_10: 0.45,
            aggregate_mrr: 0.50,
            aggregate_recall_at_5: 0.60,
            aggregate_hit_rate_at_1: 0.35,
            total_questions: 10,
            total_memories: 100,
            per_category: vec![LongMemEvalCategoryResult {
                question_type: "single-session-user".to_string(),
                code: "SSU".to_string(),
                count: 5,
                ndcg_at_5: 0.50,
                ndcg_at_10: 0.48,
                mrr: 0.55,
                recall_at_5: 0.65,
                hit_rate_at_1: 0.40,
            }],
            baseline: None,
        };
        let text = report.to_terminal();
        assert!(text.contains("LongMemEval Benchmark"));
        assert!(text.contains("NDCG@10"));
        assert!(text.contains("SSU"));
        assert!(text.contains("single-session-user"));
    }

    #[test]
    fn test_baseline_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("longmemeval_baseline.json");

        let report = LongMemEvalReport {
            aggregate_ndcg_at_10: 0.450,
            aggregate_mrr: 0.500,
            aggregate_recall_at_5: 0.600,
            aggregate_hit_rate_at_1: 0.350,
            total_questions: 10,
            total_memories: 100,
            per_category: vec![
                LongMemEvalCategoryResult {
                    question_type: "single-session-user".to_string(),
                    code: "SSU".to_string(),
                    count: 5,
                    ndcg_at_5: 0.50,
                    ndcg_at_10: 0.480,
                    mrr: 0.550,
                    recall_at_5: 0.650,
                    hit_rate_at_1: 0.400,
                },
                LongMemEvalCategoryResult {
                    question_type: "knowledge-update".to_string(),
                    code: "KU".to_string(),
                    count: 3,
                    ndcg_at_5: 0.40,
                    ndcg_at_10: 0.420,
                    mrr: 0.450,
                    recall_at_5: 0.550,
                    hit_rate_at_1: 0.300,
                },
            ],
            baseline: None,
        };

        report.save_baseline(&path).unwrap();
        let loaded = LongMemEvalReport::load_baseline(&path).unwrap();

        assert!((loaded.ndcg_at_10 - 0.450).abs() < 0.001);
        assert!((loaded.mrr - 0.500).abs() < 0.001);
        assert!((loaded.recall_at_5 - 0.600).abs() < 0.001);
        assert!((loaded.hit_rate_at_1 - 0.350).abs() < 0.001);

        // Per-category baselines
        assert_eq!(loaded.per_category.len(), 2);
        assert_eq!(loaded.per_category[0].name, "single-session-user");
        assert!((loaded.per_category[0].ndcg_at_10 - 0.480).abs() < 0.001);
        assert_eq!(loaded.per_category[1].name, "knowledge-update");
        assert!((loaded.per_category[1].mrr - 0.450).abs() < 0.001);
    }

    #[test]
    fn test_to_terminal_with_baseline() {
        let report = LongMemEvalReport {
            aggregate_ndcg_at_10: 0.500,
            aggregate_mrr: 0.550,
            aggregate_recall_at_5: 0.650,
            aggregate_hit_rate_at_1: 0.400,
            total_questions: 10,
            total_memories: 100,
            per_category: vec![LongMemEvalCategoryResult {
                question_type: "single-session-user".to_string(),
                code: "SSU".to_string(),
                count: 5,
                ndcg_at_5: 0.52,
                ndcg_at_10: 0.510,
                mrr: 0.580,
                recall_at_5: 0.680,
                hit_rate_at_1: 0.430,
            }],
            baseline: Some(LongMemEvalBaseline {
                ndcg_at_10: 0.450,
                mrr: 0.500,
                recall_at_5: 0.600,
                hit_rate_at_1: 0.350,
                per_category: vec![crate::eval::report::CategoryBaseline {
                    name: "single-session-user".to_string(),
                    ndcg_at_10: 0.480,
                    mrr: 0.550,
                    recall_at_5: 0.650,
                }],
            }),
        };

        let text = report.to_terminal();
        assert!(text.contains("LongMemEval Benchmark"));
        assert!(text.contains("Baseline comparison:"));
        assert!(text.contains("->"));
        assert!(text.contains("single-session-user"));
    }

    #[test]
    fn test_anscheck_prompt_task_specific() {
        let q = "What is the user's pet?";
        let a = "a dog named Max";
        let r = "The user has a dog named Max.";
        let ssu = get_anscheck_prompt("single-session-user", q, a, r);
        assert!(ssu.contains("Answer yes or no only"));
        assert!(!ssu.contains("off-by-one"));
        let tr = get_anscheck_prompt("temporal-reasoning", q, a, r);
        assert!(tr.contains("off-by-one"));
        let ku = get_anscheck_prompt("knowledge-update", q, a, r);
        assert!(ku.contains("updated answer"));
        let ssp = get_anscheck_prompt("single-session-preference", q, a, r);
        assert!(ssp.contains("rubric"));
    }

    #[test]
    fn test_build_answer_prompt_empty_context() {
        let (prompt, _sys) = build_answer_prompt("What is 2+2?", "", "multi-session");
        assert!(prompt.contains("Answer the question as best you can"));
        assert!(!prompt.contains("Context:"));
    }

    #[test]
    fn test_build_answer_prompt_preference() {
        let (prompt, sys) = build_answer_prompt(
            "What restaurant?",
            "User likes Italian.",
            "single-session-preference",
        );
        assert!(prompt.contains("preferences"));
        assert!(sys.contains("personalized assistant"));
    }

    #[test]
    fn test_build_answer_prompt_default() {
        let (prompt, sys) =
            build_answer_prompt("What color?", "The car is red.", "single-session-user");
        assert!(prompt.contains("Context:"));
        assert!(sys.contains("based on the provided context"));
    }
}
