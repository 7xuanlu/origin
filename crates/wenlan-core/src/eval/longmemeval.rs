// SPDX-License-Identifier: Apache-2.0
//! LongMemEval benchmark adapter — converts LongMemEval dataset into Wenlan eval cases.
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

use crate::db::{EvalTemporalSeed, MemoryDB};
use crate::error::WenlanError;
use crate::eval::fixtures::{EvalCase, SeedMemory};
use crate::eval::metrics;
use crate::quality_gate::QualityGate;
use crate::read_scope::ReadScope;
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
pub fn load_longmemeval(path: &Path) -> Result<Vec<LongMemEvalSample>, WenlanError> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| WenlanError::Generic(format!("Failed to read LongMemEval file: {e}")))?;
    let samples: Vec<LongMemEvalSample> = serde_json::from_str(&data)
        .map_err(|e| WenlanError::Generic(format!("Failed to parse LongMemEval JSON: {e}")))?;
    Ok(samples)
}

/// Truncate the loaded LongMemEval samples in place if `EVAL_LME_LIMIT` is set
/// to a positive integer. Used by every `run_longmemeval_eval*` variant so a
/// developer can run a small pre-flight subset (~30min) before committing
/// to a full multi-hour run.
fn apply_lme_limit(samples: &mut Vec<LongMemEvalSample>) {
    // EVAL_LME_STRATIFIED=N keeps the first N questions of EACH question_type so a
    // small fast run still covers all 6 categories. The fixture is sorted by type,
    // so a plain front-truncate (EVAL_LME_LIMIT) only hits the first category.
    // Takes precedence over EVAL_LME_LIMIT when set.
    if let Some(n) = std::env::var("EVAL_LME_STRATIFIED")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        let mut seen: HashMap<String, usize> = HashMap::new();
        samples.retain(|s| {
            let c = seen.entry(s.question_type.clone()).or_insert(0);
            *c += 1;
            *c <= n
        });
        log::warn!(
            "[eval/longmemeval] EVAL_LME_STRATIFIED={} active -- {} questions across {} categories",
            n,
            samples.len(),
            seen.len()
        );
        return;
    }
    apply_limit_from_env(samples, "EVAL_LME_LIMIT", "longmemeval", "questions");
}

/// Shared helper for `apply_lme_limit`. Parameterized on the env var name so
/// unit tests can exercise the behavior without racing the production var.
fn apply_limit_from_env<T>(samples: &mut Vec<T>, env_var: &str, bench_tag: &str, unit_label: &str) {
    let Some(limit) = std::env::var(env_var)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    else {
        return;
    };
    let total = samples.len();
    if limit < total {
        samples.truncate(limit);
        log::warn!(
            "[eval/{}] {}={} active -- running on {} of {} {}",
            bench_tag,
            env_var,
            limit,
            samples.len(),
            total,
            unit_label
        );
    }
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

/// Build a `{source_id -> event_date(unix seconds)}` map from LongMemEval session
/// metadata, for eval-seed `event_date` injection (T11/T20 temporal).
///
/// Turn TEXT carries no date, so classify-from-text recovers no `event_date`; the
/// per-session date lives in `haystack_dates`, parallel to `haystack_session_ids`.
/// Reuses the module's existing [`parse_lme_date`] and truncates to midnight UTC
/// (day-level matching). `source_id` mirrors the seed builder via [`memory_source_id`].
pub fn event_date_map(samples: &[LongMemEvalSample]) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    for sample in samples {
        for mem in extract_memories(sample) {
            let Some(date_str) = sample.haystack_dates.get(mem.session_idx) else {
                continue;
            };
            let Some(dt) = parse_lme_date(date_str) else {
                continue;
            };
            let Some(midnight) = dt.date_naive().and_hms_opt(0, 0, 0) else {
                continue;
            };
            map.insert(
                memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                midnight.and_utc().timestamp(),
            );
        }
    }
    map
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
                space: Some("conversation".to_string()),
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
        space: Some("conversation".to_string()),
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

/// Build a `CaseResult` for one LongMemEval QA sample. Metrics not computed by
/// the LME runner (map_at_10, precision_at_3, negatives) are zero-filled.
#[allow(clippy::too_many_arguments)]
fn build_lme_case_result(
    question: &str,
    question_type: &str,
    ndcg_5: f64,
    ndcg_10: f64,
    mrr_val: f64,
    recall_5: f64,
    hr_1: f64,
) -> crate::eval::report::CaseResult {
    crate::eval::report::CaseResult {
        query: question.to_string(),
        ndcg_at_10: ndcg_10,
        ndcg_at_5: ndcg_5,
        map_at_10: 0.0,
        mrr: mrr_val,
        recall_at_5: recall_5,
        hit_rate_at_1: hr_1,
        precision_at_3: 0.0,
        negative_leakage: 0,
        neg_above_relevant: 0,
        category: Some(category_name(question_type).to_string()),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<crate::eval::report::CoverageRecall>,
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
    /// Run environment capture (schema v1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<crate::eval::report::ReportEnv>,
    /// Per-question results for paired comparison (McNemar etc.).
    /// Populated by each runner variant; empty by default for back-compat.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_case: Vec<crate::eval::report::CaseResult>,
    /// Source-expanded coverage recall (page-channel `_from_db` runner only;
    /// None elsewhere). See [`crate::eval::report::CoverageRecall`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<crate::eval::report::CoverageRecall>,
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

        if let Some(ref cov) = self.coverage {
            out.push_str(&format!(
                "  Coverage recall (set-based, page-source-expanded):\n    blind:    {:.4}\n    expanded: {:.4}\n    delta:    {:+.4}  <- page contribution\n",
                cov.blind,
                cov.expanded,
                cov.expanded - cov.blind
            ));
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
            coverage: self.coverage.clone(),
        };
        let json = serde_json::to_string_pretty(&baseline).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load baseline from a previous run for comparison.
    pub fn load_baseline(path: &Path) -> Option<LongMemEvalBaseline> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Encode retrieval variant + provider + fixture-hash into baseline filename.
    /// Falls back to base + ".json" if env is missing (back-compat).
    pub fn baseline_filename(&self, base: &str) -> String {
        crate::eval::report::encode_baseline_filename(self.env.as_ref(), base)
    }

    /// Project this LongMemEvalReport onto the flat `EvalReport` shape so the
    /// P0b layered baseline path (`save_full_report`) can consume it.
    ///
    /// Same mapping policy as `LocomoReport::to_eval_report` — only metrics
    /// surfaced by the LongMemEval runner are populated; the rest are
    /// zero-filled, and per-case data stays on the strongly-typed report.
    pub fn to_eval_report(&self) -> crate::eval::report::EvalReport {
        let search_mode = self
            .env
            .as_ref()
            .map(|e| e.retrieval_method.clone())
            .unwrap_or_else(|| "longmemeval".to_string());
        crate::eval::report::EvalReport {
            fixture_count: self.total_questions,
            file_count: 1,
            search_mode,
            ndcg_at_10: self.aggregate_ndcg_at_10,
            ndcg_at_5: 0.0,
            map_at_5: 0.0,
            map_at_10: 0.0,
            mrr: self.aggregate_mrr,
            recall_at_1: 0.0,
            recall_at_3: 0.0,
            recall_at_5: self.aggregate_recall_at_5,
            hit_rate_at_1: self.aggregate_hit_rate_at_1,
            hit_rate_at_3: 0.0,
            precision_at_3: 0.0,
            precision_at_5: 0.0,
            neg_above_relevant: 0,
            total_negatives: 0,
            negative_leakage: 0,
            gate_content_filtered: 0,
            gate_novelty_filtered: 0,
            empty_set_count: 0,
            empty_set_false_confidence: None,
            score_gap: None,
            temporal_ordering_total: 0,
            temporal_ordering_correct: 0,
            temporal_ordering_rate: None,
            baseline: None,
            per_case: self.per_case.clone(),
            env: self.env.clone(),
            latency: None,
            total_scenarios: self.total_questions,
            skipped_scenarios: Vec::new(),
            enrichment_failures: 0,
            truncated_reason: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ReportEnv builder
// ---------------------------------------------------------------------------

/// Build a `ReportEnv` for a LongMemEval runner variant.
///
/// Fills both the legacy 9 fields (needed by `encode_baseline_filename`) and
/// the new P0a additive fields. The `llm_provider_class` / `llm_model` legacy
/// fields and the new P0a fields carry the same information so both views of
/// the data stay consistent.
fn build_lme_env(
    variant: &str,
    path: &std::path::Path,
    retrieval_method: &str,
    llm_provider_class: &str,
    llm_model: &str,
    judge_model: Option<String>,
) -> crate::eval::report::ReportEnv {
    let fixture_revision =
        crate::eval::fixtures::fixture_revision_hash(path).unwrap_or_else(|_| "unknown".into());
    let n_runs: u32 = 1;
    let run_id = Some(format!(
        "run_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let now = chrono::Utc::now();
    let timestamp_utc = Some(now.to_rfc3339());
    crate::eval::report::ReportEnv {
        // Legacy fields (needed by encode_baseline_filename and existing callers)
        fixture_revision,
        embedder_model: "BGE-Base-EN-v1.5-Q".into(),
        embedder_revision: "768d".into(),
        retrieval_method: retrieval_method.to_string(),
        llm_provider_class: llm_provider_class.to_string(),
        llm_model: llm_model.to_string(),
        judge_model: judge_model.clone(),
        origin_version: env!("CARGO_PKG_VERSION").into(),
        eval_timestamp_unix: now.timestamp(),
        // P0a additive fields
        layer: Some(crate::eval::EvalLayer::L1Db),
        task: Some("lme".to_string()),
        variant: Some(variant.to_string()),
        embed_dim: Some(768),
        similarity_fn_name: "cosine".to_string(),
        judge_model_id: judge_model,
        mcp_schema_hash: None,
        skill_prompt_hash: None,
        schema_version: 1,
        schema_db_version: Some(crate::db::SCHEMA_VERSION),
        migrations_hash: option_env!("WENLAN_MIGRATIONS_HASH").map(String::from),
        n_runs,
        is_single_run: n_runs == 1,
        run_id,
        timestamp_utc,
        git_sha: option_env!("WENLAN_GIT_SHA").map(String::from),
        warmup_iterations: 0,
        ..Default::default()
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
async fn run_longmemeval_eval_core(
    path: &Path,
    reranker: Option<std::sync::Arc<dyn crate::reranker::Reranker>>,
    variant: &str,
    retrieval_method: &str,
    llm_provider_class: &str,
    llm_model: &str,
) -> Result<LongMemEvalReport, WenlanError> {
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    // (question_type, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut per_case: Vec<crate::eval::report::CaseResult> = Vec::new();
    let mut total_memories: usize = 0;

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| WenlanError::Generic(format!("tempdir: {e}")))?;
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
                    space: Some("conversation".to_string()),
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

        let results = if let Some(reranker) = &reranker {
            db.search_memory_cross_rerank(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                Some(reranker.clone()),
            )
            .await?
        } else {
            db.search_memory(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await?
        };

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
        per_case.push(build_lme_case_result(
            &sample.question,
            &sample.question_type,
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    // Aggregate
    let per_category = aggregate_by_category(&all_scores);

    let mut report = LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories,
        per_category,
        baseline: None,
        env: None,
        per_case,
        coverage: None,
    };
    let mut env_stamp = build_lme_env(
        variant,
        path,
        retrieval_method,
        llm_provider_class,
        llm_model,
        None,
    );
    env_stamp.flags.push(format!(
        "graph_memory_stream={}",
        if crate::db::graph_memory_stream_enabled() {
            "on"
        } else {
            "off"
        }
    ));
    report.env = Some(env_stamp);
    Ok(report)
}

pub async fn run_longmemeval_eval(path: &Path) -> Result<LongMemEvalReport, WenlanError> {
    run_longmemeval_eval_core(path, None, "base", "search_memory", "none", "none").await
}

// ---------------------------------------------------------------------------
// Cross-encoder rerank benchmark runner — same as run_longmemeval_eval_reranked
// but swaps the LLM reranker for a cross-encoder model (fastembed TextRerank).
// ---------------------------------------------------------------------------

/// Same seeding/scoring logic as `run_longmemeval_eval_reranked`, but retrieval
/// uses `search_memory_cross_rerank` driven by a cross-encoder reranker
/// (model per `WENLAN_RERANKER_MODEL`; default `BGERerankerBase` since
/// 2026-06-11). Lets the eval sweep compare LLM-as-judge
/// reranking against a purpose-built cross-encoder on identical fixtures.
pub async fn run_longmemeval_eval_cross_rerank(
    path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
) -> Result<LongMemEvalReport, WenlanError> {
    let llm_model = format!("cross-encoder:{}", reranker.model_id());
    run_longmemeval_eval_core(
        path,
        Some(reranker),
        "cross_rerank",
        "search_memory_with_reranker",
        "cross-encoder",
        &llm_model,
    )
    .await
}

// ---------------------------------------------------------------------------
// Cross-encoder rerank runner against a pre-seeded consolidated DB (PR-B)
// ---------------------------------------------------------------------------

/// Like `run_longmemeval_eval_cross_rerank`, but scores against a PRE-SEEDED
/// consolidated scenario DB (no per-question ephemeral DB, no ingest).
/// Used by PR-B's page-channel eval to surface distilled pages that the
/// fullpipeline harness wrote into the cache.
///
/// `db` MUST already contain memories with `source_id` formatted as
/// `lme_<question_id>_<session_idx>_t<turn_idx>` (matches the
/// `memory_source_id` function used by the LME ephemeral seed path and
/// the fullpipeline harness).
/// Page-channel ON/OFF is controlled by the caller via the
/// `WENLAN_ENABLE_PAGE_CHANNEL` env var (read inside
/// `search_memory_cross_rerank`).
pub async fn run_longmemeval_eval_cross_rerank_from_db(
    db: &MemoryDB,
    path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
) -> Result<LongMemEvalReport, WenlanError> {
    // Reproducibility: pin the ungated rerank-pool tuning knobs read raw at
    // db.rs:8922-8926 (search_memory_cross_rerank) to their production defaults
    // (compute_rerank_fetch_pool: multiplier=1, floor=10) when unset, so two
    // cross-rerank baselines with identical env stamps can't silently measure
    // different pool depths. --test-threads=1 makes process-global set_var safe.
    if std::env::var_os("RERANK_POOL_MULTIPLIER").is_none() {
        std::env::set_var("RERANK_POOL_MULTIPLIER", "1");
    }
    if std::env::var_os("RERANK_POOL_FLOOR").is_none() {
        std::env::set_var("RERANK_POOL_FLOOR", "10");
    }
    // Threat-2 guard: assert summary_nodes is empty so an accidental future
    // populate fails loud here instead of silently demoting gold memories via
    // the global-prelude prepend at db.rs:9268. A missing table reads as zero.
    crate::eval::paired::assert_summary_nodes_empty(db).await;

    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    // (question_type, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut per_case: Vec<crate::eval::report::CaseResult> = Vec::new();
    let mut total_memories: usize = 0;
    let mut cov_blind_acc: Vec<f64> = Vec::new();
    let mut cov_expanded_acc: Vec<f64> = Vec::new();

    for sample in &samples {
        let memories = extract_memories(sample);
        total_memories += memories.len();

        // Build relevance judgments: has_answer turns are relevant.
        // source_id format matches both the ephemeral seed and fullpipeline harness.
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();

        if relevant_source_ids.is_empty() {
            continue;
        }

        let results = db
            .search_memory_cross_rerank(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                Some(reranker.clone()),
            )
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

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
        per_case.push(build_lme_case_result(
            &sample.question,
            &sample.question_type,
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));

        // Source-expanded coverage recall (page provenance, eval-only).
        // Pages contribute the memory source ids they were distilled from;
        // memories contribute their own id. Set-based + deduped so a gold
        // id counts once (no double-count). Reads the full result bundle
        // (memories in [0..limit] + appended pages) — pages ride after the
        // limit by the iter2 partition.
        let page_src_owned: Vec<(String, Vec<String>)> = {
            let mut v = Vec::new();
            for r in &results {
                if r.source == "page" {
                    let srcs: Vec<String> = db
                        .get_page_sources(&r.source_id)
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .map(|ps| ps.memory_source_id)
                        .collect();
                    v.push((r.source_id.clone(), srcs));
                }
            }
            v
        };
        let page_sources_map: HashMap<&str, Vec<&str>> = page_src_owned
            .iter()
            .map(|(pid, srcs)| (pid.as_str(), srcs.iter().map(|s| s.as_str()).collect()))
            .collect();
        let units: Vec<(&str, &str)> = results
            .iter()
            .map(|r| (r.source.as_str(), r.source_id.as_str()))
            .collect();
        let cov_blind = metrics::coverage_recall(
            &metrics::build_coverage_set(&units, &HashMap::new()),
            &relevant_set,
        );
        let cov_expanded = metrics::coverage_recall(
            &metrics::build_coverage_set(&units, &page_sources_map),
            &relevant_set,
        );
        cov_blind_acc.push(cov_blind);
        cov_expanded_acc.push(cov_expanded);
    }

    let per_category = aggregate_by_category(&all_scores);
    let coverage = if cov_blind_acc.is_empty() {
        None
    } else {
        Some(crate::eval::report::CoverageRecall {
            blind: cov_blind_acc.iter().sum::<f64>() / cov_blind_acc.len() as f64,
            expanded: cov_expanded_acc.iter().sum::<f64>() / cov_expanded_acc.len() as f64,
        })
    };
    let mut report = LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories,
        per_category,
        baseline: None,
        env: None,
        per_case,
        coverage,
    };

    let env_stamp = crate::eval::shared::stamp_cr_from_db_env(|variant_tag| {
        build_lme_env(
            variant_tag,
            path,
            "search_memory_with_reranker",
            "cross-encoder",
            &format!("cross-encoder:{}", reranker.model_id()),
            None,
        )
    });
    report.env = Some(env_stamp);
    Ok(report)
}

/// Retrieval eval over a pre-seeded DB using the base `search_memory` path
/// (vector + FTS + RRF + graph augmentation) — the path the graph gate acts on.
/// Mirrors `run_longmemeval_eval_cross_rerank_from_db` minus the cross-encoder/
/// page channel. Reports the graph-gate skip rate (queries that bypass graph)
/// when `WENLAN_ENABLE_GRAPH_GATE` is on. Used for the T3 graph-gate A/B.
pub async fn run_longmemeval_eval_from_db(
    db: &MemoryDB,
    path: &Path,
) -> Result<LongMemEvalReport, WenlanError> {
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut total_memories: usize = 0;
    let mut cov_acc: Vec<f64> = Vec::new();
    let gate_on = crate::db::graph_gate_enabled();
    let (mut gate_skipped, mut gate_total) = (0usize, 0usize);

    for sample in &samples {
        let memories = extract_memories(sample);
        total_memories += memories.len();
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }

        gate_total += 1;
        if gate_on && !crate::retrieval::signals::query_warrants_graph(&sample.question) {
            gate_skipped += 1;
        }
        let results = db
            .search_memory(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| (*id, u8::from(relevant_source_ids.contains(*id))))
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

        let units: Vec<(&str, &str)> = results
            .iter()
            .map(|r| (r.source.as_str(), r.source_id.as_str()))
            .collect();
        cov_acc.push(metrics::coverage_recall(
            &metrics::build_coverage_set(&units, &HashMap::new()),
            &relevant_set,
        ));
    }
    if gate_on {
        eprintln!(
            "[lme] graph-gate skipped {gate_skipped}/{gate_total} queries ({:.1}%)",
            100.0 * gate_skipped as f64 / gate_total.max(1) as f64
        );
    }

    let per_category = aggregate_by_category(&all_scores);
    let coverage = if cov_acc.is_empty() {
        None
    } else {
        Some(crate::eval::report::CoverageRecall {
            blind: cov_acc.iter().sum::<f64>() / cov_acc.len() as f64,
            expanded: cov_acc.iter().sum::<f64>() / cov_acc.len() as f64,
        })
    };
    let mut report = LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories,
        per_category,
        baseline: None,
        env: None,
        per_case: Vec::new(),
        coverage,
    };
    let graph_gate = if gate_on { "on" } else { "off" };
    let mut env_stamp = build_lme_env(
        if gate_on {
            "search_memory_gate_on"
        } else {
            "search_memory_gate_off"
        },
        path,
        "search_memory",
        "none",
        "none",
        None,
    );
    env_stamp.flags.push(format!("graph_gate={graph_gate}"));
    env_stamp.flags.push(format!(
        "graph_memory_stream={}",
        if crate::db::graph_memory_stream_enabled() {
            "on"
        } else {
            "off"
        }
    ));
    env_stamp.flags.push("scenario_db=consolidated".to_string());
    report.env = Some(env_stamp);
    Ok(report)
}

// ---------------------------------------------------------------------------
// Per-query collector (paired A/B apparatus v2)
// ---------------------------------------------------------------------------

/// Per-query variant of [`run_longmemeval_eval_from_db`]. Identical retrieval +
/// scoring, but emits one [`PerQueryRow`] per evaluated question (with a
/// wall-clock latency for the `search_memory` call) instead of aggregating.
async fn run_longmemeval_eval_from_db_collect_core(
    db: &MemoryDB,
    path: &Path,
    reranker: Option<std::sync::Arc<dyn crate::reranker::Reranker>>,
    feature: &str,
    flag_state: &str,
    needs_base_ids: bool,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    use crate::eval::paired::PerQueryRow;
    use std::time::Instant;

    let use_ce = reranker.is_some();
    if use_ce {
        if std::env::var_os("RERANK_POOL_MULTIPLIER").is_none() {
            std::env::set_var("RERANK_POOL_MULTIPLIER", "1");
        }
        if std::env::var_os("RERANK_POOL_FLOOR").is_none() {
            std::env::set_var("RERANK_POOL_FLOOR", "10");
        }
    }

    // No-drift eval gate: a graph/temporal A/B over an empty substrate is a null,
    // not a result. Same contract the seed orchestrator asserts (producer/consumer).
    db.assert_eval_feature_substrate_live(feature).await?;

    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let gate_on = if use_ce {
        false
    } else {
        crate::db::graph_gate_enabled()
    };
    let mut rows: Vec<PerQueryRow> = Vec::new();

    for sample in &samples {
        let memories = extract_memories(sample);
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }

        let graph_skipped =
            gate_on && !crate::retrieval::signals::query_warrants_graph(&sample.question);

        let base_ids_owned: Vec<String> = if needs_base_ids {
            db.search_memory(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await?
            .iter()
            .map(|r| r.source_id.clone())
            .collect()
        } else {
            vec![]
        };
        let base_ids: Vec<&str> = base_ids_owned.iter().map(|s| s.as_str()).collect();

        let t0 = Instant::now();
        let results = if let Some(reranker) = &reranker {
            db.search_memory_cross_rerank(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                Some(reranker.clone()),
            )
            .await?
        } else {
            db.search_memory(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await?
        };
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        let channel_touched = if use_ce {
            // After result_ids exist and outside the latency window: CE-path probe.
            crate::eval::shared::ce_channel_touched(
                db,
                feature,
                &sample.question,
                &base_ids,
                &result_ids,
            )
            .await?
        } else {
            // Outside the latency window: the base-path channel-touch probe.
            crate::eval::shared::base_channel_touched(db, feature, &sample.question).await?
        };
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| (*id, u8::from(relevant_source_ids.contains(*id))))
            .collect();
        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        // For page_channel feature: compute both blind (no expansion) and expanded coverage
        let (cov_blind, cov_expanded) = if use_ce && feature == "page_channel" {
            let page_src_owned: Vec<(String, Vec<String>)> = {
                let mut v = Vec::new();
                for r in &results {
                    if r.source == "page" {
                        let srcs: Vec<String> = db
                            .get_page_sources(&r.source_id)
                            .await
                            .unwrap_or_default()
                            .into_iter()
                            .map(|ps| ps.memory_source_id)
                            .collect();
                        v.push((r.source_id.clone(), srcs));
                    }
                }
                v
            };
            let page_sources_map: HashMap<&str, Vec<&str>> = page_src_owned
                .iter()
                .map(|(pid, srcs)| (pid.as_str(), srcs.iter().map(|s| s.as_str()).collect()))
                .collect();
            let units: Vec<(&str, &str)> = results
                .iter()
                .map(|r| (r.source.as_str(), r.source_id.as_str()))
                .collect();
            (
                Some(metrics::coverage_recall(
                    &metrics::build_coverage_set(&units, &HashMap::new()),
                    &relevant_set,
                )),
                Some(metrics::coverage_recall(
                    &metrics::build_coverage_set(&units, &page_sources_map),
                    &relevant_set,
                )),
            )
        } else {
            (None, None)
        };

        rows.push(PerQueryRow {
            feature: feature.to_string(),
            bench: "lme".to_string(),
            flag_state: flag_state.to_string(),
            query_id: sample.question_id.clone(),
            category: sample.question_type.clone(),
            ndcg10: metrics::ndcg_at_k(&result_ids, &grades, 10),
            recall5: metrics::recall_at_k(&result_ids, &relevant_set, 5),
            mrr: metrics::mrr(&result_ids, &relevant_set),
            latency_ms,
            graph_skipped: if gate_on { Some(graph_skipped) } else { None },
            temporal_touched: None,
            channel_touched,
            cov_blind,
            cov_expanded,
        });
    }

    Ok(rows)
}

pub async fn run_longmemeval_eval_from_db_collect(
    db: &MemoryDB,
    path: &Path,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    run_longmemeval_eval_from_db_collect_core(db, path, None, feature, flag_state, false).await
}

/// One row of the recall-headroom probe: where the gold evidence turns sit in
/// the base path's ranked list at three separate fetch limits.
///
/// Each limit is a SEPARATE `search_memory` call because candidate generation
/// scales with the limit (`fetch_limit = limit * 3` per channel) and dedup +
/// graph augmentation run over that scaled pool — ranks from one deep call do
/// NOT equal where gold sits in a shallower production call. `gold_in_10` is
/// production semantics (limit=10), `gold_in_30` is the CE knee window
/// (RERANK_POOL_FLOOR=30) semantics, `gold_in_100` is the deep single-query
/// fetch ceiling.
#[derive(Debug, serde::Serialize)]
pub struct HeadroomRow {
    pub query_id: String,
    pub category: String,
    pub n_gold: usize,
    /// 0-based rank of each gold id in the limit=100 call, -1 = absent.
    /// Sorted ascending with absences last so output is deterministic.
    pub gold_ranks_100: Vec<i64>,
    pub gold_in_10: usize,
    pub gold_in_30: usize,
    pub gold_in_100: usize,
    pub hit_any_10: bool,
    pub hit_any_30: bool,
    pub hit_any_100: bool,
}

/// Count gold ranks that landed inside the top-`k` (absences are `-1`).
fn gold_in_top_k(ranks: &[i64], k: usize) -> usize {
    ranks
        .iter()
        .filter(|&&r| r >= 0 && (r as usize) < k)
        .count()
}

/// Recall-headroom probe over the base `search_memory` path (Step 0 of the
/// decompose probe ladder). For each question, runs the base path at three
/// fetch limits (10 / 30 / 100) and records how many gold evidence turns each
/// list contains. Gold absent at the deep limit is recall headroom a
/// single-query fetch cannot reach — the multi-anchor / decompose case; gold
/// reachable at 30 but not 10 is CE-knee-window territory (RERANK_POOL_FLOOR);
/// gold reachable at 100 but not 30 is pool-widening territory.
/// No flag arms: this measures the substrate, not a lever, so there is no
/// substrate-liveness gate either (the base path has no starvable channel).
pub async fn run_longmemeval_headroom_probe_from_db(
    db: &MemoryDB,
    path: &Path,
) -> Result<Vec<HeadroomRow>, WenlanError> {
    const LIMITS: [usize; 3] = [10, 30, 100];
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut rows: Vec<HeadroomRow> = Vec::new();

    for sample in &samples {
        let memories = extract_memories(sample);
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }

        // One independent search per limit: per-limit candidate generation is
        // the point (see HeadroomRow docs), so ranks are computed within each
        // call's own list, never derived by truncating the deep call.
        let mut gold_in = [0usize; 3];
        let mut deep_ranks: Vec<i64> = Vec::new();
        for (i, &k) in LIMITS.iter().enumerate() {
            let results = db
                .search_memory(
                    &sample.question,
                    k,
                    None,
                    &ReadScope::Global,
                    None,
                    None,
                    None,
                    None,
                )
                .await?;
            let ranks: Vec<i64> = relevant_source_ids
                .iter()
                .map(|gid| {
                    results
                        .iter()
                        .position(|r| r.source_id == *gid)
                        .map(|p| p as i64)
                        .unwrap_or(-1)
                })
                .collect();
            gold_in[i] = gold_in_top_k(&ranks, k);
            if k == 100 {
                deep_ranks = ranks;
                deep_ranks.sort_by_key(|&r| if r < 0 { i64::MAX } else { r });
            }
        }

        rows.push(HeadroomRow {
            query_id: sample.question_id.clone(),
            category: sample.question_type.clone(),
            n_gold: relevant_source_ids.len(),
            gold_ranks_100: deep_ranks,
            gold_in_10: gold_in[0],
            gold_in_30: gold_in[1],
            gold_in_100: gold_in[2],
            hit_any_10: gold_in[0] > 0,
            hit_any_30: gold_in[1] > 0,
            hit_any_100: gold_in[2] > 0,
        });
    }

    Ok(rows)
}

/// Per-question variant of the recall-headroom probe. Opens each question's OWN
/// seeded scenario DB (the faithful per-question instrument) instead of one
/// consolidated DB, and records where gold sits in that question's own base
/// `search_memory` ranking at limits 10/30/100. This is the honest gold-rank
/// measurement for deciding whether a deeper CE rerank pool can ever help on
/// per-question LME-S: gold reachable at 30 but not 10 = CE-knee territory
/// (a deeper pool can promote it); gold reachable at 100 but not 30 =
/// pool-widening territory; gold absent at 100 = recall headroom no pool depth
/// can reach. If `gold_in_10 == gold_in_30 == gold_in_100` for (nearly) every
/// question, pool depth is structurally inert here and the adaptive pool cannot
/// add accuracy regardless of sweep size. Questions with no seeded DB under
/// `baselines_dir` are skipped (counted), not seeded — this is a read-only probe.
pub async fn run_longmemeval_headroom_probe_per_question(
    longmemeval_path: &Path,
    baselines_dir: &Path,
) -> Result<Vec<HeadroomRow>, WenlanError> {
    const LIMITS: [usize; 3] = [10, 30, 100];
    let mut samples = load_longmemeval(longmemeval_path)?;
    apply_lme_limit(&mut samples);
    let mut rows: Vec<HeadroomRow> = Vec::new();
    let mut missing = 0usize;

    for sample in &samples {
        let scope_dir =
            crate::eval::shared::scenario_db_dir(baselines_dir, "lme", &sample.question_id);
        if !scope_dir.join("origin_memory.db").exists() {
            missing += 1;
            continue;
        }
        let db = MemoryDB::new(&scope_dir, std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        let memories = extract_memories(sample);
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }

        let mut gold_in = [0usize; 3];
        let mut deep_ranks: Vec<i64> = Vec::new();
        for (i, &k) in LIMITS.iter().enumerate() {
            let results = db
                .search_memory(
                    &sample.question,
                    k,
                    None,
                    &ReadScope::Global,
                    None,
                    None,
                    None,
                    None,
                )
                .await?;
            let ranks: Vec<i64> = relevant_source_ids
                .iter()
                .map(|gid| {
                    results
                        .iter()
                        .position(|r| r.source_id == *gid)
                        .map(|p| p as i64)
                        .unwrap_or(-1)
                })
                .collect();
            gold_in[i] = gold_in_top_k(&ranks, k);
            if k == 100 {
                deep_ranks = ranks;
                deep_ranks.sort_by_key(|&r| if r < 0 { i64::MAX } else { r });
            }
        }

        rows.push(HeadroomRow {
            query_id: sample.question_id.clone(),
            category: sample.question_type.clone(),
            n_gold: relevant_source_ids.len(),
            gold_ranks_100: deep_ranks,
            gold_in_10: gold_in[0],
            gold_in_30: gold_in[1],
            gold_in_100: gold_in[2],
            hit_any_10: gold_in[0] > 0,
            hit_any_30: gold_in[1] > 0,
            hit_any_100: gold_in[2] > 0,
        });
    }
    if missing > 0 {
        eprintln!(
            "[headroom_per_question] WARN: {missing} question(s) had no seeded DB under {} (skipped)",
            baselines_dir.display()
        );
    }
    Ok(rows)
}

/// One row of the decompose-recall probe (Step 1 of the decompose probe
/// ladder): does multi-anchor fetch reach gold that single-query fetch cannot?
///
/// All arms fetch at the CE knee window (limit=30, `RERANK_POOL_FLOOR=30`
/// semantics) so the comparison is against the post-knee production shape:
/// - `gold_in_base30` — single original query (control arm)
/// - `gold_in_date30` — Zep-style question-date prefix `(date: ...) question`
/// - `gold_in_union`  — presence anywhere in the union of per-stream pools
///   (original + subqueries, 30 each): the ceiling for ANY downstream ranker
/// - `gold_in_rrf30`  — top-30 of the equal-weight RRF merge, mirroring the
///   production `search_memory_decomposed` merge: what the CURRENT merge
///   realizes of that ceiling
///
/// Pool-size control is by construction: union ≤ 4 streams × 30 = 120, and the
/// headroom probe already measured the single-query limit=100 ceiling on the
/// same questions — join the two JSONLs on `query_id` to separate anchor
/// diversity from pool size.
#[derive(Debug, serde::Serialize)]
pub struct DecomposeRecallRow {
    pub query_id: String,
    pub category: String,
    pub n_gold: usize,
    /// Subqueries from the fixture (excluding the original). 0 = atomic:
    /// the union/rrf arms degrade to the base arm by construction.
    pub n_subqueries: usize,
    pub gold_in_base30: usize,
    pub gold_in_date30: usize,
    pub gold_in_union: usize,
    pub union_size: usize,
    pub gold_in_rrf30: usize,
}

#[derive(serde::Deserialize)]
struct SubqFixtureRow {
    query_id: String,
    subqueries: Vec<String>,
}

/// Load the pre-generated subquery fixture (JSONL: `{query_id, subqueries}`).
/// Subqueries exclude the original question (the probe prepends it, mirroring
/// `retrieval::decompose::parse_subqueries`). Fails loud on malformed lines —
/// the fixture is generated input, not user data.
fn load_subquery_fixture(path: &Path) -> Result<HashMap<String, Vec<String>>, WenlanError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| WenlanError::Generic(format!("read subquery fixture: {e}")))?;
    let mut map = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let row: SubqFixtureRow = serde_json::from_str(line)
            .map_err(|e| WenlanError::Generic(format!("parse subquery fixture line: {e}")))?;
        map.insert(row.query_id, row.subqueries);
    }
    Ok(map)
}

/// Equal-weight RRF merge of per-stream ranked id lists, truncated to `k`.
/// Mirrors the `search_memory_decomposed` merge (score = Σ 1/(60+rank)), with
/// a stable id tiebreak the production code leaves to HashMap order.
fn rrf_merge_ids(streams: &[Vec<String>], k: usize) -> Vec<String> {
    let mut scores: HashMap<&str, f32> = HashMap::new();
    for ranked in streams {
        for (rank, id) in ranked.iter().enumerate() {
            *scores.entry(id.as_str()).or_default() += 1.0 / (60.0 + rank as f32);
        }
    }
    let mut merged: Vec<(&str, f32)> = scores.into_iter().collect();
    merged.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });
    merged.truncate(k);
    merged.into_iter().map(|(id, _)| id.to_string()).collect()
}

/// Decompose-recall probe over the base `search_memory` path (Step 1 of the
/// decompose probe ladder). Consumes a pre-generated subquery fixture
/// (agent-delegated decomposition — the primary lane from the 2026-05-30
/// decision) instead of calling an LLM, so the run is deterministic and
/// GPU-free. See [`DecomposeRecallRow`] for the four arms.
pub async fn run_longmemeval_decompose_recall_probe_from_db(
    db: &MemoryDB,
    path: &Path,
    subq_path: &Path,
) -> Result<Vec<DecomposeRecallRow>, WenlanError> {
    const K: usize = 30;
    let subq_map = load_subquery_fixture(subq_path)?;
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut rows: Vec<DecomposeRecallRow> = Vec::new();

    for sample in &samples {
        let memories = extract_memories(sample);
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }
        let gold_in = |ids: &[String]| {
            ids.iter()
                .filter(|id| relevant_source_ids.contains(*id))
                .count()
        };

        let subqueries = subq_map
            .get(&sample.question_id)
            .cloned()
            .unwrap_or_default();

        // Streams mirror parse_subqueries: original first, then subqueries.
        let mut streams: Vec<Vec<String>> = Vec::with_capacity(1 + subqueries.len());
        for q in std::iter::once(&sample.question).chain(subqueries.iter()) {
            let results = db
                .search_memory(q, K, None, &ReadScope::Global, None, None, None, None)
                .await?;
            streams.push(results.into_iter().map(|r| r.source_id).collect());
        }

        let date_query = format!("(date: {}) {}", sample.question_date, sample.question);
        let date_results = db
            .search_memory(
                &date_query,
                K,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await?;
        let date_ids: Vec<String> = date_results.into_iter().map(|r| r.source_id).collect();

        let union: HashSet<&str> = streams.iter().flatten().map(|s| s.as_str()).collect();
        let rrf_top = rrf_merge_ids(&streams, K);

        rows.push(DecomposeRecallRow {
            query_id: sample.question_id.clone(),
            category: sample.question_type.clone(),
            n_gold: relevant_source_ids.len(),
            n_subqueries: subqueries.len(),
            gold_in_base30: gold_in(&streams[0]),
            gold_in_date30: gold_in(&date_ids),
            gold_in_union: relevant_source_ids
                .iter()
                .filter(|gid| union.contains(gid.as_str()))
                .count(),
            union_size: union.len(),
            gold_in_rrf30: gold_in(&rrf_top),
        });
    }

    Ok(rows)
}

/// `(ndcg@10, recall@5, mrr)` for a retrieved/reranked result list against the
/// `has_answer` relevant set, binary relevance.
///
/// IDCG is normalized over ALL gold for the question (every `has_answer`
/// memory), NOT just the gold that landed in the retrieved set. Grading only the
/// retrieved ids (the legacy baseline convention) inflates IDCG as each gold is
/// surfaced, so in the multi-gold case a genuine recall improvement can register
/// as an NDCG *drop* — the metric becomes anti-correlated with the pool-depth
/// objective (boule adversarial review, 2026-06-19). All-gold IDCG makes
/// surfacing a buried gold into the top-k strictly raise NDCG. This intentionally
/// diverges from the legacy retrieved-set baselines (which measured a different,
/// pool-blind quantity on oracle/consolidated substrates).
fn score_retrieval(result_ids: &[&str], relevant_source_ids: &HashSet<String>) -> (f64, f64, f64) {
    let grades: HashMap<&str, u8> = relevant_source_ids
        .iter()
        .map(|s| (s.as_str(), 1u8))
        .collect();
    let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();
    (
        metrics::ndcg_at_k(result_ids, &grades, 10),
        metrics::recall_at_k(result_ids, &relevant_set, 5),
        metrics::mrr(result_ids, &relevant_set),
    )
}

/// Deep-S retrieval-quality probe for the CE rerank pool / model decision.
///
/// Opens each question's OWN pre-seeded deep-S scenario DB
/// (`<baselines_dir>/fullpipeline/lme/<question_id>/`) — faithful per-question
/// isolation, NOT one shared/consolidated store — and runs the PRODUCTION
/// retrieval path (`search_memory_cross_rerank_cued`) for that question, scoring
/// its top-10 against the `has_answer` relevant set. NO answer generation, NO
/// judge (phases P3/P4 skipped → minutes, deterministic). `reranker = None`
/// exercises the byte-identical non-CE baseline (CE-off); `Some` runs the CE
/// with pool / model from the live env (`RERANK_POOL_FLOOR`, `WENLAN_RERANKER_MODEL`).
/// `arm_label` is the
/// `feature` on each [`PerQueryRow`] so `analyze_paired.py` reads the output
/// directly. Caller sets the per-arm env + arm_label.
///
/// REUSE-ONLY: a per-question DB that is missing is a hard error (the probe must
/// not pay enrichment, and a half-seeded substrate must not silently score ~0).
/// Seed the deep-S substrate first; point the fixture at exactly the cached qids.
pub async fn run_longmemeval_rerank_pool_probe(
    baselines_dir: &Path,
    path: &Path,
    reranker: Option<std::sync::Arc<dyn crate::reranker::Reranker>>,
    arm_label: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    use crate::eval::paired::PerQueryRow;
    use crate::eval::shared::{eval_shared_embedder, scenario_db_dir};
    use std::time::Instant;

    let shared_embedder = eval_shared_embedder();
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let flag_state = if reranker.is_some() { "on" } else { "off" };
    let mut rows: Vec<PerQueryRow> = Vec::with_capacity(samples.len());

    for sample in &samples {
        let relevant_source_ids: HashSet<String> = extract_memories(sample)
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue; // no gradeable evidence turn
        }

        let scope_dir = scenario_db_dir(baselines_dir, "lme", &sample.question_id);
        if !scope_dir.join("origin_memory.db").exists() {
            return Err(WenlanError::Generic(format!(
                "rerank_pool_probe: no cached deep-S DB for question {} at {} — \
                 seed the substrate first (retrieval probe never seeds)",
                sample.question_id,
                scope_dir.display()
            )));
        }
        let db = MemoryDB::new_with_shared_embedder(
            &scope_dir,
            std::sync::Arc::new(crate::events::NoopEmitter),
            shared_embedder.clone(),
        )
        .await?;

        let t0 = Instant::now();
        let results = db
            .search_memory_cross_rerank_cued(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                None,
                reranker.clone(),
            )
            .await?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        let (ndcg10, recall5, mrr) = score_retrieval(&result_ids, &relevant_source_ids);

        // Source-expanded coverage (page provenance, eval-only). Positional
        // NDCG/recall score a `source="page"` result as a miss — its page id
        // never matches a memory-keyed gold id. coverage_recall over the
        // provenance-expanded set credits a page via the memory source ids it
        // was distilled from, so the page-channel-ON arm is measurable.
        // `cov_expanded - cov_blind` is the page contribution. Mirror of the
        // LoCoMo path (locomo.rs).
        let relevant_set: std::collections::HashSet<&str> =
            relevant_source_ids.iter().map(|s| s.as_str()).collect();
        let page_src_owned: Vec<(String, Vec<String>)> = {
            let mut v = Vec::new();
            for r in &results {
                if r.source == "page" {
                    let srcs: Vec<String> = db
                        .get_page_sources(&r.source_id)
                        .await
                        .unwrap_or_default()
                        .into_iter()
                        .map(|ps| ps.memory_source_id)
                        .collect();
                    v.push((r.source_id.clone(), srcs));
                }
            }
            v
        };
        let page_sources_map: std::collections::HashMap<&str, Vec<&str>> = page_src_owned
            .iter()
            .map(|(pid, srcs)| (pid.as_str(), srcs.iter().map(|s| s.as_str()).collect()))
            .collect();
        let units: Vec<(&str, &str)> = results
            .iter()
            .map(|r| (r.source.as_str(), r.source_id.as_str()))
            .collect();
        let cov_blind = crate::eval::metrics::coverage_recall(
            &crate::eval::metrics::build_coverage_set(&units, &std::collections::HashMap::new()),
            &relevant_set,
        );
        let cov_expanded = crate::eval::metrics::coverage_recall(
            &crate::eval::metrics::build_coverage_set(&units, &page_sources_map),
            &relevant_set,
        );

        rows.push(PerQueryRow {
            feature: arm_label.to_string(),
            bench: "lme".to_string(),
            flag_state: flag_state.to_string(),
            query_id: sample.question_id.clone(),
            category: sample.question_type.clone(),
            ndcg10,
            recall5,
            mrr,
            latency_ms,
            graph_skipped: None,
            temporal_touched: None,
            channel_touched: None,
            cov_blind: Some(cov_blind),
            cov_expanded: Some(cov_expanded),
        });
    }

    Ok(rows)
}

/// CE-rank a candidate pool against `query` and return source_ids in CE-score
/// order (descending, stable id tiebreak). Fail-loud on reranker errors — this
/// is probe code, not the production degrade path.
async fn ce_rank_ids(
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
    query: &str,
    pool: Vec<(String, String)>,
) -> Result<Vec<String>, WenlanError> {
    if pool.len() <= 1 {
        return Ok(pool.into_iter().map(|(id, _)| id).collect());
    }
    let q = query.to_string();
    let mut scored = tokio::task::spawn_blocking(move || reranker.rerank(&q, &pool))
        .await
        .map_err(|e| WenlanError::Generic(format!("rerank join: {e}")))??;
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    Ok(scored.into_iter().map(|(id, _)| id).collect())
}

/// Step 2 of the decompose probe ladder: CE conversion at MATCHED budget.
///
/// Both arms feed the cross-encoder exactly 30 candidates (the measured knee,
/// PR #244) and score the CE's top-10 — so any delta is pool COMPOSITION
/// (anchor diversity), never pool size:
/// - OFF arm: base `search_memory(question, 30)` pool (current CE-path shape
///   under `RERANK_POOL_FLOOR=30`)
/// - ON arm: equal-weight RRF merge of original+subquery streams (30 each,
///   per-stream dedup by source_id), truncated to 30 — the
///   `search_memory_decomposed` merge feeding the CE instead of bypassing it
///
/// Atomic questions (no subqueries in the fixture) have byte-identical pools;
/// the CE is deterministic, so the OFF metrics are emitted for both arms
/// without a second CE pass (mirrors a production gate where decompose no-ops
/// on atomic queries). `latency_ms` covers the CE call only — per-stream
/// searches are shared across arms and excluded.
///
/// Emits both arms as [`crate::eval::paired::PerQueryRow`]s
/// (feature `decompose_ce`) so `analyze_paired.py` reads the output directly.
pub async fn run_longmemeval_decompose_ce_probe_from_db(
    db: &MemoryDB,
    path: &Path,
    subq_path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    use crate::eval::paired::PerQueryRow;
    use std::time::Instant;
    const K: usize = 30;

    let subq_map = load_subquery_fixture(subq_path)?;
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut rows: Vec<PerQueryRow> = Vec::new();

    for sample in &samples {
        let memories = extract_memories(sample);
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }
        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        let subqueries = subq_map
            .get(&sample.question_id)
            .cloned()
            .unwrap_or_default();

        // Per-stream searches: ordered unique source_ids + first-seen content
        // (the CE candidate text, trimmed to 512 chars like the production path).
        let mut streams: Vec<Vec<String>> = Vec::with_capacity(1 + subqueries.len());
        let mut content_by_id: HashMap<String, String> = HashMap::new();
        for q in std::iter::once(&sample.question).chain(subqueries.iter()) {
            let results = db
                .search_memory(q, K, None, &ReadScope::Global, None, None, None, None)
                .await?;
            let mut ids = Vec::with_capacity(results.len());
            let mut seen: HashSet<&str> = HashSet::new();
            for r in &results {
                if seen.insert(r.source_id.as_str()) {
                    ids.push(r.source_id.clone());
                }
            }
            for r in &results {
                content_by_id
                    .entry(r.source_id.clone())
                    .or_insert_with(|| r.content.chars().take(512).collect());
            }
            streams.push(ids);
        }

        let pool_of = |ids: &[String]| -> Vec<(String, String)> {
            ids.iter()
                .map(|id| {
                    (
                        id.clone(),
                        content_by_id.get(id).cloned().unwrap_or_default(),
                    )
                })
                .collect()
        };
        let on_ids = rrf_merge_ids(&streams, K);
        let off_ids = &streams[0];

        let mut emit = |state: &str, ranked: &[String], latency_ms: f64| {
            let top10: Vec<&str> = ranked.iter().take(10).map(|s| s.as_str()).collect();
            let grades: HashMap<&str, u8> = top10
                .iter()
                .map(|id| (*id, u8::from(relevant_set.contains(*id))))
                .collect();
            rows.push(PerQueryRow {
                feature: "decompose_ce".to_string(),
                bench: "lme".to_string(),
                flag_state: state.to_string(),
                query_id: sample.question_id.clone(),
                category: sample.question_type.clone(),
                ndcg10: metrics::ndcg_at_k(&top10, &grades, 10),
                recall5: metrics::recall_at_k(&top10, &relevant_set, 5),
                mrr: metrics::mrr(&top10, &relevant_set),
                latency_ms,
                graph_skipped: None,
                temporal_touched: None,
                channel_touched: None,
                cov_blind: None,
                cov_expanded: None,
            });
        };

        let t0 = Instant::now();
        let off_ranked = ce_rank_ids(reranker.clone(), &sample.question, pool_of(off_ids)).await?;
        let off_latency = t0.elapsed().as_secs_f64() * 1000.0;
        emit("off", &off_ranked, off_latency);

        if subqueries.is_empty() {
            // Atomic: pools identical, CE deterministic — reuse the OFF ranking.
            emit("on", &off_ranked, off_latency);
        } else {
            let t1 = Instant::now();
            let on_ranked =
                ce_rank_ids(reranker.clone(), &sample.question, pool_of(&on_ids)).await?;
            emit("on", &on_ranked, t1.elapsed().as_secs_f64() * 1000.0);
        }
    }

    Ok(rows)
}

/// Cross-encoder variant of [`run_longmemeval_eval_from_db_collect`]. Identical
/// query set + relevance judgments + scoring, but retrieval goes through
/// `search_memory_cross_rerank` (CE rescoring over the widened pool) instead of
/// the base `search_memory` path. Pairs OFF (base collector) against ON (this
/// one) on the same snapshot DB to measure whether the cross-encoder helps.
///
/// Pins the ungated rerank-pool knobs (multiplier=1, floor=10) when unset, same
/// as `run_longmemeval_eval_cross_rerank_from_db`, so the CE pool depth is
/// reproducible. `--test-threads=1` makes the process-global set_var safe.
pub async fn run_longmemeval_eval_cross_rerank_from_db_collect(
    db: &MemoryDB,
    path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    let needs_base = feature == "rerank" || feature.starts_with("rerank_model");
    run_longmemeval_eval_from_db_collect_core(
        db,
        path,
        Some(reranker),
        feature,
        flag_state,
        needs_base,
    )
    .await
}

// ---------------------------------------------------------------------------
// T4a eval runner — temporal filter with haystack_dates event_date seeding
// ---------------------------------------------------------------------------

/// Parse LongMemEval date string "YYYY/MM/DD (DDD) HH:MM" to a UTC DateTime.
///
/// Returns `None` on malformed input so the runner degrades gracefully.
fn parse_lme_date(s: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    // Format: "2023/04/10 (Mon) 23:07"
    // We strip the weekday parenthetical and parse the rest.
    let s = s.trim();
    // Find the part before " (" and after ") "
    let without_weekday = if let (Some(a), Some(b)) = (s.find(" ("), s.rfind(") ")) {
        format!("{} {}", &s[..a], &s[b + 2..])
    } else {
        s.to_string()
    };
    // without_weekday is now "YYYY/MM/DD HH:MM"
    chrono::NaiveDateTime::parse_from_str(&without_weekday, "%Y/%m/%d %H:%M")
        .ok()
        .map(|dt| chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(dt, chrono::Utc))
}

fn t4a_event_date_seeds(
    memories: &[LongMemEvalMemory],
    haystack_dates: &[String],
) -> Vec<EvalTemporalSeed> {
    memories
        .iter()
        .filter_map(|mem| {
            let date_str = haystack_dates.get(mem.session_idx)?;
            let event_date = parse_lme_date(date_str)?.timestamp();
            Some(EvalTemporalSeed {
                source_id: memory_source_id(&mem.question_id, mem.session_idx, mem.turn_idx),
                event_date,
            })
        })
        .collect()
}

/// LongMemEval temporal eval runner (T4a).
///
/// Clones `run_longmemeval_eval` but adds two T4a-specific steps:
///
/// 1. **Event-date seeding**: after `upsert_documents`, each seeded memory is
///    stamped with the Unix timestamp of its session's `haystack_dates` entry
///    via a direct `UPDATE memories SET event_date=? WHERE source_id=?` on the
///    connection. Without this, every `event_date` is NULL and the temporal
///    hard-filter is a guaranteed no-op (all memories pass the `OR IS NULL`
///    branch).
///
/// 2. **Temporal search**: calls `search_memory_temporal` with
///    `now = parse(sample.question_date)` — NOT `Utc::now()` — so the cue
///    window aligns with the fixture clock.
///
/// # LoCoMo note
/// LoCoMo has no reliable absolute dates (only session_num), so a LoCoMo
/// temporal runner is not provided. Temporal eval on LoCoMo would require
/// synthetic date assignment, which risks fabricating the measured signal.
///
/// # Citation discipline
/// First runs are single-run scaffolds (`is_single_run = true`). Do NOT cite
/// accuracy numbers without N>=3 + stddev per AGENTS.md Eval Citation Discipline.
#[allow(dead_code)] // L7 manual, GPU-requiring eval; not wired to CI
pub async fn run_longmemeval_eval_temporal(path: &Path) -> Result<LongMemEvalReport, WenlanError> {
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut per_case: Vec<crate::eval::report::CaseResult> = Vec::new();
    let mut total_memories: usize = 0;

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| WenlanError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        // Seed all extracted memories (same as run_longmemeval_eval)
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
                    space: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                }
            })
            .collect();
        total_memories += docs.len();
        db.upsert_documents(docs).await?;

        // T4a: stamp event_date from haystack_dates (parallel to sessions/memories).
        // Without this, every event_date is NULL and the temporal filter is a no-op.
        // haystack_dates is parallel to haystack_session_ids
        // (one date per session, indexed by session_idx).
        let event_date_seeds = t4a_event_date_seeds(&memories, &sample.haystack_dates);
        db.apply_eval_temporal_seeds(&event_date_seeds).await;

        // Build relevance judgments
        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();

        if relevant_source_ids.is_empty() {
            continue;
        }

        // T4a: parse question_date as `now` so the cue window aligns with fixture clock.
        // Fall back to Utc::now() only if the date is unparseable (degraded mode).
        let now = parse_lme_date(&sample.question_date).unwrap_or_else(chrono::Utc::now);

        // T4a: use search_memory_temporal with WENLAN_ENABLE_TEMPORAL_FILTER=1
        let results = db
            .search_memory_temporal(&sample.question, 10, None, &ReadScope::Global, None, now)
            .await?;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| (*id, u8::from(relevant_source_ids.contains(*id))))
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
        per_case.push(build_lme_case_result(
            &sample.question,
            &sample.question_type,
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    let per_category = aggregate_by_category(&all_scores);
    let mut report = LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories,
        per_category,
        baseline: None,
        env: None,
        per_case,
        coverage: None,
    };
    let mut env_stamp = build_lme_env(
        "temporal",
        path,
        "search_memory_temporal",
        "none",
        "none",
        None,
    );
    env_stamp.flags.push(format!(
        "graph_memory_stream={}",
        if crate::db::graph_memory_stream_enabled() {
            "on"
        } else {
            "off"
        }
    ));
    report.env = Some(env_stamp);
    Ok(report)
}

/// Per-query variant of [`run_longmemeval_eval_temporal`] (T4a). SELF-SEEDS a
/// fresh ephemeral DB per question (stamps `event_date`), so unlike the cached-DB
/// collectors this re-runs the write path — runs are reproducible only insofar as
/// the seeding is deterministic. Emits one [`PerQueryRow`] per question with a
/// wall-clock latency for the `search_memory_temporal` call.
pub async fn run_longmemeval_eval_temporal_collect(
    path: &Path,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    use crate::eval::paired::PerQueryRow;
    use std::time::Instant;

    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut rows: Vec<PerQueryRow> = Vec::new();

    for sample in &samples {
        let memories = extract_memories(sample);

        let tmp = tempfile::tempdir().map_err(|e| WenlanError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;

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
                    space: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                }
            })
            .collect();
        db.upsert_documents(docs).await?;

        let event_date_seeds = t4a_event_date_seeds(&memories, &sample.haystack_dates);
        db.apply_eval_temporal_seeds(&event_date_seeds).await;

        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }

        let now = parse_lme_date(&sample.question_date).unwrap_or_else(chrono::Utc::now);

        // T4a per-touched: did a high-confidence temporal cue actually fire for
        // this query? Mirrors the gate in search_memory_temporal (db.rs:8438):
        // only a High-confidence cue engages the hard temporal filter; None/Low
        // degrade to plain search and leave this query a no-op. Lets the analyzer
        // isolate the ~3.4% of queries the feature truly touches.
        let cue_fired = crate::temporal_query::extract_cue(&sample.question, now)
            .map(|c| c.confidence == crate::temporal_query::CueConfidence::High)
            .unwrap_or(false);

        let t0 = Instant::now();
        let results = db
            .search_memory_temporal(&sample.question, 10, None, &ReadScope::Global, None, now)
            .await?;
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| (*id, u8::from(relevant_source_ids.contains(*id))))
            .collect();
        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        rows.push(PerQueryRow {
            feature: feature.to_string(),
            bench: "lme".to_string(),
            flag_state: flag_state.to_string(),
            query_id: sample.question_id.clone(),
            category: sample.question_type.clone(),
            ndcg10: metrics::ndcg_at_k(&result_ids, &grades, 10),
            recall5: metrics::recall_at_k(&result_ids, &relevant_set, 5),
            mrr: metrics::mrr(&result_ids, &relevant_set),
            latency_ms,
            graph_skipped: None,
            temporal_touched: Some(cue_fired),
            channel_touched: None,
            cov_blind: None,
            cov_expanded: None,
        });
    }

    Ok(rows)
}

/// Paired graph-stream collector for LongMemEval (efficient, cached). Mirrors
/// [`crate::eval::locomo::run_locomo_eval_graph_stream_collect`]: one persistent
/// populated DB per question under `<GRAPH_POP_DIR>/lme/<question_id>/`, the fine
/// entity sweep paid once, base `search_memory` (no expansion/rerank) so the query
/// loop is LLM-free. Only `WENLAN_GRAPH_MEMORY_STREAM` differs per arm.
pub async fn run_longmemeval_eval_graph_stream_collect(
    path: &Path,
    llm: std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
    prompts: &crate::prompts::PromptRegistry,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    use crate::eval::paired::PerQueryRow;

    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut rows: Vec<PerQueryRow> = Vec::new();

    for sample in &samples {
        let memories = extract_memories(sample);
        let dir = crate::eval::locomo::graph_pop_dir("lme", &sample.question_id);
        let fresh = !dir.join("origin_memory.db").exists();
        std::fs::create_dir_all(&dir)
            .map_err(|e| WenlanError::Generic(format!("graph_pop dir: {e}")))?;
        let db = MemoryDB::new(&dir, std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        if fresh {
            let docs: Vec<RawDocument> = memories
                .iter()
                .map(|mem| {
                    let memory_type = match sample.question_type.as_str() {
                        "single-session-preference" => "preference",
                        _ => "fact",
                    };
                    RawDocument {
                        content: mem.content.clone(),
                        source_id: memory_source_id(
                            &mem.question_id,
                            mem.session_idx,
                            mem.turn_idx,
                        ),
                        source: "memory".to_string(),
                        title: format!("{} session {}", mem.role, mem.session_idx),
                        memory_type: Some(memory_type.to_string()),
                        space: Some("conversation".to_string()),
                        last_modified: chrono::Utc::now().timestamp(),
                        ..Default::default()
                    }
                })
                .collect();
            db.upsert_documents(docs).await?;
            let linked =
                crate::eval::locomo::populate_memory_entities_sweep(&db, &llm, prompts).await;
            println!(
                "[graph_stream] populated lme/{}: {linked}/{} memories linked",
                sample.question_id,
                memories.len()
            );
        }

        let relevant_source_ids: HashSet<String> = memories
            .iter()
            .filter(|m| m.has_answer)
            .map(|m| memory_source_id(&m.question_id, m.session_idx, m.turn_idx))
            .collect();
        if relevant_source_ids.is_empty() {
            continue;
        }

        let results = db
            .search_memory(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await?;
        // Per-sample DB handle: feature contains graph_stream, probe fires.
        let channel_touched =
            crate::eval::shared::base_channel_touched(&db, feature, &sample.question).await?;
        let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
        let grades: HashMap<&str, u8> = result_ids
            .iter()
            .map(|id| (*id, u8::from(relevant_source_ids.contains(*id))))
            .collect();
        let relevant_set: HashSet<&str> = relevant_source_ids.iter().map(|s| s.as_str()).collect();

        rows.push(PerQueryRow {
            feature: feature.to_string(),
            bench: "lme".to_string(),
            flag_state: flag_state.to_string(),
            query_id: sample.question_id.clone(),
            category: sample.question_type.clone(),
            ndcg10: metrics::ndcg_at_k(&result_ids, &grades, 10),
            recall5: metrics::recall_at_k(&result_ids, &relevant_set, 5),
            mrr: metrics::mrr(&result_ids, &relevant_set),
            latency_ms: 0.0,
            graph_skipped: None,
            temporal_touched: None,
            channel_touched,
            cov_blind: None,
            cov_expanded: None,
        });
    }

    Ok(rows)
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
            space: Some("conversation".to_string()),
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
) -> Result<LongMemEvalReport, WenlanError> {
    let mut samples = load_longmemeval(path)?;
    apply_lme_limit(&mut samples);
    let mut all_scores: Vec<(String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut per_case: Vec<crate::eval::report::CaseResult> = Vec::new();
    let mut total_memories_inserted: usize = 0;

    let gate = match mode {
        LongMemEvalGateMode::Gated => Some(QualityGate::new(GateConfig::default())),
        _ => None,
    };

    for sample in &samples {
        let memories = extract_memories(sample);

        // Create ephemeral DB for this question
        let tmp = tempfile::tempdir().map_err(|e| WenlanError::Generic(format!("tempdir: {e}")))?;
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
                    space: Some("conversation".to_string()),
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
            .search_memory(
                &sample.question,
                10,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
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
        per_case.push(build_lme_case_result(
            &sample.question,
            &sample.question_type,
            ndcg_5,
            ndcg_10,
            mrr_val,
            recall_5,
            hr_1,
        ));
    }

    // Aggregate
    let per_category = aggregate_by_category(&all_scores);

    let mut report = LongMemEvalReport {
        aggregate_ndcg_at_10: avg_field(&all_scores, |s| s.2),
        aggregate_mrr: avg_field(&all_scores, |s| s.3),
        aggregate_recall_at_5: avg_field(&all_scores, |s| s.4),
        aggregate_hit_rate_at_1: avg_field(&all_scores, |s| s.5),
        total_questions: all_scores.len(),
        total_memories: total_memories_inserted,
        per_category,
        baseline: None,
        env: None,
        per_case,
        coverage: None,
    };
    let mut env_stamp = build_lme_env("gated", path, "search_memory", "none", "none", None);
    env_stamp.flags.push(format!(
        "graph_memory_stream={}",
        if crate::db::graph_memory_stream_enabled() {
            "on"
        } else {
            "off"
        }
    ));
    report.env = Some(env_stamp);
    Ok(report)
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
// Backward-compat re-exports (prompt functions now live in judge.rs)
// ---------------------------------------------------------------------------

pub use crate::eval::anthropic::{
    call_anthropic_api, download_batch_results, estimate_batch_cost, poll_batch, submit_batch,
};
pub use crate::eval::judge::{
    lme_anscheck_prompt as get_anscheck_prompt, lme_answer_prompt as build_answer_prompt,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_retrieval_ranks_answer_memory() {
        use std::collections::HashSet;
        let relevant: HashSet<String> = ["lme_q1_0_t0".to_string()].into_iter().collect();
        // answer at rank 1 → perfect on all three metrics
        assert_eq!(
            score_retrieval(&["lme_q1_0_t0", "d1", "d2"], &relevant),
            (1.0, 1.0, 1.0)
        );
        // answer at rank 3 → ndcg@10 = 1/log2(4) = 0.5, recall@5 = 1.0, mrr = 1/3
        let (ndcg, recall, mrr) = score_retrieval(&["d1", "d2", "lme_q1_0_t0"], &relevant);
        assert!((ndcg - 0.5).abs() < 1e-9, "ndcg={ndcg}");
        assert_eq!(recall, 1.0);
        assert!((mrr - 1.0 / 3.0).abs() < 1e-9, "mrr={mrr}");
        // answer not in the returned set → all zero
        assert_eq!(score_retrieval(&["d1", "d2"], &relevant), (0.0, 0.0, 0.0));
    }

    #[test]
    fn score_retrieval_rewards_surfacing_buried_gold() {
        use std::collections::HashSet;
        // Two gold memories for this question. A deeper pool surfacing the second
        // gold into top-10 MUST raise NDCG. Under the old retrieved-set IDCG this
        // DROPPED (1.0 → 0.92); all-gold IDCG fixes it (0.61 → 0.92). This test
        // is the regression guard for the boule-caught normalization bug.
        let relevant: HashSet<String> = ["g1".to_string(), "g2".to_string()].into_iter().collect();
        let (ndcg_one, _, _) = score_retrieval(&["g1", "d1", "d2"], &relevant); // only g1
        let (ndcg_both, _, _) = score_retrieval(&["g1", "d1", "g2"], &relevant); // g2 surfaced @3
        assert!(
            ndcg_both > ndcg_one,
            "surfacing a buried gold must RAISE ndcg, not drop it: {ndcg_one} -> {ndcg_both}"
        );
    }

    #[test]
    fn gold_in_top_k_bands() {
        // ranks: two in top-10, one at 11-30, one at 31-100, one absent
        let ranks = vec![0, 7, 22, 64, -1];
        assert_eq!(gold_in_top_k(&ranks, 10), 2);
        assert_eq!(gold_in_top_k(&ranks, 30), 3);
        assert_eq!(gold_in_top_k(&ranks, 100), 4);
        // absent-only never counts
        assert_eq!(gold_in_top_k(&[-1, -1], 100), 0);
        // boundary: rank 9 in, rank 10 out at k=10
        assert_eq!(gold_in_top_k(&[9, 10], 10), 1);
    }

    #[test]
    fn rrf_merge_prefers_multi_stream_ids() {
        // id "b" appears in both streams (ranks 1 and 0) and must beat "a"
        // (rank 0 in one stream only): 1/61 + 1/60 > 1/60.
        let streams = vec![
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            vec!["b".to_string(), "d".to_string()],
        ];
        let merged = rrf_merge_ids(&streams, 3);
        assert_eq!(merged[0], "b");
        assert_eq!(merged[1], "a");
        assert_eq!(merged.len(), 3);
        // truncation respected
        assert_eq!(rrf_merge_ids(&streams, 1), vec!["b".to_string()]);
        // single stream degrades to identity order
        let single = vec![vec!["x".to_string(), "y".to_string()]];
        assert_eq!(
            rrf_merge_ids(&single, 30),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn subquery_fixture_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("subq.jsonl");
        std::fs::write(
            &p,
            "{\"query_id\":\"q1\",\"subqueries\":[\"a\",\"b\"]}\n{\"query_id\":\"q2\",\"subqueries\":[]}\n",
        )
        .unwrap();
        let map = load_subquery_fixture(&p).unwrap();
        assert_eq!(map["q1"], vec!["a", "b"]);
        assert!(map["q2"].is_empty());
        // malformed line fails loud, not silently skipped
        std::fs::write(&p, "{\"query_id\":\"q1\"\n").unwrap();
        assert!(load_subquery_fixture(&p).is_err());
    }

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
            env: None,
            per_case: vec![],
            coverage: None,
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
            env: None,
            per_case: vec![],
            coverage: None,
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
                coverage: None,
            }),
            env: None,
            per_case: vec![],
            coverage: None,
        };

        let text = report.to_terminal();
        assert!(text.contains("LongMemEval Benchmark"));
        assert!(text.contains("Baseline comparison:"));
        assert!(text.contains("->"));
        assert!(text.contains("single-session-user"));
    }

    /// Build a vec of `n` minimal `LongMemEvalSample`s for env-limit tests.
    fn mock_samples(n: usize) -> Vec<LongMemEvalSample> {
        (0..n)
            .map(|i| {
                let json = format!(
                    r#"{{
                        "question_id": "mock-{i}",
                        "question_type": "single-session-user",
                        "question": "q?",
                        "answer": "a",
                        "question_date": "2023/04/10 (Mon) 23:07",
                        "haystack_dates": [],
                        "haystack_session_ids": [],
                        "haystack_sessions": [],
                        "answer_session_ids": []
                    }}"#
                );
                serde_json::from_str::<LongMemEvalSample>(&json).unwrap()
            })
            .collect()
    }

    #[test]
    fn eval_lme_limit_truncates_when_set() {
        // Unique env var name so the test doesn't race the real EVAL_LME_LIMIT.
        let var = "EVAL_LME_LIMIT_TEST_TRUNCATE";
        let mut samples = mock_samples(8);
        std::env::set_var(var, "3");
        apply_limit_from_env(&mut samples, var, "longmemeval", "questions");
        std::env::remove_var(var);
        assert_eq!(samples.len(), 3, "limit=3 should truncate 8 down to 3");
        assert_eq!(samples[0].question_id, "mock-0");
        assert_eq!(samples[2].question_id, "mock-2");
    }

    #[test]
    fn eval_lme_limit_no_op_when_unset() {
        let var = "EVAL_LME_LIMIT_TEST_NOOP";
        std::env::remove_var(var);
        let mut samples = mock_samples(4);
        apply_limit_from_env(&mut samples, var, "longmemeval", "questions");
        assert_eq!(samples.len(), 4, "unset env var must leave samples intact");
    }

    #[test]
    fn lme_event_date_map_parses_haystack_dates_to_source_ids() {
        let sample: LongMemEvalSample = serde_json::from_value(serde_json::json!({
            "question_id": "q1",
            "question_type": "temporal-reasoning",
            "question": "when?",
            "answer": "x",
            "question_date": "2023/04/10 (Mon) 23:07",
            "haystack_dates": ["2023/04/10 (Mon) 17:50"],
            "haystack_session_ids": ["s0"],
            "haystack_sessions": [[{ "role": "user", "content": "hi", "has_answer": false }]],
            "answer_session_ids": []
        }))
        .unwrap();
        let map = event_date_map(&[sample]);
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 4, 10)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        // source_id mirrors the seed builder via memory_source_id: lme_<qid>_<sidx>_t<tidx>.
        assert_eq!(map.get("lme_q1_0_t0"), Some(&expected));
    }

    #[test]
    fn t4a_seeds_preserve_full_time_input_order_duplicates_and_skip_bad_dates() {
        let memory = |question_id: &str, session_idx: usize, turn_idx: usize| LongMemEvalMemory {
            content: "body".to_string(),
            role: "user".to_string(),
            session_id: format!("session-{session_idx}"),
            session_idx,
            turn_idx,
            has_answer: false,
            question_id: question_id.to_string(),
        };
        let memories = vec![
            memory("later-first", 2, 5),
            memory("earlier-second", 0, 1),
            memory("later-first", 2, 5),
            memory("malformed", 1, 2),
            memory("missing", 4, 3),
        ];
        let dates = vec![
            "2023/04/10 (Mon) 09:15".to_string(),
            "not a date".to_string(),
            "2023/04/12 (Wed) 17:50".to_string(),
        ];
        let later = parse_lme_date(&dates[2]).unwrap().timestamp();
        let earlier = parse_lme_date(&dates[0]).unwrap().timestamp();

        assert_eq!(
            t4a_event_date_seeds(&memories, &dates),
            vec![
                EvalTemporalSeed {
                    source_id: "lme_later-first_2_t5".to_string(),
                    event_date: later,
                },
                EvalTemporalSeed {
                    source_id: "lme_earlier-second_0_t1".to_string(),
                    event_date: earlier,
                },
                EvalTemporalSeed {
                    source_id: "lme_later-first_2_t5".to_string(),
                    event_date: later,
                },
            ]
        );
        assert_ne!(
            later,
            parse_lme_date("2023/04/12 (Wed) 00:00")
                .unwrap()
                .timestamp(),
            "T4a seeds must retain the parsed 17:50 time rather than truncate to midnight"
        );
    }
}
