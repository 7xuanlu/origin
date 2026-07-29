// SPDX-License-Identifier: Apache-2.0
//! LoCoMo benchmark adapter — converts locomo10.json into Wenlan eval cases.
//!
//! LoCoMo (Long Conversational Memory) contains 10 conversations with
//! pre-extracted observations and ~1,986 QA pairs in 5 categories:
//!   1 = multi-hop, 2 = temporal, 3 = open-domain, 4 = single-hop, 5 = adversarial
//!
//! Dataset: <https://github.com/snap-research/locomo>

use crate::db::MemoryDB;
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
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load LoCoMo dataset from a local JSON file.
pub fn load_locomo(path: &Path) -> Result<Vec<LocomoSample>, WenlanError> {
    let data = std::fs::read_to_string(path)
        .map_err(|e| WenlanError::Generic(format!("Failed to read LoCoMo file: {e}")))?;
    let samples: Vec<LocomoSample> = serde_json::from_str(&data)
        .map_err(|e| WenlanError::Generic(format!("Failed to parse LoCoMo JSON: {e}")))?;
    Ok(samples)
}

/// Truncate the loaded LoCoMo samples in place if `EVAL_LOCOMO_LIMIT` is set
/// to a positive integer. Used by every `run_locomo_eval*` variant so a
/// developer can run a small pre-flight subset (~30min) before committing
/// to a full multi-hour run.
fn apply_locomo_limit(samples: &mut Vec<LocomoSample>) {
    apply_limit_from_env(samples, "EVAL_LOCOMO_LIMIT", "locomo", "conversations");
}

/// Shared helper for `apply_locomo_limit`. Parameterized on the env var name so
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

                memories.push(LocomoMemory {
                    content,
                    speaker: speaker.clone(),
                    session_num,
                    dia_id,
                    sample_id: sample.sample_id.clone(),
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

/// Parse a LoCoMo `session_N_date_time` value into a day-granular unix timestamp
/// (midnight UTC of that date). Input looks like `"1:56 pm on 8 May, 2023"`; only
/// the date part after `" on "` is used — time-of-day is dropped because temporal
/// matching is day-level.
fn parse_locomo_date(s: &str) -> Option<i64> {
    let date_part = s.split(" on ").nth(1).unwrap_or(s).trim();
    let nd = chrono::NaiveDate::parse_from_str(date_part, "%d %B, %Y").ok()?;
    Some(nd.and_hms_opt(0, 0, 0)?.and_utc().timestamp())
}

/// Build a `{source_id -> event_date(unix seconds)}` map from LoCoMo session
/// metadata, for eval-seed `event_date` injection (T11/T20 temporal).
///
/// LoCoMo observation TEXT is date-stripped, so classify-from-text recovers no
/// `event_date`; the date lives only in `conversation["session_N_date_time"]`.
/// `source_id` mirrors the seed builder exactly: `locomo_<sample_id>_obs_<i>`
/// where `i` is the enumerate index over [`extract_observations`].
pub fn event_date_map(samples: &[LocomoSample]) -> HashMap<String, i64> {
    let mut map = HashMap::new();
    for sample in samples {
        // session_num -> ts, parsed from the `session_N_date_time` conversation keys.
        let mut session_ts: HashMap<usize, i64> = HashMap::new();
        if let Some(conv) = sample.conversation.as_object() {
            for (k, v) in conv {
                let parsed = k
                    .strip_prefix("session_")
                    .and_then(|r| r.strip_suffix("_date_time"))
                    .and_then(|r| r.parse::<usize>().ok());
                if let (Some(n), Some(s)) = (parsed, v.as_str()) {
                    if let Some(ts) = parse_locomo_date(s) {
                        session_ts.insert(n, ts);
                    }
                }
            }
        }
        for (i, mem) in extract_observations(sample).iter().enumerate() {
            if let Some(&ts) = session_ts.get(&mem.session_num) {
                map.insert(format!("locomo_{}_obs_{}", sample.sample_id, i), ts);
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// Conversion to eval cases
// ---------------------------------------------------------------------------

/// Convert a LoCoMo sample into eval cases for Wenlan's runner.
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
            });
        }

        cases.push(EvalCase {
            query: qa.question.clone(),
            space: Some("conversation".to_string()),
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

/// Build a `CaseResult` for one LoCoMo QA pair. Metrics not computed by the
/// LoCoMo runner (map_at_10, precision_at_3, negatives) are zero-filled.
#[allow(clippy::too_many_arguments)]
fn build_locomo_case_result(
    question: &str,
    category: u8,
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
        category: Some(category_name(category).to_string()),
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<crate::eval::report::CoverageRecall>,
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
            coverage: self.coverage.clone(),
        };
        let json = serde_json::to_string_pretty(&baseline).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load baseline from a previous run for comparison.
    pub fn load_baseline(path: &Path) -> Option<LocomoBaseline> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Encode retrieval variant + provider + fixture-hash into baseline filename.
    /// Falls back to base + ".json" if env is missing (back-compat).
    pub fn baseline_filename(&self, base: &str) -> String {
        crate::eval::report::encode_baseline_filename(self.env.as_ref(), base)
    }

    /// Project this LocomoReport onto the flat `EvalReport` shape so the
    /// P0b layered baseline path (`save_full_report`) can consume it.
    ///
    /// **Mapping notes:**
    /// - LoCoMo retrieval metrics map onto `ndcg_at_10` / `mrr` / `recall_at_5`
    ///   / `hit_rate_at_1` directly.
    /// - Metrics not surfaced by the LoCoMo runner (ndcg_at_5, map_at_5/10,
    ///   recall_at_1/3, precision_at_3/5, negative analysis) are zero-filled.
    /// - `search_mode` is taken from the env stamp when available.
    /// - `per_case` is left empty; LoCoMo's per-conversation breakdown lives
    ///   on the strongly-typed report. Consumers wanting per-conversation
    ///   data should keep using `LocomoReport` directly.
    pub fn to_eval_report(&self) -> crate::eval::report::EvalReport {
        let search_mode = self
            .env
            .as_ref()
            .map(|e| e.retrieval_method.clone())
            .unwrap_or_else(|| "locomo".to_string());
        crate::eval::report::EvalReport {
            fixture_count: self.total_questions,
            file_count: self.conversations.len(),
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
            total_scenarios: self.conversations.len(),
            skipped_scenarios: Vec::new(),
            enrichment_failures: 0,
            truncated_reason: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ReportEnv builder
// ---------------------------------------------------------------------------

/// Build a `ReportEnv` for a LoCoMo runner variant.
///
/// Fills both the legacy 9 fields (needed by `encode_baseline_filename`) and
/// the new P0a additive fields. The `llm_provider_class` / `llm_model` legacy
/// fields and the new P0a fields carry the same information so both views of
/// the data stay consistent.
fn build_locomo_env(
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
        task: Some("locomo".to_string()),
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

/// Run LoCoMo benchmark. For each conversation:
/// 1. Create fresh ephemeral DB
/// 2. Seed ALL observations as memories
/// 3. For each non-adversarial QA pair, search and score
/// 4. Aggregate per-category and overall metrics
async fn run_locomo_eval_core(
    path: &Path,
    reranker: Option<std::sync::Arc<dyn crate::reranker::Reranker>>,
    variant: &str,
    retrieval_method: &str,
    llm_provider_class: &str,
    llm_model: &str,
) -> Result<LocomoReport, WenlanError> {
    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);
    let mut conversations = Vec::new();
    // (category, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();
    let mut per_case: Vec<crate::eval::report::CaseResult> = Vec::new();

    for sample in &samples {
        let memories = extract_observations(sample);

        // Create ephemeral DB for this conversation
        let tmp = tempfile::tempdir().map_err(|e| WenlanError::Generic(format!("tempdir: {e}")))?;
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
                space: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
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

            let results = if let Some(reranker) = &reranker {
                db.search_memory_cross_rerank(
                    &qa.question,
                    10,
                    None,
                    &ReadScope::Global,
                    None,
                    Some(reranker.clone()),
                )
                .await?
            } else {
                db.search_memory(
                    &qa.question,
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
            per_case.push(build_locomo_case_result(
                &qa.question,
                qa.category,
                ndcg_5,
                ndcg_10,
                mrr_val,
                recall_5,
                hr_1,
            ));
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
    let mut report = LocomoReport {
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
        env: None,
        per_case,
        coverage: None,
    };
    let mut env_stamp = build_locomo_env(
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

pub async fn run_locomo_eval(path: &Path) -> Result<LocomoReport, WenlanError> {
    run_locomo_eval_core(path, None, "base", "search_memory", "none", "none").await
}

// ---------------------------------------------------------------------------
// Cross-encoder rerank benchmark runner — same as run_locomo_eval_reranked but
// swaps the LLM reranker for a cross-encoder model (fastembed TextRerank).
// ---------------------------------------------------------------------------

/// Same seeding/scoring logic as `run_locomo_eval_reranked`, but retrieval uses
/// `search_memory_cross_rerank` driven by a cross-encoder reranker
/// (model per `WENLAN_RERANKER_MODEL`; default `BGERerankerBase` since
/// 2026-06-11). Lets the eval sweep compare LLM-as-judge
/// reranking against a purpose-built cross-encoder on identical fixtures.
pub async fn run_locomo_eval_cross_rerank(
    path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
) -> Result<LocomoReport, WenlanError> {
    let llm_model = format!("cross-encoder:{}", reranker.model_id());
    run_locomo_eval_core(
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

/// Like `run_locomo_eval_cross_rerank`, but scores against a PRE-SEEDED
/// consolidated scenario DB (no per-conversation ephemeral DB, no ingest).
/// Used by PR-B's page-channel eval to surface distilled pages that the
/// fullpipeline harness wrote into the cache.
///
/// `db` MUST already contain memories with `source_id` formatted as
/// `locomo_<sample_id>_obs_<i>` (matches both the in-tree fullpipeline
/// seed path and the ephemeral seed in `run_locomo_eval_cross_rerank`).
/// Page-channel ON/OFF is controlled by the caller via the
/// `WENLAN_ENABLE_PAGE_CHANNEL` env var (read inside
/// `search_memory_cross_rerank`).
pub async fn run_locomo_eval_cross_rerank_from_db(
    db: &MemoryDB,
    path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
) -> Result<LocomoReport, WenlanError> {
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

    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);
    let mut conversations = Vec::new();
    // (category, ndcg_5, ndcg_10, mrr, recall_5, hit_rate_1)
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();
    let mut per_case: Vec<crate::eval::report::CaseResult> = Vec::new();
    let mut cov_blind_acc: Vec<f64> = Vec::new();
    let mut cov_expanded_acc: Vec<f64> = Vec::new();

    for sample in &samples {
        let memories = extract_observations(sample);

        // Re-derive the source_id mapping — matches the fullpipeline seed path.
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
                .search_memory_cross_rerank(
                    &qa.question,
                    10,
                    None,
                    &ReadScope::Global,
                    None,
                    Some(reranker.clone()),
                )
                .await?;

            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();

            if relevant_ids.is_empty() {
                continue;
            }

            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();

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
            per_case.push(build_locomo_case_result(
                &qa.question,
                qa.category,
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

    let per_cat_agg = aggregate_by_category(&all_scores);
    let coverage = if cov_blind_acc.is_empty() {
        None
    } else {
        Some(crate::eval::report::CoverageRecall {
            blind: cov_blind_acc.iter().sum::<f64>() / cov_blind_acc.len() as f64,
            expanded: cov_expanded_acc.iter().sum::<f64>() / cov_expanded_acc.len() as f64,
        })
    };
    let mut report = LocomoReport {
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
        env: None,
        per_case,
        coverage,
    };

    let env_stamp = crate::eval::shared::stamp_cr_from_db_env(|variant_tag| {
        build_locomo_env(
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

// ---------------------------------------------------------------------------
// Per-query collector (paired A/B apparatus v2)
// ---------------------------------------------------------------------------

/// Per-query variant of [`run_locomo_eval_from_db`]. Identical retrieval +
/// scoring, but emits one [`PerQueryRow`] per evaluated question (with a
/// wall-clock latency for the `search_memory` call) instead of aggregating.
///
/// `feature` / `flag_state` are stamped onto each row for the downstream
/// paired analyzer. The graph-gate skip decision is recorded per-query so the
/// T3 "skip work, no recall regression" metric is recoverable.
async fn run_locomo_eval_from_db_collect_core(
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

    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);
    let gate_on = if use_ce {
        false
    } else {
        crate::db::graph_gate_enabled()
    };
    let mut rows: Vec<PerQueryRow> = Vec::new();

    for sample in &samples {
        let memories = extract_observations(sample);
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

        for (q_idx, qa) in sample.qa.iter().enumerate() {
            if qa.category == 5 {
                continue;
            }
            let graph_skipped =
                gate_on && !crate::retrieval::signals::query_warrants_graph(&qa.question);

            let base_ids_owned: Vec<String> = if needs_base_ids {
                db.search_memory(
                    &qa.question,
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
                    &qa.question,
                    10,
                    None,
                    &ReadScope::Global,
                    None,
                    Some(reranker.clone()),
                )
                .await?
            } else {
                db.search_memory(
                    &qa.question,
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
            let channel_touched;
            let relevant_ids: HashSet<String>;
            if use_ce {
                relevant_ids = qa
                    .evidence
                    .iter()
                    .filter_map(|did| dia_to_source.get(did).cloned())
                    .collect();
                if relevant_ids.is_empty() {
                    continue;
                }
                // After result_ids exist and outside the latency window: CE-path probe.
                channel_touched = crate::eval::shared::ce_channel_touched(
                    db,
                    feature,
                    &qa.question,
                    &base_ids,
                    &result_ids,
                )
                .await?;
            } else {
                // Outside the latency window: the base-path channel-touch probe.
                channel_touched =
                    crate::eval::shared::base_channel_touched(db, feature, &qa.question).await?;
                relevant_ids = qa
                    .evidence
                    .iter()
                    .filter_map(|did| dia_to_source.get(did).cloned())
                    .collect();
                if relevant_ids.is_empty() {
                    continue;
                }
            }
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();
            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

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
                bench: "locomo".to_string(),
                flag_state: flag_state.to_string(),
                query_id: format!("{}#q{}", sample.sample_id, q_idx),
                category: qa.category.to_string(),
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
    }

    Ok(rows)
}

pub async fn run_locomo_eval_from_db_collect(
    db: &MemoryDB,
    path: &Path,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    run_locomo_eval_from_db_collect_core(db, path, None, feature, flag_state, false).await
}

/// Cross-encoder variant of [`run_locomo_eval_from_db_collect`]. Identical query
/// set + relevance judgments + scoring, but retrieval goes through
/// `search_memory_cross_rerank` (CE rescoring over the widened pool, where the
/// page / episode / fact / global-prelude channels live) instead of the base
/// `search_memory` path. Pairs OFF (base collector) against ON (this one) on the
/// same snapshot DB so a CE-path feature flag's delta is actually measurable —
/// flipping a CE-path flag on the base collector reads zero, because the base
/// read never touches that channel (the documented T20 session-diversity trap).
///
/// Pins the ungated rerank-pool knobs (multiplier=1, floor=10) when unset, same
/// as `run_locomo_eval_cross_rerank_from_db`, so the CE pool depth is
/// reproducible. `--test-threads=1` makes the process-global set_var safe.
pub async fn run_locomo_eval_cross_rerank_from_db_collect(
    db: &MemoryDB,
    path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    let needs_base = feature == "rerank" || feature.starts_with("rerank_model");
    run_locomo_eval_from_db_collect_core(db, path, Some(reranker), feature, flag_state, needs_base)
        .await
}

// ---------------------------------------------------------------------------
// Temporal oracle probe (3-arm: Baseline / ExtractCue / Oracle)
// ---------------------------------------------------------------------------

/// Which temporal cue strategy to apply per query in the temporal oracle probe.
///
/// - `Baseline`: no temporal cue — identical to the plain cross-rerank call.
/// - `ExtractCue`: extract a cue from the question text, resolved against the
///   conversation's own timeline (arm B). High-confidence cues only; others
///   fall back to `None`.
/// - `Oracle`: use the union window of the evidence observation dates (arm C).
///   If none of the evidence observations have a known `event_date`, falls back
///   to `None` (counted separately; behaves identically to Baseline for that QA).
#[derive(Clone, Copy, Debug)]
pub enum TemporalArm {
    Baseline,
    ExtractCue,
    Oracle,
}

/// Collector for the 3-arm temporal oracle probe over a pre-seeded LoCoMo DB.
///
/// Mirrors [`run_locomo_eval_cross_rerank_from_db_collect`] verbatim for the
/// relevance judgments, NDCG@10/recall@5/MRR scoring, and `PerQueryRow`
/// construction. The only difference is how `temporal_cue` is computed per
/// query, forwarded through [`crate::db::MemoryDB::search_memory_cross_rerank_cued`].
///
/// Counters printed to stdout (informational, not returned):
/// - Arm B: QAs with no high-confidence cue (fall back to `None`).
/// - Arm C: QAs where no evidence observation has a known `event_date`
///   (fall back to `None`; behaviorally identical to Baseline for those QAs).
#[allow(clippy::too_many_arguments)]
pub async fn run_locomo_eval_cross_rerank_temporal_collect(
    db: &MemoryDB,
    path: &Path,
    reranker: std::sync::Arc<dyn crate::reranker::Reranker>,
    feature: &str,
    arm: TemporalArm,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    use crate::eval::paired::PerQueryRow;
    use crate::temporal_query::{extract_cue, CueConfidence};
    use std::time::Instant;

    if std::env::var_os("RERANK_POOL_MULTIPLIER").is_none() {
        std::env::set_var("RERANK_POOL_MULTIPLIER", "1");
    }
    if std::env::var_os("RERANK_POOL_FLOOR").is_none() {
        std::env::set_var("RERANK_POOL_FLOOR", "10");
    }

    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);

    // Build event_date_map once; used by both ExtractCue (for `now` pinning)
    // and Oracle (for evidence window construction).
    let edm = event_date_map(&samples);

    let mut rows: Vec<PerQueryRow> = Vec::new();
    let mut no_cue_count = 0usize;
    let mut no_oracle_date_count = 0usize;

    for sample in &samples {
        let memories = extract_observations(sample);
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

        // Compute `now` for this sample: the latest known session date + 1 day,
        // so relative cues like "last week" resolve against the conversation's
        // own present rather than wall-clock 2026.
        // If no dated observations exist for this sample, fall back to Utc::now().
        let sample_source_ids: Vec<String> = memories
            .iter()
            .enumerate()
            .map(|(i, _)| format!("locomo_{}_obs_{}", sample.sample_id, i))
            .collect();
        let max_session_ts: Option<i64> = sample_source_ids
            .iter()
            .filter_map(|sid| edm.get(sid).copied())
            .max();
        let sample_now = match max_session_ts {
            Some(ts) => {
                chrono::DateTime::from_timestamp(ts + 86400, 0).unwrap_or_else(chrono::Utc::now)
            }
            None => chrono::Utc::now(),
        };

        for (q_idx, qa) in sample.qa.iter().enumerate() {
            if qa.category == 5 {
                continue;
            }

            let temporal_cue: Option<crate::temporal_query::DateRange> = match arm {
                TemporalArm::Baseline => None,

                TemporalArm::ExtractCue => {
                    let cue = extract_cue(&qa.question, sample_now)
                        .filter(|c| c.confidence == CueConfidence::High)
                        .map(|c| c.range);
                    if cue.is_none() {
                        no_cue_count += 1;
                    }
                    cue
                }

                TemporalArm::Oracle => {
                    // Collect event_dates for all evidence observations.
                    let dates: Vec<i64> = qa
                        .evidence
                        .iter()
                        .filter_map(|did| dia_to_source.get(did))
                        .filter_map(|sid| edm.get(sid).copied())
                        .collect();
                    if dates.is_empty() {
                        no_oracle_date_count += 1;
                        None
                    } else {
                        let start = *dates.iter().min().unwrap();
                        let end = *dates.iter().max().unwrap();
                        Some(crate::temporal_query::DateRange { start, end })
                    }
                }
            };

            let t0 = Instant::now();
            let results = db
                .search_memory_cross_rerank_cued(
                    &qa.question,
                    10,
                    None,
                    &ReadScope::Global,
                    None,
                    temporal_cue,
                    Some(reranker.clone()),
                )
                .await?;
            let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if relevant_ids.is_empty() {
                continue;
            }
            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();
            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            // temporal_touched: for arm B, record whether a high-confidence cue fired.
            let temporal_touched = match arm {
                TemporalArm::ExtractCue => Some(temporal_cue.is_some()),
                _ => None,
            };

            rows.push(PerQueryRow {
                feature: feature.to_string(),
                bench: "locomo".to_string(),
                flag_state: flag_state.to_string(),
                query_id: format!("{}#q{}", sample.sample_id, q_idx),
                category: qa.category.to_string(),
                ndcg10: metrics::ndcg_at_k(&result_ids, &grades, 10),
                recall5: metrics::recall_at_k(&result_ids, &relevant_set, 5),
                mrr: metrics::mrr(&result_ids, &relevant_set),
                latency_ms,
                graph_skipped: None,
                temporal_touched,
                channel_touched: None,
                cov_blind: None,
                cov_expanded: None,
            });
        }
    }

    println!(
        "[temporal_probe:{arm:?}] no_cue={no_cue_count} no_oracle_date={no_oracle_date_count} \
         total_rows={}",
        rows.len()
    );
    Ok(rows)
}

// ---------------------------------------------------------------------------
/// Retrieval eval over a pre-seeded DB using the base `search_memory` path
/// (vector + FTS + RRF + graph augmentation) — the path the graph gate
/// (`WENLAN_ENABLE_GRAPH_GATE`) acts on. Mirrors `run_locomo_eval_cross_rerank_from_db`
/// but without the cross-encoder/page channel, so the only LLM-free retrieval
/// signal under test is the graph augmentation + its gate. Used for the T3
/// graph-gate A/B experiment.
pub async fn run_locomo_eval_from_db(
    db: &MemoryDB,
    path: &Path,
) -> Result<LocomoReport, WenlanError> {
    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);
    let mut conversations = Vec::new();
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();
    let mut cov_acc: Vec<f64> = Vec::new();
    let gate_on = crate::db::graph_gate_enabled();
    let (mut gate_skipped, mut gate_total) = (0usize, 0usize);

    for sample in &samples {
        let memories = extract_observations(sample);
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
            gate_total += 1;
            if gate_on && !crate::retrieval::signals::query_warrants_graph(&qa.question) {
                gate_skipped += 1;
            }
            let results = db
                .search_memory(
                    &qa.question,
                    10,
                    None,
                    &ReadScope::Global,
                    None,
                    None,
                    None,
                    None,
                )
                .await?;

            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if relevant_ids.is_empty() {
                continue;
            }
            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
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

            let units: Vec<(&str, &str)> = results
                .iter()
                .map(|r| (r.source.as_str(), r.source_id.as_str()))
                .collect();
            cov_acc.push(metrics::coverage_recall(
                &metrics::build_coverage_set(&units, &HashMap::new()),
                &relevant_set,
            ));
        }

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

    if gate_on {
        eprintln!(
            "[locomo] graph-gate skipped {gate_skipped}/{gate_total} queries ({:.1}%)",
            100.0 * gate_skipped as f64 / gate_total.max(1) as f64
        );
    }
    let per_cat_agg = aggregate_by_category(&all_scores);
    let coverage = if cov_acc.is_empty() {
        None
    } else {
        Some(crate::eval::report::CoverageRecall {
            blind: cov_acc.iter().sum::<f64>() / cov_acc.len() as f64,
            expanded: cov_acc.iter().sum::<f64>() / cov_acc.len() as f64,
        })
    };
    let mut report = LocomoReport {
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
        env: None,
        per_case: Vec::new(),
        coverage,
    };
    let graph_gate = if crate::db::graph_gate_enabled() {
        "on"
    } else {
        "off"
    };
    let mut env_stamp = build_locomo_env(
        if graph_gate == "off" {
            "search_memory_gate_off"
        } else {
            "search_memory_gate_on"
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
// #15 slice-1 paired collector — expanded path + intent routing probe
// ---------------------------------------------------------------------------

/// #15 slice-1 paired collector. Returns one `PerQueryRow` per LoCoMo QA for the
/// expanded deep path, recording `graph_skipped` (the negation of the routing
/// decision actually used). The arm (intent-LLM vs keyword gate) is selected by
/// the caller via the `WENLAN_ENABLE_INTENT_LLM` env (the expanded path reads it
/// internally; this collector records the decision it observes). `flag_state` is
/// "on"/"off". Ephemeral-per-conversation seeding, mirroring `run_locomo_eval`.
///
/// Probe-operator caveats:
/// - On the BASELINE (intent-OFF) arm, `graph_skipped` records the keyword gate's
///   *verdict* (`query_warrants_graph`). That equals the search's realized routing
///   only when `WENLAN_ENABLE_GRAPH_GATE=1`; with the gate at its default OFF,
///   `search_memory_with_cue` always augments (`do_graph = !gate || warrants`), so
///   the recorded skip would be fictional. The Task 6 probe sets
///   `WENLAN_ENABLE_GRAPH_GATE=1` on the baseline arm for this reason — run it that way.
/// - On the INTENT (intent-ON) arm, the emitter runs ~2x per QA: once here to record
///   `graph_skipped`, once inside `search_memory_expanded`. Deterministic at temp=0 so
///   the two agree; size the full-run cost/time for the doubled intent-arm LLM calls.
pub async fn run_locomo_eval_expanded_intent_collect(
    path: &std::path::Path,
    llm: std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);
    let mut rows: Vec<crate::eval::paired::PerQueryRow> = Vec::new();
    let intent_arm = crate::retrieval::intent::intent_llm_enabled();

    for sample in &samples {
        let memories = extract_observations(sample);
        let tmp = tempfile::tempdir().map_err(|e| WenlanError::Generic(format!("tempdir: {e}")))?;
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter)).await?;
        let docs: Vec<RawDocument> = memories
            .iter()
            .enumerate()
            .map(|(i, mem)| RawDocument {
                content: mem.content.clone(),
                source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
                source: "memory".to_string(),
                title: format!("{} session {}", mem.speaker, mem.session_num),
                memory_type: Some("fact".to_string()),
                space: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
                ..Default::default()
            })
            .collect();
        db.upsert_documents(docs).await?;

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

        for (qi, qa) in sample.qa.iter().enumerate() {
            if qa.category == 5 {
                continue;
            }
            // Record the routing decision this arm uses, for graph_skipped.
            // Intent arm: the LLM's use_graph (a second emit call; deterministic at
            // temp=0, so it matches the decision search_memory_expanded uses).
            // Baseline arm: the keyword gate.
            let used_graph = if intent_arm {
                crate::retrieval::intent::emit_query_intent_llm(&llm, &qa.question)
                    .await
                    .use_graph
            } else {
                crate::retrieval::signals::query_warrants_graph(&qa.question)
            };

            let results = db
                .search_memory_expanded(
                    &qa.question,
                    10,
                    None,
                    &ReadScope::Global,
                    None,
                    Some(llm.clone()),
                )
                .await?;

            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if relevant_ids.is_empty() {
                continue;
            }
            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();
            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            rows.push(crate::eval::paired::PerQueryRow {
                feature: feature.to_string(),
                bench: "locomo".to_string(),
                flag_state: flag_state.to_string(),
                query_id: format!("{}#q{}", sample.sample_id, qi),
                category: qa.category.to_string(),
                ndcg10: metrics::ndcg_at_k(&result_ids, &grades, 10),
                recall5: metrics::recall_at_k(&result_ids, &relevant_set, 5),
                mrr: metrics::mrr(&result_ids, &relevant_set),
                latency_ms: 0.0,
                graph_skipped: Some(!used_graph),
                temporal_touched: None,
                channel_touched: None,
                cov_blind: None,
                cov_expanded: None,
            });
        }
    }
    Ok(rows)
}

/// Cache root for persistent per-conversation graph-populated DBs. Override with
/// `GRAPH_POP_DIR` (default `~/.cache/origin-eval/graph_pop`). Delete to rebuild
/// (e.g. after a fixture or extractor change).
pub(crate) fn graph_pop_dir(bench: &str, sample_id: &str) -> std::path::PathBuf {
    let base = std::env::var("GRAPH_POP_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        format!("{home}/.cache/origin-eval/graph_pop")
    });
    std::path::PathBuf::from(base).join(bench).join(sample_id)
}

/// Bounded fine entity sweep over a freshly-ingested DB; populates the
/// `memory_entities` junction. Mirrors the gate's `run_capped_fine_sweep`
/// (attempted-tracking so zero-entity rows cannot re-fetch forever). Fine
/// extract-ALL is used; person/speaker hubs are excluded at retrieval time by the
/// v3 type + degree filter. Returns the count of memories that got ≥1 entity.
pub async fn populate_memory_entities_sweep(
    db: &MemoryDB,
    llm: &std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
    prompts: &crate::prompts::PromptRegistry,
) -> usize {
    let mut attempted: HashSet<String> = HashSet::new();
    let mut linked = 0usize;
    loop {
        let batch = db
            .unlinked_memories(64 + attempted.len())
            .await
            .unwrap_or_default();
        let fresh: Vec<(String, String)> = batch
            .into_iter()
            .filter(|(sid, _)| !attempted.contains(sid))
            .collect();
        if fresh.is_empty() {
            break;
        }
        for (sid, content) in fresh {
            let ents = crate::kg::entity_extraction::extract_entities_for_content(
                db,
                llm,
                prompts,
                &content,
                Some(sid.as_str()),
            )
            .await
            .unwrap_or_default();
            if !ents.is_empty() {
                let refs: Vec<&str> = ents.iter().map(|s| s.as_str()).collect();
                let _ = db.link_memory_entities(&sid, &refs).await;
                linked += 1;
            }
            attempted.insert(sid);
        }
    }
    linked
}

/// Paired graph-stream collector (efficient, cached). For each conversation it
/// keeps a PERSISTENT populated DB under `<GRAPH_POP_DIR>/locomo/<sample_id>/`:
/// on first encounter it ingests the conversation + runs the fine entity sweep
/// (the only LLM cost) and persists it; every later run — the second arm, the
/// surface-new arm, knob sweeps — reuses that DB and pays ZERO extraction.
///
/// Retrieval uses the base `search_memory` path (no expansion, no rerank), so the
/// query loop is LLM-free and deterministic. The ONLY per-arm difference is the
/// `WENLAN_GRAPH_MEMORY_STREAM` env (set by the harness), making the paired Δ a
/// clean graph-OFF-vs-ON contrast. Coverage is printed when a conversation is
/// freshly populated so a null can be checked against substrate liveness.
pub async fn run_locomo_eval_graph_stream_collect(
    path: &std::path::Path,
    llm: std::sync::Arc<dyn crate::llm_provider::LlmProvider>,
    prompts: &crate::prompts::PromptRegistry,
    feature: &str,
    flag_state: &str,
) -> Result<Vec<crate::eval::paired::PerQueryRow>, WenlanError> {
    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);
    let mut rows: Vec<crate::eval::paired::PerQueryRow> = Vec::new();

    for sample in &samples {
        let memories = extract_observations(sample);
        let dir = graph_pop_dir("locomo", &sample.sample_id);
        let fresh = !dir.join("origin_memory.db").exists();
        std::fs::create_dir_all(&dir)
            .map_err(|e| WenlanError::Generic(format!("graph_pop dir: {e}")))?;
        let db = MemoryDB::new(&dir, std::sync::Arc::new(crate::events::NoopEmitter)).await?;

        if fresh {
            let docs: Vec<RawDocument> = memories
                .iter()
                .enumerate()
                .map(|(i, mem)| RawDocument {
                    content: mem.content.clone(),
                    source_id: format!("locomo_{}_obs_{}", sample.sample_id, i),
                    source: "memory".to_string(),
                    title: format!("{} session {}", mem.speaker, mem.session_num),
                    memory_type: Some("fact".to_string()),
                    space: Some("conversation".to_string()),
                    last_modified: chrono::Utc::now().timestamp(),
                    ..Default::default()
                })
                .collect();
            db.upsert_documents(docs).await?;
            let linked = populate_memory_entities_sweep(&db, &llm, prompts).await;
            println!(
                "[graph_stream] populated locomo/{}: {linked}/{} memories linked",
                sample.sample_id,
                memories.len()
            );
        }

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

        for (qi, qa) in sample.qa.iter().enumerate() {
            if qa.category == 5 {
                continue;
            }
            let results = db
                .search_memory(
                    &qa.question,
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
                crate::eval::shared::base_channel_touched(&db, feature, &qa.question).await?;

            let relevant_ids: HashSet<String> = qa
                .evidence
                .iter()
                .filter_map(|did| dia_to_source.get(did).cloned())
                .collect();
            if relevant_ids.is_empty() {
                continue;
            }
            let result_ids: Vec<&str> = results.iter().map(|r| r.source_id.as_str()).collect();
            let grades: HashMap<&str, u8> = result_ids
                .iter()
                .map(|id| (*id, if relevant_ids.contains(*id) { 1 } else { 0 }))
                .collect();
            let relevant_set: HashSet<&str> = relevant_ids.iter().map(|s| s.as_str()).collect();

            rows.push(crate::eval::paired::PerQueryRow {
                feature: feature.to_string(),
                bench: "locomo".to_string(),
                flag_state: flag_state.to_string(),
                query_id: format!("{}#q{}", sample.sample_id, qi),
                category: qa.category.to_string(),
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
    }
    Ok(rows)
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
            space: Some("conversation".to_string()),
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
) -> Result<LocomoReport, WenlanError> {
    let mut samples = load_locomo(path)?;
    apply_locomo_limit(&mut samples);
    let mut conversations = Vec::new();
    let mut all_scores: Vec<(u8, f64, f64, f64, f64, f64)> = Vec::new();
    let mut per_case: Vec<crate::eval::report::CaseResult> = Vec::new();
    let mut total_memories_inserted: usize = 0;

    let gate = match mode {
        LocomoGateMode::Gated => Some(QualityGate::new(GateConfig::default())),
        _ => None,
    };

    for sample in &samples {
        let memories = extract_observations(sample);

        // Create ephemeral DB for this conversation
        let tmp = tempfile::tempdir().map_err(|e| WenlanError::Generic(format!("tempdir: {e}")))?;
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
                space: Some("conversation".to_string()),
                last_modified: chrono::Utc::now().timestamp(),
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
                .search_memory(
                    &qa.question,
                    10,
                    None,
                    &ReadScope::Global,
                    None,
                    None,
                    None,
                    None,
                )
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
            per_case.push(build_locomo_case_result(
                &qa.question,
                qa.category,
                ndcg_5,
                ndcg_10,
                mrr_val,
                recall_5,
                hr_1,
            ));
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
    let mut report = LocomoReport {
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
        env: None,
        per_case,
        coverage: None,
    };
    let mut env_stamp = build_locomo_env("gated", path, "search_memory", "none", "none", None);
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
            env: None,
            per_case: vec![],
            coverage: None,
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
    fn locomo_env_records_judge_model_when_batch_runs() {
        let mut report = LocomoReport {
            conversations: vec![],
            aggregate_ndcg_at_10: 0.0,
            aggregate_mrr: 0.0,
            aggregate_recall_at_5: 0.0,
            aggregate_hit_rate_at_1: 0.0,
            total_questions: 0,
            total_memories: 0,
            per_category_aggregate: vec![],
            qa_accuracy: None,
            baseline: None,
            env: None,
            per_case: vec![],
            coverage: None,
        };
        report.env = Some(crate::eval::report::ReportEnv {
            fixture_revision: "deadbeef".into(),
            embedder_model: "bge-base-en-v1.5-q".into(),
            embedder_revision: "768d".into(),
            retrieval_method: "search_memory".into(),
            llm_provider_class: "on-device".into(),
            llm_model: "qwen3-4b".into(),
            judge_model: None,
            origin_version: env!("CARGO_PKG_VERSION").into(),
            eval_timestamp_unix: 0,
            ..Default::default()
        });
        crate::eval::judge::stamp_judge_model(&mut report.env, "claude-haiku-4-5-20251001");
        assert_eq!(
            report.env.as_ref().and_then(|e| e.judge_model.clone()),
            Some("claude-haiku-4-5-20251001".to_string())
        );
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
                coverage: None,
            }),
            env: None,
            per_case: vec![],
            coverage: None,
        };

        let text = report.to_terminal();
        assert!(text.contains("LoCoMo Benchmark"));
        assert!(text.contains("NDCG@10"));
        assert!(text.contains("Baseline comparison:"));
        assert!(text.contains("open-domain"));
        // Verify delta printing is present
        assert!(text.contains("->"));
    }

    /// Build a vec of `n` minimal `LocomoSample`s for env-limit tests.
    fn mock_samples(n: usize) -> Vec<LocomoSample> {
        (0..n)
            .map(|i| {
                let json = format!(
                    r#"{{
                        "sample_id": "mock-{i}",
                        "conversation": {{}},
                        "observation": {{}},
                        "qa": []
                    }}"#
                );
                serde_json::from_str::<LocomoSample>(&json).unwrap()
            })
            .collect()
    }

    #[test]
    fn eval_locomo_limit_truncates_when_set() {
        // Unique env var name so the test doesn't race the real EVAL_LOCOMO_LIMIT.
        let var = "EVAL_LOCOMO_LIMIT_TEST_TRUNCATE";
        let mut samples = mock_samples(10);
        std::env::set_var(var, "2");
        apply_limit_from_env(&mut samples, var, "locomo", "conversations");
        std::env::remove_var(var);
        assert_eq!(samples.len(), 2, "limit=2 should truncate 10 down to 2");
        assert_eq!(samples[0].sample_id, "mock-0");
        assert_eq!(samples[1].sample_id, "mock-1");
    }

    #[test]
    fn eval_locomo_limit_no_op_when_unset() {
        let var = "EVAL_LOCOMO_LIMIT_TEST_NOOP";
        // Defensive: ensure the var is unset before the call.
        std::env::remove_var(var);
        let mut samples = mock_samples(5);
        apply_limit_from_env(&mut samples, var, "locomo", "conversations");
        assert_eq!(samples.len(), 5, "unset env var must leave samples intact");
    }

    #[test]
    fn locomo_event_date_map_parses_session_dates_to_source_ids() {
        let sample: LocomoSample = serde_json::from_value(serde_json::json!({
            "sample_id": "conv-test",
            "conversation": { "session_1_date_time": "1:56 pm on 8 May, 2023" },
            "qa": [],
            "observation": {
                "session_1_observation": { "Alice": [["Alice likes tea", "D1:1"]] }
            }
        }))
        .unwrap();
        let map = event_date_map(&[sample]);
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 8)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        // source_id mirrors the seed builder: locomo_<sample>_obs_<enumerate-index>.
        assert_eq!(map.get("locomo_conv-test_obs_0"), Some(&expected));
    }

    #[test]
    fn to_terminal_prints_coverage_delta_when_present() {
        let mut report = LocomoReport {
            conversations: vec![],
            aggregate_ndcg_at_10: 0.0,
            aggregate_mrr: 0.0,
            aggregate_recall_at_5: 0.0,
            aggregate_hit_rate_at_1: 0.0,
            total_questions: 0,
            total_memories: 0,
            per_category_aggregate: vec![],
            qa_accuracy: None,
            baseline: None,
            env: None,
            per_case: vec![],
            coverage: Some(crate::eval::report::CoverageRecall {
                blind: 0.40,
                expanded: 0.55,
            }),
        };
        let out = report.to_terminal();
        assert!(
            out.contains("Coverage recall"),
            "missing coverage header:\n{out}"
        );
        assert!(out.contains("0.4000"), "missing blind:\n{out}");
        assert!(out.contains("0.5500"), "missing expanded:\n{out}");
        assert!(out.contains("+0.1500"), "missing delta:\n{out}");
        report.coverage = None;
        assert!(!report.to_terminal().contains("Coverage recall"));
    }
}
