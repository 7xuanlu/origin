// SPDX-License-Identifier: Apache-2.0
//! Eval report formatting — terminal table and JSON output.

use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ReportEnv {
    pub fixture_revision: String,
    pub embedder_model: String,
    pub embedder_revision: String,
    pub retrieval_method: String,
    pub llm_provider_class: String,
    pub llm_model: String,
    pub judge_model: Option<String>,
    pub origin_version: String,
    pub eval_timestamp_unix: i64,

    // P0a additive fields:
    #[serde(default)]
    pub layer: Option<crate::eval::EvalLayer>,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub embed_dim: Option<u32>,
    #[serde(default = "default_similarity_fn")]
    pub similarity_fn_name: String,
    #[serde(default)]
    pub judge_model_id: Option<String>,
    #[serde(default)]
    pub mcp_schema_hash: Option<String>,
    #[serde(default)]
    pub skill_prompt_hash: Option<String>,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub schema_db_version: Option<u32>,
    #[serde(default)]
    pub migrations_hash: Option<String>,
    #[serde(default = "default_n_runs")]
    pub n_runs: u32,
    #[serde(default)]
    pub is_single_run: bool,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub timestamp_utc: Option<String>,
    #[serde(default)]
    pub git_sha: Option<String>,
    #[serde(default)]
    pub warmup_iterations: u32,
    #[serde(default)]
    pub eval_max_usd_baseline_cap: Option<f64>,
    #[serde(default)]
    pub eval_max_usd_run_cap: Option<f64>,
    #[serde(default)]
    pub eval_max_wall_secs_cap: Option<u64>,
    #[serde(default)]
    pub total_cost_usd: f64,
    #[serde(default)]
    pub total_wall_secs: u64,
    /// Runtime flags active during this eval run (e.g. `WENLAN_DISABLE_SUPERSEDE_FILTER=1`).
    /// Used to attest to non-default runtime configuration in baseline JSON files.
    #[serde(default)]
    pub flags: Vec<String>,
}

fn default_similarity_fn() -> String {
    "cosine".to_string()
}
fn default_schema_version() -> u32 {
    1
}
fn default_n_runs() -> u32 {
    1
}

impl Default for ReportEnv {
    fn default() -> Self {
        Self {
            // 9 legacy fields
            fixture_revision: String::new(),
            embedder_model: String::new(),
            embedder_revision: String::new(),
            retrieval_method: String::new(),
            llm_provider_class: String::new(),
            llm_model: String::new(),
            judge_model: None,
            origin_version: String::new(),
            eval_timestamp_unix: 0,
            // P0a additive fields — mirror serde default attributes
            layer: None,
            task: None,
            variant: None,
            embed_dim: None,
            similarity_fn_name: default_similarity_fn(),
            judge_model_id: None,
            mcp_schema_hash: None,
            skill_prompt_hash: None,
            schema_version: default_schema_version(),
            schema_db_version: None,
            migrations_hash: None,
            n_runs: default_n_runs(),
            is_single_run: false,
            run_id: None,
            timestamp_utc: None,
            git_sha: None,
            warmup_iterations: 0,
            eval_max_usd_baseline_cap: None,
            eval_max_usd_run_cap: None,
            eval_max_wall_secs_cap: None,
            total_cost_usd: 0.0,
            total_wall_secs: 0,
            flags: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
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
    // Environment capture (schema v1)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub env: Option<ReportEnv>,
    // Per-query latency summary
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub latency: Option<crate::eval::latency::LatencySummary>,
    // Guard fields (P0c)
    #[serde(default)]
    pub total_scenarios: usize,
    #[serde(default)]
    pub skipped_scenarios: Vec<String>,
    #[serde(default)]
    pub enrichment_failures: usize,
    #[serde(default)]
    pub truncated_reason: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
}

/// Source-expanded coverage recall for the page-channel eval (task #45).
///
/// `blind` is set-based recall over memory results only (a retrieved page
/// contributes only its own `page_*` id, which never matches memory-keyed
/// ground truth). `expanded` is recall after expanding each retrieved page to
/// the memory source ids it was distilled from (provenance expansion). The
/// `expanded - blind` delta is the honest "pages help" signal. Both are
/// dedup-safe set metrics — a gold id is credited once regardless of how many
/// units point to it, so a page-expanded id also retrieved directly cannot
/// inflate the score.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct CoverageRecall {
    pub blind: f64,
    pub expanded: f64,
}

/// Hash the subset of `ReportEnv` fields that determine baseline comparability.
///
/// Includes: fixture_revision, embedder_revision, llm_provider_class,
/// llm_model, mcp_schema_hash, skill_prompt_hash, schema_version,
/// schema_db_version, similarity_fn_name, flags (sorted).
///
/// Excludes: layer (path component), variant (path component), n_runs,
/// run_id, timestamp, costs, latency fields. These vary across runs of
/// the same eval setup, so cross-run comparison of metrics requires the
/// COMPARABLE subset to match.
///
/// **Contract:** any modification to this function's input set (adding,
/// removing, reordering, or changing the encoding of any field) MUST bump
/// `default_schema_version()` so old vs new baselines hash distinctly and
/// don't appear comparable.
pub fn comparable_env_hash(env: &ReportEnv) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(env.fixture_revision.as_bytes());
    h.update(b"|");
    h.update(env.embedder_revision.as_bytes());
    h.update(b"|");
    h.update(env.llm_provider_class.as_bytes());
    h.update(b"|");
    h.update(env.llm_model.as_bytes());
    h.update(b"|");
    h.update(env.mcp_schema_hash.as_deref().unwrap_or("").as_bytes());
    h.update(b"|");
    h.update(env.skill_prompt_hash.as_deref().unwrap_or("").as_bytes());
    h.update(b"|");
    h.update(env.schema_version.to_string().as_bytes());
    h.update(b"|");
    h.update(
        env.schema_db_version
            .map(|v| v.to_string())
            .unwrap_or_default()
            .as_bytes(),
    );
    h.update(b"|");
    h.update(env.similarity_fn_name.as_bytes());
    h.update(b"|");
    let mut sorted_flags = env.flags.clone();
    sorted_flags.sort();
    h.update(sorted_flags.join(",").as_bytes());
    let hex = format!("{:x}", h.finalize());
    hex.chars().take(8).collect()
}

/// Encode the on-disk path for a baseline given an env stamp.
///
/// Layout: `<root>/<layer_dir>/<task>/<variant>__<comparable_hash>.json`
///
/// E.g.: `~/.cache/origin-eval/baselines/l1_db/locomo/base__a1b2c3d4.json`
///
/// Panics if env.layer / env.task / env.variant are None — these are required
/// for any baseline that ships through `save_full_report`.
pub fn encode_baseline_path(root: &std::path::Path, env: &ReportEnv) -> std::path::PathBuf {
    let layer = env
        .layer
        .expect("encode_baseline_path: env.layer must be Some")
        .as_path_component();
    let task = env
        .task
        .as_deref()
        .expect("encode_baseline_path: env.task must be Some");
    let variant = env
        .variant
        .as_deref()
        .expect("encode_baseline_path: env.variant must be Some");
    let hash = comparable_env_hash(env);
    root.join(layer)
        .join(task)
        .join(format!("{}__{}.json", variant, hash))
}

/// Encode retrieval variant + provider + fixture-hash into a baseline filename.
/// Shared by EvalReport, LocomoReport, and LongMemEvalReport.
/// Falls back to `base + ".json"` when `env` is None (back-compat).
pub fn encode_baseline_filename(env: Option<&ReportEnv>, base: &str) -> String {
    let Some(env) = env else {
        return format!("{}.json", base);
    };
    let variant = env
        .retrieval_method
        .trim_start_matches("search_memory")
        .trim_start_matches('_');
    let variant = if variant.is_empty() { "base" } else { variant };
    format!(
        "{}__{}__{}__{}.json",
        base, variant, env.llm_provider_class, env.fixture_revision
    )
}

impl EvalReport {
    /// Encode retrieval variant + provider + fixture-hash into baseline filename.
    /// Falls back to base + ".json" if env is missing (back-compat).
    pub fn baseline_filename(&self, base: &str) -> String {
        encode_baseline_filename(self.env.as_ref(), base)
    }

    /// Format report as terminal-friendly text.
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("Wenlan Memory Eval\n");
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

    /// Return the name of the first non-finite (NaN or Inf) f64 field, if any.
    ///
    /// Used by `save_full_report` to guard against corrupt metric values before
    /// writing. `serde_json` silently converts NaN/Inf to `null` (JSON has no
    /// representation for them), so a post-serialization walk would miss them.
    pub fn first_non_finite_field(&self) -> Option<&'static str> {
        let checks: &[(&'static str, f64)] = &[
            ("ndcg_at_10", self.ndcg_at_10),
            ("ndcg_at_5", self.ndcg_at_5),
            ("map_at_5", self.map_at_5),
            ("map_at_10", self.map_at_10),
            ("mrr", self.mrr),
            ("recall_at_1", self.recall_at_1),
            ("recall_at_3", self.recall_at_3),
            ("recall_at_5", self.recall_at_5),
            ("hit_rate_at_1", self.hit_rate_at_1),
            ("hit_rate_at_3", self.hit_rate_at_3),
            ("precision_at_3", self.precision_at_3),
            ("precision_at_5", self.precision_at_5),
        ];
        for &(name, val) in checks {
            if !val.is_finite() {
                return Some(name);
            }
        }
        if let Some(v) = self.empty_set_false_confidence {
            if !v.is_finite() {
                return Some("empty_set_false_confidence");
            }
        }
        if let Some(v) = self.score_gap {
            if !v.is_finite() {
                return Some("score_gap");
            }
        }
        if let Some(v) = self.temporal_ordering_rate {
            if !v.is_finite() {
                return Some("temporal_ordering_rate");
            }
        }
        None
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

// ---------------------------------------------------------------------------
// P0c: save_full_report + save_partial_report
// ---------------------------------------------------------------------------

/// Save an `EvalReport` to the layered baselines layout with strict guards.
///
/// Path: `<root>/<layer>/<task>/<variant>__<comparable_hash>.json`
/// via [`encode_baseline_path`].
///
/// Guards (any failure returns `Err`, no file is written):
/// - `report.env` must be `Some(...)`.
/// - All metric f64 fields must be finite (no NaN, no Inf). Checked via
///   `EvalReport::first_non_finite_field()` which walks known metric fields
///   directly — serde_json silently maps NaN to JSON null, so a serialized
///   walk would miss the very value we're rejecting.
/// - `skipped_scenarios.len() / total_scenarios <= 5%` when `total_scenarios > 0`.
/// - `enrichment_failures == 0` unless `EVAL_ACCEPT_PARTIAL=1` is set in the
///   environment.
///
/// Atomic write: serialise to `<final>.tmp.<pid>.<nanos>` in the **same**
/// directory as the final path, then `std::fs::rename`. Same-filesystem rename
/// is guaranteed because the tmp file lives beside the target.
pub fn save_full_report(
    baselines_root: &std::path::Path,
    report: &EvalReport,
) -> anyhow::Result<std::path::PathBuf> {
    let env = report
        .env
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("save_full_report: env is required, got None"))?;

    // Guard: all f64 metric fields must be finite.
    // Note: serde_json serializes NaN/Inf as null (RFC 7159 has no NaN literal),
    // so a JSON-tree walk would silently miss them. Check struct fields directly.
    if let Some(bad) = report.first_non_finite_field() {
        anyhow::bail!(
            "save_full_report: non-finite metric value (NaN or Inf) in field `{}`",
            bad
        );
    }

    // Guard: skip rate <= 5 %.
    if report.total_scenarios > 0 {
        let rate = report.skipped_scenarios.len() as f64 / report.total_scenarios as f64;
        if rate > 0.05 {
            anyhow::bail!(
                "save_full_report: overall skip rate {:.2}% > 5% — write to partial/ instead",
                rate * 100.0
            );
        }
    }

    // Guard: no enrichment failures unless caller opted in.
    if report.enrichment_failures > 0 && std::env::var("EVAL_ACCEPT_PARTIAL").is_err() {
        anyhow::bail!(
            "save_full_report: {} enrichment_failure(s) present; \
             set EVAL_ACCEPT_PARTIAL=1 to override",
            report.enrichment_failures
        );
    }

    // Compute final path and ensure parent dir exists.
    let final_path = encode_baseline_path(baselines_root, env);
    let parent = final_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("baseline path has no parent: {:?}", final_path))?;
    std::fs::create_dir_all(parent)?;

    // Atomic same-directory tmp write + rename.
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_filename = format!(
        "{}.tmp.{}.{}",
        final_path.file_name().unwrap().to_string_lossy(),
        pid,
        nanos
    );
    let tmp_path = parent.join(tmp_filename);

    let json = serde_json::to_string_pretty(report)?;
    std::fs::write(&tmp_path, &json)?;
    std::fs::rename(&tmp_path, &final_path)?;

    Ok(final_path)
}

/// Save a partial / truncated / failed report to `<eval_root>/partial/`.
///
/// Never writes to the baselines directory, so scripts globbing baselines
/// never pick up garbage. Format:
/// `partial/<run_id>__<layer>__<task>__<variant>.json`
///
/// `reason` is recorded in `report.truncated_reason` in the saved file.
pub fn save_partial_report(
    eval_root: &std::path::Path,
    report: &EvalReport,
    reason: &str,
) -> anyhow::Result<std::path::PathBuf> {
    let partial_dir = eval_root.join("partial");
    std::fs::create_dir_all(&partial_dir)?;

    let env = report.env.as_ref();
    let run_id = env
        .and_then(|e| e.run_id.as_deref())
        .unwrap_or("unknown_run");
    let layer = env
        .and_then(|e| e.layer)
        .map(|l| l.as_path_component())
        .unwrap_or("unknown_layer");
    let task = env
        .and_then(|e| e.task.as_deref())
        .unwrap_or("unknown_task");
    let variant = env
        .and_then(|e| e.variant.as_deref())
        .unwrap_or("unknown_variant");

    let path = partial_dir.join(format!("{}__{}__{}__{}.json", run_id, layer, task, variant));

    let mut to_save = report.clone();
    to_save.truncated_reason = Some(reason.to_string());

    let json = serde_json::to_string_pretty(&to_save)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::EvalLayer;

    fn sample_env(layer: EvalLayer, variant: &str) -> ReportEnv {
        ReportEnv {
            layer: Some(layer),
            task: Some("locomo".to_string()),
            variant: Some(variant.to_string()),
            fixture_revision: "fixture_aaaa".to_string(),
            embedder_revision: "BGE-Base-EN-v1.5-Q".to_string(),
            llm_provider_class: "on-device".to_string(),
            llm_model: "qwen3-4b".to_string(),
            mcp_schema_hash: None,
            skill_prompt_hash: None,
            schema_version: 1,
            schema_db_version: Some(46),
            similarity_fn_name: "cosine".to_string(),
            ..ReportEnv::default()
        }
    }

    #[test]
    fn comparable_hash_excludes_layer_and_variant() {
        // L1 base vs L2 base: same comparable fields → same hash across layers.
        let h1 = comparable_env_hash(&sample_env(EvalLayer::L1Db, "base"));
        let h2 = comparable_env_hash(&sample_env(EvalLayer::L2Http, "base"));
        assert_eq!(h1, h2, "same comparable fields → same hash across layers");

        // base vs reranked at L1: also same hash (variant excluded).
        let h3 = comparable_env_hash(&sample_env(EvalLayer::L1Db, "reranked"));
        assert_eq!(h1, h3, "variant should not affect comparable hash");
    }

    #[test]
    fn comparable_hash_changes_when_fixture_changes() {
        let e1 = sample_env(EvalLayer::L1Db, "base");
        let mut e2 = sample_env(EvalLayer::L1Db, "base");
        e2.fixture_revision = "different_fixture".to_string();
        assert_ne!(comparable_env_hash(&e1), comparable_env_hash(&e2));
    }

    #[test]
    fn comparable_hash_changes_when_schema_version_bumps() {
        let e1 = sample_env(EvalLayer::L1Db, "base");
        let mut e2 = sample_env(EvalLayer::L1Db, "base");
        e2.schema_version = 2;
        assert_ne!(comparable_env_hash(&e1), comparable_env_hash(&e2));
    }

    #[test]
    fn comparable_hash_stable_across_calls() {
        let e = sample_env(EvalLayer::L1Db, "base");
        assert_eq!(comparable_env_hash(&e), comparable_env_hash(&e));
        assert_eq!(comparable_env_hash(&e).len(), 8, "expected sha256[..8]");
    }

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
            env: None,
            latency: None,
            total_scenarios: 0,
            skipped_scenarios: vec![],
            enrichment_failures: 0,
            truncated_reason: None,
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

    #[test]
    fn encode_baseline_path_layout() {
        let env = sample_env(EvalLayer::L1Db, "base");
        let root = std::path::Path::new("/tmp/baselines");
        let path = encode_baseline_path(root, &env);
        let comparable = comparable_env_hash(&env);
        // Build the expectation with `join` rather than a literal '/' string:
        // `encode_baseline_path` joins, so on Windows it produces backslashes and
        // the old string comparison failed there. The layout is what this asserts,
        // not the separator.
        let expected = root
            .join("l1_db")
            .join("locomo")
            .join(format!("base__{comparable}.json"));
        assert_eq!(path, expected);
    }

    #[test]
    fn encode_baseline_path_all_layers_distinct() {
        let base = std::path::Path::new("/tmp");
        let p1 = encode_baseline_path(base, &sample_env(EvalLayer::L1Db, "base"));
        let p2 = encode_baseline_path(base, &sample_env(EvalLayer::L2Http, "base"));
        let p3 = encode_baseline_path(base, &sample_env(EvalLayer::L3Mcp, "base"));
        assert_ne!(p1, p2);
        assert_ne!(p2, p3);
        assert_ne!(p1, p3);
    }

    #[test]
    fn encode_baseline_path_all_variants_distinct() {
        let base = std::path::Path::new("/tmp");
        let p_base = encode_baseline_path(base, &sample_env(EvalLayer::L1Db, "base"));
        let p_rerank = encode_baseline_path(base, &sample_env(EvalLayer::L1Db, "reranked"));
        let p_aq = encode_baseline_path(base, &sample_env(EvalLayer::L1Db, "answer_quality"));
        assert_ne!(p_base, p_rerank);
        assert_ne!(p_rerank, p_aq);
    }

    #[test]
    fn report_env_serializes_flags_field() {
        let env = ReportEnv {
            flags: vec!["WENLAN_DISABLE_SUPERSEDE_FILTER=1".into()],
            ..Default::default()
        };
        let json = serde_json::to_string(&env).expect("serialize");
        assert!(json.contains("\"flags\""), "json missing 'flags' key");
        assert!(
            json.contains("WENLAN_DISABLE_SUPERSEDE_FILTER=1"),
            "json missing flag value"
        );
    }

    #[test]
    fn report_env_default_flags_empty() {
        let env = ReportEnv::default();
        assert!(env.flags.is_empty(), "default flags should be empty vec");
    }

    #[test]
    fn report_env_deserializes_missing_flags_as_empty() {
        // Backward compat: existing baselines without `flags` should still deserialize.
        let env = ReportEnv::default();
        let mut value: serde_json::Value = serde_json::to_value(&env).unwrap();
        if let Some(obj) = value.as_object_mut() {
            obj.remove("flags");
        }
        let json = serde_json::to_string(&value).unwrap();
        let parsed: ReportEnv = serde_json::from_str(&json).expect("deserialize without flags");
        assert!(parsed.flags.is_empty());
    }

    #[test]
    fn comparable_env_hash_distinguishes_flags() {
        // Two envs identical in all other comparable fields but differing in flags
        // must produce distinct hashes so they don't overwrite each other's baseline.
        let e1 = ReportEnv {
            flags: vec!["page_channel=on".into()],
            ..sample_env(EvalLayer::L1Db, "base")
        };
        let e2 = ReportEnv {
            flags: vec!["page_channel=off".into()],
            ..sample_env(EvalLayer::L1Db, "base")
        };
        assert_ne!(
            comparable_env_hash(&e1),
            comparable_env_hash(&e2),
            "envs differing only in flags must hash differently"
        );
        // Also verify that empty flags vs non-empty flags differ.
        let e3 = sample_env(EvalLayer::L1Db, "base"); // flags = []
        assert_ne!(
            comparable_env_hash(&e1),
            comparable_env_hash(&e3),
            "non-empty flags must differ from empty flags"
        );
    }
}

// ---------------------------------------------------------------------------
// McNemar exact test
// ---------------------------------------------------------------------------

/// Two-sided exact McNemar's test p-value for paired binary outcomes.
///
/// `b` = count of "structured correct, decomposed wrong" flips;
/// `c` = count of "structured wrong, decomposed correct" flips.
///
/// Uses the binomial distribution at p=0.5 over the discordant pairs.
/// p_two_sided = 2 * Pr(X <= min(b,c) | n=b+c, p=0.5), clipped to 1.0.
///
/// Returns 1.0 when n=0 (no signal).
pub fn mcnemar_p_value(b: u32, c: u32) -> f64 {
    let n = (b as u64) + (c as u64);
    if n == 0 {
        return 1.0;
    }
    let m = b.min(c) as u64;
    // Compute log Pr(X <= m | n, p=0.5) via log-sum-exp over log-PMFs for stability.
    let log_half_n = -(n as f64) * std::f64::consts::LN_2;
    let mut log_cdf: f64 = f64::NEG_INFINITY;
    for k in 0..=m {
        let log_pmf = log_binomial(n, k) + log_half_n;
        log_cdf = log_sum_exp(log_cdf, log_pmf);
    }
    let one_sided = log_cdf.exp();
    (2.0 * one_sided).min(1.0)
}

/// log(C(n, k)) computed via lgamma for numerical stability.
fn log_binomial(n: u64, k: u64) -> f64 {
    if k == 0 || k == n {
        return 0.0;
    }
    let k = k.min(n - k);
    lgamma(n as f64 + 1.0) - lgamma(k as f64 + 1.0) - lgamma((n - k) as f64 + 1.0)
}

/// Lanczos approximation to lgamma. Sufficient precision for our use case
/// (b, c are nonnegative integer counts up to a few thousand).
fn lgamma(x: f64) -> f64 {
    if x < 0.5 {
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - lgamma(1.0 - x);
    }
    let g = 7.0;
    let p = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = p[0];
    for (i, &pi) in p.iter().enumerate().skip(1) {
        a += pi / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Numerically-stable log(exp(a) + exp(b)).
fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Returns the z-value for the given alpha level (two-tailed normal quantile at 1 - alpha/2).
/// Only supports alpha in {0.01, 0.05, 0.10}.
fn z_for_alpha(alpha: f64) -> f64 {
    if (alpha - 0.05).abs() < 1e-12 {
        1.959_963_984_540_054
    } else if (alpha - 0.10).abs() < 1e-12 {
        1.644_853_626_951_472_2
    } else if (alpha - 0.01).abs() < 1e-12 {
        2.575_829_303_548_900_4
    } else {
        unimplemented!("alpha must be 0.01, 0.05, or 0.10")
    }
}

/// Mid-p variant of McNemar's test (Fagerland, Lydersen, Laake 2013,
/// BMC Med Res Methodology 13:91).
///
/// Uniformly better than exact McNemar across sample sizes. Subtracts half
/// the boundary mass from the tail probability.
///
/// `b` = "method A correct, method B wrong" discordant pair count.
/// `c` = "method B correct, method A wrong" discordant pair count.
/// Returns 1.0 when n == 0.
pub fn mcnemar_mid_p(b: u32, c: u32) -> f64 {
    let n = (b as u64) + (c as u64);
    if n == 0 {
        return 1.0;
    }
    let m = b.min(c) as u64;
    let log_half_n = -(n as f64) * std::f64::consts::LN_2;

    // Pr(X = m | n, p=0.5)
    let log_pmf_m = log_binomial(n, m) + log_half_n;
    let pmf_m = log_pmf_m.exp();

    // Pr(X < m | n, p=0.5) = sum over k in 0..m (exclusive)
    let mut log_cdf_excl: f64 = f64::NEG_INFINITY;
    for k in 0..m {
        let log_pmf = log_binomial(n, k) + log_half_n;
        log_cdf_excl = log_sum_exp(log_cdf_excl, log_pmf);
    }
    let cdf_excl = if m == 0 { 0.0 } else { log_cdf_excl.exp() };

    // One-sided mid-p = Pr(X < m) + 0.5 * Pr(X = m)
    let one_sided = cdf_excl + 0.5 * pmf_m;
    (2.0 * one_sided).min(1.0)
}

/// Odds ratio for matched-pairs binary outcome (McNemar effect size).
///
/// `b` = discordant pairs where method A wins; `c` = discordant pairs where method B wins.
/// Returns 1.0 when both are 0 (no information), INFINITY when c == 0 and b > 0, 0.0 when b == 0 and c > 0.
pub fn odds_ratio_mcnemar(b: u32, c: u32) -> f64 {
    if b == 0 && c == 0 {
        return 1.0;
    }
    if c == 0 {
        return f64::INFINITY;
    }
    if b == 0 {
        return 0.0;
    }
    b as f64 / c as f64
}

/// Wilson score confidence interval for a single proportion.
/// Reference: Wilson 1927.
///
/// `successes` = number of successes; `total` = sample size; `alpha` = significance level.
/// `alpha` must be 0.01, 0.05, or 0.10.
/// Precondition: `successes <= total` (debug-asserted). Violating this causes `p_hat > 1.0`,
/// which produces a negative radicand and NaN output.
/// Returns (0.0, 1.0) when `total` == 0.
pub fn wilson_ci(successes: u32, total: u32, alpha: f64) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }
    debug_assert!(
        successes <= total,
        "successes ({successes}) exceeds total ({total})"
    );
    let n = total as f64;
    let p_hat = successes as f64 / n;
    let z = z_for_alpha(alpha);
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let spread = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n).sqrt() / denom;
    let lower = (center - spread).max(0.0);
    let upper = (center + spread).min(1.0);
    (lower, upper)
}

/// Newcombe-Wilson hybrid confidence interval for paired proportion difference.
/// Reference: Newcombe 1998, Stat Med 17:2635 (Method 10).
///
/// `n` = total paired observations; `b` = method A correct & method B wrong;
/// `c` = method A wrong & method B correct; `alpha` = significance level.
/// Returns (0.0, 0.0) when `n` == 0 or when `b` == 0 && `c` == 0.
pub fn paired_diff_ci_newcombe(n: u32, b: u32, c: u32, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    if b == 0 && c == 0 {
        return (0.0, 0.0);
    }
    let fn_ = n as f64;
    let fb = b as f64;
    let fc = c as f64;
    let delta_hat = (fb - fc) / fn_;

    let (l1, u1) = wilson_ci(b, n, alpha);
    let (l2, u2) = wilson_ci(c, n, alpha);

    let p1 = fb / fn_;
    let p2 = fc / fn_;

    let lower = (delta_hat - ((p1 - l1).powi(2) + (u2 - p2).powi(2)).sqrt()).max(-1.0);
    let upper = (delta_hat + ((u1 - p1).powi(2) + (p2 - l2).powi(2)).sqrt()).min(1.0);
    (lower, upper)
}

/// Paired-binary report for two retrieval configurations evaluated on the same
/// question set. Discordant counts feed McNemar; concordant counts inform
/// agreement-rate. Per-config Wilson CI at alpha=0.05, paired-diff Newcombe
/// Method 10 CI at alpha=0.05.
///
/// Used by the pool-expansion P0a sweep: baseline vs treatment-A (Mem0 shape).
/// Per-category stratification is the caller's responsibility — pre-filter
/// the slices to a single category before calling, or call once per category.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct PairedMcnemarReport {
    /// Number of queries matched in both baseline and treatment.
    pub n_matched: u32,
    /// Baseline-only queries (treatment missing the question).
    pub n_baseline_only: u32,
    /// Treatment-only queries (baseline missing the question).
    pub n_treatment_only: u32,
    /// Both correct.
    pub a: u32,
    /// Baseline correct, treatment wrong.
    pub b: u32,
    /// Baseline wrong, treatment correct.
    pub c: u32,
    /// Both wrong.
    pub d: u32,
    pub exact_p: f64,
    pub mid_p: f64,
    pub odds_ratio: f64,
    pub baseline_accuracy: f64,
    pub treatment_accuracy: f64,
    pub baseline_ci_95: (f64, f64),
    pub treatment_ci_95: (f64, f64),
    pub accuracy_diff_ci_95: (f64, f64),
}

/// Compute paired McNemar + Wilson CI + Newcombe paired-diff CI for two
/// per-case result vectors. Matches by `CaseResult.query` (string equality).
///
/// A question is "correct" when its `ndcg_at_10` is at or above `threshold`.
/// This is a binary projection of a continuous metric; callers should pick
/// a threshold matched to the eval contract (P0a acceptance criteria locks
/// `threshold = 0.5`).
///
/// Unmatched queries (present in only one slice) are surfaced via
/// `n_baseline_only` / `n_treatment_only` so the caller can spot missing
/// scenarios instead of silently dropping them.
pub fn paired_mcnemar(
    baseline: &[CaseResult],
    treatment: &[CaseResult],
    threshold: f64,
) -> PairedMcnemarReport {
    use std::collections::HashMap;

    let treat_map: HashMap<&str, &CaseResult> =
        treatment.iter().map(|c| (c.query.as_str(), c)).collect();
    let base_keys: std::collections::HashSet<&str> =
        baseline.iter().map(|c| c.query.as_str()).collect();

    let mut a: u32 = 0;
    let mut b: u32 = 0;
    let mut c: u32 = 0;
    let mut d: u32 = 0;
    let mut n_matched: u32 = 0;
    let mut n_baseline_only: u32 = 0;

    for base in baseline {
        match treat_map.get(base.query.as_str()) {
            Some(treat) => {
                n_matched = n_matched.saturating_add(1);
                let b_correct = base.ndcg_at_10 >= threshold;
                let t_correct = treat.ndcg_at_10 >= threshold;
                match (b_correct, t_correct) {
                    (true, true) => a = a.saturating_add(1),
                    (true, false) => b = b.saturating_add(1),
                    (false, true) => c = c.saturating_add(1),
                    (false, false) => d = d.saturating_add(1),
                }
            }
            None => n_baseline_only = n_baseline_only.saturating_add(1),
        }
    }

    let n_treatment_only: u32 = treatment
        .iter()
        .filter(|t| !base_keys.contains(t.query.as_str()))
        .count() as u32;

    let baseline_correct = a.saturating_add(b);
    let treatment_correct = a.saturating_add(c);

    let baseline_accuracy = if n_matched == 0 {
        0.0
    } else {
        baseline_correct as f64 / n_matched as f64
    };
    let treatment_accuracy = if n_matched == 0 {
        0.0
    } else {
        treatment_correct as f64 / n_matched as f64
    };

    PairedMcnemarReport {
        n_matched,
        n_baseline_only,
        n_treatment_only,
        a,
        b,
        c,
        d,
        exact_p: mcnemar_p_value(b, c),
        mid_p: mcnemar_mid_p(b, c),
        odds_ratio: odds_ratio_mcnemar(b, c),
        baseline_accuracy,
        treatment_accuracy,
        baseline_ci_95: wilson_ci(baseline_correct, n_matched, 0.05),
        treatment_ci_95: wilson_ci(treatment_correct, n_matched, 0.05),
        accuracy_diff_ci_95: paired_diff_ci_newcombe(n_matched, c, b, 0.05),
    }
}

#[cfg(test)]
mod mcnemar_tests {
    use super::*;

    #[test]
    fn mid_p_smaller_than_exact_for_small_n() {
        // For (b=10, c=0): exact two-sided p ≈ 0.00195, mid-p ≈ 0.000976
        // (mid-p subtracts half the boundary mass).
        let exact = mcnemar_p_value(10, 0);
        let mid = mcnemar_mid_p(10, 0);
        assert!(mid < exact, "expected mid-p ({mid}) < exact ({exact})");
        assert!(mid > 0.0);
    }

    #[test]
    fn mid_p_no_disagreements_returns_one() {
        assert!((mcnemar_mid_p(0, 0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mid_p_handles_balanced_discordance() {
        // b == c: two-sided mid-p should be at most 1.0
        let p = mcnemar_mid_p(5, 5);
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn odds_ratio_basics() {
        assert_eq!(odds_ratio_mcnemar(0, 0), 1.0);
        assert!(odds_ratio_mcnemar(5, 0).is_infinite());
        assert_eq!(odds_ratio_mcnemar(0, 5), 0.0);
        assert!((odds_ratio_mcnemar(6, 3) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn wilson_ci_basics() {
        // 50/100 at alpha=0.05: known CI ≈ (0.4038, 0.5962)
        let (lo, hi) = wilson_ci(50, 100, 0.05);
        assert!((lo - 0.4038).abs() < 1e-3, "lo={lo}");
        assert!((hi - 0.5962).abs() < 1e-3, "hi={hi}");
        // Boundary cases
        let (lo0, hi0) = wilson_ci(0, 0, 0.05);
        assert_eq!((lo0, hi0), (0.0, 1.0));
        let (lo1, hi1) = wilson_ci(0, 10, 0.05);
        assert!(lo1 == 0.0);
        assert!(hi1 > 0.0 && hi1 < 0.5);
    }

    #[test]
    #[should_panic(expected = "successes")]
    #[cfg(debug_assertions)]
    fn wilson_ci_panics_when_successes_exceeds_total() {
        let _ = wilson_ci(101, 100, 0.05);
    }

    fn case(query: &str, ndcg: f64) -> CaseResult {
        CaseResult {
            query: query.into(),
            ndcg_at_10: ndcg,
            ndcg_at_5: 0.0,
            map_at_10: 0.0,
            mrr: 0.0,
            recall_at_5: 0.0,
            hit_rate_at_1: 0.0,
            precision_at_3: 0.0,
            negative_leakage: 0,
            neg_above_relevant: 0,
            category: None,
        }
    }

    #[test]
    fn paired_mcnemar_clean_win() {
        // Baseline: 5 questions all under threshold. Treatment: same 5 over threshold.
        // b=0, c=5 → treatment always wins on discordant pairs.
        let base: Vec<_> = (0..5).map(|i| case(&format!("q{i}"), 0.1)).collect();
        let treat: Vec<_> = (0..5).map(|i| case(&format!("q{i}"), 0.9)).collect();
        let r = paired_mcnemar(&base, &treat, 0.5);
        assert_eq!(r.n_matched, 5);
        assert_eq!(r.b, 0);
        assert_eq!(r.c, 5);
        assert_eq!(r.a, 0);
        assert_eq!(r.d, 0);
        assert!(r.odds_ratio == 0.0);
        // p_mid one-sided ≈ 0.5 * (0.5)^5 → very small; two-sided still well below 0.05
        assert!(r.mid_p < 0.05, "mid_p={}", r.mid_p);
        assert_eq!(r.baseline_accuracy, 0.0);
        assert_eq!(r.treatment_accuracy, 1.0);
        // accuracy diff CI brackets +1.0
        assert!(r.accuracy_diff_ci_95.0 > 0.0);
    }

    #[test]
    fn paired_mcnemar_no_change() {
        // Identical scores → b=c=0, p=1.0, no signal.
        let base: Vec<_> = (0..10).map(|i| case(&format!("q{i}"), 0.7)).collect();
        let treat = base.clone();
        let r = paired_mcnemar(&base, &treat, 0.5);
        assert_eq!(r.n_matched, 10);
        assert_eq!(r.b, 0);
        assert_eq!(r.c, 0);
        assert_eq!(r.a, 10);
        assert!((r.mid_p - 1.0).abs() < 1e-12);
        assert_eq!(r.baseline_accuracy, 1.0);
        assert_eq!(r.treatment_accuracy, 1.0);
    }

    #[test]
    fn paired_mcnemar_tracks_unmatched_queries() {
        let base = vec![case("q1", 0.9), case("q2", 0.9), case("q3", 0.1)];
        let treat = vec![case("q1", 0.9), case("q4", 0.9)];
        let r = paired_mcnemar(&base, &treat, 0.5);
        assert_eq!(r.n_matched, 1);
        assert_eq!(r.n_baseline_only, 2);
        assert_eq!(r.n_treatment_only, 1);
    }

    #[test]
    fn paired_mcnemar_threshold_split() {
        // 10 questions, threshold=0.5.
        // 4 baseline-correct (above 0.5), 6 baseline-wrong.
        // Treatment: 6 correct, 4 wrong. Pattern engineered:
        // a=3 (both correct), b=1 (base only), c=3 (treat only), d=3 (both wrong)
        let base = vec![
            case("q1", 0.9), // base correct
            case("q2", 0.9),
            case("q3", 0.9),
            case("q4", 0.9), // base only correct
            case("q5", 0.1),
            case("q6", 0.1),
            case("q7", 0.1), // treat only correct
            case("q8", 0.1),
            case("q9", 0.1),
            case("q10", 0.1),
        ];
        let treat = vec![
            case("q1", 0.9),
            case("q2", 0.9),
            case("q3", 0.9),  // both correct = a
            case("q4", 0.1),  // base correct, treat wrong = b
            case("q5", 0.9),  // both? no, base 0.1, treat 0.9 = c
            case("q6", 0.9),  // c
            case("q7", 0.9),  // c
            case("q8", 0.1),  // d
            case("q9", 0.1),  // d
            case("q10", 0.1), // d
        ];
        let r = paired_mcnemar(&base, &treat, 0.5);
        assert_eq!(r.n_matched, 10);
        assert_eq!(r.a, 3);
        assert_eq!(r.b, 1);
        assert_eq!(r.c, 3);
        assert_eq!(r.d, 3);
        assert!((r.baseline_accuracy - 0.4).abs() < 1e-12);
        assert!((r.treatment_accuracy - 0.6).abs() < 1e-12);
        // Newcombe CI for (c - b) / n = (3 - 1) / 10 = +0.2 brackets the point estimate
        assert!(r.accuracy_diff_ci_95.0 < 0.2 && r.accuracy_diff_ci_95.1 > 0.2);
    }

    #[test]
    fn paired_diff_ci_basics() {
        // n=100, b=20, c=10: delta_hat = 0.10. Newcombe hybrid CI brackets 0.10.
        let (lo, hi) = paired_diff_ci_newcombe(100, 20, 10, 0.05);
        assert!(
            lo < 0.10 && hi > 0.10,
            "expected CI brackets 0.10, got [{lo}, {hi}]"
        );
        assert!(hi - lo > 0.0);
        assert!(lo >= -1.0 && hi <= 1.0);

        // No discordance: delta = 0, CI = (0, 0)
        let (lo0, hi0) = paired_diff_ci_newcombe(100, 0, 0, 0.05);
        assert_eq!((lo0, hi0), (0.0, 0.0));
    }

    #[test]
    fn wikipedia_example_b121_c59_p_below_1e_4() {
        // Wikipedia McNemar's example: 121 vs 59 disagreements.
        // Two-sided exact p ≈ 1.6e-5 per scipy.stats.mcnemar(exact=True).
        let p = mcnemar_p_value(121, 59);
        assert!(p < 1e-4, "expected p<1e-4, got {p}");
    }

    #[test]
    fn no_disagreements_p_one() {
        // 0 vs 0 disagreements: no signal, p = 1.0.
        assert!((mcnemar_p_value(0, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn one_vs_zero_p_one() {
        // 1 vs 0: tiny dataset, two-sided p = 1.0
        // (one-sided P(X<=0|n=1, p=0.5) = 0.5; two-sided = 1.0).
        assert!((mcnemar_p_value(1, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ten_vs_zero_p_below_005() {
        // n=10, all flips one direction. Two-sided exact p ≈ 0.00195. Easy <0.05.
        let p = mcnemar_p_value(10, 0);
        assert!(p < 0.05, "expected p<0.05, got {p}");
    }
}
