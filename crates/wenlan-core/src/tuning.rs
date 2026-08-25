// SPDX-License-Identifier: Apache-2.0
//! TuningConfig — intelligence-affecting numeric parameters loaded from TOML.
//! Serde defaults match the current hardcoded values exactly.

use serde::Deserialize;
use std::path::{Path, PathBuf};

// ---- serde default functions ----
fn d_085() -> f64 {
    0.85
}
fn d_60_i64() -> i64 {
    60
}
fn d_002() -> f64 {
    0.02
}
fn d_60_f64() -> f64 {
    60.0
}
fn d_80() -> usize {
    80
}
fn d_015() -> f32 {
    0.15
}
fn d_01_f32() -> f32 {
    0.1
}
fn d_0005() -> f64 {
    0.005
}
fn d_5_usize() -> usize {
    5
}
fn d_3_usize() -> usize {
    3
}
fn d_86400() -> i64 {
    86400
}
fn d_20_usize() -> usize {
    20
}
fn d_120() -> u64 {
    120
}
fn d_21600() -> i64 {
    21600
}
fn d_5_u64() -> u64 {
    5
}
fn d_100() -> usize {
    100
}
fn d_12() -> usize {
    12
}
fn d_090() -> f32 {
    0.90
}
fn d_070() -> f32 {
    0.70
}
fn d_050() -> f32 {
    0.50
}
fn d_0001() -> f64 {
    0.001
}
fn d_001() -> f64 {
    0.01
}
fn d_005() -> f64 {
    0.05
}
fn d_10_f32() -> f32 {
    1.0
}
fn d_07_f32() -> f32 {
    0.7
}
fn d_04_f32() -> f32 {
    0.4
}
fn d_30_u64() -> u64 {
    30
}
fn d_015_f64() -> f64 {
    0.15
}
fn d_03() -> f64 {
    0.3
}
fn d_true() -> bool {
    true
}
fn d_false() -> bool {
    false
}
fn d_10_usize() -> usize {
    10
}
fn d_12_usize() -> usize {
    12
}
fn d_50_usize() -> usize {
    50
}
fn d_7_usize() -> usize {
    7
}
fn d_15_f32() -> f32 {
    1.5
}
fn d_03_f32() -> f32 {
    0.3
}
fn d_60_f32() -> f32 {
    60.0
}
fn d_085_f32() -> f32 {
    0.85
}
fn d_115_f32() -> f32 {
    1.15
}
fn d_02_f32() -> f32 {
    0.2
}
fn d_25_f32() -> f32 {
    2.5
}
fn d_2_usize() -> usize {
    2
}
fn d_073() -> f64 {
    0.73
}
fn d_8000_usize() -> usize {
    8000
}
fn d_50000_usize() -> usize {
    50000
}
fn d_070_f64() -> f64 {
    0.75
}
fn d_075() -> f64 {
    0.75
}
fn d_060() -> f64 {
    0.60
}
fn d_13_f32() -> f32 {
    1.3
}
fn d_30_i64() -> i64 {
    30
}
fn d_168_u64() -> u64 {
    168
}
fn d_600_usize() -> usize {
    600
}
fn d_1024_u32() -> u32 {
    1024
}
fn d_15_u64() -> u64 {
    15
}
fn d_90_i64() -> i64 {
    90
}
fn d_01_f64() -> f64 {
    0.1
}

// ---- TuningConfig ----

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TuningConfig {
    pub router: RouterConfig,
    pub scoring: ScoringConfig,
    pub refinery: RefineryConfig,
    pub narrative: NarrativeConfig,
    pub briefing: BriefingConfig,
    pub confidence: ConfidenceConfig,
    pub eval: EvalConfig,
    pub search_scoring: SearchScoringConfig,
    #[serde(default)]
    pub distillation: DistillationConfig,
    #[serde(default)]
    pub gate: GateConfig,
    #[serde(default)]
    pub compress: ContextCompressConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RouterConfig {
    #[serde(default = "d_085")]
    pub window_dedup_threshold: f64,
    #[serde(default = "d_085")]
    pub consumer_dedup_threshold: f64,
    #[serde(default = "d_60_i64")]
    pub consumer_dedup_window_secs: i64,
    #[serde(default = "d_002")]
    pub hellinger_threshold: f64,
    #[serde(default = "d_60_f64")]
    pub afk_threshold_secs: f64,
    #[serde(default = "d_30_u64")]
    pub recent_focus_window_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScoringConfig {
    #[serde(default = "d_80")]
    pub min_text_length: usize,
    #[serde(default = "d_015")]
    pub score_threshold: f32,
    #[serde(default = "d_01_f32")]
    pub focus_bonus: f32,
    #[serde(default = "d_0005")]
    pub keyword_min_threshold: f64,
}
#[derive(Debug, Clone, Deserialize)]
pub struct RefineryConfig {
    #[serde(default = "d_5_usize")]
    pub max_proposals_per_steep: usize,
    #[serde(default = "d_3_usize")]
    pub min_memories_for_recap: usize,
    #[serde(default = "d_86400")]
    pub recap_lookback_secs: i64,
    #[serde(default = "d_20_usize")]
    pub max_reweave_per_steep: usize,
    #[serde(default = "d_120")]
    pub steep_deadline_secs: u64,
    #[serde(default = "d_015_f64")]
    pub entity_link_distance: f64,
    #[serde(default = "d_03")]
    pub consolidation_confidence_threshold: f64,
    #[serde(default = "d_3_usize")]
    pub consolidation_batch_size: usize,
    #[serde(default = "d_30_i64")]
    pub batch_window_secs: i64,
    #[serde(default = "d_168_u64")]
    pub kg_rethink_interval_hours: u64,
    #[serde(default = "d_5_usize")]
    pub entity_backfill_batch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DistillationConfig {
    #[serde(default = "d_073")]
    pub similarity_threshold: f64,
    #[serde(default = "d_060")]
    pub formation_threshold: f64,
    /// Max tokens per distillation cluster for on-device models (Qwen3-4B: 8K,
    /// Qwen3.5-9B: 16K effective synthesis window per research). Default 8000.
    #[serde(default = "d_8000_usize")]
    pub ondevice_token_limit: usize,
    /// Max tokens per distillation cluster for API models (Haiku: 32K, Sonnet:
    /// 64K effective synthesis window). Default 50000 — safe for Sonnet, within
    /// Haiku's range. Clusters exceeding this are sub-clustered.
    #[serde(default = "d_50000_usize")]
    pub api_token_limit: usize,
    #[serde(default = "d_3_usize")]
    pub max_retries: usize,
    #[serde(default = "d_20_usize")]
    pub max_clusters_per_steep: usize,
    #[serde(default = "d_3_usize", alias = "concept_min_cluster_size")]
    pub page_min_cluster_size: usize,
    #[serde(default = "d_075", alias = "concept_growth_threshold")]
    pub page_growth_threshold: f64,
    #[serde(default = "d_085")]
    pub page_match_threshold: f64,
    /// Max DETECT candidates attempted per sweep tick. DETECT is intentionally
    /// bounded separately from synthesis because it scans the staging pool.
    #[serde(default = "d_5_usize")]
    pub detect_max_candidates_per_tick: usize,
    /// Reserved for future integration into search_memory scoring pipeline.
    /// Currently pages are searched via separate search_pages endpoint.
    #[serde(default = "d_13_f32", alias = "concept_boost")]
    pub page_boost: f32,
    /// Hard cap on size of unlinked-memory clusters (no entity_id, no domain).
    /// Clusters above this size are skipped during distillation. Safety valve
    /// for the runaway-cluster failure mode (see spec 2026-04-25).
    #[serde(default = "d_50_usize")]
    pub max_unlinked_cluster_size: usize,
    /// Hard cap on size of entity- or community-grouped clusters before the
    /// agent's coherence check rejects them as grab-bags. A 12+ prose-memory
    /// cluster at the default 0.60 similarity is almost always a community
    /// pile (e.g. cid=16 "Wenlan" sweeping in unrelated sub-topics). When a
    /// grouped sub-cluster exceeds this, re-split repeatedly in +0.05
    /// threshold steps up to a ceiling of 0.92; only a sub-cluster still
    /// over cap at that ceiling is dropped.
    #[serde(default = "d_12_usize")]
    pub max_grouped_cluster_size: usize,
    /// Minimum source-memory overlap required for a page to pass the
    /// retrieval-time relevance gate. A page is included in chat context
    /// only if at least this many of its source memories appear in the
    /// search_memory results for the same query.
    ///
    /// Default 2: filters pages whose source memories don't appear in
    /// search results (LME-style noise) while keeping pages with genuine
    /// topical overlap (LoCoMo-style coherence).
    ///
    /// Tradeoffs measured 2026-04-27:
    /// - LME (noisy data): 33.7% -> 39.9% (+6.2pp) at min_overlap=2
    /// - LoCoMo (coherent data): 32.0% -> 30.5% (-1.5pp) at min_overlap=2
    ///
    /// Lower (1) preserves more pages but lets noise back in.
    /// Higher (3+) is more aggressive filtering.
    #[serde(default = "d_2_usize", alias = "concept_min_overlap")]
    pub page_min_overlap: usize,
    #[serde(default)]
    pub export_vault_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GateConfig {
    #[serde(default = "d_true")]
    pub enabled: bool,
    #[serde(default = "d_5_usize")]
    pub min_word_count: usize,
    /// Flag threshold: similarity at or above this value stores the memory and flags it as a near duplicate.
    #[serde(default = "d_070_f64")]
    pub novelty_threshold: f64,
    #[serde(default = "d_true")]
    pub noise_patterns_enabled: bool,
    #[serde(default = "d_true")]
    pub credential_check_enabled: bool,
    #[serde(default = "d_true")]
    pub log_rejections: bool,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_word_count: 5,
            novelty_threshold: 0.75,
            noise_patterns_enabled: true,
            credential_check_enabled: true,
            log_rejections: true,
        }
    }
}

/// Read-time context compression (T10, gap C6). Master enable is the env
/// flag `WENLAN_ENABLE_CONTEXT_COMPRESS` (see
/// [`crate::retrieval::compress::context_compress_enabled`]); these fields are
/// the knobs read once that gate is open. Default OFF -> byte-identical to
/// today's verbatim bundle.
#[derive(Debug, Clone, Deserialize)]
pub struct ContextCompressConfig {
    #[serde(default = "d_false")]
    pub enabled: bool,
    #[serde(default = "d_600_usize")]
    pub min_chars: usize,
    #[serde(default = "d_1024_u32")]
    pub max_output_tokens: u32,
    #[serde(default = "d_15_u64")]
    pub timeout_secs: u64,
}

impl Default for ContextCompressConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_chars: 600,
            max_output_tokens: 1024,
            timeout_secs: 15,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct NarrativeConfig {
    #[serde(default = "d_86400")]
    pub stale_secs: i64,
    #[serde(default = "d_12")]
    pub max_memories: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BriefingConfig {
    #[serde(default = "d_21600")]
    pub stale_secs: i64,
    #[serde(default = "d_5_u64")]
    pub stale_memory_delta: u64,
    #[serde(default = "d_5_usize")]
    pub max_topic_memories: usize,
    #[serde(default = "d_100")]
    pub max_memory_chars: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfidenceConfig {
    #[serde(default = "d_090")]
    pub protected_base: f32,
    #[serde(default = "d_070")]
    pub standard_base: f32,
    #[serde(default = "d_050")]
    pub ephemeral_base: f32,
    #[serde(default = "d_0001")]
    pub protected_decay: f64,
    #[serde(default = "d_001")]
    pub standard_decay: f64,
    #[serde(default = "d_005")]
    pub ephemeral_decay: f64,
    #[serde(default = "d_07_f32")]
    pub low_quality_multiplier: f32,
    #[serde(default = "d_10_f32")]
    pub full_trust_weight: f32,
    #[serde(default = "d_07_f32")]
    pub review_trust_weight: f32,
    #[serde(default = "d_04_f32")]
    pub untrusted_weight: f32,
}

/// T21 Stage 1 — archive-not-delete soft eviction policy.
///
/// Controls which stale, low-value memories the refinery's `evict` phase
/// flips to `supersede_mode='archive'` (recoverable, never hard-deleted).
/// All gates are conservative and the feature is opt-in via
/// `WENLAN_ENABLE_EVICTION`; see [`crate::db::eviction_enabled`].
#[derive(Debug, Clone, Deserialize)]
pub struct EvictionConfig {
    /// A memory is only eligible once its `last_accessed` is at least this many
    /// days in the past. Guards against archiving fresh-but-low-confidence rows.
    #[serde(default = "d_90_i64")]
    pub min_age_days: i64,
    /// A memory is only eligible when its `effective_confidence` is strictly
    /// below this floor. `effective_confidence` already folds confidence ×
    /// recency × access via the decay phase, so it acts as the salience proxy.
    #[serde(default = "d_01_f64")]
    pub evict_confidence_floor: f64,
    /// Optional per-space cap: when a space holds more non-immune memories than
    /// this, the lowest-`effective_confidence` overflow is archived even if it
    /// would otherwise pass the floor. `None` disables the cap.
    #[serde(default)]
    pub per_space_cap: Option<usize>,
    /// Forward-compat for Stage 2 (hard-delete + consolidation). In Stage 1 this
    /// is a no-op: archiving IS recovery, so nothing is ever hard-deleted.
    #[serde(default = "d_true")]
    pub recover_before_delete: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalConfig {
    #[serde(default = "d_true")]
    pub enabled: bool,
    #[serde(default = "d_true")]
    pub signal_capture: bool,
    #[serde(default = "d_false")]
    pub llm_judge_enabled: bool,
    #[serde(default = "d_10_usize")]
    pub llm_judge_sample_rate: usize,
    #[serde(default = "d_50_usize")]
    pub min_signals_for_tune: usize,
    #[serde(default = "d_7_usize")]
    pub tune_cooldown_days: usize,
    #[serde(default = "d_005")]
    pub tune_min_improvement: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchScoringConfig {
    #[serde(default = "d_25_f32")]
    pub confirmation_boost: f32,
    #[serde(default = "d_03_f32")]
    pub recap_penalty: f32,
    #[serde(default = "d_60_f32")]
    pub rrf_k: f32,
    #[serde(default = "d_15_f32")]
    pub domain_boost: f32,
    /// Weight for FTS/BM25 signal in RRF fusion (0.0-1.0). Lower values reduce
    /// keyword-matching noise that can overpower semantic similarity.
    #[serde(default = "d_02_f32")]
    pub fts_weight: f32,
    /// T8 salience prior band floor: importance=1 maps to this multiplier.
    #[serde(default = "d_085_f32")]
    pub salience_floor: f32,
    /// T8 salience prior band ceil: importance=10 maps to this multiplier.
    #[serde(default = "d_115_f32")]
    pub salience_ceil: f32,
}

// Remaining manual Default impls for sub-configs with serde default helpers.
impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            window_dedup_threshold: d_085(),
            consumer_dedup_threshold: d_085(),
            consumer_dedup_window_secs: d_60_i64(),
            hellinger_threshold: d_002(),
            afk_threshold_secs: d_60_f64(),
            recent_focus_window_secs: d_30_u64(),
        }
    }
}
impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            min_text_length: d_80(),
            score_threshold: d_015(),
            focus_bonus: d_01_f32(),
            keyword_min_threshold: d_0005(),
        }
    }
}
impl Default for RefineryConfig {
    fn default() -> Self {
        Self {
            max_proposals_per_steep: d_10_usize(),
            min_memories_for_recap: d_3_usize(),
            recap_lookback_secs: d_86400(),
            max_reweave_per_steep: d_20_usize(),
            steep_deadline_secs: d_120(),
            entity_link_distance: d_015_f64(),
            consolidation_confidence_threshold: d_03(),
            consolidation_batch_size: d_10_usize(),
            batch_window_secs: d_30_i64(),
            kg_rethink_interval_hours: d_168_u64(),
            entity_backfill_batch_size: d_5_usize(),
        }
    }
}
impl Default for NarrativeConfig {
    fn default() -> Self {
        Self {
            stale_secs: d_86400(),
            max_memories: d_12(),
        }
    }
}
impl Default for BriefingConfig {
    fn default() -> Self {
        Self {
            stale_secs: d_21600(),
            stale_memory_delta: d_5_u64(),
            max_topic_memories: d_5_usize(),
            max_memory_chars: d_100(),
        }
    }
}
impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            protected_base: d_090(),
            standard_base: d_070(),
            ephemeral_base: d_050(),
            protected_decay: d_0001(),
            standard_decay: d_001(),
            ephemeral_decay: d_005(),
            low_quality_multiplier: d_07_f32(),
            full_trust_weight: d_10_f32(),
            review_trust_weight: d_07_f32(),
            untrusted_weight: d_04_f32(),
        }
    }
}
impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            min_age_days: d_90_i64(),
            evict_confidence_floor: d_01_f64(),
            per_space_cap: None,
            recover_before_delete: d_true(),
        }
    }
}
impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            enabled: d_true(),
            signal_capture: d_true(),
            llm_judge_enabled: d_false(),
            llm_judge_sample_rate: d_10_usize(),
            min_signals_for_tune: d_50_usize(),
            tune_cooldown_days: d_7_usize(),
            tune_min_improvement: d_005(),
        }
    }
}
impl Default for SearchScoringConfig {
    fn default() -> Self {
        Self {
            confirmation_boost: 1.8,
            recap_penalty: 0.3,
            rrf_k: 60.0,
            domain_boost: 1.5,
            fts_weight: 0.5,
            salience_floor: 0.85,
            salience_ceil: 1.15,
        }
    }
}
impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: d_073(),
            formation_threshold: d_060(),
            ondevice_token_limit: d_8000_usize(),
            api_token_limit: d_50000_usize(),
            max_retries: d_3_usize(),
            max_clusters_per_steep: d_20_usize(),
            page_min_cluster_size: d_3_usize(),
            page_growth_threshold: d_075(),
            page_match_threshold: d_085(),
            detect_max_candidates_per_tick: d_5_usize(),
            page_boost: d_13_f32(),
            max_unlinked_cluster_size: d_50_usize(),
            max_grouped_cluster_size: d_12_usize(),
            page_min_overlap: d_2_usize(),
            export_vault_path: None,
        }
    }
}

impl TuningConfig {
    /// Load tuning config from a TOML file, falling back to defaults for missing keys.
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(cfg) => {
                    log::info!("[tuning] loaded config from {}", path.display());
                    cfg
                }
                Err(e) => {
                    log::warn!(
                        "[tuning] parse error in {}: {e}, using defaults",
                        path.display()
                    );
                    Self::default()
                }
            },
            Err(_) => {
                log::info!("[tuning] no config at {}, using defaults", path.display());
                Self::default()
            }
        }
    }

    /// Returns the default tuning config file path.
    pub fn config_path() -> PathBuf {
        crate::config::data_root().join("intelligence.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `WENLAN_DATA_DIR` must isolate `intelligence.toml` alongside the rest
    /// of the daemon's data-dir-scoped state (`config.rs::data_root`), not
    /// just `config.json` — a `--data-dir` scratch daemon must never read the
    /// developer's real tuning config.
    #[test]
    fn config_path_honors_wenlan_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        temp_env::with_var("WENLAN_DATA_DIR", Some(dir.path()), || {
            assert_eq!(
                TuningConfig::config_path(),
                dir.path().join("intelligence.toml")
            );
        });
    }

    #[test]
    fn test_default_matches_current_values() {
        let cfg = TuningConfig::default();
        // Router
        assert_eq!(cfg.router.window_dedup_threshold, 0.85);
        assert_eq!(cfg.router.consumer_dedup_threshold, 0.85);
        assert_eq!(cfg.router.consumer_dedup_window_secs, 60);
        assert_eq!(cfg.router.hellinger_threshold, 0.02);
        assert_eq!(cfg.router.afk_threshold_secs, 60.0);
        assert_eq!(cfg.router.recent_focus_window_secs, 30);
        // Scoring
        assert_eq!(cfg.scoring.min_text_length, 80);
        assert_eq!(cfg.scoring.score_threshold, 0.15);
        assert_eq!(cfg.scoring.focus_bonus, 0.1);
        assert_eq!(cfg.scoring.keyword_min_threshold, 0.005);
        // Refinery
        assert_eq!(cfg.refinery.max_proposals_per_steep, 10);
        assert_eq!(cfg.refinery.min_memories_for_recap, 3);
        assert_eq!(cfg.refinery.recap_lookback_secs, 86400);
        assert_eq!(cfg.refinery.max_reweave_per_steep, 20);
        assert_eq!(cfg.refinery.steep_deadline_secs, 120);
        assert_eq!(cfg.refinery.entity_link_distance, 0.15);
        assert_eq!(cfg.refinery.consolidation_confidence_threshold, 0.3);
        assert_eq!(cfg.refinery.consolidation_batch_size, 10);
        assert_eq!(cfg.refinery.batch_window_secs, 30);
        assert_eq!(cfg.refinery.kg_rethink_interval_hours, 168);
        // Narrative
        assert_eq!(cfg.narrative.stale_secs, 86400);
        assert_eq!(cfg.narrative.max_memories, 12);
        // Briefing
        assert_eq!(cfg.briefing.stale_secs, 21600);
        assert_eq!(cfg.briefing.stale_memory_delta, 5);
        assert_eq!(cfg.briefing.max_topic_memories, 5);
        assert_eq!(cfg.briefing.max_memory_chars, 100);
        // Confidence
        assert_eq!(cfg.confidence.protected_base, 0.90);
        assert_eq!(cfg.confidence.standard_base, 0.70);
        assert_eq!(cfg.confidence.ephemeral_base, 0.50);
        assert_eq!(cfg.confidence.protected_decay, 0.001);
        assert_eq!(cfg.confidence.standard_decay, 0.01);
        assert_eq!(cfg.confidence.ephemeral_decay, 0.05);
        assert_eq!(cfg.confidence.low_quality_multiplier, 0.7);
        assert_eq!(cfg.confidence.full_trust_weight, 1.0);
        assert_eq!(cfg.confidence.review_trust_weight, 0.7);
        assert_eq!(cfg.confidence.untrusted_weight, 0.4);
    }

    #[test]
    fn test_load_partial_toml_override() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("intelligence.toml");
        std::fs::write(
            &path,
            r#"
[router]
window_dedup_threshold = 0.90

[scoring]
score_threshold = 0.25
"#,
        )
        .unwrap();

        let cfg = TuningConfig::load(&path);
        assert_eq!(cfg.router.window_dedup_threshold, 0.90);
        assert_eq!(cfg.scoring.score_threshold, 0.25);
        // Non-overridden values should keep defaults
        assert_eq!(cfg.router.consumer_dedup_threshold, 0.85);
        assert_eq!(cfg.refinery.max_proposals_per_steep, 10);
    }

    #[test]
    fn test_eval_config_defaults() {
        let cfg = TuningConfig::default();
        assert!(cfg.eval.enabled);
        assert!(cfg.eval.signal_capture);
        assert!(!cfg.eval.llm_judge_enabled);
        assert_eq!(cfg.eval.llm_judge_sample_rate, 10);
        assert_eq!(cfg.eval.min_signals_for_tune, 50);
        assert_eq!(cfg.eval.tune_cooldown_days, 7);
        assert_eq!(cfg.eval.tune_min_improvement, 0.05);
    }

    #[test]
    fn test_search_scoring_config_defaults() {
        let cfg = TuningConfig::default();
        assert_eq!(cfg.search_scoring.confirmation_boost, 1.8);
        assert_eq!(cfg.search_scoring.recap_penalty, 0.3);
        assert_eq!(cfg.search_scoring.rrf_k, 60.0);
        assert_eq!(cfg.search_scoring.domain_boost, 1.5);
    }

    #[test]
    fn test_load_nonexistent_toml_returns_defaults() {
        let cfg = TuningConfig::load(Path::new("/nonexistent/intelligence.toml"));
        assert_eq!(cfg.router.window_dedup_threshold, 0.85);
    }

    #[test]
    fn test_gate_config_defaults() {
        let config: GateConfig = serde_json::from_str("{}").unwrap();
        assert!(config.enabled);
        assert_eq!(config.min_word_count, 5);
        assert!((config.novelty_threshold - 0.75).abs() < f64::EPSILON);
        assert!(config.noise_patterns_enabled);
        assert!(config.credential_check_enabled);
        assert!(config.log_rejections);
    }

    #[test]
    fn test_gate_config_override() {
        let config: GateConfig =
            serde_json::from_str(r#"{"enabled": false, "novelty_threshold": 0.92}"#).unwrap();
        assert!(!config.enabled);
        assert!((config.novelty_threshold - 0.92).abs() < f64::EPSILON);
    }

    #[test]
    fn test_concept_config_defaults() {
        let cfg = DistillationConfig::default();
        assert!((cfg.formation_threshold - 0.60).abs() < f64::EPSILON);
        assert!((cfg.page_match_threshold - 0.85).abs() < f64::EPSILON);
        assert_eq!(cfg.page_min_cluster_size, 3);
        assert!((cfg.page_growth_threshold - 0.75).abs() < 0.01);
        assert!((cfg.page_boost - 1.3).abs() < 0.01);
    }

    #[test]
    fn distillation_thresholds_load_from_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("intelligence.toml");
        std::fs::write(
            &path,
            r#"
[distillation]
formation_threshold = 0.42
page_match_threshold = 0.91
"#,
        )
        .unwrap();

        let cfg = TuningConfig::load(&path);

        assert!((cfg.distillation.formation_threshold - 0.42).abs() < f64::EPSILON);
        assert!((cfg.distillation.page_match_threshold - 0.91).abs() < f64::EPSILON);
        assert_eq!(cfg.distillation.page_min_cluster_size, 3);
        assert!((cfg.distillation.page_growth_threshold - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn distillation_config_default_has_unlinked_cluster_cap() {
        let cfg = DistillationConfig::default();
        assert_eq!(cfg.max_unlinked_cluster_size, 50);
    }

    #[test]
    fn distillation_config_default_has_grouped_cluster_cap() {
        let cfg = DistillationConfig::default();
        assert_eq!(cfg.max_grouped_cluster_size, 12);
    }

    #[test]
    fn distillation_config_default_concept_min_overlap() {
        let cfg = DistillationConfig::default();
        assert_eq!(cfg.page_min_overlap, 2);
    }

    // -- T10: context-compression tuning defaults + override --
    #[test]
    fn tuning_default_compress_disabled() {
        let cfg = TuningConfig::default();
        assert!(!cfg.compress.enabled);
        assert_eq!(cfg.compress.min_chars, 600);
        assert_eq!(cfg.compress.max_output_tokens, 1024);
        assert_eq!(cfg.compress.timeout_secs, 15);
    }

    #[test]
    fn tuning_load_compress_override() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("intelligence.toml");
        std::fs::write(&path, "[compress]\nenabled = true\n").unwrap();
        let cfg = TuningConfig::load(&path);
        // Overridden field flips; the rest keep defaults.
        assert!(cfg.compress.enabled);
        assert_eq!(cfg.compress.min_chars, 600);
        assert_eq!(cfg.compress.max_output_tokens, 1024);
        assert_eq!(cfg.compress.timeout_secs, 15);
    }

    // T21 C1 — EvictionConfig defaults + serde-default deserialization.
    #[test]
    fn test_eviction_config_defaults() {
        let cfg = EvictionConfig::default();
        assert_eq!(cfg.min_age_days, 90);
        assert!((cfg.evict_confidence_floor - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.per_space_cap, None);
        assert!(cfg.recover_before_delete);
    }

    // T21 C1 — an absent config block must deserialize to the same defaults,
    // proving every field carries #[serde(default = ...)] (mirror
    // test_eval_config_defaults).
    #[test]
    fn test_eviction_config_empty_block_is_defaults() {
        // Mirror test_gate_config_defaults: an absent config block ("{}")
        // must deserialize to defaults, proving every field is #[serde(default)].
        let cfg: EvictionConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.min_age_days, 90);
        assert!((cfg.evict_confidence_floor - 0.1).abs() < f64::EPSILON);
        assert_eq!(cfg.per_space_cap, None);
        assert!(cfg.recover_before_delete);
    }
}
