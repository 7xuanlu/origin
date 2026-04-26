// SPDX-License-Identifier: AGPL-3.0-only
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
fn d_600() -> i64 {
    600
}
fn d_7200() -> i64 {
    7200
}
fn d_300() -> u64 {
    300
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
fn d_13_f32() -> f32 {
    1.3
}
fn d_30_i64() -> i64 {
    30
}
fn d_168_u64() -> u64 {
    168
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TuningConfig {
    pub router: RouterConfig,
    pub scoring: ScoringConfig,
    pub refinery: RefineryConfig,
    pub narrative: NarrativeConfig,
    pub briefing: BriefingConfig,
    pub confidence: ConfidenceConfig,
    pub packager: PackagerConfig,
    pub eval: EvalConfig,
    pub search_scoring: SearchScoringConfig,
    #[serde(default)]
    pub distillation: DistillationConfig,
    #[serde(default)]
    pub gate: GateConfig,
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

fn d_070_topic() -> f64 {
    0.70
}
fn d_080_topic() -> f64 {
    0.80
}
fn d_090_topic() -> f64 {
    0.90
}
fn d_20_topic() -> usize {
    20
}
fn d_50_changelog() -> usize {
    50
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TopicMatchConfig {
    /// Embedding threshold when domain + type both match (high confidence).
    #[serde(default = "d_070_topic")]
    pub threshold_exact: f64,
    /// Embedding threshold when only domain OR type matches (partial context).
    #[serde(default = "d_080_topic")]
    pub threshold_partial: f64,
    /// Embedding threshold when neither domain nor type matches (semantic only).
    #[serde(default = "d_090_topic")]
    pub threshold_none: f64,
    /// Maximum number of candidate memories to consider.
    #[serde(default = "d_20_topic")]
    pub max_candidates: usize,
    /// Maximum changelog entries to retain before trimming oldest.
    #[serde(default = "d_50_changelog")]
    pub changelog_cap: usize,
}

impl Default for TopicMatchConfig {
    fn default() -> Self {
        Self {
            threshold_exact: 0.70,
            threshold_partial: 0.80,
            threshold_none: 0.90,
            max_candidates: 20,
            changelog_cap: 50,
        }
    }
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
    #[serde(default = "d_085")]
    pub dedup_similarity_threshold: f64,
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
    #[serde(default = "d_3_usize")]
    pub max_enrichment_retries: usize,
    #[serde(default)]
    pub topic_match: TopicMatchConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DistillationConfig {
    #[serde(default = "d_073")]
    pub similarity_threshold: f64,
    #[serde(default = "d_2_usize")]
    pub min_cluster_size: usize,
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
    #[serde(default = "d_3_usize")]
    pub concept_min_cluster_size: usize,
    #[serde(default = "d_075")]
    pub concept_growth_threshold: f64,
    /// Reserved for future integration into search_memory scoring pipeline.
    /// Currently concepts are searched via separate search_concepts endpoint.
    #[serde(default = "d_13_f32")]
    pub concept_boost: f32,
    /// Hard cap on size of unlinked-memory clusters (no entity_id, no domain).
    /// Clusters above this size are skipped during distillation. Safety valve
    /// for the runaway-cluster failure mode (see spec 2026-04-25).
    #[serde(default = "d_50_usize")]
    pub max_unlinked_cluster_size: usize,
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

#[derive(Debug, Clone, Deserialize)]
pub struct PackagerConfig {
    #[serde(default = "d_600")]
    pub session_gap_secs: i64,
    #[serde(default = "d_3_usize")]
    pub min_session_captures: usize,
    #[serde(default = "d_7200")]
    pub max_session_duration_secs: i64,
    #[serde(default = "d_300")]
    pub packager_interval_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalConfig {
    #[serde(default = "d_true")]
    pub enabled: bool,
    #[serde(default = "d_true")]
    pub signal_capture: bool,
    #[serde(default = "d_false")]
    pub auto_tune_enabled: bool,
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
            dedup_similarity_threshold: d_085(),
            entity_link_distance: d_015_f64(),
            consolidation_confidence_threshold: d_03(),
            consolidation_batch_size: d_10_usize(),
            batch_window_secs: d_30_i64(),
            kg_rethink_interval_hours: d_168_u64(),
            entity_backfill_batch_size: d_5_usize(),
            max_enrichment_retries: d_3_usize(),
            topic_match: TopicMatchConfig::default(),
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
impl Default for PackagerConfig {
    fn default() -> Self {
        Self {
            session_gap_secs: d_600(),
            min_session_captures: d_3_usize(),
            max_session_duration_secs: d_7200(),
            packager_interval_secs: d_300(),
        }
    }
}
impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            enabled: d_true(),
            signal_capture: d_true(),
            auto_tune_enabled: d_false(),
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
        }
    }
}
impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: d_073(),
            min_cluster_size: d_2_usize(),
            ondevice_token_limit: d_8000_usize(),
            api_token_limit: d_50000_usize(),
            max_retries: d_3_usize(),
            max_clusters_per_steep: d_20_usize(),
            concept_min_cluster_size: d_3_usize(),
            concept_growth_threshold: d_075(),
            concept_boost: d_13_f32(),
            max_unlinked_cluster_size: d_50_usize(),
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
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("origin")
            .join("intelligence.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(cfg.refinery.dedup_similarity_threshold, 0.85);
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
        // Packager
        assert_eq!(cfg.packager.session_gap_secs, 600);
        assert_eq!(cfg.packager.min_session_captures, 3);
        assert_eq!(cfg.packager.max_session_duration_secs, 7200);
        assert_eq!(cfg.packager.packager_interval_secs, 300);
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
        assert!(!cfg.eval.auto_tune_enabled);
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
        assert_eq!(cfg.concept_min_cluster_size, 3);
        assert!((cfg.concept_growth_threshold - 0.75).abs() < 0.01);
        assert!((cfg.concept_boost - 1.3).abs() < 0.01);
    }

    #[test]
    fn distillation_config_default_has_unlinked_cluster_cap() {
        let cfg = DistillationConfig::default();
        assert_eq!(cfg.max_unlinked_cluster_size, 50);
    }
}
