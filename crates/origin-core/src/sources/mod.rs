// SPDX-License-Identifier: Apache-2.0
//! Source connectors and memory type helpers.
//!
//! The core memory types (`MemoryType`, `RawDocument`, `SourceType`,
//! `StabilityTier`, `SyncStatus`) live in `origin-types` and are re-exported
//! here so intra-crate imports keep working via `crate::sources::*`.
//!
//! This module owns helpers that need `tuning::ConfidenceConfig` (confidence
//! computation, decay rates) and the `DataSource` trait plus supporting
//! structs used by connectors.
pub mod local_files;
pub mod obsidian;

use crate::error::OriginError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::path::PathBuf;

// Re-export canonical type definitions from origin-types. This keeps
// `crate::sources::RawDocument` working for origin-core modules while the
// authoritative definition lives in the shared types crate.
pub use origin_types::sources::{
    stability_tier, MemoryType, RawDocument, SourceType, StabilityTier, SyncStatus,
};

/// Return the confidence ceiling for a stability tier.
pub fn base_confidence(tier: &StabilityTier, cfg: &crate::tuning::ConfidenceConfig) -> f32 {
    match tier {
        StabilityTier::Protected => cfg.protected_base,
        StabilityTier::Standard => cfg.standard_base,
        StabilityTier::Ephemeral => cfg.ephemeral_base,
    }
}

/// Return the trust weighting for a given agent trust level.
pub fn trust_weight(trust_level: &str, cfg: &crate::tuning::ConfidenceConfig) -> f32 {
    match trust_level {
        "full" => cfg.full_trust_weight,
        "review" => cfg.review_trust_weight,
        _ => cfg.untrusted_weight,
    }
}

/// Return the decay rate for a stability tier.
pub fn decay_rate(tier: &StabilityTier, cfg: &crate::tuning::ConfidenceConfig) -> f64 {
    match tier {
        StabilityTier::Protected => cfg.protected_decay,
        StabilityTier::Standard => cfg.standard_decay,
        StabilityTier::Ephemeral => cfg.ephemeral_decay,
    }
}

/// Compute effective confidence from agent-supplied value, memory type, trust level, and quality.
/// If agent provides a value, it's clamped to the max allowed by tier * trust.
/// If agent provides no value, uses the max as default.
/// If quality is "low", the result is multiplied by low_quality_multiplier (default 0.7).
pub fn compute_effective_confidence(
    agent_confidence: Option<f32>,
    memory_type: Option<&str>,
    trust_level: &str,
    quality: Option<&str>,
    cfg: &crate::tuning::ConfidenceConfig,
) -> f32 {
    let tier = stability_tier(memory_type);
    let max_confidence = base_confidence(&tier, cfg) * trust_weight(trust_level, cfg);
    let base = match agent_confidence {
        Some(val) => val.min(max_confidence),
        None => max_confidence,
    };
    if quality == Some("low") {
        base * cfg.low_quality_multiplier
    } else {
        base
    }
}

/// Status of a connected source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStatus {
    pub name: String,
    pub connected: bool,
    pub requires_auth: bool,
    pub last_sync: Option<i64>,
    pub document_count: u64,
    pub error: Option<String>,
}

/// Persisted source configuration — stored in config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub source_type: SourceType,
    pub path: PathBuf,
    #[serde(default = "default_sync_status")]
    pub status: SyncStatus,
    pub last_sync: Option<i64>,
    #[serde(default)]
    pub file_count: u64,
    #[serde(default)]
    pub memory_count: u64,
    /// Number of files that failed to read / ingest in the last sync.
    #[serde(default)]
    pub last_sync_errors: u64,
    /// Categorized detail of last sync errors for UI display.
    /// Known values: "google_drive_offline", "file_read_errors".
    #[serde(default)]
    pub last_sync_error_detail: Option<String>,
}

fn default_sync_status() -> SyncStatus {
    SyncStatus::Active
}

/// Trait that all data source connectors must implement.
#[async_trait]
pub trait DataSource: Send + Sync {
    /// Unique name for this source ("gmail", "notion", etc.)
    fn name(&self) -> &str;

    /// Whether this source requires OAuth authentication
    fn requires_auth(&self) -> bool;

    /// Check if the source is currently connected/authenticated
    async fn is_connected(&self) -> bool;

    /// Connect/authenticate the source (triggers OAuth if needed)
    async fn connect(&mut self) -> Result<(), OriginError>;

    /// Disconnect the source (revoke tokens, cleanup)
    async fn disconnect(&mut self) -> Result<(), OriginError>;

    /// Fetch new/updated content since last sync
    async fn fetch_updates(&mut self) -> Result<Vec<RawDocument>, OriginError>;

    /// Initial full sync - fetches all available content
    async fn full_sync(&mut self) -> Result<Vec<RawDocument>, OriginError>;

    /// Get the current status of this source
    async fn status(&self) -> SourceStatus;

    /// Downcast to concrete type
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type_from_str_valid() {
        assert_eq!(
            "identity".parse::<MemoryType>().unwrap(),
            MemoryType::Identity
        );
        assert_eq!(
            "preference".parse::<MemoryType>().unwrap(),
            MemoryType::Preference
        );
        assert_eq!(
            "decision".parse::<MemoryType>().unwrap(),
            MemoryType::Decision
        );
        assert_eq!("fact".parse::<MemoryType>().unwrap(), MemoryType::Fact);
        assert_eq!("goal".parse::<MemoryType>().unwrap(), MemoryType::Goal);
        // Case insensitive
        assert_eq!(
            "IDENTITY".parse::<MemoryType>().unwrap(),
            MemoryType::Identity
        );
        assert_eq!("Fact".parse::<MemoryType>().unwrap(), MemoryType::Fact);
    }

    #[test]
    fn test_stability_tier_protected() {
        assert!(matches!(
            stability_tier(Some("identity")),
            StabilityTier::Protected
        ));
        assert!(matches!(
            stability_tier(Some("preference")),
            StabilityTier::Protected
        ));
    }

    #[test]
    fn test_stability_tier_standard() {
        assert!(matches!(
            stability_tier(Some("fact")),
            StabilityTier::Standard
        ));
        assert!(matches!(
            stability_tier(Some("decision")),
            StabilityTier::Standard
        ));
    }

    #[test]
    fn test_stability_tier_ephemeral() {
        assert!(matches!(
            stability_tier(Some("goal")),
            StabilityTier::Ephemeral
        ));
        assert!(matches!(stability_tier(None), StabilityTier::Ephemeral));
        assert!(matches!(
            stability_tier(Some("unknown_junk")),
            StabilityTier::Ephemeral
        ));
    }

    #[test]
    fn test_base_confidence_values() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        assert_eq!(base_confidence(&StabilityTier::Protected, &cfg), 0.90);
        assert_eq!(base_confidence(&StabilityTier::Standard, &cfg), 0.70);
        assert_eq!(base_confidence(&StabilityTier::Ephemeral, &cfg), 0.50);
    }

    #[test]
    fn test_trust_weight_values() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        assert_eq!(trust_weight("full", &cfg), 1.0);
        assert_eq!(trust_weight("review", &cfg), 0.7);
        assert_eq!(trust_weight("untrusted", &cfg), 0.4);
        assert_eq!(trust_weight("something_else", &cfg), 0.4);
    }

    #[test]
    fn test_compute_effective_confidence() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        // Full trust + Protected = 0.90
        assert!(
            (compute_effective_confidence(None, Some("identity"), "full", None, &cfg) - 0.90).abs()
                < 0.001
        );
        // Review trust + Protected = 0.63
        assert!(
            (compute_effective_confidence(None, Some("identity"), "review", None, &cfg) - 0.63)
                .abs()
                < 0.001
        );
        // Agent provides lower value — kept
        assert!(
            (compute_effective_confidence(Some(0.5), Some("identity"), "full", None, &cfg) - 0.5)
                .abs()
                < 0.001
        );
        // Agent provides higher value — clamped
        assert!(
            (compute_effective_confidence(Some(0.99), Some("fact"), "review", None, &cfg) - 0.49)
                .abs()
                < 0.001
        );
        // NULL memory_type → Ephemeral base
        assert!(
            (compute_effective_confidence(None, None, "full", None, &cfg) - 0.50).abs() < 0.001
        );
    }

    #[test]
    fn test_compute_effective_confidence_quality_low() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        // quality = "low" multiplies result by 0.7
        // Full trust + Protected = 0.90 * 0.7 = 0.63
        assert!(
            (compute_effective_confidence(None, Some("identity"), "full", Some("low"), &cfg)
                - 0.63)
                .abs()
                < 0.001
        );
        // Full trust + Standard = 0.70 * 0.7 = 0.49
        assert!(
            (compute_effective_confidence(None, Some("fact"), "full", Some("low"), &cfg) - 0.49)
                .abs()
                < 0.001
        );
    }

    #[test]
    fn test_compute_effective_confidence_quality_non_low() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        // quality = "medium" or "high" should NOT reduce confidence
        let base = compute_effective_confidence(None, Some("fact"), "full", None, &cfg);
        let medium = compute_effective_confidence(None, Some("fact"), "full", Some("medium"), &cfg);
        let high = compute_effective_confidence(None, Some("fact"), "full", Some("high"), &cfg);
        assert!((base - medium).abs() < 0.001);
        assert!((base - high).abs() < 0.001);
    }

    #[test]
    fn test_decay_rate_values() {
        let cfg = crate::tuning::ConfidenceConfig::default();
        assert!((decay_rate(&StabilityTier::Protected, &cfg) - 0.001).abs() < 0.0001);
        assert!((decay_rate(&StabilityTier::Standard, &cfg) - 0.01).abs() < 0.001);
        assert!((decay_rate(&StabilityTier::Ephemeral, &cfg) - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_source_roundtrip() {
        let source = Source {
            id: "obsidian-main".to_string(),
            source_type: SourceType::Obsidian,
            path: PathBuf::from("/Users/x/vault"),
            status: SyncStatus::Active,
            last_sync: Some(1712678400),
            file_count: 42,
            memory_count: 128,
            last_sync_errors: 0,
            last_sync_error_detail: None,
        };
        let json = serde_json::to_string(&source).unwrap();
        let back: Source = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "obsidian-main");
        assert_eq!(back.source_type, SourceType::Obsidian);
        assert_eq!(back.file_count, 42);
    }

    #[test]
    fn test_source_type_display() {
        assert_eq!(SourceType::Obsidian.as_str(), "obsidian");
        assert_eq!(SourceType::Directory.as_str(), "directory");
    }

    #[test]
    fn test_sync_status_variants() {
        let active = SyncStatus::Active;
        let err = SyncStatus::Error("bad path".to_string());
        let json_err = serde_json::to_string(&err).unwrap();
        assert!(json_err.contains("bad path"));
        let json_active = serde_json::to_string(&active).unwrap();
        assert!(json_active.contains("Active"));
    }
}
