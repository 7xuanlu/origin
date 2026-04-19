// SPDX-License-Identifier: Apache-2.0

//! Registry of supported on-device GGUF models.
//!
//! Two tiers, chosen by RAM:
//! - Qwen 3 4B (default) — runs on any laptop, ~3GB RAM
//! - Qwen 3.5 9B — requires 16GB+ RAM, better synthesis quality
//!
//! ## Token budget calibration (2026-04-18)
//!
//! These values are initial engineering estimates aligned with llama.cpp
//! community practice for Q4_K_M quantized models on Apple Silicon, not
//! benchmark-derived constants. Subject to revision as we collect empirical
//! data from real distillation runs.
//!
//! - `context_size`: KV cache allocation. Community norm is 4K-8K for sub-7B
//!   Q4 models, 8K-16K for 7B-13B. Constrained by Metal GPU memory, not
//!   model capability. DeltaNet (9B) handles longer contexts than standard
//!   Transformers at the same parameter count.
//! - `max_output_tokens`: quality-constrained, not context-constrained. Small
//!   quantized models lose coherence and start repeating past ~750-1000 words.
//!   9B sustains structured output (wiki format) up to ~1500 words.
//! - `synthesis_token_limit`: effective input capacity for synthesis. Set well
//!   below the trained context to account for quality degradation of quantized
//!   models at long contexts.

#[derive(Debug, Clone)]
pub struct OnDeviceModel {
    pub id: &'static str,
    pub display_name: &'static str,
    pub repo_id: &'static str,
    pub filename: &'static str,
    pub param_count: &'static str,
    pub ram_required_gb: f64,
    pub file_size_gb: f64,
    /// Max tokens this model can effectively synthesize (not just read).
    /// Research-calibrated: ~25% of context window for quantized models,
    /// accounting for quality degradation at longer contexts.
    pub synthesis_token_limit: usize,
    /// KV cache context window size for inference. Controls how many tokens
    /// (prompt + output) fit in one call. Larger = more GPU memory. Set per
    /// model to balance capacity vs memory pressure.
    pub context_size: u32,
    /// Recommended max output tokens for body-generation tasks (distillation,
    /// re-distillation). Smaller models produce thin/repetitive output at
    /// high token counts, so this is quality-constrained, not context-constrained.
    pub max_output_tokens: u32,
}

/// All available on-device models, ordered by size.
pub const MODELS: &[OnDeviceModel] = &[
    OnDeviceModel {
        id: "qwen3-4b",
        display_name: "Qwen 3 4B",
        repo_id: "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename: "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        param_count: "4B",
        ram_required_gb: 3.0,
        file_size_gb: 2.7,
        synthesis_token_limit: 6_000, // 32K context, ~19% — conservative for 4B quantized
        context_size: 8_192,          // 25% of 32K. KV cache ~2GB on Metal.
        max_output_tokens: 1_024,     // 4B quality degrades past ~750 words
    },
    OnDeviceModel {
        id: "qwen3.5-9b",
        display_name: "Qwen 3.5 9B",
        repo_id: "unsloth/Qwen3.5-9B-GGUF",
        filename: "Qwen3.5-9B-Q4_K_M.gguf",
        param_count: "9B",
        ram_required_gb: 6.0,
        file_size_gb: 5.5,
        synthesis_token_limit: 16_000, // 128K context, ~12% — DeltaNet arch helps with long ctx
        context_size: 8_192,           // Practical: distillation prompts are 2-4K tokens. 8K gives
        // 6K prompt + 2K output. Same as 4B. Avoids the 4x KV cache
        // overhead of 16K which slows prefill without benefit.
        max_output_tokens: 2_048, // 9B sustains quality up to ~1500 words
    },
];

/// Get model by ID. Returns `None` if the ID is unknown; callers should
/// fall back to `get_default_model()`.
pub fn get_model(id: &str) -> Option<&'static OnDeviceModel> {
    MODELS.iter().find(|m| m.id == id)
}

/// Get the default model (smallest tier, runs on any laptop).
pub fn get_default_model() -> &'static OnDeviceModel {
    &MODELS[0]
}

/// Resolve a possibly-unknown model id to a concrete model, falling back to
/// the default when the id is missing or not in the registry. Used on startup
/// so an outdated `config.on_device_model` value doesn't break initialization.
pub fn resolve_or_default(id: Option<&str>) -> &'static OnDeviceModel {
    match id.and_then(get_model) {
        Some(m) => m,
        None => {
            if let Some(unknown) = id {
                log::warn!(
                    "[on_device_models] unknown model id {:?}, falling back to default {}",
                    unknown,
                    get_default_model().id
                );
            }
            get_default_model()
        }
    }
}

/// Recommend a model based on total system RAM.
pub fn recommend_for_ram(total_ram_gb: f64) -> &'static OnDeviceModel {
    if total_ram_gb >= 16.0 {
        &MODELS[1] // qwen3.5-9b
    } else {
        &MODELS[0] // qwen3-4b
    }
}

/// Check whether a model is already downloaded to the hf-hub cache.
/// Does not trigger a download. Returns false if the cache can't be located
/// or the file doesn't exist.
pub fn is_cached(model: &OnDeviceModel) -> bool {
    // hf-hub caches at ~/.cache/huggingface/hub/models--{repo_with_slashes_dashified}/
    // snapshots/{hash}/{filename}
    // We scan all snapshot directories for the expected filename.
    let Some(home) = dirs::home_dir() else {
        return false;
    };
    let repo_dir = model.repo_id.replace('/', "--");
    let snapshots = home
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(format!("models--{}", repo_dir))
        .join("snapshots");
    let Ok(entries) = std::fs::read_dir(&snapshots) else {
        return false;
    };
    for entry in entries.flatten() {
        let candidate = entry.path().join(model.filename);
        if candidate.exists() {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_two_models() {
        assert_eq!(MODELS.len(), 2);
        assert_eq!(MODELS[0].id, "qwen3-4b");
        assert_eq!(MODELS[1].id, "qwen3.5-9b");
    }

    #[test]
    fn get_model_by_id() {
        assert!(get_model("qwen3-4b").is_some());
        assert!(get_model("qwen3.5-9b").is_some());
        assert!(get_model("nonexistent").is_none());
    }

    #[test]
    fn resolve_or_default_handles_unknown() {
        // Unknown id → default
        assert_eq!(resolve_or_default(Some("qwen3.5-4b")).id, "qwen3-4b");
        assert_eq!(resolve_or_default(Some("nonexistent")).id, "qwen3-4b");
        // None → default
        assert_eq!(resolve_or_default(None).id, "qwen3-4b");
        // Known id → that id
        assert_eq!(resolve_or_default(Some("qwen3.5-9b")).id, "qwen3.5-9b");
    }

    #[test]
    fn recommendation_by_ram() {
        assert_eq!(recommend_for_ram(8.0).id, "qwen3-4b");
        assert_eq!(recommend_for_ram(16.0).id, "qwen3.5-9b");
        assert_eq!(recommend_for_ram(32.0).id, "qwen3.5-9b");
    }

    #[test]
    fn default_is_qwen3_4b() {
        assert_eq!(get_default_model().id, "qwen3-4b");
    }
}
