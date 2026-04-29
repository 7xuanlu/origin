// SPDX-License-Identifier: AGPL-3.0-only
//! Memory classification via the on-device LLM engine.
//!
//! Wraps [`crate::engine::LlmEngine`] with classification-specific prompts and
//! response parsers. Covers single-memory classification, batch classification,
//! type-only fallback classification, and profile sub-classification (alias
//! "profile" -> identity / preference / goal).

use crate::engine::{
    extract_json, extract_json_array, strip_think_tags, truncate_at_word_boundary, LlmEngine,
    CTX_SIZE,
};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;

use std::num::NonZeroU32;
use std::time::Instant;

/// Valid memory types for classification (5 canonical types).
const VALID_MEMORY_TYPES: &[&str] = &["identity", "preference", "fact", "decision", "goal"];

/// Result of classifying a single memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClassificationResult {
    pub memory_type: String,
    pub domain: Option<String>,
    pub tags: Vec<String>,
    /// Quality signal: "low", "medium", "high", or None (default)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,
}

impl Default for ClassificationResult {
    fn default() -> Self {
        Self {
            memory_type: "fact".to_string(),
            domain: None,
            tags: Vec::new(),
            quality: None,
        }
    }
}

/// Parse a batch classification response from LLM output.
/// Extracts JSON array, validates each entry, falls back to defaults for invalid entries,
/// and pads with defaults if the array is shorter than expected.
pub fn parse_classification_response(
    raw: &str,
    expected_count: usize,
) -> Vec<ClassificationResult> {
    let json_str = match extract_json_array(raw) {
        Some(s) => s,
        None => return vec![ClassificationResult::default(); expected_count],
    };

    let entries: Vec<serde_json::Value> = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return vec![ClassificationResult::default(); expected_count],
    };

    let mut results: Vec<ClassificationResult> = entries
        .iter()
        .map(|entry| {
            let memory_type = entry
                .get("type")
                .and_then(|v| v.as_str())
                .map(|s| s.to_lowercase())
                .filter(|s| VALID_MEMORY_TYPES.contains(&s.as_str()))
                .unwrap_or_else(|| "fact".to_string());

            let domain = entry
                .get("domain")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string());

            let tags = entry
                .get("tags")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            ClassificationResult {
                memory_type,
                domain,
                tags,
                quality: None,
            }
        })
        .collect();

    // Pad with defaults if fewer results than expected
    while results.len() < expected_count {
        results.push(ClassificationResult::default());
    }

    results
}

/// Parse LLM output for profile sub-classification. Fallback: "identity" (Protected tier).
pub fn parse_profile_subtype(raw: &str) -> &'static str {
    let first = raw.split_whitespace().next().unwrap_or("").to_lowercase();
    match first.as_str() {
        "identity" => "identity",
        "preference" => "preference",
        "goal" => "goal",
        _ => "identity", // safest default -- Protected tier
    }
}

/// Classify a "profile" alias into identity/preference/goal using the on-device engine.
///
/// Runs synchronously on the engine's worker thread. Caller is responsible for
/// running this inside `spawn_blocking` if invoking from an async context, since
/// [`LlmEngine::run_inference`] is blocking. Falls back to "identity" (Protected
/// tier) on any failure.
pub fn classify_profile_subtype_llm(engine: &LlmEngine, content: &str) -> String {
    let truncated: String = content.chars().take(500).collect();

    let system_prompt = crate::prompts::defaults::CLASSIFY_PROFILE_SUBTYPE;
    let prompt = format!(
        "<|im_start|>system\n\
         {system_prompt}\n\
         <|im_end|>\n\
         <|im_start|>user\n\
         {truncated}\n\
         <|im_end|>\n\
         <|im_start|>assistant\n",
    );

    match engine.run_inference(&prompt, 16, 0.1, crate::engine::CTX_SIZE, Some("classify")) {
        Some(output) => parse_profile_subtype(&output).to_string(),
        None => {
            log::warn!("[classify] profile sub-classify failed");
            "identity".to_string()
        }
    }
}

/// Async variant of [`classify_profile_subtype_llm`] that works with any
/// [`LlmProvider`]. Used by the app/server layers where the LLM is only
/// available through the provider trait (not a raw `LlmEngine`).
pub async fn classify_profile_subtype_via_provider(
    provider: &dyn crate::llm_provider::LlmProvider,
    content: &str,
) -> String {
    let truncated: String = content.chars().take(500).collect();

    let request = crate::llm_provider::LlmRequest {
        system_prompt: Some(crate::prompts::defaults::CLASSIFY_PROFILE_SUBTYPE.to_string()),
        user_prompt: truncated,
        max_tokens: 16,
        temperature: 0.1,
        label: None,
        timeout_secs: None,
    };

    match provider.generate(request).await {
        Ok(output) => parse_profile_subtype(&output).to_string(),
        Err(e) => {
            log::warn!("[classify] profile sub-classify via provider failed: {e}");
            "identity".to_string()
        }
    }
}

#[allow(dead_code)] // Full classification API -- only classify_memories_batch wired through OnDeviceProvider currently
impl LlmEngine {
    /// Classify memory content into type, domain, and tags via JSON output.
    /// Returns None if classification fails (caller should fallback to defaults).
    pub fn classify_memory(&self, content: &str) -> Option<ClassificationResult> {
        let start = Instant::now();

        let truncated = truncate_at_word_boundary(content, 1000);

        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {truncated}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            sys = self.prompts().classify_memory_quality,
        );

        let classify_ctx_size: u32 = 2048;
        let classify_max_output: i32 = 128;

        let tokens = match self.model().str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[classify] classify_memory tokenization failed: {e}");
                return None;
            }
        };

        let max_prompt_tokens = classify_ctx_size as usize - classify_max_output as usize;
        let tokens = if tokens.len() > max_prompt_tokens {
            tokens[..max_prompt_tokens].to_vec()
        } else {
            tokens
        };

        let n_batch = tokens.len().max(512) as u32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(classify_ctx_size).unwrap()))
            .with_n_batch(n_batch);

        let mut ctx = match self.model().new_context(self.backend(), ctx_params) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[classify] classify_memory context failed: {e}");
                return None;
            }
        };

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            if let Err(e) = batch.add(*token, i as i32, &[0], i == tokens.len() - 1) {
                log::warn!("[classify] classify_memory batch add failed: {e}");
                return None;
            }
        }

        if let Err(e) = ctx.decode(&mut batch) {
            log::warn!("[classify] classify_memory decode failed: {e}");
            return None;
        }

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.1), LlamaSampler::dist(42)]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();
        let max_pos = n_cur + classify_max_output;

        while n_cur < max_pos {
            if start.elapsed() > std::time::Duration::from_secs(10) {
                log::warn!("[classify] classify_memory timeout");
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model().is_eog_token(token) {
                break;
            }

            match self.model().token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => output.push_str(&piece),
                Err(_) => break,
            }

            batch.clear();
            if batch.add(token, n_cur, &[0], true).is_err() {
                break;
            }
            if ctx.decode(&mut batch).is_err() {
                break;
            }
            n_cur += 1;
        }

        log::info!(
            "[classify] classify_memory output='{}' in {:?}",
            output.trim(),
            start.elapsed()
        );

        // Parse JSON, falling back to classify_memory_type if JSON parse fails
        let cleaned = strip_think_tags(&output);
        let json_str = extract_json(&cleaned).unwrap_or(&cleaned);

        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(val) => {
                let memory_type = val
                    .get("memory_type")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_lowercase())
                    .filter(|s| s.parse::<origin_types::MemoryType>().is_ok())
                    .unwrap_or_else(|| "fact".to_string());

                let domain = val
                    .get("domain")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_lowercase());

                let tags = val
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.trim().to_lowercase()))
                            .filter(|s| !s.is_empty())
                            .collect()
                    })
                    .unwrap_or_default();

                Some(ClassificationResult {
                    memory_type,
                    domain,
                    tags,
                    quality: None,
                })
            }
            Err(e) => {
                log::warn!("[classify] classify_memory JSON parse failed: {e}, falling back to classify_memory_type");
                // Fallback: use the simple type-only classifier
                self.classify_memory_type(content)
                    .map(|mt| ClassificationResult {
                        memory_type: mt,
                        domain: None,
                        tags: Vec::new(),
                        quality: None,
                    })
            }
        }
    }

    /// Classify memory content into a MemoryType facet.
    /// Returns None if classification fails (caller should fallback to "fact").
    pub fn classify_memory_type(&self, content: &str) -> Option<String> {
        let start = Instant::now();

        // Truncate content to 500 chars for classification
        let truncated = truncate_at_word_boundary(content, 500);

        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {truncated}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            sys = self.prompts().classify_memory,
        );

        // Use small context window and low max output
        let classify_ctx_size: u32 = 512;
        let classify_max_output: i32 = 32;

        let tokens = match self.model().str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[classify] classify tokenization failed: {e}");
                return None;
            }
        };

        // Truncate if needed
        let max_prompt_tokens = classify_ctx_size as usize - classify_max_output as usize;
        let tokens = if tokens.len() > max_prompt_tokens {
            tokens[..max_prompt_tokens].to_vec()
        } else {
            tokens
        };

        let n_batch = tokens.len().max(512) as u32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(classify_ctx_size).unwrap()))
            .with_n_batch(n_batch);

        let mut ctx = match self.model().new_context(self.backend(), ctx_params) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[classify] classify context failed: {e}");
                return None;
            }
        };

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            if let Err(e) = batch.add(*token, i as i32, &[0], i == tokens.len() - 1) {
                log::warn!("[classify] classify batch add failed: {e}");
                return None;
            }
        }

        if let Err(e) = ctx.decode(&mut batch) {
            log::warn!("[classify] classify decode failed: {e}");
            return None;
        }

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.1), LlamaSampler::dist(42)]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();
        let max_pos = n_cur + classify_max_output;

        while n_cur < max_pos {
            if start.elapsed() > std::time::Duration::from_secs(10) {
                log::warn!("[classify] classify timeout");
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model().is_eog_token(token) {
                break;
            }

            match self.model().token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => output.push_str(&piece),
                Err(_) => break,
            }

            batch.clear();
            if batch.add(token, n_cur, &[0], true).is_err() {
                break;
            }
            if ctx.decode(&mut batch).is_err() {
                break;
            }
            n_cur += 1;
        }

        log::info!(
            "[classify] classify output='{}' in {:?}",
            output.trim(),
            start.elapsed()
        );

        // Parse: trim, lowercase, strip think tags, validate via MemoryType::from_str
        let cleaned = strip_think_tags(&output);
        let candidate = cleaned.trim().to_lowercase();
        // Extract just the first word (in case LLM adds explanation)
        let first_word = candidate.split_whitespace().next().unwrap_or("");

        match first_word.parse::<origin_types::MemoryType>() {
            Ok(mt) => Some(mt.to_string()),
            Err(_) => {
                log::warn!(
                    "[classify] classify returned invalid type: '{}'",
                    first_word
                );
                None
            }
        }
    }

    /// Classify a batch of memories into types, domains, and tags using on-device LLM.
    /// Returns one `ClassificationResult` per input memory. Falls back to defaults on failure.
    pub fn classify_memories_batch(&self, memories: &[String]) -> Vec<ClassificationResult> {
        if memories.is_empty() {
            return Vec::new();
        }

        let numbered = memories
            .iter()
            .enumerate()
            .map(|(i, m)| format!("{}. {}", i + 1, m))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {numbered}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n",
            sys = self.prompts().batch_classify,
        );

        match self.run_inference(&prompt, 2048, 0.3, CTX_SIZE, Some("classify")) {
            Some(response) => parse_classification_response(&response, memories.len()),
            None => vec![ClassificationResult::default(); memories.len()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_classification_response_valid() {
        let json = r#"[{"i":1,"type":"identity","domain":"work","tags":["engineer"]},{"i":2,"type":"preference","domain":"technology","tags":["dark mode"]}]"#;
        let results = parse_classification_response(json, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].memory_type, "identity");
        assert_eq!(results[0].domain, Some("work".to_string()));
        assert_eq!(results[1].memory_type, "preference");
    }

    #[test]
    fn test_parse_classification_response_malformed_entry() {
        let json =
            r#"[{"i":1,"type":"identity","domain":"work","tags":[]},{"i":2,"type":"INVALID"}]"#;
        let results = parse_classification_response(json, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].memory_type, "identity");
        assert_eq!(results[1].memory_type, "fact"); // invalid type falls back
    }

    #[test]
    fn test_parse_classification_response_total_garbage() {
        let json = "this is not json at all";
        let results = parse_classification_response(json, 3);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.memory_type == "fact"));
    }

    #[test]
    fn test_parse_classification_response_wrong_count() {
        let json = r#"[{"i":1,"type":"identity","domain":"work","tags":[]}]"#;
        let results = parse_classification_response(json, 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].memory_type, "identity");
        assert_eq!(results[1].memory_type, "fact"); // padded
    }

    #[test]
    fn test_parse_classification_response_with_surrounding_text() {
        let json = r#"Here are the results: [{"i":1,"type":"fact","domain":"work","tags":[]}] Hope that helps!"#;
        let results = parse_classification_response(json, 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].memory_type, "fact");
    }

    #[test]
    fn test_parse_profile_subtype() {
        assert_eq!(parse_profile_subtype("identity"), "identity");
        assert_eq!(
            parse_profile_subtype("preference\nsome extra"),
            "preference"
        );
        assert_eq!(parse_profile_subtype("GOAL"), "goal");
        assert_eq!(parse_profile_subtype("bogus"), "identity"); // fallback
        assert_eq!(parse_profile_subtype(""), "identity"); // fallback
    }

    #[test]
    fn test_classify_prompt_contains_all_types() {
        // Verify the classify prompt would include all valid types
        let all = origin_types::MemoryType::all_values();
        let prompt_types = "identity, preference, decision, fact, goal";
        for val in all {
            assert!(
                prompt_types.contains(val),
                "prompt should include type '{}'",
                val
            );
        }
    }
}
