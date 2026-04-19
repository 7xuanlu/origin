// SPDX-License-Identifier: AGPL-3.0-only
use crate::engine::LlmEngine;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;

use std::num::NonZeroU32;
use std::time::Instant;

/// Maximum output tokens for classify (summary + space + tags + stream_name ≈ 100 tokens).
const CLASSIFY_MAX_OUTPUT_TOKENS: i32 = 256;
/// Maximum input chars for classify (less text needed for classification).
const CLASSIFY_MAX_INPUT_CHARS: usize = 3000;
/// Context window size for classify.
const CLASSIFY_CTX_SIZE: u32 = 8192;
/// Timeout for a single classify inference call.
const INFERENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

#[derive(Debug, serde::Deserialize)]
pub struct ClassifyResult {
    pub summary: String,
    pub space: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub stream_name: Option<String>,
}

/// Lightweight classification: summary + space + tags + stream_name only.
/// No formatted_text generation — ~100 tokens output vs ~1000+ for format_ocr_text.
pub fn classify_content(
    engine: &LlmEngine,
    raw_text: &str,
    app_name: &str,
    window_title: Option<&str>,
    spaces: &[String],
    classify_screen_prompt: &str,
) -> Option<ClassifyResult> {
    let start = Instant::now();

    let truncated = truncate_at_word_boundary(raw_text, CLASSIFY_MAX_INPUT_CHARS);
    let window_title = window_title.unwrap_or("Unknown");
    let spaces_str = spaces.join(", ");

    let system_prompt = classify_screen_prompt.replace("{spaces_str}", &spaces_str);
    let prompt = format!(
        "<|im_start|>system\n\
         {system_prompt}\n\
         <|im_end|>\n\
         <|im_start|>user\n\
         App: {app_name}\n\
         Window: {window_title}\n\n\
         {truncated}\n\
         <|im_end|>\n\
         <|im_start|>assistant\n",
    );

    // Tokenize
    let tokens = match engine.model.str_to_token(&prompt, AddBos::Always) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("[llm_classifier] tokenization failed: {e}");
            return None;
        }
    };

    log::info!("[llm_classifier] prompt tokens={}", tokens.len());

    // Truncate if needed
    let max_prompt_tokens = CLASSIFY_CTX_SIZE as usize - CLASSIFY_MAX_OUTPUT_TOKENS as usize;
    let tokens = if tokens.len() > max_prompt_tokens {
        log::warn!(
            "[llm_classifier] prompt tokens ({}) exceed budget ({}), truncating",
            tokens.len(),
            max_prompt_tokens
        );
        tokens[..max_prompt_tokens].to_vec()
    } else {
        tokens
    };

    // Create context
    let n_batch = tokens.len().max(512) as u32;
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(CLASSIFY_CTX_SIZE).unwrap()))
        .with_n_batch(n_batch);

    let mut ctx = match engine.model.new_context(&engine.backend, ctx_params) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("[llm_classifier] context creation failed: {e}");
            return None;
        }
    };

    // Fill and decode prompt
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (i, token) in tokens.iter().enumerate() {
        if let Err(e) = batch.add(*token, i as i32, &[0], i == tokens.len() - 1) {
            log::warn!("[llm_classifier] batch add failed: {e}");
            return None;
        }
    }

    if let Err(e) = ctx.decode(&mut batch) {
        log::warn!("[llm_classifier] prompt decode failed: {e}");
        return None;
    }

    // Generate
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::temp(0.3), LlamaSampler::dist(42)]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();
    // n_cur is the prompt length (e.g. 991). We must limit GENERATED tokens,
    // not absolute position — otherwise the loop never executes when prompt > budget.
    let max_pos = n_cur + CLASSIFY_MAX_OUTPUT_TOKENS;

    while n_cur < max_pos {
        if start.elapsed() > INFERENCE_TIMEOUT {
            log::warn!("[llm_classifier] inference timeout");
            break;
        }

        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if engine.model.is_eog_token(token) {
            break;
        }

        match engine.model.token_to_piece(token, &mut decoder, true, None) {
            Ok(piece) => output.push_str(&piece),
            Err(e) => {
                log::warn!("[llm_classifier] token decode failed: {e}");
                break;
            }
        }

        batch.clear();
        if let Err(_e) = batch.add(token, n_cur, &[0], true) {
            break;
        }

        if let Err(_e) = ctx.decode(&mut batch) {
            break;
        }

        n_cur += 1;
    }

    log::info!(
        "[llm_classifier] generated {} chars in {:?}",
        output.len(),
        start.elapsed()
    );

    // Strip any residual <think> tags (safety net for Qwen3)
    let stripped = crate::llm_provider::strip_think_tags(&output);
    // Sanitize Unicode curly quotes that break JSON parsing
    let sanitized = crate::llm_provider::sanitize_json_quotes(&stripped);
    // Extract JSON from unconstrained output
    let json_str = crate::llm_provider::extract_json(&sanitized).unwrap_or(&sanitized);
    match serde_json::from_str::<ClassifyResult>(json_str) {
        Ok(result) => {
            log::info!(
                "[llm_classifier] classified: space={:?}, tags={:?}, stream={:?}",
                result.space,
                result.tags,
                result.stream_name
            );
            Some(result)
        }
        Err(e) => {
            log::warn!(
                "[llm_classifier] JSON parse failed: {e}, output: {}",
                &output[..output.floor_char_boundary(200)]
            );
            None
        }
    }
}

/// Truncate text at a word boundary, not exceeding `max_chars` bytes.
fn truncate_at_word_boundary(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        return text;
    }
    let safe_end = text.floor_char_boundary(max_chars);
    match text[..safe_end].rfind(' ') {
        Some(pos) => &text[..pos],
        None => &text[..safe_end],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_result_deserialize() {
        let json = r#"{"summary": "User was editing code", "space": "coding", "tags": ["rust", "editor"], "stream_name": "refactoring auth"}"#;
        let result: ClassifyResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.summary, "User was editing code");
        assert_eq!(result.space.as_deref(), Some("coding"));
        assert_eq!(result.tags, vec!["rust", "editor"]);
        assert_eq!(result.stream_name.as_deref(), Some("refactoring auth"));
    }

    #[test]
    fn test_classify_result_minimal() {
        let json = r#"{"summary": "User browsing web"}"#;
        let result: ClassifyResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.summary, "User browsing web");
        assert!(result.space.is_none());
        assert!(result.tags.is_empty());
        assert!(result.stream_name.is_none());
    }

    #[test]
    fn test_classify_result_from_raw_output() {
        let raw_output = r#"Here is the classification:
{"summary": "User was reading docs about Rust async.", "space": "coding", "tags": ["rust", "async", "docs"], "stream_name": "learning async"}"#;
        let json_str = crate::llm_provider::extract_json(raw_output).unwrap();
        let result: ClassifyResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.summary, "User was reading docs about Rust async.");
        assert_eq!(result.space.as_deref(), Some("coding"));
        assert_eq!(result.tags, vec!["rust", "async", "docs"]);
        assert_eq!(result.stream_name.as_deref(), Some("learning async"));
    }

    #[test]
    fn test_sanitize_curly_quotes() {
        let raw = "```json\n{\"summary\": \"User analyzes a \u{201C}ambient OS\u{201D} project.\", \"space\": \"coding\", \"tags\": [\"rust\"], \"stream_name\": \"dev\"}";
        let sanitized = crate::llm_provider::sanitize_json_quotes(raw);
        let json_str = crate::llm_provider::extract_json(&sanitized).unwrap();
        let result: ClassifyResult = serde_json::from_str(json_str).unwrap();
        assert!(result.summary.contains("ambient OS"));
    }
}
