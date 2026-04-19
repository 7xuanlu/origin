// SPDX-License-Identifier: AGPL-3.0-only
//! LLM-based reranking of retrieval candidates.
//!
//! Given a query and a list of `(id, content)` candidates, asks the on-device
//! engine to score each candidate 0.0 - 1.0 and returns the candidates sorted
//! by descending score. Falls back to an empty vec on any failure (callers are
//! expected to keep the original ordering in that case).

use crate::engine::{strip_think_tags, truncate_at_word_boundary, LlmEngine};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::sampling::LlamaSampler;

use std::num::NonZeroU32;
use std::time::Instant;

#[allow(dead_code)] // Wired via search.rs reranked variant
impl LlmEngine {
    /// Rerank search results by query relevance using on-device LLM.
    /// Returns (id, relevance_score) pairs sorted by relevance descending.
    /// Falls back to empty vec on failure (caller should use original order).
    pub fn rerank_results(
        &self,
        query: &str,
        candidates: &[(String, String)], // (id, content) pairs
    ) -> Vec<(String, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let start = Instant::now();
        let n = candidates.len();

        // Build numbered candidate list, truncating each to ~200 chars
        let mut candidate_list = String::new();
        for (i, (_, content)) in candidates.iter().enumerate() {
            let truncated = truncate_at_word_boundary(content, 200);
            candidate_list.push_str(&format!("{}. {}\n", i + 1, truncated));
        }

        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             Query: {query}\n\n\
             {candidate_list}\
             <|im_end|>\n\
             <|im_start|>assistant\n[",
            sys = self.prompts().rerank_results,
        );

        let rerank_ctx_size: u32 = 4096;
        let rerank_max_output: i32 = 128;

        let tokens = match self.model().str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[rerank] tokenization failed: {e}");
                return Vec::new();
            }
        };

        let max_prompt_tokens = rerank_ctx_size as usize - rerank_max_output as usize;
        let tokens = if tokens.len() > max_prompt_tokens {
            tokens[..max_prompt_tokens].to_vec()
        } else {
            tokens
        };

        let n_batch = tokens.len().max(512) as u32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(rerank_ctx_size).unwrap()))
            .with_n_batch(n_batch);

        let mut ctx = match self.model().new_context(self.backend(), ctx_params) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[rerank] context failed: {e}");
                return Vec::new();
            }
        };

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            if let Err(e) = batch.add(*token, i as i32, &[0], i == tokens.len() - 1) {
                log::warn!("[rerank] batch add failed: {e}");
                return Vec::new();
            }
        }

        if let Err(e) = ctx.decode(&mut batch) {
            log::warn!("[rerank] decode failed: {e}");
            return Vec::new();
        }

        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.1), LlamaSampler::dist(42)]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        // We pre-seeded with "[" in the prompt
        let mut output = String::from("[");
        let mut n_cur = batch.n_tokens();
        let max_pos = n_cur + rerank_max_output;

        while n_cur < max_pos {
            if start.elapsed() > std::time::Duration::from_secs(10) {
                log::warn!("[rerank] timeout");
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

        log::debug!(
            "[rerank] output='{}' in {:?}",
            output.trim(),
            start.elapsed()
        );

        // Parse: strip think tags, extract JSON array of scores
        let cleaned = strip_think_tags(&output);
        // Find the JSON array
        let json_str = if let Some(start_idx) = cleaned.find('[') {
            if let Some(end_idx) = cleaned[start_idx..].find(']') {
                &cleaned[start_idx..start_idx + end_idx + 1]
            } else {
                // Try adding closing bracket
                &cleaned[start_idx..]
            }
        } else {
            log::warn!("[rerank] no JSON array found in output");
            return Vec::new();
        };

        let scores: Vec<f32> = match serde_json::from_str::<Vec<serde_json::Value>>(json_str) {
            Ok(vals) => vals
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect(),
            Err(e) => {
                log::warn!("[rerank] JSON parse failed: {e}, raw: {json_str}");
                return Vec::new();
            }
        };

        // Map scores back to candidate IDs (handle length mismatch gracefully)
        let mut results: Vec<(String, f32)> = candidates
            .iter()
            .enumerate()
            .map(|(i, (id, _))| {
                let score = scores.get(i).copied().unwrap_or(0.0);
                (id.clone(), score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        log::info!(
            "[rerank] reranked {} candidates in {:?}",
            n,
            start.elapsed()
        );

        results
    }
}
