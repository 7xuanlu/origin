// SPDX-License-Identifier: AGPL-3.0-only
//! Generic on-device LLM inference engine.
//!
//! Wraps `llama-cpp-2` with Metal GPU offload and provides a reusable
//! inference loop. Domain-specific operations (classification, KG extraction,
//! reranking, memory merging) live in sibling modules (`classify`, `extract`,
//! `rerank`, `merge`) and call through to [`LlmEngine::run_inference`] or
//! compose their own tokenize/decode loops against [`LlmEngine::model`] /
//! [`LlmEngine::backend`].

use crate::error::OriginError;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use std::sync::{Arc, OnceLock};

/// Process-wide llama.cpp backend. Initialized lazily on first use and shared
/// across every `LlmEngine` instance in the process. `LlamaBackend::init()` is
/// a one-shot — calling it twice returns `BackendAlreadyInitialized`, which is
/// why we must go through this `OnceLock`.
static LLAMA_BACKEND: OnceLock<Arc<LlamaBackend>> = OnceLock::new();

fn shared_backend() -> Result<Arc<LlamaBackend>, OriginError> {
    // Fast path: already initialized.
    if let Some(b) = LLAMA_BACKEND.get() {
        return Ok(b.clone());
    }
    // Slow path: init and store. If another thread beats us here, our new
    // backend is dropped and we use theirs.
    match LlamaBackend::init() {
        Ok(backend) => {
            let arc = Arc::new(backend);
            let _ = LLAMA_BACKEND.set(arc.clone());
            // Another thread may have raced us; prefer the stored value.
            Ok(LLAMA_BACKEND.get().cloned().unwrap_or(arc))
        }
        Err(e) => {
            // If the backend was initialized elsewhere (e.g. in a test, in a
            // previous engine that didn't use this helper), we can't recover
            // its handle — fail loudly so the caller knows.
            Err(OriginError::Llm(format!("backend init: {e}")))
        }
    }
}
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel as LlamaCppModel};
use llama_cpp_2::sampling::LlamaSampler;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Maximum input chars sent to the LLM (truncated at word boundary).
pub const MAX_INPUT_CHARS: usize = 20_000;
/// Maximum tokens the LLM can generate per request.
pub const MAX_OUTPUT_TOKENS: i32 = 2048;
/// Timeout for a single LLM inference call. 120s accommodates larger context
/// windows (16K+) where prompt prefill and generation take longer, especially
/// on first call after boot (Metal JIT shader compilation).
pub const INFERENCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
/// Context window size for the LLM.
pub const CTX_SIZE: u32 = 8192;

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
pub struct FormattedResult {
    pub formatted_text: String,
    pub summary: String,
    pub space: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub stream_name: Option<String>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
pub struct SessionSynthesisResult {
    pub summary: String,
    pub tags: Vec<String>,
}

/// Generic on-device LLM engine backed by a loaded GGUF model.
///
/// Construction is blocking — call [`LlmEngine::download_model`] and
/// [`LlmEngine::new`] from a dedicated initialization thread or
/// `spawn_blocking` context. Once constructed, inference methods are called
/// from a single worker thread; the engine is neither `Send` nor `Sync` by
/// default (the `unsafe impl` below is sound only because the app guarantees
/// single-threaded access after construction).
pub struct LlmEngine {
    /// Shared process-wide llama.cpp backend. Every `LlmEngine` holds an
    /// `Arc` to the same global backend (see [`shared_backend`]). This is
    /// required because `LlamaBackend::init()` can only be called once per
    /// process — attempting to construct a second fresh backend fails with
    /// `BackendAlreadyInitialized`.
    pub(crate) backend: Arc<LlamaBackend>,
    pub(crate) model: LlamaCppModel,
    pub(crate) prompts: crate::prompts::PromptRegistry,
}

// SAFETY: LlamaCppModel and LlamaBackend are created once on the init thread
// and then used exclusively on the worker thread. Any Arc<LlmEngine> is
// shared only so the app can hold a reference; inference is always called
// from the single worker thread.
unsafe impl Send for LlmEngine {}
unsafe impl Sync for LlmEngine {}

#[allow(dead_code)] // Full inference API -- only run_inference used currently via OnDeviceProvider
impl LlmEngine {
    /// Download a GGUF model via hf-hub by repo and filename.
    /// Uses the sync API (blocking). Cached in ~/.cache/huggingface/hub/.
    pub fn download_model_by_spec(repo_id: &str, filename: &str) -> Result<PathBuf, OriginError> {
        log::info!(
            "[llm_engine] downloading model {}/{} (cached if already present)...",
            repo_id,
            filename
        );

        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| OriginError::Llm(format!("hf-hub init: {e}")))?;

        let repo = api.model(repo_id.to_string());

        let path = repo
            .get(filename)
            .map_err(|e| OriginError::Llm(format!("model download: {e}")))?;

        log::info!("[llm_engine] model path: {}", path.display());
        Ok(path)
    }

    /// Download the default on-device model (backward compat).
    pub fn download_model() -> Result<PathBuf, OriginError> {
        let model = crate::on_device_models::get_default_model();
        Self::download_model_by_spec(model.repo_id, model.filename)
    }

    /// Load the GGUF model with full Metal GPU offload.
    pub fn new(
        model_path: &Path,
        prompts: crate::prompts::PromptRegistry,
    ) -> Result<Self, OriginError> {
        log::info!("[llm_engine] acquiring shared backend...");

        let backend = shared_backend()?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(99);

        log::info!("[llm_engine] loading model...");
        let model = LlamaCppModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| OriginError::Llm(format!("model load: {e}")))?;

        log::info!("[llm_engine] model loaded successfully");
        Ok(Self {
            backend,
            model,
            prompts,
        })
    }

    /// Probe whether Metal context creation works. Returns true if a minimal
    /// context can be allocated. Used by the auto-degrade pattern: if this fails,
    /// the caller can set GGML_METAL_BF16_DISABLE=1 and recreate the engine.
    pub fn probe_metal_context(&self) -> bool {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(512).unwrap()))
            .with_n_batch(512);
        self.model.new_context(&self.backend, ctx_params).is_ok()
    }

    /// Access the loaded llama-cpp model. Used by domain modules that need to
    /// run their own tokenize/decode loops (e.g. classification with a smaller
    /// per-call context window).
    pub(crate) fn model(&self) -> &LlamaCppModel {
        &self.model
    }

    /// Access the llama-cpp backend (needed when creating per-call contexts).
    pub(crate) fn backend(&self) -> &LlamaBackend {
        &self.backend
    }

    // Helper for methods that need to pass `&LlamaBackend` to llama_cpp_2 APIs.
    // Arc<LlamaBackend> derefs to LlamaBackend so `&*self.backend` gives the
    // right type.

    /// Access the shared prompt registry.
    pub(crate) fn prompts(&self) -> &crate::prompts::PromptRegistry {
        &self.prompts
    }

    /// Format raw OCR text using the LLM with unconstrained generation + JSON extraction.
    /// Returns None if inference fails, times out, or output is not valid JSON.
    #[allow(dead_code)]
    pub fn format_ocr_text(
        &self,
        raw_text: &str,
        app_name: &str,
        window_title: Option<&str>,
        spaces: &[String],
    ) -> Option<FormattedResult> {
        let start = Instant::now();

        // Input is already sanitized+structured from the capture pipeline
        // Truncate input at word boundary
        let truncated = truncate_at_word_boundary(raw_text, MAX_INPUT_CHARS);

        let window_title = window_title.unwrap_or("Unknown");

        // Build ChatML prompt for Qwen3 (with thinking disabled)
        let spaces_str = spaces.join(", ");
        let system_prompt = self
            .prompts
            .format_ocr_text
            .replace("{spaces}", &spaces_str);
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
        let tokens = match self.model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[llm_engine] tokenization failed: {e}");
                return None;
            }
        };

        log::info!("[llm_engine] prompt tokens={}", tokens.len());

        // Truncate prompt tokens so there's room for output within context window
        let max_prompt_tokens = CTX_SIZE as usize - MAX_OUTPUT_TOKENS as usize;
        let tokens = if tokens.len() > max_prompt_tokens {
            log::warn!(
                "[llm_engine] prompt tokens ({}) exceed budget ({}), truncating",
                tokens.len(),
                max_prompt_tokens
            );
            tokens[..max_prompt_tokens].to_vec()
        } else {
            tokens
        };

        // Create per-call context -- n_batch must be >= prompt token count
        let n_batch = tokens.len().max(512) as u32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(CTX_SIZE).unwrap()))
            .with_n_batch(n_batch);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[llm_engine] context creation failed: {e}");
                return None;
            }
        };

        // Fill batch with prompt tokens
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            if let Err(e) = batch.add(*token, i as i32, &[0], i == tokens.len() - 1) {
                log::warn!("[llm_engine] batch add failed: {e}");
                return None;
            }
        }

        // Decode prompt
        if let Err(e) = ctx.decode(&mut batch) {
            log::warn!("[llm_engine] prompt decode failed: {e}");
            return None;
        }

        // Build sampler chain -- unconstrained generation (no grammar)
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.3), LlamaSampler::dist(42)]);

        // Generate tokens
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();

        while n_cur < MAX_OUTPUT_TOKENS {
            // Check timeout
            if start.elapsed() > INFERENCE_TIMEOUT {
                log::warn!("[llm_engine] inference timeout after {:?}", start.elapsed());
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => output.push_str(&piece),
                Err(e) => {
                    log::warn!("[llm_engine] token decode failed: {e}");
                    break;
                }
            }

            batch.clear();
            if let Err(e) = batch.add(token, n_cur, &[0], true) {
                log::warn!("[llm_engine] batch add failed: {e}");
                break;
            }

            if let Err(e) = ctx.decode(&mut batch) {
                log::warn!("[llm_engine] decode failed: {e}");
                break;
            }

            n_cur += 1;
        }

        log::info!(
            "[llm_engine] generated {} chars in {:?}",
            output.len(),
            start.elapsed()
        );

        // Strip any residual <think> tags (safety net), then extract JSON
        let cleaned = strip_think_tags(&output);
        let json_str = extract_json(&cleaned).unwrap_or(&cleaned);

        // Parse JSON output
        match serde_json::from_str::<FormattedResult>(json_str) {
            Ok(result) => {
                if result.formatted_text.is_empty() {
                    log::debug!("[llm_engine] empty formatted_text, skipping");
                    return None;
                }
                Some(result)
            }
            Err(e) => {
                log::warn!(
                    "[llm_engine] JSON parse failed: {e}, output: {}",
                    &output[..output.floor_char_boundary(200)]
                );
                None
            }
        }
    }

    /// Generic inference helper: tokenize prompt, run generation, return raw output string.
    /// Used by refinement prompts (dedup merge, extract patterns, detect contradiction).
    pub fn run_inference(
        &self,
        prompt: &str,
        max_output_tokens: i32,
        temperature: f32,
        ctx_size: u32,
        label: Option<&str>,
    ) -> Option<String> {
        let start = Instant::now();

        let tokens = match self.model.str_to_token(prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[llm_engine] inference tokenization failed: {e}");
                return None;
            }
        };

        let max_prompt_tokens = (ctx_size as usize).saturating_sub(max_output_tokens as usize);
        if max_prompt_tokens == 0 {
            log::warn!(
                "[llm_engine] max_output_tokens={} >= ctx_size={}, refusing to run inference",
                max_output_tokens,
                ctx_size
            );
            return None;
        }
        let tokens = if tokens.len() > max_prompt_tokens {
            tokens[..max_prompt_tokens].to_vec()
        } else {
            tokens
        };

        let n_batch = tokens.len().max(512) as u32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()))
            .with_n_batch(n_batch);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[llm_engine] inference context failed: {e}");
                return None;
            }
        };

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            if batch
                .add(*token, i as i32, &[0], i == tokens.len() - 1)
                .is_err()
            {
                return None;
            }
        }

        if ctx.decode(&mut batch).is_err() {
            return None;
        }

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(256, 1.2, 0.0, 0.0),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(42),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();
        let max_pos = n_cur + max_output_tokens;

        while n_cur < max_pos {
            if start.elapsed() > std::time::Duration::from_secs(30) {
                log::warn!("[llm_engine] inference timeout");
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            match self.model.token_to_piece(token, &mut decoder, true, None) {
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

        let cleaned = strip_think_tags(&output);
        let trimmed = cleaned.trim().to_string();

        let label_suffix = label.map(|l| format!(" [{}]", l)).unwrap_or_default();
        log::info!(
            "[llm_engine] inference{}: {} chars in {:?}",
            label_suffix,
            trimmed.len(),
            start.elapsed()
        );

        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }

    /// Benchmark-focused inference: configurable timeout, larger context, and
    /// returns raw output (caller decides whether to strip think tags).
    pub fn run_inference_raw(
        &self,
        prompt: &str,
        max_output_tokens: i32,
        temperature: f32,
        timeout_secs: u64,
        ctx_size: u32,
    ) -> Option<String> {
        let start = Instant::now();

        let tokens = match self.model.str_to_token(prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[llm_engine] raw tokenization failed: {e}");
                return None;
            }
        };

        let max_prompt_tokens = (ctx_size as usize).saturating_sub(max_output_tokens as usize);
        if max_prompt_tokens == 0 {
            log::warn!(
                "[llm_engine] max_output_tokens={} >= ctx_size={}, refusing to run inference",
                max_output_tokens,
                ctx_size
            );
            return None;
        }
        let tokens = if tokens.len() > max_prompt_tokens {
            tokens[..max_prompt_tokens].to_vec()
        } else {
            tokens
        };

        let n_batch = tokens.len().max(512) as u32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(ctx_size).unwrap()))
            .with_n_batch(n_batch);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[llm_engine] raw context failed: {e}");
                return None;
            }
        };

        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            if batch
                .add(*token, i as i32, &[0], i == tokens.len() - 1)
                .is_err()
            {
                return None;
            }
        }

        if ctx.decode(&mut batch).is_err() {
            return None;
        }

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(256, 1.2, 0.0, 0.0),
            LlamaSampler::temp(temperature),
            LlamaSampler::dist(42),
        ]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();
        let max_pos = n_cur + max_output_tokens;

        while n_cur < max_pos {
            if start.elapsed() > std::time::Duration::from_secs(timeout_secs) {
                log::warn!("[llm_engine] raw inference timeout at {}s", timeout_secs);
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            match self.model.token_to_piece(token, &mut decoder, true, None) {
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
            "[llm_engine] raw inference: {} chars in {:?}",
            output.len(),
            start.elapsed()
        );

        if output.is_empty() {
            None
        } else {
            Some(output)
        }
    }

    /// Synthesize a session summary from an activity log.
    /// Returns None if inference fails.
    #[allow(dead_code)]
    pub fn format_session(&self, raw_text: &str, app_name: &str) -> Option<SessionSynthesisResult> {
        let start = Instant::now();

        let truncated = truncate_at_word_boundary(raw_text, MAX_INPUT_CHARS);

        // Build ChatML prompt for session synthesis
        let user_prompt = self
            .prompts
            .summarize_activity_user
            .replace("{apps}", app_name)
            .replace("{log}", truncated);
        let prompt = format!(
            "<|im_start|>system\n\
             {sys}\n\
             <|im_end|>\n\
             <|im_start|>user\n\
             {user_prompt}\n\
             <|im_end|>\n\
             <|im_start|>assistant\n{{\"summary\": \"",
            sys = self.prompts.summarize_activity_system,
        );

        // Tokenize
        let tokens = match self.model.str_to_token(&prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                log::warn!("[llm_engine] session tokenization failed: {e}");
                return None;
            }
        };

        log::info!("[llm_engine] session prompt tokens={}", tokens.len());

        // Truncate if needed
        let max_prompt_tokens = CTX_SIZE as usize - MAX_OUTPUT_TOKENS as usize;
        let tokens = if tokens.len() > max_prompt_tokens {
            tokens[..max_prompt_tokens].to_vec()
        } else {
            tokens
        };

        // Create context
        let n_batch = tokens.len().max(512) as u32;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(CTX_SIZE).unwrap()))
            .with_n_batch(n_batch);

        let mut ctx = match self.model.new_context(&self.backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[llm_engine] session context creation failed: {e}");
                return None;
            }
        };

        // Fill and decode prompt
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, token) in tokens.iter().enumerate() {
            if let Err(e) = batch.add(*token, i as i32, &[0], i == tokens.len() - 1) {
                log::warn!("[llm_engine] session batch add failed: {e}");
                return None;
            }
        }

        if let Err(e) = ctx.decode(&mut batch) {
            log::warn!("[llm_engine] session prompt decode failed: {e}");
            return None;
        }

        // Generate
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::temp(0.3), LlamaSampler::dist(42)]);

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();
        let mut n_cur = batch.n_tokens();

        while n_cur < MAX_OUTPUT_TOKENS {
            if start.elapsed() > INFERENCE_TIMEOUT {
                log::warn!("[llm_engine] session inference timeout");
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            match self.model.token_to_piece(token, &mut decoder, true, None) {
                Ok(piece) => output.push_str(&piece),
                Err(e) => {
                    log::warn!("[llm_engine] session token decode failed: {e}");
                    break;
                }
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
            "[llm_engine] session generated {} chars in {:?}",
            output.len(),
            start.elapsed()
        );

        // Strip any residual <think> tags, then parse JSON
        // Prompt starts assistant with {"summary": " so prepend it
        let cleaned = strip_think_tags(&output);
        let full_output = format!("{{\"summary\": \"{}", cleaned);
        let json_str = extract_json(&full_output).unwrap_or(&full_output);
        match serde_json::from_str::<SessionSynthesisResult>(json_str) {
            Ok(result) => {
                log::info!("[llm_engine] session synthesis: \"{}\"", result.summary);
                Some(result)
            }
            Err(e) => {
                log::warn!(
                    "[llm_engine] session JSON parse failed: {e}, output: {}",
                    &output[..output.floor_char_boundary(200)]
                );
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared parsing and text helpers
// ---------------------------------------------------------------------------

/// Strip any `<think>...</think>` blocks from LLM output (safety net for Qwen3).
pub fn strip_think_tags(text: &str) -> String {
    let mut result = text.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end_offset) = result[start..].find("</think>") {
            let end = start + end_offset + "</think>".len();
            result = format!("{}{}", &result[..start], &result[end..]);
        } else {
            // Unclosed <think> -- strip from <think> to end
            result.truncate(start);
            break;
        }
    }
    result
}

/// Try to extract a JSON object from text that may contain markdown fences or preamble.
/// Finds the first `{` and last `}` and returns the substring.
pub fn extract_json(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

/// Extract a JSON array from text that may have surrounding prose.
/// Finds the first `[` and last `]` and returns the substring.
/// Falls back to wrapping a single JSON object `{...}` in array brackets,
/// since small on-device models (e.g., Qwen3-4B) often return a single
/// object instead of an array when given a single input item.
pub fn extract_json_array(text: &str) -> Option<String> {
    // Strip markdown code fences (Qwen3.5-9B wraps output in ```json...```).
    // Find first JSON-relevant char (`[` or `{`) and last `]` or `}` to
    // narrow the window. The streaming Deserializer (Strategy 2) cannot
    // skip leading backticks on its own.
    let trimmed = {
        let json_start = text.find(['[', '{']);
        match json_start {
            Some(start) => &text[start..],
            None => return None,
        }
    };

    // Strategy 1: try array extraction `[...]` if present and parses cleanly.
    if let (Some(start), Some(end)) = (trimmed.find('['), trimmed.rfind(']')) {
        if end > start {
            let candidate = trimmed[start..=end].to_string();
            if serde_json::from_str::<Vec<serde_json::Value>>(&candidate).is_ok() {
                return Some(candidate);
            }
        }
    }
    // Strategy 2: walk brace depth to collect each complete top-level `{...}`.
    // Handles:
    //   (a) NDJSON `{...}{...}` (no enclosing array)
    //   (b) Truncated array `[{...},{...},{..` where strategy 1 fails because
    //       the closing `]` is missing
    //   (c) Comma-separated objects `{...},{...}` (which Deserializer streaming
    //       chokes on)
    // Tracks string state so braces inside JSON string literals don't confuse
    // the depth count.
    if trimmed.contains('{') {
        let slices = collect_top_level_objects(trimmed);
        let collected: Vec<serde_json::Value> = slices
            .iter()
            .filter_map(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .filter(|v| v.is_object())
            .collect();
        if !collected.is_empty() {
            if let Ok(s) = serde_json::to_string(&collected) {
                return Some(s);
            }
        }
    }
    // Strategy 3: last resort — wrap a single best-effort `{...}` slice in array brackets.
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if end > start {
            return Some(format!("[{}]", &trimmed[start..=end]));
        }
    }
    None
}

/// Walk `text` and return each complete top-level `{...}` slice in order.
/// Tracks string state (with `\\"` escape handling) so braces inside JSON
/// string literals are not counted toward depth. Truncated trailing objects
/// are skipped.
fn collect_top_level_objects(text: &str) -> Vec<&str> {
    let mut results = Vec::new();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                if depth > 0 {
                    depth -= 1;
                    if depth == 0 {
                        if let Some(s) = start {
                            results.push(&text[s..=i]);
                            start = None;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    results
}

/// Truncate text at a word boundary, not exceeding `max_chars` bytes.
/// Uses `floor_char_boundary` to avoid panicking on multi-byte UTF-8.
pub(crate) fn truncate_at_word_boundary(text: &str, max_chars: usize) -> &str {
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
    fn test_truncate_at_word_boundary_ascii() {
        let text = "hello world foo bar";
        // max=12 -> text[..12]="hello world " -> rfind(' ') at 11 -> "hello world"
        assert_eq!(truncate_at_word_boundary(text, 12), "hello world");
        // max=11 -> text[..11]="hello world" -> rfind(' ') at 5 -> "hello"
        assert_eq!(truncate_at_word_boundary(text, 11), "hello");
        assert_eq!(truncate_at_word_boundary(text, 100), text);
    }

    #[test]
    fn test_truncate_at_word_boundary_multibyte() {
        // Curly quotes: \u{201c} and \u{201d} are 3 bytes each
        let text = "he said \u{201c}hello\u{201d} world";
        // Truncating at byte 10 (inside \u{201c}) should not panic
        let result = truncate_at_word_boundary(text, 10);
        assert!(result.len() <= 10);
        assert!(result.is_char_boundary(result.len()));

        // CJK: each char is 3 bytes, no spaces -> returns at safe boundary
        let cjk = "\u{4e16}\u{754c}\u{4f60}\u{597d}";
        let result = truncate_at_word_boundary(cjk, 5);
        assert!(result.len() <= 5);
        assert!(result.is_char_boundary(result.len()));

        // Em dash (\u{2014}, 3 bytes) in text
        let text = "foo \u{2014} bar baz";
        let result = truncate_at_word_boundary(text, 6);
        assert_eq!(result, "foo");
    }

    #[test]
    fn test_truncate_at_word_boundary_exact() {
        let text = "abc def";
        assert_eq!(truncate_at_word_boundary(text, 7), text);
        assert_eq!(truncate_at_word_boundary(text, 3), "abc");
    }

    #[test]
    fn test_strip_think_tags() {
        assert_eq!(
            strip_think_tags("<think>\nsome reasoning\n</think>\n{\"key\": \"val\"}"),
            "\n{\"key\": \"val\"}"
        );
        assert_eq!(
            strip_think_tags("<think>\n\n</think>\n\n{\"a\": 1}"),
            "\n\n{\"a\": 1}"
        );
        assert_eq!(strip_think_tags("{\"key\": \"val\"}"), "{\"key\": \"val\"}");
        assert_eq!(strip_think_tags("prefix<think>dangling"), "prefix");
        assert_eq!(
            strip_think_tags("<think>a</think>mid<think>b</think>end"),
            "midend"
        );
    }

    #[test]
    fn test_extract_json() {
        assert_eq!(
            extract_json(r#"Sure! {"key": "val"}"#),
            Some(r#"{"key": "val"}"#)
        );
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json("{"), None);
    }

    #[test]
    fn test_extract_json_array_with_surrounding_text() {
        let text = r#"Here are the results: [{"i":1}] Hope that helps!"#;
        assert_eq!(extract_json_array(text), Some(r#"[{"i":1}]"#.to_string()));
    }

    #[test]
    fn test_extract_json_array_fallback_wraps_single_object() {
        let text = r#"{"i":1,"type":"fact"}"#;
        assert_eq!(
            extract_json_array(text),
            Some(r#"[{"i":1,"type":"fact"}]"#.to_string())
        );
    }

    /// Regression: single KG object with inner arrays must be wrapped, not
    /// misextracted by matching the inner `[`/`]` as the top-level array.
    #[test]
    fn test_extract_json_array_single_object_with_inner_arrays() {
        let text = r#"{"i": 0, "entities": [{"name": "caroline", "type": "person"}], "observations": [{"entity": "caroline", "content": "joined the group"}]}"#;
        let result = extract_json_array(text).unwrap();
        // Must be a valid JSON array
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
        // Inner entities must survive
        let entities = parsed[0]["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0]["name"], "caroline");
    }

    /// Array response (batch extraction) still works.
    #[test]
    fn test_extract_json_array_real_array_with_inner_arrays() {
        let text = r#"[{"i": 0, "entities": [{"name": "a"}]}, {"i": 1, "entities": []}]"#;
        let result = extract_json_array(text).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    /// NDJSON-style: model returns multiple top-level objects without enclosing
    /// `[]`. This is what Qwen3-4B emits at batch>1. Must collect all into array.
    #[test]
    fn test_extract_json_array_ndjson_multiple_objects() {
        let text = r#"{"i": 0, "entities": [{"name": "a"}]}
{"i": 1, "entities": [{"name": "b"}]}
{"i": 2, "entities": []}"#;
        let result = extract_json_array(text).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0]["i"], 0);
        assert_eq!(parsed[1]["entities"][0]["name"], "b");
    }

    /// NDJSON without separator: `{...}{...}` directly back-to-back.
    #[test]
    fn test_extract_json_array_ndjson_no_separator() {
        let text = r#"{"i":0,"entities":[]}{"i":1,"entities":[]}"#;
        let result = extract_json_array(text).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    /// Markdown code-fenced output (Qwen3.5-9B). Must strip leading fence.
    #[test]
    fn test_extract_json_array_markdown_fence() {
        let text = "```json\n[{\"i\": 0, \"entities\": [{\"name\": \"a\"}]}]\n```";
        let result = extract_json_array(text).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["entities"][0]["name"], "a");
    }

    /// Markdown fence + NDJSON inside.
    #[test]
    fn test_extract_json_array_fence_with_ndjson() {
        let text = "```json\n{\"i\": 0, \"entities\": []}\n{\"i\": 1, \"entities\": []}\n```";
        let result = extract_json_array(text).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    /// Truncated array — model wrote `[{..},{..},{..` and got cut off.
    /// Common at high batch sizes where output exceeds time budget.
    /// Strategy 2 must skip the leading `[` and stream-collect partial objects.
    #[test]
    fn test_extract_json_array_truncated_no_close() {
        let text = r#"[{"i":0,"entities":[{"name":"a"}]},{"i":1,"entities":[{"name":"b"}]},{"i":2,"entities":[{"na"#;
        let result = extract_json_array(text).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        // Should recover the 2 complete objects, drop the truncated 3rd
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["entities"][0]["name"], "a");
        assert_eq!(parsed[1]["entities"][0]["name"], "b");
    }

    /// Markdown fence + truncated array (real 9B failure case).
    #[test]
    fn test_extract_json_array_fence_truncated() {
        let text = "```json\n[\n  {\"i\": 0, \"entities\": []},\n  {\"i\": 1, \"entities\":";
        let result = extract_json_array(text).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
    }
}
