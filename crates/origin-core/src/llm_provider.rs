// SPDX-License-Identifier: AGPL-3.0-only
//! LLM provider trait and implementations.
//!
//! Provides a uniform abstraction over multiple LLM backends:
//! - [`OnDeviceProvider`] — wraps [`crate::engine::LlmEngine`] on a dedicated
//!   `std::thread` for GPU inference (Qwen3-4B via Metal).
//! - [`ApiProvider`] — calls the Anthropic Claude API via `reqwest`.
//! - `MockProvider` — test double, only compiled under `#[cfg(test)]`.
//!
//! Also hosts shared LLM output parsing helpers (`parse_classify_response`,
//! `parse_extraction_response`, `sanitize_json_quotes`) used across the
//! refinery, search, and MCP server entry points.

use async_trait::async_trait;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::engine::LlmEngine;

// ---------------------------------------------------------------------------
// Readiness hook — fires exactly once per process when an LLM provider first
// successfully serves traffic. Used by origin-server to trigger the
// `intelligence-ready` onboarding milestone.
//
// The hook is a process-level one-shot (shared across provider instances):
// main.rs registers the hook once during startup, and providers call
// [`mark_llm_ready`] from their success paths. Subsequent calls are no-ops.
// ---------------------------------------------------------------------------

/// Callback fired when an LLM provider becomes ready.
pub type ReadinessHook = Arc<dyn Fn() + Send + Sync>;

/// The process-level readiness hook. Set once from main.rs, read by providers.
pub static LLM_READINESS_HOOK: OnceLock<ReadinessHook> = OnceLock::new();

/// Guards that the hook fires at most once per process. `OnceLock::set(()).is_ok()`
/// returns true only on the first call, making [`mark_llm_ready`] idempotent.
pub static LLM_READINESS_FIRED: OnceLock<()> = OnceLock::new();

/// Signal that the LLM provider has successfully served a request or completed
/// warmup. Fires the registered readiness hook at most once per process. Safe
/// to call from any provider's success path; no-op if no hook is registered.
pub fn mark_llm_ready() {
    if LLM_READINESS_FIRED.set(()).is_ok() {
        if let Some(hook) = LLM_READINESS_HOOK.get() {
            hook();
        }
    }
}

/// A single inference request.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Human-readable label for logging (e.g. "distill_body", "classify",
    /// "title_gen"). Appears in inference log lines so operators can see
    /// what each LLM call is doing.
    pub label: Option<String>,
    /// Override the default 30s on-device inference timeout. When `Some(n)`,
    /// `OnDeviceProvider` routes through `run_inference_raw` instead of
    /// `run_inference`. Has no effect for cloud providers (Anthropic, etc.).
    pub timeout_secs: Option<u64>,
}

/// Errors returned by an [`LlmProvider`].
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("LLM not available")]
    NotAvailable,
    #[error("Inference failed: {0}")]
    InferenceFailed(String),
    #[error("Timeout")]
    Timeout,
    #[error("Pro plan required")]
    PlanRequired,
}

/// Which backend handled the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmBackend {
    OnDevice,
    Api,
}

/// Trait implemented by both on-device and API-backed LLM providers.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError>;
    fn is_available(&self) -> bool;
    fn name(&self) -> &str;
    fn backend(&self) -> LlmBackend;
    /// Max input tokens this model can effectively synthesize (not just read).
    /// Used by distillation to cap cluster sizes per model capability.
    /// Override per provider — default 8000 is conservative for unknown models.
    fn synthesis_token_limit(&self) -> usize {
        8000
    }
    /// Recommended max output tokens for body-generation tasks (distillation).
    /// Smaller models produce thin output at high counts; cloud models can do more.
    /// Callers should use this instead of hardcoding max_tokens.
    fn recommended_max_output(&self) -> u32 {
        2048
    }
    /// Context window size for on-device inference. Cloud providers ignore this
    /// (the API manages context). On-device providers derive from model spec.
    fn context_size(&self) -> u32 {
        8192
    }
}

// ---------------------------------------------------------------------------
// OnDeviceProvider — wraps LlmEngine on a dedicated GPU inference thread
// ---------------------------------------------------------------------------

/// Wrap a user/system prompt pair in the Qwen ChatML template.
///
/// The empty `<think></think>` block disables thinking mode for Qwen3.5
/// (which defaults to thinking-on). Without it, all output tokens go to
/// reasoning and `strip_think_tags` removes them, leaving empty output.
/// Harmless for Qwen3 (which has no thinking mode).
fn format_chatml_prompt(system: Option<&str>, user: &str) -> String {
    match system {
        Some(sys) => format!(
            "<|im_start|>system\n{sys}\n<|im_end|>\n\
             <|im_start|>user\n{user}\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        None => format!(
            "<|im_start|>user\n{user}\n<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
    }
}

/// Internal message sent to the inference worker thread.
struct InferenceRequest {
    prompt: String,
    system_prompt: Option<String>,
    max_tokens: i32,
    temperature: f32,
    ctx_size: u32,
    label: Option<String>,
    timeout_secs: Option<u64>,
    response_tx: tokio::sync::oneshot::Sender<Result<String, LlmError>>,
}

/// On-device LLM provider that runs inference on a dedicated `std::thread`.
///
/// Construction downloads the model (cached), loads it via Metal GPU, and
/// spawns a worker thread that processes [`InferenceRequest`]s received over a
/// bounded `SyncSender` channel (capacity 8).
pub struct OnDeviceProvider {
    tx: std::sync::mpsc::SyncSender<InferenceRequest>,
    available: Arc<AtomicBool>,
    synthesis_limit: usize,
    model_context_size: u32,
    model_max_output: u32,
}

impl OnDeviceProvider {
    /// Probe Metal context creation with retries. ggml creates a new
    /// MTLCommandQueue per context, and macOS limits live queues per process.
    /// Retrying after a delay lets the Objective-C autorelease pool drain
    /// queues from dropped contexts.
    fn probe_with_retries(engine: &crate::engine::LlmEngine, label: &str) -> bool {
        for attempt in 0..3 {
            if engine.probe_metal_context() {
                if attempt > 0 {
                    log::info!(
                        "[on_device_provider] Metal probe ({label}) succeeded on attempt {}",
                        attempt + 1
                    );
                }
                return true;
            }
            log::warn!(
                "[on_device_provider] Metal probe ({label}) failed, attempt {}/3. \
                 Waiting for autorelease pool to drain...",
                attempt + 1
            );
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        false
    }

    /// Create a new on-device provider with a specific model from the registry.
    ///
    /// If `model_id` is Some, resolves it against the registry — unknown ids
    /// fall back to the default with a warning (so stale config values from
    /// older releases don't break startup). If None, uses the default model.
    ///
    /// This downloads the model (if not cached), loads it onto the GPU, and
    /// spawns a background thread for inference. All of this is blocking, so
    /// call from a context that can block (e.g. `spawn_blocking`).
    pub fn new_with_model(model_id: Option<&str>) -> Result<Self, crate::error::OriginError> {
        let model_spec = crate::on_device_models::resolve_or_default(model_id);
        log::info!(
            "[on_device_provider] using model: {} ({})",
            model_spec.display_name,
            model_spec.id
        );
        let model_path =
            LlmEngine::download_model_by_spec(model_spec.repo_id, model_spec.filename)?;
        let prompts =
            crate::prompts::PromptRegistry::load(&crate::prompts::PromptRegistry::override_dir());

        // Auto-degrade Metal init with retries.
        //
        // ggml's Metal backend (llama-cpp-2 v0.1.143) creates a new
        // MTLCommandQueue per context. macOS enforces a per-process limit on
        // live command queues. When queues from prior probe/drop cycles haven't
        // been reclaimed by the Objective-C autorelease pool, newCommandQueue
        // returns nil. Retrying after a short delay lets the pool drain.
        //
        // Additionally, GGML_METAL_NO_RESIDENCY disables residency sets that
        // can accumulate and block queue creation under memory pressure.
        //
        // Fallback chain: BF16 with retries -> BF16 disabled with retries -> error.
        std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");

        let engine = LlmEngine::new(&model_path, prompts.clone())?;
        let engine = if Self::probe_with_retries(&engine, "BF16") {
            log::info!("[on_device_provider] Metal context probe passed (BF16 OK)");
            engine
        } else {
            log::warn!(
                "[on_device_provider] Metal context probe failed after retries. \
                 Retrying with BF16 disabled (macOS Metal compatibility)."
            );
            std::env::set_var("GGML_METAL_BF16_DISABLE", "1");
            drop(engine);
            // Sleep to let autorelease pool fully drain queues from the dropped engine.
            std::thread::sleep(std::time::Duration::from_millis(500));
            let fallback = LlmEngine::new(&model_path, prompts)?;
            if !Self::probe_with_retries(&fallback, "BF16-disabled") {
                log::error!(
                    "[on_device_provider] Metal context still fails with BF16 disabled \
                     after retries. On-device inference unavailable."
                );
                return Err(crate::error::OriginError::Llm(
                    "Metal context creation failed even with BF16 disabled".into(),
                ));
            }
            log::warn!(
                "[on_device_provider] running in degraded mode (BF16 disabled, slower inference). \
                 Update macOS to restore full Metal performance."
            );
            fallback
        };

        // Channel capacity sized for concurrent MCP bursts: a 10-store
        // `remember` burst fires ~4 LLM requests per store (classify +
        // extract at handler level + entity extraction + title generation
        // at post_ingest level) = ~40 queued inference requests. The old
        // cap of 8 dropped requests with "channel send failed: sending on
        // a full channel" under the post-S2-async load, leaving memories
        // silently unclassified. 256 absorbs bursts up to ~64 concurrent
        // stores before backpressure.
        let (tx, rx) = std::sync::mpsc::sync_channel::<InferenceRequest>(256);

        // Multi-context worker pool. ORIGIN_LLM_WORKERS controls the number of
        // parallel inference threads, each owning a persistent LlamaContext.
        // Workers compete on a shared receiver: lock is held only during recv(),
        // released before GPU inference, so contention is minimal.
        //
        // ORIGIN_LLM_PARALLEL_SEQS (Option B / S2) controls how many sequences
        // each worker decodes in parallel inside its single LlamaContext via
        // llama.cpp's continuous batching scheduler. M=1 (default) routes
        // through the single-seq persistent path and is byte-identical to S1.
        // M>1 enables the batched decode loop in
        // `LlmEngine::run_inference_continuous_batch`.
        //
        // Memory budget per worker: ~0.8 GB KV cache at ctx=8192 for M=1.
        // M sequences in one context still share the same `n_ctx` allocation,
        // so per-worker KV memory does NOT scale linearly with M — but each
        // sequence's effective per-seq budget shrinks to ctx_size / M.
        // N=4 workers = 3.2 GB KV + 2.7 GB shared weights ≈ safe on 16 GB Mac.
        // Default N=1, M=1 is identical to the previous single-thread behavior.
        //
        // GGML_METAL_NO_RESIDENCY (set above) mitigates Metal command queue
        // exhaustion from multiple simultaneous contexts.
        let n_workers = std::env::var("ORIGIN_LLM_WORKERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        let parallel_seqs = std::env::var("ORIGIN_LLM_PARALLEL_SEQS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        log::info!(
            "[on_device_provider] {n_workers} worker(s) x {parallel_seqs} seq(s) each \
             (max concurrent = {})",
            n_workers * parallel_seqs
        );

        let available = Arc::new(AtomicBool::new(true));

        // Wrap the receiver so it can be shared across worker threads.
        // std::sync::mpsc::Receiver is Send but not Sync, so we guard it with
        // a Mutex. Only one worker holds the lock at a time (during recv()),
        // then releases it before the actual GPU inference call.
        let rx_shared = Arc::new(std::sync::Mutex::new(rx));

        // Shared engine across all workers. LlmEngine holds the model weights
        // once (loaded onto the GPU); each worker creates its own LlamaContext
        // from it, so the weights are not duplicated.
        let engine = Arc::new(engine);

        for i in 0..n_workers {
            let rx_arc = Arc::clone(&rx_shared);
            let engine = Arc::clone(&engine);
            let thread_available = Arc::clone(&available);
            let m = parallel_seqs;

            std::thread::Builder::new()
                .name(format!("llm-provider-worker-{i}"))
                .spawn(move || {
                    log::info!(
                        "[on_device_provider] worker {i} thread started (parallel_seqs={m})"
                    );

                    // Persistent LlamaContext: build lazily on first request,
                    // then reuse across all subsequent requests by clearing the
                    // KV cache between calls. Saves the per-call cost of
                    // `new_context()` (KV cache allocation + Metal pipeline
                    // rebuild) which was the dominant overhead under serial
                    // inference. If a request arrives with a different
                    // ctx_size, the context is rebuilt.
                    //
                    // At M>1 the context is built with n_seq_max=M so the
                    // continuous-batching scheduler can slot up to M sequences
                    // simultaneously.
                    let mut persistent_ctx: Option<llama_cpp_2::context::LlamaContext<'_>> = None;
                    let mut persistent_ctx_size: u32 = 0;

                    loop {
                        // Block on the first request, then opportunistically
                        // pull up to M-1 more if M>1. The first recv() may
                        // sleep; subsequent try_recv() calls return Empty
                        // immediately so a single in-flight request is never
                        // delayed waiting for siblings.
                        let first_req = match rx_arc.lock().unwrap().recv() {
                            Ok(r) => r,
                            Err(_) => break, // channel closed — sender dropped
                        };

                        let mut batch_reqs: Vec<InferenceRequest> = Vec::with_capacity(m);
                        batch_reqs.push(first_req);

                        if m > 1 {
                            // Short coalescing window: try to drain up to M-1
                            // additional pending requests without blocking.
                            // Bounded by `coalesce_attempts` * `coalesce_sleep`
                            // so a lone request doesn't sit waiting for siblings.
                            let coalesce_attempts: u32 = 3;
                            let coalesce_sleep = std::time::Duration::from_millis(5);
                            for _ in 0..coalesce_attempts {
                                if batch_reqs.len() >= m {
                                    break;
                                }
                                let drained = {
                                    let rx = rx_arc.lock().unwrap();
                                    let mut local = Vec::new();
                                    while batch_reqs.len() + local.len() < m {
                                        match rx.try_recv() {
                                            Ok(r) => local.push(r),
                                            Err(_) => break,
                                        }
                                    }
                                    local
                                };
                                let did_drain = !drained.is_empty();
                                batch_reqs.extend(drained);
                                if !did_drain {
                                    std::thread::sleep(coalesce_sleep);
                                }
                            }
                        }

                        // The required ctx_size is the max across the batched
                        // requests (callers normally use a single value, but be
                        // defensive). Rebuild context if it changes.
                        let target_ctx_size = batch_reqs
                            .iter()
                            .map(|r| r.ctx_size)
                            .max()
                            .unwrap_or(0);

                        if persistent_ctx.is_none() || persistent_ctx_size != target_ctx_size {
                            let _ = persistent_ctx.take();
                            persistent_ctx = engine.build_persistent_context_with_seq_max(
                                target_ctx_size,
                                m as u32,
                            );
                            if persistent_ctx.is_some() {
                                persistent_ctx_size = target_ctx_size;
                                log::info!(
                                    "[on_device_provider] worker {i} persistent context built \
                                     (ctx_size={target_ctx_size}, n_seq_max={m})"
                                );
                            }
                        }

                        // M=1 path: byte-identical to S1 (single-seq persistent
                        // inference). This is critical for backward compat —
                        // single-request latency must not regress.
                        if m == 1 || batch_reqs.len() == 1 {
                            let req = batch_reqs.pop().expect("batch has at least 1 req");
                            let full_prompt = format_chatml_prompt(
                                req.system_prompt.as_deref(),
                                &req.prompt,
                            );
                            let (timeout_secs, strip_think) = match req.timeout_secs {
                                Some(secs) => (secs, false),
                                None => (30, true),
                            };

                            let result = match persistent_ctx.as_mut() {
                                Some(ctx) => engine.run_inference_persistent(
                                    ctx,
                                    &full_prompt,
                                    req.max_tokens,
                                    req.temperature,
                                    timeout_secs,
                                    strip_think,
                                    req.label.as_deref(),
                                ),
                                None => {
                                    log::warn!(
                                        "[on_device_provider] worker {i} persistent context \
                                         unavailable, falling back to per-call context"
                                    );
                                    match req.timeout_secs {
                                        Some(secs) => engine.run_inference_raw(
                                            &full_prompt,
                                            req.max_tokens,
                                            req.temperature,
                                            secs,
                                            req.ctx_size,
                                        ),
                                        None => engine.run_inference(
                                            &full_prompt,
                                            req.max_tokens,
                                            req.temperature,
                                            req.ctx_size,
                                            req.label.as_deref(),
                                        ),
                                    }
                                }
                            };

                            let response = match result {
                                Some(text) => {
                                    mark_llm_ready();
                                    Ok(text)
                                }
                                None => Err(LlmError::InferenceFailed(
                                    "inference returned no output".into(),
                                )),
                            };
                            let _ = req.response_tx.send(response);
                            continue;
                        }

                        // M>1 with multiple coalesced requests: run continuous
                        // batching. Each request gets a distinct seq_id slot.
                        // Outputs are demultiplexed back to per-request channels.
                        let prompts: Vec<(String, i32, f32, u64, bool, Option<String>)> = batch_reqs
                            .iter()
                            .map(|req| {
                                let prompt = format_chatml_prompt(
                                    req.system_prompt.as_deref(),
                                    &req.prompt,
                                );
                                let (timeout_secs, strip_think) = match req.timeout_secs {
                                    Some(secs) => (secs, false),
                                    None => (30, true),
                                };
                                (
                                    prompt,
                                    req.max_tokens,
                                    req.temperature,
                                    timeout_secs,
                                    strip_think,
                                    req.label.clone(),
                                )
                            })
                            .collect();

                        let outputs: Vec<Option<String>> = match persistent_ctx.as_mut() {
                            Some(ctx) => {
                                engine.run_inference_continuous_batch(ctx, &prompts)
                            }
                            None => {
                                log::warn!(
                                    "[on_device_provider] worker {i} persistent multi-seq \
                                     context unavailable, falling back to serial per-call"
                                );
                                // Serial fallback: run each request through
                                // run_inference_raw / run_inference. Slower
                                // but preserves correctness.
                                prompts
                                    .iter()
                                    .zip(batch_reqs.iter())
                                    .map(|((p, max_t, temp, timeout, _strip, label), req)| {
                                        match req.timeout_secs {
                                            Some(_) => engine.run_inference_raw(
                                                p,
                                                *max_t,
                                                *temp,
                                                *timeout,
                                                req.ctx_size,
                                            ),
                                            None => engine.run_inference(
                                                p,
                                                *max_t,
                                                *temp,
                                                req.ctx_size,
                                                label.as_deref(),
                                            ),
                                        }
                                    })
                                    .collect()
                            }
                        };

                        // Demultiplex per-seq results back to per-request channels.
                        let mut any_ok = false;
                        for (req, out) in batch_reqs.into_iter().zip(outputs) {
                            let response = match out {
                                Some(text) => {
                                    any_ok = true;
                                    Ok(text)
                                }
                                None => Err(LlmError::InferenceFailed(
                                    "continuous-batch inference returned no output".into(),
                                )),
                            };
                            let _ = req.response_tx.send(response);
                        }
                        if any_ok {
                            mark_llm_ready();
                        }
                    }

                    // Drop the persistent context before exiting so Metal
                    // queues are released cleanly.
                    drop(persistent_ctx);
                    log::info!("[on_device_provider] worker {i} thread exiting");
                    // Mark available=false only when the last worker exits.
                    // Workers share the AtomicBool; any surviving worker
                    // keeps the provider available. We use a simple approach:
                    // set false unconditionally — if another worker is still
                    // running it will not reset it, but the channel being
                    // closed means no new requests can succeed anyway.
                    thread_available.store(false, Ordering::SeqCst);
                })
                .map_err(|e| {
                    crate::error::OriginError::Llm(format!(
                        "failed to spawn inference thread {i}: {e}"
                    ))
                })?;
        }

        Ok(Self {
            tx,
            available,
            synthesis_limit: model_spec.synthesis_token_limit,
            model_context_size: model_spec.context_size,
            model_max_output: model_spec.max_output_tokens,
        })
    }

    /// Create with the default model (backward compat).
    pub fn new() -> Result<Self, crate::error::OriginError> {
        Self::new_with_model(None)
    }
}

#[async_trait]
impl LlmProvider for OnDeviceProvider {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        if !self.is_available() {
            return Err(LlmError::NotAvailable);
        }

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        let inference_req = InferenceRequest {
            prompt: request.user_prompt,
            system_prompt: request.system_prompt,
            max_tokens: request.max_tokens as i32,
            temperature: request.temperature,
            ctx_size: self.model_context_size,
            label: request.label,
            timeout_secs: request.timeout_secs,
            response_tx,
        };

        self.tx
            .try_send(inference_req)
            .map_err(|e| LlmError::InferenceFailed(format!("channel send failed: {e}")))?;

        // Match the engine-side INFERENCE_TIMEOUT (120s) to avoid the provider
        // timing out before the engine finishes. Larger context windows (16K+)
        // need more time for prompt prefill + generation.
        match tokio::time::timeout(std::time::Duration::from_secs(120), response_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(LlmError::InferenceFailed(
                "worker dropped response channel".into(),
            )),
            Err(_) => Err(LlmError::Timeout),
        }
    }

    fn is_available(&self) -> bool {
        self.available.load(Ordering::SeqCst)
    }

    fn name(&self) -> &str {
        "on_device"
    }

    fn backend(&self) -> LlmBackend {
        LlmBackend::OnDevice
    }

    fn synthesis_token_limit(&self) -> usize {
        self.synthesis_limit
    }

    fn recommended_max_output(&self) -> u32 {
        self.model_max_output
    }

    fn context_size(&self) -> u32 {
        self.model_context_size
    }
}

// ---------------------------------------------------------------------------
// Shared LLM output parsing helpers
// ---------------------------------------------------------------------------

/// Strip any `<think>...</think>` blocks from LLM output (safety net for Qwen3).
///
/// Re-exported from [`crate::engine`] so call sites can keep using
/// `llm_provider::strip_think_tags`.
pub use crate::engine::{extract_json, strip_think_tags};

/// Replace Unicode curly quotes with escaped straight quotes so JSON parsing succeeds.
pub fn sanitize_json_quotes(text: &str) -> String {
    text.replace(['\u{201C}', '\u{201D}'], "\\\"")
        .replace(['\u{2018}', '\u{2019}'], "'")
}

/// Parse LLM extraction output into structured_fields JSON string and retrieval_cue.
/// Reuses existing extract_json + sanitize_json_quotes helpers for robust LLM output parsing.
pub fn parse_extraction_response(output: &str) -> (Option<String>, Option<String>) {
    let cleaned = strip_think_tags(output);
    let sanitized = sanitize_json_quotes(&cleaned);
    let json_str = match extract_json(&sanitized) {
        Some(s) => s.to_string(),
        None => return (None, None),
    };
    let json: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return (None, None),
    };

    let retrieval_cue = json
        .get("retrieval_cue")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Remove retrieval_cue from fields — it's stored separately
    let mut fields = json.clone();
    if let Some(obj) = fields.as_object_mut() {
        obj.remove("retrieval_cue");
        if obj.is_empty() {
            return (None, retrieval_cue);
        }
    }

    (Some(fields.to_string()), retrieval_cue)
}

/// Parse a classify JSON response from the LLM into a ClassificationResult.
/// Shared by refinery, search commands, and server memory endpoints.
pub fn parse_classify_response(raw: &str) -> Option<crate::classify::ClassificationResult> {
    let stripped = strip_think_tags(raw);
    let json_str = extract_json(&stripped)?;
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;

    let memory_type = val
        .get("memory_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase())
        .filter(|s| s.parse::<crate::sources::MemoryType>().is_ok())
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

    let quality = val
        .get("quality")
        .and_then(|v| v.as_str())
        .map(|s| s.to_lowercase())
        .filter(|s| matches!(s.as_str(), "low" | "medium" | "high"));

    Some(crate::classify::ClassificationResult {
        memory_type,
        domain,
        tags,
        quality,
    })
}

// ---------------------------------------------------------------------------
// ApiProvider — Anthropic Claude API
// ---------------------------------------------------------------------------

pub const DEFAULT_ROUTINE_MODEL: &str = "claude-haiku-4-5-20251001";

pub struct ApiProvider {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl ApiProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    pub fn with_default_model(api_key: String) -> Self {
        Self::new(api_key, DEFAULT_ROUTINE_MODEL.to_string())
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl LlmProvider for ApiProvider {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        let messages = vec![serde_json::json!({"role": "user", "content": request.user_prompt})];

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        });
        if let Some(ref sys) = request.system_prompt {
            body["system"] = serde_json::json!(sys);
        }

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::InferenceFailed(format!("API request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::InferenceFailed(format!(
                "API error {}: {}",
                status, text
            )));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| LlmError::InferenceFailed(format!("API read error: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| LlmError::InferenceFailed(format!("API parse error: {}", e)))?;

        // Extract text from content[0].text
        let text = if let Some(arr) = json["content"].as_array() {
            arr.first()
                .and_then(|block| block["text"].as_str())
                .unwrap_or("")
                .to_string()
        } else {
            String::new()
        };

        // First successful API response = BYOK provider is ready to answer
        // queries. Safe to fire the onboarding readiness hook; subsequent
        // calls are no-ops.
        mark_llm_ready();

        Ok(text)
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn name(&self) -> &str {
        "api"
    }

    fn backend(&self) -> LlmBackend {
        LlmBackend::Api
    }

    fn synthesis_token_limit(&self) -> usize {
        // Model-specific effective synthesis limits (research-calibrated).
        // Context windows: Opus/Haiku 1M, Sonnet 200K (1M beta).
        // Synthesis quality degrades well before context limit — these are
        // the effective ranges where output remains coherent and faithful.
        // New models: add a branch here when adding to the supported set.
        if self.model.contains("opus") {
            100_000 // Opus: strongest synthesis, 1M context — ~10% utilization
        } else if self.model.contains("sonnet") {
            64_000 // Sonnet: 200K-1M context — ~32% utilization at 200K
        } else if self.model.contains("haiku") {
            50_000 // Haiku: 1M context — ~5% utilization, quality-limited
        } else {
            32_000 // Unknown API model — conservative default
        }
    }

    fn recommended_max_output(&self) -> u32 {
        // Cloud models sustain quality at much higher output counts than
        // on-device quantized models. 4096 is a comfortable upper bound
        // for wiki-style concept bodies without hitting API max_tokens limits.
        if self.model.contains("haiku") {
            2_048
        } else {
            4_096 // Opus and Sonnet produce high-quality long-form
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAICompatibleProvider — any OpenAI-compatible API (Ollama, LM Studio, etc.)
// ---------------------------------------------------------------------------

pub struct OpenAICompatibleProvider {
    endpoint: String,
    model: String,
    client: reqwest::Client,
}

impl OpenAICompatibleProvider {
    pub fn new(endpoint: String, model: String) -> Self {
        // Ensure endpoint doesn't have trailing slash
        let endpoint = endpoint.trim_end_matches('/').to_string();
        Self {
            endpoint,
            model,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60)) // longer timeout for local models
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl LlmProvider for OpenAICompatibleProvider {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        let mut messages = Vec::new();
        if let Some(ref sys) = request.system_prompt {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        messages.push(serde_json::json!({"role": "user", "content": request.user_prompt}));

        let body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        });

        let url = format!("{}/chat/completions", self.endpoint);
        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::InferenceFailed(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(LlmError::InferenceFailed(format!(
                "API error {}: {}",
                status, text
            )));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| LlmError::InferenceFailed(format!("Read error: {}", e)))?;
        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| LlmError::InferenceFailed(format!("Parse error: {}", e)))?;

        // OpenAI format: choices[0].message.content
        let text = json["choices"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|choice| choice["message"]["content"].as_str())
            .unwrap_or("")
            .to_string();

        // First successful response from an OpenAI-compatible endpoint
        // (Ollama, LM Studio, etc.) counts as intelligence-ready.
        mark_llm_ready();

        Ok(text)
    }

    fn is_available(&self) -> bool {
        !self.endpoint.is_empty() && !self.model.is_empty()
    }

    fn name(&self) -> &str {
        "external"
    }

    fn backend(&self) -> LlmBackend {
        LlmBackend::Api
    }
}

// ---------------------------------------------------------------------------
// ClaudeCliProvider — uses `claude -p` CLI (Max plan, no API key)
// ---------------------------------------------------------------------------

/// LLM provider that shells out to `claude -p` for inference.
/// Uses the user's Claude Max subscription via OAuth (no API key needed).
/// Intended for eval benchmarks comparing on-device vs cloud quality.
pub struct ClaudeCliProvider {
    model: String,
}

impl ClaudeCliProvider {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }

    /// Haiku via Max plan.
    pub fn haiku() -> Self {
        Self::new("haiku")
    }

    /// Sonnet via Max plan.
    pub fn sonnet() -> Self {
        Self::new("sonnet")
    }
}

#[async_trait]
impl LlmProvider for ClaudeCliProvider {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        use tokio::io::AsyncWriteExt;
        use tokio::process::Command;

        let mut args = vec![
            "-p".to_string(),
            "--model".to_string(),
            self.model.clone(),
            "--no-session-persistence".to_string(),
            "--allowedTools".to_string(),
            "".to_string(),
        ];
        if let Some(ref sys) = request.system_prompt {
            args.push("--system-prompt".to_string());
            args.push(sys.clone());
        }

        // Strip ANTHROPIC_API_KEY so claude CLI uses Max plan OAuth instead of
        // routing through an API account that may have a low credit balance.
        // The eval harness intentionally chose CLI over Batch API to avoid API costs.
        let mut child = Command::new("claude")
            .env_remove("ANTHROPIC_API_KEY")
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| LlmError::InferenceFailed(format!("claude -p spawn: {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(request.user_prompt.as_bytes())
                .await
                .map_err(|e| LlmError::InferenceFailed(format!("claude stdin write: {e}")))?;
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| LlmError::InferenceFailed(format!("claude -p wait: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(LlmError::InferenceFailed(format!(
                "claude -p exited {}: {}",
                output.status, stderr
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        &self.model
    }

    fn backend(&self) -> LlmBackend {
        LlmBackend::Api
    }

    fn synthesis_token_limit(&self) -> usize {
        32000
    }

    fn recommended_max_output(&self) -> u32 {
        4096
    }
}

/// Mock provider for testing -- returns a fixed response.
#[cfg(test)]
pub struct MockProvider {
    response: Option<String>,
    backend: LlmBackend,
}

#[cfg(test)]
impl MockProvider {
    pub fn new(response: &str) -> Self {
        Self {
            response: Some(response.to_string()),
            backend: LlmBackend::OnDevice,
        }
    }
    pub fn unavailable() -> Self {
        Self {
            response: None,
            backend: LlmBackend::OnDevice,
        }
    }
    pub fn with_backend(mut self, backend: LlmBackend) -> Self {
        self.backend = backend;
        self
    }
}

#[cfg(test)]
#[async_trait]
impl LlmProvider for MockProvider {
    async fn generate(&self, _request: LlmRequest) -> Result<String, LlmError> {
        self.response.clone().ok_or(LlmError::NotAvailable)
    }
    fn is_available(&self) -> bool {
        self.response.is_some()
    }
    fn name(&self) -> &str {
        "mock"
    }

    fn backend(&self) -> LlmBackend {
        self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_chatml_prompt_with_system() {
        let p = format_chatml_prompt(Some("you are helpful"), "hi");
        assert!(p.contains("<|im_start|>system\nyou are helpful\n<|im_end|>"));
        assert!(p.contains("<|im_start|>user\nhi\n<|im_end|>"));
        assert!(p.contains("<|im_start|>assistant\n<think>"));
    }

    #[test]
    fn test_format_chatml_prompt_no_system() {
        let p = format_chatml_prompt(None, "hello world");
        assert!(!p.contains("<|im_start|>system"));
        assert!(p.contains("<|im_start|>user\nhello world\n<|im_end|>"));
        assert!(p.contains("<|im_start|>assistant\n<think>"));
    }

    #[test]
    fn test_parallel_seqs_env_clamping() {
        // Document the clamp behavior expected by the worker init: invalid
        // / out-of-range values must clamp to [1, 8]. This characterizes the
        // env-parsing logic without spinning up an OnDeviceProvider (which
        // requires GPU + model files).
        for (input, expected) in [
            (Some("0"), 1usize),
            (Some("1"), 1),
            (Some("4"), 4),
            (Some("8"), 8),
            (Some("99"), 8),
            (Some("not_a_number"), 1),
            (None, 1),
        ] {
            let parsed = input
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 8);
            assert_eq!(parsed, expected, "input {input:?} should clamp to {expected}");
        }
    }

    #[test]
    fn test_llm_request_clone() {
        let req = LlmRequest {
            system_prompt: Some("You are helpful.".into()),
            user_prompt: "Hello".into(),
            max_tokens: 256,
            temperature: 0.1,
            label: None,
            timeout_secs: None,
        };
        let cloned = req.clone();
        assert_eq!(cloned.user_prompt, "Hello");
        assert_eq!(cloned.max_tokens, 256);
    }

    #[test]
    fn test_llm_error_display() {
        let err = LlmError::NotAvailable;
        assert_eq!(format!("{}", err), "LLM not available");
        let err = LlmError::InferenceFailed("OOM".into());
        assert!(format!("{}", err).contains("OOM"));
    }

    #[tokio::test]
    async fn test_mock_provider_generate() {
        let mock = MockProvider::new("test response");
        assert!(mock.is_available());
        assert_eq!(mock.name(), "mock");
        let result = mock
            .generate(LlmRequest {
                system_prompt: None,
                user_prompt: "test".into(),
                max_tokens: 100,
                temperature: 0.0,
                label: None,
                timeout_secs: None,
            })
            .await
            .unwrap();
        assert_eq!(result, "test response");
    }

    #[tokio::test]
    async fn test_mock_provider_unavailable() {
        let mock = MockProvider::unavailable();
        assert!(!mock.is_available());
        let result = mock
            .generate(LlmRequest {
                system_prompt: None,
                user_prompt: "test".into(),
                max_tokens: 100,
                temperature: 0.0,
                label: None,
                timeout_secs: None,
            })
            .await;
        assert!(matches!(result, Err(LlmError::NotAvailable)));
    }

    #[test]
    fn test_api_provider_no_key() {
        let provider = ApiProvider::with_default_model(String::new());
        assert!(!provider.is_available());
        assert_eq!(provider.name(), "api");
    }

    #[test]
    fn test_api_provider_with_key() {
        let provider = ApiProvider::with_default_model("sk-ant-test".to_string());
        assert!(provider.is_available());
    }

    #[tokio::test]
    async fn test_api_provider_empty_key_fails() {
        let provider = ApiProvider::with_default_model(String::new());
        let result = provider
            .generate(LlmRequest {
                system_prompt: None,
                user_prompt: "test".into(),
                max_tokens: 100,
                temperature: 0.0,
                label: None,
                timeout_secs: None,
            })
            .await;
        // Empty key → API call will fail
        assert!(result.is_err());
    }

    #[test]
    fn test_sanitize_json_quotes() {
        let input = "User analyzes a \u{201C}ambient OS\u{201D} project";
        let result = sanitize_json_quotes(input);
        assert!(result.contains("\\\"ambient OS\\\""));
    }

    #[test]
    fn test_parse_extraction_response_valid() {
        let output = r#"{"claim": "I love Rust", "evidence": "10 years exp", "retrieval_cue": "What does the user know about Rust?"}"#;
        let (fields, cue) = parse_extraction_response(output);
        assert!(fields.is_some());
        assert!(cue.is_some());
        let fields_json: serde_json::Value = serde_json::from_str(&fields.unwrap()).unwrap();
        assert_eq!(fields_json["claim"], "I love Rust");
        assert!(cue.unwrap().contains("Rust"));
    }

    #[test]
    fn test_parse_extraction_response_with_think_tags() {
        let output = "<think>reasoning</think>{\"claim\": \"test\", \"retrieval_cue\": \"q?\"}";
        let (fields, cue) = parse_extraction_response(output);
        assert!(fields.is_some());
        assert!(cue.is_some());
    }

    #[test]
    fn test_parse_extraction_response_invalid() {
        let (fields, cue) = parse_extraction_response("not json at all");
        assert!(fields.is_none());
        assert!(cue.is_none());
    }

    #[test]
    fn test_parse_extraction_response_empty_fields() {
        let output = r#"{"retrieval_cue": "just a cue"}"#;
        let (fields, cue) = parse_extraction_response(output);
        assert!(fields.is_none()); // No actual fields, just cue
        assert!(cue.is_some());
    }

    #[test]
    fn test_parse_classify_response() {
        let raw =
            r#"{"memory_type": "preference", "domain": "technology", "tags": ["dark mode", "ui"]}"#;
        let result = parse_classify_response(raw).unwrap();
        assert_eq!(result.memory_type, "preference");
        assert_eq!(result.domain, Some("technology".to_string()));
        assert_eq!(result.tags, vec!["dark mode", "ui"]);
    }

    #[test]
    fn test_parse_classify_response_with_think_tags() {
        let raw =
            r#"<think>analyzing...</think>{"memory_type": "fact", "domain": "work", "tags": []}"#;
        let result = parse_classify_response(raw).unwrap();
        assert_eq!(result.memory_type, "fact");
        assert_eq!(result.domain, Some("work".to_string()));
    }

    #[test]
    fn test_parse_classify_response_invalid() {
        assert!(parse_classify_response("not json at all").is_none());
    }

    #[test]
    fn openai_compatible_provider_stores_config() {
        let provider =
            OpenAICompatibleProvider::new("http://localhost:11434/v1".into(), "qwen3.5:9b".into());
        assert_eq!(provider.endpoint(), "http://localhost:11434/v1");
        assert_eq!(provider.model(), "qwen3.5:9b");
        assert_eq!(provider.name(), "external");
        assert_eq!(provider.backend(), LlmBackend::Api);
        assert!(provider.is_available());
    }

    #[test]
    fn openai_compatible_provider_strips_trailing_slash() {
        let provider =
            OpenAICompatibleProvider::new("http://localhost:11434/v1/".into(), "model".into());
        assert_eq!(provider.endpoint(), "http://localhost:11434/v1");
    }

    #[test]
    fn openai_compatible_provider_not_available_when_empty() {
        let provider = OpenAICompatibleProvider::new("".into(), "".into());
        assert!(!provider.is_available());
    }

    /// Characterization test for `ReadinessHook`: the on-device LLM worker
    /// (`llm-provider-worker` std::thread at `llm_provider.rs:142`) invokes
    /// `mark_llm_ready()` from its own thread after the first successful
    /// inference. That calls the registered hook synchronously in the same
    /// std::thread, which does NOT have a Tokio reactor in thread-local context.
    ///
    /// A hook that uses bare `tokio::spawn(...)` therefore panics with
    /// "there is no reactor running, must be called from the context of a
    /// Tokio 1.x runtime" — exactly what was observed at `main.rs:420` when
    /// the panic killed the LLM worker on 2026-04-16 (see `/tmp/origin-server.log`).
    ///
    /// The correct pattern is to capture a `tokio::runtime::Handle` at hook
    /// construction time (inside an async/Tokio context) and use
    /// `handle.spawn(...)` inside the closure. `Handle` carries an explicit
    /// reference to the runtime rather than relying on thread-local context.
    #[test]
    fn readiness_hook_with_captured_handle_works_from_std_thread() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let rt = tokio::runtime::Runtime::new().unwrap();
        let handle = rt.handle().clone();
        let ran = Arc::new(AtomicBool::new(false));

        let hook: ReadinessHook = {
            let handle = handle.clone();
            let ran = ran.clone();
            Arc::new(move || {
                let ran = ran.clone();
                handle.spawn(async move {
                    ran.store(true, Ordering::SeqCst);
                });
            })
        };

        // Invoke the hook from a non-Tokio std::thread, exactly as
        // `mark_llm_ready()` does from the `llm-provider-worker` thread.
        let hook_for_thread = hook.clone();
        std::thread::spawn(move || {
            hook_for_thread();
        })
        .join()
        .expect("hook must not panic when invoked from a std::thread");

        // Give the spawned task time to run on the runtime.
        rt.block_on(async {
            for _ in 0..20 {
                if ran.load(Ordering::SeqCst) {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        });

        assert!(
            ran.load(Ordering::SeqCst),
            "hook using captured Handle should have spawned the task successfully"
        );
    }
}
