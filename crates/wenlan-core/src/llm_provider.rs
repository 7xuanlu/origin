// SPDX-License-Identifier: Apache-2.0
//! LLM provider trait and implementations.
//!
//! Provides a uniform abstraction over multiple LLM backends:
//! - [`OnDeviceProvider`] — wraps [`crate::engine::LlmEngine`] on a dedicated
//!   `std::thread` for local Qwen inference.
//! - [`ApiProvider`] — calls the Anthropic Claude API via `reqwest`.
//! - `MockProvider` — test double, only compiled under `#[cfg(test)]`.
//!
//! Also hosts shared LLM output parsing helpers (`parse_classify_response`,
//! `parse_extraction_response`, `sanitize_json_quotes`) used across the
//! refinery, search, and MCP server entry points.

use async_trait::async_trait;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::engine::LlmEngine;

// ---------------------------------------------------------------------------
// Readiness hook — fires exactly once per process when an LLM provider first
// successfully serves traffic. Used by wenlan-server to trigger the
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

/// Whether a configured provider has successfully completed warmup or served traffic.
pub fn llm_provider_ready() -> bool {
    LLM_READINESS_FIRED.get().is_some()
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
    /// Short identifier for the provider class (e.g. "on-device", "byok-api-anthropic").
    /// Used by eval report env capture to label runs without GPU access.
    fn kind(&self) -> &'static str {
        "unknown"
    }
    /// Model identifier for this provider instance. May be dynamic (user-configured).
    /// Used by eval report env capture.
    fn model_id(&self) -> String {
        "unknown".to_string()
    }
    /// Immutable model artifact version when the provider can prove one.
    fn model_version(&self) -> Option<String> {
        None
    }
    /// Runtime backend selected for an on-device provider. Non-local
    /// providers return `None`; callers can expose this as optional status
    /// without downcasting the trait object.
    fn inference_runtime_info(&self) -> Option<wenlan_types::responses::OnDeviceInferenceStatus> {
        None
    }
}

/// Build an immutable model version from the hf-hub snapshot path that was loaded.
pub(crate) fn hf_snapshot_model_version(repo_id: &str, model_path: &Path) -> Option<String> {
    if repo_id.trim().is_empty() {
        return None;
    }

    for snapshot_dir in model_path.ancestors() {
        let snapshots_dir = snapshot_dir.parent()?;
        if snapshots_dir.file_name()? != "snapshots" {
            continue;
        }

        let snapshot_hash = snapshot_dir.file_name()?.to_str()?;
        let filename = model_path
            .strip_prefix(snapshot_dir)
            .ok()?
            .to_str()?
            .replace('\\', "/");
        if snapshot_hash.is_empty() || filename.is_empty() {
            return None;
        }

        return Some(format!("hf:{repo_id}@{snapshot_hash}/{filename}"));
    }

    None
}

// ---------------------------------------------------------------------------
// OnDeviceProvider — wraps LlmEngine on a dedicated inference thread
// ---------------------------------------------------------------------------

/// Wrap a user/system prompt pair in the Qwen ChatML template.
///
/// `thinking_mode` is true for models that default to thinking-on (Qwen3.5):
/// the assistant turn then opens with an empty `<think></think>` block so no
/// output tokens go to reasoning (`strip_think_tags` would otherwise remove
/// them all). Models without a thinking mode (Qwen3-4B-Instruct-2507) must
/// not get that block: they read it as an answer that already ended and, at
/// the title job's sampling temperature, stop before the first word (greedy
/// decoding still answers). Callers outside this module go through
/// [`crate::engine::LlmEngine::format_prompt`], which knows its model.
pub(crate) fn format_chatml_prompt(
    system: Option<&str>,
    user: &str,
    thinking_mode: bool,
) -> String {
    let prefill = if thinking_mode {
        "<think>\n\n</think>\n\n"
    } else {
        ""
    };
    match system {
        Some(sys) => format!(
            "<|im_start|>system\n{sys}\n<|im_end|>\n\
             <|im_start|>user\n{user}\n<|im_end|>\n\
             <|im_start|>assistant\n{prefill}"
        ),
        None => format!(
            "<|im_start|>user\n{user}\n<|im_end|>\n\
             <|im_start|>assistant\n{prefill}"
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

const MAX_LLM_WORKERS: usize = 8;
const MAX_LLM_PARALLEL_SEQS: usize = 8;
const DEFAULT_BATCH_COALESCE_MS: u64 = 0;
const MAX_BATCH_COALESCE_MS: u64 = 25;
const CONTINUOUS_BATCH_SAFETY_RESERVE_TOKENS: usize = 32;
/// Slot-backfill drain multiplier: how many `m`-slot batches' worth of
/// immediately-available requests one continuous-batch call may pull, so the
/// engine has a queue to keep its `m` slots full (avoiding the ragged M→1 decode
/// drain). Bounds per-call latency: outputs for a drained batch return together,
/// so a drained request waits at most ~this-many slot-rounds. m≤8 → cap≤32.
const BACKFILL_DRAIN_FACTOR: usize = 4;

/// Continuous-batch slot backfill (`WENLAN_LLM_SLOT_BACKFILL`). OFF by default
/// (opt-in); enable with `1`/`true`/`yes`/`on`. Default-OFF because this is a
/// structural rewrite of the shared on-device inference path that CI cannot
/// validate across every runtime backend — it ships behind an explicit opt-in for the
/// throughput-bound paths (bulk ingest, eval seed) until staged-rollout
/// confidence accrues. When ON the coalescer may drain more than `m`
/// immediately-available requests into one call so the engine keeps all `m`
/// slots full via backfill; when OFF (the default) it caps the drain at `m` —
/// one slot per request, byte-identical to the pre-backfill behavior.
fn slot_backfill_enabled() -> bool {
    match std::env::var("WENLAN_LLM_SLOT_BACKFILL") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

fn parse_clamped_usize_env(name: &str, default: usize, min: usize, max: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn parse_clamped_u64_env(name: &str, default: u64, min: u64, max: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
        .clamp(min, max)
}

fn continuous_batch_per_seq_budget(ctx_size: u32, seq_capacity: usize) -> usize {
    (ctx_size as usize) / seq_capacity.max(1)
}

fn continuous_batch_eligible(
    ctx_size: u32,
    prompt_tokens: usize,
    max_tokens: i32,
    seq_capacity: usize,
) -> bool {
    if seq_capacity <= 1 || max_tokens <= 0 {
        return false;
    }
    let per_seq_budget = continuous_batch_per_seq_budget(ctx_size, seq_capacity);

    // The request's full footprint must fit the per-seq budget. Requests that
    // do not fit run single-seq at full context instead of being truncated.
    prompt_tokens
        .saturating_add(max_tokens as usize)
        .saturating_add(CONTINUOUS_BATCH_SAFETY_RESERVE_TOKENS)
        <= per_seq_budget
}

fn request_continuous_batch_eligible(
    engine: &LlmEngine,
    req: &InferenceRequest,
    seq_capacity: usize,
) -> bool {
    if seq_capacity <= 1 || req.max_tokens <= 0 {
        return false;
    }
    let prompt_tokens = engine.count_prompt_tokens(req.system_prompt.as_deref(), &req.prompt);
    continuous_batch_eligible(req.ctx_size, prompt_tokens, req.max_tokens, seq_capacity)
}

fn context_seq_max_for_batch(batch_len: usize, parallel_seqs: usize) -> u32 {
    if batch_len > 1 {
        parallel_seqs.max(1) as u32
    } else {
        1
    }
}

fn batch_log_call_class(label: Option<&str>) -> &str {
    label.unwrap_or("unknown")
}

/// On-device LLM provider that runs inference on a dedicated `std::thread`.
///
/// Construction downloads the model (cached), loads it using the GPU backend
/// exposed by this llama.cpp binary or CPU/OpenMP when none is available, and
/// spawns a worker thread that processes [`InferenceRequest`]s received over a
/// bounded `SyncSender` channel.
pub struct OnDeviceProvider {
    tx: std::sync::mpsc::SyncSender<InferenceRequest>,
    available: Arc<AtomicBool>,
    synthesis_limit: usize,
    model_context_size: u32,
    model_max_output: u32,
    resolved_model_id: String,
    resolved_model_version: String,
    runtime_info: wenlan_types::responses::OnDeviceInferenceStatus,
}

impl OnDeviceProvider {
    /// Probe Metal context creation with retries. ggml creates a new
    /// MTLCommandQueue per context, and macOS limits live queues per process.
    /// Retrying after a delay lets the Objective-C autorelease pool drain
    /// queues from dropped contexts.
    fn probe_metal_with_retries(engine: &crate::engine::LlmEngine, label: &str) -> bool {
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
    /// This downloads the model (if not cached), loads it using the selected
    /// runtime backend, and spawns a background thread for inference. All of
    /// this is blocking, so call from a context that can block (for example
    /// `spawn_blocking`).
    pub fn new_with_model(model_id: Option<&str>) -> Result<Self, crate::error::WenlanError> {
        let model_spec = crate::on_device_models::resolve_or_default(model_id);
        log::info!(
            "[on_device_provider] using model: {} ({})",
            model_spec.display_name,
            model_spec.id
        );
        let model_path =
            LlmEngine::download_model_by_spec(model_spec.repo_id, model_spec.filename)?;
        let resolved_model_version = hf_snapshot_model_version(model_spec.repo_id, &model_path)
            .ok_or_else(|| {
                crate::error::WenlanError::Llm(format!(
                    "model path is not an hf-hub snapshot artifact: {}",
                    model_path.display()
                ))
            })?;
        let prompts =
            crate::prompts::PromptRegistry::load(&crate::prompts::PromptRegistry::override_dir());

        // This variable controls only Metal context allocation. Set it before
        // loading the model on macOS, and never export a Metal-specific setting
        // into Windows/Linux initialization.
        #[cfg(target_os = "macos")]
        std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");

        let engine = LlmEngine::new(&model_path, prompts.clone())?
            .with_thinking_mode(model_spec.thinking_mode);
        let backend_plan = engine.inference_backend_plan().clone();
        log::info!(
            "[on_device_provider] selected inference backend: {}",
            backend_plan.label()
        );

        let engine = if backend_plan.uses_metal_compatibility_fallback() {
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
            if Self::probe_metal_with_retries(&engine, "BF16") {
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
                let fallback = LlmEngine::new(&model_path, prompts)?
                    .with_thinking_mode(model_spec.thinking_mode);
                if !Self::probe_metal_with_retries(&fallback, "BF16-disabled") {
                    log::error!(
                        "[on_device_provider] Metal context still fails with BF16 disabled \
                         after retries. On-device inference unavailable."
                    );
                    return Err(crate::error::WenlanError::Llm(
                        "Metal context creation failed even with BF16 disabled".into(),
                    ));
                }
                log::warn!(
                    "[on_device_provider] running in degraded mode (BF16 disabled, slower inference). \
                     Update macOS to restore full Metal performance."
                );
                fallback
            }
        } else {
            if let Err(cause) = engine.probe_context() {
                let message = backend_plan.context_failure_message(&cause);
                if backend_plan.supports_cpu_fallback() {
                    log::warn!("[on_device_provider] {message}; reloading the model on CPU");
                    drop(engine);
                    let fallback = LlmEngine::reload_on_cpu(&model_path, prompts, message.clone())?
                        .with_thinking_mode(model_spec.thinking_mode);
                    fallback.probe_context().map_err(|cpu_cause| {
                        crate::error::WenlanError::Llm(format!(
                            "{message}; CPU fallback context creation failed: {cpu_cause}"
                        ))
                    })?;
                    log::warn!(
                        "[on_device_provider] GPU context failed; CPU fallback context probe passed"
                    );
                    fallback
                } else {
                    log::error!("[on_device_provider] {message}. On-device inference unavailable.");
                    return Err(crate::error::WenlanError::Llm(message));
                }
            } else {
                log::info!(
                    "[on_device_provider] {} context probe passed",
                    backend_plan.label()
                );
                engine
            }
        };
        let runtime_info = engine.inference_runtime_info();

        // Channel capacity sized for concurrent MCP bursts: a 10-store
        // `remember` burst fires ~4 LLM requests per store (classify +
        // extract at handler level + entity extraction + title generation
        // at post_ingest level) = ~40 queued inference requests. The old
        // cap of 8 dropped requests with "channel send failed: sending on
        // a full channel" under the post-S2-async load, leaving memories
        // silently unclassified. 256 absorbs bursts up to ~64 concurrent
        // stores before backpressure.
        let (tx, rx) = std::sync::mpsc::sync_channel::<InferenceRequest>(256);

        // Multi-context worker pool. WENLAN_LLM_WORKERS controls the number of
        // parallel inference threads, each owning a persistent LlamaContext.
        // Workers compete on a shared receiver: lock is held only during recv(),
        // released before inference, so contention is minimal.
        //
        // WENLAN_LLM_PARALLEL_SEQS (Option B / S2) controls how many sequences
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
        // N=4 workers = 3.2 GB KV + 2.7 GB shared weights ≈ safe on a 16 GB system.
        // Default N=1, M=1 is identical to the previous single-thread behavior.
        //
        // On Metal, GGML_METAL_NO_RESIDENCY (set above) mitigates command queue
        // exhaustion from multiple simultaneous contexts. Other backends ignore it.
        let n_workers = parse_clamped_usize_env("WENLAN_LLM_WORKERS", 1, 1, MAX_LLM_WORKERS);
        let parallel_seqs =
            parse_clamped_usize_env("WENLAN_LLM_PARALLEL_SEQS", 1, 1, MAX_LLM_PARALLEL_SEQS);
        let coalesce_ms = parse_clamped_u64_env(
            "WENLAN_LLM_COALESCE_MS",
            DEFAULT_BATCH_COALESCE_MS,
            0,
            MAX_BATCH_COALESCE_MS,
        );
        log::info!(
            "[on_device_provider] {n_workers} worker(s) x {parallel_seqs} seq(s) each \
             (max concurrent = {}, coalesce_ms={coalesce_ms})",
            n_workers * parallel_seqs
        );

        let available = Arc::new(AtomicBool::new(true));

        // Wrap the receiver so it can be shared across worker threads.
        // std::sync::mpsc::Receiver is Send but not Sync, so we guard it with
        // a Mutex. Only one worker holds the lock at a time (during recv()),
        // then releases it before the actual inference call.
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
            let coalesce_wait = std::time::Duration::from_millis(coalesce_ms);
            // How many requests one continuous-batch call may drain. With slot
            // backfill ON, drain past `m` so the engine keeps all `m` slots full
            // from the queue; OFF caps at `m` (pre-backfill: one slot per req).
            let drain_cap = if slot_backfill_enabled() {
                m.saturating_mul(BACKFILL_DRAIN_FACTOR).max(m)
            } else {
                m
            };
            // Process-lifetime setting: read once here rather than per batch
            // inside the worker loop below.
            let batch_log_enabled = std::env::var_os("WENLAN_BATCH_LOG").is_some();

            std::thread::Builder::new()
                .name(format!("llm-provider-worker-{i}"))
                .spawn(move || {
                    let mut deferred_reqs: std::collections::VecDeque<InferenceRequest> =
                        std::collections::VecDeque::new();
                    log::info!(
                        "[on_device_provider] worker {i} thread started \
                         (parallel_seqs={m}, drain_cap={drain_cap})"
                    );

                    // Persistent LlamaContext: build lazily on first request,
                    // then reuse across all subsequent requests by clearing the
                    // KV cache between calls. Saves the per-call cost of
                    // `new_context()` (KV cache and backend pipeline
                    // setup) which was the dominant overhead under serial
                    // inference. If a request arrives with a different
                    // ctx_size or n_seq_max requirement, the context is rebuilt.
                    //
                    // True multi-request batches use n_seq_max=M so the
                    // continuous-batching scheduler can slot up to M sequences.
                    // Singleton requests use n_seq_max=1, including high-output
                    // synthesis/distillation calls, so they retain the full
                    // context window.
                    let mut persistent_ctx: Option<llama_cpp_2::context::LlamaContext<'_>> = None;
                    let mut persistent_ctx_size: u32 = 0;
                    let mut persistent_ctx_seq_max: u32 = 0;

                    loop {
                        // Block on the first request, then opportunistically
                        // pull up to M-1 more if M>1. By default, the follow-up
                        // drain is immediate so isolated calls do not wait for
                        // siblings; WENLAN_LLM_COALESCE_MS can opt into a tiny
                        // wait for trickle workloads.
                        let first_req = match deferred_reqs.pop_front() {
                            Some(r) => r,
                            None => match rx_arc.lock().unwrap().recv() {
                                Ok(r) => r,
                                Err(_) => break, // channel closed — sender dropped
                            },
                        };

                        let mut batch_reqs: Vec<InferenceRequest> = Vec::with_capacity(drain_cap);
                        let batch_eligible =
                            request_continuous_batch_eligible(&engine, &first_req, m);
                        batch_reqs.push(first_req);

                        if m > 1 && batch_eligible {
                            // Short coalescing window: try to drain up to
                            // `drain_cap`-1 additional pending requests. By
                            // default this is an immediate drain (0ms wait) so
                            // isolated calls pay no artificial latency. Operators
                            // can set WENLAN_LLM_COALESCE_MS for trickle
                            // workloads. With slot backfill ON, drain_cap > m so
                            // the engine keeps all m slots full from the queue.
                            let coalesce_attempts: u32 =
                                if coalesce_wait.is_zero() { 1 } else { 3 };
                            for _ in 0..coalesce_attempts {
                                if batch_reqs.len() >= drain_cap {
                                    break;
                                }
                                let drained = {
                                    let rx = rx_arc.lock().unwrap();
                                    let mut local = Vec::new();
                                    while batch_reqs.len() + local.len() < drain_cap {
                                        match rx.try_recv() {
                                            Ok(r)
                                                if request_continuous_batch_eligible(
                                                    &engine, &r, m,
                                                ) =>
                                            {
                                                local.push(r)
                                            }
                                            Ok(r) => {
                                                deferred_reqs.push_back(r);
                                                break;
                                            }
                                            Err(_) => break,
                                        }
                                    }
                                    local
                                };
                                let did_drain = !drained.is_empty();
                                batch_reqs.extend(drained);
                                if !did_drain && !coalesce_wait.is_zero() {
                                    std::thread::sleep(coalesce_wait);
                                } else if coalesce_wait.is_zero() {
                                    break;
                                }
                            }
                        }

                        // The required ctx_size is the max across the batched
                        // requests (callers normally use a single value, but be
                        // defensive). Rebuild context if it changes.
                        let target_ctx_size =
                            batch_reqs.iter().map(|r| r.ctx_size).max().unwrap_or(0);
                        let target_seq_max = context_seq_max_for_batch(batch_reqs.len(), m);

                        if persistent_ctx.is_none()
                            || persistent_ctx_size != target_ctx_size
                            || persistent_ctx_seq_max != target_seq_max
                        {
                            let _ = persistent_ctx.take();
                            persistent_ctx = engine.build_persistent_context_with_seq_max(
                                target_ctx_size,
                                target_seq_max,
                            );
                            if persistent_ctx.is_some() {
                                persistent_ctx_size = target_ctx_size;
                                persistent_ctx_seq_max = target_seq_max;
                                log::info!(
                                    "[on_device_provider] worker {i} persistent context built \
                                     (ctx_size={target_ctx_size}, n_seq_max={target_seq_max})"
                                );
                            }
                        }

                        // M=1 path: byte-identical to S1 (single-seq persistent
                        // inference). This is critical for backward compat —
                        // single-request latency must not regress.
                        if m == 1 || batch_reqs.len() == 1 {
                            let batch_n_seqs = batch_reqs.len();
                            let req = batch_reqs.pop().expect("batch has at least 1 req");
                            let full_prompt =
                                engine.format_prompt(req.system_prompt.as_deref(), &req.prompt);
                            let (timeout_secs, strip_think) = match req.timeout_secs {
                                Some(secs) => (secs, false),
                                None => (30, true),
                            };

                            // Optionally time the inference only when batch_log_enabled.
                            let t = batch_log_enabled.then(std::time::Instant::now);

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

                            if let Some(t) = t {
                                // Consumed by scripts/seed-lme-s-n90.sh:166-174, which
                                // parses this line format to compute batching rate and
                                // per-class wall-time share.
                                let call_class = batch_log_call_class(req.label.as_deref());
                                eprintln!(
                                    "[batch_log] n_seqs={} call_class={} wall_ms={}",
                                    batch_n_seqs,
                                    call_class,
                                    t.elapsed().as_millis()
                                );
                            }

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
                        let prompts: Vec<(String, i32, f32, u64, bool, Option<String>)> =
                            batch_reqs
                                .iter()
                                .map(|req| {
                                    let prompt = engine
                                        .format_prompt(req.system_prompt.as_deref(), &req.prompt);
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

                        let batch_n_seqs = batch_reqs.len();
                        let batch_call_class = batch_log_call_class(
                            batch_reqs.first().and_then(|req| req.label.as_deref()),
                        );

                        // Optionally time the inference only when batch_log_enabled.
                        let t = batch_log_enabled.then(std::time::Instant::now);

                        let outputs: Vec<Option<String>> = match persistent_ctx.as_mut() {
                            Some(ctx) => engine.run_inference_continuous_batch(ctx, &prompts, m),
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
                                    .map(
                                        |((p, max_t, temp, timeout, _strip, label), req)| match req
                                            .timeout_secs
                                        {
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
                                        },
                                    )
                                    .collect()
                            }
                        };

                        if let Some(t) = t {
                            eprintln!(
                                "[batch_log] n_seqs={} call_class={} wall_ms={}",
                                batch_n_seqs,
                                batch_call_class,
                                t.elapsed().as_millis()
                            );
                        }

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

                    // Drop the persistent context before exiting so backend
                    // resources are released cleanly.
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
                    crate::error::WenlanError::Llm(format!(
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
            resolved_model_id: model_spec.id.to_string(),
            resolved_model_version,
            runtime_info,
        })
    }

    /// Create with the default model (backward compat).
    pub fn new() -> Result<Self, crate::error::WenlanError> {
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

    fn kind(&self) -> &'static str {
        "on-device"
    }

    fn inference_runtime_info(&self) -> Option<wenlan_types::responses::OnDeviceInferenceStatus> {
        Some(self.runtime_info.clone())
    }

    fn model_id(&self) -> String {
        self.resolved_model_id.clone()
    }

    fn model_version(&self) -> Option<String> {
        Some(self.resolved_model_version.clone())
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

/// Parsed output from an LLM extraction response.
#[derive(Debug, Clone, Default)]
pub struct ExtractedFields {
    pub structured_fields: Option<String>,
    pub retrieval_cue: Option<String>,
    pub event_date: Option<i64>,
    pub event_end: Option<i64>,
}

fn parse_iso_to_unix(s: &str) -> Option<i64> {
    if let Ok(d) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp());
    }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp());
    }
    None
}

/// Parse LLM extraction output into an `ExtractedFields` struct.
/// Reuses existing extract_json + sanitize_json_quotes helpers for robust LLM output parsing.
pub fn parse_extraction_response(output: &str) -> ExtractedFields {
    let cleaned = strip_think_tags(output);
    let sanitized = sanitize_json_quotes(&cleaned);
    let json_str = match extract_json(&sanitized) {
        Some(s) => s.to_string(),
        None => return ExtractedFields::default(),
    };
    let mut json: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(_) => return ExtractedFields::default(),
    };

    let obj = match json.as_object_mut() {
        Some(o) => o,
        None => return ExtractedFields::default(),
    };

    let retrieval_cue = obj
        .remove("retrieval_cue")
        .and_then(|v| v.as_str().map(|s| s.to_string()));

    let event_date = obj
        .remove("event_date")
        .and_then(|v| v.as_str().and_then(parse_iso_to_unix));

    let event_end = obj
        .remove("event_end")
        .and_then(|v| v.as_str().and_then(parse_iso_to_unix));

    let structured_fields = if obj.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(obj.clone()).to_string())
    };

    ExtractedFields {
        structured_fields,
        retrieval_cue,
        event_date,
        event_end,
    }
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

    let space = val
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

    // T8 salience prior. Defensive parse: a JSON NUMBER is clamped to [1, 10]
    // (out-of-range numbers are a miscalibrated-but-real signal). Absent / null /
    // non-numeric (string, array, object) -> None. NEVER default to a number
    // (the silent-data-loss class behind the lost-5901-memories incident).
    let importance = val
        .get("importance")
        .and_then(|v| v.as_u64())
        .map(|n| n.clamp(1, 10) as u8);

    Some(crate::classify::ClassificationResult {
        memory_type,
        space,
        tags,
        quality,
        importance,
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

    fn kind(&self) -> &'static str {
        "byok-api-anthropic"
    }

    fn model_id(&self) -> String {
        self.model.clone()
    }
}

// ---------------------------------------------------------------------------
// OpenAICompatibleProvider — any OpenAI-compatible API (Ollama, LM Studio, etc.)
// ---------------------------------------------------------------------------

pub struct OpenAICompatibleProvider {
    endpoint: String,
    model: String,
    api_key: Option<String>,
    client: reqwest::Client,
}

impl OpenAICompatibleProvider {
    pub fn new_with_key(endpoint: String, model: String, api_key: Option<String>) -> Self {
        // Ensure endpoint doesn't have trailing slash
        let endpoint = endpoint.trim_end_matches('/').to_string();
        Self {
            endpoint,
            model,
            api_key,
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
        let mut request = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body);
        if let Some(key) = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
        {
            request = request.header("authorization", format!("Bearer {key}"));
        }
        let resp = request
            .send()
            .await
            .map_err(|e| LlmError::InferenceFailed(format!("Request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            // Name the URL that answered. A 401 from LM Studio's
            // "Require Authentication" toggle reads as "wrong endpoint"
            // without it.
            return Err(LlmError::InferenceFailed(format!(
                "API error {} from {}: {}",
                status, url, text
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

    fn kind(&self) -> &'static str {
        "byok-api-openai-compat"
    }

    fn model_id(&self) -> String {
        self.model.clone()
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

    fn kind(&self) -> &'static str {
        "subscription-cli"
    }

    fn model_id(&self) -> String {
        self.model.clone()
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

    fn kind(&self) -> &'static str {
        "mock"
    }
}

/// Test provider that returns a DISTINCT response per call, advancing a cursor.
///
/// `MockProvider` returns one fixed response forever, which can't exercise the
/// CoT retrieve-reason-retrieve loop (draft round 0, validate -> followup, draft
/// round 1, validate -> complete) where each round needs a different LLM output.
/// `SequencedMockProvider` yields `responses[i]` on the i-th call; once exhausted
/// it repeats the LAST response (so an always-followup sequence drives the
/// max_iter cap test). Counts total calls for the runaway-cost guard test.
#[cfg(test)]
pub struct SequencedMockProvider {
    responses: Vec<String>,
    cursor: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl SequencedMockProvider {
    pub fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: responses.into_iter().map(|s| s.to_string()).collect(),
            cursor: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Total number of `generate` calls served so far.
    pub fn call_count(&self) -> usize {
        self.cursor.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(test)]
#[async_trait]
impl LlmProvider for SequencedMockProvider {
    async fn generate(&self, _request: LlmRequest) -> Result<String, LlmError> {
        let i = self
            .cursor
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self.responses.is_empty() {
            return Err(LlmError::NotAvailable);
        }
        let idx = i.min(self.responses.len() - 1);
        Ok(self.responses[idx].clone())
    }
    fn is_available(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "sequenced-mock"
    }
    fn backend(&self) -> LlmBackend {
        LlmBackend::OnDevice
    }
    fn kind(&self) -> &'static str {
        "mock"
    }
}

/// Test double that returns canned responses keyed by `request.system_prompt`.
///
/// Used by WAY B Phase 1 tests (Task 1.1 split-equivalence, Task 1.5 golden
/// orchestration-equivalence) to mock the on-device entity-extraction LLM
/// deterministically, isolating orchestration logic from real inference.
///
/// # Visibility
/// Gated `#[cfg(test)]` — visible to all `#[cfg(test)]` unit-test modules in
/// this crate (including `entity_extraction.rs` for Task 1.1). Task 1.5's
/// golden test must therefore be a `#[cfg(test)]` unit-test module in `src/`
/// (not under `tests/`), because `#[cfg(test)]` items in `src/` are NOT
/// visible to integration tests under `tests/`. No `test-support` feature
/// exists in `Cargo.toml`; the `#[cfg(test)]`-unit-test route is the choice.
#[cfg(test)]
pub struct CannedLlmProvider {
    default: String,
    map: std::collections::HashMap<String, String>,
    /// If any entry here is a substring of `request.user_prompt`, return an error.
    fail_on_user_substrings: Vec<String>,
}

#[cfg(test)]
impl CannedLlmProvider {
    pub fn new(default: impl Into<String>) -> Self {
        Self {
            default: default.into(),
            map: std::collections::HashMap::new(),
            fail_on_user_substrings: Vec::new(),
        }
    }

    /// Register a canned response for any `system_prompt` that equals or
    /// contains `key`.
    pub fn with(mut self, key: impl Into<String>, response: impl Into<String>) -> Self {
        self.map.insert(key.into(), response.into());
        self
    }

    /// Return `Err(LlmError::RequestFailed)` for any call whose `user_prompt`
    /// contains `substr`. Used in error-path tests to fail exactly one memory's
    /// extract without keying on the system prompt (which is shared across all
    /// memories in the batch). Only affects `generate`; `is_available` stays true.
    pub fn fail_on(mut self, substr: impl Into<String>) -> Self {
        self.fail_on_user_substrings.push(substr.into());
        self
    }
}

#[cfg(test)]
#[async_trait]
impl LlmProvider for CannedLlmProvider {
    async fn generate(&self, request: LlmRequest) -> Result<String, LlmError> {
        // Error injection: fail before checking the map so the caller sees Err.
        for substr in &self.fail_on_user_substrings {
            if request.user_prompt.contains(substr.as_str()) {
                return Err(LlmError::InferenceFailed(format!(
                    "canned error: user_prompt contains {:?}",
                    substr
                )));
            }
        }
        if let Some(sys) = &request.system_prompt {
            for (key, response) in &self.map {
                if sys == key || sys.contains(key.as_str()) {
                    return Ok(response.clone());
                }
            }
        }
        Ok(self.default.clone())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "canned"
    }

    fn backend(&self) -> LlmBackend {
        LlmBackend::OnDevice
    }

    fn kind(&self) -> &'static str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_snapshot_model_version_includes_repo_snapshot_and_filename() {
        let path = Path::new(
            "/cache/huggingface/hub/models--Qwen--Qwen3/snapshots/012345abcdef/models/qwen.gguf",
        );

        assert_eq!(
            hf_snapshot_model_version("Qwen/Qwen3", path).as_deref(),
            Some("hf:Qwen/Qwen3@012345abcdef/models/qwen.gguf")
        );
        assert!(
            hf_snapshot_model_version("Qwen/Qwen3", Path::new("/cache/blobs/qwen.gguf")).is_none()
        );
    }

    #[tokio::test]
    async fn canned_llm_returns_by_prompt() {
        let provider = CannedLlmProvider::new("DEFAULT").with("kg-system", "{\"entities\":[]}");

        // Registered key → canned response.
        let req_match = LlmRequest {
            system_prompt: Some("kg-system".into()),
            user_prompt: "extract".into(),
            max_tokens: 64,
            temperature: 0.0,
            label: None,
            timeout_secs: None,
        };
        assert_eq!(
            provider.generate(req_match).await.unwrap(),
            "{\"entities\":[]}"
        );

        // Unregistered prompt → default.
        let req_default = LlmRequest {
            system_prompt: Some("unregistered".into()),
            user_prompt: "extract".into(),
            max_tokens: 64,
            temperature: 0.0,
            label: None,
            timeout_secs: None,
        };
        assert_eq!(provider.generate(req_default).await.unwrap(), "DEFAULT");

        // No system prompt → default.
        let req_no_sys = LlmRequest {
            system_prompt: None,
            user_prompt: "extract".into(),
            max_tokens: 64,
            temperature: 0.0,
            label: None,
            timeout_secs: None,
        };
        assert_eq!(provider.generate(req_no_sys).await.unwrap(), "DEFAULT");

        // Metadata checks.
        assert!(provider.is_available());
        assert_eq!(provider.name(), "canned");
        assert_eq!(provider.backend(), LlmBackend::OnDevice);
    }

    #[test]
    fn test_format_chatml_prompt_with_system() {
        let p = format_chatml_prompt(Some("you are helpful"), "hi", true);
        assert!(p.contains("<|im_start|>system\nyou are helpful\n<|im_end|>"));
        assert!(p.contains("<|im_start|>user\nhi\n<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_format_chatml_prompt_no_system() {
        let p = format_chatml_prompt(None, "hello world", true);
        assert!(!p.contains("<|im_start|>system"));
        assert!(p.contains("<|im_start|>user\nhello world\n<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn test_format_chatml_prompt_without_thinking_mode_has_no_think_block() {
        let p = format_chatml_prompt(Some("you are helpful"), "hi", false);
        assert!(!p.contains("<think>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
        let p = format_chatml_prompt(None, "hi", false);
        assert!(!p.contains("<think>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
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
                .clamp(1, MAX_LLM_PARALLEL_SEQS);
            assert_eq!(
                parsed, expected,
                "input {input:?} should clamp to {expected}"
            );
        }
    }

    #[test]
    fn test_continuous_batch_eligibility_keeps_long_outputs_single_seq() {
        // 8K context / 8 seq slots = 1024 tokens per seq. Entity extraction
        // style calls fit; distillation/chat synthesis calls keep the full
        // single-seq context instead of being silently budget-truncated.
        assert!(continuous_batch_eligible(8192, 64, 128, 8));
        assert!(continuous_batch_eligible(8192, 64, 512, 8));
        assert!(!continuous_batch_eligible(8192, 64, 1024, 8));
        assert!(!continuous_batch_eligible(8192, 64, 2048, 8));
        assert!(!continuous_batch_eligible(8192, 64, 128, 1));
    }

    #[test]
    fn long_context_prompt_not_batched_even_with_small_output() {
        // The old bug: a context-heavy prompt (900 tok) with modest output (512) passed
        // the fixed-256 reserve gate (512+256<=1024) and was then silently truncated.
        // Now the ACTUAL prompt length is accounted: 900+512+32 > 1024 => NOT eligible
        // => runs single-seq at full context.
        assert!(!continuous_batch_eligible(8192, 900, 512, 8));
        // The same request fits at M=4 (budget 2048): 900+512+32 <= 2048.
        assert!(continuous_batch_eligible(8192, 900, 512, 4));
        // A short prompt still batches at M=8.
        assert!(continuous_batch_eligible(8192, 200, 256, 8));
        // M=1 never batches.
        assert!(!continuous_batch_eligible(8192, 10, 10, 1));
    }

    #[test]
    fn test_continuous_batch_budget_uses_configured_capacity() {
        assert_eq!(continuous_batch_per_seq_budget(8192, 8), 1024);
        assert_eq!(continuous_batch_per_seq_budget(8192, 4), 2048);
        assert_eq!(continuous_batch_per_seq_budget(8192, 0), 8192);
    }

    #[test]
    fn test_singleton_requests_use_single_seq_context() {
        assert_eq!(context_seq_max_for_batch(0, 8), 1);
        assert_eq!(context_seq_max_for_batch(1, 8), 1);
        assert_eq!(context_seq_max_for_batch(2, 8), 8);
        assert_eq!(context_seq_max_for_batch(8, 8), 8);
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
        let provider = ApiProvider::new(String::new(), DEFAULT_ROUTINE_MODEL.to_string());
        assert!(!provider.is_available());
        assert_eq!(provider.name(), "api");
    }

    #[test]
    fn test_api_provider_with_key() {
        let provider =
            ApiProvider::new("sk-ant-test".to_string(), DEFAULT_ROUTINE_MODEL.to_string());
        assert!(provider.is_available());
    }

    #[tokio::test]
    async fn test_api_provider_empty_key_fails() {
        let provider = ApiProvider::new(String::new(), DEFAULT_ROUTINE_MODEL.to_string());
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
        let out = parse_extraction_response(output);
        assert!(out.structured_fields.is_some());
        assert!(out.retrieval_cue.is_some());
        let fields_json: serde_json::Value =
            serde_json::from_str(&out.structured_fields.unwrap()).unwrap();
        assert_eq!(fields_json["claim"], "I love Rust");
        assert!(out.retrieval_cue.unwrap().contains("Rust"));
    }

    #[test]
    fn test_parse_extraction_response_with_think_tags() {
        let output = "<think>reasoning</think>{\"claim\": \"test\", \"retrieval_cue\": \"q?\"}";
        let out = parse_extraction_response(output);
        assert!(out.structured_fields.is_some());
        assert!(out.retrieval_cue.is_some());
    }

    #[test]
    fn test_parse_extraction_response_invalid() {
        let out = parse_extraction_response("not json at all");
        assert!(out.structured_fields.is_none());
        assert!(out.retrieval_cue.is_none());
    }

    #[test]
    fn test_parse_extraction_response_empty_fields() {
        let output = r#"{"retrieval_cue": "just a cue"}"#;
        let out = parse_extraction_response(output);
        assert!(out.structured_fields.is_none()); // No actual fields, just cue
        assert!(out.retrieval_cue.is_some());
    }

    #[test]
    fn parse_extraction_response_returns_extracted_fields_struct() {
        let raw = r#"{"claim": "user prefers dark mode", "event_date": "2026-05-26", "retrieval_cue": "What does the user prefer about UI theme?"}"#;
        let out = parse_extraction_response(raw);
        assert_eq!(
            out.retrieval_cue.as_deref(),
            Some("What does the user prefer about UI theme?")
        );
        assert!(out
            .structured_fields
            .as_ref()
            .unwrap()
            .contains("\"claim\""));
        // event_date should be stripped from structured_fields and parsed into its own slot.
        assert!(!out
            .structured_fields
            .as_ref()
            .unwrap()
            .contains("event_date"));
        let expected_ts = chrono::NaiveDate::from_ymd_opt(2026, 5, 26)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        assert_eq!(out.event_date, Some(expected_ts));
        assert_eq!(out.event_end, None);
    }

    #[test]
    fn parse_extraction_response_handles_event_end() {
        let raw = r#"{"claim": "trip lasted a week", "event_date": "2026-05-01", "event_end": "2026-05-08", "retrieval_cue": "When was the trip?"}"#;
        let out = parse_extraction_response(raw);
        assert!(out.event_date.is_some());
        assert!(out.event_end.is_some());
    }

    #[test]
    fn parse_extraction_response_missing_dates_returns_none() {
        let raw = r#"{"claim": "user prefers dark mode", "retrieval_cue": "..."}"#;
        let out = parse_extraction_response(raw);
        assert_eq!(out.event_date, None);
        assert_eq!(out.event_end, None);
    }

    #[test]
    fn test_parse_classify_response() {
        let raw =
            r#"{"memory_type": "preference", "domain": "technology", "tags": ["dark mode", "ui"]}"#;
        let result = parse_classify_response(raw).unwrap();
        assert_eq!(result.memory_type, "preference");
        assert_eq!(result.space, Some("technology".to_string()));
        assert_eq!(result.tags, vec!["dark mode", "ui"]);
    }

    #[test]
    fn test_parse_classify_response_with_think_tags() {
        let raw =
            r#"<think>analyzing...</think>{"memory_type": "fact", "domain": "work", "tags": []}"#;
        let result = parse_classify_response(raw).unwrap();
        assert_eq!(result.memory_type, "fact");
        assert_eq!(result.space, Some("work".to_string()));
    }

    #[test]
    fn test_parse_classify_response_invalid() {
        assert!(parse_classify_response("not json at all").is_none());
    }

    // --- T8 importance parse (L2 tests) ---

    #[test]
    fn parse_classify_extracts_importance() {
        let c = parse_classify_response(
            r#"{"memory_type": "fact", "domain": "x", "tags": [], "importance": 8}"#,
        )
        .unwrap();
        assert_eq!(c.importance, Some(8));
    }

    #[test]
    fn parse_classify_importance_clamps() {
        // >10 clamps to 10.
        let hi =
            parse_classify_response(r#"{"memory_type": "fact", "tags": [], "importance": 99}"#)
                .unwrap();
        assert_eq!(hi.importance, Some(10));
        // 0 is below the [1,10] band -> clamps to 1 (a present number is a real,
        // if miscalibrated, signal). Only non-numeric / absent values stay None.
        let lo = parse_classify_response(r#"{"memory_type": "fact", "tags": [], "importance": 0}"#)
            .unwrap();
        assert_eq!(lo.importance, Some(1));
    }

    #[test]
    fn parse_classify_importance_absent_is_none() {
        // No importance key -> None (NEVER a default number).
        let c = parse_classify_response(r#"{"memory_type": "fact", "tags": []}"#).unwrap();
        assert_eq!(c.importance, None);
    }

    #[test]
    fn parse_classify_importance_malformed_is_none() {
        // String, null, and array all -> None, no error (silent-zero-class guard).
        for body in [
            r#"{"memory_type": "fact", "tags": [], "importance": "high"}"#,
            r#"{"memory_type": "fact", "tags": [], "importance": null}"#,
            r#"{"memory_type": "fact", "tags": [], "importance": []}"#,
        ] {
            let c = parse_classify_response(body).unwrap();
            assert_eq!(c.importance, None, "body={body}");
        }
    }

    #[test]
    fn openai_compatible_provider_stores_config() {
        let provider = OpenAICompatibleProvider::new_with_key(
            "http://localhost:11434/v1".into(),
            "qwen3.5:9b".into(),
            None,
        );
        assert_eq!(provider.endpoint(), "http://localhost:11434/v1");
        assert_eq!(provider.model(), "qwen3.5:9b");
        assert_eq!(provider.name(), "external");
        assert_eq!(provider.backend(), LlmBackend::Api);
        assert!(provider.is_available());
    }

    #[test]
    fn openai_compatible_provider_strips_trailing_slash() {
        let provider = OpenAICompatibleProvider::new_with_key(
            "http://localhost:11434/v1/".into(),
            "model".into(),
            None,
        );
        assert_eq!(provider.endpoint(), "http://localhost:11434/v1");
    }

    #[test]
    fn openai_compatible_provider_not_available_when_empty() {
        let provider = OpenAICompatibleProvider::new_with_key("".into(), "".into(), None);
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
    /// the panic killed the LLM worker on 2026-04-16 (see `/tmp/wenlan-server.log`).
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

    #[test]
    fn on_device_provider_model_id_reflects_resolved_model() {
        // resolve_or_default is the same path new_with_model uses to set resolved_model_id.
        assert_eq!(
            crate::on_device_models::resolve_or_default(Some("qwen3.5-9b")).id,
            "qwen3.5-9b"
        );
    }
}
