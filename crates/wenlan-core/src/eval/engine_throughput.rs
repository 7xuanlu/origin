//! On-device continuous-batch **slot-backfill** throughput benchmark + KV-reuse
//! correctness oracle (L7 manual, needs the GPU + a downloaded GGUF model —
//! `#[ignore]`d so it never runs in CI).
//!
//! Wires the measured engine-perf speedup into the eval harness. The on-device
//! enrichment batch is decode-bound (`[batch_timing]` measured ~40% prefill /
//! ~59% decode): a continuous batch mixes short outputs (classify ~20 tok) with
//! long ones (entity ~100 tok), so decode width drains raggedly M→1 as the short
//! ones finish first. Slot backfill keeps all M slots full from a queue, holding
//! decode width near M until the queue drains.
//!
//! Two GPU tests:
//! - [`run_backfill_throughput_ab`] contrasts the **same total work** scheduled
//!   two ways: OFF = the queue in M-sized chunks (one call per chunk, ragged
//!   drain), ON = the whole queue in one call (the engine backfills freed slots).
//!   The headline is the wall-clock ratio `OFF / ON`.
//! - [`first_grounding_violation`] is the KV-reuse **correctness oracle**. Byte
//!   equivalence is the WRONG oracle for batched GPU inference — it is confounded
//!   by per-request sampler seeds (`42 + request_index`, which reslicing changes)
//!   and by batch-shape float non-determinism flipping near-tied tokens; both
//!   produce benign reorderings of identical content. The real backfill risk is
//!   different: a slot reused without clearing its KV would make a request attend
//!   to a PREVIOUS occupant's context and emit *that* prompt's entities. So each
//!   request carries a distinctive **canary** proper noun, and the oracle asserts
//!   every backfilled extract output reflects its OWN prompt's canary and leaks no
//!   FOREIGN canary — catching cross-slot KV contamination regardless of sampling.

use crate::engine::LlmEngine;
use llama_cpp_2::context::LlamaContext;
use std::time::Instant;

/// A continuous-batch prompt tuple, matching
/// `LlmEngine::run_inference_continuous_batch`'s input:
/// `(prompt, max_output_tokens, temperature, timeout_secs, strip_think, label)`.
pub type BatchPrompt = (String, i32, f32, u64, bool, Option<String>);

/// One synthetic request plus the metadata the grounding oracle needs.
#[derive(Debug, Clone)]
pub struct WorkloadItem {
    pub prompt: BatchPrompt,
    /// Distinctive proper noun from THIS request's prompt; an extract output must
    /// surface it (proves the request attended to its own KV, not a neighbour's).
    pub canary: String,
    /// Extract-style request (asks for entities, so the canary should appear).
    /// Classify-style requests emit a sentiment word and carry no grounding canary.
    pub is_extract: bool,
}

/// One arm's measurement.
#[derive(Debug, Clone)]
pub struct ThroughputArm {
    /// Wall-clock for the whole workload (ms).
    pub wall_ms: u128,
    /// Total output characters produced (work proxy; token counts are logged by
    /// `WENLAN_BATCH_LOG`'s `[batch_timing]` line, not returned by the engine).
    pub out_chars: usize,
    /// Requests that produced output (non-`None`).
    pub completed: usize,
}

/// Paired ON-vs-OFF throughput result for a single workload.
#[derive(Debug, Clone)]
pub struct ThroughputAb {
    pub on: ThroughputArm,
    pub off: ThroughputArm,
    pub n_requests: usize,
    pub m: usize,
}

impl ThroughputAb {
    /// Wall-clock speedup of backfill ON over OFF (`>1.0` = faster). Saturates to
    /// `1.0` if the ON arm somehow measured 0ms (sub-millisecond / degenerate).
    pub fn speedup(&self) -> f64 {
        if self.on.wall_ms == 0 {
            return 1.0;
        }
        self.off.wall_ms as f64 / self.on.wall_ms as f64
    }
}

/// A request whose output is not grounded in its own prompt.
#[derive(Debug, Clone)]
pub struct GroundingViolation {
    pub request: usize,
    pub reason: String,
    pub output: Option<String>,
}

fn out_chars(outputs: &[Option<String>]) -> usize {
    outputs.iter().flatten().map(|s| s.len()).sum()
}

fn completed(outputs: &[Option<String>]) -> usize {
    outputs.iter().flatten().count()
}

/// Run the same `prompts` two ways and time each: OFF = `m`-sized chunks (one
/// engine call per chunk, ragged drain), ON = one engine call over the whole
/// queue (slot backfill). The engine clears its KV cache at the start of every
/// call, so reusing `ctx` across both arms is safe.
pub fn run_backfill_throughput_ab(
    engine: &LlmEngine,
    ctx: &mut LlamaContext<'_>,
    prompts: &[BatchPrompt],
    m: usize,
) -> ThroughputAb {
    // OFF: run in m-chunks (each chunk <= m => no backfill => ragged drain).
    let off_start = Instant::now();
    let mut off_outputs: Vec<Option<String>> = Vec::with_capacity(prompts.len());
    for chunk in prompts.chunks(m.max(1)) {
        off_outputs.extend(engine.run_inference_continuous_batch(ctx, chunk, m));
    }
    let off_wall = off_start.elapsed().as_millis();

    // ON: one call over the whole queue (engine backfills freed slots).
    let on_start = Instant::now();
    let on_outputs = engine.run_inference_continuous_batch(ctx, prompts, m);
    let on_wall = on_start.elapsed().as_millis();

    ThroughputAb {
        on: ThroughputArm {
            wall_ms: on_wall,
            out_chars: out_chars(&on_outputs),
            completed: completed(&on_outputs),
        },
        off: ThroughputArm {
            wall_ms: off_wall,
            out_chars: out_chars(&off_outputs),
            completed: completed(&off_outputs),
        },
        n_requests: prompts.len(),
        m,
    }
}

/// KV-reuse correctness oracle. Runs the whole queue ONCE with slot backfill
/// engaged (`prompts.len() > m`), then checks every extract request's output is
/// grounded in its OWN prompt: it must surface its own canary and must not leak
/// any other request's canary. A slot reused without clearing its KV would make a
/// backfilled request attend to its predecessor's context and emit the wrong
/// canary — exactly what this catches, independent of sampler seed / float noise.
/// Returns the first violation, or `None` if every extract request is grounded.
pub fn first_grounding_violation(
    engine: &LlmEngine,
    ctx: &mut LlamaContext<'_>,
    items: &[WorkloadItem],
    m: usize,
) -> Option<GroundingViolation> {
    let prompts: Vec<BatchPrompt> = items.iter().map(|it| it.prompt.clone()).collect();
    let outputs = engine.run_inference_continuous_batch(ctx, &prompts, m);

    for (i, (item, out)) in items.iter().zip(&outputs).enumerate() {
        if !item.is_extract {
            continue; // classify requests carry no entity canary
        }
        let Some(text) = out else {
            return Some(GroundingViolation {
                request: i,
                reason: "extract request produced no output".to_string(),
                output: None,
            });
        };
        let low = text.to_lowercase();
        let own = item.canary.to_lowercase();
        if !low.contains(&own) {
            return Some(GroundingViolation {
                request: i,
                reason: format!(
                    "missing own canary '{}' (not grounded in own prompt)",
                    item.canary
                ),
                output: Some(text.clone()),
            });
        }
        // Leakage: a DIFFERENT request's distinctive canary appearing here can
        // only come from cross-slot KV contamination (each prompt contains only
        // its own sentence). Same-canary repeats (i and i+8) are skipped.
        for (j, other) in items.iter().enumerate() {
            if j == i {
                continue;
            }
            let o = other.canary.to_lowercase();
            if o != own && low.contains(&o) {
                return Some(GroundingViolation {
                    request: i,
                    reason: format!("leaked foreign canary '{}' from request {j}", other.canary),
                    output: Some(text.clone()),
                });
            }
        }
    }
    None
}

/// Build a synthetic mixed-length workload: alternating short "classify"-style
/// requests (one-word sentiment output) and long "extract"-style requests (entity
/// JSON). The output-length heterogeneity is what makes a non-backfilled batch
/// drain raggedly. Each base sentence leads with a DISTINCTIVE proper-noun canary
/// (the grammatical subject, so the extractor surfaces it early) for the grounding
/// oracle. Prompts are chatml-formatted exactly like the enrichment path.
pub fn synthetic_mixed_workload(n: usize, temperature: f32) -> Vec<WorkloadItem> {
    // (canary subject, sentence). Canaries are distinctive enough that a foreign
    // canary in an output can only come from KV cross-contamination.
    let base = [
        (
            "Lavazza",
            "Lavazza shipped the new espresso machine and the office finally has great coffee.",
        ),
        (
            "Berlin",
            "Berlin's airport delayed my flight three hours and I missed the connection.",
        ),
        (
            "Pixel",
            "Pixel, the rescue terrier we adopted, already learned to sit and stay.",
        ),
        (
            "Acme",
            "Acme reported quarterly numbers well below forecast again this year.",
        ),
        (
            "Tamalpais",
            "Tamalpais looked stunning at dawn as the fog burned off the valley.",
        ),
        (
            "Okafor",
            "Okafor, the landlord, still has not fixed the leaking radiator in the hall.",
        ),
        (
            "Diego",
            "Diego migrated the backend from Postgres to SQLite for the local build.",
        ),
        (
            "Beatrix",
            "Beatrix insists the cake needs exactly two eggs, never one.",
        ),
    ];
    (0..n)
        .map(|i| {
            let (canary, s) = base[i % base.len()];
            if i % 2 == 0 {
                // Short classify-style: ~one-word output.
                let user = format!(
                    "Reply with exactly one word — positive, negative, or neutral — for the \
                     overall sentiment of this sentence and nothing else:\n\"{s}\""
                );
                let prompt = crate::llm_provider::format_chatml_prompt(
                    Some("You are a terse sentiment classifier."),
                    &user,
                    crate::on_device_models::get_default_model().thinking_mode,
                );
                WorkloadItem {
                    prompt: (
                        prompt,
                        24,
                        temperature,
                        30,
                        true,
                        Some("classify".to_string()),
                    ),
                    canary: canary.to_string(),
                    is_extract: false,
                }
            } else {
                // Long extract-style: multi-entity JSON output. The subject canary
                // is the first entity, so it appears well within the token budget.
                let user = format!(
                    "Extract every person, place, organization, and object mentioned, plus any \
                     relations between them, as a JSON array of objects. Output only JSON:\n\"{s}\""
                );
                let prompt = crate::llm_provider::format_chatml_prompt(
                    Some("You extract knowledge-graph entities and relations as JSON."),
                    &user,
                    crate::on_device_models::get_default_model().thinking_mode,
                );
                WorkloadItem {
                    prompt: (
                        prompt,
                        128,
                        temperature,
                        30,
                        true,
                        Some("extract".to_string()),
                    ),
                    canary: canary.to_string(),
                    is_extract: true,
                }
            }
        })
        .collect()
}

/// Build a workload that SHARES one long instruction prefix across every request
/// (the prefix-KV cache's target shape: a homogeneous enrichment burst). All items
/// use an identical ~system prompt + identical user task framing, then a per-item
/// distinctive sentence of VARYING length. The length variance (and a queue longer
/// than `m`) exercises slot backfill, the `min_len-1` cap, and the short-backfilled-
/// request edge case together. The shared prefix is well over `PREFIX_KV_MIN_TOKENS`,
/// so the cache engages. Prompts are chatml-formatted exactly like the real path.
pub fn shared_prefix_workload(n: usize, temperature: f32) -> Vec<WorkloadItem> {
    // One fixed instruction shared by every request — this is the cacheable prefix.
    const SYSTEM: &str = "You are a precise information extractor. Read the user's \
        sentence and reply with a compact JSON object containing the single most \
        important named entity and a one-word topic. Output only JSON, no prose, no \
        markdown fences, no explanation. Always include both keys.";
    // (canary, sentence) pairs of deliberately varying length — the shortest keeps
    // the overall min length close to the shared prefix to test the cap.
    let base = [
        ("Lavazza", "Lavazza shipped espresso machines to the Milan office this spring after a long delay."),
        ("Berlin", "Berlin rained."),
        ("Pixel", "Pixel the rescue terrier learned to sit, stay, and roll over within two short weeks."),
        ("Acme", "Acme missed forecast."),
        ("Tamalpais", "Tamalpais glowed at dawn while fog slowly burned off the valley below the ridgeline trail."),
        ("Okafor", "Okafor fixed it."),
        ("Diego", "Diego migrated the entire backend from Postgres to SQLite for the offline local build target."),
        ("Beatrix", "Beatrix baked two cakes."),
    ];
    (0..n)
        .map(|i| {
            let (canary, s) = base[i % base.len()];
            let user = format!("Sentence: \"{s}\"");
            let prompt = crate::llm_provider::format_chatml_prompt(
                Some(SYSTEM),
                &user,
                crate::on_device_models::get_default_model().thinking_mode,
            );
            // Vary the output budget too, so decode width drains raggedly.
            let max_out = if i % 2 == 0 { 24 } else { 64 };
            WorkloadItem {
                prompt: (
                    prompt,
                    max_out,
                    temperature,
                    30,
                    true,
                    Some("extract".to_string()),
                ),
                canary: canary.to_string(),
                is_extract: true,
            }
        })
        .collect()
}

/// Output divergence between the prefix-KV cache OFF and ON for the SAME request
/// set / same `m` / same seeds. Because the cache changes neither scheduling nor
/// sampler seeds nor batch shape, the ON output must be byte-identical to OFF (the
/// copied prefix KV is mathematically the prefix's fresh KV) — so plain byte
/// equivalence is the correct oracle here (unlike slot backfill). At temperature
/// 0.0 sampling is greedy argmax: stable under benign sub-ULP float noise, but a
/// real wrong-logits-row or KV-corruption bug flips the argmax and diverges loudly.
/// Returns the first `(request_index, off_output, on_output)` that differs.
///
/// Toggles `WENLAN_LLM_PREFIX_KV_CACHE` around each arm. The engine clears its KV
/// at the start of every call, so reusing `ctx` across both arms is safe.
pub fn first_prefix_cache_divergence(
    engine: &LlmEngine,
    ctx: &mut LlamaContext<'_>,
    prompts: &[BatchPrompt],
    m: usize,
) -> Option<(usize, Option<String>, Option<String>)> {
    std::env::remove_var("WENLAN_LLM_PREFIX_KV_CACHE");
    let off = engine.run_inference_continuous_batch(ctx, prompts, m);
    std::env::set_var("WENLAN_LLM_PREFIX_KV_CACHE", "1");
    let on = engine.run_inference_continuous_batch(ctx, prompts, m);
    std::env::remove_var("WENLAN_LLM_PREFIX_KV_CACHE");
    off.iter()
        .zip(&on)
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| (i, a.clone(), b.clone()))
}

/// Prefix-KV cache ON-vs-OFF over the WHOLE queue in one call each (identical
/// scheduling; only the flag differs). Wall-clock barely moves when decode
/// dominates — the real signal is `prefill_ms`/`prime_ms` from the `[batch_timing]`
/// line (set `WENLAN_BATCH_LOG=1`). This A/B asserts only no-regression; the
/// prefill magnitude is read from the logs.
pub fn run_prefix_cache_throughput_ab(
    engine: &LlmEngine,
    ctx: &mut LlamaContext<'_>,
    prompts: &[BatchPrompt],
    m: usize,
) -> ThroughputAb {
    std::env::remove_var("WENLAN_LLM_PREFIX_KV_CACHE");
    let off_start = Instant::now();
    let off_outputs = engine.run_inference_continuous_batch(ctx, prompts, m);
    let off_wall = off_start.elapsed().as_millis();

    std::env::set_var("WENLAN_LLM_PREFIX_KV_CACHE", "1");
    let on_start = Instant::now();
    let on_outputs = engine.run_inference_continuous_batch(ctx, prompts, m);
    let on_wall = on_start.elapsed().as_millis();
    std::env::remove_var("WENLAN_LLM_PREFIX_KV_CACHE");

    ThroughputAb {
        on: ThroughputArm {
            wall_ms: on_wall,
            out_chars: out_chars(&on_outputs),
            completed: completed(&on_outputs),
        },
        off: ThroughputArm {
            wall_ms: off_wall,
            out_chars: out_chars(&off_outputs),
            completed: completed(&off_outputs),
        },
        n_requests: prompts.len(),
        m,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompts::PromptRegistry;

    const CTX_SIZE: u32 = 4096; // 4096 / m=8 = 512 tok/slot — fits classify+extract.
    const M: usize = 8;

    fn build_engine_and_ctx() -> Option<(&'static LlmEngine, LlamaContext<'static>)> {
        let model_path = LlmEngine::download_model().ok()?;
        let engine = LlmEngine::new(&model_path, PromptRegistry::default()).ok()?;
        // The ctx borrows the engine (via the loaded model). Leak the engine to
        // 'static so the ctx can outlive this helper; the test process exits
        // right after, so the one-time leak is harmless.
        let engine: &'static LlmEngine = Box::leak(Box::new(engine));
        let ctx = engine.build_persistent_context_with_seq_max(CTX_SIZE, M as u32)?;
        Some((engine, ctx))
    }

    /// Leak the context instead of dropping it, to dodge the per-`LlamaContext`
    /// `GGML_ASSERT([rsets->data count] == 0)` that llama-cpp-2 0.1.143's
    /// ggml-metal backend raises during residency-set teardown. NOTE: a second,
    /// process-level variant of the same assert can still fire during ggml-metal
    /// DEVICE teardown at process exit, AFTER the test has already reported
    /// `test result: ok` — that trailing SIGABRT is teardown noise, not a test
    /// failure. Read the `test result:` / `grounding OK` / `speedup` lines as
    /// authoritative; these tests are `#[ignore]`d so the abort never reaches CI.
    fn leak_ctx(ctx: LlamaContext<'static>) {
        std::mem::forget(ctx);
    }

    /// KV slot-reuse correctness: with backfill engaged (19 requests over M=8
    /// slots → 11 backfills), every extract output must be grounded in its own
    /// prompt's canary and leak no foreign canary. Run locally on Metal:
    /// `cargo test -p wenlan-core --lib eval::engine_throughput::tests::backfill_grounding -- --ignored --nocapture`
    #[test]
    #[ignore = "needs GPU + downloaded GGUF model (L7 manual)"]
    fn backfill_grounding() {
        let Some((engine, mut ctx)) = build_engine_and_ctx() else {
            eprintln!("[engine_throughput] engine unavailable (no GPU/model) — skipping");
            return;
        };
        let items = synthetic_mixed_workload(19, 0.0);
        let result = first_grounding_violation(engine, &mut ctx, &items, M);
        leak_ctx(ctx);
        match result {
            None => eprintln!(
                "[engine_throughput] grounding OK: every backfilled extract output grounded in its own prompt ({} requests, M={M})",
                items.len()
            ),
            Some(v) => panic!(
                "slot backfill corrupted request {}: {}\n  output: {:?}",
                v.request, v.reason, v.output
            ),
        }
    }

    /// Throughput A/B: report the wall-clock speedup of slot backfill ON vs OFF
    /// on a mixed M=8 workload. Run locally on Metal:
    /// `cargo test -p wenlan-core --lib eval::engine_throughput::tests::backfill_throughput_ab -- --ignored --nocapture`
    #[test]
    #[ignore = "needs GPU + downloaded GGUF model (L7 manual)"]
    fn backfill_throughput_ab() {
        let Some((engine, mut ctx)) = build_engine_and_ctx() else {
            eprintln!("[engine_throughput] engine unavailable (no GPU/model) — skipping");
            return;
        };
        let n = 40; // 5 m-rounds of mixed short/long output.
        let items = synthetic_mixed_workload(n, 0.7);
        let prompts: Vec<BatchPrompt> = items.iter().map(|it| it.prompt.clone()).collect();
        let ab = run_backfill_throughput_ab(engine, &mut ctx, &prompts, M);
        leak_ctx(ctx);
        eprintln!(
            "[engine_throughput] n={} m={}\n  OFF (m-chunks):  {} ms, {} chars, {} ok\n  ON  (backfill):  {} ms, {} chars, {} ok\n  speedup (OFF/ON): {:.2}x",
            ab.n_requests, ab.m,
            ab.off.wall_ms, ab.off.out_chars, ab.off.completed,
            ab.on.wall_ms, ab.on.out_chars, ab.on.completed,
            ab.speedup(),
        );
        // Backfill must not be slower than the ragged baseline (allow a small
        // noise margin); the headline magnitude is reported above, not asserted.
        assert!(
            ab.speedup() >= 0.95,
            "slot backfill regressed throughput: {:.2}x (OFF {} ms / ON {} ms)",
            ab.speedup(),
            ab.off.wall_ms,
            ab.on.wall_ms
        );
    }

    /// Prefix-KV cache CORRECTNESS oracle: with a long shared instruction prefix
    /// and a queue (19) longer than M=8 (so backfill engages, 11 backfills), the
    /// cache ON must produce output BYTE-IDENTICAL to OFF — the copied prefix KV is
    /// mathematically the prefix's fresh KV. Greedy (temp 0.0) so a real wrong-
    /// logits-row / KV-corruption bug flips the argmax and is caught, while benign
    /// float noise does not. Also asserts every request completed (the `min_len-1`
    /// cap must leave even the shortest backfilled request a non-empty suffix). Run:
    /// `cargo test -p wenlan-core --lib eval::engine_throughput::tests::prefix_cache_equivalence -- --ignored --nocapture`
    #[test]
    #[ignore = "needs GPU + downloaded GGUF model (L7 manual)"]
    fn prefix_cache_equivalence() {
        let Some((engine, mut ctx)) = build_engine_and_ctx() else {
            eprintln!("[engine_throughput] engine unavailable (no GPU/model) — skipping");
            return;
        };
        let items = shared_prefix_workload(19, 0.0);
        let prompts: Vec<BatchPrompt> = items.iter().map(|it| it.prompt.clone()).collect();
        // Re-run OFF once on its own to assert completeness (no silent empty-suffix
        // failure) independent of the divergence check.
        std::env::remove_var("WENLAN_LLM_PREFIX_KV_CACHE");
        let off_completed =
            completed(&engine.run_inference_continuous_batch(&mut ctx, &prompts, M));
        let divergence = first_prefix_cache_divergence(engine, &mut ctx, &prompts, M);
        leak_ctx(ctx);
        assert_eq!(
            off_completed,
            prompts.len(),
            "baseline (OFF) left {} of {} requests with no output",
            prompts.len() - off_completed,
            prompts.len()
        );
        match divergence {
            None => eprintln!(
                "[engine_throughput] prefix-KV equivalence OK: cached output byte-identical to fresh prefill ({} requests, M={M}, backfill engaged)",
                prompts.len()
            ),
            Some((i, off, on)) => panic!(
                "prefix-KV cache changed request {i}'s output\n  OFF: {off:?}\n  ON:  {on:?}"
            ),
        }
    }

    /// Prefix-KV cache prefill A/B: report wall-clock ON vs OFF over a homogeneous
    /// shared-prefix batch. The headline (prefill_ms / prime_ms drop) is in the
    /// `[batch_timing]` lines below; set WENLAN_BATCH_LOG via the test. Run:
    /// `WENLAN_BATCH_LOG=1 cargo test -p wenlan-core --lib eval::engine_throughput::tests::prefix_cache_prefill_ab -- --ignored --nocapture`
    #[test]
    #[ignore = "needs GPU + downloaded GGUF model (L7 manual)"]
    fn prefix_cache_prefill_ab() {
        let Some((engine, mut ctx)) = build_engine_and_ctx() else {
            eprintln!("[engine_throughput] engine unavailable (no GPU/model) — skipping");
            return;
        };
        std::env::set_var("WENLAN_BATCH_LOG", "1"); // surface [batch_timing] prefix_len/prime_ms/prefill_ms
        let items = shared_prefix_workload(40, 0.0);
        let prompts: Vec<BatchPrompt> = items.iter().map(|it| it.prompt.clone()).collect();
        let ab = run_prefix_cache_throughput_ab(engine, &mut ctx, &prompts, M);
        leak_ctx(ctx);
        eprintln!(
            "[engine_throughput] prefix-KV n={} m={}\n  OFF: {} ms, {} chars, {} ok\n  ON:  {} ms, {} chars, {} ok\n  wall speedup (OFF/ON): {:.2}x  (prefill detail in [batch_timing] above)",
            ab.n_requests, ab.m,
            ab.off.wall_ms, ab.off.out_chars, ab.off.completed,
            ab.on.wall_ms, ab.on.out_chars, ab.on.completed,
            ab.speedup(),
        );
        assert_eq!(
            ab.on.completed,
            prompts.len(),
            "prefix-KV ON dropped requests"
        );
        assert!(
            ab.speedup() >= 0.95,
            "prefix-KV regressed throughput: {:.2}x (OFF {} ms / ON {} ms)",
            ab.speedup(),
            ab.off.wall_ms,
            ab.on.wall_ms
        );
    }
}
