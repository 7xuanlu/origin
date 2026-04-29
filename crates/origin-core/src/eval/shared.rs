// SPDX-License-Identifier: AGPL-3.0-only
//! Shared eval infrastructure: embedder, tokenizer, entity extraction helper.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::sources::RawDocument;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::LazyLock;

/// Shared BPE tokenizer instance (cl100k_base). Initialized once, intentionally
/// leaked to avoid destructor conflicts with ONNX runtime at process exit.
static BPE: LazyLock<&'static tiktoken_rs::CoreBPE> = LazyLock::new(|| {
    let bpe = tiktoken_rs::cl100k_base().expect("failed to load cl100k_base tokenizer");
    Box::leak(Box::new(bpe))
});

/// Process-wide shared ONNX embedder for eval functions. Loaded once (1-2s),
/// intentionally leaked to avoid SIGSEGV from ONNX runtime destructor at exit.
/// We store a cloned Arc and `mem::forget` the original so the strong count never
/// reaches zero — the TextEmbedding destructor never runs.
static EVAL_EMBEDDER: LazyLock<Arc<std::sync::Mutex<fastembed::TextEmbedding>>> =
    LazyLock::new(|| {
        let opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::BGEBaseENV15Q)
            .with_show_download_progress(true);
        let embedder = fastembed::TextEmbedding::try_new(opts)
            .expect("failed to load BGE-Base-EN-v1.5-Q ONNX model");
        let arc = Arc::new(std::sync::Mutex::new(embedder));
        // Leak one strong ref so the Arc never reaches zero and the destructor never runs.
        std::mem::forget(arc.clone());
        arc
    });

/// Returns the process-wide shared embedder for eval use.
pub fn eval_shared_embedder() -> Arc<std::sync::Mutex<fastembed::TextEmbedding>> {
    EVAL_EMBEDDER.clone()
}

/// Count tokens in text using tiktoken cl100k_base encoding.
pub fn count_tokens(text: &str) -> usize {
    BPE.encode_with_special_tokens(text).len()
}

/// Probe on-device batch extraction at different batch sizes.
/// Returns vec of (batch_size, input_tokens, response_len, entities_found, observations_found).
pub async fn probe_extraction_batch_sizes(
    observations: &[(String, String)], // (source_id, content)
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
    batch_sizes: &[usize],
) -> Vec<(usize, usize, usize, usize, usize)> {
    use crate::extract::parse_kg_response;
    use crate::prompts::PromptRegistry;

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let mut results = Vec::new();

    for &batch_size in batch_sizes {
        let batch: Vec<&(String, String)> = observations.iter().take(batch_size).collect();
        if batch.is_empty() {
            continue;
        }

        // Format numbered input (same as production batch extraction)
        let numbered: String = batch
            .iter()
            .enumerate()
            .map(|(i, (_, content))| {
                let truncated: String = content.chars().take(500).collect();
                format!("{}. {}", i + 1, truncated)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let input_tokens = count_tokens(&numbered) + count_tokens(&prompts.extract_knowledge_graph);

        eprintln!(
            "[probe] batch_size={}, input_tokens={}, sending...",
            batch_size, input_tokens
        );

        let start = std::time::Instant::now();
        match llm
            .generate(crate::llm_provider::LlmRequest {
                system_prompt: Some(prompts.extract_knowledge_graph.clone()),
                user_prompt: numbered,
                max_tokens: ((batch_size * 200) as u32).max(512), // scale with input, min 512
                temperature: 0.3,
                label: Some(format!("probe_batch_{}", batch_size)),
            })
            .await
        {
            Ok(response) => {
                let elapsed = start.elapsed();
                let memories: Vec<(usize, String)> = batch
                    .iter()
                    .enumerate()
                    .map(|(i, (_, c))| (i, c.clone()))
                    .collect();
                let kg = parse_kg_response(&response, &memories);
                let total_entities: usize = kg.iter().map(|r| r.entities.len()).sum();
                let total_obs: usize = kg.iter().map(|r| r.observations.len()).sum();

                let resp_preview: String = response.chars().take(300).collect();
                eprintln!(
                    "[probe] batch_size={}: {}ms, response_len={}, entities={}, obs={}\n  preview: {}",
                    batch_size,
                    elapsed.as_millis(),
                    response.len(),
                    total_entities,
                    total_obs,
                    resp_preview,
                );
                results.push((
                    batch_size,
                    input_tokens,
                    response.len(),
                    total_entities,
                    total_obs,
                ));
            }
            Err(e) => {
                eprintln!("[probe] batch_size={}: FAILED — {}", batch_size, e);
                results.push((batch_size, input_tokens, 0, 0, 0));
            }
        }
    }

    results
}

/// Run entity-extraction enrichment via Anthropic Batch API. **Opt-in only** — production
/// uses on-device Qwen3-4B; this path over-flatters quality vs production. Set
/// `EVAL_ENRICHMENT=cloud` to use it.
///
/// Cost: not "~$1" — at LME scale (5500 memories) this single batch is ~$2-3 input + output
/// at Haiku batch rates. Combined with `run_title_enrichment_batch_api` and
/// `run_concept_distillation_batch_api` (the full enrichment trio), cumulative is
/// ~$5-8 per LME run. Per-batch `cost_cap_usd` enforced inside `submit_batch`; session-aggregate
/// caps NOT yet implemented (see Phase 2 cost ledger).
///
/// Speed: ~5 min vs ~2 hours on-device for LoCoMo (10x faster). Useful for fast iteration when
/// you knowingly want to spend.
///
/// Returns total entities created.
pub async fn run_enrichment_batch_api(
    db: &MemoryDB,
    api_key: &str,
    model: &str,
    cost_cap_usd: f64,
) -> Result<usize, OriginError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::extract::parse_kg_response;
    use crate::prompts::PromptRegistry;

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());

    // 1. Get all memories needing extraction
    // Use a large limit to get everything in one query
    let all_memories = db.get_unlinked_memories(100_000).await?;
    if all_memories.is_empty() {
        eprintln!("[batch_enrich] No unlinked memories found");
        return Ok(0);
    }
    eprintln!("[batch_enrich] {} memories to extract", all_memories.len());

    // 2. Format extraction prompts (1 per memory, same as production single-memory path)
    let mut batch_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();
    let mut memory_map: std::collections::HashMap<String, (String, String)> = // custom_id -> (source_id, content)
        std::collections::HashMap::new();

    for (idx, (source_id, content)) in all_memories.iter().enumerate() {
        let truncated: String = content.chars().take(500).collect();
        let numbered = format!("1. {}", truncated);
        let custom_id = format!("extract_{}", idx);

        batch_requests.push((
            custom_id.clone(),
            numbered,
            Some(prompts.extract_knowledge_graph.clone()),
            512,
        ));
        memory_map.insert(custom_id, (source_id.clone(), content.clone()));
    }

    // 3. Submit batch
    eprintln!(
        "[batch_enrich] Submitting {} extraction requests (model={}, cap=${:.2})",
        batch_requests.len(),
        model,
        cost_cap_usd,
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| OriginError::Generic(format!("client: {e}")))?;

    let batch_id = submit_batch(&client, api_key, batch_requests, model, cost_cap_usd)
        .await
        .map_err(|e| OriginError::Generic(format!("batch submit: {e}")))?;
    eprintln!("[batch_enrich] Batch submitted: {}", batch_id);

    let results_url = poll_batch(&client, api_key, &batch_id)
        .await
        .map_err(|e| OriginError::Generic(format!("batch poll: {e}")))?;

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| OriginError::Generic(format!("batch download: {e}")))?;

    eprintln!(
        "[batch_enrich] Downloaded {} results. Creating entities...",
        raw_results.len()
    );

    // 4. Parse results and create entities
    let mut total_entities = 0usize;
    let mut entity_cache: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for (custom_id, response) in &raw_results {
        let (source_id, content) = match memory_map.get(custom_id) {
            Some(m) => m,
            None => continue,
        };

        let batch = [(0usize, content.clone())];
        let kg_results = parse_kg_response(response, &batch);

        let mut first_entity_id: Option<String> = None;

        for kg in &kg_results {
            for entity in &kg.entities {
                match crate::importer::resolve_or_create_entity(
                    db,
                    &mut entity_cache,
                    entity,
                    "batch_eval",
                )
                .await
                {
                    Ok((id, _created)) => {
                        total_entities += 1;
                        if first_entity_id.is_none() {
                            first_entity_id = Some(id);
                        }
                    }
                    Err(e) => {
                        log::warn!("[batch_enrich] entity create failed: {e}");
                    }
                }
            }
            for obs in &kg.observations {
                if let Some(entity_id) = entity_cache.get(&obs.entity.to_lowercase()) {
                    let _ = db
                        .add_observation(entity_id, &obs.content, Some("batch_eval"), None)
                        .await;
                }
            }
            for rel in &kg.relations {
                let from_id = entity_cache.get(&rel.from.to_lowercase()).cloned();
                let to_id = entity_cache.get(&rel.to.to_lowercase()).cloned();
                if let (Some(from), Some(to)) = (from_id, to_id) {
                    let _ = db
                        .create_relation(
                            &from,
                            &to,
                            &rel.relation_type,
                            Some("batch_eval"),
                            rel.confidence,
                            rel.explanation.as_deref(),
                            Some(source_id),
                        )
                        .await;
                }
            }
        }

        // Link memory to first entity
        if let Some(ref eid) = first_entity_id {
            let _ = db.update_memory_entity_id(source_id, eid).await;
        }
    }

    // 5. Mark all memories as enriched for concept distillation
    let marked = db.mark_all_memories_enriched_for_eval().await?;
    eprintln!(
        "[batch_enrich] Done: {} entities created, {} memories marked enriched",
        total_entities, marked
    );

    Ok(total_entities)
}

/// Run entity extraction using Origin's production pipeline (refinery path).
///
/// Uses `extract_entities_from_memories` which calls `extract_single_memory_entities`
/// with the production EXTRACT_KNOWLEDGE_GRAPH prompt (PR5) and proper Qwen chat
/// template formatting. Much more reliable than the old custom JSON extraction.
///
/// Runs in batches of `batch_size` unlinked memories until all are processed.
pub async fn run_entity_extraction_for_eval(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<usize, OriginError> {
    run_entity_extraction_for_eval_concurrent(db, llm, 1).await
}

/// Like `run_entity_extraction_for_eval` but with configurable per-memory concurrency.
///
/// Fetches all unlinked memories up front, then dispatches up to `concurrency` parallel
/// `extract_single_memory_entities` calls via `buffer_unordered`. Benefit: overlaps LLM
/// inference time across memories; DB writes still serialize through the internal Mutex.
///
/// Logs progress every 20 memories processed: phase, processed/total, elapsed, rate, eta.
pub async fn run_entity_extraction_for_eval_concurrent(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
    concurrency: usize,
) -> Result<usize, OriginError> {
    use crate::prompts::PromptRegistry;
    use futures::StreamExt;

    let prompts = Arc::new(PromptRegistry::load(&PromptRegistry::override_dir()));
    let mut total = 0usize;
    let t0 = std::time::Instant::now();
    let mut processed = 0usize;

    // Drain all unlinked memories in batches, re-querying after each concurrent round
    // so new entities written by one batch don't block the next.
    loop {
        let unlinked = db.get_unlinked_memories(256).await?;
        if unlinked.is_empty() {
            break;
        }
        let batch_len = unlinked.len();
        let results: Vec<_> =
            futures::stream::iter(unlinked.into_iter().map(|(source_id, content)| {
                let llm = llm.clone();
                let prompts = prompts.clone();
                async move {
                    crate::refinery::extract_single_memory_entities(
                        db, &llm, &prompts, &source_id, &content,
                    )
                    .await
                }
            }))
            .buffer_unordered(concurrency.max(1))
            .collect()
            .await;

        let mut batch_extracted = 0usize;
        for result in results {
            match result {
                Ok(Some(_)) => batch_extracted += 1,
                Ok(None) => {}
                Err(e) => log::warn!("[entity_extract] extraction failed: {}", e),
            }
            processed += 1;
            if processed.is_multiple_of(20) {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = processed as f64 / elapsed.max(0.001);
                eprintln!(
                    "[enrich] phase=entity processed={} elapsed={:.0}s rate={:.1}/s",
                    processed, elapsed, rate,
                );
            }
        }
        total += batch_extracted;
        eprintln!(
            "    [entity_extract] batch: +{} entities from {} memories (total: {})",
            batch_extracted, batch_len, total
        );

        if batch_extracted == 0 {
            // No progress despite unlinked memories remaining — break to avoid infinite loop.
            break;
        }
    }

    // Mark all memories as enriched so find_distillation_clusters includes them.
    // In production, the async post-ingest flow writes these rows. In eval we
    // must do it explicitly after entity extraction completes.
    let marked = db.mark_all_memories_enriched_for_eval().await?;
    eprintln!(
        "    [entity_extract] marked {} memories as enriched",
        marked
    );

    Ok(total)
}

/// Batched entity extraction for on-device LLMs (Qwen3.5-9B or larger).
///
/// Packs `batch_size` memories into a single numbered prompt, makes one LLM call per chunk,
/// and parses per-memory entities from the response using `parse_kg_response`. This is the
/// preferred path when `EVAL_ENRICHMENT_BATCH_SIZE > 1` because Metal is single-device — true
/// parallelism doesn't help, but fewer round-trips per token amortizes inference overhead.
///
/// Qwen3.5-9B handles batches of 5-10 memories reliably. Qwen3-4B degrades above 1-2.
///
/// Returns total entity-linked memories (memories that got at least one entity).
///
/// Logs progress every 5 chunks: `[entity_extract_batched] chunk K/N: ...`.
pub async fn run_entity_extraction_for_eval_batched(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
    batch_size: usize,
) -> Result<usize, OriginError> {
    use crate::extract::parse_kg_response;
    use crate::prompts::PromptRegistry;

    let batch_size = batch_size.max(1);
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let mut entity_cache: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    let mut total_linked = 0usize;
    let t0 = std::time::Instant::now();

    // Collect all unlinked memories once. Re-querying per chunk would re-fetch the same
    // rows until `update_memory_entity_id` marks them linked — expensive and racy.
    let all_unlinked = db.get_unlinked_memories(100_000).await?;
    if all_unlinked.is_empty() {
        eprintln!("[entity_extract_batched] no unlinked memories — skipping");
        let marked = db.mark_all_memories_enriched_for_eval().await?;
        eprintln!(
            "[entity_extract_batched] marked {} memories as enriched",
            marked
        );
        return Ok(0);
    }

    let total_memories = all_unlinked.len();
    let chunks: Vec<&[(String, String)]> = all_unlinked.chunks(batch_size).collect();
    let num_chunks = chunks.len();

    eprintln!(
        "[entity_extract_batched] {} memories in {} chunks (batch_size={})",
        total_memories, num_chunks, batch_size
    );

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        // Format numbered prompt: "1. <content>\n2. <content>\n..."
        let numbered: String = chunk
            .iter()
            .enumerate()
            .map(|(i, (_, content))| {
                let truncated: String = content.chars().take(500).collect();
                format!("{}. {}", i + 1, truncated)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let response = llm
            .generate(crate::llm_provider::LlmRequest {
                system_prompt: Some(prompts.extract_knowledge_graph.clone()),
                user_prompt: numbered,
                max_tokens: ((chunk.len() * 256) as u32).max(512),
                temperature: 0.3,
                label: Some(format!("batch_extract_chunk_{}", chunk_idx)),
            })
            .await;

        let response = match response {
            Ok(r) => r,
            Err(e) => {
                log::warn!(
                    "[entity_extract_batched] chunk {}/{}: LLM failed: {}",
                    chunk_idx + 1,
                    num_chunks,
                    e
                );
                continue;
            }
        };

        // parse_kg_response expects (index, content) pairs — index is used to map
        // numbered sections back to individual memories.
        let indexed: Vec<(usize, String)> = chunk
            .iter()
            .enumerate()
            .map(|(i, (_, c))| (i, c.clone()))
            .collect();
        let kg_results = parse_kg_response(&response, &indexed);

        let mut chunk_entities = 0usize;
        let mut chunk_linked = 0usize;

        for (mem_idx, kg) in kg_results.iter().enumerate() {
            if mem_idx >= chunk.len() {
                break;
            }
            let (source_id, _) = &chunk[mem_idx];
            let mut first_entity_id: Option<String> = None;

            for entity in &kg.entities {
                match crate::importer::resolve_or_create_entity(
                    db,
                    &mut entity_cache,
                    entity,
                    "batch_eval",
                )
                .await
                {
                    Ok((id, _)) => {
                        chunk_entities += 1;
                        if first_entity_id.is_none() {
                            first_entity_id = Some(id);
                        }
                    }
                    Err(e) => log::warn!("[entity_extract_batched] entity create failed: {e}"),
                }
            }
            for obs in &kg.observations {
                if let Some(entity_id) = entity_cache.get(&obs.entity.to_lowercase()) {
                    let _ = db
                        .add_observation(entity_id, &obs.content, Some("batch_eval"), None)
                        .await;
                }
            }
            for rel in &kg.relations {
                let from_id = entity_cache.get(&rel.from.to_lowercase()).cloned();
                let to_id = entity_cache.get(&rel.to.to_lowercase()).cloned();
                if let (Some(from), Some(to)) = (from_id, to_id) {
                    let _ = db
                        .create_relation(
                            &from,
                            &to,
                            &rel.relation_type,
                            Some("batch_eval"),
                            rel.confidence,
                            rel.explanation.as_deref(),
                            Some(source_id),
                        )
                        .await;
                }
            }

            if let Some(ref eid) = first_entity_id {
                let _ = db.update_memory_entity_id(source_id, eid).await;
                chunk_linked += 1;
            }
        }

        total_linked += chunk_linked;

        if (chunk_idx + 1) % 5 == 0 || chunk_idx + 1 == num_chunks {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = (chunk_idx + 1) as f64 / elapsed.max(0.001);
            let eta = if rate > 0.0 {
                (num_chunks - chunk_idx - 1) as f64 / rate
            } else {
                0.0
            };
            eprintln!(
                "[entity_extract_batched] chunk {}/{}: extracted {} entities from {} memories (total_linked: {}) elapsed={:.0}s rate={:.1}chunk/s eta={:.0}s",
                chunk_idx + 1,
                num_chunks,
                chunk_entities,
                chunk.len(),
                total_linked,
                elapsed,
                rate,
                eta,
            );
        }
    }

    let marked = db.mark_all_memories_enriched_for_eval().await?;
    eprintln!(
        "[entity_extract_batched] done: {} memories linked, {} marked enriched",
        total_linked, marked
    );

    Ok(total_linked)
}

/// Resolve the per-scenario DB directory under `baselines/fullpipeline/{benchmark}/{scenario_id}/`.
///
/// `benchmark` is the short name (`"lme"` or `"locomo"`). The function prepends `"fullpipeline/"`,
/// so the result is `baselines_dir/fullpipeline/{benchmark}/{scenario_id}`.
/// `MemoryDB::new_with_shared_embedder` will create `origin_memory.db` inside the returned dir.
pub fn scenario_db_dir(baselines_dir: &Path, benchmark: &str, scenario_id: &str) -> PathBuf {
    baselines_dir
        .join("fullpipeline")
        .join(benchmark)
        .join(scenario_id)
}

/// Open existing per-scenario DB if fully enriched; otherwise seed and enrich, then return.
///
/// Seed docs are provided lazily via `seed_docs` so callers avoid materializing them
/// when the DB is already cached. Partial state (memories present but not fully enriched)
/// is wiped and restarted to guarantee consistency.
///
/// Cache hit: `mem_count > 0 && enriched_count == mem_count` - returns immediately.
/// Partial resume: `mem_count > 0 && enriched_count < mem_count` - clears and re-seeds.
/// Empty or new DB: seeds from `seed_docs()` and runs full enrichment.
pub async fn open_or_seed_scenario_db<F>(
    db_dir: &Path,
    shared_embedder: Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
    seed_docs: F,
    enrichment: &EnrichmentMode,
) -> Result<MemoryDB, OriginError>
where
    F: FnOnce() -> Vec<RawDocument>,
{
    std::fs::create_dir_all(db_dir)
        .map_err(|e| OriginError::Generic(format!("create db_dir: {e}")))?;

    let db = MemoryDB::new_with_shared_embedder(
        db_dir,
        Arc::new(crate::events::NoopEmitter),
        shared_embedder,
    )
    .await?;

    let mem_count = db.memory_count().await.unwrap_or(0);
    let enriched = db.enriched_memory_count().await.unwrap_or(0);

    if mem_count > 0 && enriched == mem_count {
        log::info!(
            "[scenario_db] cache hit: {} ({} memories, all enriched)",
            db_dir.display(),
            mem_count
        );
        return Ok(db);
    }

    if mem_count > 0 && enriched < mem_count {
        // Refuse to silently destroy data. Past incident: pooled eval DBs lost ~5901
        // memories because helper wiped on partial state with no operator confirmation.
        // Operator must opt-in via EVAL_ALLOW_WIPE=1 after inspecting the partial DB.
        if std::env::var("EVAL_ALLOW_WIPE").as_deref() != Ok("1") {
            return Err(OriginError::Generic(format!(
                "[scenario_db] refused to wipe partial state at {} ({}/{} enriched). \
                 Set EVAL_ALLOW_WIPE=1 to permit destruction, or inspect/repair the DB manually.",
                db_dir.display(),
                enriched,
                mem_count
            )));
        }
        log::warn!(
            "[scenario_db] partial state {}/{} enriched - WIPING (EVAL_ALLOW_WIPE=1): {}",
            enriched,
            mem_count,
            db_dir.display()
        );
        db.clear_all_for_eval().await?;
    }

    let docs = seed_docs();
    if docs.is_empty() {
        log::info!(
            "[scenario_db] no seed docs for {} - skipping enrichment",
            db_dir.display()
        );
        return Ok(db);
    }

    db.upsert_documents(docs).await?;

    let (entities, titles, concepts) = enrich_db_for_eval(&db, enrichment).await?;
    log::info!(
        "[scenario_db] enriched {}: {} entities, {} titles, {} concepts",
        db_dir.display(),
        entities,
        titles,
        concepts
    );

    Ok(db)
}

/// Where to source enrichment work for an eval DB.
///
/// `OnDevice` mirrors production: free, slow, uses Qwen3-4B via the on-device LlmProvider.
/// Use this when measuring what real users get.
///
/// `BatchApi` uses Anthropic Batch API: fast, paid, and over-flatters quality vs. production
/// because Haiku is more capable than Qwen-4B. Opt-in only.
pub enum EnrichmentMode {
    OnDevice(Arc<dyn crate::llm_provider::LlmProvider>),
    BatchApi {
        api_key: String,
        model: String,
        cost_cap_usd: f64,
    },
}

impl EnrichmentMode {
    /// Construct from environment.
    ///
    /// - `EVAL_ENRICHMENT=cloud` (or `batch`) → `BatchApi` (requires `ANTHROPIC_API_KEY`).
    /// - Anything else (default) → `OnDevice`, model selected by `EVAL_LOCAL_MODEL`:
    ///   - unset → `qwen3-4b` (default, fast, lower quality)
    ///   - `qwen3.5-9b` → 9B model (slower, higher quality)
    ///
    /// Returns Err if `cloud` is requested without `ANTHROPIC_API_KEY`, or if
    /// `OnDeviceProvider` init fails (model load, GPU init, etc.).
    pub fn from_env(answer_model: &str, cost_cap_usd: f64) -> Result<Self, OriginError> {
        let mode = std::env::var("EVAL_ENRICHMENT").unwrap_or_else(|_| "local".into());
        match mode.as_str() {
            "cloud" | "batch" => {
                let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                    OriginError::Generic("EVAL_ENRICHMENT=cloud requires ANTHROPIC_API_KEY".into())
                })?;
                Ok(Self::BatchApi {
                    api_key,
                    model: answer_model.to_string(),
                    cost_cap_usd,
                })
            }
            _ => {
                let model_id = std::env::var("EVAL_LOCAL_MODEL").ok();
                let provider = match model_id.as_deref() {
                    Some(m) => crate::llm_provider::OnDeviceProvider::new_with_model(Some(m)),
                    None => crate::llm_provider::OnDeviceProvider::new(),
                }
                .map_err(|e| {
                    OriginError::Generic(format!(
                        "OnDeviceProvider init failed (set EVAL_ENRICHMENT=cloud to use Batch API): {e}"
                    ))
                })?;
                eprintln!(
                    "[enrichment] mode=on-device model={}",
                    model_id.as_deref().unwrap_or("qwen3-4b (default)")
                );
                Ok(Self::OnDevice(Arc::new(provider)))
            }
        }
    }
}

/// Enrich a freshly-seeded eval DB: entity extraction + title enrichment + concept distillation.
///
/// Caller must check `mem_count == enriched_count` before calling — this function unconditionally
/// re-enriches everything it sees. Returns `(entities, titles, concepts)` counts.
pub async fn enrich_db_for_eval(
    db: &MemoryDB,
    mode: &EnrichmentMode,
) -> Result<(usize, usize, usize), OriginError> {
    match mode {
        EnrichmentMode::OnDevice(llm) => enrich_db_for_eval_local(db, llm).await,
        EnrichmentMode::BatchApi {
            api_key,
            model,
            cost_cap_usd,
        } => {
            // Batch A: extraction + titles in parallel (independent).
            let (entities_res, titles_res) = tokio::join!(
                run_enrichment_batch_api(db, api_key, model, *cost_cap_usd),
                run_title_enrichment_batch_api(db, api_key, model, *cost_cap_usd),
            );
            let entities = entities_res?;
            let titles = titles_res?;
            // Batch B: concept distillation (depends on entities + enrichment_steps).
            let concepts =
                run_concept_distillation_batch_api(db, api_key, model, *cost_cap_usd).await?;
            Ok((entities, titles, concepts))
        }
    }
}

/// On-device title enrichment via production code path.
///
/// Loops over `db.get_memories_needing_title_enrichment` candidates, calls
/// `post_ingest::enrich_title` per memory. Returns count of titles actually updated
/// (some candidates may be rejected by the LLM or skipped).
///
/// Concurrency: up to `concurrency` parallel LLM calls (pass 1 for serial).
/// Logs progress every 20 memories: phase, processed, elapsed, rate.
pub async fn run_title_enrichment_for_eval(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
    concurrency: usize,
) -> Result<usize, OriginError> {
    use futures::StreamExt;

    let candidates = db.get_memories_needing_title_enrichment().await?;
    let total_candidates = candidates.len();
    if total_candidates == 0 {
        return Ok(0);
    }

    let t0 = std::time::Instant::now();
    let mut processed = 0usize;

    let results: Vec<_> = futures::stream::iter(candidates.into_iter().map(
        |(source_id, content)| async move {
            crate::post_ingest::enrich_title(db, &source_id, &content, llm).await
        },
    ))
    .buffer_unordered(concurrency.max(1))
    .inspect(|_| {
        // Note: inspect runs before collect; count is approximate within a concurrent batch.
    })
    .collect()
    .await;

    let mut updated = 0usize;
    for result in results {
        match result {
            Ok(crate::post_ingest::TitleEnrichResult::Enriched) => updated += 1,
            Ok(_) => {}
            Err(e) => log::warn!("[title_enrich_local] title enrichment failed: {}", e),
        }
        processed += 1;
        if processed.is_multiple_of(20) {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = processed as f64 / elapsed.max(0.001);
            let eta = (total_candidates - processed) as f64 / rate.max(0.001);
            eprintln!(
                "[enrich] phase=title processed={}/{} elapsed={:.0}s rate={:.1}/s eta={:.0}s",
                processed, total_candidates, elapsed, rate, eta,
            );
        }
    }

    eprintln!(
        "    [title_enrich_local] {}/{} titles enriched",
        updated, total_candidates
    );
    Ok(updated)
}

/// On-device enrichment via production code paths. Mirrors production exactly:
/// `refinery::extract_entities_from_memories` → `post_ingest::enrich_title` per memory →
/// `refinery::distill_concepts`. Free but slow (Qwen3-4B serial ≈ several hours at LME scale).
///
/// Concurrency: entity + title phases dispatch up to `EVAL_ENRICHMENT_CONCURRENCY` parallel LLM
/// calls (default 1 = serial). Distillation stays serial due to cluster ordering dependencies:
/// `distill_concepts` builds clusters by scanning enrichment state written by prior phases, and
/// splitting it would break the FK ordering assumptions in `find_distillation_clusters`.
///
/// Batching: when `EVAL_ENRICHMENT_BATCH_SIZE > 1`, entity extraction uses
/// `run_entity_extraction_for_eval_batched` instead of the per-memory concurrent path.
/// Batch mode packs multiple memories into one LLM call, which is faster on single-device Metal
/// where true concurrency is limited. Qwen3.5-9B handles batch_size=5-10 reliably;
/// Qwen3-4B degrades above 1-2. Default (1) preserves current per-memory behavior.
///
/// For staged evals (e.g. `pipeline.rs` Flat/Enriched/Distilled), call the three sub-steps
/// independently — `run_entity_extraction_for_eval`, `run_title_enrichment_for_eval`,
/// `refinery::distill_concepts` — so each stage can be measured in isolation.
pub async fn enrich_db_for_eval_local(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<(usize, usize, usize), OriginError> {
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let concurrency: usize = std::env::var("EVAL_ENRICHMENT_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let batch_size: usize = std::env::var("EVAL_ENRICHMENT_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    eprintln!(
        "    [enrich_local] concurrency={} batch_size={}",
        concurrency, batch_size
    );

    let entities = if batch_size > 1 {
        run_entity_extraction_for_eval_batched(db, llm, batch_size).await?
    } else {
        run_entity_extraction_for_eval_concurrent(db, llm, concurrency).await?
    };
    let titles = run_title_enrichment_for_eval(db, llm, concurrency).await?;

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();
    let concepts =
        crate::refinery::distill_concepts(db, Some(llm), &prompts, &tuning, None).await?;
    eprintln!("    [distill_local] {} concepts", concepts);

    Ok((entities, titles, concepts))
}

/// Batch title enrichment via Anthropic Batch API.
///
/// Finds all memories with generic/truncated titles, generates semantic titles
/// via Haiku, updates them in DB. Improves FTS search recall.
pub async fn run_title_enrichment_batch_api(
    db: &MemoryDB,
    api_key: &str,
    model: &str,
    cost_cap_usd: f64,
) -> Result<usize, OriginError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};

    let candidates = db.get_memories_needing_title_enrichment().await?;

    if candidates.is_empty() {
        eprintln!("[batch_title] No memories need title enrichment");
        return Ok(0);
    }
    eprintln!(
        "[batch_title] {} memories need title enrichment",
        candidates.len()
    );

    let title_system = "Given a note, write a 3-5 word title. Output ONLY the title.\n\nExample: 'The system uses libsql for vector storage with DiskANN indexing' -> libsql Vector Storage\nExample: 'Google Sign-In fails with developer_error status 10' -> Google Sign-In SHA Fix".to_string();

    let batch_requests: Vec<(String, String, Option<String>, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, (_, content))| {
            let input: String = content.chars().take(300).collect();
            (
                format!("title_{}", i),
                input,
                Some(title_system.clone()),
                16,
            )
        })
        .collect();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| OriginError::Generic(format!("client: {e}")))?;

    let batch_id = submit_batch(&client, api_key, batch_requests, model, cost_cap_usd)
        .await
        .map_err(|e| OriginError::Generic(format!("title batch submit: {e}")))?;
    eprintln!("[batch_title] Batch submitted: {}", batch_id);

    let results_url = poll_batch(&client, api_key, &batch_id)
        .await
        .map_err(|e| OriginError::Generic(format!("title batch poll: {e}")))?;

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| OriginError::Generic(format!("title batch download: {e}")))?;

    let mut updated = 0usize;
    for (i, (source_id, _)) in candidates.iter().enumerate() {
        let custom_id = format!("title_{}", i);
        if let Some(title) = raw_results.get(&custom_id) {
            let clean = title.trim().trim_matches('"').trim();
            if !clean.is_empty() && clean.len() < 100 {
                db.update_title(source_id, clean).await?;
                updated += 1;
            }
        }
    }

    eprintln!("[batch_title] Updated {} titles", updated);
    Ok(updated)
}

/// Batch concept distillation via Anthropic Batch API.
///
/// Replaces production `distill_concepts` (which uses sequential on-device LLM)
/// with a batch API approach. Same DB queries and concept storage, different
/// LLM execution model.
///
/// Two batch submissions: refinement (merge/split clusters), then synthesis.
pub async fn run_concept_distillation_batch_api(
    db: &MemoryDB,
    api_key: &str,
    model: &str,
    cost_cap_usd: f64,
) -> Result<usize, OriginError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Use Haiku's synthesis limit (200K context, generous)
    let token_limit = 16_000;
    let clusters = db
        .find_distillation_clusters(
            tuning.similarity_threshold,
            tuning.concept_min_cluster_size,
            tuning.max_clusters_per_steep,
            token_limit,
            tuning.max_unlinked_cluster_size,
        )
        .await?;

    if clusters.is_empty() {
        eprintln!("[batch_distill] No clusters found for distillation");
        return Ok(0);
    }
    eprintln!("[batch_distill] {} clusters to distill", clusters.len());

    // Skip refinement for eval (it only matters when entities have 2+ clusters,
    // which is rare in a single benchmark run). Go straight to synthesis.

    // Build synthesis prompts for each cluster
    struct ClusterMeta {
        idx: usize,
        topic: String,
        entity_id: Option<String>,
        domain: Option<String>,
        source_ids: Vec<String>,
    }
    let mut batch_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();
    let mut cluster_meta: Vec<ClusterMeta> = Vec::new();

    for (idx, cluster) in clusters.iter().enumerate() {
        let topic = cluster
            .entity_name
            .as_deref()
            .or(cluster.domain.as_deref())
            .unwrap_or("general");

        // Skip if concept with similar sources exists (Jaccard > 0.8)
        let overlap = db
            .max_concept_overlap(&cluster.source_ids)
            .await
            .unwrap_or(0.0);
        if overlap > 0.8 {
            continue;
        }

        // Clean and cap memory snippets
        let memories_block: String = cluster
            .source_ids
            .iter()
            .zip(cluster.contents.iter())
            .map(|(id, content)| {
                let snippet: String = content.chars().take(800).collect();
                format!("[{}] {}", id, snippet)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // Skip thin clusters
        let total_chars: usize = cluster.contents.iter().map(|c| c.len()).sum();
        if total_chars < 200 {
            continue;
        }

        let user_prompt = format!("Topic: {}\n\n{}", topic, memories_block);

        batch_requests.push((
            format!("synth_{}", idx),
            user_prompt,
            Some(prompts.distill_concept.clone()),
            2048,
        ));
        cluster_meta.push(ClusterMeta {
            idx,
            topic: topic.to_string(),
            entity_id: cluster.entity_id.clone(),
            domain: cluster.domain.clone(),
            source_ids: cluster.source_ids.clone(),
        });
    }

    if batch_requests.is_empty() {
        eprintln!("[batch_distill] No clusters passed filtering");
        return Ok(0);
    }

    eprintln!(
        "[batch_distill] Submitting {} synthesis requests",
        batch_requests.len()
    );

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| OriginError::Generic(format!("client: {e}")))?;

    let batch_id = submit_batch(&client, api_key, batch_requests, model, cost_cap_usd)
        .await
        .map_err(|e| OriginError::Generic(format!("distill batch submit: {e}")))?;
    eprintln!("[batch_distill] Batch submitted: {}", batch_id);

    let results_url = poll_batch(&client, api_key, &batch_id)
        .await
        .map_err(|e| OriginError::Generic(format!("distill batch poll: {e}")))?;

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| OriginError::Generic(format!("distill batch download: {e}")))?;

    // Also batch title generation for concepts
    let mut title_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();
    let mut synth_results: Vec<(usize, String)> = Vec::new(); // (meta_idx, content)

    for (meta_idx, meta) in cluster_meta.iter().enumerate() {
        let custom_id = format!("synth_{}", meta.idx);
        if let Some(raw) = raw_results.get(&custom_id) {
            let cleaned = crate::llm_provider::strip_think_tags(raw);
            let content = cleaned.trim().to_string();
            if !content.is_empty() {
                let input: String = content.chars().take(300).collect();
                title_requests.push((
                    format!("ctitle_{}", meta_idx),
                    input,
                    Some(
                        "Given a note, write a 3-5 word title. Output ONLY the title.".to_string(),
                    ),
                    16,
                ));
                synth_results.push((meta_idx, content));
            }
        }
    }

    if synth_results.is_empty() {
        eprintln!("[batch_distill] No synthesis results to store");
        return Ok(0);
    }

    // Batch concept titles
    eprintln!(
        "[batch_distill] Submitting {} title requests",
        title_requests.len()
    );
    let title_batch_id = submit_batch(&client, api_key, title_requests, model, cost_cap_usd)
        .await
        .map_err(|e| OriginError::Generic(format!("ctitle batch submit: {e}")))?;

    let title_results_url = poll_batch(&client, api_key, &title_batch_id)
        .await
        .map_err(|e| OriginError::Generic(format!("ctitle batch poll: {e}")))?;

    let title_results = download_batch_results(&client, api_key, &title_results_url)
        .await
        .map_err(|e| OriginError::Generic(format!("ctitle batch download: {e}")))?;

    // Store concepts
    let mut distilled = 0usize;
    for (meta_idx, content) in &synth_results {
        let meta = &cluster_meta[*meta_idx];

        // Hallucination check via embedding similarity
        // Compare concept output against actual memory content (not source IDs)
        let source_content = meta
            .source_ids
            .iter()
            .filter_map(|sid| {
                // Look up content from the cluster data
                clusters
                    .iter()
                    .find(|c| c.source_ids.contains(sid))
                    .and_then(|c| {
                        let idx = c.source_ids.iter().position(|s| s == sid)?;
                        c.contents.get(idx).cloned()
                    })
            })
            .collect::<Vec<_>>()
            .join(" ");
        let texts = vec![content.clone(), source_content];
        if let Ok(embeddings) = db.generate_embeddings(&texts) {
            if embeddings.len() == 2 {
                let sim = crate::db::cosine_similarity(&embeddings[0], &embeddings[1]);
                if sim < 0.6 {
                    eprintln!(
                        "[batch_distill] hallucination (sim={:.2}) for '{}', skipping",
                        sim, meta.topic
                    );
                    continue;
                }
            }
        }

        let title = title_results
            .get(&format!("ctitle_{}", meta_idx))
            .map(|t| t.trim().trim_matches('"').to_string())
            .filter(|t| !t.is_empty() && t.len() < 100)
            .unwrap_or_else(|| meta.topic.clone());

        let summary = content
            .lines()
            .find(|l| l.starts_with("- "))
            .map(|l| l.trim_start_matches("- ").to_string());

        let source_refs: Vec<&str> = meta.source_ids.iter().map(|s| s.as_str()).collect();
        let now = chrono::Utc::now().to_rfc3339();
        let concept_id = crate::concepts::Concept::new_id();

        db.insert_concept(
            &concept_id,
            &title,
            summary.as_deref(),
            content,
            meta.entity_id.as_deref(),
            meta.domain.as_deref(),
            &source_refs,
            &now,
        )
        .await?;

        distilled += 1;
    }

    eprintln!("[batch_distill] Distilled {} concepts", distilled);
    Ok(distilled)
}
