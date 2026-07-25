// SPDX-License-Identifier: Apache-2.0
//! Shared eval infrastructure: embedder, tokenizer, entity extraction helper.

use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::sources::RawDocument;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::LazyLock;

pub fn stamp_cr_from_db_env<F>(build_env: F) -> crate::eval::report::ReportEnv
where
    F: FnOnce(&str) -> crate::eval::report::ReportEnv,
{
    let page_channel_state = if crate::db::page_channel_enabled() {
        "on"
    } else {
        "off"
    };
    let magfusion_state = if crate::db::magnitude_fusion_enabled() {
        "on"
    } else {
        "off"
    };
    let mut variant_tag = if page_channel_state == "off" {
        "cross_rerank_v2_no_pages".to_string()
    } else {
        "cross_rerank_v2_pages".to_string()
    };
    if magfusion_state == "on" {
        variant_tag.push_str("_magfusion");
    }
    let graph_seed_depth = if crate::db::graph_seed_enabled() {
        let depth = crate::retrieval::signals::parse_hop_depth(
            std::env::var("WENLAN_GRAPH_HOP_DEPTH").ok().as_deref(),
        );
        Some(depth)
    } else {
        None
    };
    if let Some(depth) = graph_seed_depth {
        variant_tag.push_str(&format!("__graph_seed_d{}", depth));
    }
    let graph_khop_depth = if crate::db::khop_traversal_enabled() {
        Some(crate::retrieval::traversal::parse_khop_depth(
            std::env::var("WENLAN_GRAPH_KHOP_DEPTH").ok().as_deref(),
        ))
    } else {
        None
    };
    if let Some(depth) = graph_khop_depth {
        variant_tag.push_str(&format!("__graph_khop_d{}", depth));
    }
    let query_intent_state = if crate::retrieval::query_intent::query_intent_enabled() {
        variant_tag.push_str("__query_intent");
        "on"
    } else {
        "off"
    };
    let salience_state = if crate::db::salience_prior_enabled() {
        variant_tag.push_str("__salience");
        "on"
    } else {
        "off"
    };
    let episode_state = if crate::db::episode_channel_enabled() {
        variant_tag.push_str("__episode");
        "on"
    } else {
        "off"
    };
    let fact_state = if crate::retrieval::fact_channel::fact_channel_enabled() {
        variant_tag.push_str("__fact");
        "on"
    } else {
        "off"
    };
    let mut env_stamp = build_env(&variant_tag);
    env_stamp
        .flags
        .push(format!("page_channel={}", page_channel_state));
    env_stamp
        .flags
        .push(format!("magnitude_fusion={}", magfusion_state));
    if let Some(depth) = graph_seed_depth {
        env_stamp.flags.push(format!("graph_seed=on_d{}", depth));
    } else {
        env_stamp.flags.push("graph_seed=off".to_string());
    }
    if let Some(depth) = graph_khop_depth {
        env_stamp.flags.push(format!("graph_khop=on_d{}", depth));
    } else {
        env_stamp.flags.push("graph_khop=off".to_string());
    }
    env_stamp
        .flags
        .push(format!("query_intent={}", query_intent_state));
    env_stamp
        .flags
        .push(format!("salience_prior={}", salience_state));
    env_stamp
        .flags
        .push(format!("episode_channel={}", episode_state));
    env_stamp.flags.push(format!("fact_channel={}", fact_state));
    env_stamp.flags.push(format!(
        "graph_memory_stream={}",
        if crate::db::graph_memory_stream_enabled() {
            "on"
        } else {
            "off"
        }
    ));
    env_stamp.flags.push(format!(
        "rerank_pool=mult{}_floor{}",
        std::env::var("RERANK_POOL_MULTIPLIER")
            .as_deref()
            .unwrap_or("1"),
        std::env::var("RERANK_POOL_FLOOR")
            .as_deref()
            .unwrap_or("10"),
    ));
    env_stamp.flags.push("scenario_db=consolidated".to_string());
    env_stamp
}

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
        let mut opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::BGEBaseENV15Q)
            .with_show_download_progress(true);
        // Reuse the shared on-disk fastembed cache (`WENLAN_TEST_FASTEMBED_CACHE`, else the
        // data-dir cache) instead of fastembed's cwd-relative `./.fastembed_cache` default —
        // otherwise every git worktree starts from an empty cwd cache and re-downloads the
        // model. Mirrors the db.rs production/test embedder via `resolve_fastembed_cache_dir`;
        // the `db_path` here is a sentinel, so only the env + default tiers apply.
        let cache_dir =
            crate::db::resolve_fastembed_cache_dir(std::path::Path::new(".nonexistent"));
        if let Some(cache) = &cache_dir {
            opts = opts.with_cache_dir(cache.clone());
        }
        let embedder = crate::db::init_text_embedding(opts)
            .expect("failed to load BGE-Base-EN-v1.5-Q ONNX model");
        let arc = Arc::new(std::sync::Mutex::new(embedder));
        // Leak one strong ref so the Arc never reaches zero and the destructor never runs.
        std::mem::forget(arc.clone());
        arc
    });

/// Returns the process-wide shared embedder for eval use.
///
/// # Concurrency safety
/// The underlying lock is `std::sync::Mutex`, which is safe here because
/// `MemoryDB::generate_embeddings` is a synchronous function: it acquires the
/// lock, calls `embed()`, and drops the guard before returning. The guard is
/// never held across an `.await` point, so there is no risk of deadlock even
/// when `EVAL_SCENARIO_CONCURRENCY > 1` drives multiple async tasks
/// concurrently. Tasks contend on the mutex, but they do not block the Tokio
/// executor (sync computation inside an async context is acceptable for CPU-
/// bound work that completes in milliseconds).
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
                timeout_secs: None,
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
) -> Result<usize, WenlanError> {
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
        .map_err(|e| WenlanError::Generic(format!("client: {e}")))?;

    let batch_id = submit_batch(&client, api_key, batch_requests, model, cost_cap_usd)
        .await
        .map_err(|e| WenlanError::Generic(format!("batch submit: {e}")))?;
    eprintln!("[batch_enrich] Batch submitted: {}", batch_id);

    let results_url = poll_batch(&client, api_key, &batch_id)
        .await
        .map_err(|e| WenlanError::Generic(format!("batch poll: {e}")))?;

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| WenlanError::Generic(format!("batch download: {e}")))?;

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
                match crate::importer::resolve_entity_bulk(
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
                    if let Err(e) = db
                        .add_observation(entity_id, &obs.content, Some("batch_eval"), None)
                        .await
                    {
                        log::warn!(
                            "[batch_enrich] add_observation failed for entity={} source={}: {}",
                            entity_id,
                            source_id,
                            e
                        );
                    }
                }
            }
            for rel in &kg.relations {
                let from_id = entity_cache.get(&rel.from.to_lowercase()).cloned();
                let to_id = entity_cache.get(&rel.to.to_lowercase()).cloned();
                if let (Some(from), Some(to)) = (from_id, to_id) {
                    if let Err(e) = db
                        .create_relation(
                            &from,
                            &to,
                            &rel.relation_type,
                            Some("batch_eval"),
                            rel.confidence,
                            rel.explanation.as_deref(),
                            Some(source_id),
                        )
                        .await
                    {
                        log::warn!(
                            "[batch_enrich] create_relation failed from={} to={} source={}: {}",
                            from,
                            to,
                            source_id,
                            e
                        );
                    }
                }
            }
        }

        // Link memory to first entity
        if let Some(ref eid) = first_entity_id {
            if let Err(e) = db.update_memory_entity_id(source_id, eid).await {
                log::warn!(
                    "[batch_enrich] update_memory_entity_id failed for source={} entity={}: {}",
                    source_id,
                    eid,
                    e
                );
            }
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

/// Run entity extraction using Wenlan's production pipeline (refinery path).
///
/// Uses `extract_entities_from_memories` which calls `extract_single_memory_entities`
/// with the production EXTRACT_KNOWLEDGE_GRAPH prompt (PR5) and proper Qwen chat
/// template formatting. Much more reliable than the old custom JSON extraction.
///
/// Runs in batches of `batch_size` unlinked memories until all are processed.
pub async fn run_entity_extraction_for_eval(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<usize, WenlanError> {
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
) -> Result<usize, WenlanError> {
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
/// Stamps schema-affecting invariants for a per-scenario DB cache.
///
/// Stored as `cache_env.json` next to `origin_memory.db`. If a stored stamp
/// disagrees with the current build's stamp, the cached DB is stale because
/// the underlying schema or migrations changed. The runner refuses to reuse
/// stale state unless `EVAL_ALLOW_WIPE=1` is set.
///
/// The fingerprint is intentionally narrow: only the things this crate
/// can compute on its own. Comparability across fixture / provider / model
/// is enforced at the **baseline** layer via `comparable_env_hash`, not here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ScenarioCacheEnv {
    schema_db_version: u32,
    migrations_hash: String,
    #[serde(default)]
    enricher_provider: String,
    #[serde(default)]
    enricher_model: String,
}

fn current_cache_env(enrichment: &EnrichmentMode) -> ScenarioCacheEnv {
    let (enricher_provider, enricher_model) = enrichment.provenance();
    ScenarioCacheEnv {
        schema_db_version: crate::db::SCHEMA_VERSION,
        migrations_hash: option_env!("WENLAN_MIGRATIONS_HASH")
            .unwrap_or("unknown")
            .to_string(),
        enricher_provider,
        enricher_model,
    }
}

#[derive(Debug, PartialEq, Eq)]
enum CacheStampDecision {
    Reuse,
    MigrateStale,
    Wipe,
    Refuse,
}

fn provenance_matches(stored: &ScenarioCacheEnv, want: &ScenarioCacheEnv) -> bool {
    // Empty (unstamped) fields never match a populated `want`.
    !stored.enricher_provider.is_empty()
        && !stored.enricher_model.is_empty()
        && stored.enricher_provider == want.enricher_provider
        && stored.enricher_model == want.enricher_model
}

fn decide_cache_stamp(
    stored: Option<&ScenarioCacheEnv>,
    want: &ScenarioCacheEnv,
    db_exists: bool,
    migrate_stale: bool,
    allow_wipe: bool,
) -> CacheStampDecision {
    match stored {
        Some(s) if s == want => CacheStampDecision::Reuse,
        Some(s) => {
            if migrate_stale && provenance_matches(s, want) {
                CacheStampDecision::MigrateStale
            } else if allow_wipe {
                CacheStampDecision::Wipe
            } else {
                CacheStampDecision::Refuse
            }
        }
        None => {
            if db_exists {
                if allow_wipe {
                    CacheStampDecision::Wipe
                } else {
                    CacheStampDecision::Refuse
                }
            } else {
                CacheStampDecision::Reuse
            }
        }
    }
}

pub async fn open_or_seed_scenario_db<F>(
    db_dir: &Path,
    shared_embedder: Arc<std::sync::Mutex<fastembed::TextEmbedding>>,
    seed_docs: F,
    enrichment: &EnrichmentMode,
) -> Result<MemoryDB, WenlanError>
where
    F: FnOnce() -> Vec<RawDocument>,
{
    use fs2::FileExt;

    std::fs::create_dir_all(db_dir)
        .map_err(|e| WenlanError::Generic(format!("create db_dir: {e}")))?;

    // Exclusive lock to prevent two eval runs from corrupting the same scenario DB.
    let lock_path = db_dir.join("scenario.lock");
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| WenlanError::Generic(format!("open scenario.lock: {e}")))?;
    if FileExt::try_lock_exclusive(&lock_file).is_err()
        && std::env::var("EVAL_PARALLEL_OK").as_deref() != Ok("1")
    {
        return Err(WenlanError::Generic(format!(
            "scenario.db at {} is locked by another eval run. \
             Set EVAL_PARALLEL_OK=1 to override (results may be corrupted).",
            db_dir.display()
        )));
    }

    // Cache invalidation by schema/migrations stamp. If the on-disk stamp
    // disagrees with the build's stamp, the cached DB is stale.
    let cache_env_path = db_dir.join("cache_env.json");
    let want = current_cache_env(enrichment);
    let db_file = db_dir.join("origin_memory.db");
    let db_exists = db_file.exists();
    let stored_opt: Option<ScenarioCacheEnv> = std::fs::read(&cache_env_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    let migrate_stale = std::env::var("EVAL_MIGRATE_STALE").as_deref() == Ok("1");
    let allow_wipe = std::env::var("EVAL_ALLOW_WIPE").as_deref() == Ok("1");
    let stamp_decision = decide_cache_stamp(
        stored_opt.as_ref(),
        &want,
        db_exists,
        migrate_stale,
        allow_wipe,
    );
    let attempt_migrate = match stamp_decision {
        CacheStampDecision::Reuse => false,
        CacheStampDecision::Wipe => {
            log::warn!(
                "[scenario_db] cache_env mismatch at {} — wiping (EVAL_ALLOW_WIPE=1)",
                db_dir.display()
            );
            if db_file.exists() {
                std::fs::remove_file(&db_file)
                    .map_err(|e| WenlanError::Generic(format!("remove stale scenario.db: {e}")))?;
            }
            false
        }
        CacheStampDecision::Refuse => {
            return Err(WenlanError::Generic(format!(
                "[scenario_db] cache_env mismatch at {} (schema/migrations changed). \
                 Set EVAL_ALLOW_WIPE=1 to wipe and reseed, or migrate manually.",
                db_dir.display()
            )));
        }
        CacheStampDecision::MigrateStale => true,
    };

    let db = MemoryDB::new_with_shared_embedder(
        db_dir,
        Arc::new(crate::events::NoopEmitter),
        shared_embedder,
    )
    .await?;

    let mem_count = db.memory_count().await.unwrap_or(0);
    let enriched = db.enriched_memory_count().await.unwrap_or(0);

    if attempt_migrate {
        if mem_count == 0 || enriched != mem_count {
            return Err(WenlanError::Generic(format!(
                "[scenario_db] stale DB at {} is not fully enriched ({}/{} enriched). \
                 Cannot migrate — must re-seed. Set EVAL_ALLOW_WIPE=1 to wipe and reseed.",
                db_dir.display(),
                enriched,
                mem_count
            )));
        }
        {
            let conn = db.conn.lock().await;
            crate::eval::seed_contract::assert_feature_substrate_live(&conn, "temporal").await?;
            crate::eval::seed_contract::assert_feature_substrate_live(&conn, "graph").await?;
            crate::eval::seed_contract::assert_feature_substrate_live(&conn, "pages").await?;
        }
        log::info!(
            "[scenario_db] migrate_stale: schema migrated, substrate live at {} ({} memories) — \
             falling through to shared Phase-1 classification backfill",
            db_dir.display(),
            mem_count
        );
        // Do NOT return here — fall through to the cache-hit backfill block so the
        // migrate path shares the Phase-1 classification pass (importance/quality/
        // event_date). Returning early here would skip that backfill and ship
        // training-serving skew (T8/T11/T15 starved on migrated DBs). The migrate
        // guard above already ensured mem_count > 0 && enriched == mem_count, so the
        // fall-through enters the correct branch below and never the partial-wipe path.
        // write_cache_env_stamp + return Ok(db) happen at the bottom of that block.
    }

    if mem_count > 0 && enriched == mem_count {
        // Entity/title/page enrichment is complete (`enriched_memory_count` tracks
        // that marker). But a DB seeded before the Phase-1 classification pass
        // existed — or a partially-backfilled one — can still lack importance /
        // event_date / quality, so the T8 (salience), T11/T20 (temporal), and T15
        // (fact-channel) flags would read empty columns on a cache hit and ship
        // merged-but-inert. Additively backfill the gap rather than treat the
        // cache as complete.
        //
        // This is NON-DESTRUCTIVE: `get_memories_needing_classification` filters
        // `importance IS NULL`, so the pass only touches un-classified rows and is
        // a no-op once complete (resumable). It never wipes — the wipe path below
        // is untouched.
        let needs_class = db
            .get_memories_needing_classification()
            .await
            .map(|v| v.len())
            .unwrap_or(0);
        if needs_class > 0 {
            match enrichment {
                EnrichmentMode::OnDevice(llm) => {
                    log::warn!(
                        "[scenario_db] cache hit at {} but {} memories lack Phase-1 \
                         classification — additively backfilling (resumable)",
                        db_dir.display(),
                        needs_class
                    );
                    let concurrency: usize = std::env::var("EVAL_ENRICHMENT_CONCURRENCY")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1);
                    let n = run_classification_for_eval_concurrent(&db, llm, concurrency).await?;
                    log::info!(
                        "[scenario_db] backfilled Phase-1 classification for {n} memories at {}",
                        db_dir.display()
                    );
                }
                _ => {
                    // BatchApi / Cli classify backfill is the deferred batch-classify
                    // port (see `run_classification_for_eval_concurrent` docs). Fail
                    // loud rather than silently shipping a classification-starved cache.
                    log::warn!(
                        "[scenario_db] cache hit at {} but {} memories lack Phase-1 \
                         classification; enrichment mode is not OnDevice so the batch-classify \
                         backfill is not yet wired — classification-dependent flags \
                         (T8/T11/T20/T15) will read EMPTY on this cache",
                        db_dir.display(),
                        needs_class
                    );
                }
            }
        } else {
            log::info!(
                "[scenario_db] cache hit: {} ({} memories, all enriched + classified)",
                db_dir.display(),
                mem_count
            );
        }
        // Stamp on cache hit too, in case an older run seeded the DB before
        // cache_env.json existed.
        write_cache_env_stamp(&cache_env_path, &want);
        return Ok(db);
    }

    if mem_count > 0 && enriched < mem_count {
        // Refuse to silently destroy data. Past incident: pooled eval DBs lost ~5901
        // memories because helper wiped on partial state with no operator confirmation.
        // Operator must opt-in via EVAL_ALLOW_WIPE=1 after inspecting the partial DB.
        if std::env::var("EVAL_ALLOW_WIPE").as_deref() != Ok("1") {
            return Err(WenlanError::Generic(format!(
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

    write_cache_env_stamp(&cache_env_path, &want);

    Ok(db)
}

/// Best-effort cache_env.json stamp write. Logs on failure but does not fail
/// the eval run — a missing stamp will be detected on the next open and the
/// usual EVAL_ALLOW_WIPE gate will handle it.
fn write_cache_env_stamp(path: &Path, env: &ScenarioCacheEnv) {
    let bytes = match serde_json::to_vec_pretty(env) {
        Ok(b) => b,
        Err(e) => {
            log::warn!("[scenario_db] serialize cache_env failed: {e}");
            return;
        }
    };
    if let Err(e) = std::fs::write(path, bytes) {
        log::warn!(
            "[scenario_db] write cache_env stamp to {} failed: {e}",
            path.display()
        );
    }
}

/// Where to source enrichment work for an eval DB.
///
/// `OnDevice` mirrors production: free, slow, uses Qwen3-4B via the on-device LlmProvider.
/// Use this when measuring what real users get.
///
/// `BatchApi` uses Anthropic Batch API: fast, paid, and over-flatters quality vs. production
/// because Haiku is more capable than Qwen-4B. Opt-in only.
pub enum EnrichmentMode {
    /// On-device Qwen3-4B/9B. Free + slow + production-faithful (matches what users
    /// run locally). Best for LME-scale (~10k memories, ~2h on M2 Pro).
    OnDevice(Arc<dyn crate::llm_provider::LlmProvider>),
    /// Anthropic Batch API. Paid (~$2-3 per LME run) + fast + Haiku quality. Best
    /// for production-quality fast enrichment when API balance is available.
    BatchApi {
        api_key: String,
        model: String,
        cost_cap_usd: f64,
    },
    /// `claude -p` subprocess via Max-plan OAuth. Paid in Max quota + slower than
    /// Batch API at scale + Haiku quality. Best for interactive dev iteration on
    /// LoCoMo-scale (~500 memories) when API balance is empty but Max OAuth is
    /// available. NOT recommended for LME-scale (sequential CLI runs ≈ 3+ hours).
    Cli {
        model: String,
        batch_entities: usize,
        batch_titles: usize,
        rotation: usize,
        retries: u32,
        cost_cap_usd: f64,
        /// Directory for JSONL cache files. Tests pass a tempdir; production
        /// callers pass the eval baselines directory.
        cache_dir: PathBuf,
    },
}

impl EnrichmentMode {
    fn provenance(&self) -> (String, String) {
        match self {
            EnrichmentMode::OnDevice(llm) => (llm.kind().to_string(), llm.model_id()),
            EnrichmentMode::BatchApi { model, .. } => {
                ("anthropic-batch".to_string(), model.clone())
            }
            EnrichmentMode::Cli { model, .. } => ("cli".to_string(), model.clone()),
        }
    }

    /// Construct from environment.
    ///
    /// - `EVAL_ENRICHMENT=cloud` (or `batch`) → `BatchApi` (requires `ANTHROPIC_API_KEY`).
    /// - `EVAL_ENRICHMENT=cli` → `Cli` (uses Max OAuth via `claude -p`).
    /// - Anything else (default) → `OnDevice`, model selected by `EVAL_LOCAL_MODEL`.
    ///
    /// CLI knobs:
    /// - `EVAL_ENRICHMENT_CLI_MODEL` (default `haiku`)
    /// - `EVAL_ENRICHMENT_BATCH_SIZE_ENTITIES` (default 1; ≥2 selects batched path)
    /// - `EVAL_ENRICHMENT_BATCH_SIZE_TITLES` (default 1; ≥2 selects batched path)
    /// - `EVAL_ENRICHMENT_ROTATION` (default 3 calls/session)
    /// - `EVAL_ENRICHMENT_RETRIES` (default 3)
    /// - `EVAL_ENRICHMENT_COST_CAP_USD` (default $5; suggest $20 for LME)
    pub fn from_env(answer_model: &str, cost_cap_usd: f64) -> Result<Self, WenlanError> {
        let mode = std::env::var("EVAL_ENRICHMENT").unwrap_or_else(|_| "local".into());
        match mode.as_str() {
            "cloud" | "batch" => {
                let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
                    WenlanError::Generic("EVAL_ENRICHMENT=cloud requires ANTHROPIC_API_KEY".into())
                })?;
                Ok(Self::BatchApi {
                    api_key,
                    model: answer_model.to_string(),
                    cost_cap_usd,
                })
            }
            "cli" => {
                let model =
                    std::env::var("EVAL_ENRICHMENT_CLI_MODEL").unwrap_or_else(|_| "haiku".into());
                let batch_entities: usize = std::env::var("EVAL_ENRICHMENT_BATCH_SIZE_ENTITIES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1)
                    .max(1);
                let batch_titles: usize = std::env::var("EVAL_ENRICHMENT_BATCH_SIZE_TITLES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1)
                    .max(1);
                let rotation: usize = std::env::var("EVAL_ENRICHMENT_ROTATION")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3)
                    .max(1);
                let retries: u32 = std::env::var("EVAL_ENRICHMENT_RETRIES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                let cli_cost_cap: f64 = std::env::var("EVAL_ENRICHMENT_COST_CAP_USD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5.0);
                let cache_dir: PathBuf = std::env::var("EVAL_ENRICHMENT_CACHE_DIR")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .map(PathBuf::from)
                    .or_else(eval_baselines_dir_override)
                    .unwrap_or_else(|| PathBuf::from("eval/baselines"));
                eprintln!(
                    "[enrichment] mode=cli model={} batch_entities={} batch_titles={} rotation={} retries={} cost_cap=${:.2} cache_dir={}",
                    model,
                    batch_entities,
                    batch_titles,
                    rotation,
                    retries,
                    cli_cost_cap,
                    cache_dir.display()
                );
                Ok(Self::Cli {
                    model,
                    batch_entities,
                    batch_titles,
                    rotation,
                    retries,
                    cost_cap_usd: cli_cost_cap,
                    cache_dir,
                })
            }
            _ => {
                let model_id = std::env::var("EVAL_LOCAL_MODEL").ok();
                let provider = match model_id.as_deref() {
                    Some(m) => crate::llm_provider::OnDeviceProvider::new_with_model(Some(m)),
                    None => crate::llm_provider::OnDeviceProvider::new(),
                }
                .map_err(|e| {
                    WenlanError::Generic(format!(
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

/// Override for the eval baselines directory. When set, takes precedence over
/// the per-test default (`{CARGO_MANIFEST_DIR}/eval/baselines`).
///
/// Use case: keep the per-scenario DB cache (and chained Phase 1 / Phase 3 / judge
/// JSONL caches) outside any single worktree so it survives `git worktree remove`
/// and parallel sessions can share it.
///
/// Recommended location: `~/.cache/origin-eval` (XDG-style).
///
/// Returns `None` if the variable is unset or empty (an empty string is treated
/// as unset, since `PathBuf::from("")` resolves to cwd which is rarely intended).
///
/// # Example
/// ```bash
/// export EVAL_BASELINES_DIR=$HOME/.cache/origin-eval
/// cargo test -p origin --test eval_harness generate_fullpipeline_locomo -- --ignored
/// ```
pub fn eval_baselines_dir_override() -> Option<std::path::PathBuf> {
    std::env::var("EVAL_BASELINES_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
}

/// Enrich a freshly-seeded eval DB: entity extraction + title enrichment + concept distillation.
///
/// Caller must check `mem_count == enriched_count` before calling — this function unconditionally
/// re-enriches everything it sees. Returns `(entities, titles, concepts)` counts.
pub async fn enrich_db_for_eval(
    db: &MemoryDB,
    mode: &EnrichmentMode,
) -> Result<(usize, usize, usize), WenlanError> {
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
        EnrichmentMode::Cli {
            model,
            batch_entities,
            batch_titles,
            rotation,
            retries,
            cost_cap_usd,
            cache_dir,
        } => {
            // Run entity extraction first (Phase A), then titles (Phase B). They use
            // separate sessions so the schema doesn't switch mid-conversation (the
            // model can drift if entity-schema and title-schema interleave under
            // --resume — same root cause as judge schema drift).
            let entities = run_entity_extraction_for_eval_cli(
                db,
                model,
                *batch_entities,
                *rotation,
                *retries,
                *cost_cap_usd,
                cache_dir,
            )
            .await?;
            let titles = run_title_enrichment_for_eval_cli(
                db,
                model,
                *batch_titles,
                *rotation,
                *retries,
                *cost_cap_usd,
                cache_dir,
            )
            .await?;
            // Concept distillation: not yet ported to CLI. Skip for now; user can
            // separately run on-device or Batch API distillation if needed.
            // TODO: add Cli concept distillation in a follow-up if usage warrants.
            eprintln!(
                "[enrichment-cli] entities={} titles={} concepts=0 (distillation not implemented in CLI mode)",
                entities, titles
            );
            Ok((entities, titles, 0))
        }
    }
}

// ============================================================================
// Phase 1 enrichment via `claude -p` CLI (selected by EVAL_ENRICHMENT=cli).
// ============================================================================
//
// Mirrors the judge.rs batched + persistent pattern: strict prompts, --resume
// cache reuse, session rotation every N calls, JSONL persistence, retry with
// exp backoff, cost telemetry, hard cost cap.
//
// Scope: entity extraction + title enrichment only. Observations, relations,
// and concept distillation remain on-device or Batch-API-only paths. CLI mode
// is intended for interactive dev iteration on LoCoMo-scale benchmarks where
// API balance is unavailable but Max OAuth is.
//
// JSONL caches:
// - app/eval/baselines/_enrichment_entities_cli.jsonl
// - app/eval/baselines/_enrichment_titles_cli.jsonl
// (path is db-relative — caller controls placement via env if needed).

const ENRICH_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntityRecordCli {
    name: String,
    #[serde(default)]
    entity_type: String,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntityCacheRecord {
    schema_version: u32,
    memory_id: String,
    entities: Vec<EntityRecordCli>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TitleCacheRecord {
    schema_version: u32,
    memory_id: String,
    title: String,
}

fn strict_batch_entity_prompt(memories: &[(String, String)]) -> String {
    let mut s = String::with_capacity(2048 + memories.len() * 512);
    s.push_str("You will extract named entities (people, places, organizations, events) from each numbered memory below. Be CONSERVATIVE.\n\n");
    s.push_str("Rules:\n");
    s.push_str("- Extract ONLY entities explicitly mentioned in the memory text.\n");
    s.push_str("- Do not invent entities or infer from context.\n");
    s.push_str("- If a memory has no clear entities, output an empty list.\n");
    s.push_str("- Use type labels: person, place, organization, event, other.\n\n");
    s.push_str(&format!(
        "Return JSON object {{\"results\":[{{idx, memory_id, entities:[{{name, type, confidence}}]}}]}} with exactly {} entries in input order.\n\nMemories:\n",
        memories.len()
    ));
    for (i, (mid, content)) in memories.iter().enumerate() {
        let truncated: String = content.chars().take(500).collect();
        s.push_str(&format!("[{}] memory_id={}\n{}\n\n", i, mid, truncated));
    }
    s
}

fn strict_batch_title_prompt(memories: &[(String, String)]) -> String {
    let mut s = String::with_capacity(1024 + memories.len() * 256);
    s.push_str("You will generate a short title (3-5 words) for each numbered memory below.\n\n");
    s.push_str("Rules:\n");
    s.push_str("- Title must reflect what the memory ACTUALLY states.\n");
    s.push_str("- Do not paraphrase facts the memory doesn't contain.\n");
    s.push_str("- Avoid generic titles like 'Note' or 'Conversation' — be specific.\n\n");
    s.push_str(&format!(
        "Return JSON object {{\"results\":[{{idx, memory_id, title}}]}} with exactly {} entries in input order.\n\nMemories:\n",
        memories.len()
    ));
    for (i, (mid, content)) in memories.iter().enumerate() {
        let truncated: String = content.chars().take(300).collect();
        s.push_str(&format!("[{}] memory_id={}\n{}\n\n", i, mid, truncated));
    }
    s
}

fn parse_entity_envelope(stdout: &str) -> Option<Vec<(usize, String, Vec<EntityRecordCli>)>> {
    use crate::eval::cli_batch::strip_markdown_fence;
    let trimmed = stdout.trim();
    let env: serde_json::Value = serde_json::from_str(trimmed).ok()?;

    let extract = |arr: &[serde_json::Value]| -> Vec<(usize, String, Vec<EntityRecordCli>)> {
        arr.iter()
            .filter_map(|v| {
                let idx = v.get("idx").and_then(|x| x.as_u64())? as usize;
                let memory_id = v.get("memory_id").and_then(|x| x.as_str())?.to_string();
                let entities = v
                    .get("entities")
                    .and_then(|x| x.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|e| {
                                let name = e.get("name").and_then(|x| x.as_str())?.to_string();
                                let entity_type = e
                                    .get("type")
                                    .and_then(|x| x.as_str())
                                    .unwrap_or("other")
                                    .to_string();
                                let confidence = e
                                    .get("confidence")
                                    .and_then(|x| x.as_f64())
                                    .map(|v| v as f32);
                                Some(EntityRecordCli {
                                    name,
                                    entity_type,
                                    confidence,
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Some((idx, memory_id, entities))
            })
            .collect()
    };

    if let Some(results) = env
        .get("structured_output")
        .and_then(|v| v.get("results"))
        .and_then(|v| v.as_array())
    {
        if !results.is_empty() {
            return Some(extract(results));
        }
    }
    if let Some(result_str) = env.get("result").and_then(|v| v.as_str()) {
        let stripped = strip_markdown_fence(result_str);
        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(&stripped) {
            if let Some(results) = inner.get("results").and_then(|v| v.as_array()) {
                if !results.is_empty() {
                    return Some(extract(results));
                }
            }
        }
    }
    None
}

fn parse_title_envelope(stdout: &str) -> Option<Vec<(usize, String, String)>> {
    use crate::eval::cli_batch::strip_markdown_fence;
    let trimmed = stdout.trim();
    let env: serde_json::Value = serde_json::from_str(trimmed).ok()?;

    let extract = |arr: &[serde_json::Value]| -> Vec<(usize, String, String)> {
        arr.iter()
            .filter_map(|v| {
                let idx = v.get("idx").and_then(|x| x.as_u64())? as usize;
                let memory_id = v.get("memory_id").and_then(|x| x.as_str())?.to_string();
                let title = v.get("title").and_then(|x| x.as_str())?.to_string();
                Some((idx, memory_id, title))
            })
            .collect()
    };

    if let Some(results) = env
        .get("structured_output")
        .and_then(|v| v.get("results"))
        .and_then(|v| v.as_array())
    {
        if !results.is_empty() {
            return Some(extract(results));
        }
    }
    if let Some(result_str) = env.get("result").and_then(|v| v.as_str()) {
        let stripped = strip_markdown_fence(result_str);
        if let Ok(inner) = serde_json::from_str::<serde_json::Value>(&stripped) {
            if let Some(results) = inner.get("results").and_then(|v| v.as_array()) {
                if !results.is_empty() {
                    return Some(extract(results));
                }
            }
        }
    }
    None
}

/// CLI-batched entity extraction. Batched + --resume + persistence + retry.
#[allow(clippy::too_many_arguments)]
pub async fn run_entity_extraction_for_eval_cli(
    db: &MemoryDB,
    model: &str,
    batch_size: usize,
    rotation_calls: usize,
    max_retries: u32,
    cost_cap_usd: f64,
    cache_dir: &Path,
) -> Result<usize, WenlanError> {
    use crate::eval::cli_batch::run_cli_batch_subprocess;
    use std::collections::{HashMap, HashSet};
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader, Write};

    let all_unlinked = db.get_unlinked_memories(100_000).await?;
    if all_unlinked.is_empty() {
        eprintln!("[enrich-cli-entities] no unlinked memories");
        db.mark_all_memories_enriched_for_eval().await?;
        return Ok(0);
    }

    let cache_path = cache_dir.join("_enrichment_entities_cli.jsonl");

    // Load cached records.
    let mut cached: HashMap<String, Vec<EntityRecordCli>> = HashMap::new();
    let mut bad_lines = 0usize;
    if cache_path.exists() {
        if let Ok(f) = std::fs::File::open(&cache_path) {
            for line in BufReader::new(f).lines().map_while(|l| l.ok()) {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<EntityCacheRecord>(&line) {
                    Ok(rec) if rec.schema_version == ENRICH_SCHEMA_VERSION => {
                        cached.insert(rec.memory_id, rec.entities);
                    }
                    Ok(_) => {}
                    Err(_) => bad_lines += 1,
                }
            }
        }
    }
    if bad_lines > 0 {
        eprintln!(
            "[enrich-cli-entities] WARN: skipped {} corrupt JSONL lines",
            bad_lines
        );
    }
    eprintln!(
        "[enrich-cli-entities] cache: {} existing | total memories: {}",
        cached.len(),
        all_unlinked.len()
    );

    // Filter: skip already-cached.
    let cached_ids: HashSet<&str> = cached.keys().map(|s| s.as_str()).collect();
    let todo: Vec<(String, String)> = all_unlinked
        .iter()
        .filter(|(mid, _)| !cached_ids.contains(mid.as_str()))
        .cloned()
        .collect();
    eprintln!(
        "[enrich-cli-entities] cache hits: {} | to call: {}",
        all_unlinked.len() - todo.len(),
        todo.len()
    );

    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent); // best-effort: cache dir may already exist
    }
    let mut cache_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cache_path)
        .ok();

    let json_schema = r#"{"type":"object","properties":{"results":{"type":"array","items":{"type":"object","properties":{"idx":{"type":"integer"},"memory_id":{"type":"string"},"entities":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"type":{"type":"string"},"confidence":{"type":"number"}},"required":["name","type"]}}},"required":["idx","memory_id","entities"]}}},"required":["results"]}"#;

    let total_batches = if todo.is_empty() {
        0
    } else {
        todo.len().div_ceil(batch_size)
    };
    let mut session_id: Option<String> = None;
    let mut calls_in_session = 0usize;
    let mut total_cost = 0.0f64;
    let mut succ_batches = 0usize;
    let mut fail_batches = 0usize;
    let mut retries = 0usize;
    let mut total_entities_cached: HashMap<String, Vec<EntityRecordCli>> = cached.clone();
    let mut aborted = false;

    for (batch_i, chunk) in todo.chunks(batch_size).enumerate() {
        if total_cost > cost_cap_usd {
            eprintln!(
                "[enrich-cli-entities] ABORT: cumulative cost ${:.4} > cap ${:.2}",
                total_cost, cost_cap_usd
            );
            aborted = true;
            break;
        }
        let prompt = strict_batch_entity_prompt(chunk);
        if calls_in_session >= rotation_calls {
            session_id = None;
            calls_in_session = 0;
        }

        let mut parsed_opt: Option<Vec<(usize, String, Vec<EntityRecordCli>)>> = None;
        let mut last_err: Option<String> = None;
        for attempt in 0..=max_retries {
            match run_cli_batch_subprocess(&prompt, model, json_schema, session_id.as_deref()).await
            {
                Ok((stdout, cost, sid)) => {
                    if let Some(c) = cost {
                        total_cost += c.cost_usd;
                    }
                    match parse_entity_envelope(&stdout) {
                        Some(parsed) if parsed.len() == chunk.len() => {
                            parsed_opt = Some(parsed);
                            if session_id.is_none() {
                                session_id = sid;
                                calls_in_session = 1;
                            } else {
                                calls_in_session += 1;
                            }
                            break;
                        }
                        _ => {
                            last_err = Some("parse mismatch or empty".into());
                            if attempt < max_retries {
                                retries += 1;
                                session_id = None;
                                calls_in_session = 0;
                                tokio::time::sleep(std::time::Duration::from_millis(
                                    500u64 * (1 << attempt),
                                ))
                                .await;
                            }
                        }
                    }
                }
                Err(e) => {
                    last_err = Some(e.to_string());
                    if attempt < max_retries {
                        retries += 1;
                        session_id = None;
                        calls_in_session = 0;
                        tokio::time::sleep(std::time::Duration::from_millis(
                            500u64 * (1 << attempt),
                        ))
                        .await;
                    }
                }
            }
        }

        match parsed_opt {
            Some(parsed) => {
                succ_batches += 1;
                for (i, (_idx, memory_id, entities)) in parsed.iter().enumerate() {
                    // Defensive: prefer the position-based memory_id from the chunk over
                    // the model's claimed memory_id, which may rarely drift.
                    let mid = chunk
                        .get(i)
                        .map(|t| t.0.clone())
                        .unwrap_or_else(|| memory_id.clone());
                    if let Some(file) = cache_file.as_mut() {
                        let rec = EntityCacheRecord {
                            schema_version: ENRICH_SCHEMA_VERSION,
                            memory_id: mid.clone(),
                            entities: entities.clone(),
                        };
                        if let Ok(line) = serde_json::to_string(&rec) {
                            let _ = writeln!(file, "{}", line); // best-effort: cache write
                            let _ = file.flush(); // best-effort: cache flush
                        }
                    }
                    total_entities_cached.insert(mid, entities.clone());
                }
                eprintln!(
                    "[enrich-cli-entities] batch {}/{} ok | call_in_session={} | cost ${:.4}",
                    batch_i + 1,
                    total_batches,
                    calls_in_session,
                    total_cost
                );
            }
            None => {
                fail_batches += 1;
                eprintln!(
                    "[enrich-cli-entities] batch {}/{} FAILED: {}",
                    batch_i + 1,
                    total_batches,
                    last_err.unwrap_or_else(|| "?".into())
                );
                session_id = None;
                calls_in_session = 0;
            }
        }
    }

    // Apply cached entities to DB.
    let mut entity_cache: HashMap<String, String> = HashMap::new();
    let mut total_linked = 0usize;
    let mut total_entities_count = 0usize;
    for (memory_id, entities) in &total_entities_cached {
        let mut first_id: Option<String> = None;
        for ent in entities {
            let extracted = crate::extract::ExtractedEntity {
                name: ent.name.clone(),
                entity_type: ent.entity_type.clone(),
            };
            match crate::importer::resolve_entity_bulk(
                db,
                &mut entity_cache,
                &extracted,
                "cli_eval",
            )
            .await
            {
                Ok((id, _)) => {
                    total_entities_count += 1;
                    if first_id.is_none() {
                        first_id = Some(id);
                    }
                }
                Err(e) => log::warn!("[enrich-cli-entities] entity create failed: {e}"),
            }
        }
        if let Some(eid) = first_id {
            if let Err(e) = db.update_memory_entity_id(memory_id, &eid).await {
                log::warn!(
                    "[enrich-cli-entities] update_memory_entity_id failed for memory={} entity={}: {}",
                    memory_id,
                    eid,
                    e
                );
            } else {
                total_linked += 1;
            }
        }
    }

    db.mark_all_memories_enriched_for_eval().await?;
    eprintln!(
        "[enrich-cli-entities] DONE: {} batches succ, {} failed, {} retries | aborted={} | total_cost=${:.4} | linked={} entities={}",
        succ_batches,
        fail_batches,
        retries,
        aborted,
        total_cost,
        total_linked,
        total_entities_count
    );

    Ok(total_linked)
}

/// CLI-batched title enrichment. Batched + --resume + persistence + retry.
#[allow(clippy::too_many_arguments)]
pub async fn run_title_enrichment_for_eval_cli(
    db: &MemoryDB,
    model: &str,
    batch_size: usize,
    rotation_calls: usize,
    max_retries: u32,
    cost_cap_usd: f64,
    cache_dir: &Path,
) -> Result<usize, WenlanError> {
    use crate::eval::cli_batch::run_cli_batch_subprocess;
    use std::collections::{HashMap, HashSet};
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader, Write};

    let candidates = db.get_memories_needing_title_enrichment().await?;
    if candidates.is_empty() {
        eprintln!("[enrich-cli-titles] no candidates");
        return Ok(0);
    }

    let cache_path = cache_dir.join("_enrichment_titles_cli.jsonl");
    let mut cached: HashMap<String, String> = HashMap::new();
    let mut bad_lines = 0usize;
    if cache_path.exists() {
        if let Ok(f) = std::fs::File::open(&cache_path) {
            for line in BufReader::new(f).lines().map_while(|l| l.ok()) {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<TitleCacheRecord>(&line) {
                    Ok(rec) if rec.schema_version == ENRICH_SCHEMA_VERSION => {
                        cached.insert(rec.memory_id, rec.title);
                    }
                    Ok(_) => {}
                    Err(_) => bad_lines += 1,
                }
            }
        }
    }
    if bad_lines > 0 {
        eprintln!(
            "[enrich-cli-titles] WARN: skipped {} corrupt JSONL lines",
            bad_lines
        );
    }

    let cached_ids: HashSet<&str> = cached.keys().map(|s| s.as_str()).collect();
    let todo: Vec<(String, String)> = candidates
        .iter()
        .filter(|(mid, _)| !cached_ids.contains(mid.as_str()))
        .cloned()
        .collect();
    eprintln!(
        "[enrich-cli-titles] cache hits: {} | to call: {} | total: {}",
        cached.len(),
        todo.len(),
        candidates.len()
    );

    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent); // best-effort: cache dir may already exist
    }
    let mut cache_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&cache_path)
        .ok();

    let json_schema = r#"{"type":"object","properties":{"results":{"type":"array","items":{"type":"object","properties":{"idx":{"type":"integer"},"memory_id":{"type":"string"},"title":{"type":"string"}},"required":["idx","memory_id","title"]}}},"required":["results"]}"#;

    let total_batches = if todo.is_empty() {
        0
    } else {
        todo.len().div_ceil(batch_size)
    };
    let mut session_id: Option<String> = None;
    let mut calls_in_session = 0usize;
    let mut total_cost = 0.0f64;
    let mut succ = 0usize;
    let mut fail_batches = 0usize;
    let mut retries = 0usize;
    let mut new_titles: HashMap<String, String> = cached.clone();
    let mut aborted = false;

    for (batch_i, chunk) in todo.chunks(batch_size).enumerate() {
        if total_cost > cost_cap_usd {
            eprintln!(
                "[enrich-cli-titles] ABORT: cumulative cost ${:.4} > cap ${:.2}",
                total_cost, cost_cap_usd
            );
            aborted = true;
            break;
        }
        let prompt = strict_batch_title_prompt(chunk);
        if calls_in_session >= rotation_calls {
            session_id = None;
            calls_in_session = 0;
        }

        let mut parsed_opt: Option<Vec<(usize, String, String)>> = None;
        let mut last_err: Option<String> = None;
        for attempt in 0..=max_retries {
            match run_cli_batch_subprocess(&prompt, model, json_schema, session_id.as_deref()).await
            {
                Ok((stdout, cost, sid)) => {
                    if let Some(c) = cost {
                        total_cost += c.cost_usd;
                    }
                    match parse_title_envelope(&stdout) {
                        Some(parsed) if parsed.len() == chunk.len() => {
                            parsed_opt = Some(parsed);
                            if session_id.is_none() {
                                session_id = sid;
                                calls_in_session = 1;
                            } else {
                                calls_in_session += 1;
                            }
                            break;
                        }
                        _ => {
                            last_err = Some("parse mismatch or empty".into());
                            if attempt < max_retries {
                                retries += 1;
                                session_id = None;
                                calls_in_session = 0;
                                tokio::time::sleep(std::time::Duration::from_millis(
                                    500u64 * (1 << attempt),
                                ))
                                .await;
                            }
                        }
                    }
                }
                Err(e) => {
                    last_err = Some(e.to_string());
                    if attempt < max_retries {
                        retries += 1;
                        session_id = None;
                        calls_in_session = 0;
                        tokio::time::sleep(std::time::Duration::from_millis(
                            500u64 * (1 << attempt),
                        ))
                        .await;
                    }
                }
            }
        }

        match parsed_opt {
            Some(parsed) => {
                for (i, (_idx, _claimed_id, title)) in parsed.iter().enumerate() {
                    let mid = chunk.get(i).map(|t| t.0.clone()).unwrap_or_default();
                    if let Some(file) = cache_file.as_mut() {
                        let rec = TitleCacheRecord {
                            schema_version: ENRICH_SCHEMA_VERSION,
                            memory_id: mid.clone(),
                            title: title.clone(),
                        };
                        if let Ok(line) = serde_json::to_string(&rec) {
                            let _ = writeln!(file, "{}", line); // best-effort: cache write
                            let _ = file.flush(); // best-effort: cache flush
                        }
                    }
                    new_titles.insert(mid, title.clone());
                }
                eprintln!(
                    "[enrich-cli-titles] batch {}/{} ok | cost ${:.4}",
                    batch_i + 1,
                    total_batches,
                    total_cost
                );
            }
            None => {
                fail_batches += 1;
                eprintln!(
                    "[enrich-cli-titles] batch {}/{} FAILED: {}",
                    batch_i + 1,
                    total_batches,
                    last_err.unwrap_or_else(|| "?".into())
                );
                session_id = None;
                calls_in_session = 0;
            }
        }
    }

    // Apply new titles to DB.
    for (memory_id, title) in &new_titles {
        match db.update_title(memory_id, title).await {
            Ok(_) => succ += 1,
            Err(e) => log::warn!("[enrich-cli-titles] update_title({memory_id}) failed: {e}"),
        }
    }

    eprintln!(
        "[enrich-cli-titles] DONE: {} titles applied | {} batches failed | {} retries | aborted={} | total_cost=${:.4}",
        succ, fail_batches, retries, aborted, total_cost
    );
    Ok(succ)
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
) -> Result<usize, WenlanError> {
    use futures::StreamExt;

    let candidates = db.get_memories_needing_title_enrichment().await?;
    let total_candidates = candidates.len();
    if total_candidates == 0 {
        return Ok(0);
    }

    let t0 = std::time::Instant::now();
    let mut processed = 0usize;

    let force = std::env::var("EVAL_FORCE_TITLE_ENRICHMENT").as_deref() == Ok("1");
    let results: Vec<_> = futures::stream::iter(candidates.into_iter().map(
        |(source_id, content)| async move {
            crate::post_ingest::enrich_title(db, &source_id, &content, llm, force).await
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

/// Backfill Phase-1 classification onto seeded memories by running the SHARED
/// `ingest::run_classification_enrichment` per memory with bounded concurrency.
///
/// This is the pass the eval seed previously LACKED. It writes importance (T8
/// salience), event_date (T11/T20 temporal), quality, structured_fields, and
/// retrieval_cue — the exact write-time signals the old `enrich_db_for_eval`
/// shortcut (entity + title + page only) never produced, which is why those
/// feature flags shipped merged-but-inert. Running the same `ingest` fn the
/// daemon store path runs keeps the eval seed and production in lockstep
/// (Google "Rules of ML", Rule #32).
///
/// Resumable: only memories with `importance IS NULL` are processed, so a fresh
/// seed classifies everything and a partially-classified DB fills only the gaps
/// (the Tier-2 additive-backfill case).
///
/// Uses the on-device / CLI `llm` per memory (2 calls each: classify + extract).
/// Free but slow at LME scale, like the sibling entity/title passes. The
/// Anthropic Batch API variant — one batched request set instead of ~12k live
/// calls — is a deliberate follow-up; `EVAL_ENRICHMENT=cloud` seeds currently get
/// entity/title/page via batch but would get classification through this
/// per-memory path. Track the batch-classify port separately before relying on
/// cloud-mode classification at scale.
///
/// Returns the count of memories processed.
pub async fn run_classification_for_eval_concurrent(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
    concurrency: usize,
) -> Result<usize, WenlanError> {
    use crate::prompts::PromptRegistry;
    use futures::StreamExt;

    let candidates = db.get_memories_needing_classification().await?;
    let total = candidates.len();
    if total == 0 {
        return Ok(0);
    }
    let prompts = Arc::new(PromptRegistry::load(&PromptRegistry::override_dir()));
    let t0 = std::time::Instant::now();
    let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let results: Vec<_> = futures::stream::iter(candidates.into_iter().map(
        |(source_id, content)| {
            let llm = llm.clone();
            let prompts = prompts.clone();
            let counter = counter.clone();
            async move {
                // Default opts: no agent overrides, placeholder type "fact" — the
                // classifier resolves the concrete type/domain/quality/importance
                // and the extractor resolves event_date/structured_fields/cue,
                // exactly as the production store path does for an un-typed capture.
                let opts = crate::ingest::EnrichmentOpts {
                    initial_memory_type: "fact".to_string(),
                    initial_domain: None,
                    rejected_explicit_domain: false,
                    initial_supersede_mode: "hide".to_string(),
                    initial_structured_fields: None,
                    agent_supplied_memory_type: false,
                    agent_supplied_profile_alias: false,
                    agent_supplied_structured_fields: false,
                };
                let outcome = crate::ingest::run_classification_enrichment(
                    db, &source_id, &content, &llm, &prompts, &opts,
                )
                .await;
                let n = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n.is_multiple_of(50) {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let rate = n as f64 / elapsed.max(0.001);
                    let eta = (total - n) as f64 / rate.max(0.001);
                    eprintln!(
                        "[enrich] phase=classify processed={}/{} elapsed={:.0}s rate={:.1}/s eta={:.0}s",
                        n, total, elapsed, rate, eta,
                    );
                }
                outcome
            }
        },
    ))
    .buffer_unordered(concurrency.max(1))
    .collect()
    .await;

    let processed = results.len();
    eprintln!(
        "    [classify_local] {}/{} memories classified",
        processed, total
    );
    Ok(processed)
}

/// On-device eval seed enrichment via the production code path.
///
/// Phase 1 (classification) runs the shared `ingest` classify+extract pass per
/// memory with `EVAL_ENRICHMENT_CONCURRENCY` parallelism (default 1 = serial),
/// writing importance / event_date / quality / structured_fields / retrieval_cue.
///
/// Phase 2 (entity link + extraction + title + page growth) is de-forked: it loops
/// over every primary memory in insertion order and calls the production
/// `post_ingest::run_post_ingest_enrichment` per memory — the SAME code the daemon
/// store path runs — so the eval seed and production share one enrichment path
/// (Google "Rules of ML", Rule #32). This restores the `auto_link_entity` step the
/// old eval fork lacked. Free but slow (Qwen3-4B serial ≈ several hours at LME scale).
///
/// Phase 3 (`refinery::distill_pages`) stays serial: it builds clusters by scanning
/// enrichment state written by prior phases, and splitting it would break the FK
/// ordering assumptions in `find_distillation_clusters`.
///
/// Returns REAL `(entity_links, titles, concepts)` counts via cheap COUNT queries.
///
/// For staged evals (e.g. `pipeline.rs` Flat/Enriched/Distilled), call the sub-steps
/// independently — `run_entity_extraction_for_eval`, `run_title_enrichment_for_eval`,
/// `refinery::distill_pages` — so each stage can be measured in isolation.
pub async fn enrich_db_for_eval_local(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
) -> Result<(usize, usize, usize), WenlanError> {
    use crate::prompts::PromptRegistry;
    use crate::tuning::{DistillationConfig, RefineryConfig};

    let concurrency: usize = std::env::var("EVAL_ENRICHMENT_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    eprintln!("    [enrich_local] concurrency={concurrency}");

    // Phase 1 (classification) FIRST — shared with production; write importance /
    // event_date / quality / structured_fields / retrieval_cue.
    let classified = run_classification_for_eval_concurrent(db, llm, concurrency).await?;
    eprintln!("    [enrich_local] {classified} memories classified (Phase 1)");

    // Phase 2 — de-forked: route through the production `run_post_ingest_enrichment`
    // per memory in insertion order. Matches the production single-memory path;
    // the windowed `find_recent_batch` batch-extraction branch does not fire on
    // historical-timestamp seeds (seeded rows are >30s old). This includes
    // `auto_link_entity` (the step the old fork lacked), so a verbatim entity-name
    // memory is auto-linked instead of re-extracted, matching the canonical arm.
    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let refinery = RefineryConfig::default();
    let distillation = DistillationConfig::default();

    let all_memories = db.get_all_source_memories_ordered().await?;
    let total_memories = all_memories.len();

    if batched_post_ingest_enabled() {
        let conc = std::env::var("EVAL_ENRICHMENT_CONCURRENCY")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        eprintln!(
            "    [enrich_local] Phase 2: {total_memories} memories via BATCHED post_ingest (conc={conc})..."
        );
        enrich_post_ingest_batched(
            db,
            llm,
            &prompts,
            &refinery,
            &distillation,
            &all_memories,
            conc,
        )
        .await?;
    } else {
        eprintln!(
            "    [enrich_local] Phase 2: {total_memories} memories via canonical post_ingest..."
        );
        let t2_start = std::time::Instant::now();

        for (i, (source_id, content)) in all_memories.iter().enumerate() {
            crate::post_ingest::run_post_ingest_enrichment(
                db,
                source_id,
                content,
                None, // entity_id: let auto_link_entity + extraction decide
                None,
                None,
                None,
                Some(llm),
                &prompts,
                &refinery,
                &distillation,
                None, // knowledge_path
                None, // cancel
                None, // precomputed_kg — Task 1.4 will re-route through the orchestrator
            )
            .await?;
            if (i + 1) % 20 == 0 || i + 1 == total_memories {
                let elapsed = t2_start.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 / elapsed.max(0.001);
                eprintln!(
                    "    [enrich_local] Phase 2: {}/{} done | {:.1}s elapsed | {:.1}/s",
                    i + 1,
                    total_memories,
                    elapsed,
                    rate,
                );
            }
        }

        let elapsed_phase2 = t2_start.elapsed();
        eprintln!(
            "    [enrich_local] Phase 2 done in {:.1}s ({:.2}s/memory avg)",
            elapsed_phase2.as_secs_f64(),
            elapsed_phase2.as_secs_f64() / total_memories.max(1) as f64,
        );
    }

    // Phase 3: distill pages (unchanged — already shared with production).
    let concepts =
        crate::refinery::distill_pages(db, Some(llm), &prompts, &distillation, None).await?;
    eprintln!("    [distill_local] {} concepts", concepts);

    // Report REAL counts (Phase 2 now runs entity + title together via
    // run_post_ingest_enrichment, so they're not tracked per-phase). Cheap COUNT
    // queries keep the shared seed log honest instead of fabricating memory-count
    // proxies. These are junction-edge + non-empty-title counts (logging-only); not
    // directly comparable to the BatchApi/Cli arms' entity-created semantics.
    let entity_links = db.count_memory_entity_links().await? as usize;
    let titles = db.count_nonempty_titles().await? as usize;
    Ok((entity_links, titles, concepts))
}

/// Returns true when `WENLAN_SEED_BATCHED_POSTINGEST` is set to `1`, `true`, `yes`, or `on`.
/// Default OFF — the existing serial Phase-2 loop is used when unset or any other value.
fn batched_post_ingest_enabled() -> bool {
    matches!(
        std::env::var("WENLAN_SEED_BATCHED_POSTINGEST")
            .unwrap_or_default()
            .to_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Two-pass batched Phase-2 orchestrator.
///
/// Pass 2a: pre-compute `extract_kg` for all memories in parallel via
/// `buffer_unordered(concurrency)`. Results are collected into a `HashMap`
/// keyed by `source_id`; errors are buffered (not `?`-aborted) so a single
/// memory's extraction failure does not cancel the entire batch.
///
/// Pass 2b: iterate `all_memories` in insertion order. For each memory, pull
/// the buffered `Result<Vec<KgExtractionResult>>` from the map:
/// - `Some(Ok(kg))` → pass the pre-computed KG to `run_post_ingest_enrichment`
///   (skips the inline extract call inside post_ingest).
/// - `Some(Err(_))` or missing → pass `None` (serial fallback: post_ingest
///   re-runs extract inline, recording the same entity_extract status as the
///   serial path).
///
/// Pass 2b is serial in insertion order so that `auto_link_entity` (step 1 of
/// post_ingest) sees entities created by earlier memories, preserving the same
/// DB state as the serial path (Rule #32).
pub(crate) async fn enrich_post_ingest_batched(
    db: &MemoryDB,
    llm: &Arc<dyn crate::llm_provider::LlmProvider>,
    prompts: &crate::prompts::PromptRegistry,
    refinery: &crate::tuning::RefineryConfig,
    distillation: &crate::tuning::DistillationConfig,
    all_memories: &[(String, String)], // (source_id, content), insertion order
    concurrency: usize,
) -> Result<(), WenlanError> {
    use futures::StreamExt;
    use std::collections::HashMap;

    let total = all_memories.len();
    let t2a_start = std::time::Instant::now();

    // Pass 2a — parallel KG extraction (fills GPU batch).
    // Clone source_id + content into the closure so the futures are 'static.
    let buffered: HashMap<String, Result<Vec<crate::extract::KgExtractionResult>, WenlanError>> =
        futures::stream::iter(all_memories.iter().map(|(source_id, content)| {
            let llm = llm.clone();
            let prompts_clone = prompts.clone();
            let sid = source_id.clone();
            let cnt = content.clone();
            async move {
                let result = crate::refinery::extract_kg(&llm, &prompts_clone, &cnt).await;
                (sid, result)
            }
        }))
        .buffer_unordered(concurrency.max(1))
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect();

    let elapsed_2a = t2a_start.elapsed();
    eprintln!(
        "    [enrich_local] Phase 2a (batch KG extract): {}/{} done in {:.1}s",
        buffered.len(),
        total,
        elapsed_2a.as_secs_f64(),
    );

    // Pass 2b — serial commit in insertion order.
    let t2b_start = std::time::Instant::now();
    let mut buffered = buffered; // rebind as mut so we can .remove()
    for (i, (source_id, content)) in all_memories.iter().enumerate() {
        let precomputed = match buffered.remove(source_id.as_str()) {
            Some(Ok(kg)) => Some(kg),
            _ => None, // Err or missing → serial fallback (post_ingest re-extracts inline)
        };
        crate::post_ingest::run_post_ingest_enrichment(
            db,
            source_id,
            content,
            None, // entity_id: let auto_link_entity + extraction decide
            None,
            None,
            None,
            Some(llm),
            prompts,
            refinery,
            distillation,
            None, // knowledge_path
            None, // cancel
            precomputed,
        )
        .await?;
        if (i + 1) % 20 == 0 || i + 1 == total {
            let elapsed = t2b_start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed.max(0.001);
            eprintln!(
                "    [enrich_local] Phase 2 (batched): {}/{} done | {:.1}s elapsed | {:.1}/s",
                i + 1,
                total,
                elapsed,
                rate,
            );
        }
    }

    let elapsed_2b = t2b_start.elapsed();
    eprintln!(
        "    [enrich_local] Phase 2 (batched) done in {:.1}s ({:.2}s/memory avg)",
        elapsed_2b.as_secs_f64(),
        elapsed_2b.as_secs_f64() / total.max(1) as f64,
    );

    Ok(())
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
) -> Result<usize, WenlanError> {
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
        .map_err(|e| WenlanError::Generic(format!("client: {e}")))?;

    let batch_id = submit_batch(&client, api_key, batch_requests, model, cost_cap_usd)
        .await
        .map_err(|e| WenlanError::Generic(format!("title batch submit: {e}")))?;
    eprintln!("[batch_title] Batch submitted: {}", batch_id);

    let results_url = poll_batch(&client, api_key, &batch_id)
        .await
        .map_err(|e| WenlanError::Generic(format!("title batch poll: {e}")))?;

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| WenlanError::Generic(format!("title batch download: {e}")))?;

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
/// Replaces production `distill_pages` (which uses sequential on-device LLM)
/// with a batch API approach. Same DB queries and PageWrite-backed concept
/// storage, different LLM execution model.
///
/// Two batch submissions: refinement (merge/split clusters), then synthesis.
pub async fn run_concept_distillation_batch_api(
    db: &MemoryDB,
    api_key: &str,
    model: &str,
    cost_cap_usd: f64,
) -> Result<usize, WenlanError> {
    use crate::eval::anthropic::{download_batch_results, poll_batch, submit_batch};
    use crate::prompts::PromptRegistry;
    use crate::tuning::DistillationConfig;

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let tuning = DistillationConfig::default();

    // Use Haiku's synthesis limit (200K context, generous)
    let token_limit = 16_000;
    let clusters = db
        .find_distillation_clusters(
            tuning.formation_threshold,
            tuning.page_min_cluster_size,
            tuning.max_clusters_per_steep,
            token_limit,
            tuning.max_unlinked_cluster_size,
            tuning.max_grouped_cluster_size,
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
        space: Option<String>,
        source_ids: Vec<String>,
    }
    let mut batch_requests: Vec<(String, String, Option<String>, usize)> = Vec::new();
    let mut cluster_meta: Vec<ClusterMeta> = Vec::new();

    for (idx, cluster) in clusters.iter().enumerate() {
        let topic = cluster
            .entity_name
            .as_deref()
            .or(cluster.space.as_deref())
            .unwrap_or("general");

        // Skip if concept with similar sources exists (Jaccard > 0.8)
        let overlap = db
            .max_page_overlap(&cluster.source_ids)
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
            Some(prompts.distill_page.clone()),
            2048,
        ));
        cluster_meta.push(ClusterMeta {
            idx,
            topic: topic.to_string(),
            entity_id: cluster.entity_id.clone(),
            space: cluster.space.clone(),
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
        .map_err(|e| WenlanError::Generic(format!("client: {e}")))?;

    let batch_id = submit_batch(&client, api_key, batch_requests, model, cost_cap_usd)
        .await
        .map_err(|e| WenlanError::Generic(format!("distill batch submit: {e}")))?;
    eprintln!("[batch_distill] Batch submitted: {}", batch_id);

    let results_url = poll_batch(&client, api_key, &batch_id)
        .await
        .map_err(|e| WenlanError::Generic(format!("distill batch poll: {e}")))?;

    let raw_results = download_batch_results(&client, api_key, &results_url)
        .await
        .map_err(|e| WenlanError::Generic(format!("distill batch download: {e}")))?;

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
        .map_err(|e| WenlanError::Generic(format!("ctitle batch submit: {e}")))?;

    let title_results_url = poll_batch(&client, api_key, &title_batch_id)
        .await
        .map_err(|e| WenlanError::Generic(format!("ctitle batch poll: {e}")))?;

    let title_results = download_batch_results(&client, api_key, &title_results_url)
        .await
        .map_err(|e| WenlanError::Generic(format!("ctitle batch download: {e}")))?;

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

        store_batch_distilled_page(
            db,
            BatchDistilledPage {
                title,
                summary,
                content: content.clone(),
                entity_id: meta.entity_id.clone(),
                space: meta.space.clone(),
                source_memory_ids: meta.source_ids.clone(),
            },
            &tuning,
        )
        .await?;

        distilled += 1;
    }

    eprintln!("[batch_distill] Distilled {} concepts", distilled);
    Ok(distilled)
}

struct BatchDistilledPage {
    title: String,
    summary: Option<String>,
    content: String,
    entity_id: Option<String>,
    space: Option<String>,
    source_memory_ids: Vec<String>,
}

async fn store_batch_distilled_page(
    db: &MemoryDB,
    page: BatchDistilledPage,
    tuning: &crate::tuning::DistillationConfig,
) -> Result<String, WenlanError> {
    let write = crate::post_write::create_page_with_tuning(
        db,
        wenlan_types::requests::CreateConceptRequest {
            title: page.title,
            content: page.content,
            summary: page.summary,
            entity_id: page.entity_id,
            space: page.space.clone(),
            source_memory_ids: page.source_memory_ids,
            creation_kind: Some("distilled".to_string()),
            workspace: page.space,
        },
        "system",
        None,
        tuning.page_min_cluster_size,
        tuning.page_match_threshold,
    )
    .await?;

    Ok(write.id)
}

/// Static (DB-free) part of the CE-path G3 touch probe. Separated from the
/// DB-dependent dispatch so the predicate table is unit-testable.
///   rerank_skip_pref        -> is_preference_query(question): the bypass IS the channel.
///   rerank / rerank_model_* -> CE top-k ids differ from base top-k ids.
///   everything else         -> None (no honest probe; verdict keeps its star).
pub fn ce_channel_touched_static(
    feature: &str,
    question: &str,
    base_ids: &[&str],
    ce_ids: &[&str],
) -> Option<bool> {
    if feature == "rerank_skip_pref" {
        return Some(crate::router::classify::is_preference_query(question));
    }
    if feature == "rerank" || feature.starts_with("rerank_model") {
        return Some(base_ids != ce_ids);
    }
    None
}

/// Full CE-path probe: adds the DB-dependent arms.
pub async fn ce_channel_touched(
    db: &crate::db::MemoryDB,
    feature: &str,
    question: &str,
    base_ids: &[&str],
    ce_ids: &[&str],
) -> Result<Option<bool>, WenlanError> {
    if feature == "rerank_graph_stack" {
        return Ok(Some(db.graph_stream_touches(question, 10).await?));
    }
    Ok(ce_channel_touched_static(
        feature, question, base_ids, ce_ids,
    ))
}

/// Base-path probe (search_memory collectors).
pub async fn base_channel_touched(
    db: &crate::db::MemoryDB,
    feature: &str,
    question: &str,
) -> Result<Option<bool>, WenlanError> {
    if feature.contains("graph_stream") {
        return Ok(Some(db.graph_stream_touches(question, 10).await?));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs2::FileExt;

    #[test]
    fn eval_distillation_cluster_callers_use_formation_threshold() {
        // Production callers must group distillation clusters at
        // `formation_threshold` (0.60), matching what `distill_pages_scoped`
        // ships — else eval-seed pages form at a different threshold than
        // production and reintroduce seed<->production skew (Rule #32 / ONE
        // route, ONE contract). Scan only the PRODUCTION region of each file
        // (truncated at the `#[cfg(test)]` boundary) so this guard never
        // matches its own test literals.
        let sources = [
            ("eval/shared.rs", include_str!("shared.rs")),
            ("eval/lifecycle.rs", include_str!("lifecycle.rs")),
            ("eval/pipeline.rs", include_str!("pipeline.rs")),
        ];

        let mut failures = Vec::new();
        for (path, full_source) in sources {
            let source = full_source
                .split_once("\n#[cfg(test)]")
                .map(|(prod, _)| prod)
                .unwrap_or(full_source);
            let lines: Vec<_> = source.lines().collect();
            for (idx, line) in lines.iter().enumerate() {
                if !line.contains(".find_distillation_clusters(") {
                    continue;
                }

                let call_window = lines
                    .iter()
                    .enumerate()
                    .skip(idx)
                    .take(8)
                    .map(|(line_idx, line)| (line_idx + 1, *line));

                for (line_no, call_line) in call_window {
                    if call_line.contains(".similarity_threshold") {
                        failures.push(format!("{path}:{line_no}: {call_line}"));
                    }
                }
            }
        }

        assert!(
            failures.is_empty(),
            "eval distillation cluster discovery must use formation_threshold, \
             matching production distill_pages cluster formation:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn scenario_lock_blocks_concurrent_acquire() {
        let tmp = tempfile::tempdir().unwrap();
        let lock_path = tmp.path().join("scenario.lock");
        let lock1 = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        FileExt::try_lock_exclusive(&lock1).unwrap();
        let lock2 = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .unwrap();
        assert!(FileExt::try_lock_exclusive(&lock2).is_err());
    }

    #[test]
    fn cache_env_stamp_round_trips() {
        let want = ScenarioCacheEnv {
            schema_db_version: 99,
            migrations_hash: "abc123".to_string(),
            enricher_provider: "on-device".to_string(),
            enricher_model: "qwen3-4b".to_string(),
        };
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cache_env.json");
        write_cache_env_stamp(&path, &want);
        let got: ScenarioCacheEnv = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(got, want);
    }

    #[tokio::test]
    async fn batch_distilled_page_storage_uses_page_write_provenance() {
        let dir = tempfile::TempDir::new().unwrap();
        let db = MemoryDB::new(
            dir.path().join("test.db").as_path(),
            Arc::new(crate::events::NoopEmitter),
        )
        .await
        .unwrap();

        let sources = [
            (
                "mem-eval-pagewrite-a",
                "Rust references track ownership so borrowed values remain memory safe.",
            ),
            (
                "mem-eval-pagewrite-b",
                "Rust lifetimes describe how references stay valid across function calls.",
            ),
            (
                "mem-eval-pagewrite-c",
                "Rust borrowing rules prevent mutable aliasing while references are live.",
            ),
        ];
        let docs = sources
            .iter()
            .map(|(source_id, content)| RawDocument {
                source: "memory".to_string(),
                source_id: (*source_id).to_string(),
                title: content.chars().take(40).collect(),
                summary: None,
                content: (*content).to_string(),
                url: None,
                last_modified: chrono::Utc::now().timestamp(),
                metadata: std::collections::HashMap::new(),
                memory_type: Some("fact".to_string()),
                space: Some("engineering".to_string()),
                source_agent: Some("test".to_string()),
                confidence: Some(0.9),
                confirmed: Some(true),
                stability: Some("stable".to_string()),
                supersedes: None,
                pending_revision: false,
                entity_id: None,
                quality: None,
                importance: None,
                is_recap: false,
                enrichment_status: "raw".to_string(),
                supersede_mode: "hide".to_string(),
                structured_fields: None,
                retrieval_cue: None,
                source_text: None,
                content_hash: None,
            })
            .collect();
        db.upsert_documents(docs).await.unwrap();

        let tuning = crate::tuning::DistillationConfig::default();
        let page_id = store_batch_distilled_page(
            &db,
            BatchDistilledPage {
                title: "Rust Reference Safety".to_string(),
                summary: Some("Rust reference safety".to_string()),
                content: "Rust references track ownership, borrowing, and lifetimes so borrowed values remain memory safe without mutable aliasing.".to_string(),
                entity_id: None,
                space: Some("engineering".to_string()),
                source_memory_ids: sources
                    .iter()
                    .map(|(source_id, _)| (*source_id).to_string())
                    .collect(),
            },
            &tuning,
        )
        .await
        .unwrap();

        let page = db.get_page(&page_id).await.unwrap().unwrap();
        assert_eq!(page.review_status, "unconfirmed");

        let activity = db.list_agent_activity(20, None, None).await.unwrap();
        assert!(
            activity.iter().any(|a| {
                a.action == "page_create"
                    && a.memory_ids.as_deref()
                        == Some("mem-eval-pagewrite-a,mem-eval-pagewrite-b,mem-eval-pagewrite-c")
            }),
            "expected PageWrite page_create provenance row, got: {activity:?}"
        );
    }

    /// The eval classification pass must close the gap the legacy shortcut left:
    /// run the shared `ingest::run_classification_enrichment` over seeded memories
    /// and backfill `importance` (T8) + `event_date` (T11/T20) + the refined type,
    /// so `get_memories_needing_classification` empties out afterward.
    #[tokio::test]
    async fn classification_pass_backfills_importance_via_shared_ingest() {
        use crate::llm_provider::{LlmProvider, SequencedMockProvider};
        use crate::sources::RawDocument;

        let dir = tempfile::TempDir::new().unwrap();
        let db = MemoryDB::new(
            dir.path().join("test.db").as_path(),
            Arc::new(crate::events::NoopEmitter),
        )
        .await
        .unwrap();

        db.create_space("work", None, false).await.unwrap();

        let source_id = "mem_eval_classify_1";
        let content = "Moved the launch review to 2026-02-10 since the demo slipped a week.";
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: content.chars().take(40).collect(),
            summary: None,
            content: content.to_string(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            metadata: std::collections::HashMap::new(),
            memory_type: Some("fact".to_string()),
            space: None,
            source_agent: Some("test".to_string()),
            confidence: Some(0.7),
            confirmed: Some(false),
            stability: Some("new".to_string()),
            supersedes: None,
            pending_revision: false,
            entity_id: None,
            quality: None,
            importance: None,
            is_recap: false,
            enrichment_status: "raw".to_string(),
            supersede_mode: "hide".to_string(),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
        };
        db.upsert_documents(vec![doc]).await.unwrap();

        // Before: the memory needs classification (importance IS NULL).
        let before = db.get_memories_needing_classification().await.unwrap();
        assert_eq!(before.len(), 1);

        // classify call (idx 0) then extract call (idx 1).
        let llm: Arc<dyn LlmProvider> = Arc::new(SequencedMockProvider::new(vec![
            r#"{"memory_type":"decision","domain":"work","quality":"high","importance":6,"tags":["launch"]}"#,
            r#"{"event_date":"2026-02-10","retrieval_cue":"launch review reschedule"}"#,
        ]));

        let n = run_classification_for_eval_concurrent(&db, &llm, 1)
            .await
            .unwrap();
        assert_eq!(n, 1);

        // After: importance is set, so the memory drops out of the gap query.
        let after = db.get_memories_needing_classification().await.unwrap();
        assert!(
            after.is_empty(),
            "importance should be backfilled, still pending: {after:?}"
        );

        // The refined memory_type + domain were persisted via apply_enrichment.
        let (mt, space) = db.get_memory_classification(source_id).await.unwrap();
        assert_eq!(mt.as_deref(), Some("decision"));
        assert_eq!(space.as_deref(), Some("work"));
    }

    /// STEP 5 contract test: a cached scenario DB that is entity/title-enriched
    /// (`enriched == mem_count`) but was seeded BEFORE the Phase-1 classification
    /// pass existed must NOT be treated as a complete cache hit. `open_or_seed_
    /// scenario_db` must detect the missing classification and additively backfill
    /// it (non-destructive, resumable) rather than silently ship importance-NULL
    /// rows that starve the T8/T11/T20/T15 flags.
    #[tokio::test]
    async fn cache_hit_backfills_missing_classification() {
        use crate::llm_provider::{LlmProvider, SequencedMockProvider};
        use crate::sources::RawDocument;

        let dir = tempfile::TempDir::new().unwrap();
        let emb = eval_shared_embedder();

        let source_id = "mem_step5_cache_1";
        let content = "Picked Postgres over Mongo on 2026-03-01 for the billing service.";
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: content.chars().take(40).collect(),
            summary: None,
            content: content.to_string(),
            url: None,
            last_modified: chrono::Utc::now().timestamp(),
            metadata: std::collections::HashMap::new(),
            memory_type: Some("fact".to_string()),
            space: None,
            source_agent: Some("test".to_string()),
            confidence: Some(0.7),
            confirmed: Some(false),
            stability: Some("new".to_string()),
            supersedes: None,
            pending_revision: false,
            entity_id: None,
            quality: None,
            importance: None,
            is_recap: false,
            enrichment_status: "raw".to_string(),
            supersede_mode: "hide".to_string(),
            structured_fields: None,
            retrieval_cue: None,
            source_text: None,
            content_hash: None,
        };

        // Create the mode upfront so the cache stamp and the re-open use the same provenance.
        let llm: Arc<dyn LlmProvider> = Arc::new(SequencedMockProvider::new(vec![
            r#"{"memory_type":"decision","domain":"infra","quality":"high","importance":7,"tags":["db"]}"#,
            r#"{"event_date":"2026-03-01","retrieval_cue":"db choice billing"}"#,
        ]));
        let mode = EnrichmentMode::OnDevice(llm);

        // Build the "pre-classification cache" state: seed + mark enriched (the
        // entity/title marker) but leave importance NULL, and stamp cache_env so
        // the re-open is a stamp-match cache hit (not a wipe).
        {
            let db = MemoryDB::new_with_shared_embedder(
                dir.path(),
                Arc::new(crate::events::NoopEmitter),
                emb.clone(),
            )
            .await
            .unwrap();
            db.upsert_documents(vec![doc]).await.unwrap();
            db.mark_all_memories_enriched_for_eval().await.unwrap();
            assert_eq!(db.memory_count().await.unwrap(), 1);
            assert_eq!(db.enriched_memory_count().await.unwrap(), 1);
            // Importance is NULL -> classification is incomplete.
            assert_eq!(
                db.get_memories_needing_classification()
                    .await
                    .unwrap()
                    .len(),
                1
            );
            write_cache_env_stamp(
                &dir.path().join("cache_env.json"),
                &current_cache_env(&mode),
            );
        }

        // Empty seed closure: it's a cache hit, so no re-seed should occur.
        let db = open_or_seed_scenario_db(dir.path(), emb.clone(), Vec::new, &mode)
            .await
            .unwrap();

        // The cache-hit path backfilled classification: no gap remains, and the
        // refined type landed on the row.
        assert!(
            db.get_memories_needing_classification()
                .await
                .unwrap()
                .is_empty(),
            "cache-hit backfill should have classified the memory"
        );
        let (mt, _space) = db.get_memory_classification(source_id).await.unwrap();
        assert_eq!(mt.as_deref(), Some("decision"));
    }

    #[test]
    fn plan_channel_touch_probe_dispatch() {
        // skip_pref: pure preference-classifier predicate, no DB
        assert_eq!(
            ce_channel_touched_static("rerank_skip_pref", "any tips for hiking boots?", &[], &[]),
            Some(true)
        );
        assert_eq!(
            ce_channel_touched_static("rerank_skip_pref", "when did I visit Tokyo?", &[], &[]),
            Some(false)
        );
        // plain rerank / model arms: touched = CE top-k differs from base top-k
        assert_eq!(
            ce_channel_touched_static("rerank_model_turbo", "q", &["a", "b"], &["a", "b"]),
            Some(false)
        );
        assert_eq!(
            ce_channel_touched_static("rerank_model_turbo", "q", &["a", "b"], &["b", "a"]),
            Some(true)
        );
        assert_eq!(
            ce_channel_touched_static("rerank", "q", &["a"], &["b"]),
            Some(true)
        );
        // unprobed CE channels stay None (star is honest)
        assert_eq!(
            ce_channel_touched_static("page_channel", "q", &[], &[]),
            None
        );
    }

    /// Task 2.5 — provenance-stamped EVAL_MIGRATE_STALE.
    ///
    /// `decide_cache_stamp` must allow migration ONLY when the stored stamp's
    /// enricher provenance matches the current run's provenance. A schema-59 DB
    /// stamped with "on-device / qwen3.5-9b" migrates under EVAL_MIGRATE_STALE=1.
    /// A DB stamped with "anthropic-batch / haiku" — or an UNSTAMPED legacy DB
    /// (empty enricher fields from `#[serde(default)]`) — is REFUSED even with the
    /// flag. This prevents cloud-substrate laundering onto on-device headline runs.
    #[test]
    fn migrate_stale_requires_matching_provenance() {
        let want = ScenarioCacheEnv {
            schema_db_version: crate::db::SCHEMA_VERSION,
            migrations_hash: option_env!("WENLAN_MIGRATIONS_HASH")
                .unwrap_or("unknown")
                .to_string(),
            enricher_provider: "on-device".to_string(),
            enricher_model: "qwen3.5-9b".to_string(),
        };

        // Case 1: stored is schema-59 but same provider/model → MigrateStale.
        let stored_qwen = ScenarioCacheEnv {
            schema_db_version: 59,
            migrations_hash: "old-hash".to_string(),
            enricher_provider: "on-device".to_string(),
            enricher_model: "qwen3.5-9b".to_string(),
        };
        assert_eq!(
            decide_cache_stamp(Some(&stored_qwen), &want, true, true, false),
            CacheStampDecision::MigrateStale,
            "same provenance, schema diff + migrate_stale=true → MigrateStale"
        );

        // Case 2: stored is cloud-enriched (haiku) → Refuse even with migrate_stale.
        let stored_haiku = ScenarioCacheEnv {
            schema_db_version: 59,
            migrations_hash: "old-hash".to_string(),
            enricher_provider: "anthropic-batch".to_string(),
            enricher_model: "haiku".to_string(),
        };
        assert_eq!(
            decide_cache_stamp(Some(&stored_haiku), &want, true, true, false),
            CacheStampDecision::Refuse,
            "cloud provenance mismatch → Refuse even with migrate_stale=true"
        );

        // Case 3: stored is unstamped legacy (empty enricher fields) → Refuse.
        let stored_unstamped = ScenarioCacheEnv {
            schema_db_version: 59,
            migrations_hash: "old-hash".to_string(),
            enricher_provider: String::new(),
            enricher_model: String::new(),
        };
        assert_eq!(
            decide_cache_stamp(Some(&stored_unstamped), &want, true, true, false),
            CacheStampDecision::Refuse,
            "unstamped legacy DB → Refuse (must re-seed for on-device headline)"
        );

        // Case 4: same provenance but migrate_stale=false → Refuse (flag required).
        assert_eq!(
            decide_cache_stamp(Some(&stored_qwen), &want, true, false, false),
            CacheStampDecision::Refuse,
            "matching provenance but migrate_stale=false → Refuse"
        );

        // Case 5: exact stamp match → Reuse (no wipe, no migrate needed).
        assert_eq!(
            decide_cache_stamp(Some(&want), &want, true, false, false),
            CacheStampDecision::Reuse,
            "exact match → Reuse"
        );

        // Case 6: mismatch + allow_wipe → Wipe.
        assert_eq!(
            decide_cache_stamp(Some(&stored_haiku), &want, true, false, true),
            CacheStampDecision::Wipe,
            "mismatch + allow_wipe → Wipe"
        );
    }

    /// Smoke test: `enrich_post_ingest_batched` runs Pass 2a + 2b and links at
    /// least one entity for memory "m1". Uses `CannedLlmProvider` (CPU-only, no
    /// GPU, no seed). The KG JSON response key is a 30-char prefix of the
    /// `extract_knowledge_graph` prompt (mirrors `entity_extraction.rs` tests).
    #[tokio::test]
    async fn batched_orchestrator_links_entities() {
        use crate::llm_provider::{CannedLlmProvider, LlmProvider};
        use crate::prompts::PromptRegistry;
        use crate::sources::RawDocument;
        use crate::tuning::{DistillationConfig, RefineryConfig};
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let db = MemoryDB::new(
            dir.path().join("test.db").as_path(),
            Arc::new(crate::events::NoopEmitter),
        )
        .await
        .unwrap();

        // Upsert 3 memories whose content will trigger Alice-entity extraction.
        for id in &["m1", "m2", "m3"] {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: id.to_string(),
                title: format!("Test memory {id}"),
                content: format!("Alice works on project {id}"),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        let prompts = PromptRegistry::default();
        // Key = first 30 chars of extract_knowledge_graph prompt (same idiom as
        // entity_extraction.rs tests so the canned provider matches).
        let key_fragment: String = prompts.extract_knowledge_graph.chars().take(30).collect();
        // Respond with one entity ("Alice") so post_ingest can create + link it.
        let kg_json =
            r#"[{"entities":[{"name":"Alice","type":"person"}],"observations":[],"relations":[]}]"#;
        // Also need a fallback for any other prompts (auto_link_entity / title / classify).
        let llm: Arc<dyn LlmProvider> =
            Arc::new(CannedLlmProvider::new("{}").with(key_fragment, kg_json));

        let mems = db.get_all_source_memories_ordered().await.unwrap();
        assert_eq!(mems.len(), 3, "should have 3 seeded memories");

        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();

        enrich_post_ingest_batched(&db, &llm, &prompts, &refinery, &distillation, &mems, 8)
            .await
            .unwrap();

        // At least m1 must be linked to an entity after the orchestrator runs.
        let entity_id = db.get_memory_entity_id("m1").await.unwrap();
        assert!(
            entity_id.is_some(),
            "m1 should be linked to an entity after enrich_post_ingest_batched"
        );
    }

    // ── Task 1.5: WAY B golden gate ──────────────────────────────────────────
    //
    // Two tests prove that `enrich_post_ingest_batched` (WAY B) and the serial
    // per-memory `run_post_ingest_enrichment` loop (WAY A) produce a SEMANTICALLY
    // identical substrate when given the same mock LLM.
    //
    // Fixture memories use `source_agent = None` so `find_recent_batch` always
    // returns an empty Vec (the windowed-batch branch requires source_agent to be
    // Some). This makes the single-memory extract path structurally the only
    // branch taken in both WAY A and WAY B's serial-commit pass, guaranteeing
    // apples-to-apples comparison.

    /// Semantic substrate fingerprint extracted from a MemoryDB.
    ///
    /// All raw auto-generated IDs (entity id, page id, …) are excluded because
    /// they are assigned sequentially per-DB and will differ between two separate
    /// DBs even when content is identical. We compare names, types, links, and
    /// step records instead.
    #[derive(Debug, PartialEq, Eq)]
    struct SubstrateFingerprint {
        /// Sorted set of (name, entity_type) across all entities.
        entities: Vec<(String, String)>,
        /// Sorted list of (source_id, entity_name) derived from
        /// `memories.entity_id → entities.name` for chunk_index=0 primary memories.
        memory_entity_links: Vec<(String, String)>,
        /// Sorted list of (source_id, title) for chunk_index=0 primary memories.
        titles: Vec<(String, String)>,
        /// Sorted set of active page titles.
        page_titles: Vec<String>,
        /// Sorted set of (source_id, step_name, status) across all enrichment_steps.
        enrichment_steps: Vec<(String, String, String)>,
    }

    async fn collect_fingerprint(db: &MemoryDB) -> SubstrateFingerprint {
        let conn = db.conn.lock().await;

        // 1. Entities — (name, entity_type) sorted.
        let mut entities = Vec::new();
        {
            let mut rows = conn
                .query(
                    "SELECT name, entity_type FROM entities ORDER BY name, entity_type",
                    (),
                )
                .await
                .expect("entities query");
            while let Ok(Some(row)) = rows.next().await {
                entities.push((
                    row.get::<String>(0).unwrap_or_default(),
                    row.get::<String>(1).unwrap_or_default(),
                ));
            }
        }
        entities.sort();

        // 2. Memory→entity links — (source_id, entity_name) for chunk 0 rows.
        //    Uses memories.entity_id (the primary link column written by post_ingest).
        let mut memory_entity_links = Vec::new();
        {
            let mut rows = conn
                .query(
                    "SELECT m.source_id, e.name
                     FROM memories m
                     JOIN entities e ON e.id = m.entity_id
                     WHERE m.chunk_index = 0 AND m.source = 'memory'
                     ORDER BY m.source_id",
                    (),
                )
                .await
                .expect("memory_entity_links query");
            while let Ok(Some(row)) = rows.next().await {
                memory_entity_links.push((
                    row.get::<String>(0).unwrap_or_default(),
                    row.get::<String>(1).unwrap_or_default(),
                ));
            }
        }
        memory_entity_links.sort();

        // 3. Titles — (source_id, title) for chunk 0 primary memories.
        let mut titles = Vec::new();
        {
            let mut rows = conn
                .query(
                    "SELECT source_id, title FROM memories
                     WHERE chunk_index = 0 AND source = 'memory'
                     ORDER BY source_id",
                    (),
                )
                .await
                .expect("titles query");
            while let Ok(Some(row)) = rows.next().await {
                titles.push((
                    row.get::<String>(0).unwrap_or_default(),
                    row.get::<String>(1).unwrap_or_default(),
                ));
            }
        }
        titles.sort();

        // 4. Active page titles.
        let mut page_titles = Vec::new();
        {
            let mut rows = conn
                .query(
                    "SELECT title FROM pages WHERE status = 'active' ORDER BY title",
                    (),
                )
                .await
                .expect("page_titles query");
            while let Ok(Some(row)) = rows.next().await {
                page_titles.push(row.get::<String>(0).unwrap_or_default());
            }
        }
        page_titles.sort();

        // 5. Enrichment steps — (source_id, step_name, status) sorted.
        let mut enrichment_steps = Vec::new();
        {
            let mut rows = conn
                .query(
                    "SELECT source_id, step_name, status FROM enrichment_steps
                     ORDER BY source_id, step_name",
                    (),
                )
                .await
                .expect("enrichment_steps query");
            while let Ok(Some(row)) = rows.next().await {
                enrichment_steps.push((
                    row.get::<String>(0).unwrap_or_default(),
                    row.get::<String>(1).unwrap_or_default(),
                    row.get::<String>(2).unwrap_or_default(),
                ));
            }
        }
        enrichment_steps.sort();

        SubstrateFingerprint {
            entities,
            memory_entity_links,
            titles,
            page_titles,
            enrichment_steps,
        }
    }

    /// Helper: build two fresh in-memory DBs and upsert the same N fixture
    /// memories into each. Returns (db_a, db_b, mems) where mems is the ordered
    /// (source_id, content) list suitable for both serial and batched enrichment.
    ///
    /// IMPORTANT: `source_agent = None` so `find_recent_batch` always returns
    /// an empty Vec — the windowed-batch branch never fires.
    async fn make_two_dbs_with_fixtures(
        fixture_ids: &[&str],
        contents: &[&str],
    ) -> (MemoryDB, MemoryDB, Vec<(String, String)>) {
        use crate::sources::RawDocument;

        let dir_a = tempfile::TempDir::new().unwrap();
        let dir_b = tempfile::TempDir::new().unwrap();

        let db_a = MemoryDB::new(
            dir_a.path().join("a.db").as_path(),
            Arc::new(crate::events::NoopEmitter),
        )
        .await
        .unwrap();
        let db_b = MemoryDB::new(
            dir_b.path().join("b.db").as_path(),
            Arc::new(crate::events::NoopEmitter),
        )
        .await
        .unwrap();

        for (id, content) in fixture_ids.iter().zip(contents.iter()) {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: id.to_string(),
                title: format!("Title for {id}"),
                content: content.to_string(),
                source_agent: None, // REQUIRED: prevents find_recent_batch window branch
                ..Default::default()
            };
            db_a.upsert_documents(vec![doc.clone()]).await.unwrap();
            db_b.upsert_documents(vec![doc]).await.unwrap();
        }

        let mems = db_a.get_all_source_memories_ordered().await.unwrap();
        assert_eq!(
            mems.len(),
            fixture_ids.len(),
            "all fixtures inserted into db_a"
        );

        // db_b must have the same insertion order.
        let mems_b = db_b.get_all_source_memories_ordered().await.unwrap();
        assert_eq!(mems_b, mems, "db_b same insertion order as db_a");

        // tempdir handles are dropped here but the DB files remain because they were
        // opened by path — we must keep them alive. Leak them via forget.
        std::mem::forget(dir_a);
        std::mem::forget(dir_b);

        (db_a, db_b, mems)
    }

    /// Task 1.5 — Test 1: serial vs batched produce IDENTICAL substrates.
    ///
    /// WAY A: serial `run_post_ingest_enrichment(..., precomputed_kg=None)` per memory.
    /// WAY B: `enrich_post_ingest_batched(...)`.
    /// Both receive the same `CannedLlmProvider`; both must yield the same
    /// semantic fingerprint.
    #[tokio::test]
    async fn serial_and_batched_produce_identical_substrate() {
        use crate::llm_provider::{CannedLlmProvider, LlmProvider};
        use crate::prompts::PromptRegistry;
        use crate::tuning::{DistillationConfig, RefineryConfig};

        // Fixture: 3 memories about Alice (unique token) so CannedLlm extracts
        // "Alice" as an entity for each.
        let ids = ["gold_s1", "gold_s2", "gold_s3"];
        let contents = [
            "Alice is the project lead on the Atlas initiative.",
            "Alice reviewed the Atlas roadmap last Tuesday.",
            "Alice signed off on the Atlas budget this week.",
        ];
        let (db_a, db_b, mems) = make_two_dbs_with_fixtures(&ids, &contents).await;

        let prompts = PromptRegistry::default();
        // Key on the extract_knowledge_graph system prompt prefix (same idiom as
        // the existing smoke test). Respond with Alice as a person entity.
        let key_fragment: String = prompts.extract_knowledge_graph.chars().take(30).collect();
        let kg_json =
            r#"[{"entities":[{"name":"Alice","type":"person"}],"observations":[],"relations":[]}]"#;
        // Default response for classify/title/other prompts that don't match the KG key.
        let canned = Arc::new(CannedLlmProvider::new("{}").with(key_fragment, kg_json));
        let llm: Arc<dyn LlmProvider> = canned;

        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();

        // WAY A — serial, one call per memory, precomputed_kg = None.
        for (source_id, content) in &mems {
            crate::post_ingest::run_post_ingest_enrichment(
                &db_a,
                source_id,
                content,
                None, // entity_id
                None,
                None,
                None,
                Some(&llm),
                &prompts,
                &refinery,
                &distillation,
                None, // knowledge_path
                None, // cancel
                None, // precomputed_kg
            )
            .await
            .unwrap();
        }

        // WAY B — batched orchestrator (Pass 2a parallel extract + Pass 2b serial commit).
        enrich_post_ingest_batched(&db_b, &llm, &prompts, &refinery, &distillation, &mems, 8)
            .await
            .unwrap();

        let fp_a = collect_fingerprint(&db_a).await;
        let fp_b = collect_fingerprint(&db_b).await;

        assert_eq!(
            fp_a, fp_b,
            "WAY A serial and WAY B batched substrates must be identical.\n\
             Serial:  {fp_a:#?}\n\
             Batched: {fp_b:#?}"
        );

        // Sanity: entity link must be present (not just empty on both sides).
        assert!(
            !fp_a.memory_entity_links.is_empty(),
            "expected at least one memory→entity link; got none (mock LLM may not have fired)"
        );
    }

    /// Task 1.5 — Test 2: error-path equivalence (finding B).
    ///
    /// When one memory's KG extract ERRORS, serial and batched still produce the
    /// SAME substrate: the failing memory records `(sid, "entity_extract", "failed")`
    /// in BOTH paths, and the other memories are enriched normally.
    ///
    /// `CannedLlmProvider::fail_on(substr)` returns `Err` when `user_prompt`
    /// contains the unique token that appears only in the failing memory's content.
    #[tokio::test]
    async fn error_path_serial_and_batched_produce_identical_substrate() {
        use crate::llm_provider::{CannedLlmProvider, LlmProvider};
        use crate::prompts::PromptRegistry;
        use crate::tuning::{DistillationConfig, RefineryConfig};

        // Fixture: 3 memories. Memory "err_s2" contains a unique token
        // "__INJECT_FAIL__" that the error-injecting mock will match.
        let ids = ["err_s1", "err_s2", "err_s3"];
        let contents = [
            "Alice is the project lead on the Beta initiative.",
            "Alice __INJECT_FAIL__ reviewed the Beta roadmap.", // this one errors
            "Alice signed off on the Beta budget this week.",
        ];
        let (db_a, db_b, mems) = make_two_dbs_with_fixtures(&ids, &contents).await;

        let prompts = PromptRegistry::default();
        let key_fragment: String = prompts.extract_knowledge_graph.chars().take(30).collect();
        let kg_json =
            r#"[{"entities":[{"name":"Alice","type":"person"}],"observations":[],"relations":[]}]"#;

        // Mock that errors on any call whose user_prompt contains the unique token.
        let canned = Arc::new(
            CannedLlmProvider::new("{}")
                .with(key_fragment, kg_json)
                .fail_on("__INJECT_FAIL__"),
        );
        let llm: Arc<dyn LlmProvider> = canned;

        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();

        // WAY A — serial per memory.
        for (source_id, content) in &mems {
            // Errors from run_post_ingest_enrichment are swallowed into enrichment_steps
            // (the function records "failed" and returns Ok). We call .unwrap() here
            // to fail fast if the function signature changes to propagate errors.
            crate::post_ingest::run_post_ingest_enrichment(
                &db_a,
                source_id,
                content,
                None,
                None,
                None,
                None,
                Some(&llm),
                &prompts,
                &refinery,
                &distillation,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        }

        // WAY B — batched.
        enrich_post_ingest_batched(&db_b, &llm, &prompts, &refinery, &distillation, &mems, 8)
            .await
            .unwrap();

        let fp_a = collect_fingerprint(&db_a).await;
        let fp_b = collect_fingerprint(&db_b).await;

        assert_eq!(
            fp_a, fp_b,
            "Error-path: WAY A and WAY B must still match.\n\
             Serial:  {fp_a:#?}\n\
             Batched: {fp_b:#?}"
        );

        // Both must record the failing memory's step as "failed".
        let failing_step_a = fp_a
            .enrichment_steps
            .iter()
            .find(|(sid, step, _)| sid == "err_s2" && step == "entity_extract");
        let failing_step_b = fp_b
            .enrichment_steps
            .iter()
            .find(|(sid, step, _)| sid == "err_s2" && step == "entity_extract");

        assert_eq!(
            failing_step_a.map(|(_, _, status)| status.as_str()),
            Some("failed"),
            "serial: err_s2 entity_extract should be 'failed'"
        );
        assert_eq!(
            failing_step_b.map(|(_, _, status)| status.as_str()),
            Some("failed"),
            "batched: err_s2 entity_extract should be 'failed'"
        );
    }

    // ── Task 1.6: ONE route, ONE contract ────────────────────────────────────
    //
    // Tie `enrich_post_ingest_batched` (WAY B) to the canonical `seed_contract`
    // so a silently-dropped entity-link channel fails LOUD here, not at eval
    // time.  We scope to the GRAPH floor only (`require_graph_links = true`)
    // because distillation is NOT run by the batched orchestrator — asserting
    // the pages floor would false-fail by construction.

    /// Task 1.6 — batched path satisfies the graph substrate floor.
    ///
    /// Contract: after `enrich_post_ingest_batched` the `memory_entities` table
    /// must be non-empty (graph channel alive).  Verified via the canonical
    /// `check_seed_contract` function with a `SeedExpectations` profile that
    /// carries `require_graph_links = true` and every other floor OFF (no pages,
    /// no temporal, no dupe check, no classification check) — because the
    /// orchestrator runs Phase 1+2 enrichment only; distillation (pages) runs
    /// separately and asserting it here would false-fail.
    ///
    /// If this test reports `graph_links == 0`, that is a REAL finding: the
    /// batched orchestrator stopped writing `memory_entities` links and the
    /// graph channel would ship starved.
    #[tokio::test]
    async fn batched_path_meets_graph_contract() {
        use crate::eval::seed_contract::{check_seed_contract, SeedExpectations};
        use crate::llm_provider::{CannedLlmProvider, LlmProvider};
        use crate::prompts::PromptRegistry;
        use crate::tuning::{DistillationConfig, RefineryConfig};

        // Reuse the 1.5 single-DB fixture pattern: 3 memories with source_agent=None
        // (prevents find_recent_batch windowed-batch branch) whose content yields
        // "Alice" via CannedLlmProvider.
        let ids = ["c16_s1", "c16_s2", "c16_s3"];
        let contents = [
            "Alice leads the Gamma project at headquarters.",
            "Alice reviewed the Gamma proposal last Monday.",
            "Alice approved the Gamma budget for next quarter.",
        ];
        // Only need one DB — reuse make_two_dbs_with_fixtures and discard db_a.
        let (_db_a, db, mems) = make_two_dbs_with_fixtures(&ids, &contents).await;

        let prompts = PromptRegistry::default();
        let key_fragment: String = prompts.extract_knowledge_graph.chars().take(30).collect();
        let kg_json =
            r#"[{"entities":[{"name":"Alice","type":"person"}],"observations":[],"relations":[]}]"#;
        let llm: Arc<dyn LlmProvider> =
            Arc::new(CannedLlmProvider::new("{}").with(key_fragment, kg_json));

        let refinery = RefineryConfig::default();
        let distillation = DistillationConfig::default();

        // Run the batched orchestrator (WAY B).
        enrich_post_ingest_batched(&db, &llm, &prompts, &refinery, &distillation, &mems, 8)
            .await
            .unwrap();

        // ── Canonical contract check (graph floor only) ─────────────────────
        // `complete()` also requires pages and event_dates, which the batched
        // orchestrator does NOT produce (no distill, no date injection).  Build a
        // targeted profile with ONLY the graph floor asserted so the contract
        // tests exactly the channel we care about and cannot false-fail on the
        // deliberately-absent others.
        let expect = SeedExpectations {
            variant: "batched_path_graph_floor".to_string(),
            require_no_dupes: false,
            require_full_classification: false,
            min_cue_coverage: 0.0,
            min_event_date_coverage: 0.0,
            require_graph_links: true, // ← the one floor we assert
            require_event_dates: false,
            require_pages: false, // distill not run → pages absent by design
            expect_fixture_sha256: None,
        };

        let conn = db.conn.lock().await;
        let report = check_seed_contract(&conn, &expect)
            .await
            .expect("check_seed_contract should not error");

        assert!(
            report.graph_links > 0,
            "graph substrate is empty after enrich_post_ingest_batched: \
             memory_entities links = 0. The batched path dropped the entity-link \
             channel — this is a REAL finding, not a test gap.\n\
             Contract report: {:?}",
            report
        );
        assert!(
            report.holds(),
            "seed_contract VIOLATED after enrich_post_ingest_batched:\n{}\n\
             Full report: {:?}",
            report.violations.join("; "),
            report
        );
    }
}
