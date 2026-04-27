// SPDX-License-Identifier: AGPL-3.0-only
//! Shared eval infrastructure: embedder, tokenizer, entity extraction helper.

use crate::db::MemoryDB;
use crate::error::OriginError;
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

/// Run enrichment via Anthropic Batch API: entity extraction + title enrichment.
///
/// Much faster than on-device (~5 min vs ~2 hours for LoCoMo). Better quality
/// (Haiku vs Qwen 4B). Costs ~$1 per benchmark run.
///
/// 1. Collects all memories needing extraction
/// 2. Submits extraction prompts as one Batch API request
/// 3. Parses results, creates entities/relations/observations in DB
/// 4. Marks enrichment steps for concept distillation
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
    use crate::prompts::PromptRegistry;

    let prompts = PromptRegistry::load(&PromptRegistry::override_dir());
    let batch_size = 10;
    let mut total = 0usize;

    // Keep extracting until no unlinked memories remain (or no progress)
    loop {
        let extracted =
            crate::refinery::extract_entities_from_memories(db, Some(llm), &prompts, batch_size)
                .await?;
        if extracted == 0 {
            break;
        }
        total += extracted;
        eprintln!(
            "    [entity_extract] batch: +{} entities (total: {})",
            extracted, total
        );
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
