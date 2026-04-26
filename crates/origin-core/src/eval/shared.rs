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

    Ok(total)
}
