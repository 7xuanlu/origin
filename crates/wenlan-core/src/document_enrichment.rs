// SPDX-License-Identifier: Apache-2.0
//! Canonical document-tier enrichment route (folder / multi-format ingest).
//!
//! [`run_document_enrichment_slice`] is the path the scheduler actually drives:
//! the ambient Document lane (`wenlan-server/src/scheduler/ambient.rs:309` via
//! `run_document_enrichment_slice_tick`, ambient.rs:518-530) calls it once per
//! ambient turn, spending exactly one LLM request and yielding at its durable
//! checkpoint. [`run_document_enrichment`] is the unbudgeted variant retained
//! for tests — it map-folds a whole document in one call sequence with no
//! per-turn checkpoint yield. Both are thin wrappers over the private
//! `run_document_enrichment_with_request_budget`, differing only in its
//! `requests_remaining` argument (`None` for the unbudgeted variant, `Some(1)`
//! for the slice). Sharing that inner function is what keeps seed-vs-production
//! fidelity by construction (Google "Rules of ML", Rule #32: re-use code
//! between training and serving pipelines) — mirroring the skew-discipline of
//! [`crate::ingest::run_canonical_enrichment`] for the memory tier.
//!
//! Pipeline for one file:
//! 1. **Parse** via [`crate::sources::directory::file_to_documents`], wrapped in
//!    `tokio::task::spawn_blocking` (PDF text extraction is CPU-heavy and must
//!    never run inline on an async request path).
//! 2. **Upsert** the merged file body through [`crate::db::MemoryDB::upsert_documents`]
//!    so EVERY chunk is embedded + provenance-stamped and the document is
//!    immediately searchable — before any LLM digest runs (§8-q2).
//! 3. **Map-fold**: one analysis LLM call per chunk, folding into a rolling
//!    digest capped at ~15K chars, with the fold spanning multiple ambient
//!    turns under the slice's one-request-per-turn budget. Each chunk's
//!    analysis is persisted as that chunk's summary and the queue is
//!    checkpointed AFTER every chunk ([`crate::db::MemoryDB::checkpoint_chunk`]),
//!    so a restart (or the next ambient turn) resumes mid-document without
//!    re-sending already-analyzed chunks to the LLM.
//! 4. **Outputs**: a summary + best-effort entities + exactly ONE
//!    `creation_kind='source'` page citing its own chunks (chunk-granular).
//!
//! Robustness: log-and-degrade at every step (an LLM/DB error warns and the step
//! degrades, never panics). On an LLM failure the route produces a DETERMINISTIC
//! stub SOURCE page (so the document is ALWAYS represented) and signals pause
//! (`mark_paused` with backoff) instead of burning retries in-loop — the
//! scheduler re-claims after the backoff and resumes from the checkpoint,
//! upgrading the stub to the real digest. A file that parses to nothing is
//! terminal (durable sync receipt + queue completion, no page).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::db::{DocEnrichmentQueueEntry, MemoryDB, MemoryDetail};
use crate::error::WenlanError;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::post_write::{page_write, PageWrite};
use crate::prompts::PromptRegistry;
use crate::sources::directory::{
    document_source_id, file_to_documents, provenance_path, FileOutcome,
};
#[cfg(test)]
use wenlan_types::requests::CreateConceptRequest;

/// Rolling-digest character cap (~15K).
const DIGEST_CHAR_CAP: usize = 15_000;

/// System prompt for the per-chunk map-fold analysis call. The document tier has
/// no registry prompt (the registry is memory-tier); this is a fixed, minimal
/// framing that folds each section into the running digest.
const ANALYSIS_SYSTEM_PROMPT: &str = "You are building a running digest of a document, one section at a time. \
Given the digest so far and the next section, reply with 1-3 concise sentences that capture the new section's key facts. \
Do not repeat the earlier digest; summarize only the new section.";

/// Result of enriching one queued document.
#[derive(Debug, Clone)]
pub struct DocumentEnrichmentOutcome {
    /// The canonical document `source_id` (`{source_id}::{provenance}`) under
    /// which the file's chunks live in `memories`.
    pub doc_source_id: String,
    /// The SOURCE page id. Set whenever a page (stub or digest) was written;
    /// empty when the file produced no ingestable content.
    pub page_id: String,
    /// The chunk ids the page cites (chunk-granular provenance).
    pub chunk_ids: Vec<String>,
    /// The folded map-fold digest (success), or the deterministic stub body
    /// (LLM failure / no LLM).
    pub summary: String,
    /// Best-effort entities extracted from the digest. Empty on the stub path or
    /// when extraction degrades.
    pub entities: Vec<String>,
    /// True when the map-fold ran to completion and the row was marked done.
    pub completed: bool,
    /// True when enrichment paused for retry (LLM failure or a transient DB/IO
    /// error), leaving the checkpoint intact for a later resume.
    pub paused: bool,
}

impl DocumentEnrichmentOutcome {
    fn terminal_no_page(doc_source_id: String) -> Self {
        Self {
            doc_source_id,
            page_id: String::new(),
            chunk_ids: Vec::new(),
            summary: String::new(),
            entities: Vec::new(),
            completed: false,
            paused: false,
        }
    }

    fn paused_no_page(doc_source_id: String) -> Self {
        Self {
            doc_source_id,
            page_id: String::new(),
            chunk_ids: Vec::new(),
            summary: String::new(),
            entities: Vec::new(),
            completed: false,
            paused: true,
        }
    }
}

/// Enrich a single queued document end-to-end. See the module docs for the full
/// contract. Never propagates an error: every failure mode maps to a returned
/// [`DocumentEnrichmentOutcome`] plus a queue transition
/// (`done` / `paused` / `waiting_for_provider`).
pub async fn run_document_enrichment(
    db: &MemoryDB,
    entry: &DocEnrichmentQueueEntry,
    knowledge_path: Option<&Path>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
) -> DocumentEnrichmentOutcome {
    run_document_enrichment_with_request_budget(db, entry, knowledge_path, llm, prompts, None).await
}

/// Advance a queued document by at most one LLM request, then yield its durable
/// checkpoint back to the ambient scheduler. Parsing and initial embedding are
/// still one bounded-by-file-size preparation step; map-fold and entity
/// extraction never share the same ambient turn.
pub async fn run_document_enrichment_slice(
    db: &MemoryDB,
    entry: &DocEnrichmentQueueEntry,
    knowledge_path: Option<&Path>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
) -> DocumentEnrichmentOutcome {
    run_document_enrichment_with_request_budget(db, entry, knowledge_path, llm, prompts, Some(1))
        .await
}

async fn run_document_enrichment_with_request_budget(
    db: &MemoryDB,
    entry: &DocEnrichmentQueueEntry,
    knowledge_path: Option<&Path>,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    mut requests_remaining: Option<usize>,
) -> DocumentEnrichmentOutcome {
    let source_id = entry.source_id.clone();
    let file_path = entry.file_path.clone();

    // Canonical document source_id — recomputed from provenance (a pure path op)
    // so a resumed run finds the file's chunks WITHOUT re-parsing. Shares the
    // ONE key authority (`document_source_id`) with folder-sync deletion so the
    // write-side id and the delete-side id can never drift apart.
    let provenance = provenance_path(Path::new(&file_path), knowledge_path);
    let doc_source_id = document_source_id(&source_id, Path::new(&file_path), knowledge_path);

    let is_fresh = entry.last_completed_chunk < 0;
    let mut prepared_chunks = None;
    let mut title = Path::new(&file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("document")
        .to_string();

    // ── (1)+(2) fresh run only: parse (spawn_blocking) + upsert (embed all chunks) ──
    if is_fresh {
        let parse_source_id = source_id.clone();
        let parse_path = PathBuf::from(&file_path);
        let parse_knowledge = knowledge_path.map(|p| p.to_path_buf());
        let parsed = tokio::task::spawn_blocking(move || {
            file_to_documents(&parse_source_id, &parse_path, parse_knowledge.as_deref())
        })
        .await;

        let docs = match parsed {
            Ok(FileOutcome::Ingested(docs)) => docs,
            Ok(FileOutcome::Skipped(reason)) | Ok(FileOutcome::Error(reason)) => {
                // A file that yields nothing ingestable won't improve on retry.
                // Still record sync_state so the next sync skips it instead of
                // re-parsing it every tick.
                log::warn!("[doc-enrich] {file_path}: not ingestable ({reason}); marking done");
                if !complete_with_sync_receipt(db, entry, false).await {
                    return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
                }
                return DocumentEnrichmentOutcome::terminal_no_page(doc_source_id);
            }
            Err(join_err) => {
                log::warn!("[doc-enrich] {file_path}: parse task failed: {join_err}");
                pause(db, entry, "parse task failed").await;
                return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
            }
        };

        if let Some(first) = docs.first() {
            title = first.title.clone();
        }
        // One file = one document: merge the parsed docs' bodies under the
        // canonical source_id. All chunks of a file share its content_hash.
        let body = docs
            .iter()
            .map(|d| d.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let content_hash = docs.iter().find_map(|d| d.content_hash.clone());
        let Some(parsed_hash) = content_hash.as_deref() else {
            log::warn!("[doc-enrich] {file_path}: parsed document omitted content hash; pausing");
            pause(db, entry, "parsed document omitted content hash").await;
            return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
        };
        if entry.content_hash.as_deref() != Some(parsed_hash) {
            log::info!(
                "[doc-enrich] {file_path}: file hash changed after claim; requeueing current generation"
            );
            if let Err(error) = db
                .enqueue_document(&source_id, &file_path, Some(parsed_hash))
                .await
            {
                log::warn!("[doc-enrich] {file_path}: hash requeue failed: {error}");
                pause(db, entry, "file hash changed and requeue failed").await;
                return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
            }
            return DocumentEnrichmentOutcome {
                page_id: source_page_id(&source_id, &file_path),
                doc_source_id,
                chunk_ids: Vec::new(),
                summary: String::new(),
                entities: Vec::new(),
                completed: false,
                paused: false,
            };
        }
        match db
            .prepared_document_generation(&doc_source_id, parsed_hash)
            .await
        {
            Ok(Some(chunks)) => {
                prepared_chunks = Some(chunks);
            }
            Ok(None) => {}
            Err(error) => {
                log::warn!(
                    "[doc-enrich] {file_path}: prepared generation check failed: {error}; pausing"
                );
                pause(db, entry, "prepared generation check failed").await;
                return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
            }
        }
        let last_modified = docs
            .first()
            .map(|d| d.last_modified)
            .unwrap_or_else(|| chrono::Utc::now().timestamp());
        let mut metadata = std::collections::HashMap::new();
        if let Some(ext) = docs.first().and_then(|d| d.metadata.get("extension")) {
            metadata.insert("extension".to_string(), ext.clone());
        }
        metadata.insert("path".to_string(), provenance.clone());

        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: doc_source_id.clone(),
            title: title.clone(),
            content: body,
            last_modified,
            metadata,
            source_agent: Some("folder".to_string()),
            content_hash,
            ..Default::default()
        };
        if prepared_chunks.is_none() {
            if let Err(e) = db.upsert_documents(vec![doc]).await {
                log::warn!("[doc-enrich] {file_path}: upsert failed: {e}; pausing");
                pause(db, entry, "upsert failed").await;
                return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
            }
        }
    }

    // ── read the stored chunks (ordered by chunk_index) ──
    let chunks = match prepared_chunks {
        Some(chunks) => chunks,
        None => match db.get_memories_by_source_id("memory", &doc_source_id).await {
            Ok(c) => c,
            Err(e) => {
                log::warn!("[doc-enrich] {file_path}: read chunks failed: {e}; pausing");
                pause(db, entry, "read chunks failed").await;
                return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
            }
        },
    };
    let chunk_ids: Vec<String> = chunks.iter().map(|c| c.id.clone()).collect();
    let page_id = source_page_id(&source_id, &file_path);

    if chunks.is_empty() {
        log::warn!("[doc-enrich] {file_path}: no chunks after upsert; marking done");
        if !complete_with_sync_receipt(db, entry, false).await {
            return DocumentEnrichmentOutcome::paused_no_page(doc_source_id);
        }
        return DocumentEnrichmentOutcome::terminal_no_page(doc_source_id);
    }
    if let Some(stored_title) = chunks
        .iter()
        .map(|c| c.title.trim())
        .find(|title| !title.is_empty())
    {
        title = stored_title.to_string();
    }

    // ── (3) map-fold: rebuild the digest from checkpointed summaries, then
    // analyze only the not-yet-completed chunks. ──
    let mut digest = String::new();
    for c in &chunks {
        if (c.chunk_index as i64) <= entry.last_completed_chunk {
            if let Some(s) = c.summary.as_deref() {
                fold_digest(&mut digest, s);
            }
        }
    }

    let start = (entry.last_completed_chunk + 1).max(0);
    let mut llm_failed = false;
    if let Some(llm) = llm {
        for c in chunks.iter().filter(|c| (c.chunk_index as i64) >= start) {
            if requests_remaining == Some(0) {
                return yield_document_slice(db, entry, doc_source_id, page_id, chunk_ids, digest)
                    .await;
            }
            if let Some(remaining) = requests_remaining.as_mut() {
                *remaining = remaining.saturating_sub(1);
            }
            let user_prompt = format!("Digest so far:\n{}\n\nNext section:\n{}", digest, c.content);
            match llm
                .generate(LlmRequest {
                    system_prompt: Some(ANALYSIS_SYSTEM_PROMPT.to_string()),
                    user_prompt,
                    max_tokens: 256,
                    temperature: 0.2,
                    label: Some("doc_analysis".to_string()),
                    timeout_secs: None,
                })
                .await
            {
                Ok(analysis) => {
                    let analysis = analysis.trim().to_string();
                    // The summary and resume point are one durable fact. Commit
                    // them atomically before yielding this bounded slice.
                    if let Err(e) = db
                        .persist_document_chunk_progress_at_hash(
                            &doc_source_id,
                            c.chunk_index as i64,
                            &analysis,
                            &source_id,
                            &file_path,
                            entry.content_hash.as_deref(),
                        )
                        .await
                    {
                        log::warn!(
                            "[doc-enrich] {file_path}: durable chunk checkpoint({}) failed: {e}; pausing",
                            c.chunk_index
                        );
                        pause(db, entry, "durable chunk checkpoint failed").await;
                        return DocumentEnrichmentOutcome {
                            doc_source_id,
                            page_id,
                            chunk_ids,
                            summary: digest,
                            entities: Vec::new(),
                            completed: false,
                            paused: true,
                        };
                    }
                    fold_digest(&mut digest, &analysis);
                }
                Err(e) => {
                    // LLM failure: do NOT burn retries in-loop. Fall through to the
                    // deterministic stub page + pause; the checkpoint preserves the
                    // chunks already analyzed for the retry.
                    log::warn!(
                        "[doc-enrich] {file_path}: analysis LLM failed at chunk {}: {e}; pausing",
                        c.chunk_index
                    );
                    llm_failed = true;
                    break;
                }
            }
        }
    }

    if !llm_failed && llm.is_some() && requests_remaining == Some(0) {
        return yield_document_slice(db, entry, doc_source_id, page_id, chunk_ids, digest).await;
    }

    // ── (4) outputs: exactly one SOURCE page (always), summary + entities ──
    if llm_failed || llm.is_none() {
        // Deterministic stub SOURCE page so the document is ALWAYS represented.
        let body = stub_page_body(&title, &chunks);
        if let Err(e) =
            write_document_source_page(db, entry, &page_id, &title, None, &body, &chunk_ids).await
        {
            log::warn!("[doc-enrich] {file_path}: stub source page write failed: {e}");
        }
        if llm_failed {
            pause(db, entry, "analysis LLM failed").await;
            return DocumentEnrichmentOutcome {
                doc_source_id,
                page_id,
                chunk_ids,
                summary: body,
                entities: Vec::new(),
                completed: false,
                paused: true,
            };
        }
        // The deterministic preparation is searchable, but model-derived
        // enrichment remains parked until the user authorizes a provider.
        if !complete_with_sync_receipt(db, entry, true).await {
            return DocumentEnrichmentOutcome {
                doc_source_id,
                page_id,
                chunk_ids,
                summary: body,
                entities: Vec::new(),
                completed: false,
                paused: true,
            };
        }
        return DocumentEnrichmentOutcome {
            doc_source_id,
            page_id,
            chunk_ids,
            summary: body,
            entities: Vec::new(),
            completed: false,
            paused: false,
        };
    }

    // Success: best-effort entity extraction over the digest, then the real page.
    let entities = match llm {
        Some(llm) => {
            if let Some(remaining) = requests_remaining.as_mut() {
                *remaining = remaining.saturating_sub(1);
            }
            let user_prompt: String = digest.chars().take(4000).collect();
            match llm
                .generate(LlmRequest {
                    system_prompt: Some(prompts.extract_knowledge_graph.clone()),
                    user_prompt,
                    max_tokens: 512,
                    temperature: 0.1,
                    label: Some("doc_entities".to_string()),
                    timeout_secs: None,
                })
                .await
            {
                Ok(out) => parse_entities(&out),
                Err(e) => {
                    log::warn!("[doc-enrich] {file_path}: entity extraction failed: {e}");
                    Vec::new()
                }
            }
        }
        None => Vec::new(),
    };

    let summary_line: String = digest.chars().take(280).collect();
    if let Err(e) = write_document_source_page(
        db,
        entry,
        &page_id,
        &title,
        Some(&summary_line),
        &digest,
        &chunk_ids,
    )
    .await
    {
        log::warn!("[doc-enrich] {file_path}: source page write failed: {e}; pausing");
        pause(db, entry, "source page write failed").await;
        return DocumentEnrichmentOutcome {
            doc_source_id,
            page_id,
            chunk_ids,
            summary: digest,
            entities,
            completed: false,
            paused: true,
        };
    }

    if !complete_with_sync_receipt(db, entry, false).await {
        return DocumentEnrichmentOutcome {
            doc_source_id,
            page_id,
            chunk_ids,
            summary: digest,
            entities,
            completed: false,
            paused: true,
        };
    }
    DocumentEnrichmentOutcome {
        doc_source_id,
        page_id,
        chunk_ids,
        summary: digest,
        entities,
        completed: true,
        paused: false,
    }
}

async fn yield_document_slice(
    db: &MemoryDB,
    entry: &DocEnrichmentQueueEntry,
    doc_source_id: String,
    page_id: String,
    chunk_ids: Vec<String>,
    summary: String,
) -> DocumentEnrichmentOutcome {
    match db
        .yield_document_enrichment_at_hash(
            &entry.source_id,
            &entry.file_path,
            entry.content_hash.as_deref(),
        )
        .await
    {
        Ok(true) => {}
        Ok(false) => {
            log::info!(
                "[doc-enrich] {}: stale worker did not yield a newer generation",
                entry.file_path
            );
            return DocumentEnrichmentOutcome {
                doc_source_id,
                page_id,
                chunk_ids,
                summary,
                entities: Vec::new(),
                completed: false,
                paused: false,
            };
        }
        Err(error) => {
            log::warn!(
                "[doc-enrich] {}: failed to yield ambient slice: {error}",
                entry.file_path
            );
            pause(db, entry, "ambient slice yield failed").await;
            return DocumentEnrichmentOutcome {
                doc_source_id,
                page_id,
                chunk_ids,
                summary,
                entities: Vec::new(),
                completed: false,
                paused: true,
            };
        }
    }
    DocumentEnrichmentOutcome {
        doc_source_id,
        page_id,
        chunk_ids,
        summary,
        entities: Vec::new(),
        completed: false,
        paused: false,
    }
}

/// Atomically record the file's tracked `source_sync_state` and transition the
/// queue row to its next durable state. A failed receipt pauses the row for retry.
/// The scan-side mtime+hash skip and the deletion diff both read this table:
/// without the write, a directory file is re-enqueued on every sync and its
/// chunks are never reaped after deletion.
///
/// Self-healing against mid-enrichment modification: the file is re-stat'd and
/// re-hashed NOW. If the hash still matches the enqueue-time hash, the row gets
/// the current mtime (normal). If it no longer matches — or the file vanished —
/// the row is written with `mtime_ns = 0` and the ENQUEUE-time hash, so the
/// next sync cannot mtime-skip it: it re-hashes, sees the drift, and re-enriches
/// (or, for a vanished file, the deletion diff reaps the chunks just written).
async fn complete_with_sync_receipt(
    db: &MemoryDB,
    entry: &DocEnrichmentQueueEntry,
    wait_for_provider: bool,
) -> bool {
    let path = PathBuf::from(&entry.file_path);
    let stat = tokio::task::spawn_blocking(move || {
        let meta = std::fs::metadata(&path).ok()?;
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        let bytes = std::fs::read(&path).ok()?;
        Some((mtime_ns, crate::sources::directory::sha256_hex(&bytes)))
    })
    .await
    .ok()
    .flatten();

    let enqueue_hash = entry.content_hash.as_deref();
    let (mtime_ns, hash) = match (&stat, enqueue_hash) {
        (Some((m, now)), Some(eh)) if now == eh => (*m, now.clone()),
        // No pinned enqueue hash: trust the file as it stands now.
        (Some((m, now)), None) => (*m, now.clone()),
        // Changed mid-enrichment: force the next sync to re-check content.
        (Some(_), Some(eh)) => (0, eh.to_string()),
        // Vanished mid-enrichment: still track it so the deletion diff reaps
        // the chunks this run just wrote.
        (None, eh) => (0, eh.unwrap_or_default().to_string()),
    };
    let receipt = if wait_for_provider {
        db.defer_document_enrichment_until_provider(
            &entry.source_id,
            &entry.file_path,
            mtime_ns,
            &hash,
        )
        .await
    } else {
        db.complete_document_enrichment(&entry.source_id, &entry.file_path, mtime_ns, &hash)
            .await
    };
    if let Err(e) = receipt {
        log::warn!(
            "[doc-enrich] {}: completion receipt failed: {e}; pausing",
            entry.file_path
        );
        pause(db, entry, "source sync receipt failed").await;
        return false;
    }
    true
}

/// Pause a document for retry with an exponential-ish backoff. `mark_paused`
/// bumps the attempt counter; the checkpoint is left intact.
async fn pause(db: &MemoryDB, entry: &DocEnrichmentQueueEntry, reason: &str) {
    let retry_at = chrono::Utc::now().timestamp() + retry_backoff_secs(entry.attempt_count);
    match db
        .mark_paused_at_hash(
            &entry.source_id,
            &entry.file_path,
            entry.content_hash.as_deref(),
            reason,
            Some(retry_at),
        )
        .await
    {
        Ok(true) => {}
        Ok(false) => log::info!(
            "[doc-enrich] {}: stale worker did not pause a newer generation",
            entry.file_path
        ),
        Err(e) => log::warn!("[doc-enrich] {}: mark_paused failed: {e}", entry.file_path),
    }
}

/// Return the exact claimed generation to the retry queue after an unwind at
/// an outer task boundary. The content-hash CAS prevents a stale worker from
/// pausing a newer file generation, while preserving the durable checkpoint.
pub async fn pause_document_enrichment_after_panic(db: &MemoryDB, entry: &DocEnrichmentQueueEntry) {
    pause(db, entry, "document enrichment panicked").await;
}

async fn write_document_source_page(
    db: &MemoryDB,
    entry: &DocEnrichmentQueueEntry,
    page_id: &str,
    title: &str,
    summary: Option<&str>,
    content: &str,
    chunk_ids: &[String],
) -> Result<(), WenlanError> {
    let expected_page_version = db.get_page(page_id).await?.map(|page| page.version);
    page_write(
        db,
        PageWrite::DocumentSource {
            page_id,
            title,
            summary,
            content,
            source_memory_ids: chunk_ids,
            queue_source_id: &entry.source_id,
            file_path: &entry.file_path,
            expected_content_hash: entry.content_hash.as_deref(),
            expected_page_version,
            agent: "doc-enrich",
        },
    )
    .await
    .map(|_| ())
}

/// Write (idempotently) the single `creation_kind='source'` page for a document,
/// citing its chunks. Existing machine-owned source Pages update in place so a
/// failed retry cannot delete the last valid Page or its provenance.
#[cfg(test)]
async fn write_source_page(
    db: &MemoryDB,
    page_id: &str,
    title: &str,
    summary: Option<&str>,
    content: &str,
    chunk_ids: &[String],
) -> Result<(), WenlanError> {
    if db.get_page(page_id).await?.is_some() {
        return page_write(
            db,
            PageWrite::ReplaceSource {
                page_id,
                title,
                summary,
                content,
                source_memory_ids: chunk_ids,
                agent: "doc-enrich",
            },
        )
        .await
        .map(|_| ());
    }
    let req = CreateConceptRequest {
        title: title.to_string(),
        content: content.to_string(),
        summary: summary.map(str::to_string),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: chunk_ids.to_vec(),
        creation_kind: Some("source".to_string()),
        workspace: None,
    };
    page_write(
        db,
        PageWrite::Create {
            page_id: Some(page_id),
            req,
            agent: "doc-enrich",
            knowledge_path: None,
            page_min_cluster_size: 1,
            page_match_threshold: 0.0,
            citations_json: None,
        },
    )
    .await
    .map(|_| ())
}

/// Retry backoff (seconds) for a paused document, given the attempt count BEFORE
/// this failure (`mark_paused` increments it). Exponential-ish, capped at 1h.
fn retry_backoff_secs(attempt_count: i64) -> i64 {
    let base: i64 = 60;
    let shift = attempt_count.clamp(0, 6) as u32;
    base.saturating_mul(1i64 << shift).min(3600)
}

/// Deterministic SOURCE page id for a document, stable across retries.
pub fn source_page_id(source_id: &str, file_path: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"source_page::");
    hasher.update(source_id.as_bytes());
    hasher.update(b"::");
    hasher.update(file_path.as_bytes());
    let hex = format!("{:x}", hasher.finalize());
    format!("src_{}", &hex[..16])
}

/// Deterministic stub page body (no LLM) — the file's chunk text, capped. Used
/// so a SOURCE page always exists even when enrichment has not run yet.
fn stub_page_body(title: &str, chunks: &[MemoryDetail]) -> String {
    let mut body = String::new();
    for c in chunks {
        if !body.is_empty() {
            body.push_str("\n\n");
        }
        body.push_str(&c.content);
    }
    let capped: String = body.chars().take(DIGEST_CHAR_CAP).collect();
    format!("Source document: {title}\n\n{capped}")
}

/// Fold a chunk analysis into the rolling digest, capping at [`DIGEST_CHAR_CAP`]
/// characters (UTF-8 safe — never byte-slices mid-char).
fn fold_digest(digest: &mut String, analysis: &str) {
    let trimmed = analysis.trim();
    if trimmed.is_empty() {
        return;
    }
    if !digest.is_empty() {
        digest.push('\n');
    }
    digest.push_str(trimmed);
    if digest.chars().count() > DIGEST_CHAR_CAP {
        *digest = digest.chars().take(DIGEST_CHAR_CAP).collect();
    }
}

/// Leniently pull entity names out of an LLM response. Accepts a plain array of
/// names (`["rust","tdd"]`), the `extract_knowledge_graph` shape
/// (`[{"entities":[{"name":"..."}]}]`), or a single `{"entities":[...]}` object.
/// Degrades to an empty vec on any parse failure. Names are trimmed + deduped
/// (order-preserving).
fn parse_entities(response: &str) -> Vec<String> {
    let json = crate::engine::extract_json_array(response)
        .or_else(|| crate::engine::extract_json(response).map(|s| s.to_string()));
    let Some(json) = json else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&json) else {
        return Vec::new();
    };
    let mut names = Vec::new();
    collect_entity_names(&value, &mut names);
    let mut seen = std::collections::HashSet::new();
    names.retain(|n| !n.is_empty() && seen.insert(n.clone()));
    names
}

fn collect_entity_names(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::String(s) => out.push(s.trim().to_string()),
        serde_json::Value::Array(arr) => {
            for v in arr {
                collect_entity_names(v, out);
            }
        }
        serde_json::Value::Object(map) => {
            if let Some(serde_json::Value::String(n)) = map.get("name") {
                out.push(n.trim().to_string());
            }
            if let Some(ents) = map.get("entities") {
                collect_entity_names(ents, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::NoopEmitter;
    use crate::llm_provider::{LlmBackend, LlmError, SequencedMockProvider};
    use crate::read_scope::ReadScope;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("origin_memory.db");
        let db = MemoryDB::new(db_path.as_path(), Arc::new(NoopEmitter))
            .await
            .unwrap();
        (db, dir)
    }

    /// Write a temp file whose prose reliably chunks into several (>= 3) chunks in
    /// both the char-based and token-aware chunker configurations, and enqueue it.
    /// Returns (path, canonical unique marker present in the content).
    fn write_doc(dir: &Path) -> PathBuf {
        let path = dir.join("doc.txt");
        // ~6K chars of distinct sentences; contains the unique token "Wenlanborg"
        // so a search proves the chunks are embedded + retrievable.
        let mut body = String::new();
        body.push_str("Wenlanborg is the code name for the folder ingestion subsystem.\n\n");
        for i in 0..80 {
            body.push_str(&format!(
                "Paragraph {i} describes an aspect of the document ingestion pipeline in careful, \
                 concrete detail so that the fixed-size and token-aware chunkers both split it into \
                 multiple sections rather than a single chunk. It keeps going for a while.\n\n"
            ));
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    /// Markdown fixture whose parsed chunk title differs from the file stem.
    /// This exposes resume-title drift: a resumed run should reuse the stored
    /// parsed title from chunks instead of falling back to `file_stem`.
    fn write_markdown_doc(dir: &Path) -> PathBuf {
        let path = dir.join("resume-title.md");
        let mut body = String::new();
        body.push_str("# Canonical Parsed Heading\n\n");
        body.push_str("Wenlanborg is the code name for the folder ingestion subsystem.\n\n");
        for i in 0..80 {
            body.push_str(&format!(
                "Paragraph {i} describes an aspect of the document ingestion pipeline in careful, \
                 concrete detail so that the markdown chunker splits this note into multiple \
                 sections rather than a single chunk. It keeps going for a while.\n\n"
            ));
        }
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    fn analysis_responses() -> Vec<String> {
        (0..128)
            .map(|i| format!("SECTION_ANALYSIS_{i:03}"))
            .collect()
    }

    fn file_hash(path: &Path) -> String {
        crate::sources::directory::sha256_hex(&std::fs::read(path).unwrap())
    }

    fn mock(responses: &[String]) -> Arc<dyn LlmProvider> {
        Arc::new(SequencedMockProvider::new(
            responses.iter().map(String::as_str).collect(),
        ))
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MemoryInventoryRow {
        id: String,
        content: String,
        title: String,
        source: String,
        source_id: String,
        chunk_index: i64,
        content_hash: Option<String>,
        space: Option<String>,
        version: i64,
        summary: Option<String>,
    }

    async fn memory_inventory(db: &MemoryDB, source_id: &str) -> Vec<MemoryInventoryRow> {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT id, content, title, source, source_id, chunk_index,
                        content_hash, space, version, summary
                 FROM memories
                 WHERE source = 'memory' AND source_id = ?1
                 ORDER BY chunk_index ASC, id ASC",
                [source_id],
            )
            .await
            .unwrap();
        let mut inventory = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            inventory.push(MemoryInventoryRow {
                id: row.get::<String>(0).unwrap(),
                content: row.get::<String>(1).unwrap(),
                title: row.get::<String>(2).unwrap(),
                source: row.get::<String>(3).unwrap(),
                source_id: row.get::<String>(4).unwrap(),
                chunk_index: row.get::<i64>(5).unwrap(),
                content_hash: row.get::<Option<String>>(6).unwrap(),
                space: row.get::<Option<String>>(7).unwrap(),
                version: row.get::<i64>(8).unwrap(),
                summary: row.get::<Option<String>>(9).unwrap(),
            });
        }
        inventory
    }

    #[derive(Debug, Clone, Copy)]
    enum PreparedGenerationCorruption {
        BlobContentHash,
        BlobContent,
        BlobSummary,
        NullVersion,
        MixedVersion,
        ZeroVersion,
        DuplicateChunkIndex,
        GappedChunkIndex,
        NonzeroStartChunkIndex,
    }

    impl PreparedGenerationCorruption {
        fn expected_replacement_version(self) -> i64 {
            match self {
                Self::MixedVersion => 3,
                Self::ZeroVersion => 1,
                _ => 2,
            }
        }
    }

    async fn assert_corrupt_prepared_generation_is_replaced(
        corruption: PreparedGenerationCorruption,
    ) {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let first_entry = db.claim_next_pending().await.unwrap().expect("first claim");
        let llm: Arc<dyn LlmProvider> = Arc::new(FailingProvider);
        let first = run_document_enrichment(
            &db,
            &first_entry,
            None,
            Some(&llm),
            &PromptRegistry::default(),
        )
        .await;
        assert!(
            first.paused,
            "{corruption:?}: first attempt prepares chunks"
        );
        let mut expected_inventory = memory_inventory(&db, &first.doc_source_id).await;
        assert!(
            expected_inventory.len() >= 3,
            "{corruption:?}: fixture is multi-chunk"
        );
        let first_id = expected_inventory[0].id.clone();
        let second_id = expected_inventory[1].id.clone();
        let last_id = expected_inventory.last().unwrap().id.clone();

        {
            let conn = db.test_primary_session().await;
            match corruption {
                PreparedGenerationCorruption::BlobContentHash => {
                    conn.execute(
                        "UPDATE memories SET content_hash = x'80' WHERE id = ?1",
                        [first_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::BlobContent => {
                    conn.execute(
                        "UPDATE memories SET content = x'80' WHERE id = ?1",
                        [first_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::BlobSummary => {
                    conn.execute(
                        "UPDATE memories SET summary = x'80' WHERE id = ?1",
                        [first_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::NullVersion => {
                    conn.execute(
                        "UPDATE memories SET version = NULL WHERE id = ?1",
                        [first_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::MixedVersion => {
                    conn.execute(
                        "UPDATE memories SET version = 2 WHERE id = ?1",
                        [first_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::ZeroVersion => {
                    conn.execute(
                        "UPDATE memories SET version = 0
                         WHERE source = 'memory' AND source_id = ?1",
                        [first.doc_source_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::DuplicateChunkIndex => {
                    conn.execute(
                        "UPDATE memories SET chunk_index = 0 WHERE id = ?1",
                        [second_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::GappedChunkIndex => {
                    conn.execute(
                        "UPDATE memories SET chunk_index = ?2 WHERE id = ?1",
                        libsql::params![last_id.as_str(), expected_inventory.len() as i64],
                    )
                    .await
                    .unwrap();
                }
                PreparedGenerationCorruption::NonzeroStartChunkIndex => {
                    conn.execute(
                        "UPDATE memories SET chunk_index = chunk_index + 1
                         WHERE source = 'memory' AND source_id = ?1",
                        [first.doc_source_id.as_str()],
                    )
                    .await
                    .unwrap();
                }
            }
            conn.execute(
                "UPDATE document_enrichment_queue
                 SET next_retry_at = ?3
                 WHERE source_id = ?1 AND file_path = ?2",
                libsql::params![
                    "folder-notes",
                    file_path.as_str(),
                    chrono::Utc::now().timestamp() - 1
                ],
            )
            .await
            .unwrap();
        }

        let retry_entry = db
            .claim_next_pending()
            .await
            .unwrap()
            .expect("same-hash retry claim");
        let retry = run_document_enrichment(
            &db,
            &retry_entry,
            None,
            Some(&llm),
            &PromptRegistry::default(),
        )
        .await;
        assert!(
            retry.paused,
            "{corruption:?}: replacement reaches the expected provider failure"
        );
        let queue = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            queue.error_detail.as_deref(),
            Some("analysis LLM failed"),
            "{corruption:?}: corrupt preparation falls back to replacement before inference"
        );

        let expected_version = corruption.expected_replacement_version();
        for row in &mut expected_inventory {
            row.version = expected_version;
        }
        let repaired_inventory = memory_inventory(&db, &first.doc_source_id).await;
        assert_eq!(
            repaired_inventory, expected_inventory,
            "{corruption:?}: atomic replacement restores the parsed inventory"
        );
        assert!(
            db.prepared_document_generation(&first.doc_source_id, &content_hash)
                .await
                .unwrap()
                .is_some(),
            "{corruption:?}: repaired rows form one exact generation"
        );
    }

    // ── pure-helper unit tests ───────────────────────────────────────────────

    #[test]
    fn parse_entities_plain_array() {
        assert_eq!(
            parse_entities(r#"["rust","tdd","rust"]"#),
            vec!["rust".to_string(), "tdd".to_string()],
            "plain string array → deduped names"
        );
    }

    #[test]
    fn parse_entities_kg_shape() {
        let out = parse_entities(
            r#"prose [{"i":0,"entities":[{"name":"Alice Chen","type":"person"},{"name":"rust","type":"technology"}]}] trailing"#,
        );
        assert_eq!(out, vec!["Alice Chen".to_string(), "rust".to_string()]);
    }

    #[test]
    fn parse_entities_object_shape() {
        let out = parse_entities(r#"{"entities":[{"name":"origin"}]}"#);
        assert_eq!(out, vec!["origin".to_string()]);
    }

    #[test]
    fn parse_entities_garbage_degrades_to_empty() {
        assert!(parse_entities("no json here").is_empty());
        assert!(parse_entities("").is_empty());
    }

    #[test]
    fn fold_digest_caps_at_15k_utf8_safe() {
        let mut d = String::new();
        // Multibyte char to prove the cap never byte-slices mid-char.
        let big = "é".repeat(20_000);
        fold_digest(&mut d, &big);
        assert_eq!(d.chars().count(), DIGEST_CHAR_CAP);
        // Round-trips as valid UTF-8 (would panic on a bad boundary).
        assert!(d.chars().all(|c| c == 'é'));
    }

    // ── failing-provider double: always errors ───────────────────────────────

    struct FailingProvider;
    #[async_trait::async_trait]
    impl LlmProvider for FailingProvider {
        async fn generate(&self, _req: LlmRequest) -> Result<String, LlmError> {
            Err(LlmError::InferenceFailed("boom".into()))
        }
        fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "failing"
        }
        fn backend(&self) -> LlmBackend {
            LlmBackend::OnDevice
        }
    }

    // ── hang-after double: serves N responses, then hangs forever (crash sim) ─

    struct HangAfterProvider {
        responses: Vec<String>,
        hang_after: usize,
        calls: AtomicUsize,
    }
    #[async_trait::async_trait]
    impl LlmProvider for HangAfterProvider {
        async fn generate(&self, _req: LlmRequest) -> Result<String, LlmError> {
            let i = self.calls.fetch_add(1, Ordering::SeqCst);
            if i >= self.hang_after {
                // Never resolves — the caller's future is dropped by a timeout,
                // simulating a process kill after `hang_after` chunks.
                std::future::pending::<()>().await;
            }
            Ok(self.responses[i.min(self.responses.len() - 1)].clone())
        }
        fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "hang-after"
        }
        fn backend(&self) -> LlmBackend {
            LlmBackend::OnDevice
        }
    }

    // ── integration: full happy-path enrichment ──────────────────────────────

    #[tokio::test]
    async fn enriches_multichunk_doc_end_to_end() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");
        assert_eq!(entry.last_completed_chunk, -1, "fresh claim");

        let responses = analysis_responses();
        let llm = mock(&responses);
        let prompts = PromptRegistry::default();

        let outcome = run_document_enrichment(&db, &entry, None, Some(&llm), &prompts).await;

        assert!(outcome.completed, "map-fold ran to completion");
        assert!(!outcome.paused);

        // All chunks embedded + stored.
        let chunks = db
            .get_memories_by_source_id("memory", &outcome.doc_source_id)
            .await
            .unwrap();
        let n = chunks.len();
        assert!(n >= 3, "doc should chunk into >= 3 chunks, got {n}");
        assert_eq!(outcome.chunk_ids.len(), n);

        // Searchable: the unique document token retrieves this document's chunks.
        let results = db
            .search_memory(
                "Wenlanborg",
                30,
                None,
                &ReadScope::Global,
                None,
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert!(
            results.iter().any(|r| r.source_id == outcome.doc_source_id),
            "document chunks must be searchable after upsert"
        );

        // Exactly ONE creation_kind='source' page, citing its chunks (chunk-granular).
        let page = db.get_page(&outcome.page_id).await.unwrap().expect("page");
        assert_eq!(page.creation_kind, "source");
        assert_eq!(
            page.source_memory_ids.len(),
            n,
            "page cites every chunk (chunk-granular provenance)"
        );
        assert_eq!(count_source_pages(&db).await, 1, "exactly one SOURCE page");

        // Digest folded: the page body carries multiple chunk analyses.
        assert!(page.content.contains("SECTION_ANALYSIS_000"));
        assert!(page
            .content
            .contains(&format!("SECTION_ANALYSIS_{:03}", n - 1)));

        // Queue marked done.
        let q = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(q.status, "done");
    }

    #[tokio::test]
    async fn ambient_slice_yields_after_exactly_one_llm_request() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");

        let responses = analysis_responses();
        let provider = Arc::new(SequencedMockProvider::new(
            responses.iter().map(String::as_str).collect(),
        ));
        let llm: Arc<dyn LlmProvider> = provider.clone();
        let prompts = PromptRegistry::default();

        let outcome = run_document_enrichment_slice(&db, &entry, None, Some(&llm), &prompts).await;

        assert_eq!(provider.call_count(), 1, "one slice is one LLM request");
        assert!(!outcome.completed, "more chunks remain for later slices");
        assert!(!outcome.paused, "budget yield is not a provider failure");

        let queued = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .expect("queue row");
        assert_eq!(queued.status, "pending", "the next poll can reclaim it");
        assert_eq!(queued.last_completed_chunk, 0, "the first chunk is durable");
        assert_eq!(queued.attempt_count, 0, "yield does not burn a retry");
    }

    #[tokio::test]
    async fn changed_file_requeues_new_hash_before_any_inference() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let queued_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&queued_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");
        std::fs::write(
            &path,
            "The file changed after discovery and must be requeued before inference. ".repeat(80),
        )
        .unwrap();
        let new_hash = file_hash(&path);
        assert_ne!(new_hash, queued_hash);
        let provider = Arc::new(SequencedMockProvider::new(vec!["must not be called"]));
        let llm: Arc<dyn LlmProvider> = provider.clone();

        let outcome = run_document_enrichment_slice(
            &db,
            &entry,
            None,
            Some(&llm),
            &PromptRegistry::default(),
        )
        .await;

        assert_eq!(provider.call_count(), 0);
        assert!(!outcome.completed);
        assert!(!outcome.paused);
        let queued = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(queued.status, "pending");
        assert_eq!(queued.content_hash.as_deref(), Some(new_hash.as_str()));
        assert_eq!(queued.last_completed_chunk, -1);
    }

    #[tokio::test]
    async fn changed_file_after_failed_preparation_requeues_before_any_retry_inference() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let queued_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&queued_hash))
            .await
            .unwrap();
        let first_entry = db.claim_next_pending().await.unwrap().expect("first claim");
        let failing_llm: Arc<dyn LlmProvider> = Arc::new(FailingProvider);

        let first = run_document_enrichment(
            &db,
            &first_entry,
            None,
            Some(&failing_llm),
            &PromptRegistry::default(),
        )
        .await;
        assert!(first.paused, "first attempt prepares chunks, then pauses");

        std::fs::write(
            &path,
            "The file changed after failed preparation and must be requeued before inference. "
                .repeat(80),
        )
        .unwrap();
        let new_hash = file_hash(&path);
        assert_ne!(new_hash, queued_hash);
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE document_enrichment_queue
                 SET next_retry_at = ?3
                 WHERE source_id = ?1 AND file_path = ?2",
                libsql::params![
                    "folder-notes",
                    file_path.as_str(),
                    chrono::Utc::now().timestamp() - 1
                ],
            )
            .await
            .unwrap();
        }
        let retry_entry = db.claim_next_pending().await.unwrap().expect("retry claim");
        assert_eq!(
            retry_entry.content_hash.as_deref(),
            Some(queued_hash.as_str())
        );
        let provider = Arc::new(SequencedMockProvider::new(vec!["must not be called"]));
        let retry_llm: Arc<dyn LlmProvider> = provider.clone();

        let retry = run_document_enrichment_slice(
            &db,
            &retry_entry,
            None,
            Some(&retry_llm),
            &PromptRegistry::default(),
        )
        .await;

        assert_eq!(provider.call_count(), 0);
        assert!(!retry.completed);
        assert!(!retry.paused);
        let queued = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(queued.status, "pending");
        assert_eq!(queued.content_hash.as_deref(), Some(new_hash.as_str()));
        assert_eq!(queued.last_completed_chunk, -1);
    }

    #[tokio::test]
    async fn changed_hash_replacement_preserves_assigned_space_and_invalidates_projection() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let source_id = format!("folder-notes::{file_path}");
        let initial_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&initial_hash))
            .await
            .unwrap();
        let initial_entry = db.claim_next_pending().await.unwrap().expect("claim v1");
        run_document_enrichment(&db, &initial_entry, None, None, &PromptRegistry::default()).await;

        db.update_memory_space(&source_id, "m4-live").await.unwrap();
        let from = db
            .create_entity("M4 source", "concept", Some("m4-live"))
            .await
            .unwrap();
        let to = db
            .create_entity("M4 target", "concept", Some("m4-live"))
            .await
            .unwrap();
        db.create_relation(
            &from,
            &to,
            "related_to",
            Some("document_enrichment"),
            Some(0.9),
            None,
            Some(&source_id),
        )
        .await
        .unwrap();
        let edge_id = crate::provenance::compute_edge_id(
            "relates",
            "entity",
            &from,
            "entity",
            &to,
            "related_to",
        );
        assert_eq!(
            db.edge_snapshot_for_test(&edge_id).await.unwrap()["valid_until"],
            serde_json::Value::Null,
            "fixture must begin with an active source-owned edge"
        );

        std::fs::write(
            &path,
            "The third semantic generation replaces the folder document body. ".repeat(80),
        )
        .unwrap();
        let replacement_hash = file_hash(&path);
        assert_ne!(replacement_hash, initial_hash);
        db.enqueue_document("folder-notes", &file_path, Some(&replacement_hash))
            .await
            .unwrap();
        let replacement_entry = db.claim_next_pending().await.unwrap().expect("claim v3");
        run_document_enrichment(
            &db,
            &replacement_entry,
            None,
            None,
            &PromptRegistry::default(),
        )
        .await;

        let stored = db
            .get_memories_by_source_id("memory", &source_id)
            .await
            .unwrap();
        assert!(!stored.is_empty());
        let (stored_spaces, stored_versions) = {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query(
                    "SELECT space, version FROM memories
                     WHERE source = 'memory' AND source_id = ?1
                     ORDER BY chunk_index",
                    libsql::params![source_id.as_str()],
                )
                .await
                .unwrap();
            let mut spaces = Vec::new();
            let mut versions = Vec::new();
            while let Some(row) = rows.next().await.unwrap() {
                spaces.push(row.get::<String>(0).unwrap());
                versions.push(row.get::<i64>(1).unwrap());
            }
            (spaces, versions)
        };
        assert!(
            stored_spaces.iter().all(|space| space == "m4-live"),
            "replacement without an explicit space must preserve the assigned space"
        );
        assert!(
            stored_versions.iter().all(|version| *version == 3),
            "v1 ingest + v2 assignment + replacement must produce v3"
        );
        assert!(
            db.list_relations_between(&from, &to)
                .await
                .unwrap()
                .is_empty(),
            "semantic replacement must delete the old source-owned relation"
        );
        assert!(
            db.edge_snapshot_for_test(&edge_id).await.unwrap()["valid_until"].is_number(),
            "semantic replacement must soft-invalidate the old source-owned edge"
        );
    }

    // ── integration: kill after 2 chunks (drop), then resume from checkpoint ──

    #[tokio::test]
    async fn resumes_from_checkpoint_without_reanalyzing() {
        let (db, dir) = test_db().await;
        let path = write_markdown_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");

        let run1_responses = analysis_responses();
        let provider = Arc::new(HangAfterProvider {
            responses: run1_responses.clone(),
            hang_after: 2, // serve chunks 0 and 1, hang on chunk 2 (index 2)
            calls: AtomicUsize::new(0),
        });
        let hang: Arc<dyn LlmProvider> = provider.clone();
        let prompts = PromptRegistry::default();

        // Drive the future until chunk 2's LLM call has STARTED (calls == 3 — the
        // loop awaits checkpoint_chunk(1) before generate(chunk 2), so this
        // guarantees chunks 0+1 are checkpointed), then DROP it (simulated kill).
        // Condition-based, not wall-clock: a fixed delay flakes under
        // parallel-test CPU contention.
        {
            let enrich = run_document_enrichment(&db, &entry, None, Some(&hang), &prompts);
            tokio::pin!(enrich);
            let reached_hang = tokio::time::timeout(std::time::Duration::from_secs(60), async {
                tokio::select! {
                    _ = &mut enrich => false,
                    _ = async {
                        while provider.calls.load(Ordering::SeqCst) < 3 {
                            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                        }
                    } => true,
                }
            })
            .await
            .expect("enrichment should reach chunk 2 within 60s");
            assert!(reached_hang, "run should hang on chunk 2 and be dropped");
        }

        // Checkpoint committed chunks 0 and 1 (resume point = 1); row still in_progress.
        let mid = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(mid.last_completed_chunk, 1, "checkpointed after chunk 1");
        assert_eq!(mid.status, "in_progress");

        // Chunks 0 and 1 carry their analyses (persisted); chunk 2 does not.
        let doc_source_id = format!("folder-notes::{file_path}");
        let mid_chunks = db
            .get_memories_by_source_id("memory", &doc_source_id)
            .await
            .unwrap();
        assert_eq!(
            mid_chunks[0].summary.as_deref(),
            Some("SECTION_ANALYSIS_000")
        );
        assert_eq!(
            mid_chunks[1].summary.as_deref(),
            Some("SECTION_ANALYSIS_001")
        );
        assert_eq!(mid_chunks[2].summary, None, "chunk 2 not yet analyzed");
        let stored_title = mid_chunks[0].title.clone();
        let file_stem_title = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap()
            .to_string();
        assert_ne!(
            stored_title, file_stem_title,
            "fixture must expose parsed-title vs file-stem drift"
        );
        let n = mid_chunks.len();

        // Re-run with a FRESH provider — it must be asked to analyze only chunks
        // 2..n plus the single entity call: (n-2)+1 = n-1 total calls.
        let resume_provider =
            SequencedMockProvider::new(run1_responses.iter().map(String::as_str).collect());
        // We need call_count afterwards, so hold a concrete handle.
        let resume_arc = Arc::new(resume_provider);
        let resume_dyn: Arc<dyn LlmProvider> = resume_arc.clone();
        let entry2 = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();

        let outcome =
            run_document_enrichment(&db, &entry2, None, Some(&resume_dyn), &prompts).await;
        assert!(outcome.completed);

        assert_eq!(
            resume_arc.call_count(),
            n - 1,
            "resume analyzes only chunks 2..n (+1 entity call); chunks 0-1 NOT re-analyzed"
        );

        // Chunks 0-1 still carry their ORIGINAL run-1 analyses (never overwritten).
        let final_chunks = db
            .get_memories_by_source_id("memory", &doc_source_id)
            .await
            .unwrap();
        assert_eq!(
            final_chunks[0].summary.as_deref(),
            Some("SECTION_ANALYSIS_000")
        );
        assert_eq!(
            final_chunks[1].summary.as_deref(),
            Some("SECTION_ANALYSIS_001")
        );

        // One SOURCE page, done.
        assert_eq!(count_source_pages(&db).await, 1);
        let page = db.get_page(&outcome.page_id).await.unwrap().expect("page");
        assert_eq!(
            page.title, stored_title,
            "resumed completion should reuse the parsed title stored on chunks"
        );
        let q = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(q.status, "done");
    }

    // ── integration: LLM failure → deterministic stub page + pause ────────────

    #[tokio::test]
    async fn llm_failure_writes_stub_page_and_pauses() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");

        let llm: Arc<dyn LlmProvider> = Arc::new(FailingProvider);
        let prompts = PromptRegistry::default();

        let outcome = run_document_enrichment(&db, &entry, None, Some(&llm), &prompts).await;

        // Chunks still embedded (upsert ran before the LLM).
        let chunks = db
            .get_memories_by_source_id("memory", &outcome.doc_source_id)
            .await
            .unwrap();
        assert!(chunks.len() >= 3);

        // A deterministic stub SOURCE page exists and cites the chunks.
        assert!(outcome.paused, "LLM failure signals pause");
        assert!(!outcome.completed);
        assert_eq!(count_source_pages(&db).await, 1, "stub SOURCE page exists");
        let page = db
            .get_page(&outcome.page_id)
            .await
            .unwrap()
            .expect("stub page");
        assert_eq!(page.creation_kind, "source");
        assert!(page.content.starts_with("Source document:"));
        assert_eq!(page.source_memory_ids.len(), chunks.len());

        // Pause is signaled on the queue, with a retry scheduled and attempt bumped.
        let q = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(q.status, "paused");
        assert_eq!(q.attempt_count, 1);
        assert!(q.next_retry_at.is_some());
    }

    #[tokio::test]
    async fn same_hash_retry_does_not_reupsert_prepared_generation_after_first_llm_failure() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let first_entry = db.claim_next_pending().await.unwrap().expect("first claim");
        let llm: Arc<dyn LlmProvider> = Arc::new(FailingProvider);

        let first = run_document_enrichment(
            &db,
            &first_entry,
            None,
            Some(&llm),
            &PromptRegistry::default(),
        )
        .await;
        assert!(first.paused, "first attempt prepares chunks, then pauses");
        let inventory_v1 = memory_inventory(&db, &first.doc_source_id).await;
        assert!(!inventory_v1.is_empty(), "prepared generation is durable");
        assert!(
            inventory_v1.iter().all(|row| row.version == 1),
            "fresh prepared generation starts at v1"
        );
        let first_queue = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first_queue.attempt_count, 1);

        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE document_enrichment_queue
                 SET next_retry_at = ?3
                 WHERE source_id = ?1 AND file_path = ?2",
                libsql::params![
                    "folder-notes",
                    file_path.as_str(),
                    chrono::Utc::now().timestamp() - 1
                ],
            )
            .await
            .unwrap();
        }
        let retry_entry = db
            .claim_next_pending()
            .await
            .unwrap()
            .expect("same-hash retry claim");
        assert_eq!(
            retry_entry.content_hash.as_deref(),
            Some(content_hash.as_str())
        );
        let second = run_document_enrichment(
            &db,
            &retry_entry,
            None,
            Some(&llm),
            &PromptRegistry::default(),
        )
        .await;
        assert!(second.paused, "second provider failure pauses again");

        let inventory_after_retry = memory_inventory(&db, &second.doc_source_id).await;
        assert_eq!(
            inventory_after_retry, inventory_v1,
            "same-hash retry must not semantically replace the prepared generation"
        );
        assert!(
            inventory_after_retry.iter().all(|row| row.version == 1),
            "same-hash retry keeps the source-wide generation at v1"
        );
        let retry_queue = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(retry_queue.attempt_count, 2);
    }

    #[tokio::test]
    async fn same_hash_retry_replaces_prepared_generation_with_malformed_hash_storage_type() {
        assert_corrupt_prepared_generation_is_replaced(
            PreparedGenerationCorruption::BlobContentHash,
        )
        .await;
    }

    #[tokio::test]
    async fn same_hash_retry_replaces_prepared_generation_with_malformed_required_text() {
        assert_corrupt_prepared_generation_is_replaced(PreparedGenerationCorruption::BlobContent)
            .await;
    }

    #[tokio::test]
    async fn same_hash_retry_replaces_prepared_generation_with_malformed_optional_text() {
        assert_corrupt_prepared_generation_is_replaced(PreparedGenerationCorruption::BlobSummary)
            .await;
    }

    #[tokio::test]
    async fn same_hash_retry_replaces_prepared_generation_with_invalid_version_or_chunk_shape() {
        for corruption in [
            PreparedGenerationCorruption::NullVersion,
            PreparedGenerationCorruption::MixedVersion,
            PreparedGenerationCorruption::DuplicateChunkIndex,
            PreparedGenerationCorruption::GappedChunkIndex,
            PreparedGenerationCorruption::NonzeroStartChunkIndex,
            PreparedGenerationCorruption::ZeroVersion,
        ] {
            assert_corrupt_prepared_generation_is_replaced(corruption).await;
        }
    }

    #[tokio::test]
    async fn document_source_page_creation_routes_through_pagewrite() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");
        let prompts = PromptRegistry::default();

        let outcome = run_document_enrichment(&db, &entry, None, None, &prompts).await;

        assert!(!outcome.paused);
        assert!(!outcome.page_id.is_empty());
        let page = db
            .get_page(&outcome.page_id)
            .await
            .unwrap()
            .expect("source page");
        assert_eq!(page.creation_kind, "source");
        assert_eq!(page.review_status, "unconfirmed");
        let activity = db.list_agent_activity(20, None, None).await.unwrap();
        assert!(
            activity.iter().any(|entry| {
                entry.action == "page_create"
                    && entry.memory_ids.as_deref() == Some(&outcome.chunk_ids.join(","))
            }),
            "document source-page creation must route through PageWrite and log page_create with chunk provenance, got {activity:?}"
        );
        let evidence = db.get_page_evidence(&outcome.page_id).await.unwrap();
        assert_eq!(evidence.len(), outcome.chunk_ids.len());
        assert!(
            evidence
                .iter()
                .all(|row| row.source_kind == "external_file"),
            "document source-page evidence must preserve folder chunk provenance, got {evidence:?}"
        );
        let queued = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .expect("document queue row");
        assert_eq!(queued.status, "waiting_for_provider");
        assert!(
            db.get_sync_state("folder-notes", &file_path)
                .await
                .unwrap()
                .is_some(),
            "searchable preparation must publish the sync receipt before waiting for model consent"
        );
        assert!(
            db.claim_next_pending_for_provider(false)
                .await
                .unwrap()
                .is_none(),
            "without model consent, prepared documents must stay parked instead of spinning"
        );
        let resumed = db
            .claim_next_pending_for_provider(true)
            .await
            .unwrap()
            .expect("configured model must resume the parked document");
        assert_eq!(resumed.source_id, "folder-notes");
        assert_eq!(resumed.file_path, file_path);
        assert_eq!(resumed.status, "in_progress");
    }

    #[tokio::test]
    async fn document_source_page_write_routes_through_the_hash_guard() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");
        let outcome =
            run_document_enrichment(&db, &entry, None, None, &PromptRegistry::default()).await;
        let before = db.get_page(&outcome.page_id).await.unwrap().unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE document_enrichment_queue
                 SET status='in_progress', content_hash='old-hash'
                 WHERE source_id=?1 AND file_path=?2",
                libsql::params!["folder-notes", file_path.as_str()],
            )
            .await
            .unwrap();
        }
        let stale_entry = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE document_enrichment_queue SET content_hash='new-hash'
                 WHERE source_id=?1 AND file_path=?2",
                libsql::params!["folder-notes", file_path.as_str()],
            )
            .await
            .unwrap();
        }

        write_document_source_page(
            &db,
            &stale_entry,
            &outcome.page_id,
            "Stale title",
            None,
            "Stale folded body",
            &outcome.chunk_ids,
        )
        .await
        .expect_err("the production PageWrite route must reject an old queue hash");

        let after = db.get_page(&outcome.page_id).await.unwrap().unwrap();
        assert_eq!(after.title, before.title);
        assert_eq!(after.content, before.content);
        assert_eq!(after.version, before.version);
    }

    /// G6 edges-parity repair: drives the real `write_document_source_page`
    /// path (the production route, not the `#[cfg(test)]` `write_source_page`
    /// shim) twice over the same folder-doc source page with a growing chunk
    /// set -- the 29->48 shape from the incident, reduced to a deterministic
    /// keep/drop/add triple. Both prior defects fired in this exact call
    /// path: `replace_source_page_inner`'s over-retire of carried-over
    /// evidence (fix b) and the page_sources/backfill kind-derivation
    /// disagreement with the live writer (fix a). Folder-agent sids (source_id
    /// containing "::") resolve via `resolve_page_evidence_source_kind` to
    /// `external_file` -> `dst_kind="external"`, the same rule
    /// `insert_resolved_page_evidence` uses.
    ///
    /// RED control (source mutation, run by hand 2026-08-05): reverting fix
    /// (a) alone (hard-coding `dst_kind="memory"` back into the
    /// `compute_edges_parity_report` page_sources contributor) fails the FIRST
    /// `drift_count == 0` assertion (left=2, right=0), because the sweep then
    /// expects a memory-kind edge no writer ever mints. Reverting fix (b)
    /// alone (restoring `replace_source_page_inner`'s retire loop to run
    /// AFTER `insert_resolved_page_evidence`, with the old memory-kind-only
    /// subtraction) fails the SECOND `drift_count == 0` assertion instead
    /// (left=1, right=0): the over-retire kills the carried-over `keep_sid`
    /// edge after `insert_resolved_page_evidence` re-asserts it, and the
    /// still-intact `page_sources`/`page_evidence` rows then disagree with
    /// the now-retired live edge. Both reverts confirmed to fail by hand,
    /// then the fix restored and this test re-confirmed green.
    #[tokio::test]
    async fn write_document_source_page_replace_keeps_carried_over_retires_dropped() {
        use crate::sources::RawDocument;

        let (db, _dir) = test_db().await;
        let file_path = "fake/g6-red-control-doc.md".to_string();
        db.enqueue_document("folder-notes", &file_path, Some("hash-v1"))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");

        let keep_sid = "doc_g6_red::chunk_keep";
        let drop_sid = "doc_g6_red::chunk_drop";
        let new_sid = "doc_g6_red::chunk_new";
        for sid in [keep_sid, drop_sid, new_sid] {
            let doc = RawDocument {
                source: "memory".to_string(),
                source_id: sid.to_string(),
                title: format!("chunk-{sid}"),
                summary: None,
                content: format!("Folder-doc content for {sid}."),
                url: None,
                last_modified: chrono::Utc::now().timestamp(),
                metadata: std::collections::HashMap::new(),
                memory_type: Some("fact".to_string()),
                space: Some("work".to_string()),
                source_agent: Some("folder".to_string()),
                confidence: Some(0.9),
                confirmed: Some(false),
                ..Default::default()
            };
            db.upsert_documents(vec![doc]).await.unwrap();
        }

        write_document_source_page(
            &db,
            &entry,
            "page_g6_red_control",
            "Source page v1",
            None,
            "content v1",
            &[keep_sid.to_string(), drop_sid.to_string()],
        )
        .await
        .unwrap();
        // G6 Stage 2 PR 2b: parity oracle retired, correctness carried by per-writer regression tests (item 7).

        write_document_source_page(
            &db,
            &entry,
            "page_g6_red_control",
            "Source page v2",
            None,
            "content v2, grown",
            &[keep_sid.to_string(), new_sid.to_string()],
        )
        .await
        .unwrap();
        // G6 Stage 2 PR 2b: parity oracle retired, correctness carried by per-writer regression tests (item 7).

        let conn = db.test_primary_session().await;
        let keep_edge_id = crate::provenance::compute_edge_id(
            "cites",
            "page",
            "page_g6_red_control",
            "external",
            keep_sid,
            keep_sid,
        );
        let drop_edge_id = crate::provenance::compute_edge_id(
            "cites",
            "page",
            "page_g6_red_control",
            "external",
            drop_sid,
            drop_sid,
        );
        let new_edge_id = crate::provenance::compute_edge_id(
            "cites",
            "page",
            "page_g6_red_control",
            "external",
            new_sid,
            new_sid,
        );
        for (edge_id, label, expect_active) in [
            (keep_edge_id.as_str(), "carried-over", true),
            (drop_edge_id.as_str(), "dropped", false),
            (new_edge_id.as_str(), "newly added", true),
        ] {
            let mut rows = conn
                .query(
                    "SELECT valid_until FROM edges WHERE edge_id = ?1",
                    libsql::params![edge_id],
                )
                .await
                .unwrap();
            let row = rows
                .next()
                .await
                .unwrap()
                .unwrap_or_else(|| panic!("{label} edge must exist"));
            let valid_until: Option<u64> = row.get(0).unwrap();
            assert_eq!(
                valid_until.is_none(),
                expect_active,
                "{label} locator's cites edge active-state mismatch"
            );
        }
    }

    #[tokio::test]
    async fn source_page_replacement_failure_preserves_last_valid_page_and_provenance() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");
        let outcome =
            run_document_enrichment(&db, &entry, None, None, &PromptRegistry::default()).await;
        let before = db.get_page(&outcome.page_id).await.unwrap().unwrap();
        let evidence_before = db.get_page_evidence(&outcome.page_id).await.unwrap();
        assert!(!evidence_before.is_empty());

        {
            let conn = db.test_primary_session().await;
            conn.execute_batch(&format!(
                "CREATE TRIGGER abort_source_page_replacement
                 BEFORE UPDATE OF content ON pages
                 WHEN OLD.id = '{}'
                 BEGIN SELECT RAISE(ABORT, 'blocked source page replacement'); END;",
                outcome.page_id.replace('\'', "''")
            ))
            .await
            .unwrap();
        }
        let err = write_source_page(
            &db,
            &outcome.page_id,
            &before.title,
            before.summary.as_deref(),
            "replacement body",
            &outcome.chunk_ids,
        )
        .await
        .expect_err("replacement insert fault must be returned");
        assert!(err.to_string().contains("blocked source page replacement"));
        let after_failure = db.get_page(&outcome.page_id).await.unwrap();
        let evidence_after_failure = db.get_page_evidence(&outcome.page_id).await.unwrap();

        {
            let conn = db.test_primary_session().await;
            conn.execute("DROP TRIGGER abort_source_page_replacement", ())
                .await
                .unwrap();
        }
        write_source_page(
            &db,
            &outcome.page_id,
            &before.title,
            before.summary.as_deref(),
            "replacement body",
            &outcome.chunk_ids,
        )
        .await
        .expect("connection and replacement path must remain reusable");

        assert!(
            after_failure.is_some(),
            "failed replacement must preserve the last valid source Page"
        );
        assert_eq!(after_failure.unwrap().content, before.content);
        assert_eq!(
            evidence_after_failure.len(),
            evidence_before.len(),
            "failed replacement must preserve Page provenance"
        );
    }

    #[tokio::test]
    async fn sync_receipt_failure_does_not_leave_same_hash_queue_terminal() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");
        {
            let conn = db.test_primary_session().await;
            conn.execute_batch(
                "CREATE TRIGGER abort_source_sync_receipt
                 BEFORE INSERT ON source_sync_state
                 BEGIN SELECT RAISE(ABORT, 'blocked source sync receipt'); END;",
            )
            .await
            .unwrap();
        }

        run_document_enrichment(&db, &entry, None, None, &PromptRegistry::default()).await;
        let queue_after_failure = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        let receipt_after_failure = db.get_sync_state("folder-notes", &file_path).await.unwrap();

        {
            let conn = db.test_primary_session().await;
            conn.execute("DROP TRIGGER abort_source_sync_receipt", ())
                .await
                .unwrap();
        }
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .expect("connection and queue must remain reusable");
        let queue_after_reenqueue = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();

        assert!(
            receipt_after_failure.is_some() || queue_after_failure.status != "done",
            "terminal queue state must imply a durable source-sync receipt"
        );
        assert_ne!(
            queue_after_reenqueue.status, "done",
            "same-hash re-enqueue must recover a missing sync receipt"
        );
    }

    // ── self-healing loop: LLM failure pauses with backoff, then re-claims ────

    #[tokio::test]
    async fn llm_failure_pauses_with_backoff_then_reclaims_after_retry_elapses() {
        let (db, dir) = test_db().await;
        let path = write_doc(dir.path());
        let file_path = path.to_string_lossy().to_string();
        let content_hash = file_hash(&path);
        db.enqueue_document("folder-notes", &file_path, Some(&content_hash))
            .await
            .unwrap();
        let entry = db.claim_next_pending().await.unwrap().expect("claim");

        let llm: Arc<dyn LlmProvider> = Arc::new(FailingProvider);
        let prompts = PromptRegistry::default();
        let outcome = run_document_enrichment(&db, &entry, None, Some(&llm), &prompts).await;
        assert!(outcome.paused, "LLM failure pauses (no in-loop retry burn)");

        // Paused: attempt bumped, a FUTURE retry scheduled → not yet claimable.
        let paused = db
            .get_queue_entry("folder-notes", &file_path)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(paused.status, "paused");
        assert_eq!(paused.attempt_count, 1);
        let retry_at = paused.next_retry_at.expect("backoff sets next_retry_at");
        assert!(
            retry_at > chrono::Utc::now().timestamp(),
            "backoff schedules the retry in the future"
        );
        assert!(
            db.claim_next_pending().await.unwrap().is_none(),
            "not claimable before backoff elapses"
        );

        // Advance time past next_retry_at (simulate elapsed backoff) → claimable
        // again, so the scheduler auto-resumes with no daemon restart.
        db.mark_paused(
            "folder-notes",
            &file_path,
            "analysis LLM failed",
            Some(chrono::Utc::now().timestamp() - 1),
        )
        .await
        .unwrap();
        let reclaimed = db
            .claim_next_pending()
            .await
            .unwrap()
            .expect("claimable after backoff elapses");
        assert_eq!(reclaimed.file_path, file_path);
        assert_eq!(
            reclaimed.attempt_count, 2,
            "attempts keep incrementing across retries"
        );
    }

    #[tokio::test]
    async fn exhausted_document_is_not_claimed_while_fresh_pending_claims() {
        let (db, _dir) = test_db().await;
        db.enqueue_document("folder", "/poison.md", Some("poison-hash"))
            .await
            .unwrap();
        for _ in 0..MemoryDB::DOC_ENRICHMENT_MAX_ATTEMPTS {
            db.mark_paused(
                "folder",
                "/poison.md",
                "poison document",
                Some(chrono::Utc::now().timestamp() - 1),
            )
            .await
            .unwrap();
        }
        db.enqueue_document("folder", "/fresh.md", Some("fresh-hash"))
            .await
            .unwrap();

        let claimed = db
            .claim_next_pending()
            .await
            .unwrap()
            .expect("fresh pending document remains claimable");
        assert_eq!(claimed.file_path, "/fresh.md");
        assert!(
            db.claim_next_pending().await.unwrap().is_none(),
            "the capped poison document is not claimable"
        );

        let poison = db
            .get_queue_entry("folder", "/poison.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            poison.attempt_count,
            MemoryDB::DOC_ENRICHMENT_MAX_ATTEMPTS,
            "poison row reaches the retry cap"
        );
        let status = db.document_enrichment_queue_status().await.unwrap();
        assert_eq!(status.exhausted, 1);
        // The exhausted poison row is not "waiting for another retry" (no
        // retryable paused row exists), but it must still surface as a
        // paused reason rather than reading as ordinary `Active` pending
        // work forever.
        let reason = status
            .paused_reason
            .expect("an exhausted document surfaces as paused, not silently active");
        assert!(reason.contains("exhausted"), "reason: {reason}");
    }

    // ── queue observability: status summary reflects pending + paused ─────────

    #[tokio::test]
    async fn queue_status_summarizes_pending_and_paused() {
        let (db, _dir) = test_db().await;

        // Empty queue → nothing pending, no pause.
        let empty = db.document_enrichment_queue_status().await.unwrap();
        assert_eq!(empty.pending, 0);
        assert!(empty.paused_reason.is_none());
        assert!(empty.next_retry_at.is_none());
        assert_eq!(empty.exhausted, 0);

        // Two enqueued, one paused with a reason + retry time.
        db.enqueue_document("folder", "/a.md", Some("h"))
            .await
            .unwrap();
        db.enqueue_document("folder", "/b.md", Some("h"))
            .await
            .unwrap();
        db.mark_paused(
            "folder",
            "/a.md",
            "analysis LLM failed",
            Some(1_712_678_400),
        )
        .await
        .unwrap();

        let status = db.document_enrichment_queue_status().await.unwrap();
        assert_eq!(status.pending, 2, "pending counts all not-done rows");
        assert_eq!(status.paused_reason.as_deref(), Some("analysis LLM failed"));
        assert_eq!(status.next_retry_at, Some(1_712_678_400));
        assert_eq!(status.exhausted, 0);

        // Once done, rows drop out of the pending count and the pause clears.
        db.mark_done("folder", "/a.md").await.unwrap();
        db.mark_done("folder", "/b.md").await.unwrap();
        let drained = db.document_enrichment_queue_status().await.unwrap();
        assert_eq!(drained.pending, 0);
        assert!(drained.paused_reason.is_none());
        assert!(drained.next_retry_at.is_none());
        assert_eq!(drained.exhausted, 0);
    }

    #[tokio::test]
    async fn queue_status_surfaces_all_exhausted_queue_as_paused() {
        let (db, _dir) = test_db().await;
        db.enqueue_document("folder", "/poison.md", Some("poison-hash"))
            .await
            .unwrap();
        for _ in 0..MemoryDB::DOC_ENRICHMENT_MAX_ATTEMPTS {
            db.mark_paused(
                "folder",
                "/poison.md",
                "poison document",
                Some(chrono::Utc::now().timestamp() - 1),
            )
            .await
            .unwrap();
        }

        let status = db.document_enrichment_queue_status().await.unwrap();
        assert_eq!(status.exhausted, 1);
        // The wire mapping (`crates/wenlan-server/src/routes.rs`) reports
        // `Paused` whenever `paused_reason.is_some()`; a queue holding only
        // exhausted rows must clear that bar instead of reading as `Active`
        // forever.
        let reason = status
            .paused_reason
            .expect("all-exhausted queue reports a paused reason");
        assert!(
            reason.contains("exhausted"),
            "reason mentions exhaustion: {reason}"
        );
        assert!(
            status.next_retry_at.is_none(),
            "an all-exhausted queue has no next retry to report"
        );
    }

    #[tokio::test]
    async fn queue_status_paused_reason_mentions_both_exhausted_and_retryable() {
        let (db, _dir) = test_db().await;
        db.enqueue_document("folder", "/poison.md", Some("poison-hash"))
            .await
            .unwrap();
        for _ in 0..MemoryDB::DOC_ENRICHMENT_MAX_ATTEMPTS {
            db.mark_paused(
                "folder",
                "/poison.md",
                "poison document",
                Some(chrono::Utc::now().timestamp() - 1),
            )
            .await
            .unwrap();
        }
        db.enqueue_document("folder", "/retrying.md", Some("retrying-hash"))
            .await
            .unwrap();
        db.mark_paused(
            "folder",
            "/retrying.md",
            "analysis LLM failed",
            Some(1_712_678_400),
        )
        .await
        .unwrap();

        let status = db.document_enrichment_queue_status().await.unwrap();
        assert_eq!(status.exhausted, 1);
        let reason = status
            .paused_reason
            .expect("mixed queue reports a paused reason");
        assert!(
            reason.contains("analysis LLM failed"),
            "reason mentions the retryable pause: {reason}"
        );
        assert!(
            reason.contains("exhausted"),
            "reason also mentions the exhausted document: {reason}"
        );
        assert_eq!(
            status.next_retry_at,
            Some(1_712_678_400),
            "next_retry_at still tracks the retryable row"
        );
    }

    // ── restart resume: in_progress rows are requeued (checkpoint preserved) ──

    #[tokio::test]
    async fn reset_in_progress_requeues_orphaned_docs_preserving_checkpoint() {
        let (db, _dir) = test_db().await;
        db.enqueue_document("folder", "/a.md", Some("h"))
            .await
            .unwrap();
        let claimed = db.claim_next_pending().await.unwrap().expect("claim");
        assert_eq!(claimed.status, "in_progress");
        db.checkpoint_chunk("folder", "/a.md", 4).await.unwrap();
        // Simulate a crash: the row is stuck in_progress and NOT claimable.
        assert!(db.claim_next_pending().await.unwrap().is_none());

        // A fresh daemon start requeues orphaned in_progress rows.
        let requeued = db.reset_in_progress_documents().await.unwrap();
        assert_eq!(requeued, 1, "one orphaned in_progress row requeued");
        let entry = db
            .get_queue_entry("folder", "/a.md")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entry.status, "pending");
        assert_eq!(
            entry.last_completed_chunk, 4,
            "checkpoint preserved so the resume skips analyzed chunks"
        );

        // Claimable again, resuming from the checkpoint.
        let resumed = db
            .claim_next_pending()
            .await
            .unwrap()
            .expect("claimable after reset");
        assert_eq!(resumed.last_completed_chunk, 4);
    }

    /// Count active `creation_kind='source'` pages.
    async fn count_source_pages(db: &MemoryDB) -> i64 {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM pages WHERE creation_kind = 'source' AND status = 'active'",
                (),
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
    }
}
