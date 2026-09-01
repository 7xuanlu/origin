use super::{log_activity_best_effort, WriteOutcome, WriteResult};
use crate::{
    db::{MemoryDB, UNFILED_SPACE_ID},
    error::WenlanError,
};
use std::{collections::HashSet, path::Path};
use wenlan_types::requests::CreateConceptRequest;

const VALID_PAGE_CREATION_KINDS: [&str; 5] =
    ["distilled", "authored", "research", "imported", "source"];

const PAGE_BIRTH_REVIEW_STATUS: &str = "unconfirmed";

#[cfg(test)]
type CreatePagePreDbGate = (
    String,
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
);

#[cfg(test)]
pub(super) static CREATE_PAGE_PRE_DB_GATE: std::sync::Mutex<Option<CreatePagePreDbGate>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
async fn create_page_pre_db_pause(page_id: &str) {
    let gate = {
        let mut slot = CREATE_PAGE_PRE_DB_GATE.lock().unwrap();
        if slot
            .as_ref()
            .is_some_and(|(target, _, _)| target == page_id)
        {
            slot.take()
        } else {
            None
        }
    };
    if let Some((_, parked, resume)) = gate {
        let _ = parked.send(());
        let _ = resume.await;
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn write_document_source_page_impl(
    db: &MemoryDB,
    page_id: &str,
    title: &str,
    summary: Option<&str>,
    content: &str,
    source_memory_ids: &[String],
    queue_source_id: &str,
    file_path: &str,
    expected_content_hash: Option<&str>,
    expected_page_version: Option<i64>,
    agent: &str,
) -> Result<WriteResult, WenlanError> {
    if title.trim().is_empty() || content.trim().is_empty() || source_memory_ids.is_empty() {
        return Err(WenlanError::Validation(
            "document source Page requires title, content, and source ids".into(),
        ));
    }
    let source_refs: Vec<&str> = source_memory_ids.iter().map(String::as_str).collect();
    let now = chrono::Utc::now().to_rfc3339();
    let wrote = match expected_page_version {
        Some(version) => {
            db.replace_source_page_at_document_hash(
                page_id,
                title,
                summary,
                content,
                &source_refs,
                agent,
                queue_source_id,
                file_path,
                expected_content_hash,
                version,
            )
            .await?
        }
        None => {
            db.insert_document_source_page_at_hash(
                page_id,
                title,
                summary,
                content,
                &source_refs,
                &now,
                queue_source_id,
                file_path,
                expected_content_hash,
            )
            .await?
        }
    };
    if !wrote {
        return Err(WenlanError::Conflict(format!(
            "document source Page input changed: {page_id}"
        )));
    }

    let action = if expected_page_version.is_some() {
        "page_update"
    } else {
        "page_create"
    };
    let detail = format!("title={title}, sources={}", source_memory_ids.len());
    if let Err(error) = db
        .log_agent_activity(agent, action, source_memory_ids, None, &detail)
        .await
    {
        log::warn!("[document_source_page] activity log failed: {error}");
    }
    Ok(WriteResult {
        id: page_id.to_string(),
        attached_to: None,
        warnings: Vec::new(),
        wrote: true,
        revision_card_id: None,
        gated: false,
        acknowledged: false,
        outcome: WriteOutcome::Wrote,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn replace_source_page_impl(
    db: &MemoryDB,
    page_id: &str,
    title: &str,
    summary: Option<&str>,
    content: &str,
    source_memory_ids: &[String],
    agent: &str,
) -> Result<WriteResult, WenlanError> {
    if title.trim().is_empty() || content.trim().is_empty() || source_memory_ids.is_empty() {
        return Err(WenlanError::Validation(
            "source Page replacement requires title, content, and source ids".into(),
        ));
    }
    let content = crate::export::provenance::validate_canonical_page_content(content)?;
    let current = db
        .get_page(page_id)
        .await?
        .ok_or_else(|| WenlanError::NotFound(format!("Page not found: {page_id}")))?;
    if current.creation_kind != "source" || current.user_edited {
        return Err(WenlanError::Conflict(format!(
            "source Page {page_id} is not machine-owned"
        )));
    }

    // Nothing to write: this enrichment recomputed exactly what the page
    // already says. Mirrors the same early return on the Update path, and
    // matters more here — a document whose analysis LLM is unreachable rebuilds
    // a byte-identical stub body on every retry, and without this each retry
    // would bump the version and append another identical history row.
    //
    // Canonical storage is exact source, so compare the validated incoming
    // representation directly with the stored representation.
    let stored_sources: HashSet<&str> = current
        .source_memory_ids
        .iter()
        .map(String::as_str)
        .collect();
    let incoming_sources: HashSet<&str> = source_memory_ids.iter().map(String::as_str).collect();
    if current.title == title
        && current.summary.as_deref() == summary
        && current.content == content
        && stored_sources == incoming_sources
    {
        return Ok(WriteResult {
            id: page_id.to_string(),
            attached_to: None,
            warnings: Vec::new(),
            wrote: false,
            revision_card_id: None,
            gated: false,
            acknowledged: false,
            outcome: WriteOutcome::Unchanged,
        });
    }

    let source_refs: Vec<&str> = source_memory_ids.iter().map(String::as_str).collect();
    if !db
        .replace_source_page(page_id, title, summary, content, &source_refs, agent)
        .await?
    {
        return Err(WenlanError::Conflict(format!(
            "source Page {page_id} changed ownership before replacement"
        )));
    }
    let detail = format!("title={title}, sources={}", source_memory_ids.len());
    if let Err(error) = db
        .log_agent_activity(agent, "page_update", source_memory_ids, None, &detail)
        .await
    {
        log::warn!("[replace_source_page] activity log failed: {error}");
    }
    Ok(WriteResult {
        id: page_id.to_string(),
        attached_to: None,
        warnings: Vec::new(),
        wrote: true,
        revision_card_id: None,
        gated: false,
        outcome: WriteOutcome::Wrote,
        acknowledged: false,
    })
}

pub(super) async fn attach_page_sources_impl(
    db: &MemoryDB,
    page_id: &str,
    source_memory_ids: &[String],
    link_reason: &str,
    agent: &str,
) -> Result<WriteResult, WenlanError> {
    for sid in source_memory_ids {
        db.link_page_source(page_id, sid, link_reason).await?;
    }
    log_activity_best_effort(db, agent, "page_attach", page_id).await;
    Ok(WriteResult {
        id: page_id.to_string(),
        attached_to: Some(page_id.to_string()),
        warnings: vec![],
        wrote: true,
        revision_card_id: None,
        gated: false,
        outcome: WriteOutcome::Wrote,
        acknowledged: false,
    })
}

async fn page_source_reference_exists(
    db: &MemoryDB,
    creation_kind: &str,
    source_id: &str,
) -> Result<bool, WenlanError> {
    if db.get_memory_detail(source_id).await?.is_some() {
        return Ok(true);
    }
    if creation_kind != "source" {
        return Ok(false);
    }
    db.has_active_non_episode_memory_id(source_id).await
}

pub(super) struct CreatePageInput<'a> {
    pub(super) page_id: Option<&'a str>,
    pub(super) req: CreateConceptRequest,
    pub(super) agent: &'a str,
    pub(super) knowledge_path: Option<&'a Path>,
    pub(super) page_min_cluster_size: usize,
    pub(super) page_match_threshold: f64,
    pub(super) citations_json: Option<&'a str>,
}

pub(super) async fn create_page_impl(
    db: &MemoryDB,
    input: CreatePageInput<'_>,
) -> Result<WriteResult, WenlanError> {
    let CreatePageInput {
        page_id,
        req,
        agent,
        knowledge_path,
        page_min_cluster_size,
        page_match_threshold,
        citations_json,
    } = input;

    // Pre-write validation
    if req.title.trim().is_empty() {
        return Err(WenlanError::Validation(
            "page title must not be empty".into(),
        ));
    }
    if req.content.trim().is_empty() {
        return Err(WenlanError::Validation(
            "page content must not be empty".into(),
        ));
    }
    crate::export::provenance::validate_canonical_page_content(&req.content)?;
    let creation_kind = req.creation_kind.as_deref().unwrap_or("distilled");
    if creation_kind == "distilled" && req.source_memory_ids.is_empty() {
        return Err(WenlanError::Validation(
            "distilled page must cite at least one source memory".into(),
        ));
    }
    if !VALID_PAGE_CREATION_KINDS.contains(&creation_kind) {
        return Err(WenlanError::Validation(format!(
            "invalid creation_kind '{creation_kind}' (expected one of: distilled, authored, research, imported, source)"
        )));
    }
    if creation_kind == "source" && page_id.is_none() {
        return Err(WenlanError::Validation(
            "source page requires a deterministic page id".into(),
        ));
    }
    if creation_kind == "source" && req.source_memory_ids.is_empty() {
        return Err(WenlanError::Validation(
            "source page must cite at least one source memory".into(),
        ));
    }
    let distinct_source_count = req
        .source_memory_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>()
        .len();
    if creation_kind == "distilled" && distinct_source_count < page_min_cluster_size {
        return Err(WenlanError::Validation(format!(
            "distilled page requires at least {page_min_cluster_size} distinct source memories (got {distinct_source_count})"
        )));
    }
    let review_status = PAGE_BIRTH_REVIEW_STATUS;
    // Resolution check: every source id must exist
    for sid in &req.source_memory_ids {
        if !page_source_reference_exists(db, creation_kind, sid).await? {
            return Err(WenlanError::Validation(format!(
                "source memory '{sid}' does not exist"
            )));
        }
    }
    // Hallucination guard (only when sources are present)
    if !req.source_memory_ids.is_empty() {
        let passed =
            crate::kg_quality::hallucination_guard(db, &req.content, &req.source_memory_ids)
                .await?;
        if !passed {
            return Err(WenlanError::Validation(
                "page body diverges from cited sources (cos sim < 0.6)".into(),
            ));
        }
    }

    if creation_kind == "distilled" {
        let embed_text =
            crate::pages::page_embedding_text(&req.title, req.summary.as_deref(), &req.content);
        let embedding = match db.generate_embeddings(&[embed_text]) {
            Ok(mut embeddings) => embeddings.pop(),
            Err(e) => {
                log::warn!("[create_page] dedup embedding failed: {e}");
                None
            }
        };
        let workspace = req
            .workspace
            .as_deref()
            .or(req.space.as_deref())
            .or_else(|| {
                matches!(req.space, wenlan_types::WriteSpaceTarget::Uncategorized)
                    .then_some(UNFILED_SPACE_ID)
            });
        if let (Some(embedding), Some(workspace)) = (embedding, workspace) {
            if let Some(matched) = db
                .find_matching_page_scoped(
                    req.entity_id.as_deref(),
                    &embedding,
                    page_match_threshold,
                    Some(workspace),
                    false,
                )
                .await?
            {
                let matched_id = matched.id;
                return attach_page_sources_impl(
                    db,
                    &matched_id,
                    &req.source_memory_ids,
                    "page_write_attach",
                    agent,
                )
                .await;
            }
        }
    }

    // Build page
    let id = page_id
        .map(str::to_string)
        .unwrap_or_else(crate::pages::new_page_id);
    let now = chrono::Utc::now().to_rfc3339();
    let page = crate::pages::Page {
        id: id.clone(),
        title: req.title.clone(),
        summary: req.summary.clone(),
        content: req.content.clone(),
        entity_id: req.entity_id.clone(),
        space: req.space.named().map(str::to_string),
        source_memory_ids: req.source_memory_ids.clone(),
        version: 1,
        status: "active".to_string(),
        created_at: now.clone(),
        last_compiled: now.clone(),
        last_modified: now.clone(),
        sources_updated_count: 0,
        stale_reason: None,
        pending_rebuild: None,
        refresh_blocked_reason: None,
        user_edited: false,
        relevance_score: 0.0,
        last_edited_by: None,
        last_edited_at: None,
        last_delta_summary: None,
        changelog: None,
        creation_kind: creation_kind.to_string(),
        review_status: review_status.to_string(),
        workspace: req.workspace.clone(),
        citations: citations_json
            .and_then(|raw| serde_json::from_str(raw).ok())
            .unwrap_or_default(),
        // `insert_page` never sets `kind` explicitly, so the row lands with the
        // schema DEFAULT ('concept') -- mirror that here for consistency.
        kind: "concept".to_string(),
        truth: None,
    };

    // md-first write (only if a knowledge_path was provided)
    let projection = knowledge_path.map(|path| {
        crate::export::knowledge::KnowledgeProjectionWrite::new(path.to_path_buf(), db)
    });
    if let Some(ref projection) = projection {
        let _ = projection
            .write_page_gated(db, &page)
            .await
            .map_err(|e| WenlanError::VectorDb(format!("write_page: {e}")))?;
    }

    #[cfg(test)]
    create_page_pre_db_pause(&id).await;

    // DB index
    let source_refs: Vec<&str> = req.source_memory_ids.iter().map(|s| s.as_str()).collect();
    if let Err(e) = db
        .insert_page_with_kind(
            &id,
            &req.title,
            req.summary.as_deref(),
            &req.content,
            req.entity_id.as_deref(),
            req.space.as_deref(),
            &source_refs,
            &now,
            creation_kind,
            review_status,
            req.workspace.as_deref(),
            citations_json,
        )
        .await
    {
        // Rollback md if it was written
        if let Some(ref projection) = projection {
            if let Err(rb) = projection.remove_page(&id) {
                log::warn!(
                    "[create_page] DB insert failed and md rollback also failed for {id}: db_err={e}, rollback_err={rb}"
                );
            }
        }
        return Err(e);
    }
    drop(projection);

    // Post-write enrichment
    let mut warnings: Vec<String> = Vec::new();

    if creation_kind == "distilled" {
        if let Err(e) =
            crate::maintenance::emit_keep_or_archive_card(db, &id, distinct_source_count).await
        {
            log::warn!("[create_page] keep/archive card failed for {id}: {e}");
            warnings.push(format!("keep/archive card failed: {e}"));
        }
    }

    // 1. Orphan-link resolution (best-effort)
    if let Err(e) = db.resolve_orphan_page_links().await {
        log::warn!("[create_page] orphan link resolve failed for {id}: {e}");
        warnings.push(format!("orphan link resolve failed: {e}"));
    }

    // 2. Self-retrieval verification
    if let Ok(result) = crate::kg_quality::verify_page(db, &id, &req.title).await {
        for w in &result.warnings {
            log::warn!("[create_page] {w}");
            warnings.push(w.clone());
        }
    }

    // 3. Activity log (matches synthesis/distill.rs:498 shape — source ids as the memory ids list)
    let detail = format!(
        "title={}, sources={}",
        req.title,
        req.source_memory_ids.len()
    );
    if let Err(e) = db
        .log_agent_activity(agent, "page_create", &req.source_memory_ids, None, &detail)
        .await
    {
        log::warn!("[create_page] activity log failed: {e}");
    }

    Ok(WriteResult {
        id,
        attached_to: None,
        warnings,
        wrote: true,
        revision_card_id: None,
        gated: false,
        outcome: WriteOutcome::Wrote,
        acknowledged: false,
    })
}
