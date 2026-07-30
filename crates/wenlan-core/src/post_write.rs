// SPDX-License-Identifier: Apache-2.0
//! Canonical write-path capability functions. Each fn owns the full create
//! flow for one kind: validation, resolve-or-create (where applicable),
//! storage primitive call, post-write enrichment (verify, log, refinery
//! enqueue). Both HTTP route handlers and daemon-internal extractors call
//! these -- eliminating drift between agent-LLM and daemon-LLM trigger paths.

use crate::db::{MemoryDB, UNFILED_SPACE_ID};
use crate::error::WenlanError;
use std::{collections::HashSet, path::Path, str::FromStr};
use wenlan_types::{
    repair::{RepairDigest, RepairManifest, RepairRollbackPayloadV2},
    requests::{CreateConceptRequest, UpdatePageRequest},
    MemoryType, RawDocument,
};

mod entity_graph;
mod page_revision;

pub use self::entity_graph::{
    add_observation, create_entity, create_relation, create_relation_with_span,
};
pub use self::page_revision::{
    accept_pending_revision, accept_pending_revision_with_knowledge_path, dismiss_contradiction,
    dismiss_pending_revision,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WriteResult {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attached_to: Option<String>,
    pub warnings: Vec<String>,
    pub wrote: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_card_id: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub gated: bool,
    #[serde(default, skip_serializing_if = "is_false")]
    pub acknowledged: bool,
    /// Why the write did or did not land. `wrote` says whether the page moved;
    /// this says why not, which is what a caller needs to answer the user.
    ///
    /// Branch on this, never on `warnings` — the strings are for humans reading
    /// logs and will get reworded.
    #[serde(default)]
    pub outcome: WriteOutcome,
}

/// The distinguishable ends of a page write.
///
/// Without this, every unsuccessful write looked the same to a caller: "content
/// was already correct" and "somebody else holds the page" both arrived as
/// `wrote: false` and an empty warning list. The manual-edit route guessed
/// conflict for both and told users their routine no-op save had been
/// overwritten by someone else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteOutcome {
    /// The page was updated.
    ///
    /// Default because the only `WriteResult`s that predate this field are the
    /// ones stored in idempotency receipts, and a receipt is only ever written
    /// inside a transaction that committed — so a receipt without an outcome
    /// describes a write that landed.
    #[default]
    Wrote,
    /// The page already said what the caller asked for, or the caller asked to
    /// write only if stale and it wasn't. Nothing to do, and nothing wrong.
    Unchanged,
    /// The caller declared an `expected_version` that no longer matches. Its
    /// content was computed against a version that has moved on.
    Refused,
    /// The page kept moving and the write lost every CAS attempt. The caller's
    /// content was discarded.
    Contended,
    /// A machine write to a human-owned page, preserved as a revision card
    /// rather than applied. See `revision_card_id`.
    Gated,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryUpdate<'a> {
    pub content: Option<&'a str>,
    pub space: Option<Option<&'a str>>,
    pub confirm: bool,
    pub memory_type: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairWriteProof {
    before_target_receipt: RepairDigest,
    after_target_receipt: RepairDigest,
    non_target_before: RepairDigest,
    non_target_after: RepairDigest,
    post_apply_db_digest: RepairDigest,
}

impl RepairWriteProof {
    pub(crate) fn from_parts(
        before_target_receipt: RepairDigest,
        after_target_receipt: RepairDigest,
        non_target_before: RepairDigest,
        non_target_after: RepairDigest,
        post_apply_db_digest: RepairDigest,
    ) -> Self {
        Self {
            before_target_receipt,
            after_target_receipt,
            non_target_before,
            non_target_after,
            post_apply_db_digest,
        }
    }

    pub fn before_target_receipt(&self) -> &RepairDigest {
        &self.before_target_receipt
    }

    pub fn after_target_receipt(&self) -> &RepairDigest {
        &self.after_target_receipt
    }

    pub fn non_target_before(&self) -> &RepairDigest {
        &self.non_target_before
    }

    pub fn non_target_after(&self) -> &RepairDigest {
        &self.non_target_after
    }

    pub fn post_apply_db_digest(&self) -> &RepairDigest {
        &self.post_apply_db_digest
    }
}

pub async fn reclassify_memory_cas<F>(
    db: &MemoryDB,
    source_id: &str,
    expected_receipt: &RepairDigest,
    expected_space: Option<&str>,
    review_binding: Option<&wenlan_types::repair::RepairReviewBinding>,
    after_memory_type: MemoryType,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    db.reclassify_memory_repair_cas(
        source_id,
        expected_receipt,
        expected_space,
        review_binding,
        after_memory_type,
        before_commit,
    )
    .await
}

#[cfg(test)]
pub(crate) async fn reclassify_memory_cas_with_forced_rollback_failure<F>(
    db: &MemoryDB,
    source_id: &str,
    expected_receipt: &RepairDigest,
    expected_space: Option<&str>,
    review_binding: Option<&wenlan_types::repair::RepairReviewBinding>,
    after_memory_type: MemoryType,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    db.reclassify_memory_repair_cas_with_forced_rollback_failure(
        source_id,
        expected_receipt,
        expected_space,
        review_binding,
        after_memory_type,
        before_commit,
    )
    .await
}

pub(crate) async fn complete_entity_extraction_cas<F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &RepairRollbackPayloadV2,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    db.complete_entity_extraction_repair_cas(manifest, rollback, before_commit)
        .await
}

#[cfg(test)]
pub(crate) async fn complete_entity_extraction_cas_with_forced_rollback_failure<F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &RepairRollbackPayloadV2,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    db.complete_entity_extraction_repair_cas_with_forced_rollback_failure(
        manifest,
        rollback,
        before_commit,
    )
    .await
}

pub(crate) async fn rename_page_title_cas<F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &RepairRollbackPayloadV2,
    page_root: &Path,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    crate::db::repair_page_rename::rename_page_title_cas_inner(
        db,
        manifest,
        rollback,
        page_root,
        before_commit,
    )
    .await
}

#[cfg(test)]
pub(crate) async fn rename_page_title_cas_with_projection_write_hook<F, G>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &RepairRollbackPayloadV2,
    page_root: &Path,
    after_target_write: G,
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
    G: FnOnce() -> Result<(), WenlanError> + 'static,
{
    let checkpoints = crate::db::repair_page_rename::RenamePageTitleTestCheckpoints {
        after_target_write: Some(Box::new(after_target_write)),
        ..Default::default()
    };
    crate::db::repair_page_rename::with_rename_page_title_test_checkpoints(
        checkpoints,
        crate::db::repair_page_rename::rename_page_title_cas_inner(
            db,
            manifest,
            rollback,
            page_root,
            before_commit,
        ),
    )
    .await
}

pub use crate::db::repair_deterministic::apply_deterministic_repair_cas;

pub(crate) use crate::db::repair_page_regenerate::regenerate_page_projection_cas;

pub(crate) use crate::db::repair_stale_projection::quarantine_stale_page_projection_cas_with_apply_journal;

#[cfg(test)]
pub(crate) use crate::db::repair_stale_projection::{
    quarantine_stale_page_projection_cas, quarantine_stale_page_projection_cas_with_after_pin,
    quarantine_stale_page_projection_cas_with_before_pin,
    quarantine_stale_page_projection_cas_with_before_source_stage,
};

pub async fn update_memory(
    db: &MemoryDB,
    source_id: &str,
    update: MemoryUpdate<'_>,
) -> Result<(), WenlanError> {
    let _space_write_guard = if update.space.is_some() {
        Some(db.lock_space_writes().await)
    } else {
        None
    };
    let registered_space = match update.space {
        None => None,
        Some(None) => Some(None),
        Some(Some(requested)) => Some(db.registered_space_or_none(Some(requested)).await?),
    };
    let parsed_memory_type = update
        .memory_type
        .map(MemoryType::from_str)
        .transpose()
        .map_err(WenlanError::Validation)?;
    let normalized_memory_type = parsed_memory_type.map(|memory_type| memory_type.to_string());

    db.apply_memory_update(
        source_id,
        update.content,
        registered_space
            .as_ref()
            .map(|space| space.as_ref().map(String::as_str)),
        update.confirm,
        normalized_memory_type.as_deref(),
        None,
    )
    .await
}

fn is_false(value: &bool) -> bool {
    !*value
}

const VALID_PAGE_CREATION_KINDS: [&str; 5] =
    ["distilled", "authored", "research", "imported", "source"];
const PAGE_BIRTH_REVIEW_STATUS: &str = "unconfirmed";

pub enum PageWrite<'a> {
    Attach {
        page_id: &'a str,
        source_memory_ids: &'a [String],
        link_reason: &'a str,
        agent: &'a str,
    },
    Create {
        page_id: Option<&'a str>,
        req: CreateConceptRequest,
        agent: &'a str,
        knowledge_path: Option<&'a Path>,
        page_min_cluster_size: usize,
        page_match_threshold: f64,
        citations_json: Option<String>,
    },
    Update {
        page_id: &'a str,
        req: UpdatePageRequest,
        edited_by: &'a str,
        require_stale: bool,
        expected_source_revision: Option<i64>,
        knowledge_path: Option<&'a Path>,
        citations: Option<(String, String)>,
    },
    /// Human content edit that preserves the source set from the exact Page
    /// generation selected inside the update CAS. HTTP callers do not own the
    /// source list, so it must not be snapshotted outside the gate.
    UpdatePreservingSources {
        page_id: &'a str,
        req: UpdatePageRequest,
        edited_by: &'a str,
        knowledge_path: Option<&'a Path>,
    },
    ReplaceSource {
        page_id: &'a str,
        title: &'a str,
        summary: Option<&'a str>,
        content: &'a str,
        source_memory_ids: &'a [String],
        agent: &'a str,
    },
    DocumentSource {
        page_id: &'a str,
        title: &'a str,
        summary: Option<&'a str>,
        content: &'a str,
        source_memory_ids: &'a [String],
        queue_source_id: &'a str,
        file_path: &'a str,
        expected_content_hash: Option<&'a str>,
        expected_page_version: Option<i64>,
        agent: &'a str,
    },
}

#[derive(Clone, Copy)]
struct PageGrowthCommit<'a> {
    source_id: &'a str,
    expected_memory_version: i64,
    expected_page_version: i64,
    expected_source_revision: i64,
}

pub async fn page_write(db: &MemoryDB, write: PageWrite<'_>) -> Result<WriteResult, WenlanError> {
    match write {
        PageWrite::Attach {
            page_id,
            source_memory_ids,
            link_reason,
            agent,
        } => attach_page_sources_impl(db, page_id, source_memory_ids, link_reason, agent).await,
        PageWrite::Create {
            page_id,
            req,
            agent,
            knowledge_path,
            page_min_cluster_size,
            page_match_threshold,
            citations_json,
        } => {
            create_page_impl(
                db,
                CreatePageInput {
                    page_id,
                    req,
                    agent,
                    knowledge_path,
                    page_min_cluster_size,
                    page_match_threshold,
                    citations_json: citations_json.as_deref(),
                },
            )
            .await
        }
        PageWrite::Update {
            page_id,
            req,
            edited_by,
            require_stale,
            expected_source_revision,
            knowledge_path,
            citations,
        } => {
            update_page_impl(
                db,
                page_id,
                req,
                edited_by,
                require_stale,
                expected_source_revision,
                knowledge_path,
                citations,
                None,
                false,
            )
            .await
        }
        PageWrite::UpdatePreservingSources {
            page_id,
            req,
            edited_by,
            knowledge_path,
        } => {
            update_page_impl(
                db,
                page_id,
                req,
                edited_by,
                false,
                None,
                knowledge_path,
                None,
                None,
                true,
            )
            .await
        }
        PageWrite::ReplaceSource {
            page_id,
            title,
            summary,
            content,
            source_memory_ids,
            agent,
        } => {
            replace_source_page_impl(
                db,
                page_id,
                title,
                summary,
                content,
                source_memory_ids,
                agent,
            )
            .await
        }
        PageWrite::DocumentSource {
            page_id,
            title,
            summary,
            content,
            source_memory_ids,
            queue_source_id,
            file_path,
            expected_content_hash,
            expected_page_version,
            agent,
        } => {
            write_document_source_page_impl(
                db,
                page_id,
                title,
                summary,
                content,
                source_memory_ids,
                queue_source_id,
                file_path,
                expected_content_hash,
                expected_page_version,
                agent,
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn write_document_source_page_impl(
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
async fn replace_source_page_impl(
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

async fn attach_page_sources_impl(
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

/// Best-effort activity logger used by curation-mutate capability fns.
/// Failure to log does not fail the operation — matches the pattern in
/// `create_entity`, `create_relation`, etc.
pub(crate) async fn log_activity_best_effort(
    db: &MemoryDB,
    agent: &str,
    action: &str,
    target_id: &str,
) {
    let target = target_id.to_string();
    if let Err(e) = db
        .log_agent_activity(agent, action, std::slice::from_ref(&target), None, "")
        .await
    {
        log::warn!("[{}] activity log failed: {}", action, e);
    }
}

/// Create a distilled wiki page. Canonical entry for both agent-triggered
/// (`/api/pages`) and daemon-internal distillation callers.
pub async fn create_page(
    db: &MemoryDB,
    req: CreateConceptRequest,
    agent: &str,
    knowledge_path: Option<&Path>,
) -> Result<WriteResult, WenlanError> {
    let distillation = crate::tuning::DistillationConfig::default();
    create_page_with_tuning(
        db,
        req,
        agent,
        knowledge_path,
        distillation.page_min_cluster_size,
        distillation.page_match_threshold,
    )
    .await
}

pub async fn create_page_with_floor(
    db: &MemoryDB,
    req: CreateConceptRequest,
    agent: &str,
    knowledge_path: Option<&Path>,
    page_min_cluster_size: usize,
) -> Result<WriteResult, WenlanError> {
    create_page_with_tuning(
        db,
        req,
        agent,
        knowledge_path,
        page_min_cluster_size,
        crate::tuning::DistillationConfig::default().page_match_threshold,
    )
    .await
}

pub async fn create_page_with_tuning(
    db: &MemoryDB,
    req: CreateConceptRequest,
    agent: &str,
    knowledge_path: Option<&Path>,
    page_min_cluster_size: usize,
    page_match_threshold: f64,
) -> Result<WriteResult, WenlanError> {
    page_write(
        db,
        PageWrite::Create {
            page_id: None,
            req,
            agent,
            knowledge_path,
            page_min_cluster_size,
            page_match_threshold,
            citations_json: None,
        },
    )
    .await
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

struct CreatePageInput<'a> {
    page_id: Option<&'a str>,
    req: CreateConceptRequest,
    agent: &'a str,
    knowledge_path: Option<&'a Path>,
    page_min_cluster_size: usize,
    page_match_threshold: f64,
    citations_json: Option<&'a str>,
}

async fn create_page_impl(
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

/// A daemon pipeline stage that rewrites page prose with an LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage<'a> {
    Distill,
    ReDistill,
    PageGrowth,
    RefineryMerge,
    /// A writer string the gate does not recognize, carried verbatim.
    ///
    /// Two kinds of string land here. Legitimate ones: `edited_by` values that
    /// reach `page_history` without passing this gate (`create` and
    /// `migration_84` are SQL literals in db.rs; `citation_backfill` and
    /// `revision_accept` write via the db layer). Illegitimate ones: a typo in
    /// a caller's literal. The gate cannot tell them apart, so both get the
    /// conservative answer — see `Writer::is_machine`.
    Unknown(&'a str),
}

/// Who is writing a page, as the spec's `human | agent | pipeline(stage)`.
///
/// This is a **lens over the persisted string, not a replacement for it**.
/// Every page write records provenance as text in `pages.changelog` and
/// `page_history.edited_by`, and those bytes are already in users' databases,
/// so `as_str` round-trips the exact literal that was classified.
///
/// Its job is to put the write gate's three authority questions in one place
/// with one vocabulary. They used to be three independent `matches!` lists over
/// `&str`, which could drift apart silently and had no name for the fallthrough
/// case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Writer<'a> {
    /// A person edited the page directly — in the app editor, or in the vault
    /// on disk. The only identity that may overwrite a human-owned page.
    Human(&'a str),
    /// An agent asked for a faithful re-synth of a page it already owns.
    Agent(&'a str),
    /// A daemon pipeline stage.
    Pipeline(PipelineStage<'a>),
}

impl<'a> Writer<'a> {
    /// Total over every `&str` — there is no failure mode, only `Unknown`.
    pub fn classify(edited_by: &'a str) -> Self {
        match edited_by {
            "manual_edit" | "fs_edit" => Writer::Human(edited_by),
            "agent_refresh" => Writer::Agent(edited_by),
            "distill" => Writer::Pipeline(PipelineStage::Distill),
            "re_distill" => Writer::Pipeline(PipelineStage::ReDistill),
            "page_growth" => Writer::Pipeline(PipelineStage::PageGrowth),
            "refinery_merge" => Writer::Pipeline(PipelineStage::RefineryMerge),
            other => Writer::Pipeline(PipelineStage::Unknown(other)),
        }
    }

    /// The persisted `edited_by` literal, byte-identical to what was classified.
    pub fn as_str(&self) -> &'a str {
        match self {
            Writer::Human(s) | Writer::Agent(s) => s,
            Writer::Pipeline(stage) => match stage {
                PipelineStage::Distill => "distill",
                PipelineStage::ReDistill => "re_distill",
                PipelineStage::PageGrowth => "page_growth",
                PipelineStage::RefineryMerge => "refinery_merge",
                PipelineStage::Unknown(s) => s,
            },
        }
    }

    /// Everything that is not a human. **Unknown counts as machine on purpose**:
    /// that is the fail-safe direction, because a machine write to a
    /// human-owned page is staged as a revision card rather than overwriting
    /// the human's prose. Trusting an unrecognized string as human would let it
    /// clobber that prose instead.
    fn is_machine(&self) -> bool {
        !matches!(self, Writer::Human(_))
    }

    /// Writers that bypass the hallucination guard. Incremental updates can push
    /// aggregate cosine sim below 0.6; the HTTP/MCP `agent_refresh` route
    /// historically accepted agent-provided refreshes without this guard, so
    /// routing it through PageWrite preserves that behavior.
    ///
    /// This is `is_llm_rewrite` plus `Agent` — the two used to be parallel
    /// `matches!` lists whose only difference was the `agent_refresh` arm, with
    /// a comment warning not to merge them. Here the difference is structural.
    fn skips_hallucination_guard(&self) -> bool {
        matches!(self, Writer::Agent(_)) || self.is_llm_rewrite()
    }

    /// A recognized LLM-rewrite stage, checked by the shrink-guard. `Unknown` is
    /// excluded: the shrink-guard rejects a write outright, so an unrecognized
    /// writer must not be able to opt itself into being rejected.
    fn is_llm_rewrite(&self) -> bool {
        matches!(
            self,
            Writer::Pipeline(
                PipelineStage::Distill
                    | PipelineStage::ReDistill
                    | PipelineStage::PageGrowth
                    | PipelineStage::RefineryMerge
            )
        )
    }
}

pub fn page_is_human_owned(page: &crate::pages::Page) -> bool {
    page.user_edited || page.creation_kind == "authored"
}

/// Stage a machine write to a human-owned page as a pending revision card
/// instead of overwriting the page's prose. Uses the same grammar as L3
/// doc-grounded revisions (`crate::reconcile::write_revision`): a
/// `source='memory'`, `pending_revision=1`, `supersedes=<page id>` row that
/// `list_pending_revisions` surfaces on the `/curate revisions` queue. The page
/// itself is never mutated here — the human accepts or dismisses the card.
/// Returns a gated `WriteResult` carrying the new card id.
pub async fn stage_page_revision_card(
    db: &MemoryDB,
    page: &crate::pages::Page,
    content: &str,
    source_memory_ids: &[String],
    edited_by: &str,
    retry: Option<&RetryIdentity>,
) -> Result<WriteResult, WenlanError> {
    crate::export::provenance::validate_canonical_page_content(content)?;

    let revision_card_id = format!(
        "mem_{}",
        uuid::Uuid::new_v4()
            .to_string()
            .replace('-', "")
            .chars()
            .take(12)
            .collect::<String>()
    );
    let structured = serde_json::json!({
        "revision_kind": "page_write",
        "target_kind": "page",
        "revises_page": page.id,
        "page_version": page.version,
        "edited_by": edited_by,
        "source_memory_ids": source_memory_ids,
    })
    .to_string();
    let title: String = format!("Revision: {}", page.title)
        .chars()
        .take(80)
        .collect();
    let row = RawDocument {
        source: "memory".to_string(),
        source_id: revision_card_id.clone(),
        title,
        content: content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        space: page.space.clone().or_else(|| page.workspace.clone()),
        source_agent: Some("page_write".to_string()),
        confidence: Some(0.9),
        confirmed: Some(false),
        stability: Some("new".to_string()),
        supersedes: Some(page.id.clone()),
        pending_revision: true,
        structured_fields: Some(structured.clone()),
        source_text: Some(content.to_string()),
        ..Default::default()
    };
    let result = WriteResult {
        id: page.id.clone(),
        attached_to: None,
        warnings: vec![
            "human-owned page; staged revision card instead of overwriting content".to_string(),
        ],
        wrote: false,
        revision_card_id: Some(revision_card_id.clone()),
        gated: true,
        outcome: WriteOutcome::Gated,
        acknowledged: false,
    };
    if let Some(retry_identity @ (caller, operation, digest)) = retry {
        let response = serde_json::to_string(&result)?;
        let write = db
            .upsert_documents_with_operation_receipt(
                vec![row],
                crate::db::OperationReceipt {
                    caller_id: caller,
                    operation_id: operation,
                    request_digest: digest,
                    response: &response,
                },
            )
            .await;
        match write {
            Ok(_) => {}
            Err(error @ WenlanError::Conflict(_)) => {
                return replay_matching_operation_receipt(db, retry_identity, error).await;
            }
            Err(error) => return Err(error),
        }
    } else {
        db.upsert_documents(vec![row]).await?;
    }
    if let Err(e) = db
        .log_agent_activity(
            edited_by,
            "page_revision_card",
            &[page.id.clone(), revision_card_id.clone()],
            None,
            &structured,
        )
        .await
    {
        log::warn!("[page_revision_card] activity log failed: {e}");
    }

    Ok(result)
}

/// Parse WENLAN_MERGE_SHRINK_GUARD env var as f64 threshold.
/// Returns Some(t) when set to a valid float; None when unset/unparseable
/// (guard OFF = byte-identical behavior to pre-T17).
/// Mirrors page_channel_enabled() env-read discipline in db.rs.
pub(crate) fn merge_shrink_threshold() -> Option<f64> {
    std::env::var("WENLAN_MERGE_SHRINK_GUARD")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
}

/// Update a distilled wiki page. Canonical entry for all page-update paths:
/// daemon-internal distillation, refinery re-distill, fs watcher, and
/// future agent-HTTP routes.
///
/// Two write modes via `require_stale`:
/// - `false` — unconditional write (post-ingest, distill, page_growth callers)
/// - `true`  — CAS: only writes when `stale_reason IS NOT NULL` (refinery callers).
///   Returns `Ok(WriteResult { warnings: vec![] })` without writing when not stale.
///
/// Hallucination guard runs for human writers (`manual_edit`, `fs_edit`) and
/// for any writer the gate does not recognize. `agent_refresh` and the LLM
/// rewrite stages (`distill`, `re_distill`, `page_growth`, `refinery_merge`)
/// skip it — incremental updates may push aggregate cosine sim below 0.6 and
/// would silently drop legitimate writes. See `Writer::skips_hallucination_guard`,
/// which is the single source of truth; `WRITER_TABLE` pins it per writer.
///
/// `citations`: `Some((citations_json, stats_summary))` when the caller has
/// freshly verified [N] markers against a numbered source list for this
/// exact `req.content` — persisted atomically with the content, and
/// `stats_summary` is recorded on the changelog entry. `None` always resets
/// `citations` to SQL `NULL` (a stale marker-to-source map must not survive a
/// content change without fresh verification, and the new body must remain
/// eligible for bounded annotation).
#[allow(clippy::too_many_arguments)]
pub async fn update_page(
    db: &MemoryDB,
    page_id: &str,
    req: UpdatePageRequest,
    edited_by: &str,
    require_stale: bool,
    knowledge_path: Option<&Path>,
    citations: Option<(String, String)>,
) -> Result<WriteResult, WenlanError> {
    page_write(
        db,
        PageWrite::Update {
            page_id,
            req,
            edited_by,
            require_stale,
            expected_source_revision: None,
            knowledge_path,
            citations,
        },
    )
    .await
}

/// Update only the Page body while preserving the sources from the exact
/// generation loaded inside PageWrite's CAS. This is the manual-editor seam:
/// the HTTP request does not own `source_memory_ids`, so a source attached
/// between request parsing and the write gate must survive.
pub async fn update_page_preserving_sources(
    db: &MemoryDB,
    page_id: &str,
    req: UpdatePageRequest,
    edited_by: &str,
    knowledge_path: Option<&Path>,
) -> Result<WriteResult, WenlanError> {
    page_write(
        db,
        PageWrite::UpdatePreservingSources {
            page_id,
            req,
            edited_by,
            knowledge_path,
        },
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn update_page_at_source_revision(
    db: &MemoryDB,
    page_id: &str,
    req: UpdatePageRequest,
    edited_by: &str,
    require_stale: bool,
    expected_source_revision: i64,
    knowledge_path: Option<&Path>,
    citations: Option<(String, String)>,
) -> Result<WriteResult, WenlanError> {
    page_write(
        db,
        PageWrite::Update {
            page_id,
            req,
            edited_by,
            require_stale,
            expected_source_revision: Some(expected_source_revision),
            knowledge_path,
            citations,
        },
    )
    .await
}

/// Page-growth-only update path. Unlike the generic update helper, the CAS
/// token comes from the pre-inference match and is not refreshed after the
/// model returns. The memory receipt shares the DB transaction with the Page
/// write, so neither can claim a stale inference landed.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn update_page_growth_at_versions(
    db: &MemoryDB,
    page_id: &str,
    req: UpdatePageRequest,
    expected_page_version: i64,
    expected_source_revision: i64,
    source_id: &str,
    expected_memory_version: i64,
    knowledge_path: Option<&Path>,
    citations: Option<(String, String)>,
) -> Result<WriteResult, WenlanError> {
    update_page_impl(
        db,
        page_id,
        req,
        "page_growth",
        false,
        None,
        knowledge_path,
        citations,
        Some(PageGrowthCommit {
            source_id,
            expected_memory_version,
            expected_page_version,
            expected_source_revision,
        }),
        false,
    )
    .await
}

/// Test-only seam. A test installs a `(page_id, parked, go)` handshake here and
/// `update_page_impl` uses it once for that page only, *after* deciding
/// ownership and *before* writing — i.e. in the exact window a competing edit
/// has to land in. It announces that it is parked, then blocks until released,
/// so the test can land a full competing write in between with no ordering
/// guesswork. Binding the seam to a page keeps unrelated parallel tests from
/// consuming it.
///
/// This is the only way to deterministically exercise the version CAS: with no
/// interleaving edit, a guarded write and an unguarded one behave identically.
///
/// One-shot (`take`), so the retry attempt after a CAS miss runs unblocked.
/// Compiled out entirely in non-test builds.
#[cfg(test)]
type PreWriteGate = (
    String,
    tokio::sync::oneshot::Sender<()>,
    tokio::sync::oneshot::Receiver<()>,
);

#[cfg(test)]
pub(crate) static PRE_WRITE_GATE: std::sync::Mutex<Option<PreWriteGate>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
async fn pre_write_pause(page_id: &str) {
    let gate = {
        let mut slot = PRE_WRITE_GATE.lock().unwrap();
        if slot
            .as_ref()
            .is_some_and(|(target, _, _)| target == page_id)
        {
            slot.take()
        } else {
            None
        }
    };
    if let Some((_, parked, go)) = gate {
        let _ = parked.send(());
        let _ = go.await;
    }
}

#[cfg(not(test))]
#[inline(always)]
async fn pre_write_pause(_page_id: &str) {}

#[allow(clippy::too_many_arguments)]
/// The advisory line a successful page update returns. Shared between the
/// receipt and the return value so a replay hands back exactly what the
/// original call did, rather than a lookalike rebuilt at replay time.
fn write_warnings(delta_summary: &Option<String>, from: i64, to: i64) -> Vec<String> {
    match delta_summary {
        Some(summary) => vec![format!("v{from} → v{to}: {summary}")],
        None => vec![],
    }
}

/// Fingerprint of a page write, used to tell an honest retry (same request,
/// replay the stored response) from an operation id being reused for a
/// different write (a conflict).
///
/// Covers everything that decides what the write does: which page, the body,
/// the sources, who is writing, and the version precondition. Field lengths
/// are hashed alongside the values so two different requests cannot collide by
/// shifting a boundary — `["ab","c"]` and `["a","bc"]` must not agree.
fn page_write_digest(
    page_id: &str,
    req: &UpdatePageRequest,
    edited_by: &str,
    preserve_sources: bool,
) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut field = |bytes: &[u8]| {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    field(page_id.as_bytes());
    field(req.content.as_bytes());
    field(edited_by.as_bytes());
    if preserve_sources {
        // Server-derived state is intentionally excluded: an honest retry is
        // the same caller request even if another writer attached a source
        // after the first response was lost.
        field(b"preserve-sources");
    } else {
        field(b"replace-sources");
        field(&(req.source_memory_ids.len() as u64).to_le_bytes());
        for sid in &req.source_memory_ids {
            field(sid.as_bytes());
        }
    }
    match req.expected_version {
        Some(v) => {
            field(b"v");
            field(&v.to_le_bytes());
        }
        None => field(b"-"),
    }
    format!("{:x}", hasher.finalize())
}

type RetryIdentity = (String, String, String);

/// A transaction-coupled receipt insert can lose a race only after its domain
/// mutation was rolled back. In that case the winning transaction is the
/// authoritative response for an identical digest; a different digest remains
/// an operation-id conflict.
async fn replay_matching_operation_receipt(
    db: &MemoryDB,
    retry: &RetryIdentity,
    original_error: WenlanError,
) -> Result<WriteResult, WenlanError> {
    let (caller, operation, digest) = retry;
    let Some(stored) = db.get_operation_receipt(caller, operation).await? else {
        return Err(original_error);
    };
    if stored.request_digest != *digest {
        return Err(WenlanError::Conflict(format!(
            "operation id '{operation}' was already used by '{caller}' for a \
             different page write"
        )));
    }
    serde_json::from_str::<WriteResult>(&stored.response).map_err(|e| {
        WenlanError::VectorDb(format!(
            "receipt for {caller}/{operation} is unreadable: {e}"
        ))
    })
}

/// Persist a successful terminal response that made no domain mutation. If an
/// identical concurrent attempt won the receipt race, replay what it stored;
/// a different digest remains an operation-id conflict.
async fn terminal_result_with_receipt(
    db: &MemoryDB,
    retry: Option<&RetryIdentity>,
    result: WriteResult,
) -> Result<WriteResult, WenlanError> {
    let Some(retry_identity @ (caller, operation, digest)) = retry else {
        return Ok(result);
    };
    let response = serde_json::to_string(&result)?;
    let receipt = crate::db::OperationReceipt {
        caller_id: caller,
        operation_id: operation,
        request_digest: digest,
        response: &response,
    };
    match db.record_operation_receipt(receipt).await {
        Ok(()) => Ok(result),
        Err(error @ WenlanError::Conflict(_)) => {
            replay_matching_operation_receipt(db, retry_identity, error).await
        }
        Err(error) => Err(error),
    }
}

#[allow(clippy::too_many_arguments)]
async fn update_page_impl(
    db: &MemoryDB,
    page_id: &str,
    req: UpdatePageRequest,
    edited_by: &str,
    require_stale: bool,
    expected_source_revision: Option<i64>,
    knowledge_path: Option<&Path>,
    citations: Option<(String, String)>,
    page_growth: Option<PageGrowthCommit<'_>>,
    preserve_sources: bool,
) -> Result<WriteResult, WenlanError> {
    // ── Pre-write validation ────────────────────────────────────────────────
    if req.content.trim().is_empty() {
        return Err(WenlanError::Validation(
            "page content must not be empty".into(),
        ));
    }
    // Preserve mode derives provenance from the Page generation inside the
    // CAS. It is intentionally human-only: a machine writer must declare the
    // source set its output was computed from.
    let writer = Writer::classify(edited_by);
    if preserve_sources && writer.is_machine() {
        return Err(WenlanError::Validation(
            "only human Page writes may preserve server-owned sources".into(),
        ));
    }
    // A machine replacement write must carry provenance. A human write need not: an
    // authored page is legitimately born with zero sources (create_page
    // allows exactly that), so demanding one here would reject every later
    // human edit of that page — in the app and in the vault alike.
    if !preserve_sources && req.source_memory_ids.is_empty() && writer.is_machine() {
        return Err(WenlanError::Validation(
            "page must cite at least one source memory".into(),
        ));
    }
    // Source-existence check removed. create_page validates sources at
    // creation time. Updates only carry forward or extend an already-valid
    // source list; re-checking on every update would break daemon-internal
    // callers (fs_edit, re_distill) whose sources may reference pruned
    // memories.

    // ── Conditional hallucination guard ────────────────────────────────────
    if !preserve_sources && !writer.skips_hallucination_guard() {
        let passed =
            crate::kg_quality::hallucination_guard(db, &req.content, &req.source_memory_ids)
                .await?;
        if !passed {
            return Err(WenlanError::Validation(
                "page body diverges from cited sources (cos sim < 0.6)".into(),
            ));
        }
    }

    // ── Retry identity ──────────────────────────────────────────────────────
    // A caller that sends both ids gets exactly-once semantics. The same pair
    // with the same request replays the recorded response without writing
    // again; the same pair with a different request is refused rather than
    // quietly becoming a second version. Either id alone is ignored — an
    // operation id only means anything within the caller that minted it.
    let retry = match (req.caller_id.as_deref(), req.operation_id.as_deref()) {
        (Some(caller), Some(operation)) if !caller.is_empty() && !operation.is_empty() => Some((
            caller.to_string(),
            operation.to_string(),
            page_write_digest(page_id, &req, edited_by, preserve_sources),
        )),
        _ => None,
    };
    if let Some((caller, operation, digest)) = retry.as_ref() {
        if let Some(stored) = db.get_operation_receipt(caller, operation).await? {
            if stored.request_digest != *digest {
                return Err(WenlanError::Conflict(format!(
                    "operation id '{operation}' was already used by '{caller}' for a \
                     different page write"
                )));
            }
            log::debug!("[update_page] {page_id}: replaying receipt for {caller}/{operation}");
            return serde_json::from_str::<WriteResult>(&stored.response).map_err(|e| {
                WenlanError::VectorDb(format!(
                    "receipt for {caller}/{operation} is unreadable: {e}"
                ))
            });
        }
    }

    // Preserve receipt replay/collision precedence, then reject daemon-owned
    // projection delimiters before ownership gating, revision-card staging, or
    // any canonical page/projection mutation.
    crate::export::provenance::validate_canonical_page_content(&req.content)?;

    let projection = knowledge_path.map(|path| {
        crate::export::knowledge::KnowledgeProjectionWrite::new(path.to_path_buf(), db)
    });

    // ── Load, decide ownership, and write under one version CAS ─────────────
    // The ownership decision is made from a loaded row, and the write CASes on
    // *that row's* version — so the row we decided from is provably the row we
    // wrote. An edit landing in the gap fails the CAS instead of being
    // clobbered, and we reload and re-decide rather than forcing the write
    // through: a page that became human-owned in the gap gets a revision card.
    //
    // Bounded because each retry only re-runs on a version that actually moved;
    // a caller losing three races in a row is a write storm, not a lost update,
    // and yielding is the safe answer.
    const MAX_CAS_ATTEMPTS: usize = 3;
    let no_op = |outcome: WriteOutcome, warnings: Vec<String>| WriteResult {
        id: page_id.to_string(),
        attached_to: None,
        warnings,
        wrote: false,
        revision_card_id: None,
        gated: false,
        outcome,
        acknowledged: false,
    };

    let (delta_summary, current_version, new_version) = 'cas: {
        for attempt in 1..=MAX_CAS_ATTEMPTS {
            let current = db.get_page(page_id).await?.ok_or_else(|| {
                WenlanError::Validation(format!("page '{page_id}' does not exist"))
            })?;
            let effective_sources = if preserve_sources {
                &current.source_memory_ids
            } else {
                &req.source_memory_ids
            };
            if preserve_sources && !writer.skips_hallucination_guard() {
                let passed =
                    crate::kg_quality::hallucination_guard(db, &req.content, effective_sources)
                        .await?;
                if !passed {
                    return Err(WenlanError::Validation(
                        "page body diverges from cited sources (cos sim < 0.6)".into(),
                    ));
                }
            }
            let source_refs: Vec<&str> = effective_sources.iter().map(|s| s.as_str()).collect();

            // Page Growth is computed against one exact machine-owned Page
            // generation. It must never retarget the inference to a newer
            // version or stage a card onto a Page that became human-owned.
            if let Some(guard) = page_growth {
                if current.version != guard.expected_page_version {
                    return Ok(no_op(WriteOutcome::Refused, vec![]));
                }
                if page_is_human_owned(&current) {
                    return terminal_result_with_receipt(
                        db,
                        retry.as_ref(),
                        no_op(WriteOutcome::Unchanged, vec![]),
                    )
                    .await;
                }
            }

            // `require_stale` asks "write only if this page is stale". A page
            // that is not stale is the answer to that question, not a lost
            // write — and it must short-circuit *here*, before the unchanged
            // early-return below, which would otherwise acknowledge a compile
            // against a page that was never stale to begin with.
            if require_stale && current.stale_reason.is_none() {
                return terminal_result_with_receipt(
                    db,
                    retry.as_ref(),
                    no_op(WriteOutcome::Unchanged, vec![]),
                )
                .await;
            }

            let current_version = current.version;
            let new_version = current_version + 1;

            // A caller-supplied `expected_version` is a precondition, not a retry
            // hint: once it stops matching, the write is refused outright rather
            // than re-aimed at a row the caller never saw.
            //
            // This has to come BEFORE the ownership gate, not after. Staging a
            // card first would take content the agent computed against an old
            // base and bind it to the version we just loaded — so accepting that
            // card silently reverts whatever the human wrote in between. Refusing
            // first drops no agent work either: the caller re-reads the fresh
            // content and stages a better card against it.
            if let Some(expected) = req.expected_version {
                if expected != current_version {
                    log::debug!(
                        "[update_page] {page_id}: expected_version {expected} != current {current_version}; refusing write"
                    );
                    return Ok(no_op(
                        WriteOutcome::Refused,
                        vec![format!(
                            "page moved to v{current_version} (expected v{expected}); write refused"
                        )],
                    ));
                }
            }

            // Ownership gate, re-evaluated on every attempt. Inside the CAS loop
            // it is no longer advisory: whatever it decided is what the write
            // guards on.
            if writer.is_machine() && page_is_human_owned(&current) {
                let result = stage_page_revision_card(
                    db,
                    &current,
                    &req.content,
                    effective_sources,
                    edited_by,
                    retry.as_ref(),
                )
                .await?;
                // A gated compile still consumed the staleness it was dispatched
                // for: the work landed as a revision card awaiting review, so the
                // page must not be re-compiled on the next sweep. Clearing at the
                // source revision keeps that safe — a source that moved since
                // dispatch leaves the page stale.
                if require_stale {
                    let _ = db
                        .clear_page_staleness_at_source_revision(
                            page_id,
                            current.version,
                            expected_source_revision,
                        )
                        .await?;
                }
                return Ok(result);
            }

            // Shrink-guard (T17): opt-in via WENLAN_MERGE_SHRINK_GUARD=<f64>.
            // OFF by default: unset/unparseable = None = zero regression.
            // Only fires for LLM-rewrite edited_by; human edits are never blocked.
            // Placed AFTER current page load (needs old body), BEFORE early-return.
            // NOT inside the skips_hallucination_guard block: that skips page_growth/re_distill.
            if writer.is_llm_rewrite() {
                if let Some(threshold) = merge_shrink_threshold() {
                    if !crate::retrieval::integrity::body_shrink_ok(
                        &current.content,
                        &req.content,
                        threshold,
                    ) {
                        log::warn!(
                            "[update_page] shrink-guard rejected {edited_by} on {page_id}: new body ({} chars) < {}% of old ({} chars)",
                            req.content.chars().count(),
                            (threshold * 100.0) as u32,
                            current.content.chars().count(),
                        );
                        return Err(WenlanError::Validation(format!(
                            "page body shrank below {:.0}% of original (shrink-guard); update rejected",
                            threshold * 100.0
                        )));
                    }
                }
            }

            // ── Build changelog entry ───────────────────────────────────────
            let delta_summary = crate::db::compute_page_delta_summary(
                &current.content,
                &current.source_memory_ids,
                &req.content,
                &source_refs,
                edited_by,
            );

            // Compute added sources for the changelog entry
            let old_set: std::collections::HashSet<&str> = current
                .source_memory_ids
                .iter()
                .map(|s| s.as_str())
                .collect();
            let new_set: std::collections::HashSet<&str> = source_refs.iter().copied().collect();

            // Early return: identical content and identical source set — nothing to write.
            // A stale page that recompiles to the byte-identical body still had its
            // compile done, so acknowledge it; otherwise the sweep re-dispatches the
            // same no-op work forever.
            if delta_summary.is_none() && old_set == new_set {
                let acknowledged = if require_stale {
                    if let Some((caller, operation, digest)) = retry.as_ref() {
                        let acknowledged_result = WriteResult {
                            acknowledged: true,
                            ..no_op(WriteOutcome::Unchanged, vec![])
                        };
                        let response = serde_json::to_string(&acknowledged_result)?;
                        let acknowledged = db
                            .acknowledge_page_compile_with_receipt(
                                page_id,
                                current_version,
                                expected_source_revision,
                                crate::db::OperationReceipt {
                                    caller_id: caller,
                                    operation_id: operation,
                                    request_digest: digest,
                                    response: &response,
                                },
                            )
                            .await?;
                        if acknowledged {
                            return Ok(acknowledged_result);
                        }
                        false
                    } else {
                        db.acknowledge_page_compile(
                            page_id,
                            current_version,
                            expected_source_revision,
                        )
                        .await?
                    }
                } else {
                    false
                };
                return terminal_result_with_receipt(
                    db,
                    retry.as_ref(),
                    WriteResult {
                        acknowledged,
                        ..no_op(WriteOutcome::Unchanged, vec![])
                    },
                )
                .await;
            }

            let mut added_sources: Vec<&str> = new_set.difference(&old_set).copied().collect();
            added_sources.sort_unstable();
            let added_sources_json = serde_json::Value::Array(
                added_sources
                    .iter()
                    .map(|s| serde_json::Value::String(s.to_string()))
                    .collect(),
            );

            let mut entry = serde_json::json!({
                "version": new_version,
                "at": chrono::Utc::now().timestamp(),
                "edited_by": edited_by,
                "delta_summary": delta_summary,
                "incoming_source_ids": added_sources_json,
            });
            if let Some((_, ref stats_summary)) = citations {
                entry["citations_summary"] = serde_json::Value::String(stats_summary.clone());
            }

            // Read existing changelog and append the new entry
            let existing_cl = db.get_page_changelog(page_id).await?;
            const DEFAULT_CHANGELOG_CAP: usize = 20;
            let new_changelog =
                crate::db::append_changelog_entry(&existing_cl, entry, DEFAULT_CHANGELOG_CAP)?;

            // ── Apply DB update ─────────────────────────────────────────────
            // The receipt records the response this call is about to return,
            // so a replay hands back the identical envelope rather than a
            // reconstruction. It commits inside the write's own transaction.
            let receipt_response = match retry {
                Some(_) => Some(serde_json::to_string(&WriteResult {
                    id: page_id.to_string(),
                    attached_to: None,
                    warnings: write_warnings(&delta_summary, current_version, new_version),
                    wrote: true,
                    revision_card_id: None,
                    gated: false,
                    outcome: WriteOutcome::Wrote,
                    acknowledged: false,
                })?),
                None => None,
            };
            let receipt = match (retry.as_ref(), receipt_response.as_deref()) {
                (Some((caller, operation, digest)), Some(response)) => {
                    Some(crate::db::OperationReceipt {
                        caller_id: caller,
                        operation_id: operation,
                        request_digest: digest,
                        response,
                    })
                }
                _ => None,
            };

            pre_write_pause(page_id).await;
            // citations: None -> resets `citations` to SQL NULL (no fresh
            // citation source for this write; a stale claim-map must not
            // survive a content change, and the new body re-enters bounded
            // annotation).
            // The two CAS modes are mutually exclusive — the inner write refuses a
            // call carrying both. A source-revision caller is guarded by the
            // revision it compiled against, so it passes no expected_version.
            let wrote = if let Some(guard) = page_growth {
                db.try_update_page_growth_at_versions(
                    page_id,
                    &req.content,
                    &source_refs,
                    &new_changelog,
                    citations.as_ref().map(|(json, _)| json.as_str()),
                    guard.expected_page_version,
                    guard.expected_source_revision,
                    guard.source_id,
                    guard.expected_memory_version,
                )
                .await?
            } else if let Some(source_revision) = expected_source_revision {
                db.try_update_page_content_with_changelog_at_source_revision(
                    page_id,
                    &req.content,
                    &source_refs,
                    edited_by,
                    require_stale,
                    &new_changelog,
                    citations.as_ref().map(|(json, _)| json.as_str()),
                    source_revision,
                    receipt,
                )
                .await?
            } else if require_stale {
                db.try_update_page_content_with_changelog(
                    page_id,
                    &req.content,
                    &source_refs,
                    edited_by,
                    true,
                    &new_changelog,
                    citations.as_ref().map(|(json, _)| json.as_str()),
                    Some(current_version),
                    receipt,
                )
                .await?
            } else {
                db.try_update_page_content_with_changelog_at_version(
                    page_id,
                    &req.content,
                    &source_refs,
                    edited_by,
                    &new_changelog,
                    citations.as_ref().map(|(json, _)| json.as_str()),
                    current_version,
                    receipt,
                )
                .await?
            };

            if wrote {
                break 'cas (delta_summary, current_version, new_version);
            }

            // Nothing was written. Two distinguishable causes share this branch:
            // the `require_stale` gate (row untouched) and a version conflict
            // (row moved under us). Only the latter is worth retrying.
            //
            // Losing every CAS attempt used to return a bare no-op — byte-identical
            // to the "content is already what you asked for" return above. A caller
            // whose work was thrown away under contention could not tell that apart
            // from having had nothing to do, so it now says so.
            //
            // The `require_stale` skip stays silent on purpose: the caller asked to
            // write only if the page was stale, and it wasn't. That is the answer to
            // the question it asked, not a discarded write, and it is the common
            // path on every re-distill sweep.
            let landed_version = db.get_page(page_id).await?.map(|p| p.version);
            if landed_version != Some(current_version) {
                if attempt < MAX_CAS_ATTEMPTS {
                    log::debug!(
                        "[update_page] {page_id}: version moved {current_version} -> {landed_version:?} mid-write; reloading (attempt {attempt})"
                    );
                    continue;
                }
                log::warn!(
                    "[update_page] {page_id}: gave up after {MAX_CAS_ATTEMPTS} attempts; page still moving"
                );
                return Ok(no_op(
                    WriteOutcome::Contended,
                    vec![format!(
                        "page kept moving under this write ({MAX_CAS_ATTEMPTS} attempts); nothing was written"
                    )],
                ));
            }
            return terminal_result_with_receipt(
                db,
                retry.as_ref(),
                no_op(WriteOutcome::Unchanged, vec![]),
            )
            .await;
        }
        // Unreachable: every path through the loop returns, continues, or breaks
        // with a value. The compiler cannot prove that, so the block needs a
        // tail value.
        return terminal_result_with_receipt(
            db,
            retry.as_ref(),
            no_op(WriteOutcome::Unchanged, vec![]),
        )
        .await;
    };

    // ── md re-write ─────────────────────────────────────────────────────────
    if let Some(ref projection) = projection {
        if let Ok(Some(updated_page)) = db.get_page(page_id).await {
            if let Err(e) = projection.write_page_gated(db, &updated_page).await {
                log::warn!("[update_page] md re-write failed for {page_id}: {e}");
            }
        }
    }
    drop(projection);

    Ok(WriteResult {
        id: page_id.to_string(),
        attached_to: None,
        warnings: write_warnings(&delta_summary, current_version, new_version),
        wrote: true,
        revision_card_id: None,
        gated: false,
        outcome: WriteOutcome::Wrote,
        acknowledged: false,
    })
}

#[cfg(test)]
#[path = "post_write/post_write_tests.rs"]
mod tests;
