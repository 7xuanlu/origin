use super::{
    page_create::{
        attach_page_sources_impl, create_page_impl, replace_source_page_impl,
        write_document_source_page_impl, CreatePageInput,
    },
    update_page_impl, WriteResult,
};
use crate::{db::MemoryDB, error::WenlanError};
use std::path::Path;
use wenlan_types::requests::{CreateConceptRequest, UpdatePageRequest};

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
pub(super) struct PageGrowthCommit<'a> {
    pub(super) source_id: &'a str,
    pub(super) expected_memory_version: i64,
    pub(super) expected_page_version: i64,
    pub(super) expected_source_revision: i64,
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
