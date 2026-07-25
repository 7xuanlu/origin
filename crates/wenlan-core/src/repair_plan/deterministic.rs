// SPDX-License-Identifier: Apache-2.0
use crate::{
    error::WenlanError,
    lint::{
        pages::{
            fs::{scan_page_root_controlled, PageScanControl},
            state_checks::{
                normalize_id, projection_target_is_exclusive_page_markdown,
                projection_version_mismatches, stale_projection_ownership, DbPage,
                StaleProjectionOwnership, IDENTITY_ID,
            },
        },
        snapshot::LintReadSnapshot,
    },
    repair::{
        capture_page_projection_rollback, capture_rollback, capture_stale_page_projection_rollback,
        StoredRollbackArtifact,
    },
};
use std::{collections::BTreeMap, path::Path, time::Duration};
use wenlan_types::{
    repair::{
        RepairAllowedEffects, RepairLintScope, RepairMutation, RepairScope, RepairTarget,
        RepairWriter,
    },
    repair_plan::{RepairAffectedRecord, RepairAffectedRecordKind},
};

pub(crate) const MEMORY_STATE: &str = "identity.memory_state_integrity";
pub(crate) const TAG_INTEGRITY: &str = "identity.tag_integrity";
pub(crate) const SUPERSESSION: &str = "memories.supersession_integrity";
pub(crate) const ENRICHMENT_FAILURES: &str = "memories.enrichment_failures";
pub(crate) const MEMORY_ENTITY_INTEGRITY: &str = "memory_entities.integrity";
pub(crate) const DUPLICATE_TITLES: &str = "pages.duplicate_active_titles";
pub(crate) const ORPHAN_LABELS: &str = "pages.links.orphan_labels";
pub(crate) const PROJECTION_VERSION: &str = "pages.projection.version_alignment";
pub(crate) const PROJECTION_IDENTITY: &str = "pages.projection.identity";
pub(crate) const SOURCE_PAGE_INTEGRITY: &str = "pages.source_page_integrity";
const LEGACY_ROLLBACK_FORMAT_VERSION: u16 = 1;

#[derive(Debug)]
pub(crate) struct ExactDeterministicRepair {
    pub(crate) source_check_ids: Vec<&'static str>,
    pub(crate) target: RepairTarget,
    pub(crate) expected_version: Option<i64>,
    pub(crate) writer: RepairWriter,
    pub(crate) mutation: RepairMutation,
    pub(crate) allowed_effects: RepairAllowedEffects,
    pub(crate) rollback: StoredRollbackArtifact,
}

#[derive(Debug)]
pub(crate) struct DeterministicReview {
    pub(crate) check_id: &'static str,
    pub(crate) affected_record: RepairAffectedRecord,
    pub(crate) issue: String,
    pub(crate) choices: Vec<String>,
    pub(crate) suggested_research_queries: Vec<String>,
}

#[derive(Debug)]
pub(crate) struct DeterministicBlocked {
    pub(crate) check_id: &'static str,
    pub(crate) affected_record: RepairAffectedRecord,
    pub(crate) detail: String,
    pub(crate) next_action: String,
}

#[derive(Debug)]
pub(crate) enum DeterministicResolution {
    Exact(Box<ExactDeterministicRepair>),
    Review(DeterministicReview),
    Blocked(DeterministicBlocked),
}

pub(crate) async fn resolve_current(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
    page_root: Option<&Path>,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let mut resolutions = resolve_memories(snapshot, scope).await?;
    resolutions.extend(resolve_enrichment_failures(snapshot, scope).await?);
    if matches!(scope, RepairLintScope::Global) {
        resolutions.extend(resolve_tags(snapshot).await?);
    }
    resolutions.extend(resolve_memory_entity_links(snapshot, scope).await?);
    resolutions.extend(resolve_orphan_links(snapshot, scope).await?);
    resolutions.extend(resolve_duplicate_page_titles(snapshot, scope).await?);
    resolutions.extend(resolve_source_pages(snapshot, scope).await?);
    if let Some(page_root) = page_root {
        resolutions.extend(resolve_page_projections(snapshot, scope, page_root).await?);
    }
    Ok(resolutions)
}

async fn resolve_duplicate_page_titles(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let (scope_clause, params) = page_scope_clause(scope);
    let sql = format!(
        "SELECT p.id,p.title,COUNT(collision.id)
           FROM pages p
           JOIN pages collision
             ON collision.status='active'
            AND LOWER(collision.title)=LOWER(p.title)
            AND collision.space=p.space
          WHERE p.status='active'{scope_clause}
          GROUP BY p.id,p.title,p.space
         HAVING COUNT(collision.id)>1
          ORDER BY p.id"
    );
    let mut rows = snapshot.query(&sql, params).await.map_err(|error| {
        WenlanError::VectorDb(format!("repair resolve duplicate page titles: {error}"))
    })?;
    let mut resolutions = Vec::new();
    while let Some(row) = rows.next().await.map_err(|error| {
        WenlanError::VectorDb(format!("repair resolve duplicate page titles: {error}"))
    })? {
        let page_id = row.get::<String>(0).map_err(database_error)?;
        let title = row.get::<String>(1).map_err(database_error)?;
        let duplicate_count = row.get::<i64>(2).map_err(database_error)?;
        resolutions.push(DeterministicResolution::Review(DeterministicReview {
            check_id: DUPLICATE_TITLES,
            affected_record: RepairAffectedRecord::try_new(
                RepairAffectedRecordKind::Page,
                page_id.clone(),
            )
            .map_err(contract_error)?,
            issue: format!(
                "active Page {page_id} uses duplicate same-scope title {title:?} ({duplicate_count} Pages)"
            ),
            choices: vec![
                "rename this Page to a unique title".to_string(),
                "consolidate or archive the duplicate Page".to_string(),
                "defer until the intended canonical Page is known".to_string(),
            ],
            suggested_research_queries: vec![format!(
                "find sources and links that identify the canonical Page for title {title}"
            )],
        }));
    }
    Ok(resolutions)
}

async fn resolve_enrichment_failures(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let (scope_clause, params) = scoped_memory_clause(scope);
    let sql = format!(
        "SELECT m.source_id,s.step_name,s.status,COALESCE(s.error,'')
           FROM memories m
           JOIN enrichment_steps s ON s.source_id=m.source_id
          WHERE m.source='memory' AND m.chunk_index=0
            AND s.status IN ('failed','abandoned'){scope_clause}
          ORDER BY m.source_id,s.step_name"
    );
    let mut rows = snapshot.query(&sql, params).await.map_err(|error| {
        WenlanError::VectorDb(format!("repair resolve enrichment failures: {error}"))
    })?;
    let mut resolutions = Vec::new();
    while let Some(row) = rows.next().await.map_err(|error| {
        WenlanError::VectorDb(format!("repair resolve enrichment failures: {error}"))
    })? {
        let memory_id = row.get::<String>(0).map_err(database_error)?;
        let step_name = row.get::<String>(1).map_err(database_error)?;
        let status = row.get::<String>(2).map_err(database_error)?;
        let error = row.get::<String>(3).map_err(database_error)?;
        let affected_record =
            RepairAffectedRecord::try_new(RepairAffectedRecordKind::Memory, memory_id.clone())
                .map_err(contract_error)?;
        if step_name == "entity_extract" && status == "failed" {
            resolutions.push(DeterministicResolution::Review(DeterministicReview {
                check_id: ENRICHMENT_FAILURES,
                affected_record,
                issue: format!(
                    "entity extraction failed for memory {memory_id}{}",
                    if error.trim().is_empty() {
                        String::new()
                    } else {
                        format!(": {error}")
                    }
                ),
                choices: vec![
                    "select existing same-scope entities and mark extraction complete".to_string(),
                    "retry entity extraction".to_string(),
                    "defer until the intended entities are known".to_string(),
                ],
                suggested_research_queries: vec![format!(
                    "identify existing entities represented by memory {memory_id}"
                )],
            }));
        } else {
            resolutions.push(DeterministicResolution::Blocked(DeterministicBlocked {
                check_id: ENRICHMENT_FAILURES,
                affected_record,
                detail: format!(
                    "enrichment step {step_name:?} for memory {memory_id} is {status}{}",
                    if error.trim().is_empty() {
                        String::new()
                    } else {
                        format!(": {error}")
                    }
                ),
                next_action: "retry or inspect this enrichment step, then rerun lint repair"
                    .to_string(),
            }));
        }
    }
    Ok(resolutions)
}

pub(super) async fn target_still_actionable(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
    page_root: Option<&Path>,
    target: &RepairTarget,
    writer: RepairWriter,
) -> Result<bool, WenlanError> {
    if writer == RepairWriter::CompleteEntityExtraction {
        return entity_extraction_target_still_actionable(snapshot, scope, target).await;
    }
    if writer == RepairWriter::RenamePageTitle {
        return renamed_page_title_still_actionable(snapshot, scope, target).await;
    }
    let resolutions = match writer {
        RepairWriter::NormalizeMemorySourceAgent
        | RepairWriter::ClearMemorySupersedes
        | RepairWriter::UnstageOrphanRevision => resolve_memories(snapshot, scope).await?,
        RepairWriter::DeleteTagRow => resolve_tags(snapshot).await?,
        RepairWriter::DeleteMemoryEntityLink => {
            resolve_memory_entity_links(snapshot, scope).await?
        }
        RepairWriter::BindPageLink => resolve_orphan_links(snapshot, scope).await?,
        RepairWriter::ArchiveEmptySourcePage => resolve_source_pages(snapshot, scope).await?,
        RepairWriter::RegeneratePageProjection => {
            let page_root = page_root.ok_or_else(|| {
                WenlanError::Validation("page projection repair root unavailable".to_string())
            })?;
            resolve_page_projections(snapshot, scope, page_root).await?
        }
        RepairWriter::QuarantineStalePageProjection => {
            let page_root = page_root.ok_or_else(|| {
                WenlanError::Validation("page projection repair root unavailable".to_string())
            })?;
            resolve_page_projections(snapshot, scope, page_root).await?
        }
        RepairWriter::CompleteEntityExtraction => {
            unreachable!("complete entity extraction actionability handled above")
        }
        RepairWriter::ReclassifyMemory => {
            return Err(WenlanError::Validation(
                "repair_target_assertion_unsupported".to_string(),
            ));
        }
        RepairWriter::RenamePageTitle => {
            unreachable!("page title actionability handled above")
        }
    };
    Ok(resolutions.into_iter().any(|resolution| {
        matches!(
            resolution,
            DeterministicResolution::Exact(exact)
                if exact.target == *target && exact.writer == writer
        )
    }))
}

async fn renamed_page_title_still_actionable(
    snapshot: &LintReadSnapshot<'_>,
    lint_scope: &RepairLintScope,
    target: &RepairTarget,
) -> Result<bool, WenlanError> {
    let RepairTarget::PageProjection {
        page_id,
        scope: target_scope,
    } = target
    else {
        return Err(WenlanError::Validation(
            "repair_target_assertion_unsupported".to_string(),
        ));
    };
    if matches!(target_scope, RepairScope::Global) {
        return Err(WenlanError::Validation(
            "repair_target_assertion_unsupported".to_string(),
        ));
    }
    let scope_matches = match lint_scope {
        RepairLintScope::Global => true,
        RepairLintScope::Registered { space } => target_scope.space() == Some(space.as_str()),
        RepairLintScope::Uncategorized => target_scope.space().is_none(),
    };
    if !scope_matches {
        return Err(WenlanError::Validation(
            "repair_target_assertion_unsupported".to_string(),
        ));
    }
    let mut rows = snapshot
        .query(
            "SELECT EXISTS(
                 SELECT 1 FROM pages target
                  JOIN pages collision
                    ON collision.status='active'
                   AND collision.id<>target.id
                   AND collision.space=COALESCE(?2,'00000000-0000-4000-8000-000000000001')
                   AND lower(collision.title)=lower(target.title)
                 WHERE target.id=?1 AND target.status='active'
                   AND target.space=COALESCE(?2,'00000000-0000-4000-8000-000000000001'))",
            libsql::params::Params::Positional(vec![
                libsql::Value::Text(page_id.clone()),
                target_scope
                    .space()
                    .map(|space| libsql::Value::Text(space.to_string()))
                    .unwrap_or(libsql::Value::Null),
            ]),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve page title: {error}")))?;
    let row = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve page title: {error}")))?
        .ok_or_else(|| {
            WenlanError::Validation("repair_target_assertion_unsupported".to_string())
        })?;
    Ok(row
        .get::<i64>(0)
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve page title: {error}")))?
        != 0)
}

async fn entity_extraction_target_still_actionable(
    snapshot: &LintReadSnapshot<'_>,
    lint_scope: &RepairLintScope,
    target: &RepairTarget,
) -> Result<bool, WenlanError> {
    let RepairTarget::MemoryEntityExtraction {
        memory_id,
        step: wenlan_types::repair::RepairEnrichmentStep::EntityExtract,
        scope: target_scope,
        ..
    } = target
    else {
        return Err(WenlanError::Validation(
            "repair_target_assertion_unsupported".to_string(),
        ));
    };
    if matches!(target_scope, RepairScope::Global) {
        return Err(WenlanError::Validation(
            "repair_target_assertion_unsupported".to_string(),
        ));
    }
    let scope_matches = match lint_scope {
        RepairLintScope::Global => true,
        RepairLintScope::Registered { space } => target_scope.space() == Some(space.as_str()),
        RepairLintScope::Uncategorized => target_scope.space().is_none(),
    };
    if !scope_matches {
        return Err(WenlanError::Validation(
            "repair_target_assertion_unsupported".to_string(),
        ));
    }
    // memories.space is NOT NULL since migration 91 — an unscoped memory
    // carries the reserved sentinel id, which the RepairScope above translates
    // back to None. So for an unscoped target ?2 is NULL while the row's space
    // column is the sentinel: match NULL and sentinel both, else an unscoped
    // entity-extraction repair reads as non-actionable and is dropped as stale.
    let actionable_sql = format!(
        "SELECT 1
               FROM memories m
               JOIN enrichment_steps s
                 ON s.source_id=m.source_id
                AND s.step_name='entity_extract'
                AND s.status='failed'
              WHERE m.source='memory'
                AND m.source_id=?1
                AND ((?2 IS NULL AND (m.space IS NULL OR m.space = '{sentinel}')) OR m.space=?2)
              LIMIT 2",
        sentinel = crate::db::UNFILED_SPACE_ID
    );
    let mut rows = snapshot
        .query(
            &actionable_sql,
            libsql::params::Params::Positional(vec![
                libsql::Value::Text(memory_id.clone()),
                target_scope
                    .space()
                    .map(|space| libsql::Value::Text(space.to_string()))
                    .unwrap_or(libsql::Value::Null),
            ]),
        )
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("repair resolve entity extraction: {error}"))
        })?;
    let actionable = rows
        .next()
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("repair resolve entity extraction: {error}"))
        })?
        .is_some();
    if actionable
        && rows
            .next()
            .await
            .map_err(|error| {
                WenlanError::VectorDb(format!("repair resolve entity extraction: {error}"))
            })?
            .is_some()
    {
        return Err(WenlanError::Validation(
            "repair_target_assertion_unsupported".to_string(),
        ));
    }
    Ok(actionable)
}

async fn resolve_source_pages(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let (scope_clause, params) = page_scope_clause(scope);
    let sql = format!(
        "SELECT p.id,p.version,p.space,p.source_memory_ids,
                p.review_status,COALESCE(p.user_edited,0),p.content,
                EXISTS(SELECT 1 FROM page_sources ps WHERE ps.page_id=p.id),
                EXISTS(SELECT 1 FROM page_evidence pe WHERE pe.page_id=p.id)
           FROM pages p
          WHERE p.status='active' AND p.creation_kind='source'{scope_clause}
          ORDER BY p.id"
    );
    let mut rows = snapshot
        .query(&sql, params)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve source pages: {error}")))?;
    let mut candidates = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve source pages: {error}")))?
    {
        candidates.push((
            row.get::<String>(0).map_err(database_error)?,
            row.get::<i64>(1).map_err(database_error)?,
            row.get::<Option<String>>(2).map_err(database_error)?,
            row.get::<String>(3).map_err(database_error)?,
            row.get::<String>(4).map_err(database_error)?,
            row.get::<i64>(5).map_err(database_error)? != 0,
            row.get::<String>(6).map_err(database_error)?,
            row.get::<i64>(7).map_err(database_error)? != 0,
            row.get::<i64>(8).map_err(database_error)? != 0,
        ));
    }
    drop(rows);

    let mut resolutions = Vec::new();
    for (
        page_id,
        version,
        page_scope,
        source_memory_ids,
        review_status,
        user_edited,
        content,
        has_page_source,
        has_page_evidence,
    ) in candidates
    {
        let parsed_source_ids = serde_json::from_str::<Vec<String>>(&source_memory_ids).ok();
        let has_provenance = parsed_source_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
            || has_page_source
            || has_page_evidence;
        if has_provenance {
            continue;
        }
        let target_scope = match page_scope {
            Some(scope) => RepairScope::registered(scope),
            None => Ok(RepairScope::uncategorized()),
        }
        .map_err(contract_error)?;
        let target = RepairTarget::page(page_id.clone(), target_scope).map_err(contract_error)?;
        let affected_record =
            RepairAffectedRecord::try_new(RepairAffectedRecordKind::Page, page_id.clone())
                .map_err(contract_error)?;
        let safe = parsed_source_ids.is_some()
            && review_status == "unconfirmed"
            && !user_edited
            && content.trim().is_empty();
        if safe {
            let rollback = capture_page_rollback(snapshot, &page_id).await?;
            resolutions.push(DeterministicResolution::Exact(Box::new(
                ExactDeterministicRepair {
                    source_check_ids: vec![SOURCE_PAGE_INTEGRITY, IDENTITY_ID],
                    target: target.clone(),
                    expected_version: Some(version),
                    writer: RepairWriter::ArchiveEmptySourcePage,
                    mutation: RepairMutation::archive_empty_source_page(),
                    allowed_effects: RepairAllowedEffects::page_status(target),
                    rollback,
                },
            )));
        } else {
            resolutions.push(DeterministicResolution::Review(DeterministicReview {
                check_id: SOURCE_PAGE_INTEGRITY,
                affected_record,
                issue: "source Page has no provenance but is not safe for automatic archival"
                    .to_string(),
                choices: vec![
                    "attach the missing source provenance".to_string(),
                    "review and archive the Page manually".to_string(),
                ],
                suggested_research_queries: vec![format!(
                    "find source evidence for Page {page_id}"
                )],
            }));
        }
    }
    Ok(resolutions)
}

async fn capture_page_rollback(
    snapshot: &LintReadSnapshot<'_>,
    page_id: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let mut column_rows = snapshot
        .query("PRAGMA table_info(pages)", libsql::params::Params::None)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair page schema: {error}")))?;
    let mut columns = Vec::new();
    while let Some(row) = column_rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair page schema: {error}")))?
    {
        columns.push(row.get::<String>(1).map_err(database_error)?);
    }
    drop(column_rows);
    if columns.is_empty() {
        return Err(WenlanError::Validation(
            "repair_target_schema_missing".to_string(),
        ));
    }
    let selected = columns
        .iter()
        .map(|column| format!("quote(\"{}\")", column.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(",");
    let mut rows = snapshot
        .query(
            &format!("SELECT {selected} FROM pages WHERE id=?1"),
            libsql::params::Params::Positional(vec![libsql::Value::Text(page_id.to_string())]),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair page rollback: {error}")))?;
    let row = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair page rollback: {error}")))?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let mut values = Vec::with_capacity(columns.len());
    for index in 0..columns.len() {
        values.push(
            row.get::<String>(i32::try_from(index).map_err(|_| {
                WenlanError::Validation("repair_target_schema_too_wide".to_string())
            })?)
            .map_err(database_error)?,
        );
    }
    Ok(StoredRollbackArtifact {
        format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
        table: "pages".to_string(),
        source_id: page_id.to_string(),
        columns,
        rows: vec![values],
    })
}

async fn resolve_page_projections(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
    page_root: &Path,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let scan = scan_page_root_controlled(
        page_root,
        true,
        &PageScanControl::with_timeout(Duration::from_secs(30)),
    )
    .map_err(|error| WenlanError::Validation(format!("repair projection scan: {error}")))?;
    let (sql, params) = match scope {
        RepairLintScope::Global => (
            "SELECT id,status,version,space FROM pages ORDER BY id",
            libsql::params::Params::None,
        ),
        RepairLintScope::Registered { space } => (
            "SELECT id,status,version,space FROM pages WHERE space=?1 ORDER BY id",
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        RepairLintScope::Uncategorized => (
            "SELECT id,status,version,space FROM pages WHERE space='00000000-0000-4000-8000-000000000001' ORDER BY id",
            libsql::params::Params::None,
        ),
    };
    let mut rows = snapshot
        .query(sql, params)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve projections: {error}")))?;
    let mut pages = Vec::new();
    let mut scopes = BTreeMap::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve projections: {error}")))?
    {
        let id = row.get::<String>(0).map_err(database_error)?;
        pages.push(DbPage {
            id: id.clone(),
            status: row.get::<String>(1).map_err(database_error)?,
            version: row.get::<i64>(2).map_err(database_error)?,
        });
        scopes.insert(id, row.get::<Option<String>>(3).map_err(database_error)?);
    }
    drop(rows);

    let mut resolutions = Vec::new();
    if matches!(scope, RepairLintScope::Global) {
        let db_ids = pages
            .iter()
            .map(|page| page.id.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let stale_ids = scan
            .raw_state
            .edges
            .iter()
            .map(|edge| normalize_id(&edge.state_id))
            .filter(|page_id| !db_ids.contains(page_id.as_str()))
            .collect::<std::collections::BTreeSet<_>>();
        for page_id in stale_ids {
            let affected_record =
                RepairAffectedRecord::try_new(RepairAffectedRecordKind::Page, page_id.clone())
                    .map_err(contract_error)?;
            match stale_projection_ownership(&scan, &page_id) {
                StaleProjectionOwnership::Exact {
                    source_path,
                    quarantine_path,
                } => {
                    let target =
                        RepairTarget::page_projection(page_id.clone(), RepairScope::global())
                            .map_err(contract_error)?;
                    match capture_stale_page_projection_rollback(
                        snapshot,
                        page_root,
                        &page_id,
                        &source_path,
                        &quarantine_path,
                    )
                    .await
                    {
                        Ok(rollback) => resolutions.push(DeterministicResolution::Exact(
                            Box::new(ExactDeterministicRepair {
                                source_check_ids: vec![PROJECTION_IDENTITY],
                                target: target.clone(),
                                expected_version: None,
                                writer: RepairWriter::QuarantineStalePageProjection,
                                mutation: RepairMutation::quarantine_stale_page_projection(
                                    source_path,
                                    quarantine_path,
                                )
                                .map_err(contract_error)?,
                                allowed_effects:
                                    RepairAllowedEffects::page_projection_quarantine(target),
                                rollback,
                            }),
                        )),
                        Err(error) => {
                            resolutions.push(DeterministicResolution::Blocked(
                                DeterministicBlocked {
                                    check_id: PROJECTION_IDENTITY,
                                    affected_record,
                                    detail: format!(
                                        "stale Page projection cannot be safely snapshotted: {error}"
                                    ),
                                    next_action:
                                        "repair the projection ownership/path prerequisite and rerun lint"
                                            .to_string(),
                                },
                            ))
                        }
                    }
                }
                StaleProjectionOwnership::ReviewWrongOrigin => {
                    resolutions.push(DeterministicResolution::Review(DeterministicReview {
                        check_id: PROJECTION_IDENTITY,
                        affected_record,
                        issue: "stale Page projection frontmatter does not prove ownership"
                            .to_string(),
                        choices: vec![
                            "confirm the file belongs to the missing Page and correct origin_id"
                                .to_string(),
                            "leave the file as a user-owned note".to_string(),
                        ],
                        suggested_research_queries: vec![format!(
                            "find provenance for stale Page projection {page_id}"
                        )],
                    }));
                }
                StaleProjectionOwnership::Blocked(detail) => {
                    resolutions.push(DeterministicResolution::Blocked(DeterministicBlocked {
                        check_id: PROJECTION_IDENTITY,
                        affected_record,
                        detail: detail.to_string(),
                        next_action: "repair the projection state/path prerequisite and rerun lint"
                            .to_string(),
                    }));
                }
            }
        }
    }
    for mismatch in
        projection_version_mismatches(&scan, &pages, !matches!(scope, RepairLintScope::Global))
    {
        let affected_record =
            RepairAffectedRecord::try_new(RepairAffectedRecordKind::Page, mismatch.page_id.clone())
                .map_err(contract_error)?;
        if !projection_target_is_exclusive_page_markdown(
            &scan,
            &mismatch.page_id,
            &mismatch.target_path,
        ) {
            resolutions.push(DeterministicResolution::Blocked(DeterministicBlocked {
                check_id: PROJECTION_VERSION,
                affected_record,
                detail: "Page projection target is reserved, shared, aliased, or not owned by the named Page"
                    .to_string(),
                next_action:
                    "repair the Page state/identity target and rerun lint before regenerating"
                        .to_string(),
            }));
            continue;
        }
        let target_scope = match scopes.get(&mismatch.page_id).cloned().flatten() {
            Some(scope) => RepairScope::registered(scope),
            None => Ok(RepairScope::uncategorized()),
        }
        .map_err(contract_error)?;
        let target = RepairTarget::page_projection(mismatch.page_id.clone(), target_scope)
            .map_err(contract_error)?;
        match capture_page_projection_rollback(snapshot, page_root, &mismatch).await {
            Ok(rollback) => resolutions.push(DeterministicResolution::Exact(Box::new(
                ExactDeterministicRepair {
                    source_check_ids: vec![PROJECTION_VERSION],
                    target: target.clone(),
                    expected_version: Some(mismatch.database_version),
                    writer: RepairWriter::RegeneratePageProjection,
                    mutation: RepairMutation::regenerate_page_projection(mismatch.database_version)
                        .map_err(contract_error)?,
                    allowed_effects: RepairAllowedEffects::page_projection(target),
                    rollback,
                },
            ))),
            Err(error) => {
                resolutions.push(DeterministicResolution::Blocked(DeterministicBlocked {
                    check_id: PROJECTION_VERSION,
                    affected_record,
                    detail: format!("Page projection cannot be safely snapshotted: {error}"),
                    next_action: "repair the projection root/path prerequisite and rerun lint"
                        .to_string(),
                }))
            }
        }
    }
    Ok(resolutions)
}

fn scoped_memory_clause(scope: &RepairLintScope) -> (&'static str, libsql::params::Params) {
    match scope {
        RepairLintScope::Global => ("", libsql::params::Params::None),
        RepairLintScope::Registered { space } => (
            " AND m.space=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        RepairLintScope::Uncategorized => (
            // M3 PR-1 stage e: memories.space is NOT NULL as of migration
            // 85, so an unscoped memory carries the reserved sentinel id
            // rather than NULL.
            " AND m.space='00000000-0000-4000-8000-000000000001'",
            libsql::params::Params::None,
        ),
    }
}

async fn resolve_memories(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let (scope_clause, params) = scoped_memory_clause(scope);
    // M3 PR-1 stage e: memories.space is NOT NULL as of migration 91, so an
    // unscoped memory carries the reserved sentinel id rather than NULL --
    // the sentinel check below (was `m.space IS NULL`) keeps space_exists
    // from false-flagging every unfiled memory as "references a missing
    // Space" (see the `!space_exists` check further down).
    let sql = format!(
        "SELECT m.source_id,m.version,m.space,m.source_agent,m.supersedes,
                m.confirmed,m.pinned,m.pending_revision,m.stability,
                CASE WHEN m.supersedes IS NULL OR EXISTS(
                    SELECT 1 FROM memories prior
                     WHERE prior.source='memory' AND prior.source_id=m.supersedes
                ) THEN 1 ELSE 0 END AS target_exists,
                CASE WHEN m.space='00000000-0000-4000-8000-000000000001' OR EXISTS(
                    SELECT 1 FROM spaces s WHERE s.name=m.space
                ) THEN 1 ELSE 0 END AS space_exists
           FROM memories m
          WHERE m.source='memory' AND m.chunk_index=0{scope_clause}
          ORDER BY m.source_id,m.id"
    );
    let mut rows = snapshot
        .query(&sql, params)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve memories: {error}")))?;
    let mut row_values = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve memories: {error}")))?
    {
        row_values.push((
            row.get::<String>(0).map_err(database_error)?,
            row.get::<Option<i64>>(1).map_err(database_error)?,
            row.get::<Option<String>>(2).map_err(database_error)?,
            row.get::<Option<String>>(3).map_err(database_error)?,
            row.get::<Option<String>>(4).map_err(database_error)?,
            row.get::<Option<i64>>(5).map_err(database_error)?,
            row.get::<Option<i64>>(6).map_err(database_error)?,
            row.get::<Option<i64>>(7).map_err(database_error)?,
            row.get::<Option<String>>(8).map_err(database_error)?,
            row.get::<i64>(9).map_err(database_error)? != 0,
            row.get::<i64>(10).map_err(database_error)? != 0,
        ));
    }
    drop(rows);

    let mut resolutions = Vec::new();
    for (
        source_id,
        version,
        space,
        source_agent,
        supersedes,
        confirmed,
        pinned,
        pending_revision,
        stability,
        target_exists,
        space_exists,
    ) in row_values
    {
        let target = RepairTarget::memory(
            source_id.clone(),
            // M3 PR-1 stage e: memories.space is NOT NULL as of migration
            // 85 -- translate the reserved sentinel id back to None so an
            // unscoped memory still resolves to RepairScope::Uncategorized
            // instead of a bogus RepairScope::Registered(sentinel).
            match space.clone().filter(|s| s != crate::db::UNFILED_SPACE_ID) {
                Some(space) => RepairScope::registered(space),
                None => Ok(RepairScope::uncategorized()),
            }
            .map_err(contract_error)?,
        )
        .map_err(contract_error)?;

        if let Some(blank) = source_agent.filter(|agent| agent.trim().is_empty()) {
            let rollback = capture_rollback(snapshot, &source_id).await?;
            let mutation =
                RepairMutation::normalize_memory_source_agent(blank).map_err(contract_error)?;
            resolutions.push(DeterministicResolution::Exact(Box::new(
                ExactDeterministicRepair {
                    source_check_ids: vec![MEMORY_STATE],
                    target: target.clone(),
                    expected_version: version,
                    writer: RepairWriter::NormalizeMemorySourceAgent,
                    mutation,
                    allowed_effects: RepairAllowedEffects::memory_source_agent(target.clone()),
                    rollback,
                },
            )));
        }

        if supersedes.as_deref() == Some(source_id.as_str()) {
            let rollback = capture_rollback(snapshot, &source_id).await?;
            let mutation = RepairMutation::clear_memory_supersedes(source_id.clone())
                .map_err(contract_error)?;
            resolutions.push(DeterministicResolution::Exact(Box::new(
                ExactDeterministicRepair {
                    source_check_ids: vec![MEMORY_STATE, SUPERSESSION],
                    target: target.clone(),
                    expected_version: version,
                    writer: RepairWriter::ClearMemorySupersedes,
                    mutation,
                    allowed_effects: RepairAllowedEffects::memory_supersedes(target.clone()),
                    rollback,
                },
            )));
        }

        let orphan_revision = pending_revision == Some(1) && supersedes.is_none();
        if orphan_revision {
            let rollback = capture_rollback(snapshot, &source_id).await?;
            resolutions.push(DeterministicResolution::Exact(Box::new(
                ExactDeterministicRepair {
                    source_check_ids: vec![MEMORY_STATE, SUPERSESSION],
                    target: target.clone(),
                    expected_version: version,
                    writer: RepairWriter::UnstageOrphanRevision,
                    mutation: RepairMutation::unstage_orphan_revision(),
                    allowed_effects: RepairAllowedEffects::memory_pending_revision(target.clone()),
                    rollback,
                },
            )));
        }

        let mut memory_issues = Vec::new();
        if confirmed.is_some_and(|value| !matches!(value, 0 | 1)) {
            memory_issues.push("confirmed is not boolean");
        }
        if !matches!(pinned, Some(0 | 1)) {
            memory_issues.push("pinned is not boolean");
        }
        if !matches!(pending_revision, Some(0 | 1)) {
            memory_issues.push("pending_revision is not boolean");
        }
        if pinned == Some(1) && confirmed != Some(1) {
            memory_issues.push("pinned memory is not confirmed");
        }
        if !matches!(stability.as_deref(), Some("new" | "learned" | "confirmed")) {
            memory_issues.push("stability is outside the canonical vocabulary");
        }
        if !space_exists {
            memory_issues.push("memory references a missing Space");
        }
        if !orphan_revision && pending_revision == Some(1) && !target_exists {
            memory_issues.push("pending revision has no valid predecessor");
        }
        if !memory_issues.is_empty() {
            resolutions.push(DeterministicResolution::Review(DeterministicReview {
                check_id: MEMORY_STATE,
                affected_record: RepairAffectedRecord::try_new(
                    RepairAffectedRecordKind::Memory,
                    source_id.clone(),
                )
                .map_err(contract_error)?,
                issue: memory_issues.join("; "),
                choices: vec![
                    "choose the lawful memory state".to_string(),
                    "dismiss until more context is available".to_string(),
                ],
                suggested_research_queries: vec![format!(
                    "find captures and revisions related to {source_id}"
                )],
            }));
        }

        if !orphan_revision && supersedes.is_some() && !target_exists {
            resolutions.push(DeterministicResolution::Review(DeterministicReview {
                check_id: SUPERSESSION,
                affected_record: RepairAffectedRecord::try_new(
                    RepairAffectedRecordKind::Memory,
                    source_id.clone(),
                )
                .map_err(contract_error)?,
                issue: "supersession target is missing or ambiguous".to_string(),
                choices: vec![
                    "select the intended predecessor".to_string(),
                    "clear the pending relationship".to_string(),
                ],
                suggested_research_queries: vec![format!(
                    "find temporal predecessors for {source_id}"
                )],
            }));
        }
    }
    Ok(resolutions)
}

fn scoped_memory_entity_clause(scope: &RepairLintScope) -> (&'static str, libsql::params::Params) {
    match scope {
        RepairLintScope::Global => ("", libsql::params::Params::None),
        RepairLintScope::Registered { space } => (
            " AND m.space=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        RepairLintScope::Uncategorized => (
            // M3 PR-1 stage e: memories.space is NOT NULL as of migration
            // 85, same sentinel treatment as scoped_memory_clause above.
            " AND m.space='00000000-0000-4000-8000-000000000001' AND m.source_id IS NOT NULL",
            libsql::params::Params::None,
        ),
    }
}

async fn resolve_memory_entity_links(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let (scope_clause, params) = scoped_memory_entity_clause(scope);
    let sql = format!(
        "SELECT me.memory_id,me.entity_id,m.source_id,m.space,e.id
           FROM memory_entities me
           LEFT JOIN (
                SELECT source_id,MAX(space) AS space
                  FROM memories GROUP BY source_id
           ) m ON m.source_id=me.memory_id
           LEFT JOIN entities e ON e.id=me.entity_id
          WHERE (m.source_id IS NULL OR e.id IS NULL){scope_clause}
          ORDER BY me.memory_id,me.entity_id"
    );
    let mut rows = snapshot.query(&sql, params).await.map_err(|error| {
        WenlanError::VectorDb(format!("repair resolve memory entity links: {error}"))
    })?;
    let mut values = Vec::new();
    while let Some(row) = rows.next().await.map_err(|error| {
        WenlanError::VectorDb(format!("repair resolve memory entity links: {error}"))
    })? {
        values.push((
            row.get::<String>(0).map_err(database_error)?,
            row.get::<String>(1).map_err(database_error)?,
            row.get::<Option<String>>(2).map_err(database_error)?,
            row.get::<Option<String>>(3).map_err(database_error)?,
        ));
    }
    drop(rows);

    values
        .into_iter()
        .map(|(memory_id, entity_id, memory_owner, memory_space)| {
            // M3 PR-1 stage e: memories.space is NOT NULL as of migration
            // 85 -- translate the reserved sentinel id back to None so an
            // unscoped owner still resolves to RepairScope::Uncategorized
            // instead of a bogus RepairScope::Registered(sentinel).
            let memory_space = memory_space.filter(|s| s != crate::db::UNFILED_SPACE_ID);
            let target_scope = match (memory_owner, memory_space) {
                (Some(_), Some(space)) => RepairScope::registered(space),
                (Some(_), None) => Ok(RepairScope::uncategorized()),
                (None, _) => Ok(RepairScope::global()),
            }
            .map_err(contract_error)?;
            let target = RepairTarget::memory_entity_link(
                memory_id.clone(),
                entity_id.clone(),
                target_scope,
            )
            .map_err(contract_error)?;
            let rollback = StoredRollbackArtifact {
                format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
                table: "memory_entities".to_string(),
                source_id: serde_json::to_string(&[&memory_id, &entity_id])?,
                columns: vec!["memory_id".to_string(), "entity_id".to_string()],
                rows: vec![vec![memory_id.clone(), entity_id.clone()]],
            };
            Ok(DeterministicResolution::Exact(Box::new(
                ExactDeterministicRepair {
                    source_check_ids: vec![MEMORY_ENTITY_INTEGRITY],
                    target: target.clone(),
                    expected_version: None,
                    writer: RepairWriter::DeleteMemoryEntityLink,
                    mutation: RepairMutation::delete_memory_entity_link(&memory_id, &entity_id)
                        .map_err(contract_error)?,
                    allowed_effects: RepairAllowedEffects::memory_entity_link(target),
                    rollback,
                },
            )))
        })
        .collect()
}

async fn resolve_tags(
    snapshot: &LintReadSnapshot<'_>,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let mut rows = snapshot
        .query(
            "SELECT t.source,t.source_id,t.tag
               FROM document_tags t
              WHERE TRIM(t.tag)='' OR t.source NOT IN ('memory','page')
                 OR (t.source='memory' AND NOT EXISTS(
                    SELECT 1 FROM memories m WHERE m.source_id=t.source_id))
                 OR (t.source='page' AND NOT EXISTS(
                    SELECT 1 FROM pages p WHERE p.id=t.source_id))
              ORDER BY t.source,t.source_id,t.tag",
            libsql::params::Params::None,
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve tags: {error}")))?;
    let mut resolutions = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve tags: {error}")))?
    {
        let source = row.get::<String>(0).map_err(database_error)?;
        let source_id = row.get::<String>(1).map_err(database_error)?;
        let tag = row.get::<String>(2).map_err(database_error)?;
        let target = RepairTarget::tag(source.clone(), source_id.clone(), tag.clone())
            .map_err(contract_error)?;
        let rollback = StoredRollbackArtifact {
            format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
            table: "document_tags".to_string(),
            source_id: serde_json::to_string(&[&source, &source_id, &tag])?,
            columns: vec![
                "source".to_string(),
                "source_id".to_string(),
                "tag".to_string(),
            ],
            rows: vec![vec![source.clone(), source_id.clone(), tag.clone()]],
        };
        resolutions.push(DeterministicResolution::Exact(Box::new(
            ExactDeterministicRepair {
                source_check_ids: vec![TAG_INTEGRITY],
                target: target.clone(),
                expected_version: None,
                writer: RepairWriter::DeleteTagRow,
                mutation: RepairMutation::delete_tag_row(&source, &source_id, &tag)
                    .map_err(contract_error)?,
                allowed_effects: RepairAllowedEffects::tag_row(target),
                rollback,
            },
        )));
    }
    Ok(resolutions)
}

fn page_scope_clause(scope: &RepairLintScope) -> (&'static str, libsql::params::Params) {
    match scope {
        RepairLintScope::Global => ("", libsql::params::Params::None),
        RepairLintScope::Registered { space } => (
            " AND p.space=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        RepairLintScope::Uncategorized => (
            " AND p.space='00000000-0000-4000-8000-000000000001'",
            libsql::params::Params::None,
        ),
    }
}

async fn resolve_orphan_links(
    snapshot: &LintReadSnapshot<'_>,
    scope: &RepairLintScope,
) -> Result<Vec<DeterministicResolution>, WenlanError> {
    let (scope_clause, params) = page_scope_clause(scope);
    let sql = format!(
        "SELECT pl.source_page_id,pl.label_key,p.space,
                COUNT(target.id),MIN(target.id)
           FROM page_links pl
           JOIN pages p ON p.id=pl.source_page_id
           LEFT JOIN pages target
             ON LOWER(target.title)=LOWER(pl.label_key)
            AND target.status='active'
            AND target.space=p.space
          WHERE pl.target_page_id IS NULL AND p.status='active'{scope_clause}
          GROUP BY pl.source_page_id,pl.label_key,p.space
          ORDER BY pl.source_page_id,pl.label_key"
    );
    let mut rows = snapshot
        .query(&sql, params)
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve page links: {error}")))?;
    let mut resolutions = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair resolve page links: {error}")))?
    {
        let source_page_id = row.get::<String>(0).map_err(database_error)?;
        let label_key = row.get::<String>(1).map_err(database_error)?;
        let page_scope = row.get::<Option<String>>(2).map_err(database_error)?;
        let target_count = row.get::<i64>(3).map_err(database_error)?;
        let target_page_id = row.get::<Option<String>>(4).map_err(database_error)?;
        let target = RepairTarget::page_link(
            source_page_id.clone(),
            label_key.clone(),
            match page_scope.clone() {
                Some(scope) => RepairScope::registered(scope),
                None => Ok(RepairScope::uncategorized()),
            }
            .map_err(contract_error)?,
        )
        .map_err(contract_error)?;
        let affected_record = RepairAffectedRecord::try_new(
            RepairAffectedRecordKind::PageLink,
            format!("{source_page_id}:{label_key}"),
        )
        .map_err(contract_error)?;
        if target_count == 1 {
            let target_page_id = target_page_id.ok_or_else(|| {
                WenlanError::Validation("unique page link target missing".to_string())
            })?;
            let rollback = StoredRollbackArtifact {
                format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
                table: "page_links".to_string(),
                source_id: serde_json::to_string(&[&source_page_id, &label_key])?,
                columns: vec![
                    "source_page_id".to_string(),
                    "target_page_id".to_string(),
                    "label_key".to_string(),
                ],
                rows: vec![vec![
                    source_page_id.clone(),
                    "NULL".to_string(),
                    label_key.clone(),
                ]],
            };
            resolutions.push(DeterministicResolution::Exact(Box::new(
                ExactDeterministicRepair {
                    source_check_ids: vec![ORPHAN_LABELS],
                    target: target.clone(),
                    expected_version: None,
                    writer: RepairWriter::BindPageLink,
                    mutation: RepairMutation::bind_page_link(None, target_page_id)
                        .map_err(contract_error)?,
                    allowed_effects: RepairAllowedEffects::page_link_target(target),
                    rollback,
                },
            )));
        } else if target_count == 0 {
            resolutions.push(DeterministicResolution::Blocked(DeterministicBlocked {
                check_id: ORPHAN_LABELS,
                affected_record,
                detail: format!("label {label_key:?} has no active same-scope Page target"),
                next_action:
                    "create the target Page or edit the source Page label, then rerun lint"
                        .to_string(),
            }));
        } else {
            resolutions.push(DeterministicResolution::Review(DeterministicReview {
                check_id: ORPHAN_LABELS,
                affected_record,
                issue: format!("label {label_key:?} has {target_count} active same-scope targets"),
                choices: vec![
                    "select the intended active Page".to_string(),
                    "leave the label unresolved".to_string(),
                ],
                suggested_research_queries: vec![format!(
                    "find source evidence for Page label {label_key}"
                )],
            }));
        }
    }
    Ok(resolutions)
}

fn database_error(error: libsql::Error) -> WenlanError {
    WenlanError::VectorDb(format!("repair deterministic row: {error}"))
}

fn contract_error(error: impl std::fmt::Display) -> WenlanError {
    WenlanError::Validation(error.to_string())
}

#[cfg(test)]
#[path = "deterministic_tests.rs"]
mod tests;
