// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{error::WenlanError, post_write::RepairWriteProof};
use wenlan_types::repair::{RepairManifest, RepairMutation, RepairTarget, RepairWriter};

pub async fn apply_deterministic_repair_cas<F>(
    db: &MemoryDB,
    manifest: &RepairManifest,
    prior_verified_tag_targets: &[RepairTarget],
    before_commit: F,
) -> Result<RepairWriteProof, WenlanError>
where
    F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
{
    if matches!(
        manifest.writer(),
        RepairWriter::ReclassifyMemory | RepairWriter::RegeneratePageProjection
    ) {
        return Err(WenlanError::Validation(
            "deterministic database writer mismatch".to_string(),
        ));
    }
    let conn = db.conn.lock().await;
    conn.execute("BEGIN IMMEDIATE", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair begin: {error}")))?;
    let result = async {
            crate::repair::validate_tag_record_set_on_connection(
                &conn,
                manifest,
                prior_verified_tag_targets,
                false,
            )
            .await?;
            let (before_target_receipt, _) =
                crate::repair::repair_target_receipt_on_connection(&conn, manifest.target()).await?;
            if &before_target_receipt != manifest.expected_state().canonical_receipt() {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let route_invalidation_before =
                if let (RepairTarget::Page { page_id, .. }, RepairWriter::ArchiveEmptySourcePage) =
                    (manifest.target(), manifest.writer())
                {
                    let mut rows = conn
                        .query(
                            "SELECT i.space, i.generation, s.generation
                               FROM page_community_route_inputs i
                               JOIN community_route_space_inputs s ON s.space=i.space
                              WHERE i.page_id=?1",
                            libsql::params![page_id.clone()],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "repair read page route invalidation: {error}"
                            ))
                        })?;
                    let row = rows
                        .next()
                        .await
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?
                        .ok_or_else(|| {
                            WenlanError::VectorDb(
                                "repair target page route invalidation state missing".to_string(),
                            )
                        })?;
                    Some((
                        page_id.clone(),
                        row.get::<String>(0)
                            .map_err(|error| WenlanError::VectorDb(error.to_string()))?,
                        row.get::<i64>(1)
                            .map_err(|error| WenlanError::VectorDb(error.to_string()))?,
                        row.get::<i64>(2)
                            .map_err(|error| WenlanError::VectorDb(error.to_string()))?,
                    ))
                } else {
                    None
                };
            let parity_before = crate::repair::parity_input_generation_on_connection(&conn).await?;
            let non_target_before = crate::repair::effect_guard_receipt(conn.total_changes());
            // Canonical-edge dual-writes made alongside a legacy-store repair
            // (G6 Stage 0); measured per-arm and allowed by the effect guard.
            let mut edge_dual_write_changes: u64 = 0;
            let affected = match (manifest.target(), manifest.writer(), manifest.mutation()) {
                (
                    RepairTarget::Memory { source_id, .. },
                    RepairWriter::NormalizeMemorySourceAgent,
                    RepairMutation::NormalizeMemorySourceAgent {
                        before_source_agent,
                    },
                ) => conn
                    .execute(
                        "UPDATE memories SET source_agent=NULL
                         WHERE source='memory' AND source_id=?1
                           AND source_agent=?2 AND TRIM(source_agent)=''",
                        libsql::params![source_id.clone(), before_source_agent.clone()],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("repair normalize source agent: {error}"))
                    })?,
                (
                    RepairTarget::Memory { source_id, .. },
                    RepairWriter::ClearMemorySupersedes,
                    RepairMutation::ClearMemorySupersedes { before_supersedes },
                ) if source_id == before_supersedes => conn
                    .execute(
                        "UPDATE memories SET supersedes=NULL
                         WHERE source='memory' AND source_id=?1 AND supersedes=?1",
                        libsql::params![source_id.clone()],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("repair clear supersedes: {error}"))
                    })?,
                (
                    RepairTarget::Memory { source_id, .. },
                    RepairWriter::UnstageOrphanRevision,
                    RepairMutation::UnstageOrphanRevision,
                ) => conn
                    .execute(
                        "UPDATE memories SET pending_revision=0
                         WHERE source='memory' AND source_id=?1
                           AND pending_revision=1 AND supersedes IS NULL",
                        libsql::params![source_id.clone()],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("repair unstage orphan revision: {error}"))
                    })?,
                (
                    RepairTarget::Tag {
                        source,
                        source_id,
                        tag,
                        ..
                    },
                    RepairWriter::DeleteTagRow,
                    RepairMutation::DeleteTagRow {
                        source: mutation_source,
                        source_id: mutation_source_id,
                        tag: mutation_tag,
                    },
                ) if source == mutation_source
                    && source_id == mutation_source_id
                    && tag == mutation_tag =>
                {
                    conn.execute(
                        "DELETE FROM document_tags
                         WHERE source=?1 AND source_id=?2 AND tag=?3
                           AND (TRIM(tag)='' OR source NOT IN ('memory','page')
                             OR (source='memory' AND NOT EXISTS(
                                SELECT 1 FROM memories m WHERE m.source_id=document_tags.source_id))
                             OR (source='page' AND NOT EXISTS(
                                SELECT 1 FROM pages p WHERE p.id=document_tags.source_id)))",
                        libsql::params![source.clone(), source_id.clone(), tag.clone()],
                    )
                    .await
                    .map_err(|error| WenlanError::VectorDb(format!("repair delete tag row: {error}")))?
                }
                (
                    RepairTarget::MemoryEntityLink {
                        memory_id,
                        entity_id,
                        ..
                    },
                    RepairWriter::DeleteMemoryEntityLink,
                    RepairMutation::DeleteMemoryEntityLink {
                        memory_id: mutation_memory_id,
                        entity_id: mutation_entity_id,
                    },
                ) if memory_id == mutation_memory_id && entity_id == mutation_entity_id => conn
                    .execute(
                        // G6 Stage 1.5a review fix: the planner
                        // (`resolve_memory_entity_links`, repair_plan/deterministic.rs)
                        // already reads the `kind='entity'` shadow page for this same
                        // existence decision -- the applier's guard must read the same
                        // store, or a raw-seeded `entities` row with no shadow page
                        // would plan-delete but fail to apply.
                        "DELETE FROM memory_entities
                         WHERE memory_id=?1 AND entity_id=?2
                           AND (NOT EXISTS(
                                SELECT 1 FROM memories m WHERE m.source_id=memory_entities.memory_id)
                             OR NOT EXISTS(
                                SELECT 1 FROM entity_page_map epm
                                JOIN pages p ON p.id = epm.page_id
                                WHERE epm.entity_id = memory_entities.entity_id
                                  AND p.kind = 'entity' AND p.status = 'active'))",
                        libsql::params![memory_id.clone(), entity_id.clone()],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("repair delete memory entity link: {error}"))
                    })?,
                (
                    RepairTarget::PageLink {
                        source_page_id,
                        label_key,
                        scope,
                    },
                    RepairWriter::BindPageLink,
                    RepairMutation::BindPageLink {
                        before_target_page_id: None,
                        after_target_page_id,
                    },
                ) => {
                    let mut target_rows = conn
                        .query(
                            "SELECT id FROM pages
                             WHERE LOWER(title)=LOWER(?1) AND status='active'
                               AND space=COALESCE(?2,'00000000-0000-4000-8000-000000000001')
                             ORDER BY id LIMIT 2",
                            libsql::params![label_key.clone(), scope.space()],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!("repair page link target: {error}"))
                        })?;
                    let first = target_rows
                        .next()
                        .await
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?
                        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?
                        .get::<String>(0)
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?;
                    if first != *after_target_page_id
                        || target_rows
                            .next()
                            .await
                            .map_err(|error| WenlanError::VectorDb(error.to_string()))?
                            .is_some()
                    {
                        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                    }
                    drop(target_rows);
                    let changed = conn
                        .execute(
                            "UPDATE page_links SET target_page_id=?1
                             WHERE source_page_id=?2 AND label_key=?3 AND target_page_id IS NULL",
                            libsql::params![
                                after_target_page_id.clone(),
                                source_page_id.clone(),
                                label_key.clone()
                            ],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!("repair bind page link: {error}"))
                        })?;
                    // Dual-write (G6 Stage 0): an orphan row derives no edge,
                    // so binding its target makes the canonical `links` edge
                    // implied — mint it in the same transaction, mirroring
                    // `resolve_orphan_page_links`.
                    let changes_before_mint = conn.total_changes();
                    if changed > 0 {
                        let mut label_rows = conn
                            .query(
                                "SELECT label FROM page_links
                                 WHERE source_page_id = ?1 AND label_key = ?2",
                                libsql::params![source_page_id.clone(), label_key.clone()],
                            )
                            .await
                            .map_err(|error| {
                                WenlanError::VectorDb(format!("repair bind link label: {error}"))
                            })?;
                        let label = match label_rows.next().await.map_err(|error| {
                            WenlanError::VectorDb(format!("repair bind link label row: {error}"))
                        })? {
                            Some(row) => row.get::<String>(0).map_err(|error| {
                                WenlanError::VectorDb(format!(
                                    "repair bind link label value: {error}"
                                ))
                            })?,
                            None => label_key.clone(),
                        };
                        drop(label_rows);
                        let semantic_patch = serde_json::json!({ "label": label }).to_string();
                        let mut src_rows = conn
                            .query(
                                "SELECT space FROM pages WHERE id = ?1",
                                libsql::params![source_page_id.clone()],
                            )
                            .await
                            .map_err(|error| {
                                WenlanError::VectorDb(format!("repair bind link src space: {error}"))
                            })?;
                        let src_space: Option<String> =
                            match src_rows.next().await.map_err(|error| {
                                WenlanError::VectorDb(format!("repair bind link src row: {error}"))
                            })? {
                                Some(row) => row.get(0).unwrap_or(None),
                                None => None,
                            };
                        drop(src_rows);
                        if let Some(src_space) = src_space.as_deref() {
                            let mut dst_rows = conn
                                .query(
                                    "SELECT space FROM pages WHERE id = ?1",
                                    libsql::params![after_target_page_id.clone()],
                                )
                                .await
                                .map_err(|error| {
                                    WenlanError::VectorDb(format!(
                                        "repair bind link dst space: {error}"
                                    ))
                                })?;
                            let dst_space: Option<String> =
                                match dst_rows.next().await.map_err(|error| {
                                    WenlanError::VectorDb(format!(
                                        "repair bind link dst row: {error}"
                                    ))
                                })? {
                                    Some(row) => row.get(0).unwrap_or(None),
                                    None => None,
                                };
                            drop(dst_rows);
                            let cross_space_downgrade = MemoryDB::resolved_space_downgrades(
                                dst_space.as_deref(),
                                src_space,
                            );
                            let lineage = if cross_space_downgrade {
                                "legacy"
                            } else {
                                "synthesis"
                            };
                            MemoryDB::dual_write_edge_with_payload(
                                &conn,
                                "links",
                                "page",
                                source_page_id,
                                "page",
                                after_target_page_id,
                                label_key,
                                lineage,
                                src_space,
                                cross_space_downgrade,
                                None,
                                None,
                                None,
                                Some(&semantic_patch),
                            )
                            .await
                            .map_err(|error| {
                                WenlanError::VectorDb(format!(
                                    "repair bind link edge mint: {error}"
                                ))
                            })?;
                        }
                    }
                    edge_dual_write_changes = conn
                        .total_changes()
                        .checked_sub(changes_before_mint)
                        .ok_or_else(|| {
                            WenlanError::VectorDb("repair_effect_counter_underflow".to_string())
                        })?;
                    changed
                }
                (
                    RepairTarget::Page { page_id, scope },
                    RepairWriter::ArchiveEmptySourcePage,
                    RepairMutation::ArchiveEmptySourcePage {
                        before_status,
                        after_status,
                    },
                ) if before_status == "active" && after_status == "archived" => {
                    let expected_version = manifest.expected_state().version().ok_or_else(|| {
                        WenlanError::Validation("repair page version missing".to_string())
                    })?;
                    let mut content_rows = conn
                        .query(
                            "SELECT content FROM pages WHERE id=?1 AND version=?2",
                            libsql::params![page_id.clone(), expected_version],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "repair read empty source page content: {error}"
                            ))
                        })?;
                    let content = content_rows
                        .next()
                        .await
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?
                        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?
                        .get::<String>(0)
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?;
                    drop(content_rows);
                    if !content.trim().is_empty() {
                        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                    }
                    conn.execute(
                        "UPDATE pages SET status='archived'
                         WHERE id=?1 AND version=?2 AND status='active'
                           AND creation_kind='source' AND review_status='unconfirmed'
                           AND COALESCE(user_edited,0)=0
                           AND json_valid(source_memory_ids)
                           AND json_type(source_memory_ids)='array'
                           AND json_array_length(source_memory_ids)=0
                           AND space=COALESCE(?3,'00000000-0000-4000-8000-000000000001')
                           AND NOT EXISTS(
                                SELECT 1 FROM page_sources ps WHERE ps.page_id=pages.id)
                           AND NOT EXISTS(
                                SELECT 1 FROM page_evidence pe WHERE pe.page_id=pages.id)",
                        libsql::params![page_id.clone(), expected_version, scope.space()],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!("repair archive empty source page: {error}"))
                    })?
                }
                _ => {
                    return Err(WenlanError::Validation(
                        "deterministic repair target/writer/mutation mismatch".to_string(),
                    ))
                }
            };
            if affected == 0 {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let (after_target_receipt, _) =
                crate::repair::repair_target_receipt_on_connection(&conn, manifest.target()).await?;
            if after_target_receipt == before_target_receipt {
                return Err(WenlanError::VectorDb(
                    "repair_target_write_unproven".to_string(),
                ));
            }
            let allowed_derived_changes =
                if let Some((page_id, expected_space, page_generation, space_generation)) =
                    route_invalidation_before
                {
                    let mut rows = conn
                        .query(
                            "SELECT i.space, i.generation, s.generation
                               FROM page_community_route_inputs i
                               JOIN community_route_space_inputs s ON s.space=i.space
                              WHERE i.page_id=?1",
                            libsql::params![page_id],
                        )
                        .await
                        .map_err(|error| {
                            WenlanError::VectorDb(format!(
                                "repair verify page route invalidation: {error}"
                            ))
                        })?;
                    let row = rows
                        .next()
                        .await
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?
                        .ok_or_else(|| {
                            WenlanError::VectorDb(
                                "repair target page route invalidation state missing".to_string(),
                            )
                        })?;
                    let actual_space = row
                        .get::<String>(0)
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?;
                    let actual_page_generation = row
                        .get::<i64>(1)
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?;
                    let actual_space_generation = row
                        .get::<i64>(2)
                        .map_err(|error| WenlanError::VectorDb(error.to_string()))?;
                    if actual_space != expected_space
                        || actual_page_generation != page_generation.saturating_add(1)
                        || actual_space_generation != space_generation
                    {
                        return Err(WenlanError::VectorDb(
                            "repair target page route invalidation unproven".to_string(),
                        ));
                    }
                    1
                } else {
                    0
                };
            let allowed_changes = affected
                .checked_add(allowed_derived_changes)
                .and_then(|changes| changes.checked_add(edge_dual_write_changes))
                .ok_or_else(|| WenlanError::VectorDb("repair_effect_counter_overflow".to_string()))?;
            let parity_bump = crate::repair::parity_input_generation_on_connection(&conn)
                .await?
                .checked_sub(parity_before)
                .ok_or_else(|| WenlanError::VectorDb("repair_effect_counter_underflow".to_string()))?;
            let normalized_total_changes = conn
                .total_changes()
                .checked_sub(allowed_changes)
                .and_then(|changes| changes.checked_sub(parity_bump))
                .ok_or_else(|| WenlanError::VectorDb("repair_effect_counter_underflow".to_string()))?;
            let non_target_after = crate::repair::effect_guard_receipt(normalized_total_changes);
            if non_target_after != non_target_before {
                return Err(WenlanError::VectorDb("repair_effect_escape".to_string()));
            }
            let post_apply_db_digest = crate::repair::database_content_digest(&conn).await?;
            Ok(RepairWriteProof::from_parts(
                before_target_receipt,
                after_target_receipt,
                non_target_before,
                non_target_after,
                post_apply_db_digest,
            ))
        }
        .await;

    let proof = match result {
        Ok(proof) => proof,
        Err(error) => {
            if let Err(rollback_error) = conn.execute("ROLLBACK", ()).await {
                return Err(WenlanError::VectorDb(format!(
                    "{error}; repair rollback failed: {rollback_error}"
                )));
            }
            return Err(error);
        }
    };
    if let Err(error) = before_commit(&proof) {
        if let Err(rollback_error) = conn.execute("ROLLBACK", ()).await {
            return Err(WenlanError::VectorDb(format!(
                "{error}; repair rollback failed: {rollback_error}"
            )));
        }
        return Err(error);
    }
    if let Err(error) = conn.execute("COMMIT", ()).await {
        let _ = conn.execute("ROLLBACK", ()).await;
        return Err(WenlanError::VectorDb(format!(
            "repair commit failed: {error}"
        )));
    }
    Ok(proof)
}
