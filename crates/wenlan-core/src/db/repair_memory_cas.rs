// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{error::WenlanError, post_write::RepairWriteProof};
use wenlan_types::{
    repair::{
        RepairDigest, RepairManifest, RepairMutation, RepairReviewBinding, RepairRollbackPayloadV2,
        RepairTarget, RepairWriter,
    },
    MemoryType,
};

fn recovery_required_after_rollback_failure(
    error: &WenlanError,
    rollback_error: impl std::fmt::Display,
) -> WenlanError {
    log::error!("repair outcome is uncertain after {error}; rollback failed: {rollback_error}");
    WenlanError::Conflict("repair_apply_recovery_required".to_string())
}

async fn rollback_repair_transaction(
    connection: &libsql::Connection,
    error: &WenlanError,
    force_failure: bool,
) -> Result<(), WenlanError> {
    if force_failure {
        return Err(recovery_required_after_rollback_failure(
            error,
            "forced rollback failure",
        ));
    }
    connection
        .execute("ROLLBACK", ())
        .await
        .map(|_| ())
        .map_err(|rollback_error| recovery_required_after_rollback_failure(error, rollback_error))
}

struct ReclassifyMemoryCasInput<'a> {
    source_id: &'a str,
    expected_receipt: &'a RepairDigest,
    expected_space: Option<&'a str>,
    review_binding: Option<&'a RepairReviewBinding>,
    after_memory_type: MemoryType,
    force_rollback_failure: bool,
}

impl MemoryDB {
    pub(crate) async fn reclassify_memory_repair_cas<F>(
        &self,
        source_id: &str,
        expected_receipt: &RepairDigest,
        expected_space: Option<&str>,
        review_binding: Option<&RepairReviewBinding>,
        after_memory_type: MemoryType,
        before_commit: F,
    ) -> Result<RepairWriteProof, WenlanError>
    where
        F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
    {
        self.reclassify_memory_repair_cas_inner(
            ReclassifyMemoryCasInput {
                source_id,
                expected_receipt,
                expected_space,
                review_binding,
                after_memory_type,
                force_rollback_failure: false,
            },
            before_commit,
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn reclassify_memory_repair_cas_with_forced_rollback_failure<F>(
        &self,
        source_id: &str,
        expected_receipt: &RepairDigest,
        expected_space: Option<&str>,
        review_binding: Option<&RepairReviewBinding>,
        after_memory_type: MemoryType,
        before_commit: F,
    ) -> Result<RepairWriteProof, WenlanError>
    where
        F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
    {
        self.reclassify_memory_repair_cas_inner(
            ReclassifyMemoryCasInput {
                source_id,
                expected_receipt,
                expected_space,
                review_binding,
                after_memory_type,
                force_rollback_failure: true,
            },
            before_commit,
        )
        .await
    }

    async fn reclassify_memory_repair_cas_inner<F>(
        &self,
        input: ReclassifyMemoryCasInput<'_>,
        before_commit: F,
    ) -> Result<RepairWriteProof, WenlanError>
    where
        F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
    {
        let ReclassifyMemoryCasInput {
            source_id,
            expected_receipt,
            expected_space,
            review_binding,
            after_memory_type,
            force_rollback_failure,
        } = input;
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(|error| WenlanError::VectorDb(format!("repair begin: {error}")))?;
        let result = async {
            crate::repair::validate_target_space_on_connection(&conn, source_id, expected_space)
                .await?;
            if let Some(review_binding) = review_binding {
                crate::repair::validate_reclassification_review_on_connection(
                    &conn,
                    review_binding,
                    source_id,
                )
                .await?;
            }
            let (before_target_receipt, target_rows) =
                crate::repair::target_receipt_on_connection(&conn, source_id).await?;
            if &before_target_receipt != expected_receipt {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let parity_before = crate::repair::parity_input_generation_on_connection(&conn).await?;
            let non_target_before = crate::repair::effect_guard_receipt(conn.total_changes());
            let affected = conn
                .execute(
                    "UPDATE memories SET memory_type=?1
                     WHERE source='memory' AND source_id=?2",
                    libsql::params![after_memory_type.to_string(), source_id],
                )
                .await
                .map_err(|error| WenlanError::VectorDb(format!("repair reclassify: {error}")))?;
            if affected != target_rows {
                return Err(WenlanError::VectorDb(
                    "repair_target_row_count_changed".to_string(),
                ));
            }
            let (after_target_receipt, after_rows) =
                crate::repair::target_receipt_on_connection(&conn, source_id).await?;
            if after_rows != target_rows || after_target_receipt == before_target_receipt {
                return Err(WenlanError::VectorDb(
                    "repair_target_write_unproven".to_string(),
                ));
            }
            let parity_bump = crate::repair::parity_input_generation_on_connection(&conn)
                .await?
                .checked_sub(parity_before)
                .ok_or_else(|| {
                    WenlanError::VectorDb("repair_effect_counter_underflow".to_string())
                })?;
            let normalized_total_changes = conn
                .total_changes()
                .checked_sub(target_rows)
                .and_then(|changes| changes.checked_sub(parity_bump))
                .ok_or_else(|| {
                    WenlanError::VectorDb("repair_effect_counter_underflow".to_string())
                })?;
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
                rollback_repair_transaction(&conn, &error, force_rollback_failure).await?;
                return Err(error);
            }
        };
        if let Err(error) = before_commit(&proof) {
            rollback_repair_transaction(&conn, &error, force_rollback_failure).await?;
            return Err(error);
        }
        if let Err(error) = conn.execute("COMMIT", ()).await {
            let commit_error = WenlanError::VectorDb(format!("repair commit failed: {error}"));
            rollback_repair_transaction(&conn, &commit_error, force_rollback_failure).await?;
            return Err(commit_error);
        }
        Ok(proof)
    }

    pub(crate) async fn complete_entity_extraction_repair_cas<F>(
        &self,
        manifest: &RepairManifest,
        rollback: &RepairRollbackPayloadV2,
        before_commit: F,
    ) -> Result<RepairWriteProof, WenlanError>
    where
        F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
    {
        self.complete_entity_extraction_repair_cas_inner(manifest, rollback, false, before_commit)
            .await
    }

    #[cfg(test)]
    pub(crate) async fn complete_entity_extraction_repair_cas_with_forced_rollback_failure<F>(
        &self,
        manifest: &RepairManifest,
        rollback: &RepairRollbackPayloadV2,
        before_commit: F,
    ) -> Result<RepairWriteProof, WenlanError>
    where
        F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
    {
        self.complete_entity_extraction_repair_cas_inner(manifest, rollback, true, before_commit)
            .await
    }

    async fn complete_entity_extraction_repair_cas_inner<F>(
        &self,
        manifest: &RepairManifest,
        rollback: &RepairRollbackPayloadV2,
        force_rollback_failure: bool,
        before_commit: F,
    ) -> Result<RepairWriteProof, WenlanError>
    where
        F: FnOnce(&RepairWriteProof) -> Result<(), WenlanError>,
    {
        let (
            memory_id,
            entity_ids,
            scope,
            enrichment_status,
            enrichment_error,
            enrichment_attempts,
            enrichment_updated_at,
        ) = match (manifest.target(), manifest.mutation(), rollback) {
            (
                RepairTarget::MemoryEntityExtraction {
                    memory_id,
                    entity_ids,
                    scope,
                    ..
                },
                RepairMutation::CompleteEntityExtraction {
                    entity_ids: mutation_entity_ids,
                },
                RepairRollbackPayloadV2::CompleteEntityExtraction {
                    memory_id: rollback_memory_id,
                    enrichment_status,
                    enrichment_error,
                    enrichment_attempts,
                    enrichment_updated_at,
                    ..
                },
            ) if manifest.writer() == RepairWriter::CompleteEntityExtraction
                && memory_id == rollback_memory_id
                && entity_ids == mutation_entity_ids =>
            {
                (
                    memory_id,
                    entity_ids,
                    scope,
                    enrichment_status,
                    enrichment_error,
                    *enrichment_attempts,
                    *enrichment_updated_at,
                )
            }
            _ => {
                return Err(WenlanError::Validation(
                    "entity extraction repair target/writer/mutation mismatch".to_string(),
                ))
            }
        };
        if enrichment_status != "failed" {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        let review_binding = manifest.source().review_binding().ok_or_else(|| {
            WenlanError::Validation("entity extraction repair review binding missing".to_string())
        })?;
        let expected_owner_ids = [memory_id.clone()];
        if review_binding.owner_ids() != expected_owner_ids {
            return Err(WenlanError::Validation(
                "entity extraction repair review binding mismatch".to_string(),
            ));
        }

        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(|error| WenlanError::VectorDb(format!("repair begin: {error}")))?;
        let result = async {
            crate::repair::validate_target_space_on_connection(&conn, memory_id, scope.space())
                .await?;
            crate::repair::validate_selected_entities_on_connection(
                &conn,
                entity_ids,
                scope.space(),
            )
            .await?;
            let occurrence =
                crate::repair::validate_complete_entity_extraction_review_on_connection(
                    &conn,
                    review_binding.review_id(),
                    review_binding.owner_ids(),
                )
                .await?;
            if &occurrence != review_binding.occurrence_digest() {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let before =
                crate::repair::capture_complete_entity_extraction_on_connection(&conn, memory_id)
                    .await?;
            if &before != rollback {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let before_target_receipt = crate::repair::complete_entity_extraction_receipt(&before)?;
            if &before_target_receipt != manifest.expected_state().canonical_receipt() {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let parity_before = crate::repair::parity_input_generation_on_connection(&conn).await?;
            let non_target_before = crate::repair::effect_guard_receipt(conn.total_changes());
            let mut inserted = 0_u64;
            for entity_id in entity_ids {
                inserted = inserted.saturating_add(
                    conn.execute(
                        "INSERT INTO memory_entities(memory_id,entity_id)
                         SELECT ?1,?2
                          WHERE EXISTS(
                                SELECT 1 FROM entity_page_map epm
                                 JOIN pages p ON p.id = epm.page_id
                                   AND p.kind = 'entity' AND p.status = 'active'
                                 WHERE epm.entity_id=?2
                                   AND ((?3 IS NULL AND p.space IS NULL) OR p.space=?3))
                            AND NOT EXISTS(
                                SELECT 1 FROM memory_entities
                                 WHERE memory_id=?1 AND entity_id=?2)",
                        libsql::params![
                            memory_id.clone(),
                            entity_id.clone(),
                            scope.space().map(str::to_string)
                        ],
                    )
                    .await
                    .map_err(|error| {
                        WenlanError::VectorDb(format!(
                            "repair complete entity extraction link: {error}"
                        ))
                    })?,
                );
            }
            let updated = conn
                .execute(
                    "UPDATE enrichment_steps SET status='ok',error=NULL
                     WHERE source_id=?1 AND step_name='entity_extract'
                       AND status=?2 AND error IS ?3 AND attempts=?4 AND updated_at=?5",
                    libsql::params![
                        memory_id.clone(),
                        enrichment_status.clone(),
                        enrichment_error.clone(),
                        enrichment_attempts,
                        enrichment_updated_at
                    ],
                )
                .await
                .map_err(|error| {
                    WenlanError::VectorDb(format!(
                        "repair complete entity extraction step: {error}"
                    ))
                })?;
            if updated != 1 {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            let after =
                crate::repair::capture_complete_entity_extraction_on_connection(&conn, memory_id)
                    .await?;
            let RepairRollbackPayloadV2::CompleteEntityExtraction {
                before_entity_ids: after_entity_ids,
                ..
            } = &after
            else {
                unreachable!("complete entity extraction capture returns aggregate payload")
            };
            if entity_ids
                .iter()
                .any(|entity_id| after_entity_ids.binary_search(entity_id).is_err())
            {
                return Err(WenlanError::VectorDb(
                    "repair_target_write_unproven".to_string(),
                ));
            }
            let after_target_receipt = crate::repair::complete_entity_extraction_receipt(&after)?;
            if after_target_receipt == before_target_receipt {
                return Err(WenlanError::VectorDb(
                    "repair_target_write_unproven".to_string(),
                ));
            }
            let allowed_changes = inserted.saturating_add(updated);
            let parity_bump = crate::repair::parity_input_generation_on_connection(&conn)
                .await?
                .checked_sub(parity_before)
                .ok_or_else(|| {
                    WenlanError::VectorDb("repair_effect_counter_underflow".to_string())
                })?;
            let normalized_total_changes = conn
                .total_changes()
                .checked_sub(allowed_changes)
                .and_then(|changes| changes.checked_sub(parity_bump))
                .ok_or_else(|| {
                    WenlanError::VectorDb("repair_effect_counter_underflow".to_string())
                })?;
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
                rollback_repair_transaction(&conn, &error, force_rollback_failure).await?;
                return Err(error);
            }
        };
        if let Err(error) = before_commit(&proof) {
            rollback_repair_transaction(&conn, &error, force_rollback_failure).await?;
            return Err(error);
        }
        if let Err(error) = conn.execute("COMMIT", ()).await {
            let commit_error = WenlanError::VectorDb(format!("repair commit failed: {error}"));
            rollback_repair_transaction(&conn, &commit_error, force_rollback_failure).await?;
            return Err(commit_error);
        }
        Ok(proof)
    }
}
