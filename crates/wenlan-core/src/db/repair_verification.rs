// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::{
    error::WenlanError,
    export::knowledge::{KnowledgeProjectionWrite, OwnedRepairProjectionSession},
    lint::pages::fs::{scan_page_root_controlled, PageScanControl},
    repair::{
        capture_page_projection_from_row, capture_rename_page_title_on_connection,
        effect_guard_receipt, page_projection_non_target_receipt,
        projection_page_row_on_connection, projection_rollback_paths,
        rename_page_title_excluded_paths, rename_page_title_non_target_receipt,
        rename_page_title_receipt, repair_digest, repair_target_receipt_on_connection,
        stale_page_projection_paths, target_receipt, validate_current_db_receipts,
        validate_current_page_receipts_locked, validate_current_page_receipts_on_repair_projection,
        validate_deterministic_target_resolved, validate_tag_record_set_on_connection,
        RepairArtifactStore, StoredRollbackArtifact,
    },
};
use std::{collections::BTreeSet, path::Path, time::Duration};
use wenlan_types::repair::{
    RepairApplyReceipt, RepairManifest, RepairRollbackPayloadV2, RepairTarget,
    RepairVerificationReceipt, RepairVerificationReceiptDraft, RepairWriter, VerifyRepairRequest,
};

pub(crate) struct RepairVerificationAtomicInput<'a> {
    pub(crate) store: &'a RepairArtifactStore,
    pub(crate) manifest: &'a RepairManifest,
    pub(crate) apply_receipt: &'a RepairApplyReceipt,
    pub(crate) request: &'a VerifyRepairRequest,
    pub(crate) prior_verified_tag_targets: &'a [RepairTarget],
    pub(crate) rollback: Option<&'a StoredRollbackArtifact>,
    pub(crate) rename_rollback: Option<&'a RepairRollbackPayloadV2>,
    pub(crate) page_root: Option<&'a Path>,
    pub(crate) verified_at: i64,
    pub(crate) rename_projection_session: Option<&'a OwnedRepairProjectionSession>,
}

pub(crate) async fn record_repair_verification_atomic(
    db: &MemoryDB,
    input: RepairVerificationAtomicInput<'_>,
) -> Result<RepairVerificationReceipt, WenlanError> {
    let RepairVerificationAtomicInput {
        store,
        manifest,
        apply_receipt,
        request,
        prior_verified_tag_targets,
        rollback,
        rename_rollback,
        page_root,
        verified_at,
        rename_projection_session,
    } = input;

    let connection = db.conn.lock().await;
    connection
        .execute("BEGIN IMMEDIATE", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair verify begin: {error}")))?;
    let result = async {
        validate_current_db_receipts(db, request.general_report(), request.deep_report()).await?;
        // The durable content-addressed apply receipt records an apply-time
        // effect guard and rejects unequal non_target_before/non_target_after
        // values. Verification binds a fresh report to the current DB snapshot
        // and rechecks the target receipt below. Unrelated writes after that
        // completed apply transaction must not strand an otherwise valid
        // receipt.
        validate_tag_record_set_on_connection(
            &connection,
            manifest,
            prior_verified_tag_targets,
            true,
        )
        .await?;
        validate_deterministic_target_resolved(db, manifest, page_root).await?;
        let (target_now, _) = match manifest.target() {
            RepairTarget::PageProjection { page_id, .. }
                if manifest.writer() == RepairWriter::RenamePageTitle =>
            {
                let rollback = rename_rollback.ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let projection = rename_projection_session
                    .ok_or_else(|| {
                        WenlanError::Validation(
                            "page projection repair root unavailable".to_string(),
                        )
                    })?
                    .locked();
                let current =
                    capture_rename_page_title_on_connection(&connection, &projection, page_id)
                        .await?;
                let scan = projection.scan_page_root_controlled(
                    true,
                    &PageScanControl::with_timeout(Duration::from_secs(30)),
                )?;
                let excluded = rename_page_title_excluded_paths(rollback)?;
                let non_target_now = rename_page_title_non_target_receipt(
                    &effect_guard_receipt(0),
                    scan.non_target_digest(&excluded),
                    &current,
                )?;
                if non_target_now != *apply_receipt.non_target_after() {
                    return Err(WenlanError::Conflict(
                        "repair_non_target_state_changed".to_string(),
                    ));
                }
                (rename_page_title_receipt(&current)?, 1)
            }
            RepairTarget::PageProjection { page_id, .. }
                if manifest.writer() == RepairWriter::QuarantineStalePageProjection =>
            {
                let rollback = rollback.ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let page_root = page_root.ok_or_else(|| {
                    WenlanError::Validation("page projection repair root unavailable".to_string())
                })?;
                let (source_path, quarantine_path) = stale_page_projection_paths(rollback)?;
                let target_now =
                    KnowledgeProjectionWrite::with_projection_lock(page_root, |projection| {
                        let excluded = BTreeSet::from([
                            ".wenlan".to_string(),
                            ".wenlan/state.json".to_string(),
                            ".wenlan/orphaned".to_string(),
                            source_path.clone(),
                            quarantine_path.clone(),
                        ]);
                        let scan = projection.scan_page_root_controlled(
                            true,
                            &PageScanControl::with_timeout(Duration::from_secs(30)),
                        )?;
                        let current = projection.capture_stale_page_projection_current(
                            page_id,
                            &source_path,
                            &quarantine_path,
                        )?;
                        let non_target_now = page_projection_non_target_receipt(
                            scan.non_target_digest(&excluded),
                            &current,
                        )?;
                        if non_target_now != *apply_receipt.non_target_after() {
                            return Err(WenlanError::Conflict(
                                "repair_non_target_state_changed".to_string(),
                            ));
                        }
                        target_receipt(&current)
                    })?;
                (target_now, 0)
            }
            RepairTarget::PageProjection { page_id, .. } => {
                let rollback = rollback.ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let page_root = page_root.ok_or_else(|| {
                    WenlanError::Validation("page projection repair root unavailable".to_string())
                })?;
                let paths = projection_rollback_paths(rollback)?;
                let page_row = projection_page_row_on_connection(&connection, page_id).await?;
                let target_now = KnowledgeProjectionWrite::with_projection_lock(page_root, |_| {
                    let scan = scan_page_root_controlled(
                        page_root,
                        true,
                        &PageScanControl::with_timeout(Duration::from_secs(30)),
                    )
                    .map_err(|error| {
                        WenlanError::Validation(format!("repair projection scan: {error}"))
                    })?;
                    let current = capture_page_projection_from_row(
                        page_root,
                        page_id,
                        page_row,
                        &paths,
                        &rollback.table,
                    )?;
                    let non_target_now = page_projection_non_target_receipt(
                        scan.non_target_digest(&paths),
                        &current,
                    )?;
                    if non_target_now != *apply_receipt.non_target_after() {
                        return Err(WenlanError::Conflict(
                            "repair_non_target_state_changed".to_string(),
                        ));
                    }
                    target_receipt(&current)
                })?;
                (target_now, 1)
            }
            _ => repair_target_receipt_on_connection(&connection, manifest.target()).await?,
        };
        if target_now != *apply_receipt.after_target_receipt() {
            return Err(WenlanError::Conflict(
                "repair_verification_state_changed".to_string(),
            ));
        }
        let draft = match request.deep_report() {
            Some(deep) => RepairVerificationReceiptDraft::try_new(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                verified_at,
                request.general_report().snapshots().clone(),
                deep.snapshots().clone(),
            ),
            None => RepairVerificationReceiptDraft::try_new_general_only(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                verified_at,
                request.general_report().snapshots().clone(),
            ),
        }
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
        let receipt_digest = repair_digest(&draft.canonical_bytes()?);
        let receipt = RepairVerificationReceipt::from_draft(draft, receipt_digest);
        if let Some(session) = rename_projection_session {
            validate_current_page_receipts_on_repair_projection(
                request.general_report(),
                request.deep_report(),
                &session.locked(),
            )?;
            store.persist_verification_receipt(&receipt)?;
            Ok(receipt)
        } else if let Some(page_root) = page_root {
            KnowledgeProjectionWrite::with_projection_lock(page_root, |_| {
                validate_current_page_receipts_locked(
                    request.general_report(),
                    request.deep_report(),
                    Some(page_root),
                )?;
                store.persist_verification_receipt(&receipt)?;
                Ok(receipt)
            })
        } else {
            store.persist_verification_receipt(&receipt)?;
            Ok(receipt)
        }
    }
    .await;
    let receipt = match result {
        Ok(receipt) => receipt,
        Err(error) => {
            let _ = connection.execute("ROLLBACK", ()).await;
            return Err(error);
        }
    };
    connection
        .execute("COMMIT", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair verify commit: {error}")))?;
    Ok(receipt)
}
