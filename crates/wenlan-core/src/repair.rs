// SPDX-License-Identifier: Apache-2.0
//! Durable preparation and CAS application of workflow-approved repairs.
//!
//! The exact approval phrase binds cooperating agent workflows to one manifest;
//! it is not an authentication boundary against malicious local processes.

use crate::{
    db::MemoryDB,
    error::WenlanError,
    lint::{
        context::ExecutionGate,
        pages::fs::{scan_page_root_controlled, PageScanControl},
        snapshot::{LintReadSnapshot, SnapshotError, SnapshotReceipt},
    },
};
use fs2::FileExt as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Component, Path, PathBuf},
    str::FromStr,
};
use uuid::Uuid;
use wenlan_types::{
    lint::{
        LintCheckResult, LintDigest, LintEvidenceRef, LintGateEffect, LintMetricCode,
        LintMetricValue, LintOutcome, LintProfile, LintReport, LintScope, LintScopeKind,
        LintSemanticAction,
    },
    repair::{
        ApplyRepairRequest, PrepareRepairRequest, RepairAllowedEffects, RepairApplyReceipt,
        RepairApplyReceiptDraft, RepairCheckBaseline, RepairChoice, RepairContractError,
        RepairDigest, RepairEnrichmentStep, RepairExpectedState, RepairLintScope, RepairManifest,
        RepairManifestDraft, RepairMutation, RepairPostAssertions, RepairRecordSetBaseline,
        RepairReviewBinding, RepairRollbackArtifact, RepairRollbackPayloadV2, RepairRollbackV2,
        RepairScope, RepairSource, RepairTarget, RepairVerificationReceipt,
        RepairVerificationReceiptDraft, RepairWriter, StoredRepairApplyReceipt,
        StoredRepairManifest, StoredRepairRollbackArtifact, StoredRepairVerificationReceipt,
        VerifyRepairRequest, REPAIR_CLASSIFICATION_CHECK_ID, REPAIR_ROLLBACK_FORMAT_VERSION,
    },
    repair_plan::{
        RepairAffectedRecord, RepairAffectedRecordKind, RepairPlan, RepairPlanEntriesPage,
        RepairPlanEntriesRequest, RepairPlanEntry, StoredRepairPlan,
    },
    MemoryType,
};

const MANIFEST_FILE: &str = "manifest.json";
const ROLLBACK_FILE: &str = "rollback-v1.json";
const LEGACY_ROLLBACK_FORMAT_VERSION: u16 = 1;
const APPLY_RECEIPT_FILE: &str = "apply-receipt.json";
const APPLY_RECEIPT_PENDING_FILE: &str = ".apply-receipt.json.pending";
const STALE_PAGE_PROJECTION_APPLY_JOURNAL_FILE: &str =
    ".stale-page-projection-apply-journal-v1.json";
const STALE_PAGE_PROJECTION_APPLY_JOURNAL_PENDING_FILE: &str =
    ".stale-page-projection-apply-journal-v1.json.pending";
const STALE_PAGE_PROJECTION_APPLY_JOURNAL_FORMAT_VERSION: u16 = 1;
const VERIFICATION_RECEIPT_FILE: &str = "verification-receipt.json";
const OPERATION_LOCK_FILE: &str = ".operation.lock";
const PLAN_DIR: &str = "plans";
const TAG_RECORD_SET_LOCK_PREFIX: &str = ".tag-record-set-";
const REPAIR_PLAN_ARTIFACT_MAX_BYTES: u64 = 16 * 1024 * 1024;
const PAGE_PROJECTION_ROLLBACK_TABLE: &str = "page_projection";
const PAGE_PROJECTION_ROLLBACK_TABLE_V2: &str = "page_projection_v2";
const STALE_PAGE_PROJECTION_ROLLBACK_TABLE: &str = "stale_page_projection_v1";
const PAGE_PROJECTION_ROLLBACK_MAX_BYTES: u64 = 16 * 1024 * 1024;
const REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES: u64 = 40 * 1024 * 1024;

#[cfg(test)]
mod title_rename_tests;

#[derive(Debug, Clone)]
pub struct RepairArtifactStore {
    root: PathBuf,
}

impl RepairArtifactStore {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_dir(&self, manifest_id: &str) -> Result<PathBuf, WenlanError> {
        if !safe_manifest_id(manifest_id) {
            return Err(WenlanError::Validation(
                "invalid_repair_manifest_id".to_string(),
            ));
        }
        Ok(self.root.join(manifest_id))
    }

    pub fn plan_path(&self, plan_id: &str) -> Result<PathBuf, WenlanError> {
        if !safe_plan_id(plan_id) {
            return Err(WenlanError::Validation(
                "invalid_repair_plan_id".to_string(),
            ));
        }
        Ok(self.root.join(PLAN_DIR).join(format!("{plan_id}.jsonl")))
    }

    pub(crate) fn persist_plan(&self, plan: &RepairPlan) -> Result<PathBuf, WenlanError> {
        if repair_digest(&plan.canonical_unsigned_bytes()?) != *plan.plan_digest() {
            return Err(WenlanError::Validation(
                "repair_plan_digest_mismatch".to_string(),
            ));
        }
        for (offset, entry) in plan.entries().iter().enumerate() {
            RepairPlanEntriesPage::try_new(
                plan.plan_id().to_string(),
                plan.plan_digest().clone(),
                plan.scope().clone(),
                offset,
                plan.entries().len(),
                vec![entry.clone()],
            )
            .map_err(|_| WenlanError::Validation("repair_plan_entry_too_large".to_string()))?;
        }
        ensure_private_dir(&self.root)?;
        let plan_dir = self.root.join(PLAN_DIR);
        ensure_private_dir(&plan_dir)?;
        let final_path = self.plan_path(plan.plan_id())?;
        if final_path.exists() {
            return Err(WenlanError::Conflict("repair_plan_exists".to_string()));
        }
        let pending_path = plan_dir.join(format!(".{}.tmp-{}", plan.plan_id(), Uuid::new_v4()));
        let result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&pending_path)?;
            set_private_file_permissions(&pending_path)?;
            let mut written = 0_u64;
            write_plan_jsonl_line(
                &mut file,
                &mut written,
                &serde_json::json!({
                    "record": "plan_header",
                    "plan_schema_version": plan.schema_version(),
                    "plan_id": plan.plan_id(),
                    "scope": plan.scope(),
                    "general_report_receipt": plan.general_report_receipt(),
                    "deep_report_receipt": plan.deep_report_receipt(),
                    "plan_digest": plan.plan_digest(),
                    "deterministic_complete": plan.deterministic_complete(),
                    "semantic_complete": plan.semantic_complete(),
                    "totals": plan.totals(),
                }),
            )?;
            for entry in plan.entries() {
                write_plan_jsonl_line(
                    &mut file,
                    &mut written,
                    &serde_json::json!({ "record": "entry", "entry": entry }),
                )?;
            }
            write_plan_jsonl_line(
                &mut file,
                &mut written,
                &serde_json::json!({
                    "record": "complete",
                    "entry_count": plan.entries().len(),
                    "plan_digest": plan.plan_digest(),
                }),
            )?;
            file.sync_all()?;
            drop(file);
            sync_dir(&plan_dir)?;
            publish_no_replace(&pending_path, &final_path, "repair_plan_exists")?;
            Ok::<(), WenlanError>(())
        })();
        if result.is_err() && pending_path.exists() {
            let _ = fs::remove_file(&pending_path);
        }
        result?;
        Ok(final_path)
    }

    pub fn load_plan_entries_page(
        &self,
        request: &RepairPlanEntriesRequest,
    ) -> Result<RepairPlanEntriesPage, WenlanError> {
        let path = self.plan_path(request.plan_id())?;
        let file = File::open(&path)?;
        if file.metadata()?.len() > REPAIR_PLAN_ARTIFACT_MAX_BYTES {
            return Err(WenlanError::Validation(
                "repair_plan_artifact_too_large".to_string(),
            ));
        }
        let mut lines = BufReader::new(file).lines();
        let header_line = lines
            .next()
            .ok_or_else(|| WenlanError::Validation("repair_plan_artifact_invalid".to_string()))??;
        let mut header = serde_json::from_str::<serde_json::Value>(&header_line)?;
        let header_object = header
            .as_object_mut()
            .ok_or_else(|| WenlanError::Validation("repair_plan_artifact_invalid".to_string()))?;
        if header_object
            .remove("record")
            .and_then(|record| record.as_str().map(str::to_string))
            .as_deref()
            != Some("plan_header")
        {
            return Err(WenlanError::Validation(
                "repair_plan_artifact_invalid".to_string(),
            ));
        }

        let mut entries = Vec::<RepairPlanEntry>::new();
        let mut completion = None;
        for line in lines {
            let line = line?;
            if completion.is_some() {
                return Err(WenlanError::Validation(
                    "repair_plan_artifact_invalid".to_string(),
                ));
            }
            let value = serde_json::from_str::<serde_json::Value>(&line)?;
            match value.get("record").and_then(serde_json::Value::as_str) {
                Some("entry") => {
                    #[derive(Deserialize)]
                    #[serde(deny_unknown_fields)]
                    struct EntryRecord {
                        record: String,
                        entry: RepairPlanEntry,
                    }
                    let record = serde_json::from_value::<EntryRecord>(value)?;
                    if record.record != "entry" {
                        return Err(WenlanError::Validation(
                            "repair_plan_artifact_invalid".to_string(),
                        ));
                    }
                    entries.push(record.entry);
                }
                Some("complete") => {
                    #[derive(Deserialize)]
                    #[serde(deny_unknown_fields)]
                    struct CompletionRecord {
                        record: String,
                        entry_count: usize,
                        plan_digest: RepairDigest,
                    }
                    let record = serde_json::from_value::<CompletionRecord>(value)?;
                    if record.record != "complete" {
                        return Err(WenlanError::Validation(
                            "repair_plan_artifact_invalid".to_string(),
                        ));
                    }
                    completion = Some((record.entry_count, record.plan_digest));
                }
                _ => {
                    return Err(WenlanError::Validation(
                        "repair_plan_artifact_invalid".to_string(),
                    ));
                }
            }
        }
        let (entry_count, completion_digest) = completion
            .ok_or_else(|| WenlanError::Validation("repair_plan_artifact_invalid".to_string()))?;
        if entry_count != entries.len() {
            return Err(WenlanError::Validation(
                "repair_plan_artifact_invalid".to_string(),
            ));
        }
        header_object.insert("entries".to_string(), serde_json::to_value(&entries)?);
        let stored = serde_json::from_value::<StoredRepairPlan>(header)?;
        let plan = stored
            .verify_and_try_into_current(|canonical, expected| {
                repair_digest(canonical) == *expected
            })
            .map_err(|error| WenlanError::Validation(error.to_string()))?;
        if completion_digest != *plan.plan_digest()
            || request.plan_digest() != plan.plan_digest()
            || request.offset() > plan.entries().len()
        {
            return Err(WenlanError::Validation(
                "repair_plan_digest_mismatch".to_string(),
            ));
        }
        let end = request
            .offset()
            .saturating_add(request.limit())
            .min(plan.entries().len());
        let mut page_entries = Vec::new();
        for entry in &plan.entries()[request.offset()..end] {
            let mut candidate_entries = page_entries.clone();
            candidate_entries.push(entry.clone());
            if RepairPlanEntriesPage::try_new(
                plan.plan_id().to_string(),
                plan.plan_digest().clone(),
                plan.scope().clone(),
                request.offset(),
                plan.entries().len(),
                candidate_entries,
            )
            .is_err()
            {
                if page_entries.is_empty() {
                    return Err(WenlanError::Validation(
                        "repair_plan_entry_too_large".to_string(),
                    ));
                }
                break;
            }
            page_entries.push(entry.clone());
        }
        RepairPlanEntriesPage::try_new(
            plan.plan_id().to_string(),
            plan.plan_digest().clone(),
            plan.scope().clone(),
            request.offset(),
            plan.entries().len(),
            page_entries,
        )
        .map_err(|error| WenlanError::Validation(error.to_string()))
    }

    fn read_stored_manifest(&self, manifest_id: &str) -> Result<StoredRepairManifest, WenlanError> {
        let path = self.manifest_dir(manifest_id)?.join(MANIFEST_FILE);
        let manifest = StoredRepairManifest::from_slice(&fs::read(path)?)?;
        if manifest.manifest_id() != manifest_id {
            return Err(WenlanError::Validation(
                "repair_manifest_id_mismatch".to_string(),
            ));
        }
        Ok(manifest)
    }

    pub fn load_manifest(&self, manifest_id: &str) -> Result<RepairManifest, WenlanError> {
        let manifest = self.read_stored_manifest(manifest_id)?;
        manifest
            .verify_and_try_into_current(|canonical, expected| {
                repair_digest(canonical).as_str() == expected.as_str()
            })
            .map_err(|error| match error {
                RepairContractError::InvalidDigest => {
                    WenlanError::Validation("repair_manifest_digest_mismatch".to_string())
                }
                _ => WenlanError::Validation(error.to_string()),
            })
    }

    pub(crate) fn persist_prepared(
        &self,
        manifest: &RepairManifest,
        rollback_bytes: &[u8],
    ) -> Result<(), WenlanError> {
        ensure_private_dir(&self.root)?;
        let final_dir = self.manifest_dir(manifest.manifest_id())?;
        if final_dir.exists() {
            return Err(WenlanError::Conflict("repair_manifest_exists".to_string()));
        }
        let temp_dir = self.root.join(format!(
            ".{}.tmp-{}",
            manifest.manifest_id(),
            Uuid::new_v4()
        ));
        fs::create_dir(&temp_dir)?;
        set_private_dir_permissions(&temp_dir)?;
        let result = (|| {
            write_private_file(
                &temp_dir.join(manifest.rollback().relative_path()),
                rollback_bytes,
            )?;
            write_private_file(
                &temp_dir.join(MANIFEST_FILE),
                &serde_json::to_vec_pretty(manifest)?,
            )?;
            sync_dir(&temp_dir)?;
            fs::rename(&temp_dir, &final_dir)?;
            sync_dir(&self.root)?;
            Ok::<(), WenlanError>(())
        })();
        if result.is_err() && temp_dir.exists() {
            let _ = fs::remove_dir_all(&temp_dir);
        }
        result
    }

    fn load_rollback(
        &self,
        manifest: &RepairManifest,
    ) -> Result<(StoredRollbackArtifact, Vec<u8>), WenlanError> {
        let path = self
            .manifest_dir(manifest.manifest_id())?
            .join(manifest.rollback().relative_path());
        let bytes = read_bounded_file(&path, REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES)?;
        if repair_digest(&bytes) != *manifest.rollback().digest() {
            return Err(WenlanError::Validation(
                "repair_rollback_digest_mismatch".to_string(),
            ));
        }
        let stored = StoredRepairRollbackArtifact::from_slice(&bytes)?;
        let rollback = match stored {
            StoredRepairRollbackArtifact::V1(rollback) => StoredRollbackArtifact {
                format_version: rollback.format_version(),
                table: rollback.table().to_string(),
                source_id: rollback.source_id().to_string(),
                columns: rollback.columns().to_vec(),
                rows: rollback.rows().to_vec(),
            },
            StoredRepairRollbackArtifact::V2(rollback) => match rollback.payload() {
                RepairRollbackPayloadV2::SingleTable {
                    table,
                    source_id,
                    columns,
                    rows,
                } => StoredRollbackArtifact {
                    format_version: rollback.format_version(),
                    table: table.clone(),
                    source_id: source_id.clone(),
                    columns: columns.clone(),
                    rows: rows.clone(),
                },
                RepairRollbackPayloadV2::RenamePageTitle { .. }
                | RepairRollbackPayloadV2::CompleteEntityExtraction { .. } => {
                    return Err(WenlanError::Validation(
                        "repair_rollback_writer_not_implemented".to_string(),
                    ))
                }
            },
        };
        let expected_format_version = if matches!(
            manifest.writer(),
            RepairWriter::RenamePageTitle | RepairWriter::CompleteEntityExtraction
        ) {
            REPAIR_ROLLBACK_FORMAT_VERSION
        } else {
            LEGACY_ROLLBACK_FORMAT_VERSION
        };
        if rollback.format_version != expected_format_version
            || rollback.rows.is_empty()
            || !rollback_matches_target(&rollback, manifest.target())
        {
            return Err(WenlanError::Validation(
                "repair_rollback_mismatch".to_string(),
            ));
        }
        Ok((rollback, bytes))
    }

    fn load_complete_entity_extraction_rollback(
        &self,
        manifest: &RepairManifest,
    ) -> Result<RepairRollbackPayloadV2, WenlanError> {
        if manifest.writer() != RepairWriter::CompleteEntityExtraction
            || manifest.rollback().format_version() != REPAIR_ROLLBACK_FORMAT_VERSION
        {
            return Err(WenlanError::Validation(
                "repair_rollback_writer_mismatch".to_string(),
            ));
        }
        let path = self
            .manifest_dir(manifest.manifest_id())?
            .join(manifest.rollback().relative_path());
        let bytes = read_bounded_file(&path, REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES)?;
        if repair_digest(&bytes) != *manifest.rollback().digest() {
            return Err(WenlanError::Validation(
                "repair_rollback_digest_mismatch".to_string(),
            ));
        }
        let StoredRepairRollbackArtifact::V2(rollback) =
            StoredRepairRollbackArtifact::from_slice(&bytes)?
        else {
            return Err(WenlanError::Validation(
                "repair_rollback_mismatch".to_string(),
            ));
        };
        let payload = rollback.payload().clone();
        if !matches!(
            &payload,
            RepairRollbackPayloadV2::CompleteEntityExtraction { memory_id, .. }
                if matches!(
                    manifest.target(),
                    RepairTarget::MemoryEntityExtraction {
                        memory_id: target_memory_id,
                        step: RepairEnrichmentStep::EntityExtract,
                        ..
                    } if memory_id == target_memory_id
                )
        ) {
            return Err(WenlanError::Validation(
                "repair_rollback_mismatch".to_string(),
            ));
        }
        Ok(payload)
    }

    fn load_rename_page_title_rollback(
        &self,
        manifest: &RepairManifest,
    ) -> Result<RepairRollbackPayloadV2, WenlanError> {
        if manifest.writer() != RepairWriter::RenamePageTitle
            || manifest.rollback().format_version() != REPAIR_ROLLBACK_FORMAT_VERSION
        {
            return Err(WenlanError::Validation(
                "repair_rollback_writer_mismatch".to_string(),
            ));
        }
        let path = self
            .manifest_dir(manifest.manifest_id())?
            .join(manifest.rollback().relative_path());
        let bytes = read_bounded_file(&path, REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES)?;
        if repair_digest(&bytes) != *manifest.rollback().digest() {
            return Err(WenlanError::Validation(
                "repair_rollback_digest_mismatch".to_string(),
            ));
        }
        let StoredRepairRollbackArtifact::V2(rollback) =
            StoredRepairRollbackArtifact::from_slice(&bytes)?
        else {
            return Err(WenlanError::Validation(
                "repair_rollback_mismatch".to_string(),
            ));
        };
        let payload = rollback.payload().clone();
        if !matches!(
            &payload,
            RepairRollbackPayloadV2::RenamePageTitle { page_id, .. }
                if matches!(
                    manifest.target(),
                    RepairTarget::PageProjection {
                        page_id: target_page_id,
                        ..
                    } if page_id == target_page_id
                )
        ) {
            return Err(WenlanError::Validation(
                "repair_rollback_mismatch".to_string(),
            ));
        }
        Ok(payload)
    }

    fn lock_manifest_operation(
        &self,
        manifest_id: &str,
    ) -> Result<ManifestOperationLock, WenlanError> {
        let path = self.manifest_dir(manifest_id)?.join(OPERATION_LOCK_FILE);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        set_private_file_permissions(&path)?;
        file.try_lock_exclusive().map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                WenlanError::Conflict("repair_operation_in_progress".to_string())
            } else {
                WenlanError::Io(error)
            }
        })?;
        Ok(ManifestOperationLock { _file: file })
    }

    fn lock_tag_record_set(
        &self,
        manifest: &RepairManifest,
    ) -> Result<Option<ManifestOperationLock>, WenlanError> {
        let Some(record_set) = manifest.post_assertions().target_record_set() else {
            return Ok(None);
        };
        ensure_private_dir(&self.root)?;
        let path = self.root.join(format!(
            "{TAG_RECORD_SET_LOCK_PREFIX}{}.lock",
            record_set.digest().as_str()
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        set_private_file_permissions(&path)?;
        file.try_lock_exclusive().map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                WenlanError::Conflict("repair_tag_set_in_progress".to_string())
            } else {
                WenlanError::Io(error)
            }
        })?;
        Ok(Some(ManifestOperationLock { _file: file }))
    }

    fn verified_tag_targets(
        &self,
        manifest: &RepairManifest,
    ) -> Result<Vec<RepairTarget>, WenlanError> {
        let Some(record_set) = manifest.post_assertions().target_record_set() else {
            return Ok(Vec::new());
        };
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(WenlanError::Io(error)),
        };
        let mut targets = Vec::new();
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir()
                || !entry.path().join(MANIFEST_FILE).is_file()
                || !entry.path().join(VERIFICATION_RECEIPT_FILE).is_file()
            {
                continue;
            }
            let manifest_id = entry.file_name().into_string().map_err(|_| {
                WenlanError::Validation("invalid_repair_manifest_directory".to_string())
            })?;
            if manifest_id == manifest.manifest_id() {
                continue;
            }
            let candidate = self.load_manifest(&manifest_id)?;
            if candidate.writer() != RepairWriter::DeleteTagRow
                || candidate.post_assertions().target_record_set() != Some(record_set)
            {
                continue;
            }
            let apply_receipt = self.load_apply_receipt(&candidate)?;
            self.load_verification_receipt(&candidate, &apply_receipt)?
                .ok_or_else(|| {
                    WenlanError::Validation("repair_tag_set_receipt_missing".to_string())
                })?;
            if !matches!(candidate.target(), RepairTarget::Tag { .. }) {
                return Err(WenlanError::Validation(
                    "repair_tag_set_manifest_invalid".to_string(),
                ));
            }
            targets.push(candidate.target().clone());
        }
        targets.sort_by(|left, right| tag_record_key(left).cmp(&tag_record_key(right)));
        if targets
            .windows(2)
            .any(|pair| tag_record_key(&pair[0]) == tag_record_key(&pair[1]))
        {
            return Err(WenlanError::Validation(
                "repair_tag_set_duplicate_receipt".to_string(),
            ));
        }
        Ok(targets)
    }

    fn begin_apply_receipt(&self, manifest_id: &str) -> Result<PendingApplyReceipt, WenlanError> {
        let manifest_dir = self.manifest_dir(manifest_id)?;
        let final_path = manifest_dir.join(APPLY_RECEIPT_FILE);
        if final_path.exists() {
            return Err(WenlanError::Conflict("repair_already_applied".to_string()));
        }
        let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending_path)
            .map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    WenlanError::Conflict("repair_apply_in_progress".to_string())
                } else {
                    WenlanError::Io(error)
                }
            })?;
        set_private_file_permissions(&pending_path)?;
        Ok(PendingApplyReceipt {
            file: Some(file),
            pending_path,
            final_path,
        })
    }

    fn persist_stale_page_projection_apply_journal(
        &self,
        manifest: &RepairManifest,
        rollback: &StoredRollbackArtifact,
    ) -> Result<(), WenlanError> {
        validate_stale_page_projection_apply_journal_rollback(manifest, rollback)?;
        let journal = StalePageProjectionApplyJournal::new(manifest, rollback.clone())?;
        let manifest_dir = self.manifest_dir(manifest.manifest_id())?;
        let final_path = manifest_dir.join(STALE_PAGE_PROJECTION_APPLY_JOURNAL_FILE);
        let pending_path = manifest_dir.join(STALE_PAGE_PROJECTION_APPLY_JOURNAL_PENDING_FILE);
        write_private_file(&pending_path, &serde_json::to_vec_pretty(&journal)?)?;
        sync_dir(&manifest_dir)?;
        if let Err(error) =
            publish_no_replace(&pending_path, &final_path, "repair_apply_journal_exists")
        {
            if pending_path.exists() {
                fs::remove_file(&pending_path)?;
                sync_dir(&manifest_dir)?;
            }
            return Err(error);
        }
        Ok(())
    }

    fn load_stale_page_projection_apply_journal(
        &self,
        manifest: &RepairManifest,
    ) -> Result<Option<StoredRollbackArtifact>, WenlanError> {
        let manifest_dir = self.manifest_dir(manifest.manifest_id())?;
        let final_path = manifest_dir.join(STALE_PAGE_PROJECTION_APPLY_JOURNAL_FILE);
        let pending_path = manifest_dir.join(STALE_PAGE_PROJECTION_APPLY_JOURNAL_PENDING_FILE);
        let read_journal = |path: &Path| {
            let bytes = read_bounded_file(path, REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES)?;
            let journal = serde_json::from_slice::<StalePageProjectionApplyJournal>(&bytes)
                .map_err(|_| WenlanError::Validation("repair_apply_journal_invalid".to_string()))?;
            let rollback = journal.verify(manifest)?;
            validate_stale_page_projection_apply_journal_rollback(manifest, &rollback)?;
            Ok::<StoredRollbackArtifact, WenlanError>(rollback)
        };
        if !final_path.exists() {
            if !pending_path.exists() {
                return Ok(None);
            }
            let rollback = read_journal(&pending_path)?;
            publish_no_replace(&pending_path, &final_path, "repair_apply_journal_exists")?;
            return Ok(Some(rollback));
        }
        let rollback = read_journal(&final_path)?;
        if pending_path.exists() {
            if read_journal(&pending_path)? != rollback {
                return Err(WenlanError::Validation(
                    "repair_apply_journal_invalid".to_string(),
                ));
            }
            fs::remove_file(&pending_path)?;
            sync_dir(&manifest_dir)?;
        }
        Ok(Some(rollback))
    }

    fn stale_page_projection_apply_journal_exists(
        &self,
        manifest_id: &str,
    ) -> Result<bool, WenlanError> {
        let manifest_dir = self.manifest_dir(manifest_id)?;
        if manifest_dir
            .join(STALE_PAGE_PROJECTION_APPLY_JOURNAL_FILE)
            .is_file()
        {
            return Ok(true);
        }
        Ok(manifest_dir
            .join(STALE_PAGE_PROJECTION_APPLY_JOURNAL_PENDING_FILE)
            .is_file())
    }

    fn clear_stale_page_projection_apply_journal(
        &self,
        manifest_id: &str,
    ) -> Result<(), WenlanError> {
        let manifest_dir = self.manifest_dir(manifest_id)?;
        let mut removed = false;
        for file in [
            STALE_PAGE_PROJECTION_APPLY_JOURNAL_PENDING_FILE,
            STALE_PAGE_PROJECTION_APPLY_JOURNAL_FILE,
        ] {
            let path = manifest_dir.join(file);
            if path.exists() {
                fs::remove_file(path)?;
                removed = true;
            }
        }
        if removed {
            sync_dir(&manifest_dir)?;
        }
        Ok(())
    }

    fn load_apply_receipt(
        &self,
        manifest: &RepairManifest,
    ) -> Result<RepairApplyReceipt, WenlanError> {
        let path = self
            .manifest_dir(manifest.manifest_id())?
            .join(APPLY_RECEIPT_FILE);
        let receipt = StoredRepairApplyReceipt::from_slice(&fs::read(path)?)?;
        verify_stored_apply_receipt(receipt, manifest)
    }

    fn persist_verification_receipt(
        &self,
        receipt: &RepairVerificationReceipt,
    ) -> Result<(), WenlanError> {
        let manifest_dir = self.manifest_dir(receipt.manifest_id())?;
        let final_path = manifest_dir.join(VERIFICATION_RECEIPT_FILE);
        if final_path.exists() {
            return Err(WenlanError::Conflict("repair_already_verified".to_string()));
        }
        let temp_path = manifest_dir.join(format!(
            ".{VERIFICATION_RECEIPT_FILE}.tmp-{}",
            Uuid::new_v4()
        ));
        let result = (|| {
            write_private_file(&temp_path, &serde_json::to_vec_pretty(receipt)?)?;
            publish_no_replace(&temp_path, &final_path, "repair_already_verified")?;
            Ok::<(), WenlanError>(())
        })();
        if result.is_err() && temp_path.exists() {
            let _ = fs::remove_file(temp_path);
        }
        result
    }

    fn load_verification_receipt(
        &self,
        manifest: &RepairManifest,
        apply_receipt: &RepairApplyReceipt,
    ) -> Result<Option<RepairVerificationReceipt>, WenlanError> {
        let path = self
            .manifest_dir(manifest.manifest_id())?
            .join(VERIFICATION_RECEIPT_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let receipt = StoredRepairVerificationReceipt::from_slice(&fs::read(path)?)?;
        verify_stored_verification_receipt(receipt, manifest, apply_receipt).map(Some)
    }

    /// Return every durably applied repair that still needs verification.
    ///
    /// Startup uses this to restore the daemon-owned writer fence after a
    /// process restart. Corrupt or mismatched durable artifacts fail closed
    /// instead of allowing background maintenance to resume.
    pub fn pending_verification_manifest_ids(&self) -> Result<Vec<String>, WenlanError> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(WenlanError::Io(error)),
        };
        let mut pending = Vec::new();
        for entry in entries {
            let entry = entry?;
            if !entry.file_type()?.is_dir() || !entry.path().join(MANIFEST_FILE).is_file() {
                continue;
            }
            let apply_path = entry.path().join(APPLY_RECEIPT_FILE);
            let pending_path = entry.path().join(APPLY_RECEIPT_PENDING_FILE);
            if !apply_path.is_file() && !pending_path.is_file() {
                continue;
            }
            let manifest_id = entry.file_name().into_string().map_err(|_| {
                WenlanError::Validation("invalid_repair_manifest_directory".to_string())
            })?;
            let manifest = self.load_manifest(&manifest_id)?;
            if apply_path.is_file() {
                let apply_receipt = self.load_apply_receipt(&manifest)?;
                if self
                    .load_verification_receipt(&manifest, &apply_receipt)?
                    .is_none()
                {
                    pending.push(manifest_id);
                }
                continue;
            }
            if pending_path.is_file() {
                pending.push(manifest_id);
            }
        }
        pending.sort();
        Ok(pending)
    }

    fn clear_pending_apply_receipt(&self, manifest_id: &str) -> Result<(), WenlanError> {
        let manifest_dir = self.manifest_dir(manifest_id)?;
        let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
        if pending_path.exists() {
            fs::remove_file(pending_path)?;
            sync_dir(&manifest_dir)?;
        }
        Ok(())
    }

    /// Validate and report whether a repair already has its durable terminal
    /// receipt. This keeps a lost HTTP verify response safely retryable after
    /// the in-memory writer fence has been released.
    pub fn has_completed_verification(&self, manifest_id: &str) -> Result<bool, WenlanError> {
        let manifest = self.load_manifest(manifest_id)?;
        let apply_path = self.manifest_dir(manifest_id)?.join(APPLY_RECEIPT_FILE);
        if !apply_path.is_file() {
            return Ok(false);
        }
        let apply_receipt = self.load_apply_receipt(&manifest)?;
        Ok(self
            .load_verification_receipt(&manifest, &apply_receipt)?
            .is_some())
    }
}

fn read_bounded_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, WenlanError> {
    let file = File::open(path)?;
    if file.metadata()?.len() > max_bytes {
        return Err(WenlanError::Validation(
            "repair_rollback_artifact_too_large".to_string(),
        ));
    }
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(WenlanError::Validation(
            "repair_rollback_artifact_too_large".to_string(),
        ));
    }
    Ok(bytes)
}

fn verify_stored_apply_receipt(
    receipt: StoredRepairApplyReceipt,
    manifest: &RepairManifest,
) -> Result<RepairApplyReceipt, WenlanError> {
    if receipt.manifest_id() != manifest.manifest_id()
        || receipt.manifest_digest().as_str() != manifest.manifest_digest().as_str()
    {
        return Err(WenlanError::Validation(
            "repair_apply_receipt_mismatch".to_string(),
        ));
    }
    receipt
        .verify_and_try_into_current(|canonical, expected| {
            repair_digest(canonical).as_str() == expected.as_str()
        })
        .map_err(|_| WenlanError::Validation("repair_apply_receipt_mismatch".to_string()))
}

fn verify_stored_verification_receipt(
    receipt: StoredRepairVerificationReceipt,
    manifest: &RepairManifest,
    apply_receipt: &RepairApplyReceipt,
) -> Result<RepairVerificationReceipt, WenlanError> {
    if receipt.manifest_id() != manifest.manifest_id()
        || receipt.manifest_digest().as_str() != manifest.manifest_digest().as_str()
        || receipt.apply_receipt_digest().as_str() != apply_receipt.receipt_digest().as_str()
    {
        return Err(WenlanError::Validation(
            "repair_verification_receipt_mismatch".to_string(),
        ));
    }
    receipt
        .verify_and_try_into_current(|canonical, expected| {
            repair_digest(canonical).as_str() == expected.as_str()
        })
        .map_err(|_| WenlanError::Validation("repair_verification_receipt_mismatch".to_string()))
}

struct ManifestOperationLock {
    _file: File,
}

struct PendingApplyReceipt {
    file: Option<File>,
    pending_path: PathBuf,
    final_path: PathBuf,
}

impl PendingApplyReceipt {
    fn prepare(&mut self, receipt: &RepairApplyReceipt) -> Result<(), WenlanError> {
        let bytes = serde_json::to_vec_pretty(receipt)?;
        let mut file = self.file.take().expect("pending apply receipt file");
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        if let Some(parent) = self.pending_path.parent() {
            sync_dir(parent)?;
        }
        Ok(())
    }

    fn publish(self) -> Result<(), WenlanError> {
        publish_no_replace(
            &self.pending_path,
            &self.final_path,
            "repair_already_applied",
        )
    }

    fn abort(mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(&self.pending_path);
    }

    fn retain(mut self) -> Result<(), WenlanError> {
        if let Some(file) = self.file.take() {
            file.sync_all()?;
            drop(file);
        }
        if let Some(parent) = self.pending_path.parent() {
            sync_dir(parent)?;
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct StalePageProjectionApplyJournalDraft<'a> {
    format_version: u16,
    manifest_id: &'a str,
    manifest_digest: &'a RepairDigest,
    rollback: &'a StoredRollbackArtifact,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StalePageProjectionApplyJournal {
    format_version: u16,
    manifest_id: String,
    manifest_digest: RepairDigest,
    rollback: StoredRollbackArtifact,
    journal_digest: RepairDigest,
}

impl StalePageProjectionApplyJournal {
    fn new(
        manifest: &RepairManifest,
        rollback: StoredRollbackArtifact,
    ) -> Result<Self, WenlanError> {
        let draft = StalePageProjectionApplyJournalDraft {
            format_version: STALE_PAGE_PROJECTION_APPLY_JOURNAL_FORMAT_VERSION,
            manifest_id: manifest.manifest_id(),
            manifest_digest: manifest.manifest_digest(),
            rollback: &rollback,
        };
        let journal_digest = repair_digest(&serde_json::to_vec(&draft)?);
        Ok(Self {
            format_version: draft.format_version,
            manifest_id: draft.manifest_id.to_string(),
            manifest_digest: draft.manifest_digest.clone(),
            rollback,
            journal_digest,
        })
    }

    fn verify(self, manifest: &RepairManifest) -> Result<StoredRollbackArtifact, WenlanError> {
        let draft = StalePageProjectionApplyJournalDraft {
            format_version: self.format_version,
            manifest_id: &self.manifest_id,
            manifest_digest: &self.manifest_digest,
            rollback: &self.rollback,
        };
        if self.format_version != STALE_PAGE_PROJECTION_APPLY_JOURNAL_FORMAT_VERSION
            || self.manifest_id != manifest.manifest_id()
            || self.manifest_digest != *manifest.manifest_digest()
            || repair_digest(&serde_json::to_vec(&draft)?) != self.journal_digest
        {
            return Err(WenlanError::Validation(
                "repair_apply_journal_invalid".to_string(),
            ));
        }
        Ok(self.rollback)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredRollbackArtifact {
    pub(crate) format_version: u16,
    pub(crate) table: String,
    pub(crate) source_id: String,
    pub(crate) columns: Vec<String>,
    pub(crate) rows: Vec<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyStoredRollbackArtifact {
    format_version: u16,
    table: String,
    source_id: String,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Serialize for StoredRollbackArtifact {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.format_version == 1 {
            return LegacyStoredRollbackArtifact {
                format_version: self.format_version,
                table: self.table.clone(),
                source_id: self.source_id.clone(),
                columns: self.columns.clone(),
                rows: self.rows.clone(),
            }
            .serialize(serializer);
        }
        if self.format_version != REPAIR_ROLLBACK_FORMAT_VERSION {
            return Err(serde::ser::Error::custom(
                "unsupported repair rollback format",
            ));
        }
        let payload = RepairRollbackPayloadV2::single_table(
            self.table.clone(),
            self.source_id.clone(),
            self.columns.clone(),
            self.rows.clone(),
        )
        .map_err(serde::ser::Error::custom)?;
        RepairRollbackV2::try_new(payload)
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for StoredRollbackArtifact {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let format_version = value
            .get("format_version")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| serde::de::Error::custom("missing repair rollback format"))?;
        if format_version == 1 {
            let rollback: LegacyStoredRollbackArtifact =
                serde_json::from_value(value).map_err(serde::de::Error::custom)?;
            return Ok(Self {
                format_version: rollback.format_version,
                table: rollback.table,
                source_id: rollback.source_id,
                columns: rollback.columns,
                rows: rollback.rows,
            });
        }
        let rollback: RepairRollbackV2 =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        match rollback.payload() {
            RepairRollbackPayloadV2::SingleTable {
                table,
                source_id,
                columns,
                rows,
            } => Ok(Self {
                format_version: rollback.format_version(),
                table: table.clone(),
                source_id: source_id.clone(),
                columns: columns.clone(),
                rows: rows.clone(),
            }),
            RepairRollbackPayloadV2::RenamePageTitle { .. }
            | RepairRollbackPayloadV2::CompleteEntityExtraction { .. } => Err(
                serde::de::Error::custom("aggregate repair rollback requires typed writer"),
            ),
        }
    }
}

fn rollback_matches_target(rollback: &StoredRollbackArtifact, target: &RepairTarget) -> bool {
    match target {
        RepairTarget::Memory { source_id, .. } => {
            rollback.table == "memories" && rollback.source_id == *source_id
        }
        RepairTarget::MemoryEntityLink {
            memory_id,
            entity_id,
            ..
        } => {
            rollback.table == "memory_entities"
                && rollback.columns == ["memory_id", "entity_id"]
                && rollback.rows == [vec![memory_id.clone(), entity_id.clone()]]
                && serde_json::from_str::<Vec<String>>(&rollback.source_id)
                    .is_ok_and(|key| key == [memory_id.clone(), entity_id.clone()])
        }
        RepairTarget::MemoryEntityExtraction { .. } => false,
        RepairTarget::Tag {
            source,
            source_id,
            tag,
            ..
        } => {
            rollback.table == "document_tags"
                && serde_json::from_str::<Vec<String>>(&rollback.source_id)
                    .is_ok_and(|key| key == [source.clone(), source_id.clone(), tag.clone()])
        }
        RepairTarget::PageLink {
            source_page_id,
            label_key,
            ..
        } => {
            rollback.table == "page_links"
                && serde_json::from_str::<Vec<String>>(&rollback.source_id)
                    .is_ok_and(|key| key == [source_page_id.clone(), label_key.clone()])
        }
        RepairTarget::Page { page_id, .. } => {
            rollback.table == "pages"
                && rollback.source_id == *page_id
                && rollback.rows.len() == 1
                && !rollback.columns.is_empty()
        }
        RepairTarget::PageProjection { page_id, .. } => {
            matches!(
                rollback.table.as_str(),
                PAGE_PROJECTION_ROLLBACK_TABLE
                    | PAGE_PROJECTION_ROLLBACK_TABLE_V2
                    | STALE_PAGE_PROJECTION_ROLLBACK_TABLE
            ) && rollback.source_id == *page_id
        }
    }
}

fn validate_stale_page_projection_apply_journal_rollback(
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
) -> Result<(), WenlanError> {
    let (source_path, quarantine_path) = match (manifest.writer(), manifest.mutation()) {
        (
            RepairWriter::QuarantineStalePageProjection,
            RepairMutation::QuarantineStalePageProjection {
                source_path,
                quarantine_path,
            },
        ) => (source_path, quarantine_path),
        _ => {
            return Err(WenlanError::Validation(
                "repair_apply_journal_writer_mismatch".to_string(),
            ))
        }
    };
    let (rollback_source, rollback_quarantine) = stale_page_projection_paths(rollback)?;
    if !rollback_matches_target(rollback, manifest.target())
        || rollback_source.as_str() != source_path.as_str()
        || rollback_quarantine.as_str() != quarantine_path.as_str()
        || target_receipt(rollback)? != *manifest.expected_state().canonical_receipt()
    {
        return Err(WenlanError::Validation(
            "repair_apply_journal_target_mismatch".to_string(),
        ));
    }
    Ok(())
}

struct ResolvedTarget {
    source_id: String,
    memory_type: Option<String>,
    space: Option<String>,
    version: Option<i64>,
    evidence_id: LintDigest,
}

pub fn semantic_record_digest(kind: &str, durable_id: &str) -> LintDigest {
    crate::lint::semantic_record_digest(kind, durable_id)
}

pub async fn prepare_memory_reclassification(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: PrepareRepairRequest,
    now_epoch: i64,
) -> Result<RepairManifest, WenlanError> {
    prepare_memory_reclassification_with_pages(db, store, request, None, now_epoch).await
}

pub async fn prepare_memory_reclassification_with_pages(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: PrepareRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
) -> Result<RepairManifest, WenlanError> {
    ensure_repair_artifacts_supported()?;
    if now_epoch <= 0 {
        return Err(WenlanError::Validation(
            "invalid_repair_prepared_at".to_string(),
        ));
    }
    if matches!(request.choice(), RepairChoice::RenamePageTitle { .. }) {
        return prepare_rename_page_title(db, store, request, page_root, now_epoch).await;
    }
    if matches!(
        request.choice(),
        RepairChoice::CompleteEntityExtraction { .. }
    ) {
        return prepare_complete_entity_extraction(db, store, request, page_root, now_epoch).await;
    }
    validate_selected_finding(&request)?;
    let deep = request
        .deep_report()
        .ok_or_else(|| WenlanError::Validation("repair_deep_report_missing".to_string()))?;
    let selected_finding = request
        .selected_finding()
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?;
    let after_memory_type = request
        .after_memory_type()
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?;

    let snapshot = db.open_lint_snapshot().await.map_err(snapshot_error)?;
    validate_durable_scope(&snapshot, request.lint_scope(), deep.scope()).await?;
    let target = resolve_target(&snapshot, &request).await?;
    let (review_id, review_occurrence, review_owner_ids) =
        validate_reclassification_review_on_snapshot(
            &snapshot,
            selected_finding,
            &target.source_id,
        )
        .await?;
    let rollback = capture_rollback(&snapshot, &target.source_id).await?;
    let rollback_bytes = serde_json::to_vec_pretty(&rollback)?;
    let target_receipt = target_receipt(&rollback)?;
    let snapshot_receipt = snapshot.finish().await.map_err(snapshot_error)?;
    validate_source_receipts(&request, snapshot_receipt)?;
    validate_current_page_receipts(request.general_report(), Some(deep), page_root)
        .await
        .map_err(|error| match error {
            WenlanError::Conflict(_) => {
                WenlanError::Conflict("repair_source_reports_stale".to_string())
            }
            other => other,
        })?;
    if request.general_report().producer_receipt() != deep.producer_receipt() {
        return Err(WenlanError::Conflict(
            "repair_source_producers_mismatch".to_string(),
        ));
    }

    let after_memory_type = after_memory_type.to_string();
    let mutation =
        RepairMutation::try_reclassify(target.memory_type.as_deref(), after_memory_type.as_str())
            .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let target_scope = match target.space {
        Some(space) => RepairScope::registered(space),
        None => Ok(RepairScope::uncategorized()),
    }
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let repair_target = RepairTarget::memory(target.source_id, target_scope)
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let general = request.general_report();
    let agent_work_digest = deep
        .agent_work()
        .ok_or_else(|| WenlanError::Validation("repair_agent_work_missing".to_string()))?
        .work_digest()
        .clone();
    let source = RepairSource::try_new(
        request.lint_scope().clone(),
        deep.scope().clone(),
        selected_finding.clone(),
        general.snapshots().clone(),
        deep.snapshots().clone(),
        general.producer_receipt().clone(),
        deep.producer_receipt().clone(),
        agent_work_digest,
    )
    .and_then(|source| {
        source.try_with_review_binding(RepairReviewBinding::try_new(
            review_id,
            review_occurrence,
            review_owner_ids,
        )?)
    })
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let allowed_effects = RepairAllowedEffects::memory_type(repair_target.clone());
    let rollback_contract =
        RepairRollbackArtifact::try_new(ROLLBACK_FILE.to_string(), repair_digest(&rollback_bytes))
            .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let post_assertions = RepairPostAssertions::try_new(
        target.evidence_id,
        repair_check_baseline(general)?,
        repair_check_baseline(deep)?,
        vec![],
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let manifest_id = format!("repair_{}", Uuid::new_v4());
    let draft = RepairManifestDraft::try_new(
        manifest_id,
        now_epoch,
        source,
        repair_target,
        RepairExpectedState::try_new(target.version, target_receipt)
            .map_err(|error| WenlanError::Validation(error.to_string()))?,
        RepairWriter::ReclassifyMemory,
        mutation,
        allowed_effects,
        rollback_contract,
        post_assertions,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let manifest_digest = repair_digest(&draft.canonical_bytes()?);
    let manifest = RepairManifest::try_new(draft, manifest_digest)
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    store.persist_prepared(&manifest, &rollback_bytes)?;
    Ok(manifest)
}

async fn prepare_rename_page_title(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: PrepareRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
) -> Result<RepairManifest, WenlanError> {
    let RepairChoice::RenamePageTitle {
        review_id,
        page_id,
        before_title,
        after_title,
    } = request.choice()
    else {
        return Err(WenlanError::Validation(
            "unsupported_repair_finding".to_string(),
        ));
    };
    let page_root = page_root.ok_or_else(|| {
        WenlanError::Validation("page projection repair root unavailable".to_string())
    })?;
    let general = request.general_report();
    validate_rename_page_title_lint_finding(general)?;

    let snapshot = db.open_lint_snapshot().await.map_err(snapshot_error)?;
    validate_durable_scope(&snapshot, request.lint_scope(), general.scope()).await?;
    let (page_columns, before_page_row, target_scope, version) =
        capture_rename_page_row_on_snapshot(&snapshot, page_id, before_title).await?;
    validate_rename_page_title_collision_on_snapshot(
        &snapshot,
        page_id,
        before_title,
        after_title,
        target_scope.space(),
    )
    .await?;
    let occurrence =
        validate_rename_page_title_review_on_snapshot(&snapshot, review_id, page_id).await?;
    let (projection_target_path, projection_entries) =
        crate::export::knowledge::KnowledgeProjectionWrite::with_projection_lock(
            page_root,
            |projection| projection.capture_rename_page_projection(page_id),
        )?;
    let mut embedding_rows = snapshot
        .query(
            "SELECT summary,content FROM pages WHERE id=?1 LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(page_id.clone())]),
        )
        .await
        .map_err(snapshot_error)?;
    let embedding_row = embedding_rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let summary = embedding_row
        .get::<Option<String>>(0)
        .map_err(database_error)?;
    let content = embedding_row.get::<String>(1).map_err(database_error)?;
    if embedding_rows
        .next()
        .await
        .map_err(snapshot_error)?
        .is_some()
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    drop(embedding_rows);
    let embedding_input =
        crate::pages::page_embedding_text(after_title, summary.as_deref(), &content);
    let mut embeddings = db.generate_embeddings(&[embedding_input])?;
    if embeddings.len() != 1 {
        return Err(WenlanError::Validation(
            "repair_page_embedding_invalid".to_string(),
        ));
    }
    let after_embedding_hex = encode_page_title_embedding(
        embeddings
            .pop()
            .expect("exactly one embedding was validated"),
    )?;
    let rollback = RepairRollbackPayloadV2::rename_page_title(
        page_id.clone(),
        page_columns,
        before_page_row,
        projection_target_path,
        projection_entries,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let rollback_v2 = RepairRollbackV2::try_new(rollback.clone())
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let rollback_bytes = serde_json::to_vec_pretty(&rollback_v2)?;
    if u64::try_from(rollback_bytes.len()).unwrap_or(u64::MAX) > REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES
    {
        return Err(WenlanError::Validation(
            "repair_rollback_artifact_too_large".to_string(),
        ));
    }
    let target_receipt = rename_page_title_receipt(&rollback)?;
    let snapshot_receipt = snapshot.finish().await.map_err(snapshot_error)?;
    let reports = request
        .deep_report()
        .map_or_else(|| vec![general], |deep| vec![general, deep]);
    validate_report_source_receipts(&reports, snapshot_receipt)?;
    validate_current_page_receipts(general, request.deep_report(), Some(page_root))
        .await
        .map_err(|error| match error {
            WenlanError::Conflict(_) => {
                WenlanError::Conflict("repair_source_reports_stale".to_string())
            }
            other => other,
        })?;
    if request
        .deep_report()
        .is_some_and(|deep| general.producer_receipt() != deep.producer_receipt())
    {
        return Err(WenlanError::Conflict(
            "repair_source_producers_mismatch".to_string(),
        ));
    }
    let check = general
        .checks()
        .iter()
        .find(|check| check.check_id() == "pages.duplicate_active_titles")
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?;
    let source = match request.deep_report() {
        Some(deep) => RepairSource::try_new_deterministic(
            request.lint_scope().clone(),
            general.scope().clone(),
            check.check_id().to_string(),
            check.evidence().to_vec(),
            general.snapshots().clone(),
            deep.snapshots().clone(),
            general.producer_receipt().clone(),
            deep.producer_receipt().clone(),
        ),
        None => RepairSource::try_new_general_only_deterministic(
            request.lint_scope().clone(),
            general.scope().clone(),
            check.check_id().to_string(),
            check.evidence().to_vec(),
            general.snapshots().clone(),
            general.producer_receipt().clone(),
        ),
    }
    .and_then(|source| {
        source.try_with_review_binding(RepairReviewBinding::try_new(
            review_id.clone(),
            occurrence.clone(),
            vec![page_id.clone()],
        )?)
    })
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let target = RepairTarget::page_projection(page_id.clone(), target_scope)
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let post_assertions = match request.deep_report() {
        Some(deep) => RepairPostAssertions::try_new_for_check(
            check.check_id().to_string(),
            lint_digest_from_repair_digest(&occurrence)?,
            repair_check_baseline(general)?,
            repair_check_baseline(deep)?,
            if deep
                .checks()
                .iter()
                .any(|candidate| candidate.check_id() == check.check_id())
            {
                vec![check.check_id().to_string()]
            } else {
                vec![]
            },
            vec![],
        ),
        None => RepairPostAssertions::try_new_general_only_for_check(
            check.check_id().to_string(),
            lint_digest_from_repair_digest(&occurrence)?,
            repair_check_baseline(general)?,
            vec![],
        ),
    }
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let rollback_contract = RepairRollbackArtifact::try_new_v2(
        "rollback-v2.json".to_string(),
        repair_digest(&rollback_bytes),
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let draft = RepairManifestDraft::try_new(
        format!("repair_{}", Uuid::new_v4()),
        now_epoch,
        source,
        target.clone(),
        RepairExpectedState::try_new(Some(version), target_receipt)
            .map_err(|error| WenlanError::Validation(error.to_string()))?,
        RepairWriter::RenamePageTitle,
        RepairMutation::rename_page_title(
            before_title.clone(),
            after_title.clone(),
            after_embedding_hex,
        )
        .map_err(|error| WenlanError::Validation(error.to_string()))?,
        RepairAllowedEffects::page_title_rename(target),
        rollback_contract,
        post_assertions,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let manifest = RepairManifest::try_new(draft.clone(), repair_digest(&draft.canonical_bytes()?))
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    store.persist_prepared(&manifest, &rollback_bytes)?;
    Ok(manifest)
}

async fn prepare_complete_entity_extraction(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: PrepareRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
) -> Result<RepairManifest, WenlanError> {
    let RepairChoice::CompleteEntityExtraction {
        review_id,
        memory_id,
        entity_ids,
    } = request.choice()
    else {
        return Err(WenlanError::Validation(
            "unsupported_repair_finding".to_string(),
        ));
    };
    let general = request.general_report();
    validate_entity_extraction_lint_finding(general, memory_id)?;

    let snapshot = db.open_lint_snapshot().await.map_err(snapshot_error)?;
    validate_durable_scope(&snapshot, request.lint_scope(), general.scope()).await?;
    validate_entity_extraction_evidence_on_snapshot(
        &snapshot,
        general,
        request.lint_scope(),
        memory_id,
    )
    .await?;
    let (rollback, target_scope) =
        capture_complete_entity_extraction_on_snapshot(&snapshot, memory_id).await?;
    let owner_ids = complete_entity_extraction_owner_ids(memory_id);
    let occurrence =
        validate_complete_entity_extraction_review_on_snapshot(&snapshot, review_id, &owner_ids)
            .await?;
    validate_selected_entities_on_snapshot(&snapshot, entity_ids, target_scope.space()).await?;
    let RepairRollbackPayloadV2::CompleteEntityExtraction {
        enrichment_status, ..
    } = &rollback
    else {
        unreachable!("entity extraction capture returns typed payload")
    };
    if enrichment_status != "failed" {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let rollback_v2 = RepairRollbackV2::try_new(rollback.clone())
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let rollback_bytes = serde_json::to_vec_pretty(&rollback_v2)?;
    if u64::try_from(rollback_bytes.len()).unwrap_or(u64::MAX) > REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES
    {
        return Err(WenlanError::Validation(
            "repair_rollback_artifact_too_large".to_string(),
        ));
    }
    let target_receipt = complete_entity_extraction_receipt(&rollback)?;
    let snapshot_receipt = snapshot.finish().await.map_err(snapshot_error)?;
    let reports = request
        .deep_report()
        .map_or_else(|| vec![general], |deep| vec![general, deep]);
    validate_report_source_receipts(&reports, snapshot_receipt)?;
    validate_current_page_receipts(general, request.deep_report(), page_root)
        .await
        .map_err(|error| match error {
            WenlanError::Conflict(_) => {
                WenlanError::Conflict("repair_source_reports_stale".to_string())
            }
            other => other,
        })?;
    if request
        .deep_report()
        .is_some_and(|deep| general.producer_receipt() != deep.producer_receipt())
    {
        return Err(WenlanError::Conflict(
            "repair_source_producers_mismatch".to_string(),
        ));
    }

    let check = general
        .checks()
        .iter()
        .find(|check| check.check_id() == "memories.enrichment_failures")
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?;
    let source = match request.deep_report() {
        Some(deep) => RepairSource::try_new_deterministic(
            request.lint_scope().clone(),
            general.scope().clone(),
            check.check_id().to_string(),
            check.evidence().to_vec(),
            general.snapshots().clone(),
            deep.snapshots().clone(),
            general.producer_receipt().clone(),
            deep.producer_receipt().clone(),
        ),
        None => RepairSource::try_new_general_only_deterministic(
            request.lint_scope().clone(),
            general.scope().clone(),
            check.check_id().to_string(),
            check.evidence().to_vec(),
            general.snapshots().clone(),
            general.producer_receipt().clone(),
        ),
    }
    .and_then(|source| {
        source.try_with_review_binding(RepairReviewBinding::try_new(
            review_id.clone(),
            occurrence.clone(),
            owner_ids,
        )?)
    })
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let target = RepairTarget::memory_entity_extraction(
        memory_id.clone(),
        RepairEnrichmentStep::EntityExtract,
        entity_ids.clone(),
        target_scope,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let post_assertions = match request.deep_report() {
        Some(deep) => RepairPostAssertions::try_new_for_check(
            check.check_id().to_string(),
            lint_digest_from_repair_digest(&occurrence)?,
            repair_check_baseline(general)?,
            repair_check_baseline(deep)?,
            if deep
                .checks()
                .iter()
                .any(|candidate| candidate.check_id() == check.check_id())
            {
                vec![check.check_id().to_string()]
            } else {
                vec![]
            },
            vec![],
        ),
        None => RepairPostAssertions::try_new_general_only_for_check(
            check.check_id().to_string(),
            lint_digest_from_repair_digest(&occurrence)?,
            repair_check_baseline(general)?,
            vec![],
        ),
    }
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let rollback_contract = RepairRollbackArtifact::try_new_v2(
        "rollback-v2.json".to_string(),
        repair_digest(&rollback_bytes),
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let draft = RepairManifestDraft::try_new(
        format!("repair_{}", Uuid::new_v4()),
        now_epoch,
        source,
        target.clone(),
        RepairExpectedState::try_new(None, target_receipt)
            .map_err(|error| WenlanError::Validation(error.to_string()))?,
        RepairWriter::CompleteEntityExtraction,
        RepairMutation::complete_entity_extraction(entity_ids.clone())
            .map_err(|error| WenlanError::Validation(error.to_string()))?,
        RepairAllowedEffects::complete_entity_extraction(target),
        rollback_contract,
        post_assertions,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let manifest = RepairManifest::try_new(draft.clone(), repair_digest(&draft.canonical_bytes()?))
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
    store.persist_prepared(&manifest, &rollback_bytes)?;
    Ok(manifest)
}

pub(crate) fn repair_check_baseline(
    report: &wenlan_types::lint::LintReport,
) -> Result<Vec<RepairCheckBaseline>, WenlanError> {
    report
        .checks()
        .iter()
        .map(|check| {
            let affected_records = lint_check_affected_records(check)?;
            let baseline = match affected_records {
                Some(affected_records) => RepairCheckBaseline::try_new_current(
                    check.check_id().to_string(),
                    check.outcome(),
                    check.gate_effect(),
                    affected_records,
                    check.evidence().to_vec(),
                ),
                None => RepairCheckBaseline::try_new(
                    check.check_id().to_string(),
                    check.outcome(),
                    check.gate_effect(),
                    check.evidence().to_vec(),
                ),
            };
            baseline.map_err(|error| WenlanError::Validation(error.to_string()))
        })
        .collect()
}

fn lint_check_affected_records(check: &LintCheckResult) -> Result<Option<u64>, WenlanError> {
    let mut affected_records = None;
    for metric in check
        .metrics()
        .iter()
        .filter(|metric| metric.code() == LintMetricCode::AffectedRecords)
    {
        if affected_records.is_some() {
            return Err(WenlanError::Validation(
                "repair_affected_records_metric_invalid".to_string(),
            ));
        }
        let LintMetricValue::Count { value } = metric.value() else {
            return Err(WenlanError::Validation(
                "repair_affected_records_metric_invalid".to_string(),
            ));
        };
        affected_records = Some(*value);
    }
    Ok(affected_records)
}

fn tag_record_key(target: &RepairTarget) -> Option<(&str, &str, &str)> {
    match target {
        RepairTarget::Tag {
            source,
            source_id,
            tag,
            ..
        } => Some((source.as_str(), source_id.as_str(), tag.as_str())),
        _ => None,
    }
}

fn tag_record_set_from_targets(
    targets: &[RepairTarget],
) -> Result<BTreeSet<(String, String, String)>, WenlanError> {
    let mut records = BTreeSet::new();
    for target in targets {
        let (source, source_id, tag) = tag_record_key(target)
            .ok_or_else(|| WenlanError::Validation("repair_tag_set_target_invalid".to_string()))?;
        if !records.insert((source.to_string(), source_id.to_string(), tag.to_string())) {
            return Err(WenlanError::Validation(
                "repair_tag_set_target_duplicate".to_string(),
            ));
        }
    }
    Ok(records)
}

fn tag_record_set_baseline_from_records(
    records: &BTreeSet<(String, String, String)>,
) -> Result<RepairRecordSetBaseline, WenlanError> {
    #[derive(Serialize)]
    struct DigestInput<'a> {
        contract: &'static str,
        records: &'a BTreeSet<(String, String, String)>,
    }

    let record_count = u64::try_from(records.len())
        .map_err(|_| WenlanError::Validation("repair_tag_set_too_large".to_string()))?;
    let digest = repair_digest(&serde_json::to_vec(&DigestInput {
        contract: "wenlan-repair-invalid-tag-set-v1",
        records,
    })?);
    RepairRecordSetBaseline::try_new(record_count, digest)
        .map_err(|error| WenlanError::Validation(error.to_string()))
}

pub(crate) fn tag_record_set_baseline(
    targets: &[RepairTarget],
) -> Result<RepairRecordSetBaseline, WenlanError> {
    tag_record_set_baseline_from_records(&tag_record_set_from_targets(targets)?)
}

async fn current_invalid_tag_records(
    connection: &libsql::Connection,
) -> Result<BTreeSet<(String, String, String)>, WenlanError> {
    let mut rows = connection
        .query(
            "SELECT t.source,t.source_id,t.tag
               FROM document_tags t
              WHERE TRIM(t.tag)='' OR t.source NOT IN ('memory','page')
                 OR (t.source='memory' AND NOT EXISTS(
                    SELECT 1 FROM memories m WHERE m.source_id=t.source_id))
                 OR (t.source='page' AND NOT EXISTS(
                    SELECT 1 FROM pages p WHERE p.id=t.source_id))
              ORDER BY t.source,t.source_id,t.tag",
            (),
        )
        .await
        .map_err(database_error)?;
    let mut records = BTreeSet::new();
    while let Some(row) = rows.next().await.map_err(database_error)? {
        let record = (
            row.get::<String>(0).map_err(database_error)?,
            row.get::<String>(1).map_err(database_error)?,
            row.get::<String>(2).map_err(database_error)?,
        );
        if !records.insert(record) {
            return Err(WenlanError::Validation(
                "repair_tag_set_duplicate_row".to_string(),
            ));
        }
    }
    Ok(records)
}

pub(crate) async fn validate_tag_record_set_on_connection(
    connection: &libsql::Connection,
    manifest: &RepairManifest,
    prior_verified_targets: &[RepairTarget],
    include_current_target: bool,
) -> Result<(), WenlanError> {
    let Some(expected) = manifest.post_assertions().target_record_set() else {
        if manifest.writer() == RepairWriter::DeleteTagRow {
            return Err(WenlanError::Validation(
                "repair_tag_set_contract_missing".to_string(),
            ));
        }
        return Ok(());
    };
    if manifest.writer() != RepairWriter::DeleteTagRow {
        return Err(WenlanError::Validation(
            "repair_tag_set_contract_invalid".to_string(),
        ));
    }
    let mut records = current_invalid_tag_records(connection).await?;
    for target in prior_verified_targets
        .iter()
        .chain(include_current_target.then_some(manifest.target()))
    {
        let (source, source_id, tag) = tag_record_key(target)
            .ok_or_else(|| WenlanError::Validation("repair_tag_set_target_invalid".to_string()))?;
        if !records.insert((source.to_string(), source_id.to_string(), tag.to_string())) {
            return Err(WenlanError::Conflict("repair_tag_set_changed".to_string()));
        }
    }
    let current = tag_record_set_baseline_from_records(&records)?;
    if &current != expected {
        return Err(WenlanError::Conflict("repair_tag_set_changed".to_string()));
    }
    Ok(())
}

pub async fn apply_repair(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: ApplyRepairRequest,
    now_epoch: i64,
) -> Result<RepairApplyReceipt, WenlanError> {
    apply_repair_with_pages(db, store, request, None, now_epoch).await
}

pub async fn apply_repair_with_pages(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: ApplyRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
) -> Result<RepairApplyReceipt, WenlanError> {
    apply_repair_with_pages_inner(db, store, request, page_root, now_epoch, None).await
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepairApplyFault {
    ReclassificationRollbackFailure,
    EntityRollbackFailure,
}

#[cfg(test)]
async fn apply_repair_with_pages_with_forced_rollback_failure(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: ApplyRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
    writer: RepairWriter,
) -> Result<RepairApplyReceipt, WenlanError> {
    let fault = match writer {
        RepairWriter::ReclassifyMemory => RepairApplyFault::ReclassificationRollbackFailure,
        RepairWriter::CompleteEntityExtraction => RepairApplyFault::EntityRollbackFailure,
        _ => {
            return Err(WenlanError::Validation(
                "unsupported repair rollback failure injection".to_string(),
            ))
        }
    };
    apply_repair_with_pages_inner(db, store, request, page_root, now_epoch, Some(fault)).await
}

async fn apply_repair_with_pages_inner(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: ApplyRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
    fault: Option<RepairApplyFault>,
) -> Result<RepairApplyReceipt, WenlanError> {
    ensure_repair_artifacts_supported()?;
    if now_epoch <= 0 {
        return Err(WenlanError::Validation(
            "invalid_repair_applied_at".to_string(),
        ));
    }
    let manifest = store.load_manifest(request.manifest_id())?;
    if manifest.manifest_digest() != request.approved_manifest_digest() {
        return Err(WenlanError::Conflict(
            "repair_approval_mismatch".to_string(),
        ));
    }
    let _operation_lock = store.lock_manifest_operation(manifest.manifest_id())?;
    let _tag_record_set_lock = store.lock_tag_record_set(&manifest)?;
    if manifest.writer() == RepairWriter::RenamePageTitle {
        let page_root = page_root.ok_or_else(|| {
            WenlanError::Validation("page projection repair root unavailable".to_string())
        })?;
        return apply_rename_page_title(db, store, &manifest, page_root, now_epoch).await;
    }
    if manifest.writer() == RepairWriter::CompleteEntityExtraction {
        return apply_complete_entity_extraction(
            db,
            store,
            &manifest,
            now_epoch,
            fault == Some(RepairApplyFault::EntityRollbackFailure),
        )
        .await;
    }
    let prior_verified_tag_targets = store.verified_tag_targets(&manifest)?;
    let (rollback, _) = store.load_rollback(&manifest)?;
    if target_receipt(&rollback)? != *manifest.expected_state().canonical_receipt() {
        return Err(WenlanError::Validation(
            "repair_rollback_target_mismatch".to_string(),
        ));
    }
    if let Some(receipt) = recover_apply_receipt(db, store, &manifest, &rollback, page_root).await?
    {
        return Ok(receipt);
    }
    if manifest.writer() == RepairWriter::QuarantineStalePageProjection {
        return apply_quarantine_stale_page_projection(
            db, store, &manifest, &rollback, page_root, now_epoch,
        )
        .await;
    }
    let mut pending = store.begin_apply_receipt(manifest.manifest_id())?;
    let mut prepared_receipt = None;
    let prepare_receipt = |proof: &crate::post_write::RepairWriteProof| {
        let draft = RepairApplyReceiptDraft::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            now_epoch,
            proof.before_target_receipt().clone(),
            proof.after_target_receipt().clone(),
            proof.non_target_before().clone(),
            proof.non_target_after().clone(),
            proof.post_apply_db_digest().clone(),
            manifest.allowed_effects().clone(),
            manifest.writer(),
        )
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
        let receipt_digest = repair_digest(&draft.canonical_bytes()?);
        let receipt = RepairApplyReceipt::from_draft(draft, receipt_digest);
        pending.prepare(&receipt)?;
        prepared_receipt = Some(receipt);
        if fault == Some(RepairApplyFault::ReclassificationRollbackFailure) {
            return Err(WenlanError::VectorDb(
                "forced failure after pending reclassification receipt".to_string(),
            ));
        }
        Ok(())
    };
    let write_result = match manifest.writer() {
        RepairWriter::ReclassifyMemory => {
            let after_memory_type = MemoryType::from_str(manifest.mutation().after_memory_type())
                .map_err(WenlanError::Validation)?;
            if fault == Some(RepairApplyFault::ReclassificationRollbackFailure) {
                #[cfg(test)]
                {
                    crate::post_write::reclassify_memory_cas_with_forced_rollback_failure(
                        db,
                        manifest.target().memory_source_id(),
                        manifest.expected_state().canonical_receipt(),
                        manifest.target().scope().space(),
                        manifest.source().review_binding(),
                        after_memory_type,
                        prepare_receipt,
                    )
                    .await
                }
                #[cfg(not(test))]
                {
                    unreachable!("repair rollback failure injection is test-only")
                }
            } else {
                crate::post_write::reclassify_memory_cas(
                    db,
                    manifest.target().memory_source_id(),
                    manifest.expected_state().canonical_receipt(),
                    manifest.target().scope().space(),
                    manifest.source().review_binding(),
                    after_memory_type,
                    prepare_receipt,
                )
                .await
            }
        }
        RepairWriter::NormalizeMemorySourceAgent
        | RepairWriter::ClearMemorySupersedes
        | RepairWriter::UnstageOrphanRevision
        | RepairWriter::DeleteTagRow
        | RepairWriter::DeleteMemoryEntityLink
        | RepairWriter::BindPageLink
        | RepairWriter::ArchiveEmptySourcePage => {
            crate::post_write::apply_deterministic_repair_cas(
                db,
                &manifest,
                &prior_verified_tag_targets,
                prepare_receipt,
            )
            .await
        }
        RepairWriter::RegeneratePageProjection => {
            let page_root = page_root.ok_or_else(|| {
                WenlanError::Validation("page projection repair root unavailable".to_string())
            })?;
            crate::post_write::regenerate_page_projection_cas(
                db,
                &manifest,
                &rollback,
                page_root,
                prepare_receipt,
            )
            .await
        }
        RepairWriter::QuarantineStalePageProjection => Err(WenlanError::Validation(
            "repair_writer_dispatch_bypassed".to_string(),
        )),
        RepairWriter::RenamePageTitle | RepairWriter::CompleteEntityExtraction => Err(
            WenlanError::Validation("repair_writer_not_implemented".to_string()),
        ),
    };
    let proof = match write_result {
        Ok(proof) => proof,
        Err(error) => {
            let retain_pending = should_retain_pending_apply_receipt(&error);
            if retain_pending {
                if let Err(retain_error) = pending.retain() {
                    return Err(WenlanError::VectorDb(format!(
                        "{error}; repair pending receipt retention failed: {retain_error}"
                    )));
                }
            } else {
                pending.abort();
            }
            return Err(error);
        }
    };
    let receipt = prepared_receipt.ok_or_else(|| {
        WenlanError::VectorDb("repair_receipt_not_prepared_before_commit".to_string())
    })?;
    debug_assert_eq!(proof.after_target_receipt(), receipt.after_target_receipt());
    pending.publish()?;
    Ok(receipt)
}

async fn apply_quarantine_stale_page_projection(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
    now_epoch: i64,
) -> Result<RepairApplyReceipt, WenlanError> {
    let page_root = page_root.ok_or_else(|| {
        WenlanError::Validation("page projection repair root unavailable".to_string())
    })?;
    let pending = std::sync::Mutex::new(None::<PendingApplyReceipt>);
    let prepared_receipt = std::sync::Mutex::new(None::<RepairApplyReceipt>);
    let write_result =
        crate::post_write::quarantine_stale_page_projection_cas_with_apply_journal(
            db,
            manifest,
            rollback,
            page_root,
            |current| {
                store.persist_stale_page_projection_apply_journal(manifest, current)?;
                *pending.lock().map_err(|_| {
                    WenlanError::VectorDb("repair_apply_state_poisoned".to_string())
                })? = Some(store.begin_apply_receipt(manifest.manifest_id())?);
                Ok(())
            },
            |proof| {
                let draft = RepairApplyReceiptDraft::try_new(
                    manifest.manifest_id().to_string(),
                    manifest.manifest_digest().clone(),
                    now_epoch,
                    proof.before_target_receipt().clone(),
                    proof.after_target_receipt().clone(),
                    proof.non_target_before().clone(),
                    proof.non_target_after().clone(),
                    proof.post_apply_db_digest().clone(),
                    manifest.allowed_effects().clone(),
                    manifest.writer(),
                )
                .map_err(|error| WenlanError::Validation(error.to_string()))?;
                let receipt_digest = repair_digest(&draft.canonical_bytes()?);
                let receipt = RepairApplyReceipt::from_draft(draft, receipt_digest);
                pending
                    .lock()
                    .map_err(|_| WenlanError::VectorDb("repair_apply_state_poisoned".to_string()))?
                    .as_mut()
                    .ok_or_else(|| {
                        WenlanError::VectorDb(
                            "repair_pending_receipt_not_created_before_mutation".to_string(),
                        )
                    })?
                    .prepare(&receipt)?;
                *prepared_receipt.lock().map_err(|_| {
                    WenlanError::VectorDb("repair_apply_state_poisoned".to_string())
                })? = Some(receipt);
                Ok(())
            },
        )
        .await;
    let pending = pending
        .into_inner()
        .map_err(|_| WenlanError::VectorDb("repair_apply_state_poisoned".to_string()))?;
    let proof = match write_result {
        Ok(proof) => proof,
        Err(error) => {
            if let Some(pending) = pending {
                let retain_pending = should_retain_stale_page_projection_pending_receipt(
                    &error,
                    store.stale_page_projection_apply_journal_exists(manifest.manifest_id())?,
                );
                if retain_pending {
                    if let Err(retain_error) = pending.retain() {
                        return Err(WenlanError::VectorDb(format!(
                            "{error}; repair pending receipt retention failed: {retain_error}"
                        )));
                    }
                } else {
                    pending.abort();
                }
            }
            return Err(error);
        }
    };
    let receipt = prepared_receipt
        .into_inner()
        .map_err(|_| WenlanError::VectorDb("repair_apply_state_poisoned".to_string()))?
        .ok_or_else(|| {
            WenlanError::VectorDb("repair_receipt_not_prepared_before_commit".to_string())
        })?;
    debug_assert_eq!(proof.after_target_receipt(), receipt.after_target_receipt());
    pending
        .ok_or_else(|| {
            WenlanError::VectorDb("repair_pending_receipt_not_created_before_mutation".to_string())
        })?
        .publish()?;
    store.clear_stale_page_projection_apply_journal(manifest.manifest_id())?;
    Ok(receipt)
}

async fn apply_rename_page_title(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    page_root: &Path,
    now_epoch: i64,
) -> Result<RepairApplyReceipt, WenlanError> {
    let rollback = store.load_rename_page_title_rollback(manifest)?;
    if rename_page_title_receipt(&rollback)? != *manifest.expected_state().canonical_receipt() {
        return Err(WenlanError::Validation(
            "repair_rollback_target_mismatch".to_string(),
        ));
    }
    if let Some(receipt) =
        recover_rename_page_title_apply_receipt(db, store, manifest, page_root).await?
    {
        return Ok(receipt);
    }
    let mut pending = store.begin_apply_receipt(manifest.manifest_id())?;
    let mut prepared_receipt = None;
    let prepare_receipt = |proof: &crate::post_write::RepairWriteProof| {
        let draft = RepairApplyReceiptDraft::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            now_epoch,
            proof.before_target_receipt().clone(),
            proof.after_target_receipt().clone(),
            proof.non_target_before().clone(),
            proof.non_target_after().clone(),
            proof.post_apply_db_digest().clone(),
            manifest.allowed_effects().clone(),
            manifest.writer(),
        )
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
        let receipt_digest = repair_digest(&draft.canonical_bytes()?);
        let receipt = RepairApplyReceipt::from_draft(draft, receipt_digest);
        pending.prepare(&receipt)?;
        prepared_receipt = Some(receipt);
        Ok(())
    };
    let proof = match crate::post_write::rename_page_title_cas(
        db,
        manifest,
        &rollback,
        page_root,
        prepare_receipt,
    )
    .await
    {
        Ok(proof) => proof,
        Err(error) => {
            if should_retain_pending_apply_receipt(&error) {
                pending.retain()?;
            } else {
                pending.abort();
            }
            return Err(error);
        }
    };
    let receipt = prepared_receipt.ok_or_else(|| {
        WenlanError::VectorDb("repair_receipt_not_prepared_before_commit".to_string())
    })?;
    debug_assert_eq!(proof.after_target_receipt(), receipt.after_target_receipt());
    pending.publish()?;
    Ok(receipt)
}

async fn apply_complete_entity_extraction(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    now_epoch: i64,
    force_rollback_failure: bool,
) -> Result<RepairApplyReceipt, WenlanError> {
    let rollback = store.load_complete_entity_extraction_rollback(manifest)?;
    if complete_entity_extraction_receipt(&rollback)?
        != *manifest.expected_state().canonical_receipt()
    {
        return Err(WenlanError::Validation(
            "repair_rollback_target_mismatch".to_string(),
        ));
    }
    if let Some(receipt) =
        recover_complete_entity_extraction_apply_receipt(db, store, manifest).await?
    {
        return Ok(receipt);
    }
    let mut pending = store.begin_apply_receipt(manifest.manifest_id())?;
    let mut prepared_receipt = None;
    let prepare_receipt = |proof: &crate::post_write::RepairWriteProof| {
        let draft = RepairApplyReceiptDraft::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            now_epoch,
            proof.before_target_receipt().clone(),
            proof.after_target_receipt().clone(),
            proof.non_target_before().clone(),
            proof.non_target_after().clone(),
            proof.post_apply_db_digest().clone(),
            manifest.allowed_effects().clone(),
            manifest.writer(),
        )
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
        let receipt_digest = repair_digest(&draft.canonical_bytes()?);
        let receipt = RepairApplyReceipt::from_draft(draft, receipt_digest);
        pending.prepare(&receipt)?;
        prepared_receipt = Some(receipt);
        if force_rollback_failure {
            return Err(WenlanError::VectorDb(
                "forced failure after pending entity receipt".to_string(),
            ));
        }
        Ok(())
    };
    let write_result = if force_rollback_failure {
        #[cfg(test)]
        {
            crate::post_write::complete_entity_extraction_cas_with_forced_rollback_failure(
                db,
                manifest,
                &rollback,
                prepare_receipt,
            )
            .await
        }
        #[cfg(not(test))]
        {
            unreachable!("repair rollback failure injection is test-only")
        }
    } else {
        crate::post_write::complete_entity_extraction_cas(db, manifest, &rollback, prepare_receipt)
            .await
    };
    let proof = match write_result {
        Ok(proof) => proof,
        Err(error) => {
            if should_retain_pending_apply_receipt(&error) {
                pending.retain()?;
            } else {
                pending.abort();
            }
            return Err(error);
        }
    };
    let receipt = prepared_receipt.ok_or_else(|| {
        WenlanError::VectorDb("repair_receipt_not_prepared_before_commit".to_string())
    })?;
    debug_assert_eq!(proof.after_target_receipt(), receipt.after_target_receipt());
    pending.publish()?;
    Ok(receipt)
}

fn should_retain_pending_apply_receipt(error: &WenlanError) -> bool {
    matches!(
        error,
        WenlanError::Conflict(message) if message == "repair_apply_recovery_required"
    )
}

fn should_retain_stale_page_projection_pending_receipt(
    error: &WenlanError,
    apply_journal_exists: bool,
) -> bool {
    apply_journal_exists || should_retain_pending_apply_receipt(error)
}

async fn recover_rename_page_title_apply_receipt(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    page_root: &Path,
) -> Result<Option<RepairApplyReceipt>, WenlanError> {
    let manifest_dir = store.manifest_dir(manifest.manifest_id())?;
    let final_path = manifest_dir.join(APPLY_RECEIPT_FILE);
    if final_path.exists() {
        let receipt = store.load_apply_receipt(manifest)?;
        let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
        if pending_path.exists() {
            fs::remove_file(pending_path)?;
            sync_dir(&manifest_dir)?;
        }
        return Ok(Some(receipt));
    }
    let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
    if !pending_path.exists() {
        return Ok(None);
    }
    let pending = read_bounded_file(&pending_path, REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES)?;
    let parsed = StoredRepairApplyReceipt::from_slice(&pending)
        .ok()
        .and_then(|receipt| verify_stored_apply_receipt(receipt, manifest).ok());
    let rollback = store.load_rename_page_title_rollback(manifest)?;
    let page_id = match manifest.target() {
        RepairTarget::PageProjection { page_id, .. } => page_id,
        _ => {
            return Err(WenlanError::Validation(
                "page title repair target/writer mismatch".to_string(),
            ))
        }
    };
    let session = crate::export::knowledge::KnowledgeProjectionWrite::begin_owned_repair_session(
        page_root.to_path_buf(),
        db,
    )?;
    let projection = session.locked();
    let connection = db.conn.lock().await;
    connection
        .execute("BEGIN IMMEDIATE", ())
        .await
        .map_err(|error| WenlanError::VectorDb(format!("repair recovery begin: {error}")))?;
    let current =
        match capture_rename_page_title_on_connection(&connection, &projection, page_id).await {
            Ok(current) => current,
            Err(_) => {
                let _ = connection.execute("ROLLBACK", ()).await;
                return Err(WenlanError::Conflict(
                    "repair_apply_recovery_required".to_string(),
                ));
            }
        };
    #[allow(clippy::large_enum_variant)]
    enum Recovery {
        Publish(RepairApplyReceipt),
        Retry,
        RestoreRetry,
    }
    let recovery = if let Some(receipt) = parsed {
        if rename_page_title_receipt(&current)? == *receipt.after_target_receipt() {
            Recovery::Publish(receipt)
        } else if rename_page_database_matches(&current, &rollback)
            && rename_page_projection_matches_pre(&current, &rollback)
        {
            Recovery::Retry
        } else if rename_page_database_matches(&current, &rollback)
            && rename_page_projection_matches_post(&connection, manifest, &rollback, &current)
                .await?
        {
            Recovery::RestoreRetry
        } else {
            return recovery_required_with_rollback(&connection).await;
        }
    } else if rename_page_database_matches(&current, &rollback)
        && rename_page_projection_matches_pre(&current, &rollback)
    {
        Recovery::Retry
    } else if rename_page_database_matches(&current, &rollback)
        && rename_page_projection_matches_post(&connection, manifest, &rollback, &current).await?
    {
        Recovery::RestoreRetry
    } else {
        return recovery_required_with_rollback(&connection).await;
    };
    if matches!(recovery, Recovery::RestoreRetry) {
        let RepairRollbackPayloadV2::RenamePageTitle {
            projection_target_path,
            projection_entries,
            ..
        } = &rollback
        else {
            unreachable!("typed loader returned title rollback")
        };
        if projection
            .restore_rename_page_projection(projection_target_path, projection_entries)
            .is_err()
        {
            let _ = connection.execute("ROLLBACK", ()).await;
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
    }
    connection
        .execute("COMMIT", ())
        .await
        .map_err(|_| WenlanError::Conflict("repair_apply_recovery_required".to_string()))?;
    match recovery {
        Recovery::Publish(receipt) => {
            publish_no_replace(&pending_path, &final_path, "repair_already_applied")?;
            Ok(Some(receipt))
        }
        Recovery::Retry | Recovery::RestoreRetry => {
            store.clear_pending_apply_receipt(manifest.manifest_id())?;
            Ok(None)
        }
    }
}

async fn recovery_required_with_rollback<T>(
    connection: &libsql::Connection,
) -> Result<T, WenlanError> {
    let _ = connection.execute("ROLLBACK", ()).await;
    Err(WenlanError::Conflict(
        "repair_apply_recovery_required".to_string(),
    ))
}

fn rename_page_database_matches(
    current: &RepairRollbackPayloadV2,
    expected: &RepairRollbackPayloadV2,
) -> bool {
    matches!(
        (current, expected),
        (
            RepairRollbackPayloadV2::RenamePageTitle {
                page_id,
                page_columns,
                before_page_row,
                ..
            },
            RepairRollbackPayloadV2::RenamePageTitle {
                page_id: expected_page_id,
                page_columns: expected_columns,
                before_page_row: expected_row,
                ..
            }
        ) if page_id == expected_page_id
            && page_columns == expected_columns
            && before_page_row == expected_row
    )
}

fn rename_page_projection_matches_pre(
    current: &RepairRollbackPayloadV2,
    expected: &RepairRollbackPayloadV2,
) -> bool {
    matches!(
        (current, expected),
        (
            RepairRollbackPayloadV2::RenamePageTitle {
                projection_target_path,
                projection_entries,
                ..
            },
            RepairRollbackPayloadV2::RenamePageTitle {
                projection_target_path: expected_path,
                projection_entries: expected_entries,
                ..
            }
        ) if projection_target_path == expected_path
            && projection_entries == expected_entries
    )
}

async fn rename_page_projection_matches_post(
    connection: &libsql::Connection,
    manifest: &RepairManifest,
    rollback: &RepairRollbackPayloadV2,
    current: &RepairRollbackPayloadV2,
) -> Result<bool, WenlanError> {
    let (
        RepairRollbackPayloadV2::RenamePageTitle {
            page_id,
            projection_target_path,
            projection_entries: before_entries,
            ..
        },
        RepairRollbackPayloadV2::RenamePageTitle {
            projection_target_path: current_target_path,
            projection_entries: current_entries,
            ..
        },
        RepairMutation::RenamePageTitle { after_title, .. },
    ) = (rollback, current, manifest.mutation())
    else {
        return Ok(false);
    };
    if projection_target_path != current_target_path
        || before_entries.len() != 2
        || current_entries.len() != 2
    {
        return Ok(false);
    }
    let mut after_page = crate::post_write::page_on_connection(connection, page_id).await?;
    after_page.title = after_title.clone();
    after_page.version = after_page.version.saturating_add(1);
    let expected_markdown = crate::export::knowledge::render_markdown_for(&after_page);
    let current_target = current_entries
        .iter()
        .find(|entry| entry.relative_path() == projection_target_path)
        .and_then(|entry| hex::decode(entry.content_hex()).ok());
    if current_target.as_deref() != Some(expected_markdown.as_bytes()) {
        return Ok(false);
    }
    let before_state = before_entries
        .iter()
        .find(|entry| entry.relative_path() == ".wenlan/state.json")
        .and_then(|entry| hex::decode(entry.content_hex()).ok())
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let current_state = current_entries
        .iter()
        .find(|entry| entry.relative_path() == ".wenlan/state.json")
        .and_then(|entry| hex::decode(entry.content_hex()).ok())
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    let (Some(mut before_state), Some(mut current_state)) = (before_state, current_state) else {
        return Ok(false);
    };
    let before_page = before_state
        .get_mut("pages")
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|pages| pages.remove(page_id));
    let current_page = current_state
        .get_mut("pages")
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|pages| pages.remove(page_id));
    let (Some(before_page), Some(current_page)) = (before_page, current_page) else {
        return Ok(false);
    };
    let expected_current_page = serde_json::json!({
        "file": projection_target_path,
        "version": after_page.version,
        "last_written": after_page.last_modified,
    });
    Ok(before_state == current_state
        && before_page.get("file")
            == Some(&serde_json::Value::String(projection_target_path.clone()))
        && current_page == expected_current_page)
}

async fn recover_complete_entity_extraction_apply_receipt(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
) -> Result<Option<RepairApplyReceipt>, WenlanError> {
    let manifest_dir = store.manifest_dir(manifest.manifest_id())?;
    let final_path = manifest_dir.join(APPLY_RECEIPT_FILE);
    if final_path.exists() {
        let receipt = store.load_apply_receipt(manifest)?;
        let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
        if pending_path.exists() {
            fs::remove_file(pending_path)?;
            sync_dir(&manifest_dir)?;
        }
        return Ok(Some(receipt));
    }
    let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
    if !pending_path.exists() {
        return Ok(None);
    }
    let pending = read_bounded_file(&pending_path, REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES)?;
    let parsed = StoredRepairApplyReceipt::from_slice(&pending)
        .ok()
        .and_then(|receipt| verify_stored_apply_receipt(receipt, manifest).ok());
    let connection = db.conn.lock().await;
    let (target_now, _) =
        repair_target_receipt_on_connection(&connection, manifest.target()).await?;
    drop(connection);
    if let Some(receipt) = parsed {
        if target_now == *receipt.after_target_receipt() {
            publish_no_replace(&pending_path, &final_path, "repair_already_applied")?;
            return Ok(Some(receipt));
        }
    }
    if target_now != *manifest.expected_state().canonical_receipt() {
        return Err(WenlanError::Conflict(
            "repair_apply_recovery_required".to_string(),
        ));
    }
    fs::remove_file(&pending_path)?;
    sync_dir(&manifest_dir)?;
    Ok(None)
}

async fn recover_apply_receipt(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
) -> Result<Option<RepairApplyReceipt>, WenlanError> {
    let manifest_dir = store.manifest_dir(manifest.manifest_id())?;
    let final_path = manifest_dir.join(APPLY_RECEIPT_FILE);
    if final_path.exists() {
        let receipt = store.load_apply_receipt(manifest)?;
        let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
        if pending_path.exists() {
            fs::remove_file(pending_path)?;
            sync_dir(&manifest_dir)?;
        }
        if manifest.writer() == RepairWriter::QuarantineStalePageProjection {
            store.clear_stale_page_projection_apply_journal(manifest.manifest_id())?;
        }
        return Ok(Some(receipt));
    }
    let pending_path = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
    if !pending_path.exists() {
        if manifest.writer() == RepairWriter::QuarantineStalePageProjection
            && store.stale_page_projection_apply_journal_exists(manifest.manifest_id())?
        {
            return recover_stale_page_projection_apply_receipt(
                db, store, manifest, rollback, page_root, None,
            )
            .await;
        }
        return Ok(None);
    }
    let pending = fs::read(&pending_path)?;
    let parsed = StoredRepairApplyReceipt::from_slice(&pending)
        .ok()
        .and_then(|receipt| verify_stored_apply_receipt(receipt, manifest).ok());
    if manifest.writer() == RepairWriter::QuarantineStalePageProjection {
        return recover_stale_page_projection_apply_receipt(
            db, store, manifest, rollback, page_root, parsed,
        )
        .await;
    }
    if let Some(receipt) = parsed {
        let (target_now, _) = target_receipt_current(db, manifest, rollback, page_root).await?;
        if target_now == *receipt.after_target_receipt() {
            publish_no_replace(&pending_path, &final_path, "repair_already_applied")?;
            return Ok(Some(receipt));
        }
        if target_now != *manifest.expected_state().canonical_receipt() {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
    } else {
        let (target_now, _) = target_receipt_current(db, manifest, rollback, page_root).await?;
        if target_now != *manifest.expected_state().canonical_receipt() {
            return Err(WenlanError::Conflict(
                "repair_apply_recovery_required".to_string(),
            ));
        }
    }
    fs::remove_file(&pending_path)?;
    sync_dir(&manifest_dir)?;
    Ok(None)
}

async fn recover_stale_page_projection_apply_receipt(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
    pending_receipt: Option<RepairApplyReceipt>,
) -> Result<Option<RepairApplyReceipt>, WenlanError> {
    let page_root = page_root.ok_or_else(|| {
        WenlanError::Validation("page projection repair root unavailable".to_string())
    })?;
    let page_id = match manifest.target() {
        RepairTarget::PageProjection { page_id, .. } => page_id,
        _ => {
            return Err(WenlanError::Validation(
                "stale page projection repair target/writer mismatch".to_string(),
            ))
        }
    };
    let connection = db.conn.lock().await;
    let mut owner = connection
        .query(
            "SELECT 1 FROM pages WHERE id=?1 LIMIT 1",
            libsql::params![page_id.as_str()],
        )
        .await
        .map_err(database_error)?;
    if owner.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict(
            "repair_apply_recovery_required".to_string(),
        ));
    }
    drop(owner);
    let apply_journal = store.load_stale_page_projection_apply_journal(manifest)?;
    if apply_journal.is_none() {
        let (source_path, quarantine_path) = stale_page_projection_paths(rollback)?;
        if capture_stale_page_projection_current(page_root, page_id, &source_path, &quarantine_path)
            .is_ok_and(|current| current == *rollback)
        {
            store.clear_pending_apply_receipt(manifest.manifest_id())?;
            return Ok(None);
        }
    }
    let apply_rollback = apply_journal.unwrap_or_else(|| rollback.clone());
    let restore_post = pending_receipt.is_none();
    let recovery = crate::export::knowledge::KnowledgeProjectionWrite::with_repair_lock(
        page_root.to_path_buf(),
        db,
        |write| {
            write.recover_stale_page_projection(
                &apply_rollback,
                manifest.manifest_id(),
                restore_post,
            )
        },
    )
    .unwrap_or(crate::export::knowledge::StalePageProjectionRecoveryState::Unknown);
    use crate::export::knowledge::StalePageProjectionRecoveryState as Recovery;
    match (pending_receipt, recovery) {
        (Some(receipt), Recovery::Post)
            if stale_page_projection_post_target_receipt(&apply_rollback)?
                == *receipt.after_target_receipt() =>
        {
            let manifest_dir = store.manifest_dir(manifest.manifest_id())?;
            publish_no_replace(
                &manifest_dir.join(APPLY_RECEIPT_PENDING_FILE),
                &manifest_dir.join(APPLY_RECEIPT_FILE),
                "repair_already_applied",
            )?;
            store.clear_stale_page_projection_apply_journal(manifest.manifest_id())?;
            Ok(Some(receipt))
        }
        (Some(_), Recovery::Original) | (None, Recovery::Original) => {
            store.clear_pending_apply_receipt(manifest.manifest_id())?;
            store.clear_stale_page_projection_apply_journal(manifest.manifest_id())?;
            Ok(None)
        }
        _ => Err(WenlanError::Conflict(
            "repair_apply_recovery_required".to_string(),
        )),
    }
}

async fn target_receipt_current(
    db: &MemoryDB,
    manifest: &RepairManifest,
    rollback: &StoredRollbackArtifact,
    page_root: Option<&Path>,
) -> Result<(RepairDigest, u64), WenlanError> {
    let connection = db.conn.lock().await;
    match manifest.target() {
        RepairTarget::PageProjection { page_id, .. }
            if manifest.writer() == RepairWriter::QuarantineStalePageProjection =>
        {
            let page_root = page_root.ok_or_else(|| {
                WenlanError::Validation("page projection repair root unavailable".to_string())
            })?;
            let mut owner = connection
                .query(
                    "SELECT 1 FROM pages WHERE id=?1 LIMIT 1",
                    libsql::params![page_id.as_str()],
                )
                .await
                .map_err(database_error)?;
            if owner.next().await.map_err(database_error)?.is_some() {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            drop(owner);
            let (source_path, quarantine_path) = stale_page_projection_paths(rollback)?;
            let current = capture_stale_page_projection_current(
                page_root,
                page_id,
                &source_path,
                &quarantine_path,
            )?;
            Ok((target_receipt(&current)?, 0))
        }
        RepairTarget::PageProjection { page_id, .. } => {
            let page_root = page_root.ok_or_else(|| {
                WenlanError::Validation("page projection repair root unavailable".to_string())
            })?;
            let paths = projection_rollback_paths(rollback)?;
            let current = capture_page_projection_on_connection(
                &connection,
                page_root,
                page_id,
                &paths,
                &rollback.table,
            )
            .await?;
            Ok((target_receipt(&current)?, 1))
        }
        _ => repair_target_receipt_on_connection(&connection, manifest.target()).await,
    }
}

pub async fn record_repair_verification(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: VerifyRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
) -> Result<RepairVerificationReceipt, WenlanError> {
    record_repair_verification_inner(db, store, request, page_root, now_epoch, || Ok(())).await
}

async fn record_repair_verification_inner<F>(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: VerifyRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
    after_projection_session: F,
) -> Result<RepairVerificationReceipt, WenlanError>
where
    F: FnOnce() -> Result<(), WenlanError>,
{
    ensure_repair_artifacts_supported()?;
    if now_epoch <= 0 {
        return Err(WenlanError::Validation(
            "invalid_repair_verified_at".to_string(),
        ));
    }
    let manifest = store.load_manifest(request.manifest_id())?;
    if manifest.manifest_digest() != request.manifest_digest() {
        return Err(WenlanError::Conflict(
            "repair_verification_manifest_mismatch".to_string(),
        ));
    }
    let _operation_lock = store.lock_manifest_operation(manifest.manifest_id())?;
    let _tag_record_set_lock = store.lock_tag_record_set(&manifest)?;
    let prior_verified_tag_targets = store.verified_tag_targets(&manifest)?;
    let apply_receipt = store.load_apply_receipt(&manifest)?;
    if apply_receipt.receipt_digest() != request.apply_receipt_digest() {
        return Err(WenlanError::Validation(
            "repair_apply_receipt_mismatch".to_string(),
        ));
    }
    if apply_receipt.post_apply_db_digest().is_none() {
        if let Some(receipt) = store.load_verification_receipt(&manifest, &apply_receipt)? {
            store.clear_pending_apply_receipt(manifest.manifest_id())?;
            return Ok(receipt);
        }
        return Err(WenlanError::Validation(
            "repair_legacy_apply_receipt_unverifiable".to_string(),
        ));
    }
    if apply_receipt.actual_effects() != manifest.allowed_effects()
        || apply_receipt.writer() != manifest.writer()
        || apply_receipt.before_target_receipt() != manifest.expected_state().canonical_receipt()
    {
        return Err(WenlanError::Validation(
            "repair_apply_receipt_mismatch".to_string(),
        ));
    }
    if let Some(receipt) = store.load_verification_receipt(&manifest, &apply_receipt)? {
        store.clear_pending_apply_receipt(manifest.manifest_id())?;
        return Ok(receipt);
    }
    let rename_rollback = if manifest.writer() == RepairWriter::RenamePageTitle {
        Some(store.load_rename_page_title_rollback(&manifest)?)
    } else {
        None
    };
    let rollback = if matches!(
        manifest.writer(),
        RepairWriter::RenamePageTitle | RepairWriter::CompleteEntityExtraction
    ) {
        if manifest.writer() == RepairWriter::CompleteEntityExtraction {
            store.load_complete_entity_extraction_rollback(&manifest)?;
        }
        None
    } else {
        Some(store.load_rollback(&manifest)?.0)
    };
    validate_verification_reports(&manifest, request.general_report(), request.deep_report())?;
    validate_current_page_receipts(request.general_report(), request.deep_report(), page_root)
        .await?;
    let rename_projection_session = if manifest.writer() == RepairWriter::RenamePageTitle {
        let page_root = page_root.ok_or_else(|| {
            WenlanError::Validation("page projection repair root unavailable".to_string())
        })?;
        Some(
            crate::export::knowledge::KnowledgeProjectionWrite::begin_owned_repair_session(
                page_root.to_path_buf(),
                db,
            )?,
        )
    } else {
        None
    };
    after_projection_session()?;
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
            &manifest,
            &prior_verified_tag_targets,
            true,
        )
        .await?;
        validate_deterministic_target_resolved(db, &manifest, page_root).await?;
        let (target_now, _) = match manifest.target() {
            RepairTarget::PageProjection { page_id, .. }
                if manifest.writer() == RepairWriter::RenamePageTitle =>
            {
                let rollback = rename_rollback.as_ref().ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let projection = rename_projection_session
                    .as_ref()
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
                    &PageScanControl::with_timeout(std::time::Duration::from_secs(30)),
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
                let rollback = rollback.as_ref().ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let page_root = page_root.ok_or_else(|| {
                    WenlanError::Validation("page projection repair root unavailable".to_string())
                })?;
                let (source_path, quarantine_path) = stale_page_projection_paths(rollback)?;
                let target_now =
                    crate::export::knowledge::KnowledgeProjectionWrite::with_projection_lock(
                        page_root,
                        |projection| {
                            let excluded = BTreeSet::from([
                                ".wenlan".to_string(),
                                ".wenlan/state.json".to_string(),
                                ".wenlan/orphaned".to_string(),
                                source_path.clone(),
                                quarantine_path.clone(),
                            ]);
                            let scan = projection.scan_page_root_controlled(
                                true,
                                &PageScanControl::with_timeout(std::time::Duration::from_secs(30)),
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
                        },
                    )?;
                (target_now, 0)
            }
            RepairTarget::PageProjection { page_id, .. } => {
                let rollback = rollback.as_ref().ok_or_else(|| {
                    WenlanError::Validation("repair_rollback_writer_mismatch".to_string())
                })?;
                let page_root = page_root.ok_or_else(|| {
                    WenlanError::Validation("page projection repair root unavailable".to_string())
                })?;
                let paths = projection_rollback_paths(rollback)?;
                let page_row = projection_page_row_on_connection(&connection, page_id).await?;
                let target_now =
                    crate::export::knowledge::KnowledgeProjectionWrite::with_projection_lock(
                        page_root,
                        |_| {
                            let scan = crate::lint::pages::fs::scan_page_root_controlled(
                                page_root,
                                true,
                                &crate::lint::pages::fs::PageScanControl::with_timeout(
                                    std::time::Duration::from_secs(30),
                                ),
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
                        },
                    )?;
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
                now_epoch,
                request.general_report().snapshots().clone(),
                deep.snapshots().clone(),
            ),
            None => RepairVerificationReceiptDraft::try_new_general_only(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                now_epoch,
                request.general_report().snapshots().clone(),
            ),
        }
        .map_err(|error| WenlanError::Validation(error.to_string()))?;
        let receipt_digest = repair_digest(&draft.canonical_bytes()?);
        let receipt = RepairVerificationReceipt::from_draft(draft, receipt_digest);
        if let Some(session) = rename_projection_session.as_ref() {
            validate_current_page_receipts_on_repair_projection(
                request.general_report(),
                request.deep_report(),
                &session.locked(),
            )?;
            store.persist_verification_receipt(&receipt)?;
            Ok(receipt)
        } else if let Some(page_root) = page_root {
            crate::export::knowledge::KnowledgeProjectionWrite::with_projection_lock(
                page_root,
                |_| {
                    validate_current_page_receipts_locked(
                        request.general_report(),
                        request.deep_report(),
                        Some(page_root),
                    )?;
                    store.persist_verification_receipt(&receipt)?;
                    Ok(receipt)
                },
            )
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
    store.clear_pending_apply_receipt(manifest.manifest_id())?;
    Ok(receipt)
}

#[cfg(test)]
async fn record_repair_verification_with_projection_session_hook<F>(
    db: &MemoryDB,
    store: &RepairArtifactStore,
    request: VerifyRepairRequest,
    page_root: Option<&Path>,
    now_epoch: i64,
    after_projection_session: F,
) -> Result<RepairVerificationReceipt, WenlanError>
where
    F: FnOnce() -> Result<(), WenlanError>,
{
    record_repair_verification_inner(
        db,
        store,
        request,
        page_root,
        now_epoch,
        after_projection_session,
    )
    .await
}

fn validate_verification_reports(
    manifest: &RepairManifest,
    general: &LintReport,
    deep: Option<&LintReport>,
) -> Result<(), WenlanError> {
    let deep_shape_matches = manifest.source().is_general_only_deterministic() == deep.is_none();
    if !manifest
        .source()
        .lint_scope()
        .matches_report_scope_kind(general.scope())
        || general.scope() != manifest.source().report_scope()
        || !stable_report_snapshot(general)
        || !deep_shape_matches
        || deep.is_some_and(|deep| {
            !manifest
                .source()
                .lint_scope()
                .matches_report_scope_kind(deep.scope())
                || deep.scope() != manifest.source().report_scope()
                || (manifest.source().finding().is_some() && deep.agent_work().is_none())
                || general.producer_receipt() != deep.producer_receipt()
                || !stable_report_snapshot(deep)
        })
    {
        return Err(WenlanError::Validation(
            "repair_verification_report_mismatch".to_string(),
        ));
    }
    let general_baseline = manifest.post_assertions().general_baseline();
    let deep_baseline = manifest.post_assertions().deep_baseline();
    if manifest
        .post_assertions()
        .verification_policy()
        .requires_whole_reports()
        && (!general.complete() || deep.is_none_or(|deep| !deep.complete()))
    {
        return Err(WenlanError::Validation(
            "repair_legacy_verification_not_clean".to_string(),
        ));
    }
    if let Some(required_deep_check_ids) = manifest
        .post_assertions()
        .verification_policy()
        .required_deep_check_ids()
    {
        let Some(deep) = deep else {
            return Err(WenlanError::Validation(
                "repair_required_deep_check_incomplete".to_string(),
            ));
        };
        let required_deep_complete = required_deep_check_ids.iter().all(|check_id| {
            deep.checks()
                .iter()
                .find(|check| check.check_id() == check_id)
                .is_some_and(|check| {
                    matches!(check.outcome(), LintOutcome::Pass | LintOutcome::Finding)
                })
        });
        if !required_deep_complete {
            return Err(WenlanError::Validation(
                "repair_required_deep_check_incomplete".to_string(),
            ));
        }
    }
    if general_baseline.is_empty() && deep_baseline.is_empty() {
        // The first durable v1 producer predated baseline binding. Preserve
        // those manifests without inventing unapproved baseline state: their
        // post-repair reports must instead be conservatively clean.
        if std::iter::once(general).chain(deep).any(|report| {
            report.totals().actionable_findings() != 0 || report.totals().incomplete() != 0
        }) {
            return Err(WenlanError::Validation(
                "repair_legacy_verification_not_clean".to_string(),
            ));
        }
    } else {
        validate_check_deltas(
            general_baseline,
            general,
            manifest.post_assertions().target_check_id(),
            manifest.source().finding().is_none(),
            manifest.post_assertions().allowed_non_target_check_deltas(),
        )?;
        if let Some(deep) = deep {
            validate_check_deltas(
                deep_baseline,
                deep,
                manifest.post_assertions().target_check_id(),
                manifest.source().finding().is_none(),
                manifest.post_assertions().allowed_non_target_check_deltas(),
            )?;
        } else if !deep_baseline.is_empty() {
            return Err(WenlanError::Validation(
                "repair_verification_report_mismatch".to_string(),
            ));
        }
    }
    let target_survives = manifest.source().finding().is_some()
        && std::iter::once(general)
            .chain(deep)
            .flat_map(|report| report.checks())
            .filter(|check| check.check_id() == manifest.post_assertions().target_check_id())
            .any(|check| {
                check.evidence().iter().any(|evidence| match evidence {
                    LintEvidenceRef::SemanticFinding { finding } => finding
                        .evidence_ids()
                        .contains(manifest.post_assertions().target_evidence_id()),
                    _ => false,
                })
            });
    if target_survives {
        return Err(WenlanError::Validation(
            "repair_target_assertion_failed".to_string(),
        ));
    }
    if manifest.source().finding().is_none() {
        if !deterministic_target_assertion_supported(manifest) {
            return Err(WenlanError::Validation(
                "repair_target_assertion_unsupported".to_string(),
            ));
        }
        if !deterministic_target_report_changed(manifest, general) {
            return Err(WenlanError::Validation(
                "repair_target_assertion_failed".to_string(),
            ));
        }
    }
    Ok(())
}

fn deterministic_target_report_changed(manifest: &RepairManifest, general: &LintReport) -> bool {
    let check_id = manifest.post_assertions().target_check_id();
    let before = manifest
        .post_assertions()
        .general_baseline()
        .iter()
        .find(|check| check.check_id() == check_id);
    let after = general
        .checks()
        .iter()
        .find(|check| check.check_id() == check_id);
    if check_id == "identity.tag_integrity" {
        if let (Some(_before), Some(after), Some(before_affected)) = (
            before,
            after,
            before.and_then(RepairCheckBaseline::affected_records),
        ) {
            return lint_check_affected_records(after).is_ok_and(|after_affected| {
                after_affected.is_some_and(|after_affected| after_affected < before_affected)
            });
        }
    }
    matches!(
        (before, after),
        (Some(before), Some(after))
            if before.outcome() == LintOutcome::Finding
                && (after.outcome() != before.outcome()
                    || after.evidence() != before.evidence())
    )
}

pub(crate) fn deterministic_target_assertion_supported(manifest: &RepairManifest) -> bool {
    matches!(
        (
            manifest.post_assertions().target_check_id(),
            manifest.writer()
        ),
        (
            "identity.memory_state_integrity",
            RepairWriter::NormalizeMemorySourceAgent
                | RepairWriter::ClearMemorySupersedes
                | RepairWriter::UnstageOrphanRevision
        ) | (
            "memories.supersession_integrity",
            RepairWriter::ClearMemorySupersedes | RepairWriter::UnstageOrphanRevision
        ) | ("identity.tag_integrity", RepairWriter::DeleteTagRow)
            | (
                "memory_entities.integrity",
                RepairWriter::DeleteMemoryEntityLink
            )
            | (
                "memories.enrichment_failures",
                RepairWriter::CompleteEntityExtraction
            )
            | (
                "pages.duplicate_active_titles",
                RepairWriter::RenamePageTitle
            )
            | ("pages.links.orphan_labels", RepairWriter::BindPageLink)
            | (
                "pages.source_page_integrity",
                RepairWriter::ArchiveEmptySourcePage
            )
            | (
                "pages.projection.version_alignment",
                RepairWriter::RegeneratePageProjection
            )
            | (
                "pages.projection.identity",
                RepairWriter::QuarantineStalePageProjection
            )
    )
}

async fn validate_deterministic_target_resolved(
    db: &MemoryDB,
    manifest: &RepairManifest,
    page_root: Option<&Path>,
) -> Result<(), WenlanError> {
    if manifest.source().finding().is_some() {
        return Ok(());
    }
    let snapshot = db.open_lint_snapshot().await.map_err(snapshot_error)?;
    let target_still_actionable = crate::repair_plan::deterministic_target_still_actionable(
        &snapshot,
        manifest.source().lint_scope(),
        page_root,
        manifest.target(),
        manifest.writer(),
    )
    .await?;
    let receipt = snapshot.finish().await.map_err(snapshot_error)?;
    if !receipt.is_consistent() {
        return Err(WenlanError::Conflict(
            "repair_verification_reports_stale".to_string(),
        ));
    }
    if target_still_actionable {
        return Err(WenlanError::Validation(
            "repair_target_assertion_failed".to_string(),
        ));
    }
    Ok(())
}

fn stable_report_snapshot(report: &LintReport) -> bool {
    report.snapshots().db().post_run_digest() == Some(report.snapshots().db().analysis_digest())
        && report.snapshots().pages().after_scan_digest()
            == Some(report.snapshots().pages().before_scan_digest())
}

fn validate_current_page_receipts_locked(
    general: &LintReport,
    deep: Option<&LintReport>,
    page_root: Option<&Path>,
) -> Result<(), WenlanError> {
    validate_current_page_report_receipt_locked(general, page_root)?;
    if let Some(deep) = deep {
        validate_current_page_report_receipt_locked(deep, page_root)?;
    }
    Ok(())
}

fn validate_current_page_receipts_on_repair_projection(
    general: &LintReport,
    deep: Option<&LintReport>,
    projection: &crate::export::knowledge::LockedRepairProjection<'_>,
) -> Result<(), WenlanError> {
    validate_current_page_report_receipt_on_repair_projection(general, projection)?;
    if let Some(deep) = deep {
        validate_current_page_report_receipt_on_repair_projection(deep, projection)?;
    }
    Ok(())
}

fn validate_current_page_report_receipt_on_repair_projection(
    report: &LintReport,
    projection: &crate::export::knowledge::LockedRepairProjection<'_>,
) -> Result<(), WenlanError> {
    let current_page = current_page_digest_on_repair_projection(projection, report.profile())?;
    if report.snapshots().pages().before_scan_digest() != &current_page
        || report.snapshots().pages().after_scan_digest() != Some(&current_page)
    {
        return Err(WenlanError::Conflict(
            "repair_verification_reports_stale".to_string(),
        ));
    }
    Ok(())
}

fn current_page_digest_on_repair_projection(
    projection: &crate::export::knowledge::LockedRepairProjection<'_>,
    profile: LintProfile,
) -> Result<LintDigest, WenlanError> {
    let control = PageScanControl::with_timeout(ExecutionGate::page_budget_for(profile));
    let before = projection
        .scan_page_root_controlled(profile == LintProfile::Deep, &control)?
        .normalized_bytes();
    let after = projection
        .scan_page_root_controlled(profile == LintProfile::Deep, &control)?
        .normalized_bytes();
    if before != after {
        return Err(WenlanError::Conflict(
            "repair_verification_reports_stale".to_string(),
        ));
    }
    Ok(lint_digest(before))
}

fn validate_current_page_report_receipt_locked(
    report: &LintReport,
    page_root: Option<&Path>,
) -> Result<(), WenlanError> {
    let current_page = current_page_digest_locked(page_root, report.profile())?;
    if report.snapshots().pages().before_scan_digest() != &current_page
        || report.snapshots().pages().after_scan_digest() != Some(&current_page)
    {
        return Err(WenlanError::Conflict(
            "repair_verification_reports_stale".to_string(),
        ));
    }
    Ok(())
}

fn current_page_digest_locked(
    page_root: Option<&Path>,
    profile: LintProfile,
) -> Result<LintDigest, WenlanError> {
    let Some(root) = page_root else {
        return Ok(lint_digest([0; 32]));
    };
    let control = PageScanControl::with_timeout(ExecutionGate::page_budget_for(profile));
    let scan = scan_page_root_controlled(root, profile == LintProfile::Deep, &control)
        .map_err(page_snapshot_error)?;
    let before = scan.normalized_bytes();
    let after = scan
        .verify_unchanged_with_control(root, &control)
        .map_err(page_snapshot_error)?
        .after_normalized_bytes();
    if before != after {
        return Err(WenlanError::Conflict(
            "repair_verification_reports_stale".to_string(),
        ));
    }
    Ok(lint_digest(before))
}

pub(crate) async fn validate_current_page_receipts(
    general: &LintReport,
    deep: Option<&LintReport>,
    page_root: Option<&Path>,
) -> Result<(), WenlanError> {
    validate_current_page_report_receipt(general, page_root).await?;
    if let Some(deep) = deep {
        validate_current_page_report_receipt(deep, page_root).await?;
    }
    Ok(())
}

pub(crate) async fn validate_current_page_report_receipt(
    report: &LintReport,
    page_root: Option<&Path>,
) -> Result<(), WenlanError> {
    let current_page = current_page_digest(page_root, report.profile()).await?;
    if report.snapshots().pages().before_scan_digest() != &current_page
        || report.snapshots().pages().after_scan_digest() != Some(&current_page)
    {
        return Err(WenlanError::Conflict(
            "repair_verification_reports_stale".to_string(),
        ));
    }
    Ok(())
}

async fn validate_current_db_receipts(
    db: &MemoryDB,
    general: &LintReport,
    deep: Option<&LintReport>,
) -> Result<(), WenlanError> {
    let snapshot = db.open_lint_snapshot().await.map_err(snapshot_error)?;
    let current = snapshot.finish().await.map_err(snapshot_error)?;
    if !current.is_consistent() {
        return Err(WenlanError::Conflict(
            "repair_verification_reports_stale".to_string(),
        ));
    }
    let current_db = lint_digest(current.analysis_receipt_digest().as_bytes());
    for report in std::iter::once(general).chain(deep) {
        if report.snapshots().db().analysis_digest() != &current_db
            || report.snapshots().db().post_run_digest() != Some(&current_db)
        {
            return Err(WenlanError::Conflict(
                "repair_verification_reports_stale".to_string(),
            ));
        }
    }
    Ok(())
}

async fn current_page_digest(
    page_root: Option<&Path>,
    profile: LintProfile,
) -> Result<LintDigest, WenlanError> {
    let Some(root) = page_root else {
        return Ok(lint_digest([0; 32]));
    };
    let root = root.to_path_buf();
    let timeout = ExecutionGate::page_budget_for(profile);
    let control = PageScanControl::with_timeout(timeout);
    let worker_control = control.clone();
    let mut task = tokio::task::spawn_blocking(move || {
        let scan = scan_page_root_controlled(&root, profile == LintProfile::Deep, &worker_control)
            .map_err(page_snapshot_error)?;
        let before = scan.normalized_bytes();
        let after = scan
            .verify_unchanged_with_control(&root, &worker_control)
            .map_err(page_snapshot_error)?
            .after_normalized_bytes();
        if before != after {
            return Err(WenlanError::Conflict(
                "repair_verification_reports_stale".to_string(),
            ));
        }
        Ok(lint_digest(before))
    });
    match tokio::time::timeout(timeout, &mut task).await {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => Err(WenlanError::VectorDb(format!(
            "repair page snapshot task: {error}"
        ))),
        Err(_) => {
            control.cancel();
            let _ = task.await;
            Err(WenlanError::Conflict(
                "repair_verification_reports_stale: page snapshot deadline exceeded".to_string(),
            ))
        }
    }
}

fn validate_check_deltas(
    baseline: &[RepairCheckBaseline],
    report: &LintReport,
    target_check_id: &str,
    deterministic_target: bool,
    allowed_check_deltas: &[String],
) -> Result<(), WenlanError> {
    if baseline.len() != report.checks().len() {
        return Err(WenlanError::Validation(
            "repair_verification_catalog_mismatch".to_string(),
        ));
    }
    for before in baseline {
        let Some(after) = report
            .checks()
            .iter()
            .find(|check| check.check_id() == before.check_id())
        else {
            return Err(WenlanError::Validation(
                "repair_verification_catalog_mismatch".to_string(),
            ));
        };
        if before.gate_effect() != after.gate_effect() {
            return Err(WenlanError::Validation(
                "repair_verification_catalog_mismatch".to_string(),
            ));
        }
        if allowed_check_deltas
            .binary_search_by(|value| value.as_str().cmp(after.check_id()))
            .is_ok()
        {
            continue;
        }
        let after_affected_records = lint_check_affected_records(after)?;
        let affected_records = match (before.affected_records(), after_affected_records) {
            (Some(before_affected), Some(after_affected)) => {
                Some((before_affected, after_affected))
            }
            (Some(_), None) => {
                return Err(WenlanError::Validation(
                    "repair_affected_records_metric_invalid".to_string(),
                ));
            }
            (None, _) => None,
        };
        let target_uses_capped_tag_evidence = deterministic_target
            && target_check_id == "identity.tag_integrity"
            && before.check_id() == target_check_id;
        let target_count_decreased = target_uses_capped_tag_evidence
            && affected_records
                .is_some_and(|(before_affected, after_affected)| after_affected < before_affected);
        let affected_records_increased = affected_records
            .is_some_and(|(before_affected, after_affected)| after_affected > before_affected);
        let deterministic_target_not_decreased = target_uses_capped_tag_evidence
            && affected_records
                .is_some_and(|(before_affected, after_affected)| after_affected >= before_affected);
        let new_actionable = after.gate_effect() == LintGateEffect::Actionable
            && after.outcome() == LintOutcome::Finding
            && (before.outcome() != LintOutcome::Finding
                || affected_records_increased
                || deterministic_target_not_decreased
                || after
                    .evidence()
                    .iter()
                    .any(|evidence| !before.evidence().contains(evidence))
                    && !target_count_decreased);
        if new_actionable {
            return Err(WenlanError::Validation(
                "repair_new_actionable_finding".to_string(),
            ));
        }
        let after_incomplete = matches!(
            after.outcome(),
            LintOutcome::NotRunPrerequisite
                | LintOutcome::InconsistentSnapshot
                | LintOutcome::FailedToRun
        );
        if after_incomplete
            && (before.outcome() != after.outcome() || before.evidence() != after.evidence())
        {
            return Err(WenlanError::Validation(
                "repair_new_incomplete_check".to_string(),
            ));
        }
    }
    Ok(())
}

pub(crate) async fn target_receipt_on_connection(
    connection: &libsql::Connection,
    source_id: &str,
) -> Result<(RepairDigest, u64), WenlanError> {
    let rollback = capture_rollback_on_connection(connection, source_id).await?;
    let row_count = u64::try_from(rollback.rows.len())
        .map_err(|_| WenlanError::Validation("repair_target_too_large".to_string()))?;
    Ok((target_receipt(&rollback)?, row_count))
}

fn complete_entity_extraction_owner_ids(memory_id: &str) -> Vec<String> {
    vec![memory_id.to_string()]
}

fn lint_digest_from_repair_digest(digest: &RepairDigest) -> Result<LintDigest, WenlanError> {
    let prefix = digest
        .as_str()
        .get(..16)
        .ok_or_else(|| WenlanError::Validation("repair occurrence digest truncated".to_string()))?;
    Ok(LintDigest::from_u64(
        u64::from_str_radix(prefix, 16)
            .map_err(|error| WenlanError::Validation(error.to_string()))?,
    ))
}

pub(crate) fn complete_entity_extraction_receipt(
    rollback: &RepairRollbackPayloadV2,
) -> Result<RepairDigest, WenlanError> {
    if !matches!(
        rollback,
        RepairRollbackPayloadV2::CompleteEntityExtraction { .. }
    ) {
        return Err(WenlanError::Validation(
            "repair_rollback_writer_mismatch".to_string(),
        ));
    }
    let payload_bytes = serde_json::to_vec(rollback)?;
    if u64::try_from(payload_bytes.len()).unwrap_or(u64::MAX) > REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES {
        return Err(WenlanError::Validation(
            "repair_rollback_artifact_too_large".to_string(),
        ));
    }
    let mut bytes = b"wenlan-repair-entity-extraction-v2".to_vec();
    bytes.extend(payload_bytes);
    Ok(repair_digest(&bytes))
}

pub(crate) fn rename_page_title_receipt(
    rollback: &RepairRollbackPayloadV2,
) -> Result<RepairDigest, WenlanError> {
    if !matches!(rollback, RepairRollbackPayloadV2::RenamePageTitle { .. }) {
        return Err(WenlanError::Validation(
            "repair_rollback_writer_mismatch".to_string(),
        ));
    }
    let payload_bytes = serde_json::to_vec(rollback)?;
    if u64::try_from(payload_bytes.len()).unwrap_or(u64::MAX) > REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES {
        return Err(WenlanError::Validation(
            "repair_rollback_artifact_too_large".to_string(),
        ));
    }
    let mut bytes = b"wenlan-repair-page-title-v2".to_vec();
    bytes.extend(payload_bytes);
    Ok(repair_digest(&bytes))
}

fn encode_page_title_embedding(embedding: Vec<f32>) -> Result<String, WenlanError> {
    if embedding.len() != 768 || embedding.iter().any(|value| !value.is_finite()) {
        return Err(WenlanError::Validation(
            "repair_page_embedding_invalid".to_string(),
        ));
    }
    let mut bytes = Vec::with_capacity(768 * std::mem::size_of::<f32>());
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(hex::encode(bytes))
}

pub(crate) fn rename_page_title_non_target_receipt(
    database_guard: &RepairDigest,
    filesystem_digest: [u8; 32],
    captured: &RepairRollbackPayloadV2,
) -> Result<RepairDigest, WenlanError> {
    let RepairRollbackPayloadV2::RenamePageTitle {
        page_id,
        projection_entries,
        ..
    } = captured
    else {
        return Err(WenlanError::Validation(
            "repair_rollback_writer_mismatch".to_string(),
        ));
    };
    let state = projection_entries
        .iter()
        .find(|entry| entry.relative_path() == ".wenlan/state.json")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let state_bytes = hex::decode(state.content_hex())
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let mut root = serde_json::from_slice::<BTreeMap<String, serde_json::Value>>(&state_bytes)
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let pages = root
        .remove("pages")
        .and_then(|pages| serde_json::from_value::<BTreeMap<String, serde_json::Value>>(pages).ok())
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    root.insert(
        "pages".to_string(),
        serde_json::to_value(
            pages
                .into_iter()
                .filter(|(candidate, _)| candidate != page_id)
                .collect::<BTreeMap<_, _>>(),
        )?,
    );
    let mut bytes = b"wenlan-repair-page-title-non-target-v2".to_vec();
    bytes.extend(database_guard.as_str().as_bytes());
    bytes.extend(filesystem_digest);
    bytes.extend(serde_json::to_vec(&root)?);
    Ok(repair_digest(&bytes))
}

pub(crate) fn rename_page_title_excluded_paths(
    rollback: &RepairRollbackPayloadV2,
) -> Result<BTreeSet<String>, WenlanError> {
    let RepairRollbackPayloadV2::RenamePageTitle {
        projection_target_path,
        ..
    } = rollback
    else {
        return Err(WenlanError::Validation(
            "repair_rollback_writer_mismatch".to_string(),
        ));
    };
    Ok(BTreeSet::from([
        ".wenlan".to_string(),
        ".wenlan/state.json".to_string(),
        projection_target_path.clone(),
    ]))
}

fn validate_rename_page_title_lint_finding(report: &LintReport) -> Result<(), WenlanError> {
    let check = report
        .checks()
        .iter()
        .find(|check| check.check_id() == "pages.duplicate_active_titles")
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?;
    if check.outcome() != LintOutcome::Finding || check.gate_effect() != LintGateEffect::Actionable
    {
        return Err(WenlanError::Validation(
            "unsupported_repair_finding".to_string(),
        ));
    }
    Ok(())
}

async fn capture_rename_page_row_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    page_id: &str,
    expected_title: &str,
) -> Result<(Vec<String>, Vec<String>, RepairScope, i64), WenlanError> {
    let (columns, row) = encoded_page_row_on_snapshot(snapshot, page_id).await?;
    let mut rows = snapshot
        .query(
            "SELECT title,version,status,space
               FROM pages WHERE id=?1 LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(page_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let metadata = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let title = metadata.get::<String>(0).map_err(database_error)?;
    let version = metadata.get::<i64>(1).map_err(database_error)?;
    let status = metadata.get::<String>(2).map_err(database_error)?;
    let effective_scope = metadata.get::<Option<String>>(3).map_err(database_error)?;
    if rows.next().await.map_err(snapshot_error)?.is_some()
        || title != expected_title
        || version <= 0
        || status != "active"
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let scope = match effective_scope {
        Some(scope) => RepairScope::registered(scope),
        None => Ok(RepairScope::uncategorized()),
    }
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    Ok((columns, row, scope, version))
}

async fn encoded_page_columns_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = snapshot
        .query("PRAGMA table_info(pages)", libsql::params::Params::None)
        .await
        .map_err(snapshot_error)?;
    let mut columns = Vec::new();
    let mut captured_bytes = 0_u64;
    while let Some(row) = rows.next().await.map_err(snapshot_error)? {
        let column = row.get::<String>(1).map_err(database_error)?;
        reserve_entity_extraction_capture(&mut captured_bytes, column.len())?;
        columns.push(column);
    }
    if columns.is_empty() {
        return Err(WenlanError::Validation(
            "repair_target_schema_missing".to_string(),
        ));
    }
    Ok(columns)
}

async fn encoded_page_columns_on_connection(
    connection: &libsql::Connection,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = connection
        .query("PRAGMA table_info(pages)", ())
        .await
        .map_err(database_error)?;
    let mut columns = Vec::new();
    let mut captured_bytes = 0_u64;
    while let Some(row) = rows.next().await.map_err(database_error)? {
        let column = row.get::<String>(1).map_err(database_error)?;
        reserve_entity_extraction_capture(&mut captured_bytes, column.len())?;
        columns.push(column);
    }
    if columns.is_empty() {
        return Err(WenlanError::Validation(
            "repair_target_schema_missing".to_string(),
        ));
    }
    Ok(columns)
}

async fn encoded_page_row_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    page_id: &str,
) -> Result<(Vec<String>, Vec<String>), WenlanError> {
    let columns = encoded_page_columns_on_snapshot(snapshot).await?;
    let selected_lengths = columns
        .iter()
        .map(|column| entity_extraction_encoded_length_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    let mut length_rows = snapshot
        .query(
            &format!("SELECT {selected_lengths} FROM pages WHERE id=?1 LIMIT 2"),
            libsql::params::Params::Positional(vec![libsql::Value::Text(page_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let lengths = length_rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let column_bytes = columns.iter().try_fold(0_u64, |mut total, column| {
        reserve_entity_extraction_capture(&mut total, column.len())?;
        Ok::<_, WenlanError>(total)
    })?;
    ensure_entity_extraction_memory_lengths(&lengths, columns.len(), column_bytes)?;
    if length_rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let selected = columns
        .iter()
        .map(|column| entity_extraction_encoded_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    let mut rows = snapshot
        .query(
            &format!("SELECT {selected} FROM pages WHERE id=?1 LIMIT 2"),
            libsql::params::Params::Positional(vec![libsql::Value::Text(page_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let mut values = Vec::with_capacity(columns.len());
    let mut captured_bytes = column_bytes;
    for index in 0..columns.len() {
        let value = row
            .get::<String>(i32::try_from(index).map_err(|_| {
                WenlanError::Validation("repair_target_schema_too_wide".to_string())
            })?)
            .map_err(database_error)?;
        reserve_entity_extraction_capture(&mut captured_bytes, value.len())?;
        values.push(value);
    }
    if rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok((columns, values))
}

async fn encoded_page_row_on_connection(
    connection: &libsql::Connection,
    page_id: &str,
) -> Result<(Vec<String>, Vec<String>), WenlanError> {
    let columns = encoded_page_columns_on_connection(connection).await?;
    let selected_lengths = columns
        .iter()
        .map(|column| entity_extraction_encoded_length_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    let mut length_rows = connection
        .query(
            &format!("SELECT {selected_lengths} FROM pages WHERE id=?1 LIMIT 2"),
            libsql::params![page_id],
        )
        .await
        .map_err(database_error)?;
    let lengths = length_rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let column_bytes = columns.iter().try_fold(0_u64, |mut total, column| {
        reserve_entity_extraction_capture(&mut total, column.len())?;
        Ok::<_, WenlanError>(total)
    })?;
    ensure_entity_extraction_memory_lengths(&lengths, columns.len(), column_bytes)?;
    if length_rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let selected = columns
        .iter()
        .map(|column| entity_extraction_encoded_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    let mut rows = connection
        .query(
            &format!("SELECT {selected} FROM pages WHERE id=?1 LIMIT 2"),
            libsql::params![page_id],
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let mut values = Vec::with_capacity(columns.len());
    let mut captured_bytes = column_bytes;
    for index in 0..columns.len() {
        let value = row
            .get::<String>(i32::try_from(index).map_err(|_| {
                WenlanError::Validation("repair_target_schema_too_wide".to_string())
            })?)
            .map_err(database_error)?;
        reserve_entity_extraction_capture(&mut captured_bytes, value.len())?;
        values.push(value);
    }
    if rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok((columns, values))
}

pub(crate) async fn capture_rename_page_title_on_connection(
    connection: &libsql::Connection,
    projection: &crate::export::knowledge::LockedRepairProjection<'_>,
    page_id: &str,
) -> Result<RepairRollbackPayloadV2, WenlanError> {
    let (page_columns, before_page_row) =
        encoded_page_row_on_connection(connection, page_id).await?;
    let (projection_target_path, projection_entries) =
        projection.capture_rename_page_projection(page_id)?;
    RepairRollbackPayloadV2::rename_page_title(
        page_id.to_string(),
        page_columns,
        before_page_row,
        projection_target_path,
        projection_entries,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))
}

async fn validate_rename_page_title_collision_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    page_id: &str,
    before_title: &str,
    after_title: &str,
    effective_scope: Option<&str>,
) -> Result<(), WenlanError> {
    let mut rows = snapshot
        .query(
            "SELECT
                 EXISTS(
                    SELECT 1 FROM pages
                     WHERE status='active' AND id<>?1
                       AND space=COALESCE(?4,'00000000-0000-4000-8000-000000000001')
                       AND lower(title)=lower(?2)),
                 EXISTS(
                    SELECT 1 FROM pages
                     WHERE status='active' AND id<>?1
                       AND space=COALESCE(?4,'00000000-0000-4000-8000-000000000001')
                       AND lower(title)=lower(?3))",
            libsql::params::Params::Positional(vec![
                libsql::Value::Text(page_id.to_string()),
                libsql::Value::Text(before_title.to_string()),
                libsql::Value::Text(after_title.to_string()),
                effective_scope
                    .map(|scope| libsql::Value::Text(scope.to_string()))
                    .unwrap_or(libsql::Value::Null),
            ]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    if row.get::<i64>(0).map_err(database_error)? != 1
        || row.get::<i64>(1).map_err(database_error)? != 0
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

pub(crate) async fn validate_rename_page_title_collision_on_connection(
    connection: &libsql::Connection,
    page_id: &str,
    before_title: &str,
    after_title: &str,
    effective_scope: Option<&str>,
) -> Result<(), WenlanError> {
    let mut rows = connection
        .query(
            "SELECT
                 EXISTS(
                    SELECT 1 FROM pages
                     WHERE status='active' AND id<>?1
                       AND space=COALESCE(?4,'00000000-0000-4000-8000-000000000001')
                       AND lower(title)=lower(?2)),
                 EXISTS(
                    SELECT 1 FROM pages
                     WHERE status='active' AND id<>?1
                       AND space=COALESCE(?4,'00000000-0000-4000-8000-000000000001')
                       AND lower(title)=lower(?3))",
            libsql::params![
                page_id,
                before_title,
                after_title,
                effective_scope.map(str::to_string)
            ],
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    if row.get::<i64>(0).map_err(database_error)? != 1
        || row.get::<i64>(1).map_err(database_error)? != 0
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

async fn validate_reclassification_review_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    finding: &wenlan_types::lint::LintSemanticFinding,
    memory_id: &str,
) -> Result<(String, RepairDigest, Vec<String>), WenlanError> {
    let affected_records = vec![RepairAffectedRecord::try_new(
        RepairAffectedRecordKind::Memory,
        memory_id.to_string(),
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?];
    let occurrence = crate::repair_plan::semantic_review_occurrence_digest(
        REPAIR_CLASSIFICATION_CHECK_ID,
        finding,
        &affected_records,
    )?;
    let review_id = format!("lint_review_{}", occurrence.as_str());
    let expected_source_ids = vec![memory_id.to_string()];
    let mut rows = snapshot
        .query(
            "SELECT action,source_ids,payload,status FROM refinement_queue
             WHERE id=?1 LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(review_id.clone())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let action = row.get::<String>(0).map_err(database_error)?;
    let source_ids = row.get::<String>(1).map_err(database_error)?;
    let payload = row.get::<Option<String>>(2).map_err(database_error)?;
    let status = row.get::<String>(3).map_err(database_error)?;
    if rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    validate_reclassification_review_row(
        &review_id,
        &action,
        &source_ids,
        payload.as_deref(),
        &status,
        &occurrence,
        &expected_source_ids,
    )?;
    Ok((review_id, occurrence, expected_source_ids))
}

pub(crate) async fn validate_reclassification_review_on_connection(
    connection: &libsql::Connection,
    binding: &RepairReviewBinding,
    memory_id: &str,
) -> Result<(), WenlanError> {
    let mut rows = connection
        .query(
            "SELECT action,source_ids,payload,status FROM refinement_queue
             WHERE id=?1 LIMIT 2",
            libsql::params![binding.review_id()],
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let action = row.get::<String>(0).map_err(database_error)?;
    let source_ids = row.get::<String>(1).map_err(database_error)?;
    let payload = row.get::<Option<String>>(2).map_err(database_error)?;
    let status = row.get::<String>(3).map_err(database_error)?;
    if rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    validate_reclassification_review_row(
        binding.review_id(),
        &action,
        &source_ids,
        payload.as_deref(),
        &status,
        binding.occurrence_digest(),
        &[memory_id.to_string()],
    )
}

fn validate_reclassification_review_row(
    review_id: &str,
    action: &str,
    source_ids_json: &str,
    payload: Option<&str>,
    status: &str,
    expected_occurrence: &RepairDigest,
    expected_source_ids: &[String],
) -> Result<(), WenlanError> {
    if action != "lint_repair_review" || status != "awaiting_review" {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let source_ids = serde_json::from_str::<Vec<String>>(source_ids_json)
        .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?;
    if source_ids != expected_source_ids {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let payload =
        payload.ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let decoded = crate::db::validate_lint_review_contract(review_id, &source_ids, payload)
        .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let wenlan_types::RefinementPayload::LintRepairReview {
        check_id,
        occurrence_digest,
        owner_binding_digest,
        ..
    } = decoded
    else {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    };
    if check_id != REPAIR_CLASSIFICATION_CHECK_ID
        || &occurrence_digest != expected_occurrence
        || owner_binding_digest
            != lint_review_owner_binding_digest(&occurrence_digest, &source_ids)?
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

async fn validate_rename_page_title_review_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    review_id: &str,
    page_id: &str,
) -> Result<RepairDigest, WenlanError> {
    let mut rows = snapshot
        .query(
            "SELECT action,source_ids,payload,status FROM refinement_queue
             WHERE id=?1 LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(review_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let action = row.get::<String>(0).map_err(database_error)?;
    let source_ids = row.get::<String>(1).map_err(database_error)?;
    let payload = row.get::<Option<String>>(2).map_err(database_error)?;
    let status = row.get::<String>(3).map_err(database_error)?;
    if rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    validate_rename_page_title_review_row(
        review_id,
        &action,
        &source_ids,
        payload.as_deref(),
        &status,
        page_id,
    )
}

pub(crate) async fn validate_rename_page_title_review_on_connection(
    connection: &libsql::Connection,
    review_id: &str,
    page_id: &str,
) -> Result<RepairDigest, WenlanError> {
    let mut rows = connection
        .query(
            "SELECT action,source_ids,payload,status FROM refinement_queue
             WHERE id=?1 LIMIT 2",
            libsql::params![review_id],
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let action = row.get::<String>(0).map_err(database_error)?;
    let source_ids = row.get::<String>(1).map_err(database_error)?;
    let payload = row.get::<Option<String>>(2).map_err(database_error)?;
    let status = row.get::<String>(3).map_err(database_error)?;
    if rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    validate_rename_page_title_review_row(
        review_id,
        &action,
        &source_ids,
        payload.as_deref(),
        &status,
        page_id,
    )
}

fn validate_rename_page_title_review_row(
    review_id: &str,
    action: &str,
    source_ids_json: &str,
    payload: Option<&str>,
    status: &str,
    page_id: &str,
) -> Result<RepairDigest, WenlanError> {
    if action != "lint_repair_review" || status != "awaiting_review" {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let source_ids = serde_json::from_str::<Vec<String>>(source_ids_json)
        .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?;
    if source_ids.len() != 1 || source_ids[0] != page_id {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let payload =
        payload.ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let decoded = crate::db::validate_lint_review_contract(review_id, &source_ids, payload)
        .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let wenlan_types::RefinementPayload::LintRepairReview {
        check_id,
        occurrence_digest,
        owner_binding_digest,
        ..
    } = decoded
    else {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    };
    if check_id != "pages.duplicate_active_titles"
        || owner_binding_digest
            != lint_review_owner_binding_digest(&occurrence_digest, &source_ids)?
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(occurrence_digest)
}

fn validate_entity_extraction_lint_finding(
    report: &LintReport,
    memory_id: &str,
) -> Result<(), WenlanError> {
    let check = report
        .checks()
        .iter()
        .find(|check| check.check_id() == "memories.enrichment_failures")
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?;
    if check.outcome() != LintOutcome::Finding
        || check.gate_effect() != LintGateEffect::Actionable
        || memory_id.trim() != memory_id
        || memory_id.is_empty()
    {
        return Err(WenlanError::Validation(
            "unsupported_repair_finding".to_string(),
        ));
    }
    Ok(())
}

async fn validate_entity_extraction_evidence_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    report: &LintReport,
    scope: &RepairLintScope,
    memory_id: &str,
) -> Result<(), WenlanError> {
    let (scope_clause, params) = match scope {
        RepairLintScope::Global => (
            String::new(),
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        ),
        RepairLintScope::Registered { space } => (
            " AND space=?2".to_string(),
            libsql::params::Params::Positional(vec![
                libsql::Value::Text(memory_id.to_string()),
                libsql::Value::Text(space.clone()),
            ]),
        ),
        RepairLintScope::Uncategorized => (
            // memories.space is NOT NULL since migration 91 — an unscoped
            // memory carries the reserved sentinel id, not NULL. Match both so
            // a not-yet-folded legacy NULL row still resolves.
            format!(
                " AND (space IS NULL OR space = '{}')",
                crate::db::UNFILED_SPACE_ID
            ),
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        ),
    };
    let target_sql = format!(
        "SELECT 1 FROM memories
         WHERE source='memory' AND source_id=?1{scope_clause} LIMIT 1"
    );
    let mut target_rows = snapshot
        .query(&target_sql, params.clone())
        .await
        .map_err(snapshot_error)?;
    if target_rows.next().await.map_err(snapshot_error)?.is_none() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    drop(target_rows);
    let sql = format!(
        "SELECT COUNT(DISTINCT source_id) FROM memories
          WHERE source='memory' AND source_id < ?1{scope_clause}"
    );
    let mut rows = snapshot.query(&sql, params).await.map_err(snapshot_error)?;
    let Some(row) = rows.next().await.map_err(snapshot_error)? else {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    };
    let position = usize::try_from(row.get::<i64>(0).map_err(database_error)?)
        .map_err(|_| WenlanError::Validation("unsupported_repair_finding".to_string()))?;
    let opaque_id = wenlan_types::lint::LintOpaqueId::from_sorted_position(position)
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?;
    let check = report
        .checks()
        .iter()
        .find(|check| check.check_id() == "memories.enrichment_failures")
        .expect("validated before snapshot evidence");
    if !check
        .evidence()
        .iter()
        .any(|evidence| matches!(evidence, LintEvidenceRef::OpaqueId { opaque_id: current } if *current == opaque_id))
    {
        return Err(WenlanError::Validation(
            "unsupported_repair_finding".to_string(),
        ));
    }
    Ok(())
}

async fn memory_columns_and_single_row_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    memory_id: &str,
    capture_bytes: &mut u64,
) -> Result<(Vec<String>, Vec<String>, Option<String>), WenlanError> {
    let mut column_rows = snapshot
        .query("PRAGMA table_info(memories)", libsql::params::Params::None)
        .await
        .map_err(snapshot_error)?;
    let mut columns = Vec::new();
    while let Some(row) = column_rows.next().await.map_err(snapshot_error)? {
        let column = row.get::<String>(1).map_err(database_error)?;
        reserve_entity_extraction_capture(capture_bytes, column.len())?;
        columns.push(column);
    }
    drop(column_rows);
    if columns.is_empty() {
        return Err(WenlanError::Validation(
            "repair_target_schema_missing".to_string(),
        ));
    }
    let selected = columns
        .iter()
        .map(|column| entity_extraction_encoded_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    preflight_entity_extraction_memory_on_snapshot(snapshot, memory_id, &columns, *capture_bytes)
        .await?;
    let sql = format!(
        "SELECT {selected} FROM memories
         WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id LIMIT 2"
    );
    let mut rows = snapshot
        .query(
            &sql,
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let mut values = Vec::with_capacity(columns.len());
    let mut space = None;
    for (index, column) in columns.iter().enumerate() {
        let index = i32::try_from(index)
            .map_err(|_| WenlanError::Validation("repair_target_schema_too_wide".to_string()))?;
        let value = row.get::<String>(index).map_err(database_error)?;
        reserve_entity_extraction_capture(capture_bytes, value.len())?;
        if column == "space" {
            space = decode_entity_extraction_space(&value)?;
        }
        values.push(value);
    }
    if !columns.iter().any(|column| column == "space") {
        return Err(WenlanError::Validation(
            "repair_target_schema_mismatch".to_string(),
        ));
    }
    if rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok((columns, values, space))
}

async fn capture_complete_entity_extraction_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    memory_id: &str,
) -> Result<(RepairRollbackPayloadV2, RepairScope), WenlanError> {
    let mut capture_bytes = 0_u64;
    let (memory_columns, before_memory_row, space) =
        memory_columns_and_single_row_on_snapshot(snapshot, memory_id, &mut capture_bytes).await?;
    preflight_entity_extraction_links_on_snapshot(snapshot, memory_id, capture_bytes).await?;
    let mut link_rows = snapshot
        .query(
            "SELECT entity_id FROM memory_entities
             WHERE memory_id=?1 ORDER BY entity_id",
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let mut before_entity_ids = Vec::new();
    while let Some(row) = link_rows.next().await.map_err(snapshot_error)? {
        let entity_id = row.get::<String>(0).map_err(database_error)?;
        reserve_entity_extraction_capture(&mut capture_bytes, entity_id.len())?;
        before_entity_ids.push(entity_id);
    }
    drop(link_rows);
    preflight_entity_extraction_step_on_snapshot(snapshot, memory_id, capture_bytes).await?;
    let mut step_rows = snapshot
        .query(
            "SELECT status,error,attempts,updated_at FROM enrichment_steps
             WHERE source_id=?1 AND step_name='entity_extract' LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let step = step_rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let enrichment_status = step.get::<String>(0).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: enrichment status: {error}"))
    })?;
    let enrichment_error = step.get::<Option<String>>(1).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: enrichment error: {error}"))
    })?;
    reserve_entity_extraction_capture(&mut capture_bytes, enrichment_status.len())?;
    if let Some(error) = &enrichment_error {
        reserve_entity_extraction_capture(&mut capture_bytes, error.len())?;
    }
    let enrichment_attempts = step.get::<i64>(2).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: enrichment attempts: {error}"))
    })?;
    let enrichment_updated_at = step.get::<i64>(3).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: enrichment updated_at: {error}"))
    })?;
    if step_rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let target_scope = match space {
        Some(space) => RepairScope::registered(space),
        None => Ok(RepairScope::uncategorized()),
    }
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    let payload = RepairRollbackPayloadV2::complete_entity_extraction(
        memory_id.to_string(),
        memory_columns,
        before_memory_row,
        before_entity_ids,
        enrichment_status,
        enrichment_error,
        enrichment_attempts,
        enrichment_updated_at,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))?;
    Ok((payload, target_scope))
}

pub(crate) async fn capture_complete_entity_extraction_on_connection(
    connection: &libsql::Connection,
    memory_id: &str,
) -> Result<RepairRollbackPayloadV2, WenlanError> {
    let mut capture_bytes = 0_u64;
    let (memory_columns, before_memory_row) =
        memory_columns_and_single_row_on_connection(connection, memory_id, &mut capture_bytes)
            .await?;
    preflight_entity_extraction_links_on_connection(connection, memory_id, capture_bytes).await?;
    let mut link_rows = connection
        .query(
            "SELECT entity_id FROM memory_entities
             WHERE memory_id=?1 ORDER BY entity_id",
            libsql::params![memory_id],
        )
        .await
        .map_err(database_error)?;
    let mut before_entity_ids = Vec::new();
    while let Some(row) = link_rows.next().await.map_err(database_error)? {
        let entity_id = row.get::<String>(0).map_err(database_error)?;
        reserve_entity_extraction_capture(&mut capture_bytes, entity_id.len())?;
        before_entity_ids.push(entity_id);
    }
    drop(link_rows);
    preflight_entity_extraction_step_on_connection(connection, memory_id, capture_bytes).await?;
    let mut step_rows = connection
        .query(
            "SELECT status,error,attempts,updated_at FROM enrichment_steps
             WHERE source_id=?1 AND step_name='entity_extract' LIMIT 2",
            libsql::params![memory_id],
        )
        .await
        .map_err(database_error)?;
    let step = step_rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let enrichment_status = step.get::<String>(0).map_err(database_error)?;
    let enrichment_error = step.get::<Option<String>>(1).map_err(database_error)?;
    reserve_entity_extraction_capture(&mut capture_bytes, enrichment_status.len())?;
    if let Some(error) = &enrichment_error {
        reserve_entity_extraction_capture(&mut capture_bytes, error.len())?;
    }
    let enrichment_attempts = step.get::<i64>(2).map_err(database_error)?;
    let enrichment_updated_at = step.get::<i64>(3).map_err(database_error)?;
    if step_rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    RepairRollbackPayloadV2::complete_entity_extraction(
        memory_id.to_string(),
        memory_columns,
        before_memory_row,
        before_entity_ids,
        enrichment_status,
        enrichment_error,
        enrichment_attempts,
        enrichment_updated_at,
    )
    .map_err(|error| WenlanError::Validation(error.to_string()))
}

async fn memory_columns_and_single_row_on_connection(
    connection: &libsql::Connection,
    memory_id: &str,
    capture_bytes: &mut u64,
) -> Result<(Vec<String>, Vec<String>), WenlanError> {
    let mut column_rows = connection
        .query("PRAGMA table_info(memories)", libsql::params::Params::None)
        .await
        .map_err(database_error)?;
    let mut columns = Vec::new();
    while let Some(row) = column_rows.next().await.map_err(database_error)? {
        let column = row.get::<String>(1).map_err(database_error)?;
        reserve_entity_extraction_capture(capture_bytes, column.len())?;
        columns.push(column);
    }
    drop(column_rows);
    if columns.is_empty() {
        return Err(WenlanError::Validation(
            "repair_target_schema_missing".to_string(),
        ));
    }
    let selected = columns
        .iter()
        .map(|column| entity_extraction_encoded_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    preflight_entity_extraction_memory_on_connection(
        connection,
        memory_id,
        &columns,
        *capture_bytes,
    )
    .await?;
    let sql = format!(
        "SELECT {selected} FROM memories
         WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id LIMIT 2"
    );
    let mut rows = connection
        .query(
            &sql,
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let mut values = Vec::with_capacity(columns.len());
    for index in 0..columns.len() {
        let index = i32::try_from(index)
            .map_err(|_| WenlanError::Validation("repair_target_schema_too_wide".to_string()))?;
        let value = row.get::<String>(index).map_err(database_error)?;
        reserve_entity_extraction_capture(capture_bytes, value.len())?;
        values.push(value);
    }
    if rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok((columns, values))
}

async fn preflight_entity_extraction_memory_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    memory_id: &str,
    columns: &[String],
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let selected = columns
        .iter()
        .map(|column| entity_extraction_encoded_length_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT {selected} FROM memories
         WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id LIMIT 2"
    );
    let mut rows = snapshot
        .query(
            &sql,
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    ensure_entity_extraction_memory_lengths(&row, columns.len(), captured_bytes)?;
    if rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

async fn preflight_entity_extraction_memory_on_connection(
    connection: &libsql::Connection,
    memory_id: &str,
    columns: &[String],
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let selected = columns
        .iter()
        .map(|column| entity_extraction_encoded_length_expression(column))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT {selected} FROM memories
         WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id LIMIT 2"
    );
    let mut rows = connection
        .query(
            &sql,
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    ensure_entity_extraction_memory_lengths(&row, columns.len(), captured_bytes)?;
    if rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

fn ensure_entity_extraction_memory_lengths(
    row: &libsql::Row,
    column_count: usize,
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let mut projected_bytes = captured_bytes;
    for index in 0..column_count {
        let index = i32::try_from(index)
            .map_err(|_| WenlanError::Validation("repair_target_schema_too_wide".to_string()))?;
        let encoded_length = row.get::<i64>(index).map_err(database_error)?;
        let encoded_length = u64::try_from(encoded_length).map_err(|_| {
            WenlanError::Validation("repair_rollback_artifact_too_large".to_string())
        })?;
        reserve_entity_extraction_capture_u64(&mut projected_bytes, encoded_length)?;
    }
    Ok(())
}

async fn preflight_entity_extraction_links_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    memory_id: &str,
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let mut rows = snapshot
        .query(
            "SELECT COUNT(*),COALESCE(SUM(length(CAST(entity_id AS BLOB))),0)
               FROM memory_entities WHERE memory_id=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    ensure_entity_extraction_link_lengths(&row, captured_bytes)
}

async fn preflight_entity_extraction_links_on_connection(
    connection: &libsql::Connection,
    memory_id: &str,
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let mut rows = connection
        .query(
            "SELECT COUNT(*),COALESCE(SUM(length(CAST(entity_id AS BLOB))),0)
               FROM memory_entities WHERE memory_id=?1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    ensure_entity_extraction_link_lengths(&row, captured_bytes)
}

async fn preflight_entity_extraction_step_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    memory_id: &str,
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let mut rows = snapshot
        .query(
            "SELECT length(CAST(status AS BLOB)),
                    CASE WHEN error IS NULL THEN 0
                         ELSE length(CAST(error AS BLOB)) END
               FROM enrichment_steps
              WHERE source_id=?1 AND step_name='entity_extract'
              LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    ensure_entity_extraction_step_lengths(&row, captured_bytes)?;
    if rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

async fn preflight_entity_extraction_step_on_connection(
    connection: &libsql::Connection,
    memory_id: &str,
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let mut rows = connection
        .query(
            "SELECT length(CAST(status AS BLOB)),
                    CASE WHEN error IS NULL THEN 0
                         ELSE length(CAST(error AS BLOB)) END
               FROM enrichment_steps
              WHERE source_id=?1 AND step_name='entity_extract'
              LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(memory_id.to_string())]),
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    ensure_entity_extraction_step_lengths(&row, captured_bytes)?;
    if rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

fn ensure_entity_extraction_link_lengths(
    row: &libsql::Row,
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let count = u64::try_from(row.get::<i64>(0).map_err(database_error)?)
        .map_err(|_| WenlanError::Validation("repair_rollback_artifact_too_large".to_string()))?;
    let total_length = u64::try_from(row.get::<i64>(1).map_err(database_error)?)
        .map_err(|_| WenlanError::Validation("repair_rollback_artifact_too_large".to_string()))?;
    let worst_case_json_bytes = total_length
        .checked_mul(6)
        .and_then(|length| {
            count
                .checked_mul(3)
                .and_then(|overhead| length.checked_add(overhead))
        })
        .ok_or_else(|| WenlanError::Validation("repair_rollback_artifact_too_large".to_string()))?;
    let mut projected_bytes = captured_bytes;
    reserve_entity_extraction_capture_u64(&mut projected_bytes, worst_case_json_bytes)
}

fn ensure_entity_extraction_step_lengths(
    row: &libsql::Row,
    captured_bytes: u64,
) -> Result<(), WenlanError> {
    let status_length = u64::try_from(row.get::<i64>(0).map_err(database_error)?)
        .map_err(|_| WenlanError::Validation("repair_rollback_artifact_too_large".to_string()))?;
    let error_length = u64::try_from(row.get::<i64>(1).map_err(database_error)?)
        .map_err(|_| WenlanError::Validation("repair_rollback_artifact_too_large".to_string()))?;
    let worst_case_json_bytes = status_length
        .checked_add(error_length)
        .and_then(|length| length.checked_mul(6))
        .and_then(|length| length.checked_add(6))
        .ok_or_else(|| WenlanError::Validation("repair_rollback_artifact_too_large".to_string()))?;
    let mut projected_bytes = captured_bytes;
    reserve_entity_extraction_capture_u64(&mut projected_bytes, worst_case_json_bytes)
}

fn reserve_entity_extraction_capture(
    capture_bytes: &mut u64,
    additional_bytes: usize,
) -> Result<(), WenlanError> {
    reserve_entity_extraction_capture_u64(
        capture_bytes,
        u64::try_from(additional_bytes).unwrap_or(u64::MAX),
    )
}

fn reserve_entity_extraction_capture_u64(
    capture_bytes: &mut u64,
    additional_bytes: u64,
) -> Result<(), WenlanError> {
    let next = capture_bytes
        .checked_add(additional_bytes)
        .ok_or_else(|| WenlanError::Validation("repair_rollback_artifact_too_large".to_string()))?;
    if next > REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES {
        return Err(WenlanError::Validation(
            "repair_rollback_artifact_too_large".to_string(),
        ));
    }
    *capture_bytes = next;
    Ok(())
}

fn entity_extraction_encoded_expression(column: &str) -> String {
    let column = quote_identifier(column);
    format!(
        "CASE typeof({column})
           WHEN 'null' THEN 'n:'
           WHEN 'integer' THEN 'i:' || printf('%lld', {column})
           WHEN 'real' THEN 'r:' || printf('%!.26g', {column})
           WHEN 'text' THEN 't:' || lower(hex({column}))
           WHEN 'blob' THEN 'b:' || lower(hex({column}))
           ELSE 'x:' || typeof({column}) || ':' || lower(hex({column}))
         END"
    )
}

fn entity_extraction_encoded_length_expression(column: &str) -> String {
    let column = quote_identifier(column);
    format!(
        "CASE typeof({column})
           WHEN 'null' THEN 2
           WHEN 'integer' THEN 2 + length(printf('%lld', {column}))
           WHEN 'real' THEN 2 + length(printf('%!.26g', {column}))
           WHEN 'text' THEN 2 + (2 * length(CAST({column} AS BLOB)))
           WHEN 'blob' THEN 2 + (2 * length(CAST({column} AS BLOB)))
           ELSE 3 + length(typeof({column}))
                + (2 * length(CAST({column} AS BLOB)))
         END"
    )
}

fn decode_entity_extraction_space(encoded: &str) -> Result<Option<String>, WenlanError> {
    if encoded == "n:" {
        return Ok(None);
    }
    let Some(value) = encoded.strip_prefix("t:") else {
        return Err(WenlanError::Validation(
            "repair_target_schema_mismatch".to_string(),
        ));
    };
    let bytes = hex::decode(value)
        .map_err(|_| WenlanError::Validation("repair_target_schema_mismatch".to_string()))?;
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| WenlanError::Validation("repair_target_schema_mismatch".to_string()))
}

async fn validate_selected_entities_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    entity_ids: &[String],
    space: Option<&str>,
) -> Result<(), WenlanError> {
    for entity_id in entity_ids {
        let mut rows = snapshot
            .query(
                "SELECT 1 FROM entities
                 WHERE id=?1 AND ((?2 IS NULL AND space IS NULL) OR space=?2)
                 LIMIT 2",
                libsql::params::Params::Positional(vec![
                    libsql::Value::Text(entity_id.clone()),
                    space
                        .map(|value| libsql::Value::Text(value.to_string()))
                        .unwrap_or(libsql::Value::Null),
                ]),
            )
            .await
            .map_err(snapshot_error)?;
        if rows.next().await.map_err(snapshot_error)?.is_none()
            || rows.next().await.map_err(snapshot_error)?.is_some()
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
    }
    Ok(())
}

pub(crate) async fn validate_selected_entities_on_connection(
    connection: &libsql::Connection,
    entity_ids: &[String],
    space: Option<&str>,
) -> Result<(), WenlanError> {
    for entity_id in entity_ids {
        let mut rows = connection
            .query(
                "SELECT 1 FROM entities
                 WHERE id=?1 AND ((?2 IS NULL AND space IS NULL) OR space=?2)
                 LIMIT 2",
                libsql::params![entity_id.clone(), space.map(str::to_string)],
            )
            .await
            .map_err(database_error)?;
        if rows.next().await.map_err(database_error)?.is_none()
            || rows.next().await.map_err(database_error)?.is_some()
        {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
    }
    Ok(())
}

async fn validate_complete_entity_extraction_review_on_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    review_id: &str,
    expected_source_ids: &[String],
) -> Result<RepairDigest, WenlanError> {
    let mut rows = snapshot
        .query(
            "SELECT action,source_ids,payload,status FROM refinement_queue
             WHERE id=?1 LIMIT 2",
            libsql::params::Params::Positional(vec![libsql::Value::Text(review_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    // libSQL rows may reuse their backing row buffer after `next()`. Read the
    // complete review binding before probing for an unexpected second row.
    let action = row.get::<String>(0).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: review action: {error}"))
    })?;
    let source_ids = row.get::<String>(1).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: review source_ids: {error}"))
    })?;
    let payload = row.get::<Option<String>>(2).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: review payload: {error}"))
    })?;
    let status = row.get::<String>(3).map_err(|error| {
        WenlanError::VectorDb(format!("repair database: review status: {error}"))
    })?;
    if rows.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    validate_complete_entity_extraction_review_row(
        review_id,
        &action,
        &source_ids,
        payload.as_deref(),
        &status,
        expected_source_ids,
    )
}

pub(crate) async fn validate_complete_entity_extraction_review_on_connection(
    connection: &libsql::Connection,
    review_id: &str,
    expected_source_ids: &[String],
) -> Result<RepairDigest, WenlanError> {
    let mut rows = connection
        .query(
            "SELECT action,source_ids,payload,status FROM refinement_queue
             WHERE id=?1 LIMIT 2",
            libsql::params![review_id],
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let action = row.get::<String>(0).map_err(database_error)?;
    let source_ids = row.get::<String>(1).map_err(database_error)?;
    let payload = row.get::<Option<String>>(2).map_err(database_error)?;
    let status = row.get::<String>(3).map_err(database_error)?;
    if rows.next().await.map_err(database_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    validate_complete_entity_extraction_review_row(
        review_id,
        &action,
        &source_ids,
        payload.as_deref(),
        &status,
        expected_source_ids,
    )
    .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))
}

fn validate_complete_entity_extraction_review_row(
    review_id: &str,
    action: &str,
    source_ids_json: &str,
    payload: Option<&str>,
    status: &str,
    expected_source_ids: &[String],
) -> Result<RepairDigest, WenlanError> {
    if action != "lint_repair_review" || status != "awaiting_review" {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let source_ids = serde_json::from_str::<Vec<String>>(source_ids_json)
        .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?;
    if source_ids != expected_source_ids {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let payload =
        payload.ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let decoded = crate::db::validate_lint_review_contract(review_id, &source_ids, payload)
        .map_err(|_| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let wenlan_types::RefinementPayload::LintRepairReview {
        check_id,
        occurrence_digest,
        owner_binding_digest,
        ..
    } = decoded
    else {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    };
    if check_id != "memories.enrichment_failures"
        || owner_binding_digest
            != lint_review_owner_binding_digest(&occurrence_digest, &source_ids)?
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(occurrence_digest)
}

pub(crate) async fn repair_target_receipt_on_connection(
    connection: &libsql::Connection,
    target: &RepairTarget,
) -> Result<(RepairDigest, u64), WenlanError> {
    match target {
        RepairTarget::Memory { source_id, scope } => {
            validate_target_space_on_connection(connection, source_id, scope.space()).await?;
            target_receipt_on_connection(connection, source_id).await
        }
        RepairTarget::MemoryEntityLink {
            memory_id,
            entity_id,
            scope,
        } => {
            validate_memory_entity_scope_on_connection(connection, memory_id, scope).await?;
            let rollback =
                capture_memory_entity_link_on_connection(connection, memory_id, entity_id).await?;
            let count = u64::try_from(rollback.rows.len())
                .map_err(|_| WenlanError::Validation("repair target too large".to_string()))?;
            Ok((target_receipt(&rollback)?, count))
        }
        RepairTarget::MemoryEntityExtraction {
            memory_id, scope, ..
        } => {
            validate_target_space_on_connection(connection, memory_id, scope.space()).await?;
            let rollback =
                capture_complete_entity_extraction_on_connection(connection, memory_id).await?;
            Ok((complete_entity_extraction_receipt(&rollback)?, 1))
        }
        RepairTarget::Tag {
            source,
            source_id,
            tag,
            ..
        } => {
            let mut rows = connection
                .query(
                    "SELECT source,source_id,tag FROM document_tags
                     WHERE source=?1 AND source_id=?2 AND tag=?3",
                    libsql::params![source.clone(), source_id.clone(), tag.clone()],
                )
                .await
                .map_err(database_error)?;
            let mut values = Vec::new();
            while let Some(row) = rows.next().await.map_err(database_error)? {
                values.push(vec![
                    row.get::<String>(0).map_err(database_error)?,
                    row.get::<String>(1).map_err(database_error)?,
                    row.get::<String>(2).map_err(database_error)?,
                ]);
            }
            let rollback = StoredRollbackArtifact {
                format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
                table: "document_tags".to_string(),
                source_id: serde_json::to_string(&[source, source_id, tag])?,
                columns: vec![
                    "source".to_string(),
                    "source_id".to_string(),
                    "tag".to_string(),
                ],
                rows: values,
            };
            let count = u64::try_from(rollback.rows.len())
                .map_err(|_| WenlanError::Validation("repair target too large".to_string()))?;
            Ok((target_receipt(&rollback)?, count))
        }
        RepairTarget::PageLink {
            source_page_id,
            label_key,
            scope,
        } => {
            let mut rows = connection
                .query(
                    "SELECT pl.source_page_id,pl.target_page_id,pl.label_key,
                            p.space
                       FROM page_links pl JOIN pages p ON p.id=pl.source_page_id
                      WHERE pl.source_page_id=?1 AND pl.label_key=?2",
                    libsql::params![source_page_id.clone(), label_key.clone()],
                )
                .await
                .map_err(database_error)?;
            let mut values = Vec::new();
            while let Some(row) = rows.next().await.map_err(database_error)? {
                let actual_scope = row.get::<Option<String>>(3).map_err(database_error)?;
                if actual_scope.as_deref() != scope.space() {
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
                values.push(vec![
                    row.get::<String>(0).map_err(database_error)?,
                    row.get::<Option<String>>(1)
                        .map_err(database_error)?
                        .unwrap_or_else(|| "NULL".to_string()),
                    row.get::<String>(2).map_err(database_error)?,
                ]);
            }
            let rollback = StoredRollbackArtifact {
                format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
                table: "page_links".to_string(),
                source_id: serde_json::to_string(&[source_page_id, label_key])?,
                columns: vec![
                    "source_page_id".to_string(),
                    "target_page_id".to_string(),
                    "label_key".to_string(),
                ],
                rows: values,
            };
            let count = u64::try_from(rollback.rows.len())
                .map_err(|_| WenlanError::Validation("repair target too large".to_string()))?;
            Ok((target_receipt(&rollback)?, count))
        }
        RepairTarget::Page { page_id, scope } => {
            validate_page_scope_on_connection(connection, page_id, scope).await?;
            let rollback = capture_page_on_connection(connection, page_id).await?;
            Ok((target_receipt(&rollback)?, 1))
        }
        RepairTarget::PageProjection { .. } => Err(WenlanError::Validation(
            "page projection repair receipt requires page root".to_string(),
        )),
    }
}

async fn validate_page_scope_on_connection(
    connection: &libsql::Connection,
    page_id: &str,
    scope: &RepairScope,
) -> Result<(), WenlanError> {
    let mut rows = connection
        .query(
            "SELECT space FROM pages WHERE id=?1",
            libsql::params![page_id],
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    let actual = row.get::<Option<String>>(0).map_err(database_error)?;
    if actual.as_deref() != scope.space() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

async fn capture_page_on_connection(
    connection: &libsql::Connection,
    page_id: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let mut column_rows = connection
        .query("PRAGMA table_info(pages)", ())
        .await
        .map_err(database_error)?;
    let mut columns = Vec::new();
    while let Some(row) = column_rows.next().await.map_err(database_error)? {
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
        .map(|column| format!("quote({})", quote_identifier(column)))
        .collect::<Vec<_>>()
        .join(",");
    let mut rows = connection
        .query(
            &format!("SELECT {selected} FROM pages WHERE id=?1"),
            libsql::params![page_id],
        )
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
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

async fn validate_memory_entity_scope_on_connection(
    connection: &libsql::Connection,
    memory_id: &str,
    scope: &RepairScope,
) -> Result<(), WenlanError> {
    match scope {
        RepairScope::Global => {
            let mut rows = connection
                .query(
                    "SELECT 1 FROM memories WHERE source_id=?1 LIMIT 1",
                    libsql::params![memory_id],
                )
                .await
                .map_err(database_error)?;
            if rows.next().await.map_err(database_error)?.is_some() {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            Ok(())
        }
        RepairScope::Registered { .. } | RepairScope::Uncategorized => {
            let mut rows = connection
                .query(
                    "SELECT space FROM memories
                     WHERE source_id=?1 ORDER BY chunk_index,id",
                    libsql::params![memory_id],
                )
                .await
                .map_err(database_error)?;
            let mut seen = 0_u64;
            while let Some(row) = rows.next().await.map_err(database_error)? {
                // M3 PR-1 stage e: memories.space is NOT NULL as of migration
                // 85, so an unscoped memory carries the reserved sentinel id
                // rather than NULL -- translate it back to None here so it
                // still matches an Uncategorized scope's `scope.space() ==
                // None`, the same translation prepare-time scope resolution
                // already applies (see repair_plan/deterministic.rs).
                let actual = row
                    .get::<Option<String>>(0)
                    .map_err(database_error)?
                    .filter(|s| s != crate::db::UNFILED_SPACE_ID);
                if actual.as_deref() != scope.space() {
                    return Err(WenlanError::Conflict("repair_target_stale".to_string()));
                }
                seen = seen.saturating_add(1);
            }
            if seen == 0 {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            Ok(())
        }
    }
}

async fn capture_memory_entity_link_on_connection(
    connection: &libsql::Connection,
    memory_id: &str,
    entity_id: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let mut rows = connection
        .query(
            "SELECT memory_id,entity_id FROM memory_entities
             WHERE memory_id=?1 AND entity_id=?2",
            libsql::params![memory_id, entity_id],
        )
        .await
        .map_err(database_error)?;
    let mut values = Vec::new();
    while let Some(row) = rows.next().await.map_err(database_error)? {
        values.push(vec![
            row.get::<String>(0).map_err(database_error)?,
            row.get::<String>(1).map_err(database_error)?,
        ]);
    }
    Ok(StoredRollbackArtifact {
        format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
        table: "memory_entities".to_string(),
        source_id: serde_json::to_string(&[memory_id, entity_id])?,
        columns: vec!["memory_id".to_string(), "entity_id".to_string()],
        rows: values,
    })
}

pub(crate) async fn validate_target_space_on_connection(
    connection: &libsql::Connection,
    source_id: &str,
    expected_space: Option<&str>,
) -> Result<(), WenlanError> {
    let mut rows = connection
        .query(
            "SELECT space FROM memories
             WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id",
            libsql::params![source_id],
        )
        .await
        .map_err(database_error)?;
    let mut seen = 0_u64;
    while let Some(row) = rows.next().await.map_err(database_error)? {
        // M3 PR-1 stage e: memories.space is NOT NULL as of migration 91, so
        // an unscoped memory carries the reserved sentinel id rather than
        // NULL -- translate it back to None here so it still matches an
        // Uncategorized target's `expected_space == None`, the same
        // translation prepare-time scope resolution already applies (see
        // repair_plan/deterministic.rs).
        let actual: Option<String> = row
            .get::<Option<String>>(0)
            .map_err(database_error)?
            .filter(|s| s != crate::db::UNFILED_SPACE_ID);
        if actual.as_deref() != expected_space {
            return Err(WenlanError::Conflict("repair_target_stale".to_string()));
        }
        seen = seen.saturating_add(1);
    }
    if seen == 0 {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(())
}

pub(crate) fn effect_guard_receipt(normalized_total_changes: u64) -> RepairDigest {
    let mut bytes = b"wenlan-repair-effect-guard-v1".to_vec();
    bytes.extend_from_slice(&normalized_total_changes.to_le_bytes());
    repair_digest(&bytes)
}

fn validate_selected_finding(request: &PrepareRepairRequest) -> Result<(), WenlanError> {
    let Some(selected_finding) = request.selected_finding() else {
        return Err(WenlanError::Validation(
            "unsupported_repair_finding".to_string(),
        ));
    };
    let Some(deep) = request.deep_report() else {
        return Err(WenlanError::Validation(
            "repair_deep_report_missing".to_string(),
        ));
    };
    if selected_finding.proposed_action() != LintSemanticAction::ReclassifyMemory
        || selected_finding.evidence_ids().len() != 1
        || !selected_finding.counterevidence_ids().is_empty()
    {
        return Err(WenlanError::Validation(
            "unsupported_repair_finding".to_string(),
        ));
    }
    let present = deep.checks().iter().any(|check| {
        check.check_id() == wenlan_types::repair::REPAIR_CLASSIFICATION_CHECK_ID
            && check.evidence().iter().any(|evidence| {
                matches!(evidence, LintEvidenceRef::SemanticFinding { finding } if finding == selected_finding)
            })
    });
    if !present {
        return Err(WenlanError::Validation(
            "repair_finding_not_in_deep_report".to_string(),
        ));
    }
    Ok(())
}

pub(crate) async fn validate_durable_scope(
    snapshot: &LintReadSnapshot<'_>,
    lint_scope: &RepairLintScope,
    report_scope: &LintScope,
) -> Result<(), WenlanError> {
    if !lint_scope.matches_report_scope_kind(report_scope) {
        return Err(WenlanError::Validation("repair_scope_mismatch".to_string()));
    }
    let RepairLintScope::Registered { space } = lint_scope else {
        return Ok(());
    };
    let mut rows = snapshot
        .query(
            "SELECT (SELECT COUNT(*) FROM spaces prior WHERE prior.name < current.name)
             FROM spaces current WHERE current.name=?1 LIMIT 1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        )
        .await
        .map_err(snapshot_error)?;
    let Some(row) = rows.next().await.map_err(snapshot_error)? else {
        return Err(WenlanError::Validation(
            "repair_scope_not_registered".to_string(),
        ));
    };
    let position = usize::try_from(row.get::<i64>(0).map_err(database_error)?)
        .map_err(|_| WenlanError::Validation("repair_scope_mismatch".to_string()))?;
    let expected = wenlan_types::lint::LintOpaqueId::from_sorted_position(position)
        .ok_or_else(|| WenlanError::Validation("repair_scope_mismatch".to_string()))?;
    if report_scope.kind() != LintScopeKind::Registered
        || report_scope.opaque_scope_ref() != Some(expected)
    {
        return Err(WenlanError::Validation("repair_scope_mismatch".to_string()));
    }
    Ok(())
}

async fn resolve_target(
    snapshot: &LintReadSnapshot<'_>,
    request: &PrepareRepairRequest,
) -> Result<ResolvedTarget, WenlanError> {
    let (scope_clause, params) = match request.lint_scope() {
        RepairLintScope::Global => (String::new(), libsql::params::Params::None),
        RepairLintScope::Registered { space } => (
            " AND m.space=?1".to_string(),
            libsql::params::Params::Positional(vec![libsql::Value::Text(space.clone())]),
        ),
        RepairLintScope::Uncategorized => (
            // memories.space is NOT NULL since migration 91 — an unscoped
            // memory carries the reserved sentinel id, not NULL. Match both so
            // a not-yet-folded legacy NULL row still resolves.
            format!(
                " AND (m.space IS NULL OR m.space = '{}')",
                crate::db::UNFILED_SPACE_ID
            ),
            libsql::params::Params::None,
        ),
    };
    let sql = format!(
        "SELECT m.source_id,m.memory_type,m.space,m.version
         FROM memories m
         WHERE m.source='memory' AND m.chunk_index=0 AND m.pending_revision=0
           AND COALESCE(m.is_recap,0)=0 AND m.supersede_mode!='evicted'{scope_clause}
         ORDER BY m.source_id,m.id"
    );
    let target_evidence = request
        .selected_finding()
        .ok_or_else(|| WenlanError::Validation("unsupported_repair_finding".to_string()))?
        .evidence_ids()[0]
        .clone();
    let mut rows = snapshot.query(&sql, params).await.map_err(snapshot_error)?;
    let mut matches = Vec::new();
    while let Some(row) = rows.next().await.map_err(snapshot_error)? {
        let source_id: String = row.get(0).map_err(database_error)?;
        if semantic_record_digest("memory", &source_id) == target_evidence {
            matches.push(ResolvedTarget {
                source_id,
                memory_type: row.get(1).map_err(database_error)?,
                space: row.get(2).map_err(database_error)?,
                version: row.get(3).map_err(database_error)?,
                evidence_id: target_evidence.clone(),
            });
        }
    }
    if matches.len() != 1 {
        return Err(WenlanError::Validation(
            "repair_target_not_unique".to_string(),
        ));
    }
    Ok(matches.remove(0))
}

pub(crate) async fn capture_rollback(
    snapshot: &LintReadSnapshot<'_>,
    source_id: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let mut column_rows = snapshot
        .query("PRAGMA table_info(memories)", libsql::params::Params::None)
        .await
        .map_err(snapshot_error)?;
    let mut columns = Vec::new();
    while let Some(row) = column_rows.next().await.map_err(snapshot_error)? {
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
        .map(|column| format!("quote({})", quote_identifier(column)))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT {selected} FROM memories
         WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id"
    );
    let mut query_rows = snapshot
        .query(
            &sql,
            libsql::params::Params::Positional(vec![libsql::Value::Text(source_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let mut rows = Vec::new();
    while let Some(row) = query_rows.next().await.map_err(snapshot_error)? {
        let mut values = Vec::with_capacity(columns.len());
        for index in 0..columns.len() {
            let index = i32::try_from(index).map_err(|_| {
                WenlanError::Validation("repair_target_schema_too_wide".to_string())
            })?;
            values.push(row.get::<String>(index).map_err(database_error)?);
        }
        rows.push(values);
    }
    if rows.is_empty() {
        return Err(WenlanError::NotFound("repair_target_missing".to_string()));
    }
    Ok(StoredRollbackArtifact {
        format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
        table: "memories".to_string(),
        source_id: source_id.to_string(),
        columns,
        rows,
    })
}

pub(crate) async fn capture_page_projection_rollback(
    snapshot: &LintReadSnapshot<'_>,
    page_root: &Path,
    mismatch: &crate::lint::pages::state_checks::ProjectionVersionMismatch,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let (page_row, _) = projection_page_row_from_snapshot(snapshot, &mismatch.page_id).await?;
    if page_row
        .get(7)
        .is_none_or(|version| version != &format!("{}", mismatch.database_version))
    {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    let paths = projection_closure_paths(&mismatch.target_path)?;
    projection_rollback_from_paths(
        &mismatch.page_id,
        page_row,
        page_root,
        &paths,
        PAGE_PROJECTION_ROLLBACK_TABLE_V2,
    )
}

pub(crate) async fn capture_stale_page_projection_rollback(
    snapshot: &LintReadSnapshot<'_>,
    page_root: &Path,
    page_id: &str,
    source_path: &str,
    quarantine_path: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let mut owner = snapshot
        .query(
            "SELECT 1 FROM pages WHERE id=?1 LIMIT 1",
            libsql::params::Params::Positional(vec![libsql::Value::Text(page_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    if owner.next().await.map_err(snapshot_error)?.is_some() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    drop(owner);
    crate::export::knowledge::KnowledgeProjectionWrite::with_projection_lock(
        page_root,
        |projection| {
            let scan = projection.scan_page_root_controlled(
                true,
                &PageScanControl::with_timeout(std::time::Duration::from_secs(30)),
            )?;
            if !matches!(
                crate::lint::pages::state_checks::stale_projection_ownership(&scan, page_id),
                crate::lint::pages::state_checks::StaleProjectionOwnership::Exact {
                    source_path: ref current_source,
                    quarantine_path: ref current_quarantine,
                } if current_source == source_path && current_quarantine == quarantine_path
            ) {
                return Err(WenlanError::Conflict("repair_target_stale".to_string()));
            }
            projection.capture_stale_page_projection_current(page_id, source_path, quarantine_path)
        },
    )
}

pub(crate) fn capture_stale_page_projection_current(
    page_root: &Path,
    page_id: &str,
    source_path: &str,
    quarantine_path: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    crate::export::knowledge::KnowledgeProjectionWrite::with_projection_lock(
        page_root,
        |projection| {
            projection.capture_stale_page_projection_current(page_id, source_path, quarantine_path)
        },
    )
}

pub(crate) fn capture_stale_page_projection_current_locked(
    projection: &crate::export::knowledge::LockedProjection<'_>,
    page_id: &str,
    source_path: &str,
    quarantine_path: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    for relative in [".wenlan/state.json", source_path, quarantine_path] {
        validate_projection_relative_path(relative)?;
    }
    let mut rows = Vec::new();
    let mut read_budget = crate::export::knowledge::RepairReadBudget::new();
    for relative in [".wenlan/state.json", source_path, quarantine_path] {
        let row =
            match projection.read_relative_regular_nofollow_budget(relative, &mut read_budget)? {
                Some(bytes) => vec![
                    relative.to_string(),
                    "file_hex".to_string(),
                    hex::encode(bytes),
                ],
                None => {
                    vec![relative.to_string(), "missing".to_string(), String::new()]
                }
            };
        rows.push(row);
    }
    rows.push(
        match projection.orphaned_baseline_nofollow(&mut read_budget)? {
            Some(baseline) => {
                vec![
                    ".wenlan/orphaned".to_string(),
                    "directory".to_string(),
                    hex::encode(serde_json::to_vec(&baseline)?),
                ]
            }
            None => vec![
                ".wenlan/orphaned".to_string(),
                "missing".to_string(),
                String::new(),
            ],
        },
    );
    Ok(StoredRollbackArtifact {
        format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
        table: STALE_PAGE_PROJECTION_ROLLBACK_TABLE.to_string(),
        source_id: page_id.to_string(),
        columns: vec![
            "path".to_string(),
            "kind".to_string(),
            "content_hex".to_string(),
        ],
        rows,
    })
}

pub(crate) async fn capture_page_projection_on_connection(
    connection: &libsql::Connection,
    page_root: &Path,
    page_id: &str,
    paths: &BTreeSet<String>,
    table: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let page_row = projection_page_row_on_connection(connection, page_id).await?;
    capture_page_projection_from_row(page_root, page_id, page_row, paths, table)
}

pub(crate) async fn projection_page_row_on_connection(
    connection: &libsql::Connection,
    page_id: &str,
) -> Result<Vec<String>, WenlanError> {
    let (page_row, _) = projection_page_row_from_connection(connection, page_id).await?;
    Ok(page_row)
}

pub(crate) fn capture_page_projection_from_row(
    page_root: &Path,
    page_id: &str,
    page_row: Vec<String>,
    paths: &BTreeSet<String>,
    table: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    projection_rollback_from_paths(page_id, page_row, page_root, paths, table)
}

pub(crate) fn projection_rollback_paths(
    rollback: &StoredRollbackArtifact,
) -> Result<BTreeSet<String>, WenlanError> {
    if rollback.format_version != LEGACY_ROLLBACK_FORMAT_VERSION
        || !matches!(
            rollback.table.as_str(),
            PAGE_PROJECTION_ROLLBACK_TABLE | PAGE_PROJECTION_ROLLBACK_TABLE_V2
        )
        || rollback.columns != ["path", "kind", "content"]
    {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_invalid".to_string(),
        ));
    }
    let mut paths = BTreeSet::new();
    for row in &rollback.rows {
        if row.len() != 3 {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        if row[0] == "@page_row" {
            continue;
        }
        validate_projection_relative_path(&row[0])?;
        if !matches!(row[1].as_str(), "file" | "directory" | "missing") {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        paths.insert(row[0].clone());
    }
    Ok(paths)
}

pub(crate) fn page_projection_target_path(
    rollback: &StoredRollbackArtifact,
) -> Result<String, WenlanError> {
    let paths = projection_rollback_paths(rollback)?;
    let state = rollback
        .rows
        .iter()
        .find(|row| row[0] == ".wenlan/state.json" && row[1] == "file")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let root = serde_json::from_str::<serde_json::Value>(&state[2])
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let page = root
        .get("pages")
        .and_then(|pages| pages.get(&rollback.source_id))
        .or_else(|| {
            rollback
                .source_id
                .strip_prefix("page_")
                .and_then(|suffix| root.get("concepts")?.get(format!("concept_{suffix}")))
        })
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let raw = page
        .get("file")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let normalized = crate::lint::pages::fs::normalize_target_path(raw)
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?
        .as_str()
        .to_string();
    if !paths.contains(&normalized) {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_invalid".to_string(),
        ));
    }
    Ok(normalized)
}

pub(crate) fn restore_page_projection_snapshot(
    page_root: &Path,
    rollback: &StoredRollbackArtifact,
) -> Result<(), WenlanError> {
    let _ = projection_rollback_paths(rollback)?;
    for row in rollback.rows.iter().filter(|row| row[0] != "@page_row") {
        ensure_projection_path_no_symlink(page_root, &row[0])?;
        let path = page_root.join(&row[0]);
        match row[1].as_str() {
            "file" => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, row[2].as_bytes())?;
            }
            "missing" => match fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.is_file() => fs::remove_file(path)?,
                Ok(metadata) if metadata.is_dir() => {}
                Ok(_) => {
                    return Err(WenlanError::Validation(
                        "repair_projection_rollback_unsafe_path".to_string(),
                    ))
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(WenlanError::Io(error)),
            },
            "directory" => fs::create_dir_all(path)?,
            _ => unreachable!("validated projection rollback kind"),
        }
    }
    for row in rollback
        .rows
        .iter()
        .rev()
        .filter(|row| row[0] != "@page_row" && row[1] == "missing")
    {
        let path = page_root.join(&row[0]);
        if path.is_dir() {
            match fs::remove_dir(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::DirectoryNotEmpty => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(WenlanError::Io(error)),
            }
        }
    }
    Ok(())
}

async fn projection_page_row_from_snapshot(
    snapshot: &LintReadSnapshot<'_>,
    page_id: &str,
) -> Result<(Vec<String>, Vec<String>), WenlanError> {
    let mut rows = snapshot
        .query(
            projection_page_receipt_sql(),
            libsql::params::Params::Positional(vec![libsql::Value::Text(page_id.to_string())]),
        )
        .await
        .map_err(snapshot_error)?;
    let row = rows
        .next()
        .await
        .map_err(snapshot_error)?
        .ok_or_else(|| WenlanError::NotFound("repair_target_missing".to_string()))?;
    projection_page_row_values(&row)
}

async fn projection_page_row_from_connection(
    connection: &libsql::Connection,
    page_id: &str,
) -> Result<(Vec<String>, Vec<String>), WenlanError> {
    let mut rows = connection
        .query(projection_page_receipt_sql(), libsql::params![page_id])
        .await
        .map_err(database_error)?;
    let row = rows
        .next()
        .await
        .map_err(database_error)?
        .ok_or_else(|| WenlanError::Conflict("repair_target_stale".to_string()))?;
    projection_page_row_values(&row)
}

fn projection_page_receipt_sql() -> &'static str {
    "SELECT id,title,COALESCE(summary,''),content,COALESCE(entity_id,''),
            COALESCE(space,''),COALESCE(source_memory_ids,'[]'),version,status,
            created_at,last_compiled,last_modified,COALESCE(workspace,''),
            COALESCE(citations,'[]')
       FROM pages WHERE id=?1"
}

fn projection_page_row_values(
    row: &libsql::Row,
) -> Result<(Vec<String>, Vec<String>), WenlanError> {
    let mut values = Vec::with_capacity(14);
    for index in 0..14 {
        if index == 7 {
            values.push(row.get::<i64>(index).map_err(database_error)?.to_string());
        } else {
            values.push(row.get::<String>(index).map_err(database_error)?);
        }
    }
    let source_ids = serde_json::from_str::<Vec<String>>(&values[6]).map_err(|error| {
        WenlanError::Validation(format!("repair projection source ids invalid: {error}"))
    })?;
    Ok((values, source_ids))
}

fn projection_closure_paths(target_path: &str) -> Result<BTreeSet<String>, WenlanError> {
    let paths = BTreeSet::from([
        ".wenlan".to_string(),
        ".wenlan/state.json".to_string(),
        target_path.to_string(),
    ]);
    for path in &paths {
        validate_projection_relative_path(path)?;
    }
    Ok(paths)
}

fn projection_rollback_from_paths(
    page_id: &str,
    page_row: Vec<String>,
    page_root: &Path,
    paths: &BTreeSet<String>,
    table: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    if !matches!(
        table,
        PAGE_PROJECTION_ROLLBACK_TABLE | PAGE_PROJECTION_ROLLBACK_TABLE_V2
    ) {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_invalid".to_string(),
        ));
    }
    if !page_root.is_dir() || page_root.symlink_metadata()?.file_type().is_symlink() {
        return Err(WenlanError::Validation(
            "repair_projection_root_invalid".to_string(),
        ));
    }
    let mut rows = vec![vec![
        "@page_row".to_string(),
        "value".to_string(),
        serde_json::to_string(&page_row)?,
    ]];
    let mut captured_bytes = 0_u64;
    for relative in paths {
        validate_projection_relative_path(relative)?;
        ensure_projection_path_no_symlink(page_root, relative)?;
        let path = page_root.join(relative);
        let row = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(WenlanError::Validation(
                    "repair_projection_rollback_unsafe_path".to_string(),
                ))
            }
            Ok(metadata) if metadata.is_dir() => {
                vec![relative.clone(), "directory".to_string(), String::new()]
            }
            Ok(metadata) if metadata.is_file() => {
                captured_bytes = captured_bytes.checked_add(metadata.len()).ok_or_else(|| {
                    WenlanError::Validation("repair_projection_rollback_too_large".to_string())
                })?;
                if captured_bytes > PAGE_PROJECTION_ROLLBACK_MAX_BYTES {
                    return Err(WenlanError::Validation(
                        "repair_projection_rollback_too_large".to_string(),
                    ));
                }
                let content = fs::read_to_string(&path).map_err(|error| {
                    WenlanError::Validation(format!(
                        "repair projection file is not readable UTF-8: {error}"
                    ))
                })?;
                vec![relative.clone(), "file".to_string(), content]
            }
            Ok(_) => {
                return Err(WenlanError::Validation(
                    "repair_projection_rollback_unsafe_path".to_string(),
                ))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                vec![relative.clone(), "missing".to_string(), String::new()]
            }
            Err(error) => return Err(WenlanError::Io(error)),
        };
        rows.push(row);
    }
    Ok(StoredRollbackArtifact {
        format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
        table: table.to_string(),
        source_id: page_id.to_string(),
        columns: vec![
            "path".to_string(),
            "kind".to_string(),
            "content".to_string(),
        ],
        rows,
    })
}

pub(crate) fn validate_projection_relative_path(path: &str) -> Result<(), WenlanError> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_unsafe_path".to_string(),
        ));
    }
    Ok(())
}

fn ensure_projection_path_no_symlink(root: &Path, relative: &str) -> Result<(), WenlanError> {
    validate_projection_relative_path(relative)?;
    let mut current = root.to_path_buf();
    for component in Path::new(relative).components() {
        let Component::Normal(component) = component else {
            unreachable!("projection path was validated")
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(WenlanError::Validation(
                    "repair_projection_rollback_unsafe_path".to_string(),
                ))
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(WenlanError::Io(error)),
        }
    }
    Ok(())
}

async fn capture_rollback_on_connection(
    connection: &libsql::Connection,
    source_id: &str,
) -> Result<StoredRollbackArtifact, WenlanError> {
    let mut column_rows = connection
        .query("PRAGMA table_info(memories)", ())
        .await
        .map_err(database_error)?;
    let mut columns = Vec::new();
    while let Some(row) = column_rows.next().await.map_err(database_error)? {
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
        .map(|column| format!("quote({})", quote_identifier(column)))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT {selected} FROM memories
         WHERE source='memory' AND source_id=?1 ORDER BY chunk_index,id"
    );
    let mut query_rows = connection
        .query(&sql, libsql::params![source_id])
        .await
        .map_err(database_error)?;
    let mut rows = Vec::new();
    while let Some(row) = query_rows.next().await.map_err(database_error)? {
        let mut values = Vec::with_capacity(columns.len());
        for index in 0..columns.len() {
            let index = i32::try_from(index).map_err(|_| {
                WenlanError::Validation("repair_target_schema_too_wide".to_string())
            })?;
            values.push(row.get::<String>(index).map_err(database_error)?);
        }
        rows.push(values);
    }
    if rows.is_empty() {
        return Err(WenlanError::Conflict("repair_target_stale".to_string()));
    }
    Ok(StoredRollbackArtifact {
        format_version: LEGACY_ROLLBACK_FORMAT_VERSION,
        table: "memories".to_string(),
        source_id: source_id.to_string(),
        columns,
        rows,
    })
}

pub(crate) fn target_receipt(
    rollback: &StoredRollbackArtifact,
) -> Result<RepairDigest, WenlanError> {
    if rollback.table == STALE_PAGE_PROJECTION_ROLLBACK_TABLE {
        return stale_page_projection_target_receipt(rollback);
    }
    if rollback.table == PAGE_PROJECTION_ROLLBACK_TABLE_V2 {
        return page_projection_target_receipt(rollback);
    }
    let mut bytes = b"wenlan-repair-target-v1".to_vec();
    bytes.extend(serde_json::to_vec(rollback)?);
    Ok(repair_digest(&bytes))
}

pub(crate) fn stale_page_projection_paths(
    rollback: &StoredRollbackArtifact,
) -> Result<(String, String), WenlanError> {
    if rollback.format_version != LEGACY_ROLLBACK_FORMAT_VERSION
        || rollback.table != STALE_PAGE_PROJECTION_ROLLBACK_TABLE
        || rollback.columns != ["path", "kind", "content_hex"]
        || rollback.rows.len() != 4
    {
        return Err(WenlanError::Validation(
            "repair_projection_rollback_invalid".to_string(),
        ));
    }
    for row in &rollback.rows {
        if row.len() != 3 {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
        validate_projection_relative_path(&row[0])?;
        if !matches!(row[1].as_str(), "file_hex" | "directory" | "missing")
            || (row[1] == "file_hex" && hex::decode(&row[2]).is_err())
        {
            return Err(WenlanError::Validation(
                "repair_projection_rollback_invalid".to_string(),
            ));
        }
    }
    let source = rollback
        .rows
        .iter()
        .find(|row| {
            row[0] != ".wenlan/state.json"
                && row[0] != ".wenlan/orphaned"
                && !row[0].starts_with(".wenlan/orphaned/")
        })
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let quarantine = rollback
        .rows
        .iter()
        .find(|row| row[0].starts_with(".wenlan/orphaned/"))
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    Ok((source[0].clone(), quarantine[0].clone()))
}

pub(crate) fn stale_page_projection_orphaned_baseline(
    rollback: &StoredRollbackArtifact,
) -> Result<Option<Vec<(String, String)>>, WenlanError> {
    let row = rollback
        .rows
        .iter()
        .find(|row| row[0] == ".wenlan/orphaned")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    match row[1].as_str() {
        "missing" if row[2].is_empty() => Ok(None),
        "directory" if !row[2].is_empty() => {
            let bytes = hex::decode(&row[2]).map_err(|_| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
            let mut baseline =
                serde_json::from_slice::<Vec<(String, String)>>(&bytes).map_err(|_| {
                    WenlanError::Validation("repair_projection_rollback_invalid".to_string())
                })?;
            let mut sorted = baseline.clone();
            sorted.sort();
            sorted.dedup();
            if baseline != sorted
                || baseline.iter().any(|(name, digest)| {
                    name.is_empty()
                        || name == "."
                        || name == ".."
                        || name.contains('/')
                        || name.contains('\\')
                        || RepairDigest::parse(digest).is_err()
                })
            {
                return Err(WenlanError::Validation(
                    "repair_projection_rollback_invalid".to_string(),
                ));
            }
            Ok(Some(std::mem::take(&mut baseline)))
        }
        _ => Err(WenlanError::Validation(
            "repair_projection_rollback_invalid".to_string(),
        )),
    }
}

fn stale_page_projection_target_receipt(
    rollback: &StoredRollbackArtifact,
) -> Result<RepairDigest, WenlanError> {
    let _ = stale_page_projection_paths(rollback)?;
    let state = rollback
        .rows
        .iter()
        .find(|row| row[0] == ".wenlan/state.json" && row[1] == "file_hex")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let state_bytes = hex::decode(&state[2])
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let state_target = projection_shared_json_slice_bytes(&state_bytes, &rollback.source_id)?;
    let mut bytes = b"wenlan-repair-stale-page-target-v1".to_vec();
    bytes.extend(state_target);
    for row in &rollback.rows {
        if row[0] != ".wenlan/state.json" && row[0] != ".wenlan/orphaned" {
            bytes.extend(serde_json::to_vec(row)?);
        }
    }
    Ok(repair_digest(&bytes))
}

fn stale_page_projection_post_target_receipt(
    rollback: &StoredRollbackArtifact,
) -> Result<RepairDigest, WenlanError> {
    let (source_path, quarantine_path) = stale_page_projection_paths(rollback)?;
    let state = rollback
        .rows
        .iter()
        .find(|row| row[0] == ".wenlan/state.json" && row[1] == "file_hex")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let source = rollback
        .rows
        .iter()
        .find(|row| row[0] == source_path && row[1] == "file_hex")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let original_state = hex::decode(&state[2])
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let post_state =
        crate::lint::pages::state::remove_unique_page_member(&original_state, &rollback.source_id)
            .map_err(|()| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
    let mut post = rollback.clone();
    for row in &mut post.rows {
        if row[0] == ".wenlan/state.json" {
            row[1] = "file_hex".to_string();
            row[2] = hex::encode(&post_state);
        } else if row[0] == source_path {
            row[1] = "missing".to_string();
            row[2].clear();
        } else if row[0] == quarantine_path {
            row[1] = "file_hex".to_string();
            row[2] = source[2].clone();
        }
    }
    stale_page_projection_target_receipt(&post)
}

fn projection_shared_json_slice_bytes(
    content: &[u8],
    page_id: &str,
) -> Result<Vec<u8>, WenlanError> {
    let mut root = serde_json::from_slice::<BTreeMap<String, serde_json::Value>>(content)
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let pages = root
        .remove("pages")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))
        .and_then(|pages| {
            serde_json::from_value::<BTreeMap<String, serde_json::Value>>(pages).map_err(|_| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })
        })?;
    let selected = pages
        .get(page_id)
        .cloned()
        .map(|page| BTreeMap::from([(page_id.to_string(), page)]))
        .unwrap_or_default();
    root.insert("pages".to_string(), serde_json::to_value(selected)?);
    Ok(serde_json::to_vec(&root)?)
}

fn page_projection_target_receipt(
    rollback: &StoredRollbackArtifact,
) -> Result<RepairDigest, WenlanError> {
    let _ = projection_rollback_paths(rollback)?;
    let mut target = rollback.clone();
    for row in &mut target.rows {
        if row[1] == "file" && row[0] == ".wenlan/state.json" {
            row[2] = projection_shared_json_slice(&row[2], &target.source_id)?;
        }
    }
    let mut bytes = b"wenlan-repair-page-target-v2".to_vec();
    bytes.extend(serde_json::to_vec(&target)?);
    Ok(repair_digest(&bytes))
}

fn projection_shared_json_slice(content: &str, page_id: &str) -> Result<String, WenlanError> {
    projection_shared_json_material(content, page_id, true)
}

fn projection_shared_json_without_page(
    content: &str,
    page_id: &str,
) -> Result<String, WenlanError> {
    projection_shared_json_material(content, page_id, false)
}

fn projection_shared_json_material(
    content: &str,
    page_id: &str,
    retain_target: bool,
) -> Result<String, WenlanError> {
    let mut root = serde_json::from_str::<BTreeMap<String, serde_json::Value>>(content)
        .map_err(|_| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))?;
    let pages = root
        .remove("pages")
        .ok_or_else(|| WenlanError::Validation("repair_projection_rollback_invalid".to_string()))
        .and_then(|pages| {
            serde_json::from_value::<BTreeMap<String, serde_json::Value>>(pages).map_err(|_| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })
        })?;
    let selected = if retain_target {
        pages
            .get(page_id)
            .cloned()
            .map(|page| BTreeMap::from([(page_id.to_string(), page)]))
            .unwrap_or_default()
    } else {
        pages
            .into_iter()
            .filter(|(candidate, _)| candidate != page_id)
            .collect()
    };
    root.insert("pages".to_string(), serde_json::to_value(selected)?);
    Ok(serde_json::to_string(&root)?)
}

pub(crate) fn page_projection_non_target_receipt(
    filesystem_digest: [u8; 32],
    captured: &StoredRollbackArtifact,
) -> Result<RepairDigest, WenlanError> {
    if captured.table == STALE_PAGE_PROJECTION_ROLLBACK_TABLE {
        let _ = stale_page_projection_paths(captured)?;
        let state = captured
            .rows
            .iter()
            .find(|row| row[0] == ".wenlan/state.json" && row[1] == "file_hex")
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
        let state_bytes = hex::decode(&state[2]).map_err(|_| {
            WenlanError::Validation("repair_projection_rollback_invalid".to_string())
        })?;
        let mut root = serde_json::from_slice::<BTreeMap<String, serde_json::Value>>(&state_bytes)
            .map_err(|_| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })?;
        let pages = root
            .remove("pages")
            .ok_or_else(|| {
                WenlanError::Validation("repair_projection_rollback_invalid".to_string())
            })
            .and_then(|pages| {
                serde_json::from_value::<BTreeMap<String, serde_json::Value>>(pages).map_err(|_| {
                    WenlanError::Validation("repair_projection_rollback_invalid".to_string())
                })
            })?;
        root.insert(
            "pages".to_string(),
            serde_json::to_value(
                pages
                    .into_iter()
                    .filter(|(candidate, _)| candidate != &captured.source_id)
                    .collect::<BTreeMap<_, _>>(),
            )?,
        );
        let mut bytes = b"wenlan-repair-stale-page-non-target-v1".to_vec();
        bytes.extend(filesystem_digest);
        bytes.extend(serde_json::to_vec(&root)?);
        return Ok(repair_digest(&bytes));
    }
    let _ = projection_rollback_paths(captured)?;
    if captured.table == PAGE_PROJECTION_ROLLBACK_TABLE {
        // Compatibility: durable v1 apply receipts shipped with this exact,
        // unprefixed digest. Changing it would make existing artifacts unverifiable.
        return Ok(repair_digest(&filesystem_digest));
    }
    let mut shared_rows = Vec::new();
    for row in &captured.rows {
        if row[0] == ".wenlan/state.json" {
            let mut shared = row.clone();
            if shared[1] == "file" {
                shared[2] = projection_shared_json_without_page(&shared[2], &captured.source_id)?;
            }
            shared_rows.push(shared);
        }
    }
    let mut bytes = b"wenlan-repair-page-non-target-v2".to_vec();
    bytes.extend(filesystem_digest);
    bytes.extend(captured.source_id.as_bytes());
    bytes.extend(serde_json::to_vec(&shared_rows)?);
    Ok(repair_digest(&bytes))
}

fn validate_source_receipts(
    request: &PrepareRepairRequest,
    current: SnapshotReceipt,
) -> Result<(), WenlanError> {
    let deep = request
        .deep_report()
        .ok_or_else(|| WenlanError::Validation("repair_deep_report_missing".to_string()))?;
    validate_report_source_receipts(&[request.general_report(), deep], current)
}

pub(crate) fn validate_report_source_receipts(
    reports: &[&wenlan_types::lint::LintReport],
    current: SnapshotReceipt,
) -> Result<(), WenlanError> {
    if !current.is_consistent() {
        return Err(WenlanError::Conflict(
            "repair_snapshot_inconsistent".to_string(),
        ));
    }
    let current = lint_digest(current.analysis_receipt_digest().as_bytes());
    for report in reports {
        let db = report.snapshots().db();
        if db.analysis_digest() != &current || db.post_run_digest() != Some(&current) {
            return Err(WenlanError::Conflict(
                "repair_source_reports_stale".to_string(),
            ));
        }
    }
    Ok(())
}

fn lint_digest(bytes: [u8; 32]) -> LintDigest {
    LintDigest::from_u64(u64::from_le_bytes(
        bytes[..8].try_into().expect("eight bytes"),
    ))
}

pub(crate) fn repair_digest(bytes: &[u8]) -> RepairDigest {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("write to string");
    }
    RepairDigest::parse(&encoded).expect("sha256 is lowercase hex")
}

#[derive(Serialize)]
struct LintReviewOwnerBinding<'a> {
    occurrence_digest: &'a RepairDigest,
    source_ids: &'a [String],
}

pub(crate) fn canonical_lint_review_source_ids(
    source_ids: &[String],
) -> Result<Vec<String>, WenlanError> {
    if source_ids.is_empty()
        || source_ids
            .iter()
            .any(|source_id| source_id.is_empty() || source_id.trim() != source_id)
    {
        return Err(WenlanError::Validation(
            "lint repair review source_ids must be nonempty and trim-stable".to_string(),
        ));
    }
    let mut canonical = source_ids.to_vec();
    canonical.sort();
    if canonical.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(WenlanError::Validation(
            "lint repair review source_ids must be unique".to_string(),
        ));
    }
    Ok(canonical)
}

pub(crate) fn lint_review_owner_binding_digest(
    occurrence_digest: &RepairDigest,
    source_ids: &[String],
) -> Result<RepairDigest, WenlanError> {
    let source_ids = canonical_lint_review_source_ids(source_ids)?;
    let canonical = serde_json::to_vec(&LintReviewOwnerBinding {
        occurrence_digest,
        source_ids: &source_ids,
    })?;
    Ok(repair_digest(&canonical))
}

/// Logical digest of every ordinary SQLite table, streamed one row at a time.
/// This is intentionally separate from lint's cheap structural snapshot: a
/// repair apply receipt keeps this apply-time forensic/content anchor so
/// persisted receipt versions remain auditable without loading the whole
/// database into memory.
pub(crate) async fn database_content_digest(
    connection: &libsql::Connection,
) -> Result<RepairDigest, WenlanError> {
    let mut digest = Sha256::new();
    digest_len_bytes(&mut digest, b"wenlan-repair-db-content-v1")?;

    let mut schema_rows = connection
        .query(
            "SELECT type, name, COALESCE(sql, '')
             FROM sqlite_schema
             WHERE type IN ('index', 'table', 'trigger', 'view')
               AND name NOT LIKE 'sqlite_%'
             ORDER BY type, name",
            (),
        )
        .await
        .map_err(database_error)?;
    let mut ordinary_tables = Vec::new();
    while let Some(row) = schema_rows.next().await.map_err(database_error)? {
        let object_type = row.get::<String>(0).map_err(database_error)?;
        let name = row.get::<String>(1).map_err(database_error)?;
        let sql = row.get::<String>(2).map_err(database_error)?;
        digest_len_bytes(&mut digest, object_type.as_bytes())?;
        digest_len_bytes(&mut digest, name.as_bytes())?;
        digest_len_bytes(&mut digest, sql.as_bytes())?;
        if object_type == "table"
            && !sql
                .trim_start()
                .to_ascii_lowercase()
                .starts_with("create virtual table")
        {
            ordinary_tables.push((name, sql));
        }
    }
    drop(schema_rows);

    for (table, create_sql) in ordinary_tables {
        digest_len_bytes(&mut digest, table.as_bytes())?;
        let order_by = if create_sql.to_ascii_lowercase().contains("without rowid") {
            without_rowid_order(connection, &table).await?
        } else {
            "_rowid_".to_string()
        };
        let query = format!(
            "SELECT * FROM {} ORDER BY {order_by}",
            quote_identifier(&table)
        );
        let mut rows = connection.query(&query, ()).await.map_err(database_error)?;
        let column_count = rows.column_count();
        digest.update(i64::from(column_count).to_le_bytes());
        while let Some(row) = rows.next().await.map_err(database_error)? {
            digest.update(b"row");
            for column in 0..column_count {
                digest_sql_value(&mut digest, row.get_value(column).map_err(database_error)?)?;
            }
        }
    }

    Ok(repair_digest(&digest.finalize()))
}

async fn without_rowid_order(
    connection: &libsql::Connection,
    table: &str,
) -> Result<String, WenlanError> {
    let query = format!("PRAGMA table_info({})", quote_identifier(table));
    let mut rows = connection.query(&query, ()).await.map_err(database_error)?;
    let mut primary_key = Vec::new();
    while let Some(row) = rows.next().await.map_err(database_error)? {
        let name = row.get::<String>(1).map_err(database_error)?;
        let position = row.get::<i64>(5).map_err(database_error)?;
        if position > 0 {
            primary_key.push((position, name));
        }
    }
    primary_key.sort_by_key(|(position, _)| *position);
    if primary_key.is_empty() {
        return Err(WenlanError::VectorDb(format!(
            "repair db digest: WITHOUT ROWID table {table} has no primary key"
        )));
    }
    Ok(primary_key
        .into_iter()
        .map(|(_, name)| quote_identifier(&name))
        .collect::<Vec<_>>()
        .join(","))
}

fn digest_sql_value(digest: &mut Sha256, value: libsql::Value) -> Result<(), WenlanError> {
    match value {
        libsql::Value::Null => digest.update(b"n"),
        libsql::Value::Integer(value) => {
            digest.update(b"i");
            digest.update(value.to_le_bytes());
        }
        libsql::Value::Real(value) => {
            digest.update(b"r");
            digest.update(value.to_bits().to_le_bytes());
        }
        libsql::Value::Text(value) => {
            digest.update(b"t");
            digest_len_bytes(digest, value.as_bytes())?;
        }
        libsql::Value::Blob(value) => {
            digest.update(b"b");
            digest_len_bytes(digest, &value)?;
        }
    }
    Ok(())
}

fn digest_len_bytes(digest: &mut Sha256, value: &[u8]) -> Result<(), WenlanError> {
    let length = u64::try_from(value.len())
        .map_err(|_| WenlanError::Validation("repair_db_digest_value_too_large".to_string()))?;
    digest.update(length.to_le_bytes());
    digest.update(value);
    Ok(())
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn safe_manifest_id(value: &str) -> bool {
    value.starts_with("repair_")
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && Path::new(value).components().count() == 1
}

fn safe_plan_id(value: &str) -> bool {
    safe_manifest_id(value) && value.starts_with("repair_plan_")
}

fn write_plan_jsonl_line(
    file: &mut File,
    written: &mut u64,
    value: &impl Serialize,
) -> Result<(), WenlanError> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    let line_len = u64::try_from(bytes.len())
        .map_err(|_| WenlanError::Validation("repair_plan_artifact_too_large".to_string()))?;
    let next_len = written
        .checked_add(line_len)
        .ok_or_else(|| WenlanError::Validation("repair_plan_artifact_too_large".to_string()))?;
    if next_len > REPAIR_PLAN_ARTIFACT_MAX_BYTES {
        return Err(WenlanError::Validation(
            "repair_plan_artifact_too_large".to_string(),
        ));
    }
    file.write_all(&bytes)?;
    *written = next_len;
    Ok(())
}

fn write_private_file(path: &Path, bytes: &[u8]) -> Result<(), WenlanError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    set_private_file_permissions(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn publish_no_replace(
    pending_path: &Path,
    final_path: &Path,
    exists_error: &str,
) -> Result<(), WenlanError> {
    publish_no_replace_with_hook(pending_path, final_path, exists_error, || {})
}

fn publish_no_replace_with_hook<F>(
    pending_path: &Path,
    final_path: &Path,
    exists_error: &str,
    after_final_sync: F,
) -> Result<(), WenlanError>
where
    F: FnOnce(),
{
    fs::hard_link(pending_path, final_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            WenlanError::Conflict(exists_error.to_string())
        } else {
            WenlanError::Io(error)
        }
    })?;
    if let Some(parent) = final_path.parent() {
        sync_dir(parent)?;
    }
    after_final_sync();
    fs::remove_file(pending_path)?;
    if let Some(parent) = final_path.parent() {
        sync_dir(parent)?;
    }
    Ok(())
}

#[cfg(unix)]
fn ensure_repair_artifacts_supported() -> Result<(), WenlanError> {
    Ok(())
}

#[cfg(not(unix))]
fn ensure_repair_artifacts_supported() -> Result<(), WenlanError> {
    Err(WenlanError::Validation(
        "lint_repair_unsupported_platform".to_string(),
    ))
}

fn ensure_private_dir(path: &Path) -> Result<(), WenlanError> {
    fs::create_dir_all(path)?;
    set_private_dir_permissions(path)
}

#[cfg(unix)]
fn set_private_dir_permissions(path: &Path) -> Result<(), WenlanError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_dir_permissions(_path: &Path) -> Result<(), WenlanError> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file_permissions(path: &Path) -> Result<(), WenlanError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_private_file_permissions(_path: &Path) -> Result<(), WenlanError> {
    Ok(())
}

#[cfg(unix)]
fn sync_dir(path: &Path) -> Result<(), WenlanError> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_dir(_path: &Path) -> Result<(), WenlanError> {
    Ok(())
}

fn snapshot_error(error: SnapshotError) -> WenlanError {
    WenlanError::VectorDb(format!("repair snapshot: {error}"))
}

fn page_snapshot_error(error: crate::lint::pages::fs::PageFsError) -> WenlanError {
    WenlanError::Conflict(format!("repair_verification_reports_stale: {error}"))
}

fn database_error(error: libsql::Error) -> WenlanError {
    WenlanError::VectorDb(format!("repair database: {error}"))
}

#[cfg(test)]
#[path = "repair/entity_extraction_tests.rs"]
mod entity_extraction_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        db::{tests::test_db, MemoryDB},
        lint::{
            context::{CancellationToken, LintClock},
            runner::LintRunner,
            snapshot::LintReadSnapshot,
        },
    };
    use wenlan_types::{
        lint::{
            LintAgentSubmission, LintAgentVerdict, LintEvidenceRef, LintProfile, LintQuery,
            LintSemanticCheckId, LintSemanticDecision,
        },
        repair::{
            ApplyRepairRequest, PrepareRepairRequest, RepairDigest, RepairLintScope,
            VerifyRepairRequest,
        },
        repair_plan::RepairPlanRequest,
        MemoryType,
    };

    #[test]
    fn recovery_required_keeps_reclassification_and_entity_pending_receipts() {
        let recovery_required = WenlanError::Conflict("repair_apply_recovery_required".to_string());

        assert!(should_retain_pending_apply_receipt(&recovery_required));
        assert!(should_retain_stale_page_projection_pending_receipt(
            &recovery_required,
            false,
        ));
        assert!(should_retain_stale_page_projection_pending_receipt(
            &WenlanError::Conflict("repair_target_stale".to_string()),
            true,
        ));
        assert!(!should_retain_stale_page_projection_pending_receipt(
            &WenlanError::Conflict("repair_target_stale".to_string()),
            false,
        ));
        assert!(!should_retain_pending_apply_receipt(
            &WenlanError::Conflict("repair_target_stale".to_string()),
        ));
    }

    #[test]
    fn receipt_publication_keeps_pending_link_through_final_directory_sync() {
        let root = tempfile::tempdir().unwrap();
        let pending = root.path().join("pending.json");
        let final_path = root.path().join("final.json");
        std::fs::write(&pending, b"receipt").unwrap();
        let mut observed = false;

        publish_no_replace_with_hook(&pending, &final_path, "already_exists", || {
            assert!(pending.is_file());
            assert!(final_path.is_file());
            observed = true;
        })
        .unwrap();

        assert!(observed);
        assert!(!pending.exists());
        assert_eq!(std::fs::read(final_path).unwrap(), b"receipt");
    }

    #[test]
    fn page_projection_v1_rollback_keeps_legacy_target_receipt() {
        let rollback = StoredRollbackArtifact {
            format_version: 1,
            table: PAGE_PROJECTION_ROLLBACK_TABLE.to_string(),
            source_id: "page_legacy".to_string(),
            columns: vec![
                "path".to_string(),
                "kind".to_string(),
                "content".to_string(),
            ],
            rows: vec![
                vec![
                    "@page_row".to_string(),
                    "value".to_string(),
                    "[]".to_string(),
                ],
                vec![
                    ".wenlan/state.json".to_string(),
                    "file".to_string(),
                    r#"{"schema_version":2,"pages":{"page_legacy":{"file":"legacy.md","version":1}}}"#
                        .to_string(),
                ],
            ],
        };
        let mut legacy_bytes = b"wenlan-repair-target-v1".to_vec();
        legacy_bytes.extend(serde_json::to_vec(&rollback).unwrap());

        assert_eq!(
            target_receipt(&rollback).unwrap(),
            repair_digest(&legacy_bytes)
        );
    }

    async fn fixture() -> (MemoryDB, tempfile::TempDir) {
        let (db, dir) = test_db().await;
        db.conn
            .lock()
            .await
            .execute_batch(
                "INSERT INTO spaces (id,name,created_at,updated_at)
                 VALUES ('space-personal','personal',1,1),('space-work','work',1,1);
                 INSERT INTO memories
                     (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                      pending_revision,is_recap,supersede_mode,memory_type,space)
                 VALUES ('row-target','Target decision','memory','mem_target','target',0,10,
                         'text',0,0,'hide',NULL,'work'),
                        ('row-target-2','Target detail','memory','mem_target','target',1,10,
                         'text',0,0,'hide',NULL,'work'),
                        ('row-other','Other fact','memory','mem_other','other',0,11,
                         'text',0,0,'hide','fact','personal');",
            )
            .await
            .unwrap();
        (db, dir)
    }

    async fn request(db: &MemoryDB) -> PrepareRepairRequest {
        request_for_scope(db, RepairLintScope::global(), None).await
    }

    async fn request_for_scope(
        db: &MemoryDB,
        lint_scope: RepairLintScope,
        space: Option<&str>,
    ) -> PrepareRepairRequest {
        let (general, deep, _) = semantic_reports_for_scope(db, space).await;
        let plan_root = tempfile::tempdir().unwrap();
        crate::repair_plan::prepare_repair_plan(
            db,
            &RepairArtifactStore::new(plan_root.path().to_path_buf()),
            RepairPlanRequest::try_new(lint_scope.clone(), general, Some(deep)).unwrap(),
            None,
            1_720_999_999,
        )
        .await
        .unwrap();
        let (general, deep, finding) = semantic_reports_for_scope(db, space).await;
        PrepareRepairRequest::try_new(lint_scope, general, deep, finding, MemoryType::Decision)
            .unwrap()
    }

    async fn semantic_reports_for_scope(
        db: &MemoryDB,
        space: Option<&str>,
    ) -> (
        wenlan_types::lint::LintReport,
        wenlan_types::lint::LintReport,
        wenlan_types::lint::LintSemanticFinding,
    ) {
        let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .run(
                db,
                &LintQuery::new(Some(LintProfile::General), space.map(str::to_string)),
                None,
                false,
            )
            .await
            .unwrap();
        let prepared = LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_semantic_agent_assist()
            .run(
                db,
                &LintQuery::new(Some(LintProfile::Deep), space.map(str::to_string)),
                None,
                false,
            )
            .await
            .unwrap();
        let work = prepared.agent_work().expect("agent work");
        let verdicts = work
            .candidates()
            .iter()
            .map(|candidate| {
                let finding = candidate.check_id() == LintSemanticCheckId::MemoryClassification;
                LintAgentVerdict::try_new(
                    candidate.reference(),
                    if finding {
                        LintSemanticDecision::Finding
                    } else {
                        LintSemanticDecision::Pass
                    },
                    None,
                    candidate.reason_code(),
                    9_000,
                    vec![],
                )
                .unwrap()
            })
            .collect();
        let submission =
            LintAgentSubmission::try_new(work.work_digest().clone(), verdicts).unwrap();
        let deep = LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_semantic_agent_submission(submission)
            .run(
                db,
                &LintQuery::new(Some(LintProfile::Deep), space.map(str::to_string)),
                None,
                false,
            )
            .await
            .unwrap();
        let deep = complete_deep_fixture(deep);
        let finding = deep
            .checks()
            .iter()
            .find(|check| check.check_id() == LintSemanticCheckId::MemoryClassification.as_str())
            .and_then(|check| {
                check.evidence().iter().find_map(|evidence| match evidence {
                    LintEvidenceRef::SemanticFinding { finding } => Some(finding.clone()),
                    _ => None,
                })
            })
            .expect("classification finding");
        assert!(general.complete(), "general totals: {:?}", general.totals());
        assert!(
            deep.complete(),
            "deep totals: {:?}; incomplete: {:?}",
            deep.totals(),
            deep.checks()
                .iter()
                .filter(|check| !matches!(
                    check.outcome(),
                    wenlan_types::lint::LintOutcome::Pass
                        | wenlan_types::lint::LintOutcome::Finding
                ))
                .map(|check| (check.check_id(), check.outcome()))
                .collect::<Vec<_>>()
        );
        assert!(deep.agent_work().is_some(), "final Deep lost agent work");
        (general, deep, finding)
    }

    fn complete_deep_fixture(
        report: wenlan_types::lint::LintReport,
    ) -> wenlan_types::lint::LintReport {
        let mut value = serde_json::to_value(report).unwrap();
        let checks = value["checks"].as_array_mut().unwrap();
        for check in checks.iter_mut() {
            let complete = matches!(check["outcome"].as_str(), Some("pass" | "finding"));
            if !complete {
                check["outcome"] = serde_json::json!("pass");
                check["severity"] = serde_json::json!("info");
                check["applicability"] = serde_json::json!("expected_empty");
                check["precondition"] = serde_json::json!("expected_empty");
            }
        }
        let passed = checks
            .iter()
            .filter(|check| check["outcome"] == "pass")
            .count();
        let findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding")
            .count();
        let actionable_findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding" && check["gate_effect"] == "actionable")
            .count();
        value["totals"] = serde_json::json!({
            "checks": checks.len(),
            "passed": passed,
            "findings": findings,
            "actionable_findings": actionable_findings,
            "advisory_findings": findings - actionable_findings,
            "incomplete": 0
        });
        value["complete"] = serde_json::json!(true);
        serde_json::from_value(value).unwrap()
    }

    fn fail_deep_check(
        report: wenlan_types::lint::LintReport,
        classification: bool,
    ) -> wenlan_types::lint::LintReport {
        let mut value = serde_json::to_value(report).unwrap();
        let checks = value["checks"].as_array_mut().unwrap();
        let check = checks
            .iter_mut()
            .find(|check| {
                (check["check_id"] == REPAIR_CLASSIFICATION_CHECK_ID) == classification
                    && matches!(check["outcome"].as_str(), Some("pass" | "finding"))
            })
            .expect("eligible Deep check");
        check["outcome"] = serde_json::json!("failed_to_run");
        check["severity"] = serde_json::json!("error");
        check["applicability"] = serde_json::json!("applicable");
        check["precondition"] = serde_json::json!("ready");
        let passed = checks
            .iter()
            .filter(|check| check["outcome"] == "pass")
            .count();
        let findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding")
            .count();
        let actionable_findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding" && check["gate_effect"] == "actionable")
            .count();
        let incomplete = checks.len() - passed - findings;
        value["totals"] = serde_json::json!({
            "checks": checks.len(),
            "passed": passed,
            "findings": findings,
            "actionable_findings": actionable_findings,
            "advisory_findings": findings - actionable_findings,
            "incomplete": incomplete
        });
        value["complete"] = serde_json::json!(false);
        serde_json::from_value(value).unwrap()
    }

    fn with_incomplete_deep(
        request: PrepareRepairRequest,
        classification: bool,
    ) -> Result<PrepareRepairRequest, serde_json::Error> {
        let mut value = serde_json::to_value(request).unwrap();
        let deep: wenlan_types::lint::LintReport =
            serde_json::from_value(value["deep_report"].take()).unwrap();
        value["deep_report"] = serde_json::to_value(fail_deep_check(deep, classification)).unwrap();
        serde_json::from_value(value)
    }

    async fn fingerprint(db: &MemoryDB) -> [u8; 32] {
        let snapshot = LintReadSnapshot::open(&db._db).await.unwrap();
        let fingerprint = snapshot.analysis_digest().unwrap().as_bytes();
        snapshot.finish().await.unwrap();
        fingerprint
    }

    #[tokio::test]
    async fn prepare_resolves_one_hashed_memory_without_mutating_store() {
        let (db, _db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let request = request(&db).await;
        let before = fingerprint(&db).await;

        let manifest = prepare_memory_reclassification(
            &db,
            &RepairArtifactStore::new(repair_root.path().to_path_buf()),
            request,
            1_721_000_000,
        )
        .await
        .unwrap();

        assert_eq!(manifest.target().memory_source_id(), "mem_target");
        assert_eq!(before, fingerprint(&db).await);
        let manifest_dir = repair_root.path().join(manifest.manifest_id());
        assert!(manifest_dir.join("manifest.json").is_file());
        assert!(manifest_dir
            .join(manifest.rollback().relative_path())
            .is_file());
        assert!(matches!(
            RepairArtifactStore::new(repair_root.path().to_path_buf())
                .read_stored_manifest(manifest.manifest_id())
                .unwrap(),
            wenlan_types::repair::StoredRepairManifest::V6(_)
        ));
        assert_eq!(
            RepairArtifactStore::new(repair_root.path().to_path_buf())
                .load_manifest(manifest.manifest_id())
                .unwrap(),
            manifest
        );
    }

    #[tokio::test]
    async fn prepare_accepts_complete_classification_with_unrelated_incomplete_deep_check() {
        let (db, _db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let request = with_incomplete_deep(request(&db).await, false)
            .expect("unrelated Deep incompleteness remains a valid repair request");

        let manifest = prepare_memory_reclassification(
            &db,
            &RepairArtifactStore::new(repair_root.path().to_path_buf()),
            request,
            1_721_000_000,
        )
        .await
        .expect("classification repair remains applicable");

        assert_eq!(manifest.target().memory_source_id(), "mem_target");
    }

    #[tokio::test]
    async fn prepare_rejects_incomplete_classification_check() {
        let (db, _db_dir) = fixture().await;

        assert!(with_incomplete_deep(request(&db).await, true).is_err());
    }

    #[test]
    fn artifact_store_loads_the_first_durable_v1_manifest_shape() {
        let repair_root = tempfile::tempdir().unwrap();
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let manifest_dir = repair_root.path().join(manifest_id);
        std::fs::create_dir(&manifest_dir).unwrap();
        std::fs::write(
            manifest_dir.join(MANIFEST_FILE),
            include_bytes!("../../wenlan-types/testdata/repair/v1/manifest-pre-baseline.json"),
        )
        .unwrap();

        let manifest = RepairArtifactStore::new(repair_root.path().to_path_buf())
            .load_manifest(manifest_id)
            .expect("the first daemon-persisted v1 shape remains loadable");

        assert!(manifest.post_assertions().general_baseline().is_empty());
        assert!(manifest.post_assertions().deep_baseline().is_empty());
    }

    #[tokio::test]
    async fn pre_baseline_v1_verification_requires_conservatively_clean_reports() {
        let repair_root = tempfile::tempdir().unwrap();
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let manifest_dir = repair_root.path().join(manifest_id);
        std::fs::create_dir(&manifest_dir).unwrap();
        std::fs::write(
            manifest_dir.join(MANIFEST_FILE),
            include_bytes!("../../wenlan-types/testdata/repair/v1/manifest-pre-baseline.json"),
        )
        .unwrap();
        let manifest = RepairArtifactStore::new(repair_root.path().to_path_buf())
            .load_manifest(manifest_id)
            .unwrap();
        let (db, _db_dir) = fixture().await;
        let before = request(&db).await;

        assert!(matches!(
            validate_verification_reports(
                &manifest,
                before.general_report(),
                before.deep_report()
            ),
            Err(WenlanError::Validation(code)) if code == "repair_legacy_verification_not_clean"
        ));
    }

    #[tokio::test]
    async fn unverified_v1_apply_receipt_cannot_claim_v2_non_target_proof() {
        let repair_root = tempfile::tempdir().unwrap();
        let manifest_id = "repair_550e8400-e29b-41d4-a716-446655440000";
        let manifest_dir = repair_root.path().join(manifest_id);
        std::fs::create_dir(&manifest_dir).unwrap();
        std::fs::write(
            manifest_dir.join(MANIFEST_FILE),
            include_bytes!("../../wenlan-types/testdata/repair/v1/manifest.json"),
        )
        .unwrap();
        std::fs::write(
            manifest_dir.join(APPLY_RECEIPT_FILE),
            include_bytes!("../../wenlan-types/testdata/repair/v1/apply-receipt.json"),
        )
        .unwrap();
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let manifest = store.load_manifest(manifest_id).unwrap();
        let apply_receipt = store.load_apply_receipt(&manifest).unwrap();
        assert!(apply_receipt.post_apply_db_digest().is_none());
        let (db, _db_dir) = fixture().await;
        let (general, deep) = verification_reports(&db).await;

        let result = record_repair_verification(
            &db,
            &store,
            VerifyRepairRequest::try_new(
                manifest_id.to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                general,
                deep,
            )
            .unwrap(),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Validation(message))
                if message == "repair_legacy_apply_receipt_unverifiable"
        ));
    }

    #[tokio::test]
    async fn prepare_rejects_stale_lint_receipt_without_writing_artifacts() {
        let (db, _db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let request = request(&db).await;
        db.conn
            .lock()
            .await
            .execute(
                "UPDATE memories SET title='changed' WHERE source_id='mem_other'",
                (),
            )
            .await
            .unwrap();

        let result = prepare_memory_reclassification(
            &db,
            &RepairArtifactStore::new(repair_root.path().to_path_buf()),
            request,
            1_721_000_000,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(std::fs::read_dir(repair_root.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn prepare_rejects_reports_from_different_runtime_producers() {
        let (db, _db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let mut value = serde_json::to_value(request(&db).await).unwrap();
        value["deep_report"]["producer_receipt"]["runtime_commit"] =
            serde_json::json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let mixed: PrepareRepairRequest = serde_json::from_value(value).unwrap();

        let result = prepare_memory_reclassification(
            &db,
            &RepairArtifactStore::new(repair_root.path().to_path_buf()),
            mixed,
            1_721_000_000,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Conflict(message))
                if message == "repair_source_producers_mismatch"
        ));
        assert_eq!(std::fs::read_dir(repair_root.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn prepare_rejects_reports_from_a_different_page_projection() {
        let (db, _db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let page_root = tempfile::tempdir().unwrap();
        std::fs::write(page_root.path().join("changed.md"), "# changed").unwrap();

        let result = prepare_memory_reclassification_with_pages(
            &db,
            &RepairArtifactStore::new(repair_root.path().to_path_buf()),
            request(&db).await,
            Some(page_root.path()),
            1_721_000_000,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Conflict(message)) if message == "repair_source_reports_stale"
        ));
        assert_eq!(std::fs::read_dir(repair_root.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn prepare_rejects_registered_label_bound_to_another_opaque_scope() {
        let (db, _db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let request = request_for_scope(
            &db,
            RepairLintScope::registered("work".to_string()).unwrap(),
            Some("work"),
        )
        .await;
        let mut value = serde_json::to_value(request).unwrap();
        value["lint_scope"]["space"] = serde_json::json!("personal");
        let rebound: PrepareRepairRequest = serde_json::from_value(value).unwrap();

        let result = prepare_memory_reclassification(
            &db,
            &RepairArtifactStore::new(repair_root.path().to_path_buf()),
            rebound,
            1_721_000_000,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Validation(message)) if message == "repair_scope_mismatch"
        ));
        assert_eq!(std::fs::read_dir(repair_root.path()).unwrap().count(), 0);
    }

    async fn prepared_fixture() -> (
        MemoryDB,
        tempfile::TempDir,
        tempfile::TempDir,
        RepairManifest,
    ) {
        let (db, db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let manifest = prepare_memory_reclassification(
            &db,
            &RepairArtifactStore::new(repair_root.path().to_path_buf()),
            request(&db).await,
            1_721_000_000,
        )
        .await
        .unwrap();
        (db, db_dir, repair_root, manifest)
    }

    fn exact_apply(manifest: &RepairManifest) -> ApplyRepairRequest {
        let approval = format!(
            "apply repair {} {}",
            manifest.manifest_id(),
            manifest.manifest_digest().as_str()
        );
        ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            approval,
        )
        .unwrap()
    }

    async fn target_memory_types(db: &MemoryDB) -> Vec<Option<String>> {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT memory_type FROM memories
                 WHERE source='memory' AND source_id='mem_target'
                 ORDER BY chunk_index,id",
                (),
            )
            .await
            .unwrap();
        let mut output = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            output.push(row.get(0).unwrap());
        }
        output
    }

    async fn verification_reports(
        db: &MemoryDB,
    ) -> (
        wenlan_types::lint::LintReport,
        wenlan_types::lint::LintReport,
    ) {
        verification_reports_at(db, None).await
    }

    async fn verification_reports_at(
        db: &MemoryDB,
        page_root: Option<&Path>,
    ) -> (
        wenlan_types::lint::LintReport,
        wenlan_types::lint::LintReport,
    ) {
        let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .run(
                db,
                &LintQuery::new(Some(LintProfile::General), None),
                page_root,
                page_root.is_some(),
            )
            .await
            .unwrap();
        let prepared = LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_semantic_agent_assist()
            .run(
                db,
                &LintQuery::new(Some(LintProfile::Deep), None),
                page_root,
                page_root.is_some(),
            )
            .await
            .unwrap();
        let work = prepared.agent_work().expect("agent work");
        let verdicts = work
            .candidates()
            .iter()
            .map(|candidate| {
                LintAgentVerdict::try_new(
                    candidate.reference(),
                    LintSemanticDecision::Pass,
                    None,
                    candidate.reason_code(),
                    9_000,
                    vec![],
                )
                .unwrap()
            })
            .collect();
        let submission =
            LintAgentSubmission::try_new(work.work_digest().clone(), verdicts).unwrap();
        let deep = LintRunner::new(LintClock::fixed(), CancellationToken::new())
            .with_semantic_agent_submission(submission)
            .run(
                db,
                &LintQuery::new(Some(LintProfile::Deep), None),
                page_root,
                page_root.is_some(),
            )
            .await
            .unwrap();
        (general, complete_deep_fixture(deep))
    }

    fn exact_verify(
        manifest: &RepairManifest,
        apply_receipt: &RepairApplyReceipt,
        general: wenlan_types::lint::LintReport,
        deep: wenlan_types::lint::LintReport,
    ) -> VerifyRepairRequest {
        VerifyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            apply_receipt.receipt_digest().clone(),
            general,
            deep,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn verify_request_roundtrips_optional_exact_next_apply() {
        let (db, _db_dir, _repair_root, manifest) = prepared_fixture().await;
        let (general, deep) = verification_reports(&db).await;
        let next_manifest_id = "repair_550e8400-e29b-41d4-a716-446655440001";
        let next_digest =
            RepairDigest::parse("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                .unwrap();
        let next_apply = ApplyRepairRequest::try_new(
            next_manifest_id.to_string(),
            next_digest.clone(),
            format!("apply repair {next_manifest_id} {}", next_digest.as_str()),
        )
        .unwrap();
        let request = VerifyRepairRequest::try_new_with_next_apply(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            next_digest.clone(),
            general.clone(),
            deep.clone(),
            Some(next_apply.clone()),
        )
        .unwrap();
        let roundtrip: VerifyRepairRequest =
            serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
        assert_eq!(roundtrip.next_apply(), Some(&next_apply));

        let same_apply = ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                manifest.manifest_digest().as_str()
            ),
        )
        .unwrap();
        assert!(VerifyRepairRequest::try_new_with_next_apply(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            next_digest,
            general,
            deep,
            Some(same_apply),
        )
        .is_err());
    }

    #[tokio::test]
    async fn wrong_approval_or_stale_target_performs_zero_writes() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let before = fingerprint(&db).await;
        let wrong_digest =
            RepairDigest::parse("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                .unwrap();
        let wrong = ApplyRepairRequest::try_new(
            manifest.manifest_id().to_string(),
            wrong_digest.clone(),
            format!(
                "apply repair {} {}",
                manifest.manifest_id(),
                wrong_digest.as_str()
            ),
        )
        .unwrap();

        assert!(apply_repair(&db, &store, wrong, 1_721_000_001)
            .await
            .is_err());
        assert_eq!(before, fingerprint(&db).await);

        db.conn
            .lock()
            .await
            .execute(
                "UPDATE memories SET title='stale' WHERE id='row-target'",
                (),
            )
            .await
            .unwrap();
        let stale_before = fingerprint(&db).await;
        assert!(matches!(
            apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001).await,
            Err(WenlanError::Conflict(message)) if message == "repair_target_stale"
        ));
        assert_eq!(stale_before, fingerprint(&db).await);
        assert!(!store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join("apply-receipt.json")
            .exists());
    }

    #[tokio::test]
    async fn successful_apply_changes_only_declared_owner_closure() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());

        let receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();

        assert_eq!(
            target_memory_types(&db).await,
            vec![Some("decision".to_string()), Some("decision".to_string())]
        );
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT memory_type FROM memories WHERE id='row-other'", ())
            .await
            .unwrap();
        let other: Option<String> = rows.next().await.unwrap().unwrap().get(0).unwrap();
        drop(rows);
        drop(conn);
        assert_eq!(other.as_deref(), Some("fact"));
        assert_eq!(receipt.manifest_digest(), manifest.manifest_digest());
        assert!(store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join("apply-receipt.json")
            .is_file());
    }

    #[tokio::test]
    async fn reclassification_rollback_uncertainty_retains_pending_receipt() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());

        let error = apply_repair_with_pages_with_forced_rollback_failure(
            &db,
            &store,
            exact_apply(&manifest),
            None,
            1_721_000_001,
            RepairWriter::ReclassifyMemory,
        )
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            WenlanError::Conflict(message) if message == "repair_apply_recovery_required"
        ));
        let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
        assert!(manifest_dir.join(APPLY_RECEIPT_PENDING_FILE).is_file());
        assert!(!manifest_dir.join(APPLY_RECEIPT_FILE).exists());
        db.conn.lock().await.execute("ROLLBACK", ()).await.unwrap();
        assert_eq!(target_memory_types(&db).await, vec![None, None]);
    }

    #[tokio::test]
    async fn apply_recovers_committed_receipt_after_unrelated_background_write() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
        std::fs::rename(
            manifest_dir.join(APPLY_RECEIPT_FILE),
            manifest_dir.join(APPLY_RECEIPT_PENDING_FILE),
        )
        .unwrap();
        db.conn
            .lock()
            .await
            .execute(
                "UPDATE memories SET title='background' WHERE id='row-other'",
                (),
            )
            .await
            .unwrap();

        let recovered = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();

        assert_eq!(recovered, receipt);
        assert!(manifest_dir.join(APPLY_RECEIPT_FILE).is_file());
        assert!(!manifest_dir.join(APPLY_RECEIPT_PENDING_FILE).exists());
    }

    #[tokio::test]
    async fn apply_refuses_while_manifest_operation_lock_is_held() {
        use fs2::FileExt as _;

        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let lock_path = store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join(".operation.lock");
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .unwrap();
        lock.lock_exclusive().unwrap();

        let result = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001).await;

        assert!(matches!(
            result,
            Err(WenlanError::Conflict(message)) if message == "repair_operation_in_progress"
        ));
        assert_eq!(target_memory_types(&db).await, vec![None, None]);
    }

    #[tokio::test]
    async fn apply_discards_precommit_partial_receipt_and_retries() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
        std::fs::write(manifest_dir.join(APPLY_RECEIPT_PENDING_FILE), b"partial").unwrap();

        let receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();

        assert_eq!(receipt.manifest_id(), manifest.manifest_id());
        assert!(manifest_dir.join(APPLY_RECEIPT_FILE).is_file());
        assert!(!manifest_dir.join(APPLY_RECEIPT_PENDING_FILE).exists());
    }

    #[tokio::test]
    async fn effect_escape_rolls_back_target_and_trigger_side_effect() {
        let (db, _db_dir) = fixture().await;
        db.conn
            .lock()
            .await
            .execute_batch(
                "CREATE TABLE repair_escape (value TEXT NOT NULL);
                 CREATE TRIGGER repair_escape_trigger AFTER UPDATE OF memory_type ON memories
                 WHEN NEW.source_id='mem_target'
                 BEGIN INSERT INTO repair_escape(value) VALUES ('escaped'); END;",
            )
            .await
            .unwrap();
        let repair_root = tempfile::tempdir().unwrap();
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let manifest =
            prepare_memory_reclassification(&db, &store, request(&db).await, 1_721_000_000)
                .await
                .unwrap();
        let before = fingerprint(&db).await;

        let result = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001).await;

        assert!(matches!(
            result,
            Err(WenlanError::VectorDb(message)) if message.contains("repair_effect_escape")
        ));
        assert_eq!(before, fingerprint(&db).await);
        assert_eq!(target_memory_types(&db).await, vec![None, None]);
    }

    #[tokio::test]
    async fn verification_records_receipt_only_after_post_lint_and_effect_proof() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;

        let receipt = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await
        .unwrap();

        assert_eq!(receipt.manifest_id(), manifest.manifest_id());
        assert!(store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join("verification-receipt.json")
            .is_file());
    }

    #[tokio::test]
    async fn post_repair_general_and_deep_must_share_one_producer() {
        let (db, _db_dir, _repair_root, manifest) = prepared_fixture().await;
        let (general, deep) = verification_reports(&db).await;
        let mut deep_value = serde_json::to_value(deep).unwrap();
        let replacement_commit = if general
            .producer_receipt()
            .runtime_commit()
            .is_some_and(|commit| commit.as_str() == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        {
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        } else {
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        };
        deep_value["producer_receipt"]["runtime_commit"] = serde_json::json!(replacement_commit);
        let mixed_deep: wenlan_types::lint::LintReport =
            serde_json::from_value(deep_value).unwrap();

        assert!(
            VerifyRepairRequest::try_new(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                RepairDigest::parse(
                    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                )
                .unwrap(),
                general.clone(),
                mixed_deep.clone(),
            )
            .is_err()
        );
        assert!(matches!(
            validate_verification_reports(&manifest, &general, Some(&mixed_deep)),
            Err(WenlanError::Validation(message))
                if message == "repair_verification_report_mismatch"
        ));
    }

    #[tokio::test]
    async fn pending_verification_manifest_ids_tracks_apply_until_verify() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        assert!(store
            .pending_verification_manifest_ids()
            .unwrap()
            .is_empty());

        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        assert_eq!(
            store.pending_verification_manifest_ids().unwrap(),
            vec![manifest.manifest_id().to_string()]
        );

        let (general, deep) = verification_reports(&db).await;
        record_repair_verification(
            &db,
            &store,
            VerifyRepairRequest::try_new(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                general,
                deep,
            )
            .unwrap(),
            None,
            1_721_000_002,
        )
        .await
        .unwrap();
        assert!(store
            .pending_verification_manifest_ids()
            .unwrap()
            .is_empty());
        assert!(store
            .has_completed_verification(manifest.manifest_id())
            .unwrap());
    }

    #[test]
    fn pending_verification_ignores_unapplied_unreadable_manifest() {
        let repair_root = tempfile::tempdir().unwrap();
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let manifest_id = format!("repair_{}", Uuid::new_v4());
        let manifest_dir = repair_root.path().join(manifest_id);
        std::fs::create_dir_all(&manifest_dir).unwrap();
        std::fs::write(manifest_dir.join(MANIFEST_FILE), b"{}").unwrap();

        assert!(store
            .pending_verification_manifest_ids()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn pending_verification_rejects_pending_unreadable_manifest() {
        let repair_root = tempfile::tempdir().unwrap();
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let manifest_id = format!("repair_{}", Uuid::new_v4());
        let manifest_dir = repair_root.path().join(manifest_id);
        std::fs::create_dir_all(&manifest_dir).unwrap();
        std::fs::write(manifest_dir.join(MANIFEST_FILE), b"{}").unwrap();
        std::fs::write(manifest_dir.join(APPLY_RECEIPT_PENDING_FILE), b"{}").unwrap();

        assert!(store.pending_verification_manifest_ids().is_err());
    }

    #[tokio::test]
    async fn verification_clears_crash_window_pending_apply_link() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
        std::fs::hard_link(
            manifest_dir.join(APPLY_RECEIPT_FILE),
            manifest_dir.join(APPLY_RECEIPT_PENDING_FILE),
        )
        .unwrap();

        assert_eq!(
            store.pending_verification_manifest_ids().unwrap(),
            vec![manifest.manifest_id().to_string()]
        );

        let (general, deep) = verification_reports(&db).await;
        record_repair_verification(
            &db,
            &store,
            VerifyRepairRequest::try_new(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                general,
                deep,
            )
            .unwrap(),
            None,
            1_721_000_002,
        )
        .await
        .unwrap();

        assert!(!manifest_dir.join(APPLY_RECEIPT_PENDING_FILE).exists());
        assert!(store
            .pending_verification_manifest_ids()
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn verification_retry_returns_the_existing_receipt() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        let request = exact_verify(&manifest, &apply_receipt, general, deep);

        let first = record_repair_verification(&db, &store, request.clone(), None, 1_721_000_002)
            .await
            .unwrap();
        let retried = record_repair_verification(&db, &store, request, None, 1_721_000_003)
            .await
            .unwrap();

        assert_eq!(retried, first);
    }

    #[tokio::test]
    async fn failed_post_apply_verification_can_resume_with_fresh_valid_reports() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        let failed_deep = fail_deep_check(deep.clone(), false);

        let failed = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general.clone(), failed_deep),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            failed,
            Err(WenlanError::Validation(message)) if message == "repair_new_incomplete_check"
        ));
        assert!(!store
            .has_completed_verification(manifest.manifest_id())
            .unwrap());

        let receipt = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_003,
        )
        .await
        .unwrap();

        assert_eq!(receipt.manifest_id(), manifest.manifest_id());
        assert!(store
            .has_completed_verification(manifest.manifest_id())
            .unwrap());
    }

    #[tokio::test]
    async fn verification_refuses_while_manifest_operation_lock_is_held() {
        use fs2::FileExt as _;

        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        let lock_path = store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join(OPERATION_LOCK_FILE);
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(lock_path)
            .unwrap();
        lock.lock_exclusive().unwrap();

        let result = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Conflict(message)) if message == "repair_operation_in_progress"
        ));
        assert!(!store
            .manifest_dir(manifest.manifest_id())
            .unwrap()
            .join(VERIFICATION_RECEIPT_FILE)
            .exists());
    }

    #[tokio::test]
    async fn verification_rejects_target_evidence_that_survives_post_lint() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        let mut value = serde_json::to_value(deep).unwrap();
        let checks = value["checks"].as_array_mut().unwrap();
        let target = checks
            .iter_mut()
            .find(|check| check["check_id"] == REPAIR_CLASSIFICATION_CHECK_ID)
            .unwrap();
        target["outcome"] = serde_json::json!("finding");
        target["severity"] = serde_json::json!("warning");
        target["applicability"] = serde_json::json!("applicable");
        target["precondition"] = serde_json::json!("ready");
        target["evidence"] = serde_json::json!([{
            "kind": "semantic_finding",
            "finding": manifest.source().finding()
        }]);
        target["coverage"]["evidence_returned"] = serde_json::json!(1);
        let passed = checks
            .iter()
            .filter(|check| check["outcome"] == "pass")
            .count();
        let findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding")
            .count();
        let actionable_findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding" && check["gate_effect"] == "actionable")
            .count();
        value["totals"] = serde_json::json!({
            "checks": checks.len(),
            "passed": passed,
            "findings": findings,
            "actionable_findings": actionable_findings,
            "advisory_findings": findings - actionable_findings,
            "incomplete": 0
        });
        let deep = serde_json::from_value(value).unwrap();

        let result = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Validation(message)) if message == "repair_target_assertion_failed"
        ));
    }

    #[tokio::test]
    async fn verification_accepts_unrelated_write_after_apply_when_reports_are_fresh() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        db.conn
            .lock()
            .await
            .execute(
                "UPDATE memories SET title='changed' WHERE id='row-other'",
                (),
            )
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;

        let receipt = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await
        .unwrap();

        assert_eq!(
            receipt.apply_receipt_digest(),
            apply_receipt.receipt_digest()
        );
    }

    #[tokio::test]
    async fn verification_accepts_in_place_metadata_update_when_reports_are_fresh() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        db.set_app_metadata("repair_digest_probe", "before")
            .await
            .unwrap();
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        db.set_app_metadata("repair_digest_probe", "after")
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;

        let receipt = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await
        .unwrap();

        assert_eq!(
            receipt.apply_receipt_digest(),
            apply_receipt.receipt_digest()
        );
    }

    #[tokio::test]
    async fn verification_rejects_target_owner_change_after_apply_even_with_fresh_reports() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        db.conn
            .lock()
            .await
            .execute(
                "UPDATE memories SET title='changed' WHERE id='row-target'",
                (),
            )
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;

        let result = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Conflict(message)) if message == "repair_verification_state_changed"
        ));
    }

    #[tokio::test]
    async fn verification_rejects_reports_that_are_no_longer_current() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        db.conn
            .lock()
            .await
            .execute("UPDATE memories SET title='later' WHERE id='row-other'", ())
            .await
            .unwrap();

        let result = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Conflict(message)) if message == "repair_verification_reports_stale"
        ));
    }

    #[tokio::test]
    async fn verification_rejects_reports_after_page_projection_changes() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let page_root = tempfile::tempdir().unwrap();
        let (general, deep) = verification_reports_at(&db, Some(page_root.path())).await;
        std::fs::write(page_root.path().join("changed.md"), "# changed").unwrap();

        let result = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            Some(page_root.path()),
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Conflict(message)) if message.starts_with("repair_verification_reports_stale")
        ));
    }

    #[tokio::test]
    async fn verification_rejects_new_actionable_check() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        let mut value = serde_json::to_value(deep).unwrap();
        let checks = value["checks"].as_array_mut().unwrap();
        let injected = checks
            .iter_mut()
            .find(|check| {
                check["check_id"] != REPAIR_CLASSIFICATION_CHECK_ID
                    && check["outcome"] == "pass"
                    && check["gate_effect"] == "actionable"
            })
            .unwrap();
        injected["outcome"] = serde_json::json!("finding");
        injected["severity"] = serde_json::json!("warning");
        injected["applicability"] = serde_json::json!("applicable");
        injected["precondition"] = serde_json::json!("ready");
        let passed = checks
            .iter()
            .filter(|check| check["outcome"] == "pass")
            .count();
        let findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding")
            .count();
        let actionable_findings = checks
            .iter()
            .filter(|check| check["outcome"] == "finding" && check["gate_effect"] == "actionable")
            .count();
        value["totals"] = serde_json::json!({
            "checks": checks.len(),
            "passed": passed,
            "findings": findings,
            "actionable_findings": actionable_findings,
            "advisory_findings": findings - actionable_findings,
            "incomplete": 0
        });
        let deep = serde_json::from_value(value).unwrap();

        let result = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Validation(message)) if message == "repair_new_actionable_finding"
        ));
    }

    #[tokio::test]
    async fn verification_accepts_unchanged_unrelated_deep_incomplete_baseline() {
        let (db, _db_dir) = fixture().await;
        let repair_root = tempfile::tempdir().unwrap();
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let request = with_incomplete_deep(request(&db).await, false).unwrap();
        let manifest = prepare_memory_reclassification(&db, &store, request, 1_721_000_000)
            .await
            .unwrap();
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        let deep = fail_deep_check(deep, false);

        let receipt = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await
        .unwrap();

        assert_eq!(receipt.manifest_id(), manifest.manifest_id());
    }

    #[tokio::test]
    async fn verification_rejects_new_unrelated_deep_incomplete_check() {
        let (db, _db_dir, repair_root, manifest) = prepared_fixture().await;
        let store = RepairArtifactStore::new(repair_root.path().to_path_buf());
        let apply_receipt = apply_repair(&db, &store, exact_apply(&manifest), 1_721_000_001)
            .await
            .unwrap();
        let (general, deep) = verification_reports(&db).await;
        let deep = fail_deep_check(deep, false);

        let result = record_repair_verification(
            &db,
            &store,
            exact_verify(&manifest, &apply_receipt, general, deep),
            None,
            1_721_000_002,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Validation(message)) if message == "repair_new_incomplete_check"
        ));
    }

    #[test]
    fn semantic_digest_binds_kind_and_durable_id() {
        assert_eq!(
            semantic_record_digest("memory", "mem_target"),
            semantic_record_digest("memory", "mem_target")
        );
        assert_ne!(
            semantic_record_digest("memory", "mem_target"),
            semantic_record_digest("entity", "mem_target")
        );
    }
}
