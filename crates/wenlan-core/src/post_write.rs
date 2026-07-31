// SPDX-License-Identifier: Apache-2.0
//! Canonical write-path capability functions. Each fn owns the full create
//! flow for one kind: validation, resolve-or-create (where applicable),
//! storage primitive call, post-write enrichment (verify, log, refinery
//! enqueue). Both HTTP route handlers and daemon-internal extractors call
//! these -- eliminating drift between agent-LLM and daemon-LLM trigger paths.

use crate::db::MemoryDB;
use crate::error::WenlanError;
use std::{path::Path, str::FromStr};
use wenlan_types::{
    repair::{RepairDigest, RepairManifest, RepairRollbackPayloadV2},
    MemoryType,
};

mod entity_graph;
mod page_create;
mod page_dispatch;
mod page_revision;
mod page_update;

pub use self::entity_graph::{
    add_observation, create_entity, create_relation, create_relation_with_span,
};
pub use self::page_dispatch::{
    create_page, create_page_with_floor, create_page_with_tuning, page_write, update_page,
    update_page_preserving_sources, PageWrite,
};
pub(crate) use self::page_dispatch::{
    update_page_at_source_revision, update_page_growth_at_versions,
};
pub use self::page_revision::{
    accept_pending_revision, accept_pending_revision_with_knowledge_path, dismiss_contradiction,
    dismiss_pending_revision,
};
pub(crate) use self::page_update::merge_shrink_threshold;
#[cfg(test)]
pub(crate) use self::page_update::PRE_WRITE_GATE;
pub use self::page_update::{page_is_human_owned, stage_page_revision_card, PipelineStage, Writer};

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

#[cfg(test)]
#[path = "post_write/post_write_tests.rs"]
mod tests;
