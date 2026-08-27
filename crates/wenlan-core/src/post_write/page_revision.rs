use super::log_activity_best_effort;
use crate::{
    db::{MemoryDB, PendingMemoryRevisionPayload},
    error::WenlanError,
};
use std::path::Path;

struct PageRevisionCard {
    page_id: String,
    revision_id: String,
    page_version: Option<i64>,
    source_revision: Option<i64>,
    content: String,
    source_memory_ids: Vec<String>,
}

async fn resolve_page_revision_card(
    db: &MemoryDB,
    id: &str,
) -> Result<Option<PageRevisionCard>, WenlanError> {
    let payload: Option<PendingMemoryRevisionPayload> =
        db.pending_memory_revision_payload(id).await?;
    let Some(payload) = payload else {
        return Ok(None);
    };
    let structured = payload
        .structured_fields
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

    let Some(structured) = structured else {
        return Ok(None);
    };
    if structured.get("revision_kind").and_then(|v| v.as_str()) != Some("page_write")
        || structured.get("target_kind").and_then(|v| v.as_str()) != Some("page")
    {
        return Ok(None);
    }

    // The page branch must prove itself against the database, never against
    // the card's own JSON. `structured_fields` is persisted verbatim from a
    // wire store, so routing on those three strings alone let a low-trust
    // agent stage a memory correction that carried the page markers and turn
    // the human's accept click into an overwrite of a human-authored page
    // (and their dismiss click into a deletion of a captured memory). The one
    // writer of real cards, `stage_page_revision_card`, always sets
    // `supersedes` to the page id, so this check costs a genuine card
    // nothing. It is also the same fact `list_pending_revisions_scoped` uses
    // to label the card for the human, so accept and the queue now agree.
    if db.get_page(&payload.supersedes).await?.is_none() {
        return Ok(None);
    }
    let page_id = payload.supersedes.clone();
    let source_memory_ids = structured
        .get("source_memory_ids")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let page_version = structured.get("page_version").and_then(|v| v.as_i64());
    // `stage_page_revision_card` now requires every caller to pass a real
    // counter, so a fresh card always carries a concrete number here. `None`
    // only decodes two legacy shapes: cards staged before the field existed
    // at all (missing key), and cards staged in the window where the field
    // existed but callers could still pass `None` through it (a literal JSON
    // `null`). Both are treated the same as a pre-versioning `page_version`:
    // accepted on the version fence alone rather than inventing a base to
    // check.
    let source_revision = structured.get("source_revision").and_then(|v| v.as_i64());

    Ok(Some(PageRevisionCard {
        page_id,
        revision_id: payload.revision_id,
        page_version,
        source_revision,
        content: payload.content,
        source_memory_ids,
    }))
}

// Current PageWrite cards record the Page version they were staged from, and
// `try_accept_page_revision` checks it atomically with card consumption.
// Legacy cards without `page_version` remain accepted for compatibility; their
// missing base is explicit in `PageRevisionCard` rather than silently invented.
async fn accept_page_revision_card(
    db: &MemoryDB,
    card: PageRevisionCard,
    knowledge_path: Option<&Path>,
) -> Result<wenlan_types::RevisionAcceptResponse, WenlanError> {
    let current = db
        .get_page(&card.page_id)
        .await?
        .ok_or_else(|| WenlanError::NotFound(format!("Page not found: {}", card.page_id)))?;
    let source_memory_ids = if card.source_memory_ids.is_empty() {
        current.source_memory_ids.clone()
    } else {
        card.source_memory_ids.clone()
    };
    let source_refs: Vec<&str> = source_memory_ids.iter().map(String::as_str).collect();
    let old_set: std::collections::HashSet<&str> = current
        .source_memory_ids
        .iter()
        .map(|s| s.as_str())
        .collect();
    let new_set: std::collections::HashSet<&str> = source_refs.iter().copied().collect();
    let mut added_sources: Vec<&str> = new_set.difference(&old_set).copied().collect();
    added_sources.sort_unstable();
    let added_sources_json = serde_json::Value::Array(
        added_sources
            .iter()
            .map(|s| serde_json::Value::String((*s).to_string()))
            .collect(),
    );
    let new_version = current.version + 1;
    let entry = serde_json::json!({
        "version": new_version,
        "at": chrono::Utc::now().timestamp(),
        "edited_by": "revision_accept",
        "delta_summary": crate::db::compute_page_delta_summary(
            &current.content,
            &current.source_memory_ids,
            &card.content,
            &source_refs,
            "revision_accept",
        ),
        "incoming_source_ids": added_sources_json,
    });
    let existing_cl = db.get_page_changelog(&card.page_id).await?;
    const DEFAULT_CHANGELOG_CAP: usize = 20;
    let new_changelog =
        crate::db::append_changelog_entry(&existing_cl, entry, DEFAULT_CHANGELOG_CAP)?;

    let projection = knowledge_path.map(|path| {
        crate::export::knowledge::KnowledgeProjectionWrite::new(path.to_path_buf(), db)
    });
    let wrote = db
        .try_accept_page_revision(
            &card.page_id,
            &card.content,
            &source_refs,
            &new_changelog,
            card.page_version,
            card.source_revision,
            &card.revision_id,
        )
        .await?;

    if !wrote {
        let current_version = db
            .get_page(&card.page_id)
            .await?
            .ok_or_else(|| WenlanError::NotFound(format!("Page not found: {}", card.page_id)))?
            .version;
        let Some(staged_version) = card.page_version else {
            return Err(WenlanError::Conflict(format!(
                "page revision card {} for page {} did not write",
                card.revision_id, card.page_id
            )));
        };
        let mut msg = format!(
            "page revision card {} was staged for page {} at staged version {}, but current version {} no longer matches",
            card.revision_id, card.page_id, staged_version, current_version
        );
        // A source attached after this card was staged bumps only
        // `source_revision`, leaving `version` unchanged -- so a
        // source-revision-only conflict would otherwise report identical
        // staged/current versions here with no clue why the write was
        // rejected. Name the counter that actually moved.
        if let Some(staged_source_revision) = card.source_revision {
            let current_source_revision = db.try_get_page_source_revision(&card.page_id).await?;
            msg.push_str(&format!(
                ", staged source revision {staged_source_revision}, current source revision {}",
                current_source_revision
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            ));
        }
        return Err(WenlanError::Conflict(msg));
    }

    if let Some(ref projection) = projection {
        if let Ok(Some(updated_page)) = db.get_page(&card.page_id).await {
            if let Err(e) = projection.write_page_gated(db, &updated_page).await {
                log::warn!(
                    "[accept_page_revision_card] md re-write failed for {}: {e}",
                    card.page_id
                );
            }
        }
    }
    drop(projection);

    Ok(wenlan_types::RevisionAcceptResponse {
        target_source_id: card.page_id,
        revision_source_id: card.revision_id,
        wrote: true,
    })
}

/// Accept a pending memory revision. Canonical entry for both agent-triggered
/// (`/api/memory/revision/{id}/accept`) and daemon-internal accept-dispatch.
/// Activates the revision row, suppresses the original, and logs activity.
/// Returns `NotFound` if no pending revision exists for the supplied id.
pub async fn accept_pending_revision(
    db: &MemoryDB,
    id: &str,
    agent: &str,
) -> Result<wenlan_types::RevisionAcceptResponse, WenlanError> {
    accept_pending_revision_with_knowledge_path(db, id, agent, None).await
}

pub async fn accept_pending_revision_with_knowledge_path(
    db: &MemoryDB,
    id: &str,
    agent: &str,
    knowledge_path: Option<&Path>,
) -> Result<wenlan_types::RevisionAcceptResponse, WenlanError> {
    if let Some(card) = resolve_page_revision_card(db, id).await? {
        let result = accept_page_revision_card(db, card, knowledge_path).await?;
        log_activity_best_effort(db, agent, "revision_accept", &result.target_source_id).await;
        return Ok(result);
    }

    // `id` may be the revision's own source_id (exact) or its target's (legacy);
    // the DB resolves it and returns the actual (target, revision) pair acted on.
    let (target_source_id, revision_source_id) = db.accept_pending_revision(id).await?;
    log_activity_best_effort(db, agent, "revision_accept", &target_source_id).await;

    Ok(wenlan_types::RevisionAcceptResponse {
        target_source_id,
        revision_source_id,
        wrote: true,
    })
}

/// Dismiss a pending memory revision. Canonical entry for both
/// agent-triggered (`/api/memory/revision/{id}/dismiss`) and daemon-internal
/// triggers. Unstages the pending revision (clears its false revision link,
/// keeping it as an independent row); the original is untouched.
/// Returns `NotFound` if no pending revision exists for the supplied id.
///
/// A PAGE revision card is the exception and is DELETED, mirroring the accept
/// path's `consume_revision_id`. Unstage-and-keep is right for a memory card,
/// which is a distinct capture that merely topic-matched; a page card is
/// manufactured by `stage_page_revision_card` from the page's own prose, so
/// keeping it would leave a permanent standalone copy of the page body in
/// `memories` — reachable from every memory reader, carrying none of the
/// page's truth state.
pub async fn dismiss_pending_revision(
    db: &MemoryDB,
    id: &str,
    agent: &str,
) -> Result<wenlan_types::RevisionDismissResponse, WenlanError> {
    if let Some(card) = resolve_page_revision_card(db, id).await? {
        db.delete_by_source_id("memory", &card.revision_id).await?;
        log_activity_best_effort(db, agent, "revision_dismiss", &card.page_id).await;
        return Ok(wenlan_types::RevisionDismissResponse {
            target_source_id: card.page_id,
            wrote: true,
        });
    }

    let (target_source_id, _revision_source_id) = db.dismiss_pending_revision(id).await?;
    log_activity_best_effort(db, agent, "revision_dismiss", &target_source_id).await;
    Ok(wenlan_types::RevisionDismissResponse {
        target_source_id,
        wrote: true,
    })
}

/// Dismiss all awaiting-review contradiction flags for a memory. Canonical
/// entry for both agent-triggered (`/api/memory/contradiction/{source_id}/dismiss`)
/// and daemon-internal triggers. `wrote: true` is best-effort: the DB method
/// silently no-ops when no rows match. See spec §3 for the caveat.
pub async fn dismiss_contradiction(
    db: &MemoryDB,
    source_id: &str,
    agent: &str,
) -> Result<wenlan_types::ContradictionDismissResponse, WenlanError> {
    db.dismiss_contradiction_for_source(source_id).await?;
    log_activity_best_effort(db, agent, "contradiction_dismiss", source_id).await;
    Ok(wenlan_types::ContradictionDismissResponse {
        source_id: source_id.to_string(),
        wrote: true,
    })
}
