// SPDX-License-Identifier: Apache-2.0
//! Refinement queue phase: drain pending memories, apply merge-by-tier logic.

use crate::contradiction::ContradictionResult;
use crate::db::MemoryDB;
use crate::error::WenlanError;
use crate::llm_provider::{LlmProvider, LlmRequest};
use crate::post_write::{page_write, PageWrite, WriteResult};
use crate::prompts::PromptRegistry;
use crate::synthesis::distill::apply_merge_by_tier;
use std::sync::Arc;
use wenlan_types::requests::CreateConceptRequest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveStatus {
    Dismissed,
    AwaitingReview,
    AutoApplied,
    Resolved,
}

impl ResolveStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ResolveStatus::Dismissed => "dismissed",
            ResolveStatus::AwaitingReview => "awaiting_review",
            ResolveStatus::AutoApplied => "auto_applied",
            ResolveStatus::Resolved => "resolved",
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            ResolveStatus::Dismissed | ResolveStatus::AutoApplied | ResolveStatus::Resolved
        )
    }
}

/// Resolve a refinement queue proposal. Convergent capability fn used by both the
/// daemon scheduler (refinery phase) and the HTTP reject route. Wraps
/// `db.resolve_refinement_if_open` with idempotency semantics + agent activity logging.
pub async fn resolve_proposal(
    db: &MemoryDB,
    id: &str,
    status: ResolveStatus,
    agent: &str,
) -> Result<WriteResult, WenlanError> {
    let rows = db.resolve_refinement_if_open(id, status.as_str()).await?;

    if rows == 0 {
        // Distinguish 404 (missing) vs 422 (already terminal) via a follow-up lookup.
        return match db.get_refinement_proposal(id).await? {
            None => Err(WenlanError::NotFound(format!(
                "refinement proposal {id} not found"
            ))),
            Some(p) => Err(WenlanError::Validation(format!(
                "refinement proposal {id} already resolved (status={})",
                p.status
            ))),
        };
    }

    // Read back for activity payload — proposal still exists (we just resolved it).
    if let Ok(Some(prop)) = db.get_refinement_proposal(id).await {
        let payload = serde_json::json!({
            "action": prop.action,
            "new_status": status.as_str(),
            "source_ids": prop.source_ids,
        })
        .to_string();
        let _ = db
            .log_agent_activity(agent, "refinement_resolve", &[], None, &payload)
            .await;
    }

    Ok(WriteResult {
        id: id.to_string(),
        attached_to: None,
        warnings: Vec::new(),
        wrote: true,
        revision_card_id: None,
        gated: false,
        outcome: crate::post_write::WriteOutcome::Wrote,
        acknowledged: false,
    })
}

/// Outcome of `apply_refinement`. Returned by the HTTP route and MCP tool.
#[derive(Debug)]
pub struct AcceptOutcome {
    pub id: String,
    pub action_applied: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefinementDecision {
    Accept {
        notes: Option<String>,
    },
    PickSpace {
        space: String,
        notes: Option<String>,
    },
}

/// Apply a refinement queue proposal using sensible defaults per action variant.
/// Single dispatch point for both daemon-internal accept paths (none today; reserved)
/// and agent HTTP triggers. Wraps PR1's idempotent primitives + atomic `resolve_proposal`.
///
/// Defaults:
/// - `entity_merge`: existing entity (source_ids[1]) wins as canonical, new (source_ids[0]) folds in as alias.
/// - `relation_conflict`: new relation (source_ids[0]) wins, existing (source_ids[1]) deleted.
/// - `detect_contradiction`: acknowledge-and-resolve only — neither memory is mutated
///   (a pure contradiction keeps both facts; the SUPERSEDES auto-merge is daemon-side).
/// - `suggest_entity`, `dedup_merge`, unknown: returns Validation (422).
pub async fn apply_refinement(
    db: &MemoryDB,
    id: &str,
    agent: &str,
) -> Result<AcceptOutcome, WenlanError> {
    apply_refinement_with_decision(db, id, agent, RefinementDecision::Accept { notes: None }).await
}

pub async fn apply_refinement_with_decision(
    db: &MemoryDB,
    id: &str,
    agent: &str,
    decision: RefinementDecision,
) -> Result<AcceptOutcome, WenlanError> {
    let prop = db
        .get_refinement_proposal(id)
        .await?
        .ok_or_else(|| WenlanError::NotFound(format!("refinement proposal {id} not found")))?;

    // Mirror the terminal-status set from resolve_refinement_if_open SQL.
    if matches!(
        prop.status.as_str(),
        "dismissed" | "auto_applied" | "resolved"
    ) {
        return Err(WenlanError::Validation(format!(
            "refinement proposal {id} already resolved (status={})",
            prop.status
        )));
    }

    let mut resolved_inside_action = false;
    match prop.action.as_str() {
        "entity_merge" => {
            let new_id = prop.source_ids.first().ok_or_else(|| {
                WenlanError::Validation("entity_merge missing source_ids[0]".into())
            })?;
            let existing_id = prop.source_ids.get(1).ok_or_else(|| {
                WenlanError::Validation("entity_merge missing source_ids[1]".into())
            })?;
            db.merge_entities(existing_id, new_id).await?;
        }
        "relation_conflict" => {
            let new_id = prop.source_ids.first().ok_or_else(|| {
                WenlanError::Validation("relation_conflict missing source_ids[0]".into())
            })?;
            let existing_id = prop.source_ids.get(1).ok_or_else(|| {
                WenlanError::Validation("relation_conflict missing source_ids[1]".into())
            })?;
            db.supersede_relation(existing_id, new_id).await?;
        }
        "detect_contradiction" => {
            // Validate the proposal shape, then acknowledge-and-resolve below —
            // do NOT mutate either memory. A pure contradiction keeps both facts;
            // the auto-merge (SUPERSEDES) case is handled daemon-side in
            // process_refinement_queue, never here. Flagging pending_revision
            // without a supersedes link quarantined the memory (hidden from
            // retrieval, absent from /curate, no path to clear). ponytail: accept
            // = clear it from the review queue, nothing more.
            prop.source_ids.get(1).ok_or_else(|| {
                WenlanError::Validation("detect_contradiction missing source_ids[1]".into())
            })?;
        }
        "suggest_entity" => {
            return Err(WenlanError::Validation(
                "action 'suggest_entity' has no accept path (reserved for future producer)".into(),
            ));
        }
        "dedup_merge" => {
            return Err(WenlanError::Validation(
                "action 'dedup_merge' has no accept path (deprecated stale-v1 variant)".into(),
            ));
        }
        "lint_repair_review" => {
            return Err(WenlanError::Validation(
                "action 'lint_repair_review' requires a choice-specific repair flow".into(),
            ));
        }
        "page_merge" => {
            let survivor_id = prop.source_ids.first().cloned().ok_or_else(|| {
                WenlanError::Validation("page_merge missing source_ids[0]".into())
            })?;
            let absorbed_id = prop.source_ids.get(1).cloned().ok_or_else(|| {
                WenlanError::Validation("page_merge missing source_ids[1]".into())
            })?;
            db.accept_page_merge(id, &survivor_id, &absorbed_id).await?;
            resolved_inside_action = true;
        }
        "cross_space_discovery" => match decision {
            RefinementDecision::PickSpace { space, notes: _ } => {
                apply_cross_space_discovery(db, &prop.source_ids, &space, agent).await?;
            }
            RefinementDecision::Accept { notes: _ } => {
                return Err(WenlanError::Validation(
                    "cross_space_discovery requires pick_space action".into(),
                ));
            }
        },
        "page_keep_or_archive" => match decision {
            RefinementDecision::Accept { notes: _ } => {
                let page_id = prop.source_ids.first().ok_or_else(|| {
                    WenlanError::Validation("page_keep_or_archive missing source_ids[0]".into())
                })?;
                db.archive_page(page_id).await?;
            }
            RefinementDecision::PickSpace { space: _, notes: _ } => {
                return Err(WenlanError::Validation(
                    "page_keep_or_archive requires accept or dismiss".into(),
                ));
            }
        },
        "vocab_promote" => {
            let payload = prop
                .payload
                .as_deref()
                .ok_or_else(|| WenlanError::Validation("vocab_promote missing payload".into()))?;
            let v: serde_json::Value = serde_json::from_str(payload)
                .map_err(|e| WenlanError::Validation(format!("vocab_promote payload: {e}")))?;
            let kind = v["kind"].as_str().unwrap_or_default();
            let old_value = v["old_value"].as_str().unwrap_or_default();
            let category = v["category"].as_str();
            db.promote_vocabulary_canonical(kind, old_value, category)
                .await?;
            // The entities that proposed this type were stored under the
            // default `concept` (their ids are the proposal's `source_ids`);
            // now that the type is canonical, move them onto it.
            if kind == "entity" {
                let canonical = old_value.to_lowercase();
                db.retype_entities_after_promote(&prop.source_ids, &canonical)
                    .await?;
            }
        }
        other => {
            return Err(WenlanError::Validation(format!("unknown action: {other}")));
        }
    }

    if resolved_inside_action {
        let resolve_payload = serde_json::json!({
            "action": &prop.action,
            "new_status": ResolveStatus::Resolved.as_str(),
            "source_ids": &prop.source_ids,
        })
        .to_string();
        let _ = db
            .log_agent_activity(agent, "refinement_resolve", &[], None, &resolve_payload)
            .await;
    } else {
        resolve_proposal(db, id, ResolveStatus::Resolved, agent).await?;
    }

    let payload = serde_json::json!({
        "action": &prop.action,
        "source_ids": &prop.source_ids,
    })
    .to_string();
    let _ = db
        .log_agent_activity(agent, "refinement_apply", &prop.source_ids, None, &payload)
        .await;

    Ok(AcceptOutcome {
        id: id.to_string(),
        action_applied: prop.action,
    })
}

async fn apply_cross_space_discovery(
    db: &MemoryDB,
    source_ids: &[String],
    space: &str,
    agent: &str,
) -> Result<(), WenlanError> {
    let space = space.trim();
    if space.is_empty() {
        return Err(WenlanError::Validation(
            "pick_space requires a space".into(),
        ));
    }
    let source_contents = db.get_memory_contents_by_ids(source_ids).await?;
    if source_contents.is_empty() {
        return Err(WenlanError::Validation(
            "cross_space_discovery has no readable source memories".into(),
        ));
    }
    let title = discovery_page_title(&source_contents);
    let content = discovery_page_content(&source_contents);
    let req = CreateConceptRequest {
        title,
        content,
        summary: None,
        entity_id: None,
        space: (Some(space.to_string())).into(),
        source_memory_ids: source_ids.to_vec(),
        creation_kind: Some("distilled".to_string()),
        workspace: Some(space.to_string()),
    };
    page_write(
        db,
        PageWrite::Create {
            page_id: None,
            req,
            agent,
            knowledge_path: None,
            page_min_cluster_size: crate::tuning::DistillationConfig::default()
                .page_min_cluster_size,
            page_match_threshold: crate::tuning::DistillationConfig::default().page_match_threshold,
            citations_json: None,
        },
    )
    .await?;
    Ok(())
}

fn discovery_page_title(source_contents: &[(String, String)]) -> String {
    let title_words = source_contents
        .first()
        .map(|(_, content)| {
            content
                .split_whitespace()
                .take(8)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_else(|| "Cross-space discovery".to_string());
    format!("Discovered topic: {title_words}")
}

fn discovery_page_content(source_contents: &[(String, String)]) -> String {
    source_contents
        .iter()
        .map(|(_, content)| content.trim())
        .filter(|content| !content.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Process pending refinement queue items via LLM.
pub(crate) async fn process_refinement_queue(
    db: &MemoryDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    tuning: &crate::tuning::RefineryConfig,
) -> Result<usize, WenlanError> {
    let pending = db.get_pending_refinements().await?;
    let mut processed = 0usize;

    for proposal in pending
        .iter()
        .filter(|proposal| proposal.status == "pending")
        .take(tuning.max_proposals_per_steep)
    {
        match proposal.action.as_str() {
            "dedup_merge" => {
                // Stale v1 proposal — dismiss (distillation handles merges now)
                resolve_proposal(db, &proposal.id, ResolveStatus::Dismissed, "daemon").await?;
                processed += 1;
            }
            "detect_contradiction" => {
                if let Some(llm) = llm {
                    let contents = db.get_memory_contents(&proposal.source_ids).await?;
                    if contents.len() < 2 {
                        resolve_proposal(db, &proposal.id, ResolveStatus::Dismissed, "daemon")
                            .await?;
                        continue;
                    }

                    let existing_content = contents.get(1).cloned().unwrap_or_default();
                    let new_content = contents.first().cloned().unwrap_or_default();

                    let response = llm
                        .generate(LlmRequest {
                            system_prompt: Some(prompts.detect_contradiction.clone()),
                            user_prompt: format!(
                                "Existing: {}\nNew: {}",
                                existing_content, new_content
                            ),
                            max_tokens: 256,
                            temperature: 0.1,
                            label: None,
                            timeout_secs: None,
                        })
                        .await;

                    if let Ok(r) = response {
                        let r = crate::llm_provider::strip_think_tags(&r);
                        let r = r.trim().to_string();
                        let result = if r.starts_with("CONTRADICTS:") {
                            ContradictionResult::Contradicts {
                                explanation: r
                                    .strip_prefix("CONTRADICTS:")
                                    .unwrap_or("")
                                    .trim()
                                    .to_string(),
                            }
                        } else if r.starts_with("SUPERSEDES:") {
                            ContradictionResult::Supersedes {
                                merged_content: r
                                    .strip_prefix("SUPERSEDES:")
                                    .unwrap_or("")
                                    .trim()
                                    .to_string(),
                            }
                        } else {
                            ContradictionResult::Consistent
                        };

                        match result {
                            ContradictionResult::Consistent => {
                                resolve_proposal(
                                    db,
                                    &proposal.id,
                                    ResolveStatus::Dismissed,
                                    "daemon",
                                )
                                .await?;
                            }
                            ContradictionResult::Contradicts { explanation } => {
                                log::info!("[refinery] contradiction detected: {}", explanation);
                                resolve_proposal(
                                    db,
                                    &proposal.id,
                                    ResolveStatus::AwaitingReview,
                                    "daemon",
                                )
                                .await?;
                            }
                            ContradictionResult::Supersedes { merged_content } => {
                                let tier = db.get_highest_tier(&proposal.source_ids).await?;
                                apply_merge_by_tier(
                                    db,
                                    &proposal.source_ids,
                                    &merged_content,
                                    &proposal.id,
                                    &tier,
                                )
                                .await?;
                            }
                        }
                        processed += 1;
                    }
                }
            }
            "suggest_entity" => {
                // Entity suggestion: payload contains the suggested entity name.
                // Mark as awaiting_review so the UI can surface it for approval.
                resolve_proposal(db, &proposal.id, ResolveStatus::AwaitingReview, "daemon").await?;
                log::info!(
                    "[refinery] entity suggestion queued for review: {:?}",
                    proposal.payload
                );
                processed += 1;
            }
            "entity_merge" | "relation_conflict" => {
                // Producer wrote pending; surface for human review (Spec A list endpoint
                // filters by awaiting_review status). Accept dispatch is agent-triggered,
                // not daemon-auto.
                //
                // Guard on status=='pending' to avoid spamming refinement_resolve activity
                // rows every tick on already-promoted proposals: get_pending_refinements
                // returns proposals with status NOT IN (terminal), so awaiting_review
                // proposals would re-match and re-log without this guard.
                if proposal.status == "pending" {
                    resolve_proposal(db, &proposal.id, ResolveStatus::AwaitingReview, "daemon")
                        .await?;
                }
                processed += 1;
            }
            _ => {
                log::debug!("[refinery] unknown action: {}", proposal.action);
            }
        }
    }
    Ok(processed)
}

/// Outcome of `resolve_dual_pool` — what the dual-pool resolution did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ResolveOutcome {
    /// source_ids of existing memories soft-suppressed (incoming won).
    pub invalidated: Vec<String>,
    /// True when the INCOMING memory was itself expired (existing had a
    /// strictly-later valid-time — out-of-order backfill).
    pub expired_incoming: bool,
    /// source_ids of existing memories flagged for human review instead of
    /// auto-mutated (because one side was protected).
    pub flagged_for_review: Vec<String>,
    /// refinement proposal ids filed for near-duplicate consolidation.
    pub dedup_proposals: Vec<String>,
}

/// Minimum count of new content tokens an incoming memory must add over an
/// existing near-duplicate before the re-capture earns a human merge card.
/// Below this the two read as ~identical and dedup silently.
const MIN_NEW_CONTENT_TOKENS: usize = 4;

/// Content tokens of `s`: lowercased alphanumeric words of length >= 4. Short
/// function words ("now", "the", "use") are dropped so a trivial restatement
/// reads as ~identical to the original.
// ponytail: lexical token-overlap, not semantics — a faithful paraphrase that
// reuses no surface words reads as "richer", a keyword-stuffed restatement reads
// as "identical". Upgrade path: the resolve LLM already sees both texts; have it
// emit a `richer:bool` alongside `duplicates[]` once a judge variant is wired.
fn content_tokens(s: &str) -> std::collections::HashSet<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 4)
        .map(|w| w.to_string())
        .collect()
}

/// True when `incoming` adds at least `MIN_NEW_CONTENT_TOKENS` content tokens
/// absent from `existing` — the re-capture carries materially new information,
/// not a restatement. Gates whether a near-duplicate earns a consolidation card.
fn is_materially_richer(incoming: &str, existing: &str) -> bool {
    let existing_tokens = content_tokens(existing);
    content_tokens(incoming)
        .difference(&existing_tokens)
        .count()
        >= MIN_NEW_CONTENT_TOKENS
}

/// T14 — resolve an incoming memory against two candidate pools in ONE LLM call.
///
/// Pool A = near-duplicates (vector ~0.88). Pool B = same-entity/domain,
/// possibly-contradicting. The single LLM call returns
/// `{"duplicates":[...],"invalidates":[...]}`. We act on the result:
///   - invalidate (Pool B only) -> soft-suppress the older side (bidirectional
///     temporal expiry; newer valid-time wins). Protected side -> flag
///     `pending_revision` instead of auto-mutating (rule 2).
///   - duplicate -> file a consolidation refinement proposal (do NOT collapse).
///
/// SAFETY: never deletes (soft-suppress only); silent-zero parse guard via
/// `parse_dual_pool`; protected rows never auto-mutated. Best-effort and
/// timeout-bounded. No-op (returns default) when the flag is off, no LLM is
/// wired, the incoming row is missing, or both pools are empty.
pub async fn resolve_dual_pool(
    db: &MemoryDB,
    incoming_source_id: &str,
    llm: Option<&Arc<dyn LlmProvider>>,
    prompts: &PromptRegistry,
    emitter: &Arc<dyn crate::events::EventEmitter>,
) -> Result<ResolveOutcome, WenlanError> {
    use crate::retrieval::resolve::{
        expiry_direction, index_to_pool, parse_dual_pool, pool_b_offset, valid_time, Candidate,
        ExpiryDirection, IncomingMemory, Pool, ResolutionConfig,
    };

    let mut outcome = ResolveOutcome::default();

    // Master switch + LLM presence: byte-identical no-op when off.
    if !crate::db::dual_pool_resolve_enabled() {
        return Ok(outcome);
    }
    let Some(llm) = llm else {
        return Ok(outcome);
    };

    // Fetch the incoming memory's metadata; missing row -> no-op.
    let Some(incoming): Option<IncomingMemory> =
        db.get_incoming_for_resolution(incoming_source_id).await?
    else {
        return Ok(outcome);
    };

    let cfg = ResolutionConfig::default();
    let (pool_a, pool_b) = db.build_resolution_pools(&incoming, cfg).await?;
    let a_len = pool_a.len();
    let b_len = pool_b.len();
    if a_len == 0 && b_len == 0 {
        return Ok(outcome);
    }
    let total_len = a_len + b_len;

    // Build the continuous-index prompt: Pool A first, then a divider, then Pool B.
    let user_prompt = build_resolution_prompt(&incoming.content, &pool_a, &pool_b);

    // ONE LLM call, timeout-bounded (mirrors page-channel/enrichment discipline).
    let raw = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        llm.generate(LlmRequest {
            system_prompt: Some(prompts.resolve_dual_pool.clone()),
            user_prompt,
            max_tokens: 256,
            temperature: 0.1,
            label: Some("resolve_dual_pool".to_string()),
            timeout_secs: None,
        }),
    )
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            log::warn!("[resolve_dual_pool] llm error: {e}");
            return Ok(outcome);
        }
        Err(_) => {
            log::warn!("[resolve_dual_pool] llm timed out after 10s");
            return Ok(outcome);
        }
    };

    // Defensive parse (silent-zero guard): empty decision on any failure.
    let decision = parse_dual_pool(&raw, total_len);

    let inc_valid = valid_time(incoming.event_date, incoming.created_at);

    // --- Invalidations: act ONLY on Pool B (Pool A indices are duplicates,
    //     never contradictions, so ignore them in `invalidates`). ---
    //
    // Two-phase + ORDER-INDEPENDENT. An incoming memory can conflict with two
    // existings in OPPOSITE valid-time directions (one older, one newer). If
    // ANY conflicting existing has a strictly-later valid-time, the incoming is
    // the LOSER and must be expired — and it must NOT then go on to soft-suppress
    // the older existing(s) (it didn't win, it lost). Iteration order over Pool B
    // is `last_modified DESC`, unrelated to event_date, so a single in-loop guard
    // flag would be order-sensitive (B-then-A would suppress before expiring).
    // Phase 1 classifies every invalidated target; Phase 2 acts on the verdict.
    let mut protected_targets: Vec<&Candidate> = Vec::new();
    let mut supersede_targets: Vec<&Candidate> = Vec::new();
    let mut expiring_targets: Vec<&Candidate> = Vec::new();
    for idx in &decision.invalidates {
        if index_to_pool(*idx, a_len, b_len) != Some(Pool::B) {
            continue;
        }
        let Some(offset) = pool_b_offset(*idx, a_len, b_len) else {
            continue;
        };
        let existing: &Candidate = &pool_b[offset];
        // Rule 2: protected memories are NEVER auto-mutated.
        if existing.is_protected() {
            protected_targets.push(existing);
            continue;
        }
        let ex_valid = valid_time(existing.event_date, existing.created_at);
        match expiry_direction(inc_valid, ex_valid) {
            ExpiryDirection::IncomingExpired => expiring_targets.push(existing),
            ExpiryDirection::ExistingSuperseded => supersede_targets.push(existing),
        }
    }

    // Protected collisions: flag the incoming for human review, link to each
    // protected existing so /brief can surface it. Never auto-mutate either side.
    for existing in &protected_targets {
        if let Err(e) = db.flag_memory_for_revision(incoming_source_id).await {
            log::warn!(
                "[resolve_dual_pool] protected-collision review-flag failed for \
                 {incoming_source_id}: {e}; skipping link"
            );
            continue;
        }
        let _ = db
            .resolve_link_pending_revision(incoming_source_id, &existing.source_id)
            .await;
        outcome.flagged_for_review.push(existing.source_id.clone());
        emit_resolution_event(
            emitter,
            "flagged_for_review",
            incoming_source_id,
            &existing.source_id,
        );
    }

    if expiring_targets.is_empty() {
        // Incoming wins (or ties) against every conflicting existing -> soft-suppress
        // each older existing. Incoming stays confirmed.
        for existing in &supersede_targets {
            if let Err(e) = db
                .resolve_supersede_existing(incoming_source_id, &existing.source_id)
                .await
            {
                log::warn!("[resolve_dual_pool] supersede_existing failed: {e}");
                continue;
            }
            outcome.invalidated.push(existing.source_id.clone());
            emit_resolution_event(
                emitter,
                "existing_superseded",
                incoming_source_id,
                &existing.source_id,
            );
        }
    } else {
        // At least one existing has a strictly-later valid-time: the incoming is
        // the loser. Expire the incoming ONCE (link it to the newest such
        // existing) and suppress NOTHING else. The `supersede_targets` (older
        // existings) are left untouched and stay confirmed=1 — the incoming did
        // not win against them, so it cannot suppress them.
        let survivor = expiring_targets
            .iter()
            .max_by_key(|c| valid_time(c.event_date, c.created_at))
            .expect("expiring_targets is non-empty");
        if let Err(e) = db
            .resolve_expire_incoming(incoming_source_id, &survivor.source_id)
            .await
        {
            log::warn!("[resolve_dual_pool] expire_incoming failed: {e}");
        } else {
            outcome.expired_incoming = true;
            emit_resolution_event(
                emitter,
                "incoming_expired",
                incoming_source_id,
                &survivor.source_id,
            );
        }
    }

    // --- Duplicates: file a consolidation proposal, do NOT collapse. ---
    for idx in decision.duplicates {
        // Resolve the candidate's source_id + content from whichever pool the index hits.
        let dup = match index_to_pool(idx, a_len, b_len) {
            Some(Pool::A) => pool_a
                .get(idx)
                .map(|c| (c.source_id.clone(), c.content.clone())),
            Some(Pool::B) => pool_b_offset(idx, a_len, b_len)
                .and_then(|o| pool_b.get(o))
                .map(|c| (c.source_id.clone(), c.content.clone())),
            None => None,
        };
        let Some((dup_id, dup_content)) = dup else {
            continue;
        };
        // Narrow-the-gate (v1): only a materially-richer re-capture earns a human
        // merge card; a ~identical restatement dedups silently (no revision row).
        if !is_materially_richer(&incoming.content, &dup_content) {
            continue;
        }
        let prop_id = format!("ref_dual_{}", uuid::Uuid::new_v4().simple());
        if let Err(e) = db
            .insert_refinement_proposal(
                &prop_id,
                "consolidate_duplicate",
                &[incoming_source_id.to_string(), dup_id.clone()],
                None,
                0.85,
            )
            .await
        {
            log::warn!("[resolve_dual_pool] insert dedup proposal failed: {e}");
            continue;
        }
        outcome.dedup_proposals.push(prop_id);
    }

    Ok(outcome)
}

/// Render the numbered candidate list for the resolve prompt. Pool A is listed
/// first (indices `0..a_len`), then a divider, then Pool B
/// (`a_len..a_len+b_len`). Continuous numbering matches `index_to_pool`.
fn build_resolution_prompt(
    incoming_content: &str,
    pool_a: &[crate::retrieval::resolve::Candidate],
    pool_b: &[crate::retrieval::resolve::Candidate],
) -> String {
    let mut s = String::new();
    s.push_str("Incoming memory:\n");
    s.push_str(incoming_content);
    s.push_str("\n\nCandidates:\n");
    s.push_str(&format!(
        "--- DUPLICATES range (indices 0..{}): near-duplicate restatements ---\n",
        pool_a.len()
    ));
    for (i, c) in pool_a.iter().enumerate() {
        s.push_str(&format!("[{i}] {}\n", c.content));
    }
    s.push_str(&format!(
        "--- CONFLICTS range (indices {}..{}): same topic, possibly-contradicting ---\n",
        pool_a.len(),
        pool_a.len() + pool_b.len()
    ));
    for (j, c) in pool_b.iter().enumerate() {
        let idx = pool_a.len() + j;
        s.push_str(&format!("[{idx}] {}\n", c.content));
    }
    s
}

/// Emit one EventEmitter receipt per resolution mutation so /brief can surface
/// it for human review. Best-effort; emit failures are swallowed.
fn emit_resolution_event(
    emitter: &Arc<dyn crate::events::EventEmitter>,
    direction: &str,
    incoming_id: &str,
    existing_id: &str,
) {
    let payload = serde_json::json!({
        "kind": "dual_pool_resolution",
        "direction": direction,
        "incoming": incoming_id,
        "existing": existing_id,
    })
    .to_string();
    let _ = emitter.emit("memory_resolution", &payload);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::MemoryDB;
    use crate::events::NoopEmitter;
    use std::sync::Arc;
    use tempfile::TempDir;

    async fn test_db() -> (MemoryDB, TempDir) {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test.db");
        let db = MemoryDB::new(&db_path, Arc::new(NoopEmitter))
            .await
            .unwrap();
        (db, tmp)
    }

    async fn insert_page_source_memory(db: &MemoryDB, source_id: &str, content: &str, space: &str) {
        let conn = db.test_primary_session().await;
        conn.execute(
            "INSERT INTO memories (
                id, content, source, source_id, title, chunk_index, last_modified,
                chunk_type, memory_type, space, source_agent, confidence,
                confirmed, word_count, enrichment_status, quality, is_recap,
                supersede_mode, stability
             )
             VALUES (
                ?1, ?2, 'memory', ?1, ?3, 0, ?4, 'document', 'fact',
                ?5, 'codex-test', 0.9, 1, 8, 'enriched', 'high', 0,
                'hide', 'learned'
             )",
            libsql::params![
                source_id,
                content,
                content.chars().take(40).collect::<String>(),
                chrono::Utc::now().timestamp(),
                space,
            ],
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn apply_refinement_vocab_promote_adds_canonical() {
        let (db, _tmp) = test_db().await;
        // A dirty relation type exists in the queue as a promote candidate.
        db.insert_vocab_promote_proposal("relation", "design_inspiration", None)
            .await
            .unwrap();
        let id = crate::db::MemoryDB::vocab_proposal_fingerprint("relation", "design_inspiration");
        apply_refinement(&db, &id, "test").await.unwrap();
        // It is now canonical: resolve returns itself.
        assert_eq!(
            db.resolve_relation_type("design_inspiration")
                .await
                .unwrap(),
            Some("design_inspiration".into())
        );
    }

    // Appendix E miss: an entity whose extracted type was unknown is stored
    // as `concept` and its id recorded on the vocab proposal; accepting the
    // proposal must re-type that entity onto the new canonical.
    #[tokio::test]
    async fn apply_refinement_vocab_promote_retypes_proposing_entities() {
        let (db, _tmp) = test_db().await;
        // "widget" is not a canonical, alias, or safe transform of one.
        let (id, created) = db
            .resolve_or_create_entity("Sprocket Widget", "widget", None, Some("test"), None)
            .await
            .unwrap();
        assert!(created);
        let stored = db.get_entity_name_type(&id).await.unwrap().unwrap();
        assert_eq!(stored.1, "concept", "unknown type is stored as the default");

        let proposal_id = crate::db::MemoryDB::vocab_proposal_fingerprint("entity", "widget");
        let prop = db
            .get_refinement_proposal(&proposal_id)
            .await
            .unwrap()
            .expect("vocab_promote proposal was queued");
        assert_eq!(
            prop.source_ids,
            vec![id.clone()],
            "the created entity is recorded on the proposal"
        );

        apply_refinement(&db, &proposal_id, "test").await.unwrap();
        assert_eq!(
            db.resolve_entity_type("widget").await.unwrap(),
            Some("widget".into())
        );
        let retyped = db.get_entity_name_type(&id).await.unwrap().unwrap();
        assert_eq!(
            retyped.1, "widget",
            "accepting the vocabulary re-types the entity that proposed it"
        );
        // Ledgered like fold_entity_type so the re-type is reversible.
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM vocab_heal_ledger WHERE kind='entity' AND old_value='concept' AND new_value='widget'",
                (),
            )
            .await
            .unwrap();
        let n: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(n, 1);
    }

    #[tokio::test]
    async fn resolve_proposal_dismissed_updates_status() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_test_1",
            "detect_contradiction",
            &["mem_a".to_string(), "mem_b".to_string()],
            None,
            0.8,
        )
        .await
        .unwrap();

        let result = resolve_proposal(&db, "ref_test_1", ResolveStatus::Dismissed, "test-agent")
            .await
            .unwrap();
        assert_eq!(result.id, "ref_test_1");
        assert!(result.warnings.is_empty());

        let pending = db.get_pending_refinements().await.unwrap();
        assert!(
            !pending.iter().any(|p| p.id == "ref_test_1"),
            "dismissed proposal should not appear in pending list"
        );
    }

    #[tokio::test]
    async fn resolve_proposal_not_found_returns_not_found_error() {
        let (db, _tmp) = test_db().await;
        let err = resolve_proposal(
            &db,
            "ref_nonexistent",
            ResolveStatus::Dismissed,
            "test-agent",
        )
        .await
        .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
    }

    #[tokio::test]
    async fn resolve_proposal_already_terminal_returns_validation_error() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_test_2",
            "entity_merge",
            &["e1".to_string(), "e2".to_string()],
            None,
            0.87,
        )
        .await
        .unwrap();
        resolve_proposal(&db, "ref_test_2", ResolveStatus::Dismissed, "test-agent")
            .await
            .unwrap();

        let err = resolve_proposal(&db, "ref_test_2", ResolveStatus::Dismissed, "test-agent")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::Validation(_)),
            "expected Validation, got {err:?}"
        );
    }

    #[tokio::test]
    async fn daemon_dismiss_logs_activity_via_convergence() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_daemon_1",
            "dedup_merge",
            &["a".to_string(), "b".to_string()],
            None,
            0.5,
        )
        .await
        .unwrap();

        resolve_proposal(&db, "ref_daemon_1", ResolveStatus::Dismissed, "daemon")
            .await
            .unwrap();

        let acts = db
            .list_agent_activity(10, Some("daemon"), None)
            .await
            .unwrap_or_default();
        assert!(
            acts.iter().any(|a| a.action == "refinement_resolve"),
            "daemon path should log refinement_resolve via convergence"
        );
    }

    #[tokio::test]
    async fn resolve_proposal_logs_activity() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_test_3",
            "suggest_entity",
            &["mem_x".to_string()],
            Some("{\"name\":\"Acme\"}"),
            0.9,
        )
        .await
        .unwrap();

        resolve_proposal(&db, "ref_test_3", ResolveStatus::Dismissed, "test-agent")
            .await
            .unwrap();

        let acts = db
            .list_agent_activity(10, Some("test-agent"), None)
            .await
            .unwrap_or_default();
        assert!(
            acts.iter().any(|a| a.action == "refinement_resolve"),
            "expected refinement_resolve activity row, found: {acts:?}"
        );
    }

    #[tokio::test]
    async fn resolve_proposal_concurrent_rejects_logs_once() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_race_1",
            "entity_merge",
            &["a".into(), "b".into()],
            None,
            0.85,
        )
        .await
        .unwrap();

        let db_arc = std::sync::Arc::new(db);
        let h1 = {
            let d = db_arc.clone();
            tokio::spawn(async move {
                resolve_proposal(&d, "ref_race_1", ResolveStatus::Dismissed, "client-a").await
            })
        };
        let h2 = {
            let d = db_arc.clone();
            tokio::spawn(async move {
                resolve_proposal(&d, "ref_race_1", ResolveStatus::Dismissed, "client-b").await
            })
        };
        let r1 = h1.await.unwrap();
        let r2 = h2.await.unwrap();

        let ok_count = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
        let err_count = [&r1, &r2].iter().filter(|r| r.is_err()).count();
        assert_eq!(
            ok_count, 1,
            "exactly one resolve should succeed, got {r1:?} / {r2:?}"
        );
        assert_eq!(err_count, 1, "exactly one should fail with Validation");
        let err = if let Err(e) = r1 { e } else { r2.unwrap_err() };
        assert!(
            matches!(err, crate::error::WenlanError::Validation(_)),
            "concurrent loser should see Validation, got {err:?}"
        );

        let acts = db_arc
            .list_agent_activity(10, None, None)
            .await
            .unwrap_or_default();
        let resolve_count = acts
            .iter()
            .filter(|a| a.action == "refinement_resolve")
            .count();
        assert_eq!(
            resolve_count, 1,
            "activity log should record exactly one resolve, got {resolve_count}"
        );
    }

    // ==================== apply_refinement ====================

    async fn seed_entity_merge_proposal(db: &MemoryDB, id: &str) -> (String, String) {
        let new_ent = db
            .create_entity("Acme Corporation", "organization", None)
            .await
            .unwrap();
        let existing_ent = db
            .create_entity("Acme Corp", "organization", None)
            .await
            .unwrap();
        db.insert_refinement_proposal(
            id,
            "entity_merge",
            &[new_ent.clone(), existing_ent.clone()],
            None,
            0.87,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE refinement_queue SET status = 'awaiting_review' WHERE id = ?1",
                libsql::params![id],
            )
            .await
            .unwrap();
        }
        (new_ent, existing_ent)
    }

    #[tokio::test]
    async fn apply_refinement_entity_merge_default_existing_wins() {
        let (db, _tmp) = test_db().await;
        let (new_ent, existing_ent) = seed_entity_merge_proposal(&db, "ref_em_1").await;

        let outcome = apply_refinement(&db, "ref_em_1", "test-agent")
            .await
            .unwrap();
        assert_eq!(outcome.action_applied, "entity_merge");

        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_page_map WHERE entity_id = ?1",
                libsql::params![existing_ent],
            )
            .await
            .unwrap();
        let count: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(count, 1, "existing entity should remain canonical");

        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM entity_page_map WHERE entity_id = ?1",
                libsql::params![new_ent],
            )
            .await
            .unwrap();
        let count: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(count, 0, "new entity should be merged away");
    }

    #[tokio::test]
    async fn apply_refinement_relation_conflict_default_new_wins() {
        let (db, _tmp) = test_db().await;
        let from = db.create_entity("Alice", "person", None).await.unwrap();
        let to = db
            .create_entity("Acme", "organization", None)
            .await
            .unwrap();
        let existing_rel = db
            .create_relation(&from, &to, "works_at", None, Some(0.5), None, None)
            .await
            .unwrap();
        let new_rel = db
            .create_relation(&from, &to, "leads", None, Some(0.9), None, None)
            .await
            .unwrap();
        db.insert_refinement_proposal(
            "ref_rc_1",
            "relation_conflict",
            &[new_rel.clone(), existing_rel.clone()],
            None,
            0.7,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE refinement_queue SET status = 'awaiting_review' WHERE id = ?1",
                libsql::params!["ref_rc_1"],
            )
            .await
            .unwrap();
        }

        let outcome = apply_refinement(&db, "ref_rc_1", "test-agent")
            .await
            .unwrap();
        assert_eq!(outcome.action_applied, "relation_conflict");

        // G6 Stage 2 PR 2b: `relations` is frozen -- `supersede_relation` is
        // keyed on `edges.edge_id` now (item 3 compat note), and it
        // retracts (not deletes) the loser edge, so assert `valid_until`
        // instead of a `relations` row count.
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT valid_until FROM edges WHERE edge_id = ?1",
                libsql::params![new_rel],
            )
            .await
            .unwrap();
        let valid_until: Option<i64> = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert!(
            valid_until.is_none(),
            "new relation (winner) should remain active"
        );

        let mut rows = conn
            .query(
                "SELECT valid_until FROM edges WHERE edge_id = ?1",
                libsql::params![existing_rel],
            )
            .await
            .unwrap();
        let valid_until: Option<i64> = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert!(
            valid_until.is_some(),
            "existing relation (loser) should be retracted"
        );
    }

    #[tokio::test]
    async fn apply_refinement_detect_contradiction_does_not_quarantine() {
        let (db, _tmp) = test_db().await;
        let new_mem = format!("mem_{}", uuid::Uuid::new_v4().simple());
        let existing_mem = format!("mem_{}", uuid::Uuid::new_v4().simple());
        {
            let conn = db.test_primary_session().await;
            for sid in &[new_mem.clone(), existing_mem.clone()] {
                conn.execute(
                    "INSERT INTO memories (id, content, source, source_id, title, chunk_index, \
                                            chunk_type, source_agent, space, confidence, confirmed, \
                                            last_modified, memory_type, pending_revision) \
                     VALUES (?1, ?2, 'memory', ?3, 'test', 0, 'text', NULL, 'general', 1.0, 0, 1712707200, 'fact', 0)",
                    libsql::params![sid.clone(), "x".to_string(), sid.clone()],
                )
                .await
                .unwrap();
            }
        }
        db.insert_refinement_proposal(
            "ref_dc_1",
            "detect_contradiction",
            &[new_mem.clone(), existing_mem.clone()],
            None,
            0.8,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE refinement_queue SET status = 'awaiting_review' WHERE id = ?1",
                libsql::params!["ref_dc_1"],
            )
            .await
            .unwrap();
        }

        let outcome = apply_refinement(&db, "ref_dc_1", "test-agent")
            .await
            .unwrap();
        assert_eq!(outcome.action_applied, "detect_contradiction");

        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT pending_revision FROM memories WHERE source_id = ?1",
                libsql::params![existing_mem],
            )
            .await
            .unwrap();
        let f: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(
            f, 0,
            "existing memory must NOT be quarantined: flagging pending_revision \
             without a supersedes link hides it from /curate and all retrieval \
             with no way to clear (the contradiction-accept quarantine bug)"
        );

        let mut rows = conn
            .query(
                "SELECT pending_revision FROM memories WHERE source_id = ?1",
                libsql::params![new_mem],
            )
            .await
            .unwrap();
        let f: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(f, 0, "new memory should NOT be flagged either");
    }

    #[tokio::test]
    async fn apply_refinement_page_merge_unions_evidence_archives_absorbed_and_repoints_links() {
        let (db, _tmp) = test_db().await;
        let now = chrono::Utc::now().to_rfc3339();

        db.insert_page(
            "page_survivor",
            "Survivor Topic",
            None,
            "Survivor prose.",
            None,
            None,
            &["mem_keep"],
            &now,
        )
        .await
        .unwrap();
        db.insert_page(
            "page_absorbed",
            "Absorbed Topic",
            None,
            "Absorbed prose.",
            None,
            None,
            &["mem_absorbed", "mem_extra"],
            &now,
        )
        .await
        .unwrap();
        db.insert_page(
            "page_referrer",
            "Referrer",
            None,
            "See [[Absorbed Topic]] for the current notes.",
            None,
            None,
            &[],
            &now,
        )
        .await
        .unwrap();
        db.set_page_citations_for_test("page_survivor", "[]")
            .await
            .unwrap();
        db.insert_refinement_proposal(
            "ref_page_merge_1",
            "page_merge",
            &["page_survivor".to_string(), "page_absorbed".to_string()],
            Some(
                r#"{"action":"page_merge","left_page_id":"page_survivor","right_page_id":"page_absorbed","source_overlap":1,"source_overlap_ratio":0.5}"#,
            ),
            0.95,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE refinement_queue SET status = 'awaiting_review' WHERE id = ?1",
                libsql::params!["ref_page_merge_1"],
            )
            .await
            .unwrap();
        }

        let outcome = apply_refinement(&db, "ref_page_merge_1", "test-agent")
            .await
            .unwrap();

        assert_eq!(outcome.action_applied, "page_merge");
        let survivor = db
            .get_page("page_survivor")
            .await
            .unwrap()
            .expect("survivor page remains active");
        assert_eq!(survivor.status, "active");
        assert_eq!(survivor.version, 2, "merge must advance the survivor");
        assert_eq!(survivor.stale_reason.as_deref(), Some("source_updated"));
        assert!(
            ["mem_keep", "mem_absorbed", "mem_extra"]
                .iter()
                .all(|sid| survivor.source_memory_ids.iter().any(|s| s == sid)),
            "survivor must carry the unioned source evidence, got {:?}",
            survivor.source_memory_ids
        );

        let absorbed = db
            .get_page("page_absorbed")
            .await
            .unwrap()
            .expect("absorbed page is archived, not deleted");
        assert_eq!(absorbed.status, "archived");
        assert_eq!(
            absorbed.version, 2,
            "archive must advance the absorbed Page"
        );
        {
            let conn = db.test_primary_session().await;
            let mut rows = conn
                .query("SELECT citations FROM pages WHERE id = 'page_survivor'", ())
                .await
                .unwrap();
            assert_eq!(
                rows.next()
                    .await
                    .unwrap()
                    .unwrap()
                    .get::<Option<String>>(0)
                    .unwrap(),
                None,
                "merged evidence must reopen citation annotation"
            );
        }

        let survivor_sources = db.get_page_sources("page_survivor").await.unwrap();
        assert!(
            ["mem_keep", "mem_absorbed", "mem_extra"].iter().all(|sid| {
                survivor_sources
                    .iter()
                    .any(|source| source.memory_source_id == *sid)
            }),
            "page_sources must carry the unioned evidence, got {:?}",
            survivor_sources
        );

        let survivor_links = db
            .get_page_inbound_links_scoped("page_survivor", &crate::read_scope::ReadScope::Global)
            .await
            .unwrap();
        assert!(
            survivor_links
                .iter()
                .any(|(source_page_id, label)| source_page_id == "page_referrer"
                    && label == "Absorbed Topic"),
            "wikilinks to the absorbed title must point at the survivor now, got {:?}",
            survivor_links
        );
        let absorbed_links = db
            .get_page_inbound_links_scoped("page_absorbed", &crate::read_scope::ReadScope::Global)
            .await
            .unwrap();
        assert!(
            !absorbed_links
                .iter()
                .any(|(source_page_id, _)| source_page_id == "page_referrer"),
            "absorbed page must not retain backlinks after merge, got {:?}",
            absorbed_links
        );
    }

    #[tokio::test]
    async fn apply_refinement_cross_space_discovery_pick_space_creates_page_in_selected_space() {
        let (db, _tmp) = test_db().await;
        for (source_id, space) in [
            ("disc_pick_work_a", "work"),
            ("disc_pick_personal_a", "personal"),
            ("disc_pick_work_b", "work"),
        ] {
            insert_page_source_memory(
                &db,
                source_id,
                "Incremental Rust compilation cache tuning for shared developer machines.",
                space,
            )
            .await;
        }
        let source_ids = vec![
            "disc_pick_work_a".to_string(),
            "disc_pick_personal_a".to_string(),
            "disc_pick_work_b".to_string(),
        ];
        db.insert_refinement_proposal(
            "ref_cross_space_discovery_1",
            "cross_space_discovery",
            &source_ids,
            Some(
                r#"{"action":"cross_space_discovery","memory_count":3,"spaces":["personal","work"],"allowed_actions":["dismiss","pick_space"]}"#,
            ),
            1.0,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE refinement_queue SET status = 'awaiting_review' WHERE id = ?1",
                libsql::params!["ref_cross_space_discovery_1"],
            )
            .await
            .unwrap();
        }

        let outcome = apply_refinement_with_decision(
            &db,
            "ref_cross_space_discovery_1",
            "test-agent",
            RefinementDecision::PickSpace {
                space: "work".to_string(),
                notes: Some("Prefer the engineering workspace.".to_string()),
            },
        )
        .await
        .unwrap();

        assert_eq!(outcome.action_applied, "cross_space_discovery");
        let pages = db.list_pages("active", 10, 0).await.unwrap();
        assert_eq!(pages.len(), 1, "pick_space should create exactly one page");
        let page = &pages[0];
        assert_eq!(page.workspace.as_deref(), Some("work"));
        assert_eq!(page.space.as_deref(), Some("work"));
        assert_eq!(page.creation_kind, "distilled");
        assert_eq!(page.review_status, "unconfirmed");
        assert!(
            source_ids
                .iter()
                .all(|sid| page.source_memory_ids.iter().any(|source| source == sid)),
            "PageWrite should persist every discovery source, got {:?}",
            page.source_memory_ids
        );
        let resolved = db
            .get_refinement_proposal("ref_cross_space_discovery_1")
            .await
            .unwrap()
            .expect("proposal remains after resolution");
        assert_eq!(resolved.status, "resolved");
    }

    #[tokio::test]
    async fn apply_refinement_page_keep_or_archive_accept_archives_page() {
        let (db, _tmp) = test_db().await;
        insert_page_source_memory(&db, "stub_source", "Small source.", "work").await;
        let now = chrono::Utc::now().to_rfc3339();
        db.insert_page_with_kind(
            "stub_page",
            "Stub Page",
            None,
            "Thin machine-owned page.",
            None,
            None,
            &["stub_source"],
            &now,
            "distilled",
            "confirmed",
            Some("work"),
            Some("[]"),
        )
        .await
        .unwrap();
        db.insert_refinement_proposal(
            "ref_keep_or_archive_1",
            "page_keep_or_archive",
            &["stub_page".to_string()],
            Some(
                r#"{"action":"page_keep_or_archive","page_id":"stub_page","source_count":1,"allowed_actions":["dismiss","accept"]}"#,
            ),
            1.0,
        )
        .await
        .unwrap();
        db.resolve_refinement_if_open("ref_keep_or_archive_1", "awaiting_review")
            .await
            .unwrap();

        let outcome = apply_refinement(&db, "ref_keep_or_archive_1", "test-agent")
            .await
            .unwrap();

        assert_eq!(outcome.action_applied, "page_keep_or_archive");
        let page = db
            .get_page("stub_page")
            .await
            .unwrap()
            .expect("page remains after archive");
        assert_eq!(page.status, "archived");
        let resolved = db
            .get_refinement_proposal("ref_keep_or_archive_1")
            .await
            .unwrap()
            .expect("proposal remains after resolution");
        assert_eq!(resolved.status, "resolved");
    }

    #[tokio::test]
    async fn apply_refinement_suggest_entity_returns_422() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_se_1",
            "suggest_entity",
            &["x".into()],
            Some("{\"name\":\"Acme\"}"),
            0.9,
        )
        .await
        .unwrap();
        let err = apply_refinement(&db, "ref_se_1", "test-agent")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::Validation(_)),
            "expected Validation for suggest_entity, got {err:?}"
        );
    }

    #[tokio::test]
    async fn apply_refinement_dedup_merge_returns_422() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_dm_1",
            "dedup_merge",
            &["a".into(), "b".into()],
            None,
            0.95,
        )
        .await
        .unwrap();
        let err = apply_refinement(&db, "ref_dm_1", "test-agent")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::Validation(_)),
            "expected Validation for dedup_merge, got {err:?}"
        );
    }

    #[tokio::test]
    async fn apply_lint_repair_review_requires_choice_specific_repair_flow() {
        let (db, _tmp) = test_db().await;
        let digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let id = format!("lint_review_{digest}");
        let payload = serde_json::json!({
            "action": "lint_repair_review",
            "check_id": "pages.links.orphan_labels",
            "occurrence_digest": digest,
            "owner_binding_digest": crate::repair::lint_review_owner_binding_digest(
                &wenlan_types::repair::RepairDigest::parse(digest).unwrap(),
                &["page-a".to_string()],
            )
            .unwrap(),
            "issue": "Review the ambiguous target.",
            "choices": ["keep", "retarget", "remove"],
            "suggested_research_queries": [],
        })
        .to_string();
        db.insert_lint_review_if_absent(&id, &["page-a".into()], &payload)
            .await
            .unwrap();

        let err = apply_refinement(&db, &id, "test-agent").await.unwrap_err();
        let crate::error::WenlanError::Validation(message) = err else {
            panic!("expected Validation");
        };
        assert_eq!(
            message,
            "action 'lint_repair_review' requires a choice-specific repair flow"
        );
        assert_eq!(
            db.get_refinement_proposal(&id)
                .await
                .unwrap()
                .unwrap()
                .status,
            "awaiting_review"
        );
    }

    #[tokio::test]
    async fn apply_refinement_unknown_action_returns_422() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal("ref_uk_1", "future_action_xyz", &["a".into()], None, 0.5)
            .await
            .unwrap();
        let err = apply_refinement(&db, "ref_uk_1", "test-agent")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::Validation(_)),
            "expected Validation for unknown action, got {err:?}"
        );
    }

    #[tokio::test]
    async fn apply_refinement_missing_id_returns_404() {
        let (db, _tmp) = test_db().await;
        let err = apply_refinement(&db, "ref_nonexistent", "test-agent")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
    }

    #[tokio::test]
    async fn apply_refinement_terminal_proposal_returns_422() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_term_1",
            "entity_merge",
            &["a".into(), "b".into()],
            None,
            0.85,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE refinement_queue SET status = 'resolved' WHERE id = ?1",
                libsql::params!["ref_term_1"],
            )
            .await
            .unwrap();
        }
        let err = apply_refinement(&db, "ref_term_1", "test-agent")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::Validation(_)),
            "expected Validation (terminal), got {err:?}"
        );
    }

    #[tokio::test]
    async fn apply_refinement_logs_both_apply_and_resolve_activity() {
        let (db, _tmp) = test_db().await;
        let _ = seed_entity_merge_proposal(&db, "ref_log_1").await;

        apply_refinement(&db, "ref_log_1", "test-agent")
            .await
            .unwrap();

        let acts = db
            .list_agent_activity(20, Some("test-agent"), None)
            .await
            .unwrap_or_default();
        let apply_count = acts
            .iter()
            .filter(|a| a.action == "refinement_apply")
            .count();
        let resolve_count = acts
            .iter()
            .filter(|a| a.action == "refinement_resolve")
            .count();
        assert_eq!(
            apply_count, 1,
            "exactly one refinement_apply row, got {apply_count}"
        );
        assert_eq!(
            resolve_count, 1,
            "exactly one refinement_resolve row, got {resolve_count}"
        );
    }

    #[tokio::test]
    async fn apply_refinement_idempotent_via_resolve_race() {
        let (db, _tmp) = test_db().await;
        let _ = seed_entity_merge_proposal(&db, "ref_race_1").await;

        apply_refinement(&db, "ref_race_1", "client-a")
            .await
            .unwrap();
        let err = apply_refinement(&db, "ref_race_1", "client-b")
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::error::WenlanError::Validation(_)),
            "second accept should 422, got {err:?}"
        );
    }

    #[tokio::test]
    async fn daemon_arm_promotes_pending_entity_merge_to_awaiting_review() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_pending_em",
            "entity_merge",
            &["a".into(), "b".into()],
            None,
            0.87,
        )
        .await
        .unwrap();
        // status defaults to 'pending'
        let tuning = crate::tuning::RefineryConfig::default();
        let prompts = crate::prompts::PromptRegistry::default();
        let processed = process_refinement_queue(&db, None, &prompts, &tuning)
            .await
            .unwrap();
        assert!(
            processed >= 1,
            "process_refinement_queue should report >=1 processed"
        );

        let prop = db
            .get_refinement_proposal("ref_pending_em")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            prop.status, "awaiting_review",
            "pending entity_merge should promote to awaiting_review"
        );
    }

    #[tokio::test]
    async fn daemon_arm_promotes_pending_relation_conflict_to_awaiting_review() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_pending_rc",
            "relation_conflict",
            &["a".into(), "b".into()],
            None,
            0.7,
        )
        .await
        .unwrap();
        let tuning = crate::tuning::RefineryConfig::default();
        let prompts = crate::prompts::PromptRegistry::default();
        let processed = process_refinement_queue(&db, None, &prompts, &tuning)
            .await
            .unwrap();
        assert!(processed >= 1);

        let prop = db
            .get_refinement_proposal("ref_pending_rc")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(prop.status, "awaiting_review");
    }

    #[tokio::test]
    async fn awaiting_review_backlog_does_not_starve_pending_proposals() {
        let (db, _tmp) = test_db().await;
        for index in 0..11 {
            let digest = format!("{:064x}", index + 1);
            let payload = serde_json::json!({
                "action": "lint_repair_review",
                "check_id": "pages.links.orphan_labels",
                "occurrence_digest": digest,
                "owner_binding_digest": crate::repair::lint_review_owner_binding_digest(
                    &wenlan_types::repair::RepairDigest::parse(&digest).unwrap(),
                    &["page-a".to_string()],
                )
                .unwrap(),
                "issue": "Review the ambiguous target.",
                "choices": ["keep", "retarget", "remove"],
                "suggested_research_queries": [],
            })
            .to_string();
            db.insert_lint_review_if_absent(
                &format!("lint_review_{digest}"),
                &["page-a".into()],
                &payload,
            )
            .await
            .unwrap();
        }
        db.test_primary_session()
            .await
            .execute(
                "UPDATE refinement_queue
                    SET created_at='2023-11-14 22:13:20'
                  WHERE action='lint_repair_review'",
                libsql::params::Params::None,
            )
            .await
            .unwrap();
        db.insert_refinement_proposal(
            "later-pending",
            "entity_merge",
            &["a".into(), "b".into()],
            None,
            0.87,
        )
        .await
        .unwrap();

        let tuning = crate::tuning::RefineryConfig::default();
        let prompts = crate::prompts::PromptRegistry::default();
        let processed = process_refinement_queue(&db, None, &prompts, &tuning)
            .await
            .unwrap();

        assert_eq!(processed, 1);
        assert_eq!(
            db.get_refinement_proposal("later-pending")
                .await
                .unwrap()
                .unwrap()
                .status,
            "awaiting_review"
        );
    }

    #[tokio::test]
    async fn daemon_arm_skips_already_awaiting_review() {
        let (db, _tmp) = test_db().await;
        db.insert_refinement_proposal(
            "ref_aw_em",
            "entity_merge",
            &["a".into(), "b".into()],
            None,
            0.87,
        )
        .await
        .unwrap();
        {
            let conn = db.test_primary_session().await;
            conn.execute(
                "UPDATE refinement_queue SET status = 'awaiting_review' WHERE id = ?1",
                libsql::params!["ref_aw_em"],
            )
            .await
            .unwrap();
        }

        let tuning = crate::tuning::RefineryConfig::default();
        let prompts = crate::prompts::PromptRegistry::default();
        let _ = process_refinement_queue(&db, None, &prompts, &tuning)
            .await
            .unwrap();

        // Verify NO new refinement_resolve activity row was logged by the daemon
        // for this already-awaiting proposal (status guard skipped resolve_proposal).
        // get_pending_refinements returns proposals with status NOT IN (terminal),
        // so awaiting_review still shows; the test asserts no spurious daemon activity.
        let acts = db
            .list_agent_activity(50, Some("daemon"), None)
            .await
            .unwrap_or_default();
        // No activity rows produced for this proposal at all (it never went through
        // resolve_proposal in this tick). Check by counting "refinement_resolve" rows.
        let count = acts
            .iter()
            .filter(|a| a.action == "refinement_resolve")
            .count();
        assert_eq!(
            count, 0,
            "no spurious resolve activity for already-awaiting proposal"
        );
    }

    // ==================== T14 dual-pool resolution ====================

    use crate::events::EventEmitter;
    use crate::llm_provider::{LlmBackend, LlmError, LlmProvider, LlmRequest};
    use crate::prompts::PromptRegistry;
    use wenlan_types::sources::RawDocument;

    /// LLM stub returning a fixed canned response (or an error).
    struct StubLlm {
        response: Result<String, ()>,
    }

    #[async_trait::async_trait]
    impl LlmProvider for StubLlm {
        async fn generate(&self, _req: LlmRequest) -> Result<String, LlmError> {
            self.response
                .clone()
                .map_err(|_| LlmError::InferenceFailed("stub".into()))
        }
        fn is_available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "stub"
        }
        fn backend(&self) -> LlmBackend {
            LlmBackend::OnDevice
        }
    }

    fn stub_llm(resp: &str) -> Arc<dyn LlmProvider> {
        Arc::new(StubLlm {
            response: Ok(resp.to_string()),
        })
    }

    /// EventEmitter test double that records every (event, payload) pair.
    struct RecordingEmitter {
        events: std::sync::Mutex<Vec<(String, String)>>,
    }
    impl RecordingEmitter {
        fn new() -> Self {
            Self {
                events: std::sync::Mutex::new(Vec::new()),
            }
        }
        fn count(&self) -> usize {
            self.events.lock().unwrap().len()
        }
    }
    impl EventEmitter for RecordingEmitter {
        fn emit(&self, event: &str, payload: &str) -> anyhow::Result<()> {
            self.events
                .lock()
                .unwrap()
                .push((event.to_string(), payload.to_string()));
            Ok(())
        }
    }

    /// Insert a confirmed `source='memory'` row through the real upsert path
    /// (so the embedder fills `embedding`), then patch fields the RawDocument
    /// path can't set directly (event_date, pinned).
    #[allow(clippy::too_many_arguments)]
    async fn seed_memory(
        db: &MemoryDB,
        source_id: &str,
        content: &str,
        memory_type: &str,
        domain: Option<&str>,
        entity_id: Option<&str>,
        structured_fields: Option<&str>,
        last_modified: i64,
        event_date: Option<i64>,
        pinned: bool,
        stability: &str,
    ) {
        let doc = RawDocument {
            source: "memory".to_string(),
            source_id: source_id.to_string(),
            title: format!("title-{source_id}"),
            content: content.to_string(),
            memory_type: Some(memory_type.to_string()),
            space: domain.map(|s| s.to_string()),
            entity_id: entity_id.map(|s| s.to_string()),
            structured_fields: structured_fields.map(|s| s.to_string()),
            confirmed: Some(true),
            stability: Some(stability.to_string()),
            last_modified,
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();
        // confirmed=1 is needed for build_resolution_pools; upsert may store the
        // caller-supplied confirmed but force it deterministically here.
        let conn = db.test_primary_session().await;
        conn.execute(
            "UPDATE memories SET confirmed = 1, pinned = ?1, event_date = ?2 \
             WHERE source_id = ?3 AND source = 'memory'",
            libsql::params![if pinned { 1_i64 } else { 0_i64 }, event_date, source_id],
        )
        .await
        .unwrap();
    }

    async fn confirmed_of(db: &MemoryDB, source_id: &str) -> i64 {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT confirmed FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                libsql::params![source_id],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        row.get::<i64>(0).unwrap()
    }

    async fn supersedes_of(db: &MemoryDB, source_id: &str) -> Option<String> {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT supersedes FROM memories WHERE source_id = ?1 AND source = 'memory' LIMIT 1",
                libsql::params![source_id],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        row.get::<Option<String>>(0).unwrap()
    }

    async fn row_count(db: &MemoryDB) -> i64 {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memories WHERE source = 'memory'",
                libsql::params![],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        row.get::<i64>(0).unwrap()
    }

    fn noop_emitter() -> Arc<dyn EventEmitter> {
        Arc::new(crate::events::NoopEmitter)
    }

    // ---- build_resolution_pools (Tests 16-20) ----

    #[tokio::test]
    async fn test_build_pools_duplicate_in_pool_a() {
        let (db, _tmp) = test_db().await;
        // Existing memory.
        seed_memory(
            &db,
            "mem_a",
            "I use the Rust programming language for backend work",
            "fact",
            Some("engineering"),
            None,
            None,
            1000,
            None,
            false,
            "confirmed",
        )
        .await;
        // Near-identical incoming memory.
        seed_memory(
            &db,
            "mem_b",
            "I use the Rust programming language for backend work daily",
            "fact",
            Some("engineering"),
            None,
            None,
            2000,
            None,
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_b")
            .await
            .unwrap()
            .unwrap();
        let (pool_a, pool_b) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        let a_ids: Vec<&str> = pool_a.iter().map(|c| c.source_id.as_str()).collect();
        assert!(
            a_ids.contains(&"mem_a"),
            "near-identical memory should be in Pool A, got A={a_ids:?} B={:?}",
            pool_b.iter().map(|c| &c.source_id).collect::<Vec<_>>(),
        );
        // It is a duplicate, NOT a field-contradiction -> excluded from Pool B.
        assert!(
            !pool_b.iter().any(|c| c.source_id == "mem_a"),
            "duplicate must not also appear in Pool B (disjoint)"
        );
    }

    #[tokio::test]
    async fn test_build_pools_empty_db_returns_empty() {
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_inc",
            "totally unique content about quantum widgets",
            "fact",
            Some("physics"),
            None,
            None,
            1000,
            None,
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_inc")
            .await
            .unwrap()
            .unwrap();
        let (pool_a, pool_b) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        assert!(pool_a.is_empty(), "no other memories -> empty Pool A");
        assert!(pool_b.is_empty(), "no other memories -> empty Pool B");
    }

    #[tokio::test]
    async fn test_build_pools_contradiction_in_pool_b() {
        let (db, _tmp) = test_db().await;
        // Existing claim.
        seed_memory(
            &db,
            "mem_old",
            "libSQL is built on top of SQLite",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses SQLite","domain":"databases"}"#),
            1000,
            Some(100),
            false,
            "confirmed",
        )
        .await;
        // Incoming contradicting claim (same domain, different claim, low sim by
        // different wording but fields_may_contradict passes).
        seed_memory(
            &db,
            "mem_new",
            "libSQL is actually based on PostgreSQL internals",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses PostgreSQL","domain":"databases"}"#),
            2000,
            Some(200),
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_new")
            .await
            .unwrap()
            .unwrap();
        let (pool_a, pool_b) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        let b_ids: Vec<&str> = pool_b.iter().map(|c| c.source_id.as_str()).collect();
        assert!(
            b_ids.contains(&"mem_old"),
            "contradicting same-domain claim should be in Pool B, got A={:?} B={:?}",
            pool_a.iter().map(|c| &c.source_id).collect::<Vec<_>>(),
            b_ids,
        );
    }

    #[tokio::test]
    async fn test_build_pools_respects_candidate_cap() {
        let (db, _tmp) = test_db().await;
        // Seed many near-duplicate memories so Pool A would overflow without the cap.
        for i in 0..20 {
            seed_memory(
                &db,
                &format!("mem_dup_{i}"),
                "I really enjoy using the Rust programming language daily",
                "fact",
                Some("engineering"),
                None,
                None,
                1000 + i as i64,
                None,
                false,
                "confirmed",
            )
            .await;
        }
        seed_memory(
            &db,
            "mem_probe",
            "I really enjoy using the Rust programming language daily too",
            "fact",
            Some("engineering"),
            None,
            None,
            5000,
            None,
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_probe")
            .await
            .unwrap()
            .unwrap();
        let cfg = crate::retrieval::resolve::ResolutionConfig::default();
        let (pool_a, pool_b) = db.build_resolution_pools(&incoming, cfg).await.unwrap();
        assert!(
            pool_a.len() + pool_b.len() <= cfg.combined_cap,
            "combined pools must respect cap {}, got {}",
            cfg.combined_cap,
            pool_a.len() + pool_b.len()
        );
    }

    // ---- resolve_dual_pool orchestrator (Tests 21-28) ----

    #[tokio::test]
    async fn test_dual_pool_disabled_by_default() {
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_x",
            "some fact",
            "fact",
            Some("d"),
            None,
            None,
            1000,
            None,
            false,
            "confirmed",
        )
        .await;
        let llm = stub_llm(r#"{"duplicates":[],"invalidates":[0]}"#);
        let emitter = Arc::new(RecordingEmitter::new());
        let dyn_em: Arc<dyn EventEmitter> = emitter.clone();
        let prompts = PromptRegistry::default();
        // Flag unset -> no-op.
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", None::<&str>)], async {
                resolve_dual_pool(&db, "mem_x", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(
            outcome,
            ResolveOutcome::default(),
            "flag off -> default no-op outcome"
        );
        assert_eq!(emitter.count(), 0, "flag off -> no events");
    }

    #[tokio::test]
    async fn test_dual_pool_llm_none_noop() {
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_x",
            "some fact",
            "fact",
            Some("d"),
            None,
            None,
            1000,
            None,
            false,
            "confirmed",
        )
        .await;
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_x", None, &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(outcome, ResolveOutcome::default(), "no LLM -> no-op");
    }

    #[tokio::test]
    async fn test_dual_pool_both_pools_empty_noop() {
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_solo",
            "unique content about narwhals",
            "fact",
            Some("zoology"),
            None,
            None,
            1000,
            None,
            false,
            "confirmed",
        )
        .await;
        let llm = stub_llm(r#"{"duplicates":[0],"invalidates":[1]}"#);
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_solo", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(
            outcome,
            ResolveOutcome::default(),
            "both pools empty -> no-op (no LLM action)"
        );
    }

    #[tokio::test]
    async fn test_apply_invalidate_existing_soft_suppressed() {
        let (db, _tmp) = test_db().await;
        // Existing older claim (event_date=100).
        seed_memory(
            &db,
            "mem_old",
            "libSQL is built on top of SQLite",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses SQLite","domain":"databases"}"#),
            1000,
            Some(100),
            false,
            "confirmed",
        )
        .await;
        // Incoming newer contradicting claim (event_date=200).
        seed_memory(
            &db,
            "mem_new",
            "libSQL is actually based on PostgreSQL internals",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses PostgreSQL","domain":"databases"}"#),
            2000,
            Some(200),
            false,
            "confirmed",
        )
        .await;
        // LLM marks Pool-B index (a_len + 0) as invalidates. Pool A is empty here
        // (claims differ enough), so existing lands at index 0.
        let incoming = db
            .get_incoming_for_resolution("mem_new")
            .await
            .unwrap()
            .unwrap();
        let (pa, pb) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        let inval_idx = pa.len(); // first Pool-B index
        assert!(!pb.is_empty(), "precondition: Pool B has the contradiction");
        let resp = format!(r#"{{"duplicates":[],"invalidates":[{inval_idx}]}}"#);
        let llm = stub_llm(&resp);
        let prompts = PromptRegistry::default();
        let emitter = Arc::new(RecordingEmitter::new());
        let dyn_em: Arc<dyn EventEmitter> = emitter.clone();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_new", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(
            outcome.invalidated,
            vec!["mem_old".to_string()],
            "existing should be invalidated"
        );
        assert_eq!(
            confirmed_of(&db, "mem_old").await,
            0,
            "existing soft-suppressed (confirmed=0)"
        );
        assert_eq!(
            confirmed_of(&db, "mem_new").await,
            1,
            "incoming stays confirmed"
        );
        // NOT deleted.
        assert_eq!(
            row_count(&db).await,
            2,
            "no row deleted -- soft-suppress only"
        );
        assert_eq!(
            supersedes_of(&db, "mem_new").await.as_deref(),
            Some("mem_old"),
            "incoming linked as revision-of"
        );
        assert_eq!(emitter.count(), 1, "one expiry event emitted");
    }

    #[tokio::test]
    async fn test_apply_invalidate_incoming_expired_when_existing_later() {
        let (db, _tmp) = test_db().await;
        // Existing has a STRICTLY-LATER valid-time (event_date=300).
        seed_memory(
            &db,
            "mem_existing",
            "libSQL is built on top of SQLite",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses SQLite","domain":"databases"}"#),
            1000,
            Some(300),
            false,
            "confirmed",
        )
        .await;
        // Incoming is an out-of-order backfill (event_date=100, older).
        seed_memory(
            &db,
            "mem_incoming",
            "libSQL is actually based on PostgreSQL internals",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses PostgreSQL","domain":"databases"}"#),
            2000,
            Some(100),
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_incoming")
            .await
            .unwrap()
            .unwrap();
        let (pa, _pb) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        let inval_idx = pa.len();
        let resp = format!(r#"{{"duplicates":[],"invalidates":[{inval_idx}]}}"#);
        let llm = stub_llm(&resp);
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_incoming", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert!(
            outcome.expired_incoming,
            "incoming should be expired (existing newer)"
        );
        assert_eq!(
            confirmed_of(&db, "mem_incoming").await,
            0,
            "incoming soft-suppressed (confirmed=0)"
        );
        assert_eq!(
            confirmed_of(&db, "mem_existing").await,
            1,
            "existing untouched, stays confirmed"
        );
        assert_eq!(row_count(&db).await, 2, "no deletion");
    }

    #[tokio::test]
    async fn test_apply_protected_never_auto_mutated() {
        let (db, _tmp) = test_db().await;
        // Existing is PROTECTED (pinned).
        seed_memory(
            &db,
            "mem_prot",
            "libSQL is built on top of SQLite",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses SQLite","domain":"databases"}"#),
            1000,
            Some(100),
            true,
            "confirmed",
        )
        .await;
        seed_memory(
            &db,
            "mem_inc",
            "libSQL is actually based on PostgreSQL internals",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses PostgreSQL","domain":"databases"}"#),
            2000,
            Some(200),
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_inc")
            .await
            .unwrap()
            .unwrap();
        let (pa, _pb) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        let inval_idx = pa.len();
        let resp = format!(r#"{{"duplicates":[],"invalidates":[{inval_idx}]}}"#);
        let llm = stub_llm(&resp);
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_inc", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(
            confirmed_of(&db, "mem_prot").await,
            1,
            "protected existing NEVER auto-suppressed"
        );
        assert_eq!(
            outcome.flagged_for_review,
            vec!["mem_prot".to_string()],
            "protected -> flagged for review"
        );
        assert_eq!(
            outcome.invalidated,
            Vec::<String>::new(),
            "no auto-invalidation on protected"
        );
        // Incoming flagged pending_revision + linked to the protected existing.
        assert_eq!(
            supersedes_of(&db, "mem_inc").await.as_deref(),
            Some("mem_prot")
        );
    }

    #[tokio::test]
    async fn test_apply_duplicate_emits_consolidation_proposal_not_collapse() {
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_dup",
            "I use the Rust programming language for all backend services",
            "fact",
            Some("engineering"),
            None,
            None,
            1000,
            None,
            false,
            "confirmed",
        )
        .await;
        // Materially richer re-capture: keeps the original clause (so it stays a
        // Pool-A near-duplicate) but adds new content tokens -> earns a merge card.
        seed_memory(
            &db,
            "mem_inc",
            "I use the Rust programming language for all backend services, \
             specifically the tokio async runtime and libsql persistence",
            "fact",
            Some("engineering"),
            None,
            None,
            2000,
            None,
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_inc")
            .await
            .unwrap()
            .unwrap();
        let (pa, _pb) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        assert!(!pa.is_empty(), "precondition: near-duplicate is in Pool A");
        // LLM marks Pool-A index 0 as a duplicate.
        let llm = stub_llm(r#"{"duplicates":[0],"invalidates":[]}"#);
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let before = row_count(&db).await;
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_inc", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(
            outcome.dedup_proposals.len(),
            1,
            "one consolidation proposal filed"
        );
        // BOTH memories remain confirmed, nothing collapsed.
        assert_eq!(confirmed_of(&db, "mem_dup").await, 1);
        assert_eq!(confirmed_of(&db, "mem_inc").await, 1);
        assert_eq!(row_count(&db).await, before, "no rows collapsed");
        // Proposal exists with the consolidation action.
        let pending = db.get_pending_refinements().await.unwrap();
        assert!(
            pending.iter().any(|p| p.action == "consolidate_duplicate"),
            "consolidation proposal should be in the queue"
        );
    }

    #[tokio::test]
    async fn test_apply_near_identical_duplicate_files_no_proposal() {
        // A re-capture that adds nothing material ("now" is the only delta) must
        // dedup silently — no consolidation card for the human to clear.
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_dup",
            "I use the Rust programming language for all backend services",
            "fact",
            Some("engineering"),
            None,
            None,
            1000,
            None,
            false,
            "confirmed",
        )
        .await;
        seed_memory(
            &db,
            "mem_inc",
            "I use the Rust programming language for all backend services now",
            "fact",
            Some("engineering"),
            None,
            None,
            2000,
            None,
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_inc")
            .await
            .unwrap()
            .unwrap();
        let (pa, _pb) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        assert!(!pa.is_empty(), "precondition: near-duplicate is in Pool A");
        // LLM still flags Pool-A index 0 as a duplicate; the richer-gate, not the
        // LLM, is what suppresses the card.
        let llm = stub_llm(r#"{"duplicates":[0],"invalidates":[]}"#);
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_inc", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert!(
            outcome.dedup_proposals.is_empty(),
            "~identical re-capture should dedup silently, no consolidation card"
        );
        let pending = db.get_pending_refinements().await.unwrap();
        assert!(
            !pending.iter().any(|p| p.action == "consolidate_duplicate"),
            "no consolidation proposal should be queued for a ~identical re-capture"
        );
    }

    #[test]
    fn materially_richer_predicate() {
        // Identical text adds nothing -> not richer.
        assert!(!is_materially_richer(
            "I use the Rust programming language for all backend services",
            "I use the Rust programming language for all backend services"
        ));
        // A trivial restatement (only the short word "now" added) -> not richer.
        assert!(!is_materially_richer(
            "I use the Rust programming language for all backend services now",
            "I use the Rust programming language for all backend services"
        ));
        // Substantial new content (several new content tokens) -> richer.
        assert!(is_materially_richer(
            "I use the Rust programming language for all backend services, \
             specifically tokio async runtime with libsql database persistence and axum routing",
            "I use the Rust programming language for all backend services"
        ));
    }

    #[tokio::test]
    async fn test_apply_empty_arrays_is_noop() {
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_old",
            "libSQL is built on top of SQLite",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses SQLite","domain":"databases"}"#),
            1000,
            Some(100),
            false,
            "confirmed",
        )
        .await;
        seed_memory(
            &db,
            "mem_new",
            "libSQL is actually based on PostgreSQL internals",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses PostgreSQL","domain":"databases"}"#),
            2000,
            Some(200),
            false,
            "confirmed",
        )
        .await;
        // LLM returns empty arrays -> no action.
        let llm = stub_llm(r#"{"duplicates":[],"invalidates":[]}"#);
        let prompts = PromptRegistry::default();
        let emitter = Arc::new(RecordingEmitter::new());
        let dyn_em: Arc<dyn EventEmitter> = emitter.clone();
        let before = row_count(&db).await;
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_new", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(
            outcome,
            ResolveOutcome::default(),
            "empty arrays -> idempotent no-op"
        );
        assert_eq!(confirmed_of(&db, "mem_old").await, 1);
        assert_eq!(confirmed_of(&db, "mem_new").await, 1);
        assert_eq!(row_count(&db).await, before);
        assert_eq!(emitter.count(), 0, "no events on no-op");
    }

    #[tokio::test]
    async fn test_apply_malformed_llm_is_silent_zero() {
        let (db, _tmp) = test_db().await;
        seed_memory(
            &db,
            "mem_old",
            "libSQL is built on top of SQLite",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses SQLite","domain":"databases"}"#),
            1000,
            Some(100),
            false,
            "confirmed",
        )
        .await;
        seed_memory(
            &db,
            "mem_new",
            "libSQL is actually based on PostgreSQL internals",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"libSQL uses PostgreSQL","domain":"databases"}"#),
            2000,
            Some(200),
            false,
            "confirmed",
        )
        .await;
        // Garbage LLM output -> silent-zero guard -> no mutation.
        let llm = stub_llm("the model is confused and returns prose only");
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_new", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;
        assert_eq!(
            outcome,
            ResolveOutcome::default(),
            "malformed LLM -> no mutation (silent-zero)"
        );
        assert_eq!(
            confirmed_of(&db, "mem_old").await,
            1,
            "no fact wrongly expired on garbage"
        );
    }

    #[tokio::test]
    async fn test_build_pools_subtracts_overlap() {
        let (db, _tmp) = test_db().await;
        // A memory that is BOTH a near-duplicate (high cosine) AND a
        // field-contradiction (same domain+type, differing claim) must end up in
        // Pool A only -- never double-counted in Pool B.
        seed_memory(
            &db,
            "mem_both",
            "I prefer dark mode in my code editor for long sessions",
            "fact",
            Some("preferences"),
            None,
            Some(r#"{"claim":"prefers dark mode editor","domain":"preferences"}"#),
            1000,
            Some(100),
            false,
            "confirmed",
        )
        .await;
        seed_memory(
            &db,
            "mem_inc",
            "I prefer dark mode in my code editor for long sessions now",
            "fact",
            Some("preferences"),
            None,
            Some(r#"{"claim":"prefers light mode editor","domain":"preferences"}"#),
            2000,
            Some(200),
            false,
            "confirmed",
        )
        .await;
        let incoming = db
            .get_incoming_for_resolution("mem_inc")
            .await
            .unwrap()
            .unwrap();
        let (pool_a, pool_b) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        let in_a = pool_a.iter().any(|c| c.source_id == "mem_both");
        let in_b = pool_b.iter().any(|c| c.source_id == "mem_both");
        assert!(in_a, "high-sim row should be in Pool A");
        assert!(
            !in_b,
            "row in Pool A must be subtracted from Pool B (disjoint)"
        );
    }

    #[tokio::test]
    async fn multi_invalidate_incoming_expired_does_not_suppress_other_existings() {
        // Incoming conflicts with TWO existings in OPPOSITE valid-time
        // directions: existing_A is NEWER than incoming (-> incoming should be
        // expired), existing_B is OLDER than incoming (-> incoming would
        // normally supersede it). Because the incoming LOSES to A, it must NOT
        // suppress B. Regression for the multi-invalidation short-circuit bug.
        let (db, _tmp) = test_db().await;
        // existing_A: newer valid-time (event_date=300).
        seed_memory(
            &db,
            "mem_a",
            "Postgres is my primary production database now",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"primary db is Postgres","domain":"databases"}"#),
            1000,
            Some(300),
            false,
            "confirmed",
        )
        .await;
        // existing_B: older valid-time (event_date=100).
        seed_memory(
            &db,
            "mem_b",
            "I rely on MySQL as the main datastore for everything",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"primary db is MySQL","domain":"databases"}"#),
            1100,
            Some(100),
            false,
            "confirmed",
        )
        .await;
        // incoming: valid-time=200 (between A and B).
        seed_memory(
            &db,
            "mem_inc",
            "These days SQLite is what I use as the primary database",
            "fact",
            Some("databases"),
            None,
            Some(r#"{"claim":"primary db is SQLite","domain":"databases"}"#),
            2000,
            Some(200),
            false,
            "confirmed",
        )
        .await;

        let incoming = db
            .get_incoming_for_resolution("mem_inc")
            .await
            .unwrap()
            .unwrap();
        let (pa, pb) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();
        // Both contradicting existings must be in Pool B (precondition).
        let b_ids: std::collections::HashSet<&str> =
            pb.iter().map(|c| c.source_id.as_str()).collect();
        assert!(
            b_ids.contains("mem_a") && b_ids.contains("mem_b"),
            "both existings should be in Pool B, got A={:?} B={b_ids:?}",
            pa.iter().map(|c| &c.source_id).collect::<Vec<_>>(),
        );
        // LLM invalidates BOTH Pool-B indices.
        let idx_a = pa.len() + pb.iter().position(|c| c.source_id == "mem_a").unwrap();
        let idx_b = pa.len() + pb.iter().position(|c| c.source_id == "mem_b").unwrap();
        let resp = format!(r#"{{"duplicates":[],"invalidates":[{idx_a},{idx_b}]}}"#);
        let llm = stub_llm(&resp);
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();
        let outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_inc", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;

        // Incoming LOST to existing_A (newer) -> expired.
        assert!(
            outcome.expired_incoming,
            "incoming should be expired (existing_A newer)"
        );
        assert_eq!(
            confirmed_of(&db, "mem_inc").await,
            0,
            "incoming soft-suppressed"
        );
        // existing_A is the winner -> untouched.
        assert_eq!(
            confirmed_of(&db, "mem_a").await,
            1,
            "newer existing_A stays confirmed"
        );
        // existing_B must NOT be suppressed: the incoming lost, so it can't win
        // against B either. THIS is the bug the fix prevents.
        assert_eq!(
            confirmed_of(&db, "mem_b").await,
            1,
            "older existing_B must NOT be suppressed by a losing incoming"
        );
        assert_eq!(
            outcome.invalidated,
            Vec::<String>::new(),
            "no existing should be invalidated"
        );
        // No deletion.
        assert_eq!(
            row_count(&db).await,
            3,
            "no rows deleted -- soft-suppress only"
        );
    }

    #[tokio::test]
    async fn test_document_chunks_excluded_from_resolution_pools() {
        let (db, _tmp) = test_db().await;

        // SETUP: Create a probe memory (the incoming for resolution).
        // This is a confirmed memory that will be used as the incoming.
        seed_memory(
            &db,
            "mem_probe",
            "I use Rust for backend services, tokio workers, and libsql persistence",
            "fact",
            Some("engineering"),
            None,
            None,
            1_710_000_000,
            None,
            false,
            "confirmed",
        )
        .await;

        // POSITIVE CONTROL: Create a confirmed capture that is a near-duplicate of the probe.
        // This should appear in Pool A (high cosine similarity >= 0.88).
        // Uses the same content to ensure high similarity.
        seed_memory(
            &db,
            "mem_duplicate",
            "I use Rust for backend services, tokio workers, and libsql persistence",
            "fact",
            Some("engineering"),
            None,
            None,
            1_710_000_050,
            None,
            false,
            "confirmed",
        )
        .await;

        // NEGATIVE CONTROL: Ingest a folder document with multiple chunks (unconfirmed).
        // Craft the content to be a near-duplicate of the probe so that IF the
        // `confirmed = 1` filter was removed, it WOULD surface. This ensures the test
        // can go RED if the invariant is violated.
        let folder_doc_body = std::iter::repeat_n(
            "I use Rust for backend services, tokio workers, and libsql persistence",
            10,
        )
        .collect::<Vec<_>>()
        .join(" ");

        db.upsert_documents(vec![RawDocument {
            source: "memory".to_string(),
            source_id: "folder_doc_1".to_string(),
            title: "Folder Decision Notes".to_string(),
            content: folder_doc_body,
            memory_type: Some("fact".to_string()),
            space: Some("engineering".to_string()),
            source_agent: Some("folder".to_string()),
            confirmed: None,
            content_hash: Some("folder-hash-1".to_string()),
            last_modified: 1_710_000_100,
            ..Default::default()
        }])
        .await
        .unwrap();

        // Verify preconditions: folder document chunks exist and are ALL unconfirmed.
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memories WHERE source = 'memory' AND source_id = ?1",
                libsql::params!["folder_doc_1"],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let chunk_count: i64 = row.get(0).unwrap();
        drop(rows);

        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM memories WHERE source = 'memory' AND source_id = ?1 AND confirmed IS NULL",
                libsql::params!["folder_doc_1"],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let unconfirmed_count: i64 = row.get(0).unwrap();
        drop(rows);
        drop(conn);

        assert!(
            chunk_count > 0,
            "folder document must exist in memories, got {chunk_count} rows"
        );
        assert_eq!(
            unconfirmed_count, chunk_count,
            "all folder document chunks must be unconfirmed (confirmed IS NULL), \
             got {unconfirmed_count}/{chunk_count}"
        );

        // Get the probe memory as the incoming for resolution.
        let incoming = db
            .get_incoming_for_resolution("mem_probe")
            .await
            .unwrap()
            .expect("probe memory must be resolvable");

        // Build resolution pools using the probe as the incoming.
        // The key invariant: build_resolution_pools filters on:
        //   chunk_index = 0 AND source = 'memory' AND confirmed = 1 AND source_id != incoming
        // This means:
        //   - mem_duplicate (confirmed=1, chunk_index=0) SHOULD appear in Pool A (cosine >= 0.88)
        //   - folder_doc_1 chunks (confirmed IS NULL) are excluded by the `confirmed = 1` filter
        let (pool_a, pool_b) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();

        // POSITIVE CONTROL ASSERTION: mem_duplicate should appear in Pool A.
        let pool_a_ids: Vec<&str> = pool_a.iter().map(|c| c.source_id.as_str()).collect();
        assert!(
            pool_a_ids.contains(&"mem_duplicate"),
            "confirmed duplicate must appear in Pool A (cosine >= 0.88), got {pool_a_ids:?}"
        );

        // NEGATIVE CONTROL ASSERTION: folder document chunks must NOT appear in either pool.
        // This is what we're locking: unconfirmed documents are immutable and never candidates.
        let pool_b_ids: Vec<&str> = pool_b.iter().map(|c| c.source_id.as_str()).collect();

        assert!(
            !pool_a_ids.contains(&"folder_doc_1"),
            "unconfirmed folder document chunks must not appear in Pool A (confirmed != 1 excludes them)"
        );
        assert!(
            !pool_b_ids.contains(&"folder_doc_1"),
            "unconfirmed folder document chunks must not appear in Pool B (confirmed != 1 excludes them)"
        );

        // REQUIREMENT (2): Exercise the refinement-queue candidate path.
        // Call resolve_dual_pool to verify that the refinement-queue candidate selection also
        // excludes document chunks. This uses the same build_resolution_pools query, so both
        // entry points are protected by the same predicate.
        let llm = stub_llm(r#"{"duplicates":[0],"invalidates":[]}"#);
        let prompts = PromptRegistry::default();
        let dyn_em = noop_emitter();

        let _outcome =
            temp_env::async_with_vars([("WENLAN_ENABLE_DUAL_POOL_RESOLVE", Some("1"))], async {
                resolve_dual_pool(&db, "mem_probe", Some(&llm), &prompts, &dyn_em)
                    .await
                    .unwrap()
            })
            .await;

        // Verify once more that resolve_dual_pool did not surface any folder document chunks
        // in its candidate selection (it calls build_resolution_pools internally).
        let (pool_a_final, pool_b_final) = db
            .build_resolution_pools(
                &incoming,
                crate::retrieval::resolve::ResolutionConfig::default(),
            )
            .await
            .unwrap();

        let pool_a_final_ids: Vec<&str> =
            pool_a_final.iter().map(|c| c.source_id.as_str()).collect();
        let pool_b_final_ids: Vec<&str> =
            pool_b_final.iter().map(|c| c.source_id.as_str()).collect();

        assert!(
            !pool_a_final_ids.contains(&"folder_doc_1"),
            "refinement-queue path: folder document chunks must not appear in Pool A"
        );
        assert!(
            !pool_b_final_ids.contains(&"folder_doc_1"),
            "refinement-queue path: folder document chunks must not appear in Pool B"
        );
    }
}
