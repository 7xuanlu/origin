use super::{WriteOutcome, WriteResult};
use crate::{db::MemoryDB, error::WenlanError};
use wenlan_types::requests::{AddObservationRequest, CreateEntityRequest, CreateRelationRequest};

/// Create or resolve an entity. Canonical entry point for both
/// agent-triggered (`/api/memory/entities`) and daemon-internal
/// (`kg/entity_extraction.rs`) writes.
///
/// Resolution order (4-step, matches `importer::resolve_entity_bulk` used for bulk/eval paths):
///   1. Alias lookup
///   2. Exact name search
///   3. Vector similarity (distance < 0.1 => sim > 0.9)
///   4. Create new
///
/// Post-write enrichment fires only on newly-created entities. Resolved-existing
/// returns immediately with empty warnings.
pub async fn create_entity(
    db: &MemoryDB,
    req: CreateEntityRequest,
    agent: &str,
) -> Result<WriteResult, WenlanError> {
    // Pre-write validation
    let name = req.name.trim();
    if name.is_empty() {
        return Err(WenlanError::Validation(
            "entity name must not be empty".into(),
        ));
    }
    let entity_type = req.entity_type.trim();
    if entity_type.is_empty() {
        return Err(WenlanError::Validation(
            "entity_type must not be empty".into(),
        ));
    }
    if let Some(c) = req.confidence {
        if !(0.0..=1.0).contains(&c) {
            return Err(WenlanError::Validation(format!(
                "confidence {c} out of range [0.0, 1.0]"
            )));
        }
    }

    // Resolve-then-write: the canonical cascade (M3 PR-1 stage d) shared with
    // `importer::resolve_entity_bulk`, terminating in `store_entity`.
    let (id, created) = db
        .resolve_or_create_entity(
            name,
            entity_type,
            req.space.as_deref(),
            req.source_agent.as_deref(),
            req.confidence,
        )
        .await?;
    if !created {
        return Ok(WriteResult {
            id,
            attached_to: None,
            warnings: vec![],
            wrote: false,
            revision_card_id: None,
            gated: false,
            outcome: WriteOutcome::Unchanged,
            acknowledged: false,
        });
    }

    // Post-write enrichment (LLM-free, non-blocking) -- only for a genuinely
    // new entity; a resolved match already went through this ring when it
    // was first created.
    let mut warnings: Vec<String> = Vec::new();

    // 1. Self-retrieval verification
    if let Ok(result) = crate::kg_quality::verify_entity(db, &id, name).await {
        for w in &result.warnings {
            log::warn!("[create_entity] {w}");
            warnings.push(w.clone());
        }
    }

    // 2. Merge-candidate refinery enqueue: similar entity in [0.85, 0.9) with same type
    if let Ok(results) = db.search_entities_by_vector(name, 5).await {
        for r in &results {
            if r.entity.id == id {
                continue;
            }
            if r.entity.entity_type != entity_type {
                continue;
            }
            let sim = 1.0 - r.distance;
            if (0.85..0.9).contains(&sim) {
                let id_len = id.len().min(8);
                let r_id_len = r.entity.id.len().min(8);
                let proposal_id = format!("merge_{}_{}", &id[..id_len], &r.entity.id[..r_id_len]);
                let payload = serde_json::json!({
                    "existing_id": r.entity.id,
                    "new_id": id,
                    "similarity": sim,
                })
                .to_string();
                let _ = db
                    .insert_refinement_proposal(
                        &proposal_id,
                        "entity_merge",
                        &[id.clone(), r.entity.id.clone()],
                        Some(&payload),
                        sim as f64,
                    )
                    .await;
            }
        }
    }

    // 3. Activity log
    let detail = format!("name={name}, type={entity_type}");
    if let Err(e) = db
        .log_agent_activity(
            agent,
            "entity_create",
            std::slice::from_ref(&id),
            None,
            &detail,
        )
        .await
    {
        log::warn!("[create_entity] activity log failed: {e}");
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

/// Create a directed relation between two entities. Canonical entry for
/// both agent-triggered (`/api/memory/relations`) and daemon-internal
/// extraction.
pub async fn create_relation(
    db: &MemoryDB,
    req: CreateRelationRequest,
    agent: &str,
) -> Result<WriteResult, WenlanError> {
    create_relation_with_span(db, req, agent, None).await
}

/// Like [`create_relation`], plus the verbatim source-memory text the
/// relation's `span` quote (if any) was extracted from. Used by
/// daemon-internal KG extraction to ground `req.span` into char offsets
/// (M3g span capture, §2.3). `source_content` should be the exact string
/// the extraction model saw -- never re-fetched from the DB, since a
/// batch-extraction `source_memory_id` may not map 1:1 to that content.
pub async fn create_relation_with_span(
    db: &MemoryDB,
    req: CreateRelationRequest,
    agent: &str,
    source_content: Option<&str>,
) -> Result<WriteResult, WenlanError> {
    // Pre-write validation
    if !db.entity_exists(&req.from_entity).await? {
        return Err(WenlanError::Validation(format!(
            "from_entity '{}' does not exist",
            req.from_entity
        )));
    }
    if !db.entity_exists(&req.to_entity).await? {
        return Err(WenlanError::Validation(format!(
            "to_entity '{}' does not exist",
            req.to_entity
        )));
    }
    let rt = req.relation_type.trim();
    if !is_valid_snake_case_relation(rt) {
        return Err(WenlanError::Validation(format!(
            "relation_type '{rt}' must be lowercase snake_case (^[a-z][a-z0-9_]*$)"
        )));
    }

    // Source-less retries can return immediately. A source-backed retry must
    // reach the DB upsert so an existing unbacked triple gains provenance and
    // the canonical edge dual-write runs.
    if req.source_memory_id.is_none() {
        if let Ok(existing) = db
            .list_relations_between(&req.from_entity, &req.to_entity)
            .await
        {
            if let Some((existing_id, _)) = existing.into_iter().find(|(_, t)| t == rt) {
                return Ok(WriteResult {
                    id: existing_id,
                    attached_to: None,
                    warnings: vec![],
                    wrote: false,
                    revision_card_id: None,
                    gated: false,
                    outcome: WriteOutcome::Unchanged,
                    acknowledged: false,
                });
            }
        }
    }

    let id = db
        .create_relation_with_span(
            &req.from_entity,
            &req.to_entity,
            rt,
            req.source_agent.as_deref(),
            req.confidence,
            req.explanation.as_deref(),
            req.source_memory_id.as_deref(),
            req.span.as_deref(),
            source_content,
            req.model_version.as_deref(),
            req.prompt_version.as_deref(),
        )
        .await?;

    // Post-write enrichment
    let mut warnings: Vec<String> = Vec::new();

    // Conflict check: existing relation between same (from, to) with different
    // type → auto-supersede (last-write-wins). The /refinery skill no longer
    // surfaces queue proposals to users (PR #109), so enqueuing for human review
    // would silently accumulate forever. The same outcome the user would have
    // hand-applied via `accept_refinement(relation_conflict)` runs immediately
    // here. Activity log records the daemon's decision for power-user audit
    // (queryable via list_agent_activity).
    if let Ok(existing) = db
        .list_relations_between(&req.from_entity, &req.to_entity)
        .await
    {
        for (existing_id, existing_type) in &existing {
            if existing_id != &id && existing_type != rt {
                match db.supersede_relation(existing_id, &id).await {
                    Ok(archived) => {
                        warnings.push(format!(
                            "auto-superseded existing relation ({}-{}-{}); newer relation now active",
                            req.from_entity, existing_type, req.to_entity
                        ));
                        let payload = serde_json::json!({
                            "existing_id": existing_id,
                            "new_id": id,
                            "from": req.from_entity,
                            "to": req.to_entity,
                            "old_type": existing_type,
                            "new_type": rt,
                            "archived": archived,
                        })
                        .to_string();
                        if let Err(e) = db
                            .log_agent_activity(
                                agent,
                                "relation_supersede_auto",
                                &[id.clone(), existing_id.clone()],
                                None,
                                &payload,
                            )
                            .await
                        {
                            log::warn!("[create_relation] auto-supersede activity log failed: {e}");
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "[create_relation] auto-supersede of {} → {} failed: {e}",
                            existing_id,
                            id
                        );
                        warnings.push(format!(
                            "conflicting relation exists ({}-{}-{}); auto-supersede failed",
                            req.from_entity, existing_type, req.to_entity
                        ));
                    }
                }
            }
        }
    }

    // Activity log
    let detail = format!(
        "from={}, to={}, type={}",
        req.from_entity, req.to_entity, rt
    );
    if let Err(e) = db
        .log_agent_activity(
            agent,
            "relation_create",
            std::slice::from_ref(&id),
            None,
            &detail,
        )
        .await
    {
        log::warn!("[create_relation] activity log failed: {e}");
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

/// Add an observation to an existing entity. Canonical entry for both
/// agent-triggered (`/api/memory/observations`) and daemon-internal callers.
pub async fn add_observation(
    db: &MemoryDB,
    req: AddObservationRequest,
    agent: &str,
) -> Result<WriteResult, WenlanError> {
    // Pre-write validation
    if !db.entity_exists(&req.entity_id).await? {
        return Err(WenlanError::Validation(format!(
            "entity_id '{}' does not exist",
            req.entity_id
        )));
    }
    let content = req.content.trim();
    if content.chars().count() < 5 {
        return Err(WenlanError::Validation(
            "observation content must be at least 5 characters".into(),
        ));
    }
    if let Some(c) = req.confidence {
        if !(0.0..=1.0).contains(&c) {
            return Err(WenlanError::Validation(format!(
                "confidence {c} out of range [0.0, 1.0]"
            )));
        }
    }

    let id = db
        .add_observation(
            &req.entity_id,
            content,
            req.source_agent.as_deref(),
            req.confidence,
        )
        .await?;

    // Activity log (no verify step yet — observations have no canonical quality check)
    let detail = format!("entity_id={}, content_len={}", req.entity_id, content.len());
    if let Err(e) = db
        .log_agent_activity(
            agent,
            "observation_add",
            std::slice::from_ref(&id),
            None,
            &detail,
        )
        .await
    {
        log::warn!("[add_observation] activity log failed: {e}");
    }

    Ok(WriteResult {
        id,
        attached_to: None,
        warnings: vec![],
        wrote: true,
        revision_card_id: None,
        gated: false,
        outcome: WriteOutcome::Wrote,
        acknowledged: false,
    })
}

fn is_valid_snake_case_relation(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}
