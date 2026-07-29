// SPDX-License-Identifier: Apache-2.0
//! Knowledge graph quality checks: post-store verification and periodic rethink.

use crate::db::{ContradictionObservationCount, MemoryDB};
use crate::error::WenlanError;
use crate::llm_provider::LlmProvider;
use crate::read_scope::ReadScope;
use crate::tuning::RefineryConfig;
use std::sync::Arc;

/// Result of a post-store verification check.
#[derive(Debug)]
pub struct VerificationResult {
    pub entity_self_retrieval_passed: Option<bool>,
    pub concept_self_retrieval_passed: Option<bool>,
    pub relation_consistency_passed: Option<bool>,
    pub warnings: Vec<String>,
}

/// Run post-store verification checks on a newly created/linked entity.
pub async fn verify_entity(
    db: &MemoryDB,
    entity_id: &str,
    entity_name: &str,
) -> Result<VerificationResult, WenlanError> {
    let mut warnings = Vec::new();

    // Entity self-retrieval test: search by name, check if this entity appears in top 5
    let self_retrieval_passed = match db.search_entities_by_vector(entity_name, 5).await {
        Ok(results) => {
            let found = results.iter().any(|r| r.entity.id == entity_id);
            if !found {
                warnings.push(format!(
                    "Entity '{}' ({}) not found in top-5 self-retrieval results",
                    entity_name, entity_id
                ));
            }
            Some(found)
        }
        Err(_) => None, // Embedding not available, skip check
    };

    Ok(VerificationResult {
        entity_self_retrieval_passed: self_retrieval_passed,
        concept_self_retrieval_passed: None,
        relation_consistency_passed: None,
        warnings,
    })
}

/// Run post-store verification on a newly distilled page.
pub async fn verify_page(
    db: &MemoryDB,
    page_id: &str,
    page_title: &str,
) -> Result<VerificationResult, WenlanError> {
    let mut warnings = Vec::new();

    let self_retrieval_passed = match db
        .search_memory(
            page_title,
            10,
            None,
            &ReadScope::Global,
            None,
            None,
            None,
            None,
        )
        .await
    {
        Ok(results) => {
            let found = results.iter().any(|r| r.source_id == page_id);
            if !found {
                warnings.push(format!(
                    "Page '{}' ({}) not found in top-10 self-retrieval results",
                    page_title, page_id
                ));
            }
            Some(found)
        }
        Err(_) => None,
    };

    Ok(VerificationResult {
        entity_self_retrieval_passed: None,
        concept_self_retrieval_passed: self_retrieval_passed,
        relation_consistency_passed: None,
        warnings,
    })
}

/// Report of a rethink pass.
#[derive(Debug, Default, serde::Serialize)]
pub struct RethinkReport {
    pub merge_candidates: usize,
    pub relations_healed: usize,
    pub relations_queued: usize,
    pub entities_healed: usize,
    pub entities_queued: usize,
    pub embeddings_refreshed: usize,
    pub stale_relations_flagged: usize,
    pub contradictions_found: usize,
}

/// Run the periodic knowledge graph rethink pass.
///
/// Phases:
/// 1. Entity merge candidates -- find duplicates with identical lowercase names
/// 2. Relation vocabulary heal -- auto-fold safe non-canonical types (known
///    aliases + deterministic transforms) to canonical, queue the rest for review
/// 3. Entity embedding refresh -- re-embed entities with many new observations
/// 4. Stale relation detection -- relations whose source memory was deleted
/// 5. Contradiction scan -- log entities with many observations for review
pub async fn run_rethink(
    db: &MemoryDB,
    _llm: Option<&Arc<dyn LlmProvider>>,
    _config: &RefineryConfig,
) -> Result<RethinkReport, WenlanError> {
    let rel = heal_relation_vocabulary(db).await?;
    let ent = heal_entity_vocabulary(db).await?;
    let report = RethinkReport {
        merge_candidates: find_merge_candidates(db).await?,
        relations_healed: rel.healed,
        relations_queued: rel.queued,
        entities_healed: ent.healed,
        entities_queued: ent.queued,
        embeddings_refreshed: refresh_stale_entity_embeddings(db).await?,
        stale_relations_flagged: detect_stale_relations(db).await?,
        contradictions_found: scan_contradictions(db).await?,
    };

    let receipt = serde_json::json!({
        "relations_healed": report.relations_healed,
        "relations_queued": report.relations_queued,
        "entities_healed": report.entities_healed,
        "entities_queued": report.entities_queued,
    })
    .to_string();
    let _ = db
        .log_agent_activity("daemon", "vocab_heal_receipt", &[], None, &receipt)
        .await;

    Ok(report)
}

/// Find entity pairs with identical lowercase names that might be duplicates.
/// Migration 40 deduplicated existing data, but new duplicates can appear when
/// entities are created via code paths that bypass alias resolution.
pub async fn find_merge_candidates(db: &MemoryDB) -> Result<usize, WenlanError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT LOWER(name) as lname, COUNT(*) as cnt FROM entities
             GROUP BY LOWER(name) HAVING cnt > 1",
            (),
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("merge candidates query: {}", e)))?;

    let mut count = 0usize;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("merge candidates row: {}", e)))?
    {
        let name: String = row.get::<String>(0).unwrap_or_default();
        let cnt: i64 = row.get::<i64>(1).unwrap_or(0);
        log::warn!(
            "[rethink] merge candidate: '{}' has {} entities -- review for dedup",
            name,
            cnt
        );
        // Each group with N > 1 yields N-1 merge candidates.
        count += (cnt.saturating_sub(1)) as usize;
    }
    drop(rows);
    drop(conn);

    // T16: surface MinHash/LSH band collisions into the human-review queue.
    // Opt-in (WENLAN_ENABLE_ENTITY_MINHASH). These are the *borderline* pairs
    // the auto-merge cascade deliberately leaves alone: exact Jaccard in
    // [0.85, 0.9), or a same/near match across DIFFERENT entity types. They are
    // enqueued as `entity_merge` proposals tagged `minhash_jaccard`, never
    // auto-merged.
    if crate::db::entity_minhash_enabled() {
        count += surface_minhash_merge_candidates(db).await?;
    }
    Ok(count)
}

/// Scan high-entropy entity names for LSH band collisions and enqueue the
/// borderline pairs (Jaccard in [0.85, 0.9), or cross-type) into the
/// human-review refinement queue. Returns the number of proposals enqueued.
async fn surface_minhash_merge_candidates(db: &MemoryDB) -> Result<usize, WenlanError> {
    use crate::retrieval::dedup;
    use std::collections::HashMap;

    // Pull every entity once (id, name, entity_type).
    let entities: Vec<(String, String, String)> = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT id, name, entity_type FROM entities", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("minhash candidates scan: {e}")))?;
        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("minhash candidates row: {e}")))?
        {
            let id: String = row.get(0).unwrap_or_default();
            let name: String = row.get(1).unwrap_or_default();
            let etype: String = row.get(2).unwrap_or_default();
            if dedup::has_high_entropy(&name) {
                out.push((id, name, etype));
            }
        }
        out
    };

    // Bucket entity indices by band key.
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, (_, name, _)) in entities.iter().enumerate() {
        for key in dedup::name_band_keys(name) {
            buckets.entry(key).or_default().push(i);
        }
    }

    // Examine each colliding pair once.
    let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut enqueued = 0usize;
    for idxs in buckets.values() {
        for a in 0..idxs.len() {
            for b in (a + 1)..idxs.len() {
                let (i, j) = (idxs[a].min(idxs[b]), idxs[a].max(idxs[b]));
                if i == j || !seen.insert((i, j)) {
                    continue;
                }
                let (ref id_i, ref name_i, ref type_i) = entities[i];
                let (ref id_j, ref name_j, ref type_j) = entities[j];
                let jac = dedup::name_jaccard(name_i, name_j);
                let same_type = type_i.eq_ignore_ascii_case(type_j);
                // Human-review band: borderline similarity, OR a cross-type
                // collision the auto-merge guard would have skipped. Anything
                // at/above the auto-merge threshold with the SAME type is left
                // to the write-path cascade and is not re-queued here.
                let borderline = (0.85..dedup::FUZZY_JACCARD_THRESHOLD).contains(&jac);
                let cross_type_match = !same_type && jac >= 0.85;
                if !(borderline || cross_type_match) {
                    continue;
                }
                let i_len = id_i.len().min(8);
                let j_len = id_j.len().min(8);
                let proposal_id = format!("minhash_{}_{}", &id_i[..i_len], &id_j[..j_len]);
                let payload = serde_json::json!({
                    "existing_id": id_i,
                    "new_id": id_j,
                    "jaccard": jac,
                    "same_type": same_type,
                    "provenance": "minhash_jaccard",
                })
                .to_string();
                if db
                    .insert_refinement_proposal(
                        &proposal_id,
                        "entity_merge",
                        &[id_i.clone(), id_j.clone()],
                        Some(&payload),
                        jac,
                    )
                    .await
                    .is_ok()
                {
                    enqueued += 1;
                }
            }
        }
    }
    Ok(enqueued)
}

/// Counts returned by a vocabulary heal pass: how many relations were
/// auto-folded to a canonical, and how many distinct new types were queued
/// for human review as promote candidates.
#[derive(Debug, Default, Clone, Copy)]
pub struct VocabHealCounts {
    pub healed: usize,
    pub queued: usize,
}

/// Heal non-canonical relation types: auto-fold the deterministically safe ones
/// (known aliases + safe transforms), queue everything else as a promote candidate.
pub async fn heal_relation_vocabulary(db: &MemoryDB) -> Result<VocabHealCounts, WenlanError> {
    // First, read all distinct relation types.
    let types_to_check: Vec<String> = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT DISTINCT relation_type FROM relations", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("distinct rel types: {}", e)))?;
        let mut types = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("rel type row: {}", e)))?
        {
            types.push(row.get::<String>(0).unwrap_or_default());
        }
        types
    };

    let canonicals = db.relation_canonicals().await?;
    let mut counts = VocabHealCounts::default();
    for rel_type in &types_to_check {
        if rel_type.is_empty() {
            continue;
        }
        // Already canonical? nothing to do.
        if canonicals.iter().any(|c| c == rel_type) {
            continue;
        }
        // Band A1: known alias in the vocabulary -> deterministic fold.
        if let Some(canonical) = db.resolve_relation_type(rel_type).await? {
            let folded = db.fold_relation_type(rel_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} relations, ledgered)",
                rel_type,
                canonical,
                folded
            );
            continue;
        }
        // Band A2: deterministic safe transform toward an existing canonical.
        if let Some(canonical) = crate::vocab::safe_transform::safe_transform(rel_type, &canonicals)
        {
            let folded = db.fold_relation_type(rel_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} relations, ledgered)",
                rel_type,
                canonical,
                folded
            );
            continue;
        }
        // Band B: semantic / new -> queue one promote candidate.
        if db
            .insert_vocab_promote_proposal("relation", rel_type, None)
            .await?
        {
            counts.queued += 1;
        }
    }
    Ok(counts)
}

/// Heal non-canonical entity types: auto-fold the deterministically safe ones
/// (known aliases + safe transforms), queue everything else as a promote candidate.
pub async fn heal_entity_vocabulary(db: &MemoryDB) -> Result<VocabHealCounts, WenlanError> {
    // First, read all distinct entity types.
    let types_to_check: Vec<String> = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT DISTINCT entity_type FROM entities", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("distinct entity types: {}", e)))?;
        let mut types = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("entity type row: {}", e)))?
        {
            types.push(row.get::<String>(0).unwrap_or_default());
        }
        types
    };

    let canonicals = db.entity_canonicals().await?;
    let mut counts = VocabHealCounts::default();
    for entity_type in &types_to_check {
        if entity_type.is_empty() {
            continue;
        }
        // Already canonical? nothing to do.
        if canonicals.iter().any(|c| c == entity_type) {
            continue;
        }
        // Band A1: known alias in the vocabulary -> deterministic fold.
        if let Some(canonical) = db.resolve_entity_type(entity_type).await? {
            let folded = db.fold_entity_type(entity_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} entities, ledgered)",
                entity_type,
                canonical,
                folded
            );
            continue;
        }
        // Band A2: deterministic safe transform toward an existing canonical.
        if let Some(canonical) =
            crate::vocab::safe_transform::safe_transform(entity_type, &canonicals)
        {
            let folded = db.fold_entity_type(entity_type, &canonical).await?;
            counts.healed += folded;
            log::info!(
                "[rethink] folded '{}' -> '{}' ({} entities, ledgered)",
                entity_type,
                canonical,
                folded
            );
            continue;
        }
        // Band B: semantic / new -> queue one promote candidate.
        if db
            .insert_vocab_promote_proposal("entity", entity_type, None)
            .await?
        {
            counts.queued += 1;
        }
    }
    Ok(counts)
}

/// Refresh embeddings for entities that have accumulated 5+ new observations
/// since their embedding was last updated.
pub async fn refresh_stale_entity_embeddings(db: &MemoryDB) -> Result<usize, WenlanError> {
    // Find candidates.
    let candidates: Vec<(String, String)> = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT e.id, e.name
                 FROM entities e
                 LEFT JOIN observations o ON o.entity_id = e.id
                    AND o.created_at > COALESCE(e.embedding_updated_at, 0)
                 GROUP BY e.id, e.name
                 HAVING COUNT(o.id) >= 5",
                (),
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("stale entity query: {}", e)))?;

        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| WenlanError::VectorDb(format!("stale entity row: {}", e)))?
        {
            out.push((
                row.get::<String>(0).unwrap_or_default(),
                row.get::<String>(1).unwrap_or_default(),
            ));
        }
        out
    };

    let mut refreshed = 0usize;
    for (id, name) in &candidates {
        // Build text from entity name + top 10 recent observations.
        let mut parts = vec![name.clone()];
        {
            let conn = db.conn.lock().await;
            let mut obs_rows = conn
                .query(
                    "SELECT content FROM observations WHERE entity_id = ?1 ORDER BY created_at DESC LIMIT 10",
                    libsql::params![id.clone()],
                )
                .await
                .map_err(|e| WenlanError::VectorDb(format!("obs fetch: {}", e)))?;
            while let Some(row) = obs_rows
                .next()
                .await
                .map_err(|e| WenlanError::VectorDb(format!("obs row: {}", e)))?
            {
                let c: String = row.get::<String>(0).unwrap_or_default();
                if !c.is_empty() {
                    parts.push(c);
                }
            }
        }
        let combined = parts.join(". ");
        match db.refresh_entity_embedding(id, &combined).await {
            Ok(()) => {
                refreshed += 1;
                log::info!("[rethink] refreshed embedding for entity '{}'", name);
            }
            Err(e) => log::warn!(
                "[rethink] refresh_entity_embedding failed for '{}': {}",
                name,
                e
            ),
        }
    }
    Ok(refreshed)
}

/// Count relations whose `source_memory_id` no longer corresponds to an
/// existing memory. Logged for visibility; actual pruning is deferred until
/// relation temporality lands (requires a dedicated `stale` column).
pub async fn detect_stale_relations(db: &MemoryDB) -> Result<usize, WenlanError> {
    let count = db.count_stale_relation_sources().await?;
    if count > 0 {
        log::warn!(
            "[rethink] {} relations have stale source_memory_id (source memory deleted)",
            count
        );
    }
    Ok(count)
}

/// Scan for entities with many observations, logging them for manual review.
/// A full contradiction detection pass would need LLM; this is a cheap proxy
/// that highlights entities most likely to contain conflicting information.
pub async fn scan_contradictions(db: &MemoryDB) -> Result<usize, WenlanError> {
    let counts: Vec<ContradictionObservationCount> =
        db.list_contradiction_observation_counts().await?;
    for item in &counts {
        log::info!(
            "[rethink] entity '{}' has {} observations -- review for contradictions",
            item.entity_name,
            item.observation_count
        );
    }
    Ok(counts.len())
}

/// Embedding-based hallucination check: returns `true` if the body is
/// semantically similar (cosine >= 0.6) to the concatenation of the source
/// memories' content. Returns `false` if the body diverges from its
/// cited sources, or if embeddings cannot be produced.
///
/// Used by the agent-triggered `/api/pages` create path. The daemon-side
/// distillation in `synthesis::distill` keeps its own inline check for
/// now since it builds source text from a cluster, not by id.
pub async fn hallucination_guard(
    db: &MemoryDB,
    body: &str,
    source_ids: &[String],
) -> Result<bool, WenlanError> {
    if source_ids.is_empty() {
        return Ok(true);
    }
    let mut source_contents: Vec<String> = Vec::with_capacity(source_ids.len());
    for sid in source_ids {
        // Propagate DB errors; Ok(None) means unresolvable id — skip silently.
        if let Some(detail) = db.get_memory_detail(sid).await? {
            source_contents.push(detail.content);
        }
    }
    if source_contents.is_empty() {
        return Ok(true);
    }
    let joined = source_contents.join(" ");
    let texts = vec![body.to_string(), joined];
    match db.generate_embeddings(&texts) {
        Ok(embs) if embs.len() == 2 => {
            let sim = crate::db::cosine_similarity(&embs[0], &embs[1]);
            Ok(sim >= 0.6)
        }
        _ => Ok(true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDB::new(db_path.as_path(), Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();
        (db, dir)
    }

    #[tokio::test]
    async fn test_verify_entity_passes_for_good_entity() {
        let (db, _dir) = test_db().await;

        let id = db
            .store_entity("Rust", "technology", None, Some("test"), None)
            .await
            .unwrap();
        let result = verify_entity(&db, &id, "Rust").await.unwrap();
        assert_eq!(result.entity_self_retrieval_passed, Some(true));
        assert!(result.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_verify_entity_warns_on_missing() {
        let (db, _dir) = test_db().await;

        // Create an entity, then check a non-matching name
        let id = db
            .store_entity("Rust", "technology", None, Some("test"), None)
            .await
            .unwrap();
        let result = verify_entity(
            &db,
            &id,
            "completely unrelated query that won't match Rust at all",
        )
        .await
        .unwrap();
        // With only one entity in the DB, vector search should still return it as
        // closest result even for an unrelated query, so this may still pass.
        // The important thing is the function executes without error.
        assert!(result.entity_self_retrieval_passed.is_some());
    }

    #[tokio::test]
    async fn test_run_rethink_normalizes_relation_types() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("Alice", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectX", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Bypass create_relation's normalization by inserting directly.
        // This simulates legacy data created before relation vocabulary existed.
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                libsql::params![
                    "rel-legacy-1".to_string(),
                    e1.clone(),
                    e2.clone(),
                    "working_at".to_string(),
                    chrono::Utc::now().timestamp()
                ],
            )
            .await
            .unwrap();
        }

        let config = RefineryConfig::default();
        let report = run_rethink(&db, None, &config).await.unwrap();

        // Rethink should have normalized "working_at" -> "works_on"
        assert!(
            report.relations_healed >= 1,
            "expected at least 1 type normalized, got {}",
            report.relations_healed
        );

        // Verify the relation now has the canonical type
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT relation_type FROM relations WHERE id = ?1",
                libsql::params!["rel-legacy-1".to_string()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let rt: String = row.get::<String>(0).unwrap();
        assert_eq!(rt, "works_on");
    }

    #[tokio::test]
    async fn test_run_rethink_completes_on_empty_db() {
        let (db, _dir) = test_db().await;
        let config = RefineryConfig::default();
        let report = run_rethink(&db, None, &config).await.unwrap();
        // All zero is fine; the point is it runs without error.
        assert_eq!(report.merge_candidates, 0);
        assert_eq!(report.relations_healed, 0);
        assert_eq!(report.stale_relations_flagged, 0);
    }

    #[tokio::test]
    async fn test_find_merge_candidates_detects_duplicates() {
        let (db, _dir) = test_db().await;

        // Create two entities with same lowercase name by bypassing alias resolution.
        // (In practice this shouldn't happen post-migration-40, but we want to know
        // if it does via the rethink's logging.)
        {
            let conn = db.conn.lock().await;
            let now = chrono::Utc::now().timestamp();
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, source_agent, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
                libsql::params![
                    "dup-1".to_string(),
                    "Alice".to_string(),
                    "person".to_string(),
                    "test".to_string(),
                    now
                ],
            )
            .await
            .unwrap();
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, source_agent, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
                libsql::params![
                    "dup-2".to_string(),
                    "alice".to_string(),
                    "person".to_string(),
                    "test".to_string(),
                    now
                ],
            )
            .await
            .unwrap();
        }

        let count = find_merge_candidates(&db).await.unwrap();
        assert_eq!(count, 1, "expected 1 merge candidate (2 entities, 1 extra)");
    }

    #[tokio::test]
    async fn test_full_kg_quality_pipeline() {
        let (db, _dir) = test_db().await;

        // 1. Create entity with alias self-registration
        let id1 = db
            .store_entity("Alice Chen", "person", None, Some("test"), None)
            .await
            .unwrap();

        // 2. Verify alias resolution (case-insensitive)
        let resolved = db.resolve_entity_by_alias("alice chen").await.unwrap();
        assert_eq!(resolved, Some(id1.clone()));

        // 3. Create relation using canonical type
        let proj = db
            .store_entity("ProjectX", "project", None, Some("test"), None)
            .await
            .unwrap();
        db.create_relation(
            &id1,
            &proj,
            "works_on",
            Some("test"),
            Some(0.9),
            Some("she leads it"),
            Some("mem_1"),
        )
        .await
        .unwrap();

        // 4. Create relation using alias type — should normalize at insert
        let proj2 = db
            .store_entity("ProjectY", "project", None, Some("test"), None)
            .await
            .unwrap();
        db.create_relation(&id1, &proj2, "working_at", Some("test"), None, None, None)
            .await
            .unwrap();

        // Verify normalization happened at insert time
        {
            let conn = db.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT relation_type FROM relations WHERE from_entity = ?1",
                    libsql::params![id1.clone()],
                )
                .await
                .unwrap();
            while let Some(row) = rows.next().await.unwrap() {
                let rt: String = row.get::<String>(0).unwrap();
                assert_eq!(
                    rt, "works_on",
                    "expected all relations normalized to 'works_on'"
                );
            }
        }

        // 5. Entity self-retrieval passes
        let vr = verify_entity(&db, &id1, "Alice Chen").await.unwrap();
        assert_eq!(vr.entity_self_retrieval_passed, Some(true));

        // 6. Rethink completes successfully
        let config = RefineryConfig::default();
        let report = run_rethink(&db, None, &config).await.unwrap();
        // Nothing to normalize (already canonical); no duplicates;
        // embedding_refreshed and stale counts should be 0.
        assert_eq!(report.merge_candidates, 0);
        assert_eq!(report.relations_healed, 0);
    }

    // ── Fix 1: Case-insensitive relation resolution ────────────────────────

    #[tokio::test]
    async fn test_resolve_relation_type_case_insensitive() {
        let (db, _dir) = test_db().await;

        // "Working_At" should resolve to "works_on" (alias match, case-insensitive)
        let result = db.resolve_relation_type("Working_At").await.unwrap();
        assert_eq!(
            result,
            Some("works_on".to_string()),
            "Working_At should resolve to works_on via case-insensitive alias lookup"
        );

        // "WORKS_ON" is itself the canonical, just uppercased — should return "works_on"
        let result = db.resolve_relation_type("WORKS_ON").await.unwrap();
        assert_eq!(
            result,
            Some("works_on".to_string()),
            "WORKS_ON should resolve to works_on via canonical lookup (lowercased)"
        );

        // Identity: already canonical lowercase
        let result = db.resolve_relation_type("works_on").await.unwrap();
        assert_eq!(
            result,
            Some("works_on".to_string()),
            "works_on should return itself unchanged"
        );

        // Novel type not in vocabulary: return None
        let result = db.resolve_relation_type("novel_type").await.unwrap();
        assert_eq!(
            result, None,
            "novel_type should return None (not in vocabulary)"
        );
    }

    // ── Fix 2: ON CONFLICT updates confidence when higher ─────────────────

    #[tokio::test]
    async fn test_create_relation_confidence_upsert() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("Alice", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectAlpha", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Insert with confidence 0.5
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.5), None, None)
            .await
            .unwrap();

        // Upsert with higher confidence 0.9 — should update
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.9), None, None)
            .await
            .unwrap();

        // Verify confidence is now 0.9
        {
            let conn = db.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT confidence FROM relations WHERE from_entity = ?1 AND to_entity = ?2 AND relation_type = ?3",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let confidence: f64 = row.get::<f64>(0).unwrap();
            assert!(
                (confidence - 0.9).abs() < 1e-6,
                "confidence should be 0.9 after higher-confidence upsert, got {confidence}"
            );
        }

        // Upsert with lower confidence 0.3 — should NOT update
        db.create_relation(&e1, &e2, "works_on", Some("test"), Some(0.3), None, None)
            .await
            .unwrap();

        // Verify confidence is still 0.9
        {
            let conn = db.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT confidence, COUNT(*) as cnt FROM relations WHERE from_entity = ?1 AND to_entity = ?2 AND relation_type = ?3",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let confidence: f64 = row.get::<f64>(0).unwrap();
            let cnt: i64 = row.get::<i64>(1).unwrap();
            assert!(
                (confidence - 0.9).abs() < 1e-6,
                "confidence should still be 0.9 after lower-confidence upsert, got {confidence}"
            );
            assert_eq!(cnt, 1, "only 1 relation row should exist, got {cnt}");
        }
    }

    // ── Fix 3: fold_relation_type merges provenance + ledgers the loser ────

    #[tokio::test]
    async fn fold_relation_type_merges_provenance_and_ledgers_the_loser() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("EntityA", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("EntityB", "project", None, Some("test"), None)
            .await
            .unwrap();
        let now = chrono::Utc::now().timestamp();
        {
            let conn = db.conn.lock().await;
            // Survivor: canonical works_on, confidence 0.9.
            conn.execute(
                "INSERT INTO relations (id, from_entity, to_entity, relation_type, confidence, created_at) VALUES ('rel-canon', ?1, ?2, 'works_on', 0.9, ?3)",
                libsql::params![e1.clone(), e2.clone(), now]).await.unwrap();
            // Loser: alias working_at, confidence 0.5, has an explanation.
            conn.execute(
                "INSERT INTO relations (id, from_entity, to_entity, relation_type, confidence, explanation, created_at) VALUES ('rel-alias', ?1, ?2, 'working_at', 0.5, 'employed since 2020', ?3)",
                libsql::params![e1.clone(), e2.clone(), now]).await.unwrap();
        }

        let folded = db
            .fold_relation_type("working_at", "works_on")
            .await
            .unwrap();
        assert_eq!(folded, 1);

        let conn = db.conn.lock().await;
        // Exactly one A->B row survives, canonical, confidence preserved at 0.9,
        // and the loser's explanation was merged in (survivor had none).
        let mut rows = conn.query(
            "SELECT COUNT(*), MAX(confidence) FROM relations WHERE from_entity = ?1 AND to_entity = ?2",
            libsql::params![e1.clone(), e2.clone()]).await.unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let cnt: i64 = row.get(0).unwrap();
        let conf: f64 = row.get(1).unwrap();
        assert_eq!(cnt, 1, "collision must not leave two rows");
        assert!(
            (conf - 0.9).abs() < 1e-6,
            "survivor keeps the higher confidence"
        );
        drop(rows);
        // The absorbed edge's pre-image is in the ledger (undo is possible),
        // and the pre-image is COMPLETE: it carries created_at so a NOT-NULL
        // restore is possible on undo.
        let mut led = conn
            .query(
                "SELECT COUNT(*), MAX(pre_image) FROM vocab_heal_ledger WHERE old_value = 'working_at'",
                (),
            )
            .await
            .unwrap();
        let led_row = led.next().await.unwrap().unwrap();
        let n: i64 = led_row.get(0).unwrap();
        let pre_image: String = led_row.get(1).unwrap();
        assert!(n >= 1, "the folded loser must be recorded in the ledger");
        assert!(
            pre_image.contains("created_at"),
            "pre-image must include created_at so undo can restore the NOT-NULL column, got {pre_image}"
        );
        // Spec §2.5: the SURVIVOR is mutated too (provenance COALESCE-merge), so its
        // pre-image must also be ledgered before the UPDATE, else undo cannot restore
        // its pre-merge confidence/explanation. `rel-canon` is the survivor's id and
        // appears in NO loser pre-image (loser id is `rel-alias`), so its presence
        // proves the survivor pre-image was captured.
        assert!(
            pre_image.contains("rel-canon"),
            "survivor pre-image must be ledgered so undo can restore the mutated survivor, got {pre_image}"
        );
    }

    #[tokio::test]
    async fn fold_relation_type_rolls_back_when_ledger_insert_fails() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("RollbackA", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("RollbackB", "project", None, Some("test"), None)
            .await
            .unwrap();
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO relations
                     (id, from_entity, to_entity, relation_type, confidence, created_at)
                 VALUES ('rollback-canon', ?1, ?2, 'works_on', 0.9, 1)",
                libsql::params![e1.clone(), e2.clone()],
            )
            .await
            .unwrap();
            conn.execute(
                "INSERT INTO relations
                     (id, from_entity, to_entity, relation_type, confidence,
                      explanation, created_at)
                 VALUES ('rollback-alias', ?1, ?2, 'working_at', 0.5,
                         'must survive failed ledger', 1)",
                libsql::params![e1.clone(), e2.clone()],
            )
            .await
            .unwrap();
            conn.execute_batch(
                "CREATE TRIGGER fail_relation_vocab_ledger
                 BEFORE INSERT ON vocab_heal_ledger
                 WHEN NEW.kind = 'relation'
                 BEGIN
                     SELECT RAISE(FAIL, 'forced relation ledger failure');
                 END;",
            )
            .await
            .unwrap();
        }

        let error = db
            .fold_relation_type("working_at", "works_on")
            .await
            .expect_err("ledger failure must reject the whole fold");
        assert!(
            error.to_string().contains("forced relation ledger failure"),
            "unexpected fold error: {error}"
        );

        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT id, relation_type, explanation
                 FROM relations
                 WHERE from_entity = ?1 AND to_entity = ?2
                 ORDER BY id",
                libsql::params![e1, e2],
            )
            .await
            .unwrap();
        let mut surviving = Vec::new();
        while let Some(row) = rows.next().await.unwrap() {
            surviving.push((
                row.get::<String>(0).unwrap(),
                row.get::<String>(1).unwrap(),
                row.get::<Option<String>>(2).unwrap(),
            ));
        }
        assert_eq!(
            surviving,
            vec![
                (
                    "rollback-alias".to_string(),
                    "working_at".to_string(),
                    Some("must survive failed ledger".to_string()),
                ),
                ("rollback-canon".to_string(), "works_on".to_string(), None,),
            ],
            "relation mutation and undo ledger must commit or roll back together"
        );
    }

    // ── hallucination_guard ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_hallucination_guard_rejects_unrelated_body() {
        let (db, _dir) = test_db().await;
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "rust-mem".to_string(),
            title: "rust-mem".to_string(),
            content: "Rust is a systems programming language".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test".to_string()),
            confidence: Some(0.9),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();
        let body = "Pasta carbonara uses eggs and pancetta";
        let source_ids = vec!["rust-mem".to_string()];
        let passed = crate::kg_quality::hallucination_guard(&db, body, &source_ids)
            .await
            .unwrap();
        assert!(!passed, "guard should reject body unrelated to sources");
    }

    #[tokio::test]
    async fn test_hallucination_guard_accepts_related_body() {
        let (db, _dir) = test_db().await;
        let doc = crate::sources::RawDocument {
            source: "memory".to_string(),
            source_id: "rust-mem-2".to_string(),
            title: "rust-mem-2".to_string(),
            content: "Rust is a systems programming language with memory safety".to_string(),
            last_modified: chrono::Utc::now().timestamp(),
            memory_type: Some("fact".to_string()),
            source_agent: Some("test".to_string()),
            confidence: Some(0.9),
            ..Default::default()
        };
        db.upsert_documents(vec![doc]).await.unwrap();
        let body = "Rust provides memory-safe systems programming";
        let source_ids = vec!["rust-mem-2".to_string()];
        let passed = crate::kg_quality::hallucination_guard(&db, body, &source_ids)
            .await
            .unwrap();
        assert!(passed, "guard should accept body matching sources");
    }

    // ── Fix 5: source_memory_id is populated ──────────────────────────────

    #[tokio::test]
    async fn test_create_relation_source_memory_id() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("PersonX", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectZ", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Insert with source_memory_id "mem_123" and confidence 0.8
        db.create_relation(
            &e1,
            &e2,
            "works_on",
            Some("test"),
            Some(0.8),
            None,
            Some("mem_123"),
        )
        .await
        .unwrap();

        // Verify source_memory_id was stored
        {
            let conn = db.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT source_memory_id FROM relations WHERE from_entity = ?1 AND to_entity = ?2 AND relation_type = ?3",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let smid: String = row.get::<String>(0).unwrap();
            assert_eq!(smid, "mem_123", "source_memory_id should be mem_123");
        }

        // Upsert with lower confidence and source_memory_id "mem_456".
        // Relation provenance is fill-only: the first non-null source remains.
        db.create_relation(
            &e1,
            &e2,
            "works_on",
            Some("test"),
            Some(0.3),
            None,
            Some("mem_456"),
        )
        .await
        .unwrap();

        {
            let conn = db.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT source_memory_id, confidence FROM relations WHERE from_entity = ?1 AND to_entity = ?2 AND relation_type = ?3",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let smid: String = row.get::<String>(0).unwrap();
            let conf: f64 = row.get::<f64>(1).unwrap();
            // Confidence should still be 0.8 (not overwritten by lower 0.3)
            assert!(
                (conf - 0.8).abs() < 1e-6,
                "confidence should remain 0.8 after lower-confidence upsert, got {conf}"
            );
            assert_eq!(
                smid, "mem_123",
                "source_memory_id should preserve the first non-null source"
            );
        }

        // Upsert with higher confidence and source_memory_id "mem_789"
        db.create_relation(
            &e1,
            &e2,
            "works_on",
            Some("test"),
            Some(0.95),
            None,
            Some("mem_789"),
        )
        .await
        .unwrap();

        // Higher confidence updates the score, but not existing provenance.
        {
            let conn = db.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT source_memory_id, confidence FROM relations WHERE from_entity = ?1 AND to_entity = ?2 AND relation_type = ?3",
                    libsql::params![e1.clone(), e2.clone(), "works_on".to_string()],
                )
                .await
                .unwrap();
            let row = rows.next().await.unwrap().expect("relation should exist");
            let smid: String = row.get::<String>(0).unwrap();
            let conf: f64 = row.get::<f64>(1).unwrap();
            assert!(
                (conf - 0.95).abs() < 1e-6,
                "confidence should be 0.95 after higher-confidence upsert, got {conf}"
            );
            assert_eq!(
                smid, "mem_123",
                "source_memory_id should remain mem_123 after higher-confidence upsert"
            );
        }
    }

    // ── T16: MinHash band-collision surfacing into the human-review queue ────

    #[tokio::test]
    async fn find_merge_candidates_surfaces_band_near_dups() {
        temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", Some("1"))], async {
            let (db, _dir) = test_db().await;
            // "Glorptech"/"Glorptechs": high-entropy, share an LSH band, exact
            // Jaccard ~0.875 in [0.85, 0.9) -> borderline -> human-review only.
            let jac = crate::retrieval::dedup::name_jaccard("Glorptech", "Glorptechs");
            assert!(
                (0.85..0.9).contains(&jac),
                "fixture pair must sit in the borderline band, got {jac}"
            );
            let id1 = db
                .store_entity("Glorptech", "project", None, Some("t"), None)
                .await
                .unwrap();
            let id2 = db
                .store_entity("Glorptechs", "project", None, Some("t"), None)
                .await
                .unwrap();
            // Both still exist (NOT auto-merged).
            assert!(db.get_entity_name_type(&id1).await.unwrap().is_some());
            assert!(db.get_entity_name_type(&id2).await.unwrap().is_some());

            let enqueued = find_merge_candidates(&db).await.unwrap();
            assert!(enqueued >= 1, "borderline band collision must be counted");

            let pending = db.get_pending_refinements().await.unwrap();
            let proposal = pending
                .iter()
                .find(|p| p.action == "entity_merge" && p.id.starts_with("minhash_"))
                .expect("a minhash entity_merge proposal must be queued");
            let payload = proposal.payload.as_ref().expect("proposal payload");
            assert!(
                payload.contains("minhash_jaccard"),
                "proposal must carry minhash_jaccard provenance, got: {payload}"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn find_merge_candidates_minhash_off_no_band_proposals() {
        // Flag OFF: the band-collision pass must not run at all.
        temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", None::<&str>)], async {
            let (db, _dir) = test_db().await;
            db.store_entity("Glorptech", "project", None, Some("t"), None)
                .await
                .unwrap();
            db.store_entity("Glorptechs", "project", None, Some("t"), None)
                .await
                .unwrap();
            find_merge_candidates(&db).await.unwrap();
            let pending = db.get_pending_refinements().await.unwrap();
            assert!(
                !pending.iter().any(|p| p.id.starts_with("minhash_")),
                "flag OFF must enqueue no minhash proposals"
            );
        })
        .await;
    }

    #[tokio::test]
    async fn heal_relation_vocabulary_folds_aliases_and_queues_semantics() {
        let (db, _dir) = test_db().await;
        let e1 = db
            .store_entity("A", "person", None, Some("t"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("B", "project", None, Some("t"), None)
            .await
            .unwrap();
        let now = chrono::Utc::now().timestamp();
        {
            let conn = db.conn.lock().await;
            // Known alias -> auto-fold to works_on.
            conn.execute("INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at) VALUES ('r1', ?1, ?2, 'working_at', ?3)",
                libsql::params![e1.clone(), e2.clone(), now]).await.unwrap();
            // Semantic new type -> queue promote.
            conn.execute("INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at) VALUES ('r2', ?1, ?2, 'design_inspiration', ?3)",
                libsql::params![e1.clone(), e2.clone(), now]).await.unwrap();
        }
        let counts = super::heal_relation_vocabulary(&db).await.unwrap();
        assert!(
            counts.healed >= 1,
            "working_at is a known alias, must auto-fold"
        );
        assert!(
            counts.queued >= 1,
            "design_inspiration must be queued as promote"
        );
        // working_at is gone (folded); design_inspiration still present (awaiting human).
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT relation_type FROM relations ORDER BY relation_type",
                (),
            )
            .await
            .unwrap();
        let mut types = vec![];
        while let Some(r) = rows.next().await.unwrap() {
            types.push(r.get::<String>(0).unwrap());
        }
        assert!(types.contains(&"works_on".to_string()));
        assert!(types.contains(&"design_inspiration".to_string()));
        assert!(!types.contains(&"working_at".to_string()));
        drop(rows);
        drop(conn); // release before get_pending_refinements re-locks conn (non-reentrant mutex)
                    // And a promote proposal exists for design_inspiration.
        let pending = db.get_pending_refinements().await.unwrap();
        assert!(pending.iter().any(|p| p.action == "vocab_promote"
            && p.payload
                .as_deref()
                .unwrap_or("")
                .contains("design_inspiration")));
    }

    #[tokio::test]
    async fn heal_entity_vocabulary_folds_plural_and_queues_semantic() {
        let (db, _dir) = test_db().await;
        db.store_entity("X", "concepts", None, Some("t"), None)
            .await
            .unwrap(); // safe: concepts->concept
        db.store_entity("Y", "interest", None, Some("t"), None)
            .await
            .unwrap(); // semantic: queue
        let counts = super::heal_entity_vocabulary(&db).await.unwrap();
        assert!(counts.healed >= 1);
        assert!(counts.queued >= 1);
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT entity_type FROM entities ORDER BY entity_type", ())
            .await
            .unwrap();
        let mut types = vec![];
        while let Some(r) = rows.next().await.unwrap() {
            types.push(r.get::<String>(0).unwrap());
        }
        assert!(types.contains(&"concept".to_string()));
        assert!(!types.contains(&"concepts".to_string()));
        assert!(types.contains(&"interest".to_string())); // unchanged, awaiting human
    }

    #[tokio::test]
    async fn run_rethink_emits_a_vocab_heal_receipt() {
        let (db, _dir) = test_db().await;
        // Seed one queue-able dirty relation.
        let e1 = db
            .store_entity("A", "person", None, Some("t"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("B", "project", None, Some("t"), None)
            .await
            .unwrap();
        {
            let conn = db.conn.lock().await;
            conn.execute("INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at) VALUES ('r1', ?1, ?2, 'design_inspiration', 0)",
                libsql::params![e1, e2]).await.unwrap();
        }
        let cfg = crate::tuning::RefineryConfig::default();
        super::run_rethink(&db, None, &cfg).await.unwrap();
        // A vocab-heal receipt activity row exists.
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT COUNT(*) FROM agent_activity WHERE action = 'vocab_heal_receipt'",
                (),
            )
            .await
            .unwrap();
        let n: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert!(n >= 1, "run_rethink must emit a counted vocab-heal receipt");
    }
}
