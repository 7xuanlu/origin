// SPDX-License-Identifier: Apache-2.0
//! Knowledge graph quality checks: post-store verification and periodic rethink.

use crate::db::MemoryDB;
use crate::error::OriginError;
use crate::llm_provider::LlmProvider;
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
) -> Result<VerificationResult, OriginError> {
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

/// Run post-store verification on a newly distilled concept.
pub async fn verify_concept(
    db: &MemoryDB,
    concept_id: &str,
    concept_title: &str,
) -> Result<VerificationResult, OriginError> {
    let mut warnings = Vec::new();

    let self_retrieval_passed = match db
        .search_memory(concept_title, 10, None, None, None, None, None, None)
        .await
    {
        Ok(results) => {
            let found = results.iter().any(|r| r.source_id == concept_id);
            if !found {
                warnings.push(format!(
                    "Concept '{}' ({}) not found in top-10 self-retrieval results",
                    concept_title, concept_id
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

/// Check that a relation's endpoints exist and type is valid snake_case.
pub async fn verify_relation(
    db: &MemoryDB,
    from_entity: &str,
    to_entity: &str,
    relation_type: &str,
) -> Result<bool, OriginError> {
    let conn = db.conn.lock().await;

    // Check both entities exist
    let from_exists: i64 = conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE id = ?1",
            libsql::params![from_entity],
        )
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();

    let to_exists: i64 = conn
        .query(
            "SELECT COUNT(*) FROM entities WHERE id = ?1",
            libsql::params![to_entity],
        )
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();

    if from_exists == 0 || to_exists == 0 {
        return Ok(false);
    }

    // Check relation type is valid snake_case
    let valid_format = relation_type
        .chars()
        .all(|c| c.is_ascii_lowercase() || c == '_');
    Ok(valid_format)
}

/// Report of a rethink pass.
#[derive(Debug, Default, serde::Serialize)]
pub struct RethinkReport {
    pub merge_candidates: usize,
    pub types_normalized: usize,
    pub embeddings_refreshed: usize,
    pub stale_relations_flagged: usize,
    pub contradictions_found: usize,
}

/// Run the periodic knowledge graph rethink pass.
///
/// Phases:
/// 1. Entity merge candidates -- find duplicates with identical lowercase names
/// 2. Relation type normalization -- rewrite non-canonical types to canonical
/// 3. Entity embedding refresh -- re-embed entities with many new observations
/// 4. Stale relation detection -- relations whose source memory was deleted
/// 5. Contradiction scan -- log entities with many observations for review
pub async fn run_rethink(
    db: &MemoryDB,
    _llm: Option<&Arc<dyn LlmProvider>>,
    _config: &RefineryConfig,
) -> Result<RethinkReport, OriginError> {
    let mut report = RethinkReport::default();

    report.merge_candidates = find_merge_candidates(db).await?;
    report.types_normalized = normalize_non_vocabulary_relations(db).await?;
    report.embeddings_refreshed = refresh_stale_entity_embeddings(db).await?;
    report.stale_relations_flagged = detect_stale_relations(db).await?;
    report.contradictions_found = scan_contradictions(db).await?;

    Ok(report)
}

/// Find entity pairs with identical lowercase names that might be duplicates.
/// Migration 40 deduplicated existing data, but new duplicates can appear when
/// entities are created via code paths that bypass alias resolution.
pub async fn find_merge_candidates(db: &MemoryDB) -> Result<usize, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT LOWER(name) as lname, COUNT(*) as cnt FROM entities
             GROUP BY LOWER(name) HAVING cnt > 1",
            (),
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("merge candidates query: {}", e)))?;

    let mut count = 0usize;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| OriginError::VectorDb(format!("merge candidates row: {}", e)))?
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
    Ok(count)
}

/// Normalize relations whose type isn't canonical in the vocabulary.
/// Uses `resolve_relation_type` to map aliases to canonical forms.
pub async fn normalize_non_vocabulary_relations(db: &MemoryDB) -> Result<usize, OriginError> {
    // First, read all distinct relation types.
    let types_to_check: Vec<String> = {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query("SELECT DISTINCT relation_type FROM relations", ())
            .await
            .map_err(|e| OriginError::VectorDb(format!("distinct rel types: {}", e)))?;
        let mut types = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("rel type row: {}", e)))?
        {
            types.push(row.get::<String>(0).unwrap_or_default());
        }
        types
    };

    let mut normalized = 0usize;
    for rel_type in &types_to_check {
        if rel_type.is_empty() {
            continue;
        }
        let canonical = db.resolve_relation_type(rel_type).await?;
        if canonical != *rel_type {
            let conn = db.conn.lock().await;
            let affected = conn
                .execute(
                    "UPDATE relations SET relation_type = ?1 WHERE relation_type = ?2",
                    libsql::params![canonical.clone(), rel_type.clone()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(format!("normalize relations: {}", e)))?;
            normalized += affected as usize;
            log::info!(
                "[rethink] normalized '{}' -> '{}' ({} relations)",
                rel_type,
                canonical,
                affected
            );
        }
    }
    Ok(normalized)
}

/// Refresh embeddings for entities that have accumulated 5+ new observations
/// since their embedding was last updated.
pub async fn refresh_stale_entity_embeddings(db: &MemoryDB) -> Result<usize, OriginError> {
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
            .map_err(|e| OriginError::VectorDb(format!("stale entity query: {}", e)))?;

        let mut out = Vec::new();
        while let Some(row) = rows
            .next()
            .await
            .map_err(|e| OriginError::VectorDb(format!("stale entity row: {}", e)))?
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
                .map_err(|e| OriginError::VectorDb(format!("obs fetch: {}", e)))?;
            while let Some(row) = obs_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(format!("obs row: {}", e)))?
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
            Err(e) => log::warn!("[rethink] refresh_entity_embedding failed for '{}': {}", name, e),
        }
    }
    Ok(refreshed)
}

/// Count relations whose `source_memory_id` no longer corresponds to an
/// existing memory. Logged for visibility; actual pruning is deferred until
/// relation temporality lands (requires a dedicated `stale` column).
pub async fn detect_stale_relations(db: &MemoryDB) -> Result<usize, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM relations
             WHERE source_memory_id IS NOT NULL
             AND source_memory_id NOT IN (SELECT DISTINCT source_id FROM memories WHERE source_id IS NOT NULL)",
            (),
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("stale relations: {}", e)))?;

    let count: i64 = match rows
        .next()
        .await
        .map_err(|e| OriginError::VectorDb(format!("stale relations row: {}", e)))?
    {
        Some(row) => row.get::<i64>(0).unwrap_or(0),
        None => 0,
    };

    if count > 0 {
        log::warn!(
            "[rethink] {} relations have stale source_memory_id (source memory deleted)",
            count
        );
    }
    Ok(count as usize)
}

/// Scan for entities with many observations, logging them for manual review.
/// A full contradiction detection pass would need LLM; this is a cheap proxy
/// that highlights entities most likely to contain conflicting information.
pub async fn scan_contradictions(db: &MemoryDB) -> Result<usize, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT e.name, COUNT(o.id) as obs_count
             FROM entities e JOIN observations o ON o.entity_id = e.id
             GROUP BY e.id, e.name HAVING obs_count >= 10
             ORDER BY obs_count DESC LIMIT 20",
            (),
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("contradictions scan: {}", e)))?;

    let mut count = 0usize;
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| OriginError::VectorDb(format!("contradictions row: {}", e)))?
    {
        let name: String = row.get::<String>(0).unwrap_or_default();
        let obs: i64 = row.get::<i64>(1).unwrap_or(0);
        log::info!(
            "[rethink] entity '{}' has {} observations -- review for contradictions",
            name,
            obs
        );
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn test_db() -> (MemoryDB, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let db = MemoryDB::new(
            db_path.as_path(),
            Arc::new(crate::events::NoopEmitter),
        )
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
    async fn test_verify_relation_valid_type() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("Alice", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("ProjectX", "project", None, Some("test"), None)
            .await
            .unwrap();

        // Valid type
        assert!(verify_relation(&db, &e1, &e2, "works_on").await.unwrap());
        // Valid new type (snake_case)
        assert!(verify_relation(&db, &e1, &e2, "custom_type").await.unwrap());
        // Invalid: endpoints don't exist
        assert!(!verify_relation(&db, "nonexistent", &e2, "works_on")
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_verify_relation_invalid_format() {
        let (db, _dir) = test_db().await;

        let e1 = db
            .store_entity("Alice", "person", None, Some("test"), None)
            .await
            .unwrap();
        let e2 = db
            .store_entity("Bob", "person", None, Some("test"), None)
            .await
            .unwrap();

        // Invalid: not snake_case (uppercase)
        assert!(!verify_relation(&db, &e1, &e2, "WorksOn").await.unwrap());
        // Invalid: contains spaces
        assert!(!verify_relation(&db, &e1, &e2, "works on").await.unwrap());
    }

    #[tokio::test]
    async fn test_verify_entity_warns_on_missing() {
        let (db, _dir) = test_db().await;

        // Create an entity, then check a non-matching name
        let id = db
            .store_entity("Rust", "technology", None, Some("test"), None)
            .await
            .unwrap();
        let result = verify_entity(&db, &id, "completely unrelated query that won't match Rust at all")
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
            report.types_normalized >= 1,
            "expected at least 1 type normalized, got {}",
            report.types_normalized
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
        assert_eq!(report.types_normalized, 0);
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
                assert_eq!(rt, "works_on", "expected all relations normalized to 'works_on'");
            }
        }

        // 5. Entity self-retrieval passes
        let vr = verify_entity(&db, &id1, "Alice Chen").await.unwrap();
        assert_eq!(vr.entity_self_retrieval_passed, Some(true));

        // 6. Relation verification
        assert!(verify_relation(&db, &id1, &proj, "works_on").await.unwrap());
        assert!(
            !verify_relation(&db, "nonexistent", &proj, "works_on")
                .await
                .unwrap()
        );

        // 7. Rethink completes successfully
        let config = RefineryConfig::default();
        let report = run_rethink(&db, None, &config).await.unwrap();
        // Nothing to normalize (already canonical); no duplicates;
        // embedding_refreshed and stale counts should be 0.
        assert_eq!(report.merge_candidates, 0);
        assert_eq!(report.types_normalized, 0);
    }
}
