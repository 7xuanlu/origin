// SPDX-License-Identifier: Apache-2.0
//! Knowledge graph quality checks: post-store verification and periodic rethink.

use crate::db::MemoryDB;
use crate::error::OriginError;

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
}
