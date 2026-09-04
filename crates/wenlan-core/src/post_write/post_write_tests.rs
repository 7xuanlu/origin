use super::*;
use crate::events::NoopEmitter;
use std::collections::HashSet;
use std::sync::Arc;
use wenlan_types::requests::{
    AddObservationRequest, CreateConceptRequest, CreateEntityRequest, CreateRelationRequest,
    UpdatePageRequest,
};

// Serialize env-var-sensitive tests to avoid races.
// Uses tokio::sync::Mutex so the guard can safely span .await points.
async fn env_lock() -> tokio::sync::MutexGuard<'static, ()> {
    static ENV_MUTEX: tokio::sync::OnceCell<tokio::sync::Mutex<()>> =
        tokio::sync::OnceCell::const_new();
    ENV_MUTEX
        .get_or_init(|| async { tokio::sync::Mutex::new(()) })
        .await
        .lock()
        .await
}

async fn test_db() -> (MemoryDB, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");
    let db = MemoryDB::new(&path, Arc::new(NoopEmitter)).await.unwrap();
    (db, dir)
}

#[tokio::test]
async fn create_entity_rejects_empty_name() {
    let (db, _dir) = test_db().await;
    let req = CreateEntityRequest {
        name: "".to_string(),
        entity_type: "person".to_string(),
        space: wenlan_types::WriteSpaceTarget::Uncategorized,
        source_agent: Some("test".to_string()),
        confidence: None,
    };
    assert!(matches!(
        create_entity(&db, req, "test").await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn create_entity_rejects_empty_type() {
    let (db, _dir) = test_db().await;
    let req = CreateEntityRequest {
        name: "Alice".to_string(),
        entity_type: "".to_string(),
        space: (None).into(),
        source_agent: Some("test".to_string()),
        confidence: None,
    };
    assert!(matches!(
        create_entity(&db, req, "test").await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn create_entity_rejects_out_of_range_confidence() {
    let (db, _dir) = test_db().await;
    let req = CreateEntityRequest {
        name: "Alice".to_string(),
        entity_type: "person".to_string(),
        space: (None).into(),
        source_agent: Some("test".to_string()),
        confidence: Some(1.5),
    };
    assert!(matches!(
        create_entity(&db, req, "test").await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn create_entity_happy_path_returns_id() {
    let (db, _dir) = test_db().await;
    let req = CreateEntityRequest {
        name: "Alice".to_string(),
        entity_type: "person".to_string(),
        space: (None).into(),
        source_agent: Some("test".to_string()),
        confidence: Some(0.9),
    };
    let result = create_entity(&db, req, "test").await.unwrap();
    assert!(!result.id.is_empty());
}

#[tokio::test]
async fn create_entity_resolves_to_existing_by_name() {
    let (db, _dir) = test_db().await;
    let req1 = CreateEntityRequest {
        name: "Alice".to_string(),
        entity_type: "person".to_string(),
        space: (None).into(),
        source_agent: Some("test".to_string()),
        confidence: None,
    };
    let first = create_entity(&db, req1, "test").await.unwrap();
    let req2 = CreateEntityRequest {
        name: "Alice".to_string(),
        entity_type: "person".to_string(),
        space: (None).into(),
        source_agent: Some("test".to_string()),
        confidence: None,
    };
    let second = create_entity(&db, req2, "test").await.unwrap();
    assert_eq!(first.id, second.id);
}

#[tokio::test]
async fn create_relation_rejects_missing_from_entity() {
    let (db, _dir) = test_db().await;
    let req = CreateRelationRequest {
        from_entity: "missing-1".to_string(),
        to_entity: "missing-2".to_string(),
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    assert!(matches!(
        create_relation(&db, req, "test").await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn create_relation_rejects_bad_relation_type() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let bob = db
        .store_entity("Bob", "person", None, Some("test"), None)
        .await
        .unwrap();
    let req = CreateRelationRequest {
        from_entity: alice,
        to_entity: bob,
        relation_type: "Knows!".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    assert!(matches!(
        create_relation(&db, req, "test").await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn create_relation_happy_path() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let bob = db
        .store_entity("Bob", "person", None, Some("test"), None)
        .await
        .unwrap();
    let req = CreateRelationRequest {
        from_entity: alice,
        to_entity: bob,
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let result = create_relation(&db, req, "test").await.unwrap();
    assert!(!result.id.is_empty());
}

#[tokio::test]
async fn create_relation_idempotent_no_double_log() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let bob = db
        .store_entity("Bob", "person", None, Some("test"), None)
        .await
        .unwrap();
    let req1 = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let first = create_relation(&db, req1, "agent-x").await.unwrap();
    let req2 = CreateRelationRequest {
        from_entity: alice,
        to_entity: bob,
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let second = create_relation(&db, req2, "agent-x").await.unwrap();
    // Idempotent re-post must resolve to the same relation id.
    // The second call returns early before logging, so no duplicate activity row.
    assert_eq!(
        first.id, second.id,
        "should resolve to existing relation id"
    );
    assert!(
        second.warnings.is_empty(),
        "idempotent resolve should have no warnings"
    );
}

#[tokio::test]
async fn identical_unbacked_relation_becomes_source_backed() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let wenlan = db
        .store_entity("Wenlan", "project", None, Some("test"), None)
        .await
        .unwrap();

    let first = create_relation(
        &db,
        CreateRelationRequest {
            from_entity: alice.clone(),
            to_entity: wenlan.clone(),
            relation_type: "works_on".to_string(),
            source_agent: Some("test".to_string()),
            confidence: None,
            explanation: None,
            source_memory_id: None,
            span: None,
            model_version: None,
            prompt_version: None,
        },
        "test-agent",
    )
    .await
    .unwrap();

    {
        let conn = db.test_primary_session().await;
        conn.execute(
            "DELETE FROM edges
             WHERE edge_type = 'relates' AND src_id = ?1 AND dst_id = ?2",
            libsql::params![alice.clone(), wenlan.clone()],
        )
        .await
        .unwrap();
    }

    let second = create_relation(
        &db,
        CreateRelationRequest {
            from_entity: alice.clone(),
            to_entity: wenlan.clone(),
            relation_type: "works_on".to_string(),
            source_agent: Some("test".to_string()),
            confidence: None,
            explanation: None,
            source_memory_id: Some("mem_explicit_relation".to_string()),
            span: None,
            model_version: None,
            prompt_version: None,
        },
        "test-agent",
    )
    .await
    .unwrap();

    assert_eq!(second.id, first.id, "upsert must preserve the relation id");
    let conn = db.test_primary_session().await;
    // G6 Stage 2 PR 2b: `relations` is frozen -- `first.id`/`second.id` are
    // now the content-addressed `edges.edge_id` (item 3 compat note), so
    // round-trip through `edges` instead of the old `relations.id`.
    let mut rows = conn
        .query(
            "SELECT json_extract(payload, '$.source_memory_id') FROM edges WHERE edge_id = ?1",
            libsql::params![first.id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("relates edge");
    let source_memory_id: Option<String> = row.get(0).unwrap();
    assert_eq!(
        source_memory_id.as_deref(),
        Some("mem_explicit_relation"),
        "the source-backed retry must attach provenance to the existing triple"
    );
    drop(rows);
    let mut edge_rows = conn
        .query(
            "SELECT COUNT(*) FROM edges
             WHERE edge_type = 'relates' AND src_id = ?1 AND dst_id = ?2",
            libsql::params![alice, wenlan],
        )
        .await
        .unwrap();
    let edge_count = edge_rows
        .next()
        .await
        .unwrap()
        .unwrap()
        .get::<i64>(0)
        .unwrap();
    assert_eq!(
        edge_count, 1,
        "the source-backed retry must restore the canonical edge dual-write"
    );
}

#[tokio::test]
async fn create_relation_conflict_auto_supersedes_existing() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let bob = db
        .store_entity("Bob", "person", None, Some("test"), None)
        .await
        .unwrap();

    // Create existing relation: A-knows-B
    let req_knows = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let knows_result = create_relation(&db, req_knows, "test-agent").await.unwrap();
    let knows_id = knows_result.id.clone();

    // Create conflicting relation: A-likes-B (different type, same pair)
    let req_likes = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "likes".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let likes_result = create_relation(&db, req_likes, "test-agent").await.unwrap();

    // Warning should indicate auto-supersede
    assert!(
        likes_result
            .warnings
            .iter()
            .any(|w| w.contains("auto-superseded existing relation")),
        "expected auto-supersede warning, got: {:?}",
        likes_result.warnings
    );

    // Activity log should contain relation_supersede_auto entry
    let activity = db.list_agent_activity(50, None, None).await.unwrap();
    assert!(
        activity
            .iter()
            .any(|a| a.action == "relation_supersede_auto"),
        "expected relation_supersede_auto in activity log"
    );

    // The old knows relation should be gone (superseded / deleted)
    let active = db.list_relations_between(&alice, &bob).await.unwrap();
    let active_ids: Vec<&str> = active.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        !active_ids.contains(&knows_id.as_str()),
        "old knows relation should be archived/deleted"
    );
    assert!(
        active_ids.contains(&likes_result.id.as_str()),
        "new likes relation should be active"
    );

    // No relation_conflict proposal should have been inserted
    let pending = db.get_pending_refinements().await.unwrap();
    assert!(
        !pending.iter().any(|p| p.action == "relation_conflict"),
        "no relation_conflict proposal should be queued"
    );
}

#[tokio::test]
async fn create_relation_conflict_payload_contains_archived_snapshot() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let bob = db
        .store_entity("Bob", "person", None, Some("test"), None)
        .await
        .unwrap();

    // Existing relation carries full metadata that hard-delete would lose.
    let req_knows = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: Some(0.72),
        explanation: Some("met at offsite".to_string()),
        source_memory_id: Some("mem_seed".to_string()),
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let knows_id = create_relation(&db, req_knows, "test-agent")
        .await
        .unwrap()
        .id;

    // Conflicting different-type relation triggers auto-supersede.
    let req_likes = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "likes".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    create_relation(&db, req_likes, "test-agent").await.unwrap();

    let activity = db.list_agent_activity(50, None, None).await.unwrap();
    let entry = activity
        .iter()
        .find(|a| a.action == "relation_supersede_auto")
        .expect("relation_supersede_auto activity entry");

    let detail = entry.detail.as_ref().expect("payload detail present");
    let payload: serde_json::Value = serde_json::from_str(detail).expect("payload is JSON");
    let archived = &payload["archived"];
    assert_eq!(archived["id"], serde_json::json!(knows_id));
    assert_eq!(archived["relation_type"], serde_json::json!("knows"));
    assert_eq!(archived["confidence"], serde_json::json!(0.72));
    assert_eq!(archived["explanation"], serde_json::json!("met at offsite"));
    assert_eq!(archived["source_memory_id"], serde_json::json!("mem_seed"));
    assert_eq!(archived["source_agent"], serde_json::json!("test"));
    assert!(
        archived["created_at"].is_i64(),
        "archived created_at present"
    );
}

#[tokio::test]
async fn create_relation_conflict_no_op_when_existing_same_type() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let bob = db
        .store_entity("Bob", "person", None, Some("test"), None)
        .await
        .unwrap();

    // Create A-knows-B
    let req1 = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let first = create_relation(&db, req1, "test-agent").await.unwrap();

    // Create A-knows-B again (same type → idempotent early return)
    let req2 = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "knows".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let second = create_relation(&db, req2, "test-agent").await.unwrap();

    // Should resolve to same id, no supersede warning
    assert_eq!(first.id, second.id, "idempotent call should return same id");
    assert!(
        !second
            .warnings
            .iter()
            .any(|w| w.contains("auto-superseded")),
        "no supersede warning expected for same-type idempotent call"
    );

    // No relation_supersede_auto activity
    let activity = db.list_agent_activity(50, None, None).await.unwrap();
    assert!(
        !activity
            .iter()
            .any(|a| a.action == "relation_supersede_auto"),
        "no relation_supersede_auto activity expected for same-type call"
    );
}

#[tokio::test]
async fn create_relation_no_conflict_when_no_existing_relation() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let bob = db
        .store_entity("Bob", "person", None, Some("test"), None)
        .await
        .unwrap();

    // First relation — no prior relation exists
    let req = CreateRelationRequest {
        from_entity: alice.clone(),
        to_entity: bob.clone(),
        relation_type: "likes".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
        explanation: None,
        source_memory_id: None,
        span: None,
        model_version: None,
        prompt_version: None,
    };
    let result = create_relation(&db, req, "test-agent").await.unwrap();

    assert!(!result.id.is_empty());
    assert!(
        !result
            .warnings
            .iter()
            .any(|w| w.contains("auto-superseded") || w.contains("supersede")),
        "no supersede warning expected when no prior relation exists"
    );

    // No relation_supersede_auto activity
    let activity = db.list_agent_activity(50, None, None).await.unwrap();
    assert!(
        !activity
            .iter()
            .any(|a| a.action == "relation_supersede_auto"),
        "no relation_supersede_auto activity expected on first relation create"
    );
}

#[tokio::test]
async fn add_observation_rejects_missing_entity() {
    let (db, _dir) = test_db().await;
    let req = AddObservationRequest {
        entity_id: "no-such-entity".to_string(),
        content: "Alice prefers Rust".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
    };
    assert!(matches!(
        add_observation(&db, req, "test", &crate::read_scope::ReadScope::Global).await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn add_observation_rejects_short_content() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let req = AddObservationRequest {
        entity_id: alice,
        content: "hi".to_string(),
        source_agent: Some("test".to_string()),
        confidence: None,
    };
    assert!(matches!(
        add_observation(&db, req, "test", &crate::read_scope::ReadScope::Global).await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn add_observation_happy_path() {
    let (db, _dir) = test_db().await;
    let alice = db
        .store_entity("Alice", "person", None, Some("test"), None)
        .await
        .unwrap();
    let req = AddObservationRequest {
        entity_id: alice.clone(),
        content: "Alice prefers Rust over Python".to_string(),
        source_agent: Some("test".to_string()),
        confidence: Some(0.9),
    };
    let result = add_observation(&db, req, "test", &crate::read_scope::ReadScope::Global)
        .await
        .unwrap();
    assert!(!result.id.is_empty());

    // Verify the observation was actually persisted
    let observations = db
        .get_observations_for_entities(&[alice], 10)
        .await
        .unwrap();
    assert_eq!(observations.len(), 1);
    assert!(observations[0].content.contains("Alice prefers Rust"));
}

// ── create_page ──────────────────────────────────────────────────────────

#[tokio::test]
async fn create_page_rejects_missing_source_memory() {
    let (db, _dir) = test_db().await;
    let req = CreateConceptRequest {
        title: "Some Page".to_string(),
        content: "body content that is long enough".to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec!["mem_does_not_exist".to_string()],
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };
    assert!(matches!(
        create_page(&db, req, "test", None).await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn create_page_rejects_hallucinated_body() {
    let (db, _dir) = test_db().await;
    seed_memory(&db, "mem-rust-a", "Rust is a systems programming language").await;
    seed_memory(&db, "mem-rust-b", "Rust has ownership and borrowing").await;
    seed_memory(&db, "mem-rust-c", "Rust supports memory-safe concurrency").await;
    let req = CreateConceptRequest {
        title: "Cooking".to_string(),
        content: "Pasta carbonara needs eggs and pancetta".to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![
            "mem-rust-a".to_string(),
            "mem-rust-b".to_string(),
            "mem-rust-c".to_string(),
        ],
        creation_kind: None,
        workspace: None,
    };
    // Hallucination guard should reject (cos sim < 0.6)
    assert!(matches!(
        create_page(&db, req, "test", None).await,
        Err(WenlanError::Validation(_))
    ));
}

#[tokio::test]
async fn create_page_happy_path() {
    let (db, _dir) = test_db().await;
    seed_memory(
        &db,
        "mem-rust-happy-a",
        "Rust is a systems programming language with memory safety guarantees",
    )
    .await;
    seed_memory(
        &db,
        "mem-rust-happy-b",
        "Rust provides ownership and borrowing for memory safety",
    )
    .await;
    seed_memory(
        &db,
        "mem-rust-happy-c",
        "Rust supports systems programming with safe concurrency",
    )
    .await;
    let req = CreateConceptRequest {
        title: "Rust".to_string(),
        content: "Rust is a systems programming language providing memory safety guarantees"
            .to_string(),
        summary: Some("memory-safe systems language".to_string()),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![
            "mem-rust-happy-a".to_string(),
            "mem-rust-happy-b".to_string(),
            "mem-rust-happy-c".to_string(),
        ],
        creation_kind: None,
        workspace: None,
    };
    let result = create_page(&db, req, "test", None).await.unwrap();
    assert!(result.id.starts_with("page_"));
}

#[tokio::test]
async fn create_page_with_floor_rejects_distilled_below_configured_floor() {
    let (db, _dir) = test_db().await;
    seed_memory(
        &db,
        "mem-rust-floor-a",
        "Rust has ownership and borrowing for memory safety",
    )
    .await;
    seed_memory(
        &db,
        "mem-rust-floor-b",
        "Rust uses lifetimes to validate borrowed references",
    )
    .await;
    seed_memory(
        &db,
        "mem-rust-floor-c",
        "Rust tracks reference validity through lifetimes",
    )
    .await;
    let req = CreateConceptRequest {
        title: "Rust Memory Safety".to_string(),
        content: "Rust has ownership, borrowing, lifetimes, reference validity, and memory safety"
            .to_string(),
        summary: Some("Rust memory safety".to_string()),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![
            "mem-rust-floor-a".to_string(),
            "mem-rust-floor-b".to_string(),
            "mem-rust-floor-c".to_string(),
        ],
        creation_kind: Some("distilled".to_string()),
        workspace: None,
    };

    let result = create_page_with_floor(&db, req, "test", None, 4).await;

    match result {
        Err(WenlanError::Validation(message)) => assert_eq!(
            message,
            "distilled page requires at least 4 distinct source memories (got 3)"
        ),
        other => panic!("expected distinct-source floor validation error, got {other:?}"),
    }
}

#[tokio::test]
async fn create_page_counts_distinct_sources_for_distilled_floor() {
    let (db, _dir) = test_db().await;
    seed_memory(
        &db,
        "mem-rust-distinct-a",
        "Rust ownership prevents memory safety bugs",
    )
    .await;
    seed_memory(
        &db,
        "mem-rust-distinct-b",
        "Rust borrowing validates references at compile time",
    )
    .await;
    let req = CreateConceptRequest {
        title: "Rust Safety".to_string(),
        content: "Rust ownership and borrowing validate memory-safe references".to_string(),
        summary: Some("Rust source floor".to_string()),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![
            "mem-rust-distinct-a".to_string(),
            "mem-rust-distinct-a".to_string(),
            "mem-rust-distinct-b".to_string(),
        ],
        creation_kind: Some("distilled".to_string()),
        workspace: None,
    };

    let result = create_page(&db, req, "test", None).await;

    match result {
        Err(WenlanError::Validation(message)) => assert_eq!(
            message,
            "distilled page requires at least 3 distinct source memories (got 2)"
        ),
        other => panic!("expected distinct-source floor validation error, got {other:?}"),
    }
}

#[tokio::test]
async fn create_page_allows_authored_below_distilled_floor() {
    let (db, _dir) = test_db().await;
    seed_memory(
        &db,
        "mem-rust-authored-a",
        "Rust ownership prevents memory safety bugs",
    )
    .await;
    let req = CreateConceptRequest {
        title: "Rust Authored Note".to_string(),
        content: "Rust ownership prevents memory safety bugs".to_string(),
        summary: Some("Rust authored page".to_string()),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec!["mem-rust-authored-a".to_string()],
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };

    let result = create_page(&db, req, "test", None).await.unwrap();

    assert!(result.id.starts_with("page_"));
}

#[tokio::test]
async fn create_page_rejects_zero_source_distilled_with_preexisting_message() {
    let (db, _dir) = test_db().await;
    let req = CreateConceptRequest {
        title: "Rust".to_string(),
        content: "Rust is a systems programming language".to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![],
        creation_kind: Some("distilled".to_string()),
        workspace: None,
    };

    let result = create_page(&db, req, "test", None).await;

    match result {
        Err(WenlanError::Validation(message)) => assert_eq!(
            message, "distilled page must cite at least one source memory",
            "zero-source distilled must keep the pre-existing message, not the distinct-source floor message"
        ),
        other => panic!("expected zero-source validation error, got {other:?}"),
    }
}

#[tokio::test]
async fn create_page_allows_authored_with_zero_sources() {
    let (db, _dir) = test_db().await;
    let req = CreateConceptRequest {
        title: "Rust Authored Note".to_string(),
        content: "Rust ownership prevents memory safety bugs".to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![],
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };

    let result = create_page(&db, req, "test", None).await.unwrap();

    assert!(result.id.starts_with("page_"));
}

#[tokio::test]
async fn create_page_rejects_reserved_delimiter_before_projection_or_db_mutation() {
    use crate::export::provenance::SOURCES_BLOCK_START;

    let (db, _dir) = test_db().await;
    let knowledge = tempfile::tempdir().unwrap();
    let req = CreateConceptRequest {
        title: "Rejected authored page".to_string(),
        content: format!("before {SOURCES_BLOCK_START} after"),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: Vec::new(),
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };

    let error = create_page(&db, req, "test", Some(knowledge.path()))
        .await
        .unwrap_err();
    assert!(matches!(error, WenlanError::Validation(_)));
    assert!(db.list_pages("active", 10, 0).await.unwrap().is_empty());
    assert!(
        std::fs::read_dir(knowledge.path())
            .unwrap()
            .next()
            .is_none(),
        "validation must happen before projection creates any artifact"
    );
}

#[tokio::test]
async fn create_page_db_insert_failure_rolls_back_scratch_projection() {
    let (db, _dir) = test_db().await;
    let knowledge = tempfile::tempdir().unwrap();
    let page_id = "page_create_projection_rollback";
    let existing_content = "the existing database page remains authoritative";
    let candidate_content = "the duplicate candidate reaches projection before DB rejection";

    let existing_req = CreateConceptRequest {
        title: "Existing authoritative page".to_string(),
        content: existing_content.to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: Vec::new(),
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };
    page_write(
        &db,
        PageWrite::Create {
            page_id: Some(page_id),
            req: existing_req,
            agent: "test",
            knowledge_path: None,
            page_min_cluster_size: 3,
            page_match_threshold: 0.86,
            citations_json: None,
        },
    )
    .await
    .unwrap();
    let existing = db.get_page(page_id).await.unwrap().unwrap();

    let mut projected_candidate = existing.clone();
    projected_candidate.title = "Duplicate projected candidate".to_string();
    projected_candidate.content = candidate_content.to_string();
    let projection = crate::export::knowledge::KnowledgeProjectionWrite::new(
        knowledge.path().to_path_buf(),
        &db,
    );
    let filename = projection
        .write_page_gated(&db, &projected_candidate)
        .await
        .expect("scratch projection write must be valid")
        .expect("generation zero must permit the positive-control projection");
    let projected_path = knowledge.path().join(filename);
    assert!(
        projected_path.is_file(),
        "positive control: the same scratch projection must be writable"
    );
    projection.remove_page(page_id).unwrap();
    assert!(
        !projected_path.exists(),
        "positive control cleanup must establish an empty pre-trigger path"
    );
    drop(projection);

    let duplicate_req = CreateConceptRequest {
        title: projected_candidate.title.clone(),
        content: candidate_content.to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: Vec::new(),
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };
    let (parked_tx, parked_rx) = tokio::sync::oneshot::channel();
    let (resume_tx, resume_rx) = tokio::sync::oneshot::channel();
    *page_create::CREATE_PAGE_PRE_DB_GATE.lock().unwrap() =
        Some((page_id.to_string(), parked_tx, resume_rx));

    let create = page_write(
        &db,
        PageWrite::Create {
            page_id: Some(page_id),
            req: duplicate_req,
            agent: "test",
            knowledge_path: Some(knowledge.path()),
            page_min_cluster_size: 3,
            page_match_threshold: 0.86,
            citations_json: None,
        },
    );
    let observe_projection = async {
        tokio::time::timeout(std::time::Duration::from_secs(5), parked_rx)
            .await
            .expect("production create must reach the post-projection pause")
            .expect("production create dropped the post-projection pause");
        let markdown = std::fs::read_to_string(&projected_path)
            .expect("production create must have written its candidate projection");
        assert!(
            markdown.contains(candidate_content),
            "paused production projection must contain the duplicate candidate"
        );
        resume_tx
            .send(())
            .expect("paused production create must still be waiting");
    };
    let (result, ()) = tokio::join!(create, observe_projection);
    let error = result.expect_err("the duplicate page id must fail after the projection write");
    assert!(
        matches!(error, WenlanError::VectorDb(_)),
        "duplicate DB insert must surface its storage error, got {error:?}"
    );
    assert!(
        !projected_path.exists(),
        "DB failure must roll back the just-written markdown projection"
    );

    let after = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(after.title, existing.title);
    assert_eq!(after.content, existing_content);
    assert_eq!(after.version, existing.version);
    assert_eq!(after.source_memory_ids, existing.source_memory_ids);
    assert_eq!(db.list_pages("active", 10, 0).await.unwrap().len(), 1);
}

#[tokio::test]
async fn create_page_rejects_reserved_delimiter_before_distilled_dedup_attachment() {
    use crate::export::provenance::SOURCES_BLOCK_END;

    let (db, _dir) = test_db().await;
    let existing_source = "mem-reserved-dedup-existing";
    let candidate_source = "mem-reserved-dedup-candidate";
    let grounded = "Rust workspace members share package metadata and one dependency lockfile";
    seed_memory(&db, existing_source, grounded).await;
    seed_memory(&db, candidate_source, grounded).await;
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page_with_kind(
        "page_reserved_dedup_existing",
        "Rust Workspace",
        None,
        grounded,
        None,
        Some("work"),
        &[existing_source],
        &now,
        "distilled",
        "confirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();
    let before = db
        .get_page("page_reserved_dedup_existing")
        .await
        .unwrap()
        .unwrap();
    let evidence_before = db
        .get_page_evidence("page_reserved_dedup_existing")
        .await
        .unwrap();
    let history_before = db
        .list_page_history("page_reserved_dedup_existing", 10)
        .await
        .unwrap();
    let req = CreateConceptRequest {
        title: "Rust Workspace".to_string(),
        content: format!("{grounded}\n\n{SOURCES_BLOCK_END}"),
        summary: None,
        entity_id: None,
        space: (Some("work".to_string())).into(),
        source_memory_ids: vec![candidate_source.to_string()],
        creation_kind: Some("distilled".to_string()),
        workspace: Some("work".to_string()),
    };

    let error = create_page_with_tuning(&db, req, "test", None, 1, -1.0)
        .await
        .unwrap_err();
    assert!(matches!(error, WenlanError::Validation(_)));
    let after = db
        .get_page("page_reserved_dedup_existing")
        .await
        .unwrap()
        .unwrap();
    let evidence_after = db
        .get_page_evidence("page_reserved_dedup_existing")
        .await
        .unwrap();
    let history_after = db
        .list_page_history("page_reserved_dedup_existing", 10)
        .await
        .unwrap();
    assert_eq!(after.content, before.content);
    assert_eq!(after.version, before.version);
    assert_eq!(after.source_memory_ids, before.source_memory_ids);
    assert_eq!(after.stale_reason, before.stale_reason);
    assert_eq!(evidence_after.len(), evidence_before.len());
    assert_eq!(history_after.len(), history_before.len());
}

#[tokio::test]
async fn create_page_borns_distilled_unconfirmed() {
    let (db, _dir) = test_db().await;
    let docs = [
        (
            "mem-rust-birth-a",
            "Rust ownership helps prevent memory safety bugs",
        ),
        (
            "mem-rust-birth-b",
            "Rust borrowing validates references at compile time",
        ),
        (
            "mem-rust-birth-c",
            "Rust lifetimes describe how long references remain valid",
        ),
    ]
    .into_iter()
    .map(|(source_id, content)| crate::sources::RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        content: content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confidence: Some(0.9),
        ..Default::default()
    })
    .collect::<Vec<_>>();
    db.upsert_documents(docs).await.unwrap();
    let req = CreateConceptRequest {
        title: "Rust References".to_string(),
        content: "Rust ownership, borrowing, and lifetimes keep references memory safe".to_string(),
        summary: Some("Rust reference safety".to_string()),
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![
            "mem-rust-birth-a".to_string(),
            "mem-rust-birth-b".to_string(),
            "mem-rust-birth-c".to_string(),
        ],
        creation_kind: Some("distilled".to_string()),
        workspace: None,
    };

    let result = create_page(&db, req, "test", None).await.unwrap();
    let page = db.get_page(&result.id).await.unwrap().unwrap();

    assert_eq!(page.review_status, "unconfirmed");
    let keep_cards: Vec<_> = db
        .get_pending_refinements()
        .await
        .unwrap()
        .into_iter()
        .filter(|proposal| {
            proposal.action == "page_keep_or_archive"
                && proposal.source_ids.iter().any(|id| id == &result.id)
        })
        .collect();
    assert_eq!(
        keep_cards.len(),
        1,
        "distilled page birth must mint exactly one keep/archive card"
    );
    let payload = keep_cards[0].payload.as_deref().unwrap_or_default();
    assert!(
        payload.contains("\"source_count\":3"),
        "keep/archive card should preserve source count, got {payload}"
    );
}

#[tokio::test]
async fn create_page_borns_authored_without_keep_card() {
    let (db, _dir) = test_db().await;
    let req = CreateConceptRequest {
        title: "Authored Rust Notes".to_string(),
        content: "Authored notes about Rust references and workspace conventions.".to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![],
        creation_kind: Some("authored".to_string()),
        workspace: None,
    };

    let result = create_page(&db, req, "test", None).await.unwrap();

    let keep_cards: Vec<_> = db
        .get_pending_refinements()
        .await
        .unwrap()
        .into_iter()
        .filter(|proposal| {
            proposal.action == "page_keep_or_archive"
                && proposal.source_ids.iter().any(|id| id == &result.id)
        })
        .collect();
    assert!(
        keep_cards.is_empty(),
        "authored page birth must not mint a keep/archive card"
    );
}

#[tokio::test]
async fn create_page_attaches_same_workspace_near_duplicate_without_new_page() {
    let (db, _dir) = test_db().await;
    let existing_sources = [
        (
            "mem-pagewrite-existing-a",
            "Rust workspaces can share a single Cargo lockfile across related crates",
        ),
        (
            "mem-pagewrite-existing-b",
            "Rust workspace members inherit shared package metadata from the root",
        ),
        (
            "mem-pagewrite-existing-c",
            "Rust workspace builds can check all member crates together",
        ),
    ];
    for (source_id, content) in existing_sources {
        seed_memory(&db, source_id, content).await;
    }
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page_with_kind(
        "page_pagewrite_existing",
        "Rust Workspace Operations",
        Some("Rust workspace operations"),
        "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
        None,
        Some("recap"),
        &[
            "mem-pagewrite-existing-a",
            "mem-pagewrite-existing-b",
            "mem-pagewrite-existing-c",
        ],
        &now,
        "distilled",
        "confirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();

    for (source_id, content) in [
        (
            "mem-pagewrite-candidate-a",
            "Rust workspaces share one Cargo lockfile for related crates",
        ),
        (
            "mem-pagewrite-candidate-b",
            "Rust workspace members can inherit shared package metadata",
        ),
        (
            "mem-pagewrite-candidate-c",
            "Rust workspace checks can validate every member crate together",
        ),
    ] {
        seed_memory(&db, source_id, content).await;
    }
    let before_pages = db.list_pages("active", 10, 0).await.unwrap();
    assert_eq!(before_pages.len(), 1, "precondition: one active page");
    let req = CreateConceptRequest {
        title: "Rust Workspace Operations".to_string(),
        content: "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks"
            .to_string(),
        summary: Some("Rust workspace operations".to_string()),
        entity_id: None,
        space: (Some("recap".to_string())).into(),
        source_memory_ids: vec![
            "mem-pagewrite-candidate-a".to_string(),
            "mem-pagewrite-candidate-b".to_string(),
            "mem-pagewrite-candidate-c".to_string(),
        ],
        creation_kind: Some("distilled".to_string()),
        workspace: Some("work".to_string()),
    };

    let result = create_page(&db, req, "test", None).await.unwrap();

    assert_eq!(
        result.id, "page_pagewrite_existing",
        "near-duplicate create must resolve to the existing page id"
    );
    let result_json = serde_json::to_value(&result).unwrap();
    assert_eq!(
        result_json.get("attached_to").and_then(|v| v.as_str()),
        Some("page_pagewrite_existing"),
        "response must expose the attach target"
    );
    let after_pages = db.list_pages("active", 10, 0).await.unwrap();
    assert_eq!(
        after_pages.len(),
        1,
        "same-workspace near-duplicate must not mint a second page"
    );
    let evidence = db
        .get_page_evidence("page_pagewrite_existing")
        .await
        .unwrap();
    let locators = evidence
        .iter()
        .filter(|ev| ev.source_kind == "memory")
        .filter_map(|ev| ev.locator.as_deref())
        .collect::<HashSet<_>>();
    for expected in [
        "mem-pagewrite-existing-a",
        "mem-pagewrite-existing-b",
        "mem-pagewrite-existing-c",
        "mem-pagewrite-candidate-a",
        "mem-pagewrite-candidate-b",
        "mem-pagewrite-candidate-c",
    ] {
        assert!(
            locators.contains(expected),
            "page_evidence must include {expected}; got {locators:?}"
        );
    }
}

#[tokio::test]
async fn create_page_does_not_attach_no_space_candidate_to_workspace_page() {
    let (db, _dir) = test_db().await;
    for (source_id, content) in [
        (
            "mem-pagewrite-cross-existing-a",
            "Rust workspaces can share a single Cargo lockfile across related crates",
        ),
        (
            "mem-pagewrite-cross-existing-b",
            "Rust workspace members inherit shared package metadata from the root",
        ),
        (
            "mem-pagewrite-cross-existing-c",
            "Rust workspace builds can check all member crates together",
        ),
        (
            "mem-pagewrite-cross-candidate-a",
            "Rust workspaces share one Cargo lockfile for related crates",
        ),
        (
            "mem-pagewrite-cross-candidate-b",
            "Rust workspace members can inherit shared package metadata",
        ),
        (
            "mem-pagewrite-cross-candidate-c",
            "Rust workspace checks can validate every member crate together",
        ),
    ] {
        seed_memory(&db, source_id, content).await;
    }
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page_with_kind(
        "page_pagewrite_cross_existing",
        "Rust Workspace Operations",
        Some("Rust workspace operations"),
        "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
        None,
        Some("recap"),
        &[
            "mem-pagewrite-cross-existing-a",
            "mem-pagewrite-cross-existing-b",
            "mem-pagewrite-cross-existing-c",
        ],
        &now,
        "distilled",
        "confirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();
    db.insert_page_with_kind(
        "page_pagewrite_uncategorized_existing",
        "Rust Workspace Operations",
        Some("Rust workspace operations"),
        "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
        None,
        None,
        &[
            "mem-pagewrite-cross-existing-a",
            "mem-pagewrite-cross-existing-b",
            "mem-pagewrite-cross-existing-c",
        ],
        &now,
        "distilled",
        "confirmed",
        None,
        None,
    )
    .await
    .unwrap();
    let req = CreateConceptRequest {
        title: "Rust Workspace Operations".to_string(),
        content: "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks"
            .to_string(),
        summary: Some("Rust workspace operations".to_string()),
        entity_id: None,
        space: wenlan_types::WriteSpaceTarget::Uncategorized,
        source_memory_ids: vec![
            "mem-pagewrite-cross-candidate-a".to_string(),
            "mem-pagewrite-cross-candidate-b".to_string(),
            "mem-pagewrite-cross-candidate-c".to_string(),
        ],
        creation_kind: Some("distilled".to_string()),
        workspace: None,
    };

    let result = create_page(&db, req, "test", None).await.unwrap();

    assert_eq!(
        result.attached_to.as_deref(),
        Some("page_pagewrite_uncategorized_existing"),
        "Uncategorized dedup must attach to the Uncategorized page"
    );
    assert_ne!(result.id, "page_pagewrite_cross_existing");
    let pages = db.list_pages("active", 10, 0).await.unwrap();
    assert_eq!(
        pages.len(),
        2,
        "Uncategorized dedup must not mint a third page or attach across Space"
    );
}

#[tokio::test]
async fn create_page_does_not_attach_different_space_candidate() {
    let (db, _dir) = test_db().await;
    for (source_id, content) in [
        (
            "mem-pagewrite-diffspace-existing-a",
            "Rust workspaces can share a single Cargo lockfile across related crates",
        ),
        (
            "mem-pagewrite-diffspace-existing-b",
            "Rust workspace members inherit shared package metadata from the root",
        ),
        (
            "mem-pagewrite-diffspace-existing-c",
            "Rust workspace builds can check all member crates together",
        ),
        (
            "mem-pagewrite-diffspace-candidate-a",
            "Rust workspaces share one Cargo lockfile for related crates",
        ),
        (
            "mem-pagewrite-diffspace-candidate-b",
            "Rust workspace members can inherit shared package metadata",
        ),
        (
            "mem-pagewrite-diffspace-candidate-c",
            "Rust workspace checks can validate every member crate together",
        ),
    ] {
        seed_memory(&db, source_id, content).await;
    }
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page_with_kind(
        "page_pagewrite_diffspace_existing",
        "Rust Workspace Operations",
        Some("Rust workspace operations"),
        "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks",
        None,
        Some("recap"),
        &[
            "mem-pagewrite-diffspace-existing-a",
            "mem-pagewrite-diffspace-existing-b",
            "mem-pagewrite-diffspace-existing-c",
        ],
        &now,
        "distilled",
        "confirmed",
        Some("work"),
        None,
    )
    .await
    .unwrap();
    // Same content, but scoped to a DIFFERENT workspace ("personal") — the
    // scoped matcher's `space = ?` filter (M1 honest columns) must exclude
    // the "work" page, so this mints a new page rather than attaching.
    let req = CreateConceptRequest {
        title: "Rust Workspace Operations".to_string(),
        content: "Rust workspaces share Cargo lockfiles, inherited metadata, and all-crate checks"
            .to_string(),
        summary: Some("Rust workspace operations".to_string()),
        entity_id: None,
        space: (Some("recap".to_string())).into(),
        source_memory_ids: vec![
            "mem-pagewrite-diffspace-candidate-a".to_string(),
            "mem-pagewrite-diffspace-candidate-b".to_string(),
            "mem-pagewrite-diffspace-candidate-c".to_string(),
        ],
        creation_kind: Some("distilled".to_string()),
        workspace: Some("personal".to_string()),
    };

    let result = create_page(&db, req, "test", None).await.unwrap();

    assert_ne!(
        result.id, "page_pagewrite_diffspace_existing",
        "space-scoped dedup must not attach a different-space candidate to a work page"
    );
    assert_eq!(
        result.attached_to, None,
        "different-space create must report a new page, not an attachment"
    );
    let pages = db.list_pages("active", 10, 0).await.unwrap();
    assert_eq!(
        pages.len(),
        2,
        "different-space near-duplicate must mint a second page"
    );
}

// ── update_page ──────────────────────────────────────────────────────────

/// Helper: seed a memory and return its source_id.
async fn seed_memory(db: &MemoryDB, source_id: &str, content: &str) {
    let doc = crate::sources::RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        content: content.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confidence: Some(0.9),
        ..Default::default()
    };
    db.upsert_documents(vec![doc]).await.unwrap();
}

/// Helper: create a page via create_page for an existing memory, return page id.
async fn seed_page(db: &MemoryDB, source_id: &str, content: &str) -> String {
    let req = CreateConceptRequest {
        title: format!("Page {source_id}"),
        content: content.to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![source_id.to_string()],
        creation_kind: Some("research".to_string()),
        workspace: None,
    };
    create_page(db, req, "test", None).await.unwrap().id
}

#[tokio::test]
async fn page_write_attach_marks_page_source_updated() {
    let (db, _dir) = test_db().await;
    seed_memory(&db, "mem_attach_a", "first explicit source").await;
    seed_memory(&db, "mem_attach_b", "second explicit source").await;
    let page_id = seed_page(&db, "mem_attach_a", "first explicit source").await;

    page_write(
        &db,
        PageWrite::Attach {
            page_id: &page_id,
            source_memory_ids: &["mem_attach_b".to_string()],
            link_reason: "topic_overlap",
            agent: "test",
        },
    )
    .await
    .unwrap();

    assert_eq!(
        db.get_page_stale_reason(&page_id).await.unwrap(),
        Some("source_updated".to_string())
    );
}

#[tokio::test]
async fn update_page_round_trip() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-rpt-1";
    let content_v1 = "Rust is a systems language with memory safety";
    seed_memory(&db, mem_id, content_v1).await;
    let page_id = seed_page(&db, mem_id, content_v1).await;

    // First update → version=2
    let content_v2 = "Rust is a systems language with memory safety and zero-cost abstractions";
    let req2 = UpdatePageRequest {
        content: content_v2.to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let r2 = update_page(&db, &page_id, req2, "re_distill", false, None, None)
        .await
        .unwrap();
    assert_eq!(r2.id, page_id);

    // Second update → version=3
    let content_v3 = "Rust is a systems language with memory safety, zero-cost abstractions and concurrency without data races";
    let req3 = UpdatePageRequest {
        content: content_v3.to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let r3 = update_page(&db, &page_id, req3, "re_distill", false, None, None)
        .await
        .unwrap();

    // Check page version=3
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.version, 3);

    // Changelog has 2 entries (v1→v2 and v2→v3)
    let cl = db.get_page_changelog(&page_id).await.unwrap();
    let entries: Vec<serde_json::Value> = serde_json::from_str(&cl).unwrap();
    assert_eq!(entries.len(), 2, "expected 2 changelog entries");
    assert!(
        !r3.warnings.is_empty(),
        "warnings should carry delta summary"
    );
}

#[test]
fn non_stale_page_write_uses_loaded_version_cas() {
    let source = include_str!("page_update.rs");
    let update_impl = source
        .split("async fn update_page_impl")
        .nth(1)
        .expect("update_page_impl source");
    assert!(
        update_impl.contains("try_update_page_content_with_changelog_at_version"),
        "PageWrite must commit against the current.version snapshot it already loaded"
    );
}

#[tokio::test]
async fn update_page_cas_skips_when_not_stale() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-cas-skip";
    let content = "Rust is a systems language with memory safety";
    seed_memory(&db, mem_id, content).await;
    let page_id = seed_page(&db, mem_id, content).await;

    // Page has no stale_reason — CAS with require_stale=true should skip
    let req = UpdatePageRequest {
        content: "Rust is a systems language with memory safety and performance".to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "re_distill", true, None, None)
        .await
        .unwrap();

    // Version unchanged (page stays at v1)
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.version, 1, "version should not change when CAS skips");
    assert!(!result.wrote, "wrote must be false when CAS skips");
    assert!(result.warnings.is_empty(), "no warnings on CAS skip");
}

#[tokio::test]
async fn refresh_revision_cas_preserves_source_attached_during_synthesis() {
    let (db, _dir) = test_db().await;
    seed_memory(&db, "mem-refresh-a", "first source").await;
    seed_memory(&db, "mem-refresh-b", "second source").await;
    let page_id = seed_page(&db, "mem-refresh-a", "first source").await;
    let expected_revision = db.get_page_source_revision(&page_id).await.unwrap();

    page_write(
        &db,
        PageWrite::Attach {
            page_id: &page_id,
            source_memory_ids: &["mem-refresh-b".to_string()],
            link_reason: "attached_during_synthesis",
            agent: "test",
        },
    )
    .await
    .unwrap();
    let result = update_page_at_source_revision(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "compiled from only the first source".to_string(),
            source_memory_ids: vec!["mem-refresh-a".to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        true,
        expected_revision,
        None,
        None,
    )
    .await
    .unwrap();

    assert!(!result.wrote);
    assert!(!result.acknowledged);
    assert_eq!(
        db.get_page_sources(&page_id)
            .await
            .unwrap()
            .into_iter()
            .map(|source| source.memory_source_id)
            .collect::<std::collections::BTreeSet<_>>(),
        ["mem-refresh-a".to_string(), "mem-refresh-b".to_string()]
            .into_iter()
            .collect()
    );
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.version, 1);
    assert_eq!(page.stale_reason.as_deref(), Some("source_updated"));
}

/// G2(c): `handle_refresh_page`'s non-human (machine) branch used to write
/// through the generic, unfenced `update_page` -- a source attached between
/// the route's own `source_revision` read and its write would be silently
/// clobbered by the refresh, the same class of race
/// `refresh_revision_cas_preserves_source_attached_during_synthesis` above
/// proves for the `re_distill` path. This exercises the exact wrapper and
/// call shape `handle_refresh_page` now uses (`edited_by: "agent_refresh"`,
/// `require_stale: false`) -- the shape that had zero CAS protection before
/// the fix, since the route used to pass no fence at all.
///
/// A true interleaved race cannot be driven from wenlan-server's own
/// integration tests: the deterministic pause hook this crate uses for
/// races (`PRE_WRITE_GATE`) is `#[cfg(test)]` and `pub(crate)` to
/// wenlan-core, so it is not compiled into the rlib wenlan-server links
/// against even in wenlan-server's own test builds. This proves the same
/// staleness at the seam the route calls into instead -- see the round-7
/// report for why the route-level test was not attempted.
#[tokio::test]
async fn agent_refresh_cas_rejects_a_source_attached_after_the_route_read_its_fence() {
    let (db, _dir) = test_db().await;
    seed_memory(&db, "mem-agent-refresh-a", "first source").await;
    seed_memory(&db, "mem-agent-refresh-b", "second source").await;
    let page_id = seed_page(&db, "mem-agent-refresh-a", "first source").await;
    let source_revision = db.get_page_source_revision(&page_id).await.unwrap();

    // A source attach lands in the window between the route's own counter
    // read and its write -- exactly the race the fenced write must catch.
    page_write(
        &db,
        PageWrite::Attach {
            page_id: &page_id,
            source_memory_ids: &["mem-agent-refresh-b".to_string()],
            link_reason: "attached_during_agent_refresh",
            agent: "test",
        },
    )
    .await
    .unwrap();

    let result = update_page_at_source_revision(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "refreshed body computed before the source attach".to_string(),
            source_memory_ids: vec!["mem-agent-refresh-a".to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "agent_refresh",
        false,
        source_revision,
        None,
        None,
    )
    .await
    .unwrap();

    assert!(
        !result.wrote,
        "a stale agent_refresh must not overwrite a page whose sources moved after the route read its fence"
    );
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        page.content, "first source",
        "the pre-race content must survive"
    );
    assert_eq!(
        db.get_page_sources(&page_id)
            .await
            .unwrap()
            .into_iter()
            .map(|source| source.memory_source_id)
            .collect::<std::collections::BTreeSet<_>>(),
        [
            "mem-agent-refresh-a".to_string(),
            "mem-agent-refresh-b".to_string()
        ]
        .into_iter()
        .collect(),
        "the attach that won the race must survive"
    );
}

#[tokio::test]
async fn identical_refresh_cannot_acknowledge_a_new_source_revision() {
    let (db, _dir) = test_db().await;
    seed_memory(&db, "mem-ack-a", "first source").await;
    seed_memory(&db, "mem-ack-b", "second source").await;
    let page_id = seed_page(&db, "mem-ack-a", "first source").await;
    db.set_page_stale(&page_id, "source_updated").await.unwrap();
    let expected_revision = db.get_page_source_revision(&page_id).await.unwrap();
    db.link_page_source(&page_id, "mem-ack-b", "attached_during_synthesis")
        .await
        .unwrap();

    let result = update_page_at_source_revision(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "first source".to_string(),
            source_memory_ids: vec!["mem-ack-a".to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        true,
        expected_revision,
        None,
        None,
    )
    .await
    .unwrap();

    assert!(!result.wrote);
    assert!(!result.acknowledged);
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.stale_reason.as_deref(), Some("source_updated"));
    assert_eq!(page.sources_updated_count, 1);
    assert_eq!(
        db.get_page_source_revision(&page_id).await.unwrap(),
        expected_revision + 1
    );
}

#[tokio::test]
async fn refresh_revision_cas_rejects_sources_updated_count_aba() {
    let (db, _dir) = test_db().await;
    seed_memory(&db, "mem-aba-a", "first source").await;
    seed_memory(&db, "mem-aba-b", "second source").await;
    seed_memory(&db, "mem-aba-c", "third source").await;
    let page_id = seed_page(&db, "mem-aba-a", "first source").await;

    db.link_page_source(&page_id, "mem-aba-b", "first_pending_attach")
        .await
        .unwrap();
    let stale_snapshot = db.get_page(&page_id).await.unwrap().unwrap();
    let expected_revision = db.get_page_source_revision(&page_id).await.unwrap();
    assert_eq!(stale_snapshot.sources_updated_count, 1);

    assert!(db
        .acknowledge_page_compile(&page_id, stale_snapshot.version, Some(expected_revision))
        .await
        .unwrap());
    db.link_page_source(&page_id, "mem-aba-c", "attached_after_ack")
        .await
        .unwrap();
    assert_eq!(
        db.get_page(&page_id)
            .await
            .unwrap()
            .unwrap()
            .sources_updated_count,
        1
    );

    let result = update_page_at_source_revision(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "compiled without the third source".to_string(),
            source_memory_ids: vec!["mem-aba-a".to_string(), "mem-aba-b".to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        true,
        expected_revision,
        None,
        None,
    )
    .await
    .unwrap();

    assert!(!result.wrote);
    assert_eq!(
        db.get_page_sources(&page_id)
            .await
            .unwrap()
            .into_iter()
            .map(|source| source.memory_source_id)
            .collect::<std::collections::BTreeSet<_>>(),
        [
            "mem-aba-a".to_string(),
            "mem-aba-b".to_string(),
            "mem-aba-c".to_string(),
        ]
        .into_iter()
        .collect()
    );
}

#[tokio::test]
async fn verified_identical_refresh_acknowledges_compile_without_retry() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-cas-identical";
    let content = "Rust ownership keeps references memory safe";
    seed_memory(&db, mem_id, content).await;
    let page_id = seed_page(&db, mem_id, content).await;
    db.set_page_stale(&page_id, "source_updated").await.unwrap();
    let before = db.get_page(&page_id).await.unwrap().unwrap();

    let result = update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        true,
        None,
        None,
    )
    .await
    .unwrap();
    let after = db.get_page(&page_id).await.unwrap().unwrap();

    assert!(!result.wrote);
    assert!(result.acknowledged);
    assert_eq!(after.version, before.version);
    assert!(after.stale_reason.is_none());
    assert!(after.last_compiled >= before.last_compiled);
}

#[tokio::test]
async fn update_page_cas_writes_when_stale() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-cas-write";
    let content = "Rust is a systems language with memory safety";
    seed_memory(&db, mem_id, content).await;
    let page_id = seed_page(&db, mem_id, content).await;

    // Mark page stale
    db.set_page_stale(&page_id, "source_updated").await.unwrap();

    // CAS with require_stale=true should write when stale
    let new_content = "Rust is a systems language with memory safety and ownership model";
    let req = UpdatePageRequest {
        content: new_content.to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "re_distill", true, None, None)
        .await
        .unwrap();

    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.version, 2, "version should bump on CAS write");
    assert!(
        page.stale_reason.is_none(),
        "successful CAS clears staleness"
    );
    assert_eq!(page.sources_updated_count, 0);
    assert!(result.wrote, "wrote must be true on CAS write");
    assert!(
        !result.warnings.is_empty(),
        "warnings should carry delta summary"
    );
}

#[tokio::test]
async fn update_page_hallucination_guard_manual_edit_rejects() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-guard-reject";
    let rust_content = "Rust is a systems programming language with memory safety";
    seed_memory(&db, mem_id, rust_content).await;
    let page_id = seed_page(&db, mem_id, rust_content).await;

    // Body completely unrelated to the Rust memory source
    let req = UpdatePageRequest {
        content: "Pasta carbonara needs eggs pancetta and pecorino romano cheese".to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "manual_edit", false, None, None).await;
    assert!(
        matches!(result, Err(WenlanError::Validation(_))),
        "hallucination guard should reject manual_edit with unrelated body"
    );
}

#[tokio::test]
async fn update_page_skip_guard_re_distill() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-guard-skip";
    let rust_content = "Rust is a systems programming language with memory safety";
    seed_memory(&db, mem_id, rust_content).await;
    let page_id = seed_page(&db, mem_id, rust_content).await;

    // Same unrelated body — but re_distill skips the guard
    let req = UpdatePageRequest {
        content: "Pasta carbonara needs eggs pancetta and pecorino romano cheese".to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    // Should succeed without hallucination check
    update_page(&db, &page_id, req, "re_distill", false, None, None)
        .await
        .unwrap();
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.version, 2);
}

#[tokio::test]
async fn update_page_user_edit_flag_set() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-flag-test";
    let content = "Rust is a systems language with memory safety features and ownership";
    seed_memory(&db, mem_id, content).await;
    let page_id = seed_page(&db, mem_id, content).await;

    // fs_edit should set user_edited=1
    let req = UpdatePageRequest {
        content: "Rust is a systems language with memory safety features, ownership and borrowing"
            .to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    update_page(&db, &page_id, req, "fs_edit", false, None, None)
        .await
        .unwrap();
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert!(page.user_edited, "user_edited should be true for fs_edit");
}

#[tokio::test]
async fn update_page_fs_edit_with_nonexistent_source_succeeds() {
    // Regression: update_page must not reject fs_edit (or any daemon-internal
    // caller) when source_memory_ids references a memory that no longer exists.
    // The source list is carried forward from the existing page; re-validating
    // on update would break page_watcher for pages whose sources were pruned.
    // Insert the page directly (bypassing create_page validation) to simulate
    // a page whose source was valid at creation but since pruned.
    let (db, _dir) = test_db().await;
    let ghost_source = "mem-ghost-pruned";
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_ghost_src_test";
    db.insert_page(
        page_id,
        "Ghost Source Page",
        None,
        "Rust is a systems language with memory safety",
        None,
        None,
        &[ghost_source],
        &now,
    )
    .await
    .unwrap();

    // fs_edit carrying the non-existent source id must succeed.
    let req = UpdatePageRequest {
        content: "Rust is a systems language with memory safety (user edited)".to_string(),
        source_memory_ids: vec![ghost_source.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    update_page(&db, page_id, req, "fs_edit", false, None, None)
        .await
        .unwrap();
    let page = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(page.version, 2);
}

#[tokio::test]
async fn update_page_warnings_carry_delta() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-warn-delta";
    let content = "Rust is a systems language";
    seed_memory(&db, mem_id, content).await;
    let page_id = seed_page(&db, mem_id, content).await;

    let new_content = "Rust is a systems language with memory safety and zero-cost abstractions for high performance systems programming";
    let req = UpdatePageRequest {
        content: new_content.to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "re_distill", false, None, None)
        .await
        .unwrap();

    assert!(
        !result.warnings.is_empty(),
        "warnings should be non-empty when content changes"
    );
    let warning = &result.warnings[0];
    assert!(
        warning.contains("v1") && warning.contains("v2"),
        "warning should reference version transition, got: {warning}"
    );
}

#[tokio::test]
async fn update_page_idempotent_warnings_shape() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-idem-shape";
    let content_v1 = "Rust is a systems language with ownership model";
    seed_memory(&db, mem_id, content_v1).await;
    let page_id = seed_page(&db, mem_id, content_v1).await;

    // First call: v1 → v2
    let r1 = update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "Rust is a systems language with ownership model and borrow checker"
                .to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(r1.wrote, "first call should write");
    assert_eq!(
        r1.warnings.len(),
        1,
        "first call should produce exactly one warning"
    );
    let w1 = &r1.warnings[0];
    assert!(w1.starts_with('v'), "warning must start with 'v': {w1}");
    assert!(w1.contains('→'), "warning must contain '→': {w1}");
    assert!(
        w1.contains("v1") && w1.contains("v2"),
        "first warning should show v1 → v2: {w1}"
    );

    // Second call with different content: v2 → v3
    let r2 = update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content:
                "Rust is a systems language with ownership model, borrow checker, and lifetimes"
                    .to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(r2.wrote, "second call should write");
    assert_eq!(
        r2.warnings.len(),
        1,
        "second call should produce exactly one warning"
    );
    let w2 = &r2.warnings[0];
    assert!(w2.starts_with('v'), "warning must start with 'v': {w2}");
    assert!(w2.contains('→'), "warning must contain '→': {w2}");
    assert!(
        w2.contains("v2") && w2.contains("v3"),
        "second warning should show v2 → v3: {w2}"
    );
}

#[tokio::test]
async fn update_page_noop_returns_wrote_false() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-noop-1";
    let content = "Rust is a systems language with memory safety";
    seed_memory(&db, mem_id, content).await;
    let page_id = seed_page(&db, mem_id, content).await;

    // Fetch baseline version before no-op call
    let page_before = db.get_page(&page_id).await.unwrap().unwrap();
    let version_before = page_before.version;

    // Call update_page with identical content and identical sources
    let req = UpdatePageRequest {
        content: content.to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "re_distill", false, None, None)
        .await
        .unwrap();

    assert!(!result.wrote, "identical-content call must not write");
    assert!(result.warnings.is_empty(), "no-op must produce no warnings");

    // Version must be unchanged
    let page_after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        page_after.version, version_before,
        "version must not bump on no-op"
    );
}

#[tokio::test]
async fn page_write_update_user_edited_machine_write_creates_revision_card_without_overwrite() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-pagewrite-owned";
    let source_content = "Rust ownership keeps memory safety rules explicit in systems code";
    seed_memory(&db, mem_id, source_content).await;
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_pagewrite_owned";
    db.insert_page(
        page_id,
        "Rust Ownership",
        None,
        source_content,
        None,
        None,
        &[mem_id],
        &now,
    )
    .await
    .unwrap();

    let human_content =
        "Rust ownership keeps memory safety rules explicit in systems code, with human notes";
    page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: human_content.to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: None,
                caller_id: None,
                operation_id: None,
            },
            edited_by: "fs_edit",
            require_stale: false,
            expected_source_revision: None,
            knowledge_path: None,
            citations: None,
        },
    )
    .await
    .unwrap();

    let before = db.get_page(page_id).await.unwrap().unwrap();
    assert!(
        before.user_edited,
        "precondition: fs_edit marks human ownership"
    );

    let machine_content =
        "Rust ownership lets the compiler enforce memory safety during page refresh";
    let result = page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: machine_content.to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: None,
                caller_id: None,
                operation_id: None,
            },
            edited_by: "re_distill",
            require_stale: false,
            expected_source_revision: None,
            knowledge_path: None,
            citations: None,
        },
    )
    .await
    .unwrap();

    let after = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(result.id, page_id);
    assert!(!result.wrote, "gated PageWrite must report wrote=false");
    assert!(result.gated, "gated PageWrite must expose gated=true");
    assert_eq!(result.attached_to, None);
    assert_eq!(
        result.warnings,
        vec!["human-owned page; staged revision card instead of overwriting content"],
        "gated PageWrite must explain that the page prose was not overwritten"
    );
    assert_eq!(
        after.content, before.content,
        "machine PageWrite must not overwrite human-owned page prose"
    );
    assert_eq!(
        after.content, human_content,
        "machine PageWrite must leave the human-authored bytes unchanged"
    );
    assert_eq!(
        after.source_memory_ids, before.source_memory_ids,
        "gated PageWrite must not mutate the protected page source set"
    );
    assert_eq!(
        after.version, before.version,
        "gated PageWrite must not bump the protected page version"
    );
    assert!(
        after.user_edited,
        "gated PageWrite must preserve the human ownership marker"
    );

    let result_json = serde_json::to_value(&result).unwrap();
    assert_eq!(result_json.get("gated"), Some(&serde_json::json!(true)));
    let revision_card_id = result_json
        .get("revision_card_id")
        .and_then(|v| v.as_str())
        .expect("gated response must include revision_card_id");

    let revisions = db.list_pending_revisions(10).await.unwrap();
    assert_eq!(
        revisions.len(),
        1,
        "gated PageWrite must stage exactly one pending revision card"
    );
    let card = revisions
        .iter()
        .find(|r| r.revision_source_id == revision_card_id)
        .expect("revision card must be visible in pending revisions");
    assert_eq!(card.target_source_id, page_id);
    assert_eq!(card.revision_content, machine_content);
    assert_eq!(card.source_agent.as_deref(), Some("page_write"));

    let conn = db.test_primary_session().await;
    let mut rows = conn
        .query(
            "SELECT source, supersedes, pending_revision, confirmed, stability, \
                    structured_fields, source_text, memory_type \
             FROM memories WHERE source_id = ?1",
            libsql::params![revision_card_id.to_string()],
        )
        .await
        .unwrap();
    let row = rows
        .next()
        .await
        .unwrap()
        .expect("revision card row must be persisted");
    assert_eq!(row.get::<String>(0).unwrap(), "memory");
    assert_eq!(row.get::<String>(1).unwrap(), page_id);
    assert_eq!(row.get::<i64>(2).unwrap(), 1);
    assert_eq!(row.get::<i64>(3).unwrap(), 0);
    assert_eq!(row.get::<String>(4).unwrap(), "new");
    let structured_fields = row.get::<String>(5).unwrap();
    assert_eq!(
        row.get::<Option<String>>(6).unwrap().as_deref(),
        Some(machine_content)
    );
    assert_eq!(row.get::<String>(7).unwrap(), "fact");
    assert!(
        rows.next().await.unwrap().is_none(),
        "revision_card_id must identify one persisted card row"
    );
    drop(rows);
    drop(conn);

    let structured: serde_json::Value = serde_json::from_str(&structured_fields).unwrap();
    assert_eq!(structured["revision_kind"], "page_write");
    assert_eq!(structured["target_kind"], "page");
    assert_eq!(structured["revises_page"], page_id);
    assert_eq!(structured["page_version"], before.version);
    assert_eq!(structured["edited_by"], "re_distill");
    assert_eq!(structured["source_memory_ids"], serde_json::json!([mem_id]));
}

#[tokio::test]
async fn page_write_update_rejects_reserved_delimiter_before_revision_staging() {
    use crate::export::provenance::SOURCES_BLOCK_START;

    let (db, _dir) = test_db().await;
    let mem_id = "mem-pagewrite-reserved-staging";
    let source_content = "Rust ownership keeps memory safety rules explicit in systems code";
    seed_memory(&db, mem_id, source_content).await;
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_pagewrite_reserved_staging";
    db.insert_page_with_kind(
        page_id,
        "Rust Ownership",
        None,
        source_content,
        None,
        None,
        &[mem_id],
        &now,
        "authored",
        "confirmed",
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(page_id).await.unwrap().unwrap();
    let history_before = db.list_page_history(page_id, 10).await.unwrap();
    assert!(db.list_pending_revisions(10).await.unwrap().is_empty());

    let error = page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: format!("{source_content}\n\n{SOURCES_BLOCK_START}"),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: Some(before.version),
                caller_id: None,
                operation_id: None,
            },
            edited_by: "re_distill",
            require_stale: false,
            expected_source_revision: None,
            knowledge_path: None,
            citations: None,
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(error, WenlanError::Validation(_)));

    let after = db.get_page(page_id).await.unwrap().unwrap();
    let history_after = db.list_page_history(page_id, 10).await.unwrap();
    assert_eq!(after.content, before.content);
    assert_eq!(after.version, before.version);
    assert_eq!(after.source_memory_ids, before.source_memory_ids);
    assert_eq!(history_after, history_before);
    assert!(
        db.list_pending_revisions(10).await.unwrap().is_empty(),
        "rejected source must not survive as a pending revision card"
    );
}

#[tokio::test]
async fn gated_refresh_does_not_clear_a_newer_source_revision() {
    let (db, _dir) = test_db().await;
    for (source_id, content) in [
        (
            "mem-gated-revision-a",
            "Rust ownership keeps memory safety rules explicit in systems code",
        ),
        (
            "mem-gated-revision-b",
            "Borrow checking rejects invalid aliasing before runtime",
        ),
    ] {
        seed_memory(&db, source_id, content).await;
    }
    let page_id = seed_page(
        &db,
        "mem-gated-revision-a",
        "Rust ownership keeps memory safety rules explicit in systems code",
    )
    .await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "Human-owned notes about Rust ownership and memory safety".to_string(),
            source_memory_ids: vec!["mem-gated-revision-a".to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    db.set_page_stale(&page_id, "source_updated").await.unwrap();
    let expected_revision = db.get_page_source_revision(&page_id).await.unwrap();

    db.link_page_source(
        &page_id,
        "mem-gated-revision-b",
        "attached_during_synthesis",
    )
    .await
    .unwrap();
    let result = update_page_at_source_revision(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "Machine proposal based on the old source snapshot".to_string(),
            source_memory_ids: vec!["mem-gated-revision-a".to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        true,
        expected_revision,
        None,
        None,
    )
    .await
    .unwrap();

    assert!(result.gated);
    assert!(!result.acknowledged);
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.stale_reason.as_deref(), Some("source_updated"));
    assert_eq!(page.sources_updated_count, 1);
}

// ── accept_pending_revision ──────────────────────────────────────────────

async fn seed_pending_revision(db: &MemoryDB, target: &str, revision: &str) {
    let now = chrono::Utc::now().timestamp();
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) VALUES (?1, ?1, ?1, 'original content', 0, 'text', 'fact', 'test', 'claude-code', ?2, ?2, 1, 'confirmed', 'memory')",
        libsql::params![target.to_string(), now],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source, supersedes, pending_revision) VALUES (?1, ?1, ?1, 'revised content', 0, 'text', 'fact', 'test', 'claude-code', ?2, ?2, 0, 'new', 'memory', ?3, 1)",
        libsql::params![revision.to_string(), now, target.to_string()],
    )
    .await
    .unwrap();
}

/// M0 write gate: a content write whose `expected_version` no longer matches
/// the stored row must not land. Without the guard, a writer that loaded v1
/// and decided its ownership branch there overwrites whatever landed at v2.
#[tokio::test]
async fn page_write_update_with_stale_expected_version_does_not_clobber() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-cas-stale";
    let source_content = "Version guards keep concurrent page writers from losing updates";
    seed_memory(&db, mem_id, source_content).await;
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_cas_stale";
    db.insert_page(
        page_id,
        "Version Guards",
        None,
        source_content,
        None,
        None,
        &[mem_id],
        &now,
    )
    .await
    .unwrap();

    let loaded = db.get_page(page_id).await.unwrap().unwrap();
    let stale_version = loaded.version;

    // A concurrent writer lands first, advancing the stored version.
    let winner_content =
        "Version guards keep concurrent page writers from losing updates, landed first";
    page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: winner_content.to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: Some(stale_version),
                caller_id: None,
                operation_id: None,
            },
            edited_by: "re_distill",
            require_stale: false,
            knowledge_path: None,
            citations: None,
            expected_source_revision: None,
        },
    )
    .await
    .unwrap();

    let after_winner = db.get_page(page_id).await.unwrap().unwrap();
    assert!(
        after_winner.version > stale_version,
        "precondition: the first writer advanced the page version"
    );

    // The writer still holding the pre-write version must be refused.
    let result = page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: "Stale writer body that must never land".to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: Some(stale_version),
                caller_id: None,
                operation_id: None,
            },
            edited_by: "re_distill",
            require_stale: false,
            knowledge_path: None,
            citations: None,
            expected_source_revision: None,
        },
    )
    .await
    .unwrap();

    assert!(
        !result.wrote,
        "a write carrying a stale expected_version must report wrote=false"
    );
    let after = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, winner_content,
        "the stale writer must not overwrite the content that landed first"
    );
    assert_eq!(
        after.version, after_winner.version,
        "a refused write must not bump the page version"
    );
}

/// Every page write appends exactly one immutable `page_history` row, and
/// the snapshot it stores is the body at *that* version — so a page's
/// evolution is reconstructable rather than inferred.
///
/// Mutation check: drop the history INSERT from `try_update_page_content`
/// and the version sequence collapses to `[1]`; store the pre-write body
/// instead of the post-write one and the content assertions invert.
#[tokio::test]
async fn page_write_records_one_history_row_per_version() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-history";
    let source_content = "Every version of a page is recorded as its own history row";
    seed_memory(&db, mem_id, source_content).await;
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_history_seq";
    db.insert_page(
        page_id,
        "History",
        None,
        source_content,
        None,
        None,
        &[mem_id],
        &now,
    )
    .await
    .unwrap();

    // Creation is itself a version: the timeline is never empty.
    let at_create = db.list_page_history(page_id, 10).await.unwrap();
    assert_eq!(
        at_create.len(),
        1,
        "a newly created page must already have its v1 history row"
    );
    assert_eq!(at_create[0].content, source_content);

    let v2 = "Every version of a page is recorded as its own history row, appended in the write";
    let v3 = "Every version of a page is recorded as its own history row, appended in the write transaction";
    for body in [v2, v3] {
        let result = page_write(
            &db,
            PageWrite::Update {
                page_id,
                req: UpdatePageRequest {
                    content: body.to_string(),
                    source_memory_ids: vec![mem_id.to_string()],
                    expected_version: None,
                    caller_id: None,
                    operation_id: None,
                },
                edited_by: "re_distill",
                require_stale: false,
                knowledge_path: None,
                citations: None,
                expected_source_revision: None,
            },
        )
        .await
        .unwrap();
        assert!(result.wrote, "each update must land");
    }

    let history = db.list_page_history(page_id, 10).await.unwrap();
    let versions: Vec<i64> = history.iter().map(|h| h.version).collect();
    assert_eq!(
        versions,
        vec![3, 2, 1],
        "one row per version, newest first, no gaps and no duplicates"
    );

    // Each row holds the body *at* its version, not the body it replaced.
    assert_eq!(history[0].content, v3);
    assert_eq!(history[1].content, v2);
    assert_eq!(history[2].content, source_content);
    assert_eq!(history[0].edited_by, "re_distill");
    assert_eq!(history[2].edited_by, "create");

    // The history head must agree with the page itself — that is the whole
    // point of writing them in one transaction.
    let page = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(page.version, history[0].version);
    assert_eq!(page.content, history[0].content);
}

/// Seed the `creation_kind='source'` page that a re-enriched document
/// replaces, and return `(page_id, memory_id, body)`.
async fn seed_source_page(db: &MemoryDB, page_id: &'static str) -> (&'static str, String) {
    let mem_id = "mem-source-page";
    let body = "A source page is the machine-owned projection of one ingested document";
    seed_memory(db, mem_id, body).await;
    let req = CreateConceptRequest {
        title: "Ingested Document".to_string(),
        content: body.to_string(),
        summary: None,
        entity_id: None,
        space: (None).into(),
        source_memory_ids: vec![mem_id.to_string()],
        creation_kind: Some("source".to_string()),
        workspace: None,
    };
    page_write(
        db,
        PageWrite::Create {
            page_id: Some(page_id),
            req,
            agent: "doc-enrich",
            knowledge_path: None,
            page_min_cluster_size: 1,
            page_match_threshold: 0.0,
            citations_json: None,
        },
    )
    .await
    .unwrap();
    (mem_id, body.to_string())
}

/// `ReplaceSource` is a page write like any other: re-enriching a document
/// bumps the page version, so it must leave the same immutable `page_history`
/// row behind. Without this the M0-B invariant — one durable row per page
/// version — is simply false for every source page in the corpus.
///
/// Mutation check: drop the history INSERT from `replace_source_page` and the
/// version sequence collapses to `[1]`.
#[tokio::test]
async fn replace_source_page_records_history_row_for_new_version() {
    let (db, _dir) = test_db().await;
    let page_id = "page_source_history";
    let (mem_id, v1) = seed_source_page(&db, page_id).await;

    let at_create = db.list_page_history(page_id, 10).await.unwrap();
    assert_eq!(
        at_create.len(),
        1,
        "a freshly created source page must already have its v1 history row"
    );

    let v2 = "\u{feff}\r\n  A source page keeps exact source  \r\n\r\n";
    assert_ne!(
        v2.trim_end(),
        v2,
        "positive control: trimming must change this fixture"
    );
    let result = page_write(
        &db,
        PageWrite::ReplaceSource {
            page_id,
            title: "Ingested Document",
            summary: Some("the second enrichment"),
            content: v2,
            source_memory_ids: &[mem_id.to_string()],
            agent: "doc-enrich",
        },
    )
    .await
    .unwrap();
    assert!(result.wrote, "replacing a source page must land");

    let history = db.list_page_history(page_id, 10).await.unwrap();
    let versions: Vec<i64> = history.iter().map(|h| h.version).collect();
    assert_eq!(
        versions,
        vec![2, 1],
        "a source-page replacement must append one history row for the version it created"
    );
    assert_eq!(
        history[0].content, v2,
        "the row holds the body at its version"
    );
    assert_eq!(history[1].content, v1);
    assert_eq!(history[0].edited_by, "doc-enrich");

    // The head of the history must agree with the page — the reason both are
    // written in one transaction.
    let page = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(page.version, history[0].version);
    assert_eq!(page.content, history[0].content);
}

/// A re-enrichment that recomputes the body it already wrote must not
/// become a second version.
///
/// This is not hypothetical: when the analysis LLM is unreachable,
/// `document_enrichment` writes a deterministic stub page and pauses for
/// retry, and every retry rebuilds the same stub from the same stored
/// chunks. Without this guard that document's page version climbs once per
/// retry forever — and now that a version also appends a `page_history`
/// row, it would grow an uncapped history of byte-identical snapshots.
///
/// Mutation check: drop the unchanged-guard and the version reaches 3.
#[tokio::test]
async fn replace_source_page_with_identical_content_writes_no_new_version() {
    let (db, _dir) = test_db().await;
    let page_id = "page_source_unchanged";
    let (mem_id, _v1) = seed_source_page(&db, page_id).await;
    let sources = [mem_id.to_string()];

    let body = "\u{feff}\r\n  A source page retry stays byte-identical  \t\r\n\r\n";
    let replace = |content: &'static str| {
        page_write(
            &db,
            PageWrite::ReplaceSource {
                page_id,
                title: "Ingested Document",
                summary: Some("stub"),
                content,
                source_memory_ids: &sources,
                agent: "doc-enrich",
            },
        )
    };

    let first = replace(body).await.unwrap();
    assert!(first.wrote, "the first enrichment must land");
    let after_first = db.get_page(page_id).await.unwrap().unwrap();

    let retry = replace(body).await.unwrap();
    assert!(
        !retry.wrote,
        "recomputing the same body must not be reported as a write"
    );
    assert_eq!(
        retry.outcome,
        WriteOutcome::Unchanged,
        "a no-op retry is Unchanged, not a conflict and not a write"
    );

    let after_retry = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(
        after_retry.version, after_first.version,
        "an identical replacement must not bump the version"
    );
    let versions: Vec<i64> = db
        .list_page_history(page_id, 10)
        .await
        .unwrap()
        .iter()
        .map(|h| h.version)
        .collect();
    assert_eq!(
        versions,
        vec![2, 1],
        "no version means no history row — the retry appends nothing"
    );
}

#[tokio::test]
async fn replace_source_page_rejects_reserved_delimiters_without_mutation() {
    use crate::export::provenance::{SOURCES_BLOCK_END, SOURCES_BLOCK_START};

    let (db, _dir) = test_db().await;
    let page_id = "page_source_reserved";
    let (mem_id, _v1) = seed_source_page(&db, page_id).await;
    let sources = [mem_id.to_string()];
    let before = db.get_page(page_id).await.unwrap().unwrap();
    let history_before = db.list_page_history(page_id, 10).await.unwrap();
    let source_ids_before: HashSet<_> = db
        .get_page_sources(page_id)
        .await
        .unwrap()
        .into_iter()
        .map(|source| source.memory_source_id)
        .collect();
    let cases = [
        format!("before {SOURCES_BLOCK_START} after"),
        format!("before {SOURCES_BLOCK_END} after"),
        format!("{SOURCES_BLOCK_START}\nowned\n{SOURCES_BLOCK_END}"),
        format!(
            "{SOURCES_BLOCK_START}\none\n{SOURCES_BLOCK_END}\n\
             {SOURCES_BLOCK_START}\ntwo\n{SOURCES_BLOCK_END}"
        ),
        format!("```md\n{SOURCES_BLOCK_START}\n```\nkept prose"),
    ];

    for content in cases {
        let error = page_write(
            &db,
            PageWrite::ReplaceSource {
                page_id,
                title: "Changed title",
                summary: Some("Changed summary"),
                content: &content,
                source_memory_ids: &sources,
                agent: "doc-enrich",
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(error, WenlanError::Validation(_)));

        let after = db.get_page(page_id).await.unwrap().unwrap();
        let history_after = db.list_page_history(page_id, 10).await.unwrap();
        let source_ids_after: HashSet<_> = db
            .get_page_sources(page_id)
            .await
            .unwrap()
            .into_iter()
            .map(|source| source.memory_source_id)
            .collect();
        assert_eq!(after.title, before.title);
        assert_eq!(after.summary, before.summary);
        assert_eq!(after.content, before.content);
        assert_eq!(after.version, before.version);
        assert_eq!(history_after.len(), history_before.len());
        assert_eq!(source_ids_after, source_ids_before);
    }
}

/// Seed a page and return `(db, tempdir, page_id, memory_id)` for the
/// retry-receipt tests below.
async fn receipt_fixture(
    page_id: &'static str,
) -> (MemoryDB, tempfile::TempDir, &'static str, &'static str) {
    let (db, dir) = test_db().await;
    let mem_id = "mem-receipt";
    let body = "A retried write must not become a second version of the page";
    seed_memory(&db, mem_id, body).await;
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page(page_id, "Receipts", None, body, None, None, &[mem_id], &now)
        .await
        .unwrap();
    (db, dir, page_id, mem_id)
}

fn retry_req(content: &str, mem_id: &str, operation_id: &str) -> UpdatePageRequest {
    UpdatePageRequest {
        content: content.to_string(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: Some("app".to_string()),
        operation_id: Some(operation_id.to_string()),
    }
}

#[tokio::test]
async fn update_projection_failure_after_commit_preserves_db_history_and_receipt_authority() {
    let (db, _dir, page_id, mem_id) =
        receipt_fixture("page_update_post_commit_projection_failure").await;
    let knowledge = tempfile::tempdir().unwrap();
    let before = db.get_page(page_id).await.unwrap().unwrap();
    let history_before = db.list_page_history(page_id, 10).await.unwrap();
    let updated_content =
        "The database, history, and receipt stay authoritative after projection failure";
    let operation_id = "op-post-commit-projection-failure";

    let (parked_tx, parked_rx) = tokio::sync::oneshot::channel();
    let (resume_tx, resume_rx) = tokio::sync::oneshot::channel();
    *page_update::POST_COMMIT_PROJECTION_GATE.lock().unwrap() =
        Some((page_id.to_string(), parked_tx, resume_rx));

    let update = update_page(
        &db,
        page_id,
        retry_req(updated_content, mem_id, operation_id),
        "re_distill",
        false,
        Some(knowledge.path()),
        None,
    );
    let observe_committed_authority = async {
        tokio::time::timeout(std::time::Duration::from_secs(5), parked_rx)
            .await
            .expect("production update must reach the post-commit projection pause")
            .expect("production update dropped the post-commit projection pause");

        let committed = db.get_page(page_id).await.unwrap().unwrap();
        assert_eq!(
            committed.content, updated_content,
            "the DB update must commit before projection can fail"
        );
        assert_eq!(committed.version, before.version + 1);

        let history = db.list_page_history(page_id, 10).await.unwrap();
        assert_eq!(history.len(), history_before.len() + 1);
        assert_eq!(history[0].version, committed.version);
        assert_eq!(history[0].content, updated_content);
        assert_eq!(history[0].edited_by, "re_distill");

        let receipt = db
            .get_operation_receipt("app", operation_id)
            .await
            .unwrap()
            .expect("the page write receipt must commit with the DB update");
        let receipt_result: WriteResult = serde_json::from_str(&receipt.response).unwrap();
        assert!(receipt_result.wrote);
        assert_eq!(receipt_result.outcome, WriteOutcome::Wrote);

        let writer =
            crate::export::knowledge::KnowledgeWriter::new(knowledge.path().to_path_buf(), &db);
        let filename = writer
            .page_filename(page_id)
            .expect("the real production projection call must write the page");
        let markdown = std::fs::read_to_string(knowledge.path().join(filename)).unwrap();
        assert!(
            markdown.contains(updated_content),
            "the production projection site must run before the injected failure"
        );

        resume_tx
            .send(())
            .expect("the production update must still be paused");
        (committed, history, receipt.response)
    };

    let (result, (committed, history, receipt_response)) =
        tokio::join!(update, observe_committed_authority);
    let result = result.expect("the post-commit projection failure must stay best-effort");
    assert!(result.wrote);
    assert_eq!(result.outcome, WriteOutcome::Wrote);
    assert_eq!(serde_json::to_string(&result).unwrap(), receipt_response);
    let after = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(after.content, committed.content);
    assert_eq!(after.version, committed.version);
    assert_eq!(after.source_memory_ids, committed.source_memory_ids);
    assert_eq!(after.changelog, committed.changelog);
    assert_eq!(db.list_page_history(page_id, 10).await.unwrap(), history);
}

/// The lost-response case: the client never saw the reply and sent the very
/// same write again. It must get the original response back and the page
/// must be untouched — one version, one history row, not two.
#[tokio::test]
async fn page_write_same_operation_id_replays_instead_of_writing_again() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_replay").await;
    let body = "A retried write must not become a second version of the page, ever";

    let first = update_page(
        &db,
        page_id,
        retry_req(body, mem_id, "op-1"),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(first.wrote);
    let after_first = db.get_page(page_id).await.unwrap().unwrap();

    let replay = update_page(
        &db,
        page_id,
        retry_req(body, mem_id, "op-1"),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        serde_json::to_string(&replay).unwrap(),
        serde_json::to_string(&first).unwrap(),
        "a replay returns the recorded response verbatim"
    );
    let after_replay = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(
        after_replay.version, after_first.version,
        "the replay must not bump the version"
    );
    let versions: Vec<i64> = db
        .list_page_history(page_id, 10)
        .await
        .unwrap()
        .iter()
        .map(|h| h.version)
        .collect();
    assert_eq!(
        versions,
        vec![2, 1],
        "the replay must not append a second history row"
    );
}

/// The manual HTTP route carries content, not a replacement source list.
/// A source attached after the editor loaded must therefore survive the
/// save even when the caller omitted `expected_version`.
#[tokio::test]
async fn page_write_preserve_sources_uses_the_cas_generation_source_set() {
    let (db, _dir) = test_db().await;
    for (source_id, content) in [
        (
            "mem-preserve-a",
            "The editor originally loaded the Page with source A.",
        ),
        (
            "mem-preserve-b",
            "A concurrent writer attached source B before the save.",
        ),
        (
            "mem-preserve-c",
            "Another source was attached after the first response was lost.",
        ),
    ] {
        seed_memory(&db, source_id, content).await;
    }
    let page_id = seed_page(
        &db,
        "mem-preserve-a",
        "The editor originally loaded the Page with source A.",
    )
    .await;

    // This is the route's old TOCTOU shape: its request snapshot had only
    // A, but B landed before PageWrite loaded the generation it will CAS.
    db.link_page_source(&page_id, "mem-preserve-b", "concurrent_attach")
        .await
        .unwrap();

    let result = update_page_preserving_sources(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "The editor saved prose after another writer attached a source.".to_string(),
            source_memory_ids: vec![],
            expected_version: None,
            caller_id: Some("app".to_string()),
            operation_id: Some("op-preserve".to_string()),
        },
        "manual_edit",
        None,
    )
    .await
    .unwrap();

    assert!(result.wrote);
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        page.source_memory_ids,
        vec!["mem-preserve-a".to_string(), "mem-preserve-b".to_string()],
        "preserve mode must derive sources from the same Page generation its CAS updates"
    );

    db.link_page_source(&page_id, "mem-preserve-c", "after_lost_response")
        .await
        .unwrap();
    let replay = update_page_preserving_sources(
        &db,
        &page_id,
        UpdatePageRequest {
            content: "The editor saved prose after another writer attached a source.".to_string(),
            source_memory_ids: vec![],
            expected_version: None,
            caller_id: Some("app".to_string()),
            operation_id: Some("op-preserve".to_string()),
        },
        "manual_edit",
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        serde_json::to_string(&replay).unwrap(),
        serde_json::to_string(&result).unwrap(),
        "server-derived source changes must not turn an honest retry into a digest conflict"
    );
    assert_eq!(
        db.get_page(&page_id)
            .await
            .unwrap()
            .unwrap()
            .source_memory_ids,
        vec![
            "mem-preserve-a".to_string(),
            "mem-preserve-b".to_string(),
            "mem-preserve-c".to_string(),
        ],
        "receipt replay must not touch sources attached after the first response"
    );
}

/// A no-op is still a terminal response for an identified operation. If
/// that response is lost, a later retry must replay it instead of turning
/// into a write against whatever Page generation exists by then.
#[tokio::test]
async fn page_write_noop_receipt_replays_after_an_intervening_edit() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_noop").await;
    let original = db.get_page(page_id).await.unwrap().unwrap();

    let first = update_page(
        &db,
        page_id,
        retry_req(&original.content, mem_id, "op-noop"),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(first.outcome, WriteOutcome::Unchanged);
    assert!(
        db.get_operation_receipt("app", "op-noop")
            .await
            .unwrap()
            .is_some(),
        "a successful no-op must be replayable"
    );

    let newer = "A later operation changed the Page after the no-op response was lost";
    update_page(
        &db,
        page_id,
        retry_req(newer, mem_id, "op-newer"),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let replay = update_page(
        &db,
        page_id,
        retry_req(&original.content, mem_id, "op-noop"),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        serde_json::to_string(&replay).unwrap(),
        serde_json::to_string(&first).unwrap(),
        "the lost no-op response must replay verbatim"
    );
    assert_eq!(
        db.get_page(page_id).await.unwrap().unwrap().content,
        newer,
        "replaying the earlier no-op must not overwrite the intervening edit"
    );
}

#[tokio::test]
async fn page_write_acknowledged_noop_commits_and_replays_its_receipt() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_acknowledged").await;
    let original = db.get_page(page_id).await.unwrap().unwrap();
    db.set_page_stale(page_id, "source_updated").await.unwrap();

    let first = update_page(
        &db,
        page_id,
        retry_req(&original.content, mem_id, "op-acknowledged"),
        "re_distill",
        true,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(first.acknowledged);
    assert!(db
        .get_operation_receipt("app", "op-acknowledged")
        .await
        .unwrap()
        .is_some());

    // New stale work is a different durable event. Replaying the already
    // completed operation must return its old response without clearing it.
    db.set_page_stale(page_id, "new_source_update")
        .await
        .unwrap();
    let replay = update_page(
        &db,
        page_id,
        retry_req(&original.content, mem_id, "op-acknowledged"),
        "re_distill",
        true,
        None,
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        serde_json::to_string(&replay).unwrap(),
        serde_json::to_string(&first).unwrap()
    );
    assert_eq!(
        db.get_page(page_id)
            .await
            .unwrap()
            .unwrap()
            .stale_reason
            .as_deref(),
        Some("new_source_update"),
        "receipt replay must not acknowledge later stale work"
    );
}

/// Gating is a durable terminal outcome too. Retrying the same identified
/// machine write must return the first card id and leave one review item.
#[tokio::test]
async fn page_write_gated_receipt_deduplicates_revision_card_retry() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_gated").await;
    update_page(
        &db,
        page_id,
        UpdatePageRequest {
            content: "A human took ownership of this Page before synthesis.".to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let gated_req = || {
        retry_req(
            "The machine proposal must appear as exactly one review card.",
            mem_id,
            "op-gated",
        )
    };
    let first = update_page(&db, page_id, gated_req(), "re_distill", false, None, None)
        .await
        .unwrap();
    assert_eq!(first.outcome, WriteOutcome::Gated);

    let replay = update_page(&db, page_id, gated_req(), "re_distill", false, None, None)
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_string(&replay).unwrap(),
        serde_json::to_string(&first).unwrap(),
        "the retry must replay the first revision-card id"
    );
    assert_eq!(
        db.list_pending_revisions(10).await.unwrap().len(),
        1,
        "one operation may stage only one pending revision card"
    );
    assert!(
        db.get_operation_receipt("app", "op-gated")
            .await
            .unwrap()
            .is_some(),
        "the gated response must be durable"
    );
}

/// Both attempts may pass PageWrite's initial receipt lookup before either
/// transaction commits. The losing card transaction rolls back, but it must
/// still replay the winner instead of leaking the receipt PK conflict.
#[tokio::test]
async fn page_write_concurrent_gated_retry_replays_the_winner() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_gated_concurrent").await;
    let page = db.get_page(page_id).await.unwrap().unwrap();
    let sources = vec![mem_id.to_string()];
    let retry = (
        "app".to_string(),
        "op-gated-concurrent".to_string(),
        "same-request-digest".to_string(),
    );

    let (left, right) = tokio::join!(
        stage_page_revision_card(
            &db,
            &page,
            "The concurrent proposal must converge on one review card.",
            &sources,
            0, // never accepted in this test
            "re_distill",
            Some(&retry),
        ),
        stage_page_revision_card(
            &db,
            &page,
            "The concurrent proposal must converge on one review card.",
            &sources,
            0, // never accepted in this test
            "re_distill",
            Some(&retry),
        ),
    );
    let left = left.unwrap();
    let right = right.unwrap();

    assert_eq!(
        serde_json::to_string(&right).unwrap(),
        serde_json::to_string(&left).unwrap(),
        "the losing transaction must replay the winner's response"
    );
    assert_eq!(
        db.list_pending_revisions(10).await.unwrap().len(),
        1,
        "the receipt conflict must roll the losing card back"
    );
}

/// Page ids may be deterministic. Deleting and recreating one must start a
/// new history generation rather than exposing the deleted Page's bodies.
#[tokio::test]
async fn page_delete_then_recreate_same_id_starts_fresh_history() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_history_recreate").await;
    update_page(
        &db,
        page_id,
        retry_req(
            "The first Page generation reached version two before deletion.",
            mem_id,
            "op-old-generation",
        ),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    db.delete_page(page_id).await.unwrap();

    let new_source = "mem-recreated-page";
    let new_body = "The recreated Page is a new generation with unrelated content.";
    seed_memory(&db, new_source, new_body).await;
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page(
        page_id,
        "Recreated Page",
        None,
        new_body,
        None,
        None,
        &[new_source],
        &now,
    )
    .await
    .unwrap();

    let history = db.list_page_history(page_id, 10).await.unwrap();
    assert_eq!(
        history.len(),
        1,
        "deleted generations must not leak forward"
    );
    assert_eq!(history[0].version, 1);
    assert_eq!(history[0].content, new_body);
    assert_eq!(history[0].source_memory_ids, vec![new_source.to_string()]);
}

/// The same operation id carrying a *different* write is not a retry — it
/// is an id being reused. Replaying the old response would tell the caller
/// their new text was saved when it never was, so refuse instead.
#[tokio::test]
async fn page_write_same_operation_id_with_different_body_is_a_conflict() {
    use crate::export::provenance::SOURCES_BLOCK_START;

    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_conflict").await;

    update_page(
        &db,
        page_id,
        retry_req(
            "A retried write must not become a second version of the page, ever",
            mem_id,
            "op-1",
        ),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    let after_first = db.get_page(page_id).await.unwrap().unwrap();
    let changed_body = format!(
        "A retried write must not become a second version of the page — \
         different text\n\n{SOURCES_BLOCK_START}"
    );

    let err = update_page(
        &db,
        page_id,
        retry_req(&changed_body, mem_id, "op-1"),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .expect_err("reusing an operation id for a different write must be refused");
    assert!(
        matches!(err, WenlanError::Conflict(_)),
        "receipt collision must win before canonical-content validation; got {err:?}"
    );

    let unchanged = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(unchanged.content, after_first.content);
    assert_eq!(unchanged.version, after_first.version);
}

/// A distinct operation id from the same caller is a distinct write and
/// must land normally — the receipt table must not become a write blocker.
#[tokio::test]
async fn page_write_different_operation_id_writes_normally() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_distinct").await;

    for (op, body) in [
        (
            "op-1",
            "A retried write must not become a second version of the page, once",
        ),
        (
            "op-2",
            "A retried write must not become a second version of the page, twice",
        ),
    ] {
        let r = update_page(
            &db,
            page_id,
            retry_req(body, mem_id, op),
            "re_distill",
            false,
            None,
            None,
        )
        .await
        .unwrap();
        assert!(r.wrote, "operation {op} must land");
    }

    let versions: Vec<i64> = db
        .list_page_history(page_id, 10)
        .await
        .unwrap()
        .iter()
        .map(|h| h.version)
        .collect();
    assert_eq!(versions, vec![3, 2, 1]);
}

/// A write refused by the version precondition must leave no receipt, or
/// the caller's next honest attempt would replay a response for a write
/// that never happened. (This pins the refusal path only — the refusal
/// returns before the receipt is ever written. The transaction claim is
/// pinned by `page_write_crash_before_commit_leaves_no_receipt`.)
#[tokio::test]
async fn page_write_refused_by_cas_records_no_receipt() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_rollback").await;
    let current = db.get_page(page_id).await.unwrap().unwrap();

    let mut req = retry_req(
        "A retried write must not become a second version of the page, refused",
        mem_id,
        "op-doomed",
    );
    req.expected_version = Some(current.version + 99);

    let result = update_page(&db, page_id, req, "re_distill", false, None, None)
        .await
        .unwrap();
    assert!(!result.wrote, "precondition: the write must be refused");

    assert!(
        db.get_operation_receipt("app", "op-doomed")
            .await
            .unwrap()
            .is_none(),
        "a refused write must leave no receipt behind"
    );
}

/// The crash window: the page row, the history row, and the receipt are all
/// staged, and the process dies before COMMIT. Nothing may survive — least
/// of all the receipt, which would answer the caller's retry with "already
/// done" about a version that does not exist. This is the only test that
/// can tell an in-transaction receipt from an after-COMMIT one.
#[tokio::test]
async fn page_write_crash_before_commit_leaves_no_receipt() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_crash").await;
    let before = db.get_page(page_id).await.unwrap().unwrap();

    *crate::db::FAIL_BEFORE_COMMIT.lock().unwrap() = Some(page_id.to_string());
    let result = update_page(
        &db,
        page_id,
        retry_req(
            "A retried write must not become a second version of the page, crashed",
            mem_id,
            "op-crash",
        ),
        "re_distill",
        false,
        None,
        None,
    )
    .await;
    assert!(
        result.is_err(),
        "precondition: the injected fault must abort the write; got {result:?}"
    );

    let after = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(
        (after.content, after.version),
        (before.content, before.version),
        "the aborted write must have rolled back entirely"
    );
    assert!(
        db.get_operation_receipt("app", "op-crash")
            .await
            .unwrap()
            .is_none(),
        "a receipt must not outlive the transaction that wrote it"
    );

    // The retry now behaves as if the first attempt never happened.
    let retry = update_page(
        &db,
        page_id,
        retry_req(
            "A retried write must not become a second version of the page, crashed",
            mem_id,
            "op-crash",
        ),
        "re_distill",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(
        retry.wrote,
        "after a rolled-back attempt the same operation id must still be usable"
    );
}

/// One id without the other is not a retry identity, and must not silently
/// behave like one — otherwise two callers could collide on "op-1".
#[tokio::test]
async fn page_write_partial_retry_identity_is_ignored() {
    let (db, _dir, page_id, mem_id) = receipt_fixture("page_receipt_partial").await;

    let mut req = retry_req(
        "A retried write must not become a second version of the page, partial",
        mem_id,
        "op-partial",
    );
    req.caller_id = None;

    update_page(&db, page_id, req, "re_distill", false, None, None)
        .await
        .unwrap();

    assert!(
        db.get_operation_receipt("", "op-partial")
            .await
            .unwrap()
            .is_none(),
        "an operation id with no caller must not be recorded"
    );
}

/// A write that loses the CAS must leave no trace: no version bump and no
/// history row. Otherwise the timeline would record edits that never
/// happened, which is worse than no timeline at all.
#[tokio::test]
async fn page_write_refused_by_cas_appends_no_history_row() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-history-refused";
    let source_content = "A refused write must not appear in the page timeline";
    seed_memory(&db, mem_id, source_content).await;
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_history_refused";
    db.insert_page(
        page_id,
        "Refused",
        None,
        source_content,
        None,
        None,
        &[mem_id],
        &now,
    )
    .await
    .unwrap();
    let stale_version = db.get_page(page_id).await.unwrap().unwrap().version;

    let result = page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: "A refused write must not appear in the page timeline at all".to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: Some(stale_version + 5),
                caller_id: None,
                operation_id: None,
            },
            edited_by: "re_distill",
            require_stale: false,
            knowledge_path: None,
            citations: None,
            expected_source_revision: None,
        },
    )
    .await
    .unwrap();

    assert!(!result.wrote, "precondition: the write must be refused");
    let history = db.list_page_history(page_id, 10).await.unwrap();
    assert_eq!(
        history.len(),
        1,
        "a refused write must not append a history row"
    );
    assert_eq!(history[0].version, stale_version);
}

/// The load-bearing test for the M0 write gate: an edit that lands *inside*
/// the window between the ownership decision and the write must not be
/// clobbered.
///
/// This is the only test here that fails when `Some(current_version)` is
/// dropped from the write — with no interleaving, a guarded write and an
/// unguarded one are byte-identical. Mutation check: pass `None` as the
/// guard and this test overwrites the human edit and reports v3.
#[tokio::test]
async fn page_write_update_edit_landing_mid_write_is_not_clobbered() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-cas-interleave";
    let source_content = "A write must land on the row its ownership decision was made from";
    seed_memory(&db, mem_id, source_content).await;
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_cas_interleave";
    db.insert_page(
        page_id,
        "Interleaved Write",
        None,
        source_content,
        None,
        None,
        &[mem_id],
        &now,
    )
    .await
    .unwrap();
    let start_version = db.get_page(page_id).await.unwrap().unwrap().version;

    // Arm the seam: the machine write below parks after deciding ownership
    // (page is still machine-owned) and before writing.
    let (parked_tx, parked_rx) = tokio::sync::oneshot::channel();
    let (go_tx, go_rx) = tokio::sync::oneshot::channel();
    *PRE_WRITE_GATE.lock().unwrap() = Some((page_id.to_string(), parked_tx, go_rx));

    // Close to the seeded source: `fs_edit` is not exempt from the
    // hallucination guard, so an unrelated body would be rejected before
    // ever reaching the write this test is about.
    let human_content =
        "A write must land on the row its ownership decision was made from, typed by hand";
    let machine_write = page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: "Machine body computed from the pre-edit row".to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: None,
                caller_id: None,
                operation_id: None,
            },
            edited_by: "re_distill",
            require_stale: false,
            knowledge_path: None,
            citations: None,
            expected_source_revision: None,
        },
    );

    // Runs to completion while the machine write is parked — a genuine
    // interleaving, not a simulated one.
    let human_edit = async {
        parked_rx
            .await
            .expect("machine write must reach the pre-write gate");
        let result = page_write(
            &db,
            PageWrite::Update {
                page_id,
                req: UpdatePageRequest {
                    content: human_content.to_string(),
                    source_memory_ids: vec![mem_id.to_string()],
                    expected_version: None,
                    caller_id: None,
                    operation_id: None,
                },
                edited_by: "fs_edit",
                require_stale: false,
                knowledge_path: None,
                citations: None,
                expected_source_revision: None,
            },
        )
        .await
        .unwrap();
        // Only now release the parked machine write, so it resumes against a
        // page that has definitively moved.
        go_tx.send(()).expect("machine write must still be parked");
        result
    };

    let (machine_result, human_result) = tokio::join!(machine_write, human_edit);
    let machine_result = machine_result.unwrap();

    assert!(human_result.wrote, "the human edit itself must land");
    assert!(
        !machine_result.wrote,
        "the machine write lost the CAS and must not report a write"
    );

    let after = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, human_content,
        "the edit that landed mid-write must survive"
    );
    assert_eq!(
        after.version,
        start_version + 1,
        "only the human edit bumped the version; the losing write must not have applied"
    );
    assert!(
        machine_result.revision_card_id.is_some(),
        "on reload the page is human-owned, so the machine body is preserved as a card"
    );
}

/// A machine writer that DECLARED the version it read (`expected_version`)
/// and lost the race to a human edit is refused outright — it does not get a
/// revision card.
///
/// This reverses what this test asserted when M0-C landed. Carding looked
/// like the conservative choice ("never drop agent work"), but a card is not
/// inert: a card computed from a stale base must not be staged as if it were
/// current. Current cards do record and re-check their staged Page version,
/// but refusing at the declared precondition remains the earlier and more
/// truthful boundary. The caller sees the conflict, re-reads, and stages a
/// proposal against the real content.
///
/// Only writers that declare a base get this. A machine write with
/// `expected_version: None` never told us what it read, so a card is still
/// the best available answer — see
/// `page_write_update_edit_landing_mid_write_is_not_clobbered`, which is
/// unaffected by this and still stages one.
#[tokio::test]
async fn page_write_update_stale_version_on_human_owned_page_is_refused() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem-cas-owned";
    let source_content = "Ownership is re-decided against the row the write lands on";
    seed_memory(&db, mem_id, source_content).await;
    let now = chrono::Utc::now().to_rfc3339();
    let page_id = "page_cas_owned";
    db.insert_page(
        page_id,
        "Ownership Recheck",
        None,
        source_content,
        None,
        None,
        &[mem_id],
        &now,
    )
    .await
    .unwrap();

    // The machine writer's view of the world: machine-owned, at this version.
    let machine_view = db.get_page(page_id).await.unwrap().unwrap();
    assert!(
        !machine_view.user_edited,
        "precondition: page starts machine-owned"
    );

    // A human edit lands underneath it, taking ownership and bumping version.
    let human_content = "Ownership is re-decided against the row the write lands on, by hand";
    page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: human_content.to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: Some(machine_view.version),
                caller_id: None,
                operation_id: None,
            },
            edited_by: "fs_edit",
            require_stale: false,
            knowledge_path: None,
            citations: None,
            expected_source_revision: None,
        },
    )
    .await
    .unwrap();
    let owned = db.get_page(page_id).await.unwrap().unwrap();
    assert!(
        owned.user_edited,
        "precondition: the human edit took ownership"
    );

    let pending_before = db.list_pending_revisions(10).await.unwrap().len();

    // The machine writer proceeds from its stale view.
    let result = page_write(
        &db,
        PageWrite::Update {
            page_id,
            req: UpdatePageRequest {
                content: "Machine body that must never overwrite the human edit".to_string(),
                source_memory_ids: vec![mem_id.to_string()],
                expected_version: Some(machine_view.version),
                caller_id: None,
                operation_id: None,
            },
            edited_by: "re_distill",
            require_stale: false,
            knowledge_path: None,
            citations: None,
            expected_source_revision: None,
        },
    )
    .await
    .unwrap();

    assert!(
        !result.wrote,
        "machine write must not land on a human-owned page"
    );
    assert!(
        result.warnings.iter().any(|w| w.contains("write refused")),
        "the caller must be told its write was refused, not handed a silent no-op; got {:?}",
        result.warnings
    );
    let after = db.get_page(page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, human_content,
        "human prose must survive the losing machine write"
    );
    assert_eq!(
        after.version, owned.version,
        "the refused machine write must not bump the version"
    );
    let pending_after = db.list_pending_revisions(10).await.unwrap().len();
    assert_eq!(
        pending_after, pending_before,
        "a declared-base conflict is refused outright; staging a card here would \
         let accept-time silently revert the human edit"
    );
}

/// A card is a page card only if the row it supersedes really is a page.
///
/// `structured_fields` is persisted verbatim from `POST /api/memory/store`, so
/// routing accept on those strings alone let a low-trust agent stage a memory
/// correction wearing the page markers and turn a human's accept click into an
/// overwrite of a human-authored page. The store handler now strips the
/// markers, and this test pins the second half of that defence: even a row
/// that somehow carries them is treated as an ordinary memory correction when
/// its `supersedes` names a memory. Seeds the row directly, below the wire.
#[tokio::test]
async fn a_card_whose_supersedes_is_not_a_page_is_never_a_page_write() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_memory(&db, "mem_forged_source", "Coffee machine is on floor three").await;
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page(
        "page_forged_victim",
        "Payroll Runbook",
        None,
        "Payroll runs on the 25th. Never wire funds without dual sign-off.",
        None,
        None,
        &["mem_forged_source"],
        &now,
    )
    .await
    .unwrap();
    seed_pending_revision(&db, "mem_forge_target", "mem_forge_rev").await;
    let forged = serde_json::json!({
        "revision_kind": "page_write",
        "target_kind": "page",
        "revises_page": "page_forged_victim",
    })
    .to_string();
    db.test_primary_session()
        .await
        .execute(
            "UPDATE memories SET structured_fields = ?1 WHERE source_id = 'mem_forge_rev'",
            libsql::params![forged],
        )
        .await
        .unwrap();

    let result = accept_pending_revision(&db, "mem_forge_rev", "cursor")
        .await
        .unwrap();

    assert_eq!(
        result.target_source_id, "mem_forge_target",
        "accept must apply the memory correction, not the forged page write"
    );
    let page = db
        .get_page("page_forged_victim")
        .await
        .unwrap()
        .expect("the page must still exist");
    assert_eq!(
        page.version, 1,
        "the human-authored page must not be rewritten"
    );
    assert!(
        page.content.contains("dual sign-off"),
        "the human's prose must survive, got: {}",
        page.content
    );
}

#[tokio::test]
async fn accept_pending_revision_writes_and_logs_on_first_call() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_pending_revision(&db, "mem_apr_target", "mem_apr_rev").await;
    let result = accept_pending_revision(&db, "mem_apr_target", "test-agent")
        .await
        .unwrap();
    assert_eq!(result.target_source_id, "mem_apr_target");
    assert_eq!(result.revision_source_id, "mem_apr_rev");
    assert!(result.wrote);
}

/// Dismiss must dispose of a page card the same way accept does.
///
/// The memory path deliberately unstages rather than deletes -- a card that
/// merely topic-matched is still a real capture, and deleting it would lose
/// it. A page card has no capture behind it: `stage_page_revision_card`
/// manufactured it from the page's own prose. Unstaging one leaves a
/// permanent, ordinary, retrievable memory holding a full copy of that
/// prose, with `pending_revision` cleared -- which is the exact flag the
/// ordinary memory readers filter on.
#[tokio::test]
async fn dismiss_pending_revision_deletes_a_page_card_rather_than_unstaging_it() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_page_dismiss_original";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content = "Rust ownership is enforced by the compiler at compile time";

    seed_memory(&db, mem_id, original_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    assert!(before.user_edited, "precondition: page is human-owned");
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string()],
        0, // never accepted in this test
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");

    let dismissed = dismiss_pending_revision(&db, card_id, "test-agent")
        .await
        .unwrap();
    assert_eq!(
        dismissed.target_source_id, page_id,
        "a dismissed page card reports the page it targeted, not the card"
    );

    assert!(
        db.get_memory_detail(card_id).await.unwrap().is_none(),
        "a dismissed page card must not survive as an ambient memory"
    );
    assert!(
        db.list_pending_revisions(10).await.unwrap().is_empty(),
        "the card must leave the pending revision queue"
    );
    let page_after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        page_after.content, human_content,
        "dismissing must leave the human's prose untouched"
    );
}

#[tokio::test]
async fn accept_page_revision_projection_failure_preserves_committed_db_authority() {
    let (db, dir) = test_db().await;
    let projection_root = dir.path().join("projection-root");
    std::fs::write(
        &projection_root,
        "a regular file cannot be a projection directory",
    )
    .unwrap();
    let mem_id = "mem_page_accept_projection_failure_original";
    let new_mem_id = "mem_page_accept_projection_failure_new";
    let original_content = "original page content before projection failure";
    let proposed_content = "accepted page content remains authoritative after projection failure";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(&db, new_mem_id, proposed_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    let before = db.get_page(&page_id).await.unwrap().unwrap();

    let projection =
        crate::export::knowledge::KnowledgeProjectionWrite::new(projection_root.clone(), &db);
    projection
        .write_page_gated(&db, &before)
        .await
        .expect_err("a regular file must make the projection write fail");
    drop(projection);

    let expected_sources = vec![mem_id.to_string(), new_mem_id.to_string()];
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &expected_sources,
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");

    let accepted = accept_pending_revision_with_knowledge_path(
        &db,
        card_id,
        "test-agent",
        Some(&projection_root),
    )
    .await
    .unwrap();
    assert!(accepted.wrote);
    assert_eq!(accepted.target_source_id, page_id);
    assert_eq!(accepted.revision_source_id, card_id);

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(after.content, proposed_content);
    assert_eq!(after.version, before.version + 1);
    assert_eq!(after.source_memory_ids, expected_sources);
    assert!(
        db.get_memory_detail(card_id).await.unwrap().is_none(),
        "the committed accept must consume its transient revision card"
    );
    assert!(
        db.list_pending_revisions(10).await.unwrap().is_empty(),
        "projection failure must not resurrect the committed card"
    );
}

#[tokio::test]
async fn accept_pending_revision_page_write_card_updates_page_content() {
    let (db, _dir) = test_db().await;
    let knowledge_dir = tempfile::tempdir().unwrap();
    let mem_id = "mem_page_accept_original";
    let new_mem_id = "mem_page_accept_new";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content =
        "Rust ownership lets the compiler enforce memory safety during page refresh";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(&db, new_mem_id, proposed_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    assert!(before.user_edited, "precondition: page is human-owned");

    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string(), new_mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");

    let accepted = accept_pending_revision_with_knowledge_path(
        &db,
        card_id,
        "test-agent",
        Some(knowledge_dir.path()),
    )
    .await
    .unwrap();
    assert_eq!(accepted.target_source_id, page_id);
    assert_eq!(accepted.revision_source_id, card_id);
    assert!(accepted.wrote);

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, proposed_content,
        "accepting a page-write card must apply the proposed prose to the page"
    );
    assert_eq!(
        after.source_memory_ids,
        vec![mem_id.to_string(), new_mem_id.to_string()],
        "accepting a page-write card must apply its proposed source set"
    );
    assert_eq!(
        after.version,
        before.version + 1,
        "accepting a page-write card must bump the page version"
    );
    assert!(
        db.list_pending_revisions(10).await.unwrap().is_empty(),
        "accepted page-write card must leave the pending revision queue"
    );
    assert!(
        db.get_memory_detail(card_id).await.unwrap().is_none(),
        "accepted Page proposal is a transient card, not a new ambient memory"
    );

    let writer =
        crate::export::knowledge::KnowledgeWriter::new(knowledge_dir.path().to_path_buf(), &db);
    let filename = writer
        .page_filename(&page_id)
        .expect("accepted page-write card must refresh the markdown projection");
    let markdown = std::fs::read_to_string(knowledge_dir.path().join(filename)).unwrap();
    assert!(
        markdown.contains(proposed_content),
        "markdown projection must contain the accepted page prose"
    );
    assert!(
        markdown.contains(&format!("origin_version: {}", after.version)),
        "markdown projection must carry the accepted page version"
    );
}

/// The desktop Review lane reads `/api/memory/pending-revisions`, which is
/// `list_pending_revisions_scoped`. A gated page write stages its card against
/// a `pages.id`, but that reader only accepted a target that exists in
/// `memories`, so every page card was filtered out: the daemon gated the write
/// and staged the card, and then nothing ever asked the human about it.
///
/// The dangling card in this test is the other half of the contract: a
/// revision whose target exists in neither table must still stay off the queue,
/// so widening the check must not become "no check".
#[tokio::test]
async fn pending_revision_queue_lists_a_gated_page_card_and_accept_applies_it() {
    let (db, _dir) = test_db().await;
    let knowledge_dir = tempfile::tempdir().unwrap();
    let mem_id = "mem_page_queue_original";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content = "Rust ownership lets the compiler enforce memory safety at compile time";

    seed_memory(&db, mem_id, original_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(
        db.get_page(&page_id).await.unwrap().unwrap().user_edited,
        "precondition: the page is human-owned"
    );

    // A machine write to a human-owned page is gated into a card rather than
    // overwriting the prose -- this is the live receipt's `gated: true`.
    let gated = update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: proposed_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "page_growth",
        false,
        None,
        None,
    )
    .await
    .unwrap();
    assert!(gated.gated, "machine write to a human page must be gated");
    assert!(!gated.wrote, "gated write must leave the page prose alone");
    let card_id = gated
        .revision_card_id
        .clone()
        .expect("a gated page write must stage a revision card");
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().content,
        human_content,
        "precondition: the gated write did not touch the page"
    );

    // A card whose target exists in neither `memories` nor `pages`.
    db.upsert_documents(vec![crate::sources::RawDocument {
        source: "memory".to_string(),
        source_id: "mem_page_queue_dangling".to_string(),
        title: "dangling".to_string(),
        content: "revision of a target that does not exist".to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        supersedes: Some("page_does_not_exist".to_string()),
        pending_revision: true,
        ..Default::default()
    }])
    .await
    .unwrap();

    let listed = db
        .list_pending_revisions_scoped(10, &crate::read_scope::ReadScope::Global)
        .await
        .unwrap();
    let card = listed
        .iter()
        .find(|item| item.revision_source_id == card_id)
        .expect("a staged page revision card must reach the pending-revisions queue");
    assert_eq!(
        card.target_source_id, page_id,
        "the queue must name the page as the card's target, so accept/dismiss resolve it"
    );
    assert_eq!(card.revision_content, proposed_content);
    assert_eq!(card.source_agent.as_deref(), Some("page_write"));
    assert_eq!(
        card.target_kind,
        wenlan_types::responses::RevisionTargetKind::Page,
        "the reader must label a page card so its client knows to fetch a page"
    );
    assert!(
        listed
            .iter()
            .filter(|item| item.revision_source_id != card_id)
            .all(|item| item.target_kind == wenlan_types::responses::RevisionTargetKind::Memory),
        "only a page-target card may be labelled Page"
    );
    assert!(
        !listed
            .iter()
            .any(|item| item.revision_source_id == "mem_page_queue_dangling"),
        "a revision whose target exists in neither table must stay off the queue"
    );

    // An unfiled page is visible to the uncategorized scope and to no named one.
    assert!(
        db.list_pending_revisions_scoped(10, &crate::read_scope::ReadScope::Uncategorized)
            .await
            .unwrap()
            .iter()
            .any(|item| item.revision_source_id == card_id),
        "a card for a page with no workspace belongs to the uncategorized scope"
    );
    assert!(
        db.list_pending_revisions_scoped(
            10,
            &crate::read_scope::ReadScope::Space("work".to_string())
        )
        .await
        .unwrap()
        .is_empty(),
        "a card for a page with no workspace must not leak into a named Space"
    );

    // The app sends the queue's `target_source_id` to accept -- here, the page.
    let accepted = accept_pending_revision_with_knowledge_path(
        &db,
        &card.target_source_id,
        "test-agent",
        Some(knowledge_dir.path()),
    )
    .await
    .unwrap();
    assert_eq!(accepted.target_source_id, page_id);
    assert_eq!(accepted.revision_source_id, card_id);
    assert!(accepted.wrote);
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().content,
        proposed_content,
        "accepting from the queue must apply the proposed prose to the page"
    );
    assert!(
        !db.list_pending_revisions_scoped(10, &crate::read_scope::ReadScope::Global)
            .await
            .unwrap()
            .iter()
            .any(|item| item.revision_source_id == card_id),
        "an accepted card must leave the queue"
    );
}

/// A page card is scoped by the page's own `workspace`, the column every other
/// scoped page read gates on. The card's copied `space` is not the authority:
/// `stage_page_revision_card` fills it from `pages.space`, which is the page's
/// category column on distilled pages, not its scope.
#[tokio::test]
async fn pending_revision_queue_scopes_a_page_card_by_the_page_workspace() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_page_scope_original";
    let original_content = "Postgres row locks serialize concurrent updates";
    let human_content = "Postgres row locks serialize concurrent updates, with human notes";
    let proposed_content = "Postgres takes a row lock so concurrent updates cannot interleave";

    seed_memory(&db, mem_id, original_content).await;
    let page_id = create_page(
        &db,
        CreateConceptRequest {
            title: "Postgres row locks".to_string(),
            content: original_content.to_string(),
            summary: None,
            entity_id: None,
            space: (None).into(),
            source_memory_ids: vec![mem_id.to_string()],
            creation_kind: Some("research".to_string()),
            workspace: Some("work".to_string()),
        },
        "test",
        None,
    )
    .await
    .unwrap()
    .id;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.workspace.as_deref(), Some("work"));
    let card_id = stage_page_revision_card(
        &db,
        &page,
        proposed_content,
        &[mem_id.to_string()],
        0, // never accepted in this test
        "page_growth",
        None,
    )
    .await
    .unwrap()
    .revision_card_id
    .expect("staged page card must return an id");

    async fn lists_card(db: &MemoryDB, scope: crate::read_scope::ReadScope, card_id: &str) -> bool {
        db.list_pending_revisions_scoped(10, &scope)
            .await
            .unwrap()
            .iter()
            .any(|item| item.revision_source_id == card_id)
    }
    assert!(
        lists_card(
            &db,
            crate::read_scope::ReadScope::Space("work".to_string()),
            &card_id
        )
        .await,
        "the card must be listed in the page's own Space"
    );
    assert!(
        !lists_card(
            &db,
            crate::read_scope::ReadScope::Space("personal".to_string()),
            &card_id
        )
        .await,
        "the card must not be listed in another Space"
    );
    assert!(
        !lists_card(&db, crate::read_scope::ReadScope::Uncategorized, &card_id).await,
        "a workspace-bound page's card is not uncategorized"
    );
    assert!(
        lists_card(&db, crate::read_scope::ReadScope::Global, &card_id).await,
        "the card must be listed globally"
    );

    // Dismiss resolves the same `target_source_id` the queue hands the app.
    let dismissed = dismiss_pending_revision(&db, &page_id, "test-agent")
        .await
        .unwrap();
    assert_eq!(dismissed.target_source_id, page_id);
    assert!(
        !lists_card(
            &db,
            crate::read_scope::ReadScope::Space("work".to_string()),
            &card_id
        )
        .await,
        "a dismissed card must leave the queue"
    );
    assert_eq!(
        db.get_page(&page_id).await.unwrap().unwrap().content,
        human_content,
        "dismissing must leave the human's prose untouched"
    );
}

#[tokio::test]
async fn accept_page_revision_consume_failure_keeps_page_retryable() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_page_accept_abort_original";
    let new_mem_id = "mem_page_accept_abort_new";
    let original_content = "original page content before failed revision acceptance";
    let proposed_content = "proposed page content must commit with card consumption";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(&db, new_mem_id, proposed_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string(), new_mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card.revision_card_id.unwrap();

    {
        let conn = db.test_primary_session().await;
        conn.execute_batch(&format!(
            "CREATE TRIGGER abort_page_revision_consume
             BEFORE DELETE ON memories
             WHEN OLD.source_id = '{}' AND OLD.pending_revision = 1
             BEGIN SELECT RAISE(ABORT, 'blocked revision consume'); END;",
            card_id.replace('\'', "''")
        ))
        .await
        .unwrap();
    }

    let err = accept_pending_revision(&db, &card_id, "test-agent")
        .await
        .expect_err("consume fault must fail the acceptance");
    assert!(err.to_string().contains("blocked revision consume"));
    let after_failure = db.get_page(&page_id).await.unwrap().unwrap();
    let pending = db.list_pending_revisions(10).await.unwrap();
    assert!(pending
        .iter()
        .any(|revision| revision.revision_source_id == card_id));
    {
        let conn = db.test_primary_session().await;
        conn.execute("DROP TRIGGER abort_page_revision_consume", ())
            .await
            .unwrap();
    }
    let retry = accept_pending_revision(&db, &card_id, "test-agent").await;
    assert_eq!(
        after_failure.content, before.content,
        "failed card consumption must not commit Page content first"
    );
    assert_eq!(
        after_failure.version, before.version,
        "failed card consumption must leave the Page version retryable"
    );
    assert!(
        retry.is_ok(),
        "retry after the fault is removed must converge"
    );
    assert!(db.list_pending_revisions(10).await.unwrap().is_empty());
}

#[tokio::test]
async fn accept_page_revision_source_failure_keeps_page_retryable() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_page_source_abort_original";
    let new_mem_id = "mem_page_source_abort_new";
    let original_content = "original page content before source attachment failure";
    let proposed_content = "proposed page content must commit with exact sources";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(&db, new_mem_id, proposed_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string(), new_mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card.revision_card_id.unwrap();

    {
        let conn = db.test_primary_session().await;
        // G6 Stage 2 PR 2b: `insert_resolved_page_evidence` stopped writing
        // `page_sources` -- `edges` is the sole live producer of `cites`
        // edges now. Re-point the fault at the same-transaction edges INSERT
        // so this still exercises a mid-acceptance source-attachment abort.
        conn.execute_batch(&format!(
            "CREATE TRIGGER abort_page_revision_source
             BEFORE INSERT ON edges
             WHEN NEW.edge_type = 'cites' AND NEW.dst_id = '{}'
             BEGIN SELECT RAISE(ABORT, 'blocked revision source attachment'); END;",
            new_mem_id.replace('\'', "''")
        ))
        .await
        .unwrap();
    }

    let err = accept_pending_revision(&db, &card_id, "test-agent")
        .await
        .expect_err("source attachment fault must fail the acceptance");
    assert!(err
        .to_string()
        .contains("blocked revision source attachment"));
    let after_failure = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(after_failure.content, before.content);
    assert_eq!(after_failure.version, before.version);
    assert_eq!(after_failure.source_memory_ids, before.source_memory_ids);
    assert!(db
        .list_pending_revisions(10)
        .await
        .unwrap()
        .iter()
        .any(|revision| revision.revision_source_id == card_id));

    {
        let conn = db.test_primary_session().await;
        conn.execute("DROP TRIGGER abort_page_revision_source", ())
            .await
            .unwrap();
    }
    accept_pending_revision(&db, &card_id, "test-agent")
        .await
        .expect("retry after the source fault must converge");
    assert!(db.list_pending_revisions(10).await.unwrap().is_empty());
}

#[tokio::test]
async fn accept_pending_revision_page_write_card_conflicts_when_page_version_changed() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_page_accept_conflict_original";
    let new_mem_id = "mem_page_accept_conflict_new";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content =
        "Rust ownership lets the compiler enforce memory safety during page refresh";
    let newer_human_content =
        "Rust ownership keeps memory safety rules explicit, with newer human notes";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(&db, new_mem_id, proposed_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_version = before.version;
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string(), new_mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");

    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: newer_human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let err = accept_pending_revision(&db, card_id, "test-agent")
        .await
        .unwrap_err();
    match err {
        WenlanError::Conflict(msg) => {
            assert!(
                msg.contains(&format!("staged version {staged_version}")),
                "conflict message must name the staged version, got: {msg}"
            );
            assert!(
                msg.contains(&format!("current version {}", staged_version + 1)),
                "conflict message must name the current version, got: {msg}"
            );
        }
        other => panic!("expected version conflict, got {other:?}"),
    }

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, newer_human_content,
        "stale page-write card must not overwrite newer human prose"
    );
    assert!(
        db.list_pending_revisions(10)
            .await
            .unwrap()
            .iter()
            .any(|row| row.revision_source_id == card_id),
        "conflicted page-write card must remain pending"
    );
}

/// Round-5 finding F1 (MEDIUM): a human-owned page's staged revision card
/// used to fence acceptance on `page_version` alone. `link_page_source`
/// bumps only `source_revision`, leaving `version` unchanged, so a source
/// attached after the card was staged (but before the human accepted it)
/// passed the version-only CAS and was silently dropped when the card's
/// (now-stale) source list overwrote it. Same lost-update class rounds 3/4
/// fixed for other write paths.
#[tokio::test]
async fn accept_pending_revision_page_write_card_conflicts_when_a_source_attached_after_staging() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_card_fence_original";
    let attached_mem_id = "mem_attached_after_staging";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content =
        "Rust ownership lets the compiler enforce memory safety during page refresh";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(
        &db,
        attached_mem_id,
        "A source attached after the card was staged",
    )
    .await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_version = before.version;
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .clone()
        .expect("staged page card must return an id");

    db.link_page_source(&page_id, attached_mem_id, "concurrent_attach")
        .await
        .unwrap();
    let mid = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        mid.version, staged_version,
        "sanity: a source attach must not bump version -- only source_revision"
    );

    let err = accept_pending_revision(&db, &card_id, "test-agent")
        .await
        .unwrap_err();
    match err {
        WenlanError::Conflict(msg) => {
            assert!(
                msg.contains("staged source revision"),
                "conflict message must name the source-revision fence, got: {msg}"
            );
        }
        other => panic!("expected a source-revision conflict, got {other:?}"),
    }

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, human_content,
        "a card staged against a stale source_revision must not overwrite the page"
    );
    assert!(
        after
            .source_memory_ids
            .contains(&attached_mem_id.to_string()),
        "the concurrently attached source must survive the rejected accept"
    );
    assert!(
        db.list_pending_revisions(10)
            .await
            .unwrap()
            .iter()
            .any(|row| row.revision_source_id == card_id),
        "conflicted page-write card must remain pending"
    );
}

/// Companion to the conflict test above: when NOTHING attaches between
/// staging and accept, the same source-revision fence must let the write
/// through exactly as before.
#[tokio::test]
async fn accept_pending_revision_page_write_card_with_matching_source_revision_writes() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_card_fence_match_original";
    let original_content = "Rust borrowing enforces aliasing rules at compile time";
    let human_content = "Rust borrowing enforces aliasing rules at compile time, with human notes";
    let proposed_content = "Rust borrowing lets the compiler enforce aliasing during page refresh";

    seed_memory(&db, mem_id, original_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .clone()
        .expect("staged page card must return an id");

    accept_pending_revision(&db, &card_id, "test-agent")
        .await
        .unwrap();

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, proposed_content,
        "an unchanged source_revision must accept and overwrite the page"
    );
    assert_eq!(after.source_memory_ids, vec![mem_id.to_string()]);
    assert!(
        !db.list_pending_revisions(10)
            .await
            .unwrap()
            .iter()
            .any(|row| row.revision_source_id == card_id),
        "an accepted card must be consumed"
    );
}

/// Issue #650: a page long enough to chunk must survive its own accept.
///
/// `memories` stores one row per chunk, and every chunk of a staged card
/// shares the card's `source_id` and `last_modified`. A read that takes one
/// row therefore returns an arbitrary fragment, which the accept path then
/// writes as the page's complete new body -- observed live turning a 21,038
/// character page into 1,814. The staged card is not at fault:
/// `stage_page_revision_card` puts the whole body in `source_text` on every
/// chunk row, so the whole body is always there to be read.
///
/// The chunking assertion below is the test's own gate: a fixture that never
/// split would pass against the broken read and prove nothing. The splitters
/// cap a chunk at 1,500 characters for markdown, 512 for plain text, or 510
/// BGE tokens with a tokenizer loaded, so this fixture is sized past all of
/// them.
#[tokio::test]
async fn accepting_a_multi_chunk_page_card_keeps_the_whole_body() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_650_multi_chunk";
    let original_content = "Ownership rules make Rust memory safety explicit";
    // Kept close to the cited source: the human write runs through the same
    // hallucination guard, and an unrelated sentence fails it before the test
    // reaches the behaviour under test.
    let human_content =
        "Ownership rules make Rust memory safety explicit. A human maintains this page.";

    // Long enough to split, and structured so the split lands on section
    // boundaries the way a real wiki page does.
    let proposed_content = (1..=8)
        .map(|section| {
            format!(
                "## Section {section}\n\n{}\n",
                format!(
                    "Paragraph {section} of the proposed revision carries enough prose to \
                     push this page past the chunker's size budget so the write lands as \
                     several rows rather than one. "
                )
                .repeat(4)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        proposed_content.len() > 1500,
        "fixture must exceed the character splitter's budget, got {} chars",
        proposed_content.len()
    );

    seed_memory(&db, mem_id, original_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        &proposed_content,
        &[mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");

    // Gate: the card must really have chunked, or the assertions below would
    // pass against the broken read and prove nothing. The chunk this row
    // carries being shorter than what was staged is exactly that proof.
    let payload = db
        .pending_memory_revision_payload(card_id)
        .await
        .unwrap()
        .expect("the staged card must resolve");
    assert!(
        payload.content.len() < proposed_content.len(),
        "fixture must stage a card that actually chunked: one row already holds \
         all {} characters",
        proposed_content.len()
    );
    assert_eq!(
        payload.full_body.as_deref(),
        Some(proposed_content.as_str()),
        "the row must expose the whole staged body beside its chunk"
    );

    // The queue a human reads before clicking accept must show the same body
    // the accept will write. Previewing one chunk hides what is about to
    // happen at the exact moment the human is deciding.
    let queued = db
        .list_pending_revisions_scoped(10, &crate::read_scope::ReadScope::Global)
        .await
        .unwrap();
    let queued_card = queued
        .iter()
        .find(|item| item.revision_source_id == card_id)
        .expect("the staged card must appear in the pending queue");
    assert_eq!(
        queued_card.revision_content,
        proposed_content,
        "the review queue must preview the whole staged body, not one chunk \
         (got {} chars, staged {} chars)",
        queued_card.revision_content.len(),
        proposed_content.len()
    );

    accept_pending_revision(&db, card_id, "test-agent")
        .await
        .unwrap();

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content,
        proposed_content,
        "accepting a chunked card must write the whole staged body, not one chunk \
         (got {} chars, staged {} chars)",
        after.content.len(),
        proposed_content.len()
    );
}

/// Recovering the whole staged body must not smuggle raw personal data past
/// the redaction the chunk rows already carry.
///
/// `upsert_documents` runs `redact_pii` on `content` before chunking but copies
/// `source_text` in verbatim, so a read that recovers the body from
/// `source_text` and hands it straight to the accept path would write an email
/// address or a card number into the page and into its exported markdown --
/// worse than the truncation of issue #650, and silent in the same way.
#[tokio::test]
async fn accepting_a_chunked_page_card_keeps_the_redaction_its_chunks_carry() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_650_redaction";
    let original_content = "Ownership rules make Rust memory safety explicit";
    let human_content =
        "Ownership rules make Rust memory safety explicit. A human maintains this page.";

    let leak = "alice@example.com";
    let proposed_content = (1..=8)
        .map(|section| {
            format!(
                "## Section {section}\n\n{}Reach the maintainer at {leak} for details.\n",
                "Paragraph of the proposed revision, long enough that this page \
                 splits into several rows rather than one. "
                    .repeat(4)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    seed_memory(&db, mem_id, original_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        &proposed_content,
        &[mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");

    accept_pending_revision(&db, card_id, "test-agent")
        .await
        .unwrap();

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert!(
        !after.content.contains(leak),
        "the raw address must never reach the page body"
    );
    assert!(
        after.content.contains("[REDACTED:EMAIL]"),
        "the page must carry the same redaction marker the chunk rows do"
    );
    assert_eq!(
        after.content,
        crate::privacy::redact_pii(&proposed_content),
        "the page must equal the staged body with exactly the write-time \
         redaction applied, nothing more and nothing less"
    );
}

/// A human edit to a staged card is what its accept must write.
///
/// `apply_memory_update` rewrites chunk zero and deletes the rest without
/// touching `source_text`, so an edited card is one row whose `source_text`
/// still holds the body from before the edit. Recovering the body from that
/// column unconditionally would revert the edit at the moment of accepting it
/// -- the same silent wrong-body write as issue #650, pointed the other way.
#[tokio::test]
async fn accepting_an_edited_page_card_writes_the_edit_not_the_staged_body() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_650_edited";
    let original_content = "Ownership rules make Rust memory safety explicit";
    let human_content =
        "Ownership rules make Rust memory safety explicit. A human maintains this page.";

    // Staged long enough to chunk, so the row starts out in exactly the shape
    // issue #650 is about.
    let proposed_content = (1..=8)
        .map(|section| {
            format!(
                "## Section {section}\n\n{}\n",
                "Paragraph of the proposed revision, long enough that this page \
                 splits into several rows rather than one. "
                    .repeat(4)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    seed_memory(&db, mem_id, original_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        &proposed_content,
        &[mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");

    // Gate: the card chunked, so before the edit the whole body is only
    // reachable through `source_text`.
    let staged = db
        .pending_memory_revision_payload(card_id)
        .await
        .unwrap()
        .expect("the staged card must resolve");
    assert_eq!(
        staged.full_body.as_deref(),
        Some(proposed_content.as_str()),
        "a chunked card must expose its whole staged body"
    );

    let edited_content = format!("{proposed_content}\n## Reviewed\n\nA human rewrote the end.\n");
    update_memory(
        &db,
        card_id,
        MemoryUpdate {
            content: Some(&edited_content),
            space: None,
            confirm: false,
            memory_type: None,
        },
    )
    .await
    .unwrap();

    // The edit collapsed the card to one row, so `content` is authoritative
    // again and the stale `source_text` must be left alone.
    let edited = db
        .pending_memory_revision_payload(card_id)
        .await
        .unwrap()
        .expect("the edited card must resolve");
    assert_eq!(
        edited.content, edited_content,
        "the edited card's own row must hold the whole edited body"
    );
    assert_eq!(
        edited.full_body, None,
        "an unchunked card must not offer a body recovered from the stale \
         `source_text` it kept from before the edit"
    );

    let queued = db
        .list_pending_revisions_scoped(10, &crate::read_scope::ReadScope::Global)
        .await
        .unwrap();
    let queued_card = queued
        .iter()
        .find(|item| item.revision_source_id == card_id)
        .expect("the edited card must appear in the pending queue");
    assert_eq!(
        queued_card.revision_content, edited_content,
        "the review queue must preview the edit, not the body staged before it"
    );

    accept_pending_revision(&db, card_id, "test-agent")
        .await
        .unwrap();

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, edited_content,
        "accepting an edited card must write the edit, not the staged body it \
         replaced"
    );
}

/// The queue must keep previewing a memory card's stored `content`.
///
/// Only a page card previews `source_text`. On a memory card `source_text` is
/// the prose the row was distilled from, which its accept never writes, so
/// previewing it would show the human text that accepting cannot produce.
#[tokio::test]
async fn a_memory_revision_card_still_previews_its_stored_content() {
    let (db, _dir) = test_db().await;
    let target_id = "mem_650_memory_target";
    seed_memory(&db, target_id, "Postgres stores JSON in a JSONB column").await;

    let stored = "Postgres stores JSON in a JSONB column, which is indexable";
    let pre_distillation = "So anyway I looked it up and Postgres has this JSONB thing";
    let card = crate::sources::RawDocument {
        source: "memory".to_string(),
        source_id: "mem_650_memory_card".to_string(),
        title: "Revision: Postgres".to_string(),
        content: stored.to_string(),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confidence: Some(0.9),
        supersedes: Some(target_id.to_string()),
        pending_revision: true,
        source_text: Some(pre_distillation.to_string()),
        ..Default::default()
    };
    db.upsert_documents(vec![card]).await.unwrap();

    let queued = db
        .list_pending_revisions_scoped(10, &crate::read_scope::ReadScope::Global)
        .await
        .unwrap();
    let item = queued
        .iter()
        .find(|item| item.revision_source_id == "mem_650_memory_card")
        .expect("the memory card must appear in the pending queue");
    assert_eq!(
        item.revision_content, stored,
        "a memory card previews the text its accept stores, not the prose it \
         was distilled from"
    );
}

/// Proves the source_revision fence is actually wired end-to-end through
/// the daemon-internal path a machine refresh uses to reach a human-owned
/// page's gate: `update_page_at_source_revision` (the same wrapper
/// `synthesis::distill`'s refresh calls) threads `expected_source_revision`
/// into `stage_page_revision_card`, which must record it on the staged
/// card's `structured_fields` rather than silently dropping it.
#[tokio::test]
async fn stage_page_revision_card_via_fenced_update_records_source_revision() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_fenced_card_source_revision";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content = "Rust ownership lets the compiler enforce memory safety during refresh";

    seed_memory(&db, mem_id, original_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let source_revision = db.get_page_source_revision(&page_id).await.unwrap();

    let result = update_page_at_source_revision(
        &db,
        &page_id,
        UpdatePageRequest {
            content: proposed_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "re_distill",
        false,
        source_revision,
        None,
        None,
    )
    .await
    .unwrap();

    assert!(
        result.gated,
        "a machine write to a human-owned page must be gated as a revision card"
    );
    let revision_card_id = result
        .revision_card_id
        .expect("gated write must return a card id");

    let conn = db.test_primary_session().await;
    let mut rows = conn
        .query(
            "SELECT structured_fields FROM memories WHERE source_id = ?1",
            libsql::params![revision_card_id.clone()],
        )
        .await
        .unwrap();
    let row = rows
        .next()
        .await
        .unwrap()
        .expect("revision card row must be persisted");
    let structured_fields = row.get::<String>(0).unwrap();
    drop(rows);
    drop(conn);

    let structured: serde_json::Value = serde_json::from_str(&structured_fields).unwrap();
    assert_eq!(
        structured["source_revision"], source_revision,
        "a card staged through the fenced daemon-internal update path must record the \
         source_revision it was staged from, structured_fields: {structured_fields}"
    );
}

#[tokio::test]
async fn accept_pending_revision_legacy_page_write_card_without_version_still_accepts() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_page_accept_legacy_original";
    let new_mem_id = "mem_page_accept_legacy_new";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content =
        "Rust ownership lets the compiler enforce memory safety during page refresh";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(&db, new_mem_id, proposed_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string(), new_mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id");
    {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT structured_fields FROM memories WHERE source_id = ?1",
                libsql::params![card_id.to_string()],
            )
            .await
            .unwrap();
        let row = rows
            .next()
            .await
            .unwrap()
            .expect("revision card row must exist");
        let structured_fields = row.get::<String>(0).unwrap();
        drop(rows);

        let mut structured: serde_json::Value = serde_json::from_str(&structured_fields).unwrap();
        structured
            .as_object_mut()
            .expect("structured_fields must be an object")
            .remove("page_version");
        conn.execute(
            "UPDATE memories SET structured_fields = ?1 WHERE source_id = ?2",
            libsql::params![structured.to_string(), card_id.to_string()],
        )
        .await
        .unwrap();
    }

    let accepted = accept_pending_revision(&db, card_id, "test-agent")
        .await
        .unwrap();
    assert_eq!(accepted.target_source_id, page_id);
    assert_eq!(accepted.revision_source_id, card_id);
    assert!(accepted.wrote);

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, proposed_content,
        "legacy page-write cards without page_version must still accept"
    );
}

/// A card staged before source-revision fencing (PR #598) records no base to
/// check its evidence against, and `version` cannot stand in for one: attaching
/// or detaching a source leaves `version` untouched. Accepting such a card on
/// the version fence alone would write prose citing sources the page may no
/// longer hold. It is refused, deleted, and the page re-queued instead.
#[tokio::test]
async fn accept_pending_revision_discards_a_page_card_staged_before_source_revision_fencing() {
    let (db, _dir) = test_db().await;
    let mem_id = "mem_page_accept_prefence_original";
    let new_mem_id = "mem_page_accept_prefence_new";
    let original_content = "Rust ownership keeps memory safety rules explicit";
    let human_content = "Rust ownership keeps memory safety rules explicit, with human notes";
    let proposed_content =
        "Rust ownership lets the compiler enforce memory safety during page refresh";

    seed_memory(&db, mem_id, original_content).await;
    seed_memory(&db, new_mem_id, proposed_content).await;
    let page_id = seed_page(&db, mem_id, original_content).await;
    update_page(
        &db,
        &page_id,
        UpdatePageRequest {
            content: human_content.to_string(),
            source_memory_ids: vec![mem_id.to_string()],
            expected_version: None,
            caller_id: None,
            operation_id: None,
        },
        "fs_edit",
        false,
        None,
        None,
    )
    .await
    .unwrap();

    let before = db.get_page(&page_id).await.unwrap().unwrap();
    let staged_source_revision = db.get_page_source_revision(&page_id).await.unwrap();
    let card = stage_page_revision_card(
        &db,
        &before,
        proposed_content,
        &[mem_id.to_string(), new_mem_id.to_string()],
        staged_source_revision,
        "page_growth",
        None,
    )
    .await
    .unwrap();
    let card_id = card
        .revision_card_id
        .as_deref()
        .expect("staged page card must return an id")
        .to_string();

    // Age the card back to a pre-#598 staging: the field is gone entirely,
    // which is what the cards already sitting in real queues look like.
    {
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query(
                "SELECT structured_fields FROM memories WHERE source_id = ?1",
                libsql::params![card_id.clone()],
            )
            .await
            .unwrap();
        let row = rows
            .next()
            .await
            .unwrap()
            .expect("revision card row must exist");
        let structured_fields = row.get::<String>(0).unwrap();
        drop(rows);

        let mut structured: serde_json::Value = serde_json::from_str(&structured_fields).unwrap();
        structured
            .as_object_mut()
            .expect("structured_fields must be an object")
            .remove("source_revision");
        conn.execute(
            "UPDATE memories SET structured_fields = ?1 WHERE source_id = ?2",
            libsql::params![structured.to_string(), card_id.clone()],
        )
        .await
        .unwrap();
    }

    let err = accept_pending_revision(&db, &card_id, "test-agent")
        .await
        .unwrap_err();
    let WenlanError::Conflict(message) = err else {
        panic!("a card carrying no source_revision must be refused as a conflict: {err:?}");
    };
    assert!(
        message.contains(&card_id) && message.contains(&page_id),
        "the refusal must name the card and the page: {message}"
    );
    assert!(
        message.contains("discarded") && message.contains("re-queued"),
        "the refusal is permanent, so it must say the card is gone and the page will be \
         regenerated rather than invite a retry that can never succeed: {message}"
    );

    let after = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        after.content, human_content,
        "the stale card's prose must not reach the page"
    );
    assert_eq!(
        after.stale_reason.as_deref(),
        Some("source_updated"),
        "the page must be re-queued for regeneration"
    );
    assert!(
        db.get_page_source_revision(&page_id).await.unwrap() > staged_source_revision,
        "re-queueing must bump the source revision so no other card staged against the \
         old evidence can win a later race"
    );
    assert!(
        db.pending_memory_revision_payload(&card_id)
            .await
            .unwrap()
            .is_none(),
        "the discarded card must be gone from the pending queue"
    );
}

#[tokio::test]
async fn accept_pending_revision_returns_not_found_on_missing_id() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let err = accept_pending_revision(&db, "mem_nope", "test-agent")
        .await
        .unwrap_err();
    assert!(matches!(err, WenlanError::NotFound(_)));
}

#[tokio::test]
async fn accept_pending_revision_returns_not_found_on_re_call_after_success() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_pending_revision(&db, "mem_arr_target", "mem_arr_rev").await;
    accept_pending_revision(&db, "mem_arr_target", "test-agent")
        .await
        .unwrap();
    let err = accept_pending_revision(&db, "mem_arr_target", "test-agent")
        .await
        .unwrap_err();
    assert!(matches!(err, WenlanError::NotFound(_)));
}

// ── proposal gate (task #7): memory-target revision accept/dismiss ──────

/// Accept must both suppress the original and make the correction the one
/// search returns -- checking `confirmed`/`pending_revision` alone would
/// miss a retrieval-side predicate drifting out of sync with them.
#[tokio::test]
async fn accepting_a_gated_memory_correction_makes_it_live_and_hides_the_target() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_pending_revision(&db, "mem_gate_accept_target", "mem_gate_accept_rev").await;

    accept_pending_revision_with_knowledge_path(&db, "mem_gate_accept_target", "test-agent", None)
        .await
        .unwrap();

    let target = db
        .get_memory_detail("mem_gate_accept_target")
        .await
        .unwrap()
        .expect("the superseded original must still exist");
    assert!(!target.confirmed, "accept must suppress the original");

    let revision = db
        .get_memory_detail("mem_gate_accept_rev")
        .await
        .unwrap()
        .expect("the accepted revision must now be non-pending");
    assert!(revision.confirmed, "the accepted correction must be live");

    let hits = db
        .search_memory(
            "revised content",
            10,
            None,
            &crate::read_scope::ReadScope::Global,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    let ids: Vec<_> = hits.into_iter().map(|r| r.source_id).collect();
    assert!(
        ids.contains(&"mem_gate_accept_rev".to_string()),
        "search must surface the now-live correction, got: {ids:?}"
    );
    assert!(
        !ids.contains(&"mem_gate_accept_target".to_string()),
        "search must no longer surface the superseded original, got: {ids:?}"
    );
}

/// Default 6: two staged corrections on one target are resolved by
/// `ORDER BY last_modified DESC` when addressed by target id. Accepting the
/// newest must not touch the older sibling's pending state.
#[tokio::test]
async fn accepting_the_latest_of_two_staged_corrections_leaves_the_older_pending() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let now = chrono::Utc::now().timestamp();
    let conn = db.test_primary_session().await;
    conn.execute(
        "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source) VALUES ('mem_two_target', 'mem_two_target', 'mem_two_target', 'original content', 0, 'text', 'fact', 'test', 'claude-code', ?1, ?1, 1, 'confirmed', 'memory')",
        libsql::params![now],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source, supersedes, pending_revision) VALUES ('mem_two_rev_older', 'mem_two_rev_older', 'mem_two_rev_older', 'older revision content', 0, 'text', 'fact', 'test', 'claude-code', ?1, ?1, 0, 'new', 'memory', 'mem_two_target', 1)",
        libsql::params![now],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO memories (id, source_id, title, content, chunk_index, chunk_type, memory_type, space, source_agent, created_at, last_modified, confirmed, stability, source, supersedes, pending_revision) VALUES ('mem_two_rev_newer', 'mem_two_rev_newer', 'mem_two_rev_newer', 'newer revision content', 0, 'text', 'fact', 'test', 'claude-code', ?1, ?1, 0, 'new', 'memory', 'mem_two_target', 1)",
        libsql::params![now + 1],
    )
    .await
    .unwrap();
    drop(conn);

    let accepted = accept_pending_revision(&db, "mem_two_target", "test-agent")
        .await
        .unwrap();
    assert_eq!(
        accepted.revision_source_id, "mem_two_rev_newer",
        "addressing by target id must resolve to the latest staged revision"
    );

    let newer = db
        .get_memory_detail("mem_two_rev_newer")
        .await
        .unwrap()
        .expect("the accepted revision must be non-pending");
    assert!(
        newer.confirmed,
        "the accepted (latest) revision must be live"
    );

    // get_memory_detail filters pending_revision = 1 rows out entirely, so
    // its continued absence here is itself the "still pending" assertion.
    assert!(
        db.get_memory_detail("mem_two_rev_older")
            .await
            .unwrap()
            .is_none(),
        "a still-pending sibling must not surface through get_memory_detail"
    );
    let queue = db.list_pending_revisions(10).await.unwrap();
    assert!(
        queue
            .iter()
            .any(|item| item.revision_source_id == "mem_two_rev_older"),
        "the older sibling must remain in the pending-revisions queue, got: {queue:?}"
    );
}

// ── dismiss_pending_revision ─────────────────────────────────────────────

#[tokio::test]
async fn dismiss_pending_revision_writes_and_logs_on_first_call() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_pending_revision(&db, "mem_dpr_target", "mem_dpr_rev").await;
    let result = dismiss_pending_revision(&db, "mem_dpr_target", "test-agent")
        .await
        .unwrap();
    assert_eq!(result.target_source_id, "mem_dpr_target");
    assert!(result.wrote);
}

#[tokio::test]
async fn dismiss_pending_revision_returns_not_found_on_missing_id() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let err = dismiss_pending_revision(&db, "mem_nope", "test-agent")
        .await
        .unwrap_err();
    assert!(matches!(err, WenlanError::NotFound(_)));
}

#[tokio::test]
async fn dismiss_pending_revision_returns_not_found_on_re_call() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_pending_revision(&db, "mem_dpr2_target", "mem_dpr2_rev").await;
    dismiss_pending_revision(&db, "mem_dpr2_target", "test-agent")
        .await
        .unwrap();
    let err = dismiss_pending_revision(&db, "mem_dpr2_target", "test-agent")
        .await
        .unwrap_err();
    assert!(matches!(err, WenlanError::NotFound(_)));
}

/// Regression guard for the spec's read of the brief: dismiss unstages, it
/// does not delete. Both rows must survive, the false `supersedes` link
/// must clear, and the target must be untouched.
#[tokio::test]
async fn dismissing_a_gated_memory_correction_keeps_both_rows() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    seed_pending_revision(&db, "mem_gate_dismiss_target", "mem_gate_dismiss_rev").await;

    dismiss_pending_revision(&db, "mem_gate_dismiss_target", "test-agent")
        .await
        .unwrap();

    let revision = db
        .get_memory_detail("mem_gate_dismiss_rev")
        .await
        .unwrap()
        .expect("dismiss must unstage, not delete, the correction");
    assert!(
        revision.supersedes.is_none(),
        "dismiss must clear the supersedes link"
    );
    assert!(
        !revision.confirmed,
        "an unstaged correction is an ordinary new memory, not auto-confirmed"
    );

    let target = db
        .get_memory_detail("mem_gate_dismiss_target")
        .await
        .unwrap()
        .expect("the target must remain after dismiss");
    assert!(target.confirmed, "dismiss must leave the target untouched");
}

// ── dismiss_contradiction ────────────────────────────────────────────────

#[tokio::test]
async fn dismiss_contradiction_writes_and_returns_wrote_true() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let result = dismiss_contradiction(&db, "mem_any_source_id", "test-agent")
        .await
        .unwrap();
    assert_eq!(result.source_id, "mem_any_source_id");
    assert!(result.wrote);
}

#[tokio::test]
async fn dismiss_contradiction_logs_activity_once_per_call() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    dismiss_contradiction(&db, "mem_one", "test-agent")
        .await
        .unwrap();
    let conn = db.test_primary_session().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM agent_activity WHERE action = 'contradiction_dismiss' AND memory_ids = 'mem_one'",
            libsql::params![],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let count: i64 = row.get(0).unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn dismiss_contradiction_swallows_no_rows_matched() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    // No contradiction rows seeded — DB method is silent-idempotent
    let result = dismiss_contradiction(&db, "mem_no_contradictions", "test-agent")
        .await
        .unwrap();
    assert!(
        result.wrote,
        "wrote=true even with no rows matched (best-effort signal per §3 caveat)"
    );
}

// ── T16: MinHash/LSH entity near-dedup cascade (Step 2.5) ────────────────
//
// Test pair "Vorpalblade Jabberwock Inc" / "Vorpalblade Jabberwock Ino" is chosen so the char-trigram
// Jaccard is >= 0.9 (MinHash auto-merges) while the BGE vector distance is
// ~0.13 (> 0.1, so the existing vector step does NOT merge them). That
// separation is what lets the flag-OFF noop test prove byte-identity:
// without MinHash these stay two distinct entities.

fn entity_req(name: &str, etype: &str) -> CreateEntityRequest {
    CreateEntityRequest {
        name: name.to_string(),
        entity_type: etype.to_string(),
        space: (None).into(),
        source_agent: Some("test".to_string()),
        confidence: None,
    }
}

#[tokio::test]
async fn create_entity_minhash_merges_abbreviation() {
    temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", Some("1"))], async {
        let (db, _dir) = test_db().await;
        let first = create_entity(
            &db,
            entity_req("Vorpalblade Jabberwock Inc", "project"),
            "test",
        )
        .await
        .unwrap();
        let second = create_entity(
            &db,
            entity_req("Vorpalblade Jabberwock Ino", "project"),
            "test",
        )
        .await
        .unwrap();
        assert_eq!(
            first.id, second.id,
            "near-dup must resolve to the first entity id"
        );
        assert!(
            !second.wrote,
            "resolved-existing must not write a new entity"
        );
        // A "minhash" alias must have been recorded for the second name.
        let resolved = db
            .resolve_entity_by_alias(&"Vorpalblade Jabberwock Ino".to_lowercase())
            .await
            .unwrap();
        assert_eq!(resolved, Some(first.id));
    })
    .await;
}

#[tokio::test]
async fn create_entity_minhash_respects_type_guard() {
    temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", Some("1"))], async {
        let (db, _dir) = test_db().await;
        let first = create_entity(
            &db,
            entity_req("Vorpalblade Jabberwock Inc", "project"),
            "test",
        )
        .await
        .unwrap();
        // Same near-dup name but a DIFFERENT entity type must not auto-merge.
        let second = create_entity(
            &db,
            entity_req("Vorpalblade Jabberwock Ino", "person"),
            "test",
        )
        .await
        .unwrap();
        assert_ne!(
            first.id, second.id,
            "cross-type near-dup must NOT auto-merge (same-type guard)"
        );
        assert!(second.wrote, "a new entity should have been created");
    })
    .await;
}

#[tokio::test]
async fn create_entity_minhash_short_name_skips_fuzzy() {
    temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", Some("1"))], async {
        let (db, _dir) = test_db().await;
        // "API"/"APIs" are below the entropy gate, so Step 2.5 must punt them
        // to the vector step and never record a "minhash" alias.
        let first = create_entity(&db, entity_req("API", "concept"), "test")
            .await
            .unwrap();
        let second = create_entity(&db, entity_req("APIs", "concept"), "test")
            .await
            .unwrap();
        // No band rows are written for low-entropy names, regardless of how
        // the vector step resolved them.
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM entity_minhash_bands", ())
            .await
            .unwrap();
        let band_count: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(
            band_count, 0,
            "low-entropy names must not be indexed into entity_minhash_bands"
        );
        drop(rows);
        // `conn` holds the primary connection's tokio Mutex guard
        // (`test_primary_session` -> `lock_owned`); drop it before the next
        // `db.*` call below re-locks the same mutex internally, or the two
        // acquisitions on this task self-deadlock.
        drop(conn);
        // G6 Stage 2 PR 2c sub-step 3 item 4: `entity_aliases` stops being
        // written, and `pages.aliases` carries no per-alias provenance
        // field, so the minhash-source distinction this test used to check
        // is no longer expressible. Assert the behavior that matters
        // instead: a short name must not have been cross-registered as the
        // other entity's alias via the page payload.
        assert_ne!(
            first.id, second.id,
            "short names must not have been minhash-merged into one entity"
        );
        assert_eq!(
            db.resolve_entity_by_alias("apis").await.unwrap(),
            Some(second.id),
            "APIs must resolve only to its own entity, never as API's alias"
        );
    })
    .await;
}

#[tokio::test]
async fn create_entity_minhash_disabled_is_noop() {
    // CRITICAL regression guard: with the flag OFF, the near-dup pair must
    // stay TWO separate entities (vector distance ~0.13 > 0.1 so the vector
    // step does not merge them), and NO minhash alias / band row is written.
    temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", None::<&str>)], async {
        let (db, _dir) = test_db().await;
        let first = create_entity(
            &db,
            entity_req("Vorpalblade Jabberwock Inc", "project"),
            "test",
        )
        .await
        .unwrap();
        let second = create_entity(
            &db,
            entity_req("Vorpalblade Jabberwock Ino", "project"),
            "test",
        )
        .await
        .unwrap();
        assert_ne!(
            first.id, second.id,
            "flag OFF must leave near-dups as distinct entities (byte-identity)"
        );
        assert!(second.wrote, "flag OFF must create a second entity");
        let conn = db.test_primary_session().await;
        let mut rows = conn
            .query("SELECT COUNT(*) FROM entity_minhash_bands", ())
            .await
            .unwrap();
        let band_count: i64 = rows.next().await.unwrap().unwrap().get(0).unwrap();
        assert_eq!(band_count, 0, "flag OFF must write zero band rows");
        drop(rows);
        // `conn` holds the primary connection's tokio Mutex guard
        // (`test_primary_session` -> `lock_owned`); drop it before the next
        // `db.*` call below re-locks the same mutex internally, or the two
        // acquisitions on this task self-deadlock.
        drop(conn);
        // G6 Stage 2 PR 2c sub-step 3 item 4: see the short-name test above
        // -- the minhash-source distinction is no longer expressible, so
        // assert the near-dup stayed unresolved via the page payload
        // instead of checking `entity_aliases.source`.
        assert_eq!(
            db.resolve_entity_by_alias("vorpalblade jabberwock ino")
                .await
                .unwrap(),
            Some(second.id),
            "flag OFF must leave the near-dup resolving only to its own entity"
        );
    })
    .await;
}

#[tokio::test]
async fn resolve_entity_bulk_minhash_mirrors_create_entity() {
    use crate::extract::ExtractedEntity;
    use std::collections::HashMap;
    temp_env::async_with_vars([("WENLAN_ENABLE_ENTITY_MINHASH", Some("1"))], async {
        let (db, _dir) = test_db().await;
        let mut cache: HashMap<String, String> = HashMap::new();
        let e1 = ExtractedEntity {
            name: "Vorpalblade Jabberwock Inc".to_string(),
            entity_type: "project".to_string(),
        };
        let (id1, new1) = crate::importer::resolve_entity_bulk(&db, &mut cache, &e1, "test")
            .await
            .unwrap();
        assert!(new1, "first bulk entity is newly created");
        // Fresh cache so the in-batch shortcut does not mask Step 2.5.
        let mut cache2: HashMap<String, String> = HashMap::new();
        let e2 = ExtractedEntity {
            name: "Vorpalblade Jabberwock Ino".to_string(),
            entity_type: "project".to_string(),
        };
        let (id2, new2) = crate::importer::resolve_entity_bulk(&db, &mut cache2, &e2, "test")
            .await
            .unwrap();
        assert_eq!(id1, id2, "bulk path must mirror create_entity merge");
        assert!(!new2, "bulk near-dup must resolve to existing, not create");
    })
    .await;
}

// Integration tests: update_page shrink-guard

#[tokio::test]
async fn update_page_shrink_guard_rejects_truncation() {
    let _lock = env_lock().await;
    // Guard ON + LLM-rewrite caller + body shrinks below threshold -> Err + DB unchanged
    std::env::set_var("WENLAN_MERGE_SHRINK_GUARD", "0.7");
    let (db, _dir) = test_db().await;
    let mem_id = "mem-sg-reject";
    // 100-char body
    let old_body = "a".repeat(100);
    seed_memory(&db, mem_id, &old_body).await;
    let page_id = seed_page(&db, mem_id, &old_body).await;

    // New body is only 60 chars: 60 < 100 * 0.7 = 70 -> should reject
    let short_body = "a".repeat(60);
    let req = UpdatePageRequest {
        content: short_body,
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "distill", false, None, None).await;
    assert!(
        matches!(result, Err(WenlanError::Validation(_))),
        "shrink-guard must reject truncated LLM rewrite"
    );

    // DB must still have the ORIGINAL body
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        page.content,
        "a".repeat(100),
        "body must be unchanged after rejection"
    );
    assert_eq!(page.version, 1, "version must not bump on rejection");
    std::env::remove_var("WENLAN_MERGE_SHRINK_GUARD");
}

#[tokio::test]
async fn update_page_shrink_guard_allows_growth() {
    let _lock = env_lock().await;
    // Guard ON + LLM-rewrite caller + body grows -> Ok
    std::env::set_var("WENLAN_MERGE_SHRINK_GUARD", "0.7");
    let (db, _dir) = test_db().await;
    let mem_id = "mem-sg-grow";
    let old_body = "a".repeat(50);
    seed_memory(&db, mem_id, &old_body).await;
    let page_id = seed_page(&db, mem_id, &old_body).await;

    let long_body = "a".repeat(200);
    let req = UpdatePageRequest {
        content: long_body.clone(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "page_growth", false, None, None).await;
    assert!(result.is_ok(), "shrink-guard must allow growing body");
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(page.content, long_body);
    std::env::remove_var("WENLAN_MERGE_SHRINK_GUARD");
}

#[tokio::test]
async fn update_page_shrink_guard_off_by_default() {
    let _lock = env_lock().await;
    // Guard UNSET: even extreme truncation must succeed (zero regression)
    std::env::remove_var("WENLAN_MERGE_SHRINK_GUARD");
    let (db, _dir) = test_db().await;
    let mem_id = "mem-sg-off";
    let old_body = "a".repeat(100);
    seed_memory(&db, mem_id, &old_body).await;
    let page_id = seed_page(&db, mem_id, &old_body).await;

    let tiny_body = "a".repeat(5); // 5 < 100 * 0.7 = 70, would fail if guard were ON
    let req = UpdatePageRequest {
        content: tiny_body.clone(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    let result = update_page(&db, &page_id, req, "distill", false, None, None)
        .await
        .unwrap();
    assert!(result.wrote, "guard OFF must allow any size update");
    let page = db.get_page(&page_id).await.unwrap().unwrap();
    assert_eq!(
        page.content, tiny_body,
        "content must update when guard is OFF"
    );
}

#[tokio::test]
async fn update_page_shrink_guard_skips_human_edits() {
    let _lock = env_lock().await;
    // Guard ON + human edited_by: guard never fires, update goes through
    std::env::set_var("WENLAN_MERGE_SHRINK_GUARD", "0.7");
    let (db, _dir) = test_db().await;
    let mem_id = "mem-sg-human";
    let old_body = "a".repeat(100);
    seed_memory(&db, mem_id, &old_body).await;
    let page_id = seed_page(&db, mem_id, &old_body).await;

    // 5 chars: would fail guard if LLM rewrite, but "manual_edit" is human
    let tiny_body = "a".repeat(5);
    let req = UpdatePageRequest {
        content: tiny_body.clone(),
        source_memory_ids: vec![mem_id.to_string()],
        expected_version: None,
        caller_id: None,
        operation_id: None,
    };
    // manual_edit bypasses hallucination guard AND is NOT an LLM rewrite
    // so shrink-guard must NOT fire even though the body shrinks drastically
    // (hallucination guard WILL fire for manual_edit -- seed with real-ish content)
    // Actually manual_edit triggers hallucination guard, so use fs_edit instead
    let result = update_page(&db, &page_id, req, "fs_edit", false, None, None).await;
    // fs_edit IS guarded by hallucination guard and will likely fail cos-sim check.
    // The key assertion: if it fails, it must NOT be a shrink-guard Validation error.
    // If it succeeds, the body must be updated.
    match result {
        Ok(wr) => {
            // Succeeded: body updated (hallucination guard passed)
            if wr.wrote {
                let page = db.get_page(&page_id).await.unwrap().unwrap();
                assert_eq!(page.content, tiny_body);
            }
        }
        Err(WenlanError::Validation(msg)) => {
            // Hallucination guard may reject: ensure it is NOT a shrink-guard message
            assert!(
                !msg.contains("shrink-guard"),
                "human edit must not be rejected by shrink-guard; got: {msg}"
            );
        }
        Err(e) => panic!("unexpected error: {e:?}"),
    }
    std::env::remove_var("WENLAN_MERGE_SHRINK_GUARD");
}

// merge_shrink_threshold parse tests

// These three read and write the same process-global env var as the
// `update_page_shrink_guard_*` integration tests above, but were the only
// members of that group that never took `env_lock()`. Run in parallel they
// raced each other (one test's `remove_var` landing between another's
// `set_var` and its assertion), which surfaces as whichever of the three
// happened to read at the wrong moment. They take the same lock now.
#[tokio::test]
async fn merge_shrink_threshold_unset_returns_none() {
    let _lock = env_lock().await;
    std::env::remove_var("WENLAN_MERGE_SHRINK_GUARD");
    assert!(merge_shrink_threshold().is_none());
}

#[tokio::test]
async fn merge_shrink_threshold_valid_float() {
    let _lock = env_lock().await;
    std::env::set_var("WENLAN_MERGE_SHRINK_GUARD", "0.7");
    assert_eq!(merge_shrink_threshold(), Some(0.7));
    std::env::remove_var("WENLAN_MERGE_SHRINK_GUARD");
}

#[tokio::test]
async fn merge_shrink_threshold_garbage_returns_none() {
    let _lock = env_lock().await;
    std::env::set_var("WENLAN_MERGE_SHRINK_GUARD", "garbage");
    assert!(merge_shrink_threshold().is_none());
    std::env::remove_var("WENLAN_MERGE_SHRINK_GUARD");
}

// is_llm_rewrite tests

#[test]
fn is_llm_rewrite_distill_true() {
    assert!(Writer::classify("distill").is_llm_rewrite());
    assert!(Writer::classify("re_distill").is_llm_rewrite());
    assert!(Writer::classify("page_growth").is_llm_rewrite());
    assert!(Writer::classify("refinery_merge").is_llm_rewrite());
}

#[test]
fn is_llm_rewrite_user_false() {
    assert!(!Writer::classify("user").is_llm_rewrite());
    assert!(!Writer::classify("manual_edit").is_llm_rewrite());
    assert!(!Writer::classify("fs_edit").is_llm_rewrite());
    assert!(!Writer::classify("api").is_llm_rewrite());
    assert!(!Writer::classify("").is_llm_rewrite());
}

// ── Writer classification ───────────────────────────────────────────────

/// Every `edited_by` value this build persists, paired with the three
/// authority answers the write gate derives from it. This table is the
/// characterization pin: it is exactly what the string helpers
/// (`is_machine_page_write` / `skip_hallucination_guard` /
/// `is_llm_rewrite`) returned before `Writer` replaced them, so a drift in
/// any single classification fails here rather than silently changing who
/// wins an ownership decision inside the CAS.
const WRITER_TABLE: &[(&str, bool, bool, bool)] = &[
    // edited_by            machine  skips_guard  llm_rewrite
    ("manual_edit", false, false, false),
    ("fs_edit", false, false, false),
    ("distill", true, true, true),
    ("re_distill", true, true, true),
    ("page_growth", true, true, true),
    ("refinery_merge", true, true, true),
    ("agent_refresh", true, true, false),
    // Persisted in `page_history.edited_by` / `pages.changelog` but never
    // routed through the write gate today — these paths write via the db
    // layer (db.rs `create`/`migration_84`, citations.rs, revision accept).
    // Unrecognized by the classifier, and unrecognized means machine.
    ("create", true, false, false),
    ("revision_accept", true, false, false),
    ("citation_backfill", true, false, false),
    ("migration_84", true, false, false),
];

#[test]
fn writer_table_pins_gate_classification() {
    for &(edited_by, machine, skips_guard, llm_rewrite) in WRITER_TABLE {
        let w = Writer::classify(edited_by);
        assert_eq!(w.is_machine(), machine, "is_machine for {edited_by:?}");
        assert_eq!(
            w.skips_hallucination_guard(),
            skips_guard,
            "skips_hallucination_guard for {edited_by:?}"
        );
        assert_eq!(
            w.is_llm_rewrite(),
            llm_rewrite,
            "is_llm_rewrite for {edited_by:?}"
        );
    }
}

/// The persisted-string contract: these bytes are already in users'
/// databases, so classification must never rewrite them.
#[test]
fn writer_round_trips_persisted_string() {
    for &(edited_by, ..) in WRITER_TABLE {
        assert_eq!(
            Writer::classify(edited_by).as_str(),
            edited_by,
            "round-trip for {edited_by:?}"
        );
    }
    // Unrecognized strings round-trip too — the type is a lens over the
    // string, not a replacement for it.
    for odd in ["manual_edt", "", "Distill", "totally_new_stage"] {
        assert_eq!(
            Writer::classify(odd).as_str(),
            odd,
            "round-trip for {odd:?}"
        );
    }
}

/// The bug this type exists to bound: an unrecognized writer string used to
/// fall through `!matches!(..)` into "machine" with no diagnostic. That
/// direction is preserved deliberately — machine is the fail-safe answer,
/// because a machine write to a human-owned page is staged as a revision
/// card instead of overwriting the human's prose.
#[test]
fn unknown_writer_is_machine_and_guarded() {
    let typo = Writer::classify("manual_edt");
    assert!(
        typo.is_machine(),
        "a typo'd human writer must not be trusted"
    );
    assert!(
        !typo.skips_hallucination_guard(),
        "an unknown writer must not skip the hallucination guard"
    );
    assert!(!typo.is_llm_rewrite());
    assert!(matches!(typo, Writer::Pipeline(PipelineStage::Unknown(_))));
}

#[test]
fn known_writers_are_not_unknown() {
    for &(edited_by, ..) in WRITER_TABLE {
        let is_unknown = matches!(
            Writer::classify(edited_by),
            Writer::Pipeline(PipelineStage::Unknown(_))
        );
        // Only the four db-layer writers are outside the gate's vocabulary.
        let expected_unknown = matches!(
            edited_by,
            "create" | "revision_accept" | "citation_backfill" | "migration_84"
        );
        assert_eq!(
            is_unknown, expected_unknown,
            "unknown-ness of {edited_by:?}"
        );
    }
}
