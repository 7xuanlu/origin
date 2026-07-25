// SPDX-License-Identifier: Apache-2.0

use super::tests::test_db;
use crate::read_scope::ReadScope;
use crate::sources::RawDocument;
use std::collections::HashMap;

fn memory_doc(source_id: &str, space: &str) -> RawDocument {
    RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: "linked graph memory".to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        metadata: HashMap::new(),
        memory_type: Some("fact".to_string()),
        space: Some(space.to_string()),
        source_agent: None,
        confidence: Some(0.9),
        confirmed: Some(true),
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    }
}

#[tokio::test]
async fn search_entities_scopes_before_vector_limit() {
    let (db, _tmp) = test_db().await;
    let work_id = db
        .store_entity("Work entity", "project", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let mut personal_ids = Vec::new();
    for index in 0..8 {
        personal_ids.push(
            db.store_entity(
                &format!("Personal entity {index}"),
                "project",
                Some("personal"),
                None,
                Some(0.9),
            )
            .await
            .unwrap(),
        );
    }

    let query_embedding = db.get_or_compute_embedding("quasar nebula").unwrap();
    let exact = super::MemoryDB::vec_to_sql(&query_embedding);
    let opposite = super::MemoryDB::vec_to_sql(
        &query_embedding
            .iter()
            .map(|value| -*value)
            .collect::<Vec<_>>(),
    );
    let conn = db.conn.lock().await;
    conn.execute(
        "UPDATE entities SET embedding = vector32(?1) WHERE id = ?2",
        libsql::params![opposite, work_id.clone()],
    )
    .await
    .unwrap();
    for id in personal_ids {
        conn.execute(
            "UPDATE entities SET embedding = vector32(?1) WHERE id = ?2",
            libsql::params![exact.clone(), id],
        )
        .await
        .unwrap();
    }
    drop(conn);

    let results = db
        .search_entities_by_vector_scoped("quasar nebula", 1, &ReadScope::Space("work".to_string()))
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].entity.id, work_id);
}

#[tokio::test]
async fn graph_memory_fetch_excludes_cross_scope_rows() {
    let (db, _tmp) = test_db().await;
    let entity_id = db
        .create_entity("Shared topic", "topic", Some("work"))
        .await
        .unwrap();
    db.upsert_documents(vec![
        memory_doc("work-memory", "work"),
        memory_doc("personal-memory", "personal"),
    ])
    .await
    .unwrap();
    for source_id in ["work-memory", "personal-memory"] {
        db.link_memory_entities(source_id, &[entity_id.as_str()])
            .await
            .unwrap();
    }

    let results = db
        .get_memories_for_entities_scoped(
            std::slice::from_ref(&entity_id),
            10,
            &ReadScope::Space("work".to_string()),
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].source_id, "work-memory");
}

#[tokio::test]
async fn selected_khop_keeps_in_scope_neighbor_and_drops_cross_scope_endpoint() {
    let (db, _tmp) = test_db().await;
    let seed = db
        .store_entity("Work seed", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let work_neighbor = db
        .store_entity("Work neighbor", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let personal_neighbor = db
        .store_entity(
            "Personal neighbor",
            "topic",
            Some("personal"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    db.create_relation(&seed, &work_neighbor, "related_to", None, None, None, None)
        .await
        .unwrap();
    db.create_relation(
        &seed,
        &personal_neighbor,
        "related_to",
        None,
        None,
        None,
        None,
    )
    .await
    .unwrap();

    let expanded = db
        .expand_entities_khop_scoped(
            std::slice::from_ref(&seed),
            1,
            64,
            &ReadScope::Space("work".to_string()),
        )
        .await
        .unwrap();

    assert!(expanded.contains(&seed));
    assert!(expanded.contains(&work_neighbor));
    assert!(!expanded.contains(&personal_neighbor));
}

#[tokio::test]
async fn selected_graph_stream_keeps_positive_and_drops_cross_scope_memory() {
    let (db, _tmp) = test_db().await;
    let work_entity = db
        .store_entity("Scoped graph topic", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let personal_entity = db
        .store_entity(
            "Scoped graph topic",
            "topic",
            Some("personal"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    db.upsert_documents(vec![
        memory_doc("work-graph-memory", "work"),
        memory_doc("personal-graph-memory", "personal"),
    ])
    .await
    .unwrap();
    db.link_memory_entities("work-graph-memory", &[work_entity.as_str()])
        .await
        .unwrap();
    db.link_memory_entities("personal-graph-memory", &[personal_entity.as_str()])
        .await
        .unwrap();

    let results = temp_env::async_with_vars(
        [
            ("WENLAN_GRAPH_MEMORY_STREAM", Some("1")),
            ("WENLAN_GRAPH_SURFACE_NEW", Some("1")),
        ],
        async {
            db.augment_with_graph_gated(
                "Scoped graph topic",
                Vec::new(),
                10,
                true,
                &ReadScope::Space("work".to_string()),
            )
            .await
            .unwrap()
        },
    )
    .await;

    assert!(results
        .iter()
        .any(|row| row.source_id == "work-graph-memory"));
    assert!(results
        .iter()
        .all(|row| row.source_id != "personal-graph-memory"));
}

#[tokio::test]
async fn selected_episode_channel_keeps_only_matching_scope() {
    let (db, _tmp) = test_db().await;
    let mut work = memory_doc("work-episode", "work");
    work.source = "episode".to_string();
    let mut personal = memory_doc("personal-episode", "personal");
    personal.source = "episode".to_string();
    db.upsert_documents(vec![work, personal]).await.unwrap();

    let results = db
        .search_episodes_scoped(
            "linked graph memory",
            10,
            &ReadScope::Space("work".to_string()),
        )
        .await
        .unwrap();

    assert!(results.iter().any(|row| row.source_id == "work-episode"));
    assert!(results
        .iter()
        .all(|row| row.source_id != "personal-episode"));
}

#[tokio::test]
async fn selected_fact_channel_rehydrates_only_matching_scope() {
    let (db, _tmp) = test_db().await;
    db.upsert_documents(vec![
        memory_doc("work-fact-parent", "work"),
        memory_doc("personal-fact-parent", "personal"),
    ])
    .await
    .unwrap();
    let embedding = db.get_or_compute_embedding("fact child query").unwrap();
    let vector = super::MemoryDB::vec_to_sql(&embedding);
    let conn = db.conn.lock().await;
    for (id, parent_id) in [
        ("work-child", "work-fact-parent"),
        ("personal-child", "personal-fact-parent"),
    ] {
        conn.execute(
            "INSERT INTO child_vectors (id, parent_kind, parent_id, field, content, embedding) \
             VALUES (?1, 'memory', ?2, 'fact', 'child fact', vector32(?3))",
            libsql::params![id, parent_id, vector.clone()],
        )
        .await
        .unwrap();
    }
    drop(conn);

    let results = db
        .search_facts_channel(
            "fact child query",
            10,
            &ReadScope::Space("work".to_string()),
        )
        .await
        .unwrap();

    assert!(results
        .iter()
        .any(|row| row.source_id == "work-fact-parent"));
    assert!(results
        .iter()
        .all(|row| row.source_id != "personal-fact-parent"));
}

#[tokio::test]
async fn list_entities_scoped_distinguishes_null_from_literal_uncategorized() {
    let (db, _tmp) = test_db().await;
    let work = db
        .store_entity("Scoped list work", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let personal = db
        .store_entity(
            "Scoped list personal",
            "topic",
            Some("personal"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    let null = db
        .store_entity("Scoped list null", "topic", None, None, Some(0.9))
        .await
        .unwrap();
    let literal = db
        .store_entity(
            "Scoped list literal",
            "topic",
            Some("uncategorized"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();

    let work_rows = db
        .list_entities_scoped(None, &ReadScope::Space("work".to_string()))
        .await
        .unwrap();
    assert_eq!(
        work_rows
            .iter()
            .map(|entity| entity.id.as_str())
            .collect::<Vec<_>>(),
        vec![work.as_str()]
    );
    let null_rows = db
        .list_entities_scoped(None, &ReadScope::Uncategorized)
        .await
        .unwrap();
    assert_eq!(
        null_rows
            .iter()
            .map(|entity| entity.id.as_str())
            .collect::<Vec<_>>(),
        vec![null.as_str()]
    );
    let global = db
        .list_entities_scoped(None, &ReadScope::Global)
        .await
        .unwrap();
    let global_ids = global
        .iter()
        .map(|entity| entity.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    for id in [&work, &personal, &null, &literal] {
        assert!(global_ids.contains(id.as_str()));
    }
}

#[tokio::test]
async fn get_entity_detail_scoped_requires_matching_relation_endpoints() {
    let (db, _tmp) = test_db().await;
    let work = db
        .store_entity("Detail work", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let work_peer = db
        .store_entity("Detail work peer", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let personal = db
        .store_entity(
            "Detail personal",
            "topic",
            Some("personal"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    let visible = db
        .create_relation(&work, &work_peer, "related_to", None, None, None, None)
        .await
        .unwrap();
    let hidden = db
        .create_relation(&work, &personal, "related_to", None, None, None, None)
        .await
        .unwrap();

    let detail = db
        .get_entity_detail_scoped(&work, &ReadScope::Space("work".to_string()))
        .await
        .unwrap();
    assert!(detail
        .relations
        .iter()
        .any(|relation| relation.id == visible));
    assert!(detail
        .relations
        .iter()
        .all(|relation| relation.id != hidden));
    assert!(matches!(
        db.get_entity_detail_scoped(&personal, &ReadScope::Space("work".to_string()))
            .await,
        Err(crate::WenlanError::NotFound(message)) if message == "entity not found"
    ));
}

#[tokio::test]
async fn list_recent_relations_scoped_requires_both_endpoints() {
    let (db, _tmp) = test_db().await;
    let work = db
        .store_entity("Relation work", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let work_peer = db
        .store_entity("Relation work peer", "topic", Some("work"), None, Some(0.9))
        .await
        .unwrap();
    let personal = db
        .store_entity(
            "Relation personal",
            "topic",
            Some("personal"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    let visible = db
        .create_relation(&work, &work_peer, "related_to", None, None, None, None)
        .await
        .unwrap();
    let hidden = db
        .create_relation(&work, &personal, "related_to", None, None, None, None)
        .await
        .unwrap();

    let selected = db
        .list_recent_relations_scoped(20, None, &ReadScope::Space("work".to_string()))
        .await
        .unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|relation| relation.id.as_str())
            .collect::<Vec<_>>(),
        vec![visible.as_str()]
    );
    let global = db
        .list_recent_relations_scoped(20, None, &ReadScope::Global)
        .await
        .unwrap();
    assert!(global.iter().any(|relation| relation.id == hidden));
}

#[tokio::test]
async fn list_entity_suggestions_scoped_excludes_invalid_and_mixed_owner_sets() {
    let (db, _tmp) = test_db().await;
    db.upsert_documents(vec![
        memory_doc("suggest-work", "work"),
        memory_doc("suggest-personal", "personal"),
    ])
    .await
    .unwrap();
    for (id, sources) in [
        ("suggest-work-only", vec!["suggest-work".to_string()]),
        (
            "suggest-mixed",
            vec!["suggest-work".to_string(), "suggest-personal".to_string()],
        ),
        ("suggest-missing", vec!["suggest-absent".to_string()]),
        ("suggest-empty", Vec::new()),
    ] {
        db.insert_refinement_proposal(id, "suggest_entity", &sources, Some(id), 0.9)
            .await
            .unwrap();
    }
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO refinement_queue (id, action, source_ids, payload, confidence) \
         VALUES ('suggest-malformed', 'suggest_entity', 'not-json', 'malformed', 0.9)",
        (),
    )
    .await
    .unwrap();
    drop(conn);

    let selected = db
        .list_entity_suggestions_scoped(&ReadScope::Space("work".to_string()))
        .await
        .unwrap();
    assert_eq!(
        selected
            .iter()
            .map(|proposal| proposal.id.as_str())
            .collect::<Vec<_>>(),
        vec!["suggest-work-only"]
    );

    let global = db
        .list_entity_suggestions_scoped(&ReadScope::Global)
        .await
        .unwrap();
    let global_ids = global
        .iter()
        .map(|proposal| proposal.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    for id in [
        "suggest-work-only",
        "suggest-mixed",
        "suggest-missing",
        "suggest-empty",
        "suggest-malformed",
    ] {
        assert!(global_ids.contains(id));
    }
}

// ===== M3 PR-2 stage c: `scoped_entities` vanguard flip =====
//
// Differential-oracle coverage for the six scoped fns' per-call gate: seed
// a DB, run a fn with the "scoped_entities" consumer OFF (legacy
// `entities`), flip it ON under a clean, current parity watermark
// (`reconcile_entity_page_parity`), run again, and assert the two outputs
// are byte-identical. None of `Entity`/`EntityDetail`/`RelationWithEntity`/
// `RecentRelation`/`RefinementProposal`/`SearchResult` derive `PartialEq`
// (they are `wenlan-types` wire types stage c must not touch), so `Debug`
// string equality stands in for full struct equality -- it is exact and
// order-sensitive the same way `assert_eq!` would be. One shared
// gate-closure test (not per-fn) proves a shadow corrupted after the proof
// re-reconciles dirty and the reader transparently falls back to legacy,
// mirroring stage b's `entity_reader_gate_blocked_by_nonzero_drift`.

/// Seed entities covering the edge shapes the hybrid read must reproduce:
/// a named space + NULL space, confirmed + unconfirmed, and an entity with
/// an added (non-self) alias. Returns (work_id, work_peer_id,
/// uncategorized_id); `work` is confirmed and aliased, `work_peer` and
/// `uncategorized` are not.
async fn stage_c_seed_entities(db: &super::MemoryDB) -> (String, String, String) {
    let work = db
        .store_entity(
            "Stage C Work",
            "person",
            Some("stage_c_work"),
            None,
            Some(0.8),
        )
        .await
        .unwrap();
    db.confirm_entity(&work, true).await.unwrap();
    db.add_entity_alias("stage c nickname", &work, "test")
        .await
        .unwrap();
    let work_peer = db
        .store_entity(
            "Stage C Work Peer",
            "project",
            Some("stage_c_work"),
            None,
            None,
        )
        .await
        .unwrap();
    let uncategorized = db
        .store_entity("Stage C Unfiled", "person", None, None, Some(0.4))
        .await
        .unwrap();
    (work, work_peer, uncategorized)
}

/// Flip the "scoped_entities" consumer on and prove the gate actually
/// opened (a clean, current parity watermark), so a bug that silently
/// leaves every fn on legacy can't pass a differential test vacuously.
async fn stage_c_enable_cutover_clean(db: &super::MemoryDB) {
    db.set_entity_reader_cutover(super::MemoryDB::SCOPED_ENTITIES_CONSUMER, true)
        .await
        .unwrap();
    assert_eq!(
        db.reconcile_entity_page_parity().await.unwrap().drift_count,
        0,
        "seed must reconcile clean before the gate can open"
    );
    let conn = db.conn.lock().await;
    assert!(
        super::MemoryDB::reader_uses_entity_pages(&conn, super::MemoryDB::SCOPED_ENTITIES_CONSUMER)
            .await
            .unwrap(),
        "sanity: a clean, current watermark must open the gate"
    );
}

#[tokio::test]
async fn list_entities_scoped_hybrid_matches_legacy() {
    let (db, _tmp) = test_db().await;
    stage_c_seed_entities(&db).await;
    let work_scope = ReadScope::Space("stage_c_work".to_string());

    let legacy_scoped = db.list_entities_scoped(None, &work_scope).await.unwrap();
    let legacy_typed = db
        .list_entities_scoped(Some("person"), &work_scope)
        .await
        .unwrap();
    let legacy_uncategorized = db
        .list_entities_scoped(None, &ReadScope::Uncategorized)
        .await
        .unwrap();

    stage_c_enable_cutover_clean(&db).await;

    let hybrid_scoped = db.list_entities_scoped(None, &work_scope).await.unwrap();
    let hybrid_typed = db
        .list_entities_scoped(Some("person"), &work_scope)
        .await
        .unwrap();
    let hybrid_uncategorized = db
        .list_entities_scoped(None, &ReadScope::Uncategorized)
        .await
        .unwrap();

    assert_eq!(legacy_scoped.len(), 2, "sanity: two work entities seeded");
    assert_eq!(
        format!("{legacy_scoped:?}"),
        format!("{hybrid_scoped:?}"),
        "hybrid list_entities_scoped must be byte-identical to legacy"
    );
    assert_eq!(
        format!("{legacy_typed:?}"),
        format!("{hybrid_typed:?}"),
        "hybrid list_entities_scoped (entity_type filter) must be byte-identical to legacy"
    );
    assert_eq!(
        format!("{legacy_uncategorized:?}"),
        format!("{hybrid_uncategorized:?}"),
        "hybrid list_entities_scoped (Uncategorized) must be byte-identical to legacy"
    );
}

#[tokio::test]
async fn get_entity_detail_scoped_hybrid_matches_legacy() {
    let (db, _tmp) = test_db().await;
    let (work, work_peer, _uncategorized) = stage_c_seed_entities(&db).await;
    let personal = db
        .store_entity(
            "Stage C Personal",
            "person",
            Some("stage_c_personal"),
            None,
            None,
        )
        .await
        .unwrap();
    db.create_relation(&work, &work_peer, "related_to", None, None, None, None)
        .await
        .unwrap();
    db.create_relation(&work, &personal, "related_to", None, None, None, None)
        .await
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO observations (id, entity_id, content, source_agent, confidence, confirmed, created_at) \
             VALUES ('stage_c_obs_1', ?1, 'confirmed observation', NULL, 0.9, 1, unixepoch())",
            libsql::params![work.clone()],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO observations (id, entity_id, content, source_agent, confidence, confirmed, created_at) \
             VALUES ('stage_c_obs_2', ?1, 'unconfirmed observation', NULL, NULL, 0, unixepoch())",
            libsql::params![work.clone()],
        )
        .await
        .unwrap();
    }

    let scope = ReadScope::Space("stage_c_work".to_string());
    let legacy = db.get_entity_detail_scoped(&work, &scope).await.unwrap();

    stage_c_enable_cutover_clean(&db).await;

    let hybrid = db.get_entity_detail_scoped(&work, &scope).await.unwrap();

    assert_eq!(
        legacy.observations.len(),
        2,
        "sanity: two observations seeded"
    );
    assert_eq!(
        legacy.relations.len(),
        1,
        "sanity: only the in-scope relation is visible"
    );
    assert_eq!(
        format!("{legacy:?}"),
        format!("{hybrid:?}"),
        "hybrid get_entity_detail_scoped must be byte-identical to legacy"
    );
}

#[tokio::test]
async fn list_recent_relations_scoped_hybrid_matches_legacy() {
    let (db, _tmp) = test_db().await;
    let (work, work_peer, _uncategorized) = stage_c_seed_entities(&db).await;
    let personal = db
        .store_entity(
            "Stage C Personal Relation",
            "person",
            Some("stage_c_personal"),
            None,
            None,
        )
        .await
        .unwrap();
    db.create_relation(&work, &work_peer, "related_to", None, None, None, None)
        .await
        .unwrap();
    db.create_relation(&work, &personal, "related_to", None, None, None, None)
        .await
        .unwrap();

    let scope = ReadScope::Space("stage_c_work".to_string());
    let legacy = db
        .list_recent_relations_scoped(20, None, &scope)
        .await
        .unwrap();

    stage_c_enable_cutover_clean(&db).await;

    let hybrid = db
        .list_recent_relations_scoped(20, None, &scope)
        .await
        .unwrap();

    assert_eq!(
        legacy.len(),
        1,
        "sanity: only the in-scope relation is visible"
    );
    assert_eq!(
        format!("{legacy:?}"),
        format!("{hybrid:?}"),
        "hybrid list_recent_relations_scoped must be byte-identical to legacy"
    );
}

/// `list_entity_suggestions_scoped` sources no entity/page-mirrored field
/// (a suggestion describes an entity that does not exist yet, so its query
/// touches only `refinement_queue`+`memories`) -- the per-call gate is
/// still consulted for audit uniformity, but there is no hybrid branch to
/// diverge into. This test proves the consult is genuinely a no-op:
/// legacy SQL, byte-identical output, before and after the flip.
#[tokio::test]
async fn list_entity_suggestions_scoped_hybrid_matches_legacy() {
    let (db, _tmp) = test_db().await;
    db.upsert_documents(vec![memory_doc(
        "stage-c-suggest-work",
        "stage_c_suggest_work",
    )])
    .await
    .unwrap();
    let sources = vec!["stage-c-suggest-work".to_string()];
    db.insert_refinement_proposal(
        "stage-c-suggest",
        "suggest_entity",
        &sources,
        Some("stage-c-suggest"),
        0.9,
    )
    .await
    .unwrap();

    let scope = ReadScope::Space("stage_c_suggest_work".to_string());
    let legacy = db.list_entity_suggestions_scoped(&scope).await.unwrap();

    stage_c_enable_cutover_clean(&db).await;

    let hybrid = db.list_entity_suggestions_scoped(&scope).await.unwrap();

    assert_eq!(legacy.len(), 1, "sanity: the proposal is visible");
    assert_eq!(
        format!("{legacy:?}"),
        format!("{hybrid:?}"),
        "list_entity_suggestions_scoped sources no entity/page field -- \
         the gate must not change it"
    );
}

#[tokio::test]
async fn search_entities_by_vector_scoped_hybrid_matches_legacy() {
    let (db, _tmp) = test_db().await;
    let work_id = db
        .store_entity(
            "Stage C Vector Work",
            "project",
            Some("stage_c_vec_work"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    db.confirm_entity(&work_id, true).await.unwrap();
    db.store_entity(
        "Stage C Vector Personal",
        "project",
        Some("stage_c_vec_personal"),
        None,
        Some(0.9),
    )
    .await
    .unwrap();

    let scope = ReadScope::Space("stage_c_vec_work".to_string());
    let legacy = db
        .search_entities_by_vector_scoped("stage c vector query", 5, &scope)
        .await
        .unwrap();

    stage_c_enable_cutover_clean(&db).await;

    let hybrid = db
        .search_entities_by_vector_scoped("stage c vector query", 5, &scope)
        .await
        .unwrap();

    assert_eq!(
        legacy.len(),
        1,
        "sanity: only the in-scope entity is visible"
    );
    assert_eq!(legacy[0].entity.id, work_id);
    assert_eq!(
        format!("{legacy:?}"),
        format!("{hybrid:?}"),
        "hybrid search_entities_by_vector_scoped must be byte-identical to legacy"
    );
}

/// `get_memories_for_entities_scoped` returns MEMORIES linked to the given
/// entity ids (`memories`/`memory_entities` only) -- like
/// `list_entity_suggestions_scoped`, it sources no entity/page-mirrored
/// field, so the gate consult is a no-op here too.
#[tokio::test]
async fn get_memories_for_entities_scoped_hybrid_matches_legacy() {
    let (db, _tmp) = test_db().await;
    let entity_id = db
        .store_entity(
            "Stage C Memory Topic",
            "topic",
            Some("stage_c_mem_work"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    db.upsert_documents(vec![memory_doc("stage-c-mem", "stage_c_mem_work")])
        .await
        .unwrap();
    db.link_memory_entities("stage-c-mem", &[entity_id.as_str()])
        .await
        .unwrap();

    let scope = ReadScope::Space("stage_c_mem_work".to_string());
    let legacy = db
        .get_memories_for_entities_scoped(std::slice::from_ref(&entity_id), 10, &scope)
        .await
        .unwrap();

    stage_c_enable_cutover_clean(&db).await;

    let hybrid = db
        .get_memories_for_entities_scoped(std::slice::from_ref(&entity_id), 10, &scope)
        .await
        .unwrap();

    assert_eq!(legacy.len(), 1, "sanity: the linked memory is visible");
    assert_eq!(
        format!("{legacy:?}"),
        format!("{hybrid:?}"),
        "get_memories_for_entities_scoped sources no entity/page field -- \
         the gate must not change it"
    );
}

/// THE gate-closure test (one, not per-fn, per the stage-c contract):
/// corrupting a shadow and re-reconciling makes the watermark dirty, so
/// the reader must transparently fall back to legacy even though the
/// consumer is still enabled. Mirrors stage b's
/// `entity_reader_gate_blocked_by_nonzero_drift`; `list_entities_scoped`
/// stands in as the representative fn.
#[tokio::test]
async fn scoped_entities_gate_closes_on_drift_serves_legacy_transparently() {
    let (db, _tmp) = test_db().await;
    let entity_id = db
        .store_entity(
            "Stage C Drift",
            "person",
            Some("stage_c_drift"),
            None,
            Some(0.9),
        )
        .await
        .unwrap();
    stage_c_enable_cutover_clean(&db).await;

    let scope = ReadScope::Space("stage_c_drift".to_string());
    let clean_hybrid = db.list_entities_scoped(None, &scope).await.unwrap();
    assert_eq!(clean_hybrid.len(), 1);
    assert_eq!(clean_hybrid[0].entity_type, "person");

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET entity_type = 'organization' \
             WHERE id = (SELECT page_id FROM entity_page_map WHERE entity_id = ?1)",
            libsql::params![entity_id.clone()],
        )
        .await
        .unwrap();
    }
    let report = db.reconcile_entity_page_parity().await.unwrap();
    assert!(
        report.drift_count >= 1,
        "sanity: reconcile must see the drift"
    );

    let gated = db.list_entities_scoped(None, &scope).await.unwrap();
    assert_eq!(
        gated.len(),
        1,
        "the entity must still be visible via the legacy fallback"
    );
    assert_eq!(
        gated[0].entity_type, "person",
        "a dirty watermark must transparently serve the uncorrupted legacy \
         value, not the drifted shadow"
    );
}

async fn stage_c_query_plan_detail(conn: &libsql::Connection, sql: &str) -> String {
    let mut rows = conn
        .query(&format!("EXPLAIN QUERY PLAN {sql}"), ())
        .await
        .unwrap();
    let mut details = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        let detail: String = row.get(3).unwrap_or_default();
        details.push(detail);
    }
    details.join(" | ")
}

/// `list_entities_scoped`'s flipped query must reach the shadow page
/// through the `entity_page_map` UNIQUE index / pages PK, never a bare
/// `SCAN pages`.
#[tokio::test]
async fn list_entities_scoped_flipped_query_uses_index_not_pages_scan() {
    let (db, _tmp) = test_db().await;
    stage_c_seed_entities(&db).await;
    stage_c_enable_cutover_clean(&db).await;

    let conn = db.conn.lock().await;
    // Mirrors exactly the flipped SELECT `list_entities_scoped` builds
    // once `reader_uses_entity_pages` is true (Space scope, no
    // entity_type filter).
    let plan = stage_c_query_plan_detail(
        &conn,
        "SELECT e.id, p.title, p.entity_type, e.space, e.source_agent, p.confidence, \
                p.entity_confirmed, e.created_at, e.updated_at \
         FROM entities e \
         JOIN entity_page_map m ON m.entity_id = e.id \
         JOIN pages p ON p.id = m.page_id AND p.kind = 'entity' AND p.status = 'active' \
         WHERE e.space = 'stage_c_work' ORDER BY e.updated_at DESC, e.id ASC",
    )
    .await;

    for line in plan.split(" | ") {
        let upper = line.to_uppercase();
        if upper.contains(" P ") || upper.contains("PAGES") {
            assert!(
                upper.contains("USING"),
                "pages access must use the entity_page_map UNIQUE index or \
                 the pages PK, not a bare scan: {line} (full plan: {plan})"
            );
        }
    }
    assert!(
        plan.to_uppercase().contains("USING"),
        "the flipped query must use at least one index: {plan}"
    );
}
