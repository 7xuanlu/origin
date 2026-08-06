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
    db.create_relation(&work, &work_peer, "related_to", None, None, None, None)
        .await
        .unwrap();
    db.create_relation(&work, &personal, "related_to", None, None, None, None)
        .await
        .unwrap();
    // G6 Stage 1.2 Trap 2: the reader's relation `id` is now the active
    // edge's edge_id, not create_relation's relations-row uuid return value.
    let visible = crate::provenance::compute_edge_id(
        "relates",
        "entity",
        &work,
        "entity",
        &work_peer,
        "related_to",
    );
    let hidden = crate::provenance::compute_edge_id(
        "relates",
        "entity",
        &work,
        "entity",
        &personal,
        "related_to",
    );

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
    db.create_relation(&work, &work_peer, "related_to", None, None, None, None)
        .await
        .unwrap();
    db.create_relation(&work, &personal, "related_to", None, None, None, None)
        .await
        .unwrap();
    // G6 Stage 1.2 Trap 2: the reader's relation `id` is now the active
    // edge's edge_id, not create_relation's relations-row uuid return value.
    let visible = crate::provenance::compute_edge_id(
        "relates",
        "entity",
        &work,
        "entity",
        &work_peer,
        "related_to",
    );
    let hidden = crate::provenance::compute_edge_id(
        "relates",
        "entity",
        &work,
        "entity",
        &personal,
        "related_to",
    );

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
// re-reconciles dirty and the reader transparently falls back to legacy --
// stage b's mirror-image coverage of the same cutover-lever predicate,
// `entity_reader_gate_blocked_by_nonzero_drift`, retired with the gate
// itself in G6 Stage 2 PR 2a (see the retirement note at
// `main_tests.rs:44591`).

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

// G6 Stage 1.5b Part 3: the `stage_c_enable_cutover_clean` helper that used
// to flip the "scoped_entities" consumer before a differential read (proving
// the gate actually opened) has no callers left -- every gated hybrid in
// this file collapsed onto an unconditional hard cutover (spec item 9), so
// there is no more gate to flip in a test. Removed rather than left dead:
// this was a test-local helper, so it carried no Stage 2 retirement
// obligation of its own -- the production `reader_uses_entity_pages`/
// `SCOPED_ENTITIES_CONSUMER` surface it used to flip has since been
// deleted outright in G6 Stage 2 PR 2a, completing that retirement.

// G6 Stage 1.5b Part 3: `list_entities_scoped`/`get_entity_detail_scoped`
// collapsed onto an unconditional hard cutover (spec item 9) -- there is no
// more legacy/hybrid branch for these two fns to differ across, so the
// former `*_hybrid_matches_legacy` A/B tests (which called each fn once
// before and once after flipping the now-deleted gate check) retired as
// vacuous: both calls run the identical code path. Coverage that these
// readers produce correct results lives on in
// `list_entities_scoped_distinguishes_null_from_literal_uncategorized` and
// `get_entity_detail_scoped_requires_matching_relation_endpoints`; coverage
// that they read the shadow page rather than legacy `entities` lives on in
// `list_entities_scoped_reads_shadow_page_directly` and
// `get_entity_detail_scoped_reads_shadow_page_directly` below.

// G6 Stage 1.5b Part 3: `list_recent_relations_scoped` collapsed onto an
// unconditional hard cutover (spec item 9) and its join was structurally
// rebuilt on shadow pages (spec item 7), so there is no more legacy/hybrid
// branch for it to differ across -- the former `*_hybrid_matches_legacy` A/B
// test (which called the fn once before and once after flipping the
// now-deleted gate check) retired as vacuous: both calls run the identical
// code path. Coverage that this reader produces correct results lives on in
// `get_entity_detail_scoped_and_list_recent_relations_scoped_read_edges`
// below; coverage that it reads the shadow page rather than legacy
// `entities` lives on in `list_recent_relations_scoped_reads_shadow_page_directly`.

/// G6 Stage 1.2 (relations-readers migration,
/// docs/plans/2026-08-05-g6-stage12-relations-readers-spec.md): the scoped
/// variants must read the same `relates` edge fields as their unscoped
/// counterparts -- edge_id as `id`, entity names via the join -- and the
/// scope filter must still exclude a relation from a different space.
#[tokio::test]
async fn get_entity_detail_scoped_and_list_recent_relations_scoped_read_edges() {
    let (db, _tmp) = test_db().await;
    let scope = ReadScope::Space("g6_scoped_space_a".to_string());
    let alice = db
        .create_entity("G6 Scoped Alice", "person", Some("g6_scoped_space_a"))
        .await
        .unwrap();
    let wenlan = db
        .create_entity("G6 Scoped Wenlan", "project", Some("g6_scoped_space_a"))
        .await
        .unwrap();
    db.create_relation(
        &alice,
        &wenlan,
        "works_on",
        Some("claude"),
        Some(0.9),
        Some("seen"),
        None,
    )
    .await
    .unwrap();

    let expected_edge_id = crate::provenance::compute_edge_id(
        "relates", "entity", &alice, "entity", &wenlan, "works_on",
    );

    let detail = db.get_entity_detail_scoped(&alice, &scope).await.unwrap();
    assert_eq!(detail.relations.len(), 1);
    assert_eq!(
        detail.relations[0].id, expected_edge_id,
        "scoped detail relation id must be the active edge's edge_id"
    );
    assert_eq!(detail.relations[0].entity_name, "G6 Scoped Wenlan");

    let recent = db
        .list_recent_relations_scoped(10, None, &scope)
        .await
        .unwrap();
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].id, expected_edge_id);
    assert_eq!(recent[0].from_entity_name, "G6 Scoped Alice");
    assert_eq!(recent[0].to_entity_name, "G6 Scoped Wenlan");

    // A relation in a different space must not leak into this scope.
    let outside_a = db
        .create_entity("G6 Scoped Outside A", "person", Some("g6_scoped_space_b"))
        .await
        .unwrap();
    let outside_b = db
        .create_entity("G6 Scoped Outside B", "project", Some("g6_scoped_space_b"))
        .await
        .unwrap();
    db.create_relation(&outside_a, &outside_b, "knows", None, None, None, None)
        .await
        .unwrap();
    let recent_after = db
        .list_recent_relations_scoped(10, None, &scope)
        .await
        .unwrap();
    assert_eq!(
        recent_after.len(),
        1,
        "a relation in a different space must not leak into this scope"
    );
}

// G6 Stage 1.5b Part 3: `search_entities_by_vector_scoped`'s hydration
// overlay collapsed onto an unconditional run (spec item 9) -- there is no
// more legacy/hybrid branch for it to differ across, so the former
// `*_hybrid_matches_legacy` A/B test (which called the fn once before and
// once after flipping the now-deleted gate check) retired as vacuous: both
// calls run the identical code path. Coverage that the overlay reads the
// shadow page rather than legacy `entities` lives on in
// `search_entities_by_vector_scoped_reads_shadow_page_directly` above.

// G6 Stage 1.5b Part 3: the former "THE gate-closure test (one, not per-fn)"
// used `list_entities_scoped` as its representative fn -- that fn collapsed
// onto an unconditional hard cutover (spec item 9) and no longer falls back
// to legacy on drift, so it can no longer stand in for the gate-closure
// contract. The underlying `reader_uses_entity_pages` predicate and its
// `entity_reader_gate_blocked_by_nonzero_drift` coverage both retired
// outright in G6 Stage 2 PR 2a (see the retirement note at
// `main_tests.rs:44591`) -- there is no gate left for any
// scoped_entities.rs consumer to fall back through.

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

/// `list_entities_scoped`'s query must reach the shadow page through the
/// `entity_page_map` UNIQUE index / pages PK, never a bare `SCAN pages`.
#[tokio::test]
async fn list_entities_scoped_flipped_query_uses_index_not_pages_scan() {
    let (db, _tmp) = test_db().await;
    stage_c_seed_entities(&db).await;

    let conn = db.conn.lock().await;
    // Mirrors exactly the SELECT `list_entities_scoped` builds (G6 Stage
    // 1.5b Part 3: unconditional hard cutover, Space scope, no
    // entity_type filter).
    let plan = stage_c_query_plan_detail(
        &conn,
        "SELECT epm.entity_id, p.title, p.entity_type, p.space, p.source_agent, p.confidence, \
                p.entity_confirmed, p.entity_created_at, p.entity_updated_at \
         FROM entity_page_map epm \
         JOIN pages p ON p.id = epm.page_id \
         WHERE p.kind = 'entity' AND p.status = 'active' AND p.space = 'stage_c_work' \
         ORDER BY p.entity_updated_at DESC, epm.entity_id ASC",
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

// ===== M3 PR-2 stage f: tie-safe flipped reads (Sol review fix 3+4) =====
//
// G6 Stage 1.5b Part 3: both `list_recent_relations_scoped` and
// `search_entities_by_vector_scoped` collapsed onto an unconditional hard
// cutover (spec item 9), so the `*_tie_heavy_selection_matches_legacy` A/B
// tests that used to live here (proving the tied LIMIT boundary lands the
// same row set/order OFF vs ON) retired as vacuous for the same reason as
// the `*_hybrid_matches_legacy` tests above: there is only one code path
// left to run, twice, against itself. Neither fn's `ORDER BY` clause
// changed in this migration (still an untiebroken `DESC LIMIT`), so no
// stability coverage was lost -- SQLite's own tie-break determinism for a
// fixed query plan is not a G6 concern.

/// Positive control (G6 Stage 1.5b Part 3): mutating a shadow page's title
/// directly (without touching `entities`) must be visible through
/// `list_entities_scoped`, proving the unconditional hard cutover reads the
/// mirror -- not legacy `entities.name` -- since there is no flip left to
/// stamp a watermark under.
#[tokio::test]
async fn list_entities_scoped_reads_shadow_page_directly() {
    let (db, _tmp) = test_db().await;
    let entity_id = db
        .store_entity(
            "Stage F List Live",
            "person",
            Some("stage_f_list_live"),
            None,
            Some(0.5),
        )
        .await
        .unwrap();
    let scope = ReadScope::Space("stage_f_list_live".to_string());

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET title = 'Mutated List Title' \
             WHERE id = (SELECT page_id FROM entity_page_map WHERE entity_id = ?1)",
            libsql::params![entity_id.clone()],
        )
        .await
        .unwrap();
    }

    let entities = db.list_entities_scoped(None, &scope).await.unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(
        entities[0].name, "Mutated List Title",
        "list_entities_scoped must read the shadow page's title, not legacy entities.name"
    );
}

/// Positive control for the other migrated fn: same pattern as above, for
/// `get_entity_detail_scoped`'s primary entity row.
#[tokio::test]
async fn get_entity_detail_scoped_reads_shadow_page_directly() {
    let (db, _tmp) = test_db().await;
    let entity_id = db
        .store_entity(
            "Stage F Detail Live",
            "person",
            Some("stage_f_detail_live"),
            None,
            Some(0.5),
        )
        .await
        .unwrap();
    let scope = ReadScope::Space("stage_f_detail_live".to_string());

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET title = 'Mutated Detail Title' \
             WHERE id = (SELECT page_id FROM entity_page_map WHERE entity_id = ?1)",
            libsql::params![entity_id.clone()],
        )
        .await
        .unwrap();
    }

    let detail = db
        .get_entity_detail_scoped(&entity_id, &scope)
        .await
        .unwrap();
    assert_eq!(
        detail.entity.name, "Mutated Detail Title",
        "get_entity_detail_scoped must read the shadow page's title, not legacy entities.name"
    );
}

/// Positive control (G6 Stage 1.5b Part 3): mutating a shadow page's title
/// directly (without touching `entities`) must be visible through
/// `list_recent_relations_scoped`, proving the rebuilt join (spec item 7)
/// reads the mirror -- not legacy `entities.name` -- since there is no flip
/// left to stamp a watermark under. The former second half of this test
/// (re-reconcile, then assert a dirty watermark falls back to the legacy
/// name) no longer holds: the join has no legacy branch left to fall back
/// to, so that assertion would now be simply wrong, not merely untested.
#[tokio::test]
async fn list_recent_relations_scoped_reads_shadow_page_directly() {
    let (db, _tmp) = test_db().await;
    let from = db
        .store_entity(
            "Stage F Relation From",
            "person",
            Some("stage_f_relation_live"),
            None,
            Some(0.5),
        )
        .await
        .unwrap();
    let to = db
        .store_entity(
            "Stage F Relation To",
            "person",
            Some("stage_f_relation_live"),
            None,
            Some(0.5),
        )
        .await
        .unwrap();
    db.create_relation(&from, &to, "related_to", None, None, None, None)
        .await
        .unwrap();
    let scope = ReadScope::Space("stage_f_relation_live".to_string());

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET title = 'Mutated From Title' \
             WHERE id = (SELECT page_id FROM entity_page_map WHERE entity_id = ?1)",
            libsql::params![from.clone()],
        )
        .await
        .unwrap();
    }

    let live = db
        .list_recent_relations_scoped(20, None, &scope)
        .await
        .unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!(
        live[0].from_entity_name, "Mutated From Title",
        "list_recent_relations_scoped must read the shadow page's title, not legacy entities.name"
    );
    assert_eq!(live[0].to_entity_name, "Stage F Relation To");
}

/// Positive control: proves the hydration overlay reads live shadow state
/// for every mirrored field (name/entity_type/confidence/confirmed)
/// unconditionally (G6 Stage 1.5b Part 3 collapsed the gate this overlay
/// used to sit behind -- spec item 9).
#[tokio::test]
async fn search_entities_by_vector_scoped_reads_shadow_page_directly() {
    let (db, _tmp) = test_db().await;
    let entity_id = db
        .store_entity(
            "Stage F Vector Live",
            "project",
            Some("stage_f_vector_live"),
            None,
            Some(0.5),
        )
        .await
        .unwrap();
    let scope = ReadScope::Space("stage_f_vector_live".to_string());

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET title = 'Mutated Vector Title', entity_type = 'organization', \
                    confidence = 0.75, entity_confirmed = 1 \
             WHERE id = (SELECT page_id FROM entity_page_map WHERE entity_id = ?1)",
            libsql::params![entity_id.clone()],
        )
        .await
        .unwrap();
    }

    let results = db
        .search_entities_by_vector_scoped("stage f vector live query", 5, &scope)
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].entity.name, "Mutated Vector Title");
    assert_eq!(results[0].entity.entity_type, "organization");
    assert_eq!(results[0].entity.confidence, Some(0.75));
    assert!(results[0].entity.confirmed);
}
