// SPDX-License-Identifier: Apache-2.0
//
// G6 Stage 1.5a: RED-control tests for the CLEAN entity readers migrated
// onto `kind='entity'` shadow pages (see
// docs/plans/2026-08-05-g6-stage15a-entity-clean-readers-spec.md, "Tests"
// section, items 1-3). Item 4 (parity watermark unchanged, still drift 0) is
// covered by the pre-existing, untouched
// `db::tests::reconcile_clean_dual_write_reports_zero_drift_and_stamps_watermark`
// fixture -- no new test needed there.

use super::MemoryDB;
use crate::read_scope::ReadScope;

async fn insert_edge(db: &MemoryDB, src: &str, dst: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO edges (edge_id,src_id,src_kind,dst_id,dst_kind,edge_type,lineage,grounded,space,created_at,semantic_type)
         VALUES (?1,?2,'entity',?3,'entity','relates','legacy',0,'00000000-0000-4000-8000-000000000001',1,'related')",
        libsql::params![format!("edge-{src}-{dst}"), src, dst],
    )
    .await
    .unwrap();
}

/// Test category 1 (spec "Tests" item 1): equivalence per migrated
/// product-adjacent reader -- khop, name_type, counts -- seeded via the real
/// writer (`store_entity`), field-by-field against what was written.
#[tokio::test]
async fn migrated_readers_return_fields_that_match_what_store_entity_wrote() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    let anchor = db
        .store_entity("Anchor Corp", "organization", None, None, Some(0.62))
        .await
        .unwrap();
    let neighbor = db
        .store_entity("Neighbor LLC", "organization", None, None, Some(0.81))
        .await
        .unwrap();
    insert_edge(&db, &anchor, &neighbor).await;

    // count_entities: plain COUNT via the shadow page, must equal the
    // number of real writes.
    assert_eq!(db.count_entities().await.unwrap(), 2);

    // get_entity_name_type: name/type hydration must equal the exact
    // strings passed to store_entity, not a derived or truncated form.
    let (anchor_name, anchor_type) = db.get_entity_name_type(&anchor).await.unwrap().unwrap();
    assert_eq!(anchor_name, "Anchor Corp");
    assert_eq!(anchor_type, "organization");
    let (neighbor_name, neighbor_type) = db.get_entity_name_type(&neighbor).await.unwrap().unwrap();
    assert_eq!(neighbor_name, "Neighbor LLC");
    assert_eq!(neighbor_type, "organization");

    // khop: one-hop expansion from the anchor must reach the neighbor via
    // the shadow-page-joined edge traversal.
    let expanded = db
        .expand_entities_khop_scoped(std::slice::from_ref(&anchor), 1, 50, &ReadScope::Global)
        .await
        .unwrap();
    assert!(
        expanded.contains(&neighbor),
        "khop expansion from {anchor} must reach {neighbor} via the relates edge, got {expanded:?}"
    );
}

/// Test category 2 (spec "Tests" item 2), respecified at G6 Stage 3.
///
/// The original test seeded the divergence directly: an entity present only
/// as a shadow page (its `entities` row raw-deleted) had to stay visible,
/// and two entities present only as raw `entities` rows had to stay
/// invisible. Migration 123 dropped `entities`, so the raw-only half is no
/// longer a state the database can reach -- every entity is shadow-only now,
/// and seeding the legacy half would test a substrate production does not
/// have.
///
/// What survives is the reader contract that half was written to pin:
/// resolution goes through `entity_page_map` alone, so an entity id with no
/// map row resolves to `None` and counts for nothing, no matter that it is a
/// well-formed id.
///
/// RED-control (recorded at Stage 1.5a, still the control for the positive
/// assertion): reverting `count_entities`'s SQL to the legacy
/// `SELECT COUNT(*) FROM entities` made this fail. Post-123 that revert no
/// longer compiles against a real database -- the drop itself is now the
/// control.
#[tokio::test]
async fn migrated_readers_resolve_only_through_entity_page_map() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    let shadow_only = db
        .store_entity("Shadow Only", "concept", None, None, Some(0.5))
        .await
        .unwrap();

    assert_eq!(
        db.count_entities().await.unwrap(),
        1,
        "count_entities must count the one mapped shadow page and nothing else"
    );
    assert_eq!(
        db.get_entity_name_type(&shadow_only).await.unwrap(),
        Some(("Shadow Only".to_string(), "concept".to_string())),
        "get_entity_name_type must resolve the shadow-only entity via entity_page_map"
    );
    for unmapped in ["raw-only-1", "raw-only-2"] {
        assert_eq!(
            db.get_entity_name_type(unmapped).await.unwrap(),
            None,
            "get_entity_name_type must return None for an id with no entity_page_map row"
        );
    }
}

/// Test category 3 (spec "Tests" item 3): the mirror invariant itself --
/// `pages.title`/`entity_type`/`confidence` must track updates made through
/// the writer-coupled mutators (`store_entity` at creation,
/// `merge_entities`'s type-promotion at update), not just the initial
/// insert. This is the invariant every reader migrated in this stage now
/// load-bears.
#[tokio::test]
async fn shadow_page_title_type_and_confidence_track_store_entity_updates() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    // "entity" is merge_entities's generic-type marker (its `is_generic`
    // predicate matches only "entity"/"unknown"/""), so this canonical is
    // eligible for the alias's concrete type to be promoted onto it.
    let canonical = db
        .store_entity("Origin Corp", "entity", None, None, Some(0.4))
        .await
        .unwrap();
    let alias = db
        .store_entity("Origin Corp Inc", "database", None, None, Some(0.9))
        .await
        .unwrap();

    async fn read_shadow(db: &MemoryDB, entity_id: &str) -> (String, String, Option<f64>) {
        let conn = db.conn.lock().await;
        let mut rows = conn
            .query(
                "SELECT p.title, p.entity_type, p.confidence FROM entity_page_map epm
                 JOIN pages p ON p.id = epm.page_id
                 WHERE epm.entity_id = ?1 AND p.kind = 'entity' AND p.status = 'active'",
                libsql::params![entity_id.to_string()],
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let title: String = row.get(0).unwrap();
        let entity_type: String = row.get(1).unwrap();
        let confidence: Option<f64> = row.get(2).ok();
        (title, entity_type, confidence)
    }

    // Baseline: the initial insert already mirrors title/type/confidence.
    let (title, entity_type, confidence) = read_shadow(&db, &canonical).await;
    assert_eq!(title, "Origin Corp");
    assert_eq!(entity_type, "entity");
    assert!((confidence.unwrap() - 0.4).abs() < 1e-6);

    // Update: merge_entities promotes the canonical's generic type to the
    // alias's concrete type and re-syncs the shadow page.
    db.merge_entities(&canonical, &alias).await.unwrap();

    let (title, entity_type, confidence) = read_shadow(&db, &canonical).await;
    assert_eq!(title, "Origin Corp", "title must be unchanged by the merge");
    assert_eq!(
        entity_type, "database",
        "entity_type must round-trip the merge's generic->concrete promotion"
    );
    assert!(
        (confidence.unwrap() - 0.4).abs() < 1e-6,
        "confidence must still track the entities row after the re-sync"
    );
}

/// `count_relations` reads live `relates` edges, not the frozen legacy
/// `relations` table (which `create_relation` stopped writing at G6 Stage 2
/// PR 2b). Five entities plus one live relation is exactly the shape the
/// GraphAlive onboarding gate (`onboarding.rs`) needs to fire.
#[tokio::test]
async fn count_relations_counts_live_relates_edges() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    let mut ids = Vec::new();
    for name in ["Ent One", "Ent Two", "Ent Three", "Ent Four", "Ent Five"] {
        ids.push(
            db.store_entity(name, "concept", None, None, Some(0.5))
                .await
                .unwrap(),
        );
    }
    assert_eq!(db.count_entities().await.unwrap(), 5);
    assert_eq!(
        db.count_relations().await.unwrap(),
        0,
        "entities alone must count as zero relations"
    );

    db.create_relation(
        &ids[0],
        &ids[1],
        "related_to",
        Some("test"),
        None,
        None,
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        db.count_relations().await.unwrap(),
        1,
        "one live relation created through the writer must be counted"
    );
}
