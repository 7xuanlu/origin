// SPDX-License-Identifier: Apache-2.0
//
// #708 "detected entities are an index, not a to-do": the three lifecycle
// states (detected / established / archived), the two ways an entity gets
// established on its own, what the page-browse surfaces show, and the bulk
// archive/restore round trip.

use super::tests::test_db;
use super::{MemoryDB, UNFILED_SPACE_ID};
use crate::read_scope::ReadScope;
use crate::sources::RawDocument;
use std::collections::HashMap;
use wenlan_types::requests::{EntitySelection, ListEntitiesRequest};
use wenlan_types::EntityStatus;

fn memory_doc(source_id: &str) -> RawDocument {
    RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: format!("A memory recorded for {source_id}."),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        metadata: HashMap::new(),
        memory_type: Some("fact".to_string()),
        space: None,
        source_agent: None,
        confidence: Some(0.9),
        confirmed: Some(true),
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    }
}

/// Link `count` freshly stored memories to `entity_id`, one call each, so the
/// promotion threshold is crossed the way production crosses it -- one link at
/// a time, not one batch.
async fn link_memories(db: &MemoryDB, entity_id: &str, prefix: &str, count: usize) {
    for index in 0..count {
        let source_id = format!("{prefix}-{index}");
        db.upsert_documents(vec![memory_doc(&source_id)])
            .await
            .unwrap();
        db.link_memory_entities(&source_id, &[entity_id])
            .await
            .unwrap();
    }
}

async fn entity_row(db: &MemoryDB, entity_id: &str) -> wenlan_types::Entity {
    db.get_entity_detail(entity_id).await.unwrap().entity
}

/// `legacy` lineage keeps the edge out of `edges_space_fence`, so this seeds a
/// relation without also asserting the fence's behavior -- the round trip under
/// test is archive/restore, not the fence.
async fn insert_edge(db: &MemoryDB, edge_id: &str, src: &str, dst: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO edges (edge_id,src_id,src_kind,dst_id,dst_kind,edge_type,lineage,grounded,space,created_at,semantic_type)
         VALUES (?1,?2,'entity',?3,'entity','relates','legacy',0,?4,1,'related')",
        libsql::params![edge_id, src, dst, UNFILED_SPACE_ID],
    )
    .await
    .unwrap();
}

async fn edge_is_active(db: &MemoryDB, edge_id: &str) -> bool {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT valid_until IS NULL FROM edges WHERE edge_id = ?1",
            libsql::params![edge_id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("edge row");
    row.get::<i64>(0).unwrap() == 1
}

async fn shadow_page_id(db: &MemoryDB, entity_id: &str) -> String {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT page_id FROM entity_page_map WHERE entity_id = ?1",
            libsql::params![entity_id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("entity_page_map row");
    row.get::<String>(0).unwrap()
}

async fn page_status(db: &MemoryDB, entity_id: &str) -> String {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT p.status FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id \
             WHERE epm.entity_id = ?1",
            libsql::params![entity_id],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().expect("shadow page row");
    row.get::<String>(0).unwrap()
}

/// A/C 1: the page-browse surfaces are the Wiki. A detected entity is a guess
/// the system has not earned the right to show there, an archived one is a
/// guess the person rejected, and an established one must arrive carrying the
/// `entity_id` that makes it open as an entity rather than as a blank page.
#[tokio::test]
async fn browse_scoped_pages_hide_detected_entities_and_link_established_ones() {
    let (db, _tmp) = test_db().await;
    let now = chrono::Utc::now().to_rfc3339();

    let detected_ids = {
        let mut ids = Vec::new();
        for name in ["Detected One", "Detected Two", "Detected Three"] {
            ids.push(
                db.store_entity(name, "concept", None, None, Some(0.5))
                    .await
                    .unwrap(),
            );
        }
        ids
    };
    let established = db
        .store_entity("Established Org", "organization", None, None, Some(0.9))
        .await
        .unwrap();
    db.confirm_entity(&established, true).await.unwrap();

    db.insert_page_with_kind(
        "distilled-page",
        "A distilled page",
        None,
        "page prose",
        None,
        None,
        &[],
        &now,
        "distilled",
        "confirmed",
        None,
        None,
    )
    .await
    .unwrap();

    for pages in [
        db.list_pages_browse("active", 50, 0).await.unwrap(),
        db.list_pages_scoped_browse("active", 50, 0, &ReadScope::Global)
            .await
            .unwrap(),
    ] {
        let ids: Vec<&str> = pages.iter().map(|page| page.id.as_str()).collect();
        assert_eq!(
            pages.len(),
            2,
            "browse must show the distilled page and the established entity only, got {ids:?}"
        );
        assert!(ids.contains(&"distilled-page"));

        let entity_row = pages
            .iter()
            .find(|page| page.id != "distilled-page")
            .expect("the established entity's page");
        assert_eq!(
            entity_row.entity_id.as_deref(),
            Some(established.as_str()),
            "an established entity's row must carry its entity id so it opens as an entity"
        );

        // `kind` is an internal classification; the wire shape must not leak it.
        let encoded = serde_json::to_value(entity_row).unwrap();
        assert!(
            encoded.get("kind").is_none(),
            "Page must not serialize `kind`, got {encoded}"
        );
    }

    // The fenced twin is the mutation/validation surface and still sees no
    // entity rows at all, established or not.
    let fenced = db.list_pages("active", 50, 0).await.unwrap();
    assert_eq!(fenced.len(), 1, "fenced list must show only the real page");
    assert_eq!(fenced[0].id, "distilled-page");

    // Archiving the established entity takes it back out of browse.
    db.archive_entity(&established).await.unwrap();
    let after_archive = db.list_pages_browse("active", 50, 0).await.unwrap();
    assert_eq!(after_archive.len(), 1);
    assert_eq!(after_archive[0].id, "distilled-page");

    for id in &detected_ids {
        assert_eq!(entity_row(&db, id).await.status, EntityStatus::Detected);
    }
}

/// A/C 2: an entity the system keeps seeing has earned its place. The default
/// threshold is three linked memories, and the promotion has to happen in the
/// same transaction as the link that crosses it.
#[tokio::test]
async fn entity_establishes_on_the_third_memory_link_with_auto_memories() {
    let (db, _tmp) = test_db().await;
    let entity = db
        .store_entity("Threshold Corp", "organization", None, None, Some(0.7))
        .await
        .unwrap();

    link_memories(&db, &entity, "below", 2).await;
    let below = entity_row(&db, &entity).await;
    assert_eq!(
        below.status,
        EntityStatus::Detected,
        "two links is not three"
    );
    assert_eq!(below.memory_count, 2);
    assert_eq!(below.established_by, None);
    assert!(!below.confirmed);

    link_memories(&db, &entity, "third", 1).await;
    let crossed = entity_row(&db, &entity).await;
    assert_eq!(crossed.status, EntityStatus::Established);
    assert_eq!(crossed.memory_count, 3);
    assert_eq!(crossed.established_by.as_deref(), Some("auto:memories"));
    assert!(crossed.confirmed);

    // Further links must not re-stamp an entity that is already established.
    link_memories(&db, &entity, "extra", 1).await;
    let later = entity_row(&db, &entity).await;
    assert_eq!(later.memory_count, 4);
    assert_eq!(later.established_by.as_deref(), Some("auto:memories"));
}

/// A/C 2: a distilled page that cites an entity is stronger evidence than any
/// number of mentions, so a citation establishes it whatever the count.
#[tokio::test]
async fn entity_establishes_by_citation_regardless_of_memory_count() {
    let (db, _tmp) = test_db().await;
    let entity = db
        .store_entity("Cited Person", "person", None, None, Some(0.6))
        .await
        .unwrap();
    assert_eq!(
        entity_row(&db, &entity).await.status,
        EntityStatus::Detected
    );

    assert!(db.establish_entity_by_citation(&entity).await.unwrap());
    let cited = entity_row(&db, &entity).await;
    assert_eq!(cited.status, EntityStatus::Established);
    assert_eq!(cited.memory_count, 0, "no memory link was needed");
    assert_eq!(cited.established_by.as_deref(), Some("auto:citation"));

    // Already established: nothing to do, and the reason must not be rewritten.
    assert!(!db.establish_entity_by_citation(&entity).await.unwrap());
    assert_eq!(
        entity_row(&db, &entity).await.established_by.as_deref(),
        Some("auto:citation")
    );
}

/// A person confirming an entity is the third way it gets established, and the
/// record has to say it was them rather than the system.
#[tokio::test]
async fn confirm_entity_records_established_by_manual() {
    let (db, _tmp) = test_db().await;
    let entity = db
        .store_entity("Confirmed Place", "location", None, None, Some(0.5))
        .await
        .unwrap();

    db.confirm_entity(&entity, true).await.unwrap();
    let confirmed = entity_row(&db, &entity).await;
    assert_eq!(confirmed.status, EntityStatus::Established);
    assert_eq!(confirmed.established_by.as_deref(), Some("manual"));

    // Un-confirming puts it back where it was, reason and all.
    db.confirm_entity(&entity, false).await.unwrap();
    let reverted = entity_row(&db, &entity).await;
    assert_eq!(reverted.status, EntityStatus::Detected);
    assert_eq!(reverted.established_by, None);
}

/// A/C 3: archiving in bulk is the whole point of the Entities view, and it is
/// only safe to offer if it is exactly reversible -- page status, edges, and
/// the entity's own fields all have to come back as they were.
#[tokio::test]
async fn archive_entities_then_restore_entities_round_trips_pages_and_edges() {
    let (db, _tmp) = test_db().await;
    let alpha = db
        .store_entity("Alpha Bulk", "organization", None, None, Some(0.8))
        .await
        .unwrap();
    let beta = db
        .store_entity("Beta Bulk", "organization", None, None, Some(0.8))
        .await
        .unwrap();
    link_memories(&db, &alpha, "alpha-mem", 3).await;
    insert_edge(&db, "edge-alpha-beta", &alpha, &beta).await;

    let before_alpha = entity_row(&db, &alpha).await;
    assert_eq!(before_alpha.status, EntityStatus::Established);
    assert!(edge_is_active(&db, "edge-alpha-beta").await);

    let selection = EntitySelection {
        ids: Some(vec![alpha.clone(), beta.clone()]),
        filter: None,
    };
    let archived = db
        .archive_entities(&selection, &ReadScope::Global, false)
        .await
        .unwrap();
    assert_eq!(archived.count, 2);
    assert!(!archived.dry_run);
    assert_eq!(page_status(&db, &alpha).await, "archived");
    assert_eq!(page_status(&db, &beta).await, "archived");
    // `get_entity_detail` is a live-entity reader, so an archived entity is
    // read back through the Entities view's own reader.
    let archived_filter = ListEntitiesRequest {
        status: Some(EntityStatus::Archived),
        ..Default::default()
    };
    let (archived_rows, archived_total) = db
        .query_entities_scoped(&archived_filter, &ReadScope::Global)
        .await
        .unwrap();
    assert_eq!(archived_total, 2);
    assert!(archived_rows
        .iter()
        .all(|row| row.status == EntityStatus::Archived));
    assert!(
        !edge_is_active(&db, "edge-alpha-beta").await,
        "archiving an endpoint must retire its edges"
    );

    // Re-archiving the same selection changes nothing: already-archived
    // entities are skipped, not re-stamped or errored.
    let again = db
        .archive_entities(&selection, &ReadScope::Global, false)
        .await
        .unwrap();
    assert_eq!(again.count, 0);
    assert!(again.entity_ids.is_empty());

    let restored = db
        .restore_entities(&selection, &ReadScope::Global, false)
        .await
        .unwrap();
    assert_eq!(restored.count, 2);
    assert_eq!(page_status(&db, &alpha).await, "active");
    assert_eq!(page_status(&db, &beta).await, "active");
    assert!(
        edge_is_active(&db, "edge-alpha-beta").await,
        "restore must bring the archive-retired edge back"
    );

    let after_alpha = entity_row(&db, &alpha).await;
    assert_eq!(after_alpha.status, before_alpha.status);
    assert_eq!(after_alpha.established_by, before_alpha.established_by);
    assert_eq!(after_alpha.memory_count, before_alpha.memory_count);
    assert_eq!(after_alpha.confirmed, before_alpha.confirmed);
    assert_eq!(after_alpha.name, before_alpha.name);
    assert_eq!(after_alpha.entity_type, before_alpha.entity_type);
}

/// A bulk action a person cannot preview is one they will not run. `dry_run`
/// has to answer "how many, and which" without touching anything.
#[tokio::test]
async fn archive_entities_dry_run_mutates_nothing() {
    let (db, _tmp) = test_db().await;
    let entity = db
        .store_entity("Preview Only", "concept", None, None, Some(0.4))
        .await
        .unwrap();

    let selection = EntitySelection {
        ids: Some(vec![entity.clone()]),
        filter: None,
    };
    let preview = db
        .archive_entities(&selection, &ReadScope::Global, true)
        .await
        .unwrap();
    assert_eq!(preview.count, 1);
    assert!(preview.dry_run);
    assert_eq!(preview.entity_ids, vec![entity.clone()]);
    assert_eq!(
        page_status(&db, &entity).await,
        "active",
        "a dry run must not archive anything"
    );

    // Neither half of the pair may mutate under dry_run.
    db.archive_entities(&selection, &ReadScope::Global, false)
        .await
        .unwrap();
    let restore_preview = db
        .restore_entities(&selection, &ReadScope::Global, true)
        .await
        .unwrap();
    assert_eq!(restore_preview.count, 1);
    assert!(restore_preview.dry_run);
    assert_eq!(page_status(&db, &entity).await, "archived");
}

/// A selection is either explicit ids or a filter. Core refuses both and
/// neither, because "neither" would mean archiving the entire scope.
#[tokio::test]
async fn archive_entities_refuses_an_ambiguous_selection() {
    let (db, _tmp) = test_db().await;
    let empty = EntitySelection::default();
    let both = EntitySelection {
        ids: Some(vec!["some-id".to_string()]),
        filter: Some(ListEntitiesRequest::default()),
    };
    for selection in [empty, both] {
        let error = db
            .archive_entities(&selection, &ReadScope::Global, true)
            .await
            .expect_err("an ambiguous selection must be refused");
        assert!(
            matches!(error, crate::WenlanError::Validation(_)),
            "expected a validation refusal, got {error:?}"
        );
    }
}

/// The Entities view filters, and then hands the same filter to the bulk
/// action. Both sides must select the same rows, or "archive everything I am
/// looking at" archives something else.
#[tokio::test]
async fn query_entities_scoped_and_archive_entities_agree_on_a_filter() {
    let (db, _tmp) = test_db().await;
    let quiet = db
        .store_entity("Quiet Signal", "concept", None, None, Some(0.3))
        .await
        .unwrap();
    let busy = db
        .store_entity("Busy Signal", "concept", None, None, Some(0.9))
        .await
        .unwrap();
    link_memories(&db, &busy, "busy-mem", 3).await;

    let detected_only = ListEntitiesRequest {
        status: Some(EntityStatus::Detected),
        ..Default::default()
    };
    let (entities, total) = db
        .query_entities_scoped(&detected_only, &ReadScope::Global)
        .await
        .unwrap();
    assert_eq!(total, 1);
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].id, quiet);
    assert_eq!(entities[0].memory_count, 0);

    // A name filter is a case-insensitive substring over name and aliases.
    let by_name = ListEntitiesRequest {
        query: Some("signal".to_string()),
        ..Default::default()
    };
    let (matched, matched_total) = db
        .query_entities_scoped(&by_name, &ReadScope::Global)
        .await
        .unwrap();
    assert_eq!(matched_total, 2, "both entities carry 'Signal' in the name");
    assert_eq!(matched.len(), 2);

    // A memory-count bound reads the same count the row displays.
    let linked = ListEntitiesRequest {
        min_memories: Some(3),
        ..Default::default()
    };
    let (busy_rows, _) = db
        .query_entities_scoped(&linked, &ReadScope::Global)
        .await
        .unwrap();
    assert_eq!(busy_rows.len(), 1);
    assert_eq!(busy_rows[0].id, busy);
    assert_eq!(busy_rows[0].memory_count, 3);

    // The same filter through the bulk action selects exactly that one row.
    let by_filter = EntitySelection {
        ids: None,
        filter: Some(detected_only),
    };
    let archived = db
        .archive_entities(&by_filter, &ReadScope::Global, false)
        .await
        .unwrap();
    assert_eq!(archived.count, 1);
    assert_eq!(archived.entity_ids, vec![quiet.clone()]);
    assert_eq!(page_status(&db, &quiet).await, "archived");
    assert_eq!(
        page_status(&db, &busy).await,
        "active",
        "the established entity was never in the filter"
    );
}

/// `limit` defaults to 100 and clamps at 1000, and `total` counts every match
/// rather than the page -- the view needs both to say "100 of 4,312".
#[tokio::test]
async fn query_entities_scoped_pages_and_reports_the_unpaged_total() {
    let (db, _tmp) = test_db().await;
    for index in 0..5 {
        db.store_entity(
            &format!("Paged Entity {index}"),
            "concept",
            None,
            None,
            None,
        )
        .await
        .unwrap();
    }

    let first_page = ListEntitiesRequest {
        limit: Some(2),
        ..Default::default()
    };
    let (page, total) = db
        .query_entities_scoped(&first_page, &ReadScope::Global)
        .await
        .unwrap();
    assert_eq!(page.len(), 2);
    assert_eq!(total, 5);

    let second_page = ListEntitiesRequest {
        limit: Some(2),
        offset: Some(4),
        ..Default::default()
    };
    let (tail, tail_total) = db
        .query_entities_scoped(&second_page, &ReadScope::Global)
        .await
        .unwrap();
    assert_eq!(tail.len(), 1, "one row is left after an offset of four");
    assert_eq!(tail_total, 5);
}

/// The page surface must refuse to archive or delete an entity's page, and the
/// refusal has to name the entity: "shadow page" is not a thing anyone outside
/// this file has heard of.
#[tokio::test]
async fn page_mutations_refuse_an_entity_page_and_name_the_entity() {
    let (db, _tmp) = test_db().await;
    let entity = db
        .store_entity("Guarded Entity", "organization", None, None, Some(0.9))
        .await
        .unwrap();
    let owner = shadow_page_id(&db, &entity).await;

    let error = db
        .archive_page(&owner)
        .await
        .expect_err("archiving an entity's page must be refused");
    let message = error.to_string();
    assert!(
        message.contains(
            "This page belongs to the entity 'Guarded Entity'. Archive or delete it from Entities"
        ),
        "guard message must name the entity and where to go, got: {message}"
    );
    assert!(
        message.contains(&entity),
        "guard message must carry the entity id for callers that only see a string, got: {message}"
    );

    let delete_error = db
        .delete_page(&owner)
        .await
        .expect_err("deleting an entity's page must be refused");
    assert!(delete_error
        .to_string()
        .contains("Archive or delete it from Entities"));

    // A real page is untouched by the guard.
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page_with_kind(
        "ordinary-page",
        "Ordinary",
        None,
        "prose",
        None,
        None,
        &[],
        &now,
        "distilled",
        "confirmed",
        None,
        None,
    )
    .await
    .unwrap();
    db.archive_page("ordinary-page").await.unwrap();
}

/// How many `kind='entity'` shadow pages carry this lowercased title, whatever
/// their status. The duplicate an archived entity used to spawn shows up here
/// as a second row and nowhere else, because the second row is invisible to
/// every live reader.
async fn entity_shadow_count(db: &MemoryDB, lower_title: &str) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM entity_page_map epm JOIN pages p ON p.id = epm.page_id \
             WHERE p.kind = 'entity' AND LOWER(p.title) = ?1",
            libsql::params![lower_title],
        )
        .await
        .unwrap();
    rows.next()
        .await
        .unwrap()
        .expect("count row")
        .get::<i64>(0)
        .unwrap()
}

/// A non-`legacy` edge, so `edges_space_fence` actually runs on it -- the
/// opposite of `insert_edge`, which is deliberately exempt.
async fn insert_fenced_edge(
    db: &MemoryDB,
    edge_id: &str,
    src: &str,
    dst: &str,
    space: &str,
) -> Result<(), String> {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO edges (edge_id,src_id,src_kind,dst_id,dst_kind,edge_type,lineage,grounded,space,created_at,semantic_type)
         VALUES (?1,?2,'entity',?3,'entity','relates','assertion',1,?4,1,'related')",
        libsql::params![edge_id, src, dst, space],
    )
    .await
    .map(|_| ())
    .map_err(|e| e.to_string())
}

/// The read the Entities view uses for an archived entity, since
/// `get_entity_detail` is a live-only reader.
async fn archived_entity(db: &MemoryDB, entity_id: &str) -> wenlan_types::Entity {
    let filter = ListEntitiesRequest {
        status: Some(EntityStatus::Archived),
        ..Default::default()
    };
    let (rows, _) = db
        .query_entities_scoped(&filter, &ReadScope::Global)
        .await
        .unwrap();
    rows.into_iter()
        .find(|row| row.id == entity_id)
        .expect("the archived entity must be readable through the Entities view")
}

/// A/C: archive is a prune, not a delete, so an archived entity has to absorb
/// a later mention of itself. Before #708 it could not: resolution skipped it,
/// a duplicate was created beside it, and a bulk archive at onboarding was
/// undone by the next import. Below the establish threshold the archived
/// entity just accumulates links; the mention that carries it over the
/// threshold brings it back, edges and all.
#[tokio::test]
async fn an_archived_entity_absorbs_mentions_and_returns_at_the_threshold() {
    let (db, _tmp) = test_db().await;
    let upside = db
        .store_entity("Upside", "concept", None, None, Some(0.6))
        .await
        .unwrap();
    let neighbour = db
        .store_entity("Downside", "concept", None, None, Some(0.6))
        .await
        .unwrap();
    insert_edge(&db, "upside-neighbour", &upside, &neighbour).await;

    db.archive_entity(&upside).await.unwrap();
    assert_eq!(page_status(&db, &upside).await, "archived");
    assert!(
        !edge_is_active(&db, "upside-neighbour").await,
        "archiving retires the edges incident to the entity"
    );

    // The mention. Resolution must land on the row the person archived.
    let (resolved, created) = db
        .resolve_or_create_entity("Upside", "concept", None, Some("test"), None)
        .await
        .unwrap();
    assert_eq!(
        resolved, upside,
        "a mention must resolve to the archived row"
    );
    assert!(!created, "no second entity may be created beside it");
    assert_eq!(
        entity_shadow_count(&db, "upside").await,
        1,
        "exactly one shadow page may carry this name"
    );

    link_memories(&db, &upside, "upside-mention", 1).await;
    assert_eq!(
        page_status(&db, &upside).await,
        "archived",
        "one mention is below the threshold, so the entity stays archived"
    );
    let still_archived = archived_entity(&db, &upside).await;
    assert_eq!(still_archived.memory_count, 1);
    assert_eq!(still_archived.status, EntityStatus::Archived);
    assert_eq!(still_archived.established_by, None);
    assert!(
        !edge_is_active(&db, "upside-neighbour").await,
        "an archived entity's edges stay retired while it is still archived"
    );

    // Two more carry it over `entity_establish_min_memories` (3).
    link_memories(&db, &upside, "upside-again", 2).await;
    assert_eq!(page_status(&db, &upside).await, "active");
    let back = entity_row(&db, &upside).await;
    assert_eq!(back.status, EntityStatus::Established);
    assert_eq!(back.established_by.as_deref(), Some("auto:memories"));
    assert_eq!(back.memory_count, 3);
    assert!(
        edge_is_active(&db, "upside-neighbour").await,
        "coming back restores the edges the archive retired"
    );
    assert_eq!(
        entity_shadow_count(&db, "upside").await,
        1,
        "and still without a duplicate"
    );
}

/// A/C: migration 130. The space fence must judge an edge endpoint by its
/// space, which is what it exists for, and not by whether the entity is
/// archived -- otherwise an archived entity can absorb a mention but not the
/// relation that came with it. The cross-space arm is the control: the fence
/// is live and still refuses the thing it was built to refuse.
#[tokio::test]
async fn the_space_fence_accepts_an_archived_endpoint_and_still_refuses_a_cross_space_one() {
    let (db, _tmp) = test_db().await;
    let archived = db
        .store_entity("Fenced Archived", "concept", None, None, Some(0.6))
        .await
        .unwrap();
    let live = db
        .store_entity("Fenced Live", "concept", None, None, Some(0.6))
        .await
        .unwrap();
    db.archive_entity(&archived).await.unwrap();
    assert_eq!(page_status(&db, &archived).await, "archived");

    insert_fenced_edge(&db, "fence-archived", &live, &archived, UNFILED_SPACE_ID)
        .await
        .expect("an edge whose endpoint is archived must be accepted");

    let refused = insert_fenced_edge(&db, "fence-cross", &live, &archived, "some-other-space")
        .await
        .expect_err("the fence must still refuse a cross-space edge");
    assert!(
        refused.contains("edges_space_fence"),
        "the refusal must come from the fence itself: {refused}"
    );
}
