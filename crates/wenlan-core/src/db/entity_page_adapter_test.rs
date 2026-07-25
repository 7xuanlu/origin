// SPDX-License-Identifier: Apache-2.0

use super::entity_page_adapter::{entity_id_for_page, page_id_for_entity};
use super::tests::test_db;

/// M3 stage F acceptance property: the adapter round-trips `entity_id <->
/// page_id` for every entity seeded through the real `store_entity`
/// dual-write path (migration 90's `entity_page_map`).
#[tokio::test]
async fn adapter_round_trips_entity_id_and_page_id() {
    let (db, _tmp) = test_db().await;
    let mut entity_ids = Vec::new();
    for i in 0..5 {
        let id = db
            .store_entity(
                &format!("Adapter Entity {i}"),
                "person",
                Some("adapter_seam"),
                None,
                Some(0.7),
            )
            .await
            .unwrap();
        entity_ids.push(id);
    }

    let conn = db.conn.lock().await;
    for entity_id in &entity_ids {
        let page_id = page_id_for_entity(&conn, entity_id)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("{entity_id} must have a mapped page_id"));
        let round_tripped = entity_id_for_page(&conn, &page_id).await.unwrap();
        assert_eq!(
            round_tripped.as_deref(),
            Some(entity_id.as_str()),
            "entity_id_for_page(page_id_for_entity(id)) must round-trip to id"
        );
    }
}

#[tokio::test]
async fn adapter_returns_none_for_nonexistent_id() {
    let (db, _tmp) = test_db().await;
    let conn = db.conn.lock().await;
    assert_eq!(
        page_id_for_entity(&conn, "nonexistent_entity")
            .await
            .unwrap(),
        None,
        "page_id_for_entity must return Ok(None) for an unmapped entity_id"
    );
    assert_eq!(
        entity_id_for_page(&conn, "nonexistent_page").await.unwrap(),
        None,
        "entity_id_for_page must return Ok(None) for an unmapped page_id"
    );
}
