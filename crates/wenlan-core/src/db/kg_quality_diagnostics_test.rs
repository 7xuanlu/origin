// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::db::TestEntity;

async fn insert_entity(db: &MemoryDB, id: &str, name: &str) {
    // G6 Stage 3: `list_contradiction_observation_counts` reads the
    // `kind='entity'` shadow page, and migration 123 dropped `entities`, so
    // the shadow IS the seed.
    db.test_seed_entity_shadow_page(TestEntity::new(id, name, "test"))
        .await
        .unwrap();
}

async fn insert_memory_source(db: &MemoryDB, id: &str, source_id: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories (
             id, content, source, source_id, title, chunk_index, last_modified, chunk_type
         ) VALUES (?1, 'content', 'memory', ?2, 'title', 0, 1, 'text')",
        libsql::params![id, source_id],
    )
    .await
    .unwrap();
}

async fn insert_relation(
    db: &MemoryDB,
    id: &str,
    from_entity: &str,
    to_entity: &str,
    source_memory_id: Option<&str>,
) {
    // Live `relates` edge; `source_memory_id` lives in the payload JSON.
    let payload = match source_memory_id {
        Some(source) => serde_json::json!({ "source_memory_id": source }).to_string(),
        None => "{}".to_string(),
    };
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO edges (edge_id,src_id,src_kind,dst_id,dst_kind,edge_type,lineage,grounded,space,payload,created_at,semantic_type)
         VALUES (?1,?2,'entity',?3,'entity','relates','legacy',0,'00000000-0000-4000-8000-000000000001',?4,1,'related_to')",
        libsql::params![id, from_entity, to_entity, payload],
    )
    .await
    .unwrap();
}

async fn insert_observations(db: &MemoryDB, entity_id: &str, count: usize) {
    let conn = db.conn.lock().await;
    let transaction = conn.transaction().await.unwrap();
    for index in 0..count {
        transaction
            .execute(
                "INSERT INTO observations (id, entity_id, content, created_at)
             VALUES (?1, ?2, 'observation', 1)",
                libsql::params![format!("{entity_id}-obs-{index:02}"), entity_id],
            )
            .await
            .unwrap();
    }
    transaction.commit().await.unwrap();
}

#[tokio::test]
async fn stale_relation_count_distinguishes_matched_missing_and_null_sources() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    for (id, name) in [
        ("from-a", "From A"),
        ("from-b", "From B"),
        ("to-a", "To A"),
        ("to-b", "To B"),
    ] {
        insert_entity(&db, id, name).await;
    }
    insert_memory_source(&db, "memory-row", "live-source").await;
    insert_relation(&db, "matched", "from-a", "to-a", Some("live-source")).await;
    insert_relation(&db, "missing", "from-b", "to-a", Some("missing-source")).await;
    insert_relation(&db, "null-source", "from-a", "to-b", None).await;

    assert_eq!(db.count_stale_relation_sources().await.unwrap(), 1);
}

#[tokio::test]
async fn contradiction_counts_include_ten_and_exclude_nine() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_entity(&db, "below-threshold", "Below threshold").await;
    insert_observations(&db, "below-threshold", 9).await;
    insert_entity(&db, "at-threshold", "At threshold").await;
    insert_observations(&db, "at-threshold", 10).await;

    let counts = db.list_contradiction_observation_counts().await.unwrap();
    assert_eq!(counts.len(), 1);
    assert_eq!(counts[0].entity_name, "At threshold");
    assert_eq!(counts[0].observation_count, 10);
}

#[tokio::test]
async fn contradiction_counts_apply_descending_order_and_limit() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    for count in 10..=31 {
        let id = format!("entity-{count:02}");
        insert_entity(&db, &id, &format!("Entity {count:02}")).await;
        insert_observations(&db, &id, count).await;
    }

    let counts = db.list_contradiction_observation_counts().await.unwrap();
    assert_eq!(counts.len(), 20);
    assert_eq!(
        counts
            .iter()
            .map(|item| item.observation_count)
            .collect::<Vec<_>>(),
        (12_i64..=31).rev().collect::<Vec<_>>()
    );
    assert_eq!(counts.first().unwrap().entity_name, "Entity 31");
    assert_eq!(counts.last().unwrap().entity_name, "Entity 12");
}
