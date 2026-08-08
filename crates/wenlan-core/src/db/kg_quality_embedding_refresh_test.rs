// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::db::TestEntity;

async fn insert_entity(db: &MemoryDB, id: &str, name: &str, embedding_updated_at: Option<i64>) {
    // G6 Stage 3: `stale_entity_embedding_candidates_for_refresh` reads the
    // shadow page, and migration 123 dropped `entities`, so the shadow IS the
    // seed.
    let mut entity = TestEntity::new(id, name, "test");
    if let Some(at) = embedding_updated_at {
        entity = entity.embedding_updated_at(at);
    }
    db.test_seed_entity_shadow_page(entity).await.unwrap();
}

async fn insert_observations(db: &MemoryDB, entity_id: &str, observations: &[(i64, &str)]) {
    let conn = db.conn.lock().await;
    conn.execute("BEGIN", ()).await.unwrap();
    for (index, (created_at, content)) in observations.iter().enumerate() {
        conn.execute(
            "INSERT INTO observations (id, entity_id, content, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            libsql::params![
                format!("{entity_id}-obs-{index:02}"),
                entity_id,
                *content,
                *created_at,
            ],
        )
        .await
        .unwrap();
    }
    conn.execute("COMMIT", ()).await.unwrap();
}

#[tokio::test]
async fn stale_candidates_preserve_strict_timestamp_and_five_observation_boundary() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_entity(&db, "null-five", "Null Five", None).await;
    insert_observations(
        &db,
        "null-five",
        &[(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")],
    )
    .await;
    insert_entity(&db, "null-four", "Null Four", None).await;
    insert_observations(&db, "null-four", &[(1, "a"), (2, "b"), (3, "c"), (4, "d")]).await;
    insert_entity(&db, "equal-five", "Equal Five", Some(100)).await;
    insert_observations(
        &db,
        "equal-five",
        &[(100, "a"), (100, "b"), (100, "c"), (100, "d"), (100, "e")],
    )
    .await;
    insert_entity(&db, "newer-five", "Newer Five", Some(100)).await;
    insert_observations(
        &db,
        "newer-five",
        &[(101, "a"), (102, "b"), (103, "c"), (104, "d"), (105, "e")],
    )
    .await;

    let mut candidates = db
        .stale_entity_embedding_candidates_for_refresh()
        .await
        .unwrap();
    candidates.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].entity_id, "newer-five");
    assert_eq!(candidates[0].entity_name, "Newer Five");
    assert_eq!(candidates[1].entity_id, "null-five");
    assert_eq!(candidates[1].entity_name, "Null Five");
}

#[tokio::test]
async fn recent_contents_keep_empty_and_return_exact_newest_ten() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_entity(&db, "recent", "Recent", None).await;
    insert_observations(
        &db,
        "recent",
        &[
            (1, "oldest-1"),
            (2, "oldest-2"),
            (3, "content-3"),
            (4, "content-4"),
            (5, "content-5"),
            (6, "content-6"),
            (7, "content-7"),
            (8, "content-8"),
            (9, "content-9"),
            (10, "content-10"),
            (11, ""),
            (12, "content-12"),
        ],
    )
    .await;

    let contents = db
        .recent_entity_observation_contents_for_embedding_refresh("recent")
        .await
        .unwrap();
    assert_eq!(
        contents,
        vec![
            "content-12",
            "",
            "content-10",
            "content-9",
            "content-8",
            "content-7",
            "content-6",
            "content-5",
            "content-4",
            "content-3",
        ]
    );
}

#[tokio::test]
async fn refresh_caller_embeds_name_plus_nonempty_newest_ten_exactly() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_entity(&db, "refresh-target", "Refresh Target", None).await;
    insert_observations(
        &db,
        "refresh-target",
        &[
            (1, "oldest-excluded"),
            (2, "content-2"),
            (3, "content-3"),
            (4, "content-4"),
            (5, "content-5"),
            (6, "content-6"),
            (7, "content-7"),
            (8, "content-8"),
            (9, "content-9"),
            (10, ""),
            (11, "content-11"),
        ],
    )
    .await;
    let expected = [
        "Refresh Target",
        "content-11",
        "content-9",
        "content-8",
        "content-7",
        "content-6",
        "content-5",
        "content-4",
        "content-3",
        "content-2",
    ]
    .join(". ");

    assert_eq!(
        crate::kg_quality::refresh_stale_entity_embeddings(&db)
            .await
            .unwrap(),
        1
    );
    let hits = db.search_entities_by_vector(&expected, 1).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].entity.id, "refresh-target");
    assert!(
        hits[0].distance <= 1e-5,
        "stored embedding must match exact caller text; distance={}",
        hits[0].distance
    );
    assert_eq!(
        crate::kg_quality::refresh_stale_entity_embeddings(&db)
            .await
            .unwrap(),
        0
    );
}
