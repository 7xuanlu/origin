// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

async fn insert_entity(db: &MemoryDB, id: &str, entity_type: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO entities (id, name, entity_type, created_at, updated_at)
         VALUES (?1, ?1, ?2, 1, 1)",
        libsql::params![id, entity_type],
    )
    .await
    .unwrap();
}

async fn insert_relation(
    db: &MemoryDB,
    id: &str,
    from_entity: &str,
    to_entity: &str,
    relation_type: &str,
) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO relations (id, from_entity, to_entity, relation_type, created_at)
         VALUES (?1, ?2, ?3, ?4, 1)",
        libsql::params![id, from_entity, to_entity, relation_type],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn distinct_relation_types_keep_empty_and_collapse_duplicates() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    for id in ["entity-a", "entity-b", "entity-c", "entity-d"] {
        insert_entity(&db, id, "control").await;
    }
    insert_relation(&db, "novel-a", "entity-a", "entity-b", "novel_relation").await;
    insert_relation(&db, "novel-b", "entity-a", "entity-c", "novel_relation").await;
    insert_relation(&db, "empty", "entity-a", "entity-d", "").await;
    insert_relation(&db, "canonical", "entity-b", "entity-c", "related_to").await;

    let mut types = db
        .distinct_relation_types_for_vocabulary_heal()
        .await
        .unwrap();
    types.sort();
    assert_eq!(types, vec!["", "novel_relation", "related_to"]);
}

#[tokio::test]
async fn distinct_entity_types_keep_empty_and_collapse_duplicates() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_entity(&db, "novel-a", "novel_entity").await;
    insert_entity(&db, "novel-b", "novel_entity").await;
    insert_entity(&db, "empty", "").await;
    insert_entity(&db, "canonical", "person").await;

    let mut types = db
        .distinct_entity_types_for_vocabulary_heal()
        .await
        .unwrap();
    types.sort();
    assert_eq!(types, vec!["", "novel_entity", "person"]);
}
