// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

async fn insert_entity(db: &MemoryDB, id: &str, name: &str, entity_type: &str) {
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO entities (id, name, entity_type, created_at, updated_at)
             VALUES (?1, ?2, ?3, 1, 1)",
            libsql::params![id, name, entity_type],
        )
        .await
        .unwrap();
    }
    // G6 Stage 2 PR 2c sub-step 3 item 1: both readers under test now read
    // the `kind='entity'` shadow page, not `entities` directly -- backfill
    // it for this raw-seeded row, same as every other raw-SQL entity
    // fixture since the 1.5a port.
    db.test_seed_entity_shadow_page(id).await.unwrap();
}

#[tokio::test]
async fn duplicate_name_groups_case_fold_counts_and_exclude_singletons() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    for (id, name) in [
        ("alice-1", "Alice"),
        ("alice-2", "alice"),
        ("alice-3", "ALICE"),
        ("carol-1", "Carol"),
        ("carol-2", "carol"),
        ("bob-single", "Bob"),
    ] {
        insert_entity(&db, id, name, "person").await;
    }

    let mut groups = db
        .duplicate_name_groups_for_merge_candidates()
        .await
        .unwrap();
    groups.sort_by(|left, right| left.lower_name.cmp(&right.lower_name));
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].lower_name, "alice");
    assert_eq!(groups[0].entity_count, 3);
    assert_eq!(groups[1].lower_name, "carol");
    assert_eq!(groups[1].entity_count, 2);
    assert!(
        groups.iter().all(|group| group.lower_name != "bob"),
        "the HAVING clause must exclude singleton names"
    );
}

#[tokio::test]
async fn minhash_candidate_inputs_keep_low_entropy_rows_and_all_fields() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_entity(&db, "low", "A", "person").await;
    insert_entity(&db, "high", "Distributed Systems Architecture", "project").await;

    let mut inputs = db.entities_for_minhash_merge_candidates().await.unwrap();
    inputs.sort_by(|left, right| left.id.cmp(&right.id));
    assert_eq!(inputs.len(), 2);
    assert_eq!(inputs[0].id, "high");
    assert_eq!(inputs[0].name, "Distributed Systems Architecture");
    assert_eq!(inputs[0].entity_type, "project");
    assert_eq!(inputs[1].id, "low");
    assert_eq!(inputs[1].name, "A");
    assert_eq!(inputs[1].entity_type, "person");
}
