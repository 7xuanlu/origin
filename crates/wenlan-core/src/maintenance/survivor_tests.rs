// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::db::MemoryDB;
use crate::events::NoopEmitter;

async fn new_test_db() -> (MemoryDB, tempfile::TempDir) {
    let db_dir = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(db_dir.path(), Arc::new(NoopEmitter))
        .await
        .unwrap();
    (db, db_dir)
}

async fn store_test_memory(db: &MemoryDB, id: &str) {
    db.upsert_documents(vec![wenlan_types::RawDocument {
        source: "memory".to_string(),
        source_id: id.to_string(),
        title: id.to_string(),
        content: format!("{id} evidence for deterministic page merge survivor tests."),
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        source_agent: Some("test".to_string()),
        confirmed: Some(true),
        ..Default::default()
    }])
    .await
    .unwrap();
}

async fn insert_test_page(db: &MemoryDB, id: &str, content: &str, source_ids: &[&str]) {
    let now = chrono::Utc::now().to_rfc3339();
    db.insert_page_with_kind(
        id,
        id,
        None,
        content,
        None,
        None,
        source_ids,
        &now,
        "research",
        "confirmed",
        Some("work"),
        Some("[]"),
    )
    .await
    .unwrap();
}

async fn set_page_selection_fixtures(
    db: &MemoryDB,
    id: &str,
    created_at: &str,
    last_modified: &str,
    user_edited: bool,
) {
    let conn = db.test_primary_session().await;
    conn.execute(
        "UPDATE pages SET created_at = ?2, last_modified = ?3, user_edited = ?4 WHERE id = ?1",
        libsql::params![id, created_at, last_modified, i64::from(user_edited)],
    )
    .await
    .unwrap();
}

fn config() -> MaintenanceTickConfig {
    MaintenanceTickConfig {
        page_match_threshold: 0.85,
        formation_threshold: 0.60,
        page_min_cluster_size: 3,
        token_limit: 3500,
        max_unlinked_cluster_size: 20,
        max_grouped_cluster_size: 20,
        max_per_tick: 5,
    }
}

async fn emitted_page_merge_source_ids(db: &MemoryDB) -> Vec<String> {
    let result = run_maintenance_tick(db, None, &PromptRegistry::default(), &config(), None)
        .await
        .unwrap();
    assert_eq!(result.merge_cards_emitted, 1);

    let cards: Vec<_> = db
        .get_pending_refinements()
        .await
        .unwrap()
        .into_iter()
        .filter(|card| card.action == "page_merge")
        .collect();
    assert_eq!(cards.len(), 1);
    cards[0].source_ids.clone()
}

#[tokio::test]
async fn page_merge_card_survivor_prefers_user_edited_page_when_iterator_puts_it_right() {
    let (db, _db_dir) = new_test_db().await;
    for id in ["mem_user_a", "mem_user_b"] {
        store_test_memory(&db, id).await;
    }
    insert_test_page(
        &db,
        "page_a_machine",
        "Shared Rust ownership evidence page from distillation.",
        &["mem_user_a", "mem_user_b"],
    )
    .await;
    insert_test_page(
        &db,
        "page_z_human",
        "Shared Rust ownership evidence page with human edits.",
        &["mem_user_a", "mem_user_b"],
    )
    .await;
    set_page_selection_fixtures(
        &db,
        "page_a_machine",
        "2026-01-02T00:00:00Z",
        "2026-01-02T00:00:00Z",
        false,
    )
    .await;
    set_page_selection_fixtures(
        &db,
        "page_z_human",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        true,
    )
    .await;

    let source_ids = emitted_page_merge_source_ids(&db).await;

    assert_eq!(source_ids, vec!["page_z_human", "page_a_machine"]);
}

#[tokio::test]
async fn page_merge_card_survivor_prefers_page_with_more_source_memories() {
    let (db, _db_dir) = new_test_db().await;
    for id in ["mem_count_a", "mem_count_b", "mem_count_c"] {
        store_test_memory(&db, id).await;
    }
    insert_test_page(
        &db,
        "page_a_two_sources",
        "Shared survivor evidence with two source memories.",
        &["mem_count_a", "mem_count_b"],
    )
    .await;
    insert_test_page(
        &db,
        "page_z_three_sources",
        "Shared survivor evidence with three source memories.",
        &["mem_count_a", "mem_count_b", "mem_count_c"],
    )
    .await;
    set_page_selection_fixtures(
        &db,
        "page_a_two_sources",
        "2026-01-02T00:00:00Z",
        "2026-01-02T00:00:00Z",
        false,
    )
    .await;
    set_page_selection_fixtures(
        &db,
        "page_z_three_sources",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        false,
    )
    .await;

    let source_ids = emitted_page_merge_source_ids(&db).await;

    assert_eq!(
        source_ids,
        vec!["page_z_three_sources", "page_a_two_sources"]
    );
}
