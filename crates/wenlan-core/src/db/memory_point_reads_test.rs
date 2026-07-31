// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

struct TestMemoryRow<'a> {
    id: &'a str,
    source: &'a str,
    source_id: &'a str,
    chunk_index: i64,
    last_modified: i64,
    pending_revision: bool,
    supersedes: Option<&'a str>,
    structured_fields: Option<&'a str>,
    content_hash: Option<&'a str>,
}

async fn insert_memory_row(db: &MemoryDB, row: TestMemoryRow<'_>) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories (
             id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, pending_revision, supersedes, structured_fields, content_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'text', ?8, ?9, ?10, ?11)",
        libsql::params![
            row.id,
            format!("content:{}", row.id),
            row.source,
            row.source_id,
            format!("title:{}", row.id),
            row.chunk_index,
            row.last_modified,
            row.pending_revision as i64,
            row.supersedes,
            row.structured_fields,
            row.content_hash,
        ],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn active_non_episode_lookup_uses_physical_id_without_source_kind_narrowing() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "physical_active",
            source: "web",
            source_id: "logical_active",
            chunk_index: 0,
            last_modified: 1,
            pending_revision: false,
            supersedes: None,
            structured_fields: None,
            content_hash: None,
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "physical_pending",
            source: "memory",
            source_id: "logical_pending",
            chunk_index: 0,
            last_modified: 2,
            pending_revision: true,
            supersedes: None,
            structured_fields: None,
            content_hash: None,
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "physical_episode",
            source: "episode",
            source_id: "logical_episode",
            chunk_index: 0,
            last_modified: 3,
            pending_revision: false,
            supersedes: None,
            structured_fields: None,
            content_hash: None,
        },
    )
    .await;

    assert!(db
        .has_active_non_episode_memory_id("physical_active")
        .await
        .unwrap());
    assert!(
        !db.has_active_non_episode_memory_id("logical_active")
            .await
            .unwrap(),
        "the fallback is keyed by physical memories.id, not source_id"
    );
    assert!(!db
        .has_active_non_episode_memory_id("physical_pending")
        .await
        .unwrap());
    assert!(!db
        .has_active_non_episode_memory_id("physical_episode")
        .await
        .unwrap());
    assert!(!db
        .has_active_non_episode_memory_id("physical_missing")
        .await
        .unwrap());
}

#[tokio::test]
async fn pending_revision_prefers_exact_source_id_and_keeps_structured_json_raw() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    let raw_structured = r#"{"raw":"exact","revision_kind":"not-interpreted-here"}"#;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "exact_row",
            source: "memory",
            source_id: "revision_key",
            chunk_index: 0,
            last_modified: 10,
            pending_revision: true,
            supersedes: Some("page_exact"),
            structured_fields: Some(raw_structured),
            content_hash: None,
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "newer_legacy_row",
            source: "memory",
            source_id: "newer_revision",
            chunk_index: 0,
            last_modified: 999,
            pending_revision: true,
            supersedes: Some("revision_key"),
            structured_fields: Some(r#"{"raw":"newer supersedes match"}"#),
            content_hash: None,
        },
    )
    .await;

    let payload = db
        .pending_memory_revision_payload("revision_key")
        .await
        .unwrap()
        .expect("exact source_id revision");
    assert_eq!(payload.revision_id, "revision_key");
    assert_eq!(payload.supersedes, "page_exact");
    assert_eq!(payload.content, "content:exact_row");
    assert_eq!(
        payload.structured_fields.as_deref(),
        Some(raw_structured),
        "the DB seam must not interpret or normalize structured_fields"
    );

    assert!(db
        .pending_memory_revision_payload("missing_revision")
        .await
        .unwrap()
        .is_none());

    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "null_structured_row",
            source: "memory",
            source_id: "null_structured_revision",
            chunk_index: 0,
            last_modified: 15,
            pending_revision: true,
            supersedes: Some("null_structured_target"),
            structured_fields: None,
            content_hash: None,
        },
    )
    .await;
    let null_structured = db
        .pending_memory_revision_payload("null_structured_revision")
        .await
        .unwrap()
        .expect("pending revision with SQL NULL structured_fields");
    assert_eq!(null_structured.structured_fields, None);

    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "legacy_older_row",
            source: "memory",
            source_id: "legacy_older_revision",
            chunk_index: 0,
            last_modified: 20,
            pending_revision: true,
            supersedes: Some("legacy_target"),
            structured_fields: Some(r#"{"raw":"older eligible fallback"}"#),
            content_hash: None,
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "legacy_newer_row",
            source: "memory",
            source_id: "legacy_newer_revision",
            chunk_index: 0,
            last_modified: 30,
            pending_revision: true,
            supersedes: Some("legacy_target"),
            structured_fields: Some(r#"{"raw":"newer eligible fallback"}"#),
            content_hash: None,
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "legacy_wrong_source",
            source: "episode",
            source_id: "legacy_wrong_source_revision",
            chunk_index: 0,
            last_modified: 50,
            pending_revision: true,
            supersedes: Some("legacy_target"),
            structured_fields: Some(r#"{"raw":"wrong source"}"#),
            content_hash: None,
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "legacy_not_pending",
            source: "memory",
            source_id: "legacy_not_pending_revision",
            chunk_index: 0,
            last_modified: 40,
            pending_revision: false,
            supersedes: Some("legacy_target"),
            structured_fields: Some(r#"{"raw":"not pending"}"#),
            content_hash: None,
        },
    )
    .await;

    let fallback = db
        .pending_memory_revision_payload("legacy_target")
        .await
        .unwrap()
        .expect("newest eligible supersedes fallback");
    assert_eq!(fallback.revision_id, "legacy_newer_revision");
    assert_eq!(fallback.supersedes, "legacy_target");
    assert_eq!(fallback.content, "content:legacy_newer_row");
    assert_eq!(
        fallback.structured_fields.as_deref(),
        Some(r#"{"raw":"newer eligible fallback"}"#)
    );
}

#[tokio::test]
async fn content_hash_read_keeps_chunk_zero_nulls_and_omits_nonmatches() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "null_chunk_zero",
            source: "memory",
            source_id: "source_null",
            chunk_index: 0,
            last_modified: 1,
            pending_revision: false,
            supersedes: None,
            structured_fields: None,
            content_hash: None,
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "hash_chunk_zero",
            source: "memory",
            source_id: "source_hash",
            chunk_index: 0,
            last_modified: 2,
            pending_revision: false,
            supersedes: None,
            structured_fields: None,
            content_hash: Some("hash-zero"),
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "hash_chunk_one",
            source: "memory",
            source_id: "source_chunk_one",
            chunk_index: 1,
            last_modified: 3,
            pending_revision: false,
            supersedes: None,
            structured_fields: None,
            content_hash: Some("hash-one"),
        },
    )
    .await;
    insert_memory_row(
        &db,
        TestMemoryRow {
            id: "hash_episode",
            source: "episode",
            source_id: "source_episode",
            chunk_index: 0,
            last_modified: 4,
            pending_revision: false,
            supersedes: None,
            structured_fields: None,
            content_hash: Some("hash-episode"),
        },
    )
    .await;

    let source_ids = [
        "source_null",
        "source_hash",
        "source_chunk_one",
        "source_episode",
        "source_missing",
    ]
    .map(str::to_string);
    let hashes = db
        .memory_content_hashes_for_source_ids(&source_ids)
        .await
        .unwrap();

    assert_eq!(hashes.len(), 2);
    assert_eq!(hashes.get("source_null"), Some(&None));
    assert_eq!(
        hashes.get("source_hash"),
        Some(&Some("hash-zero".to_string()))
    );
    assert!(!hashes.contains_key("source_chunk_one"));
    assert!(!hashes.contains_key("source_episode"));
    assert!(!hashes.contains_key("source_missing"));
    assert!(db
        .memory_content_hashes_for_source_ids(&[])
        .await
        .unwrap()
        .is_empty());
}
