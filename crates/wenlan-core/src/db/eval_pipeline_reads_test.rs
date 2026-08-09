// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;

struct MemoryRow<'a> {
    id: &'a str,
    source_id: &'a str,
    source: &'a str,
    content: &'a str,
    chunk_index: i64,
    supersede_mode: &'a str,
    entity_id: Option<&'a str>,
}

async fn insert_memory(db: &MemoryDB, row: MemoryRow<'_>) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories (
             id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, supersede_mode, entity_id
        ) VALUES (?1, ?2, ?3, ?4, 'title', ?5, 1, 'text', ?6, ?7)",
        libsql::params![
            row.id,
            row.content,
            row.source,
            row.source_id,
            row.chunk_index,
            row.supersede_mode,
            row.entity_id,
        ],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn corpus_contents_keep_exact_head_and_nonarchive_membership() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    for row in [
        MemoryRow {
            id: "memory-head",
            source_id: "memory-head",
            source: "memory",
            content: "duplicate",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: None,
        },
        MemoryRow {
            id: "document-head",
            source_id: "document-head",
            source: "document",
            content: "duplicate",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: None,
        },
        MemoryRow {
            id: "empty-head",
            source_id: "empty-head",
            source: "memory",
            content: "",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: None,
        },
        MemoryRow {
            id: "archive-head",
            source_id: "archive-head",
            source: "memory",
            content: "archived",
            chunk_index: 0,
            supersede_mode: "archive",
            entity_id: None,
        },
        MemoryRow {
            id: "memory-child",
            source_id: "memory-child",
            source: "memory",
            content: "child",
            chunk_index: 1,
            supersede_mode: "hide",
            entity_id: None,
        },
    ] {
        insert_memory(&db, row).await;
    }

    let mut contents = db.eval_pipeline_corpus_contents().await.unwrap();
    contents.sort();
    assert_eq!(
        contents,
        vec![
            "".to_string(),
            "duplicate".to_string(),
            "duplicate".to_string(),
        ],
        "document and empty heads stay; duplicate contents are not collapsed"
    );
}

#[tokio::test]
async fn evidence_pairs_keep_exact_legacy_join_boundaries_and_duplicates() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "CREATE TABLE concepts (
                 id TEXT PRIMARY KEY,
                 entity_id TEXT,
                 status TEXT NOT NULL
             );
             CREATE TABLE concept_sources (
                 concept_id TEXT NOT NULL,
                 memory_source_id TEXT NOT NULL
             );
             INSERT INTO concepts (id, entity_id, status) VALUES
                 ('active-good', 'entity-good', 'active'),
                 ('inactive', 'entity-inactive', 'archived'),
                 ('active-prefix', 'entity-prefix', 'active'),
                 ('active-chunk', 'entity-chunk', 'active'),
                 ('active-entity', 'entity-expected', 'active');
             INSERT INTO concept_sources (concept_id, memory_source_id) VALUES
                 ('active-good', 'evidence-good'),
                 ('inactive', 'evidence-inactive'),
                 ('active-prefix', 'evidence-prefix'),
                 ('active-chunk', 'evidence-chunk'),
                 ('active-entity', 'evidence-entity');",
        )
        .await
        .unwrap();
    }

    for row in [
        MemoryRow {
            id: "good-1",
            source_id: "merged_good",
            source: "memory",
            content: "body",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: Some("entity-good"),
        },
        MemoryRow {
            id: "good-2",
            source_id: "merged_good",
            source: "memory",
            content: "body",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: Some("entity-good"),
        },
        MemoryRow {
            id: "good-wildcard",
            source_id: "mergedXwild",
            source: "memory",
            content: "body",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: Some("entity-good"),
        },
        MemoryRow {
            id: "inactive",
            source_id: "merged_inactive",
            source: "memory",
            content: "body",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: Some("entity-inactive"),
        },
        MemoryRow {
            id: "prefix",
            source_id: "plain_prefix",
            source: "memory",
            content: "body",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: Some("entity-prefix"),
        },
        MemoryRow {
            id: "chunk",
            source_id: "merged_chunk",
            source: "memory",
            content: "body",
            chunk_index: 1,
            supersede_mode: "hide",
            entity_id: Some("entity-chunk"),
        },
        MemoryRow {
            id: "entity",
            source_id: "merged_entity",
            source: "memory",
            content: "body",
            chunk_index: 0,
            supersede_mode: "hide",
            entity_id: Some("entity-other"),
        },
    ] {
        insert_memory(&db, row).await;
    }

    let pairs = db.eval_pipeline_evidence_pairs().await.unwrap();
    assert_eq!(pairs.len(), 3);
    assert_eq!(
        pairs
            .iter()
            .filter(|(memory_sid, concept_sid)| {
                memory_sid == "evidence-good" && concept_sid == "merged_good"
            })
            .count(),
        2,
        "two matching memory rows with one source id must retain duplicate pairs"
    );
    assert!(pairs.iter().any(|(memory_sid, concept_sid)| {
        memory_sid == "evidence-good" && concept_sid == "mergedXwild"
    }));
}
