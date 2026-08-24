// SPDX-License-Identifier: Apache-2.0
use crate::db::TestEntity;

use super::MemoryDB;

struct MemoryFixture<'a> {
    id: &'a str,
    source_id: &'a str,
    source: &'a str,
    chunk_index: i64,
    content: &'a str,
    supersedes: Option<&'a str>,
    supersede_mode: &'a str,
}

async fn insert_memory(db: &MemoryDB, fixture: MemoryFixture<'_>) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO memories (
             id, content, source, source_id, title, chunk_index, last_modified,
             chunk_type, supersedes, supersede_mode
         ) VALUES (?1, ?2, ?3, ?4, 'title', ?5, 1, 'text', ?6, ?7)",
        libsql::params![
            fixture.id,
            fixture.content,
            fixture.source,
            fixture.source_id,
            fixture.chunk_index,
            fixture.supersedes,
            fixture.supersede_mode,
        ],
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn supersedes_and_merged_id_scans_preserve_raw_selection_and_deduplication() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    for fixture in [
        MemoryFixture {
            id: "merged-new-row-1",
            source_id: "merged_new",
            source: "memory",
            chunk_index: 0,
            content: "new",
            supersedes: Some("seed_new"),
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "merged-new-row-2",
            source_id: "merged_new",
            source: "memory",
            chunk_index: 1,
            content: "duplicate",
            supersedes: Some("seed_new"),
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "merged-preexisting",
            source_id: "merged_preexisting",
            source: "memory",
            chunk_index: 0,
            content: "preexisting",
            supersedes: Some("seed_old"),
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "merged-null",
            source_id: "merged_null",
            source: "memory",
            chunk_index: 0,
            content: "null",
            supersedes: None,
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "not-merged",
            source_id: "plain_id",
            source: "memory",
            chunk_index: 0,
            content: "plain",
            supersedes: Some("seed_plain"),
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "merged-non-memory",
            source_id: "merged_document",
            source: "document",
            chunk_index: 0,
            content: "document",
            supersedes: Some("seed_document"),
            supersede_mode: "hide",
        },
    ] {
        insert_memory(&db, fixture).await;
    }

    let inputs = db.eval_lifecycle_supersedes_inputs().await.unwrap();
    assert_eq!(inputs.len(), 3, "DISTINCT must collapse duplicate pairs");
    assert!(inputs.iter().any(|input| {
        input.source_id == "merged_new" && input.supersedes.as_deref() == Some("seed_new")
    }));
    assert!(inputs.iter().any(|input| {
        input.source_id == "merged_preexisting" && input.supersedes.as_deref() == Some("seed_old")
    }));
    assert!(inputs
        .iter()
        .any(|input| input.source_id == "merged_null" && input.supersedes.is_none()));

    let merged_ids = db.eval_lifecycle_merged_ids().await.unwrap();
    assert_eq!(
        merged_ids,
        ["merged_new", "merged_null", "merged_preexisting"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
}

#[tokio::test]
async fn lifecycle_state_counts_preserve_all_four_legacy_queries() {
    let (db, _tmp) = crate::db::tests::test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute_batch(
            "CREATE TABLE concepts (id TEXT PRIMARY KEY, status TEXT NOT NULL);
             INSERT INTO concepts (id, status) VALUES
                 ('concept-active', 'active'),
                 ('concept-archived', 'archived');",
        )
        .await
        .unwrap();
    }
    // G6 Stage 3: entity_count reads the `kind='entity'` shadow page, and
    // migration 123 dropped `entities`, so the shadow IS the seed.
    db.test_seed_entity_shadow_page(TestEntity::new("entity-1", "One", "concept"))
        .await
        .unwrap();
    db.test_seed_entity_shadow_page(TestEntity::new("entity-2", "Two", "concept"))
        .await
        .unwrap();

    for fixture in [
        MemoryFixture {
            id: "memory-active",
            source_id: "memory-active",
            source: "memory",
            chunk_index: 0,
            content: "active",
            supersedes: None,
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "memory-archive",
            source_id: "memory-archive",
            source: "memory",
            chunk_index: 0,
            content: "archive",
            supersedes: None,
            supersede_mode: "archive",
        },
        MemoryFixture {
            id: "document-archive",
            source_id: "document-archive",
            source: "document",
            chunk_index: 0,
            content: "not memory",
            supersedes: None,
            supersede_mode: "archive",
        },
    ] {
        insert_memory(&db, fixture).await;
    }

    let counts = db.eval_lifecycle_state_counts().await.unwrap();
    assert_eq!(counts.memory_count, 2);
    assert_eq!(counts.archived_count, 1);
    assert_eq!(counts.entity_count, 2);
    assert_eq!(counts.concept_count, 1);
}

#[tokio::test]
async fn archive_scan_keeps_only_archived_memory_heads_and_retains_empty_content() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    for fixture in [
        MemoryFixture {
            id: "archive-head",
            source_id: "archive-head",
            source: "memory",
            chunk_index: 0,
            content: "archived body",
            supersedes: None,
            supersede_mode: "archive",
        },
        MemoryFixture {
            id: "archive-empty",
            source_id: "archive-empty",
            source: "memory",
            chunk_index: 0,
            content: "",
            supersedes: None,
            supersede_mode: "archive",
        },
        MemoryFixture {
            id: "active-head",
            source_id: "active-head",
            source: "memory",
            chunk_index: 0,
            content: "active",
            supersedes: None,
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "archive-child",
            source_id: "archive-child",
            source: "memory",
            chunk_index: 1,
            content: "child",
            supersedes: None,
            supersede_mode: "archive",
        },
        MemoryFixture {
            id: "archive-document",
            source_id: "archive-document",
            source: "document",
            chunk_index: 0,
            content: "document",
            supersedes: None,
            supersede_mode: "archive",
        },
    ] {
        insert_memory(&db, fixture).await;
    }

    let inputs = db.eval_lifecycle_archived_inputs().await.unwrap();
    assert_eq!(inputs.len(), 2);
    assert!(inputs
        .iter()
        .any(|input| input.source_id == "archive-head" && input.content == "archived body"));
    assert!(inputs
        .iter()
        .any(|input| input.source_id == "archive-empty" && input.content.is_empty()));
}

#[tokio::test]
async fn supersedes_scan_surfaces_mid_scan_step_errors() {
    let (db, _tmp) = crate::db::tests::test_db().await;

    for fixture in [
        MemoryFixture {
            id: "merged-ok",
            source_id: "merged_ok",
            source: "memory",
            chunk_index: 0,
            content: "ok",
            supersedes: Some("seed_ok"),
            supersede_mode: "hide",
        },
        MemoryFixture {
            id: "merged-poison",
            source_id: "merged_poison",
            source: "memory",
            chunk_index: 0,
            content: "poison",
            supersedes: Some("seed_poison"),
            supersede_mode: "hide",
        },
    ] {
        insert_memory(&db, fixture).await;
    }

    // Healthy control: both fixture rows are visible before the poison lands.
    let healthy = db.eval_lifecycle_supersedes_inputs().await.unwrap();
    assert_eq!(healthy.len(), 2);

    // Swap the table for a view whose `supersedes` cell raises at one row
    // (json_extract over malformed JSON) -- a real mid-scan step error. The
    // old `while let Ok(Some(..))` loop returned Ok with silently truncated
    // results here.
    {
        let conn = db.conn.lock().await;
        conn.execute("ALTER TABLE memories RENAME TO memories_base", ())
            .await
            .unwrap();
        conn.execute(
            "CREATE VIEW memories AS SELECT source_id, source, \
             CASE WHEN source_id = 'merged_poison' THEN json_extract('{', '$.x') \
                  ELSE supersedes END AS supersedes \
             FROM memories_base",
            (),
        )
        .await
        .unwrap();
    }

    let err = match db.eval_lifecycle_supersedes_inputs().await {
        Ok(rows) => panic!(
            "expected the poisoned scan to error, got {} rows",
            rows.len()
        ),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("supersedes row scan"),
        "expected the surfaced scan error, got: {err}"
    );
}
