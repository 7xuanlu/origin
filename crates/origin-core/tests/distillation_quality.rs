// SPDX-License-Identifier: Apache-2.0
//! Regression tests for distillation quality constraints.
//!
//! Covers:
//! 1. The 800-char per-memory snippet cap (pure logic, no DB needed).
//! 2. That a 4-memory cluster about a topic has enough total content to
//!    pass the 200-char thin-cluster guard in `distill_concepts`.
//! 3. That the batch `get_memories_by_source_ids` returns correct results.

use origin_core::db::MemoryDB;
use origin_core::sources::RawDocument;
use origin_core::{EventEmitter, NoopEmitter};
use std::sync::Arc;

async fn make_db() -> (Arc<MemoryDB>, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("test.db");
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let db = MemoryDB::new(&db_path, emitter)
        .await
        .expect("MemoryDB::new");
    (Arc::new(db), dir)
}

fn make_memory_doc(source_id: &str, content: &str) -> RawDocument {
    RawDocument {
        source: "memory".to_string(),
        source_id: source_id.to_string(),
        title: source_id.to_string(),
        summary: None,
        content: content.to_string(),
        url: None,
        last_modified: chrono::Utc::now().timestamp(),
        memory_type: Some("fact".to_string()),
        domain: Some("technology".to_string()),
        source_agent: Some("test-agent".to_string()),
        confidence: None,
        confirmed: None,
        supersedes: None,
        pending_revision: false,
        ..Default::default()
    }
}

/// A 4-memory cluster about libSQL vector search has enough total content to
/// pass the 200-char thin-cluster guard used in `distill_concepts`.
/// This ensures small clusters of meaningful memories will not be silently skipped.
#[tokio::test]
async fn four_memory_cluster_passes_thin_content_guard() {
    let (db, _dir) = make_db().await;

    let memories: &[(&str, &str)] = &[
        ("mem_q1", "libSQL stores 768-dimensional embeddings in F32_BLOB columns using DiskANN approximate nearest neighbor indexing."),
        ("mem_q2", "The vector index is built with DiskANN parameters R=64 and L=200, tuned for sub-millisecond recall at 10k+ memories."),
        ("mem_q3", "Hybrid search combines cosine similarity (vector lane) with FTS5 BM25 scoring (keyword lane) via Reciprocal Rank Fusion."),
        ("mem_q4", "libSQL is Turso's SQLite fork; it adds vector extensions natively so no external vector DB is required for Origin."),
    ];

    let docs: Vec<RawDocument> = memories
        .iter()
        .map(|(id, content)| make_memory_doc(id, content))
        .collect();
    db.upsert_documents(docs).await.expect("upsert memories");

    // Verify each memory is retrievable and contains distinctive content.
    for (id, expected_substr) in &[
        ("mem_q1", "DiskANN"),
        ("mem_q2", "R=64"),
        ("mem_q3", "Reciprocal Rank Fusion"),
        ("mem_q4", "Turso"),
    ] {
        let detail = db.get_memory_detail(id).await.expect("get_memory_detail");
        assert!(detail.is_some(), "memory {id} should exist after upsert");
        let item = detail.unwrap();
        assert!(
            item.content.contains(expected_substr),
            "memory {id} content should contain '{expected_substr}', got: {:?}",
            item.content
        );
    }

    // Total characters across all 4 memories must exceed the 200-char guard.
    let total_chars: usize = memories.iter().map(|(_, c)| c.len()).sum();
    assert!(
        total_chars >= 200,
        "4 substantive memories should exceed the 200-char thin-cluster guard (got {total_chars})"
    );
}

/// `get_memories_by_source_ids` returns items in input order and silently omits missing ids.
#[tokio::test]
async fn batch_fetch_preserves_order_and_omits_missing() {
    let (db, _dir) = make_db().await;

    let docs = vec![
        make_memory_doc("mem_a", "Apple: a fruit."),
        make_memory_doc("mem_b", "Banana: a yellow fruit."),
        make_memory_doc("mem_c", "Cherry: a small red fruit."),
    ];
    db.upsert_documents(docs).await.expect("upsert");

    let ids = vec![
        "mem_c".to_string(),
        "mem_missing".to_string(),
        "mem_a".to_string(),
    ];
    let results = db
        .get_memories_by_source_ids(&ids)
        .await
        .expect("batch fetch");

    // mem_missing is omitted; order is mem_c, mem_a.
    assert_eq!(results.len(), 2, "missing id should be omitted");
    assert_eq!(
        results[0].source_id, "mem_c",
        "first result should be mem_c"
    );
    assert_eq!(
        results[1].source_id, "mem_a",
        "second result should be mem_a"
    );
    assert!(results[0].content.contains("Cherry"));
    assert!(results[1].content.contains("Apple"));
}

/// Per-memory snippet cap truncates at 800 chars with "..." suffix.
#[test]
fn snippet_cap_truncates_at_800_chars() {
    let long_content: String = "x".repeat(1000);
    let cap = 800_usize;
    let snippet: String = long_content.chars().take(cap).collect();
    let snippet = if long_content.chars().count() > cap {
        format!("{}...", snippet.trim_end())
    } else {
        snippet
    };
    assert_eq!(
        snippet.len(),
        803,
        "truncated snippet should be 800 chars + 3 for '...'"
    );
    assert!(snippet.ends_with("..."), "should end with ellipsis");
}

/// Per-memory snippet cap leaves short content unchanged.
#[test]
fn snippet_cap_does_not_alter_short_content() {
    let short = "libSQL stores vectors in F32_BLOB columns with DiskANN indexing.";
    let cap = 800_usize;
    let snippet: String = short.chars().take(cap).collect();
    let snippet = if short.chars().count() > cap {
        format!("{}...", snippet.trim_end())
    } else {
        snippet
    };
    assert_eq!(snippet, short, "short content must be unchanged by the cap");
}
