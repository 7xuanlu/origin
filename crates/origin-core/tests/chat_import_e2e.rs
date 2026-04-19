// SPDX-License-Identifier: Apache-2.0
//! End-to-end integration test: build a synthetic Claude export ZIP, parse it,
//! ingest into a fresh MemoryDB, verify storage, and confirm dedup on reimport.

use std::io::Write;
use std::sync::Arc;

use origin_core::chat_import::bulk_ingest::bulk_import_conversations;
use origin_core::chat_import::dispatch_parse;
use origin_core::chat_import::types::Vendor;
use origin_core::db::MemoryDB;
use origin_core::{EventEmitter, NoopEmitter};
use serde_json::json;

/// Build an in-memory ZIP archive from a list of `(filename, data)` entries.
fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut buf = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut buf);
        let mut zip = zip::ZipWriter::new(cursor);
        let options: zip::write::FileOptions<()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, data) in entries {
            zip.start_file(*name, options).unwrap();
            zip.write_all(data).unwrap();
        }
        zip.finish().unwrap();
    }
    buf
}

/// Minimal Claude conversations.json with one conversation containing one
/// human + one assistant message (which pair into a single ParsedMessage).
const CONVERSATIONS_JSON: &str = r#"[
  {
    "uuid": "conv-e2e-1",
    "name": "E2E Test Conversation",
    "summary": "Testing end-to-end import",
    "created_at": "2026-04-10T10:00:00.000Z",
    "updated_at": "2026-04-10T10:00:05.000Z",
    "account": {"uuid": "acc-e2e"},
    "chat_messages": [
      {
        "uuid": "msg-h1",
        "text": "hello",
        "content": [{"type": "text", "text": "hello"}],
        "sender": "human",
        "created_at": "2026-04-10T10:00:00.000Z",
        "updated_at": "2026-04-10T10:00:00.000Z",
        "attachments": [],
        "files": [],
        "parent_message_uuid": "00000000-0000-4000-8000-000000000000"
      },
      {
        "uuid": "msg-a1",
        "text": "world",
        "content": [{"type": "text", "text": "world"}],
        "sender": "assistant",
        "created_at": "2026-04-10T10:00:05.000Z",
        "updated_at": "2026-04-10T10:00:05.000Z",
        "attachments": [],
        "files": [],
        "parent_message_uuid": "msg-h1"
      }
    ]
  }
]"#;

#[tokio::test]
async fn e2e_claude_import_stores_memories_and_skips_on_reimport() {
    // ---- Step 1: Build a synthetic Claude export ZIP ----
    let zip_bytes = make_zip(&[
        ("users.json", b"[]"),
        ("conversations.json", CONVERSATIONS_JSON.as_bytes()),
    ]);

    // ---- Step 2: Parse the ZIP via dispatch_parse ----
    let batch = dispatch_parse(&zip_bytes).expect("dispatch_parse should succeed");
    assert_eq!(batch.vendor, Vendor::Claude);
    assert_eq!(batch.conversations.len(), 1, "expected 1 conversation");

    let conv = &batch.conversations[0];
    assert_eq!(conv.external_id, "conv-e2e-1");
    assert_eq!(conv.vendor, Vendor::Claude);
    assert_eq!(conv.title.as_deref(), Some("E2E Test Conversation"));
    // Human + assistant are paired into 1 message.
    assert_eq!(conv.messages.len(), 1, "expected 1 paired message");
    assert_eq!(conv.messages[0].content, "User: hello\nAssistant: world");

    // ---- Step 3: Ingest into a fresh MemoryDB ----
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let db = MemoryDB::new(tmp.path(), emitter.clone())
        .await
        .expect("MemoryDB::new should succeed");
    let db_arc = Arc::new(db);

    let result = bulk_import_conversations(
        db_arc.clone(),
        &batch.conversations,
        emitter.clone(),
        "e2e_test_import",
    )
    .await
    .expect("first import should succeed");

    assert_eq!(result.conversations_ingested, 1);
    assert_eq!(result.conversations_skipped_existing, 0);
    assert_eq!(result.memories_stored, 1);

    // ---- Step 4: Verify dedup — reimport the same batch ----
    let result2 = bulk_import_conversations(
        db_arc.clone(),
        &batch.conversations,
        emitter.clone(),
        "e2e_test_import",
    )
    .await
    .expect("reimport should succeed");

    assert_eq!(
        result2.conversations_ingested, 0,
        "reimport should ingest 0 conversations"
    );
    assert_eq!(
        result2.conversations_skipped_existing, 1,
        "reimport should skip 1 existing conversation"
    );
    assert_eq!(
        result2.memories_stored, 0,
        "reimport should store 0 new memories"
    );

    // ---- Step 5: Verify via check_existing_import_source_ids ----
    let expected_source_id = "import_claude_conv-e2e-1".to_string();
    let existing = db_arc
        .check_existing_import_source_ids(std::slice::from_ref(&expected_source_id))
        .await
        .expect("check_existing_import_source_ids should succeed");

    assert!(
        existing.contains(&expected_source_id),
        "source_id '{}' should exist in DB after import",
        expected_source_id
    );
}

// ---- ChatGPT end-to-end test ----

/// Minimal ChatGPT conversations.json with 2 conversations, each with one
/// user + assistant pair.
fn chatgpt_conversations_json() -> String {
    json!([
        {
            "id": "gpt-conv-e2e-1",
            "title": "First GPT Conversation",
            "create_time": 1_749_800_000.0_f64,
            "current_node": "gpt1-asst",
            "is_do_not_remember": false,
            "mapping": {
                "gpt1-user": {
                    "id": "gpt1-user",
                    "parent": null,
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": ["What is Rust?"],
                            "text": null
                        },
                        "create_time": 1_749_800_001.0_f64
                    }
                },
                "gpt1-asst": {
                    "id": "gpt1-asst",
                    "parent": "gpt1-user",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Rust is a systems programming language focused on safety."],
                            "text": null
                        },
                        "create_time": 1_749_800_002.0_f64
                    }
                }
            }
        },
        {
            "id": "gpt-conv-e2e-2",
            "title": "Second GPT Conversation",
            "create_time": 1_749_810_000.0_f64,
            "current_node": "gpt2-asst",
            "is_do_not_remember": false,
            "mapping": {
                "gpt2-user": {
                    "id": "gpt2-user",
                    "parent": null,
                    "message": {
                        "author": {"role": "user"},
                        "content": {
                            "content_type": "text",
                            "parts": ["What is memory safety?"],
                            "text": null
                        },
                        "create_time": 1_749_810_001.0_f64
                    }
                },
                "gpt2-asst": {
                    "id": "gpt2-asst",
                    "parent": "gpt2-user",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Memory safety means preventing invalid memory access."],
                            "text": null
                        },
                        "create_time": 1_749_810_002.0_f64
                    }
                }
            }
        }
    ])
    .to_string()
}

#[tokio::test]
async fn chatgpt_end_to_end_imports_and_dedupes() {
    // ---- Step 1: Build a synthetic ChatGPT export ZIP ----
    let chatgpt_json = chatgpt_conversations_json();
    let zip_bytes = make_zip(&[
        ("conversations.json", chatgpt_json.as_bytes()),
        ("message_feedback.json", b"[]"),
    ]);

    // ---- Step 2: Parse via dispatch_parse ----
    let batch = dispatch_parse(&zip_bytes).expect("dispatch_parse should succeed for ChatGPT zip");
    assert_eq!(
        batch.vendor,
        Vendor::ChatGpt,
        "dispatcher should identify ChatGPT vendor"
    );
    assert_eq!(
        batch.conversations.len(),
        2,
        "expected 2 conversations from ChatGPT zip"
    );

    // Each conversation should have exactly 1 paired message (user+assistant).
    for conv in &batch.conversations {
        assert_eq!(
            conv.messages.len(),
            1,
            "conversation '{}' should have 1 paired message",
            conv.external_id
        );
        assert_eq!(conv.vendor, Vendor::ChatGpt);
        let source_id = Vendor::ChatGpt.build_source_id(&conv.external_id);
        assert!(
            source_id.starts_with("import_chatgpt_"),
            "source_id '{}' should start with import_chatgpt_",
            source_id
        );
    }

    // ---- Step 3: Ingest into a fresh MemoryDB ----
    let tmp = tempfile::tempdir().expect("failed to create temp dir");
    let emitter: Arc<dyn EventEmitter> = Arc::new(NoopEmitter);
    let db = MemoryDB::new(tmp.path(), emitter.clone())
        .await
        .expect("MemoryDB::new should succeed");
    let db_arc = Arc::new(db);

    let result = bulk_import_conversations(
        db_arc.clone(),
        &batch.conversations,
        emitter.clone(),
        "e2e_chatgpt_import",
    )
    .await
    .expect("first ChatGPT import should succeed");

    assert_eq!(
        result.conversations_ingested, 2,
        "first import should ingest both conversations"
    );
    assert_eq!(
        result.conversations_skipped_existing, 0,
        "first import should skip nothing"
    );
    assert!(
        result.memories_stored > 0,
        "first import should store at least one memory"
    );

    // ---- Step 4: Verify source_id prefix on stored memories ----
    let candidate_ids: Vec<String> = batch
        .conversations
        .iter()
        .map(|c| Vendor::ChatGpt.build_source_id(&c.external_id))
        .collect();
    let existing = db_arc
        .check_existing_import_source_ids(&candidate_ids)
        .await
        .expect("check_existing_import_source_ids should succeed");

    for sid in &candidate_ids {
        assert!(
            existing.contains(sid),
            "source_id '{}' should exist in DB after import",
            sid
        );
        assert!(
            sid.starts_with("import_chatgpt_"),
            "stored source_id '{}' should have import_chatgpt_ prefix",
            sid
        );
    }

    // ---- Step 5: Re-import same ZIP — all should be deduped ----
    let result2 = bulk_import_conversations(
        db_arc.clone(),
        &batch.conversations,
        emitter.clone(),
        "e2e_chatgpt_import",
    )
    .await
    .expect("reimport should succeed");

    assert_eq!(
        result2.conversations_ingested, 0,
        "reimport should ingest 0 conversations"
    );
    assert_eq!(
        result2.conversations_skipped_existing, 2,
        "reimport should skip both existing conversations"
    );
    assert_eq!(
        result2.memories_stored, 0,
        "reimport should store 0 new memories"
    );
}

/// Smoke test against the user's real ChatGPT export zip (~155 MB).
///
/// Ignored by default — run with:
///   cargo test -p origin-core --test chat_import_e2e --release \
///       -- --ignored chatgpt_real_zip_smoke --nocapture
///
/// Set `ORIGIN_CHATGPT_ZIP=/path/to/export.zip` to override the path.
/// Verifies: (a) dispatch picks the ChatGPT parser, (b) parsing completes
/// without panic, (c) conversation count is non-zero and in a sane range.
#[test]
#[ignore]
fn chatgpt_real_zip_smoke() {
    let default_path = "/Users/lucian/Downloads/b50fdbe7b5a67df02e91af8f5d75a53ea4796a12b5c7b726b8b2562a233c2905-2026-04-11-20-28-58-326de17fc3cb48bb91504f88fe0ded37.zip";
    let path = std::env::var("ORIGIN_CHATGPT_ZIP").unwrap_or_else(|_| default_path.to_string());

    if !std::path::Path::new(&path).exists() {
        eprintln!("zip not found at {path}; skipping");
        return;
    }

    let bytes = std::fs::read(&path).expect("reading zip should succeed");
    eprintln!("read {} bytes from {}", bytes.len(), path);

    let batch = dispatch_parse(&bytes).expect("dispatch_parse should succeed");
    assert_eq!(batch.vendor, Vendor::ChatGpt, "vendor should be ChatGPT");
    eprintln!("parsed {} conversations", batch.conversations.len());

    assert!(
        !batch.conversations.is_empty(),
        "real export should contain at least one conversation"
    );
    // Per profile of this zip: expected ~77 conversations. Give a wide range.
    assert!(
        batch.conversations.len() >= 10,
        "expected at least 10 conversations in real export, got {}",
        batch.conversations.len()
    );

    let total_messages: usize = batch.conversations.iter().map(|c| c.messages.len()).sum();
    eprintln!("parsed {total_messages} paired messages total");
    assert!(
        total_messages > 0,
        "parsed conversations should yield at least some paired messages"
    );

    // Spot-check: every conversation has the right vendor tag and a non-empty external_id.
    for conv in &batch.conversations {
        assert_eq!(conv.vendor, Vendor::ChatGpt);
        assert!(!conv.external_id.is_empty());
    }
}
