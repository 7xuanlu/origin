// SPDX-License-Identifier: Apache-2.0
//! §6 end-to-end integration test for folder / multi-format ingest.
//!
//! A fixture folder — markdown with wikilinks (in a subdirectory), a plain-text
//! report, a tiny real text PDF, and a no-extractable-text PDF (empty content
//! stream, standing in for the scanned/image-only case) — is ingested into a temp
//! `MemoryDB` and the full L1 pipeline is asserted:
//!
//!   1. `scan_directory` walks recursively and extension-filters (md/txt/pdf).
//!   2. Each changed file is enqueued, then drained through the ONE canonical
//!      `run_document_enrichment` route (skew discipline, spec §6).
//!   3. Every ingested document: correct chunk count, an embedding on every
//!      chunk, and exactly one `creation_kind='source'` page citing its chunks.
//!   4. The image-only PDF is Skipped — no chunks, no page — and counted, not
//!      ingested (§5 "no OCR in v1").
//!   5. Deletion propagation: a file that vanishes under a LIVE root has its
//!      chunks reaped; siblings are retained.
//!   6. Root-unreachable guard (§4/§5): an unmounted root deletes ZERO chunks;
//!      the source is reported unavailable, everything already ingested survives.
//!   7. A buried fixture sentence is findable via search after ingest.
//!
//! ## Why the sync diff is reconstructed here
//!
//! The production sync routine (`sync_directory_source`) lives in `wenlan-server`
//! and is `pub(crate)`, so a `wenlan-core` integration test cannot call it —
//! `wenlan-core` must not depend on `wenlan-server`. This test therefore drives
//! the REAL core pipeline (`scan_directory`, `file_to_documents` via
//! `run_document_enrichment`, `document_source_id`, `upsert_documents` embedding,
//! `delete_by_source_id`) and reconstructs only the thin sync-diff orchestration
//! (root-liveness guard + scan/mtime/hash diff + vanished-file reap) that the
//! server layers on top. The reconstructed `directory_sync` mirrors
//! `sync_directory_source`'s deletion + root-guard semantics; the behaviors under
//! test (reap-under-live-root, zero-delete-on-unmount, one canonical enrichment
//! route) are the production ones.
//!
//! The LLM is `None`: `run_document_enrichment` writes a deterministic stub
//! SOURCE page, publishes the sync receipt, and parks model-derived work at
//! `waiting_for_provider`. This keeps the test deterministic while still
//! exercising all-chunk embedding (embedding happens in `upsert_documents`,
//! before any LLM call) and the source-page contract.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sha2::{Digest, Sha256};
use wenlan_core::db::MemoryDB;
use wenlan_core::document_enrichment::{run_document_enrichment, DocumentEnrichmentOutcome};
use wenlan_core::events::NoopEmitter;
use wenlan_core::prompts::PromptRegistry;
use wenlan_core::read_scope::ReadScope;
use wenlan_core::sources::directory::{document_source_id, scan_directory};

const SOURCE_ID: &str = "directory-fixtures";

// ── helpers ──────────────────────────────────────────────────────────────────

fn fixtures_folder() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/folder")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn mtime_ns(path: &Path) -> i64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0)
}

/// Recursively copy a directory tree (files + subdirs) into `dst`.
fn copy_dir_all(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir_all(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}

/// The canonical document `source_id` a file's chunks live under, computed the
/// same way the write side (`run_document_enrichment`) and the delete side
/// (`directory_sync`) both do — provenance relative to `knowledge`.
fn doc_id(root: &Path, file: &Path) -> String {
    document_source_id(SOURCE_ID, file, Some(root))
}

/// Outcome of one reconstructed sync pass over a Directory root.
#[derive(Debug, Default)]
struct SyncOutcome {
    /// False when the root is missing/unreadable (unmounted) — the diff is
    /// skipped and zero deletions occur. Mirrors the server's `unavailable` mark.
    root_live: bool,
    files_found: usize,
    /// Files newly enqueued for enrichment this pass (changed / new content).
    enqueued: usize,
    /// Files skipped because their mtime+hash matched the tracked sync state.
    skipped: usize,
    /// Vanished-under-live-root files whose chunks were reaped.
    deleted: usize,
    errors: usize,
}

/// Reconstruction of the core of `wenlan-server`'s `sync_directory_source`:
/// root-liveness guard, scan + per-file mtime/hash skip, enqueue of changed
/// files, and deletion propagation for tracked files that vanished under a live
/// root. `source_sync_state` is deliberately NOT written here: production
/// writes it in `run_document_enrichment` after searchable preparation, whether
/// the queue becomes terminal or waits for a provider (§4). The drain — not the
/// sync loop — therefore populates the tracked set the skip and deletion diffs
/// read. Rename optimization is intentionally omitted — it is not part of the
/// §6 assertions.
async fn directory_sync(db: &MemoryDB, root: &Path) -> SyncOutcome {
    // Root-guard (§4/§5): a missing/unreadable root means "source unavailable",
    // NOT "every file deleted". Diff nothing, delete zero rows.
    let root_live = match std::fs::metadata(root) {
        Ok(m) if m.is_dir() => std::fs::read_dir(root).is_ok(),
        Ok(m) => m.is_file(),
        Err(_) => false,
    };
    if !root_live {
        return SyncOutcome::default(); // root_live = false, all counts 0
    }

    let files = scan_directory(root);
    let scanned: HashSet<String> = files
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    let mut out = SyncOutcome {
        root_live: true,
        files_found: files.len(),
        ..SyncOutcome::default()
    };

    // Deletion propagation: any tracked file NOT in the current scan vanished
    // under the live root → reap its chunks, sync state, and any pending work.
    if let Ok(tracked) = db.list_sync_state_paths(SOURCE_ID).await {
        for tracked_path in tracked {
            if scanned.contains(&tracked_path) {
                continue;
            }
            let gone = doc_id(root, Path::new(&tracked_path));
            db.delete_by_source_id("memory", &gone).await.unwrap();
            db.delete_sync_state(SOURCE_ID, &tracked_path)
                .await
                .unwrap();
            db.dequeue_document(SOURCE_ID, &tracked_path).await.unwrap();
            out.deleted += 1;
        }
    }

    for file in &files {
        let key = file.to_string_lossy().to_string();
        let mtime = mtime_ns(file);
        let existing = db.get_sync_state(SOURCE_ID, &key).await.ok().flatten();
        if let Some(ref ss) = existing {
            if ss.mtime_ns == mtime {
                out.skipped += 1;
                continue;
            }
        }
        let bytes = match std::fs::read(file) {
            Ok(b) => b,
            Err(_) => {
                out.errors += 1;
                continue;
            }
        };
        let hash = sha256_hex(&bytes);
        if let Some(ref ss) = existing {
            if ss.content_hash == hash {
                db.upsert_sync_state(SOURCE_ID, &key, mtime, &hash)
                    .await
                    .unwrap();
                out.skipped += 1;
                continue;
            }
        }
        db.enqueue_document(SOURCE_ID, &key, Some(&hash))
            .await
            .unwrap();
        out.enqueued += 1;
    }
    out
}

/// Drain the enrichment queue through the ONE canonical route, returning the
/// per-file outcome keyed by file path. `None` LLM → deterministic stub source
/// pages; embedding still happens (in `upsert_documents`, pre-LLM).
async fn drain_queue(
    db: &MemoryDB,
    root: &Path,
    prompts: &PromptRegistry,
    expected_max: usize,
) -> Vec<(String, DocumentEnrichmentOutcome)> {
    let mut out = Vec::new();
    while let Some(entry) = db.claim_next_pending_for_provider(false).await.unwrap() {
        assert!(
            out.len() < expected_max,
            "queue drain claimed more than the expected {expected_max} documents; \
             a terminal or provider-waiting row was likely reclaimed"
        );
        let file_path = entry.file_path.clone();
        let outcome = run_document_enrichment(db, &entry, Some(root), None, prompts).await;
        // With no LLM the route never pauses; guard against a stuck loop anyway.
        assert!(
            !outcome.paused,
            "None-LLM enrichment must not pause: {file_path}"
        );
        out.push((file_path, outcome));
    }
    out
}

#[tokio::test]
async fn no_provider_drain_stops_after_searchable_preparation() {
    let db_dir = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(db_dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open temp MemoryDB");
    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("vault");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("note.md");
    std::fs::write(&path, include_str!("fixtures/folder/notes/linked.md")).unwrap();

    let synced = directory_sync(&db, &root).await;
    assert_eq!(synced.enqueued, 1);
    let outcomes = drain_queue(&db, &root, &PromptRegistry::default(), 1).await;

    assert_eq!(outcomes.len(), 1);
    let queue = db
        .get_queue_entry(SOURCE_ID, &path.to_string_lossy())
        .await
        .unwrap()
        .expect("document queue row");
    assert_eq!(queue.status, "waiting_for_provider");
}

async fn chunk_count(db: &MemoryDB, doc_source_id: &str) -> usize {
    db.get_memories_by_source_id("memory", doc_source_id)
        .await
        .unwrap()
        .len()
}

// ── the end-to-end test ──────────────────────────────────────────────────────

#[tokio::test]
async fn folder_ingest_full_pipeline_e2e() {
    // ── setup: temp DB + working copy of the fixture folder ──────────────────
    let db_dir = tempfile::tempdir().unwrap();
    let db = MemoryDB::new(db_dir.path(), Arc::new(NoopEmitter))
        .await
        .expect("open temp MemoryDB");

    let work = tempfile::tempdir().unwrap();
    let root = work.path().join("vault");
    copy_dir_all(&fixtures_folder(), &root);

    let md_path = root.join("notes/linked.md");
    let txt_path = root.join("report.txt");
    let pdf_path = root.join("paper.pdf");
    let image_pdf_path = root.join("scan.pdf");
    for p in [&md_path, &txt_path, &pdf_path, &image_pdf_path] {
        assert!(p.exists(), "fixture must be present: {}", p.display());
    }

    let md_doc = doc_id(&root, &md_path);
    let txt_doc = doc_id(&root, &txt_path);
    let pdf_doc = doc_id(&root, &pdf_path);
    let image_doc = doc_id(&root, &image_pdf_path);

    let prompts = PromptRegistry::default();

    // ── Phase A: first sync + enrichment drain ───────────────────────────────
    let synced = directory_sync(&db, &root).await;
    assert!(synced.root_live, "a present root is live");
    assert_eq!(
        synced.files_found, 4,
        "scan finds md + txt + 2 pdf, recursively"
    );
    assert_eq!(synced.enqueued, 4, "all four files enqueued on first sync");
    assert_eq!(synced.skipped, 0);
    assert_eq!(synced.deleted, 0);
    assert_eq!(synced.errors, 0);

    let outcomes = drain_queue(&db, &root, &prompts, 4).await;
    assert_eq!(outcomes.len(), 4, "every enqueued file is processed once");

    // Per-file chunk counts.
    let md_chunks = chunk_count(&db, &md_doc).await;
    let txt_chunks = chunk_count(&db, &txt_doc).await;
    let pdf_chunks = chunk_count(&db, &pdf_doc).await;
    let image_chunks = chunk_count(&db, &image_doc).await;
    assert!(
        md_chunks >= 1,
        "markdown yields at least one chunk, got {md_chunks}"
    );
    assert!(
        txt_chunks >= 3,
        "the large report chunks into >= 3, got {txt_chunks}"
    );
    assert!(
        pdf_chunks >= 1,
        "the tiny text PDF yields at least one chunk, got {pdf_chunks}"
    );
    assert_eq!(image_chunks, 0, "image-only PDF must not ingest any chunk");

    // Embeddings present on every chunk of every ingested document.
    for (label, doc) in [("md", &md_doc), ("txt", &txt_doc), ("pdf", &pdf_doc)] {
        let missing = db.count_unembedded_chunks("memory", doc).await.unwrap();
        assert_eq!(missing, 0, "{label} document has an unembedded chunk");
    }

    // Skip / error accounting matches §5: exactly the image-only PDF is skipped
    // (no page, no chunks); the other three are ingested (page + chunks). The
    // image-only skip surfaces at the ENRICHMENT tier (parse is deferred to the
    // queue), so it is asserted on the enrichment outcome, not the sync counts.
    let page_of = |name: &str| -> String {
        outcomes
            .iter()
            .find(|(fp, _)| fp.ends_with(name))
            .map(|(_, o)| o.page_id.clone())
            .unwrap_or_default()
    };
    let md_page = page_of("linked.md");
    let txt_page = page_of("report.txt");
    let pdf_page = page_of("paper.pdf");
    let image_page = page_of("scan.pdf");
    assert!(
        image_page.is_empty(),
        "skipped image-only PDF gets NO source page"
    );
    for pid in [&md_page, &txt_page, &pdf_page] {
        assert!(
            !pid.is_empty(),
            "each ingested document gets a source page id"
        );
    }

    // Exactly one SOURCE page per ingested document, each citing its own chunks.
    let mut seen_pages = HashSet::new();
    for (doc, pid, n) in [
        (&md_doc, &md_page, md_chunks),
        (&txt_doc, &txt_page, txt_chunks),
        (&pdf_doc, &pdf_page, pdf_chunks),
    ] {
        assert!(
            seen_pages.insert(pid.clone()),
            "each document has a distinct source page"
        );
        let page = db
            .get_page(pid)
            .await
            .unwrap()
            .unwrap_or_else(|| panic!("source page for {doc}"));
        assert_eq!(page.creation_kind, "source", "auto page is a SOURCE page");
        assert_eq!(
            page.source_memory_ids.len(),
            n,
            "source page cites every chunk (chunk-granular provenance)"
        );
    }
    // Globally: three active pages, one per ingested document, none for the
    // skipped image-only PDF.
    assert_eq!(
        db.count_active_pages().await.unwrap(),
        3,
        "exactly one SOURCE page per successfully-ingested document"
    );

    // Deterministic preparation is searchable, but model-derived work remains
    // parked until a provider is explicitly available. The image-only skip is
    // terminal because it has no model-derived work to perform.
    for (fp, expected_status) in [
        (
            md_path.to_string_lossy().to_string(),
            "waiting_for_provider",
        ),
        (
            txt_path.to_string_lossy().to_string(),
            "waiting_for_provider",
        ),
        (
            pdf_path.to_string_lossy().to_string(),
            "waiting_for_provider",
        ),
        (image_pdf_path.to_string_lossy().to_string(), "done"),
    ] {
        let q = db.get_queue_entry(SOURCE_ID, &fp).await.unwrap().unwrap();
        assert_eq!(q.status, expected_status, "unexpected queue state for {fp}");
    }

    // Retrieval: each ingested document is findable by a marker unique to it,
    // proving its chunks were embedded and indexed.
    assert_hits(&db, "Zorblatt marker", &md_doc, "markdown marker").await;
    assert_hits(
        &db,
        "Antikythera eclipses bronze gears",
        &txt_doc,
        "txt marker",
    )
    .await;
    assert_hits(&db, "Wenlanborg", &pdf_doc, "pdf marker").await;
    // A sentence buried deep in the report is retrievable, and the hit is the
    // Section-Eight chunk ITSELF ("millennia" appears only in Section Eight),
    // not an earlier section that shares the bronze/shipwreck vocabulary.
    let buried = db
        .search_memory(
            "separated by two millennia engaged in the same bargain",
            30,
            None,
            &ReadScope::Global,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        buried
            .iter()
            .any(|r| r.source_id == txt_doc && r.content.contains("millennia")),
        "the buried Section-Eight chunk itself must surface"
    );

    // ── Phase B: idempotent re-sync — nothing changed → everything skipped ────
    let resynced = directory_sync(&db, &root).await;
    assert!(resynced.root_live);
    assert_eq!(resynced.files_found, 4);
    assert_eq!(resynced.enqueued, 0, "unchanged files are not re-enqueued");
    assert_eq!(
        resynced.skipped, 4,
        "all four match tracked mtime → skipped"
    );
    assert_eq!(resynced.deleted, 0, "no deletions when nothing vanished");
    // No new queue work.
    let drained_again = drain_queue(&db, &root, &prompts, 0).await;
    assert!(
        drained_again.is_empty(),
        "a no-op sync enqueues nothing to drain"
    );

    // ── Phase B2: modify a file → re-enqueued, re-enriched, chunks replaced ──
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&txt_path)
            .unwrap();
        writeln!(
            f,
            "\nSection Nine. Addendum: the veridian-substrate-9271 clause \
             records that revised reports must re-enrich."
        )
        .unwrap();
    }
    let modified = directory_sync(&db, &root).await;
    assert_eq!(modified.enqueued, 1, "the modified file is re-enqueued");
    assert_eq!(modified.skipped, 3, "unchanged siblings still skip");
    assert_eq!(modified.deleted, 0);
    let redrained = drain_queue(&db, &root, &prompts, 1).await;
    assert_eq!(redrained.len(), 1, "only the modified file re-enriches");
    let txt_chunks = chunk_count(&db, &txt_doc).await;
    assert!(
        txt_chunks >= 3,
        "re-enriched report still chunks fully, got {txt_chunks}"
    );
    assert_hits(
        &db,
        "veridian-substrate-9271 clause revised reports",
        &txt_doc,
        "modified content is searchable after re-enrichment",
    )
    .await;
    // Still exactly one SOURCE page per document (deterministic page id reused).
    assert_eq!(
        db.count_active_pages().await.unwrap(),
        3,
        "re-enrichment must not duplicate the document's SOURCE page"
    );

    // ── Phase C: delete a file under the LIVE root → its chunks are reaped ────
    std::fs::remove_file(&txt_path).unwrap();
    let after_delete = directory_sync(&db, &root).await;
    assert!(
        after_delete.root_live,
        "root is still live after deleting one file"
    );
    assert_eq!(
        after_delete.files_found, 3,
        "three files remain under the live root"
    );
    assert_eq!(after_delete.deleted, 1, "the vanished file is reaped");
    assert_eq!(after_delete.enqueued, 0);

    assert_eq!(
        chunk_count(&db, &txt_doc).await,
        0,
        "deleted file's chunks are gone"
    );
    assert_eq!(
        chunk_count(&db, &md_doc).await,
        md_chunks,
        "sibling md chunks retained"
    );
    assert_eq!(
        chunk_count(&db, &pdf_doc).await,
        pdf_chunks,
        "sibling pdf chunks retained"
    );
    assert!(
        db.get_sync_state(SOURCE_ID, &txt_path.to_string_lossy())
            .await
            .unwrap()
            .is_none(),
        "the reaped file's sync state is cleared"
    );
    assert!(
        db.get_sync_state(SOURCE_ID, &md_path.to_string_lossy())
            .await
            .unwrap()
            .is_some(),
        "the surviving file's sync state remains"
    );
    // The deleted document is no longer retrievable; siblings still are.
    let txt_gone = db
        .search_memory(
            "Antikythera eclipses bronze gears",
            30,
            None,
            &ReadScope::Global,
            None,
            None,
            None,
            None,
        )
        .await
        .unwrap();
    assert!(
        !txt_gone.iter().any(|r| r.source_id == txt_doc),
        "the reaped document must not resurface in search"
    );
    assert_hits(
        &db,
        "Wenlanborg",
        &pdf_doc,
        "pdf still searchable after a sibling delete",
    )
    .await;

    // ── Phase D: unmount the root → RETAIN chunks, delete nothing ────────────
    let md_before = chunk_count(&db, &md_doc).await;
    let pdf_before = chunk_count(&db, &pdf_doc).await;
    std::fs::remove_dir_all(&root).unwrap(); // root vanishes: unmounted / renamed

    let unmounted = directory_sync(&db, &root).await;
    assert!(
        !unmounted.root_live,
        "a missing root is reported unavailable"
    );
    assert_eq!(
        unmounted.files_found, 0,
        "no files scanned under a gone root"
    );
    assert_eq!(
        unmounted.deleted, 0,
        "root-gone != file-gone: ZERO deletions"
    );

    assert_eq!(
        chunk_count(&db, &md_doc).await,
        md_before,
        "md chunks survive an unmounted root"
    );
    assert_eq!(
        chunk_count(&db, &pdf_doc).await,
        pdf_before,
        "pdf chunks survive an unmounted root"
    );
    assert!(
        db.get_sync_state(SOURCE_ID, &md_path.to_string_lossy())
            .await
            .unwrap()
            .is_some(),
        "tracked sync state is untouched by an unmounted-root sync"
    );
    // Still retrievable — the corpus was not silently wiped by a gone root.
    assert_hits(
        &db,
        "Wenlanborg",
        &pdf_doc,
        "pdf searchable after root unmount",
    )
    .await;
}

/// Assert `query` returns at least one result whose `source_id` is `doc`.
async fn assert_hits(db: &MemoryDB, query: &str, doc: &str, label: &str) {
    let results = db
        .search_memory(query, 30, None, &ReadScope::Global, None, None, None, None)
        .await
        .unwrap();
    assert!(
        results.iter().any(|r| r.source_id == doc),
        "{label}: query {query:?} should retrieve document {doc}"
    );
}

// ── one-time PDF fixture generator (run with `--ignored`) ────────────────────
//
// The two PDF fixtures under tests/fixtures/folder/ are byte-valid PDFs produced
// by lopdf (the same writer pdf-extract reads), mirroring the convention in
// `sources::directory`'s unit tests. `paper.pdf` draws real text via a `Tj`
// operator; `scan.pdf` has an empty content stream (no text operators) — the
// image-only / no-extractable-text case (no OCR in v1). Regenerate with:
//   cargo test -p wenlan-core --test folder_ingest_e2e -- --ignored generate_pdf_fixtures

#[test]
#[ignore = "one-time fixture generator; run explicitly to (re)write the PDF fixtures"]
fn generate_pdf_fixtures() {
    let dir = fixtures_folder();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("paper.pdf"),
        build_pdf(Some(
            "Wenlanborg is the tiny PDF marker and this paper has enough useful words for folder ingestion.",
        )),
    )
    .unwrap();
    std::fs::write(dir.join("scan.pdf"), build_pdf(None)).unwrap();
}

/// A valid one-page PDF whose content stream draws `text` via `Tj`. `None` gives
/// an empty content stream (no text operators, no embedded image) — it
/// exercises the same no-extractable-text skip a scanned/image-only PDF hits.
fn build_pdf(text: Option<&str>) -> Vec<u8> {
    use lopdf::content::{Content, Operation};
    use lopdf::{dictionary, Document, Object, Stream};

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();

    let font_id = doc.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let resources_id = doc.add_object(dictionary! {
        "Font" => dictionary! { "F1" => font_id },
    });

    let operations = match text {
        Some(t) => vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![20.into(), 100.into()]),
            Operation::new("Tj", vec![Object::string_literal(t)]),
            Operation::new("ET", vec![]),
        ],
        None => Vec::new(),
    };
    let content = Content { operations };
    let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));

    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => resources_id,
        "MediaBox" => vec![0.into(), 0.into(), 300.into(), 144.into()],
    });

    let pages = dictionary! {
        "Type" => "Pages",
        "Kids" => vec![page_id.into()],
        "Count" => 1,
    };
    doc.objects.insert(pages_id, Object::Dictionary(pages));

    let catalog_id = doc.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    doc.trailer.set("Root", catalog_id);

    let mut buf = Vec::new();
    doc.save_to(&mut buf).unwrap();
    buf
}
