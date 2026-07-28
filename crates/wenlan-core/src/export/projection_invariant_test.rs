// SPDX-License-Identifier: Apache-2.0
//! The projection-directory invariant, proven at the directory.
//!
//! `wenlan pages` reads Markdown straight off `knowledge_path_or_default()`
//! (`crates/wenlan-cli/src/commands/pages.rs`), so frontmatter and wire
//! negotiation cannot protect it. The assertions here are therefore about what
//! `read_dir` finds, not about what any function returned -- an invariant that
//! held in a return value while the file stayed on disk would be a lie about
//! the only surface that matters.
//!
//! Binding spec: `docs/plans/2026-07-27-m5-reader-manifest.md` section 5, and
//! the section 9 mutation row "leave provisional files in the legacy projection
//! directory".

use crate::db::tests::test_db;
use crate::db::MemoryDB;
use crate::export::knowledge::KnowledgeProjectionWrite;
use crate::pages::Page;
use std::path::Path;

/// `p1` supported, `p2` provisional, `p3` with no truth row at all -- the same
/// post-migration shape `db/truth_exposure_test.rs` seeds, where absence of a
/// support record is the normal case and reads as unsupported.
///
/// The cutover stays at 0 here; each test advances it itself, so the inert case
/// and the live case are seeded identically and differ in one value.
async fn db_with_truth_rows() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    for id in ["p1", "p2", "p3"] {
        db.insert_page(id, id, None, "", None, None, &[], "2026-07-27T00:00:00Z")
            .await
            .unwrap();
    }
    {
        let conn = db.conn.lock().await;
        set_truth(&conn, "p1", "supported").await;
        set_truth(&conn, "p2", "provisional").await;
    }
    (db, temp)
}

async fn set_truth(conn: &libsql::Connection, page_id: &str, status: &str) {
    conn.execute(
        "INSERT INTO page_truth_state
            (page_id,page_version,support_status,human_reviewed,updated_at)
         VALUES (?1,1,?2,0,1)",
        libsql::params![page_id, status],
    )
    .await
    .unwrap();
}

fn page(id: &str) -> Page {
    Page {
        id: id.to_string(),
        title: id.to_string(),
        summary: None,
        content: "body".to_string(),
        entity_id: None,
        space: None,
        source_memory_ids: Vec::new(),
        version: 1,
        status: "active".to_string(),
        created_at: "2026-07-27T00:00:00Z".to_string(),
        last_compiled: "2026-07-27T00:00:00Z".to_string(),
        last_modified: "2026-07-27T00:00:00Z".to_string(),
        sources_updated_count: 0,
        stale_reason: None,
        pending_rebuild: None,
        user_edited: false,
        relevance_score: 0.0,
        last_edited_by: None,
        last_edited_at: None,
        last_delta_summary: None,
        changelog: None,
        creation_kind: "distilled".to_string(),
        review_status: "confirmed".to_string(),
        workspace: None,
        citations: Vec::new(),
        kind: "concept".to_string(),
    }
}

/// Exactly what `wenlan pages` would list: top-level `*.md`, by filename stem.
fn readable_pages(dir: &Path) -> Vec<String> {
    let mut stems: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|x| x.to_str()) == Some("md"))
        .filter_map(|path| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(str::to_string)
        })
        .collect();
    stems.sort();
    stems
}

fn project_all(db: &MemoryDB, root: &Path) -> KnowledgeProjectionWrite {
    let projection = KnowledgeProjectionWrite::new(root.to_path_buf(), db);
    for id in ["p1", "p2", "p3"] {
        projection.write_page(&page(id)).unwrap();
    }
    assert_eq!(
        readable_pages(root),
        ["p1", "p2", "p3"],
        "the seed itself must put all three on disk, or the eviction proves nothing"
    );
    projection
}

/// The PR-B production configuration. The pass runs, and the directory keeps
/// every page -- including the provisional one and the one with no truth row at
/// all. If this ever goes RED, the cutover has happened without PR-C's ceremony.
#[tokio::test]
async fn at_generation_zero_the_projection_keeps_every_page() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());

    let removed = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(
        readable_pages(root.path()),
        ["p1", "p2", "p3"],
        "PR-B ships this pass inert"
    );
    assert_eq!(removed, 0);
}

/// After the cutover the directory is the boundary, so the provisional page and
/// the page with no truth row both leave it. `p3` is the post-migration normal
/// case: absence of a support record is not evidence of support.
#[tokio::test]
async fn after_the_cutover_the_projection_holds_supported_pages_only() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    let removed = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    // The directory first: it is the invariant. The count is a receipt for it,
    // so it is asserted second and never gets to mask a wrong directory.
    assert_eq!(
        readable_pages(root.path()),
        ["p1"],
        "a provisional page left on disk is readable by `wenlan pages`, which \
         cannot negotiate -- the file has to be gone, not merely unlisted"
    );
    assert_eq!(removed, 2);
}

/// The pass is idempotent and does not eat what it may not eat: running it
/// again after the eviction leaves the supported page exactly where it is.
#[tokio::test]
async fn a_second_pass_removes_nothing_more() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();
    let removed = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(removed, 0);
    assert_eq!(readable_pages(root.path()), ["p1"]);
}
