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

use crate::db::test_support::TestDbSession;
use crate::db::tests::test_db;
use crate::db::MemoryDB;
use crate::export::knowledge::KnowledgeProjectionWrite;
use crate::pages::Page;
use std::path::Path;

/// `p1` supported, `p2` and `p3` judged and found wanting -- the same shape
/// `db/truth_exposure_test.rs` seeds. Two failing pages, not one, because the
/// error path below has to prove the pass keeps going after a page it cannot
/// move.
///
/// Every row here carries `evaluated_at`, because these tests are about what a
/// *verdict* costs a page. A page nobody has judged keeps its file by design, so
/// seeding one here would quietly turn every eviction assertion into a test of
/// nothing; `an_unjudged_page_keeps_its_file_after_the_cutover` seeds that case
/// on purpose and asserts the opposite.
///
/// The cutover stays at 0 here; each test advances it itself, so the inert case
/// and the live case are seeded identically and differ in one value.
/// The body every fixture page is stored with. Named because a review has to be
/// stamped with the digest the page actually hashes to -- a placeholder digest
/// is a review of text that is not there, and `page_truth_states` will not count
/// it.
const FIXTURE_PAGE_BODY: &str = "the page body";

async fn db_with_truth_rows() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    for id in ["p1", "p2", "p3"] {
        db.insert_page(
            id,
            id,
            None,
            FIXTURE_PAGE_BODY,
            None,
            None,
            &[],
            "2026-07-27T00:00:00Z",
        )
        .await
        .unwrap();
    }
    {
        let conn = db.test_primary_session().await;
        set_truth(&conn, "p1", "supported").await;
        set_truth(&conn, "p2", "provisional").await;
        set_truth(&conn, "p3", "provisional").await;
    }
    (db, temp)
}

async fn set_truth(conn: &TestDbSession, page_id: &str, status: &str) {
    conn.execute(
        "INSERT INTO page_truth_state
            (page_id,page_version,support_status,human_reviewed,updated_at,
             evaluated_at)
         VALUES (?1,1,?2,0,1,1)",
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
        truth: None,
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

/// The other half of the invariant: a boot-time sweep alone would let the very
/// next distillation put an unsupported page straight back on disk, where it
/// stays readable until the daemon happens to restart. So the write declines
/// too, and the pair is what makes the invariant hold continuously rather than
/// at boot.
///
/// Declining is not an error. A gated page is a page this projection has no
/// business writing, not a failure to write one -- so it returns `Ok(None)` and
/// the caller carries on.
#[tokio::test]
async fn a_gated_write_declines_instead_of_projecting_or_failing() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = KnowledgeProjectionWrite::new(root.path().to_path_buf(), &db);

    // Generation 0: pass-through, both pages land.
    for id in ["p1", "p2"] {
        let written = projection
            .write_page_gated(&db, &page(id))
            .await
            .unwrap_or_else(|e| panic!("{id} must write at generation 0: {e}"));
        assert!(written.is_some(), "{id} was gated while the cutover is off");
    }
    assert_eq!(readable_pages(root.path()), ["p1", "p2"]);

    // Generation 1: same substrate, and only the supported page may be rewritten.
    db.set_truth_cutover_generation(1).await.unwrap();
    std::fs::remove_file(root.path().join("p2.md")).unwrap();

    assert!(
        projection
            .write_page_gated(&db, &page("p1"))
            .await
            .expect("a supported page still writes")
            .is_some(),
        "the cutover must not stop supported pages from projecting"
    );
    assert!(
        projection
            .write_page_gated(&db, &page("p2"))
            .await
            .expect("a gated write is a decision, not an error")
            .is_none(),
        "an unsupported page must not be projected"
    );
    assert_eq!(
        readable_pages(root.path()),
        ["p1"],
        "the declined write must leave nothing on disk for `wenlan pages` to read"
    );
}

/// The PR-B production configuration. The pass runs, and the directory keeps
/// every page -- including the two the judge failed. If this ever goes RED, the
/// cutover has happened without PR-C's ceremony.
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

/// After the cutover the directory is the boundary, so both judged-unsupported
/// pages leave it.
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

/// The reason the pass enumerates the directory and not `state.json`.
///
/// A projected file that `state.json` has lost is still a file on disk, and
/// `wenlan pages` reads the disk. Enumerating `state.json` walks straight past
/// it. Enumerating the directory but evicting through `remove_page` is no
/// better and is arguably worse: that function looks the id up in `state.pages`
/// and returns `Ok(())` *without touching the disk* when it is absent, so the
/// pass would count a silent no-op as a removal -- a false success in the
/// receipt, with the file still readable.
#[tokio::test]
async fn a_file_state_json_has_forgotten_is_still_evicted() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    // Total amnesia: every entry dropped, every file left exactly where it is.
    std::fs::write(
        root.path().join(".wenlan/state.json"),
        r#"{"schema_version":2,"pages":{}}"#,
    )
    .unwrap();

    let removed = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(
        readable_pages(root.path()),
        ["p1"],
        "eviction must not depend on the projection remembering it wrote the file"
    );
    assert_eq!(removed, 2);
}

/// The frontmatter scan reads a bounded prefix of each file, and that bound can
/// land in the middle of a multi-byte character. `read_to_string` fails the
/// entire read when it does, and the failure is swallowed -- so the file drops
/// out of the scan and a page `state.json` has forgotten keeps its `.md`
/// through the eviction that exists to remove it. Fail OPEN, on a disclosure
/// boundary, for any page carrying non-ASCII prose past the cap.
///
/// Three paddings, because only one alignment in three splits a 3-byte
/// character at the cap and the rendered frontmatter's own length is not a
/// constant this test should have to know. All three must evict.
#[tokio::test]
async fn a_page_with_multibyte_prose_past_the_scan_cap_is_still_evicted() {
    for pad in 0..3 {
        let (db, _tmp) = db_with_truth_rows().await;
        let root = tempfile::tempdir().unwrap();
        let projection = KnowledgeProjectionWrite::new(root.path().to_path_buf(), &db);

        let mut p2 = page("p2");
        // Far past the 8 KiB cap, so the truncation lands deep inside the CJK
        // run whatever the header measures.
        p2.content = "x".repeat(pad) + &"文".repeat(8 * 1024);
        projection.write_page(&page("p1")).unwrap();
        projection.write_page(&p2).unwrap();
        assert_eq!(readable_pages(root.path()), ["p1", "p2"], "pad {pad}: seed");

        db.set_truth_cutover_generation(1).await.unwrap();
        // The scan has to be the only enumeration path: with `state.json` intact
        // the id comes back through its keys and the bug stays hidden.
        std::fs::write(
            root.path().join(".wenlan/state.json"),
            r#"{"schema_version":2,"pages":{}}"#,
        )
        .unwrap();

        let removed = projection
            .enforce_projection_directory_invariant(&db)
            .await
            .unwrap();

        assert_eq!(
            readable_pages(root.path()),
            ["p1"],
            "pad {pad}: a page with multi-byte prose escaped eviction"
        );
        assert_eq!(removed, 1, "pad {pad}");
    }
}

/// `knowledge_path` can point at the user's own Obsidian vault, so a top-level
/// `.md` carrying no `origin_id` is somebody else's note -- not a page this
/// projection failed to track.
///
/// It is skipped: never deleted, and never reported as a stuck eviction either.
/// The second half matters as much as the first, because at generation >= 1 a
/// returned error takes the daemon down (section 4) -- so counting a stranger's
/// file as a failure would turn one unrecognized note into a refusal to start.
/// Fail closed on the decision, never on the user's data.
#[tokio::test]
async fn a_file_with_no_origin_id_is_left_alone() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    let stranger = root.path().join("someone-elses-note.md");
    std::fs::write(&stranger, "# Just a note\n\nNo frontmatter at all.\n").unwrap();

    let removed = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .expect("an unattributable file is not a stuck eviction");

    assert_eq!(removed, 2, "only the two unsupported pages are evicted");
    assert!(
        stranger.is_file(),
        "a file the projection cannot attribute must survive untouched"
    );
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

/// One page that cannot be removed must not save the others.
///
/// `remove_page` refuses a projection target that is not a regular file
/// (`knowledge.rs`, the `page_projection_target_invalid` arm), so replacing
/// `p2.md` with a directory of the same name is a portable way to make exactly
/// one eviction fail. `p3` sorts after `p2`, so if the pass still aborted on the
/// first error, `p3.md` would survive -- readable by `wenlan pages`, which is
/// the whole failure this guards.
#[tokio::test]
async fn a_page_that_cannot_be_evicted_does_not_shelter_the_next_one() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    // Turn p2's projection target into a directory: same name, not a file.
    let blocked = root.path().join("p2.md");
    std::fs::remove_file(&blocked).unwrap();
    std::fs::create_dir(&blocked).unwrap();

    let error = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .expect_err("a failed eviction must be reported, not swallowed");

    assert!(
        !root.path().join("p3.md").exists(),
        "p3 sorts after the page that failed; aborting the pass there would leave \
         it readable by `wenlan pages`"
    );
    let message = error.to_string();
    assert!(
        message.contains("p2"),
        "the failure has to name the page still on disk -- got: {message}"
    );
    assert!(
        blocked.is_dir(),
        "the obstruction itself is left alone -- the pass removes files, not whatever \
         happens to occupy the name"
    );
    assert!(
        root.path().join("p1.md").is_file(),
        "the supported page is untouched"
    );
}

// ---- the ceremony: plan, then apply -------------------------------------
//
// Same directory-level standard as everything above. A plan that named the
// right pages while the wrong files stayed on disk would be exactly the lie
// this file exists to catch.

use crate::export::knowledge::KnowledgeWriter;

/// The dry run has to answer a question the live gate cannot: what *would*
/// happen. At generation 0 `page_visibility` short-circuits every page to
/// `Full`, so a plan built on it would report "nothing would be evicted" and
/// the operator would advance blind.
#[tokio::test]
async fn a_dry_run_names_the_unsupported_pages_while_the_cutover_is_still_off() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let _projection = project_all(&db, root.path());
    let writer = KnowledgeWriter::new(root.path().to_path_buf(), &db);

    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
    let plan = writer.plan_truth_cutover(&db, 1).await.unwrap();

    let named: Vec<String> = plan.evictions.iter().map(|e| e.page_id.clone()).collect();
    assert_eq!(named, ["p2", "p3"], "the two pages the judge failed");
    assert_eq!(plan.projected, 3);
    assert!(plan.evictions.iter().all(|e| !e.files.is_empty()));
    assert_eq!(
        readable_pages(root.path()),
        ["p1", "p2", "p3"],
        "a dry run that deleted anything is not a dry run"
    );
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
}

/// The support status underneath a plan changed, so the plan is not the one the
/// operator read.
#[tokio::test]
async fn an_apply_refuses_after_a_support_status_changes() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let _projection = project_all(&db, root.path());
    let writer = KnowledgeWriter::new(root.path().to_path_buf(), &db);

    let stale = writer.plan_truth_cutover(&db, 1).await.unwrap();
    {
        let conn = db.test_primary_session().await;
        conn.execute(
            "UPDATE page_truth_state SET support_status = 'supported' WHERE page_id = 'p2'",
            (),
        )
        .await
        .unwrap();
    }

    let refused = writer.run_truth_cutover(&db, 1, &stale.digest).await;
    assert!(refused.is_err(), "a stale plan must not be applied");
    assert_eq!(
        readable_pages(root.path()),
        ["p1", "p2", "p3"],
        "the refusal must happen before anything is deleted"
    );
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
    // The lease is taken BEFORE the re-plan, so a digest refusal happens with
    // the fence held and gives it straight back. Off is what matters -- writers
    // are not left wedged by a refusal -- and the epoch has moved, which is how
    // a released fence always looks.
    let fence = db.cutover_fence().await.unwrap();
    assert_eq!(fence.phase, crate::db::CutoverPhase::Off);
    assert!(
        fence.epoch > 0,
        "the refusal took the lease and returned it, so the epoch cannot be untouched"
    );
    assert!(crate::truth_adapter::page_write_permit(&db, "p1")
        .await
        .unwrap()
        .is_some());
}

/// Directory first, then the commit, then the fence. Asserted at the directory,
/// because that is the surface `wenlan pages` reads.
#[tokio::test]
async fn the_ceremony_evicts_then_commits_then_releases_the_fence() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let _projection = project_all(&db, root.path());
    let writer = KnowledgeWriter::new(root.path().to_path_buf(), &db);

    let plan = writer.plan_truth_cutover(&db, 1).await.unwrap();
    writer
        .run_truth_cutover(&db, 1, &plan.digest)
        .await
        .unwrap();

    assert_eq!(
        readable_pages(root.path()),
        ["p1"],
        "only the supported page may still be readable off the directory"
    );
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 1);
    let fence = db.cutover_fence().await.unwrap();
    assert_eq!(fence.phase, crate::db::CutoverPhase::Committed);
    assert!(
        fence.epoch >= 2,
        "begin and commit each bump the epoch, so a committed fence is past 1"
    );
}

/// After the ceremony, ordinary page writes are allowed again -- the fence is
/// released, not left holding. A ceremony that permanently wedged the writers
/// would be safe and useless.
#[tokio::test]
async fn writers_are_allowed_back_after_the_commit() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let _projection = project_all(&db, root.path());
    let writer = KnowledgeWriter::new(root.path().to_path_buf(), &db);

    let plan = writer.plan_truth_cutover(&db, 1).await.unwrap();
    writer
        .run_truth_cutover(&db, 1, &plan.digest)
        .await
        .unwrap();

    assert!(
        crate::truth_adapter::page_write_permit(&db, "p1")
            .await
            .unwrap()
            .is_some(),
        "the supported page must still be projectable after the ceremony"
    );
    assert!(
        crate::truth_adapter::page_write_permit(&db, "p2")
            .await
            .unwrap()
            .is_none(),
        "and the unsupported one must now be refused on its own merits, not by \
         the fence"
    );
}

/// The digest covers the **whole projection**, not the eviction list.
///
/// This is the case a digest over the evictions alone cannot catch: a page that
/// appears between the dry run and the apply, and is supported, leaves the
/// eviction list byte-identical. The operator approved a plan over a three-page
/// directory; applying it against a four-page directory puts a page they never
/// saw inside the ceremony's scope. Deletions here are unrecoverable, so
/// "the list happens to match" is not consent.
#[tokio::test]
async fn an_apply_refuses_when_a_page_appeared_that_the_plan_never_covered() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    let writer = KnowledgeWriter::new(root.path().to_path_buf(), &db);

    let plan = writer.plan_truth_cutover(&db, 1).await.unwrap();

    // A fourth page lands, supported, so the eviction list does not move.
    db.insert_page(
        "p4",
        "p4",
        None,
        "",
        None,
        None,
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    {
        let conn = db.test_primary_session().await;
        set_truth(&conn, "p4", "supported").await;
    }
    projection.write_page(&page("p4")).unwrap();

    let fresh = writer.plan_truth_cutover(&db, 1).await.unwrap();
    let before: Vec<&str> = plan.evictions.iter().map(|e| e.page_id.as_str()).collect();
    let after: Vec<&str> = fresh.evictions.iter().map(|e| e.page_id.as_str()).collect();
    assert_eq!(
        before, after,
        "the premise of this test is that the eviction list is unchanged"
    );
    assert_ne!(
        plan.digest, fresh.digest,
        "an identical eviction list over a changed projection must still change \
         the digest, or the apply gate is blind to exactly this case"
    );

    let refused = writer.run_truth_cutover(&db, 1, &plan.digest).await;
    assert!(
        refused.is_err(),
        "a plan that predates p4 must not be applied"
    );
    assert_eq!(readable_pages(root.path()), ["p1", "p2", "p3", "p4"]);
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
}

/// The commit is downstream of the directory work, and conditional on it.
///
/// The happy path cannot prove this on its own: the reconcile pass that runs
/// after the commit deletes the same files, so a ceremony that committed first
/// and evicted second would leave an identical directory. The ordering is only
/// observable when the directory work FAILS -- and then the generation must
/// still read 0, because a committed generation with the prose still on disk is
/// precisely the "false trust" state the directory-first rule exists to prevent.
///
/// Unix-only: the forced failure is a read-only projection root, and Windows
/// permits unlinking from a read-only directory.
#[cfg(unix)]
#[tokio::test]
async fn a_failed_eviction_leaves_the_generation_uncommitted() {
    use std::os::unix::fs::PermissionsExt;

    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let _projection = project_all(&db, root.path());
    let writer = KnowledgeWriter::new(root.path().to_path_buf(), &db);

    let plan = writer.plan_truth_cutover(&db, 1).await.unwrap();
    assert!(!plan.evictions.is_empty(), "nothing to fail at otherwise");

    let readonly = std::fs::Permissions::from_mode(0o555);
    let writable = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(root.path(), readonly).unwrap();
    let outcome = writer.run_truth_cutover(&db, 1, &plan.digest).await;
    std::fs::set_permissions(root.path(), writable).unwrap();

    assert!(outcome.is_err(), "the eviction could not have succeeded");
    assert_eq!(
        db.truth_cutover_generation().await.unwrap(),
        0,
        "the generation was committed even though the prose is still on disk"
    );
    assert_eq!(
        readable_pages(root.path()),
        ["p1", "p2", "p3"],
        "nothing was removed, which is the premise of the assertion above"
    );

    // And the lease was given back, at a new epoch, so writers are not wedged.
    let fence = db.cutover_fence().await.unwrap();
    assert_eq!(fence.phase, crate::db::CutoverPhase::Off);
    assert!(fence.epoch > 0, "the abort must not reuse the old epoch");
    assert!(crate::truth_adapter::page_write_permit(&db, "p1")
        .await
        .unwrap()
        .is_some());
}

/// The reason `Support` has three states instead of a bool.
///
/// Every page predating claim derivation is backfilled unjudged, so a cutover
/// that read "no verdict" as "failed" would empty the whole vault on day one --
/// 511 pages on the author's own machine -- and call it enforcement. A page
/// nobody has judged keeps its file, at every generation, forever.
#[tokio::test]
async fn an_unjudged_page_keeps_its_file_after_the_cutover() {
    let (db, _tmp) = db_with_truth_rows().await;
    db.insert_page(
        "p4",
        "p4",
        None,
        "",
        None,
        None,
        &[],
        "2026-07-27T00:00:00Z",
    )
    .await
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    projection.write_page(&page("p4")).unwrap();
    db.set_truth_cutover_generation(1).await.unwrap();

    projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(
        readable_pages(root.path()),
        ["p1", "p4"],
        "p4 has no truth row at all, which is not a verdict against it"
    );
}

/// A human who approved this page outranks the machine that failed it.
///
/// `human_reviewed` has existed since migration 98 and was read into
/// `PageTruth` and then dropped on the floor -- no visibility or eviction path
/// consulted it -- so a page its owner had personally approved was archived
/// exactly like one nobody had ever seen.
///
/// The version and digest are not decoration: migration 98's CHECK refuses a
/// review that does not name the exact page version and bytes a person signed
/// off, which is what stops "reviewed" from being a sticky bit someone sets once.
/// The read now enforces the same thing the CHECK does — see the sibling test
/// below — so this fixture records the digest of the text p2 actually holds
/// rather than a placeholder.
#[tokio::test]
async fn a_human_reviewed_page_keeps_its_file_after_a_failed_judgement() {
    let (db, _tmp) = db_with_truth_rows().await;
    approve_p2(
        &db,
        &crate::provenance::revision_content_digest(FIXTURE_PAGE_BODY),
    )
    .await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(
        readable_pages(root.path()),
        ["p1", "p2"],
        "p2 failed the same judgement p3 did, and a person had already approved it"
    );
}

/// Record a human approval of p2 naming `digest` as the text that was read.
async fn approve_p2(db: &MemoryDB, digest: &str) {
    let conn = db.test_primary_session().await;
    conn.execute(
        "UPDATE page_truth_state
            SET human_reviewed=1, reviewed_page_version=1,
                reviewed_page_digest=?1
          WHERE page_id='p2'",
        libsql::params![digest],
    )
    .await
    .unwrap();
}

/// The other half, and the one that makes human authority a fact rather than a
/// flag: an approval that names text the page no longer holds grants nothing.
///
/// Human review OUTRANKS a failed machine judgement, so an unverifiable approval
/// is the strongest stale claim in the system — it keeps a file that every other
/// signal says should go. Trusting the raw bit made "reviewed" exactly the sticky
/// bit the schema CHECK was written to prevent, just set one layer up.
#[tokio::test]
async fn a_human_approval_of_text_the_page_no_longer_holds_keeps_nothing() {
    let (db, _tmp) = db_with_truth_rows().await;
    approve_p2(&db, "a digest of some text this page does not hold").await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(
        readable_pages(root.path()),
        ["p1"],
        "an approval that cannot be checked against the live page is not authority"
    );
}

/// Hiding a page is a retrieval decision, not a licence to destroy prose.
///
/// The bytes have to survive somewhere a human can find them, because
/// `knowledge_path` is very often the user's own vault and they may have edited
/// the file by hand. `archive/` is plain and visible for exactly that reason,
/// and it is out of `wenlan pages` because that reader takes one directory level
/// of `*.md`.
#[tokio::test]
async fn an_evicted_page_is_moved_into_archive_not_deleted() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    let before = std::fs::read_to_string(root.path().join("p2.md")).unwrap();
    db.set_truth_cutover_generation(1).await.unwrap();

    projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(readable_pages(root.path()), ["p1"]);
    let archived = root.path().join("archive").join("p2.md");
    assert_eq!(
        std::fs::read_to_string(&archived).unwrap(),
        before,
        "the archived file must be the page, byte for byte, not a stub"
    );
    assert!(root.path().join("archive").join("p3.md").is_file());
}

/// An archived page must not be rediscovered on the next pass.
///
/// `archive/` has no `.md` extension, so the directory scan skips it. If that
/// ever stopped being true the second pass would find the archived file, try to
/// archive it again, and collide with itself.
#[tokio::test]
async fn a_second_pass_leaves_the_archive_alone() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();
    let removed_again = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(removed_again, 0, "the second pass found nothing left to do");
    assert!(root.path().join("archive").join("p2.md").is_file());
    assert!(
        !root.path().join("archive").join("archive").exists(),
        "the archive archived itself"
    );
}

/// Nothing on disk enforces one file per page.
///
/// A sync conflict copies the whole `.md`, `origin_id` frontmatter and all. The
/// scan used to keep one filename per page, so the eviction archived one copy
/// and left the other sitting in the projection root -- readable by
/// `wenlan pages`, which is the exact disclosure the cutover exists to close.
#[tokio::test]
async fn every_copy_of_a_page_leaves_the_projection_not_just_one() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    let conflict = root.path().join("p2 (conflicted copy).md");
    std::fs::copy(root.path().join("p2.md"), &conflict).unwrap();
    db.set_truth_cutover_generation(1).await.unwrap();

    projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();

    assert_eq!(
        readable_pages(root.path()),
        ["p1"],
        "the conflicted copy carries p2's origin_id and is just as readable"
    );
    assert!(root
        .path()
        .join("archive")
        .join("p2 (conflicted copy).md")
        .is_file());
}

/// A stale `state.json` entry whose `.md` is already gone must actually be
/// dropped from the state file, even when it is the ONLY thing being evicted.
///
/// The eviction loop gated its `save_state_cap` on `removed > 0`, and the
/// file-less branch never increments `removed` -- there is no file to move. So
/// with every evictable page file-less, the `state.pages.remove` calls were
/// computed and then thrown away with the unsaved state, and the entry came
/// back on the next pass forever. That is precisely the cleanup the branch
/// exists to perform, and it was the one case it could not do.
#[tokio::test]
async fn a_stale_state_entry_with_no_file_is_dropped_even_when_nothing_moves() {
    let (db, _tmp) = db_with_truth_rows().await;
    let root = tempfile::tempdir().unwrap();
    let projection = project_all(&db, root.path());
    db.set_truth_cutover_generation(1).await.unwrap();

    // Delete the evictable pages' files but leave state.json naming them, so
    // every page the cutover wants to evict is file-less and `removed` stays 0.
    for name in ["p2.md", "p3.md"] {
        let path = root.path().join(name);
        if path.exists() {
            std::fs::remove_file(&path).unwrap();
        }
    }

    let removed = projection
        .enforce_projection_directory_invariant(&db)
        .await
        .unwrap();
    assert_eq!(removed, 0, "no file on disk means nothing to move");

    let state = std::fs::read_to_string(root.path().join(".wenlan/state.json")).unwrap();
    for gone in ["p2", "p3"] {
        assert!(
            !state.contains(&format!("\"{gone}\"")),
            "stale entry {gone} survived the eviction that exists to remove it: {state}"
        );
    }
    assert!(
        state.contains("\"p1\""),
        "the supported page's entry must stay: {state}"
    );
}
