use super::*;
use crate::{
    db::{
        repair_verification::{
            with_repair_verification_test_control, RepairVerificationDbContender,
            RepairVerificationTestControl,
        },
        tests::test_db,
        MemoryDB,
    },
    error::WenlanError,
    export::knowledge::KnowledgeProjectionWrite,
    lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    },
};
use std::fs::OpenOptions;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wenlan_types::{
    lint::{LintProfile, LintQuery},
    repair::{
        ApplyRepairRequest, PrepareRepairRequest, RepairChoice, RepairMutation,
        RepairRollbackPayloadV2, RepairTarget, RepairWriter, VerifyRepairRequest,
    },
    RefinementPayload,
};

const OCCURRENCE: &str = "cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd";

struct RenameFixture {
    db: MemoryDB,
    _db_dir: tempfile::TempDir,
    page_root: tempfile::TempDir,
    repair_root: tempfile::TempDir,
    request: PrepareRepairRequest,
    review_id: String,
}

fn assert_projection_file_lock_contended(page_root: &Path) {
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(page_root.join(".wenlan/.projection.lock"))
        .unwrap();
    assert_eq!(
        lock.try_lock_exclusive().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
}

fn assert_projection_file_lock_acquirable(page_root: &Path) {
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .open(page_root.join(".wenlan/.projection.lock"))
        .unwrap();
    lock.try_lock_exclusive().unwrap();
    lock.unlock().unwrap();
}

async fn rename_fixture() -> RenameFixture {
    let (db, db_dir) = test_db().await;
    db.conn
        .lock()
        .await
        .execute_batch(
            "INSERT INTO spaces (id,name,created_at,updated_at)
             VALUES ('space-work','work',1,1);
             INSERT INTO pages
                 (id,title,summary,content,space,source_memory_ids,version,status,embedding,
                  created_at,last_compiled,last_modified,workspace,creation_kind,review_status)
             VALUES
                 ('page-a','Origin','A summary','A body','work','[]',4,'active',
                  NULL,'2026-07-17T00:00:00Z','2026-07-17T00:00:00Z',
                  '2026-07-17T00:00:00Z','work','distilled','confirmed'),
                 ('page-b','origin','B summary','B body','work','[]',2,'active',
                  NULL,'2026-07-17T00:00:00Z','2026-07-17T00:00:00Z',
                  '2026-07-17T00:00:00Z','work','distilled','confirmed'),
                 ('page-other','Origin Git Workflow Gotchas','Other summary','Other body',
                  'personal','[]',3,'active',
                  NULL,'2026-07-17T00:00:00Z','2026-07-17T00:00:00Z',
                  '2026-07-17T00:00:00Z','personal','distilled','confirmed');",
        )
        .await
        .unwrap();
    let page_root = tempfile::tempdir().unwrap();
    for page_id in ["page-a", "page-b", "page-other"] {
        let page = db.get_page(page_id).await.unwrap().unwrap();
        KnowledgeProjectionWrite::new(page_root.path().to_path_buf(), &db)
            .write_page(&page)
            .unwrap();
    }
    std::fs::write(page_root.path().join("unrelated.txt"), b"untouched").unwrap();

    let occurrence = RepairDigest::parse(OCCURRENCE).unwrap();
    let review_id = format!("lint_review_{OCCURRENCE}");
    let source_ids = vec!["page-a".to_string()];
    let payload = RefinementPayload::LintRepairReview {
        check_id: "pages.duplicate_active_titles".to_string(),
        occurrence_digest: occurrence.clone(),
        owner_binding_digest: lint_review_owner_binding_digest(&occurrence, &source_ids).unwrap(),
        issue: "Rename the narrower duplicate Page.".to_string(),
        choices: vec!["Origin Git Workflow Gotchas".to_string()],
        suggested_research_queries: vec![],
    };
    db.insert_lint_review_if_absent(
        &review_id,
        &source_ids,
        &serde_json::to_string(&payload).unwrap(),
    )
    .await
    .unwrap();

    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(page_root.path()),
            true,
        )
        .await
        .unwrap();
    let request = PrepareRepairRequest::try_new_with_choice(
        RepairLintScope::global(),
        general,
        None,
        RepairChoice::rename_page_title(
            review_id.clone(),
            "page-a".to_string(),
            "Origin".to_string(),
            "Origin Git Workflow Gotchas".to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    RenameFixture {
        db,
        _db_dir: db_dir,
        page_root,
        repair_root: tempfile::tempdir().unwrap(),
        request,
        review_id,
    }
}

async fn prepare(fixture: &RenameFixture) -> RepairManifest {
    prepare_memory_reclassification_with_pages(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        fixture.request.clone(),
        Some(fixture.page_root.path()),
        1_721_000_001,
    )
    .await
    .unwrap()
}

fn exact_apply(manifest: &RepairManifest) -> ApplyRepairRequest {
    ApplyRepairRequest::try_new(
        manifest.manifest_id().to_string(),
        manifest.manifest_digest().clone(),
        format!(
            "apply repair {} {}",
            manifest.manifest_id(),
            manifest.manifest_digest().as_str()
        ),
    )
    .unwrap()
}

fn target_filename(root: &Path, page_id: &str) -> String {
    serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(root.join(".wenlan/state.json")).unwrap(),
    )
    .unwrap()["pages"][page_id]["file"]
        .as_str()
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn preparation_binds_exact_review_page_scope_row_projection_and_embedding() {
    let fixture = rename_fixture().await;
    let filename = target_filename(fixture.page_root.path(), "page-a");
    let expected_embedding = encode_page_title_embedding(
        fixture
            .db
            .generate_embeddings(&[crate::pages::page_embedding_text(
                "Origin Git Workflow Gotchas",
                Some("A summary"),
                "A body",
            )])
            .unwrap()
            .pop()
            .unwrap(),
    )
    .unwrap();
    let manifest = prepare(&fixture).await;

    assert_eq!(manifest.writer(), RepairWriter::RenamePageTitle);
    assert_eq!(manifest.expected_state().version(), Some(4));
    assert!(matches!(
        manifest.target(),
        RepairTarget::PageProjection { page_id, scope }
            if page_id == "page-a" && scope.space() == Some("work")
    ));
    assert!(matches!(
        manifest.mutation(),
        RepairMutation::RenamePageTitle {
            before_title,
            after_title,
            after_embedding_hex
        } if before_title == "Origin"
            && after_title == "Origin Git Workflow Gotchas"
            && after_embedding_hex == &expected_embedding
    ));
    let binding = manifest.source().review_binding().unwrap();
    assert_eq!(binding.review_id(), fixture.review_id);
    assert_eq!(binding.owner_ids(), ["page-a"]);
    assert_eq!(binding.occurrence_digest().as_str(), OCCURRENCE);
    let rollback = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf())
        .load_rename_page_title_rollback(&manifest)
        .unwrap();
    let RepairRollbackPayloadV2::RenamePageTitle {
        page_id,
        page_columns,
        before_page_row,
        projection_target_path,
        projection_entries,
    } = rollback
    else {
        panic!("expected title rollback");
    };
    assert_eq!(page_id, "page-a");
    assert_eq!(page_columns.len(), before_page_row.len());
    assert!(page_columns.iter().any(|column| column == "embedding"));
    assert_eq!(projection_target_path, filename);
    assert_eq!(projection_entries.len(), 2);
}

#[tokio::test]
async fn capture_rejects_a_duplicate_key_in_non_target_projection_state() {
    let fixture = rename_fixture().await;
    let page_a = target_filename(fixture.page_root.path(), "page-a");
    let page_b = target_filename(fixture.page_root.path(), "page-b");
    let page_other = target_filename(fixture.page_root.path(), "page-other");
    std::fs::write(
        fixture.page_root.path().join(".wenlan/state.json"),
        format!(
            r#"{{
  "schema_version": 2,
  "pages": {{
    "page-a": {{
      "file": "{page_a}",
      "version": 4,
      "last_written": "2026-07-17T00:00:00Z"
    }},
    "page-b": {{
      "file": "{page_b}",
      "version": 2,
      "last_written": "2026-07-17T00:00:00Z"
    }},
    "page-other": {{
      "file": "{page_other}",
      "version": 3,
      "last_written": "first duplicate value"
    }},
    "page-other": {{
      "file": "{page_other}",
      "version": 3,
      "last_written": "silently canonicalized duplicate"
    }}
  }}
}}"#
        ),
    )
    .unwrap();

    let error = KnowledgeProjectionWrite::with_repair_lock(
        fixture.page_root.path().to_path_buf(),
        &fixture.db,
        |projection| {
            projection
                .capture_rename_page_projection("page-a")
                .map(|_| ())
        },
    )
    .expect_err("duplicate non-target state keys must make title capture fail closed");

    assert!(matches!(
        error,
        WenlanError::Conflict(message) if message == "repair_target_stale"
    ));
}

#[tokio::test]
async fn apply_changes_only_title_version_embedding_and_target_projection() {
    let fixture = rename_fixture().await;
    let filename = target_filename(fixture.page_root.path(), "page-a");
    let other_state_before =
        std::fs::read(fixture.page_root.path().join(".wenlan/state.json")).unwrap();
    let other_file = target_filename(fixture.page_root.path(), "page-b");
    let other_file_before = std::fs::read(fixture.page_root.path().join(&other_file)).unwrap();
    let manifest = prepare(&fixture).await;

    apply_repair_with_pages(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        exact_apply(&manifest),
        Some(fixture.page_root.path()),
        1_721_000_002,
    )
    .await
    .unwrap();

    let page = fixture.db.get_page("page-a").await.unwrap().unwrap();
    assert_eq!(page.title, "Origin Git Workflow Gotchas");
    assert_eq!(page.version, 5);
    assert_eq!(
        target_filename(fixture.page_root.path(), "page-a"),
        filename
    );
    assert!(
        std::fs::read_to_string(fixture.page_root.path().join(filename))
            .unwrap()
            .contains("title: \"Origin Git Workflow Gotchas\"")
    );
    assert_eq!(
        std::fs::read(fixture.page_root.path().join(other_file)).unwrap(),
        other_file_before
    );
    assert_eq!(
        std::fs::read(fixture.page_root.path().join("unrelated.txt")).unwrap(),
        b"untouched"
    );
    assert_ne!(
        std::fs::read(fixture.page_root.path().join(".wenlan/state.json")).unwrap(),
        other_state_before
    );
}

#[tokio::test]
async fn apply_then_verification_reuses_the_owned_projection_session() {
    let fixture = rename_fixture().await;
    let manifest = prepare(&fixture).await;
    let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());
    let apply_receipt = apply_repair_with_pages(
        &fixture.db,
        &store,
        exact_apply(&manifest),
        Some(fixture.page_root.path()),
        1_721_000_002,
    )
    .await
    .unwrap();
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &fixture.db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(fixture.page_root.path()),
            true,
        )
        .await
        .unwrap();
    let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
    let final_receipt = manifest_dir.join(VERIFICATION_RECEIPT_FILE);
    let pending = manifest_dir.join(APPLY_RECEIPT_PENDING_FILE);
    std::fs::hard_link(manifest_dir.join(APPLY_RECEIPT_FILE), &pending).unwrap();
    let root_after_session = fixture.page_root.path().to_path_buf();
    let root_after_begin = root_after_session.clone();
    let root_before_persist = root_after_session.clone();
    let root_after_persist = root_after_session.clone();
    let root_after_commit = root_after_session.clone();
    let contender = RepairVerificationDbContender::new(&fixture.db);
    let start_contender = contender.clone();
    let contender_before_persist = contender.clone();
    let contender_after_persist = contender.clone();
    let receipt_before_persist = final_receipt.clone();
    let receipt_after_persist = final_receipt.clone();
    let receipt_after_commit = final_receipt.clone();
    let pending_after_commit = pending.clone();

    let receipt = with_repair_verification_test_control(
        RepairVerificationTestControl {
            after_rename_session_acquired: Some(Box::new(move || {
                assert_projection_file_lock_contended(&root_after_session);
            })),
            after_begin: Some(Box::new(move || {
                start_contender.start_and_observe_pending();
                assert_projection_file_lock_contended(&root_after_begin);
            })),
            before_receipt_persist: Some(Box::new(move || {
                contender_before_persist.assert_pending();
                assert_projection_file_lock_contended(&root_before_persist);
                assert!(!receipt_before_persist.exists());
            })),
            after_receipt_persist: Some(Box::new(move || {
                contender_after_persist.assert_pending();
                assert_projection_file_lock_contended(&root_after_persist);
                assert!(receipt_after_persist.is_file());
            })),
            after_commit_before_pending_clear: Some(Box::new(move || {
                assert_projection_file_lock_contended(&root_after_commit);
                assert!(receipt_after_commit.is_file());
                assert!(pending_after_commit.is_file());
            })),
            ..Default::default()
        },
        record_repair_verification(
            &fixture.db,
            &store,
            VerifyRepairRequest::try_new_general_only(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                general,
            )
            .unwrap(),
            Some(fixture.page_root.path()),
            1_721_000_003,
        ),
    )
    .await
    .expect("title rename apply and verification must not contend with its own projection session");

    assert_eq!(receipt.manifest_id(), manifest.manifest_id());
    assert!(store
        .has_completed_verification(manifest.manifest_id())
        .unwrap());
    assert!(!pending.exists());
    contender.assert_entered_after_verification();
    assert_projection_file_lock_acquirable(fixture.page_root.path());
}

#[tokio::test]
async fn verification_page_receipts_stay_on_the_pinned_root_after_ancestor_swap() {
    let fixture = rename_fixture().await;
    let manifest = prepare(&fixture).await;
    let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());
    let apply_receipt = apply_repair_with_pages(
        &fixture.db,
        &store,
        exact_apply(&manifest),
        Some(fixture.page_root.path()),
        1_721_000_002,
    )
    .await
    .unwrap();
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &fixture.db,
            &LintQuery::new(Some(LintProfile::General), None),
            Some(fixture.page_root.path()),
            true,
        )
        .await
        .unwrap();
    let page_root = fixture.page_root.path().to_path_buf();
    let pinned_root = page_root.with_extension("verification-pinned");

    let result = with_repair_verification_test_control(
        RepairVerificationTestControl {
            after_rename_session_acquired: Some(Box::new({
                let page_root = page_root.clone();
                let pinned_root = pinned_root.clone();
                move || {
                    std::fs::rename(&page_root, &pinned_root).unwrap();
                    std::fs::create_dir(&page_root).unwrap();
                    std::fs::write(page_root.join("replacement-canary.md"), b"replacement")
                        .unwrap();
                }
            })),
            ..Default::default()
        },
        record_repair_verification(
            &fixture.db,
            &store,
            VerifyRepairRequest::try_new_general_only(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                apply_receipt.receipt_digest().clone(),
                general,
            )
            .unwrap(),
            Some(&page_root),
            1_721_000_003,
        ),
    )
    .await;

    std::fs::remove_dir_all(&page_root).unwrap();
    std::fs::rename(&pinned_root, &page_root).unwrap();
    let receipt = result.expect("verification must read the root pinned before the ancestor swap");
    assert_eq!(receipt.manifest_id(), manifest.manifest_id());
}

#[tokio::test]
async fn stale_dimensions_are_zero_mutation_and_cross_scope_same_title_is_allowed() {
    for stale_case in [
        "same_scope_collision",
        "page_changed",
        "projection_changed",
        "unsafe_projection",
        "version",
        "title",
        "review",
    ] {
        let fixture = rename_fixture().await;
        let manifest = prepare(&fixture).await;
        let filename = target_filename(fixture.page_root.path(), "page-a");
        match stale_case {
            "same_scope_collision" => {
                fixture
                    .db
                    .conn
                    .lock()
                    .await
                    .execute(
                        // Single-axis (spec §1): a raw INSERT bypassing the
                        // insert_page_with_kind ladder must still bind both
                        // scope columns to the same value, mirroring how the
                        // fixture seeds page-a/page-b ('work'/'work'). Omitting
                        // `space` would default it to 'unfiled', so the
                        // same-scope collision the read-collapse checks on
                        // `space` alone would go undetected.
                        "INSERT INTO pages
                         (id,title,content,source_memory_ids,version,status,created_at,last_compiled,
                          last_modified,space,workspace,creation_kind,review_status)
                         VALUES ('page-new','Origin Git Workflow Gotchas','body','[]',1,'active',
                                 'now','now','now','work','work','distilled','confirmed')",
                        (),
                    )
                    .await
                    .unwrap();
            }
            "page_changed" => {
                fixture
                    .db
                    .conn
                    .lock()
                    .await
                    .execute("UPDATE pages SET summary='changed' WHERE id='page-a'", ())
                    .await
                    .unwrap();
            }
            "projection_changed" => std::fs::write(
                fixture.page_root.path().join(&filename),
                b"changed projection",
            )
            .unwrap(),
            "unsafe_projection" => {
                std::fs::remove_file(fixture.page_root.path().join(&filename)).unwrap();
                #[cfg(unix)]
                std::os::unix::fs::symlink(
                    fixture.page_root.path().join("unrelated.txt"),
                    fixture.page_root.path().join(&filename),
                )
                .unwrap();
            }
            "version" => {
                fixture
                    .db
                    .conn
                    .lock()
                    .await
                    .execute("UPDATE pages SET version=version+1 WHERE id='page-a'", ())
                    .await
                    .unwrap();
            }
            "title" => {
                fixture
                    .db
                    .conn
                    .lock()
                    .await
                    .execute("UPDATE pages SET title='changed' WHERE id='page-a'", ())
                    .await
                    .unwrap();
            }
            "review" => {
                fixture
                    .db
                    .conn
                    .lock()
                    .await
                    .execute(
                        "UPDATE refinement_queue SET status='dismissed' WHERE id=?1",
                        libsql::params![fixture.review_id.clone()],
                    )
                    .await
                    .unwrap();
            }
            _ => unreachable!(),
        }
        let db_before = {
            let conn = fixture.db.conn.lock().await;
            database_content_digest(&conn).await.unwrap()
        };
        let tree_before = crate::lint::pages::fs::scan_page_root(fixture.page_root.path()).unwrap();
        let result = apply_repair_with_pages(
            &fixture.db,
            &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
            exact_apply(&manifest),
            Some(fixture.page_root.path()),
            1_721_000_002,
        )
        .await;
        assert!(
            matches!(result, Err(WenlanError::Conflict(ref message)) if message == "repair_target_stale"),
            "unexpected {stale_case}: {result:?}"
        );
        let conn = fixture.db.conn.lock().await;
        assert_eq!(database_content_digest(&conn).await.unwrap(), db_before);
        drop(conn);
        assert_eq!(
            crate::lint::pages::fs::scan_page_root(fixture.page_root.path())
                .unwrap()
                .normalized_bytes(),
            tree_before.normalized_bytes()
        );
    }
}

#[tokio::test]
async fn pre_commit_failure_restores_raw_projection_and_rolls_back_db() {
    let fixture = rename_fixture().await;
    let manifest = prepare(&fixture).await;
    let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());
    let rollback = store.load_rename_page_title_rollback(&manifest).unwrap();
    let state_before = std::fs::read(fixture.page_root.path().join(".wenlan/state.json")).unwrap();
    let filename = target_filename(fixture.page_root.path(), "page-a");
    let file_before = std::fs::read(fixture.page_root.path().join(&filename)).unwrap();

    let result = crate::post_write::rename_page_title_cas(
        &fixture.db,
        &manifest,
        &rollback,
        fixture.page_root.path(),
        |_| {
            Err(WenlanError::Io(std::io::Error::other(
                "forced pre-commit failure",
            )))
        },
    )
    .await;

    assert!(result.is_err());
    let page = fixture.db.get_page("page-a").await.unwrap().unwrap();
    assert_eq!(page.title, "Origin");
    assert_eq!(page.version, 4);
    assert_eq!(
        std::fs::read(fixture.page_root.path().join(".wenlan/state.json")).unwrap(),
        state_before
    );
    assert_eq!(
        std::fs::read(fixture.page_root.path().join(filename)).unwrap(),
        file_before
    );
}

#[tokio::test]
async fn mid_projection_write_failure_restores_target_state_and_database() {
    let fixture = rename_fixture().await;
    let manifest = prepare(&fixture).await;
    let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());
    let rollback = store.load_rename_page_title_rollback(&manifest).unwrap();
    let state_before = std::fs::read(fixture.page_root.path().join(".wenlan/state.json")).unwrap();
    let filename = target_filename(fixture.page_root.path(), "page-a");
    let file_before = std::fs::read(fixture.page_root.path().join(&filename)).unwrap();

    let result = crate::post_write::rename_page_title_cas_with_projection_write_hook(
        &fixture.db,
        &manifest,
        &rollback,
        fixture.page_root.path(),
        || {
            Err(WenlanError::Io(std::io::Error::other(
                "forced failure after target write before state write",
            )))
        },
        |_| Ok(()),
    )
    .await;

    assert!(result.is_err());
    let page = fixture.db.get_page("page-a").await.unwrap().unwrap();
    assert_eq!(page.title, "Origin");
    assert_eq!(page.version, 4);
    assert_eq!(
        std::fs::read(fixture.page_root.path().join(".wenlan/state.json")).unwrap(),
        state_before
    );
    assert_eq!(
        std::fs::read(fixture.page_root.path().join(filename)).unwrap(),
        file_before
    );

    let owned = rename_fixture().await;
    let owned_manifest = prepare(&owned).await;
    let owned_store = RepairArtifactStore::new(owned.repair_root.path().to_path_buf());
    let owned_rollback = owned_store
        .load_rename_page_title_rollback(&owned_manifest)
        .unwrap();
    let owned_state_before =
        std::fs::read(owned.page_root.path().join(".wenlan/state.json")).unwrap();
    let owned_filename = target_filename(owned.page_root.path(), "page-a");
    let owned_file_before = std::fs::read(owned.page_root.path().join(&owned_filename)).unwrap();
    let hook_fired = Arc::new(AtomicBool::new(false));
    let hook_fired_inside = Arc::clone(&hook_fired);

    let owned_result = crate::post_write::rename_page_title_cas_with_projection_write_hook(
        &owned.db,
        &owned_manifest,
        &owned_rollback,
        owned.page_root.path(),
        move || {
            hook_fired_inside.store(true, Ordering::SeqCst);
            Err(WenlanError::Io(std::io::Error::other(
                "forced owned-capture failure after target write before state write",
            )))
        },
        |_| Ok(()),
    )
    .await;

    assert!(owned_result.is_err());
    assert!(hook_fired.load(Ordering::SeqCst));
    let owned_page = owned.db.get_page("page-a").await.unwrap().unwrap();
    assert_eq!(owned_page.title, "Origin");
    assert_eq!(owned_page.version, 4);
    assert_eq!(
        std::fs::read(owned.page_root.path().join(".wenlan/state.json")).unwrap(),
        owned_state_before
    );
    assert_eq!(
        std::fs::read(owned.page_root.path().join(owned_filename)).unwrap(),
        owned_file_before
    );
}

#[tokio::test]
async fn recovery_publishes_only_exact_committed_post_and_restores_known_pre_db_mixed_state() {
    let committed = rename_fixture().await;
    let committed_manifest = prepare(&committed).await;
    let committed_store = RepairArtifactStore::new(committed.repair_root.path().to_path_buf());
    let committed_rollback = committed_store
        .load_rename_page_title_rollback(&committed_manifest)
        .unwrap();
    let mut pending = committed_store
        .begin_apply_receipt(committed_manifest.manifest_id())
        .unwrap();
    let mut prepared = None;
    crate::post_write::rename_page_title_cas(
        &committed.db,
        &committed_manifest,
        &committed_rollback,
        committed.page_root.path(),
        |proof| {
            let draft = RepairApplyReceiptDraft::try_new(
                committed_manifest.manifest_id().to_string(),
                committed_manifest.manifest_digest().clone(),
                1_721_000_002,
                proof.before_target_receipt().clone(),
                proof.after_target_receipt().clone(),
                proof.non_target_before().clone(),
                proof.non_target_after().clone(),
                proof.post_apply_db_digest().clone(),
                committed_manifest.allowed_effects().clone(),
                committed_manifest.writer(),
            )
            .unwrap();
            let receipt = RepairApplyReceipt::from_draft(
                draft.clone(),
                repair_digest(&draft.canonical_bytes().unwrap()),
            );
            pending.prepare(&receipt)?;
            prepared = Some(receipt);
            Ok(())
        },
    )
    .await
    .unwrap();
    drop(pending);
    let recovered = apply_repair_with_pages(
        &committed.db,
        &committed_store,
        exact_apply(&committed_manifest),
        Some(committed.page_root.path()),
        1_721_000_002,
    )
    .await
    .unwrap();
    assert_eq!(Some(recovered), prepared);

    let mixed = rename_fixture().await;
    let mixed_manifest = prepare(&mixed).await;
    let mixed_store = RepairArtifactStore::new(mixed.repair_root.path().to_path_buf());
    drop(
        mixed_store
            .begin_apply_receipt(mixed_manifest.manifest_id())
            .unwrap(),
    );
    let mut projected_post = mixed.db.get_page("page-a").await.unwrap().unwrap();
    projected_post.title = "Origin Git Workflow Gotchas".to_string();
    projected_post.version += 1;
    KnowledgeProjectionWrite::new(mixed.page_root.path().to_path_buf(), &mixed.db)
        .write_page(&projected_post)
        .unwrap();
    apply_repair_with_pages(
        &mixed.db,
        &mixed_store,
        exact_apply(&mixed_manifest),
        Some(mixed.page_root.path()),
        1_721_000_002,
    )
    .await
    .expect("known projection-post/database-pre state restores then retries");
    assert_eq!(
        mixed.db.get_page("page-a").await.unwrap().unwrap().title,
        "Origin Git Workflow Gotchas"
    );

    let unknown = rename_fixture().await;
    let unknown_manifest = prepare(&unknown).await;
    let unknown_store = RepairArtifactStore::new(unknown.repair_root.path().to_path_buf());
    drop(
        unknown_store
            .begin_apply_receipt(unknown_manifest.manifest_id())
            .unwrap(),
    );
    let filename = target_filename(unknown.page_root.path(), "page-a");
    std::fs::write(
        unknown.page_root.path().join(filename),
        b"unrecognized mixed projection",
    )
    .unwrap();
    let error = apply_repair_with_pages(
        &unknown.db,
        &unknown_store,
        exact_apply(&unknown_manifest),
        Some(unknown.page_root.path()),
        1_721_000_002,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        WenlanError::Conflict(message) if message == "repair_apply_recovery_required"
    ));
    assert!(unknown
        .repair_root
        .path()
        .join(unknown_manifest.manifest_id())
        .join(APPLY_RECEIPT_PENDING_FILE)
        .exists());
}

#[tokio::test]
async fn recovery_clears_a_valid_pending_receipt_when_target_is_exactly_pre_state() {
    let fixture = rename_fixture().await;
    let manifest = prepare(&fixture).await;
    let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());
    let rollback = store.load_rename_page_title_rollback(&manifest).unwrap();
    let mut pending = store.begin_apply_receipt(manifest.manifest_id()).unwrap();
    let interrupted = crate::post_write::rename_page_title_cas(
        &fixture.db,
        &manifest,
        &rollback,
        fixture.page_root.path(),
        |proof| {
            let draft = RepairApplyReceiptDraft::try_new(
                manifest.manifest_id().to_string(),
                manifest.manifest_digest().clone(),
                1_721_000_002,
                proof.before_target_receipt().clone(),
                proof.after_target_receipt().clone(),
                proof.non_target_before().clone(),
                proof.non_target_after().clone(),
                proof.post_apply_db_digest().clone(),
                manifest.allowed_effects().clone(),
                manifest.writer(),
            )
            .unwrap();
            let receipt = RepairApplyReceipt::from_draft(
                draft.clone(),
                repair_digest(&draft.canonical_bytes().unwrap()),
            );
            pending.prepare(&receipt)?;
            Err(WenlanError::Io(std::io::Error::other(
                "forced interruption after durable pending receipt",
            )))
        },
    )
    .await;
    assert!(interrupted.is_err());
    pending.retain().unwrap();
    let pending_path = fixture
        .repair_root
        .path()
        .join(manifest.manifest_id())
        .join(APPLY_RECEIPT_PENDING_FILE);
    assert!(pending_path.is_file());
    assert_eq!(
        fixture.db.get_page("page-a").await.unwrap().unwrap().title,
        "Origin"
    );

    let receipt = apply_repair_with_pages(
        &fixture.db,
        &store,
        exact_apply(&manifest),
        Some(fixture.page_root.path()),
        1_721_000_003,
    )
    .await
    .expect("valid pending receipt plus exact pre-state must clear and retry");

    assert_eq!(receipt.manifest_id(), manifest.manifest_id());
    assert!(!pending_path.exists());
    assert_eq!(
        fixture.db.get_page("page-a").await.unwrap().unwrap().title,
        "Origin Git Workflow Gotchas"
    );
}
