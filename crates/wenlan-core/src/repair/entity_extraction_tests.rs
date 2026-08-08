use super::*;
use crate::{
    db::{tests::test_db, MemoryDB},
    error::WenlanError,
    lint::{
        context::{CancellationToken, LintClock},
        runner::LintRunner,
    },
};
use wenlan_types::{
    lint::{LintProfile, LintQuery},
    repair::{
        ApplyRepairRequest, PrepareRepairRequest, RepairChoice, RepairEnrichmentStep,
        RepairMutation, RepairRollbackPayloadV2, RepairScope, RepairTarget, RepairWriter,
        VerifyRepairRequest,
    },
    RefinementPayload,
};

const OCCURRENCE: &str = "abababababababababababababababababababababababababababababababab";

struct EntityExtractionFixture {
    db: MemoryDB,
    _db_dir: tempfile::TempDir,
    repair_root: tempfile::TempDir,
    request: PrepareRepairRequest,
    review_id: String,
}

async fn entity_extraction_fixture() -> EntityExtractionFixture {
    entity_extraction_fixture_in_space(Some("work")).await
}

/// G6 Stage 2 PR 2c, independent-review finding 2: the same fixture with the
/// memory and its entities UNCATEGORIZED. `memories.space` is NOT NULL as of
/// migration 91, so an unfiled memory carries the reserved sentinel id -- and
/// `entities.space` is NOT NULL for the same reason, so the entity and its
/// shadow page carry it too. Every existing
/// entity-extraction test used a registered space, which is why nothing caught
/// the CAS link INSERT reading the sentinel as a mismatch.
async fn entity_extraction_fixture_in_space(space: Option<&str>) -> EntityExtractionFixture {
    let (db, db_dir) = test_db().await;
    let memory_space = space.unwrap_or(crate::db::UNFILED_SPACE_ID);
    let entity_space = space.unwrap_or(crate::db::UNFILED_SPACE_ID);
    db.test_primary_session()
        .await
        .execute_batch(&format!(
            "INSERT INTO spaces (id,name,created_at,updated_at)
             VALUES ('space-work','work',1,1),('space-personal','personal',1,1);
             INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,space)
             VALUES ('row-entity','Target memory','memory','mem-entity','target',0,10,
                     'text',0,0,'hide','fact','{memory_space}');
             INSERT INTO entities
                 (id,name,entity_type,space,created_at,updated_at)
             VALUES ('ent-existing','Existing','concept','{entity_space}',1,1),
                    ('ent-new','New','concept','{entity_space}',1,1),
                    ('ent-extra','Extra','concept','{entity_space}',1,1);
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES ('mem-entity','ent-existing');
             INSERT INTO enrichment_steps
                 (source_id,step_name,status,error,attempts,updated_at)
             VALUES ('mem-entity','entity_extract','failed','transient',2,1721000000);"
        ))
        .await
        .unwrap();
    // G6 Stage 1.5a review fix round fallout: `entities` is raw-SQL-seeded
    // above (bypassing `store_entity`), so without a shadow page these three
    // entities are invisible to the readers this stage migrated
    // (`link_integrity`, `entity_integrity`, ...) -- `mem-entity`/
    // `ent-existing`'s link would read as broken before this repair ever
    // runs, corrupting the before/after lint comparison in
    // `record_repair_verification`. Backfill the invariant the same way a
    // real write would.
    for entity_id in ["ent-existing", "ent-new", "ent-extra"] {
        db.test_seed_entity_shadow_page(entity_id).await.unwrap();
    }

    let occurrence = RepairDigest::parse(OCCURRENCE).unwrap();
    let review_id = format!("lint_review_{OCCURRENCE}");
    let source_ids = vec!["mem-entity".to_string()];
    let payload = RefinementPayload::LintRepairReview {
        check_id: "memories.enrichment_failures".to_string(),
        occurrence_digest: occurrence.clone(),
        owner_binding_digest: lint_review_owner_binding_digest(&occurrence, &source_ids).unwrap(),
        issue: "Complete the failed entity extraction.".to_string(),
        choices: vec!["link ent-new".to_string()],
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
            None,
            false,
        )
        .await
        .unwrap();
    let request = PrepareRepairRequest::try_new_with_choice(
        RepairLintScope::global(),
        general,
        None,
        RepairChoice::complete_entity_extraction(
            review_id.clone(),
            "mem-entity".to_string(),
            vec!["ent-new".to_string()],
        )
        .unwrap(),
    )
    .unwrap();

    EntityExtractionFixture {
        db,
        _db_dir: db_dir,
        repair_root: tempfile::tempdir().unwrap(),
        request,
        review_id,
    }
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

async fn prepare(fixture: &EntityExtractionFixture) -> RepairManifest {
    prepare_memory_reclassification(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        fixture.request.clone(),
        1_721_000_001,
    )
    .await
    .unwrap()
}

async fn entity_state(db: &MemoryDB) -> (Vec<String>, (String, Option<String>, i64, i64)) {
    let session = db.test_primary_session().await;
    let mut rows = session
        .query(
            "SELECT entity_id FROM memory_entities
             WHERE memory_id='mem-entity' ORDER BY entity_id",
            (),
        )
        .await
        .unwrap();
    let mut entity_ids = Vec::new();
    while let Some(row) = rows.next().await.unwrap() {
        entity_ids.push(row.get::<String>(0).unwrap());
    }
    drop(rows);
    let mut rows = session
        .query(
            "SELECT status,error,attempts,updated_at FROM enrichment_steps
             WHERE source_id='mem-entity' AND step_name='entity_extract'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let step = (
        row.get::<String>(0).unwrap(),
        row.get::<Option<String>>(1).unwrap(),
        row.get::<i64>(2).unwrap(),
        row.get::<i64>(3).unwrap(),
    );
    (entity_ids, step)
}

#[tokio::test]
async fn preparation_binds_review_owner_failed_step_links_and_selected_entities() {
    let fixture = entity_extraction_fixture().await;
    let manifest = prepare(&fixture).await;

    assert_eq!(manifest.writer(), RepairWriter::CompleteEntityExtraction);
    assert!(matches!(
        manifest.target(),
        RepairTarget::MemoryEntityExtraction {
            memory_id,
            step: RepairEnrichmentStep::EntityExtract,
            entity_ids,
            ..
        } if memory_id == "mem-entity" && entity_ids.as_slice() == ["ent-new"]
    ));
    assert!(matches!(
        manifest.mutation(),
        RepairMutation::CompleteEntityExtraction { entity_ids }
            if entity_ids.as_slice() == ["ent-new"]
    ));
    let binding = manifest.source().review_binding().unwrap();
    assert_eq!(binding.review_id(), fixture.review_id);
    assert_eq!(binding.occurrence_digest().as_str(), OCCURRENCE);
    assert_eq!(binding.owner_ids(), ["mem-entity"]);

    let rollback = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf())
        .load_complete_entity_extraction_rollback(&manifest)
        .unwrap();
    let RepairRollbackPayloadV2::CompleteEntityExtraction {
        memory_id,
        before_entity_ids,
        enrichment_status,
        enrichment_error,
        enrichment_attempts,
        enrichment_updated_at,
        ..
    } = rollback
    else {
        panic!("expected entity-extraction rollback");
    };
    assert_eq!(memory_id, "mem-entity");
    assert_eq!(before_entity_ids, ["ent-existing"]);
    assert_eq!(enrichment_status, "failed");
    assert_eq!(enrichment_error.as_deref(), Some("transient"));
    assert_eq!(enrichment_attempts, 2);
    assert_eq!(enrichment_updated_at, 1_721_000_000);
}

#[tokio::test]
async fn apply_preserves_existing_link_adds_approved_link_and_completes_only_selected_step() {
    let fixture = entity_extraction_fixture().await;
    let manifest = prepare(&fixture).await;
    fixture
        .db
        .test_primary_session()
        .await
        .execute(
            "INSERT INTO enrichment_steps
                 (source_id,step_name,status,error,attempts,updated_at)
             VALUES ('mem-entity','classify','failed','keep me',7,1721000000)",
            (),
        )
        .await
        .unwrap();

    let receipt = apply_repair(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        exact_apply(&manifest),
        1_721_000_002,
    )
    .await
    .unwrap();

    assert_eq!(
        entity_state(&fixture.db).await,
        (
            vec!["ent-existing".to_string(), "ent-new".to_string()],
            ("ok".to_string(), None, 2, 1_721_000_000)
        )
    );
    let session = fixture.db.test_primary_session().await;
    let mut rows = session
        .query(
            "SELECT status,error,attempts,updated_at FROM enrichment_steps
             WHERE source_id='mem-entity' AND step_name='classify'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "failed");
    assert_eq!(
        row.get::<Option<String>>(1).unwrap().as_deref(),
        Some("keep me")
    );
    assert_eq!(row.get::<i64>(2).unwrap(), 7);
    assert_eq!(row.get::<i64>(3).unwrap(), 1_721_000_000);
    assert_eq!(
        receipt.before_target_receipt(),
        manifest.expected_state().canonical_receipt()
    );
    assert_ne!(
        receipt.after_target_receipt(),
        receipt.before_target_receipt()
    );
}

/// G6 Stage 2 PR 2c, independent-review finding 2: a complete-entity-extraction
/// repair whose target is UNCATEGORIZED must actually apply. Selection-time
/// validation treats an entity page as unfiled when its space is NULL *or* the
/// reserved sentinel, because the M1 fold stores the sentinel; the CAS link
/// INSERT used to accept only NULL, so the guarded INSERT matched no row, the
/// link was silently never written, and the post-write proof failed the whole
/// repair with `repair_target_write_unproven`. Registered-space coverage cannot
/// see this -- the two predicates agree whenever a real space id is involved.
#[tokio::test]
async fn uncategorized_target_completes_entity_extraction() {
    let fixture = entity_extraction_fixture_in_space(None).await;
    let manifest = prepare(&fixture).await;
    assert_eq!(
        manifest.target().scope(),
        &RepairScope::uncategorized(),
        "the fixture must actually resolve to an uncategorized target"
    );

    apply_repair(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        exact_apply(&manifest),
        1_721_000_002,
    )
    .await
    .expect("an uncategorized complete-entity-extraction repair must apply");

    assert_eq!(
        entity_state(&fixture.db).await,
        (
            vec!["ent-existing".to_string(), "ent-new".to_string()],
            ("ok".to_string(), None, 2, 1_721_000_000)
        ),
        "the approved link must be written and the failed step completed"
    );
}

#[tokio::test]
async fn entity_rollback_uncertainty_retains_pending_receipt() {
    let fixture = entity_extraction_fixture().await;
    let manifest = prepare(&fixture).await;
    let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());

    let error = apply_repair_with_pages_with_forced_rollback_failure(
        &fixture.db,
        &store,
        exact_apply(&manifest),
        None,
        1_721_000_002,
        RepairWriter::CompleteEntityExtraction,
    )
    .await
    .unwrap_err();

    assert!(matches!(
        error,
        WenlanError::Conflict(message) if message == "repair_apply_recovery_required"
    ));
    let manifest_dir = store.manifest_dir(manifest.manifest_id()).unwrap();
    assert!(manifest_dir.join(APPLY_RECEIPT_PENDING_FILE).is_file());
    assert!(!manifest_dir.join(APPLY_RECEIPT_FILE).exists());
    fixture
        .db
        .test_primary_session()
        .await
        .execute("ROLLBACK", ())
        .await
        .unwrap();
    assert_eq!(
        entity_state(&fixture.db).await,
        (
            vec!["ent-existing".to_string()],
            (
                "failed".to_string(),
                Some("transient".to_string()),
                2,
                1_721_000_000,
            ),
        )
    );
}

#[tokio::test]
async fn apply_recovers_empty_original_and_valid_committed_pending_receipts() {
    let empty_fixture = entity_extraction_fixture().await;
    let empty_manifest = prepare(&empty_fixture).await;
    let empty_store = RepairArtifactStore::new(empty_fixture.repair_root.path().to_path_buf());
    drop(
        empty_store
            .begin_apply_receipt(empty_manifest.manifest_id())
            .unwrap(),
    );

    apply_repair(
        &empty_fixture.db,
        &empty_store,
        exact_apply(&empty_manifest),
        1_721_000_002,
    )
    .await
    .expect("an empty marker at the exact original aggregate is retried");
    let empty_manifest_dir = empty_store
        .manifest_dir(empty_manifest.manifest_id())
        .unwrap();
    assert!(
        complete_entity_recovery_unlocked_check_observed(
            empty_manifest.manifest_id(),
            CompleteEntityRecoveryArtifactSite::Remove,
        ),
        "original-state pending removal must run only after the DB mutex is released"
    );
    assert!(!empty_manifest_dir.join(APPLY_RECEIPT_PENDING_FILE).exists());
    assert!(empty_manifest_dir.join(APPLY_RECEIPT_FILE).is_file());

    let committed_fixture = entity_extraction_fixture().await;
    let committed_manifest = prepare(&committed_fixture).await;
    let committed_store =
        RepairArtifactStore::new(committed_fixture.repair_root.path().to_path_buf());
    let rollback = committed_store
        .load_complete_entity_extraction_rollback(&committed_manifest)
        .unwrap();
    let mut pending = committed_store
        .begin_apply_receipt(committed_manifest.manifest_id())
        .unwrap();
    let mut prepared = None;
    crate::post_write::complete_entity_extraction_cas(
        &committed_fixture.db,
        &committed_manifest,
        &rollback,
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

    let recovered = apply_repair(
        &committed_fixture.db,
        &committed_store,
        exact_apply(&committed_manifest),
        1_721_000_002,
    )
    .await
    .expect("a committed aggregate with a valid pending receipt is published");
    assert_eq!(Some(recovered), prepared);
    let committed_manifest_dir = committed_store
        .manifest_dir(committed_manifest.manifest_id())
        .unwrap();
    assert!(
        complete_entity_recovery_unlocked_check_observed(
            committed_manifest.manifest_id(),
            CompleteEntityRecoveryArtifactSite::Publish,
        ),
        "committed pending publication must run only after the DB mutex is released"
    );
    assert!(!committed_manifest_dir
        .join(APPLY_RECEIPT_PENDING_FILE)
        .exists());
    assert!(committed_manifest_dir.join(APPLY_RECEIPT_FILE).is_file());
}

#[tokio::test]
async fn oversized_typed_rollback_is_rejected_before_json_parse() {
    let fixture = entity_extraction_fixture().await;
    let manifest = prepare(&fixture).await;
    let rollback_path = fixture
        .repair_root
        .path()
        .join(manifest.manifest_id())
        .join(manifest.rollback().relative_path());
    std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(rollback_path)
        .unwrap()
        .set_len(REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES + 1)
        .unwrap();

    let error = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf())
        .load_complete_entity_extraction_rollback(&manifest)
        .unwrap_err();

    assert!(matches!(
        error,
        WenlanError::Validation(message) if message == "repair_rollback_artifact_too_large"
    ));
}

#[tokio::test]
async fn aggregate_cas_rejects_every_stale_dimension_without_database_mutation() {
    for stale_case in [
        "link_set",
        "step_changed",
        "step_absent",
        "memory",
        "entity_absent",
        "entity_scope",
        "review_binding",
    ] {
        let fixture = entity_extraction_fixture().await;
        let manifest = prepare(&fixture).await;
        let session = fixture.db.test_primary_session().await;
        match stale_case {
            "link_set" => {
                session
                    .execute(
                        "INSERT INTO memory_entities(memory_id,entity_id)
                     VALUES ('mem-entity','ent-extra')",
                        (),
                    )
                    .await
                    .unwrap();
            }
            "step_changed" => {
                session
                    .execute(
                        "UPDATE enrichment_steps SET attempts=3
                     WHERE source_id='mem-entity' AND step_name='entity_extract'",
                        (),
                    )
                    .await
                    .unwrap();
            }
            "step_absent" => {
                session
                    .execute(
                        "DELETE FROM enrichment_steps
                     WHERE source_id='mem-entity' AND step_name='entity_extract'",
                        (),
                    )
                    .await
                    .unwrap();
            }
            "memory" => {
                session
                    .execute(
                        "UPDATE memories SET title='changed' WHERE source_id='mem-entity'",
                        (),
                    )
                    .await
                    .unwrap();
            }
            // G6 Stage 2 PR 2c item 5/6: the selected-entity guard reads the
            // canonical shadow page, so "absent"/"rescoped" must be simulated
            // there -- deleting or rescoping only the legacy `entities` row
            // leaves the shadow live and the guard passes.
            "entity_absent" => {
                session
                    .execute(
                        "DELETE FROM pages WHERE kind='entity' AND id=(
                             SELECT page_id FROM entity_page_map WHERE entity_id='ent-new')",
                        (),
                    )
                    .await
                    .unwrap();
                session
                    .execute("DELETE FROM entity_page_map WHERE entity_id='ent-new'", ())
                    .await
                    .unwrap();
                session
                    .execute("DELETE FROM entities WHERE id='ent-new'", ())
                    .await
                    .unwrap();
            }
            "entity_scope" => {
                session
                    .execute(
                        "UPDATE pages SET space='personal'
                         WHERE kind='entity' AND id=(
                             SELECT page_id FROM entity_page_map WHERE entity_id='ent-new')",
                        (),
                    )
                    .await
                    .unwrap();
                session
                    .execute(
                        "UPDATE entities SET space='personal' WHERE id='ent-new'",
                        (),
                    )
                    .await
                    .unwrap();
            }
            "review_binding" => {
                session
                    .execute(
                        "UPDATE refinement_queue
                     SET source_ids='[\"ent-new\",\"mem-entity\"]'
                     WHERE id=?1",
                        libsql::params![fixture.review_id.clone()],
                    )
                    .await
                    .unwrap();
            }
            _ => unreachable!(),
        }
        let before = session.repair_database_content_digest().await.unwrap();
        drop(session);

        let result = apply_repair(
            &fixture.db,
            &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
            exact_apply(&manifest),
            1_721_000_002,
        )
        .await;

        assert!(
            matches!(result, Err(WenlanError::Conflict(ref message)) if message == "repair_target_stale"),
            "unexpected {stale_case} result: {result:?}"
        );
        let session = fixture.db.test_primary_session().await;
        assert_eq!(
            session.repair_database_content_digest().await.unwrap(),
            before,
            "stale case {stale_case} mutated the database"
        );
    }
}

#[tokio::test]
async fn insert_update_and_receipt_failures_roll_back_links_and_step_together() {
    for failure in ["insert", "update", "receipt"] {
        let fixture = entity_extraction_fixture().await;
        let manifest = prepare(&fixture).await;
        let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());
        let rollback = store
            .load_complete_entity_extraction_rollback(&manifest)
            .unwrap();
        if failure != "receipt" {
            let sql = match failure {
                "insert" => {
                    "CREATE TRIGGER fail_entity_insert
                     BEFORE INSERT ON memory_entities
                     BEGIN SELECT RAISE(ABORT,'forced insert failure'); END;"
                }
                "update" => {
                    "CREATE TRIGGER fail_step_update
                     BEFORE UPDATE ON enrichment_steps
                     BEGIN SELECT RAISE(ABORT,'forced update failure'); END;"
                }
                _ => unreachable!(),
            };
            fixture
                .db
                .test_primary_session()
                .await
                .execute_batch(sql)
                .await
                .unwrap();
        }

        let result = crate::post_write::complete_entity_extraction_cas(
            &fixture.db,
            &manifest,
            &rollback,
            |_| {
                if failure == "receipt" {
                    Err(WenlanError::Io(std::io::Error::other(
                        "forced receipt failure",
                    )))
                } else {
                    Ok(())
                }
            },
        )
        .await;

        assert!(result.is_err(), "{failure} unexpectedly succeeded");
        assert_eq!(
            entity_state(&fixture.db).await,
            (
                vec!["ent-existing".to_string()],
                (
                    "failed".to_string(),
                    Some("transient".to_string()),
                    2,
                    1_721_000_000
                )
            ),
            "{failure} left a partial aggregate mutation"
        );
    }
}

#[tokio::test]
async fn generic_review_item_cannot_prepare_or_mutate_entity_extraction() {
    let fixture = entity_extraction_fixture().await;
    let occurrence =
        RepairDigest::parse("cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd")
            .unwrap();
    let review_id = format!("lint_review_{}", occurrence.as_str());
    let source_ids = vec!["mem-entity".to_string()];
    let payload = RefinementPayload::LintRepairReview {
        check_id: "pages.links.orphan_labels".to_string(),
        occurrence_digest: occurrence.clone(),
        owner_binding_digest: lint_review_owner_binding_digest(&occurrence, &source_ids).unwrap(),
        issue: "Generic review.".to_string(),
        choices: vec!["keep".to_string()],
        suggested_research_queries: vec![],
    };
    fixture
        .db
        .insert_lint_review_if_absent(
            &review_id,
            &source_ids,
            &serde_json::to_string(&payload).unwrap(),
        )
        .await
        .unwrap();
    let mut value = serde_json::to_value(&fixture.request).unwrap();
    value["choice"]["review_id"] = serde_json::json!(review_id);
    let request: PrepareRepairRequest = serde_json::from_value(value).unwrap();
    let before = entity_state(&fixture.db).await;

    let result = prepare_memory_reclassification(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        request,
        1_721_000_001,
    )
    .await;

    assert!(result.is_err());
    assert_eq!(entity_state(&fixture.db).await, before);
    assert_eq!(
        std::fs::read_dir(fixture.repair_root.path())
            .unwrap()
            .count(),
        0
    );
}

#[tokio::test]
async fn complete_entity_extraction_verifies_end_to_end_with_unrelated_failure_remaining() {
    let mut fixture = entity_extraction_fixture().await;
    fixture
        .db
        .test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO memories
                 (id,content,source,source_id,title,chunk_index,last_modified,chunk_type,
                  pending_revision,is_recap,supersede_mode,memory_type,space)
             VALUES ('row-unrelated','Other memory','memory','mem-unrelated','other',0,10,
                     'text',0,0,'hide','fact','work');
             INSERT INTO enrichment_steps
                 (source_id,step_name,status,error,attempts,updated_at)
             VALUES ('mem-unrelated','entity_extract','failed','unrelated',1,1721000000);",
        )
        .await
        .unwrap();
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &fixture.db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    fixture.request = PrepareRepairRequest::try_new_with_choice(
        RepairLintScope::global(),
        general,
        None,
        fixture.request.choice().clone(),
    )
    .unwrap();
    let manifest = prepare(&fixture).await;
    let store = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf());
    let apply_receipt = apply_repair(&fixture.db, &store, exact_apply(&manifest), 1_721_000_002)
        .await
        .unwrap();
    let post_general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &fixture.db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    assert!(post_general.checks().iter().any(|check| {
        check.check_id() == "memories.enrichment_failures"
            && check.outcome() == wenlan_types::lint::LintOutcome::Finding
    }));

    let receipt = record_repair_verification(
        &fixture.db,
        &store,
        VerifyRepairRequest::try_new_general_only(
            manifest.manifest_id().to_string(),
            manifest.manifest_digest().clone(),
            apply_receipt.receipt_digest().clone(),
            post_general,
        )
        .unwrap(),
        None,
        1_721_000_003,
    )
    .await
    .unwrap();

    assert_eq!(receipt.manifest_id(), manifest.manifest_id());
    assert!(store
        .has_completed_verification(manifest.manifest_id())
        .unwrap());
}

#[tokio::test]
async fn entity_extraction_actionability_is_exact_and_mismatched_targets_fail_closed() {
    let fixture = entity_extraction_fixture().await;
    let manifest = prepare(&fixture).await;
    let before = fixture.db.open_lint_snapshot().await.unwrap();
    assert!(crate::repair_plan::deterministic_target_still_actionable(
        &before,
        manifest.source().lint_scope(),
        None,
        manifest.target(),
        manifest.writer(),
    )
    .await
    .unwrap());
    before.finish().await.unwrap();

    apply_repair(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        exact_apply(&manifest),
        1_721_000_002,
    )
    .await
    .unwrap();
    let after = fixture.db.open_lint_snapshot().await.unwrap();
    assert!(!crate::repair_plan::deterministic_target_still_actionable(
        &after,
        manifest.source().lint_scope(),
        None,
        manifest.target(),
        manifest.writer(),
    )
    .await
    .unwrap());
    let mismatched = RepairTarget::memory(
        "mem-entity".to_string(),
        RepairScope::registered("work".to_string()).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        crate::repair_plan::deterministic_target_still_actionable(
            &after,
            manifest.source().lint_scope(),
            None,
            &mismatched,
            RepairWriter::CompleteEntityExtraction,
        )
        .await,
        Err(WenlanError::Validation(message))
            if message == "repair_target_assertion_unsupported"
    ));
    after.finish().await.unwrap();
}

#[tokio::test]
async fn suppressed_approved_link_insert_rolls_back_step_completion() {
    let fixture = entity_extraction_fixture().await;
    let manifest = prepare(&fixture).await;
    fixture
        .db
        .test_primary_session()
        .await
        .execute_batch(
            "CREATE TRIGGER suppress_approved_entity_link
             BEFORE INSERT ON memory_entities
             WHEN NEW.entity_id='ent-new'
             BEGIN
                 SELECT RAISE(IGNORE);
             END;",
        )
        .await
        .unwrap();
    let before = entity_state(&fixture.db).await;

    let result = apply_repair(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        exact_apply(&manifest),
        1_721_000_002,
    )
    .await;

    assert!(matches!(
        result,
        Err(WenlanError::VectorDb(message))
            if message == "repair_target_write_unproven"
    ));
    assert_eq!(entity_state(&fixture.db).await, before);
}

#[tokio::test]
async fn aggregate_memory_receipt_distinguishes_null_blob_and_embedded_nul_text() {
    let mut fixture = entity_extraction_fixture().await;
    let vector = serde_json::to_string(&vec![0.25_f32; 768]).unwrap();
    fixture
        .db
        .test_primary_session()
        .await
        .execute(
            "UPDATE memories
             SET summary=NULL,
                 structured_fields=CAST(X'610062' AS TEXT),
                 embedding=vector32(?1)
             WHERE source_id='mem-entity'",
            libsql::params![vector],
        )
        .await
        .unwrap();
    let general = LintRunner::new(LintClock::fixed(), CancellationToken::new())
        .run(
            &fixture.db,
            &LintQuery::new(Some(LintProfile::General), None),
            None,
            false,
        )
        .await
        .unwrap();
    fixture.request = PrepareRepairRequest::try_new_with_choice(
        RepairLintScope::global(),
        general,
        None,
        fixture.request.choice().clone(),
    )
    .unwrap();
    let manifest = prepare(&fixture).await;
    let rollback = RepairArtifactStore::new(fixture.repair_root.path().to_path_buf())
        .load_complete_entity_extraction_rollback(&manifest)
        .unwrap();
    let RepairRollbackPayloadV2::CompleteEntityExtraction {
        memory_columns,
        before_memory_row,
        ..
    } = rollback
    else {
        panic!("expected complete entity extraction rollback");
    };
    let value = |column: &str| {
        let index = memory_columns
            .iter()
            .position(|candidate| candidate == column)
            .unwrap();
        before_memory_row[index].as_str()
    };

    assert_eq!(value("summary"), "n:");
    assert_eq!(value("structured_fields"), "t:610062");
    assert!(value("embedding").starts_with("b:"));
    assert!(value("embedding").len() > 2);
}

#[tokio::test]
async fn prepare_rejects_oversized_memory_receipt_without_persisting_manifest() {
    let fixture = entity_extraction_fixture().await;
    fixture
        .db
        .test_primary_session()
        .await
        .execute(
            "UPDATE memories SET source_text=zeroblob(?1)
             WHERE source_id='mem-entity'",
            libsql::params![i64::try_from(REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES + 1).unwrap()],
        )
        .await
        .unwrap();

    let result = prepare_memory_reclassification(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        fixture.request.clone(),
        1_721_000_001,
    )
    .await;

    assert!(matches!(
        result,
        Err(WenlanError::Validation(message))
            if message == "repair_rollback_artifact_too_large"
    ));
    assert_eq!(
        std::fs::read_dir(fixture.repair_root.path())
            .unwrap()
            .count(),
        0
    );
}

#[tokio::test]
async fn prepare_rejects_oversized_link_receipt_without_persisting_manifest() {
    let fixture = entity_extraction_fixture().await;
    let raw_bytes = (REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES / 2) + 1;
    fixture
        .db
        .test_primary_session()
        .await
        .execute_batch(&format!(
            "INSERT INTO entities(id,name,entity_type,space,created_at,updated_at)
             SELECT lower(hex(zeroblob({raw_bytes}))),'Huge','concept','work',1,1;
             INSERT INTO memory_entities(memory_id,entity_id)
             SELECT 'mem-entity',id FROM entities WHERE name='Huge';"
        ))
        .await
        .unwrap();

    let result = prepare_memory_reclassification(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        fixture.request.clone(),
        1_721_000_001,
    )
    .await;

    assert!(matches!(
        result,
        Err(WenlanError::Validation(message))
            if message == "repair_rollback_artifact_too_large"
    ));
    assert_eq!(
        std::fs::read_dir(fixture.repair_root.path())
            .unwrap()
            .count(),
        0
    );
}

#[tokio::test]
async fn memory_preflight_counts_multibyte_and_embedded_nul_text_bytes() {
    let fixture = entity_extraction_fixture().await;
    let session = fixture.db.test_primary_session().await;
    session
        .execute(
            "UPDATE memories SET structured_fields=CAST(X'e7958c0078' AS TEXT)
         WHERE source_id='mem-entity'",
            (),
        )
        .await
        .unwrap();
    let expression = entity_extraction_encoded_length_expression("structured_fields");
    let mut rows = session
        .query(
            &format!("SELECT {expression} FROM memories WHERE source_id='mem-entity'"),
            (),
        )
        .await
        .unwrap();
    let encoded_length = rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap();

    assert_eq!(encoded_length, 12);
}

#[tokio::test]
async fn link_preflight_counts_multibyte_and_embedded_nul_id_bytes() {
    let fixture = entity_extraction_fixture().await;
    let entity_id = "界\0x";
    fixture
        .db
        .test_primary_session()
        .await
        .execute_batch(
            "INSERT INTO entities(id,name,entity_type,space,created_at,updated_at)
             VALUES (CAST(X'e7958c0078' AS TEXT),'Byte ID','concept','work',1,1);
             INSERT INTO memory_entities(memory_id,entity_id)
             VALUES ('mem-entity',CAST(X'e7958c0078' AS TEXT));",
        )
        .await
        .unwrap();
    assert_eq!(entity_id.len(), 5);
    let snapshot = fixture.db.open_lint_snapshot().await.unwrap();

    let result = preflight_entity_extraction_links_on_snapshot(
        &snapshot,
        "mem-entity",
        REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES - 96,
    )
    .await;

    assert!(matches!(
        result,
        Err(WenlanError::Validation(message))
            if message == "repair_rollback_artifact_too_large"
    ));
    snapshot.finish().await.unwrap();
}

#[tokio::test]
async fn prepare_rejects_oversized_multibyte_and_nul_text_without_manifest() {
    for value_sql in [
        format!(
            "replace(hex(zeroblob({})), '00', '界')",
            (REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES / 6) + 1
        ),
        format!(
            "CAST(zeroblob({}) AS TEXT)",
            (REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES / 2) + 1
        ),
    ] {
        let fixture = entity_extraction_fixture().await;
        fixture
            .db
            .test_primary_session()
            .await
            .execute(
                &format!(
                    "UPDATE memories SET structured_fields={value_sql}
                     WHERE source_id='mem-entity'"
                ),
                (),
            )
            .await
            .unwrap();

        let result = prepare_memory_reclassification(
            &fixture.db,
            &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
            fixture.request.clone(),
            1_721_000_001,
        )
        .await;

        assert!(matches!(
            result,
            Err(WenlanError::Validation(message))
                if message == "repair_rollback_artifact_too_large"
        ));
        assert_eq!(
            std::fs::read_dir(fixture.repair_root.path())
                .unwrap()
                .count(),
            0
        );
    }
}

#[tokio::test]
async fn prepare_rejects_oversized_step_error_before_materializing_it() {
    let fixture = entity_extraction_fixture().await;
    fixture
        .db
        .test_primary_session()
        .await
        .execute(
            "UPDATE enrichment_steps SET error=CAST(zeroblob(?1) AS TEXT)
             WHERE source_id='mem-entity' AND step_name='entity_extract'",
            libsql::params![i64::try_from(REPAIR_ROLLBACK_ARTIFACT_MAX_BYTES + 1).unwrap()],
        )
        .await
        .unwrap();

    let result = prepare_memory_reclassification(
        &fixture.db,
        &RepairArtifactStore::new(fixture.repair_root.path().to_path_buf()),
        fixture.request.clone(),
        1_721_000_001,
    )
    .await;

    assert!(matches!(
        result,
        Err(WenlanError::Validation(message))
            if message == "repair_rollback_artifact_too_large"
    ));
    assert_eq!(
        std::fs::read_dir(fixture.repair_root.path())
            .unwrap()
            .count(),
        0
    );
}
