use super::{check, metric, run_at, REFINEMENTS};
use crate::db::tests::test_db;
use crate::synthesis::refinement_queue::process_refinement_queue;
use sha2::{Digest as _, Sha256};
use wenlan_types::lint::{LintMetricCode, LintOutcome};

async fn process(db: &crate::db::MemoryDB) -> usize {
    process_refinement_queue(
        db,
        None,
        &crate::prompts::PromptRegistry::default(),
        &crate::tuning::RefineryConfig::default(),
    )
    .await
    .unwrap()
}

fn owner_binding_digest(digest: &str, source_ids: &[String]) -> String {
    Sha256::digest(
        serde_json::to_vec(&serde_json::json!({
            "occurrence_digest": digest,
            "source_ids": source_ids,
        }))
        .unwrap(),
    )
    .iter()
    .map(|byte| format!("{byte:02x}"))
    .collect()
}

fn lint_review_payload(digest: &str, source_ids: &[String]) -> String {
    serde_json::json!({
        "action": "lint_repair_review",
        "check_id": "pages.links.orphan_labels",
        "occurrence_digest": digest,
        "owner_binding_digest": owner_binding_digest(digest, source_ids),
        "issue": "Review the ambiguous target.",
        "choices": ["keep", "retarget", "remove"],
        "suggested_research_queries": [],
    })
    .to_string()
}

#[tokio::test]
async fn daemon_dismissed_dedup_merge_is_valid_inventory() {
    // Given
    let (db, _tmp) = test_db().await;
    db.insert_refinement_proposal(
        "dedup",
        "dedup_merge",
        &["left".into(), "right".into()],
        None,
        0.8,
    )
    .await
    .unwrap();

    // When
    assert_eq!(process(&db).await, 1);
    let report = run_at(&db, &[], chrono::Utc::now().timestamp()).await;

    // Then
    assert_eq!(
        db.get_refinement_proposal("dedup")
            .await
            .unwrap()
            .unwrap()
            .status,
        "dismissed"
    );
    assert_eq!(check(&report, REFINEMENTS).outcome(), LintOutcome::Pass);
}

#[tokio::test]
async fn daemon_promoted_suggest_entity_is_valid_inventory() {
    // Given
    let (db, _tmp) = test_db().await;
    db.insert_refinement_proposal(
        "suggest",
        "suggest_entity",
        &["memory".into()],
        Some("Entity"),
        0.8,
    )
    .await
    .unwrap();

    // When
    assert_eq!(process(&db).await, 1);
    let report = run_at(&db, &[], chrono::Utc::now().timestamp()).await;

    // Then
    assert_eq!(
        db.get_refinement_proposal("suggest")
            .await
            .unwrap()
            .unwrap()
            .status,
        "awaiting_review"
    );
    assert_eq!(check(&report, REFINEMENTS).outcome(), LintOutcome::Pass);
}

#[tokio::test]
async fn lint_repair_review_cards_are_valid_inventory_for_nonempty_source_sets() {
    // Given
    let (db, _tmp) = test_db().await;
    for (index, mut source_ids) in [
        vec!["one".to_string()],
        vec!["one".to_string(), "two".to_string()],
        vec!["one".to_string(), "two".to_string(), "three".to_string()],
    ]
    .into_iter()
    .enumerate()
    {
        source_ids.sort();
        let digest = format!("{:064x}", index + 1);
        let payload = lint_review_payload(&digest, &source_ids);
        assert!(db
            .insert_lint_review_if_absent(&format!("lint_review_{digest}"), &source_ids, &payload,)
            .await
            .unwrap());
    }

    // When
    let report = run_at(&db, &[], chrono::Utc::now().timestamp()).await;

    // Then
    let result = check(&report, REFINEMENTS);
    assert_eq!(result.outcome(), LintOutcome::Pass);
    assert_eq!(metric(result, LintMetricCode::ObservedRecords), 3);
    assert_eq!(metric(result, LintMetricCode::OperationAwaitingReview), 3);
    assert_eq!(metric(result, LintMetricCode::AffectedRecords), 0);
    assert_eq!(metric(result, LintMetricCode::OperationInvalidStates), 0);
}

#[tokio::test]
async fn malformed_lint_repair_review_contracts_are_findings() {
    // Given
    let (db, _tmp) = test_db().await;
    let digest_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let digest_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let digest_c = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    let digest_d = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    let digest_e = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    let digest_f = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    let one_owner = vec!["owner".to_string()];
    let rows = vec![
        (
            "lint-review-invalid".to_string(),
            "[\"owner\"]".to_string(),
            lint_review_payload(digest_a, &one_owner),
        ),
        (
            format!("lint_review_{digest_b}"),
            "[\"owner\"]".to_string(),
            "not-json".to_string(),
        ),
        (
            format!("lint_review_{digest_c}"),
            "[\"owner\"]".to_string(),
            lint_review_payload(digest_d, &one_owner),
        ),
        (
            format!("lint_review_{digest_e}"),
            "[\" owner \"]".to_string(),
            serde_json::json!({
                "action": "lint_repair_review",
                "check_id": "pages.links.orphan_labels",
                "occurrence_digest": digest_e,
                "owner_binding_digest": digest_a,
                "issue": "Review the ambiguous target.",
                "choices": ["keep", "retarget", "remove"],
                "suggested_research_queries": [],
            })
            .to_string(),
        ),
        (
            format!("lint_review_{digest_f}"),
            "[\"owner\"]".to_string(),
            serde_json::json!({
                "action": "lint_repair_review",
                "check_id": "pages.links.orphan_labels",
                "occurrence_digest": digest_f,
                "issue": "Review the ambiguous target.",
                "choices": ["keep", "retarget", "remove"],
                "suggested_research_queries": [],
            })
            .to_string(),
        ),
        (
            format!("lint_review_{:064x}", 17),
            "[\"owner\"]".to_string(),
            serde_json::json!({
                "action": "lint_repair_review",
                "check_id": "pages.links.orphan_labels",
                "occurrence_digest": format!("{:064x}", 17),
                "owner_binding_digest": digest_a,
                "issue": "Review the ambiguous target.",
                "choices": ["keep", "retarget", "remove"],
                "suggested_research_queries": [],
            })
            .to_string(),
        ),
        (
            format!("lint_review_{:064x}", 18),
            "[\"two\",\"one\"]".to_string(),
            lint_review_payload(
                &format!("{:064x}", 18),
                &["one".to_string(), "two".to_string()],
            ),
        ),
        (
            format!("lint_review_{:064x}", 19),
            "[\"owner\",\"owner\"]".to_string(),
            lint_review_payload(&format!("{:064x}", 19), &one_owner),
        ),
    ];
    let conn = db.test_primary_session().await;
    for (id, source_ids, payload) in rows {
        conn.execute(
            "INSERT INTO refinement_queue
                 (id,action,source_ids,payload,status,created_at)
             VALUES (?1,'lint_repair_review',?2,?3,'awaiting_review','2023-11-14 22:13:20')",
            libsql::params![id, source_ids, payload],
        )
        .await
        .unwrap();
    }
    drop(conn);

    // When
    let report = run_at(&db, &[], chrono::Utc::now().timestamp()).await;

    // Then
    let result = check(&report, REFINEMENTS);
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(metric(result, LintMetricCode::ObservedRecords), 8);
    assert_eq!(metric(result, LintMetricCode::AffectedRecords), 8);
    assert_eq!(metric(result, LintMetricCode::OperationInvalidStates), 8);
}

#[tokio::test]
async fn unexecutable_consolidate_duplicate_pending_is_a_finding() {
    // Given
    let (db, _tmp) = test_db().await;
    db.insert_refinement_proposal(
        "consolidate",
        "consolidate_duplicate",
        &["incoming".into(), "duplicate".into()],
        None,
        0.85,
    )
    .await
    .unwrap();

    // When
    assert_eq!(process(&db).await, 0);
    let report = run_at(&db, &[], chrono::Utc::now().timestamp()).await;

    // Then
    assert_eq!(
        db.get_refinement_proposal("consolidate")
            .await
            .unwrap()
            .unwrap()
            .status,
        "pending"
    );
    let result = check(&report, REFINEMENTS);
    assert_eq!(result.outcome(), LintOutcome::Finding);
    assert_eq!(metric(result, LintMetricCode::OperationPending), 1);
    assert_eq!(metric(result, LintMetricCode::OperationInvalidStates), 1);
}
