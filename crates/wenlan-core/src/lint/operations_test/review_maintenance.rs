use super::{check, metric, run, IMPORTS, MAINTENANCE, NOW, REFINEMENTS, REJECTIONS};
use crate::db::tests::test_db;
use crate::lint::test_support::assert_no_privacy_canaries;
use sha2::{Digest as _, Sha256};
use wenlan_types::lint::{LintMetricCode, LintOutcome};

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

#[tokio::test]
async fn terminal_imports_and_every_review_state_are_inventory_or_findings() {
    // Given
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    for (index, stage) in ["parsing", "stage_a", "stage_b", "done", "error"]
        .iter()
        .enumerate()
    {
        conn.execute(
            "INSERT INTO import_state
             (id,vendor,source_path,total_conversations,processed_conversations,stage,error_message,started_at,updated_at)
             VALUES (?1,'claude','/Users/canary/CANARY_HOME_7f31',10,?2,?3,?4,'2023-11-14T22:13:20Z','2023-11-14T22:13:20Z')",
            libsql::params![format!("import-{index}"), i64::try_from(index).unwrap(), *stage,
                (*stage == "error").then_some("nested error at /private/CANARY_ERROR_7f31")],
        ).await.unwrap();
    }
    for (index, status) in [
        "pending",
        "awaiting_review",
        "auto_applied",
        "resolved",
        "dismissed",
    ]
    .iter()
    .enumerate()
    {
        conn.execute(
            "INSERT INTO refinement_queue (id,action,source_ids,status,created_at) VALUES (?1,'detect_contradiction','[\"left\",\"right\"]',?2,'2023-11-14 22:13:20')",
            libsql::params![format!("ref-{index}"), *status],
        ).await.unwrap();
    }
    conn.execute(
        "INSERT INTO rejected_memories (id,content,rejection_reason,rejection_detail,created_at)
         VALUES ('rej','CANARY_MEMORY_CONTENT_7f31','noise_pattern','CANARY_SESSION_SUMMARY_7f31',?1)",
        libsql::params![NOW],
    ).await.unwrap();
    drop(conn);

    // When
    let report = run(&db, &[]).await;

    // Then
    assert_eq!(check(&report, IMPORTS).outcome(), LintOutcome::Finding);
    assert_eq!(
        metric(
            check(&report, IMPORTS),
            LintMetricCode::OperationTerminalFailures
        ),
        1
    );
    assert_eq!(
        metric(
            check(&report, REFINEMENTS),
            LintMetricCode::OperationPending
        ),
        1
    );
    assert_eq!(
        metric(
            check(&report, REFINEMENTS),
            LintMetricCode::OperationAwaitingReview
        ),
        1
    );
    assert_eq!(
        metric(
            check(&report, REFINEMENTS),
            LintMetricCode::OperationTerminal
        ),
        3
    );
    assert_eq!(
        metric(
            check(&report, REFINEMENTS),
            LintMetricCode::OperationInvalidStates
        ),
        0
    );
    assert_eq!(check(&report, REJECTIONS).outcome(), LintOutcome::Pass);
    assert_eq!(
        metric(check(&report, REJECTIONS), LintMetricCode::ObservedRecords),
        1
    );
    assert_no_privacy_canaries(&serde_json::to_string(&report).unwrap());
}

#[tokio::test]
async fn refinement_actions_enforce_closed_set_and_source_id_shape() {
    let (db, _tmp) = test_db().await;
    let conn = db.test_primary_session().await;
    for (id, action, source_ids, status) in [
        (
            "valid-binary",
            "entity_merge",
            "[\"new\",\"existing\"]",
            "awaiting_review",
        ),
        (
            "valid-page",
            "page_keep_or_archive",
            "[\"page\"]",
            "resolved",
        ),
        (
            "lint_review_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "lint_repair_review",
            "[\"one\"]",
            "awaiting_review",
        ),
        (
            "lint_review_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "lint_repair_review",
            "[\"one\",\"two\"]",
            "resolved",
        ),
        (
            "lint_review_cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "lint_repair_review",
            "[\"one\",\"three\",\"two\"]",
            "dismissed",
        ),
        (
            "lint_review_dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "lint_repair_review",
            "[\"one\"]",
            "pending",
        ),
        (
            "lint_review_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "lint_repair_review",
            "[\"one\"]",
            "auto_applied",
        ),
        ("malformed", "page_merge", "not-json", "awaiting_review"),
        ("unknown", "invented_action", "[\"a\",\"b\"]", "pending"),
        (
            "bad-cardinality",
            "detect_contradiction",
            "[\"only\"]",
            "pending",
        ),
        (
            "blank-id",
            "cross_space_discovery",
            "[\"a\",\"\"]",
            "awaiting_review",
        ),
        (
            "bad-dedup-state",
            "dedup_merge",
            "[\"a\",\"b\"]",
            "resolved",
        ),
        ("bad-suggest-state", "suggest_entity", "[\"a\"]", "resolved"),
        (
            "bad-consolidate-state",
            "consolidate_duplicate",
            "[\"a\",\"b\"]",
            "awaiting_review",
        ),
    ] {
        let payload = (action == "lint_repair_review").then(|| {
            let digest = id.strip_prefix("lint_review_").unwrap();
            let parsed_source_ids = serde_json::from_str::<Vec<String>>(source_ids).unwrap();
            serde_json::json!({
                "action": "lint_repair_review",
                "check_id": "pages.links.orphan_labels",
                "occurrence_digest": digest,
                "owner_binding_digest": owner_binding_digest(digest, &parsed_source_ids),
                "issue": "Review the ambiguous target.",
                "choices": ["keep", "retarget", "remove"],
                "suggested_research_queries": [],
            })
            .to_string()
        });
        conn.execute(
            "INSERT INTO refinement_queue (id,action,source_ids,payload,status,created_at)
             VALUES (?1,?2,?3,?4,?5,'2023-11-14 22:13:20')",
            libsql::params![id, action, source_ids, payload, status],
        )
        .await
        .unwrap();
    }
    drop(conn);

    let report = run(&db, &[]).await;
    let result = check(&report, REFINEMENTS);
    assert_eq!(metric(result, LintMetricCode::ObservedRecords), 14);
    assert_eq!(metric(result, LintMetricCode::AffectedRecords), 9);
    assert_eq!(metric(result, LintMetricCode::OperationInvalidStates), 9);
}

#[tokio::test]
async fn import_affected_records_count_unique_rows() {
    let (db, _tmp) = test_db().await;
    db.test_primary_session().await.execute(
        "INSERT INTO import_state
         (id,vendor,source_path,processed_conversations,stage,error_message,started_at,updated_at)
         VALUES ('one','invalid-vendor','opaque',-1,'error','failure','2023-11-14T22:13:20Z','2023-11-14T22:13:20Z')",
        libsql::params::Params::None,
    ).await.unwrap();

    let report = run(&db, &[]).await;
    let result = check(&report, IMPORTS);
    assert_eq!(metric(result, LintMetricCode::ObservedRecords), 1);
    assert_eq!(metric(result, LintMetricCode::AffectedRecords), 1);
    assert_eq!(metric(result, LintMetricCode::OperationTerminalFailures), 1);
    assert_eq!(metric(result, LintMetricCode::OperationInvalidStates), 1);
}

#[tokio::test]
async fn durable_failure_oracle_finds_no_progress_and_absence_is_a_limitation() {
    // Given
    let (db, _tmp) = test_db().await;

    // When
    let limited = run(&db, &[]).await;
    db.test_primary_session().await.execute_batch(
        "INSERT INTO app_metadata(key,value) VALUES
         ('reconcile_frontier_docs','{\"ts\":1,\"id\":\"opaque\",\"chunk\":0,\"stuck_id\":\"private-host-7f31.invalid\",\"failures\":2}'),
         ('compile_queue_depth_v1','7'),
         ('last_daily_steep_ts','1699996400');"
    ).await.unwrap();
    let proven = run(&db, &[]).await;

    // Then
    let limitation = check(&limited, MAINTENANCE);
    assert_eq!(limitation.outcome(), LintOutcome::Pass);
    assert_eq!(
        metric(limitation, LintMetricCode::OperationMissingProgressOracles),
        2
    );
    let finding = check(&proven, MAINTENANCE);
    assert_eq!(finding.outcome(), LintOutcome::Finding);
    assert_eq!(
        metric(finding, LintMetricCode::OperationDurableNoProgress),
        1
    );
    assert_eq!(metric(finding, LintMetricCode::PendingRecords), 7);
    assert_no_privacy_canaries(&serde_json::to_string(&proven).unwrap());
}
