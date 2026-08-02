// SPDX-License-Identifier: Apache-2.0
//! Executable controls for the D2/D3 frontier-policy amendment.

use super::frontier_policy::{
    bind_frontier_groups_to_card, dismiss_card_to_suppression,
    dismiss_card_to_suppression_with_failpoint, ensure_frontier_policy_tables,
    lift_quarantine_to_frontier, quarantine_frontier_group, reconcile_expired_suppressions,
    suppress_frontier_group, CardGroup, DismissFailpoint, LIVE_CARD_BINDING_PREDICATE,
    LIVE_QUARANTINE_PREDICATE, LIVE_SUPPRESSION_PREDICATE, SPACE_RENAME_TABLES,
};

struct TestDb {
    _database: libsql::Database,
    connection: libsql::Connection,
}

impl TestDb {
    async fn new() -> Self {
        let database = libsql::Builder::new_local(":memory:")
            .build()
            .await
            .expect("build in-memory database");
        let connection = database.connect().expect("connect in-memory database");
        let tx = connection
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .expect("begin schema transaction");
        tx.execute_batch(
            "CREATE TABLE genesis_frontier (
                 space TEXT NOT NULL,
                 independence_group_id TEXT NOT NULL,
                 coverage_epoch INTEGER NOT NULL,
                 first_seen_at INTEGER NOT NULL,
                 next_scan_at INTEGER NOT NULL,
                 PRIMARY KEY(space, independence_group_id, coverage_epoch)
             );",
        )
        .await
        .expect("create migration-108 frontier prerequisite");
        ensure_frontier_policy_tables(&tx)
            .await
            .expect("create D2/D3 substrate");
        tx.commit().await.expect("commit schema transaction");
        Self {
            _database: database,
            connection,
        }
    }

    async fn seed_frontier(&self, group: &str, first_seen_at: i64) {
        self.connection
            .execute(
                "INSERT INTO genesis_frontier (
                     space, independence_group_id, coverage_epoch,
                     first_seen_at, next_scan_at
                 ) VALUES ('space-a', ?1, 7, ?2, unixepoch())",
                libsql::params![group, first_seen_at],
            )
            .await
            .expect("seed frontier group");
    }

    async fn live_reason_count(&self, group: &str) -> i64 {
        let sql = format!(
            "SELECT
                 EXISTS(
                     SELECT 1 FROM genesis_frontier
                      WHERE space = 'space-a'
                        AND independence_group_id = ?1
                        AND coverage_epoch = 7
                 )
               + EXISTS(
                     SELECT 1 FROM genesis_card_binding
                      WHERE space = 'space-a'
                        AND independence_group_id = ?1
                        AND coverage_epoch = 7
                        AND {LIVE_CARD_BINDING_PREDICATE}
                 )
               + EXISTS(
                     SELECT 1 FROM genesis_suppression
                      WHERE space = 'space-a'
                        AND independence_group_id = ?1
                        AND coverage_epoch = 7
                        AND {LIVE_SUPPRESSION_PREDICATE}
                 )
               + EXISTS(
                     SELECT 1 FROM genesis_quarantine
                      WHERE space = 'space-a'
                        AND independence_group_id = ?1
                        AND coverage_epoch = 7
                        AND {LIVE_QUARANTINE_PREDICATE}
                 )"
        );
        scalar(&self.connection, &sql, libsql::params![group]).await
    }
}

async fn scalar(
    connection: &libsql::Connection,
    sql: &str,
    params: impl libsql::params::IntoParams,
) -> i64 {
    let mut rows = connection.query(sql, params).await.expect("scalar query");
    rows.next()
        .await
        .expect("read scalar row")
        .expect("scalar row present")
        .get(0)
        .expect("scalar integer")
}

#[tokio::test]
async fn suppression_lapses_before_frontier_restore_and_can_repeat() {
    let db = TestDb::new().await;
    db.seed_frontier("group-a", 123).await;

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    suppress_frontier_group(&tx, "space-a", "group-a", 7, "m6p_topic", "dismissed")
        .await
        .expect("first suppression");
    tx.commit().await.unwrap();
    assert_eq!(db.live_reason_count("group-a").await, 1);

    db.connection
        .execute(
            "UPDATE genesis_suppression
                SET suppressed_at = unixepoch() - 15552001,
                    expires_at = unixepoch() - 1
              WHERE space = 'space-a'
                AND independence_group_id = 'group-a'
                AND coverage_epoch = 7
                AND lapsed_at IS NULL",
            (),
        )
        .await
        .expect("age first suppression past expiry");

    assert_eq!(
        db.live_reason_count("group-a").await,
        1,
        "wall-clock expiry alone must not create a zero-reason window"
    );

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    assert_eq!(
        reconcile_expired_suppressions(&tx, "space-a", 7)
            .await
            .expect("lapse then restore frontier"),
        1
    );
    tx.commit().await.unwrap();
    assert_eq!(db.live_reason_count("group-a").await, 1);
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT first_seen_at FROM genesis_frontier
              WHERE space = 'space-a'
                AND independence_group_id = 'group-a'
                AND coverage_epoch = 7",
            (),
        )
        .await,
        123,
        "F11 must preserve the original within-epoch surfacing clock"
    );

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    suppress_frontier_group(&tx, "space-a", "group-a", 7, "m6p_topic", "dismissed-again")
        .await
        .expect("repeat suppression in the same epoch");
    tx.commit().await.unwrap();

    assert_eq!(db.live_reason_count("group-a").await, 1);
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT COUNT(*) FROM genesis_suppression
              WHERE space = 'space-a'
                AND independence_group_id = 'group-a'
                AND coverage_epoch = 7
                AND page_id = 'm6p_topic'",
            (),
        )
        .await,
        2,
        "both page-ID suppression identities must remain queryable"
    );
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT COUNT(*) FROM genesis_suppression
              WHERE space = 'space-a'
                AND independence_group_id = 'group-a'
                AND coverage_epoch = 7
                AND lapsed_at IS NULL",
            (),
        )
        .await,
        1
    );
}

#[tokio::test]
async fn the_partial_unique_indexes_reject_second_live_reasons() {
    let db = TestDb::new().await;
    db.connection
        .execute(
            "INSERT INTO genesis_suppression (
                 space, independence_group_id, coverage_epoch, page_id, reason,
                 first_seen_at, suppressed_at, expires_at
             ) VALUES ('space-a', 'group-a', 7, 'm6p_a', 'first', 10,
                       unixepoch(), unixepoch() + 15552000)",
            (),
        )
        .await
        .expect("first live suppression");
    let duplicate_suppression = db
        .connection
        .execute(
            "INSERT INTO genesis_suppression (
                 space, independence_group_id, coverage_epoch, page_id, reason,
                 first_seen_at, suppressed_at, expires_at
             ) VALUES ('space-a', 'group-a', 7, 'm6p_a', 'second', 10,
                       unixepoch() + 1, unixepoch() + 15552001)",
            (),
        )
        .await;
    assert!(
        duplicate_suppression.is_err(),
        "at most one unlapsed suppression may exist per group/epoch"
    );
    let blank_page_identity = db
        .connection
        .execute(
            "INSERT INTO genesis_suppression (
                 space, independence_group_id, coverage_epoch, page_id, reason,
                 first_seen_at, suppressed_at, expires_at
             ) VALUES ('space-a', 'group-page-id', 7, '  ', 'dismissed', 10,
                       unixepoch(), unixepoch() + 15552000)",
            (),
        )
        .await;
    assert!(
        blank_page_identity.is_err(),
        "suppression identity must be a non-empty page ID"
    );

    db.connection
        .execute(
            "INSERT INTO genesis_card_binding (
                 space, independence_group_id, coverage_epoch, card_id, page_id,
                 first_seen_at, created_at
             ) VALUES ('space-a', 'group-b', 7, 'card-a', 'm6p_b', 20, unixepoch())",
            (),
        )
        .await
        .expect("first live card binding");
    let duplicate_binding = db
        .connection
        .execute(
            "INSERT INTO genesis_card_binding (
                 space, independence_group_id, coverage_epoch, card_id, page_id,
                 first_seen_at, created_at
             ) VALUES ('space-a', 'group-b', 7, 'card-b', 'm6p_b', 20, unixepoch())",
            (),
        )
        .await;
    assert!(
        duplicate_binding.is_err(),
        "at most one open card binding may exist per group/epoch"
    );
}

#[tokio::test]
async fn shared_card_dismissal_is_atomic_at_every_post_card_boundary() {
    let db = TestDb::new().await;
    db.seed_frontier("group-a", 101).await;
    db.seed_frontier("group-b", 202).await;

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    bind_frontier_groups_to_card(
        &tx,
        "space-a",
        7,
        "card-shared",
        &[
            CardGroup::new("group-a", "m6p_a"),
            CardGroup::new("group-b", "m6p_b"),
        ],
    )
    .await
    .expect("bind both groups to one card");
    tx.commit().await.unwrap();
    assert_eq!(db.live_reason_count("group-a").await, 1);
    assert_eq!(db.live_reason_count("group-b").await, 1);

    for failpoint in [
        DismissFailpoint::AfterBindingsClosed,
        DismissFailpoint::AfterSuppression(1),
        DismissFailpoint::AfterSuppression(2),
    ] {
        let tx = db
            .connection
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .unwrap();
        let failure = dismiss_card_to_suppression_with_failpoint(
            &tx,
            "card-shared",
            "card-dismissed",
            failpoint,
        )
        .await;
        assert!(failure.is_err(), "failpoint {failpoint:?} must abort");
        tx.rollback().await.expect("roll back injected dismissal");

        assert_eq!(
            db.live_reason_count("group-a").await,
            1,
            "group-a lost or doubled its reason at {failpoint:?}"
        );
        assert_eq!(
            db.live_reason_count("group-b").await,
            1,
            "group-b lost or doubled its reason at {failpoint:?}"
        );
        assert_eq!(
            scalar(
                &db.connection,
                "SELECT COUNT(*) FROM genesis_card_binding
                  WHERE card_id = 'card-shared' AND closed_at IS NULL",
                (),
            )
            .await,
            2,
            "an injected failure must retain every live binding"
        );
    }

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    assert_eq!(
        dismiss_card_to_suppression(&tx, "card-shared", "card-dismissed")
            .await
            .expect("dismiss shared card"),
        2
    );
    tx.commit().await.unwrap();

    assert_eq!(db.live_reason_count("group-a").await, 1);
    assert_eq!(db.live_reason_count("group-b").await, 1);
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT COUNT(*) FROM genesis_card_binding
              WHERE card_id = 'card-shared' AND closed_at IS NOT NULL",
            (),
        )
        .await,
        2,
        "one dismissal must close every retained per-group binding"
    );
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT COUNT(*) FROM genesis_suppression
              WHERE reason = 'card-dismissed' AND lapsed_at IS NULL",
            (),
        )
        .await,
        2,
        "every closed binding must receive its post-card live reason"
    );
}

#[tokio::test]
async fn quarantine_lift_retains_history_and_allows_reactivation() {
    assert_eq!(
        SPACE_RENAME_TABLES,
        [
            "genesis_suppression",
            "genesis_card_binding",
            "genesis_quarantine"
        ]
    );
    let db = TestDb::new().await;
    db.seed_frontier("group-a", 303).await;

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    quarantine_frontier_group(&tx, "space-a", "group-a", 7, "policy violation")
        .await
        .expect("first quarantine");
    tx.commit().await.unwrap();
    assert_eq!(db.live_reason_count("group-a").await, 1);

    let duplicate = db
        .connection
        .execute(
            "INSERT INTO genesis_quarantine (
                 space, independence_group_id, coverage_epoch, reason,
                 first_seen_at, quarantined_at
             ) VALUES ('space-a', 'group-a', 7, 'duplicate', 303, unixepoch() + 1)",
            (),
        )
        .await;
    assert!(
        duplicate.is_err(),
        "a second live quarantine must be rejected"
    );

    let blank_reason = db
        .connection
        .execute(
            "INSERT INTO genesis_quarantine (
                 space, independence_group_id, coverage_epoch, reason,
                 first_seen_at, quarantined_at
             ) VALUES ('space-a', 'group-b', 7, '  ', 404, unixepoch())",
            (),
        )
        .await;
    assert!(blank_reason.is_err(), "quarantine reason must be explicit");

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    assert_eq!(
        lift_quarantine_to_frontier(&tx, "space-a", "group-a", 7)
            .await
            .expect("lift quarantine into frontier"),
        1
    );
    tx.commit().await.unwrap();
    assert_eq!(db.live_reason_count("group-a").await, 1);
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT first_seen_at FROM genesis_frontier
              WHERE space = 'space-a'
                AND independence_group_id = 'group-a'
                AND coverage_epoch = 7",
            (),
        )
        .await,
        303
    );

    let tx = db
        .connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .unwrap();
    quarantine_frontier_group(&tx, "space-a", "group-a", 7, "repeat violation")
        .await
        .expect("repeat quarantine");
    tx.commit().await.unwrap();

    assert_eq!(db.live_reason_count("group-a").await, 1);
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT COUNT(*) FROM genesis_quarantine
              WHERE space = 'space-a'
                AND independence_group_id = 'group-a'
                AND coverage_epoch = 7",
            (),
        )
        .await,
        2,
        "lifted quarantine history must remain queryable"
    );
    assert_eq!(
        scalar(
            &db.connection,
            "SELECT COUNT(*) FROM genesis_quarantine
              WHERE space = 'space-a'
                AND independence_group_id = 'group-a'
                AND coverage_epoch = 7
                AND lifted_at IS NULL",
            (),
        )
        .await,
        1
    );
}
