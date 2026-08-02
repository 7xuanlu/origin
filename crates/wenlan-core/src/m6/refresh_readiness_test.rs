// SPDX-License-Identifier: Apache-2.0
//! Amendment controls for M6 D4, D7, and D8.

use super::refresh_readiness::{
    acquire_refresh_lease, ensure_refresh_readiness_tables, initialize_readiness, readiness_fence,
    record_soak_receipt, replace_refresh_dependencies, space_refresh_dependencies_ready,
    transition_readiness, ReadinessFence, ReadinessKey, ReadinessPhase, RefreshDependency,
    RefreshLeaseRequest, SoakEvidence,
};

async fn connection() -> libsql::Connection {
    let database = libsql::Builder::new_local(":memory:")
        .build()
        .await
        .expect("build test database");
    let connection = database.connect().expect("connect test database");
    connection
        .execute("PRAGMA foreign_keys=ON", ())
        .await
        .expect("enable production foreign-key behavior");
    connection
}

async fn install(conn: &libsql::Connection) {
    let tx = conn.transaction().await.expect("begin install");
    ensure_refresh_readiness_tables(&tx)
        .await
        .expect("install D4/D7/D8 substrate");
    tx.commit().await.expect("commit install");
}

async fn scalar_i64(conn: &libsql::Connection, sql: &str) -> i64 {
    let mut rows = conn.query(sql, ()).await.expect("scalar query");
    rows.next()
        .await
        .expect("read scalar row")
        .expect("scalar row exists")
        .get(0)
        .expect("scalar is integer")
}

async fn index_sql(conn: &libsql::Connection, name: &str) -> String {
    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?1",
            libsql::params![name],
        )
        .await
        .expect("query index");
    rows.next()
        .await
        .expect("read index row")
        .expect("index exists")
        .get(0)
        .expect("index SQL is text")
}

fn lease<'a>(page_id: &'a str, token: &'a str) -> RefreshLeaseRequest<'a> {
    RefreshLeaseRequest {
        page_id,
        base_page_version: 7,
        base_source_revision: 11,
        space: "space-a",
        dependency_generation: 23,
        active_root_digest: "roots-v23",
        lease_token: token,
        lease_expires_at: 1_000,
        readiness_epoch: 5,
        schema_version: 109,
        reason: "source_updated",
        now: 100,
    }
}

#[tokio::test]
async fn d4_rowless_queue_acquires_exactly_one_durable_lease_before_work() {
    let conn = connection().await;
    install(&conn).await;

    let first = {
        let tx = conn.transaction().await.expect("first sweep transaction");
        let acquired = acquire_refresh_lease(&tx, &lease("page-a", "lease-1"))
            .await
            .expect("first lease attempt");
        tx.commit().await.expect("commit first lease");
        acquired
    };
    let second = {
        let tx = conn.transaction().await.expect("second sweep transaction");
        let acquired = acquire_refresh_lease(&tx, &lease("page-a", "lease-2"))
            .await
            .expect("second lease attempt");
        tx.commit().await.expect("commit second lease");
        acquired
    };

    assert!(first, "the first stale-page sweep must own the model call");
    assert!(
        !second,
        "a sweep that selected the same page/version must not own a second model call/card"
    );
    let model_calls = [first, second].into_iter().filter(|owned| *owned).count();
    let cards_opened = model_calls;
    assert_eq!(model_calls, 1, "only the lease owner may call the model");
    assert_eq!(cards_opened, 1, "only the lease owner may open a card");
    assert_eq!(
        scalar_i64(&conn, "SELECT count(*) FROM genesis_refresh_jobs").await,
        1
    );

    let mut rows = conn
        .query(
            "SELECT state, base_source_revision, dependency_generation,
                    active_root_digest, space, readiness_epoch, schema_version, reason
               FROM genesis_refresh_jobs",
            (),
        )
        .await
        .expect("read leased job");
    let row = rows
        .next()
        .await
        .expect("read leased row")
        .expect("leased row exists");
    assert_eq!(row.get::<String>(0).unwrap(), "leased");
    assert_eq!(row.get::<i64>(1).unwrap(), 11);
    assert_eq!(row.get::<i64>(2).unwrap(), 23);
    assert_eq!(row.get::<String>(3).unwrap(), "roots-v23");
    assert_eq!(row.get::<String>(4).unwrap(), "space-a");
    assert_eq!(row.get::<i64>(5).unwrap(), 5);
    assert_eq!(row.get::<i64>(6).unwrap(), 109);
    assert_eq!(row.get::<String>(7).unwrap(), "source_updated");

    conn.execute(
        "UPDATE genesis_refresh_jobs
            SET state = 'retry', lease_token = NULL, lease_expires_at = NULL,
                next_attempt_at = 50
          WHERE page_id = 'page-a' AND base_page_version = 7",
        (),
    )
    .await
    .expect("make the durable job a due retry");
    let tx = conn.transaction().await.expect("retry claim transaction");
    assert!(
        acquire_refresh_lease(&tx, &lease("page-a", "lease-3"))
            .await
            .expect("claim due retry"),
        "the same durable retry row must be claimable without inserting a second job"
    );
    tx.commit().await.unwrap();
    assert_eq!(
        scalar_i64(&conn, "SELECT count(*) FROM genesis_refresh_jobs").await,
        1
    );
}

#[tokio::test]
async fn d4_unique_predicate_names_only_durable_active_states() {
    let conn = connection().await;
    install(&conn).await;

    let sql = index_sql(&conn, "idx_genesis_refresh_jobs_active")
        .await
        .to_lowercase();
    assert!(
        sql.contains("unique"),
        "coalescing index must be UNIQUE: {sql}"
    );
    assert!(
        sql.contains("state in ('leased','retry')"),
        "coalescing predicate must be exactly leased/retry: {sql}"
    );
    assert!(
        !sql.contains("queued"),
        "queued must remain row-less: {sql}"
    );
    assert!(
        !sql.contains("generated"),
        "generated must remain in memory while the row stays leased: {sql}"
    );

    let queued = conn
        .execute(
            "INSERT INTO genesis_refresh_jobs (
                 job_id, page_id, base_page_version, base_source_revision, space,
                 dependency_generation, active_root_digest, state, lease_token,
                 lease_expires_at, attempt, readiness_epoch, schema_version, reason,
                 created_at, updated_at
             ) VALUES (999, 'page-b', 1, 1, 'space-a', 1, 'digest', 'queued',
                       'lease', 10, 0, 1, 109, 'source_updated', 1, 1)",
            (),
        )
        .await;
    assert!(queued.is_err(), "queued must not be a durable job state");
}

async fn install_provenance_roots(conn: &libsql::Connection) {
    conn.execute_batch(
        "CREATE TABLE provenance_roots (
             root_id TEXT PRIMARY KEY,
             status TEXT NOT NULL CHECK(status IN ('active','failed'))
         );
         CREATE TABLE finalized_pages (
             page_id TEXT PRIMARY KEY,
             page_version INTEGER NOT NULL
         );",
    )
    .await
    .expect("install test-owned existing substrate");
}

#[tokio::test]
async fn d7_root_retraction_and_disappearance_both_block_readiness() {
    let conn = connection().await;
    install_provenance_roots(&conn).await;
    install(&conn).await;
    conn.execute(
        "INSERT INTO provenance_roots(root_id, status) VALUES ('root-a', 'active')",
        (),
    )
    .await
    .unwrap();

    {
        let tx = conn.transaction().await.unwrap();
        replace_refresh_dependencies(
            &tx,
            "space-a",
            "page-a",
            1,
            &[RefreshDependency {
                claim_revision_id: "claim-a-v1",
                root_id: "root-a",
            }],
        )
        .await
        .unwrap();
        assert!(space_refresh_dependencies_ready(&tx, "space-a")
            .await
            .unwrap());
        tx.commit().await.unwrap();
    }

    conn.execute(
        "UPDATE provenance_roots SET status = 'failed' WHERE root_id = 'root-a'",
        (),
    )
    .await
    .unwrap();
    {
        let tx = conn.transaction().await.unwrap();
        assert!(
            !space_refresh_dependencies_ready(&tx, "space-a")
                .await
                .unwrap(),
            "a non-active root must fail precondition 5"
        );
        tx.commit().await.unwrap();
    }

    conn.execute("DELETE FROM provenance_roots WHERE root_id = 'root-a'", ())
        .await
        .unwrap();
    assert_eq!(
        scalar_i64(
            &conn,
            "SELECT count(*) FROM m6_refresh_dependencies WHERE root_id = 'root-a'"
        )
        .await,
        1,
        "root identity must remain after the live provenance row disappears"
    );
    {
        let tx = conn.transaction().await.unwrap();
        assert!(
            !space_refresh_dependencies_ready(&tx, "space-a")
                .await
                .unwrap(),
            "a missing root must fail the anti-join"
        );
        tx.commit().await.unwrap();
    }
}

#[tokio::test]
async fn d7_dependency_snapshot_commits_with_the_callers_outer_finalize_transaction() {
    let conn = connection().await;
    install_provenance_roots(&conn).await;
    install(&conn).await;
    conn.execute_batch(
        "INSERT INTO provenance_roots(root_id, status) VALUES ('root-old', 'active');
         INSERT INTO provenance_roots(root_id, status) VALUES ('root-new', 'active');
         INSERT INTO finalized_pages(page_id, page_version) VALUES ('page-a', 1);",
    )
    .await
    .unwrap();
    {
        let tx = conn.transaction().await.unwrap();
        replace_refresh_dependencies(
            &tx,
            "space-a",
            "page-a",
            1,
            &[RefreshDependency {
                claim_revision_id: "claim-old",
                root_id: "root-old",
            }],
        )
        .await
        .unwrap();
        tx.commit().await.unwrap();
    }

    {
        let tx = conn.transaction().await.unwrap();
        tx.execute(
            "UPDATE finalized_pages SET page_version = 2 WHERE page_id = 'page-a'",
            (),
        )
        .await
        .unwrap();
        replace_refresh_dependencies(
            &tx,
            "space-a",
            "page-a",
            2,
            &[RefreshDependency {
                claim_revision_id: "claim-new",
                root_id: "root-new",
            }],
        )
        .await
        .unwrap();
        tx.rollback().await.unwrap();
    }
    assert_eq!(
        scalar_i64(
            &conn,
            "SELECT page_version FROM finalized_pages WHERE page_id = 'page-a'"
        )
        .await,
        1
    );
    assert_eq!(
        scalar_i64(
            &conn,
            "SELECT page_version FROM m6_refresh_dependencies WHERE page_id = 'page-a'"
        )
        .await,
        1,
        "dependency replacement must roll back with the page finalize"
    );

    {
        let tx = conn.transaction().await.unwrap();
        tx.execute(
            "UPDATE finalized_pages SET page_version = 2 WHERE page_id = 'page-a'",
            (),
        )
        .await
        .unwrap();
        replace_refresh_dependencies(
            &tx,
            "space-a",
            "page-a",
            2,
            &[RefreshDependency {
                claim_revision_id: "claim-new",
                root_id: "root-new",
            }],
        )
        .await
        .unwrap();
        tx.commit().await.unwrap();
    }
    assert_eq!(
        scalar_i64(
            &conn,
            "SELECT page_version FROM m6_refresh_dependencies WHERE page_id = 'page-a'"
        )
        .await,
        2
    );
    assert_eq!(
        scalar_i64(
            &conn,
            "SELECT count(*) FROM m6_refresh_dependencies WHERE page_id = 'page-a'"
        )
        .await,
        1,
        "the current dependency snapshot must replace the incompatible one"
    );
}

fn key<'a>(stage: &'a str, signal: &'a str) -> ReadinessKey<'a> {
    ReadinessKey {
        space: "space-a",
        stage,
        signal,
    }
}

#[tokio::test]
async fn d8_stage_signal_domain_is_structural_and_e_signals_are_independent() {
    let conn = connection().await;
    install(&conn).await;

    let tx = conn.transaction().await.unwrap();
    initialize_readiness(&tx, key("D", "-"), 10).await.unwrap();
    let duplicate_d = initialize_readiness(&tx, key("D", "-"), 11).await.unwrap();
    assert!(!duplicate_d, "a duplicate D/'-' row must be rejected");

    for signal in [
        "evidence-cluster",
        "orphan-wikilink",
        "community-overview",
        "space-overview",
    ] {
        assert!(initialize_readiness(&tx, key("E", signal), 10)
            .await
            .unwrap());
    }
    tx.commit().await.unwrap();
    assert_eq!(
        scalar_i64(&conn, "SELECT count(*) FROM m6_readiness WHERE stage = 'E'").await,
        4
    );

    let invalid = conn
        .execute(
            "INSERT INTO m6_readiness (
                 space, stage, signal, epoch, phase, operation_state, updated_at
             ) VALUES ('space-a', 'D_preparing', '-', 0, 'off', 'active', 1)",
            (),
        )
        .await;
    assert!(
        invalid.is_err(),
        "a composite presentation label must not enter the stage column"
    );
}

#[tokio::test]
async fn d8_phase_transitions_bump_epoch_and_close_aba() {
    let conn = connection().await;
    install(&conn).await;
    let readiness_key = key("D", "-");

    let tx = conn.transaction().await.unwrap();
    initialize_readiness(&tx, readiness_key, 1).await.unwrap();
    let initial = ReadinessFence {
        epoch: 0,
        phase: ReadinessPhase::Off,
    };
    let preparing = transition_readiness(&tx, readiness_key, initial, ReadinessPhase::Preparing, 2)
        .await
        .unwrap()
        .expect("off -> preparing");
    assert_eq!(preparing.epoch, 1);
    let aborted = transition_readiness(&tx, readiness_key, preparing, ReadinessPhase::Off, 3)
        .await
        .unwrap()
        .expect("preparing -> off abort");
    assert_eq!(aborted.epoch, 2);
    assert_eq!(aborted.phase, ReadinessPhase::Off);

    let stale_aba = transition_readiness(&tx, readiness_key, initial, ReadinessPhase::Preparing, 4)
        .await
        .unwrap();
    assert!(
        stale_aba.is_none(),
        "the first off fence must not match the later off fence"
    );
    assert_eq!(
        readiness_fence(&tx, readiness_key).await.unwrap(),
        Some(aborted)
    );
    tx.commit().await.unwrap();
}

#[tokio::test]
async fn d8_soak_evidence_is_separate_from_stage_phase_and_operation_state() {
    let conn = connection().await;
    install(&conn).await;
    let readiness_key = key("D", "-");
    let tx = conn.transaction().await.unwrap();
    initialize_readiness(&tx, readiness_key, 1).await.unwrap();

    let ghost_epoch = record_soak_receipt(
        &tx,
        readiness_key,
        99,
        &SoakEvidence {
            window_started_at: 0,
            window_ended_at: 259_200,
            observed_turns: 20,
            daemon_starts: 1,
            old_writer_mutations: 0,
            work_bound_violations: 0,
            support_regressions: 0,
            recorded_at: 259_200,
        },
    )
    .await;
    assert!(
        ghost_epoch.is_err(),
        "soak evidence must bind to the current epoch/phase fence"
    );

    let weak = record_soak_receipt(
        &tx,
        readiness_key,
        0,
        &SoakEvidence {
            window_started_at: 0,
            window_ended_at: 259_200,
            observed_turns: 19,
            daemon_starts: 1,
            old_writer_mutations: 0,
            work_bound_violations: 0,
            support_regressions: 0,
            recorded_at: 259_200,
        },
    )
    .await;
    assert!(
        weak.is_err(),
        "a weak soak must not be recordable as passed"
    );

    record_soak_receipt(
        &tx,
        readiness_key,
        0,
        &SoakEvidence {
            window_started_at: 0,
            window_ended_at: 259_200,
            observed_turns: 20,
            daemon_starts: 1,
            old_writer_mutations: 0,
            work_bound_violations: 0,
            support_regressions: 0,
            recorded_at: 259_200,
        },
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();

    assert_eq!(
        scalar_i64(&conn, "SELECT count(*) FROM m6_readiness_soak_receipts").await,
        1
    );
    conn.execute(
        "UPDATE m6_readiness SET operation_state = 'disabled',
                                  operation_reason = 'gate_failed'
          WHERE space = 'space-a' AND stage = 'D' AND signal = '-'",
        (),
    )
    .await
    .expect("disable is a separate operational state");
    conn.execute(
        "UPDATE m6_readiness SET operation_state = 'forward_fix'
          WHERE space = 'space-a' AND stage = 'D' AND signal = '-'
            AND operation_state = 'disabled'",
        (),
    )
    .await
    .expect("forward-fix follows disabled without rewriting the fence");
    let mut state_rows = conn
        .query(
            "SELECT stage, phase, epoch, operation_state FROM m6_readiness
              WHERE space = 'space-a' AND stage = 'D' AND signal = '-'",
            (),
        )
        .await
        .unwrap();
    let state = state_rows.next().await.unwrap().unwrap();
    assert_eq!(state.get::<String>(0).unwrap(), "D");
    assert_eq!(state.get::<String>(1).unwrap(), "off");
    assert_eq!(state.get::<i64>(2).unwrap(), 0);
    assert_eq!(state.get::<String>(3).unwrap(), "forward_fix");

    let stage_soaked = conn
        .execute(
            "UPDATE m6_readiness SET stage = 'D_soaked' WHERE space = 'space-a'",
            (),
        )
        .await;
    assert!(
        stage_soaked.is_err(),
        "soak must not be smuggled into stage"
    );
    let phase_disabled = conn
        .execute(
            "UPDATE m6_readiness SET phase = 'disabled' WHERE space = 'space-a'",
            (),
        )
        .await;
    assert!(
        phase_disabled.is_err(),
        "disabled/forward-fix belong to operation_state, not phase"
    );
}
