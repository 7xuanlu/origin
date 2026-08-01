// SPDX-License-Identifier: Apache-2.0
//! Teeth for migration 108.
//!
//! Two halves. The first is migration discipline: a migrated database carries
//! every object, the constant matches what the chain stamps, a replay is inert,
//! and a database from a newer daemon is refused. The second is the only
//! behavior this migration actually has — the two partial unique indexes of
//! §5, which are D4's entire exclusion mechanism (I-2) and the reason a witness
//! row can never be read as coverage (I-3).
//!
//! Every index test asserts against a REJECTION, not against a query result: an
//! exclusion mechanism that is merely queried correctly is a convention, and a
//! convention is what the contract says must not be the enforcement.

use crate::db::tests::test_db;
use crate::db::{MemoryDB, SCHEMA_VERSION};

/// Every object migration 108 promises, by name and by kind.
const GENESIS_TABLES: [&str; 5] = [
    "genesis_candidates",
    "genesis_candidate_roots",
    "genesis_coverage_state",
    "genesis_group_coverage",
    "genesis_frontier",
];

const GENESIS_INDEXES: [&str; 2] = ["idx_genesis_root_claim", "idx_genesis_group_claim"];

async fn object_exists(db: &MemoryDB, kind: &str, name: &str) -> bool {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT 1 FROM sqlite_master WHERE type = ?1 AND name = ?2",
            libsql::params![kind, name],
        )
        .await
        .expect("query sqlite_master");
    rows.next().await.expect("read sqlite_master").is_some()
}

/// The stored SQL of an index, so a test can assert on its PREDICATE rather
/// than merely on its existence — an index that lost its `WHERE` is still an
/// index with the right name.
async fn index_sql(db: &MemoryDB, name: &str) -> String {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?1",
            libsql::params![name],
        )
        .await
        .expect("query index sql");
    let row = rows
        .next()
        .await
        .expect("read index sql")
        .unwrap_or_else(|| panic!("index {name} does not exist"));
    row.get::<String>(0).expect("index sql column")
}

async fn user_version(db: &MemoryDB) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query("PRAGMA user_version", ())
        .await
        .expect("read user_version");
    rows.next()
        .await
        .expect("user_version row")
        .expect("user_version present")
        .get::<i64>(0)
        .expect("user_version int")
}

/// A candidate row, minimal but valid. The claim tests need one because
/// `genesis_candidate_roots.candidate_id` is a foreign key and the test
/// database opens with `PRAGMA foreign_keys=ON`.
async fn insert_candidate(db: &MemoryDB, candidate_id: &str) {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO genesis_candidates (
             candidate_id, slot_id, page_id, space, signal_kind, coverage_epoch,
             input_generation, active_root_digest, state, attempt,
             created_at, updated_at
         ) VALUES (?1, ?2, ?3, 'space-1', 'evidence-cluster', 1, 1, 'digest', 'observed', 0,
                   unixepoch(), unixepoch())",
        libsql::params![
            candidate_id,
            format!("slot-for-{candidate_id}"),
            format!("m6p_{candidate_id}")
        ],
    )
    .await
    .expect("insert candidate");
}

/// One claim row. Returns the raw result so a test can assert on a rejection.
async fn insert_claim(
    db: &MemoryDB,
    candidate_id: &str,
    root_id: &str,
    group_id: &str,
    epoch: i64,
    claim_role: &str,
) -> Result<u64, libsql::Error> {
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO genesis_candidate_roots (
             candidate_id, root_id, independence_group_id, coverage_epoch,
             claim_role, created_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, unixepoch())",
        libsql::params![candidate_id, root_id, group_id, epoch, claim_role],
    )
    .await
}

// ---------------------------------------------------------------- discipline

#[tokio::test]
async fn a_migrated_database_carries_every_genesis_object() {
    let (db, _tmp) = test_db().await;
    for table in GENESIS_TABLES {
        assert!(
            object_exists(&db, "table", table).await,
            "migration 108 promised table {table} and it is not there"
        );
    }
    for index in GENESIS_INDEXES {
        assert!(
            object_exists(&db, "index", index).await,
            "migration 108 promised index {index} and it is not there"
        );
    }
}

/// The defect this catches is real and recent: a migration that stamps
/// `user_version = N` while `SCHEMA_VERSION` still reads `N-1` makes the
/// refuse-newer guard hard-fail every boot after the first migrated restart.
#[tokio::test]
async fn the_schema_version_constant_matches_what_the_chain_stamps() {
    let (db, _tmp) = test_db().await;
    assert_eq!(
        user_version(&db).await,
        i64::from(SCHEMA_VERSION),
        "the migration chain and SCHEMA_VERSION disagree about the current schema"
    );
}

/// Both indexes must keep their predicates. An index that lost its `WHERE`
/// would still exist under the same name, and would then forbid the witness
/// coexistence D4 requires — so existence alone is not the assertion.
#[tokio::test]
async fn both_exclusion_indexes_are_partial_and_unique() {
    let (db, _tmp) = test_db().await;
    for index in GENESIS_INDEXES {
        let sql = index_sql(&db, index).await.to_lowercase();
        assert!(
            sql.contains("unique"),
            "{index} is not UNIQUE; the exclusion mechanism is gone: {sql}"
        );
        assert!(
            sql.contains("where") && sql.contains("claim_role = 'concept'"),
            "{index} lost its concept-only predicate: {sql}"
        );
        assert!(
            sql.contains("released_at is null"),
            "{index} lost its active-only predicate: {sql}"
        );
    }
}

#[tokio::test]
async fn replaying_the_migration_chain_changes_nothing() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-replay").await;

    db.run_migrations(&crate::events::NoopEmitter)
        .await
        .expect("replaying the migration chain must succeed");

    assert_eq!(user_version(&db).await, i64::from(SCHEMA_VERSION));
    for table in GENESIS_TABLES {
        assert!(object_exists(&db, "table", table).await, "{table} vanished");
    }
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query("SELECT COUNT(*) FROM genesis_candidates", ())
        .await
        .expect("count candidates");
    let count: i64 = rows
        .next()
        .await
        .expect("count row")
        .expect("count present")
        .get(0)
        .expect("count int");
    assert_eq!(count, 1, "a replay dropped and recreated a populated table");
}

/// The refuse-newer guard has been correct since it was written and has never
/// had a test. A database migrated by a newer daemon is not ours to write to.
#[tokio::test]
async fn a_database_from_a_newer_daemon_is_refused() {
    let (db, _tmp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(&format!("PRAGMA user_version = {}", SCHEMA_VERSION + 1), ())
            .await
            .expect("stamp a newer version");
    }

    let refusal = db.run_migrations(&crate::events::NoopEmitter).await;
    let error = refusal.expect_err("a newer schema must be refused, not written to");
    let message = error.to_string();
    assert!(
        message.contains("newer than this build supports"),
        "refused for the wrong reason: {message}"
    );
}

// ------------------------------------------------------------------ exclusion

#[tokio::test]
async fn two_live_concept_claims_on_one_root_cannot_coexist() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-a").await;
    insert_candidate(&db, "cand-b").await;

    insert_claim(&db, "cand-a", "root-1", "group-a", 1, "concept")
        .await
        .expect("the first concept claim takes the root");
    let second = insert_claim(&db, "cand-b", "root-1", "group-b", 1, "concept").await;
    assert!(
        second.is_err(),
        "two live concept candidates both claimed root-1; I-2 says one slot publishes at most one page"
    );
}

#[tokio::test]
async fn two_live_concept_claims_on_one_group_cannot_coexist() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-a").await;
    insert_candidate(&db, "cand-b").await;

    insert_claim(&db, "cand-a", "root-1", "group-a", 1, "concept")
        .await
        .expect("the first concept claim takes the group");
    let second = insert_claim(&db, "cand-b", "root-2", "group-a", 1, "concept").await;
    assert!(
        second.is_err(),
        "two live concept candidates both claimed group-a; overlapping candidates must not both mint"
    );
}

/// B4/B6: a released reservation returns its group to the frontier, and the
/// next candidate must be able to claim it. If `released_at` did not leave the
/// index predicate, every crash would permanently burn a root.
#[tokio::test]
async fn releasing_a_concept_claim_frees_its_root_and_group() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-a").await;
    insert_candidate(&db, "cand-b").await;
    insert_claim(&db, "cand-a", "root-1", "group-a", 1, "concept")
        .await
        .expect("first claim");

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE genesis_candidate_roots SET released_at = unixepoch()
             WHERE candidate_id = 'cand-a'",
            (),
        )
        .await
        .expect("release the claim");
    }

    insert_claim(&db, "cand-b", "root-1", "group-a", 1, "concept")
        .await
        .expect("a released root and group must be claimable again");
}

/// I-3, the positive control for the predicate: witness rows fall outside both
/// indexes, which is the mechanical form of D4's "witness-only overview
/// candidates may coexist without stealing evidence."
#[tokio::test]
async fn witness_claims_coexist_on_a_root_a_concept_already_holds() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-concept").await;
    insert_candidate(&db, "cand-witness-1").await;
    insert_candidate(&db, "cand-witness-2").await;

    insert_claim(&db, "cand-concept", "root-1", "group-a", 1, "concept")
        .await
        .expect("concept claim");
    insert_claim(&db, "cand-witness-1", "root-1", "group-a", 1, "witness")
        .await
        .expect("a witness must not be excluded by a concept claim");
    insert_claim(&db, "cand-witness-2", "root-1", "group-a", 1, "witness")
        .await
        .expect("witnesses must not exclude each other");
}

/// The claims are epoch-keyed (S0-7). A new epoch's claim is a different
/// claim, which is what makes D3's forward mapping possible at all.
#[tokio::test]
async fn a_concept_claim_in_another_epoch_is_not_excluded() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-a").await;
    insert_candidate(&db, "cand-b").await;

    insert_claim(&db, "cand-a", "root-1", "group-a", 1, "concept")
        .await
        .expect("epoch 1 claim");
    insert_claim(&db, "cand-b", "root-1", "group-a", 2, "concept")
        .await
        .expect("epoch 2 is a different claim");
}

// -------------------------------------------------------------- value domains

#[tokio::test]
async fn a_third_claim_role_is_rejected() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-a").await;
    let rejected = insert_claim(&db, "cand-a", "root-1", "group-a", 1, "observer").await;
    assert!(
        rejected.is_err(),
        "a third claim_role would be invisible to both index predicates"
    );
}

#[tokio::test]
async fn an_eleventh_candidate_state_is_rejected() {
    let (db, _tmp) = test_db().await;
    insert_candidate(&db, "cand-a").await;
    let conn = db.conn.lock().await;
    let rejected = conn
        .execute(
            "UPDATE genesis_candidates SET state = 'retry_exhausted' WHERE candidate_id = 'cand-a'",
            (),
        )
        .await;
    assert!(
        rejected.is_err(),
        "S0-12: exhausted retry is `stale` carrying a reason, never an eleventh state"
    );
}

#[tokio::test]
async fn an_unknown_signal_kind_is_rejected() {
    let (db, _tmp) = test_db().await;
    let conn = db.conn.lock().await;
    let rejected = conn
        .execute(
            "INSERT INTO genesis_candidates (
                 candidate_id, slot_id, page_id, space, signal_kind, coverage_epoch,
                 input_generation, active_root_digest, state, attempt, created_at, updated_at
             ) VALUES ('cand-x', 'slot-x', 'm6p_x', 'space-1', 'entity-cluster', 1, 1,
                       'digest', 'observed', 0, unixepoch(), unixepoch())",
            (),
        )
        .await;
    assert!(
        rejected.is_err(),
        "the signal set is D5's four; a fifth is a contract change"
    );
}

/// D1: a space row is created genesis-DISABLED at epoch 1, and a space with no
/// row at all is disabled too. Both halves are what make an empty table mean
/// "nothing runs".
#[tokio::test]
async fn a_new_space_row_defaults_to_disabled_at_epoch_one() {
    let (db, _tmp) = test_db().await;
    let conn = db.conn.lock().await;
    conn.execute(
        "INSERT INTO genesis_coverage_state (space, opened_at) VALUES ('space-1', unixepoch())",
        (),
    )
    .await
    .expect("insert coverage state naming only the space");

    let mut rows = conn
        .query(
            "SELECT coverage_epoch, epoch_state, genesis_enabled
             FROM genesis_coverage_state WHERE space = 'space-1'",
            (),
        )
        .await
        .expect("read coverage state");
    let row = rows
        .next()
        .await
        .expect("coverage row")
        .expect("coverage row present");
    assert_eq!(row.get::<i64>(0).unwrap(), 1, "epochs start at 1");
    assert_eq!(row.get::<String>(1).unwrap(), "active");
    assert_eq!(
        row.get::<i64>(2).unwrap(),
        0,
        "a space must be genesis-disabled until its own cutover enables it"
    );
}

/// I-7: coverage rows are epoch-keyed and immortal, so D3's forward mapping
/// writes the new epoch's row beside the old one rather than over it.
#[tokio::test]
async fn coverage_rows_of_two_epochs_coexist() {
    let (db, _tmp) = test_db().await;
    let conn = db.conn.lock().await;
    for epoch in [1, 2] {
        conn.execute(
            "INSERT INTO genesis_group_coverage (
                 space, independence_group_id, coverage_epoch, page_id, candidate_id, covered_at
             ) VALUES ('space-1', 'group-a', ?1, 'm6p_a', 'cand-a', unixepoch())",
            libsql::params![epoch],
        )
        .await
        .unwrap_or_else(|error| panic!("coverage row for epoch {epoch}: {error}"));
    }
}
