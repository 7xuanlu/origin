// SPDX-License-Identifier: Apache-2.0

use super::MemoryDB;
use crate::error::WenlanError;
use crate::eval::seed_contract::{SeedContractReport, SeedExpectations};
use crate::lint::snapshot::{LintReadSnapshot, SnapshotError, StructuralDigest};
use std::marker::PhantomData;
use std::sync::Arc;
use wenlan_types::repair::RepairDigest;

enum TestDbConnection {
    Primary(tokio::sync::OwnedMutexGuard<libsql::Connection>),
    Secondary(libsql::Connection),
}

impl TestDbConnection {
    fn connection(&self) -> &libsql::Connection {
        match self {
            Self::Primary(connection) => connection,
            Self::Secondary(connection) => connection,
        }
    }

    fn is_secondary(&self) -> bool {
        matches!(self, Self::Secondary(_))
    }
}

enum TestDbSessionState {
    Connection(TestDbConnection),
    Transaction {
        transaction: libsql::Transaction,
        connection_owner: TestDbConnection,
        query_only: bool,
    },
}

/// Test-only database access with no way to recover a raw libSQL handle.
pub(crate) struct TestDbSession {
    state: TestDbSessionState,
}

impl TestDbSession {
    fn connection(&self) -> &libsql::Connection {
        match &self.state {
            TestDbSessionState::Connection(connection) => connection.connection(),
            TestDbSessionState::Transaction { transaction, .. } => transaction,
        }
    }

    fn is_secondary(&self) -> bool {
        match &self.state {
            TestDbSessionState::Connection(connection) => connection.is_secondary(),
            TestDbSessionState::Transaction {
                connection_owner, ..
            } => connection_owner.is_secondary(),
        }
    }

    pub(crate) async fn execute(
        &self,
        sql: &str,
        params: impl libsql::params::IntoParams,
    ) -> libsql::Result<u64> {
        self.connection().execute(sql, params).await
    }

    pub(crate) async fn execute_batch(&self, sql: &str) -> libsql::Result<()> {
        self.connection().execute_batch(sql).await?;
        Ok(())
    }

    pub(crate) async fn query<'session>(
        &'session self,
        sql: &str,
        params: impl libsql::params::IntoParams,
    ) -> libsql::Result<TestDbRows<'session>> {
        let rows = self.connection().query(sql, params).await?;
        Ok(TestDbRows {
            rows,
            _session: PhantomData,
        })
    }

    async fn begin(self, behavior: libsql::TransactionBehavior) -> libsql::Result<Self> {
        let connection = match self.state {
            TestDbSessionState::Connection(connection) => connection,
            TestDbSessionState::Transaction { .. } => {
                return Err(libsql::Error::Misuse(
                    "test session already owns a managed transaction".to_string(),
                ));
            }
        };
        if !connection.is_secondary() {
            return Err(libsql::Error::Misuse(
                "managed test transactions require an isolated secondary connection".to_string(),
            ));
        }
        let query_only = matches!(behavior, libsql::TransactionBehavior::ReadOnly);
        if query_only {
            connection
                .connection()
                .execute("PRAGMA query_only = ON", ())
                .await?;
        }
        let transaction_result = connection
            .connection()
            .transaction_with_behavior(behavior)
            .await;
        let transaction = match transaction_result {
            Ok(transaction) => transaction,
            Err(error) => {
                if query_only {
                    let _ = connection
                        .connection()
                        .execute("PRAGMA query_only = OFF", ())
                        .await;
                }
                return Err(error);
            }
        };
        Ok(Self {
            state: TestDbSessionState::Transaction {
                transaction,
                connection_owner: connection,
                query_only,
            },
        })
    }

    pub(crate) async fn begin_immediate(self) -> libsql::Result<Self> {
        self.begin(libsql::TransactionBehavior::Immediate).await
    }

    pub(crate) async fn begin_read_only(self) -> libsql::Result<Self> {
        self.begin(libsql::TransactionBehavior::ReadOnly).await
    }

    pub(crate) async fn commit(self) -> libsql::Result<Self> {
        match self.state {
            TestDbSessionState::Transaction {
                transaction,
                connection_owner,
                query_only,
            } => {
                let result = transaction.commit().await;
                let reset_result = if query_only {
                    connection_owner
                        .connection()
                        .execute("PRAGMA query_only = OFF", ())
                        .await
                        .map(|_| ())
                } else {
                    Ok(())
                };
                result?;
                reset_result?;
                Ok(Self {
                    state: TestDbSessionState::Connection(connection_owner),
                })
            }
            TestDbSessionState::Connection(connection) => {
                if connection.connection().is_autocommit() {
                    return Err(libsql::Error::Misuse(
                        "test session has no active raw SQL transaction to commit".to_string(),
                    ));
                }
                connection.connection().execute("COMMIT", ()).await?;
                Ok(Self {
                    state: TestDbSessionState::Connection(connection),
                })
            }
        }
    }

    pub(crate) async fn rollback(self) -> libsql::Result<Self> {
        match self.state {
            TestDbSessionState::Transaction {
                transaction,
                connection_owner,
                query_only,
            } => {
                let result = transaction.rollback().await;
                let reset_result = if query_only {
                    connection_owner
                        .connection()
                        .execute("PRAGMA query_only = OFF", ())
                        .await
                        .map(|_| ())
                } else {
                    Ok(())
                };
                result?;
                reset_result?;
                Ok(Self {
                    state: TestDbSessionState::Connection(connection_owner),
                })
            }
            TestDbSessionState::Connection(connection) => {
                if connection.connection().is_autocommit() {
                    return Err(libsql::Error::Misuse(
                        "test session has no active raw SQL transaction to roll back".to_string(),
                    ));
                }
                connection.connection().execute("ROLLBACK", ()).await?;
                Ok(Self {
                    state: TestDbSessionState::Connection(connection),
                })
            }
        }
    }

    pub(crate) async fn structural_digest(&self) -> Result<StructuralDigest, SnapshotError> {
        if !self.is_secondary() {
            return Err(SnapshotError::Database(libsql::Error::Misuse(
                "structural digest requires an isolated secondary connection".to_string(),
            )));
        }
        crate::lint::snapshot::structural_digest(self.connection()).await
    }

    pub(crate) async fn repair_database_content_digest(&self) -> Result<RepairDigest, WenlanError> {
        crate::repair::database_content_digest(self.connection()).await
    }

    pub(crate) async fn check_seed_contract(
        &self,
        expectations: &SeedExpectations,
    ) -> Result<SeedContractReport, WenlanError> {
        crate::eval::seed_contract::check_seed_contract(self.connection(), expectations).await
    }
}

/// Rows stay borrow-tied to the session that produced them even though libSQL
/// currently owns its row cursor.
pub(crate) struct TestDbRows<'session> {
    rows: libsql::Rows,
    _session: PhantomData<&'session TestDbSession>,
}

impl TestDbRows<'_> {
    pub(crate) async fn next(&mut self) -> libsql::Result<Option<TestDbRow>> {
        Ok(self.rows.next().await?.map(|row| TestDbRow { row }))
    }
}

pub(crate) struct TestDbRow {
    row: libsql::Row,
}

mod sealed {
    pub trait Sealed {}
}

trait TestDbValue: sealed::Sealed + Sized {
    fn get(row: &libsql::Row, index: i32) -> libsql::Result<Self>;
}

#[allow(private_bounds)]
impl TestDbRow {
    pub(crate) fn get<T: TestDbValue>(&self, index: i32) -> libsql::Result<T> {
        T::get(&self.row, index)
    }
}

macro_rules! concrete_test_db_value {
    ($type:ty) => {
        impl sealed::Sealed for $type {}

        impl TestDbValue for $type {
            fn get(row: &libsql::Row, index: i32) -> libsql::Result<Self> {
                row.get::<$type>(index)
            }
        }
    };
}

concrete_test_db_value!(i64);
concrete_test_db_value!(i32);
concrete_test_db_value!(u64);
concrete_test_db_value!(u32);
concrete_test_db_value!(f64);
concrete_test_db_value!(String);
concrete_test_db_value!(Vec<u8>);

impl<T: TestDbValue> sealed::Sealed for Option<T> {}

impl<T: TestDbValue> TestDbValue for Option<T> {
    fn get(row: &libsql::Row, index: i32) -> libsql::Result<Self> {
        if row.get_value(index)? == libsql::Value::Null {
            Ok(None)
        } else {
            T::get(row, index).map(Some)
        }
    }
}

impl MemoryDB {
    pub(crate) async fn test_primary_session(&self) -> TestDbSession {
        let connection = Arc::clone(&self.conn).lock_owned().await;
        TestDbSession {
            state: TestDbSessionState::Connection(TestDbConnection::Primary(connection)),
        }
    }

    pub(crate) fn test_secondary_session(&self) -> Result<TestDbSession, WenlanError> {
        let connection = self._db.connect().map_err(|error| {
            WenlanError::VectorDb(format!("test secondary connection: {error}"))
        })?;
        Ok(TestDbSession {
            state: TestDbSessionState::Connection(TestDbConnection::Secondary(connection)),
        })
    }

    pub(crate) fn primary_mutex_available(&self) -> bool {
        self.conn.try_lock().is_ok()
    }

    pub(crate) async fn open_isolated_lint_snapshot_for_test(
        &self,
    ) -> Result<LintReadSnapshot<'_>, SnapshotError> {
        LintReadSnapshot::open(&self._db).await
    }
}

#[tokio::test]
async fn primary_managed_transactions_fail_and_release_the_mutex() {
    let (db, _tmp) = super::tests::test_db().await;
    let immediate = db.test_primary_session().await.begin_immediate().await;
    assert!(
        immediate.is_err(),
        "managed transactions must require an isolated secondary connection"
    );
    assert!(
        db.primary_mutex_available(),
        "rejecting a primary managed transaction must release its owned guard"
    );
    let read_only = db.test_primary_session().await.begin_read_only().await;
    assert!(
        read_only.is_err(),
        "read-only transactions must require an isolated secondary connection"
    );
    assert!(
        db.primary_mutex_available(),
        "rejecting a primary read-only transaction must release its owned guard"
    );
}

#[tokio::test]
async fn structural_digest_requires_a_secondary_session() {
    let (db, _tmp) = super::tests::test_db().await;
    let primary = db.test_primary_session().await;
    assert!(
        primary.structural_digest().await.is_err(),
        "structural digest must refuse the primary connection"
    );
}

#[tokio::test]
async fn opaque_test_sessions_cover_primary_secondary_rows_and_transactions() {
    let (db, _tmp) = super::tests::test_db().await;

    let primary = db.test_primary_session().await;
    assert!(!db.primary_mutex_available());
    primary
        .execute_batch(
            "CREATE TEMP TABLE r4_session_probe(value INTEGER, note TEXT, bytes BLOB);
             INSERT INTO r4_session_probe(value, note, bytes) VALUES (41, NULL, X'0102');",
        )
        .await
        .expect("seed primary session");
    let mut rows = primary
        .query("SELECT value, note, bytes FROM r4_session_probe", ())
        .await
        .expect("query primary session");
    let row = rows
        .next()
        .await
        .expect("read primary row")
        .expect("primary row exists");
    assert_eq!(row.get::<i64>(0).expect("decode primary value"), 41);
    assert_eq!(
        row.get::<Option<String>>(1).expect("decode nullable text"),
        None
    );
    assert_eq!(
        row.get::<Vec<u8>>(2).expect("decode blob"),
        vec![1_u8, 2_u8]
    );
    drop(rows);

    let secondary_while_primary_is_held = db.test_secondary_session().expect("secondary session");
    let mut concurrent_rows = secondary_while_primary_is_held
        .query("SELECT 1", ())
        .await
        .expect("secondary query while primary mutex is held");
    assert_eq!(
        concurrent_rows
            .next()
            .await
            .expect("secondary row")
            .expect("secondary row exists")
            .get::<i64>(0)
            .expect("secondary value"),
        1
    );
    drop(concurrent_rows);
    drop(secondary_while_primary_is_held);

    primary
        .execute_batch(
            "CREATE TABLE r4_transaction_probe(
                 value INTEGER PRIMARY KEY
             );",
        )
        .await
        .expect("create transaction probe");
    drop(primary);
    assert!(db.primary_mutex_available());

    let rollback = db
        .test_secondary_session()
        .expect("open rollback session")
        .begin_immediate()
        .await
        .expect("begin rollback transaction");
    rollback
        .execute("INSERT INTO r4_transaction_probe(value) VALUES (1)", ())
        .await
        .expect("insert rollback probe");
    rollback
        .rollback()
        .await
        .expect("roll back immediate transaction");

    let committed = db
        .test_secondary_session()
        .expect("open commit session")
        .begin_immediate()
        .await
        .expect("begin commit transaction");
    committed
        .execute("INSERT INTO r4_transaction_probe(value) VALUES (2)", ())
        .await
        .expect("insert commit probe");
    committed
        .commit()
        .await
        .expect("commit immediate transaction");

    let verify = db.test_secondary_session().expect("verification session");
    let mut rows = verify
        .query("SELECT value FROM r4_transaction_probe ORDER BY value", ())
        .await
        .expect("query transaction outcome");
    let only_row = rows
        .next()
        .await
        .expect("read committed row")
        .expect("committed row exists");
    assert_eq!(only_row.get::<i64>(0).expect("committed value"), 2);
    assert!(
        rows.next().await.expect("read end of rows").is_none(),
        "rolled-back insert must be absent"
    );
    drop(rows);
    drop(verify);

    let read_only = db
        .test_secondary_session()
        .expect("open read-only session")
        .begin_read_only()
        .await
        .expect("begin read-only transaction");
    assert!(
        read_only
            .execute("INSERT INTO r4_transaction_probe(value) VALUES (3)", ())
            .await
            .is_err(),
        "read-only transaction must reject mutation"
    );
    let writable_after_rollback = read_only
        .rollback()
        .await
        .expect("roll back read-only transaction");
    writable_after_rollback
        .execute("INSERT INTO r4_transaction_probe(value) VALUES (3)", ())
        .await
        .expect("query_only reset on the same connection after read-only rollback");

    let read_only = db
        .test_secondary_session()
        .expect("open second read-only session")
        .begin_read_only()
        .await
        .expect("begin second read-only transaction");
    let writable_after_commit = read_only
        .commit()
        .await
        .expect("commit read-only transaction");
    writable_after_commit
        .execute("INSERT INTO r4_transaction_probe(value) VALUES (4)", ())
        .await
        .expect("query_only reset on the same connection after read-only commit");
    let mut rows = writable_after_commit
        .query(
            "SELECT COUNT(*) FROM r4_transaction_probe WHERE value IN (3, 4)",
            (),
        )
        .await
        .expect("query post-read-only write");
    assert_eq!(
        rows.next()
            .await
            .expect("read post-read-only count")
            .expect("post-read-only count exists")
            .get::<i64>(0)
            .expect("post-read-only count"),
        2
    );
    drop(rows);
    drop(writable_after_commit);

    assert!(
        db.test_secondary_session()
            .expect("autocommit session")
            .commit()
            .await
            .is_err(),
        "commit without a raw SQL transaction must fail closed"
    );
    assert!(
        db.test_secondary_session()
            .expect("autocommit session")
            .rollback()
            .await
            .is_err(),
        "rollback without a raw SQL transaction must fail closed"
    );
}

#[tokio::test]
async fn opaque_test_session_supports_preexisting_raw_sql_transaction_cleanup() {
    let (db, _tmp) = super::tests::test_db().await;
    let session = db.test_secondary_session().expect("secondary session");
    session
        .execute_batch(
            "CREATE TABLE r4_raw_transaction_probe(
                 value INTEGER PRIMARY KEY
             );",
        )
        .await
        .expect("create raw transaction probe");
    session
        .execute("BEGIN IMMEDIATE", ())
        .await
        .expect("begin raw test transaction");
    session
        .execute("INSERT INTO r4_raw_transaction_probe(value) VALUES (1)", ())
        .await
        .expect("insert raw rollback probe");
    session.rollback().await.expect("roll back raw transaction");

    let session = db.test_secondary_session().expect("secondary session");
    session
        .execute("BEGIN IMMEDIATE", ())
        .await
        .expect("begin raw test transaction");
    session
        .execute("INSERT INTO r4_raw_transaction_probe(value) VALUES (2)", ())
        .await
        .expect("insert raw commit probe");
    session.commit().await.expect("commit raw transaction");

    let verify = db.test_secondary_session().expect("verification session");
    let mut rows = verify
        .query(
            "SELECT value FROM r4_raw_transaction_probe ORDER BY value",
            (),
        )
        .await
        .expect("query raw transaction outcomes");
    let only_row = rows
        .next()
        .await
        .expect("read raw committed row")
        .expect("raw committed row exists");
    assert_eq!(only_row.get::<i64>(0).expect("raw committed value"), 2);
    assert!(
        rows.next().await.expect("read raw end of rows").is_none(),
        "raw rolled-back insert must be absent"
    );
}

#[tokio::test]
async fn opaque_domain_operations_and_isolated_lint_snapshot_are_live() {
    let (db, _tmp) = super::tests::test_db().await;
    let primary = db.test_primary_session().await;
    primary
        .execute_batch(
            "CREATE TABLE r4_digest_probe(
                 id INTEGER PRIMARY KEY,
                 payload TEXT NOT NULL
             );
             INSERT INTO r4_digest_probe(id, payload) VALUES (1, 'before');",
        )
        .await
        .expect("seed digest probe");
    let repair_before = primary
        .repair_database_content_digest()
        .await
        .expect("repair content digest");
    let expectations = SeedExpectations {
        variant: "r4_test_support".to_string(),
        require_no_dupes: false,
        require_full_classification: false,
        min_cue_coverage: 0.0,
        min_event_date_coverage: 0.0,
        require_graph_links: false,
        require_event_dates: false,
        require_pages: false,
        expect_fixture_sha256: None,
    };
    primary
        .check_seed_contract(&expectations)
        .await
        .expect("seed contract");
    drop(primary);

    let secondary = db
        .test_secondary_session()
        .expect("secondary digest session");
    let structural_before = secondary
        .structural_digest()
        .await
        .expect("structural digest");
    secondary
        .execute(
            "UPDATE r4_digest_probe SET payload = 'after' WHERE id = 1",
            (),
        )
        .await
        .expect("mutate payload without changing structure");
    let structural_after = secondary
        .structural_digest()
        .await
        .expect("post-update structural digest");
    drop(secondary);

    let primary = db.test_primary_session().await;
    let repair_after = primary
        .repair_database_content_digest()
        .await
        .expect("post-update repair content digest");
    assert_eq!(
        structural_before, structural_after,
        "structural digest must ignore payload-only changes"
    );
    assert_ne!(
        repair_before, repair_after,
        "repair digest must observe payload-only changes"
    );
    drop(primary);

    let first_snapshot = db
        .open_isolated_lint_snapshot_for_test()
        .await
        .expect("first fresh isolated lint snapshot");
    let first = first_snapshot
        .finish()
        .await
        .expect("close first isolated snapshot");
    let second_snapshot = db
        .open_isolated_lint_snapshot_for_test()
        .await
        .expect("second fresh isolated lint snapshot");
    let second = second_snapshot
        .finish()
        .await
        .expect("close second isolated snapshot");
    assert_eq!(first.analysis_digest(), second.analysis_digest());
    assert_ne!(
        first.analysis_receipt_digest(),
        second.analysis_receipt_digest(),
        "fresh isolated snapshots must carry distinct observer epochs"
    );
}
