// SPDX-License-Identifier: Apache-2.0
//! Migration-109 integration teeth for the complete inert M6 substrate.

use crate::db::tests::test_db;
use crate::db::{MemoryDB, SCHEMA_VERSION};

const FOLLOWUP_TABLES: [&str; 14] = [
    "genesis_suppression",
    "genesis_card_binding",
    "genesis_quarantine",
    "genesis_refresh_jobs",
    "m6_refresh_dependencies",
    "m6_readiness",
    "m6_readiness_soak_receipts",
    "m6_overview_subscriptions",
    "m6_pair_stats",
    "m6_adjacency",
    "m6_deploy_attestation",
    "m6_genesis_provenance",
    "page_projection_outbox",
    "m6_counters",
];

async fn table_exists(db: &MemoryDB, table: &str) -> bool {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
            libsql::params![table],
        )
        .await
        .expect("inspect migration-109 table");
    rows.next().await.expect("read table existence").is_some()
}

async fn column_exists(db: &MemoryDB, table: &str, expected: &str) -> bool {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(&format!("PRAGMA table_info({table})"), ())
        .await
        .expect("inspect migration-109 column");
    while let Some(row) = rows.next().await.expect("read table column") {
        if row.get::<String>(1).expect("column name") == expected {
            return true;
        }
    }
    false
}

async fn user_version(db: &MemoryDB) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn.query("PRAGMA user_version", ()).await.unwrap();
    rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap()
}

#[tokio::test]
async fn migrated_database_carries_all_fourteen_followup_tables_and_counter() {
    let (db, _temp) = test_db().await;
    assert_eq!(SCHEMA_VERSION, 110);
    assert_eq!(user_version(&db).await, i64::from(SCHEMA_VERSION));
    for table in FOLLOWUP_TABLES {
        assert!(
            table_exists(&db, table).await,
            "migration 109 omitted {table}"
        );
    }
    assert!(
        column_exists(&db, "genesis_coverage_state", "m6_mutation_count").await,
        "migration 109 omitted the per-space monotone M6 mutation counter"
    );
    assert!(
        column_exists(&db, "genesis_coverage_state", "space_id").await,
        "migration 109 omitted the stable space identity binding"
    );
}

#[tokio::test]
async fn schema_108_upgrades_through_the_whole_followup_inventory() {
    let (db, _temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        for table in FOLLOWUP_TABLES {
            conn.execute(&format!("DROP TABLE {table}"), ())
                .await
                .unwrap_or_else(|error| panic!("drop {table}: {error}"));
        }
        conn.execute("PRAGMA user_version = 108", ())
            .await
            .expect("restore the pre-followup schema boundary");
    }

    db.run_migrations(&crate::events::NoopEmitter)
        .await
        .expect("schema 108 must advance through migration 109");

    assert_eq!(user_version(&db).await, i64::from(SCHEMA_VERSION));
    for table in FOLLOWUP_TABLES {
        assert!(
            table_exists(&db, table).await,
            "{table} missing after upgrade"
        );
    }
}

#[tokio::test]
async fn replay_after_schema_commit_before_version_bump_preserves_rows() {
    let (db, _temp) = test_db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO m6_deploy_attestation
                 (attestation_id, attested_at, signature, binary_digest)
             VALUES ('attestation-replay', 100, 'signed', 'binary')",
            (),
        )
        .await
        .expect("seed durable migration-109 evidence");
        conn.execute("PRAGMA user_version = 108", ())
            .await
            .expect("simulate a crash before the version bump");
    }

    db.run_migrations(&crate::events::NoopEmitter)
        .await
        .expect("idempotent migration-109 replay");

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query("SELECT COUNT(*) FROM m6_deploy_attestation", ())
        .await
        .unwrap();
    assert_eq!(
        rows.next().await.unwrap().unwrap().get::<i64>(0).unwrap(),
        1
    );
}
