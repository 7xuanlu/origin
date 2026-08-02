// SPDX-License-Identifier: Apache-2.0

use super::remaining_substrate::ensure_remaining_substrate;

async fn test_connection() -> (libsql::Database, libsql::Connection) {
    let db = libsql::Builder::new_local(":memory:")
        .build()
        .await
        .expect("build remaining-substrate database");
    let conn = db.connect().expect("connect remaining-substrate database");
    conn.execute_batch(
        "CREATE TABLE spaces (
             id    TEXT PRIMARY KEY,
             name  TEXT NOT NULL UNIQUE
         );
         INSERT INTO spaces(id, name) VALUES
             ('space-id-existing', 'existing-space'),
             ('space-id-a', 'space-a');
         CREATE TABLE genesis_coverage_state (
             space             TEXT    PRIMARY KEY,
             coverage_epoch    INTEGER NOT NULL DEFAULT 1,
             epoch_state       TEXT    NOT NULL DEFAULT 'active',
             opened_at         INTEGER NOT NULL,
             migration_cursor  TEXT,
             genesis_enabled   INTEGER NOT NULL DEFAULT 0
         );
         INSERT INTO genesis_coverage_state (space, opened_at)
         VALUES ('existing-space', 10);",
    )
    .await
    .expect("install migration-108 coverage table");
    let tx = conn.transaction().await.expect("begin remaining substrate");
    ensure_remaining_substrate(&tx)
        .await
        .expect("install remaining substrate");
    tx.commit().await.expect("commit remaining substrate");
    (db, conn)
}

#[tokio::test]
async fn j5_migration_refuses_an_unresolvable_existing_space_identity() {
    let db = libsql::Builder::new_local(":memory:")
        .build()
        .await
        .unwrap();
    let conn = db.connect().unwrap();
    conn.execute_batch(
        "CREATE TABLE spaces (
             id TEXT PRIMARY KEY,
             name TEXT NOT NULL UNIQUE
         );
         CREATE TABLE genesis_coverage_state (
             space TEXT PRIMARY KEY,
             coverage_epoch INTEGER NOT NULL DEFAULT 1,
             epoch_state TEXT NOT NULL DEFAULT 'active',
             opened_at INTEGER NOT NULL,
             migration_cursor TEXT,
             genesis_enabled INTEGER NOT NULL DEFAULT 0
         );
         INSERT INTO genesis_coverage_state(space, opened_at)
         VALUES ('deleted-space', 10);",
    )
    .await
    .unwrap();
    let tx = conn.transaction().await.unwrap();
    let error = ensure_remaining_substrate(&tx)
        .await
        .expect_err("migration must not invent a stable id for an orphan proof");
    assert!(error.to_string().contains("stable space id"), "{error}");
    tx.rollback().await.unwrap();
}

#[tokio::test]
async fn j4_deploy_attestation_identity_is_collision_safe() {
    let (_db, conn) = test_connection().await;
    conn.execute(
        "INSERT INTO m6_deploy_attestation
             (attestation_id, attested_at, signature, binary_digest)
         VALUES ('attestation-a', 100, 'signature-a', 'binary-a')",
        (),
    )
    .await
    .expect("first deploy attestation");
    conn.execute(
        "INSERT INTO m6_deploy_attestation
             (attestation_id, attested_at, signature, binary_digest)
         VALUES ('attestation-b', 100, 'signature-b', 'binary-b')",
        (),
    )
    .await
    .expect("two legitimate attestations may share one Unix second");
    assert!(
        conn.execute(
            "INSERT INTO m6_deploy_attestation
                 (attestation_id, attested_at, signature, binary_digest)
             VALUES ('attestation-a', 101, 'signature-c', 'binary-c')",
            (),
        )
        .await
        .is_err(),
        "one durable attestation identity cannot be reused"
    );
    assert!(
        conn.execute(
            "INSERT INTO m6_deploy_attestation
                 (attestation_id, attested_at, signature, binary_digest)
             VALUES ('attestation-without-time', NULL, 'signature-d', 'binary-d')",
            (),
        )
        .await
        .is_err(),
        "attestation time is required data, never an implicit rowid"
    );
}

#[tokio::test]
async fn j5_mutation_counter_backfills_and_cannot_move_below_zero() {
    let (_db, conn) = test_connection().await;
    let mut rows = conn
        .query(
            "SELECT m6_mutation_count, space_id FROM genesis_coverage_state
              WHERE space = 'existing-space'",
            (),
        )
        .await
        .expect("read backfilled M6 mutation counter");
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<i64>(0).unwrap(), 0);
    assert_eq!(row.get::<String>(1).unwrap(), "space-id-existing");
    conn.execute(
        "UPDATE genesis_coverage_state SET m6_mutation_count = 2
          WHERE space = 'existing-space'",
        (),
    )
    .await
    .expect("advance the durable mutation counter");
    assert!(
        conn.execute(
            "UPDATE genesis_coverage_state SET m6_mutation_count = 1
              WHERE space = 'existing-space'",
            (),
        )
        .await
        .is_err(),
        "the durable mutation count may never decrease"
    );
    assert!(
        conn.execute(
            "INSERT OR REPLACE INTO genesis_coverage_state
                 (space, opened_at, m6_mutation_count, space_id)
             VALUES ('existing-space', 10, 1, 'space-id-existing')",
            (),
        )
        .await
        .is_err(),
        "replacement may not bypass mutation-count monotonicity"
    );
    assert!(
        conn.execute(
            "DELETE FROM genesis_coverage_state WHERE space = 'existing-space'",
            (),
        )
        .await
        .is_err(),
        "deletion may not erase the durable mutation proof"
    );
}

#[tokio::test]
async fn j5_mutation_counter_key_move_cannot_replace_destination_proof() {
    let (_db, conn) = test_connection().await;
    conn.execute(
        "UPDATE genesis_coverage_state SET m6_mutation_count = 2
          WHERE space = 'existing-space'",
        (),
    )
    .await
    .expect("advance source mutation proof");
    conn.execute(
        "INSERT INTO spaces(id, name)
         VALUES ('space-id-destination', 'destination-space')",
        (),
    )
    .await
    .expect("register destination space without renaming the source");
    conn.execute(
        "INSERT INTO genesis_coverage_state
             (space, opened_at, m6_mutation_count, space_id)
         VALUES ('destination-space', 20, 100, 'space-id-destination')",
        (),
    )
    .await
    .expect("seed higher destination mutation proof");

    assert!(
        conn.execute(
            "UPDATE OR REPLACE genesis_coverage_state
                SET space = 'destination-space'
              WHERE space = 'existing-space'",
            (),
        )
        .await
        .is_err(),
        "moving a lower proof may not replace a higher destination proof"
    );

    let mut rows = conn
        .query(
            "SELECT space, m6_mutation_count
               FROM genesis_coverage_state
              ORDER BY space",
            (),
        )
        .await
        .expect("read both mutation proofs after refused move");
    let destination = rows.next().await.unwrap().unwrap();
    assert_eq!(destination.get::<String>(0).unwrap(), "destination-space");
    assert_eq!(destination.get::<i64>(1).unwrap(), 100);
    let source = rows.next().await.unwrap().unwrap();
    assert_eq!(source.get::<String>(0).unwrap(), "existing-space");
    assert_eq!(source.get::<i64>(1).unwrap(), 2);
    assert!(rows.next().await.unwrap().is_none());
}

#[tokio::test]
async fn j5_mutation_counter_cannot_park_then_recreate_original_identity() {
    let (_db, conn) = test_connection().await;
    conn.execute(
        "UPDATE genesis_coverage_state SET m6_mutation_count = 5
          WHERE space = 'existing-space'",
        (),
    )
    .await
    .expect("advance original mutation proof");
    conn.execute(
        "INSERT INTO spaces(id, name)
         VALUES ('space-id-parked', 'parked-space')",
        (),
    )
    .await
    .expect("register a non-colliding destination space");

    let parked = conn
        .execute(
            "UPDATE OR REPLACE genesis_coverage_state
                SET space = 'parked-space', m6_mutation_count = 6
              WHERE space = 'existing-space'",
            (),
        )
        .await;
    let reset = conn
        .execute(
            "INSERT INTO genesis_coverage_state
                 (space, opened_at, m6_mutation_count, space_id)
             VALUES ('existing-space', 30, 0, 'space-id-existing')",
            (),
        )
        .await;

    assert!(
        parked.is_err(),
        "a normal DML statement cannot park a proof under an unoccupied key"
    );
    assert!(
        reset.is_err(),
        "the original identity must remain occupied and reject a zero reset"
    );
    let mut rows = conn
        .query(
            "SELECT space, m6_mutation_count FROM genesis_coverage_state",
            (),
        )
        .await
        .expect("read immutable mutation-proof identity");
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "existing-space");
    assert_eq!(row.get::<i64>(1).unwrap(), 5);
    assert!(rows.next().await.unwrap().is_none());
}

#[tokio::test]
async fn j5_mutation_counter_requires_nonnull_space_identity() {
    let (_insert_db, insert_conn) = test_connection().await;
    let null_insert = insert_conn
        .execute(
            "INSERT INTO genesis_coverage_state (space, opened_at)
             VALUES (NULL, 20)",
            (),
        )
        .await;

    let (_update_db, update_conn) = test_connection().await;
    let null_update = update_conn
        .execute(
            "UPDATE genesis_coverage_state SET space = NULL
              WHERE space = 'existing-space'",
            (),
        )
        .await;

    assert!(
        null_insert.is_err(),
        "a NULL space cannot create an identity-less mutation proof"
    );
    assert!(
        null_update.is_err(),
        "a space-key update cannot discard the mutation-proof identity"
    );
}

#[tokio::test]
async fn j6_genesis_provenance_retains_one_origin_per_page() {
    let (_db, conn) = test_connection().await;
    conn.execute(
        "INSERT INTO m6_genesis_provenance
             (page_id, content_digest, page_version, minted_at)
         VALUES ('page-a', 'digest-a', 3, 100)",
        (),
    )
    .await
    .expect("first genesis provenance row");
    assert!(
        conn.execute(
            "INSERT INTO m6_genesis_provenance
                 (page_id, content_digest, page_version, minted_at)
             VALUES ('page-a', 'digest-b', 4, 200)",
            (),
        )
        .await
        .is_err(),
        "genesis provenance is immutable page identity, not replaceable current state"
    );
    assert!(
        conn.execute(
            "UPDATE m6_genesis_provenance
                SET content_digest = 'rewritten'
              WHERE page_id = 'page-a'",
            (),
        )
        .await
        .is_err(),
        "genesis provenance content cannot be rewritten"
    );
    assert!(
        conn.execute(
            "DELETE FROM m6_genesis_provenance WHERE page_id = 'page-a'",
            (),
        )
        .await
        .is_err(),
        "genesis provenance cannot be discarded"
    );
    assert!(
        conn.execute(
            "INSERT OR REPLACE INTO m6_genesis_provenance
                 (page_id, content_digest, page_version, minted_at)
             VALUES ('page-a', 'replacement', 4, 200)",
            (),
        )
        .await
        .is_err(),
        "replacement syntax cannot bypass immutable provenance"
    );
    assert!(
        conn.execute(
            "INSERT INTO m6_genesis_provenance
                 (page_id, content_digest, page_version, minted_at)
             VALUES (NULL, 'digest-null', 1, 100)",
            (),
        )
        .await
        .is_err(),
        "a NULL page identity cannot enter provenance"
    );
}

#[tokio::test]
async fn j7_projection_outbox_is_version_keyed_and_state_checked() {
    let (_db, conn) = test_connection().await;
    for version in [1_i64, 2_i64] {
        conn.execute(
            "INSERT INTO page_projection_outbox
                 (page_id, page_version, state, content_digest)
             VALUES ('page-a', ?1, 'pending', ?2)",
            libsql::params![version, format!("digest-{version}")],
        )
        .await
        .expect("one projection intent per page version");
    }
    assert!(
        conn.execute(
            "INSERT INTO page_projection_outbox
                 (page_id, page_version, state, content_digest)
             VALUES ('page-a', 2, 'pending', 'duplicate')",
            (),
        )
        .await
        .is_err(),
        "a retry must reuse the durable projection intent"
    );
    assert!(
        conn.execute(
            "INSERT INTO page_projection_outbox
                 (page_id, page_version, state, content_digest)
             VALUES ('page-b', 1, 'unknown', 'digest-b')",
            (),
        )
        .await
        .is_err(),
        "only pending, handed_off, and complete are durable states"
    );
}

#[tokio::test]
async fn j10_counters_are_unique_and_nonnegative_per_space_and_name() {
    let (_db, conn) = test_connection().await;
    conn.execute(
        "INSERT INTO m6_counters(space_id, space, name, value)
         VALUES ('space-id-a', 'space-a', 'relevance_generation', 2)",
        (),
    )
    .await
    .expect("first named counter");
    assert!(
        conn.execute(
            "INSERT INTO m6_counters(space_id, space, name, value)
             VALUES ('space-id-a', 'space-a', 'relevance_generation', 1)",
            (),
        )
        .await
        .is_err(),
        "one counter identity must not fork into two rows"
    );
    assert!(
        conn.execute(
            "UPDATE m6_counters SET value = 1
              WHERE space = 'space-a' AND name = 'relevance_generation'",
            (),
        )
        .await
        .is_err(),
        "named counters may never decrease"
    );
    assert!(
        conn.execute(
            "INSERT OR REPLACE INTO m6_counters(space_id, space, name, value)
             VALUES ('space-id-a', 'space-a', 'relevance_generation', 1)",
            (),
        )
        .await
        .is_err(),
        "replacement may not bypass named-counter monotonicity"
    );
    assert!(
        conn.execute(
            "DELETE FROM m6_counters
              WHERE space = 'space-a' AND name = 'relevance_generation'",
            (),
        )
        .await
        .is_err(),
        "deletion may not erase a named counter"
    );
    assert!(
        conn.execute(
            "INSERT INTO m6_counters(space_id, space, name, value)
             VALUES ('space-id-a', 'space-a', 'normalization_rejection:R2', -1)",
            (),
        )
        .await
        .is_err(),
        "counters may never move below zero"
    );
}

#[tokio::test]
async fn j10_counter_key_moves_cannot_replace_destination_proofs() {
    let (_db, conn) = test_connection().await;
    conn.execute_batch(
        "INSERT INTO spaces(id, name) VALUES
             ('space-id-source', 'source-space'),
             ('space-id-destination', 'destination-space');
         INSERT INTO m6_counters(space_id, space, name, value) VALUES
             ('space-id-source', 'source-space', 'generation', 2),
             ('space-id-destination', 'destination-space', 'generation', 100),
             ('space-id-source', 'source-space', 'other', 200);",
    )
    .await
    .expect("seed source and destination counter proofs");

    assert!(
        conn.execute(
            "UPDATE OR REPLACE m6_counters
                SET space = 'destination-space'
              WHERE space = 'source-space' AND name = 'generation'",
            (),
        )
        .await
        .is_err(),
        "moving a counter may not replace an existing destination-space proof"
    );
    assert!(
        conn.execute(
            "UPDATE OR REPLACE m6_counters
                SET name = 'other'
              WHERE space = 'source-space' AND name = 'generation'",
            (),
        )
        .await
        .is_err(),
        "renaming a counter may not replace an existing destination-name proof"
    );

    let mut rows = conn
        .query(
            "SELECT space, name, value FROM m6_counters
              ORDER BY space, name",
            (),
        )
        .await
        .expect("read all counter proofs after refused moves");
    let expected = [
        ("destination-space", "generation", 100_i64),
        ("source-space", "generation", 2_i64),
        ("source-space", "other", 200_i64),
    ];
    for (expected_space, expected_name, expected_value) in expected {
        let row = rows.next().await.unwrap().unwrap();
        assert_eq!(row.get::<String>(0).unwrap(), expected_space);
        assert_eq!(row.get::<String>(1).unwrap(), expected_name);
        assert_eq!(row.get::<i64>(2).unwrap(), expected_value);
    }
    assert!(rows.next().await.unwrap().is_none());
}

#[tokio::test]
async fn j10_counter_cannot_park_then_recreate_original_identity() {
    let (_db, conn) = test_connection().await;
    conn.execute_batch(
        "INSERT INTO spaces(id, name) VALUES
             ('space-id-source', 'source-space'),
             ('space-id-parked', 'parked-space');
         INSERT INTO m6_counters(space_id, space, name, value)
         VALUES ('space-id-source', 'source-space', 'generation', 5);",
    )
    .await
    .expect("seed one counter proof and a non-colliding destination space");

    let parked = conn
        .execute(
            "UPDATE OR REPLACE m6_counters
                SET space = 'parked-space', name = 'parked', value = 6
              WHERE space = 'source-space' AND name = 'generation'",
            (),
        )
        .await;
    let reset = conn
        .execute(
            "INSERT INTO m6_counters(space_id, space, name, value)
             VALUES ('space-id-source', 'source-space', 'generation', 0)",
            (),
        )
        .await;

    assert!(
        parked.is_err(),
        "normal DML cannot change a counter's space/name identity to park it"
    );
    assert!(
        reset.is_err(),
        "the original counter identity must remain occupied and reject a zero reset"
    );
    let mut rows = conn
        .query("SELECT space, name, value FROM m6_counters", ())
        .await
        .expect("read immutable counter identity");
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "source-space");
    assert_eq!(row.get::<String>(1).unwrap(), "generation");
    assert_eq!(row.get::<i64>(2).unwrap(), 5);
    assert!(rows.next().await.unwrap().is_none());
}
