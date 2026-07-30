use super::{
    assert_no_privacy_canaries, normalize_json, population_after_sample_cap, validate_population,
    DbBytesFingerprint, DbSemanticFingerprint, LintClock, PageBytesFingerprint, PrivacyCanaries,
};
use crate::db::tests::test_db;
use crate::lint::snapshot::LintReadSnapshot;
use serde_json::json;
use std::fs;
use std::path::Path;

#[tokio::test]
async fn semantic_fingerprint_detects_same_count_mutation_in_every_catalogued_table() {
    // Given: three user tables, including telemetry and an otherwise unqueried table.
    let dir = tempfile::tempdir().expect("temporary database root");
    let db = libsql::Builder::new_local(dir.path().join("catalog.db"))
        .build()
        .await
        .expect("test database opens");
    let connection = db.connect().expect("test connection opens");
    connection
        .execute_batch(
            "CREATE TABLE access_log (id INTEGER PRIMARY KEY, value BLOB);\
             CREATE TABLE agent_activity (id INTEGER PRIMARY KEY, value BLOB);\
             CREATE TABLE \"unqueried table\" (id INTEGER PRIMARY KEY, value BLOB);\
             INSERT INTO access_log VALUES (1, X'00');\
             INSERT INTO agent_activity VALUES (1, X'00');\
             INSERT INTO \"unqueried table\" VALUES (1, X'00');",
        )
        .await
        .expect("catalog fixture seeds");
    let baseline = semantic_fingerprint(&db).await;

    // When: each catalogued table receives a same-row-count one-row mutation.
    for table in baseline.table_names() {
        connection
            .execute(
                &format!("UPDATE {} SET value = X'01' WHERE id = 1", quoted(table)),
                (),
            )
            .await
            .expect("catalogued table mutates");
        let changed = semantic_fingerprint(&db).await;

        // Then: the semantic oracle detects that table's mutation.
        assert_ne!(baseline, changed, "mutation escaped table {table}");
        assert_ne!(baseline.table(table), changed.table(table));
        connection
            .execute(
                &format!("UPDATE {} SET value = X'00' WHERE id = 1", quoted(table)),
                (),
            )
            .await
            .expect("catalogued table restores");
    }
}

#[test]
fn durable_db_fingerprint_detects_db_and_wal_bytes_but_ignores_shm() {
    // Given: the durable DB/WAL pair and the transient shared-memory file.
    let dir = tempfile::tempdir().expect("temporary database bytes root");
    write(dir.path(), "origin_memory.db", b"db-0");
    write(dir.path(), "origin_memory.db-wal", b"wal-0");
    write(dir.path(), "origin_memory.db-shm", b"shm-0");
    let baseline = DbBytesFingerprint::capture(dir.path()).expect("baseline DB bytes");

    // When/Then: either durable file changes the oracle.
    for (path, changed) in [
        ("origin_memory.db", b"db-1".as_slice()),
        ("origin_memory.db-wal", b"wal-1".as_slice()),
    ] {
        let original = fs::read(dir.path().join(path)).expect("durable fixture reads");
        write(dir.path(), path, changed);
        assert_ne!(
            baseline,
            DbBytesFingerprint::capture(dir.path()).expect("changed DB bytes")
        );
        write(dir.path(), path, &original);
    }

    // When/Then: an SHM-only difference is explicitly ignored.
    write(dir.path(), "origin_memory.db-shm", b"shm-1");
    assert_eq!(
        baseline,
        DbBytesFingerprint::capture(dir.path()).expect("SHM-ignored DB bytes")
    );
}

#[test]
fn page_fingerprint_detects_one_byte_in_every_projection_artifact() {
    // Given: every durable Page projection class plus an unrelated nested artifact.
    let dir = tempfile::tempdir().expect("temporary Page root");
    let artifacts = [
        (".wenlan/state.json", b"{\"pages\":{}}".as_slice()),
        ("page.md", b"---\norigin_id: page_a\n---\nbody\n".as_slice()),
        ("_sources/.manifest.json", b"{\"pages\":{}}".as_slice()),
        ("_sources/mem_a.md", b"origin_stub: true\n".as_slice()),
        ("nested/other.bin", b"other-0".as_slice()),
    ];
    for (path, bytes) in artifacts {
        write(dir.path(), path, bytes);
    }
    let baseline = PageBytesFingerprint::capture(dir.path()).expect("baseline Page bytes");

    // When: one byte changes in each artifact.
    for (path, _) in artifacts {
        let original = fs::read(dir.path().join(path)).expect("Page artifact reads");
        let mut changed = original.clone();
        changed[0] ^= 1;
        write(dir.path(), path, &changed);

        // Then: the full-tree byte oracle detects every class.
        assert_ne!(
            baseline,
            PageBytesFingerprint::capture(dir.path()).expect("changed Page bytes"),
            "mutation escaped Page artifact {path}"
        );
        write(dir.path(), path, &original);
    }
}

#[test]
fn privacy_canaries_are_found_in_negative_controls() {
    // Given: content, filename, malformed identity/value, path, host, env, and error canaries.
    for canary in PrivacyCanaries::all() {
        // When: a negative-control output contains one forbidden value.
        let output = format!("safe-prefix {canary} safe-suffix");
        let failure = std::panic::catch_unwind(|| assert_no_privacy_canaries(&output));

        // Then: the privacy oracle fails closed for that value.
        assert!(failure.is_err(), "privacy canary escaped: {canary}");
    }
}

#[test]
fn json_normalization_is_deterministic_with_fixed_lint_clock() {
    // Given: equivalent reports with different observations and durations.
    let clock = LintClock::fixed();
    let first = json!({
        "observed_at": 1,
        "duration_ms": 99,
        "nested": {"step_duration_ms": 12, "stable": "value"}
    });
    let second = json!({
        "duration_ms": 2,
        "nested": {"stable": "value", "step_duration_ms": 77},
        "observed_at": 9
    });

    // When: both are normalized against one captured clock.
    let first = normalize_json(&first, clock);
    let second = normalize_json(&second, clock);

    // Then: serialized bytes are stable while nonvolatile data remains.
    assert_eq!(first, second);
    assert_eq!(first["observed_at"], clock.observed_at());
    assert_eq!(first["duration_ms"], 0);
    assert_eq!(first["nested"]["stable"], "value");
    assert_eq!(
        serde_json::to_vec(&first).expect("first normalized JSON serializes"),
        serde_json::to_vec(&second).expect("second normalized JSON serializes")
    );
}

#[test]
fn defect_after_example_cap_still_fails_full_population() {
    // Given: 101 defective rows and an evidence cap of 100.
    let population = population_after_sample_cap();

    // When: validation covers the complete population.
    let result = validate_population(&population);

    // Then: row 101 contributes to failure while returned examples stay bounded.
    assert!(result.failed());
    assert_eq!(result.population_total(), 101);
    assert_eq!(result.validated_total(), 101);
    assert!(result.validated_row(101));
    assert_eq!(result.examples().len(), 100);
    assert!(result.truncated());
}

#[tokio::test]
async fn telemetry_writer_fails_oracle_and_empty_lint_read_passes() {
    // Given: one isolated migrated database and its semantic fingerprint.
    let (db, _dir) = test_db().await;
    let before_writer = memory_db_semantic_fingerprint(&db).await;

    // When: the known search-telemetry writer records an access.
    db.log_accesses(&["mem_telemetry_canary".to_string()])
        .await
        .expect("telemetry writer succeeds");
    let after_writer = memory_db_semantic_fingerprint(&db).await;

    // Then: the non-mutation oracle fails the negative control.
    assert_ne!(before_writer, after_writer);
    assert_ne!(
        before_writer.table("access_log"),
        after_writer.table("access_log")
    );

    // Given: the post-writer state is now the baseline.
    let before_read = after_writer;
    let snapshot = db.open_lint_snapshot().await.expect("lint snapshot opens");

    // When: an empty lint-shaped read runs and closes its Todo 2 snapshot.
    let mut rows = snapshot
        .query(
            "SELECT COUNT(*) FROM memories",
            libsql::params::Params::None,
        )
        .await
        .expect("empty lint query prepares");
    let count = rows
        .next()
        .await
        .expect("empty lint query steps")
        .expect("empty lint query returns one row")
        .get::<i64>(0)
        .expect("empty lint count is an integer");
    assert_eq!(count, 0);
    drop(rows);
    snapshot.finish().await.expect("lint snapshot closes");

    // Then: the same oracle proves the read changed no user table.
    assert_eq!(before_read, memory_db_semantic_fingerprint(&db).await);
}

async fn memory_db_semantic_fingerprint(db: &crate::db::MemoryDB) -> DbSemanticFingerprint {
    let snapshot = db
        .open_isolated_lint_snapshot_for_test()
        .await
        .expect("semantic snapshot opens");
    let fingerprint = DbSemanticFingerprint::capture(&snapshot)
        .await
        .expect("semantic fingerprint succeeds");
    snapshot.finish().await.expect("semantic snapshot closes");
    fingerprint
}

async fn semantic_fingerprint(database: &libsql::Database) -> DbSemanticFingerprint {
    let snapshot = LintReadSnapshot::open(database)
        .await
        .expect("semantic snapshot opens");
    let fingerprint = DbSemanticFingerprint::capture(&snapshot)
        .await
        .expect("semantic fingerprint succeeds");
    snapshot.finish().await.expect("semantic snapshot closes");
    fingerprint
}

fn quoted(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn write(root: &Path, relative: &str, bytes: &[u8]) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("fixture parent creates");
    }
    fs::write(path, bytes).expect("fixture artifact writes");
}
