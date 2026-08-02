// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::io::{self, Write};
#[cfg(feature = "eval-harness")]
use std::path::{Path, PathBuf};
#[cfg(feature = "eval-harness")]
use std::process::Command;
use wenlan_core::eval::m5_bench_corpus::{
    corpus_summary, merge_sparse_buckets, parse_accuracy_jsonl, verify_manifest_digest,
    write_corpus_stream, AccuracyCategory, PageSizeBucket, PageSizeDistribution,
    M5_BENCH_MEMORY_COUNT, M5_BENCH_PAGE_COUNT, M5_BENCH_SEED, M5_CORPUS_ENCODING, PAGE_SIZE_K_MIN,
    PAGE_SIZE_SCHEMA_VERSION,
};

const DISTRIBUTION_BYTES: &[u8] = include_bytes!("fixtures/m5_page_size_dist.json");
const ACCURACY_BYTES: &[u8] = include_bytes!("fixtures/m5_judge_accuracy.jsonl");
const MANIFEST_DIGEST_BYTES: &[u8] = include_bytes!("fixtures/m5_bench_corpus.sha256");

fn distribution() -> PageSizeDistribution {
    PageSizeDistribution::from_json_bytes(DISTRIBUTION_BYTES).unwrap()
}

#[test]
fn frozen_seed_cardinality_stream_and_digest_are_exact_and_deterministic() {
    assert_eq!(M5_BENCH_SEED, 0x4d35_0001);
    assert_eq!(M5_BENCH_MEMORY_COUNT, 100_000);
    assert_eq!(M5_BENCH_PAGE_COUNT, 5_000);

    let first = corpus_summary(&distribution()).unwrap();
    let second = corpus_summary(&distribution()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.encoding, M5_CORPUS_ENCODING);
    assert_eq!(first.seed, M5_BENCH_SEED);
    assert_eq!(first.memory_count, 100_000);
    assert_eq!(first.page_count, 5_000);

    let mut counter = CountingWriter::default();
    write_corpus_stream(&distribution(), &mut counter).unwrap();
    assert_eq!(counter.lines, 2 + 5_000 + 100_000);
    assert!(counter.bytes > 1_000_000);
}

#[test]
fn distribution_fixture_is_strict_k_anonymous_and_privacy_minimal() {
    let parsed = distribution();
    assert_eq!(parsed.sample_size, 1019);
    assert_eq!(
        parsed
            .buckets
            .iter()
            .map(|bucket| bucket.count)
            .sum::<u64>(),
        1019
    );
    assert!(parsed
        .buckets
        .iter()
        .all(|bucket| bucket.count >= parsed.k_min));

    let value: serde_json::Value = serde_json::from_slice(DISTRIBUTION_BYTES).unwrap();
    let root_keys: BTreeSet<_> = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        root_keys,
        BTreeSet::from(["buckets", "k_min", "sample_size", "schema_version", "unit"])
    );
    for bucket in value["buckets"].as_array().unwrap() {
        let keys: BTreeSet<_> = bucket
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            BTreeSet::from(["count", "max_exclusive", "min_inclusive"])
        );
    }

    let mut mutated = value;
    mutated
        .as_object_mut()
        .unwrap()
        .insert("page_ids".into(), serde_json::json!(["private"]));
    assert!(PageSizeDistribution::from_json_bytes(&serde_json::to_vec(&mutated).unwrap()).is_err());
}

#[test]
fn accuracy_fixture_is_strict_and_has_every_required_adversarial_case() {
    let cases = parse_accuracy_jsonl(ACCURACY_BYTES).unwrap();
    for required in ["N1", "N8", "N11", "N12"] {
        assert!(cases.iter().any(|case| case.case_id == required));
    }
    for hard_veto in ["N1", "N8"] {
        let case = cases.iter().find(|case| case.case_id == hard_veto).unwrap();
        assert!(!case.entails, "{hard_veto} must be a negative judgement");
    }
    assert!(cases.iter().any(|case| {
        case.category == AccuracyCategory::NegationN1
            && case.claim == "The alarm is not armed."
            && case.span == "The alarm is armed."
            && !case.entails
    }));
    assert!(cases.iter().any(|case| {
        case.category == AccuracyCategory::QuantifierN8
            && case.claim.starts_with("All ")
            && case.span.starts_with("Some ")
            && !case.entails
    }));
    assert!(cases.iter().any(|case| {
        case.category == AccuracyCategory::TerminatorN11
            && case.claim.ends_with('?')
            && case.span.ends_with('.')
            && !case.entails
    }));
    assert!(cases.iter().any(|case| {
        case.category == AccuracyCategory::AcronymCaseN12
            && case.claim.starts_with("us ")
            && case.span.starts_with("US ")
            && !case.entails
    }));

    let extra_field = b"{\"case_id\":\"X\",\"category\":\"positive_exact\",\"claim\":\"x\",\"span\":\"x\",\"entails\":true,\"content\":\"private\"}\n";
    assert!(parse_accuracy_jsonl(extra_field).is_err());
}

#[test]
fn sparse_buckets_merge_right_tail_merges_left_and_too_small_refuses() {
    let merged = merge_sparse_buckets(
        vec![
            bucket(0, 256, 894),
            bucket(256, 512, 1),
            bucket(512, 1024, 11),
        ],
        5,
    )
    .unwrap();
    assert_eq!(merged, vec![bucket(0, 256, 894), bucket(256, 1024, 12)]);

    let tail = merge_sparse_buckets(vec![bucket(0, 256, 10), bucket(256, 512, 1)], 5).unwrap();
    assert_eq!(tail, vec![bucket(0, 512, 11)]);

    assert!(merge_sparse_buckets(vec![bucket(0, 256, 4)], 5).is_err());
}

#[test]
fn open_tail_generation_is_finite_bounded_and_deterministic() {
    let pure_open = PageSizeDistribution {
        schema_version: PAGE_SIZE_SCHEMA_VERSION,
        unit: "utf8_bytes".into(),
        k_min: PAGE_SIZE_K_MIN,
        sample_size: 5,
        buckets: vec![PageSizeBucket {
            min_inclusive: 1_000,
            max_exclusive: None,
            count: 5,
        }],
    };
    assert_open_tail_deterministic(&pure_open, 1_000, 2_000);

    let merged = merge_sparse_buckets(
        vec![
            bucket(0, 256, 10),
            PageSizeBucket {
                min_inclusive: 256,
                max_exclusive: None,
                count: 1,
            },
        ],
        PAGE_SIZE_K_MIN,
    )
    .unwrap();
    assert_eq!(
        merged,
        vec![PageSizeBucket {
            min_inclusive: 0,
            max_exclusive: None,
            count: 11,
        }]
    );
    let merged_open = PageSizeDistribution {
        schema_version: PAGE_SIZE_SCHEMA_VERSION,
        unit: "utf8_bytes".into(),
        k_min: PAGE_SIZE_K_MIN,
        sample_size: 11,
        buckets: merged,
    };
    assert_open_tail_deterministic(&merged_open, 1, 2);
}

#[test]
fn manifest_covers_generated_corpus_and_exact_fixture_bytes() {
    let corpus = corpus_summary(&distribution()).unwrap();
    verify_manifest_digest(
        MANIFEST_DIGEST_BYTES,
        &corpus.sha256,
        DISTRIBUTION_BYTES,
        ACCURACY_BYTES,
    )
    .unwrap();

    let changed_corpus = "0".repeat(64);
    assert!(verify_manifest_digest(
        MANIFEST_DIGEST_BYTES,
        &changed_corpus,
        DISTRIBUTION_BYTES,
        ACCURACY_BYTES,
    )
    .is_err());

    let mut changed_distribution = DISTRIBUTION_BYTES.to_vec();
    changed_distribution[0] ^= 1;
    assert!(verify_manifest_digest(
        MANIFEST_DIGEST_BYTES,
        &corpus.sha256,
        &changed_distribution,
        ACCURACY_BYTES,
    )
    .is_err());

    let mut changed_accuracy = ACCURACY_BYTES.to_vec();
    changed_accuracy[0] ^= 1;
    assert!(verify_manifest_digest(
        MANIFEST_DIGEST_BYTES,
        &corpus.sha256,
        DISTRIBUTION_BYTES,
        &changed_accuracy,
    )
    .is_err());
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn exporter_emits_canonical_aggregate_to_stdout() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("source.db");
    create_pages_db(&db_path, &five_ascii_pages()).await;

    let result = Command::new(env!("CARGO_BIN_EXE_m5_export_page_size_dist"))
        .arg("--db")
        .arg(&db_path)
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = PageSizeDistribution::from_json_bytes(&result.stdout).unwrap();
    assert_eq!(output.sample_size, 5);
    assert_eq!(output.buckets, vec![bucket(0, 256, 5)]);
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn exporter_is_read_only_and_emits_only_aggregate_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("fixture.db");
    create_pages_db(
        &db_path,
        &[
            page("a".repeat(255), "active"),
            page("b".repeat(255), "active"),
            page("c".repeat(255), "active"),
            page("d".repeat(255), "active"),
            page("e".repeat(255), "active"),
            page("é".repeat(128), "active"),
            page("界".repeat(200), "active"),
            page("語".repeat(200), "active"),
            page("文".repeat(200), "active"),
            page("字".repeat(200), "active"),
            page("漢".repeat(200), "active"),
            page("測".repeat(200), "active"),
            page("試".repeat(200), "active"),
            page("界".repeat(200), "active"),
            page("語".repeat(200), "active"),
            page("文".repeat(200), "active"),
            page("字".repeat(200), "active"),
            page("界".repeat(6_000), "inactive"),
        ],
    )
    .await;
    let db_before = std::fs::read(&db_path).unwrap();
    let first = run_exporter(&db_path);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(std::fs::read(&db_path).unwrap(), db_before);
    let exported = PageSizeDistribution::from_json_bytes(&first.stdout).unwrap();
    assert_eq!(exported.sample_size, 17);
    assert_eq!(
        exported.buckets,
        vec![bucket(0, 256, 5), bucket(256, 1024, 12)]
    );

    let too_small_db = directory.path().join("too-small.db");
    create_pages_db(
        &too_small_db,
        &[
            page("x".repeat(10), "active"),
            page("x".repeat(10), "active"),
            page("x".repeat(10), "active"),
            page("x".repeat(10), "active"),
        ],
    )
    .await;
    let refused = run_exporter(&too_small_db);
    assert!(!refused.status.success());
    assert!(refused.stdout.is_empty());
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn shared_page_size_connection_rejects_dml_and_ddl() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("read-only.db");
    create_pages_db(&db_path, &five_ascii_pages()).await;
    let database = wenlan_core::db::M5PageSizeSnapshotDb::open(&db_path)
        .await
        .unwrap();
    let probe = database.mutation_probe_for_test().await.unwrap();
    assert!(probe.dml_refused);
    assert!(probe.ddl_refused);
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn exporter_rejects_file_output_arguments_without_touching_files() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("source.db");
    create_pages_db(&db_path, &five_ascii_pages()).await;
    let output = directory.path().join("existing.json");
    create_pages_db(&output, &[page("y".repeat(300), "active")]).await;
    let before = std::fs::read(&output).unwrap();

    let output_result = exporter_command(&db_path)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    let overwrite_result = exporter_command(&db_path)
        .arg("--overwrite")
        .output()
        .unwrap();

    for result in [output_result, overwrite_result] {
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr)
            .contains("usage: m5_export_page_size_dist --db PATH"));
    }
    assert_eq!(std::fs::read(&output).unwrap(), before);
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn exporter_reads_live_wal_without_mutating_source() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("live.db");
    let (database, connection, wal_before) = create_live_wal_db(&db_path).await;
    let wal_path = wal_path(&db_path);

    let result = run_exporter(&db_path);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = PageSizeDistribution::from_json_bytes(&result.stdout).unwrap();
    assert_eq!(output.sample_size, 5);
    assert_eq!(std::fs::read(&wal_path).unwrap(), wal_before);

    let observer = wenlan_core::db::M5PageSizeSnapshotDb::open(&db_path)
        .await
        .unwrap();
    assert_eq!(
        observer.fixed_counts().await.unwrap(),
        [5, 0, 0, 0, 0, 0, 0, 0]
    );

    drop(connection);
    drop(database);
}

fn bucket(min_inclusive: u64, max_exclusive: u64, count: u64) -> PageSizeBucket {
    PageSizeBucket {
        min_inclusive,
        max_exclusive: Some(max_exclusive),
        count,
    }
}

#[cfg(feature = "eval-harness")]
struct PageFixture {
    content: String,
    status: &'static str,
}

#[cfg(feature = "eval-harness")]
fn page(content: String, status: &'static str) -> PageFixture {
    PageFixture { content, status }
}

#[cfg(feature = "eval-harness")]
fn five_ascii_pages() -> Vec<PageFixture> {
    (0..5).map(|_| page("x".repeat(100), "active")).collect()
}

#[cfg(feature = "eval-harness")]
async fn create_pages_db(path: &Path, pages: &[PageFixture]) {
    let database = libsql::Builder::new_local(path).build().await.unwrap();
    let connection = database.connect().unwrap();
    connection
        .execute(
            "CREATE TABLE pages (id TEXT PRIMARY KEY, content TEXT NOT NULL, status TEXT NOT NULL)",
            (),
        )
        .await
        .unwrap();
    for (index, page) in pages.iter().enumerate() {
        connection
            .execute(
                "INSERT INTO pages (id, content, status) VALUES (?1, ?2, ?3)",
                libsql::params![format!("p{index}"), page.content.clone(), page.status],
            )
            .await
            .unwrap();
    }
    drop(connection);
    drop(database);
}

#[cfg(feature = "eval-harness")]
async fn create_live_wal_db(path: &Path) -> (libsql::Database, libsql::Connection, Vec<u8>) {
    let database = libsql::Builder::new_local(path).build().await.unwrap();
    let connection = database.connect().unwrap();

    let mut journal_mode = connection
        .query("PRAGMA journal_mode=WAL", ())
        .await
        .unwrap();
    assert_eq!(
        journal_mode
            .next()
            .await
            .unwrap()
            .unwrap()
            .get::<String>(0)
            .unwrap()
            .to_ascii_lowercase(),
        "wal"
    );
    drop(journal_mode);
    let mut autocheckpoint = connection
        .query("PRAGMA wal_autocheckpoint=0", ())
        .await
        .unwrap();
    autocheckpoint.next().await.unwrap().unwrap();
    drop(autocheckpoint);
    connection
        .execute(
            "CREATE TABLE pages (id TEXT PRIMARY KEY, content TEXT NOT NULL, status TEXT NOT NULL)",
            (),
        )
        .await
        .unwrap();
    let mut checkpoint = connection
        .query("PRAGMA wal_checkpoint(TRUNCATE)", ())
        .await
        .unwrap();
    checkpoint.next().await.unwrap().unwrap();
    drop(checkpoint);
    let main_before_commit = std::fs::read(path).unwrap();

    connection.execute("BEGIN IMMEDIATE", ()).await.unwrap();
    for index in 0..5 {
        connection
            .execute(
                "INSERT INTO pages (id, content, status) VALUES (?1, ?2, 'active')",
                libsql::params![format!("p{index}"), "x".repeat(100)],
            )
            .await
            .unwrap();
    }
    connection.execute("COMMIT", ()).await.unwrap();

    let wal_before = std::fs::read(wal_path(path)).unwrap();
    assert!(!wal_before.is_empty());
    assert_eq!(std::fs::read(path).unwrap(), main_before_commit);
    (database, connection, wal_before)
}

#[cfg(feature = "eval-harness")]
fn wal_path(db_path: &Path) -> PathBuf {
    let mut path = db_path.as_os_str().to_os_string();
    path.push("-wal");
    PathBuf::from(path)
}

#[cfg(feature = "eval-harness")]
fn exporter_command(db: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_m5_export_page_size_dist"));
    command.arg("--db").arg(db);
    command
}

#[cfg(feature = "eval-harness")]
fn run_exporter(db: &Path) -> std::process::Output {
    exporter_command(db).output().unwrap()
}

#[derive(Default)]
struct CountingWriter {
    bytes: usize,
    lines: usize,
}

impl Write for CountingWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes += bytes.len();
        self.lines += bytes.iter().filter(|byte| **byte == b'\n').count();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn assert_open_tail_deterministic(
    distribution: &PageSizeDistribution,
    lower_inclusive: u64,
    upper_exclusive: u64,
) {
    let first = corpus_summary(distribution).unwrap();
    let second = corpus_summary(distribution).unwrap();
    assert_eq!(first, second);
    let mut inspector = PageSizeInspector::new(lower_inclusive, upper_exclusive);
    write_corpus_stream(distribution, &mut inspector).unwrap();
    inspector.finish();
}

struct PageSizeInspector {
    pending: Vec<u8>,
    lower_inclusive: u64,
    upper_exclusive: u64,
    pages: usize,
}

impl PageSizeInspector {
    fn new(lower_inclusive: u64, upper_exclusive: u64) -> Self {
        Self {
            pending: Vec::new(),
            lower_inclusive,
            upper_exclusive,
            pages: 0,
        }
    }

    fn inspect_line(&mut self, line: &[u8]) {
        if line.first() != Some(&b'P') {
            return;
        }
        let text = std::str::from_utf8(line).unwrap();
        let size: u64 = text.split('\t').nth(2).unwrap().parse().unwrap();
        assert!(size >= self.lower_inclusive);
        assert!(size < self.upper_exclusive);
        self.pages += 1;
    }

    fn finish(self) {
        assert!(self.pending.is_empty());
        assert_eq!(self.pages, M5_BENCH_PAGE_COUNT as usize);
    }
}

impl Write for PageSizeInspector {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        for byte in bytes {
            if *byte == b'\n' {
                let line = std::mem::take(&mut self.pending);
                self.inspect_line(&line);
            } else {
                self.pending.push(*byte);
            }
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
