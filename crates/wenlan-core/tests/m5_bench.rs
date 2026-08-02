// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::io::{self, Write};
#[cfg(feature = "eval-harness")]
use std::path::Path;
#[cfg(feature = "eval-harness")]
use std::process::Command;
#[cfg(feature = "eval-harness")]
use wenlan_core::eval::m5_bench_corpus::distribution_from_fixed_counts;
use wenlan_core::eval::m5_bench_corpus::{
    canonical_manifest_digest, corpus_summary, merge_sparse_buckets, parse_accuracy_jsonl,
    verify_manifest_digest, write_corpus_stream, AccuracyCategory, PageSizeBucket,
    PageSizeDistribution, M5_BENCH_MEMORY_COUNT, M5_BENCH_PAGE_COUNT, M5_BENCH_SEED,
    M5_CORPUS_ENCODING, PAGE_SIZE_K_MIN, PAGE_SIZE_SCHEMA_VERSION,
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

    let expected =
        canonical_manifest_digest(&corpus.sha256, DISTRIBUTION_BYTES, ACCURACY_BYTES).unwrap();
    let changed_corpus = "0".repeat(64);
    assert_ne!(
        canonical_manifest_digest(&changed_corpus, DISTRIBUTION_BYTES, ACCURACY_BYTES).unwrap(),
        expected
    );

    let mut changed_distribution = DISTRIBUTION_BYTES.to_vec();
    changed_distribution[0] ^= 1;
    assert_ne!(
        canonical_manifest_digest(&corpus.sha256, &changed_distribution, ACCURACY_BYTES).unwrap(),
        expected
    );

    let mut changed_accuracy = ACCURACY_BYTES.to_vec();
    changed_accuracy[0] ^= 1;
    assert_ne!(
        canonical_manifest_digest(&corpus.sha256, DISTRIBUTION_BYTES, &changed_accuracy).unwrap(),
        expected
    );
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn exporter_is_read_only_aggregate_only_atomic_and_no_clobber() {
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
    let output_path = directory.path().join("distribution.json");

    let first = run_exporter(&db_path, &output_path, false);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert_eq!(std::fs::read(&db_path).unwrap(), db_before);
    let output_before = std::fs::read(&output_path).unwrap();
    let exported = PageSizeDistribution::from_json_bytes(&output_before).unwrap();
    assert_eq!(exported.sample_size, 17);
    assert_eq!(
        exported.buckets,
        vec![bucket(0, 256, 5), bucket(256, 1024, 12)]
    );

    let no_clobber = run_exporter(&db_path, &output_path, false);
    assert!(!no_clobber.status.success());
    assert_eq!(std::fs::read(&output_path).unwrap(), output_before);
    assert_eq!(std::fs::read(&db_path).unwrap(), db_before);

    let overwrite = run_exporter(&db_path, &output_path, true);
    assert!(overwrite.status.success());
    assert_eq!(std::fs::read(&output_path).unwrap(), output_before);
    assert_eq!(std::fs::read(&db_path).unwrap(), db_before);

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
    let refused_path = directory.path().join("refused.json");
    let refused = run_exporter(&too_small_db, &refused_path, false);
    assert!(!refused.status.success());
    assert!(!refused_path.exists());
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn shared_page_size_connection_rejects_insert_and_ddl() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("read-only.db");
    create_pages_db(&db_path, &five_ascii_pages()).await;
    let database = wenlan_core::db::M5PageSizeSnapshotDb::open(&db_path)
        .await
        .unwrap();
    let probe = database.mutation_probe_for_test().await.unwrap();
    assert!(probe.insert_refused);
    assert!(probe.ddl_refused);
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn exporter_rejects_database_output_aliases_before_data_loss() {
    let directory = tempfile::tempdir().unwrap();
    let nested = directory.path().join("nested");
    std::fs::create_dir(&nested).unwrap();

    let exact_db = directory.path().join("exact.db");
    create_pages_db(&exact_db, &five_ascii_pages()).await;
    assert_collision_refused_and_preserved(&exact_db, &exact_db);

    let dotdot_db = directory.path().join("dotdot.db");
    create_pages_db(&dotdot_db, &five_ascii_pages()).await;
    let dotdot_alias = nested.join("..").join("dotdot.db");
    assert_collision_refused_and_preserved(&dotdot_db, &dotdot_alias);

    #[cfg(unix)]
    {
        let symlink_db = directory.path().join("symlink.db");
        create_pages_db(&symlink_db, &five_ascii_pages()).await;
        let alias = directory.path().join("symlink-alias.db");
        std::os::unix::fs::symlink(&symlink_db, &alias).unwrap();
        assert_collision_refused_and_preserved(&symlink_db, &alias);
    }
}

#[cfg(all(feature = "eval-harness", unix))]
#[tokio::test]
async fn prepared_snapshot_refuses_parent_retarget_before_temp_creation() {
    let directory = tempfile::tempdir().unwrap();
    let safe_dir = directory.path().join("safe");
    let db_dir = directory.path().join("db");
    std::fs::create_dir(&safe_dir).unwrap();
    std::fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("origin_memory.db");
    create_pages_db(&db_path, &five_ascii_pages()).await;
    let db_bytes = std::fs::read(&db_path).unwrap();
    let db_metadata = std::fs::metadata(&db_path).unwrap();
    let parent_link = directory.path().join("parent-link");
    std::os::unix::fs::symlink(&safe_dir, &parent_link).unwrap();
    let output = parent_link.join("origin_memory.db");

    let prepared =
        wenlan_core::eval::m5_snapshot_io::prepare_m5_snapshot(&db_path, &output, true).unwrap();
    std::fs::remove_file(&parent_link).unwrap();
    std::os::unix::fs::symlink(&db_dir, &parent_link).unwrap();

    assert!(prepared.write(b"replacement\n").is_err());
    let after = std::fs::metadata(&db_path).unwrap();
    assert_eq!(after.len(), db_metadata.len());
    assert_eq!(after.modified().unwrap(), db_metadata.modified().unwrap());
    assert_eq!(std::fs::read(&db_path).unwrap(), db_bytes);
    assert_eq!(std::fs::read_dir(&safe_dir).unwrap().count(), 0);
    assert_eq!(std::fs::read_dir(&db_dir).unwrap().count(), 1);
}

#[cfg(feature = "eval-harness")]
#[tokio::test]
async fn concurrent_export_is_complete_for_no_clobber_and_overwrite() {
    let directory = tempfile::tempdir().unwrap();
    let first_db = directory.path().join("first.db");
    create_pages_db(&first_db, &five_ascii_pages()).await;
    let first_expected = distribution_from_fixed_counts([5, 0, 0, 0, 0, 0, 0, 0])
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();

    let no_clobber_output = directory.path().join("no-clobber.json");
    let left = spawn_exporter(&first_db, &no_clobber_output, false);
    let right = spawn_exporter(&first_db, &no_clobber_output, false);
    let left = left.wait_with_output().unwrap();
    let right = right.wait_with_output().unwrap();
    assert_eq!(
        usize::from(left.status.success()) + usize::from(right.status.success()),
        1
    );
    assert_eq!(std::fs::read(&no_clobber_output).unwrap(), first_expected);

    let second_db = directory.path().join("second.db");
    create_pages_db(
        &second_db,
        &(0..10)
            .map(|_| page("界".repeat(200), "active"))
            .collect::<Vec<_>>(),
    )
    .await;
    let second_expected = distribution_from_fixed_counts([0, 0, 10, 0, 0, 0, 0, 0])
        .unwrap()
        .to_canonical_json_bytes()
        .unwrap();
    let overwrite_output = directory.path().join("overwrite.json");
    std::fs::write(&overwrite_output, b"old\n").unwrap();
    let left = spawn_exporter(&first_db, &overwrite_output, true);
    let right = spawn_exporter(&second_db, &overwrite_output, true);
    let left = left.wait_with_output().unwrap();
    let right = right.wait_with_output().unwrap();
    assert!(left.status.success());
    assert!(right.status.success());
    let final_bytes = std::fs::read(&overwrite_output).unwrap();
    assert!(final_bytes == first_expected || final_bytes == second_expected);
    PageSizeDistribution::from_json_bytes(&final_bytes).unwrap();
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
fn exporter_command(db: &Path, output: &Path, overwrite: bool) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_m5_export_page_size_dist"));
    command.arg("--db").arg(db).arg("--output").arg(output);
    if overwrite {
        command.arg("--overwrite");
    }
    command
}

#[cfg(feature = "eval-harness")]
fn run_exporter(db: &Path, output: &Path, overwrite: bool) -> std::process::Output {
    exporter_command(db, output, overwrite).output().unwrap()
}

#[cfg(feature = "eval-harness")]
fn spawn_exporter(db: &Path, output: &Path, overwrite: bool) -> std::process::Child {
    exporter_command(db, output, overwrite).spawn().unwrap()
}

#[cfg(feature = "eval-harness")]
fn assert_collision_refused_and_preserved(db: &Path, output_alias: &Path) {
    let bytes = std::fs::read(db).unwrap();
    let metadata = std::fs::metadata(db).unwrap();
    let modified = metadata.modified().unwrap();
    let result = run_exporter(db, output_alias, true);
    assert!(!result.status.success());
    let after = std::fs::metadata(db).unwrap();
    assert_eq!(after.len(), metadata.len());
    assert_eq!(after.modified().unwrap(), modified);
    assert_eq!(std::fs::read(db).unwrap(), bytes);
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
