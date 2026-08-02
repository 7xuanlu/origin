// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use wenlan_core::eval::m5_bench_corpus::{
    canonical_manifest_digest, corpus_summary, merge_sparse_buckets, parse_accuracy_jsonl,
    verify_manifest_digest, write_corpus_stream, AccuracyCategory, PageSizeBucket,
    PageSizeDistribution, M5_BENCH_MEMORY_COUNT, M5_BENCH_PAGE_COUNT, M5_BENCH_SEED,
    M5_CORPUS_ENCODING,
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

#[tokio::test]
async fn exporter_is_read_only_aggregate_only_atomic_and_no_clobber() {
    let directory = tempfile::tempdir().unwrap();
    let db_path = directory.path().join("fixture.db");
    create_pages_db(
        &db_path,
        &[
            100, 100, 100, 100, 100, 300, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
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
    create_pages_db(&too_small_db, &[10, 10, 10, 10]).await;
    let refused_path = directory.path().join("refused.json");
    let refused = run_exporter(&too_small_db, &refused_path, false);
    assert!(!refused.status.success());
    assert!(!refused_path.exists());
}

fn bucket(min_inclusive: u64, max_exclusive: u64, count: u64) -> PageSizeBucket {
    PageSizeBucket {
        min_inclusive,
        max_exclusive: Some(max_exclusive),
        count,
    }
}

async fn create_pages_db(path: &Path, lengths: &[usize]) {
    let database = libsql::Builder::new_local(path).build().await.unwrap();
    let connection = database.connect().unwrap();
    connection
        .execute(
            "CREATE TABLE pages (id TEXT PRIMARY KEY, content TEXT NOT NULL, status TEXT NOT NULL)",
            (),
        )
        .await
        .unwrap();
    for (index, length) in lengths.iter().enumerate() {
        connection
            .execute(
                "INSERT INTO pages (id, content, status) VALUES (?1, ?2, 'active')",
                libsql::params![format!("p{index}"), "x".repeat(*length)],
            )
            .await
            .unwrap();
    }
    drop(connection);
    drop(database);
}

fn run_exporter(db: &Path, output: &Path, overwrite: bool) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_m5_export_page_size_dist"));
    command.arg("--db").arg(db).arg("--output").arg(output);
    if overwrite {
        command.arg("--overwrite");
    }
    command.output().unwrap()
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
