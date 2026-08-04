// SPDX-License-Identifier: Apache-2.0
//! Frozen-corpus gate for the M6 relevance benchmark.
//!
//! C0 ships the corpus and this target only. The relevance sweep it will
//! measure does not exist yet — that is C1 — and the ordering is deliberate:
//! §8.6's count limits are hard pass/fail gates, and a gate that arrives after
//! the code it gates has never once failed. The single proof this target
//! carries is **corpus reproducibility**: a reviewer regenerates the corpus
//! byte-for-byte, and a mismatch makes the bench refuse to run rather than
//! report a number nobody can compare against.

use std::io::{self, Write};
use wenlan_core::eval::m6_bench_corpus::{
    canonical_manifest_digest, summarize_corpus, verify_manifest_digest, write_corpus_stream,
    M6_BENCH_SEED, M6_CORPUS_ENCODING, M6_GENERATED_ROOT_PERCENT, M6_GROUP_COUNT, M6_HUB_DEGREES,
    M6_MEMORY_COUNT, M6_PAGES_PER_GROUP_MEAN, M6_PAGE_COUNT, M6_RETRACTED_EDGE_PERCENT,
    M6_ROOT_AGE_MAX_DAYS, M6_SPACE_SPLIT_PERCENT, M6_ZIPF_TRUNCATION,
};

const MANIFEST_DIGEST_BYTES: &[u8] = include_bytes!("fixtures/m6_bench_corpus.sha256");

#[test]
fn the_frozen_identity_is_exactly_s0_97() {
    // Each constant is asserted verbatim so an amendment has to change this
    // test too, and therefore has to say what it is changing (S0-99).
    assert_eq!(M6_BENCH_SEED, 0x6D36_0000, "R-CORPUS-SEED");
    assert_eq!(M6_MEMORY_COUNT, 100_000, "R-CORPUS-MEM");
    assert_eq!(M6_PAGE_COUNT, 5_000, "R-CORPUS-PAGE");
    assert_eq!(
        M6_SPACE_SPLIT_PERCENT,
        [40, 20, 12, 10, 8, 5, 3, 2],
        "R-CORPUS-SPACE"
    );
    assert_eq!(M6_GROUP_COUNT, 12_000, "R-CORPUS-GROUP");
    assert_eq!(M6_ZIPF_TRUNCATION, 5_000, "R-CORPUS-ZIPF");
    assert_eq!(M6_HUB_DEGREES, [5_000, 1_024, 65], "R-CORPUS-HUB");
    assert_eq!(M6_PAGES_PER_GROUP_MEAN, 8, "R-CORPUS-FANOUT");
    assert_eq!(M6_ROOT_AGE_MAX_DAYS, 720, "R-CORPUS-AGE");
    assert_eq!(M6_GENERATED_ROOT_PERCENT, 15, "R-CORPUS-GENFRAC");
    assert_eq!(M6_RETRACTED_EDGE_PERCENT, 5, "R-CORPUS-RETFRAC");
}

#[test]
fn the_corpus_regenerates_byte_for_byte() {
    let first = summarize_corpus().expect("first generation");
    let second = summarize_corpus().expect("second generation");
    assert_eq!(
        first, second,
        "the corpus is not reproducible from its seed; it cannot serve as a gate"
    );
    assert_eq!(first.encoding, M6_CORPUS_ENCODING);
    assert_eq!(first.seed, M6_BENCH_SEED);
}

#[test]
fn the_manifest_matches_the_checked_in_fixture() {
    let summary = summarize_corpus().expect("corpus");
    // This is the refusal the whole slice exists for. On mismatch the message
    // carries both digests, so a reviewer can tell drift from regression.
    verify_manifest_digest(MANIFEST_DIGEST_BYTES, &summary.sha256)
        .expect("checked-in M6 manifest digest does not match the regenerated corpus");
}

#[test]
fn the_manifest_refuses_tampering() {
    let summary = summarize_corpus().expect("corpus");
    let digest = canonical_manifest_digest(&summary.sha256).expect("manifest");

    // Missing trailing LF.
    assert!(verify_manifest_digest(digest.as_bytes(), &summary.sha256).is_err());
    // Right shape, wrong value.
    let mut wrong = vec![b'0'; 64];
    wrong.push(b'\n');
    assert!(verify_manifest_digest(&wrong, &summary.sha256).is_err());
    // Uppercase hex is not the frozen encoding.
    let mut upper = digest.to_ascii_uppercase().into_bytes();
    upper.push(b'\n');
    assert!(verify_manifest_digest(&upper, &summary.sha256).is_err());
    // A corpus digest that is not 64 lowercase hex characters.
    assert!(canonical_manifest_digest("not-a-digest").is_err());
}

#[test]
fn the_stream_is_well_formed_and_carries_every_r_corpus_constant() {
    let mut counter = RecordCounter::default();
    write_corpus_stream(&mut counter).expect("stream");
    counter.finish();
}

/// Streaming validator: counts records without ever holding the corpus.
#[derive(Default)]
struct RecordCounter {
    pending: Vec<u8>,
    header_seen: bool,
    spaces: u64,
    groups: u64,
    pages: u64,
    edges: u64,
    memories: u64,
    space_memory_total: u64,
    degree_total: u64,
    fanout_total: u64,
    retracted_total: u64,
    generated_groups: u64,
    hub_degrees_seen: Vec<u64>,
    max_root_age: u64,
    last_kind: u8,
}

impl RecordCounter {
    fn inspect_line(&mut self, line: &[u8]) {
        let text = std::str::from_utf8(line).expect("corpus is ASCII");
        if !self.header_seen {
            assert_eq!(text, M6_CORPUS_ENCODING, "first line must be the encoding");
            self.header_seen = true;
            return;
        }
        let mut fields = text.split('\t');
        let kind = fields.next().expect("record kind").as_bytes()[0];
        let numbers: Vec<u64> = fields
            .map(|field| field.parse::<u64>().expect("base-10 integer field"))
            .collect();

        // Records must appear in encoding order: S C G P E M.
        let rank = b"SCGPEM"
            .iter()
            .position(|candidate| *candidate == kind)
            .expect("known record kind") as u8;
        let last_rank = if self.last_kind == 0 {
            0
        } else {
            b"SCGPEM"
                .iter()
                .position(|candidate| *candidate == self.last_kind)
                .expect("known record kind") as u8
        };
        assert!(
            rank >= last_rank,
            "records are out of encoding order at {text}"
        );
        self.last_kind = kind;

        match kind {
            b'S' => {
                assert_eq!(numbers[0], M6_BENCH_SEED);
                assert_eq!(numbers[1], M6_MEMORY_COUNT);
                assert_eq!(numbers[2], M6_PAGE_COUNT);
                assert_eq!(numbers[3], M6_SPACE_SPLIT_PERCENT.len() as u64);
                assert_eq!(numbers[4], M6_GROUP_COUNT);
            }
            b'C' => {
                self.spaces += 1;
                self.space_memory_total += numbers[1];
            }
            b'G' => {
                let (degree, fanout, age, generated, retracted) =
                    (numbers[1], numbers[2], numbers[3], numbers[4], numbers[5]);
                self.groups += 1;
                self.degree_total += degree;
                self.fanout_total += fanout;
                self.retracted_total += retracted;
                self.generated_groups += generated;
                self.max_root_age = self.max_root_age.max(age);
                assert!(
                    (1..=M6_ZIPF_TRUNCATION).contains(&degree),
                    "degree {degree} outside the truncated Zipf support"
                );
                assert!(
                    age <= M6_ROOT_AGE_MAX_DAYS,
                    "root age {age} exceeds R-CORPUS-AGE"
                );
                assert!(generated <= 1, "generated flag must be 0 or 1");
                if M6_HUB_DEGREES.contains(&degree) {
                    self.hub_degrees_seen.push(degree);
                }
            }
            b'P' => {
                self.pages += 1;
                assert!(numbers[1] < M6_SPACE_SPLIT_PERCENT.len() as u64);
            }
            b'E' => {
                self.edges += 1;
                assert!(numbers[1] < M6_PAGE_COUNT);
            }
            b'M' => {
                self.memories += 1;
                assert!(numbers[1] < M6_SPACE_SPLIT_PERCENT.len() as u64);
                assert!(numbers[2] < M6_GROUP_COUNT);
            }
            other => panic!("unknown record kind {}", other as char),
        }
    }

    fn finish(mut self) {
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.inspect_line(&pending);
        }
        assert_eq!(self.spaces, M6_SPACE_SPLIT_PERCENT.len() as u64);
        assert_eq!(self.groups, M6_GROUP_COUNT, "R-CORPUS-GROUP");
        assert_eq!(self.pages, M6_PAGE_COUNT, "R-CORPUS-PAGE");
        assert_eq!(self.memories, M6_MEMORY_COUNT, "R-CORPUS-MEM");
        assert_eq!(self.space_memory_total, M6_MEMORY_COUNT, "R-CORPUS-SPACE");
        assert_eq!(self.edges, self.fanout_total, "every fanout emits an edge");

        // R-CORPUS-FANOUT: mean pages per group is 8 over the groups the
        // balanced draw covers. The three hubs are not among them — a hub's
        // page degree is its hub degree (R-CORPUS-HUB), which is what makes
        // `hub_weight`'s `64/d` branch reachable at all — and 11,997 does not
        // divide the fanout span, so the remainder is carried across the low
        // values. 95,958 over 11,997 groups is a realized mean of 7.9985.
        let hub_pages: u64 = M6_HUB_DEGREES.iter().sum();
        let non_hub_groups = M6_GROUP_COUNT - M6_HUB_DEGREES.len() as u64;
        assert_eq!(non_hub_groups, 11_997);
        assert_eq!(
            self.fanout_total,
            95_958 + hub_pages,
            "R-CORPUS-FANOUT mean drifted"
        );
        assert_eq!(
            (95_958 * 10 + non_hub_groups / 2) / non_hub_groups,
            M6_PAGES_PER_GROUP_MEAN * 10,
            "the carried remainder must still round to R-CORPUS-FANOUT's mean"
        );
        // R-CORPUS-GENFRAC: exactly 15% of groups.
        assert_eq!(
            self.generated_groups,
            M6_GROUP_COUNT * M6_GENERATED_ROOT_PERCENT / 100,
            "R-CORPUS-GENFRAC"
        );
        // R-CORPUS-RETFRAC: exactly 5% of all root edges.
        assert_eq!(
            self.retracted_total,
            self.degree_total * M6_RETRACTED_EDGE_PERCENT / 100,
            "R-CORPUS-RETFRAC"
        );
        // R-CORPUS-HUB: all three present, including the off-by-one boundary.
        for hub in M6_HUB_DEGREES {
            assert!(
                self.hub_degrees_seen.contains(&hub),
                "hub degree {hub} is absent; a corpus without 65 lets a `>` \
                 vs `>=` top-64 truncation bug pass"
            );
        }
    }
}

impl Write for RecordCounter {
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
