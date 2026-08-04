// SPDX-License-Identifier: Apache-2.0
//! Frozen-corpus gate for the M6 relevance benchmark.
//!
//! C0 shipped the corpus and this target's reproducibility proof; C1 adds the
//! two count gates that consume it. The ordering was deliberate: §8.6's count
//! limits are hard pass/fail gates, and a gate that arrives after the code it
//! gates has never once failed. The proof C0 carries is **corpus
//! reproducibility**: a reviewer regenerates the corpus byte-for-byte, and a
//! mismatch makes the bench refuse to run rather than report a number nobody
//! can compare against.
//!
//! **S0-99 governs every number below: a benchmark failure stops for a
//! Sol-reviewed contract amendment and may not be answered by tuning a
//! constant.** `R-BENCH-Q` (4) and `R-BENCH-ROWS` (512) are frozen. If a sweep
//! breaches them, the thing that moves is the sweep — `SWEEP_ENDPOINTS_PER_TURN`
//! is the parameter the contract left free, and it exists precisely so the
//! frozen ones never have to.
//!
//! `R-BENCH-MAX`/`P50`/`P99` are wall-clock and are **not** asserted here:
//! §8.6 rules the count limits into CI and leaves the timing ones to a local
//! receipt under S0-98's specified conditions (warm cache, one evaluation at a
//! time, no concurrent refinery turn), which a hosted runner does not provide.
//! The `sweep_turn_*_ms` fields the receipt prints are also not `R-BENCH-MAX`:
//! that limit times the *route evaluation* over `m6_adjacency`/`m6_pair_stats`,
//! and C1 builds those tables' writer, not their reader.
//!
//! ## Open: the row budget holds and the work does not (C1 measurement)
//!
//! Local receipt, `aarch64-macos`, release profile, corpus `06f0ee5c62d5e40c`:
//!
//! ```text
//! space-0  2,000 pages  10,795 groups  60 turns  3 queries  459 rows  p50 1355ms  max 1912ms
//! space-7    105 pages   1,874 groups   8 turns  3 queries  163 rows  p50  122ms  max  127ms
//! ```
//!
//! Both count gates pass with room to spare, and a bounded turn still costs
//! over a second. That gap is the defect S0-95 names in its own text — "a
//! query that scans a million rows and returns 512" — and it is why that
//! decision made `FULLSCAN_STEP` normative for *visited work* rather than the
//! decoded-row counter. `candidate_endpoints` builds its `support` and
//! `group_size` CTEs over the whole space before `stale`'s `LIMIT` cuts the
//! result to a handful of endpoints, so per-turn cost tracks corpus size while
//! the instrument that gates it does not.
//!
//! It is recorded rather than fixed here because nothing C1 ships runs on its
//! own: the lane that would hold the single `MemoryDB` connection mutex for
//! that second is C3's, and §12 orders C3 last for exactly this kind of reason.
//! It is a real precondition for that lane, not a footnote — `WENLAN_ENABLE_
//! ENTITY_PAGE_RECONCILE` is default-OFF today over the same failure mode at a
//! larger constant. Per S0-99 this is an amendment question, and the number
//! above is the evidence it needs.

use std::io::{self, Write};
use wenlan_core::eval::m6_bench_corpus::{
    canonical_manifest_digest, summarize_corpus, verify_manifest_digest, write_corpus_stream,
    M6_BENCH_SEED, M6_CORPUS_ENCODING, M6_GENERATED_ROOT_PERCENT, M6_GROUP_COUNT, M6_HUB_DEGREES,
    M6_MEMORY_COUNT, M6_PAGES_PER_GROUP_MEAN, M6_PAGE_COUNT, M6_RETRACTED_EDGE_PERCENT,
    M6_ROOT_AGE_MAX_DAYS, M6_SPACE_SPLIT_PERCENT, M6_ZIPF_TRUNCATION,
};
#[cfg(feature = "eval-harness")]
use wenlan_core::eval::m6_relevance_harness::run_relevance_budget_bench;

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

/// `R-BENCH-Q` and `R-BENCH-ROWS` over the frozen corpus.
///
/// The unit fixtures already assert both counters, but on a dozen pages, where
/// one turn sees every group and the fan-out that actually spends the budget
/// never happens. This is the same assertion against the corpus D9 states the
/// budget over — 5,000 pages, 12,000 groups, a Zipf degree tail.
///
/// **Two legs, because neither alone is the gate.** The dense 40% space is
/// where the budget is actually spent, and draining its 2,000 pages takes
/// ~400 turns at ~3.5s each — too slow to be a required check. The 2% space
/// drains completely and cheaply, but its pages are sparse. So: sample the
/// dense space deep enough to hit real fan-out, and drain the small one to
/// exhaustion so at least one leg proves the invariant holds all the way to
/// `Idle` rather than for a prefix.
///
/// What the sampled leg does **not** prove is a maximum. It could not: an
/// earlier fixed-slice design passed 24 turns and breached at turn 102, which
/// is exactly why the bound is now enforced by `pack_within_budget` rather
/// than hoped for. The residual case the packing cannot avoid — a single
/// endpoint whose own fan-out exceeds the remaining budget, which the turn
/// takes anyway so the sweep can make progress — is asserted directly in
/// `relevance_sweep_test.rs`, not left to drain luck.
#[cfg(feature = "eval-harness")]
#[test]
fn the_bounded_sweep_stays_inside_the_frozen_count_budgets() {
    // The corpus must be the committed one, or the numbers are incomparable.
    let summary = summarize_corpus().expect("corpus");
    verify_manifest_digest(MANIFEST_DIGEST_BYTES, &summary.sha256)
        .expect("corpus digest does not match the committed manifest; the bench refuses to run");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    for (space_index, max_turns, drains) in [(0usize, 60usize, false), (7, 4096, true)] {
        // The database is built here, not inside the harness:
        // `repository_module_graph_matches_r4_25_group_6_census` requires every
        // raw `libsql` origin under `crates/*/src` to be `#[cfg(test)]`-gated,
        // and the harness is behind a cargo feature instead -- which the guard
        // does not accept, and should not, since a feature can be switched on
        // in a shipped build. This target lives outside `src`, so owning the
        // connection here keeps the invariant intact.
        let receipt = runtime
            .block_on(async {
                let database = libsql::Builder::new_local(":memory:")
                    .build()
                    .await
                    .expect("build in-memory database");
                let connection = database.connect().expect("connect");
                run_relevance_budget_bench(&connection, space_index, max_turns).await
            })
            .expect("relevance budget bench");
        let widths: Vec<usize> = receipt
            .bounded_turns
            .iter()
            .map(|turn| turn.endpoints_considered)
            .collect();
        println!(
            "[m6-relevance-bench] corpus={} space={} pages={} groups={} pairs={} turns={} \
             endpoints={} width={:?}..{:?} peak_queries={} peak_rows={} \
             sweep_turn_max_ms={:.1} sweep_turn_p50_ms={:.1} machine={}",
            &summary.sha256[..16],
            receipt.space,
            receipt.pages_in_space,
            receipt.groups_in_space,
            receipt.rereference_pairs,
            receipt.bounded_turns.len(),
            widths.iter().sum::<usize>(),
            widths.iter().min(),
            widths.iter().max(),
            receipt.peak_queries(),
            receipt.peak_rows(),
            // Recorded, never asserted. These time the sweep turn -- the
            // writer C1 builds -- and are deliberately not named after
            // `R-BENCH-MAX`, which times the route evaluation that reads the
            // two tables the sweep writes. S0-99 binds the ungated half too:
            // a number here that looks wrong is a contract amendment
            // question, not a constant to relax.
            receipt.max_turn_ms(),
            receipt.p50_turn_ms(),
            receipt.machine(),
        );

        assert!(
            !receipt.bounded_turns.is_empty(),
            "space {space_index} ran no bounded turn, so neither budget was \
             exercised; a vacuous pass is what this gate exists to prevent"
        );
        if drains {
            assert!(
                receipt.bounded_turns.len() < max_turns,
                "space {space_index} never reached Idle in {max_turns} turns, so \
                 the draining leg proved nothing about convergence"
            );
        }
        assert!(
            receipt.peak_queries() <= 4,
            "R-BENCH-Q: a bounded turn on space {space_index} issued {} queries \
             against the frozen corpus; the cap is 4 and S0-99 forbids relaxing it",
            receipt.peak_queries(),
        );
        assert!(
            receipt.peak_rows() <= 512,
            "R-BENCH-ROWS: a bounded turn on space {space_index} materialized {} \
             rows against the frozen corpus; the cap is 512 and S0-99 forbids \
             relaxing it",
            receipt.peak_rows(),
        );
    }
}
