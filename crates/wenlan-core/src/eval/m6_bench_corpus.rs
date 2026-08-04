// SPDX-License-Identifier: Apache-2.0
//! Frozen, allocation-bounded data plane for the M6 relevance benchmark.
//!
//! This is the sibling of [`super::m5_bench_corpus`], not an extension of it:
//! M5's public surface is page-shaped end to end and none of S0-97's graph
//! layer (provenance roots, independence groups, a Zipf degree distribution,
//! hub topology, generated/retracted fractions) is expressible in it. What is
//! reused is the freeze machinery — `SplitMix64`, `sha256_hex`,
//! `validate_sha256_hex`, `DigestWriter` — which is generic.
//!
//! # Corpus encoding `wenlan-m6-bench-corpus-v1`
//!
//! ASCII with LF line endings. Integer fields are base-10, separated by one
//! tab, with no padding. Records appear in exactly this order:
//!
//! ```text
//! wenlan-m6-bench-corpus-v1
//! S <seed> <memories> <pages> <spaces> <groups>
//! C <space> <memory_count>                                        x 8
//! G <group> <degree> <fanout> <root_age_days> <generated> <retracted>  x 12000
//! P <page> <space>                                                x 5000
//! E <group> <page>                                                x sum(fanout)
//! M <memory> <space> <group>                                      x 100000
//! ```
//!
//! The digest is SHA-256 over those exact bytes. The stream is written
//! directly into the digest, so the ~213k records are never retained as a
//! corpus-sized allocation.
//!
//! # Determinism across platforms
//!
//! The Zipf weights are computed in **integer** arithmetic only. A frozen
//! corpus has to reproduce byte-for-byte on macOS and Windows (both are
//! scheduled by the `m6-platform` CI filter), and `f64::powf` is not
//! guaranteed bit-identical across platform libm implementations. A last-ULP
//! difference would land as a hard digest mismatch on one platform only — the
//! refusal would be correct but the cause would be near-impossible to read.
//! `fixed_tenth_root` sidesteps it: `k^1.1 == k * k^0.1`, and `k^0.1` is an
//! integer 10th root computed by binary search in Q20 fixed point.
//!
//! # Manifest encoding `wenlan-m6-bench-manifest-v1`
//!
//! Hashes the corpus digest together with every R-CORPUS-* constant, so an
//! amendment to the composition moves the manifest digest even in the case
//! where the corpus stream itself happened to collide. S0-99 forbids answering
//! a benchmark failure by tuning a constant; this makes such a tune visible.

use anyhow::{bail, Result};
use sha2::Sha256;
use std::io::Write;

use super::m5_bench_corpus::{sha256_hex, validate_sha256_hex, DigestWriter, SplitMix64};

/// `R-CORPUS-SEED` — a `const`, never a flag or an env var.
pub const M6_BENCH_SEED: u64 = 0x6D36_0000;
/// `R-CORPUS-MEM`
pub const M6_MEMORY_COUNT: u64 = 100_000;
/// `R-CORPUS-PAGE`
pub const M6_PAGE_COUNT: u64 = 5_000;
/// `R-CORPUS-SPACE` — 8 spaces, sizes as a percentage split of the memories.
pub const M6_SPACE_SPLIT_PERCENT: [u64; 8] = [40, 20, 12, 10, 8, 5, 3, 2];
/// `R-CORPUS-GROUP`
pub const M6_GROUP_COUNT: u64 = 12_000;
/// `R-CORPUS-ZIPF` — exponent 1.1 as an exact rational, so no float is needed.
pub const M6_ZIPF_EXPONENT_NUMERATOR: u64 = 11;
/// `R-CORPUS-ZIPF`
pub const M6_ZIPF_EXPONENT_DENOMINATOR: u64 = 10;
/// `R-CORPUS-ZIPF` — the distribution is truncated here.
pub const M6_ZIPF_TRUNCATION: u64 = 5_000;
/// `R-CORPUS-HUB` — three explicit hub groups at exactly these degrees.
///
/// 5,000 is G6's named worst case; 1,024 exercises `64/d` well inside the cap;
/// **65 is the off-by-one boundary** — the smallest degree at which top-64
/// selection actually truncates, and therefore the case an implementation with
/// `>` instead of `>=` gets wrong. A corpus omitting 65 lets that bug pass.
///
/// The degree is the group's **page** degree, not its root count, because that
/// is what the cap is taken over: `hub_weight(d)` reads `d` = pages the group
/// touches, and `groups_touching` ranks and cuts pages. Setting only the root
/// count leaves every group's page degree drawn from [`FANOUT_SPAN`], whose
/// maximum is 15 — so `hub_weight` returns exactly `1.0` for all
/// [`M6_GROUP_COUNT`] groups, the `64/d` branch never executes, and
/// `R-HUB-TOPK` / `R-HUB-WEIGHT` / `R-HUB-MAXPAIRS` go untested by the very
/// benchmark that names them. A hub therefore carries its degree in both
/// places, and its page edges are emitted **distinct** (see
/// [`write_corpus_stream`]): 65 probes the boundary only if it lands on
/// exactly 65 pages, and sampling 65 of 5,000 with replacement lands on about
/// 64.6.
pub const M6_HUB_DEGREES: [u64; 3] = [5_000, 1_024, 65];
/// `R-CORPUS-FANOUT` — mean pages per group, over the non-hub groups. The three
/// hubs are specified by `R-CORPUS-HUB` instead and are excluded from the
/// balanced draw.
pub const M6_PAGES_PER_GROUP_MEAN: u64 = 8;
/// `R-CORPUS-AGE` — root age is uniform over `0..=720` days.
pub const M6_ROOT_AGE_MAX_DAYS: u64 = 720;
/// `R-CORPUS-GENFRAC`
pub const M6_GENERATED_ROOT_PERCENT: u64 = 15;
/// `R-CORPUS-RETFRAC`
pub const M6_RETRACTED_EDGE_PERCENT: u64 = 5;

pub const M6_CORPUS_ENCODING: &str = "wenlan-m6-bench-corpus-v1";
pub const M6_MANIFEST_ENCODING: &str = "wenlan-m6-bench-manifest-v1";

/// Fanout is drawn uniformly over `1..=15`, whose mean is exactly
/// [`M6_PAGES_PER_GROUP_MEAN`].
const FANOUT_SPAN: u64 = 2 * M6_PAGES_PER_GROUP_MEAN - 1;

/// Fixed-point fractional bits for [`fixed_tenth_root`].
const Q: u32 = 20;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct M6CorpusSummary {
    pub encoding: &'static str,
    pub seed: u64,
    pub memory_count: u64,
    pub page_count: u64,
    pub group_count: u64,
    pub total_root_edges: u64,
    pub total_retracted_edges: u64,
    pub total_group_page_edges: u64,
    pub generated_group_count: u64,
    pub sha256: String,
}

/// One independence group's frozen shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Group {
    degree: u64,
    fanout: u64,
    root_age_days: u64,
    generated: bool,
    retracted: u64,
}

/// Integer 10th root in Q20 fixed point: returns `r` with `r ≈ k^(1/10) * 2^Q`.
///
/// Binary search rather than a float root, for the cross-platform reason in the
/// module docs. `k^0.1 <= 5000^0.1 < 2.35`, so the search window is tight and
/// every intermediate stays far inside `u128`.
fn fixed_tenth_root(k: u64) -> u64 {
    debug_assert!(k >= 1);
    let one = 1u128 << Q;
    let target = (k as u128) << Q;

    let mut low = one;
    let mut high = 3u128 << Q;
    while low < high {
        // Round the midpoint up, so the `low = mid` branch always advances.
        let mid = (low + high).div_ceil(2);
        // mid^10, rescaled to Q after each multiply so nothing grows.
        let mut acc = mid;
        for _ in 0..9 {
            acc = (acc * mid) >> Q;
        }
        if acc <= target {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low as u64
}

/// Cumulative integer weights for a Zipf(11/10) truncated at
/// [`M6_ZIPF_TRUNCATION`]. `w_k ∝ 1 / k^1.1`, computed as
/// `2^60 / (k * k^0.1)` entirely in integers.
fn zipf_cumulative_weights() -> Vec<u128> {
    debug_assert_eq!(M6_ZIPF_EXPONENT_NUMERATOR, 11);
    debug_assert_eq!(M6_ZIPF_EXPONENT_DENOMINATOR, 10);

    let mut cumulative = Vec::with_capacity(M6_ZIPF_TRUNCATION as usize);
    let mut running = 0u128;
    for k in 1..=M6_ZIPF_TRUNCATION {
        let root = fixed_tenth_root(k) as u128;
        // k^1.1 in Q20 == k * k^0.1(Q20). Weight is its reciprocal, scaled.
        let weight = (1u128 << 60) / ((k as u128) * root);
        running += weight;
        cumulative.push(running);
    }
    cumulative
}

/// Inverse-transform sample from a cumulative weight table.
fn sample_cumulative(cumulative: &[u128], rng: &mut SplitMix64) -> usize {
    let total = *cumulative.last().expect("non-empty cumulative table");
    let target = (rng.next_u64() as u128) % total;
    cumulative.partition_point(|entry| *entry <= target)
}

/// Memories per space, from the percentage split. Exact: the split sums to 100
/// and [`M6_MEMORY_COUNT`] is a multiple of 100.
fn space_memory_counts() -> [u64; 8] {
    let mut counts = [0u64; 8];
    for (slot, percent) in counts.iter_mut().zip(M6_SPACE_SPLIT_PERCENT) {
        *slot = M6_MEMORY_COUNT * percent / 100;
    }
    counts
}

/// The frozen group table. RNG consumption order is part of encoding v1.
fn build_groups(rng: &mut SplitMix64) -> Vec<Group> {
    let cumulative = zipf_cumulative_weights();
    let group_count = M6_GROUP_COUNT as usize;

    // 1. Degrees: the three hubs are exact, the remainder is Zipf.
    let mut degrees = Vec::with_capacity(group_count);
    degrees.extend_from_slice(&M6_HUB_DEGREES);
    for _ in M6_HUB_DEGREES.len()..group_count {
        degrees.push(sample_cumulative(&cumulative, rng) as u64 + 1);
    }

    // 2. Fanout: a *balanced* draw, not an independent one. Each value in
    //    `1..=FANOUT_SPAN` appears exactly `group_count / FANOUT_SPAN` times,
    //    then the assignment is shuffled so it does not correlate with group
    //    index. Sum is therefore exactly `group_count * mean`.
    //
    //    An independent uniform draw has mean 8 only in expectation: the
    //    realized total came out 95,464 rather than 96,000, which would make
    //    R-CORPUS-FANOUT checkable only to within sampling error. Exact by
    //    construction matches how R-CORPUS-RETFRAC is handled below.
    //
    //    The draw covers the non-hub groups only. A hub's page degree is its
    //    hub degree (R-CORPUS-HUB), which is the whole point of a hub — see
    //    `M6_HUB_DEGREES` — so it must not also take a value from this pool.
    //    `group_count - hubs` does not divide `FANOUT_SPAN`, so the remainder
    //    is carried across the low values, exactly as R-CORPUS-RETFRAC carries
    //    below; the total stays exact rather than approximate.
    let drawn = group_count - M6_HUB_DEGREES.len();
    let per_value = drawn / FANOUT_SPAN as usize;
    let carry = drawn - per_value * FANOUT_SPAN as usize;
    let mut fanouts: Vec<u64> = (1..=FANOUT_SPAN)
        .flat_map(|value| {
            let extra = usize::from((value as usize) <= carry);
            std::iter::repeat_n(value, per_value + extra)
        })
        .collect();
    debug_assert_eq!(fanouts.len(), drawn);
    for index in (1..fanouts.len()).rev() {
        let swap = (rng.next_u64() % (index as u64 + 1)) as usize;
        fanouts.swap(index, swap);
    }
    // The hubs take the front slots, matching how `degrees` is laid out.
    let mut fanouts_with_hubs = M6_HUB_DEGREES.to_vec();
    fanouts_with_hubs.extend(fanouts);
    let fanouts = fanouts_with_hubs;

    // 3. Root age, uniform over 0..=M6_ROOT_AGE_MAX_DAYS.
    let ages: Vec<u64> = (0..group_count)
        .map(|_| rng.next_u64() % (M6_ROOT_AGE_MAX_DAYS + 1))
        .collect();

    // 4. Generated roots: exactly 15% of groups, chosen by a Fisher-Yates
    //    prefix so the flag does not correlate with group index (which would
    //    otherwise make all three hubs generated).
    let generated_target = (M6_GROUP_COUNT * M6_GENERATED_ROOT_PERCENT / 100) as usize;
    let mut order: Vec<u32> = (0..group_count as u32).collect();
    for index in (1..group_count).rev() {
        let swap = (rng.next_u64() % (index as u64 + 1)) as usize;
        order.swap(index, swap);
    }
    let mut generated = vec![false; group_count];
    for picked in &order[..generated_target] {
        generated[*picked as usize] = true;
    }

    // 5. Retracted edges: exactly 5% of all root edges globally. The carry
    //    redistributes each group's flooring remainder forward, so the total is
    //    `floor(total_edges * 5 / 100)` rather than a sum of independent floors
    //    (which would land well under 5%, since most Zipf degrees are tiny).
    let mut carry = 0u64;
    let mut groups = Vec::with_capacity(group_count);
    for index in 0..group_count {
        carry += degrees[index] * M6_RETRACTED_EDGE_PERCENT;
        let retracted = carry / 100;
        carry %= 100;
        groups.push(Group {
            degree: degrees[index],
            fanout: fanouts[index],
            root_age_days: ages[index],
            generated: generated[index],
            retracted,
        });
    }
    groups
}

/// Write the frozen corpus into `writer` in encoding v1.
pub fn write_corpus_stream<W: Write>(mut writer: W) -> Result<()> {
    let mut rng = SplitMix64::new(M6_BENCH_SEED);
    let groups = build_groups(&mut rng);
    let space_counts = space_memory_counts();

    writeln!(writer, "{M6_CORPUS_ENCODING}")?;
    writeln!(
        writer,
        "S\t{}\t{}\t{}\t{}\t{}",
        M6_BENCH_SEED,
        M6_MEMORY_COUNT,
        M6_PAGE_COUNT,
        M6_SPACE_SPLIT_PERCENT.len(),
        M6_GROUP_COUNT
    )?;

    for (space, count) in space_counts.iter().enumerate() {
        writeln!(writer, "C\t{space}\t{count}")?;
    }

    for (index, group) in groups.iter().enumerate() {
        writeln!(
            writer,
            "G\t{}\t{}\t{}\t{}\t{}\t{}",
            index,
            group.degree,
            group.fanout,
            group.root_age_days,
            u8::from(group.generated),
            group.retracted
        )?;
    }

    // Pages carry a space drawn from the same split as memories.
    let space_cumulative: Vec<u128> = M6_SPACE_SPLIT_PERCENT
        .iter()
        .scan(0u128, |running, percent| {
            *running += *percent as u128;
            Some(*running)
        })
        .collect();
    let mut page_spaces = Vec::with_capacity(M6_PAGE_COUNT as usize);
    for page in 0..M6_PAGE_COUNT {
        let space = sample_cumulative(&space_cumulative, &mut rng);
        page_spaces.push(space);
        writeln!(writer, "P\t{page}\t{space}")?;
    }

    // Group -> page edges. A hub's edges are distinct pages; an ordinary
    // group's are drawn with replacement.
    //
    // The distinction is not cosmetic. `support` collapses duplicates with
    // `GROUP BY gid, page_id`, so the cap is taken over *distinct* pages, and
    // drawing 65 of 5,000 with replacement lands on about 64.6 of them — which
    // makes the 65-hub a coin flip about whether it sits above or below the
    // top-64 cut rather than the exact boundary probe R-CORPUS-HUB specifies.
    for (index, group) in groups.iter().enumerate() {
        if index < M6_HUB_DEGREES.len() {
            let take = group.fanout.min(M6_PAGE_COUNT) as usize;
            let mut deck: Vec<u64> = (0..M6_PAGE_COUNT).collect();
            for slot in 0..take {
                let swap = slot + (rng.next_u64() % (deck.len() - slot) as u64) as usize;
                deck.swap(slot, swap);
                let page = deck[slot];
                writeln!(writer, "E\t{index}\t{page}")?;
            }
        } else {
            for _ in 0..group.fanout {
                let page = rng.next_u64() % M6_PAGE_COUNT;
                writeln!(writer, "E\t{index}\t{page}")?;
            }
        }
    }

    // Memories: space from the split, group proportional to degree.
    let degree_cumulative: Vec<u128> = groups
        .iter()
        .scan(0u128, |running, group| {
            *running += group.degree as u128;
            Some(*running)
        })
        .collect();
    for memory in 0..M6_MEMORY_COUNT {
        let space = sample_cumulative(&space_cumulative, &mut rng);
        let group = sample_cumulative(&degree_cumulative, &mut rng);
        writeln!(writer, "M\t{memory}\t{space}\t{group}")?;
    }

    Ok(())
}

/// Generate the corpus and summarize it without retaining it.
pub fn summarize_corpus() -> Result<M6CorpusSummary> {
    let mut rng = SplitMix64::new(M6_BENCH_SEED);
    let groups = build_groups(&mut rng);

    let mut writer = DigestWriter(Sha256::default());
    write_corpus_stream(&mut writer)?;

    Ok(M6CorpusSummary {
        encoding: M6_CORPUS_ENCODING,
        seed: M6_BENCH_SEED,
        memory_count: M6_MEMORY_COUNT,
        page_count: M6_PAGE_COUNT,
        group_count: M6_GROUP_COUNT,
        total_root_edges: groups.iter().map(|group| group.degree).sum(),
        total_retracted_edges: groups.iter().map(|group| group.retracted).sum(),
        total_group_page_edges: groups.iter().map(|group| group.fanout).sum(),
        generated_group_count: groups.iter().filter(|group| group.generated).count() as u64,
        sha256: hex::encode(sha2::Digest::finalize(writer.0)),
    })
}

/// The composition line the manifest hashes alongside the corpus digest.
///
/// Every R-CORPUS-* constant appears, so amending one moves the manifest even
/// if the corpus stream did not change.
fn composition_line() -> String {
    format!(
        "seed={M6_BENCH_SEED}\nmemories={M6_MEMORY_COUNT}\npages={M6_PAGE_COUNT}\nspaces={:?}\ngroups={M6_GROUP_COUNT}\nzipf={M6_ZIPF_EXPONENT_NUMERATOR}/{M6_ZIPF_EXPONENT_DENOMINATOR}@{M6_ZIPF_TRUNCATION}\nhubs={:?}\nfanout={M6_PAGES_PER_GROUP_MEAN}\nage={M6_ROOT_AGE_MAX_DAYS}\ngenfrac={M6_GENERATED_ROOT_PERCENT}\nretfrac={M6_RETRACTED_EDGE_PERCENT}\n",
        M6_SPACE_SPLIT_PERCENT, M6_HUB_DEGREES
    )
}

pub fn canonical_manifest_digest(corpus_sha256: &str) -> Result<String> {
    validate_sha256_hex(corpus_sha256)?;
    let manifest = format!(
        "{M6_MANIFEST_ENCODING}\ncorpus-sha256={corpus_sha256}\n{}",
        composition_line()
    );
    Ok(sha256_hex(manifest.as_bytes()))
}

/// Refuse-on-mismatch, matching M5's shape: the bench declines to run rather
/// than reporting a number nobody can compare against.
pub fn verify_manifest_digest(expected_digest_file: &[u8], corpus_sha256: &str) -> Result<()> {
    if expected_digest_file.len() != 65 || expected_digest_file[64] != b'\n' {
        bail!("manifest digest file must be 64 lowercase hex bytes followed by LF");
    }
    let expected = std::str::from_utf8(&expected_digest_file[..64])?;
    validate_sha256_hex(expected)?;
    let actual = canonical_manifest_digest(corpus_sha256)?;
    if actual != expected {
        bail!("M6 benchmark manifest digest mismatch: expected {expected}, actual {actual}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenth_root_is_exact_at_perfect_powers() {
        // 2^10 == 1024, so the 10th root of 1024 is exactly 2.
        assert_eq!(fixed_tenth_root(1024), 2 << Q);
        assert_eq!(fixed_tenth_root(1), 1 << Q);
    }

    #[test]
    fn tenth_root_is_monotone_over_the_truncation_range() {
        let mut previous = 0;
        for k in 1..=M6_ZIPF_TRUNCATION {
            let root = fixed_tenth_root(k);
            assert!(root >= previous, "10th root went backwards at k={k}");
            previous = root;
        }
        // 5000^0.1 ~ 2.3446
        let top = fixed_tenth_root(M6_ZIPF_TRUNCATION) as f64 / (1u64 << Q) as f64;
        assert!((top - 2.3446).abs() < 0.001, "unexpected top root {top}");
    }

    #[test]
    fn zipf_weights_are_strictly_decreasing() {
        let cumulative = zipf_cumulative_weights();
        assert_eq!(cumulative.len(), M6_ZIPF_TRUNCATION as usize);
        let mut previous = u128::MAX;
        for index in 0..cumulative.len() {
            let weight = if index == 0 {
                cumulative[0]
            } else {
                cumulative[index] - cumulative[index - 1]
            };
            assert!(weight > 0, "zero weight at rank {}", index + 1);
            assert!(
                weight <= previous,
                "weight rose at rank {} ({weight} > {previous})",
                index + 1
            );
            previous = weight;
        }
    }

    #[test]
    fn the_space_split_covers_every_memory() {
        let counts = space_memory_counts();
        assert_eq!(counts.iter().sum::<u64>(), M6_MEMORY_COUNT);
        assert_eq!(M6_SPACE_SPLIT_PERCENT.iter().sum::<u64>(), 100);
    }

    #[test]
    fn the_three_hub_degrees_are_generated_exactly() {
        let mut rng = SplitMix64::new(M6_BENCH_SEED);
        let groups = build_groups(&mut rng);
        for (index, expected) in M6_HUB_DEGREES.iter().enumerate() {
            assert_eq!(
                groups[index].degree, *expected,
                "hub {index} is not at its frozen degree"
            );
        }
        // 65 is the off-by-one boundary and must be present verbatim.
        assert!(groups.iter().any(|group| group.degree == 65));
    }

    #[test]
    fn generated_roots_are_exactly_fifteen_percent_and_not_index_correlated() {
        let mut rng = SplitMix64::new(M6_BENCH_SEED);
        let groups = build_groups(&mut rng);
        let generated = groups.iter().filter(|group| group.generated).count() as u64;
        assert_eq!(generated, M6_GROUP_COUNT * M6_GENERATED_ROOT_PERCENT / 100);
        // A prefix-flagging bug would make every hub generated.
        assert!(
            !groups[..M6_HUB_DEGREES.len()]
                .iter()
                .all(|group| group.generated),
            "generated flag correlates with group index"
        );
    }

    #[test]
    fn fanout_totals_exactly_the_frozen_mean_and_is_not_index_correlated() {
        let mut rng = SplitMix64::new(M6_BENCH_SEED);
        let groups = build_groups(&mut rng);
        let drawn = M6_GROUP_COUNT as usize - M6_HUB_DEGREES.len();
        let per_value = drawn / FANOUT_SPAN as usize;
        let carry = drawn - per_value * FANOUT_SPAN as usize;

        // The balanced pool covers the non-hub groups; the remainder is
        // carried across the low values, so this is exact, not approximate.
        let non_hub = &groups[M6_HUB_DEGREES.len()..];
        let total: u64 = non_hub.iter().map(|group| group.fanout).sum();
        let expected: u64 = (1..=FANOUT_SPAN)
            .map(|value| value * (per_value as u64 + u64::from((value as usize) <= carry)))
            .sum();
        // 95,958 over 11,997 groups — a realized mean of 7.9985, which rounds
        // to R-CORPUS-FANOUT's 8. Exact by construction, so this is the whole
        // claim; a tolerance band next to it would only be a second, looser
        // statement of the same thing.
        assert_eq!(total, expected);
        for value in 1..=FANOUT_SPAN {
            let count = non_hub.iter().filter(|group| group.fanout == value).count();
            let extra = usize::from((value as usize) <= carry);
            assert_eq!(count, per_value + extra);
        }
        // Unshuffled, the first non-hub groups would all carry fanout 1.
        assert!(
            non_hub[..64].iter().any(|group| group.fanout != 1),
            "fanout assignment correlates with group index"
        );
    }

    /// The benchmark can only exercise the hub cap if some group's **page**
    /// degree crosses it. Wiring `M6_HUB_DEGREES` to the root count alone left
    /// every group's page degree inside `FANOUT_SPAN` (max 15), so
    /// `hub_weight` returned exactly `1.0` for all 12,000 groups and the
    /// `64/d` branch never ran — `R-HUB-TOPK` / `R-HUB-WEIGHT` /
    /// `R-HUB-MAXPAIRS` were untested by the corpus that names them.
    #[test]
    fn the_hubs_carry_their_degree_as_pages_touched_and_cross_the_top_64_cut() {
        let mut rng = SplitMix64::new(M6_BENCH_SEED);
        let groups = build_groups(&mut rng);
        for (index, expected) in M6_HUB_DEGREES.iter().enumerate() {
            assert_eq!(
                groups[index].fanout, *expected,
                "hub {index} does not touch as many pages as its degree"
            );
        }
        assert!(
            M6_HUB_DEGREES.iter().all(|degree| *degree > 64),
            "every hub must sit strictly above the top-64 cut"
        );
        assert!(
            groups[M6_HUB_DEGREES.len()..]
                .iter()
                .all(|group| group.fanout <= FANOUT_SPAN),
            "a non-hub group must not be silently promoted into the cap"
        );
    }

    /// The 65-hub is the `>` vs `>=` probe, so it has to land on exactly 65
    /// distinct pages. Drawn with replacement it lands on about 64.6 — right
    /// where the boundary it exists to test is, which would make the probe
    /// decide the question by luck of the seed.
    #[test]
    fn hub_page_edges_are_distinct_so_the_boundary_hub_lands_on_exactly_65() {
        let mut corpus = Vec::new();
        write_corpus_stream(&mut corpus).expect("corpus");
        let text = String::from_utf8(corpus).expect("utf8");

        for (index, degree) in M6_HUB_DEGREES.iter().enumerate() {
            let prefix = format!("E\t{index}\t");
            let pages: Vec<&str> = text
                .lines()
                .filter_map(|line| line.strip_prefix(prefix.as_str()))
                .collect();
            let distinct: std::collections::BTreeSet<&str> = pages.iter().copied().collect();
            assert_eq!(
                pages.len() as u64,
                *degree,
                "hub {index} emitted the wrong edge count"
            );
            assert_eq!(
                distinct.len() as u64,
                *degree,
                "hub {index} repeated a page, so its page degree is below its hub degree"
            );
        }
    }

    #[test]
    fn retracted_edges_are_five_percent_of_all_root_edges() {
        let mut rng = SplitMix64::new(M6_BENCH_SEED);
        let groups = build_groups(&mut rng);
        let total: u64 = groups.iter().map(|group| group.degree).sum();
        let retracted: u64 = groups.iter().map(|group| group.retracted).sum();
        // The carry makes this exact, not approximate.
        assert_eq!(retracted, total * M6_RETRACTED_EDGE_PERCENT / 100);
    }

    #[test]
    fn the_corpus_digest_is_stable_across_regeneration() {
        let first = summarize_corpus().expect("first corpus generation");
        let second = summarize_corpus().expect("second corpus generation");
        assert_eq!(first, second);
        assert_eq!(first.sha256.len(), 64);
    }

    #[test]
    fn the_manifest_refuses_a_malformed_fixture() {
        let summary = summarize_corpus().expect("corpus");
        let digest = canonical_manifest_digest(&summary.sha256).expect("manifest");

        let mut good = digest.clone().into_bytes();
        good.push(b'\n');
        assert!(verify_manifest_digest(&good, &summary.sha256).is_ok());

        // No trailing LF.
        assert!(verify_manifest_digest(digest.as_bytes(), &summary.sha256).is_err());
        // Wrong digest.
        let mut wrong = vec![b'0'; 64];
        wrong.push(b'\n');
        assert!(verify_manifest_digest(&wrong, &summary.sha256).is_err());
    }

    #[test]
    fn amending_a_composition_constant_moves_the_manifest() {
        // The manifest hashes the composition, so this line is what makes an
        // S0-99 constant-tune visible rather than silent.
        let summary = summarize_corpus().expect("corpus");
        let real = canonical_manifest_digest(&summary.sha256).expect("manifest");
        let tuned = sha256_hex(
            format!(
                "{M6_MANIFEST_ENCODING}\ncorpus-sha256={}\n{}",
                summary.sha256,
                composition_line().replace("genfrac=15", "genfrac=16")
            )
            .as_bytes(),
        );
        assert_ne!(real, tuned);
    }
}
