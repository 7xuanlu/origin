// SPDX-License-Identifier: Apache-2.0
//! Frozen, allocation-bounded data plane for the M5 entailment benchmark.
//!
//! Corpus encoding `wenlan-m5-bench-corpus-v1` is ASCII with LF line endings:
//! one version header, one seed/cardinality header, then `P` page records and
//! `M` memory records in ascending numeric order. Integer fields are base-10,
//! separated by one tab, with no padding. The digest is SHA-256 over those
//! exact bytes. The stream is written directly into the digest, so the 100k
//! records are never retained as a corpus-sized allocation.
//!
//! Manifest encoding `wenlan-m5-bench-manifest-v1` is also ASCII/LF. It hashes
//! the generated corpus digest plus SHA-256 of the exact checked-out bytes of
//! the distribution JSON and accuracy JSONL fixtures. `.gitattributes` pins
//! those bytes to LF on every platform.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

pub const M5_BENCH_SEED: u64 = 0x4d35_0001;
pub const M5_BENCH_MEMORY_COUNT: u64 = 100_000;
pub const M5_BENCH_PAGE_COUNT: u64 = 5_000;
pub const M5_CORPUS_ENCODING: &str = "wenlan-m5-bench-corpus-v1";
pub const M5_MANIFEST_ENCODING: &str = "wenlan-m5-bench-manifest-v1";
pub const PAGE_SIZE_SCHEMA_VERSION: u32 = 1;
pub const PAGE_SIZE_K_MIN: u64 = 5;

/// Half-open UTF-8 byte ranges used by the privacy-safe SQL exporter.
pub const FIXED_BUCKET_BOUNDS: [u64; 8] = [0, 256, 512, 1024, 2048, 4096, 8192, 16384];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PageSizeDistribution {
    pub schema_version: u32,
    pub unit: String,
    pub k_min: u64,
    pub sample_size: u64,
    pub buckets: Vec<PageSizeBucket>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PageSizeBucket {
    pub min_inclusive: u64,
    pub max_exclusive: Option<u64>,
    pub count: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AccuracyCategory {
    PositiveExact,
    PositiveReflow,
    NegationN1,
    QuantifierN8,
    TerminatorN11,
    AcronymCaseN12,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeAccuracyCase {
    pub case_id: String,
    pub category: AccuracyCategory,
    pub claim: String,
    pub span: String,
    pub entails: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorpusSummary {
    pub encoding: &'static str,
    pub seed: u64,
    pub memory_count: u64,
    pub page_count: u64,
    pub sha256: String,
}

impl PageSizeDistribution {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let distribution: Self =
            serde_json::from_slice(bytes).context("parse strict M5 page-size distribution")?;
        distribution.validate()?;
        Ok(distribution)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != PAGE_SIZE_SCHEMA_VERSION {
            bail!(
                "unsupported page-size schema version {}",
                self.schema_version
            );
        }
        if self.unit != "utf8_bytes" {
            bail!("page-size unit must be utf8_bytes");
        }
        if self.k_min != PAGE_SIZE_K_MIN {
            bail!("page-size k_min must be {PAGE_SIZE_K_MIN}");
        }
        if self.buckets.is_empty() {
            bail!("page-size distribution has no buckets");
        }

        let mut total = 0_u64;
        let mut prior_max = None;
        for bucket in &self.buckets {
            if bucket.count < self.k_min {
                bail!("page-size bucket count is below k_min");
            }
            if let Some(max) = bucket.max_exclusive {
                if max <= bucket.min_inclusive {
                    bail!("page-size bucket is empty or reversed");
                }
            }
            if let Some(max) = prior_max {
                if bucket.min_inclusive < max {
                    bail!("page-size buckets overlap or are unsorted");
                }
            }
            if prior_max.is_none() && total != 0 {
                bail!("open-ended page-size bucket must be last");
            }
            total = total
                .checked_add(bucket.count)
                .context("page-size sample count overflow")?;
            prior_max = bucket.max_exclusive;
        }
        if total != self.sample_size {
            bail!(
                "page-size sample_size {} does not match bucket total {total}",
                self.sample_size
            );
        }
        Ok(())
    }

    pub fn to_canonical_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut bytes = serde_json::to_vec_pretty(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

/// Merge non-empty adjacent published buckets until every emitted count is at
/// least `k_min`. A sparse bucket prefers its right neighbour; a sparse tail
/// merges left. An aggregate smaller than k is refused instead of published.
pub fn merge_sparse_buckets(
    buckets: Vec<PageSizeBucket>,
    k_min: u64,
) -> Result<Vec<PageSizeBucket>> {
    if k_min == 0 {
        bail!("k_min must be positive");
    }
    let total = buckets.iter().try_fold(0_u64, |sum, bucket| {
        sum.checked_add(bucket.count)
            .context("bucket count overflow")
    })?;
    if total < k_min {
        bail!("aggregate sample size {total} is below k_min {k_min}");
    }

    let mut merged: Vec<_> = buckets
        .into_iter()
        .filter(|bucket| bucket.count > 0)
        .collect();
    if merged.is_empty() {
        bail!("aggregate has no non-empty buckets");
    }
    for pair in merged.windows(2) {
        let left_max = pair[0]
            .max_exclusive
            .context("open-ended bucket must be last")?;
        if pair[1].min_inclusive < left_max {
            bail!("buckets overlap or are unsorted");
        }
    }

    loop {
        let Some(index) = merged.iter().position(|bucket| bucket.count < k_min) else {
            return Ok(merged);
        };
        if merged.len() == 1 {
            bail!("aggregate cannot satisfy k_min");
        }
        if index + 1 < merged.len() {
            let right = merged.remove(index + 1);
            merged[index].max_exclusive = right.max_exclusive;
            merged[index].count = merged[index]
                .count
                .checked_add(right.count)
                .context("bucket count overflow")?;
        } else {
            let tail = merged.remove(index);
            let left = merged.last_mut().expect("sparse tail has a left neighbour");
            left.max_exclusive = tail.max_exclusive;
            left.count = left
                .count
                .checked_add(tail.count)
                .context("bucket count overflow")?;
        }
    }
}

pub fn distribution_from_fixed_counts(counts: [u64; 8]) -> Result<PageSizeDistribution> {
    let mut raw = Vec::with_capacity(counts.len());
    for (index, count) in counts.into_iter().enumerate() {
        let min_inclusive = if index < FIXED_BUCKET_BOUNDS.len() {
            FIXED_BUCKET_BOUNDS[index]
        } else {
            *FIXED_BUCKET_BOUNDS
                .last()
                .expect("fixed bounds are non-empty")
        };
        let max_exclusive = FIXED_BUCKET_BOUNDS.get(index + 1).copied();
        raw.push(PageSizeBucket {
            min_inclusive,
            max_exclusive,
            count,
        });
    }
    let sample_size = raw.iter().map(|bucket| bucket.count).sum();
    let distribution = PageSizeDistribution {
        schema_version: PAGE_SIZE_SCHEMA_VERSION,
        unit: "utf8_bytes".to_string(),
        k_min: PAGE_SIZE_K_MIN,
        sample_size,
        buckets: merge_sparse_buckets(raw, PAGE_SIZE_K_MIN)?,
    };
    distribution.validate()?;
    Ok(distribution)
}

pub fn parse_accuracy_jsonl(bytes: &[u8]) -> Result<Vec<JudgeAccuracyCase>> {
    let text = std::str::from_utf8(bytes).context("accuracy fixture is not UTF-8")?;
    if !text.ends_with('\n') {
        bail!("accuracy fixture must end with LF");
    }
    let mut cases = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() {
            bail!("accuracy fixture contains a blank line at {}", index + 1);
        }
        let case = serde_json::from_str(line)
            .with_context(|| format!("parse strict accuracy case at line {}", index + 1))?;
        cases.push(case);
    }
    if cases.is_empty() {
        bail!("accuracy fixture contains no cases");
    }
    Ok(cases)
}

pub fn corpus_summary(distribution: &PageSizeDistribution) -> Result<CorpusSummary> {
    distribution.validate()?;
    let mut writer = DigestWriter(Sha256::new());
    write_corpus_stream(distribution, &mut writer)?;
    Ok(CorpusSummary {
        encoding: M5_CORPUS_ENCODING,
        seed: M5_BENCH_SEED,
        memory_count: M5_BENCH_MEMORY_COUNT,
        page_count: M5_BENCH_PAGE_COUNT,
        sha256: hex::encode(writer.0.finalize()),
    })
}

pub fn write_corpus_stream<W: Write>(
    distribution: &PageSizeDistribution,
    mut writer: W,
) -> Result<()> {
    distribution.validate()?;
    let mut rng = SplitMix64::new(M5_BENCH_SEED);
    writeln!(writer, "{M5_CORPUS_ENCODING}")?;
    writeln!(
        writer,
        "S\t{}\t{}\t{}",
        M5_BENCH_SEED, M5_BENCH_MEMORY_COUNT, M5_BENCH_PAGE_COUNT
    )?;

    let memories_per_page = M5_BENCH_MEMORY_COUNT / M5_BENCH_PAGE_COUNT;
    for page_index in 0..M5_BENCH_PAGE_COUNT {
        let page_bytes = draw_page_size(distribution, &mut rng)?;
        let first_memory = page_index * memories_per_page;
        writeln!(
            writer,
            "P\t{page_index}\t{page_bytes}\t{first_memory}\t{memories_per_page}"
        )?;
        for offset in 0..memories_per_page {
            let memory_index = first_memory + offset;
            let payload_bytes = 32 + rng.next_u64() % 481;
            let token = rng.next_u64();
            writeln!(
                writer,
                "M\t{memory_index}\t{page_index}\t{payload_bytes}\t{token}"
            )?;
        }
    }
    Ok(())
}

pub fn canonical_manifest_digest(
    corpus_sha256: &str,
    distribution_bytes: &[u8],
    accuracy_bytes: &[u8],
) -> Result<String> {
    validate_sha256_hex(corpus_sha256)?;
    let distribution_sha256 = sha256_hex(distribution_bytes);
    let accuracy_sha256 = sha256_hex(accuracy_bytes);
    let manifest = format!(
        "{M5_MANIFEST_ENCODING}\ncorpus-sha256={corpus_sha256}\npage-size-dist-sha256={distribution_sha256}\njudge-accuracy-sha256={accuracy_sha256}\n"
    );
    Ok(sha256_hex(manifest.as_bytes()))
}

pub fn verify_manifest_digest(
    expected_digest_file: &[u8],
    corpus_sha256: &str,
    distribution_bytes: &[u8],
    accuracy_bytes: &[u8],
) -> Result<()> {
    if expected_digest_file.len() != 65 || expected_digest_file[64] != b'\n' {
        bail!("manifest digest file must be 64 lowercase hex bytes followed by LF");
    }
    let expected = std::str::from_utf8(&expected_digest_file[..64])?;
    validate_sha256_hex(expected)?;
    let actual = canonical_manifest_digest(corpus_sha256, distribution_bytes, accuracy_bytes)?;
    if actual != expected {
        bail!("M5 benchmark manifest digest mismatch: expected {expected}, actual {actual}");
    }
    Ok(())
}

fn draw_page_size(distribution: &PageSizeDistribution, rng: &mut SplitMix64) -> Result<u64> {
    let target = rng.next_u64() % distribution.sample_size;
    let mut cumulative = 0_u64;
    for bucket in &distribution.buckets {
        cumulative += bucket.count;
        if target < cumulative {
            let (lower, upper) = match bucket.max_exclusive {
                Some(max) => (bucket.min_inclusive, max),
                None => {
                    // Corpus encoding v1 gives an open tail a finite,
                    // deterministic synthetic draw range wholly inside it:
                    // [max(min, 1), 2 * max(min, 1)). Checked doubling refuses
                    // an unrepresentable tail instead of wrapping.
                    let lower = bucket.min_inclusive.max(1);
                    let upper = lower
                        .checked_mul(2)
                        .context("open page-size bucket synthetic range overflow")?;
                    (lower, upper)
                }
            };
            return Ok(lower + rng.next_u64() % (upper - lower));
        }
    }
    bail!("page-size distribution selection fell outside cumulative count")
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(crate) fn validate_sha256_hex(value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("SHA-256 must be exactly 64 lowercase hex characters");
    }
    Ok(())
}

/// Small fixed PRNG whose algorithm is part of corpus encoding v1.
///
/// `pub(crate)` so M6's sibling corpus reuses the freeze machinery rather than
/// re-deriving it; the algorithm is frozen for both encodings.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

pub(crate) struct DigestWriter(pub(crate) Sha256);

impl io::Write for DigestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Digest::update(&mut self.0, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
