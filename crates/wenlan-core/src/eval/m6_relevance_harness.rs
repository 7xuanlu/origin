// SPDX-License-Identifier: Apache-2.0
//! Materializes the frozen S0-97 corpus into a database and drives the C1
//! relevance sweep over it, so §10's two count limits are measured against the
//! corpus they are specified against rather than against a unit fixture.
//!
//! **Why this lives in `eval/` and not in the bench target.** `m6::relevance`
//! and `m6::relevance_sweep` are `pub(crate)`; an integration test under
//! `tests/` cannot reach them. Widening the whole module to `pub` to let one
//! bench in would ship speculative public surface for a default-OFF feature.
//! One public entry point here is the smaller seam.
//!
//! **What is loaded and what is not.** The sweep reads `edges`,
//! `provenance_roots` and `pages`; it never reads `memories`, so the corpus's
//! 100,000 `M` records are counted and discarded rather than inserted. That
//! does not weaken the corpus's identity — identity is the stream digest,
//! asserted separately by `the_manifest_matches_the_checked_in_fixture`, and
//! the stream this harness parses is the same stream that digest is taken
//! over.
//!
//! The M6 tables come from the shipped DDL (`ensure_relevance_tables`,
//! `ensure_remaining_substrate`, `COMMUNITY_SUBSTRATE_DDL`), for the reason
//! `genesis_test_support` gives: a hand-copied `m6_pair_stats` would let the
//! gate pass against a substrate production never runs. The primary-evidence
//! tables are fixture shapes carrying only the columns the M6 readers touch,
//! matching that same fixture's decision.

use crate::db::{MemoryDB, GENESIS_SUBSTRATE_DDL};
use crate::m6::relevance::ensure_relevance_tables;
use crate::m6::relevance_sweep::{run_relevance_sweep, RelevanceTurn};
use crate::m6::remaining_substrate::ensure_remaining_substrate;
use anyhow::{Context, Result};

use super::m6_bench_corpus::{write_corpus_stream, M6_ROOT_AGE_MAX_DAYS};

/// One sweep turn's S0-95 instrument readings, plus its wall clock.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepTurnReceipt {
    pub kind: &'static str,
    pub endpoints_considered: usize,
    pub queries: u32,
    pub rows: u32,
    /// Wall clock for the turn, transaction included.
    ///
    /// **Not `R-BENCH-MAX`.** That limit times the *route evaluation* — S0-96's
    /// queries (a)-(d) over `m6_adjacency` and `m6_pair_stats` — and C1 builds
    /// the writers of those two tables, not the reader. Timing the writer and
    /// reporting it under the reader's name would be a number that looks like
    /// a gate result and answers a different question. Recorded, never gated,
    /// per §8.6's determinism split.
    pub elapsed_ms: f64,
}

/// What one bench run observed.
#[derive(Debug, Clone)]
pub struct RelevanceBudgetReceipt {
    pub space: String,
    pub pages_in_space: usize,
    pub groups_in_space: usize,
    /// The unbounded pin-establishing pass. Reported, never gated: it is the
    /// pass the incremental bound explicitly does not apply to.
    pub rereference_pairs: usize,
    pub bounded_turns: Vec<SweepTurnReceipt>,
}

impl RelevanceBudgetReceipt {
    pub fn peak_queries(&self) -> u32 {
        self.bounded_turns
            .iter()
            .map(|t| t.queries)
            .max()
            .unwrap_or(0)
    }

    pub fn peak_rows(&self) -> u32 {
        self.bounded_turns.iter().map(|t| t.rows).max().unwrap_or(0)
    }

    /// Slowest bounded turn, in milliseconds.
    pub fn max_turn_ms(&self) -> f64 {
        self.bounded_turns
            .iter()
            .map(|turn| turn.elapsed_ms)
            .fold(0.0, f64::max)
    }

    /// Median bounded turn, lower of the two middles on an even count — the
    /// receipt is a record, not an estimator, so interpolating would invent a
    /// value no turn actually took.
    pub fn p50_turn_ms(&self) -> f64 {
        if self.bounded_turns.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.bounded_turns.iter().map(|t| t.elapsed_ms).collect();
        sorted.sort_by(f64::total_cmp);
        sorted[(sorted.len() - 1) / 2]
    }

    /// What the numbers above were measured on. A timing without its machine
    /// is not evidence (§8.6).
    pub fn machine(&self) -> String {
        format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
    }
}

/// A group as the corpus defines it, before it reaches SQL.
struct CorpusGroup {
    generated: bool,
    root_age_days: u64,
    retracted: u64,
}

/// Parse the frozen stream into the three relations the sweep reads.
struct Corpus {
    /// `page index -> space index`
    page_space: Vec<u64>,
    groups: Vec<CorpusGroup>,
    /// `(group index, page index)` in stream order.
    edges: Vec<(u64, u64)>,
    memories_seen: u64,
}

fn parse_corpus() -> Result<Corpus> {
    let mut buffer: Vec<u8> = Vec::with_capacity(8 << 20);
    write_corpus_stream(&mut buffer).context("generate corpus")?;
    let text = String::from_utf8(buffer).context("corpus is not UTF-8")?;

    let mut corpus = Corpus {
        page_space: Vec::new(),
        groups: Vec::new(),
        edges: Vec::new(),
        memories_seen: 0,
    };

    for line in text.lines().skip(1) {
        let mut parts = line.split('\t');
        let kind = parts.next().unwrap_or_default();
        let numbers: Vec<u64> = parts.filter_map(|field| field.parse().ok()).collect();
        match kind {
            "G" => {
                // G <index> <degree> <fanout> <age_days> <generated> <retracted>
                corpus.groups.push(CorpusGroup {
                    root_age_days: numbers[3],
                    generated: numbers[4] == 1,
                    retracted: numbers[5],
                });
            }
            "P" => corpus.page_space.push(numbers[1]),
            "E" => corpus.edges.push((numbers[0], numbers[1])),
            "M" => corpus.memories_seen += 1,
            _ => {}
        }
    }
    Ok(corpus)
}

/// Install the schema the sweep needs.
async fn install_schema(connection: &libsql::Connection) -> Result<()> {
    connection
        .execute_batch(
            "CREATE TABLE spaces (
                 id   TEXT PRIMARY KEY,
                 name TEXT NOT NULL UNIQUE
             );
             CREATE TABLE provenance_roots (
                 root_id               TEXT PRIMARY KEY,
                 root_kind             TEXT NOT NULL,
                 independence_group_id TEXT NOT NULL,
                 status                TEXT NOT NULL,
                 created_at            INTEGER NOT NULL
             );
             CREATE TABLE edges (
                 edge_id     TEXT PRIMARY KEY,
                 src_kind    TEXT NOT NULL,
                 src_id      TEXT NOT NULL,
                 dst_kind    TEXT NOT NULL,
                 dst_id      TEXT NOT NULL,
                 edge_type   TEXT NOT NULL DEFAULT 'relates',
                 grounded    INTEGER NOT NULL,
                 root_id     TEXT,
                 space       TEXT NOT NULL,
                 valid_until INTEGER
             );
             CREATE TABLE pages (
                 id      TEXT PRIMARY KEY,
                 title   TEXT NOT NULL,
                 content TEXT NOT NULL DEFAULT '',
                 version INTEGER NOT NULL DEFAULT 1,
                 space   TEXT NOT NULL DEFAULT '',
                 status  TEXT NOT NULL DEFAULT 'active',
                 -- `every_production_page_insert_names_kind` requires the seed
                 -- insert below to name `kind`, so the column has to exist
                 -- here. Without it the whole count-budget gate dies on
                 -- `no such column` before it measures anything -- which is a
                 -- gate that cannot fail, the exact thing §8.6 orders C0 ahead
                 -- of C1 to avoid.
                 kind    TEXT NOT NULL DEFAULT 'page'
             );
             CREATE INDEX idx_edges_active_grounded_space_type
                 ON edges(space, edge_type) WHERE valid_until IS NULL AND grounded = 1;
             CREATE INDEX idx_edges_src ON edges(src_kind, src_id);
             CREATE INDEX idx_edges_root ON edges(root_id);
             CREATE INDEX idx_pages_space ON pages(space);",
        )
        .await
        .context("install primary-evidence fixture schema")?;

    connection
        .execute_batch(MemoryDB::COMMUNITY_SUBSTRATE_DDL)
        .await
        .context("install the shipped grouping_leases registry")?;
    connection
        .execute_batch(GENESIS_SUBSTRATE_DDL)
        .await
        .context("install the shipped migration-108 genesis substrate")?;

    let tx = connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .context("begin schema transaction")?;
    ensure_relevance_tables(&tx)
        .await
        .context("install the shipped relevance substrate")?;
    ensure_remaining_substrate(&tx)
        .await
        .context("install the rest of migration 109")?;
    tx.commit().await.context("commit schema")?;
    Ok(())
}

/// Chunked multi-row inserts; one statement per row would dominate the run.
async fn insert_batched(
    connection: &libsql::Connection,
    prefix: &str,
    columns: usize,
    rows: &[Vec<libsql::Value>],
) -> Result<()> {
    const CHUNK: usize = 256;
    let tx = connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .context("begin load transaction")?;
    for chunk in rows.chunks(CHUNK) {
        let tuple = format!("({})", vec!["?"; columns].join(", "));
        let sql = format!("{prefix} VALUES {}", vec![tuple; chunk.len()].join(", "));
        let flat: Vec<libsql::Value> = chunk.iter().flatten().cloned().collect();
        tx.execute(&sql, libsql::params_from_iter(flat))
            .await
            .with_context(|| format!("load batch: {prefix}"))?;
    }
    tx.commit().await.context("commit load")?;
    Ok(())
}

fn text(value: impl Into<String>) -> libsql::Value {
    libsql::Value::Text(value.into())
}

/// Load the corpus and drive `max_turns` bounded sweep turns over one space.
///
/// `space_index` selects from `M6_SPACE_SPLIT_PERCENT`; index 0 is the 40%
/// space, the worst case and the one `R-BENCH-HUB` cares about.
///
/// The caller owns the connection, and that is not a style choice:
/// `repository_module_graph_matches_r4_25_group_6_census`
/// forbids a raw `libsql` origin anywhere under `crates/*/src` that is not
/// behind `#[cfg(test)]`, and this module is behind a cargo feature instead --
/// which the guard does not accept, and should not, since a feature can be
/// switched on in a shipped build. The bench target lives outside `src`, so
/// building the database there keeps the invariant intact rather than widening
/// the guard to admit a new exemption.
pub async fn run_relevance_budget_bench(
    connection: &libsql::Connection,
    space_index: usize,
    max_turns: usize,
) -> Result<RelevanceBudgetReceipt> {
    let corpus = parse_corpus()?;
    debug_assert!(corpus.memories_seen > 0);

    let connection = connection.clone();
    install_schema(&connection).await?;

    let space_name = format!("space-{space_index}");
    let space_id = format!("space-id-{space_index}");

    // Spaces.
    let space_rows: Vec<Vec<libsql::Value>> = (0..8)
        .map(|index| {
            vec![
                text(format!("space-id-{index}")),
                text(format!("space-{index}")),
            ]
        })
        .collect();
    insert_batched(&connection, "INSERT INTO spaces (id, name)", 2, &space_rows).await?;

    // Roots: one per group. `created_at` is the decay clock's input, so a
    // corpus age of `d` days becomes a timestamp `d` days before a fixed
    // present. The present is a constant, not `now()`: a bench whose input
    // moves with the wall clock is not reproducible.
    const PRESENT: i64 = 1_800_000_000;
    let root_rows: Vec<Vec<libsql::Value>> = corpus
        .groups
        .iter()
        .enumerate()
        .map(|(index, group)| {
            let age = (group.root_age_days.min(M6_ROOT_AGE_MAX_DAYS) as i64) * 86_400;
            vec![
                text(format!("root-{index}")),
                text(if group.generated {
                    "generated"
                } else {
                    "document_ingest"
                }),
                text(format!("g{index}")),
                text("active"),
                libsql::Value::Integer(PRESENT - age),
            ]
        })
        .collect();
    insert_batched(
        &connection,
        "INSERT INTO provenance_roots (root_id, root_kind, independence_group_id, status, created_at)",
        5,
        &root_rows,
    )
    .await?;

    // Pages.
    let page_rows: Vec<Vec<libsql::Value>> = corpus
        .page_space
        .iter()
        .enumerate()
        .map(|(index, space)| {
            vec![
                text(format!("p{index}")),
                text(format!("Page {index}")),
                text(format!("space-{space}")),
                // Named rather than left to the NOT NULL DEFAULT, per
                // `every_production_page_insert_names_kind`. The corpus stands
                // in for ordinary distilled pages, which is what
                // `page_kind_for` returns for any creation kind it does not
                // recognise -- and the sweep's own eligibility predicate drops
                // pages titled `overview`, so a corpus that minted them would
                // be measuring a filtered-out population.
                text("concept".to_string()),
            ]
        })
        .collect();
    insert_batched(
        &connection,
        "INSERT INTO pages (id, title, space, kind)",
        4,
        &page_rows,
    )
    .await?;

    // Group -> page edges. `R-CORPUS-RETFRAC`'s retraction budget is stated
    // over root edges; the sweep only ever sees page edges, so the budget is
    // spent on them here (capped at the group's fanout). Retracted edges get a
    // `valid_until`, which is exactly what the eligibility predicate drops.
    let mut retracted_left: Vec<u64> = corpus.groups.iter().map(|g| g.retracted).collect();
    let mut edge_rows: Vec<Vec<libsql::Value>> = Vec::with_capacity(corpus.edges.len());
    for (ordinal, (group, page)) in corpus.edges.iter().enumerate() {
        let space = corpus.page_space[*page as usize];
        let slot = &mut retracted_left[*group as usize];
        let retracted = if *slot > 0 {
            *slot -= 1;
            true
        } else {
            false
        };
        edge_rows.push(vec![
            text(format!("edge-{ordinal}")),
            text("page"),
            text(format!("p{page}")),
            text("entity"),
            text(format!("e{group}")),
            libsql::Value::Integer(1),
            text(format!("root-{group}")),
            text(format!("space-{space}")),
            if retracted {
                libsql::Value::Integer(PRESENT)
            } else {
                libsql::Value::Null
            },
        ]);
    }
    insert_batched(
        &connection,
        "INSERT INTO edges (edge_id, src_kind, src_id, dst_kind, dst_id, grounded, root_id, space, valid_until)",
        9,
        &edge_rows,
    )
    .await?;

    let pages_in_space = corpus
        .page_space
        .iter()
        .filter(|space| **space as usize == space_index)
        .count();
    let groups_in_space = {
        let mut seen = std::collections::BTreeSet::new();
        for (group, page) in &corpus.edges {
            if corpus.page_space[*page as usize] as usize == space_index {
                seen.insert(*group);
            }
        }
        seen.len()
    };

    // Turn 1 establishes the pin (unbounded by construction).
    let tx = connection
        .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
        .await
        .context("begin rereference")?;
    let first = run_relevance_sweep(&tx, &space_id, &space_name, 7, 1)
        .await
        .context("rereference pass")?;
    tx.commit().await.context("commit rereference")?;
    let rereference_pairs = match first {
        RelevanceTurn::Rereferenced(pass) => pass.pairs_written,
        other => anyhow::bail!("first turn was {other:?}, expected the pin-establishing pass"),
    };

    // Every later turn is bounded, and every one of them is measured.
    let mut bounded_turns = Vec::new();
    for attempt in 0..max_turns {
        let tx = connection
            .transaction_with_behavior(libsql::TransactionBehavior::Immediate)
            .await
            .context("begin bounded turn")?;
        let started = std::time::Instant::now();
        let turn = run_relevance_sweep(&tx, &space_id, &space_name, 8, attempt as i64 + 2).await;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let receipt = match &turn {
            Ok(RelevanceTurn::Swept(sweep)) => SweepTurnReceipt {
                kind: "swept",
                endpoints_considered: sweep.endpoints_considered,
                queries: sweep.budget.queries,
                rows: sweep.budget.rows,
                elapsed_ms,
            },
            Ok(RelevanceTurn::Idle) => {
                let _ = tx.commit().await;
                break;
            }
            Ok(other) => anyhow::bail!("bounded turn returned {other:?}"),
            // A budget breach is returned as an error by the counters
            // themselves. That is the measurement, not a harness failure, so
            // it is surfaced rather than swallowed.
            Err(error) => anyhow::bail!("bounded turn {attempt} breached its budget: {error}"),
        };
        tx.commit().await.context("commit bounded turn")?;
        bounded_turns.push(receipt);
    }

    Ok(RelevanceBudgetReceipt {
        space: space_name,
        pages_in_space,
        groups_in_space,
        rereference_pairs,
        bounded_turns,
    })
}
