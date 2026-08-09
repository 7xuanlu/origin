// SPDX-License-Identifier: Apache-2.0
//! S0-144 — catalog completeness is a **test**, not a checklist.
//!
//! The mutation catalog names every weakening the G3 and G5 families must catch.
//! A row nobody executes is a gate that exists only in prose, and the way that
//! happens is never a decision — it is a row added to the catalog after the
//! suite was written, or a test renamed out from under one. Both are structural,
//! so both are caught here:
//!
//! 1. every `G3.*`/`G5.*` ID in the catalog is registered below, exactly once;
//! 2. every registration names a real catalog ID (no phantom rows);
//! 3. every registration bound to a gate names a test function that **exists**
//!    in the M6 tree, found by reading the source rather than by trusting the
//!    string.
//!
//! Deferred rows are declared, not omitted. `Coverage::Deferred` carries the
//! slice that owns the row, so an unexecuted gate is a visible, attributed
//! entry instead of a silent absence — which is the failure S0-144 exists to
//! make impossible.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// How a catalog row is accounted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coverage {
    /// The name of a `#[tokio::test]`/`#[test]` function in `crates/wenlan-core/src/m6/`.
    Gate(&'static str),
    /// Not executable in this slice, with the reason and the owner.
    Deferred(&'static str),
}

/// The registry. Every catalog ID appears here exactly once.
///
/// `G3.3` was PR-B2's second deferral and is bound as of PR-B3: machine E mints
/// the receipt, so "a duplicate retry writes a second receipt" is now a
/// falsifiable claim about running code. `G3.4` stays deferred — it is a
/// property **of publication**, and PR-B publishes nothing, so a gate over it
/// could only assert that nothing happened, which passes against an empty
/// implementation and a broken one alike.
const REGISTRY: [(&str, Coverage); 14] = [
    (
        "G3.1",
        Coverage::Gate("two_candidates_cannot_both_claim_one_group"),
    ),
    (
        "G3.2",
        Coverage::Gate("page_id_is_a_function_of_the_slot_alone"),
    ),
    ("G3.3", Coverage::Gate("a_retry_reuses_the_same_receipt")),
    (
        "G3.4",
        Coverage::Deferred(
            "coverage is written by publication, and PR-B's dry run publishes nothing — PR-C",
        ),
    ),
    (
        "G3.5",
        Coverage::Gate("the_reservation_exit_matrix_is_a_total_function"),
    ),
    (
        "G3.6",
        Coverage::Gate("ownership_requires_an_unexpired_row"),
    ),
    (
        "G3.7",
        Coverage::Gate("a_manual_run_and_the_scheduler_join_one_operation"),
    ),
    (
        "G5.1",
        Coverage::Gate("an_unaccounted_group_reads_zero_and_is_repaired"),
    ),
    (
        "G5.2",
        Coverage::Gate("the_cursor_round_trips_and_clears_on_wrap"),
    ),
    (
        "G5.3",
        Coverage::Gate("recovery_is_idempotent_across_a_restart"),
    ),
    (
        "G5.4",
        Coverage::Gate("a_cap_hit_leaves_the_remainder_frontier_visible"),
    ),
    (
        "G5.5",
        Coverage::Gate("a_newly_eligible_group_is_repaired_on_the_next_pass"),
    ),
    (
        "G5.6",
        Coverage::Gate("compaction_nulls_the_payload_and_keeps_the_row"),
    ),
    (
        "G5.7",
        Coverage::Gate("a_permanently_small_space_surfaces_once"),
    ),
];

const CATALOG: &str = "docs/contracts/m6-mutation-catalog.md";

/// Repo root, resolved from this crate's manifest dir — the same derivation
/// `drift_guard::repo_root` uses, kept local because that one is private.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("resolve repo root")
}

/// Every `G3.x` / `G5.x` ID in the catalog's table rows, in document order.
fn catalog_ids() -> Vec<String> {
    let catalog = std::fs::read_to_string(repo_root().join(CATALOG))
        .unwrap_or_else(|error| panic!("read {CATALOG}: {error}"));
    let mut ids = Vec::new();
    for line in catalog.lines() {
        let trimmed = line.trim_start();
        // Table rows only: a prose mention of a row ID is a reference, not a
        // declaration, and counting one would make the gate depend on how the
        // catalog's prose is worded.
        let Some(rest) = trimmed.strip_prefix("| ") else {
            continue;
        };
        let Some(cell) = rest.split('|').next().map(str::trim) else {
            continue;
        };
        let is_row_id = (cell.starts_with("G3.") || cell.starts_with("G5."))
            && cell[3..].chars().all(|c| c.is_ascii_digit())
            && !cell[3..].is_empty();
        if is_row_id {
            ids.push(cell.to_string());
        }
    }
    assert!(!ids.is_empty(), "no G3/G5 rows parsed out of {CATALOG}");
    ids
}

/// Every test-function name declared under `crates/wenlan-core/src/m6/`.
fn m6_test_function_names() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/m6");
    let mut names = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read m6 dir") {
        let path = entry.expect("read m6 dir entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read m6 source");
        for line in source.lines() {
            let trimmed = line.trim_start();
            let after_fn = trimmed
                .strip_prefix("async fn ")
                .or_else(|| trimmed.strip_prefix("fn "));
            if let Some(rest) = after_fn {
                if let Some(name) = rest.split('(').next() {
                    names.push(name.trim().to_string());
                }
            }
        }
    }
    names
}

/// (1) and (2): the registry and the catalog name the same set of rows.
#[tokio::test]
async fn every_catalog_row_is_registered_exactly_once() {
    let catalog = catalog_ids();
    let mut registered: BTreeMap<&str, usize> = BTreeMap::new();
    for (id, _) in REGISTRY {
        *registered.entry(id).or_default() += 1;
    }

    let duplicates: Vec<&&str> = registered
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|(id, _)| id)
        .collect();
    assert!(duplicates.is_empty(), "registered twice: {duplicates:?}");

    let missing: Vec<&String> = catalog
        .iter()
        .filter(|id| !registered.contains_key(id.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "catalog rows with no registered gate: {missing:?}"
    );

    let phantom: Vec<&&str> = registered
        .keys()
        .filter(|id| !catalog.iter().any(|row| row == *id))
        .collect();
    assert!(
        phantom.is_empty(),
        "registered gates for rows the catalog does not have: {phantom:?}"
    );
    assert_eq!(catalog.len(), REGISTRY.len());
}

/// (3): a registration that names a test must name one that exists. This is the
/// half that keeps the registry from decaying into a wish list — renaming a gate
/// without updating its row fails here rather than quietly unbinding it.
#[tokio::test]
async fn every_registered_gate_names_a_real_test() {
    let names = m6_test_function_names();
    let dangling: Vec<(&str, &str)> = REGISTRY
        .iter()
        .filter_map(|(id, coverage)| match coverage {
            Coverage::Gate(test) if !names.iter().any(|name| name == test) => Some((*id, *test)),
            _ => None,
        })
        .collect();
    assert!(
        dangling.is_empty(),
        "registered gates naming tests that do not exist: {dangling:?}"
    );
}

/// Deferrals are budgeted, not open-ended: each carries a reason, and the count
/// is pinned so a new one is a deliberate edit rather than a quiet slide.
#[tokio::test]
async fn deferrals_are_declared_and_bounded() {
    let deferred: Vec<(&str, &str)> = REGISTRY
        .iter()
        .filter_map(|(id, coverage)| match coverage {
            Coverage::Deferred(reason) => Some((*id, *reason)),
            Coverage::Gate(_) => None,
        })
        .collect();
    assert_eq!(
        deferred.len(),
        1,
        "the PR-B3 deferral budget is G3.4 alone: {deferred:?}"
    );
    assert!(deferred.iter().all(|(_, reason)| !reason.trim().is_empty()));
}
