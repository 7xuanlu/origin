// SPDX-License-Identifier: Apache-2.0
//! Teeth for the M5 reader manifest.
//!
//! The source-side positive control is deliberately **not** here. A test that
//! re-scans `crates/wenlan-server/src` for `.route(` would be a second, weaker
//! copy of `TrackedRouter`, which is the exact hand-copied-canonical-thing
//! pattern that produced three wrong route counts during Stage 0. `TrackedRouter`
//! already fails closed on both sides -- `route()` refuses an unclassified path,
//! `finish()` asserts set equality -- so building the router *is* the scan, and
//! it runs in `wenlan-server`'s own tests where the router lives.
//!
//! What is left for this file is what `TrackedRouter` cannot see: that the
//! committed inventory document still describes this table, and that the
//! classification columns obey their own rules.

use super::*;
use std::collections::BTreeSet;

/// The inventory document, compiled in so it cannot be edited out from under
/// the table without this test noticing.
const INVENTORY: &str = include_str!("../contracts/m5-reader-manifest-inventory.md");

fn clean(cell: &str) -> String {
    cell.replace(['*', '`'], "").trim().to_string()
}

/// Rows of the markdown table whose header starts with `header` and has `ncols`
/// columns. Stops at the first non-table line, so prose between tables is never
/// absorbed -- an earlier extraction keyed on line numbers and swallowed a
/// prose table's cells.
fn doc_table(header: &str, ncols: usize) -> Vec<Vec<String>> {
    let lines: Vec<&str> = INVENTORY.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim()
            .trim_matches('|')
            .split('|')
            .map(clean)
            .collect();
        if cells.first().map(String::as_str) != Some(header) || cells.len() != ncols {
            continue;
        }
        let mut rows = Vec::new();
        let mut j = i + 1;
        // Skip the |---|---| separator.
        if lines
            .get(j)
            .is_some_and(|l| l.starts_with('|') && l.chars().all(|c| matches!(c, '|' | '-' | ' ')))
        {
            j += 1;
        }
        while let Some(line) = lines.get(j) {
            if !line.starts_with('|') {
                break;
            }
            let cells: Vec<String> = line
                .trim()
                .trim_matches('|')
                .split('|')
                .map(clean)
                .collect();
            if cells.len() != ncols {
                break;
            }
            rows.push(cells);
            j += 1;
        }
        return rows;
    }
    panic!("inventory table with header {header:?} and {ncols} columns not found");
}

fn method_name(m: ReaderMethod) -> &'static str {
    match m {
        ReaderMethod::Get => "GET",
        ReaderMethod::Post => "POST",
        ReaderMethod::Put => "PUT",
        ReaderMethod::Delete => "DELETE",
        ReaderMethod::Patch => "PATCH",
    }
}

fn builder_name(b: Builder) -> &'static str {
    match b {
        Builder::Main => "main",
        Builder::Repair => "repair",
        Builder::MainAndRepair => "main + repair",
    }
}

fn bearing_name(p: PageBearing) -> &'static str {
    match p {
        PageBearing::Yes => "yes",
        PageBearing::No => "no",
    }
}

fn class_name(c: TruthClass) -> &'static str {
    match c {
        TruthClass::Automatic => "automatic",
        TruthClass::NotApplicable => "not_applicable",
    }
}

fn shape_name(s: MarkerShape) -> &'static str {
    match s {
        MarkerShape::None => "none",
        MarkerShape::Collection => "collection",
        MarkerShape::NamedPage => "named_page",
        MarkerShape::CollectionAndNamedPage => "collection + named_page",
    }
}

/// Positive control, document side: the table and the committed inventory must
/// describe the same set, cell for cell. Extra and missing both fail.
///
/// Weakening this (comparing counts, or only one direction) lets the document
/// rot into a snapshot while the table moves, which is what it exists to prevent.
#[test]
fn http_table_equals_inventory_document() {
    let doc: BTreeSet<Vec<String>> = doc_table("Method", 8).into_iter().collect();
    let table: BTreeSet<Vec<String>> = HTTP_READERS
        .iter()
        .map(|r| {
            vec![
                method_name(r.method).to_string(),
                r.path.to_string(),
                builder_name(r.builder).to_string(),
                bearing_name(r.page_bearing).to_string(),
                class_name(r.class).to_string(),
                shape_name(r.marker_shape).to_string(),
                r.adapter.to_string(),
                r.evidence.to_string(),
            ]
        })
        .collect();

    let only_doc: Vec<_> = doc.difference(&table).collect();
    let only_table: Vec<_> = table.difference(&doc).collect();
    assert!(
        only_doc.is_empty() && only_table.is_empty(),
        "HTTP manifest drifted from the inventory document.\n\
         only in document: {only_doc:#?}\nonly in table: {only_table:#?}"
    );
}

#[test]
fn mcp_table_equals_inventory_document() {
    let doc: BTreeSet<Vec<String>> = doc_table("Tool", 5).into_iter().collect();
    let table: BTreeSet<Vec<String>> = MCP_READERS
        .iter()
        .map(|r| {
            vec![
                r.tool.to_string(),
                bearing_name(r.page_bearing).to_string(),
                class_name(r.class).to_string(),
                shape_name(r.marker_shape).to_string(),
                r.adapter.to_string(),
            ]
        })
        .collect();
    assert_eq!(
        doc, table,
        "MCP manifest drifted from the inventory document"
    );
}

#[test]
fn cli_table_equals_inventory_document() {
    let doc: BTreeSet<Vec<String>> = doc_table("Subcommand", 5).into_iter().collect();
    let table: BTreeSet<Vec<String>> = CLI_READERS
        .iter()
        .map(|r| {
            vec![
                r.subcommand.to_string(),
                bearing_name(r.page_bearing).to_string(),
                class_name(r.class).to_string(),
                shape_name(r.marker_shape).to_string(),
                r.adapter.to_string(),
            ]
        })
        .collect();
    assert_eq!(
        doc, table,
        "CLI manifest drifted from the inventory document"
    );
}

/// The counts the binding spec states, asserted rather than trusted. Three
/// earlier enumerations were wrong; a bare count is the cheapest thing that
/// catches a bulk edit.
#[test]
fn manifest_counts_match_the_spec() {
    // 167, not the 162 the Stage 0 document was generated with: PR #399 added
    // the three `/api/spaces/default` routes and PR #407 added POST/PATCH
    // `/api/brief`. The router coverage assertion caught both drifts.
    assert_eq!(
        HTTP_READERS.len(),
        168,
        "registered (method, path, handler) triples"
    );
    assert_eq!(MCP_READERS.len(), 29, "#[tool( declarations");
    assert_eq!(CLI_READERS.len(), 19, "Commands variants");

    let entries: Vec<_> = runtime_entries().collect();
    assert_eq!(
        entries.len(),
        172,
        "(builder, method, path) runtime entries"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|(b, _, _)| *b == Builder::Main)
            .count(),
        166,
        "main builder entries"
    );
    assert_eq!(
        entries
            .iter()
            .filter(|(b, _, _)| *b == Builder::Repair)
            .count(),
        6,
        "repair builder entries"
    );

    let bearing = HTTP_READERS
        .iter()
        .filter(|r| r.page_bearing == PageBearing::Yes)
        .count();
    assert_eq!(bearing, 59, "page-bearing HTTP routes");
}

#[test]
fn brief_routes_have_explicit_truth_shapes() {
    let post = HTTP_READERS
        .iter()
        .find(|row| row.method == ReaderMethod::Post && row.path == "/api/brief")
        .expect("POST /api/brief manifest row");
    assert_eq!(post.page_bearing, PageBearing::Yes);
    assert_eq!(post.class, TruthClass::Automatic);
    assert_eq!(post.marker_shape, MarkerShape::None);

    let patch = HTTP_READERS
        .iter()
        .find(|row| row.method == ReaderMethod::Patch && row.path == "/api/brief")
        .expect("PATCH /api/brief manifest row");
    assert_eq!(patch.page_bearing, PageBearing::No);
    assert_eq!(patch.class, TruthClass::NotApplicable);
    assert_eq!(patch.marker_shape, MarkerShape::None);
}

/// No runtime identity may be registered twice. `(method, path)` alone collides
/// legitimately across builders, so this keys on the full triple -- if this ever
/// fails, `http_reader` would be resolving a route to an arbitrary one of two
/// classifications.
#[test]
fn runtime_entries_are_unique() {
    let mut seen = BTreeSet::new();
    for entry in runtime_entries() {
        assert!(seen.insert(entry), "duplicate runtime entry: {entry:?}");
    }
}

/// Teeth 6: `explicit` is a per-call signal and never a route property.
///
/// The enum has no `Explicit` variant, so the interesting half is the pairing:
/// a page-bearing reader is `automatic` (something must gate it) and a
/// non-page-bearing one is `not_applicable` (there is nothing to gate). A row
/// that says "carries prose" and "needs no adapter" is a contradiction.
#[test]
fn class_agrees_with_page_bearing_on_every_surface() {
    for r in HTTP_READERS {
        let expected = match r.page_bearing {
            PageBearing::Yes => TruthClass::Automatic,
            PageBearing::No => TruthClass::NotApplicable,
        };
        assert_eq!(
            r.class,
            expected,
            "{} {} class disagrees with page_bearing",
            method_name(r.method),
            r.path
        );
    }
    for r in MCP_READERS {
        let expected = match r.page_bearing {
            PageBearing::Yes => TruthClass::Automatic,
            PageBearing::No => TruthClass::NotApplicable,
        };
        assert_eq!(r.class, expected, "mcp {} class disagrees", r.tool);
    }
    for r in CLI_READERS {
        let expected = match r.page_bearing {
            PageBearing::Yes => TruthClass::Automatic,
            PageBearing::No => TruthClass::NotApplicable,
        };
        assert_eq!(r.class, expected, "cli {} class disagrees", r.subcommand);
    }
}

/// A reader that carries no page prose has nothing to gate, so it must not
/// carry a marker shape either. The converse is not asserted: a page-bearing
/// route is `none` by default and that is the fail-closed direction.
#[test]
fn only_page_bearing_readers_carry_a_marker_shape() {
    for r in HTTP_READERS {
        if r.page_bearing == PageBearing::No {
            assert_eq!(
                r.marker_shape,
                MarkerShape::None,
                "{} {} is not page-bearing but carries a marker shape",
                method_name(r.method),
                r.path
            );
        }
    }
}

/// Teeth 7: the marker-shape allowlist is fail-closed.
///
/// Every route not named here must be `none`. Adding a route cannot grant it a
/// shape by accident -- the new row defaults to `none` and this test keeps it
/// there until someone edits this list on purpose.
#[test]
fn marker_shape_allowlist_is_fail_closed() {
    // `/api/pages/recent` and `/api/pages/recent-changes` were `collection`
    // until PR-C and are deliberately not any more. `Collection` is documented
    // as being only for item types that can carry a page identity *and* both
    // axes, never prose -- and `RecentActivityItem` carries a prose `snippet`
    // with no axes at all, `PageChange` a bare title. The shape promised a
    // carve-out their wire types cannot honour, which is a trap rather than a
    // leak: the adapters are Full-only, so nothing over-exposed, but the next
    // person to "fix" them to honour the promise would reduce a page into a
    // struct with nowhere to put the axes. `none` is the honest shape until
    // those types grow them.
    //
    // `GET /api/pages/{id}/map` left `NAMED_PAGE` for the same reason in the
    // ceremony PR: `PageMapNode`/`PageMapEdge` carry a bare `label` and neither
    // axis, and hiding a page inside a map is a graph transformation (incident
    // edges are keyed by node ids) rather than a filter. `none` refuses; it does
    // not downgrade.
    const COLLECTION: &[(ReaderMethod, &str)] = &[
        (ReaderMethod::Get, "/api/pages"),
        (ReaderMethod::Post, "/api/pages/search"),
    ];
    const NAMED_PAGE: &[(ReaderMethod, &str)] = &[
        (ReaderMethod::Get, "/api/pages/{id}"),
        (ReaderMethod::Get, "/api/pages/{id}/links"),
        (ReaderMethod::Get, "/api/pages/{id}/revisions"),
        (ReaderMethod::Get, "/api/pages/{id}/sources"),
    ];

    for r in HTTP_READERS {
        let key = (r.method, r.path);
        let expected = if COLLECTION.contains(&key) {
            MarkerShape::Collection
        } else if NAMED_PAGE.contains(&key) {
            MarkerShape::NamedPage
        } else {
            MarkerShape::None
        };
        assert_eq!(
            r.marker_shape,
            expected,
            "{} {} marker shape is not fail-closed",
            method_name(r.method),
            r.path
        );
    }

    assert_eq!(
        HTTP_READERS
            .iter()
            .filter(|r| r.marker_shape == MarkerShape::Collection)
            .count(),
        2
    );
    assert_eq!(
        HTTP_READERS
            .iter()
            .filter(|r| r.marker_shape == MarkerShape::NamedPage)
            .count(),
        4
    );
    assert_eq!(
        HTTP_READERS
            .iter()
            .filter(|r| r.marker_shape == MarkerShape::None)
            .count(),
        162
    );
}

/// The shapes D3 excludes unconditionally. A marker must never do anything at
/// automatic context, search, or an export -- these are the routes whose whole
/// purpose is feeding an agent without a human in the loop.
#[test]
fn context_search_and_exports_refuse_every_marker() {
    for path in [
        "/api/context",
        "/api/brief",
        "/api/search",
        "/api/memory/search",
        "/api/pages/export",
        "/api/pages/{id}/export",
    ] {
        let rows: Vec<_> = HTTP_READERS.iter().filter(|r| r.path == path).collect();
        assert!(!rows.is_empty(), "{path} is not in the manifest at all");
        for r in rows {
            assert_eq!(
                r.marker_shape,
                MarkerShape::None,
                "{} {} must refuse a marker",
                method_name(r.method),
                r.path
            );
        }
    }
}

/// Teeth 12: `collection` is conditional on the item type carrying a page
/// identity **and** both truth axes. `OrphanLink { label, count }` carries
/// neither, so the route stays `none` -- it was on the allowlist for one draft.
#[test]
fn orphan_links_is_not_a_collection() {
    let row = HTTP_READERS
        .iter()
        .find(|r| r.path == "/api/pages/orphan-links")
        .expect("orphan-links must be in the manifest");
    assert_eq!(
        row.marker_shape,
        MarkerShape::None,
        "OrphanLink carries no page id and no axes, so it cannot receive a collection grant"
    );
}

/// MCP has no human gesture behind it -- the agent is the caller -- so no tool
/// may carry a shape. This is the hard half of the per-surface rule; the
/// transmission half is tested on the MCP surface itself.
#[test]
fn no_mcp_tool_carries_a_marker_shape() {
    for r in MCP_READERS {
        assert_eq!(
            r.marker_shape,
            MarkerShape::None,
            "mcp tool {} must not carry a marker shape",
            r.tool
        );
    }
}

/// `wenlan pages` is the only compound shape, and it is compound because one
/// subcommand both lists and opens. Pinning it here means a second compound row
/// has to be argued for rather than copied.
#[test]
fn wenlan_pages_is_the_only_compound_cli_shape() {
    let compound: Vec<_> = CLI_READERS
        .iter()
        .filter(|r| r.marker_shape == MarkerShape::CollectionAndNamedPage)
        .map(|r| r.subcommand)
        .collect();
    assert_eq!(compound, vec!["wenlan pages"]);
}

/// The adapter column is an enforcement address, not a note. A cell that does
/// not look like a function name (the `move` a closure produced, an empty cell
/// on a page-bearing row) is the drift this catches; resolving it to a real
/// symbol is teeth 4, which needs the language server and lands with the
/// adapters themselves.
#[test]
fn page_bearing_rows_carry_an_adapter_address() {
    for r in HTTP_READERS {
        match r.page_bearing {
            PageBearing::Yes => {
                assert!(
                    !r.adapter.is_empty() && r.adapter != "—" && r.adapter != "move",
                    "{} {} is page-bearing but its adapter cell is {:?}",
                    method_name(r.method),
                    r.path,
                    r.adapter
                );
                assert!(
                    r.adapter
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':'),
                    "{} {} adapter {:?} is not a symbol",
                    method_name(r.method),
                    r.path,
                    r.adapter
                );
            }
            PageBearing::No => {}
        }
        assert!(
            !r.evidence.is_empty(),
            "{} {} carries no evidence for its classification",
            method_name(r.method),
            r.path
        );
    }
}

/// A `MainAndRepair` call site answers for both builders; a `Main` one does not
/// answer for `repair`. Getting this backwards would make the drift assert in
/// `finish_restricted` vacuous.
#[test]
fn builder_membership_is_exact() {
    assert!(Builder::Main.installs_into(Builder::Main));
    assert!(!Builder::Main.installs_into(Builder::Repair));
    assert!(Builder::Repair.installs_into(Builder::Repair));
    assert!(!Builder::Repair.installs_into(Builder::Main));
    assert!(Builder::MainAndRepair.installs_into(Builder::Main));
    assert!(Builder::MainAndRepair.installs_into(Builder::Repair));

    // The four triples that land in both builders, keyed on (method, path)
    // because `/api/lint` contributes two of them. They come from exactly two
    // call sites -- `lint_routes::register` and `repair_routes::register_execution`
    // -- and this is the +4 that makes 162 call sites into 166 runtime entries.
    let both: BTreeSet<(&str, &str)> = HTTP_READERS
        .iter()
        .filter(|r| r.builder == Builder::MainAndRepair)
        .map(|r| (method_name(r.method), r.path))
        .collect();
    assert_eq!(
        both,
        BTreeSet::from([
            ("GET", "/api/lint"),
            ("POST", "/api/lint"),
            ("POST", "/api/repairs/apply"),
            ("POST", "/api/repairs/verify"),
        ]),
        "the set of call sites installed into both builders changed"
    );
}

/// Lookup resolves per builder, so a repair-only route is invisible from main.
#[test]
fn http_reader_lookup_respects_builder() {
    let health = http_reader(Builder::Main, ReaderMethod::Get, "/api/health");
    assert!(health.is_some(), "/api/health must resolve in main");

    for (builder, method, path) in runtime_entries() {
        assert!(
            http_reader(builder, method, path).is_some(),
            "{builder:?} {method:?} {path} is a runtime entry but does not resolve"
        );
    }
}
