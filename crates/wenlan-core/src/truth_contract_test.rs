// SPDX-License-Identifier: Apache-2.0
//! The four cases of binding-spec section 7, plus the two ways a marker is
//! supposed to be inert.
//!
//! These are the pure half of the contract. The wire half -- that a refusal is
//! an HTTP status and that a marked call leaves a row behind -- lives in
//! `wenlan-server`'s `truth_guard` tests, because neither is observable from
//! here.

use super::*;

fn pages(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| (*s).to_string()).collect()
}

// ---- section 6: contract negotiation -------------------------------------

#[test]
fn only_the_exact_served_version_declares_the_contract() {
    assert_eq!(parse_contract(Some("1")), ClientContract::V1);
    assert_eq!(parse_contract(Some(" 1 ")), ClientContract::V1);
}

#[test]
fn absent_malformed_and_unknown_versions_are_all_legacy() {
    for raw in [
        None,
        Some(""),
        Some("v1"),
        Some("banana"),
        Some("0"),
        Some("-1"),
    ] {
        assert_eq!(
            parse_contract(raw),
            ClientContract::Legacy,
            "expected legacy for {raw:?}"
        );
    }
}

/// Trust contracts do not negotiate upward. A client claiming a version we have
/// never shipped tells us nothing about what it renders, and the permissive
/// reading of "newer must be at least as good" is how an unrendered axis ships.
#[test]
fn a_version_newer_than_we_serve_is_legacy_not_supported() {
    assert_eq!(
        parse_contract(Some(&(TRUTH_CONTRACT_VERSION + 1).to_string())),
        ClientContract::Legacy
    );
    assert_eq!(parse_contract(Some("99")), ClientContract::Legacy);
}

// ---- the marker itself ----------------------------------------------------

#[test]
fn only_the_exact_marker_value_counts_as_a_gesture() {
    assert!(marker_present(Some("explicit")));
    assert!(marker_present(Some("  Explicit ")));
}

#[test]
fn a_malformed_marker_is_absent_not_present() {
    for raw in [
        None,
        Some(""),
        Some("yes"),
        Some("true"),
        Some("explicit-ish"),
    ] {
        assert!(!marker_present(raw), "expected absent for {raw:?}");
    }
}

// ---- section 7: the four cases, on a named-page route ---------------------

#[test]
fn case_1_no_contract_declared_gets_automatic() {
    assert_eq!(
        resolve(
            MarkerShape::NamedPage,
            ClientContract::Legacy,
            true,
            &pages(&["pg1"])
        ),
        TruthDecision::Grant(TruthGrant::Automatic)
    );
}

/// The case the previous design could not express at all: it had no way to tell
/// an M5-aware client's human click from its 10-second poll.
#[test]
fn case_2_contract_declared_but_no_marker_gets_automatic() {
    assert_eq!(
        resolve(
            MarkerShape::NamedPage,
            ClientContract::V1,
            false,
            &pages(&["pg1"])
        ),
        TruthDecision::Grant(TruthGrant::Automatic)
    );
}

#[test]
fn case_3_contract_and_marker_together_grant_the_named_page() {
    assert_eq!(
        resolve(
            MarkerShape::NamedPage,
            ClientContract::V1,
            true,
            &pages(&["pg1"])
        ),
        TruthDecision::Grant(TruthGrant::NamedPages(pages(&["pg1"])))
    );
}

#[test]
fn case_4_an_unknown_contract_version_gets_automatic() {
    assert_eq!(
        resolve(
            MarkerShape::NamedPage,
            parse_contract(Some("7")),
            true,
            &pages(&["pg1"])
        ),
        TruthDecision::Grant(TruthGrant::Automatic)
    );
}

// ---- section 4: the shape gate -------------------------------------------

/// The half that needs no client cooperation. `/api/context`, `/api/search` and
/// both exports are `None`-shaped, so this is what stops an agent wired to
/// transmit the marker from contaminating automatic context.
#[test]
fn a_marker_at_a_none_shaped_route_is_refused() {
    assert_eq!(
        resolve(
            MarkerShape::None,
            ClientContract::V1,
            true,
            &pages(&["pg1"])
        ),
        TruthDecision::Refuse
    );
}

/// Refusal is a fact about the route, not about the caller, so a client that
/// never declared the contract is refused on the same routes.
#[test]
fn refusal_does_not_depend_on_the_client_declaring_anything() {
    assert_eq!(
        resolve(MarkerShape::None, ClientContract::Legacy, true, &[]),
        TruthDecision::Refuse
    );
}

/// Refusing an *ignored* marker is the whole point; a request that never sent
/// one must still be served.
#[test]
fn an_unmarked_call_at_a_none_shaped_route_is_served_not_refused() {
    assert_eq!(
        resolve(MarkerShape::None, ClientContract::V1, false, &[]),
        TruthDecision::Grant(TruthGrant::Automatic)
    );
}

#[test]
fn a_marked_collection_call_lists_entries() {
    assert_eq!(
        resolve(MarkerShape::Collection, ClientContract::V1, true, &[]),
        TruthDecision::Grant(TruthGrant::CollectionEntries)
    );
}

/// Spraying the marker on every request buys nothing on a route that names no
/// page. This is the structural half of "the marker binds to the pages the call
/// names" -- the cheap defence that does not depend on the marker being honest.
#[test]
fn a_marker_naming_no_page_grants_nothing_on_a_named_page_route() {
    assert_eq!(
        resolve(MarkerShape::NamedPage, ClientContract::V1, true, &[]),
        TruthDecision::Grant(TruthGrant::Automatic)
    );
}

#[test]
fn the_compound_shape_lists_when_bare_and_opens_when_it_names_a_page() {
    assert_eq!(
        resolve(
            MarkerShape::CollectionAndNamedPage,
            ClientContract::V1,
            true,
            &[]
        ),
        TruthDecision::Grant(TruthGrant::CollectionEntries)
    );
    assert_eq!(
        resolve(
            MarkerShape::CollectionAndNamedPage,
            ClientContract::V1,
            true,
            &pages(&["pg1"])
        ),
        TruthDecision::Grant(TruthGrant::NamedPages(pages(&["pg1"])))
    );
}

// ---- section 8: the adapters are installed, and inert ---------------------

/// The single line that makes PR-B a no-op. Every other test in this file
/// describes what happens *after* PR-C; this one is why none of it happens yet.
#[test]
fn at_generation_zero_every_adapter_is_pass_through() {
    for grant in [
        TruthGrant::Automatic,
        TruthGrant::CollectionEntries,
        TruthGrant::NamedPages(pages(&["other"])),
    ] {
        for supported in [true, false] {
            assert_eq!(
                visible_at(0, &grant, "pg1", supported),
                Visibility::Full,
                "generation 0 filtered {grant:?} (supported={supported}) -- PR-B is no longer inert"
            );
        }
    }
}

#[test]
fn a_supported_page_is_always_fully_visible() {
    for grant in [
        TruthGrant::Automatic,
        TruthGrant::CollectionEntries,
        TruthGrant::NamedPages(vec![]),
    ] {
        assert_eq!(visible_at(1, &grant, "pg1", true), Visibility::Full);
    }
}

#[test]
fn an_unsupported_page_is_hidden_from_an_automatic_reader() {
    assert_eq!(
        visible_at(1, &TruthGrant::Automatic, "pg1", false),
        Visibility::Hidden
    );
}

/// Discovery without prose. A human cannot name a page they cannot see, so the
/// entry is allowed -- but only as an entry.
#[test]
fn an_unsupported_page_appears_as_an_entry_in_a_marked_collection() {
    assert_eq!(
        visible_at(1, &TruthGrant::CollectionEntries, "pg1", false),
        Visibility::EntryOnly
    );
}

#[test]
fn a_named_page_grant_opens_the_page_it_named() {
    assert_eq!(
        visible_at(1, &TruthGrant::NamedPages(pages(&["pg1"])), "pg1", false),
        Visibility::Full
    );
}

/// The embedded-other-page rule. A grant for page A never covers page B's title
/// riding along in A's links, map, or sources -- B carries no axes there, so it
/// would appear as unearned trust.
#[test]
fn a_named_page_grant_does_not_cover_a_page_riding_along_beside_it() {
    assert_eq!(
        visible_at(1, &TruthGrant::NamedPages(pages(&["pg1"])), "pg2", false),
        Visibility::Hidden
    );
}

// ---- inventory teeth 9: the surfaces that may never transmit --------------

/// Every `.rs` file under one of this workspace's crate source trees.
fn crate_sources(crate_name: &str) -> Vec<(std::path::PathBuf, String)> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join(crate_name)
        .join("src");
    assert!(root.is_dir(), "{} has no src/", root.display());

    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read_dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let body = std::fs::read_to_string(&path).expect("read source");
                out.push((path, body));
            }
        }
    }
    assert!(!out.is_empty(), "scanned {crate_name} and found no sources");
    out
}

/// MCP tools and the CLI may **never** transmit the intent marker.
///
/// There is no human gesture behind a tool call -- the agent is the caller --
/// and a scripted CLI subcommand is not browsing. This is the cooperative half
/// of the gate and it is soft by construction: the shape gate holds even when
/// this one is bypassed. It is worth a test anyway, because the failure it
/// catches is a careless integration, which is the failure that actually
/// happens.
///
/// Scanning source rather than behavior is deliberate. The alternative is
/// booting each surface and inspecting its outbound headers, which tests the one
/// code path someone remembered to exercise; a source scan covers every call
/// site including the one added next week.
#[test]
fn no_mcp_tool_or_cli_subcommand_transmits_the_intent_marker() {
    for crate_name in ["wenlan-mcp", "wenlan-cli"] {
        for (path, body) in crate_sources(crate_name) {
            assert!(
                !body.to_ascii_lowercase().contains(INTENT_HEADER),
                "{} names the reader-intent header. {crate_name} is a \
                 never-transmit surface: there is no human gesture behind an \
                 agent tool call or a scripted subcommand.",
                path.display()
            );
        }
    }
}

/// The other direction. Declaring the truth contract is a claim about
/// *rendering*, and a surface that cannot render two axes must not claim it can
/// -- so the never-transmit surfaces stay off the contract header too, and get
/// legacy treatment, which is exactly right for them.
#[test]
fn no_mcp_tool_or_cli_subcommand_declares_the_truth_contract() {
    for crate_name in ["wenlan-mcp", "wenlan-cli"] {
        for (path, body) in crate_sources(crate_name) {
            assert!(
                !body.to_ascii_lowercase().contains(CONTRACT_HEADER),
                "{} declares the truth contract. {crate_name} renders no truth \
                 axes, so declaring support for them would be a false claim.",
                path.display()
            );
        }
    }
}

// ---- audit outcomes -------------------------------------------------------

#[test]
fn every_decision_maps_to_a_distinct_audit_outcome() {
    let cases = [
        (TruthDecision::Refuse, MarkerOutcome::Refused),
        (
            TruthDecision::Grant(TruthGrant::Automatic),
            MarkerOutcome::Automatic,
        ),
        (
            TruthDecision::Grant(TruthGrant::CollectionEntries),
            MarkerOutcome::GrantedCollection,
        ),
        (
            TruthDecision::Grant(TruthGrant::NamedPages(pages(&["pg1"]))),
            MarkerOutcome::GrantedNamedPage,
        ),
    ];
    let mut seen = std::collections::BTreeSet::new();
    for (decision, expected) in cases {
        assert_eq!(MarkerOutcome::of(&decision), expected);
        assert!(
            seen.insert(expected.as_str()),
            "two outcomes serialize the same"
        );
    }
}

/// Every marker shape the manifest can carry must have an answer here. A new
/// shape added without a resolve arm would be a compile error inside `resolve`;
/// this catches the other half -- a shape that compiles but silently resolves
/// to something inert.
#[test]
fn every_marker_shape_has_a_defined_marked_outcome() {
    for shape in [
        MarkerShape::None,
        MarkerShape::Collection,
        MarkerShape::NamedPage,
        MarkerShape::CollectionAndNamedPage,
    ] {
        let decision = resolve(shape, ClientContract::V1, true, &pages(&["pg1"]));
        match shape {
            MarkerShape::None => assert_eq!(decision, TruthDecision::Refuse),
            _ => assert_ne!(
                decision,
                TruthDecision::Grant(TruthGrant::Automatic),
                "{shape:?} accepted a marker naming a page and granted nothing"
            ),
        }
    }
}
