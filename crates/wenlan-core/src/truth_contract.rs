// SPDX-License-Identifier: Apache-2.0
//! What one call is allowed to see of an unsupported page.
//!
//! D3 splits the question in two, and the split is the whole design:
//!
//! | Gate | Question | Enforcement |
//! |---|---|---|
//! | route shape | may a marker do anything here, and what? | server-side, hard. `None` **refuses**. |
//! | marker authenticity | did a human actually gesture? | cooperative. The daemon cannot tell forged from real. |
//!
//! The shape gate needs no client cooperation, so it holds against a careless
//! integration -- an MCP tool wired to send the marker still gets nothing out of
//! `/api/context`, `/api/search`, or an export, because those routes refuse it
//! whoever is calling. It does **not** bound total exposure: `Collection` and
//! `NamedPage` compose, so a caller willing to forge the marker can enumerate
//! provisional IDs and then fetch each one. That attacker is conceded (artifact
//! 5 section 1, T11, hostile same-user); the durable audit row is what makes the
//! pattern visible rather than invisible, and it is the only compensating
//! control there is.
//!
//! Two signals, both required, deliberately separate:
//!
//! - the **contract version** says the client renders both truth axes. It is a
//!   property of the client.
//! - the **intent marker** says a human named a page on *this* call. It is a
//!   property of the call.
//!
//! Neither implies the other, which is the point. `SpaceList.tsx:76` polls
//! `GET /api/pages` every 10s from a fully M5-aware app; that poll is a machine
//! read and gets automatic behavior. Reading intent off the route instead --
//! which two drafts did, in opposite directions -- cannot express that.
//!
//! Binding spec: `docs/plans/2026-07-27-m5-reader-manifest.md` sections 4 and 6.

use crate::truth_manifest::MarkerShape;

/// The newest truth-contract version this daemon serves.
///
/// A client declaring a *higher* version is treated as legacy. Trust contracts
/// do not negotiate upward: a version we have never heard of tells us nothing
/// about what that client renders, and guessing in the permissive direction is
/// exactly the unearned trust this rung exists to prevent.
pub const TRUTH_CONTRACT_VERSION: u32 = 1;

/// Header a client sets to declare it renders both truth axes.
pub const CONTRACT_HEADER: &str = "x-wenlan-truth-contract";

/// Header carrying the per-call human-intent marker.
pub const INTENT_HEADER: &str = "x-wenlan-reader-intent";

/// The only value [`INTENT_HEADER`] accepts. Anything else is not the marker.
pub const INTENT_MARKER_VALUE: &str = "explicit";

/// What the client declared about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientContract {
    /// Absent, malformed, unknown, or newer than we serve. All four collapse to
    /// one state on purpose: each means we cannot know the client renders both
    /// axes, and the response is the same in every case.
    Legacy,
    /// Declared support for contract version 1.
    V1,
}

/// Parse a declared contract version.
///
/// Everything that is not exactly a version this daemon serves is `Legacy`.
pub fn parse_contract(raw: Option<&str>) -> ClientContract {
    match raw.map(str::trim).and_then(|v| v.parse::<u32>().ok()) {
        Some(1) => ClientContract::V1,
        _ => ClientContract::Legacy,
    }
}

/// Did this call carry the human-intent marker?
///
/// A malformed value counts as *absent*, not as a marker. Absent is the
/// fail-closed direction -- it yields automatic behavior -- so garbage in the
/// header can only ever under-expose.
pub fn marker_present(raw: Option<&str>) -> bool {
    raw.map(str::trim)
        .is_some_and(|v| v.eq_ignore_ascii_case(INTENT_MARKER_VALUE))
}

/// What this call may see of unsupported pages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TruthGrant {
    /// Unsupported pages excluded entirely -- not the title, not a summary, not
    /// prose, not an excerpt, not an ID with content hanging off it. The default
    /// for every call, and what the overwhelming majority of calls get.
    Automatic,
    /// Unsupported pages may be *listed*: page ID, title, and both truth axes
    /// per entry. Never prose.
    ///
    /// The carve-out exists because named-page-only is a dead end after
    /// migration -- every page is unsupported, lists exclude unsupported, and a
    /// human cannot name a page they cannot see. The per-item axes are its
    /// precondition, not a nicety: an entry that appears without its state is
    /// the unearned trust this rung exists to prevent.
    CollectionEntries,
    /// Full prose, both axes, for exactly these page IDs -- the ones the call
    /// itself named in its path. Never for a page that merely rides along in the
    /// payload; a grant for page A does not cover page B's title in A's links.
    NamedPages(Vec<String>),
}

/// The guard's verdict for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TruthDecision {
    Grant(TruthGrant),
    /// A marker arrived at a route where it may do nothing.
    ///
    /// Refusing rather than ignoring is deliberate. An ignored marker is a
    /// wiring mistake that behaves correctly today and silently wrong after a
    /// refactor; a refused one fails at the first integration test.
    Refuse,
}

/// Resolve one call's grant from the route's shape, the client's declaration,
/// and whether a human gestured.
///
/// `named_pages` is what the *call* named -- path parameters, not a
/// client-supplied list. A client-supplied list could disagree with the path,
/// and then there would be two answers to "which page did this call name".
pub fn resolve(
    shape: MarkerShape,
    contract: ClientContract,
    marker: bool,
    named_pages: &[String],
) -> TruthDecision {
    // No marker, no grant -- whatever the client declared about itself. This is
    // the 10-second poll from a fully M5-aware app.
    if !marker {
        return TruthDecision::Grant(TruthGrant::Automatic);
    }

    // The hard gate, and the only one that needs no cooperation. Checked before
    // the contract because it is a fact about the route, not about the caller: a
    // legacy client pointing a marker at `/api/context` is the same wiring
    // mistake as an M5-aware one doing it.
    if shape == MarkerShape::None {
        return TruthDecision::Refuse;
    }

    // Declaring the contract is necessary and not sufficient; the marker above
    // is the other half. Undeclared means we do not know the client renders both
    // axes, so it sees no unsupported content however hard it gestures.
    if contract == ClientContract::Legacy {
        return TruthDecision::Grant(TruthGrant::Automatic);
    }

    let named = || {
        if named_pages.is_empty() {
            // A marker that names no page grants nothing. This is what stops the
            // marker from working as a blanket header: spray it everywhere and
            // it is structurally inert on every route that names no page.
            TruthGrant::Automatic
        } else {
            TruthGrant::NamedPages(named_pages.to_vec())
        }
    };

    TruthDecision::Grant(match shape {
        // Refused above. Spelled out rather than `unreachable!` so that
        // weakening the refusal into "ignore it" -- the exact mutation section 9
        // names -- degrades to automatic instead of panicking, and the test that
        // catches it reports a wrong answer rather than a crash.
        MarkerShape::None => TruthGrant::Automatic,
        MarkerShape::Collection => TruthGrant::CollectionEntries,
        MarkerShape::NamedPage => named(),
        // Bare it lists, with a page it opens. Only `wenlan pages` carries this
        // shape, and its real enforcement is the projection-directory invariant;
        // the arm exists so the wire path is total rather than panicking.
        MarkerShape::CollectionAndNamedPage => {
            if named_pages.is_empty() {
                TruthGrant::CollectionEntries
            } else {
                named()
            }
        }
    })
}

/// How much of one page this call may see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Everything the reader would have returned before M5.
    Full,
    /// The page exists, and here is its state -- ID, title, both axes. No prose,
    /// no summary, no excerpt.
    EntryOnly,
    /// The page is not in the response at all. Not its title, not an ID with
    /// content hanging off it.
    Hidden,
}

/// What the evidence says about a page's prose.
///
/// Three states, not a bool, because "we looked and the evidence does not back
/// this" and "nobody has ever looked" are different claims and only the first
/// is a verdict. A bool collapses them, and the collapse is not neutral: every
/// page predating claim derivation is backfilled unjudged, so reading unjudged
/// as unsupported would empty a whole vault on the day the generation advances
/// -- 511 pages on the author's own machine -- and call it enforcement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Judged, and the cited evidence backs the prose.
    Supported,
    /// Judged, and it does not. The only state that costs a page its file.
    Unsupported,
    /// Never judged. Not a verdict, and never treated as one.
    Unevaluated,
}

/// The one decision every adapter makes, for one page.
///
/// Pure on purpose: an adapter that has resolved its grant and looked up a
/// page's support status has everything it needs, so the rule lives in one
/// testable place instead of being re-derived at 79 call sites.
///
/// `generation` is the durable cutover generation. **`0` means every adapter is
/// pass-through**, which is the entire reason PR-B changes no behavior: the
/// adapters are installed and mutation-tested against a generation the tests
/// advance themselves, while production stays at 0 until PR-C's fenced
/// ceremony. Any weakening of this line makes PR-B a live cutover.
///
/// `human_reviewed` outranks the evidence. It is set only by a person approving
/// a specific page version against a specific digest, and a machine judgment
/// does not get to overturn that. The column has existed since migration 98 and
/// was read into [`crate::db::PageTruth`] and then dropped on the floor -- no
/// visibility or eviction path consulted it -- so a page its owner had
/// personally approved was archived exactly like one nobody had ever seen.
pub fn visible_at(
    generation: i64,
    grant: &TruthGrant,
    page_id: &str,
    support: Support,
    human_reviewed: bool,
) -> Visibility {
    if generation == 0 {
        return Visibility::Full;
    }
    if human_reviewed {
        return Visibility::Full;
    }
    match support {
        Support::Supported | Support::Unevaluated => return Visibility::Full,
        Support::Unsupported => {}
    }
    match grant {
        TruthGrant::Automatic => Visibility::Hidden,
        TruthGrant::CollectionEntries => Visibility::EntryOnly,
        // The grant covers the page the call named and nothing riding along
        // beside it. A linked page's title in page A's payload is page B's
        // title, and page B was never named.
        TruthGrant::NamedPages(ids) => {
            if ids.iter().any(|id| id == page_id) {
                Visibility::Full
            } else {
                Visibility::Hidden
            }
        }
    }
}

/// How a marked call came out, for the durable audit row.
///
/// Refusals are recorded alongside grants. The tooth only requires granted
/// calls, but a refusal is the signature of a misconfigured integration and
/// costs one row to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerOutcome {
    Refused,
    /// Marked, shaped, but the client had not declared the contract -- or named
    /// no page. The marker did nothing.
    Automatic,
    GrantedCollection,
    GrantedNamedPage,
}

impl MarkerOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Refused => "refused",
            Self::Automatic => "automatic",
            Self::GrantedCollection => "granted_collection",
            Self::GrantedNamedPage => "granted_named_page",
        }
    }

    /// The outcome a decision records.
    pub fn of(decision: &TruthDecision) -> Self {
        match decision {
            TruthDecision::Refuse => Self::Refused,
            TruthDecision::Grant(TruthGrant::Automatic) => Self::Automatic,
            TruthDecision::Grant(TruthGrant::CollectionEntries) => Self::GrantedCollection,
            TruthDecision::Grant(TruthGrant::NamedPages(_)) => Self::GrantedNamedPage,
        }
    }
}

#[cfg(test)]
#[path = "truth_contract_test.rs"]
mod tests;
