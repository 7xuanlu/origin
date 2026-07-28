// SPDX-License-Identifier: Apache-2.0
//! Provenance root identity & independence grouping (KG unified-model M2,
//! spec v3 §1, Q6-locked recipe).
//!
//! Pure, deterministic helpers for `provenance_roots` -- the content-addressed
//! substrate `edges.root_id` references. Two concerns, cleanly split per the
//! Q6 decision draft:
//!
//! - **Identity** ([`identity_digest`]): exact/near-exact content equivalence,
//!   deterministic and replica-convergent -- nothing semantic. Source-instance
//!   identity (file path, fetch time, importing agent) never enters this, so
//!   two byte-identical imports converge on one root (spec §1, §6.7).
//! - **Independence** ([`base_independence_key`], [`content_minhash_bands`]):
//!   fuzzy-mergeable grouping by source signal, with a mandatory near-dup
//!   MinHash/LSH overlay reusing `retrieval::dedup` (T16), re-pointed from
//!   entity names to root content. The overlay itself (band-table lookup,
//!   union into an existing group) needs DB access and lives in `db.rs`,
//!   mirroring the existing split between this crate's pure `dedup.rs` and
//!   its `db.rs` accessors (`minhash_resolve_candidate` et al.).
//!
//! `edge_id` content-addressing ([`compute_edge_id`]) lives here too -- it is
//! the same "pure deterministic identity hash" concern, just for edges
//! instead of roots (assignment-matrix "Retry identity" section,
//! `docs/plans/2026-07-21-m2-edge-assignment-matrix.md`).

use crate::retrieval::dedup;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

/// Q6-locked: the root digest's version tag. Bump only alongside a
/// deliberate canonicalization-recipe change -- a version bump changes every
/// digest, so old and new roots never collide and replica convergence never
/// silently breaks across a canonicalization change (mirrors spec §6.6's "a
/// model upgrade re-derives before it re-judges", applied to content
/// identity instead of model scores).
pub const IDENTITY_VERSION: i64 = 1;

/// Character shingle width for content near-dup detection. Wider than the
/// entity-name default (`SHINGLE_K = 3` in `retrieval::dedup`) because
/// content is much longer than a name -- a k=3 shingle set over a whole
/// document is dominated by common short substrings and would over-match.
const CONTENT_SHINGLE_K: usize = 8;

/// Q6-locked canonicalization: Unicode NFC + line-ending normalization +
/// trailing/collapsed-whitespace trim. **Nothing semantic** -- this is what
/// keeps replica convergence trivially safe; a whitespace-tweaked mirror is a
/// DISTINCT root (re-unified only by the independence-group near-dup
/// overlay, never by identity itself).
pub fn canonicalize_content(raw: &str) -> String {
    // Line-ending normalization first (CRLF/CR -> LF) so NFC never has to
    // reason about carriage returns.
    let unix_lines = raw.replace("\r\n", "\n").replace('\r', "\n");
    // Unicode NFC.
    let nfc: String = unix_lines.nfc().collect();
    // Per-line trailing-whitespace trim + collapse of internal whitespace
    // runs to a single space, then trim the whole document. `split_whitespace`
    // already drops leading/trailing whitespace per line as a side effect.
    let collapsed: String = nfc
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n");
    collapsed.trim().to_string()
}

/// `canonical_content_digest` -- the content-only half of the root digest
/// (spec §1: "hash(identity_version, root_kind, canonical_content_digest)").
/// Exposed separately from [`identity_digest`] because it is the honest
/// canonicalized counterpart to `memories.content_hash` (migration 65's
/// whole-file SHA-256 over raw bytes, which does NOT canonicalize).
pub fn canonical_content_digest(raw_content: &str) -> String {
    let canonical = canonicalize_content(raw_content);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// SHA-256 over the **exact bytes** of a page revision's content -- the
/// `base_content_digest` a human save binds itself to (D4; presence threat
/// model §3 "base/revision digests: exact viewed content", T6).
///
/// Deliberately NOT [`canonical_content_digest`]. The two answer different
/// questions and must not be swapped:
///
/// - canonical asks *"is this the same content?"* and is whitespace-tolerant on
///   purpose, so a reformatted mirror converges on one identity;
/// - this asks *"is this the exact text the human was looking at?"* and must be
///   whitespace-INtolerant, because the human-edit delta is computed by
///   comparing lines. A reflow that canonical would forgive changes every line
///   boundary, so forgiving it would attribute the reflowed text to the human
///   as new prose.
pub fn revision_content_digest(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Root digest = `hash(identity_version, root_kind, canonical_content_digest)`
/// (Q6, spec §1). Source-instance identity is excluded, so two
/// byte-identical imports converge on one root via `INSERT ... ON CONFLICT
/// ... RETURNING` (§6.7) -- same content, same root, regardless of file path,
/// fetch time, or importing agent.
pub fn identity_digest(root_kind: &str, raw_content: &str) -> String {
    let content_digest = canonical_content_digest(raw_content);
    let mut hasher = Sha256::new();
    hasher.update(IDENTITY_VERSION.to_le_bytes());
    hasher.update(b":");
    hasher.update(root_kind.as_bytes());
    hasher.update(b":");
    hasher.update(content_digest.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Content near-dup shingle set, reusing `retrieval::dedup::char_shingles`
/// re-pointed from entity names ([`CONTENT_SHINGLE_K`] instead of the
/// name-tuned trigram width).
pub fn content_shingles(canonical_content: &str) -> HashSet<String> {
    dedup::char_shingles(canonical_content, CONTENT_SHINGLE_K)
}

/// LSH band keys for a root's canonicalized content -- the same
/// MinHash-signature-then-band pipeline `retrieval::dedup` already uses for
/// entity names, applied to content shingles instead. The DB-side overlay
/// (persisting bands, looking up candidates, confirming with exact Jaccard)
/// lives in `db.rs` alongside the existing entity-minhash accessors.
pub fn content_minhash_bands(canonical_content: &str) -> Vec<u64> {
    let shingles = content_shingles(canonical_content);
    let sig = dedup::minhash_signature(&shingles);
    dedup::lsh_bands(&sig)
}

/// Q6-locked near-dup threshold, reused verbatim from `retrieval::dedup`
/// ("high threshold = near-identical only, never topical").
pub const CONTENT_NEAR_DUP_THRESHOLD: f64 = dedup::FUZZY_JACCARD_THRESHOLD;

/// Exact-Jaccard confirmation between two roots' content, reusing
/// `retrieval::dedup::jaccard` over content shingles rather than name
/// shingles. Callers pass already-canonicalized content (the DB layer
/// canonicalizes once per root and reuses that string for both the identity
/// digest and this overlay, rather than canonicalizing twice).
pub fn content_jaccard(canonical_a: &str, canonical_b: &str) -> f64 {
    dedup::jaccard(
        &content_shingles(canonical_a),
        &content_shingles(canonical_b),
    )
}

/// The four independence signals in Q6's precedence order (spec Q6 decision
/// draft, part B): distinct external source identity > agent turn/session >
/// import batch/container (fallback only when per-item source identity is
/// absent). The near-dup MinHash overlay is a separate, DB-backed union step
/// (`db.rs`) applied on top of this base key -- it is not a signal an
/// implementer of this pure function can see.
pub struct IndependenceSignals<'a> {
    /// Distinct external source identity: URL/domain, distinct file origin,
    /// or distinct author. Highest precedence.
    pub source_identity: Option<&'a str>,
    /// Agent turn/session id, for generated or captured content. Second
    /// precedence -- used only when `source_identity` is absent.
    pub agent_turn: Option<&'a str>,
    /// Import batch/container id. Fallback only, used only when both of the
    /// above are absent.
    pub import_batch: Option<&'a str>,
}

/// Base independence-group key by Q6's precedence rule. Returns `None` when
/// no signal is establishable -- per Q6 B.4 ("un-establishable independence
/// routes to human review, never auto-genesis"), a `None` here is a signal
/// to the caller to route to human review rather than mint a group.
pub fn base_independence_key(signals: &IndependenceSignals) -> Option<String> {
    signals
        .source_identity
        .map(|s| format!("src:{s}"))
        .or_else(|| signals.agent_turn.map(|s| format!("turn:{s}")))
        .or_else(|| signals.import_batch.map(|s| format!("batch:{s}")))
}

/// Content-addressed `edge_id` (assignment-matrix "Retry identity" section):
/// `sha256(edge_type, src_kind, src_id, dst_kind, dst_id, discriminator)`.
/// A retried write recomputes the same `edge_id`; `INSERT ... ON
/// CONFLICT(edge_id) DO NOTHING` converges on the one edge -- no duplicate
/// voter, exactly the §6.1 retry-identity guarantee at the edge grain.
///
/// Each part is **length-prefixed** (its byte length as a u64 LE, then the
/// bytes) so no two distinct tuples can collide by concatenation ambiguity.
/// A bare NUL *separator* is not enough: a part may itself contain a NUL byte
/// (Rust `&str` is arbitrary UTF-8, and locators/labels are unvalidated), so
/// `("x\0", "y")` and `("x", "\0y")` would hash identically under a
/// separator-only scheme. Length-prefixing removes the separator's meaning
/// entirely -- the boundary is the declared length, not a sentinel byte.
pub fn compute_edge_id(
    edge_type: &str,
    src_kind: &str,
    src_id: &str,
    dst_kind: &str,
    dst_id: &str,
    discriminator: &str,
) -> String {
    let mut hasher = Sha256::new();
    for part in [edge_type, src_kind, src_id, dst_kind, dst_id, discriminator] {
        let bytes = part.as_bytes();
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_normalizes_line_endings() {
        assert_eq!(canonicalize_content("a\r\nb\rc\nd"), "a\nb\nc\nd");
    }

    #[test]
    fn canonicalize_collapses_and_trims_whitespace() {
        assert_eq!(
            canonicalize_content("  hello   world  \n\n  foo\tbar  "),
            "hello world\n\nfoo bar"
        );
    }

    #[test]
    fn canonicalize_applies_nfc() {
        // "e" + combining acute accent (NFD) canonicalizes to precomposed "é" (NFC).
        let nfd = "e\u{0301}";
        let nfc = "\u{00e9}";
        assert_eq!(canonicalize_content(nfd), canonicalize_content(nfc));
    }

    #[test]
    fn identity_digest_is_deterministic() {
        let a = identity_digest("document_ingest", "hello world");
        let b = identity_digest("document_ingest", "hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn identity_digest_same_content_different_whitespace_converges() {
        // Byte-identical-after-canonicalization content must produce the
        // same digest (source instance excluded, spec §1).
        let a = identity_digest("document_ingest", "hello   world");
        let b = identity_digest("document_ingest", "hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn identity_digest_differs_by_root_kind() {
        // A byte-identical human_capture and document_ingest must NOT
        // converge -- root_kind is part of the digest input.
        let a = identity_digest("document_ingest", "hello world");
        let b = identity_digest("human_capture", "hello world");
        assert_ne!(a, b);
    }

    #[test]
    fn identity_digest_differs_by_content() {
        let a = identity_digest("document_ingest", "hello world");
        let b = identity_digest("document_ingest", "goodbye world");
        assert_ne!(a, b);
    }

    #[test]
    fn identity_digest_formatting_mirror_is_distinct_from_source_root() {
        // A formatting-tweaked mirror (adds semantic markup, not just
        // whitespace) is a DISTINCT root by identity -- re-unified only by
        // the independence-group near-dup overlay, never by the digest
        // itself (Q6 equivalence matrix row 2).
        let a = identity_digest("document_ingest", "# Title\n\nBody text.");
        let b = identity_digest("document_ingest", "Title\n\nBody text.");
        assert_ne!(a, b);
    }

    #[test]
    fn base_independence_key_precedence_source_over_turn_over_batch() {
        let all_three = IndependenceSignals {
            source_identity: Some("https://example.com/doc"),
            agent_turn: Some("turn-1"),
            import_batch: Some("batch-1"),
        };
        assert_eq!(
            base_independence_key(&all_three),
            Some("src:https://example.com/doc".to_string())
        );

        let turn_and_batch = IndependenceSignals {
            source_identity: None,
            agent_turn: Some("turn-1"),
            import_batch: Some("batch-1"),
        };
        assert_eq!(
            base_independence_key(&turn_and_batch),
            Some("turn:turn-1".to_string())
        );

        let batch_only = IndependenceSignals {
            source_identity: None,
            agent_turn: None,
            import_batch: Some("batch-1"),
        };
        assert_eq!(
            base_independence_key(&batch_only),
            Some("batch:batch-1".to_string())
        );
    }

    #[test]
    fn base_independence_key_none_when_unestablishable() {
        let none = IndependenceSignals {
            source_identity: None,
            agent_turn: None,
            import_batch: None,
        };
        assert_eq!(base_independence_key(&none), None);
    }

    #[test]
    fn content_jaccard_near_identical_above_threshold() {
        // b = a with a short suffix appended: every shingle of `a` is also
        // a shingle of `b` at the same offset (b has a as a verbatim
        // prefix), so intersection = |shingles(a)| exactly and the only new
        // shingles come from the short suffix + boundary. `a` is long,
        // non-repeating prose (a repeated short phrase would collapse to a
        // handful of distinct shingles via the HashSet dedup, understating
        // the shared-shingle count), so the boundary addition is a small
        // fraction of the total -- a near-dup mirror, precisely bounded
        // rather than hand-guessed.
        let a = "the annual harvest festival draws visitors from every \
                 nearby town to admire the painted carts, sample the \
                 preserved fruits, and watch the evening lantern parade \
                 wind slowly through the cobblestone streets past the old \
                 mill and the riverside market stalls selling wool, honey, \
                 and hand carved wooden toys to children waiting eagerly \
                 near the fountain square";
        let b = format!("{a} and then everyone walked home together");
        assert!(content_jaccard(a, &b) >= CONTENT_NEAR_DUP_THRESHOLD);
    }

    #[test]
    fn content_jaccard_unrelated_below_threshold() {
        let a = "the quick brown fox jumps over the lazy dog";
        let b = "completely different subject matter about astronomy";
        assert!(content_jaccard(a, b) < CONTENT_NEAR_DUP_THRESHOLD);
    }

    #[test]
    fn compute_edge_id_is_deterministic_and_retry_safe() {
        let a = compute_edge_id("relates", "entity", "e1", "entity", "e2", "friend_of");
        let b = compute_edge_id("relates", "entity", "e1", "entity", "e2", "friend_of");
        assert_eq!(a, b);
    }

    #[test]
    fn compute_edge_id_differs_by_discriminator() {
        let a = compute_edge_id("relates", "entity", "e1", "entity", "e2", "friend_of");
        let b = compute_edge_id("relates", "entity", "e1", "entity", "e2", "enemy_of");
        assert_ne!(a, b);
    }

    #[test]
    fn compute_edge_id_no_concatenation_ambiguity() {
        // ("ab", "c") vs ("a", "bc") must not collide despite concatenating
        // to the same string without a separator.
        let a = compute_edge_id("cites", "page", "ab", "memory", "c", "loc");
        let b = compute_edge_id("cites", "page", "a", "memory", "bc", "loc");
        assert_ne!(a, b);
    }

    #[test]
    fn compute_edge_id_embedded_nul_no_collision() {
        // A part may itself contain a NUL byte, so a NUL *separator* alone
        // would let ("x\0", "y") and ("x", "\0y") hash identically. The
        // length-prefix must keep them distinct.
        let a = compute_edge_id("cites", "page", "p", "memory", "x\0", "y");
        let b = compute_edge_id("cites", "page", "p", "memory", "x", "\0y");
        assert_ne!(
            a, b,
            "an embedded NUL must not collide with a shifted split -- length-prefix, not separator"
        );
    }
}
