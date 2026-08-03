//! Frozen M6 constants that PR-A consumes.
//!
//! Every value here is fixed by a Stage-0 contract artifact and cited to its
//! decision. Changing one is a contract amendment, not a tuning knob — the
//! digest domains and signal tags in particular are hashed inputs, so editing a
//! byte re-keys every `slot_id`, hence every `page_id` and `candidate_id`.
//!
//! Constants belonging to later rungs (D9 relevance weights, D8 admission
//! thresholds, lease TTLs) are deliberately not here: PR-A has no code that
//! reads them, and a constant with no consumer is a value nothing checks.

// ---------------------------------------------------------------------------
// Digest domain tags (artifact 4)
// ---------------------------------------------------------------------------

/// Domain tag for `slot_id` (artifact 4 §3.1).
pub const DOMAIN_SLOT: &str = "m6-slot-v1";

/// Domain tag for `page_id` (artifact 4 §4, S0-32).
pub const DOMAIN_PAGE: &str = "m6-page-v1";

/// Domain tag for `candidate_id` (S0-40).
pub const DOMAIN_CANDIDATE: &str = "m6-candidate-v1";

/// Domain tag for `candidate_fingerprint` (artifact 4 §5).
pub const DOMAIN_FINGERPRINT: &str = "m6-fingerprint-v1";

/// Domain tag for the active-root digest, fingerprint field 8.
///
/// **Disclosed gap-closure.** Artifact 4 §5 specifies field 8 as "a `m6_digest`
/// over `sorted_set` of `(root_id, status)` pairs" but never names its domain
/// tag. PR-A picks this spelling to match the `m6-<thing>-v1` family. Nothing
/// downstream has been built against another value.
pub const DOMAIN_ACTIVE_ROOTS: &str = "m6-active-roots-v1";

/// `page_id` prefix (S0-32). The full 64-hex digest follows and is never
/// truncated — the orphan-wikilink slot derives from an attacker-written label,
/// so a 64-bit ID would be grindable at ~2^32 work.
pub const PAGE_ID_PREFIX: &str = "m6p_";

// ---------------------------------------------------------------------------
// Signal tags (artifact 4 §3.1) — frozen because they are digest inputs
// ---------------------------------------------------------------------------

/// Signal tag literal (artifact 4 §3.1).
pub const SIGNAL_TAG_EVIDENCE_CLUSTER: &str = "evidence-cluster";
/// See [`SIGNAL_TAG_EVIDENCE_CLUSTER`].
pub const SIGNAL_TAG_ORPHAN_WIKILINK: &str = "orphan-wikilink";
/// See [`SIGNAL_TAG_EVIDENCE_CLUSTER`].
pub const SIGNAL_TAG_COMMUNITY_OVERVIEW: &str = "community-overview";
/// See [`SIGNAL_TAG_EVIDENCE_CLUSTER`].
pub const SIGNAL_TAG_SPACE_OVERVIEW: &str = "space-overview";

// ---------------------------------------------------------------------------
// D8 wikilink normalization bounds (artifact 4 §6)
// ---------------------------------------------------------------------------

/// R0 pre-cap: reject a raw target longer than this many Unicode scalars,
/// before any normalization work (S0-37). 8x the post-normalization cap.
pub const LABEL_RAW_PRECAP_SCALARS: usize = 1024;

/// R3 lower bound, measured after normalization.
pub const LABEL_KEY_MIN_SCALARS: usize = 1;

/// R3 upper bound, measured after normalization (artifact 4 §6.1 step 7).
pub const LABEL_KEY_MAX_SCALARS: usize = 128;

// ---------------------------------------------------------------------------
// D8 hard caps (artifact 4 §6, spec §3.6) — enforced at the read that
// produces the bounded set, never by truncating after the fact.
// ---------------------------------------------------------------------------

/// Roots per candidate. A signal's supporting root set is bounded with a
/// SQL `LIMIT` on the query that produces it, not by collecting every root
/// and truncating the `Vec` afterward (§3.6: caps are enforced at the read).
/// What happens to the excess (staying frontier-visible rather than lost,
/// §3.6's "no cap may terminalize") is machine F's concern, not this
/// reader's — PR-B1 makes no frontier writes.
pub const ROOTS_PER_CANDIDATE_CAP: usize = 64;
