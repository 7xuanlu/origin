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

/// Pending (non-terminal) candidates per space (§3.6, artifact 5 §7). A hit
/// refuses to *start* new work and writes nothing — S0-54 forbids a cap from
/// terminalizing a candidate or retiring a frontier row.
pub const PENDING_CANDIDATES_PER_SPACE_CAP: usize = 128;

/// Candidate prepares per space per cycle (§3.6). Same S0-54 semantics as
/// [`PENDING_CANDIDATES_PER_SPACE_CAP`]: the 17th group keeps its frontier row.
pub const CANDIDATE_PREPARES_PER_CYCLE_CAP: usize = 16;

// ---------------------------------------------------------------------------
// D6 lease TTLs (S0-3), in seconds
// ---------------------------------------------------------------------------

/// `genesis` phase TTL. Spans one inference plus one entailment check in the
/// eventual real path; vacuously long in the shadow, which makes no model call.
/// S0-3's binding rule (TTL > model timeout + finalize budget) is still
/// un-re-derived — spec §11 Q10.
pub const GENESIS_LEASE_TTL_SECONDS: i64 = 900;

/// `frontier` phase TTL. Shortest of the four: the scan is a pure differential
/// query, so a crashed scan should be retryable within one refinery interval.
pub const FRONTIER_LEASE_TTL_SECONDS: i64 = 120;

// ---------------------------------------------------------------------------
// S0-2 retry backoff and S0-151 stale re-entry delays, in seconds
// ---------------------------------------------------------------------------

/// `attempt` strictly greater than this is exhaustion (S0-2/S0-12): the
/// candidate lands in `stale` carrying `retry_exhausted`, never an 11th state.
pub const RETRY_ATTEMPT_LIMIT: i64 = 5;

/// `delay = 60s * 4^(attempt-1)`, **no jitter** (S0-2). One daemon per data
/// root means there is no herd to disperse, and a deterministic schedule is
/// what lets a crash matrix assert a specific next-attempt time.
pub const RETRY_BASE_DELAY_SECONDS: i64 = 60;

/// The S0-2 backoff ceiling. The sequence is 60s, 4m, 16m, 64m, 4h.
pub const RETRY_MAX_DELAY_SECONDS: i64 = 4 * 3_600;

/// S0-151 delay for `reason = 'retry_exhausted'`: five model attempts were
/// already burned, so a day is the cheapest delay that is obviously not a loop.
pub const STALE_RETRY_EXHAUSTED_DELAY_SECONDS: i64 = 86_400;

/// S0-151 delay for `reason = 'card_expired'`, matching F7's suppression
/// window: a human saw the card and let it lapse.
pub const STALE_CARD_EXPIRED_DELAY_SECONDS: i64 = 180 * 86_400;

// ---------------------------------------------------------------------------
// D7 frontier clocks and bounds, in seconds unless noted
// ---------------------------------------------------------------------------

/// S0-46: `next_scan_at` may never be set more than 24h ahead of
/// `unixepoch()`, so the 7-day below-floor timer is evaluated at least seven
/// times before it can fire and no path can defer a row past its own deadline.
pub const NEXT_SCAN_MAX_AHEAD_SECONDS: i64 = 86_400;

/// F3: below-floor evidence older than this surfaces one coalesced
/// unformed-topic card per `(space, coverage_epoch)` (S0-47).
pub const BELOW_FLOOR_SURFACING_SECONDS: i64 = 7 * 86_400;

/// S0-10/S0-53: a terminal candidate's `payload` is nulled after this long and
/// `payload_compacted_at` stamped. The row itself is never deleted.
pub const PAYLOAD_COMPACTION_SECONDS: i64 = 90 * 86_400;

/// Rows examined by one frontier reconciliation slice (spec §4.2). A cap on
/// work per turn, never on what stays accounted for.
pub const FRONTIER_SCAN_SLICE_ROWS: usize = 512;

// ---------------------------------------------------------------------------
// Unformed-topic card identity (artifact 5 §3.2)
// ---------------------------------------------------------------------------

/// Digest domain for the space component of an unformed-topic card ID.
///
/// **Disclosed gap-closure.** Artifact 5 §3.2 fixes the card ID as
/// `m6_unformed_topic_<space-digest>_<epoch>` and says the space digest is "an
/// `m6_digest`" so a space name never appears in an ID, but names no domain
/// tag. PR-B2 picks this spelling to match the `m6-<thing>-v1` family. Nothing
/// downstream has been built against another value.
pub const DOMAIN_CARD_SPACE: &str = "m6-card-space-v1";

/// Unformed-topic card ID prefix (artifact 5 §3.2).
pub const CARD_ID_PREFIX: &str = "m6_unformed_topic_";
