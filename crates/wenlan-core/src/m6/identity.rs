//! M6 deterministic identity derivations (artifact 4 §3-§5).
//!
//! Three identities, each a pure function of contract-named inputs:
//!
//! * `slot_id` — what a page *is*. Stable across restarts, recomputation, and
//!   root lifecycle.
//! * `page_id` — `m6p_` + a digest of the slot, so slot and page identity can
//!   version independently.
//! * `candidate_fingerprint` — whether a slot's answer is *stale*. This is the
//!   freshness axis, deliberately separate from the identity axis.
//!
//! Every space input is `spaces.id`, the daemon-minted UUID — never the mutable
//! name stored in `communities.space`. Digesting the stable ID is exactly what
//! makes a space rename unable to re-key a slot or a page (S0-161).

use super::constants::{
    DOMAIN_ACTIVE_ROOTS, DOMAIN_CANDIDATE, DOMAIN_FINGERPRINT, DOMAIN_PAGE, DOMAIN_SLOT,
    PAGE_ID_PREFIX, SIGNAL_TAG_COMMUNITY_OVERVIEW, SIGNAL_TAG_EVIDENCE_CLUSTER,
    SIGNAL_TAG_ORPHAN_WIKILINK, SIGNAL_TAG_SPACE_OVERVIEW,
};
use super::digest::{int_part, m6_digest_owned, sorted_set, ABSENT_PART};

/// The four D5 signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SignalKind {
    /// Evidence cluster — keyed on the *initial* independence-group set.
    EvidenceCluster,
    /// Orphan wikilink — keyed on the normalized label.
    OrphanWikilink,
    /// Community overview — keyed on the durable M4 community ID.
    CommunityOverview,
    /// Space overview — keyed on the space alone.
    SpaceOverview,
}

impl SignalKind {
    /// The frozen tag literal. This is a hashed input, so it is read from
    /// [`super::constants`] rather than spelled inline at each call site.
    pub fn tag(self) -> &'static str {
        match self {
            SignalKind::EvidenceCluster => SIGNAL_TAG_EVIDENCE_CLUSTER,
            SignalKind::OrphanWikilink => SIGNAL_TAG_ORPHAN_WIKILINK,
            SignalKind::CommunityOverview => SIGNAL_TAG_COMMUNITY_OVERVIEW,
            SignalKind::SpaceOverview => SIGNAL_TAG_SPACE_OVERVIEW,
        }
    }

    /// Inverse of [`SignalKind::tag`]. `genesis_candidates.signal_kind` stores
    /// the tag, and the CHECK constraint there is what keeps a fifth value from
    /// reaching this — an unknown tag is a contract change, not a value change,
    /// so it returns `None` rather than defaulting to a signal.
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            SIGNAL_TAG_EVIDENCE_CLUSTER => Some(SignalKind::EvidenceCluster),
            SIGNAL_TAG_ORPHAN_WIKILINK => Some(SignalKind::OrphanWikilink),
            SIGNAL_TAG_COMMUNITY_OVERVIEW => Some(SignalKind::CommunityOverview),
            SIGNAL_TAG_SPACE_OVERVIEW => Some(SignalKind::SpaceOverview),
            _ => None,
        }
    }
}

/// `slot_id` for an evidence cluster.
///
/// `groups` is the set snapshotted at the machine-A `observed` transition and
/// never recomputed (S0-29): a slot that tracked the live set would change
/// identity every time the cluster gained evidence, which is backwards. Growth
/// is a *fingerprint* input instead.
pub fn slot_id_evidence_cluster(space_id: &str, groups: &[String]) -> String {
    let mut parts = common_frame(SignalKind::EvidenceCluster, space_id);
    parts.extend(sorted_set(groups));
    m6_digest_owned(DOMAIN_SLOT, &parts)
}

/// `slot_id` for an orphan wikilink. `label_key` must already have passed
/// [`super::label_key::normalize_label_key`].
pub fn slot_id_orphan_wikilink(space_id: &str, label_key: &str) -> String {
    let mut parts = common_frame(SignalKind::OrphanWikilink, space_id);
    parts.push(label_key.as_bytes().to_vec());
    m6_digest_owned(DOMAIN_SLOT, &parts)
}

/// `slot_id` for a community overview.
///
/// Stable exactly as long as M4's rebinding keeps `community_id` stable, and no
/// longer. A split half treated as new gets a fresh UUID, hence a new slot and
/// a new page; machine F's detach rule retires the old one, not a subscription
/// (S0-71).
pub fn slot_id_community_overview(space_id: &str, community_id: &str) -> String {
    let mut parts = common_frame(SignalKind::CommunityOverview, space_id);
    parts.push(community_id.as_bytes().to_vec());
    m6_digest_owned(DOMAIN_SLOT, &parts)
}

/// `slot_id` for a space overview — one per space, empty per-signal vector.
///
/// The empty vector is why the signal tag must be its own part: without it this
/// slot would collide with any other zero-part signal for the same space.
pub fn slot_id_space_overview(space_id: &str) -> String {
    let parts = common_frame(SignalKind::SpaceOverview, space_id);
    m6_digest_owned(DOMAIN_SLOT, &parts)
}

/// `page_id = "m6p_" + m6_digest("m6-page-v1", [slot_id])` (S0-32).
pub fn page_id(slot_id: &str) -> String {
    format!(
        "{PAGE_ID_PREFIX}{}",
        m6_digest_owned(DOMAIN_PAGE, &[slot_id.as_bytes().to_vec()])
    )
}

/// `candidate_id = m6_digest("m6-candidate-v1", [slot_id, coverage_epoch])`
/// (S0-40).
///
/// Depends on nothing a crash or a lease expiry changes, which is what lets a
/// re-prepare after staleness land on the same row rather than colliding with a
/// durable one (machine A, transition A19).
pub fn candidate_id(slot_id: &str, coverage_epoch: i64) -> String {
    m6_digest_owned(
        DOMAIN_CANDIDATE,
        &[slot_id.as_bytes().to_vec(), int_part(coverage_epoch)],
    )
}

/// Fingerprint field 8 — a digest over the `(root_id, status)` pairs.
///
/// Field 6 carries the root IDs alone, so a root going `active -> failed`
/// changes the answer without changing field 6. This field is what notices.
///
/// **Disclosed gap-closure.** Artifact 4 §5 says "`sorted_set` of
/// `(root_id, status)` pairs" without fixing how a pair renders as a set
/// element. PR-A sorts the pairs, dedups them, emits the pair count as one
/// integer part, then emits **two** parts per pair. Fixed arity plus the
/// leading count keeps this unambiguous, and it avoids joining the two fields
/// with a separator — which would reintroduce exactly the collision the
/// length-prefix framing exists to remove.
pub fn active_root_digest(roots: &[(String, String)]) -> String {
    let mut sorted: Vec<(&str, &str)> = roots
        .iter()
        .map(|(id, status)| (id.as_str(), status.as_str()))
        .collect();
    sorted.sort_unstable();
    sorted.dedup();

    let mut parts = Vec::with_capacity(sorted.len() * 2 + 1);
    parts.push(int_part(sorted.len() as i64));
    for (root_id, status) in sorted {
        parts.push(root_id.as_bytes().to_vec());
        parts.push(status.as_bytes().to_vec());
    }

    m6_digest_owned(DOMAIN_ACTIVE_ROOTS, &parts)
}

/// The eleven fixed-arity fingerprint fields (artifact 4 §5).
///
/// Fields 3 and 4 are `Option` because they apply only to evidence-cluster and
/// community-overview signals. They serialize as a zero-length part when
/// absent and are **never omitted** — omitting a part shortens the vector, so
/// an orphan-wikilink fingerprint could align its remaining fields with a
/// community-overview one. Length-prefixing does not save you from a different
/// number of parts; fixed arity does (S0-34).
#[derive(Debug, Clone)]
pub struct CandidateFingerprint<'a> {
    /// 1 — binds the fingerprint to its slot.
    pub slot_id: &'a str,
    /// 2 — per-signal Stage-0 constant; a signal-logic change invalidates every
    /// candidate it produced.
    pub signal_version: i64,
    /// 3 — M4 community ID, absent for the two non-community signals.
    pub community_id: Option<&'a str>,
    /// 4 — M4 published generation, absent alongside field 3.
    pub published_generation: Option<i64>,
    /// 5 — coverage epoch; a contract-version epoch change invalidates all.
    pub coverage_epoch: i64,
    /// 6 — the actual evidence, not just its group summary.
    pub root_ids: &'a [String],
    /// 7 — the M4 lease/CAS generation the prepare read under.
    pub input_generation: i64,
    /// 8 — see [`active_root_digest`].
    pub active_root_digest: &'a str,
    /// 9 — the pinned LLM identifier; a model swap must re-derive.
    pub model_version: &'a str,
    /// 10 — `communities.projection_version`, M4's own re-derivation axis.
    pub projection_version: &'a str,
    /// 11 — opaque monotone prompt tag. D13 forbids the prompt *text* from
    /// entering anything durable, so a version tag is the only legal carrier
    /// (S0-33).
    pub prompt_version: &'a str,
}

impl CandidateFingerprint<'_> {
    /// `m6_digest("m6-fingerprint-v1", [11 fields])`.
    pub fn digest(&self) -> String {
        // Field order is the contract. Fields 1-5 are fixed-arity scalars, 6 is
        // a variable-length set, 7-11 are fixed-arity scalars again.
        let mut parts: Vec<Vec<u8>> = vec![
            self.slot_id.as_bytes().to_vec(), // 1
            int_part(self.signal_version),    // 2
            match self.community_id {
                Some(id) => id.as_bytes().to_vec(),
                None => ABSENT_PART.to_vec(),
            }, // 3
            match self.published_generation {
                Some(generation) => int_part(generation),
                None => ABSENT_PART.to_vec(),
            }, // 4
            int_part(self.coverage_epoch),    // 5
        ];

        parts.extend(sorted_set(self.root_ids)); // 6
        parts.push(int_part(self.input_generation)); // 7
        parts.push(self.active_root_digest.as_bytes().to_vec()); // 8
        parts.push(self.model_version.as_bytes().to_vec()); // 9
        parts.push(self.projection_version.as_bytes().to_vec()); // 10
        parts.push(self.prompt_version.as_bytes().to_vec()); // 11

        m6_digest_owned(DOMAIN_FINGERPRINT, &parts)
    }
}

/// `[signal_tag, space]` — the frame every slot shares (artifact 4 §3.1).
fn common_frame(signal: SignalKind, space_id: &str) -> Vec<Vec<u8>> {
    vec![
        signal.tag().as_bytes().to_vec(),
        space_id.as_bytes().to_vec(),
    ]
}
