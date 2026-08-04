// SPDX-License-Identifier: Apache-2.0
//! The incremental-vs-full recomputation oracle (spec §6.1).
//!
//! [`recompute_full`] derives genesis state for a space **from primary
//! evidence only** — edges, roots, pages, communities — never from a
//! `genesis_*` row. [`snapshot_incremental`] reads the durable `genesis_*`
//! rows and nothing else. [`diff`] compares the two. Any disagreement is a bug
//! in the incremental path, which is the only path that can drift.
//!
//! **What "full" means here, precisely.** A recomputation that ignores every
//! `genesis_*` row cannot know which candidates *happened* to publish, so it
//! cannot reproduce an arbitrary live state. What it reproduces is the
//! **from-empty quiescent state**: the state the durable tables would hold if
//! the shadow had started from an empty substrate against today's evidence and
//! run to quiescence. In PR-B nothing publishes, so full coverage is empty by
//! construction and the comparison is exact. Once machine E lands, the caller
//! must either seed the full snapshot's coverage from the published set or
//! restrict the diff to the sections that remain derivable — silently
//! comparing a live snapshot against a from-empty one would report divergence
//! on every published group.
//!
//! **Scope note.** [`GenesisSnapshot`] carries all four signals.
//! `evidence-cluster` and `community-overview` additionally depend on
//! `community_partition_durable` (see `signals.rs`'s module doc) — a
//! caller-supplied proxy for the community durable-reader gate, which has no
//! M6 consumer registered yet (task #10, Gap B). `recompute_full` threads
//! that flag through rather than guessing a value for it.

use super::candidates::{claim_role_for, ClaimRole, TERMINAL_STATES};
use super::identity::{self, SignalKind};
use super::signals::{self, CandidateProposal};
use crate::WenlanError;
use std::collections::{BTreeMap, BTreeSet};

/// A candidate's identity-bearing fields.
///
/// **Disclosed deviation from §6.1**, which says the snapshot must include
/// `state`. A from-primary-evidence recomputation cannot derive lifecycle
/// state — nothing in `edges` or `provenance_roots` records that a candidate
/// reached `inferencing`. Including a field the full side must fabricate would
/// make every comparison of it meaningless. `state` is used here as a
/// **membership filter** instead: [`snapshot_incremental`] admits non-terminal
/// candidates only, so a candidate reaching a terminal state shows up as a
/// membership divergence rather than a field one, and the information §6.1
/// wanted is still in the diff.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CandidateIdentity {
    pub candidate_id: String,
    pub slot_id: String,
    pub page_id: String,
    pub signal_kind: SignalKind,
    pub coverage_epoch: i64,
    /// Fingerprint field 8 over the candidate's supporting roots. The full side
    /// recomputes it from today's evidence; the incremental side reads what was
    /// stored at `observed`. They disagree exactly when the candidate's inputs
    /// moved under it — which is the A3 `input_moved` condition, and the reason
    /// §6.1 lists this field as identity-bearing.
    pub active_root_digest: String,
}

/// One live claim, keyed by the pair the partial unique indexes key on.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ClaimIdentity {
    pub root_id: String,
    pub independence_group_id: String,
    pub claim_role: ClaimRole,
    pub candidate_id: String,
}

/// A normalized, comparable genesis state for one space.
///
/// Normalization matters (§6.1): every field here is identity-bearing, and
/// nothing that legitimately differs between an incremental build and a
/// from-scratch one (`created_at`, `attempt`, `lease_token`, `next_attempt_at`,
/// ...) is carried. Each section is sorted so two independently-built
/// snapshots of the same state compare equal regardless of read order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenesisSnapshot {
    /// What the four D5 signals propose. Sorted by `slot_id`.
    pub proposals: Vec<CandidateProposal>,
    /// Non-terminal candidates. Sorted by `candidate_id`.
    pub candidates: Vec<CandidateIdentity>,
    /// Live (unreleased) claims. Sorted by `(root_id, candidate_id)`.
    pub claims: Vec<ClaimIdentity>,
    /// Groups with a durable coverage row. Sorted.
    pub coverage: Vec<String>,
    /// Groups waiting in the frontier. Sorted.
    pub frontier: Vec<String>,
}

/// Derive the entire genesis state for one space from primary evidence,
/// ignoring every `genesis_*` row.
///
/// `space_id` is the daemon-minted `spaces.id` (identity.rs's rule); see
/// [`signals::space_overview`] for why callers never also pass the name.
/// `community_partition_durable` is forwarded to
/// [`signals::evidence_cluster`] and [`signals::community_overview`]
/// unchanged.
pub async fn recompute_full(
    tx: &libsql::Transaction,
    space_id: &str,
    coverage_epoch: i64,
    community_partition_durable: bool,
) -> Result<GenesisSnapshot, WenlanError> {
    let mut proposals = Vec::new();
    proposals.extend(signals::evidence_cluster(tx, space_id, community_partition_durable).await?);
    proposals.extend(signals::orphan_wikilink(tx, space_id).await?);
    proposals.extend(signals::community_overview(tx, space_id, community_partition_durable).await?);
    if let Some(proposal) = signals::space_overview(tx, space_id).await? {
        proposals.push(proposal);
    }
    proposals.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    let mut candidates = Vec::new();
    let mut claims = Vec::new();
    let mut concept_groups = BTreeSet::new();
    for proposal in &proposals {
        let candidate_id = identity::candidate_id(&proposal.slot_id, coverage_epoch);
        candidates.push(CandidateIdentity {
            candidate_id: candidate_id.clone(),
            slot_id: proposal.slot_id.clone(),
            page_id: proposal.page_id.clone(),
            signal_kind: proposal.signal_kind,
            coverage_epoch,
            // Every root a signal proposes passed `status = 'active'` in the
            // query that produced it (R2), so the pair set is the root set at
            // `active`. A retracted root leaves `root_ids` entirely rather than
            // reappearing here with another status.
            active_root_digest: identity::active_root_digest(
                &proposal
                    .root_ids
                    .iter()
                    .map(|root_id| (root_id.clone(), "active".to_string()))
                    .collect::<Vec<_>>(),
            ),
        });
        let role = claim_role_for(proposal.signal_kind);
        for (root_id, independence_group_id) in root_groups(tx, &proposal.root_ids).await? {
            if role == ClaimRole::Concept {
                concept_groups.insert(independence_group_id.clone());
            }
            claims.push(ClaimIdentity {
                root_id,
                independence_group_id,
                claim_role: role,
                candidate_id: candidate_id.clone(),
            });
        }
    }
    candidates.sort();
    claims.sort();

    // The from-empty quiescent frontier: every eligible group that no derived
    // concept claim accounts for. Coverage is empty because PR-B publishes
    // nothing, so those are the only two of the six reasons a full
    // recomputation can produce — a suppression, a card, or a quarantine is a
    // decision, not a derivation, and none of them exist from empty.
    let space = signals::resolve_space_name(tx, space_id).await?;
    let frontier: Vec<String> = eligible_groups(tx, &space)
        .await?
        .into_iter()
        .filter(|group| !concept_groups.contains(group))
        .collect();

    Ok(GenesisSnapshot {
        proposals,
        candidates,
        claims,
        coverage: Vec::new(),
        frontier,
    })
}

/// Read the same state from the durable `genesis_*` rows, and nothing else.
///
/// The proposals section is *reconstructed* from `genesis_candidates` plus its
/// claim rows rather than re-derived from signals — reading the signal path
/// here would compare the full snapshot against itself and the oracle would
/// agree with a completely empty substrate.
pub async fn snapshot_incremental(
    tx: &libsql::Transaction,
    space_id: &str,
    coverage_epoch: i64,
) -> Result<GenesisSnapshot, WenlanError> {
    let space = signals::resolve_space_name(tx, space_id).await?;
    let candidates = durable_candidates(tx, &space, coverage_epoch).await?;
    let claims = durable_claims(tx, &space, coverage_epoch).await?;

    let mut roots_by_candidate: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut groups_by_candidate: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for claim in &claims {
        roots_by_candidate
            .entry(claim.candidate_id.as_str())
            .or_default()
            .push(claim.root_id.clone());
        groups_by_candidate
            .entry(claim.candidate_id.as_str())
            .or_default()
            .insert(claim.independence_group_id.as_str());
    }

    let mut proposals: Vec<CandidateProposal> = candidates
        .iter()
        .map(|candidate| {
            let mut root_ids = roots_by_candidate
                .get(candidate.candidate_id.as_str())
                .cloned()
                .unwrap_or_default();
            root_ids.sort();
            CandidateProposal {
                slot_id: candidate.slot_id.clone(),
                page_id: candidate.page_id.clone(),
                signal_kind: candidate.signal_kind,
                root_ids,
                group_count: groups_by_candidate
                    .get(candidate.candidate_id.as_str())
                    .map(|groups| groups.len() as i64)
                    .unwrap_or_default(),
            }
        })
        .collect();
    proposals.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));

    Ok(GenesisSnapshot {
        proposals,
        candidates,
        claims,
        coverage: durable_groups(
            tx,
            "SELECT independence_group_id FROM genesis_group_coverage
              WHERE space = ?1 AND coverage_epoch = ?2
              ORDER BY independence_group_id",
            &space,
            coverage_epoch,
        )
        .await?,
        frontier: durable_groups(
            tx,
            "SELECT independence_group_id FROM genesis_frontier
              WHERE space = ?1 AND coverage_epoch = ?2
              ORDER BY independence_group_id",
            &space,
            coverage_epoch,
        )
        .await?,
    })
}

/// One field-level disagreement between two snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    /// Which section of [`GenesisSnapshot`] disagreed. Keys are only unique
    /// within a section, so a divergence without one is ambiguous.
    pub section: &'static str,
    /// The section's identity for the disagreeing item — `slot_id` for
    /// proposals, `candidate_id` for candidates, `root_id|candidate_id` for
    /// claims, the group ID for coverage and frontier.
    pub key: String,
    pub detail: String,
}

/// Structural diff between two snapshots. Empty iff they agree.
///
/// Field census (§6.1's "the field list is itself asserted by a test"): every
/// comparable field of every section is compared, and the census is enforced by
/// the comparators destructuring each struct exhaustively rather than reaching
/// for fields by name — a new field makes that pattern non-exhaustive and the
/// crate stops compiling. Reporting agreement over a field it never read is the
/// one failure this oracle cannot be allowed to have, so the guarantee is
/// structural rather than a convention. The section list itself is guarded the
/// same way, by destructuring [`GenesisSnapshot`].
pub fn diff(a: &GenesisSnapshot, b: &GenesisSnapshot) -> Vec<Divergence> {
    // Section census. A new section on `GenesisSnapshot` breaks this pattern.
    let GenesisSnapshot {
        proposals,
        candidates,
        claims,
        coverage,
        frontier,
    } = a;
    let GenesisSnapshot {
        proposals: b_proposals,
        candidates: b_candidates,
        claims: b_claims,
        coverage: b_coverage,
        frontier: b_frontier,
    } = b;

    let mut divergences = Vec::new();
    divergences.extend(keyed_diff(
        "proposals",
        proposals,
        b_proposals,
        |proposal| proposal.slot_id.clone(),
        proposal_divergences,
    ));
    divergences.extend(keyed_diff(
        "candidates",
        candidates,
        b_candidates,
        |candidate| candidate.candidate_id.clone(),
        candidate_divergences,
    ));
    divergences.extend(keyed_diff(
        "claims",
        claims,
        b_claims,
        |claim| format!("{}|{}", claim.root_id, claim.candidate_id),
        claim_divergences,
    ));
    divergences.extend(set_diff("coverage", coverage, b_coverage));
    divergences.extend(set_diff("frontier", frontier, b_frontier));
    divergences
}

/// Match two sections by key, report presence differences, and hand matched
/// pairs to a field comparator.
fn keyed_diff<T, K, C>(
    section: &'static str,
    a: &[T],
    b: &[T],
    key_of: K,
    compare: C,
) -> Vec<Divergence>
where
    K: Fn(&T) -> String,
    C: Fn(&'static str, &str, &T, &T) -> Vec<Divergence>,
{
    let mut b_by_key: BTreeMap<String, &T> = b.iter().map(|item| (key_of(item), item)).collect();
    let mut divergences = Vec::new();
    for a_item in a {
        let key = key_of(a_item);
        match b_by_key.remove(&key) {
            None => divergences.push(Divergence {
                section,
                key,
                detail: "present in a, absent in b".to_string(),
            }),
            Some(b_item) => divergences.extend(compare(section, &key, a_item, b_item)),
        }
    }
    for (key, _) in b_by_key {
        divergences.push(Divergence {
            section,
            key,
            detail: "present in b, absent in a".to_string(),
        });
    }
    divergences
}

fn set_diff(section: &'static str, a: &[String], b: &[String]) -> Vec<Divergence> {
    let a_set: BTreeSet<&str> = a.iter().map(String::as_str).collect();
    let b_set: BTreeSet<&str> = b.iter().map(String::as_str).collect();
    let mut divergences = Vec::new();
    for key in a_set.difference(&b_set) {
        divergences.push(Divergence {
            section,
            key: (*key).to_string(),
            detail: "present in a, absent in b".to_string(),
        });
    }
    for key in b_set.difference(&a_set) {
        divergences.push(Divergence {
            section,
            key: (*key).to_string(),
            detail: "present in b, absent in a".to_string(),
        });
    }
    divergences
}

fn proposal_divergences(
    section: &'static str,
    key: &str,
    a: &CandidateProposal,
    b: &CandidateProposal,
) -> Vec<Divergence> {
    // The census is this destructuring, not a convention: a new field on
    // `CandidateProposal` makes both patterns non-exhaustive and the crate
    // stops compiling until it is compared below.
    let CandidateProposal {
        slot_id: _,
        page_id,
        signal_kind,
        root_ids,
        group_count,
    } = a;
    let CandidateProposal {
        slot_id: _,
        page_id: b_page_id,
        signal_kind: b_signal_kind,
        root_ids: b_root_ids,
        group_count: b_group_count,
    } = b;

    let mut fields = FieldReport::new(section, key);
    fields.compare("page_id", page_id, b_page_id);
    fields.compare("signal_kind", signal_kind, b_signal_kind);
    fields.compare("root_ids", root_ids, b_root_ids);
    fields.compare("group_count", group_count, b_group_count);
    fields.finish()
}

fn candidate_divergences(
    section: &'static str,
    key: &str,
    a: &CandidateIdentity,
    b: &CandidateIdentity,
) -> Vec<Divergence> {
    let CandidateIdentity {
        candidate_id: _,
        slot_id,
        page_id,
        signal_kind,
        coverage_epoch,
        active_root_digest,
    } = a;
    let CandidateIdentity {
        candidate_id: _,
        slot_id: b_slot_id,
        page_id: b_page_id,
        signal_kind: b_signal_kind,
        coverage_epoch: b_coverage_epoch,
        active_root_digest: b_active_root_digest,
    } = b;

    let mut fields = FieldReport::new(section, key);
    fields.compare("slot_id", slot_id, b_slot_id);
    fields.compare("page_id", page_id, b_page_id);
    fields.compare("signal_kind", signal_kind, b_signal_kind);
    fields.compare("coverage_epoch", coverage_epoch, b_coverage_epoch);
    fields.compare(
        "active_root_digest",
        active_root_digest,
        b_active_root_digest,
    );
    fields.finish()
}

fn claim_divergences(
    section: &'static str,
    key: &str,
    a: &ClaimIdentity,
    b: &ClaimIdentity,
) -> Vec<Divergence> {
    let ClaimIdentity {
        root_id: _,
        independence_group_id,
        claim_role,
        candidate_id: _,
    } = a;
    let ClaimIdentity {
        root_id: _,
        independence_group_id: b_group,
        claim_role: b_role,
        candidate_id: _,
    } = b;

    let mut fields = FieldReport::new(section, key);
    fields.compare("independence_group_id", independence_group_id, b_group);
    fields.compare("claim_role", claim_role, b_role);
    fields.finish()
}

/// Collects per-field disagreements for one matched pair.
struct FieldReport<'a> {
    section: &'static str,
    key: &'a str,
    divergences: Vec<Divergence>,
}

impl<'a> FieldReport<'a> {
    fn new(section: &'static str, key: &'a str) -> Self {
        Self {
            section,
            key,
            divergences: Vec::new(),
        }
    }

    fn compare<T: PartialEq + std::fmt::Debug>(&mut self, field: &str, a: &T, b: &T) {
        if a != b {
            self.divergences.push(Divergence {
                section: self.section,
                key: self.key.to_string(),
                detail: format!("{field}: {a:?} vs {b:?}"),
            });
        }
    }

    fn finish(self) -> Vec<Divergence> {
        self.divergences
    }
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

/// `(root_id, independence_group_id)` for a bounded set of roots, in the order
/// the caller listed them. Primary evidence — `provenance_roots` — so this is
/// legal on the full side.
async fn root_groups(
    tx: &libsql::Transaction,
    root_ids: &[String],
) -> Result<Vec<(String, String)>, WenlanError> {
    if root_ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = (1..=root_ids.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let params: Vec<libsql::Value> = root_ids
        .iter()
        .map(|root_id| libsql::Value::Text(root_id.clone()))
        .collect();
    let mut rows = tx
        .query(
            &format!(
                "SELECT root_id, independence_group_id FROM provenance_roots
                  WHERE root_id IN ({placeholders})
                  ORDER BY root_id"
            ),
            libsql::params_from_iter(params),
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle root groups: {error}")))?;
    let mut pairs = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle root groups row: {error}")))?
    {
        let decode = |error: libsql::Error| {
            WenlanError::VectorDb(format!("m6 oracle root groups decode: {error}"))
        };
        pairs.push((row.get(0).map_err(decode)?, row.get(1).map_err(decode)?));
    }
    Ok(pairs)
}

/// Artifact 5 §1.2's eligible set, restricted to one space.
async fn eligible_groups(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT DISTINCT r.independence_group_id
               FROM edges e
               JOIN provenance_roots r ON r.root_id = e.root_id
              WHERE e.space = ?1
                AND e.grounded = 1
                AND e.valid_until IS NULL
                AND r.status = 'active'
                AND r.root_kind <> 'generated'
              ORDER BY r.independence_group_id",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle eligible: {error}")))?;
    let mut groups = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle eligible row: {error}")))?
    {
        groups.push(row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("m6 oracle eligible decode: {error}"))
        })?);
    }
    Ok(groups)
}

async fn durable_candidates(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
) -> Result<Vec<CandidateIdentity>, WenlanError> {
    let terminal = TERMINAL_STATES
        .iter()
        .map(|state| format!("'{}'", state.as_str()))
        .collect::<Vec<_>>()
        .join(", ");
    let mut rows = tx
        .query(
            // The terminal list is interpolated from `TERMINAL_STATES`, whose
            // members are `&'static str` literals from this crate — never user
            // input. Everything a caller supplies is bound.
            &format!(
                "SELECT candidate_id, slot_id, page_id, signal_kind, coverage_epoch,
                        active_root_digest
                   FROM genesis_candidates
                  WHERE space = ?1 AND coverage_epoch = ?2
                    AND state NOT IN ({terminal})
                  ORDER BY candidate_id"
            ),
            libsql::params![space, coverage_epoch],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle candidates: {error}")))?;
    let mut candidates = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle candidates row: {error}")))?
    {
        let decode = |error: libsql::Error| {
            WenlanError::VectorDb(format!("m6 oracle candidates decode: {error}"))
        };
        let tag: String = row.get(3).map_err(decode)?;
        candidates.push(CandidateIdentity {
            candidate_id: row.get(0).map_err(decode)?,
            slot_id: row.get(1).map_err(decode)?,
            page_id: row.get(2).map_err(decode)?,
            signal_kind: SignalKind::from_tag(&tag).ok_or_else(|| {
                WenlanError::VectorDb(format!("m6 oracle unknown signal_kind {tag:?}"))
            })?,
            coverage_epoch: row.get(4).map_err(decode)?,
            active_root_digest: row.get(5).map_err(decode)?,
        });
    }
    Ok(candidates)
}

async fn durable_claims(
    tx: &libsql::Transaction,
    space: &str,
    coverage_epoch: i64,
) -> Result<Vec<ClaimIdentity>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT gr.root_id, gr.independence_group_id, gr.claim_role, gr.candidate_id
               FROM genesis_candidate_roots gr
               JOIN genesis_candidates c ON c.candidate_id = gr.candidate_id
              WHERE c.space = ?1 AND gr.coverage_epoch = ?2 AND gr.released_at IS NULL
              ORDER BY gr.root_id, gr.candidate_id",
            libsql::params![space, coverage_epoch],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle claims: {error}")))?;
    let mut claims = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle claims row: {error}")))?
    {
        let decode = |error: libsql::Error| {
            WenlanError::VectorDb(format!("m6 oracle claims decode: {error}"))
        };
        let role: String = row.get(2).map_err(decode)?;
        claims.push(ClaimIdentity {
            root_id: row.get(0).map_err(decode)?,
            independence_group_id: row.get(1).map_err(decode)?,
            claim_role: ClaimRole::parse(&role).ok_or_else(|| {
                WenlanError::VectorDb(format!("m6 oracle unknown claim_role {role:?}"))
            })?,
            candidate_id: row.get(3).map_err(decode)?,
        });
    }
    Ok(claims)
}

async fn durable_groups(
    tx: &libsql::Transaction,
    sql: &str,
    space: &str,
    coverage_epoch: i64,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = tx
        .query(sql, libsql::params![space, coverage_epoch])
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle groups: {error}")))?;
    let mut groups = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 oracle groups row: {error}")))?
    {
        groups.push(
            row.get::<String>(0).map_err(|error| {
                WenlanError::VectorDb(format!("m6 oracle groups decode: {error}"))
            })?,
        );
    }
    Ok(groups)
}
