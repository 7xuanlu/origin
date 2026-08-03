// SPDX-License-Identifier: Apache-2.0
//! The incremental-vs-full recomputation oracle (spec §6.1).
//!
//! `recompute_full` derives genesis state for a space **from primary
//! evidence only** — edges, roots, pages, communities — never from a
//! `genesis_*` row. PR-B1 ships this half only: `snapshot_incremental` and
//! `diff` need the durable candidate/claim/coverage/frontier state that
//! `candidates.rs`/`frontier.rs` (PR-B2) write, so there is nothing yet to
//! read incrementally or compare against.
//!
//! **Scope note.** [`GenesisSnapshot`] carries all four signals.
//! `evidence-cluster` and `community-overview` additionally depend on
//! `community_partition_durable` (see `signals.rs`'s module doc) — a
//! caller-supplied proxy for the community durable-reader gate, which has no
//! M6 consumer registered yet (task #10, Gap B). `recompute_full` threads
//! that flag through rather than guessing a value for it.

use super::signals::{self, CandidateProposal};
use crate::WenlanError;

/// A normalized, comparable genesis state for one space.
///
/// Normalization matters (§6.1): this holds only identity-bearing fields
/// (`slot_id`, `page_id`, `signal_kind`, `root_ids`, `group_count`) and
/// nothing that legitimately differs between an incremental build and a
/// from-scratch one (`created_at`, `attempt`, `lease_token`, ...) — none of
/// which exist on a bare [`CandidateProposal`] in the first place, since
/// those are candidate-row fields `candidates.rs` (PR-B2) adds.
///
/// Sorted by `slot_id` so two independently-built snapshots of the same
/// state compare equal regardless of read order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenesisSnapshot {
    pub proposals: Vec<CandidateProposal>,
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
    Ok(GenesisSnapshot { proposals })
}

/// One field-level disagreement between two snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    pub slot_id: String,
    pub detail: String,
}

/// Structural diff between two snapshots. Empty iff they agree.
///
/// Field census (§6.1's "the field list is itself asserted by a test"):
/// every comparable field on [`CandidateProposal`] (`slot_id`, `page_id`,
/// `signal_kind`, `root_ids`, `group_count`) is compared, and the census is
/// enforced by [`field_divergences`] destructuring the struct exhaustively
/// rather than reaching for fields by name — a new field makes that pattern
/// non-exhaustive and the crate stops compiling. Reporting agreement over a
/// field it never read is the one failure this oracle cannot be allowed to
/// have, so the guarantee is structural rather than a convention.
pub fn diff(a: &GenesisSnapshot, b: &GenesisSnapshot) -> Vec<Divergence> {
    let mut divergences = Vec::new();
    let mut b_by_slot: std::collections::HashMap<&str, &CandidateProposal> = b
        .proposals
        .iter()
        .map(|p| (p.slot_id.as_str(), p))
        .collect();

    for a_proposal in &a.proposals {
        match b_by_slot.remove(a_proposal.slot_id.as_str()) {
            None => divergences.push(Divergence {
                slot_id: a_proposal.slot_id.clone(),
                detail: "present in a, absent in b".to_string(),
            }),
            Some(b_proposal) => {
                divergences.extend(field_divergences(a_proposal, b_proposal));
            }
        }
    }
    for (slot_id, _) in b_by_slot {
        divergences.push(Divergence {
            slot_id: slot_id.to_string(),
            detail: "present in b, absent in a".to_string(),
        });
    }
    divergences
}

fn field_divergences(a: &CandidateProposal, b: &CandidateProposal) -> Vec<Divergence> {
    // The census is this destructuring, not a convention: a new field on
    // `CandidateProposal` makes both patterns non-exhaustive and the crate
    // stops compiling until it is compared below.
    let CandidateProposal {
        slot_id,
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

    let mut divergences = Vec::new();
    let mut note = |detail: String| {
        divergences.push(Divergence {
            slot_id: slot_id.clone(),
            detail,
        });
    };
    if page_id != b_page_id {
        note(format!("page_id: {page_id} vs {b_page_id}"));
    }
    if signal_kind != b_signal_kind {
        note(format!("signal_kind: {signal_kind:?} vs {b_signal_kind:?}"));
    }
    if root_ids != b_root_ids {
        note(format!("root_ids: {root_ids:?} vs {b_root_ids:?}"));
    }
    if group_count != b_group_count {
        note(format!("group_count: {group_count} vs {b_group_count}"));
    }
    divergences
}
