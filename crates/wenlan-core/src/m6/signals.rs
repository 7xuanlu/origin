// SPDX-License-Identifier: Apache-2.0
//! The four D5 genesis signals (spec §3.2). Each function is a **pure
//! reader**: it takes a `&libsql::Transaction` and returns proposals,
//! opening no transaction of its own and writing nothing (§2.1).
//!
//! **Scope note.** All four signals are wired. [`evidence_cluster`] and
//! [`community_overview`] take a caller-supplied `community_partition_durable:
//! bool` rather than calling `community_reader_durable_gate_sql`
//! (`crates/wenlan-core/src/db.rs:2663`) directly: that gate compiles to one
//! all-or-nothing predicate per consumer — there is no `proof.space`
//! parameter anywhere in its SQL, only two consumer-scoped subqueries
//! universally quantified over every proven space — so a scalar input is
//! faithful to the gate's actual shape, not a simplification of it. There is
//! no M6 consumer registered there yet; registering one (`genesis_candidates`,
//! per lead ruling) is a `db.rs` migration + reconciler-output change tracked
//! separately (task #10, Gap B) rather than guessed here. Until that consumer
//! exists, both signals admit nothing from these calls on any production
//! corpus — correct behavior given M4's grounded subgraph is empty in
//! production anyway, not a defect introduced here.
//!
//! `orphan-wikilink`'s D1 R8 claim-revision-to-link binding does not exist as
//! a stored column, but per
//! `docs/plans/2026-08-03-m6-pr-a-followup-2-scope.md` §2 (lead-accepted 2026-08-03,
//! reversing the earlier PR-A-new deferral) it is derivable at read time from
//! `page_version_claims` + `claim_anchors` + `pages.content`, verified
//! against the live body's digest — the same idiom the derivation aligner
//! already uses. No schema, no migration, no backfill.

use super::constants::ROOTS_PER_CANDIDATE_CAP;
use super::identity::{self, SignalKind};
use super::independence::{self, GroupCountScope};
use crate::WenlanError;

/// Signal 3/4's extra structural floor: at least this many distinct
/// grounded nodes, on top of D1's 3-group floor (§3.2).
const MIN_GROUNDED_NODES_OVERVIEW: usize = 5;

/// One dry-run candidate proposal — a pure function of its signal's inputs
/// (§2.1 item 1). Determinism is structural: `slot_id`/`page_id` are
/// digests of contract-named inputs (`identity.rs`), so two runs over
/// identical state produce identical proposals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateProposal {
    pub slot_id: String,
    pub page_id: String,
    pub signal_kind: SignalKind,
    /// The supporting root set, sorted and deduped — the same shape
    /// `active_root_digest` and fingerprint field 6 expect.
    pub root_ids: Vec<String>,
    /// The D1 count this proposal cleared. Never below
    /// [`independence::INDEPENDENCE_GROUP_FLOOR`].
    pub group_count: i64,
}

/// Signal 4 — space-overview (§3.2 row 4).
///
/// `space_id` is the daemon-minted `spaces.id`, never the renameable name
/// (identity.rs's rule) — this function resolves the current name itself
/// for the queries that need it, so a caller never has to carry both.
///
/// Admits when the space carries no active space-overview subscription
/// (`m6_overview_subscriptions`, scope_kind='space') and its grounded
/// evidence clears both the D1 group floor and the 5-grounded-node floor.
/// Returns `None` on either miss; never partial state.
pub async fn space_overview(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<Option<CandidateProposal>, WenlanError> {
    let space = resolve_space_name(tx, space_id).await?;

    if has_active_overview_subscription(tx, "space", &space).await? {
        return Ok(None);
    }

    let scope = GroupCountScope {
        predicate: "e.space = ?1",
        params: vec![libsql::Value::Text(space)],
    };
    let group_count = independence::distinct_group_count(tx, &scope).await?;
    if !independence::admits(group_count) {
        return Ok(None);
    }

    let root_ids = scoped_root_ids(tx, &scope).await?;
    let node_count = scoped_grounded_node_count(tx, &scope).await?;
    if node_count < MIN_GROUNDED_NODES_OVERVIEW {
        return Ok(None);
    }

    let slot_id = identity::slot_id_space_overview(space_id);
    let page_id = identity::page_id(&slot_id);
    Ok(Some(CandidateProposal {
        slot_id,
        page_id,
        signal_kind: SignalKind::SpaceOverview,
        root_ids,
        group_count,
    }))
}

/// Signal 1 — evidence-cluster (§3.2 row 1).
///
/// One candidate per current (non-retired) M4 community in the space, scoped
/// to that community's `attachment = 'core'` members only (S0-16) — a
/// peripheral member's evidence never counts toward this signal's group or
/// root set.
///
/// `community_partition_durable` proxies `community_reader_durable_gate_sql`
/// (module doc); `false` admits nothing, structural preconditions aside.
pub async fn evidence_cluster(
    tx: &libsql::Transaction,
    space_id: &str,
    community_partition_durable: bool,
) -> Result<Vec<CandidateProposal>, WenlanError> {
    if !community_partition_durable {
        return Ok(Vec::new());
    }
    let space = resolve_space_name(tx, space_id).await?;

    let mut proposals = Vec::new();
    for community_id in current_communities(tx, &space).await? {
        let member_ids = community_member_node_ids(tx, &space, &community_id, Some("core")).await?;
        if member_ids.is_empty() {
            continue;
        }
        let (predicate, params) = member_predicate(&member_ids);
        let scope = GroupCountScope {
            predicate: &predicate,
            params,
        };

        let group_count = independence::distinct_group_count(tx, &scope).await?;
        if !independence::admits(group_count) {
            continue;
        }
        let root_ids = scoped_root_ids(tx, &scope).await?;
        let group_ids = scoped_group_ids(tx, &scope).await?;

        let slot_id = identity::slot_id_evidence_cluster(space_id, &group_ids);
        let page_id = identity::page_id(&slot_id);
        proposals.push(CandidateProposal {
            slot_id,
            page_id,
            signal_kind: SignalKind::EvidenceCluster,
            root_ids,
            group_count,
        });
    }
    proposals.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    Ok(proposals)
}

/// Signal 3 — community-overview (§3.2 row 3).
///
/// One candidate per current published M4 community with no active
/// community-overview subscription, scoped to every member of that
/// community — unlike [`evidence_cluster`], no `attachment` filter, so a
/// peripheral member's evidence counts here.
///
/// `community_partition_durable` carries the same meaning as
/// [`evidence_cluster`]'s parameter of the same name.
pub async fn community_overview(
    tx: &libsql::Transaction,
    space_id: &str,
    community_partition_durable: bool,
) -> Result<Vec<CandidateProposal>, WenlanError> {
    if !community_partition_durable {
        return Ok(Vec::new());
    }
    let space = resolve_space_name(tx, space_id).await?;

    let mut proposals = Vec::new();
    for community_id in current_communities(tx, &space).await? {
        if has_active_overview_subscription(tx, "community", &community_id).await? {
            continue;
        }
        let member_ids = community_member_node_ids(tx, &space, &community_id, None).await?;
        if member_ids.is_empty() {
            continue;
        }
        let (predicate, params) = member_predicate(&member_ids);
        let scope = GroupCountScope {
            predicate: &predicate,
            params,
        };

        let group_count = independence::distinct_group_count(tx, &scope).await?;
        if !independence::admits(group_count) {
            continue;
        }
        let node_count = scoped_grounded_node_count(tx, &scope).await?;
        if node_count < MIN_GROUNDED_NODES_OVERVIEW {
            continue;
        }
        let root_ids = scoped_root_ids(tx, &scope).await?;

        let slot_id = identity::slot_id_community_overview(space_id, &community_id);
        let page_id = identity::page_id(&slot_id);
        proposals.push(CandidateProposal {
            slot_id,
            page_id,
            signal_kind: SignalKind::CommunityOverview,
            root_ids,
            group_count,
        });
    }
    proposals.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    Ok(proposals)
}

/// Every non-retired M4 community in `space`, ordered for determinism.
async fn current_communities(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT community_id FROM communities
              WHERE space = ?1 AND retired_at IS NULL
              ORDER BY community_id",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 current communities: {error}")))?;
    let mut community_ids = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 current communities row: {error}")))?
    {
        community_ids.push(row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("m6 current communities decode: {error}"))
        })?);
    }
    Ok(community_ids)
}

/// `community_members.node_id` for `community_id`, optionally narrowed to
/// one `attachment` value (`Some("core")` for signal 1; `None` for signal 3,
/// which counts every member regardless of attachment).
async fn community_member_node_ids(
    tx: &libsql::Transaction,
    space: &str,
    community_id: &str,
    attachment: Option<&str>,
) -> Result<Vec<String>, WenlanError> {
    let mut rows = match attachment {
        Some(attachment) => {
            tx.query(
                "SELECT node_id FROM community_members
                  WHERE space = ?1 AND community_id = ?2 AND attachment = ?3
                  ORDER BY node_id",
                libsql::params![space, community_id, attachment],
            )
            .await
        }
        None => {
            tx.query(
                "SELECT node_id FROM community_members
                  WHERE space = ?1 AND community_id = ?2
                  ORDER BY node_id",
                libsql::params![space, community_id],
            )
            .await
        }
    }
    .map_err(|error| WenlanError::VectorDb(format!("m6 community member node ids: {error}")))?;
    let mut node_ids = Vec::new();
    while let Some(row) = rows.next().await.map_err(|error| {
        WenlanError::VectorDb(format!("m6 community member node ids row: {error}"))
    })? {
        node_ids.push(row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("m6 community member node ids decode: {error}"))
        })?);
    }
    Ok(node_ids)
}

/// The `GroupCountScope` predicate/params for "an edge touching any of
/// `member_ids` as an entity endpoint" — shared by signals 1 and 3, which
/// differ only in which member set they pass in. The src and dst branches
/// get distinct placeholder ranges (never the same number reused across two
/// textual positions) so this composes safely with [`scoped_grounded_node_count`],
/// which embeds the predicate twice.
fn member_predicate(member_ids: &[String]) -> (String, Vec<libsql::Value>) {
    let n = member_ids.len();
    let src_placeholders = (1..=n)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let dst_placeholders = (n + 1..=2 * n)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let predicate = format!(
        "((e.src_kind = 'entity' AND e.src_id IN ({src_placeholders}))
          OR (e.dst_kind = 'entity' AND e.dst_id IN ({dst_placeholders})))"
    );
    let mut params: Vec<libsql::Value> = member_ids
        .iter()
        .cloned()
        .map(libsql::Value::Text)
        .collect();
    params.extend(member_ids.iter().cloned().map(libsql::Value::Text));
    (predicate, params)
}

/// The distinct `independence_group_id`s behind a scoped count — signal 1's
/// `slot_id` input (identity.rs: "keyed on the *initial* independence-group
/// set"). Same R1-R5 shape as [`independence::distinct_group_count`], just
/// projecting the group id instead of counting it.
async fn scoped_group_ids(
    tx: &libsql::Transaction,
    scope: &GroupCountScope<'_>,
) -> Result<Vec<String>, WenlanError> {
    let sql = format!(
        "SELECT DISTINCT r.independence_group_id
           FROM edges e
           JOIN provenance_roots r ON r.root_id = e.root_id
          WHERE e.grounded = 1
            AND e.valid_until IS NULL
            AND r.status = 'active'
            AND r.root_kind <> 'generated'
            AND NOT EXISTS (
                SELECT 1 FROM pages p
                 WHERE ((e.src_kind = 'page' AND p.id = e.src_id)
                        OR (e.dst_kind = 'page' AND p.id = e.dst_id))
                   AND lower(p.title) = 'overview'
            )
            AND ({predicate})
          ORDER BY r.independence_group_id",
        predicate = scope.predicate,
    );
    let mut rows = tx
        .query(&sql, libsql::params_from_iter(scope.params.clone()))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 scoped group ids: {error}")))?;
    let mut group_ids = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 scoped group ids row: {error}")))?
    {
        group_ids.push(row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("m6 scoped group ids decode: {error}"))
        })?);
    }
    Ok(group_ids)
}

/// Signal 2's structural precondition (§3.2 row 2): a candidate label needs
/// at least this many distinct referring pages.
const MIN_ORPHAN_REFERRING_PAGES: usize = 2;

/// Signal 2 — orphan-wikilink (§3.2 row 2, R8).
///
/// Unlike the other three signals, one space can carry many orphan-wikilink
/// candidates — one per surviving normalized label — so this returns the
/// whole admitted set rather than a single `Option`.
///
/// A label's underlying groups are the union, over its qualifying referring
/// pages, of active grounded evidence reachable through a live `supports`
/// edge on a claim revision whose anchored span contains that page's
/// occurrence of the link (R8: "the exact current claim revisions that
/// contain the link", never the whole page's evidence). A page qualifies
/// only while it is active, non-overview, and `supported` **at its current
/// version** (`page_truth_state.page_version = pages.version`, not merely
/// `support_status = 'supported'` — a page whose derivation lagged its body
/// has stale-version anchors, and `page_version = p.version` is what keeps
/// its stale claims from being trusted).
pub async fn orphan_wikilink(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<Vec<CandidateProposal>, WenlanError> {
    let space = resolve_space_name(tx, space_id).await?;
    let pages_by_label = orphan_candidate_pages(tx, &space).await?;

    let mut proposals = Vec::new();
    for (label_key, page_ids) in pages_by_label {
        if page_ids.len() < MIN_ORPHAN_REFERRING_PAGES {
            continue;
        }
        if let Some(proposal) = orphan_wikilink_slot(tx, space_id, &label_key, &page_ids).await? {
            proposals.push(proposal);
        }
    }
    proposals.sort_by(|a, b| a.slot_id.cmp(&b.slot_id));
    Ok(proposals)
}

/// D8-normalized orphan labels in `space`, mapped to their distinct, active,
/// non-overview, current-version-`supported` referring pages. A raw target
/// `normalize_label_key` rejects is dropped, matching the write-time drop
/// semantics (S0-39) rather than treated as an error.
async fn orphan_candidate_pages(
    tx: &libsql::Transaction,
    space: &str,
) -> Result<std::collections::BTreeMap<String, Vec<String>>, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT pl.source_page_id, pl.label
               FROM page_links pl
               JOIN pages p ON p.id = pl.source_page_id
               JOIN page_truth_state pts ON pts.page_id = p.id
              WHERE pl.target_page_id IS NULL
                AND p.space = ?1
                AND p.status = 'active'
                AND lower(p.title) <> 'overview'
                AND pts.support_status = 'supported'
                AND pts.page_version = p.version",
            libsql::params![space],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan candidate pages: {error}")))?;

    let mut by_label: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan candidate pages row: {error}")))?
    {
        let page_id = row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("m6 orphan candidate pages decode page_id: {error}"))
        })?;
        let label = row.get::<String>(1).map_err(|error| {
            WenlanError::VectorDb(format!("m6 orphan candidate pages decode label: {error}"))
        })?;
        let Ok(label_key) = super::label_key::normalize_label_key(&label) else {
            continue;
        };
        let pages = by_label.entry(label_key).or_default();
        if !pages.contains(&page_id) {
            pages.push(page_id);
        }
    }
    Ok(by_label)
}

/// One orphan-wikilink candidate slot: gather the containing revisions
/// across `page_ids`, scope D1's count to their `supports` roots, and admit
/// or reject.
async fn orphan_wikilink_slot(
    tx: &libsql::Transaction,
    space_id: &str,
    label_key: &str,
    page_ids: &[String],
) -> Result<Option<CandidateProposal>, WenlanError> {
    let mut revision_ids = Vec::new();
    for page_id in page_ids {
        revision_ids.extend(containing_revisions(tx, page_id, label_key).await?);
    }
    revision_ids.sort();
    revision_ids.dedup();
    if revision_ids.is_empty() {
        return Ok(None);
    }

    // R8's scope: e's provenance root is one a live `supports` edge on a
    // containing revision names. `e` stays a grounded, active, non-generated,
    // non-overview edge — R1-R5 stay owned by independence.rs.
    let placeholders = (1..=revision_ids.len())
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let predicate = format!(
        "EXISTS (
             SELECT 1 FROM edges s
              WHERE s.edge_type   = 'supports'
                AND s.src_kind    = 'claim_revision'
                AND s.src_id IN ({placeholders})
                AND s.valid_until IS NULL
                AND s.root_id     = e.root_id
         )"
    );
    let params = revision_ids
        .iter()
        .cloned()
        .map(libsql::Value::Text)
        .collect();
    let scope = GroupCountScope {
        predicate: &predicate,
        params,
    };

    let group_count = independence::distinct_group_count(tx, &scope).await?;
    if !independence::admits(group_count) {
        return Ok(None);
    }
    let root_ids = scoped_root_ids(tx, &scope).await?;

    let slot_id = identity::slot_id_orphan_wikilink(space_id, label_key);
    let page_id = identity::page_id(&slot_id);
    Ok(Some(CandidateProposal {
        slot_id,
        page_id,
        signal_kind: SignalKind::OrphanWikilink,
        root_ids,
        group_count,
    }))
}

/// Claim revisions in `page_id`'s **current version** whose anchored span
/// contains an occurrence of `label_key` in the page's live body. Each span
/// is verified against the live body's digest before it is trusted (§2.1
/// step 4, the same check the derivation aligner performs against a prior
/// version) — a stale or tampered anchor contributes nothing.
async fn containing_revisions(
    tx: &libsql::Transaction,
    page_id: &str,
    label_key: &str,
) -> Result<Vec<String>, WenlanError> {
    let mut page_rows = tx
        .query(
            "SELECT content, version FROM pages WHERE id = ?1",
            libsql::params![page_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan page body: {error}")))?;
    let Some(row) = page_rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan page body row: {error}")))?
    else {
        return Ok(Vec::new());
    };
    let content = row
        .get::<String>(0)
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan page content: {error}")))?;
    let version = row
        .get::<i64>(1)
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan page version: {error}")))?;

    // The label's occurrences in the live body — the write path's own
    // extractor, so this reuses what it derived rather than re-deriving it.
    let occurrences: Vec<std::ops::Range<usize>> =
        crate::sources::obsidian::extract_wikilinks_with_spans(&content)
            .into_iter()
            .filter(|link| {
                super::label_key::normalize_label_key(&link.target)
                    .map(|key| key == label_key)
                    .unwrap_or(false)
            })
            .map(|link| link.target_range)
            .collect();
    if occurrences.is_empty() {
        return Ok(Vec::new());
    }

    let mut rows = tx
        .query(
            "SELECT pvc.claim_revision_id, ca.span_start, ca.span_end, ca.span_digest
               FROM page_version_claims pvc
               JOIN claim_anchors ca
                 ON ca.claim_revision_id = pvc.claim_revision_id
                AND ca.source_doc_id     = pvc.page_id
                AND ca.source_version    = pvc.page_version
              WHERE pvc.page_id = ?1 AND pvc.page_version = ?2",
            libsql::params![page_id, version],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan revisions: {error}")))?;

    let mut revision_ids = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 orphan revisions row: {error}")))?
    {
        let claim_revision_id = row
            .get::<String>(0)
            .map_err(|error| WenlanError::VectorDb(format!("m6 orphan revision id: {error}")))?;
        let span_start = row
            .get::<i64>(1)
            .map_err(|error| WenlanError::VectorDb(format!("m6 orphan span start: {error}")))?
            as usize;
        let span_end = row
            .get::<i64>(2)
            .map_err(|error| WenlanError::VectorDb(format!("m6 orphan span end: {error}")))?
            as usize;
        let span_digest = row
            .get::<String>(3)
            .map_err(|error| WenlanError::VectorDb(format!("m6 orphan span digest: {error}")))?;

        let Some(span_text) = content.get(span_start..span_end) else {
            continue;
        };
        if crate::provenance::revision_content_digest(span_text) != span_digest {
            continue; // stale or tampered anchor: verified per §2.1 step 4
        }
        if occurrences
            .iter()
            .any(|occ| occ.start >= span_start && occ.end <= span_end)
        {
            revision_ids.push(claim_revision_id);
        }
    }
    Ok(revision_ids)
}

/// The current mutable `spaces.name` for an immutable `spaces.id`. Every
/// evidence table (`edges`, `provenance_roots`, `community_members`,
/// `m6_overview_subscriptions`) keys on the name, not the id, so a reader
/// that only receives the id resolves it once, here, rather than every
/// caller repeating the join.
pub(super) async fn resolve_space_name(
    tx: &libsql::Transaction,
    space_id: &str,
) -> Result<String, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT name FROM spaces WHERE id = ?1",
            libsql::params![space_id],
        )
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 resolve space name: {error}")))?;
    rows.next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 resolve space name row: {error}")))?
        .map(|row| row.get::<String>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 resolve space name decode: {error}")))?
        .ok_or_else(|| {
            WenlanError::VectorDb(format!(
                "m6 resolve space name: no space with id {space_id}"
            ))
        })
}

/// `NOT EXISTS` on `m6_overview_subscriptions` for one live scope — shared
/// by signals 3 and 4, which differ only in `scope_kind`/`scope_id`.
async fn has_active_overview_subscription(
    tx: &libsql::Transaction,
    scope_kind: &str,
    scope_id: &str,
) -> Result<bool, WenlanError> {
    let mut rows = tx
        .query(
            "SELECT EXISTS (
                 SELECT 1 FROM m6_overview_subscriptions
                  WHERE scope_kind = ?1 AND scope_id = ?2 AND state = 'active'
             )",
            libsql::params![scope_kind, scope_id],
        )
        .await
        .map_err(|error| {
            WenlanError::VectorDb(format!("m6 overview subscription check: {error}"))
        })?;
    rows.next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 overview subscription row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| {
            WenlanError::VectorDb(format!("m6 overview subscription decode: {error}"))
        })?
        .map(|flag| flag != 0)
        .ok_or_else(|| WenlanError::VectorDb("m6 overview subscription: no row returned".into()))
}

/// The root ids behind a scoped count, same R1-R5 shape as
/// [`independence::distinct_group_count`] — sorted and deduped, capped at
/// the D8 roots-per-candidate bound (§3.6).
async fn scoped_root_ids(
    tx: &libsql::Transaction,
    scope: &GroupCountScope<'_>,
) -> Result<Vec<String>, WenlanError> {
    let sql = format!(
        "SELECT DISTINCT r.root_id
           FROM edges e
           JOIN provenance_roots r ON r.root_id = e.root_id
          WHERE e.grounded = 1
            AND e.valid_until IS NULL
            AND r.status = 'active'
            AND r.root_kind <> 'generated'
            AND NOT EXISTS (
                SELECT 1 FROM pages p
                 WHERE ((e.src_kind = 'page' AND p.id = e.src_id)
                        OR (e.dst_kind = 'page' AND p.id = e.dst_id))
                   AND lower(p.title) = 'overview'
            )
            AND ({predicate})
          ORDER BY r.root_id
          LIMIT {limit}",
        predicate = scope.predicate,
        limit = ROOTS_PER_CANDIDATE_CAP,
    );
    let mut rows = tx
        .query(&sql, libsql::params_from_iter(scope.params.clone()))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 scoped root ids: {error}")))?;
    let mut root_ids = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 scoped root ids row: {error}")))?
    {
        root_ids.push(row.get::<String>(0).map_err(|error| {
            WenlanError::VectorDb(format!("m6 scoped root ids decode: {error}"))
        })?);
    }
    Ok(root_ids)
}

/// Distinct `entity` endpoints touched by the same qualifying edge set —
/// signal 3/4's "grounded nodes" floor. A separate aggregate from
/// [`independence::distinct_group_count`] on purpose: nodes and
/// independence groups are different axes (one entity's evidence can span
/// several groups, and one group can touch several entities), so counting
/// one would misreport the other.
async fn scoped_grounded_node_count(
    tx: &libsql::Transaction,
    scope: &GroupCountScope<'_>,
) -> Result<usize, WenlanError> {
    let sql = format!(
        "SELECT COUNT(*) FROM (
             SELECT e.src_id AS node_id
               FROM edges e
               JOIN provenance_roots r ON r.root_id = e.root_id
              WHERE e.src_kind = 'entity'
                AND e.grounded = 1
                AND e.valid_until IS NULL
                AND r.status = 'active'
                AND r.root_kind <> 'generated'
                AND ({predicate})
             UNION
             SELECT e.dst_id AS node_id
               FROM edges e
               JOIN provenance_roots r ON r.root_id = e.root_id
              WHERE e.dst_kind = 'entity'
                AND e.grounded = 1
                AND e.valid_until IS NULL
                AND r.status = 'active'
                AND r.root_kind <> 'generated'
                AND ({predicate})
         )",
        predicate = scope.predicate,
    );
    let mut params = scope.params.clone();
    params.extend(scope.params.clone());
    let mut rows = tx
        .query(&sql, libsql::params_from_iter(params))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 grounded node count: {error}")))?;
    let count: i64 = rows
        .next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 grounded node count row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 grounded node count decode: {error}")))?
        .ok_or_else(|| WenlanError::VectorDb("m6 grounded node count: no row returned".into()))?;
    Ok(count.max(0) as usize)
}
