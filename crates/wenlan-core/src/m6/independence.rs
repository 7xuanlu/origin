// SPDX-License-Identifier: Apache-2.0
//! D1 — the relaxed independence floor (spec §3.1).
//!
//! One canonical distinct-group count expression. Every signal in
//! `signals.rs` narrows it with a per-signal scope and reads the result;
//! there is no second count anywhere in the M6 tree (I-2's lesson: the two
//! partial unique indexes are the *entire* D4 exclusion mechanism, and a
//! duplicate code-level check would be a second, weaker one — the same
//! reasoning applies here to independence counting).
//!
//! ```sql
//! COUNT(DISTINCT r.independence_group_id)
//!   FROM edges e JOIN provenance_roots r ON r.root_id = e.root_id
//!  WHERE e.grounded = 1 AND e.valid_until IS NULL
//!    AND r.status = 'active' AND r.root_kind <> 'generated'
//!    AND <per-signal scope>
//! ```
//!
//! R1-R5 are fixed in this module's SQL and shared by every scope. R6-R8
//! are not code here: R6 (unknown independence) is the join silently
//! dropping an edge whose `root_id` does not resolve, which leaves the
//! candidate below the floor rather than admitted (S0-20/Q5 names this gap
//! without closing it — nothing in PR-B papers over it); R7 (human capture
//! collapsing to one group) is a write-time property of
//! `provenance_roots.independence_group_id` this module only reads; R8
//! (page-mediated scope) is a caller-supplied [`GroupCountScope`].

use crate::WenlanError;

/// D1's admission floor (§3.1): a candidate needs at least this many
/// distinct independence groups over its supporting evidence.
pub const INDEPENDENCE_GROUP_FLOOR: i64 = 3;

/// R1-R5 as one string, so a second reader of the eligibility spine cannot
/// drift from the count.
///
/// This module's header says there is no second count anywhere in the M6 tree.
/// C1's relevance sweep is not a second *count* — it reads the (group, page)
/// support relation rather than `COUNT(DISTINCT …)` — but it must select over
/// exactly the same eligible edges, or a group could be independent enough to
/// clear D1's floor and simultaneously contribute nothing to a pair cell. The
/// predicate lives here rather than being spelled twice, because two spellings
/// that must agree forever is the shape the header warns about.
///
/// References only `e` (`edges`) and `r` (`provenance_roots`), the same
/// contract [`GroupCountScope`] carries.
pub const ELIGIBLE_EVIDENCE_PREDICATE: &str = "e.grounded = 1
            AND e.valid_until IS NULL
            AND r.status = 'active'
            AND r.root_kind <> 'generated'
            AND NOT EXISTS (
                SELECT 1 FROM pages p
                 WHERE ((e.src_kind = 'page' AND p.id = e.src_id)
                        OR (e.dst_kind = 'page' AND p.id = e.dst_id))
                   AND lower(p.title) = 'overview'
            )";

/// A per-signal narrowing of the canonical count, ANDed onto R1-R5.
///
/// `predicate` may reference only `e` (`edges`) and `r` (`provenance_roots`)
/// — this module owns R1-R5 and the `FROM`/`JOIN` shape, so a scope that
/// needed a wider join could silently relax one of them. A scope that needs
/// another table reaches it through a correlated `EXISTS` subquery instead
/// (see the per-signal scopes in `signals.rs`).
pub struct GroupCountScope<'a> {
    pub predicate: &'a str,
    pub params: Vec<libsql::Value>,
}

impl GroupCountScope<'_> {
    /// No extra narrowing — every grounded, active, non-generated,
    /// non-overview root counts once. Used by tests that exercise R1-R5 in
    /// isolation, and by any caller that already scoped its edges upstream.
    pub fn unscoped() -> GroupCountScope<'static> {
        GroupCountScope {
            predicate: "1=1",
            params: Vec::new(),
        }
    }
}

/// D1's canonical distinct-group count (§3.1).
///
/// `tx` is caller-owned (this module opens no transaction, per §2.1: PR-B's
/// reader modules take a `&libsql::Transaction` and write nothing).
pub async fn distinct_group_count(
    tx: &libsql::Transaction,
    scope: &GroupCountScope<'_>,
) -> Result<i64, WenlanError> {
    let sql = format!(
        "SELECT COUNT(DISTINCT r.independence_group_id)
           FROM edges e
           JOIN provenance_roots r ON r.root_id = e.root_id
          WHERE {ELIGIBLE_EVIDENCE_PREDICATE}
            AND ({predicate})",
        predicate = scope.predicate,
    );
    let mut rows = tx
        .query(&sql, libsql::params_from_iter(scope.params.clone()))
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 independence count: {error}")))?;
    rows.next()
        .await
        .map_err(|error| WenlanError::VectorDb(format!("m6 independence count row: {error}")))?
        .map(|row| row.get::<i64>(0))
        .transpose()
        .map_err(|error| WenlanError::VectorDb(format!("m6 independence count decode: {error}")))?
        .ok_or_else(|| WenlanError::VectorDb("m6 independence count: no row returned".into()))
}

/// Whether a count clears D1's floor.
pub fn admits(group_count: i64) -> bool {
    group_count >= INDEPENDENCE_GROUP_FLOOR
}
