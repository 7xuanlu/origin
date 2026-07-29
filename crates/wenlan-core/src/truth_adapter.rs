// SPDX-License-Identifier: Apache-2.0
//! The protecting half of the M5 exposure contract.
//!
//! PR-B installed the decision (`truth_contract`), the durable substrate
//! (`db::truth_exposure`) and the wire guard (`wenlan-server::truth_guard`). What
//! it did not install was anything that *acts* on the verdict: `page_visibility`
//! had exactly one production caller, the projection invariant, which deletes
//! files. The destructive half was wired and the protective half was not, so
//! advancing the cutover generation would have evicted pages from the vault while
//! every page route kept serving them.
//!
//! This module is that missing half: four operations, each taking the grant the
//! guard resolved, that every page-bearing reader routes its pages through.
//!
//! | operation | shape | callers |
//! |---|---|---|
//! | [`filter_pages`] | `Vec<Page>` -> `Vec<Page>` | list + search readers |
//! | [`filter_page`] | `Option<Page>` -> `Option<Page>` | by-id readers |
//! | [`filter_page_refs`] | `Vec<T>` keyed by page id -> `Vec<T>` | revisions, links, changes, retrievals |
//! | [`page_write_permit`] | one page id -> [`PagePermit`] | projection writes, exports, re-distillation |
//! | [`redact_page_activity_detail`] | rows naming a page with no id -> blanked | `/api/activities` |
//!
//! **Still inert.** At generation 0 `page_visibility` answers `Full` for every
//! page, so all four are the identity and this PR changes no behavior — the same
//! property PR-B had, now held by considerably more code. The generation is
//! advanced by the ceremony, which comes after this, deliberately.
//!
//! Binding spec: `docs/plans/2026-07-28-m5-prc-adapters.md`.

use crate::db::MemoryDB;
use crate::pages::Page;
use crate::truth_contract::{TruthGrant, Visibility};
use crate::WenlanError;
use std::collections::HashMap;

/// Strip every prose-bearing field, keep identity and state.
///
/// What survives is what `TruthGrant::CollectionEntries` is defined to allow:
/// the page ID, its title, and both truth axes. Everything a reader could quote
/// goes -- summary, body, citations, changelog, the delta summary, and the
/// stale/rebuild strings, which name why a page changed and are prose about the
/// page even though they are not the page.
///
/// `truth` is set here and nowhere else. An entry without both axes is the
/// unearned trust this whole rung exists to prevent, so the reduction that
/// creates entries is the only thing allowed to mint one.
fn reduce_to_entry(mut page: Page, truth: wenlan_types::pages::PageTruth) -> Page {
    page.summary = None;
    page.content = String::new();
    page.citations = Vec::new();
    page.changelog = None;
    page.last_delta_summary = None;
    page.stale_reason = None;
    page.pending_rebuild = None;
    // Source IDs are not prose, but they are the join key back to the memories a
    // provisional page was distilled from, and handing them out turns a listing
    // into a retrieval plan.
    page.source_memory_ids = Vec::new();
    page.truth = Some(truth);
    page
}

/// The verdict for a batch, plus the axes needed to render any entry it allows.
///
/// One `page_visibility` call always; a second query for the axes only when the
/// verdict actually produced an entry, which happens on a marked collection call
/// after the cutover and never before it.
async fn verdicts(
    db: &MemoryDB,
    grant: &TruthGrant,
    ids: &[String],
) -> Result<
    (
        HashMap<String, Visibility>,
        HashMap<String, wenlan_types::pages::PageTruth>,
    ),
    WenlanError,
> {
    let visibility = db.page_visibility(grant, ids).await?;
    let entry_ids: Vec<String> = visibility
        .iter()
        .filter(|(_, v)| **v == Visibility::EntryOnly)
        .map(|(id, _)| id.clone())
        .collect();
    if entry_ids.is_empty() {
        return Ok((visibility, HashMap::new()));
    }
    let axes = db
        .page_truth_states(&entry_ids)
        .await?
        .into_iter()
        .map(|(id, state)| {
            (
                id,
                wenlan_types::pages::PageTruth {
                    supported: state.supported,
                    human_reviewed: state.human_reviewed,
                },
            )
        })
        .collect();
    Ok((visibility, axes))
}

/// The axes for a page that has no `page_truth_state` row.
///
/// Unsupported and unreviewed. The absence of a support record is not evidence
/// of support, and post-migration that absence is the normal case rather than an
/// anomaly -- so the default has to be the restrictive one, not "we do not know".
fn unknown_axes() -> wenlan_types::pages::PageTruth {
    wenlan_types::pages::PageTruth {
        supported: false,
        human_reviewed: false,
    }
}

/// Reduce a page batch to what this call may see.
///
/// `Hidden` drops the page entirely -- not its title, not an ID with content
/// hanging off it. `EntryOnly` reduces it. `Full` is the identity.
///
/// A page whose ID somehow carries no verdict is dropped rather than served: the
/// map is total over its input by construction, so this only fires if that ever
/// stops being true, and a missing verdict is not permission.
pub async fn filter_pages(
    db: &MemoryDB,
    grant: &TruthGrant,
    pages: Vec<Page>,
) -> Result<Vec<Page>, WenlanError> {
    if pages.is_empty() {
        return Ok(pages);
    }
    let ids: Vec<String> = pages.iter().map(|p| p.id.clone()).collect();
    let (visibility, axes) = verdicts(db, grant, &ids).await?;
    Ok(pages
        .into_iter()
        .filter_map(|page| match visibility.get(&page.id) {
            Some(Visibility::Full) => Some(page),
            Some(Visibility::EntryOnly) => {
                let truth = axes.get(&page.id).copied().unwrap_or_else(unknown_axes);
                Some(reduce_to_entry(page, truth))
            }
            _ => None,
        })
        .collect())
}

/// The by-id form. `None` in, `None` out; hidden in, `None` out.
///
/// Collapsing hidden to `None` is what makes the by-id readers return their
/// existing 404 rather than needing a new status of their own -- and a 404 is
/// the honest answer, because for this caller the page is not there.
pub async fn filter_page(
    db: &MemoryDB,
    grant: &TruthGrant,
    page: Option<Page>,
) -> Result<Option<Page>, WenlanError> {
    let Some(page) = page else {
        return Ok(None);
    };
    Ok(filter_pages(db, grant, vec![page])
        .await?
        .into_iter()
        .next())
}

/// Drop every item whose page the caller may not see in full.
///
/// For things that hang off a page rather than being one: revisions, links,
/// changelog entries, retrieval records, map nodes. There is no entry form for
/// these -- a revision body is prose and a link exposes a title -- so anything
/// short of `Full` drops. `EntryOnly` is a carve-out for listings of pages, not
/// a licence to serve their contents in another shape.
pub async fn filter_page_refs<T>(
    db: &MemoryDB,
    grant: &TruthGrant,
    items: Vec<T>,
    page_id: impl Fn(&T) -> &str,
) -> Result<Vec<T>, WenlanError> {
    if items.is_empty() {
        return Ok(items);
    }
    let ids: Vec<String> = items.iter().map(|i| page_id(i).to_string()).collect();
    let visibility = db.page_visibility(grant, &ids).await?;
    Ok(items
        .into_iter()
        .filter(|item| visibility.get(page_id(item)) == Some(&Visibility::Full))
        .collect())
}

/// Proof that one page may be written where an automatic reader will find it.
///
/// Minted only by [`page_write_permit`] -- the field is private, so a caller
/// cannot fabricate one, and a write path that takes `&PagePermit` cannot be
/// reached without having asked. That is the point: the projection write, the
/// export write and the re-distillation lane all sit outside the request path,
/// where there is no guard to forget to call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagePermit {
    page_id: String,
}

impl PagePermit {
    /// Which page this permit is for. A write path must check it against the
    /// page in hand; a permit for page A is not permission to write page B.
    pub fn page_id(&self) -> &str {
        &self.page_id
    }
}

/// May this page be written somewhere an automatic reader can reach it?
///
/// `Automatic` is the only honest grant here. These are not request paths: a
/// projection file, an exported vault entry and a re-distilled body are all read
/// later by someone who declared no contract and made no gesture, so the write
/// has to satisfy the reader who will show up, not the process doing the writing.
///
/// `None` means no. It is not an error -- declining to project an unsupported
/// page is the contract working, and callers log it and move on.
pub async fn page_write_permit(
    db: &MemoryDB,
    page_id: &str,
) -> Result<Option<PagePermit>, WenlanError> {
    let ids = vec![page_id.to_string()];
    let visibility = db.page_visibility(&TruthGrant::Automatic, &ids).await?;
    Ok(if visibility.get(page_id) == Some(&Visibility::Full) {
        Some(PagePermit {
            page_id: page_id.to_string(),
        })
    } else {
        None
    })
}

/// Blank activity detail that names a page but carries no page id.
///
/// `agent_activity` is `id, timestamp, agent_name, action, memory_ids, query,
/// detail` -- no page reference anywhere, and `memory_ids` holds memory ids.
/// Four write sites format a page title straight into the free-text `detail`
/// (`post_write.rs` x3, `post_ingest.rs` x1), so by the time a reader asks, the
/// title is a sentence and the identity that would let `page_visibility` rule on
/// it is gone.
///
/// There is nothing to key the decision on, so the only fail-closed answer is to
/// refuse the whole sentence rather than guess which page it names. The prefix
/// test is `page_`, not an enumerated list of the four actions, so an action
/// added later is covered by default rather than leaking until someone
/// remembers to extend a match arm.
///
/// **Coarse on purpose, and it takes no grant.** A reader holding a marker loses
/// the detail line too, because "which page is this about" is not a question the
/// row can answer -- no grant can be applied to an unidentified subject. The
/// alternative considered was a `page_id` column plus a migration nulling
/// historical `detail`; it was rejected because it destroys existing activity
/// text and still leaves every row written before it shipped.
///
/// Inert at generation 0, like the other four operations.
pub async fn redact_page_activity_detail(
    db: &MemoryDB,
    mut rows: Vec<wenlan_types::memory::AgentActivityRow>,
) -> Result<Vec<wenlan_types::memory::AgentActivityRow>, WenlanError> {
    if db.truth_cutover_generation().await? < 1 {
        return Ok(rows);
    }
    for row in &mut rows {
        if row.action.starts_with("page_") {
            row.detail = None;
        }
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "truth_adapter_test.rs"]
mod tests;
