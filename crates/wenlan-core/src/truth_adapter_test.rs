// SPDX-License-Identifier: Apache-2.0
//! Does the protecting half actually protect, and is it still inert at 0.
//!
//! Every test here runs the adapters at BOTH generations against the same
//! substrate. The pair is the point: one half proves the filter bites, the other
//! proves PR-C ships without changing behavior. A test that only ran at
//! generation 1 would pass a build that had quietly gone live.

use crate::db::tests::test_db;
use crate::db::MemoryDB;
use crate::pages::Page;
use crate::truth_adapter::{filter_page, filter_page_refs, filter_pages, page_write_permit};
use crate::truth_contract::TruthGrant;

/// `p1` supported, `p2` provisional, `p3` with no truth row at all -- the
/// post-migration shape, where most pages have no row yet.
async fn db_with_truth_rows() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    for id in ["p1", "p2", "p3"] {
        db.insert_page(id, id, None, "", None, None, &[], "2026-07-28T00:00:00Z")
            .await
            .unwrap();
    }
    {
        let conn = db.test_primary_session().await;
        // `human_reviewed = 1` is CHECK-bound to the version + digest it was
        // reviewed at, so a review cannot outlive the prose it approved.
        conn.execute(
            "INSERT INTO page_truth_state
                (page_id,page_version,support_status,human_reviewed,
                 reviewed_page_version,reviewed_page_digest,updated_at)
             VALUES ('p1',1,'supported',1,1,'digest-of-p1-v1',1)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_truth_state
                (page_id,page_version,support_status,human_reviewed,updated_at)
             VALUES ('p2',1,'provisional',0,1)",
            (),
        )
        .await
        .unwrap();
    }
    db.set_truth_cutover_generation(1).await.unwrap();
    (db, temp)
}

fn page(id: &str) -> Page {
    Page {
        id: id.to_string(),
        title: format!("Title of {id}"),
        summary: Some("a summary".to_string()),
        content: "the body prose".to_string(),
        entity_id: None,
        space: None,
        source_memory_ids: vec!["mem_1".to_string()],
        version: 1,
        status: "active".to_string(),
        created_at: "2026-07-28T00:00:00Z".to_string(),
        last_compiled: "2026-07-28T00:00:00Z".to_string(),
        last_modified: "2026-07-28T00:00:00Z".to_string(),
        sources_updated_count: 0,
        stale_reason: Some("source_updated".to_string()),
        pending_rebuild: None,
        user_edited: false,
        relevance_score: 0.0,
        last_edited_by: None,
        last_edited_at: None,
        last_delta_summary: Some("what changed last time".to_string()),
        changelog: Some("[]".to_string()),
        workspace: None,
        creation_kind: "distilled".to_string(),
        review_status: "confirmed".to_string(),
        citations: Vec::new(),
        kind: "concept".to_string(),
        truth: None,
    }
}

fn pages() -> Vec<Page> {
    vec![page("p1"), page("p2"), page("p3")]
}

// ---- the inertness gate, on every operation -----------------------------

/// The property PR-C is required to preserve. Four operations, one assertion
/// each: at generation 0 nothing filters, nothing reduces, and every permit is
/// granted. If any of these ever goes red, PR-C became a live cutover.
#[tokio::test]
async fn every_adapter_is_pass_through_at_generation_zero() {
    let (db, _tmp) = db_with_truth_rows().await;
    db.set_truth_cutover_generation(0).await.unwrap();

    let kept = filter_pages(&db, &TruthGrant::Automatic, pages())
        .await
        .unwrap();
    assert_eq!(kept.len(), 3, "filter_pages dropped a page at generation 0");
    assert!(
        kept.iter().all(|p| p.content == "the body prose"),
        "filter_pages reduced a page at generation 0"
    );
    assert!(
        kept.iter().all(|p| p.truth.is_none()),
        "generation 0 must not mint truth axes onto the wire"
    );

    assert!(filter_page(&db, &TruthGrant::Automatic, Some(page("p2")))
        .await
        .unwrap()
        .is_some());

    let refs = filter_page_refs(
        &db,
        &TruthGrant::Automatic,
        vec![("p1", 1), ("p2", 2), ("p3", 3)],
        |item| item.0,
    )
    .await
    .unwrap();
    assert_eq!(refs.len(), 3);

    for id in ["p1", "p2", "p3"] {
        assert!(
            page_write_permit(&db, id).await.unwrap().is_some(),
            "a permit was refused at generation 0 for {id}"
        );
    }
}

// ---- filter_pages -------------------------------------------------------

/// The default reader after the cutover: unsupported pages are gone entirely,
/// and a page with no truth row counts as unsupported.
#[tokio::test]
async fn an_automatic_reader_loses_unsupported_pages_entirely() {
    let (db, _tmp) = db_with_truth_rows().await;
    let kept = filter_pages(&db, &TruthGrant::Automatic, pages())
        .await
        .unwrap();
    let ids: Vec<&str> = kept.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(ids, vec!["p1"]);
    assert_eq!(
        kept[0].content, "the body prose",
        "a supported page must come through untouched"
    );
}

/// The collection carve-out, and its precondition. An entry keeps identity and
/// gains both axes; every prose-bearing field is emptied. This is the test that
/// fails if someone "simplifies" the reduction into just blanking `content`.
#[tokio::test]
async fn a_collection_grant_reduces_rather_than_drops_and_carries_both_axes() {
    let (db, _tmp) = db_with_truth_rows().await;
    let kept = filter_pages(&db, &TruthGrant::CollectionEntries, pages())
        .await
        .unwrap();
    assert_eq!(kept.len(), 3, "a collection grant lists every page");

    let p1 = kept.iter().find(|p| p.id == "p1").unwrap();
    assert_eq!(p1.content, "the body prose", "p1 is supported: untouched");
    assert!(p1.truth.is_none(), "a full page is not an entry");

    for id in ["p2", "p3"] {
        let entry = kept.iter().find(|p| p.id == id).unwrap();
        assert_eq!(entry.title, format!("Title of {id}"), "the title survives");
        assert!(entry.content.is_empty(), "{id} kept its prose");
        assert!(entry.summary.is_none(), "{id} kept its summary");
        assert!(entry.changelog.is_none(), "{id} kept its changelog");
        assert!(
            entry.last_delta_summary.is_none(),
            "{id} kept its delta summary"
        );
        assert!(entry.stale_reason.is_none(), "{id} kept its stale reason");
        assert!(
            entry.source_memory_ids.is_empty(),
            "{id} kept the join key back to its evidence"
        );
        let truth = entry
            .truth
            .expect("an entry without both axes is the unearned trust M5 forbids");
        assert!(!truth.supported, "{id} is provisional");
    }

    let p3 = kept.iter().find(|p| p.id == "p3").unwrap();
    assert_eq!(
        p3.truth.unwrap(),
        wenlan_types::pages::PageTruth {
            supported: false,
            human_reviewed: false,
        },
        "a page with no truth row reads as unsupported and unreviewed, not unknown"
    );
}

/// The grant covers the page the call named and nothing riding along beside it.
#[tokio::test]
async fn a_named_grant_opens_only_the_page_the_call_named() {
    let (db, _tmp) = db_with_truth_rows().await;
    let kept = filter_pages(
        &db,
        &TruthGrant::NamedPages(vec!["p2".to_string()]),
        pages(),
    )
    .await
    .unwrap();
    let ids: Vec<&str> = kept.iter().map(|p| p.id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["p1", "p2"],
        "p3 is just as unsupported as p2 and was never named"
    );
    assert_eq!(kept[1].content, "the body prose", "the named page opens");
}

// ---- filter_page --------------------------------------------------------

#[tokio::test]
async fn a_hidden_page_becomes_a_miss_rather_than_an_empty_body() {
    let (db, _tmp) = db_with_truth_rows().await;
    assert!(
        filter_page(&db, &TruthGrant::Automatic, Some(page("p2")))
            .await
            .unwrap()
            .is_none(),
        "an unsupported page must read as absent, so the reader 404s"
    );
    assert!(filter_page(&db, &TruthGrant::Automatic, Some(page("p1")))
        .await
        .unwrap()
        .is_some());
    assert!(filter_page(&db, &TruthGrant::Automatic, None)
        .await
        .unwrap()
        .is_none());
}

// ---- filter_page_refs ---------------------------------------------------

/// There is no entry form for a thing hanging off a page. A revision body is
/// prose; a link exposes a title. So `EntryOnly` drops here, and a collection
/// grant buys nothing.
#[tokio::test]
async fn things_hanging_off_a_page_have_no_entry_form() {
    let (db, _tmp) = db_with_truth_rows().await;
    for grant in [TruthGrant::Automatic, TruthGrant::CollectionEntries] {
        let kept = filter_page_refs(
            &db,
            &grant,
            vec![("p1", "rev1"), ("p2", "rev2"), ("p3", "rev3")],
            |item| item.0,
        )
        .await
        .unwrap();
        assert_eq!(
            kept.len(),
            1,
            "only the supported page's revisions survive under {grant:?}"
        );
        assert_eq!(kept[0].0, "p1");
    }
}

// ---- page_write_permit --------------------------------------------------

/// The seam for the paths with no request behind them: the projection write, the
/// export write, the re-distillation lane. `Automatic` is the only honest grant
/// -- the reader who shows up later declared no contract and made no gesture.
#[tokio::test]
async fn a_permit_is_only_minted_for_a_page_the_automatic_reader_may_see() {
    let (db, _tmp) = db_with_truth_rows().await;
    let permit = page_write_permit(&db, "p1").await.unwrap();
    assert_eq!(permit.expect("p1 is supported").page_id(), "p1");
    assert!(
        page_write_permit(&db, "p2").await.unwrap().is_none(),
        "a provisional page may not be projected where `wenlan pages` reads it"
    );
    assert!(
        page_write_permit(&db, "p3").await.unwrap().is_none(),
        "no truth row is not permission"
    );
}

// ---- empties ------------------------------------------------------------

/// The short-circuits are not just a speed trick: an empty batch must not reach
/// `page_visibility` and come back as a verdict about nothing.
#[tokio::test]
async fn empty_batches_stay_empty_without_asking() {
    let (db, _tmp) = db_with_truth_rows().await;
    assert!(filter_pages(&db, &TruthGrant::Automatic, Vec::new())
        .await
        .unwrap()
        .is_empty());
    let empty: Vec<(&str, u8)> = Vec::new();
    assert!(
        filter_page_refs(&db, &TruthGrant::Automatic, empty, |item| item.0)
            .await
            .unwrap()
            .is_empty()
    );
}
