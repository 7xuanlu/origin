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

/// `p1` supported and human-reviewed, `p2` and `p3` both judged and failed.
///
/// `p3` used to have no truth row at all, standing in for "the post-migration
/// shape, where most pages have no row yet". It cannot play the unsupported
/// role any more: a page with no row has never been judged, and an unjudged
/// page keeps its prose. It is a second *failed* page now, so the assertions
/// below still test what they claim to. The no-row case has its own tooth --
/// see `a_page_with_no_truth_row_is_unjudged_not_condemned`.
async fn db_with_truth_rows() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    for id in ["p1", "p2", "p3"] {
        db.insert_page(
            id,
            id,
            None,
            "the body prose",
            None,
            None,
            &[],
            "2026-07-28T00:00:00Z",
        )
        .await
        .unwrap();
    }
    {
        let conn = db.test_primary_session().await;
        // `human_reviewed = 1` is CHECK-bound to the version + digest it was
        // reviewed at, so a review cannot outlive the prose it approved. The
        // digest here has to be the real one for the page's real body -- a
        // review of anything else is a review of text that is not on the page,
        // and `page_truth_states` will not count it.
        let digest = crate::provenance::revision_content_digest("the body prose");
        conn.execute(
            &format!(
                "INSERT INTO page_truth_state
                    (page_id,page_version,support_status,human_reviewed,
                     reviewed_page_version,reviewed_page_digest,updated_at)
                 VALUES ('p1',1,'supported',1,1,'{digest}',1)"
            ),
            (),
        )
        .await
        .unwrap();
        // `evaluated_at` is what makes p2 a *failed* judgement rather than an
        // unjudged page. Without it these tests would assert hiding against a
        // page that is merely unexamined, and unexamined pages are deliberately
        // never hidden -- see `an_unjudged_page_is_never_hidden_no_matter_who_is_asking`.
        conn.execute(
            "INSERT INTO page_truth_state
                (page_id,page_version,support_status,human_reviewed,updated_at,
                 evaluated_at)
             VALUES ('p2',1,'provisional',0,1,1)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_truth_state
                (page_id,page_version,support_status,human_reviewed,updated_at,
                 evaluated_at)
             VALUES ('p3',1,'provisional',0,1,1)",
            (),
        )
        .await
        .unwrap();
        // A verdict is only readable while the derivation it came out of still
        // describes the page, so each of these rows needs the marker
        // derivation would have written before finalizing. A truth row without
        // one is a state
        // the pipeline does not produce, and seeding it would leave every
        // assertion below testing the missing-marker path instead of its
        // subject. Digest over the page's actual prose and this build's
        // extractor, so an edit or a version bump invalidates it exactly as it
        // would in production.
        for id in ["p1", "p2", "p3"] {
            conn.execute(
                "INSERT OR REPLACE INTO claim_derivation_markers
                     (page_id, page_version, page_version_digest, extractor_version,
                      inventory_count, created_at)
                 VALUES (?1, 1, ?2, ?3, 1, 0)",
                libsql::params![
                    id,
                    crate::provenance::revision_content_digest("the body prose"),
                    crate::db::EXTRACTOR_VERSION
                ],
            )
            .await
            .unwrap();
        }

        // `supported` is a materialized receipt, not authority by itself. The
        // read gate also requires the current claim inventory and one live edge
        // from an eligible judge triple, so the positive p1 fixture must carry
        // that real substrate.
        conn.execute(
            "INSERT INTO claims (claim_id, page_id, created_at)
             VALUES ('claim_p1', 'p1', 0)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO claim_revisions
                 (claim_revision_id, claim_id, predecessor_revision_id,
                  canonical_text, canonical_text_digest, claim_kind,
                  extractor_version, created_at)
             VALUES ('revision_p1', 'claim_p1', '', 'claim', ?1, 'factual', ?2, 0)",
            libsql::params![
                crate::provenance::revision_content_digest("claim"),
                crate::db::EXTRACTOR_VERSION
            ],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO page_version_claims
                 (page_id, page_version, claim_revision_id, ordinal)
             VALUES ('p1', 1, 'revision_p1', 0)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO claim_judge_eligibility
                 (model_id, model_version, prompt_version, state, threshold, generation)
             VALUES ('adapter-judge', 'v1', 'p1', 'active', 0.75, 1)",
            (),
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO memories
                 (id, content, source, source_id, title, chunk_index,
                  last_modified, chunk_type, space)
             VALUES ('adapter_memory_p1', 'evidence', 'memory',
                     'adapter_memory_p1', 'evidence', 0, 0, 'text', ?1)",
            libsql::params![crate::db::UNFILED_SPACE_ID],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO provenance_roots
                 (root_id, identity_version, identity_digest, root_kind,
                  independence_group_id, status, created_at)
             VALUES ('adapter_root_p1', ?1, ?2, 'document_ingest',
                     'adapter_group_p1', 'active', 0)",
            libsql::params![
                crate::provenance::IDENTITY_VERSION,
                crate::provenance::identity_digest("document_ingest", "evidence")
            ],
        )
        .await
        .unwrap();
        let payload = serde_json::json!({
            "model_id": "adapter-judge",
            "model_version": "v1",
            "prompt_version": "p1",
            "score": 0.9,
            "threshold_at_write": 0.75,
        })
        .to_string();
        conn.execute(
            "INSERT INTO edges
                 (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type,
                  lineage, grounded, root_id, space, payload, created_at,
                  superseded_by, valid_until)
             VALUES ('adapter_edge_p1', 'revision_p1', 'claim_revision',
                     'adapter_memory_p1', 'memory', 'supports', 'evidence', 0,
                     'adapter_root_p1', ?1, ?2, 0, NULL, NULL)",
            libsql::params![crate::db::UNFILED_SPACE_ID, payload],
        )
        .await
        .unwrap();
    }
    db.set_truth_cutover_generation(1).await.unwrap();
    // These tests exercise the strict adapter contract. Production promoter
    // results remain advisory unless this independent switch is explicit.
    db.set_app_metadata("claim_promoter_enforcement", "1")
        .await
        .unwrap();
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
        refresh_blocked_reason: None,
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

/// The other half of what `NamedPages` is defined to be: "Full prose, both
/// axes, for exactly these page IDs."
///
/// The prose half was enforced and the axes half was not, because `Full`
/// visibility was treated as the identity — so a client that declared the
/// contract, gestured, and opened one page got the page and no way to tell
/// whether it was supported or reviewed. That is the same unearned trust the
/// collection carve-out spells out for entries, arriving through the door that
/// serves the whole page.
#[tokio::test]
async fn a_named_grant_carries_both_axes_for_exactly_the_named_pages() {
    let (db, _tmp) = db_with_truth_rows().await;
    let kept = filter_pages(
        &db,
        &TruthGrant::NamedPages(vec!["p1".to_string(), "p2".to_string()]),
        pages(),
    )
    .await
    .unwrap();

    let p1 = kept.iter().find(|p| p.id == "p1").unwrap();
    assert_eq!(
        p1.content, "the body prose",
        "a named page keeps every prose field"
    );
    assert_eq!(
        p1.truth.expect("a named page must carry both axes"),
        wenlan_types::pages::PageTruth {
            supported: true,
            human_reviewed: true,
        },
    );

    let p2 = kept.iter().find(|p| p.id == "p2").unwrap();
    assert_eq!(
        p2.content, "the body prose",
        "an unsupported page the call named still opens in full"
    );
    assert_eq!(
        p2.truth.expect("a named page must carry both axes"),
        wenlan_types::pages::PageTruth {
            supported: false,
            human_reviewed: false,
        },
        "and says so, rather than opening silently"
    );
}

/// "For exactly these page IDs" is a limit as well as a promise. A page that
/// survives on its own support while riding along beside a named one was never
/// named, so it gets no axes — the same boundary that stops a named grant from
/// opening it.
#[tokio::test]
async fn a_page_riding_along_beside_a_named_one_gets_no_axes() {
    let (db, _tmp) = db_with_truth_rows().await;
    let kept = filter_pages(
        &db,
        &TruthGrant::NamedPages(vec!["p2".to_string()]),
        pages(),
    )
    .await
    .unwrap();

    let p1 = kept.iter().find(|p| p.id == "p1").unwrap();
    assert!(
        p1.truth.is_none(),
        "p1 came through on its own support, not on a grant that named it"
    );
    assert!(
        kept.iter().find(|p| p.id == "p2").unwrap().truth.is_some(),
        "the page the call actually named carries its axes"
    );
}

/// The inertness gate, held for the new minting site too. Before the cutover
/// there is no verdict to report, and a client that saw `supported: false` on
/// every page would be reading a judgement nobody has made.
#[tokio::test]
async fn a_named_grant_mints_no_axes_before_the_cutover() {
    let (db, _tmp) = db_with_truth_rows().await;
    db.set_truth_cutover_generation(0).await.unwrap();

    let kept = filter_pages(
        &db,
        &TruthGrant::NamedPages(vec!["p1".to_string(), "p2".to_string()]),
        pages(),
    )
    .await
    .unwrap();
    assert_eq!(kept.len(), 3, "nothing is filtered at generation 0");
    assert!(
        kept.iter().all(|p| p.truth.is_none()),
        "generation 0 must not mint truth axes onto the wire, named or not"
    );
}

/// An automatic reader's wire stays byte-identical after the cutover. It
/// declared no contract and made no gesture, so it renders no axes and must not
/// be handed any.
#[tokio::test]
async fn an_automatic_reader_gets_no_axes_after_the_cutover() {
    let (db, _tmp) = db_with_truth_rows().await;
    let kept = filter_pages(&db, &TruthGrant::Automatic, pages())
        .await
        .unwrap();
    assert!(
        kept.iter().all(|p| p.truth.is_none()),
        "an automatic reader must see the pre-M5 page shape"
    );
}

// ---- filter_page --------------------------------------------------------

/// The by-id reader is the one a person actually opens a page through, and it
/// routes through `filter_pages`, so the axes have to survive the single-page
/// path as well as the batch one.
#[tokio::test]
async fn the_by_id_reader_carries_both_axes_for_the_page_it_named() {
    let (db, _tmp) = db_with_truth_rows().await;
    let opened = filter_page(
        &db,
        &TruthGrant::NamedPages(vec!["p2".to_string()]),
        Some(page("p2")),
    )
    .await
    .unwrap()
    .expect("a named unsupported page opens");

    assert_eq!(opened.content, "the body prose");
    assert_eq!(
        opened
            .truth
            .expect("the page a call named must arrive with both axes"),
        wenlan_types::pages::PageTruth {
            supported: false,
            human_reviewed: false,
        },
    );
}

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

// ---- activity detail, the subject that cannot be named -------------------

fn activity(action: &str, detail: &str) -> wenlan_types::memory::AgentActivityRow {
    wenlan_types::memory::AgentActivityRow {
        id: 1,
        timestamp: 0,
        agent_name: "claude".into(),
        action: action.into(),
        memory_ids: None,
        query: None,
        detail: Some(detail.into()),
        memory_titles: Vec::new(),
    }
}

/// Paired like the rest: inert at 0, biting at 1.
///
/// The `page_grow` row carries a title in prose with no page id anywhere on the
/// row, which is the whole reason this adapter refuses instead of ruling. The
/// `memory_store` row is the control -- refusing every activity detail would
/// pass a blanket-nulling implementation just as happily.
#[tokio::test]
async fn page_activity_detail_survives_at_zero_and_is_refused_at_one() {
    let (db, _tmp) = db_with_truth_rows().await;
    let rows = || {
        vec![
            activity("page_grow", "grew \"Q3 Revenue Plan\""),
            activity("page_create", "title=Q3 Revenue Plan, sources=4"),
            activity("memory_store", "stored a capture"),
        ]
    };

    db.set_truth_cutover_generation(0).await.unwrap();
    let kept = crate::truth_adapter::redact_page_activity_detail(&db, rows())
        .await
        .unwrap();
    assert!(
        kept.iter().all(|r| r.detail.is_some()),
        "generation 0 must not redact anything"
    );

    db.set_truth_cutover_generation(1).await.unwrap();
    let redacted = crate::truth_adapter::redact_page_activity_detail(&db, rows())
        .await
        .unwrap();
    assert_eq!(
        redacted[0].detail, None,
        "a page title reached a reader through agent_activity.detail"
    );
    assert_eq!(
        redacted[1].detail, None,
        "a page title reached a reader through agent_activity.detail"
    );
    assert!(
        redacted[2].detail.is_some(),
        "a non-page activity lost its detail — the adapter is nulling blindly"
    );
}

/// The prefix test, not an enumerated list. A page action added after this ships
/// is covered by default; the alternative leaks until someone remembers to
/// extend a match arm.
#[tokio::test]
async fn an_unknown_page_action_is_covered_by_the_prefix() {
    let (db, _tmp) = db_with_truth_rows().await;
    db.set_truth_cutover_generation(1).await.unwrap();
    let redacted = crate::truth_adapter::redact_page_activity_detail(
        &db,
        vec![activity(
            "page_something_invented_later",
            "title=Secret Page",
        )],
    )
    .await
    .unwrap();
    assert_eq!(redacted[0].detail, None);
}
