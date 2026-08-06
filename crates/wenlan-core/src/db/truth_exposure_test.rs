// SPDX-License-Identifier: Apache-2.0
//! The durable half: does a marked call leave a trail, and is the cutover
//! genuinely off until someone advances it.

use crate::db::tests::test_db;
use crate::db::MemoryDB;
use crate::truth_contract::{MarkerOutcome, TruthGrant, Visibility};

#[tokio::test]
async fn a_fresh_database_has_the_cutover_off() {
    let (db, _tmp) = test_db().await;
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
}

/// The value PR-B ships with. If this ever reads non-zero on a fresh database,
/// every adapter has quietly gone live without the PR-C ceremony.
#[tokio::test]
async fn the_cutover_generation_round_trips_and_rolls_back() {
    let (db, _tmp) = test_db().await;
    db.set_truth_cutover_generation(1).await.unwrap();
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 1);
    db.set_truth_cutover_generation(0).await.unwrap();
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
}

/// A generation written as garbage reads as off, not as on. The gate fails
/// toward inert in every ambiguous case, mirroring the retired cutover
/// levers' fail-toward-inert convention.
#[tokio::test]
async fn an_unparseable_generation_reads_as_off() {
    let (db, _tmp) = test_db().await;
    db.set_app_metadata(super::TRUTH_CUTOVER_GENERATION_KEY, "soon")
        .await
        .unwrap();
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
}

#[tokio::test]
async fn a_marked_call_leaves_caller_pages_and_a_timestamp() {
    let (db, _tmp) = test_db().await;
    let before = chrono::Utc::now().timestamp();
    db.record_truth_marker(
        "claude-code",
        "GET",
        "/api/pages/{id}",
        MarkerOutcome::GrantedNamedPage,
        &["pg1".to_string()],
    )
    .await
    .unwrap();

    let rows = db.recent_truth_markers(10).await.unwrap();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.caller, "claude-code");
    assert_eq!(row.method, "GET");
    assert_eq!(row.path, "/api/pages/{id}");
    assert_eq!(row.outcome, "granted_named_page");
    assert_eq!(row.page_ids, vec!["pg1".to_string()]);
    assert!(row.marked_at >= before);
}

/// The enumerate-then-fetch pattern the shape gate concedes. The audit log is
/// the only thing that makes it visible, so it has to survive being read back
/// in a form that shows the shape of the walk.
#[tokio::test]
async fn page_at_a_time_extraction_reads_back_as_a_pattern() {
    let (db, _tmp) = test_db().await;
    db.record_truth_marker(
        "agent-x",
        "GET",
        "/api/pages",
        MarkerOutcome::GrantedCollection,
        &[],
    )
    .await
    .unwrap();
    for id in ["pg1", "pg2", "pg3"] {
        db.record_truth_marker(
            "agent-x",
            "GET",
            "/api/pages/{id}",
            MarkerOutcome::GrantedNamedPage,
            &[id.to_string()],
        )
        .await
        .unwrap();
    }

    let rows = db.recent_truth_markers(10).await.unwrap();
    assert_eq!(rows.len(), 4);
    let named: Vec<String> = rows
        .iter()
        .filter(|r| r.outcome == "granted_named_page")
        .flat_map(|r| r.page_ids.clone())
        .collect();
    assert_eq!(
        named.len(),
        3,
        "each fetched page is separately attributable"
    );
    assert!(rows.iter().all(|r| r.caller == "agent-x"));
    assert!(
        rows.iter().any(|r| r.outcome == "granted_collection"),
        "the enumerating call is recorded too, or the walk has no beginning"
    );
}

#[tokio::test]
async fn a_refused_call_is_recorded_as_such() {
    let (db, _tmp) = test_db().await;
    db.record_truth_marker(
        "agent-x",
        "POST",
        "/api/context",
        MarkerOutcome::Refused,
        &[],
    )
    .await
    .unwrap();
    let rows = db.recent_truth_markers(10).await.unwrap();
    assert_eq!(rows[0].outcome, "refused");
    assert!(rows[0].page_ids.is_empty());
}

// ---- the read path, exercised against the real table --------------------
//
// Strict-policy tests explicitly enable both the cutover and machine
// enforcement. The shadow control removes only the latter and proves the same
// durable verdict cannot hide a page by default.

/// `p1` supported, `p2` provisional, `p3` with no truth row at all -- the
/// post-migration shape, where most pages have no row yet.
async fn db_with_truth_rows() -> (MemoryDB, tempfile::TempDir) {
    let (db, temp) = test_db().await;
    for id in ["p1", "p2", "p3"] {
        db.insert_page(id, id, None, "", None, None, &[], "2026-07-27T00:00:00Z")
            .await
            .unwrap();
    }
    {
        let conn = db.conn.lock().await;
        let tx = conn.transaction().await.unwrap();
        MemoryDB::ensure_claim_identity_tables(&tx).await.unwrap();
        tx.commit().await.unwrap();
        set_truth(&conn, "p1", "supported").await;
        set_truth(&conn, "p2", "provisional").await;
    }
    db.set_truth_cutover_generation(1).await.unwrap();
    // Most tests below exercise the strict policy itself. Production keeps
    // promoter results advisory until this separate switch is explicitly on.
    db.set_app_metadata("claim_promoter_enforcement", "1")
        .await
        .unwrap();
    (db, temp)
}

/// Seed a page whose truth has actually been decided.
///
/// `evaluated_at` is always stamped, because these tests are about what a
/// *verdict* does. An unjudged page (NULL) keeps full visibility by design, so
/// seeding one here would quietly turn every hiding assertion below into a test
/// of nothing.
///
/// The derivation marker goes in for the same reason: the read gate refuses to
/// expose a verdict whose derivation cannot be shown to describe the live text,
/// and finalization is what writes the marker in production. A truth row without
/// one is a state the pipeline does not produce, and seeding it would leave
/// every assertion below testing the missing-marker path instead of its subject.
/// The digest is taken from the page's own content rather than hardcoded, so a
/// test that rewrites the prose invalidates the marker exactly as an edit does.
async fn set_truth(conn: &libsql::Connection, page_id: &str, status: &str) {
    conn.execute(
        "INSERT INTO page_truth_state
            (page_id,page_version,support_status,human_reviewed,updated_at,
             evaluated_at)
         VALUES (?1,1,?2,0,1,1)",
        libsql::params![page_id, status],
    )
    .await
    .unwrap();
    let content: String = {
        let mut rows = conn
            .query(
                "SELECT content FROM pages WHERE id = ?1",
                libsql::params![page_id],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };
    conn.execute(
        "INSERT OR REPLACE INTO claim_derivation_markers
             (page_id, page_version, page_version_digest, extractor_version,
              inventory_count, created_at)
         VALUES (?1, 1, ?2, ?3, 1, 0)",
        libsql::params![
            page_id,
            crate::provenance::revision_content_digest(&content),
            crate::db::claim_derivation::EXTRACTOR_VERSION
        ],
    )
    .await
    .unwrap();

    if status == "supported" {
        seed_live_support(conn, page_id).await;
    }
}

/// The smallest real substrate a stored `supported` row must still be able to
/// prove at read time: one current claim, one live support edge, and the exact
/// judge triple registered under its current threshold.
async fn seed_live_support(conn: &libsql::Connection, page_id: &str) {
    let claim_id = format!("claim_{page_id}");
    let revision_id = format!("revision_{page_id}");
    let memory_id = format!("support_{page_id}");
    let edge_id = format!("edge_{page_id}");
    let space: String = {
        let mut rows = conn
            .query(
                "SELECT space FROM pages WHERE id = ?1",
                libsql::params![page_id],
            )
            .await
            .unwrap();
        rows.next().await.unwrap().unwrap().get(0).unwrap()
    };

    conn.execute(
        "INSERT INTO claims (claim_id, page_id, created_at) VALUES (?1, ?2, 0)",
        libsql::params![claim_id.clone(), page_id],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO claim_revisions
             (claim_revision_id, claim_id, predecessor_revision_id, canonical_text,
              canonical_text_digest, claim_kind, extractor_version, created_at)
         VALUES (?1, ?2, '', 'claim', ?3, 'factual', ?4, 0)",
        libsql::params![
            revision_id.clone(),
            claim_id,
            crate::provenance::revision_content_digest("claim"),
            crate::db::claim_derivation::EXTRACTOR_VERSION
        ],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO page_version_claims
             (page_id, page_version, claim_revision_id, ordinal)
         VALUES (?1, 1, ?2, 0)",
        libsql::params![page_id, revision_id.clone()],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO memories
             (id, content, source, source_id, title, chunk_index, last_modified,
              chunk_type, space)
         VALUES (?1, 'evidence', 'memory', ?1, 'evidence', 0, 0, 'text', ?2)",
        libsql::params![memory_id.clone(), space.clone()],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT OR IGNORE INTO provenance_roots
             (root_id, identity_version, identity_digest, root_kind,
              independence_group_id, status, created_at)
         VALUES ('truth_read_root', ?1, ?2, 'document_ingest',
                 'truth_read_group', 'active', 0)",
        libsql::params![
            crate::provenance::IDENTITY_VERSION,
            crate::provenance::identity_digest("document_ingest", "evidence")
        ],
    )
    .await
    .unwrap();
    conn.execute(
        "INSERT INTO claim_judge_eligibility
             (model_id, model_version, prompt_version, state, threshold, generation)
         VALUES ('truth-read-judge', 'v1', 'p1', 'active', 0.75, 1)
         ON CONFLICT(model_id, model_version, prompt_version) DO UPDATE SET
             state = excluded.state,
             threshold = excluded.threshold,
             generation = excluded.generation",
        (),
    )
    .await
    .unwrap();
    let payload = serde_json::json!({
        "model_id": "truth-read-judge",
        "model_version": "v1",
        "prompt_version": "p1",
        "score": 0.9,
        "threshold_at_write": 0.75,
    })
    .to_string();
    conn.execute(
        "INSERT INTO edges
             (edge_id, src_id, src_kind, dst_id, dst_kind, edge_type, lineage,
              grounded, root_id, space, payload, created_at, superseded_by, valid_until)
         VALUES (?1, ?2, 'claim_revision', ?3, 'memory', 'supports', 'evidence',
                 0, 'truth_read_root', ?4, ?5, 0, NULL, NULL)",
        libsql::params![edge_id, revision_id, memory_id, space, payload],
    )
    .await
    .unwrap();
}

async fn stored_support_status(conn: &libsql::Connection, page_id: &str) -> String {
    let mut rows = conn
        .query(
            "SELECT support_status FROM page_truth_state WHERE page_id = ?1",
            libsql::params![page_id],
        )
        .await
        .unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

#[test]
fn truth_reader_binds_frontier_enforcement_and_registry_to_one_connection_guard() {
    let source = include_str!("truth_exposure.rs");
    let start = source
        .find("pub async fn page_truth_states(")
        .expect("page_truth_states entry point");
    let end = source[start..]
        .find("pub async fn page_visibility(")
        .map(|offset| start + offset)
        .expect("page_visibility follows page_truth_states");
    let body = &source[start..end];
    let guard = body
        .find("let conn = self.conn.lock().await;")
        .expect("truth reader must take the one DB connection guard");
    let frontier = body
        .find("SUPPORT_RECONCILE_FRONTIER_KEY")
        .expect("frontier must be read directly under that guard");
    let enforcement = body
        .find("CLAIM_PROMOTER_ENFORCEMENT_KEY")
        .expect("shadow/enforcement state must be read under that guard");
    let registry = body
        .find("SELECT t.page_id, t.support_status")
        .expect("truth and eligibility rows must be read under that guard");

    assert!(
        guard < frontier && frontier < registry && guard < enforcement && enforcement < registry,
        "frontier/enforcement and the registry-backed truth query must share one locked snapshot"
    );
    assert!(
        !body[..guard].contains("support_reconcile_frontier().await")
            && !body[..guard].contains("claim_promoter_enforcement_enabled().await"),
        "self-locking control reads before the row query create a torn snapshot"
    );
}

/// The default reader, post-cutover. Supported prose flows, a failed judgement
/// does not, and a page with no truth row at all is unjudged rather than
/// condemned.
///
/// That last clause used to read the other way -- no row meant unsupported, on
/// the principle that absence of a support record is not evidence of support.
/// It sounds fail-closed and it is not, because of what actually populates the
/// table: the ONLY writer is migration 99's backfill, and `insert_page` creates
/// no row. So every page created after that migration has no row, forever. The
/// old rule did not fail closed on a rare gap, it hid one hundred percent of
/// future pages the instant the generation advanced -- and hid them silently,
/// since nothing would ever arrive to un-hide them. A rule whose fail-closed
/// case is *every* case is not a safety property, it is an outage.
#[tokio::test]
async fn an_automatic_reader_sees_only_supported_pages_after_the_cutover() {
    let (db, _tmp) = db_with_truth_rows().await;
    let seen = db
        .page_visibility(&TruthGrant::Automatic, &ids(&["p1", "p2", "p3"]))
        .await
        .unwrap();
    assert_eq!(seen["p1"], Visibility::Full);
    assert_eq!(seen["p2"], Visibility::Hidden);
    assert_eq!(
        seen["p3"],
        Visibility::Full,
        "a page with no truth row has not been judged, and an unjudged page keeps its prose"
    );
}

#[tokio::test]
async fn promoter_results_default_to_advisory_on_an_existing_cutover() {
    let (db, _tmp) = db_with_truth_rows().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "DELETE FROM app_metadata WHERE key = 'claim_promoter_enforcement'",
            (),
        )
        .await
        .unwrap();
    }

    let states = db.page_truth_states(&ids(&["p2"])).await.unwrap();
    assert_eq!(
        states["p2"].support,
        crate::truth_contract::Support::Unevaluated,
        "an evaluated provisional row remains measurable but must not enforce by default"
    );
    let seen = db
        .page_visibility(&TruthGrant::Automatic, &ids(&["p2"]))
        .await
        .unwrap();
    assert_eq!(
        seen["p2"],
        Visibility::Full,
        "shadow promotion must not hide prose or archive its projection"
    );

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT support_status, evaluated_at IS NOT NULL
               FROM page_truth_state WHERE page_id = 'p2'",
            (),
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    assert_eq!(row.get::<String>(0).unwrap(), "provisional");
    assert_eq!(row.get::<i64>(1).unwrap(), 1);
}

#[tokio::test]
async fn retiring_the_exact_judge_triple_is_visible_on_the_next_read() {
    let (db, _tmp) = db_with_truth_rows().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_judge_eligibility SET state = 'retired'
              WHERE model_id = 'truth-read-judge'
                AND model_version = 'v1' AND prompt_version = 'p1'",
            (),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.page_truth_states(&ids(&["p1"])).await.unwrap()["p1"].support,
        crate::truth_contract::Support::Unevaluated
    );
    let conn = db.conn.lock().await;
    assert_eq!(stored_support_status(&conn, "p1").await, "supported");
}

#[tokio::test]
async fn raising_the_live_threshold_is_visible_on_the_next_read() {
    let (db, _tmp) = db_with_truth_rows().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_judge_eligibility SET threshold = 0.95
              WHERE model_id = 'truth-read-judge'
                AND model_version = 'v1' AND prompt_version = 'p1'",
            (),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.page_truth_states(&ids(&["p1"])).await.unwrap()["p1"].support,
        crate::truth_contract::Support::Unevaluated
    );
    let conn = db.conn.lock().await;
    assert_eq!(stored_support_status(&conn, "p1").await, "supported");
}

#[tokio::test]
async fn a_draining_judge_remains_eligible_on_the_next_read() {
    let (db, _tmp) = db_with_truth_rows().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_judge_eligibility SET state = 'draining'
              WHERE model_id = 'truth-read-judge'
                AND model_version = 'v1' AND prompt_version = 'p1'",
            (),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.page_truth_states(&ids(&["p1"])).await.unwrap()["p1"].support,
        crate::truth_contract::Support::Supported
    );
    let conn = db.conn.lock().await;
    assert_eq!(stored_support_status(&conn, "p1").await, "supported");
}

#[tokio::test]
async fn an_unregistered_judge_triple_is_unevaluated_on_the_next_read() {
    let (db, _tmp) = db_with_truth_rows().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE edges
                SET payload = json_set(payload, '$.model_version', 'unregistered')
              WHERE edge_type = 'supports' AND src_id = 'revision_p1'",
            (),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.page_truth_states(&ids(&["p1"])).await.unwrap()["p1"].support,
        crate::truth_contract::Support::Unevaluated
    );
    let conn = db.conn.lock().await;
    assert_eq!(stored_support_status(&conn, "p1").await, "supported");
}

#[tokio::test]
async fn a_zero_claim_marker_is_unevaluated_on_the_next_read() {
    let (db, _tmp) = db_with_truth_rows().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_markers SET inventory_count = 0
              WHERE page_id = 'p1' AND page_version = 1",
            (),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.page_truth_states(&ids(&["p1"])).await.unwrap()["p1"].support,
        crate::truth_contract::Support::Unevaluated
    );
    let conn = db.conn.lock().await;
    assert_eq!(stored_support_status(&conn, "p1").await, "supported");
}

#[tokio::test]
async fn a_partial_claim_inventory_is_unevaluated_on_the_next_read() {
    let (db, _tmp) = db_with_truth_rows().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_markers SET inventory_count = 2
              WHERE page_id = 'p1' AND page_version = 1",
            (),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.page_truth_states(&ids(&["p1"])).await.unwrap()["p1"].support,
        crate::truth_contract::Support::Unevaluated
    );
    let conn = db.conn.lock().await;
    assert_eq!(stored_support_status(&conn, "p1").await, "supported");
}

/// The collection carve-out: a page whose judgement failed may be *listed* with
/// its state, which is what keeps the review loop from having no entry point.
/// Never prose. An unjudged page is untouched by any of this.
#[tokio::test]
async fn a_collection_grant_degrades_unsupported_pages_to_entry_only() {
    let (db, _tmp) = db_with_truth_rows().await;
    let seen = db
        .page_visibility(&TruthGrant::CollectionEntries, &ids(&["p1", "p2", "p3"]))
        .await
        .unwrap();
    assert_eq!(seen["p1"], Visibility::Full);
    assert_eq!(seen["p2"], Visibility::EntryOnly);
    assert_eq!(seen["p3"], Visibility::Full);
}

/// The grant covers the page the call named and nothing riding along beside it.
/// `p2` is named and opens.
///
/// `p3` can no longer carry this test -- it has no truth row, so it is unjudged
/// and visible on its own account, which would pass whether or not the
/// named-grant rule held. The real check is a *judged-and-failed* page that was
/// not named, so the assertion moved onto one.
#[tokio::test]
async fn a_named_grant_opens_the_named_page_and_nothing_else() {
    let (db, _tmp) = db_with_truth_rows().await;
    // `insert_page` takes the connection mutex itself, so it has to happen
    // before this scope opens one -- holding the guard across it self-deadlocks.
    db.insert_page(
        "p4",
        "p4",
        None,
        "",
        None,
        None,
        &[],
        "2026-07-28T00:00:00Z",
    )
    .await
    .ok();
    {
        let conn = db.conn.lock().await;
        set_truth(&conn, "p4", "provisional").await;
    }
    let seen = db
        .page_visibility(
            &TruthGrant::NamedPages(ids(&["p2"])),
            &ids(&["p2", "p3", "p4"]),
        )
        .await
        .unwrap();
    assert_eq!(seen["p2"], Visibility::Full);
    assert_eq!(
        seen["p4"],
        Visibility::Hidden,
        "a grant for one page must not cover another page in the same payload"
    );
}

/// The inertness gate, proven where it actually runs rather than only in the
/// pure unit test: the same substrate at generation 0 hides nothing.
#[tokio::test]
async fn the_same_substrate_at_generation_zero_hides_nothing() {
    let (db, _tmp) = db_with_truth_rows().await;
    db.set_truth_cutover_generation(0).await.unwrap();
    let seen = db
        .page_visibility(&TruthGrant::Automatic, &ids(&["p1", "p2", "p3"]))
        .await
        .unwrap();
    assert!(seen.values().all(|v| *v == Visibility::Full));
}

/// The CHECK constraint is the tooth that stops a future caller inventing a
/// fifth outcome that no reader knows how to count.
#[tokio::test]
async fn an_unknown_outcome_string_is_refused_by_the_schema() {
    let (db, _tmp) = test_db().await;
    let conn = db.conn.lock().await;
    let result = conn
        .execute(
            "INSERT INTO truth_marker_audit (marked_at, caller, method, path, outcome, page_ids)
             VALUES (1, 'x', 'GET', '/api/pages', 'granted_everything', '[]')",
            (),
        )
        .await;
    assert!(result.is_err(), "the outcome CHECK constraint is missing");
}

// ---- the cutover setter has no production caller -------------------------

/// Every `.rs` file under every crate's `src/` in this workspace.
fn workspace_sources() -> Vec<(std::path::PathBuf, String)> {
    let crates_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/");
    let mut stack: Vec<std::path::PathBuf> = std::fs::read_dir(crates_dir)
        .expect("read crates/")
        .flatten()
        .map(|entry| entry.path().join("src"))
        .filter(|path| path.is_dir())
        .collect();
    assert!(
        !stack.is_empty(),
        "found no crate sources under {}",
        crates_dir.display()
    );

    let mut out = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read_dir").flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let body = std::fs::read_to_string(&path).expect("read source");
                out.push((path, body));
            }
        }
    }
    out
}

/// The structural half of the docstring on `set_truth_cutover_generation`.
///
/// Advancing that integer past 0 is what turns
/// `enforce_projection_directory_invariant` from an inert pass into a mass
/// delete over the user's Markdown vault, so PR-B ships it with zero production
/// callers on purpose. The docstring says nothing there should be read as
/// permission to flip it; a docstring is prose. This is the guarantee.
///
/// Scanning source rather than behavior, and scanning every crate rather than
/// the ones we expect to be clean, mirrors
/// `truth_contract_test.rs` "inventory teeth 9": the failure worth catching is
/// the call site added next week, not the ones already known.
///
/// Test callers are excused by FILE, not by `#[cfg(test)]` region: every test
/// module in this workspace lives in a `*_test.rs` pulled in under
/// `#[cfg(test)] #[path = "..."] mod tests;`, and a scan that instead tried to
/// recognize test regions inside a source file would have to decide where one
/// ends. Getting that wrong fails OPEN -- an early `#[cfg(test)]` in a long
/// file would hide every call site below it, which is the whole point of the
/// test evaporating silently. So the rule is coarse and fails CLOSED: a call
/// from an inline `#[cfg(test)] mod tests { ... }` is reported, and the fix is
/// to move it to the `*_test.rs` file the rest of the crate uses.
/// **Amended for PR-D, deliberately, as the PR-B message instructed.** The
/// ceremony has landed, so the setter now has exactly one production caller:
/// `commit_cutover`, in this module's own source file, which reaches it only
/// while holding a [`super::CutoverLease`]. Every other call site in every crate
/// is still a failure, and a *second* call inside `truth_exposure.rs` is a
/// failure too -- otherwise the allowance would widen into a hole the moment
/// someone adds a convenience wrapper next to the one that earned it.
#[test]
fn the_cutover_setter_has_one_production_caller() {
    const SETTER: &str = "set_truth_cutover_generation";
    const ALLOWED_FILE: &str = "truth_exposure.rs";
    let declaration = format!("fn {SETTER}");

    let mut callers = Vec::new();
    let mut allowed = Vec::new();
    for (path, body) in workspace_sources() {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name.ends_with("_test.rs") {
            continue;
        }
        for (offset, line) in body.lines().enumerate() {
            let trimmed = line.trim_start();
            // The declaration itself is not a call, and neither is prose.
            if !line.contains(SETTER) || trimmed.starts_with("//") || trimmed.contains(&declaration)
            {
                continue;
            }
            let site = format!("{}:{}", path.display(), offset + 1);
            if name == ALLOWED_FILE {
                allowed.push(site);
            } else {
                callers.push(site);
            }
        }
    }

    assert!(
        callers.is_empty(),
        "{SETTER} has a caller outside `{ALLOWED_FILE}` and outside every \
         `*_test.rs` file: {callers:?}. Advancing the cutover generation makes \
         the projection pass delete every unsupported page's file out of the \
         user's Markdown vault, so the ONE production path is `commit_cutover`, \
         which requires a `CutoverLease` and releases the writer fence only \
         after the generation is committed. Route the new caller through the \
         ceremony rather than widening this allowance. If this is a test caller, \
         move it into the crate's `*_test.rs` module, which is where every other \
         one lives."
    );
    assert_eq!(
        allowed.len(),
        1,
        "expected exactly one {SETTER} call inside `{ALLOWED_FILE}` -- the one in \
         `commit_cutover` -- and found {allowed:?}. A second call there is a \
         second way to advance the generation, and the fence only guards the \
         first."
    );
}

/// Every production page-prose write goes through a permit.
///
/// `write_page` is the ungated form -- it renders a page's body to a `.md` in a
/// directory `wenlan pages` reads with no gate in front of it. The gated forms
/// (`write_page_gated`, `write_page_permitted`) are what the truth contract
/// actually covers, and the difference is one word at the call site.
///
/// This matters more than it looks because
/// `enforce_projection_directory_invariant` -- the pass that evicts hidden pages
/// -- has exactly ONE production caller, at daemon startup (`main.rs`). There is
/// no sweep behind an ungated write. A page re-projected past the permit sits in
/// the user's vault until the next restart, which for a daemon is indefinitely.
///
/// Scanned by source rather than enforced by visibility because the ungated form
/// is legitimately `pub(crate)` for the writer's own internals and for test
/// fixtures; the failure worth catching is the production call site added next
/// week, same reasoning as the setter tooth above.
#[test]
fn no_production_code_projects_a_page_without_a_permit() {
    // Every projection-write entry point starts with this. Enumerating the
    // PERMITTED forms rather than the forbidden ones is the point: the first
    // draft of this tooth matched the literal `.write_page(`, which let
    // `.write_page_with_after_target_write(` -- a real, production-reachable,
    // ungated page write on the repair path -- straight through. A new
    // `write_page_*` variant is now flagged until someone adds it below, so
    // widening the writer surface cannot silently widen the hole.
    const PREFIX: &str = ".write_page";
    const PERMITTED: [&str; 3] = [
        ".write_page_gated(",
        ".write_page_permitted(",
        ".write_page_with_after_target_write_permitted(",
    ];
    // The module that owns the writer. Its own internals reach the raw form on
    // purpose -- that is what the gated forms are implemented in terms of.
    const OWNER: &str = "knowledge.rs";
    // Its only hit is a fixture inside an inline `#[cfg(test)] mod tests`. The
    // count below is the fail-closed half: a second hit here is reported.
    const INLINE_TEST_FIXTURE: &str = "page_watcher.rs";

    let mut callers = Vec::new();
    let mut fixture_hits = 0usize;
    for (path, body) in workspace_sources() {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        if name.ends_with("_test.rs") || name.ends_with("_tests.rs") || name == OWNER {
            continue;
        }
        // Only the crates that can hold a `KnowledgeWriter`. `wenlan-mcp` and
        // `wenlan-cli` reach pages over HTTP and never touch the projection --
        // the crate boundary in AGENTS.md, not a convenience -- so their
        // `write_page_impl` request wrappers are not projection writes.
        let owned = path.to_string_lossy().replace('\\', "/");
        if !owned.contains("/crates/wenlan-core/") && !owned.contains("/crates/wenlan-server/") {
            continue;
        }
        for (offset, line) in body.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for (at, _) in line.match_indices(PREFIX) {
                let call = &line[at..];
                if PERMITTED.iter().any(|form| call.starts_with(form)) {
                    continue;
                }
                if name == INLINE_TEST_FIXTURE {
                    fixture_hits += 1;
                    continue;
                }
                callers.push(format!("{}:{}", path.display(), offset + 1));
            }
        }
    }

    assert!(
        callers.is_empty(),
        "production code projects a page without a truth permit: {callers:?}. Use \
         `write_page_gated` (async, asks for the permit itself) or \
         `write_page_permitted` (takes one already in hand, for paths that hold \
         the connection mutex). The startup projection invariant is the only pass \
         that evicts hidden pages, so an ungated write outlives the cutover \
         ceremony that was supposed to remove it. If you have added a new \
         permitted `write_page_*` form, add it to PERMITTED above -- this tooth \
         is deliberately fail-closed on names it does not recognise."
    );
    assert_eq!(
        fixture_hits, 1,
        "expected exactly one ungated `write_page` in `{INLINE_TEST_FIXTURE}` -- \
         the `project_page` fixture in its inline test module -- and found \
         {fixture_hits}. If this is new production code, gate it; if it is a new \
         fixture, move the test module into a `*_test.rs` file like the rest of \
         the crate so this allowance can go away."
    );
}

// ---- the writer fence ----------------------------------------------------

use super::{CutoverFence, CutoverPhase};

#[tokio::test]
async fn a_fresh_database_has_no_ceremony_in_flight() {
    let (db, _tmp) = test_db().await;
    assert_eq!(db.cutover_fence().await.unwrap(), CutoverFence::INITIAL);
}

/// The §4.1 mutation check: a writer must not be able to skip the fence.
///
/// `page_write_permit` is the one seam every page writer goes through -- the two
/// `write_page_gated` projection writers and the re-distillation gate. While a
/// ceremony prepares, all of them get `None`, at generation 0, where the truth
/// state would otherwise grant every page.
#[tokio::test]
async fn a_page_write_is_refused_while_a_cutover_prepares() {
    let (db, _tmp) = test_db().await;
    assert!(
        crate::truth_adapter::page_write_permit(&db, "p1")
            .await
            .unwrap()
            .is_some(),
        "generation 0 with no ceremony in flight must grant"
    );

    let lease = db.begin_cutover().await.unwrap();
    assert!(
        crate::truth_adapter::page_write_permit(&db, "p1")
            .await
            .unwrap()
            .is_none(),
        "a page written mid-ceremony lands in the vault behind the pass that just \
         examined it"
    );

    db.abort_cutover(lease).await.unwrap();
    assert!(
        crate::truth_adapter::page_write_permit(&db, "p1")
            .await
            .unwrap()
            .is_some(),
        "aborting must give the writers back"
    );
}

/// The ABA test named in §6. `off -> preparing -> off` returns the *phase* to
/// where it started, and a bare enum could not tell the two apart.
#[tokio::test]
async fn a_stale_off_capture_fails_against_a_higher_epoch() {
    let (db, _tmp) = test_db().await;
    let stale = db.cutover_fence().await.unwrap();

    let lease = db.begin_cutover().await.unwrap();
    db.abort_cutover(lease).await.unwrap();

    let now = db.cutover_fence().await.unwrap();
    assert_eq!(now.phase, stale.phase, "the phase really did return to off");
    assert!(now.epoch > stale.epoch, "but the epoch moved");
    assert_ne!(
        now, stale,
        "a writer holding the pre-ceremony capture must not match the fence it \
         would swap against"
    );
    assert!(
        !db.swap_cutover_fence(stale, stale.next(CutoverPhase::Preparing))
            .await
            .unwrap(),
        "the stale capture won a compare-and-swap it should have lost"
    );
}

/// §7.5: the generation is committed first, the fence released second.
#[tokio::test]
async fn committing_advances_the_generation_and_then_releases_the_fence() {
    let (db, _tmp) = test_db().await;
    let lease = db.begin_cutover().await.unwrap();
    assert_eq!(
        db.truth_cutover_generation().await.unwrap(),
        0,
        "preparing must not have advanced anything yet"
    );

    db.commit_cutover(lease, 1).await.unwrap();
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 1);
    assert_eq!(
        db.cutover_fence().await.unwrap().phase,
        CutoverPhase::Committed
    );
}

// `a_spent_lease_cannot_commit_again` used to live here. It cannot be written
// any more: `commit_cutover` consumes the lease, so replaying it is a borrow
// error rather than a runtime refusal. The compiler is the stronger tooth, and
// the runtime fence check it used to exercise is still covered by
// `a_released_fence_invalidates_the_lease_it_took_back`, where the fence moves
// underneath a lease that has NOT been spent -- the case a linear type cannot
// rule out.

/// §6: `committed` is forward-only. Rebuilding the legacy directory would put
/// every provisional page's prose back at a path `wenlan pages` reads directly.
#[tokio::test]
async fn a_committed_cutover_cannot_be_rolled_back_to_off() {
    let (db, _tmp) = test_db().await;
    let lease = db.begin_cutover().await.unwrap();
    db.commit_cutover(lease, 1).await.unwrap();

    // Aborting with the spent lease is not expressible; a fresh ceremony is the
    // only way anyone could try to walk it back, and it must refuse.
    assert!(
        db.begin_cutover().await.is_err(),
        "a second ceremony must not be able to walk the fence back to off"
    );
    assert_eq!(
        db.cutover_fence().await.unwrap().phase,
        CutoverPhase::Committed
    );
}

/// A ceremony killed between `begin_cutover` and `commit_cutover` strands the
/// fence at `preparing`, which refuses every page write. The lease died with the
/// process, so `abort_cutover` cannot be the way out -- the daemon's startup
/// release is.
#[tokio::test]
async fn a_ceremony_killed_mid_flight_does_not_wedge_page_writes_forever() {
    let (db, _tmp) = test_db().await;
    // Bound and then never spent: the process that held the lease is gone, and
    // the fence it took is all that is left on disk. `drop` would say this more
    // loudly, but the lease owns nothing, so clippy reads that as dead code.
    let _lease = db.begin_cutover().await.unwrap();

    assert!(
        crate::truth_adapter::page_write_permit(&db, "p1")
            .await
            .unwrap()
            .is_none(),
        "a stranded fence must refuse writes; that is the state being recovered from"
    );
    assert!(
        db.begin_cutover().await.is_err(),
        "and it cannot be retried"
    );

    assert!(db.release_stranded_cutover_fence().await.unwrap());
    assert_eq!(db.cutover_fence().await.unwrap().phase, CutoverPhase::Off);
    assert!(crate::truth_adapter::page_write_permit(&db, "p1")
        .await
        .unwrap()
        .is_some());
}

/// The release is not a general reset. `committed` is forward-only, and `off`
/// has nothing to release -- reporting `false` is how the daemon stays quiet on
/// every ordinary boot.
#[tokio::test]
async fn only_a_stranded_preparing_fence_is_releasable() {
    let (db, _tmp) = test_db().await;
    assert!(!db.release_stranded_cutover_fence().await.unwrap());

    let lease = db.begin_cutover().await.unwrap();
    db.commit_cutover(lease, 1).await.unwrap();
    assert!(!db.release_stranded_cutover_fence().await.unwrap());
    assert_eq!(
        db.cutover_fence().await.unwrap().phase,
        CutoverPhase::Committed
    );
}

/// A released lease cannot come back and commit.
///
/// Deliberately NOT claiming the epoch bump is what stops it -- probing that
/// mutation showed it is not: `commit_cutover` compares the whole `(epoch,
/// phase)` tuple, and a lease is always `preparing`, so the phase alone already
/// refuses. The bump is there to keep the epoch monotone, which `next` and the
/// ABA reasoning for *writers* both assume, and this test pins the behaviour
/// that actually protects the generation.
#[tokio::test]
async fn a_released_fence_invalidates_the_lease_it_took_back() {
    let (db, _tmp) = test_db().await;
    let lease = db.begin_cutover().await.unwrap();
    assert!(db.release_stranded_cutover_fence().await.unwrap());

    assert!(
        db.commit_cutover(lease, 1).await.is_err(),
        "the released lease must not be able to advance the generation"
    );
    assert_eq!(db.truth_cutover_generation().await.unwrap(), 0);
}

/// Unlike the generation, an unreadable fence is an error rather than a
/// default. "Inert" for a fence means letting writers through, which is exactly
/// what an indeterminate ceremony state must not do.
#[tokio::test]
async fn an_unreadable_fence_refuses_writes_instead_of_defaulting_to_off() {
    let (db, _tmp) = test_db().await;
    db.set_app_metadata(super::TRUTH_CUTOVER_FENCE_KEY, "soon")
        .await
        .unwrap();

    assert!(db.cutover_fence().await.is_err());
    assert!(
        crate::truth_adapter::page_write_permit(&db, "p1")
            .await
            .is_err(),
        "an unreadable fence must not read as permission to write"
    );
}

/// The projection pass is tested; its **wiring** was not.
///
/// `enforce_projection_directory_invariant` has a full test suite, and deleting
/// the one line in `wenlan-server/src/main.rs` that calls it leaves every one of
/// those tests green -- the pass would simply never run, and the directory
/// `wenlan pages` reads would keep every unsupported page's file after the
/// cutover. A pass nobody calls is a comment.
///
/// This is the cheapest possible tooth for that: the call site exists, in
/// production source, outside a test file. It cannot prove the call is reached
/// at runtime, and it does not try to -- a source scan is exactly strong enough
/// to catch a deletion, which is the failure that actually happened to be
/// invisible.
#[test]
fn the_projection_invariant_is_wired_into_the_daemon() {
    const PASS: &str = "enforce_projection_directory_invariant";
    let declaration = format!("fn {PASS}");

    let call_sites: Vec<String> = workspace_sources()
        .into_iter()
        .filter(|(path, _)| {
            !path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("_test.rs"))
        })
        .flat_map(|(path, body)| {
            body.lines()
                .enumerate()
                .filter(|(_, line)| {
                    let trimmed = line.trim_start();
                    line.contains(PASS)
                        && !trimmed.starts_with("//")
                        && !trimmed.starts_with("///")
                        && !trimmed.contains(&declaration)
                })
                .map(|(offset, _)| format!("{}:{}", path.display(), offset + 1))
                .collect::<Vec<_>>()
        })
        .collect();

    assert!(
        call_sites.iter().any(|site| site.contains("wenlan-server")),
        "no production call to {PASS} in wenlan-server: {call_sites:?}. The \
         projection directory is the whole enforcement for `wenlan pages`, which \
         reads Markdown off disk where no wire gate can reach it. Without this \
         call the pass is a comment and every unsupported page stays readable \
         after the cutover. If the daemon genuinely stopped owning this, move the \
         call and update this test to name its new home -- do not delete it."
    );
}

/// F1(b): a review is a statement about the exact page version and text a
/// person actually read, so it cannot outlive either.
///
/// Nothing clears `human_reviewed` when a page is edited — the update bumps
/// `version` and rewrites `content`, and the truth row is not in its reach. So
/// the stored bit is a historical receipt, and whether a review is still *in
/// force* has to be computed where truth is read: the recorded version and
/// digest against the current page. An edited page falls back to unreviewed on
/// its own, with no new write path and nothing to migrate.
///
/// Both consequences are checked here, because they come from one place: the
/// axis the wire reports, and the visibility verdict that axis outranks.
#[tokio::test]
async fn an_edit_after_a_review_retires_the_review() {
    let (db, _tmp) = test_db().await;
    let page = "page_00000000-0000-4000-8000-0000000000b1";
    db.create_page_draft_with_id(page, "A Page", "the prose a human read", None, None)
        .await
        .unwrap();

    // Stand in for the review mutation: mark it against the text as it is now.
    let digest = crate::provenance::revision_content_digest("the prose a human read");
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO page_truth_state
                 (page_id, page_version, support_status, evaluated_at, human_reviewed,
                  reviewed_page_version, reviewed_page_digest, updated_at)
             VALUES (?1, 1, 'provisional', 1700000000, 1, 1, ?2, 1700000000)",
            libsql::params![page, digest.as_str()],
        )
        .await
        .unwrap();
    }

    let ids = vec![page.to_string()];
    assert!(
        db.page_truth_states(&ids).await.unwrap()[page].human_reviewed,
        "the review is in force while the page still says what it said"
    );

    // Post-cutover, a human-reviewed page is fully visible even though the
    // machine axis says unsupported.
    db.set_truth_cutover_generation(1).await.unwrap();
    let grant = TruthGrant::Automatic;
    assert_eq!(
        db.page_visibility(&grant, &ids).await.unwrap()[page],
        Visibility::Full,
        "human review outranks the machine verdict"
    );

    // Now somebody edits the page.
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET content = 'entirely different prose', version = version + 1
              WHERE id = ?1",
            libsql::params![page],
        )
        .await
        .unwrap();
    }

    let after_edit = db.page_truth_states(&ids).await.unwrap()[page];
    assert!(
        !after_edit.human_reviewed,
        "the review was of a version and text that are no longer current"
    );
    assert_eq!(
        after_edit.support,
        crate::truth_contract::Support::Unevaluated,
        "the machine verdict was also about the superseded page version"
    );
    assert_eq!(
        db.page_visibility(&grant, &ids).await.unwrap()[page],
        Visibility::Full,
        "the edited page remains visible because it is now unevaluated, not because the old review survived"
    );

    // Putting the text back does not restore a review of an older page version.
    // The truth-state contract binds approval to both exact version and digest.
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE pages SET content = 'the prose a human read' WHERE id = ?1",
            libsql::params![page],
        )
        .await
        .unwrap();
    }
    assert!(
        !db.page_truth_states(&ids).await.unwrap()[page].human_reviewed,
        "restoring bytes does not make an approval of page version 1 apply to version 2"
    );
}

/// A page nobody reviewed must not start reading as reviewed because of how the
/// digest comparison is written — a missing page, or a NULL digest, is not a
/// match.
#[tokio::test]
async fn an_unreviewed_page_is_unaffected_by_the_digest_check() {
    let (db, _tmp) = test_db().await;
    let page = "page_00000000-0000-4000-8000-0000000000b2";
    db.create_page_draft_with_id(page, "A Page", "prose", None, None)
        .await
        .unwrap();
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO page_truth_state
                 (page_id, page_version, support_status, evaluated_at, human_reviewed, updated_at)
             VALUES (?1, 1, 'supported', 1700000000, 0, 1700000000)",
            libsql::params![page],
        )
        .await
        .unwrap();
        conn.execute(
            "INSERT INTO claim_derivation_markers
                 (page_id, page_version, page_version_digest, extractor_version,
                  inventory_count, created_at)
             VALUES (?1, 1, ?2, ?3, 1, 1700000000)",
            libsql::params![
                page,
                crate::provenance::revision_content_digest("prose"),
                crate::db::EXTRACTOR_VERSION
            ],
        )
        .await
        .unwrap();
        seed_live_support(&conn, page).await;
    }
    let states = db.page_truth_states(&[page.to_string()]).await.unwrap();
    assert!(!states[page].human_reviewed);
    assert_eq!(
        states[page].support,
        crate::truth_contract::Support::Supported,
        "the machine axis is untouched by the review check"
    );
}

/// Matrix row 8. A page edited after being judged carries a verdict about text
/// it no longer holds. The enqueue trigger queues the WORK; until a worker runs
/// — on a substrate whose producer does not exist yet, never — `supported` keeps
/// the new prose readable on the strength of a judgement of the old.
///
/// Asserted in both directions, because the fail-safe has to cut both ways. A
/// stale `supported` must stop granting, and a stale failed verdict must stop
/// hiding: v1's refutation is not evidence about v2, and letting it stand would
/// archive a page over a judgement nobody ever made of it.
#[tokio::test]
async fn a_verdict_about_a_superseded_version_does_not_answer_for_the_new_one() {
    let (db, _tmp) = db_with_truth_rows().await;
    let before = db.page_truth_states(&ids(&["p1", "p2"])).await.unwrap();
    assert_eq!(
        before.get("p1").unwrap().support,
        crate::truth_contract::Support::Supported
    );
    assert_eq!(
        before.get("p2").unwrap().support,
        crate::truth_contract::Support::Unsupported
    );

    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE pages SET version = 2 WHERE id IN ('p1','p2')", ())
            .await
            .unwrap();
    }

    let after = db.page_truth_states(&ids(&["p1", "p2"])).await.unwrap();
    assert_eq!(
        after.get("p1").unwrap().support,
        crate::truth_contract::Support::Unevaluated,
        "a v1 `supported` verdict may not answer for v2's prose"
    );
    assert_eq!(
        after.get("p2").unwrap().support,
        crate::truth_contract::Support::Unevaluated,
        "and a v1 refutation may not archive v2 either"
    );

    // The same call through the gate every adapter actually uses: the page that
    // was being hidden on a superseded verdict gets its prose back.
    let visible = db
        .page_visibility(&TruthGrant::Automatic, &ids(&["p2"]))
        .await
        .unwrap();
    assert_eq!(visible.get("p2"), Some(&Visibility::Full));
}

/// The read gate's half of §1 condition 1: a verdict is readable only while the
/// derivation it came out of still exists.
///
/// Defaulting a MISSING marker to agreement was the hole. `evaluate_page_support`
/// calls a page with no marker `Unevaluated` — condition 1 unproven — so writer
/// and reader disagreed about the same page and the reader was the lenient one.
/// A marker that never landed, or was lost, therefore kept a stored `supported`
/// readable indefinitely, and on a branch with no producer nothing would ever
/// come along to disturb it.
///
/// Asserted in both directions, because a rule that hides everything is not a
/// fail-safe. Losing the marker must stop `supported` granting AND stop a stored
/// refutation hiding — the page falls back to unjudged, which keeps its file.
#[tokio::test]
async fn a_verdict_whose_derivation_marker_is_gone_stops_being_readable() {
    let (db, _tmp) = db_with_truth_rows().await;
    let before = db.page_truth_states(&ids(&["p1", "p2"])).await.unwrap();
    assert_eq!(
        before.get("p1").unwrap().support,
        crate::truth_contract::Support::Supported,
        "the marker is present, so the stored verdict reads -- the change below \
         is about losing it and nothing else"
    );
    assert_eq!(
        before.get("p2").unwrap().support,
        crate::truth_contract::Support::Unsupported
    );

    {
        let conn = db.conn.lock().await;
        conn.execute(
            "DELETE FROM claim_derivation_markers WHERE page_id IN ('p1','p2')",
            (),
        )
        .await
        .unwrap();
    }

    let after = db.page_truth_states(&ids(&["p1", "p2"])).await.unwrap();
    assert_eq!(
        after.get("p1").unwrap().support,
        crate::truth_contract::Support::Unevaluated,
        "a `supported` with no derivation behind it may not answer for the page"
    );
    assert_eq!(
        after.get("p2").unwrap().support,
        crate::truth_contract::Support::Unevaluated,
        "and the same rule must release the page it was hiding"
    );
    assert_eq!(
        db.page_visibility(&TruthGrant::Automatic, &ids(&["p2"]))
            .await
            .unwrap()
            .get("p2"),
        Some(&Visibility::Full),
        "falling back to unjudged keeps the file, which is the safe direction"
    );
}

/// The read gate's half of the extractor clause. `claim_derivation_markers`
/// records `extractor_version` because identical page text under a changed
/// extractor yields a DIFFERENT claim set — so an old marker no longer describes
/// the inventory the verdict was reached over, even though every digest matches.
///
/// Bumping `EXTRACTOR_VERSION` re-queues every page, and a queue entry is not a
/// retraction: until this check, the stored `supported` went on being served for
/// the entire time that queue took to drain, which on a vault with no producer is
/// forever.
#[tokio::test]
async fn a_verdict_from_a_superseded_extractor_stops_being_readable() {
    let (db, _tmp) = db_with_truth_rows().await;
    assert_eq!(
        db.page_truth_states(&ids(&["p1"]))
            .await
            .unwrap()
            .get("p1")
            .unwrap()
            .support,
        crate::truth_contract::Support::Supported
    );

    {
        // The page's text is untouched, so every digest still agrees. Only the
        // extractor that produced the claim inventory has moved on.
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE claim_derivation_markers SET extractor_version = ?1 WHERE page_id = 'p1'",
            libsql::params![crate::db::claim_derivation::EXTRACTOR_VERSION - 1],
        )
        .await
        .unwrap();
    }

    assert_eq!(
        db.page_truth_states(&ids(&["p1"]))
            .await
            .unwrap()
            .get("p1")
            .unwrap()
            .support,
        crate::truth_contract::Support::Unevaluated,
        "a verdict over an inventory this binary would not extract is not a \
         verdict about this page"
    );
}

/// N2. The runtime backlog continuation WRITES, so it may not run in repair
/// recovery, and no exhaustion loop may return to pre-serve startup.
///
/// Repair recovery opens the database through `open_for_repair` precisely so
/// that startup performs no ordinary side effect: an operator inspecting a
/// damaged database must see only what the repair itself does. The runtime
/// sweep deletes stale `done` jobs and inserts pending ones, so an unfenced
/// worker would put queue mutations into the one mode that promised none.
///
/// Startup must not call `drain_stale_derivation_jobs`: looping to exhaustion
/// before `serve` has no wall-clock bound. `MemoryDB::new` does run migration
/// 105's one-time, named 500-row seed before serving; the optional runtime
/// worker continues that bounded exception one sweep per turn and yields under
/// the existing repair-mode fence. The durable queue makes the continuation
/// restart-safe.
#[test]
fn the_runtime_backlog_continuation_is_fenced_out_of_repair_mode() {
    let sources = workspace_sources();
    let startup = sources
        .iter()
        .find(|(path, _)| path.ends_with("crates/wenlan-server/src/main/startup.rs"))
        .map(|(_, body)| body.as_str())
        .expect("startup.rs must be in the production source inventory");
    let runtime = sources
        .iter()
        .find(|(path, _)| path.ends_with("crates/wenlan-server/src/main/runtime.rs"))
        .map(|(_, body)| body.as_str())
        .expect("runtime.rs must be in the production source inventory");
    let db = sources
        .iter()
        .find(|(path, _)| path.ends_with("crates/wenlan-core/src/db.rs"))
        .map(|(_, body)| body.as_str())
        .expect("db.rs must be in the production source inventory");
    let claim_derivation = sources
        .iter()
        .find(|(path, _)| path.ends_with("crates/wenlan-core/src/db/claim_derivation.rs"))
        .map(|(_, body)| body.as_str())
        .expect("claim_derivation.rs must be in the production source inventory");

    assert!(
        !startup.contains("drain_stale_derivation_jobs(")
            && !startup.contains("enqueue_stale_derivation_jobs("),
        "startup.rs must not directly run data-sized backlog work"
    );
    assert!(
        startup.contains("wenlan_core::db::MemoryDB::new("),
        "the placement tooth must trace the normal constructor alias, not only direct calls"
    );
    assert!(
        claim_derivation.contains("pub(super) const MIGRATION_105_BACKLOG_SEED_LIMIT: i64 = 500;"),
        "migration 105's one permitted pre-serve queue write needs an explicit bounded contract"
    );
    assert!(
        db.contains(
            "enqueue_stale_derivation_jobs(claim_derivation::MIGRATION_105_BACKLOG_SEED_LIMIT)"
        ),
        "MemoryDB::new reaches migration 105, whose pre-serve queue seed must use the named bound"
    );
    let enqueue_start = claim_derivation
        .find("pub async fn enqueue_stale_derivation_jobs(")
        .expect("the bounded sweep entry point must exist");
    let sweep_start = claim_derivation
        .find("async fn sweep_stale_derivation_jobs(")
        .expect("the bounded sweep implementation must exist");
    let sweep_end = claim_derivation[sweep_start..]
        .find("pub async fn drain_stale_derivation_jobs(")
        .map(|offset| sweep_start + offset)
        .expect("the drain must follow the bounded sweep");
    let enqueue = &claim_derivation[enqueue_start..sweep_start];
    let validation = enqueue
        .find("if limit <= 0")
        .expect("the signed sweep bound must be validated");
    let transaction = enqueue
        .find("transaction_with_behavior")
        .expect("the bounded sweep must remain transactional");
    assert!(
        validation < transaction,
        "a non-positive limit must fail before any transactional or temp-table mutation"
    );

    let sweep = &claim_derivation[sweep_start..sweep_end];
    assert_eq!(
        sweep.matches("CREATE TEMP TABLE").count(),
        1,
        "stale-marker and drift work must share one overall batch table"
    );
    assert!(
        !sweep.contains("claim_derivation_backlog_batch")
            && !sweep.contains("claim_derivation_drift_batch"),
        "the former independently bounded batches must not return"
    );
    let capture = sweep
        .find("INSERT INTO claim_derivation_sweep_batch")
        .expect("all sweep candidates must first enter the unified batch");
    let first_cleanup = sweep
        .find("DELETE FROM claim_derivation_jobs")
        .expect("the captured batch must replace stale done jobs");
    assert!(
        capture < first_cleanup,
        "the unified bounded set must be captured before any cleanup write can touch jobs"
    );
    assert!(
        sweep.contains("UNION")
            && sweep.contains("ORDER BY c.page_id, c.page_version")
            && sweep.contains("LIMIT ?2"),
        "the distinct stale/drift union needs one deterministic overall bound"
    );
    assert_eq!(
        sweep.matches("FROM claim_derivation_sweep_batch").count(),
        4,
        "one batch reset plus done-job cleanup, pending insertion, and demotion must be the only unified-batch consumers"
    );
    assert!(
        sweep.contains("WHERE needs_demotion = 1"),
        "the unified batch must retain which captured rows need synchronous demotion"
    );
    assert!(
        sweep.contains("Ok(captured as usize)"),
        "the returned progress must count distinct rows in the unified batch"
    );
    let fence = runtime
        .find("if optional_runtime_workers_allowed(repair_recovery_pending)")
        .expect("repair recovery must fence optional runtime workers");
    let continuation = runtime
        .find("enqueue_stale_derivation_jobs(")
        .expect("the runtime worker must continue migration 105's truncated backlog");
    assert!(
        fence < continuation,
        "the backlog writer must remain inside the optional-worker repair fence"
    );
}

/// Startup atomically opens the reconciliation frontier before serving. That
/// gate WITHHOLDS a stored `supported` until the runtime evaluator re-proves it.
/// If opening the gate errors, the daemon must refuse startup rather than serve
/// exactly the stale `supported` the frontier exists to hide.
///
/// That is the failure this asserts against: a `warn!`-and-continue arm. The
/// gate must be able to refuse the boot, which on this codebase means reaching
/// `report_bootstrap_error` and returning an error, the same two moves the
/// projection-invariant pass directly above it makes for the same reason.
///
/// The data-sized evaluator belongs only to the shutdown-aware runtime worker.
/// A source scan establishes both placements; frontier behaviour tests pin the
/// resulting `Unevaluated` reads.
///
/// The generation-0 carve-out inside the arm is deliberate and is not what this
/// checks: below the cutover every adapter is pass-through and no truth state
/// reaches a reader, so there is no restriction to fail closed on.
#[test]
fn a_failed_startup_reconciliation_refuses_to_serve() {
    const GATE: &str = "begin_support_reconcile_pass(";

    let mut checked = 0usize;
    let mut runtime_continuations = 0usize;
    for (path, body) in workspace_sources() {
        let file_name = path.file_name().and_then(|name| name.to_str());
        if file_name == Some("runtime.rs") {
            runtime_continuations += body.matches("reconcile_supported_pages(").count();
        }
        if file_name != Some("startup.rs") {
            continue;
        }
        assert!(
            !body.contains("reconcile_supported_pages("),
            "startup.rs must not run a page evaluator whose per-page work is unbounded"
        );
        let lines: Vec<&str> = body.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if !code.contains(GATE) || code.trim_start().starts_with("pub async fn") {
                continue;
            }
            // The handling of this call's result, bounded generously. Anything
            // further away than this is a different statement, and an arm that
            // needed 80 lines to decide whether to serve would be its own
            // finding.
            let handler: String = lines[index..(index + 80).min(lines.len())].join("\n");
            assert!(
                handler.contains("report_bootstrap_error"),
                "{}: the startup reconciliation gate is handled without ever reaching \
                 `report_bootstrap_error`. If the frontier cannot open, stored support cannot \
                 be served safely.",
                path.display()
            );
            assert!(
                handler.contains("return Err("),
                "{}: the startup reconciliation gate is handled without ever returning an \
                 error. Logging and serving anyway exposes the stale state it should withhold.",
                path.display()
            );
            checked += 1;
        }
    }

    assert!(
        checked > 0,
        "no production startup call to {GATE}; without the frontier, stored `supported` is \
         exposed before the runtime evaluator reaches it"
    );
    assert!(
        runtime_continuations > 0,
        "no runtime reconciliation continuation; the frontier would withhold support forever"
    );
}

/// Blocker 4. Human approval outranks every machine verdict, so it is the one
/// flag that must never be trusted raw.
///
/// A person approves a SPECIFIC text — the schema stores the version and digest
/// they signed off, and the CHECK constraint guarantees both are present. Read
/// as a bare boolean, that approval silently became perpetual: edit the prose
/// afterwards and the page kept full visibility on the strength of a review of
/// text it no longer holds, which is the human-axis twin of matrix row 8.
///
/// Asserted through `page_visibility`, the gate adapters actually call, because
/// the flag's whole effect is that `visible_at` returns `Full` on it alone.
#[tokio::test]
async fn a_human_approval_of_older_text_stops_granting_visibility() {
    let (db, _tmp) = db_with_truth_rows().await;
    // p2 is provisional with a stamped verdict: Unsupported, and hidden from an
    // automatic reader unless a human approval overrides it.
    let approved_digest = crate::provenance::revision_content_digest("");
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "UPDATE page_truth_state
                SET human_reviewed = 1, reviewed_page_version = 1,
                    reviewed_page_digest = ?2
              WHERE page_id = ?1",
            libsql::params!["p2", approved_digest.clone()],
        )
        .await
        .unwrap();
    }

    let granted = db.page_truth_states(&ids(&["p2"])).await.unwrap();
    assert!(
        granted.get("p2").unwrap().human_reviewed,
        "an approval of the text the page actually holds must still grant"
    );
    assert_eq!(
        db.page_visibility(&TruthGrant::Automatic, &ids(&["p2"]))
            .await
            .unwrap()
            .get("p2"),
        Some(&Visibility::Full),
        "the approval is what keeps this unsupported page readable"
    );

    // Rewrite the prose. The version is deliberately left alone: a bumped
    // version would be caught by the version comparison, and the hole being
    // closed here is the one where the numbers still agree.
    //
    // The derivation marker moves with the text, standing in for a re-derivation
    // that reached the same provisional verdict on the new prose. Without that
    // the edit would stale the MACHINE verdict too, the page would read
    // `Unevaluated`, and the final assertion would pass on the wrong axis —
    // full visibility because nothing was judged, rather than the human axis
    // being the thing under test.
    {
        let conn = db.conn.lock().await;
        conn.execute("UPDATE pages SET content = 'rewritten' WHERE id = 'p2'", ())
            .await
            .unwrap();
        conn.execute(
            "UPDATE claim_derivation_markers SET page_version_digest = ?1
              WHERE page_id = 'p2'",
            libsql::params![crate::provenance::revision_content_digest("rewritten")],
        )
        .await
        .unwrap();
    }

    let after = db.page_truth_states(&ids(&["p2"])).await.unwrap();
    assert!(
        !after.get("p2").unwrap().human_reviewed,
        "an approval of superseded text may not go on granting authority"
    );
    assert_eq!(
        db.page_visibility(&TruthGrant::Automatic, &ids(&["p2"]))
            .await
            .unwrap()
            .get("p2"),
        Some(&Visibility::Hidden),
        "with the stale approval withdrawn the page falls back to its machine \
         verdict, which is what an unsupported page is owed"
    );
}
