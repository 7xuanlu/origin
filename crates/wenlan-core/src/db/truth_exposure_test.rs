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
/// toward inert in every ambiguous case, mirroring `reader_uses_edges`.
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
// Every other test in this file sits at generation 0, where `page_visibility`
// short-circuits and `page_truth_states` never runs. That is the production
// configuration, which is exactly why these exist: without them the only SQL
// that matters after the PR-C cutover would ship unexecuted, and a misspelled
// column would surface as a live disclosure rather than a red build.

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
    (db, temp)
}

async fn set_truth(conn: &libsql::Connection, page_id: &str, status: &str) {
    conn.execute(
        "INSERT INTO page_truth_state
            (page_id,page_version,support_status,human_reviewed,updated_at)
         VALUES (?1,1,?2,0,1)",
        libsql::params![page_id, status],
    )
    .await
    .unwrap();
}

fn ids(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

/// The default reader, post-cutover. Supported prose flows; unsupported prose
/// does not, and a page with no truth row is unsupported -- absence of a support
/// record is not evidence of support.
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
        Visibility::Hidden,
        "a page with no truth row must read as unsupported"
    );
}

/// The collection carve-out: an unsupported page may be *listed* with its state,
/// which is what keeps the review loop from having no entry point. Never prose.
#[tokio::test]
async fn a_collection_grant_degrades_unsupported_pages_to_entry_only() {
    let (db, _tmp) = db_with_truth_rows().await;
    let seen = db
        .page_visibility(&TruthGrant::CollectionEntries, &ids(&["p1", "p2", "p3"]))
        .await
        .unwrap();
    assert_eq!(seen["p1"], Visibility::Full);
    assert_eq!(seen["p2"], Visibility::EntryOnly);
    assert_eq!(seen["p3"], Visibility::EntryOnly);
}

/// The grant covers the page the call named and nothing riding along beside it.
/// `p2` is named and opens; `p3` is just as unsupported and stays shut.
#[tokio::test]
async fn a_named_grant_opens_the_named_page_and_nothing_else() {
    let (db, _tmp) = db_with_truth_rows().await;
    let seen = db
        .page_visibility(&TruthGrant::NamedPages(ids(&["p2"])), &ids(&["p2", "p3"]))
        .await
        .unwrap();
    assert_eq!(seen["p2"], Visibility::Full);
    assert_eq!(
        seen["p3"],
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
#[test]
fn the_cutover_setter_has_no_production_caller() {
    const SETTER: &str = "set_truth_cutover_generation";
    let declaration = format!("fn {SETTER}");

    let mut callers = Vec::new();
    for (path, body) in workspace_sources() {
        let is_test_file = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_test.rs"));
        if is_test_file {
            continue;
        }
        for (offset, line) in body.lines().enumerate() {
            let trimmed = line.trim_start();
            // The declaration itself is not a call, and neither is prose.
            if !line.contains(SETTER) || trimmed.starts_with("//") || trimmed.contains(&declaration)
            {
                continue;
            }
            callers.push(format!("{}:{}", path.display(), offset + 1));
        }
    }

    assert!(
        callers.is_empty(),
        "{SETTER} has a caller outside a `*_test.rs` file: {callers:?}. Advancing \
         the cutover generation makes the projection pass delete every \
         unsupported page's file out of the user's Markdown vault, so the setter \
         is a test-only lever in PR-B. PR-C's two-phase fenced ceremony is the \
         intended way to advance it -- if you are landing that ceremony, change \
         this test deliberately to name the one allowed call site rather than \
         deleting it. If this is a test caller, move it into the crate's \
         `*_test.rs` module, which is where every other one lives."
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
