// SPDX-License-Identifier: Apache-2.0
//! The §8 rows that are database questions: replay, collision, one-shot
//! nonces, and whether consumption really shares the mutation's transaction.
//!
//! These live here rather than beside the HTTP handler because every one of
//! them is about what survives in the tables, and because the ordering they
//! pin is enforced in `review_page_with_presence` — the handler only reports
//! its answer.

use super::MemoryDB;
use crate::presence::{self, PresenceAction, PresenceRefusal};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use wenlan_types::requests::PresenceCapability;

const SECRET: &[u8; 32] = b"01234567890123456789012345678901";
const PAGE: &str = "page_00000000-0000-4000-8000-0000000000a1";
const BODY: &str = "the prose a human read";

/// The daemon's data root, holding both the database and the install secret —
/// the same directory the app writes the secret into.
async fn db() -> (MemoryDB, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let emitter: std::sync::Arc<dyn crate::events::EventEmitter> =
        std::sync::Arc::new(crate::events::NoopEmitter);
    let db = MemoryDB::new(tmp.path(), emitter).await.expect("MemoryDB");
    db.create_page_draft_with_id(PAGE, "A Page", BODY, None, None)
        .await
        .expect("page");
    plant_secret(tmp.path());
    (db, tmp)
}

/// Writes the secret with the mode the app writes. The default depends on the
/// umask of whoever runs the tests, and a secret other users can read is
/// refused.
fn plant_secret(root: &std::path::Path) {
    let path = presence::secret_path(root);
    std::fs::write(&path, SECRET).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
}

/// The app's minting, reproduced independently of the encoder under test. If
/// the two ever disagree the tests below stop verifying, which is the signal
/// wanted — the same cross-check the app's own copy provides.
#[allow(clippy::too_many_arguments)]
fn mint(
    action: &str,
    targets: &[&str],
    base_digest: &str,
    caller_id: &str,
    operation_id: &str,
    nonce: &[u8],
    minted_at: u64,
    secret: &[u8],
) -> PresenceCapability {
    fn push(out: &mut Vec<u8>, bytes: &[u8]) {
        out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
        out.extend_from_slice(bytes);
    }
    let expires_at = minted_at + 60;
    let mut out = Vec::new();
    push(&mut out, &1_u32.to_be_bytes());
    push(&mut out, action.as_bytes());
    out.extend_from_slice(&(targets.len() as u64).to_be_bytes());
    for target in targets {
        push(&mut out, target.as_bytes());
    }
    push(&mut out, base_digest.as_bytes());
    push(&mut out, caller_id.as_bytes());
    push(&mut out, operation_id.as_bytes());
    push(&mut out, nonce);
    push(&mut out, &minted_at.to_be_bytes());
    push(&mut out, &expires_at.to_be_bytes());

    let mut mac = <Hmac<Sha256>>::new_from_slice(secret).unwrap();
    mac.update(&out);
    PresenceCapability {
        protocol_version: 1,
        action: action.to_string(),
        target_ids: targets.iter().map(|t| (*t).to_string()).collect(),
        base_digest: base_digest.to_string(),
        caller_id: caller_id.to_string(),
        operation_id: operation_id.to_string(),
        minted_at,
        expires_at,
        nonce: hex::encode(nonce),
        mac: hex::encode(mac.finalize().into_bytes()),
    }
}

fn body_digest() -> String {
    crate::provenance::revision_content_digest(BODY)
}

fn gesture(operation_id: &str, nonce: u8) -> PresenceCapability {
    mint(
        "review_page",
        &[PAGE],
        &body_digest(),
        "app",
        operation_id,
        &[nonce; 16],
        1_000,
        SECRET,
    )
}

async fn reviewed(db: &MemoryDB) -> bool {
    db.page_truth_states(&[PAGE.to_string()])
        .await
        .unwrap()
        .get(PAGE)
        .map(|t| t.human_reviewed)
        .unwrap_or(false)
}

async fn count(db: &MemoryDB, table: &str) -> i64 {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(&format!("SELECT count(*) FROM {table}"), ())
        .await
        .unwrap();
    rows.next().await.unwrap().unwrap().get(0).unwrap()
}

#[tokio::test]
async fn a_gesture_marks_the_page_and_leaves_a_receipt() {
    let (db, root) = db().await;
    assert!(!reviewed(&db).await, "nothing has reviewed this page yet");

    let outcome = db
        .review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    let super::ReviewOutcome::Applied(receipt) = outcome else {
        panic!("a valid capability inside its window must apply: {outcome:?}");
    };

    assert!(receipt.human_reviewed);
    assert_eq!(receipt.reviewed_page_digest, body_digest());
    assert_eq!(receipt.reviewed_page_version, 1);
    assert!(reviewed(&db).await, "the page carries the human axis now");
    assert_eq!(count(&db, "presence_receipts").await, 1);
    assert_eq!(count(&db, "presence_nonces").await, 1);
}

/// Nothing creates a `page_truth_state` row for a page distilled after
/// migration 99's backfill, so the newest pages are exactly the ones a plain
/// UPDATE would silently mark nothing for.
#[tokio::test]
async fn a_page_with_no_truth_row_yet_still_gets_marked() {
    let (db, root) = db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM page_truth_state", ())
            .await
            .unwrap();
    }
    db.review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert!(reviewed(&db).await);
}

/// A human saying "I read this" is not evidence the machine found support. The
/// row this creates must stay *unevaluated* on the machine axis, not become
/// supported or unsupported.
#[tokio::test]
async fn reviewing_a_page_says_nothing_about_its_support() {
    let (db, root) = db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute("DELETE FROM page_truth_state", ())
            .await
            .unwrap();
    }
    db.review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    let states = db.page_truth_states(&[PAGE.to_string()]).await.unwrap();
    assert_eq!(
        states.get(PAGE).unwrap().support,
        crate::truth_contract::Support::Unevaluated,
        "the machine axis must be untouched by a human review"
    );
}

/// T7. The retry arrives after the 60-second window has closed, which is the
/// normal case for a client that retries at all — and it must still be told the
/// write succeeded.
///
/// This is the row the whole §4 ordering exists for. Validate before the
/// receipt lookup and this returns `presence_expired`, at which point a client
/// that already got its write concludes it failed.
#[tokio::test]
async fn a_retry_after_the_capability_expired_replays_the_stored_answer() {
    let (db, root) = db().await;
    let cap = gesture("op-1", 0xa1);
    let first = db
        .review_page_with_presence(&cap, PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    let super::ReviewOutcome::Applied(applied) = first else {
        panic!("first execution must apply");
    };

    let retry = db
        .review_page_with_presence(&cap, PAGE, Some(root.path()), 9_999)
        .await
        .unwrap();
    let super::ReviewOutcome::Replayed(replayed) = retry else {
        panic!("a retry of an applied mutation must replay: {retry:?}");
    };
    assert_eq!(replayed.nonce_digest, applied.nonce_digest);
    assert_eq!(replayed.verified_at, applied.verified_at);
    assert_eq!(
        count(&db, "presence_nonces").await,
        1,
        "a replay must not consume a second nonce"
    );
}

/// T8. Same caller, same operation ID, different content — two different
/// mutations wearing one idempotency key. Silently overwriting would apply the
/// second under the first's authority.
#[tokio::test]
async fn reusing_an_operation_id_for_different_content_is_a_conflict() {
    let (db, root) = db().await;
    db.review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();

    let elsewhere = mint(
        "review_page",
        &[PAGE],
        "a digest of some other text",
        "app",
        "op-1",
        &[0xb2; 16],
        1_000,
        SECRET,
    );
    let outcome = db
        .review_page_with_presence(&elsewhere, PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert!(
        matches!(
            outcome,
            super::ReviewOutcome::Refused(PresenceRefusal::Conflict)
        ),
        "expected a conflict, got {outcome:?}"
    );
    assert_eq!(
        count(&db, "presence_receipts").await,
        1,
        "a conflict writes nothing"
    );
}

/// A re-mint before retrying is still a retry: same intent, new nonce. Reading
/// it as a collision would refuse a write that had already succeeded, which is
/// the same failure T7 names by a different route.
#[tokio::test]
async fn a_re_minted_retry_of_the_same_intent_still_replays() {
    let (db, root) = db().await;
    db.review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    let re_minted = mint(
        "review_page",
        &[PAGE],
        &body_digest(),
        "app",
        "op-1",
        &[0xc3; 16],
        2_000,
        SECRET,
    );
    let outcome = db
        .review_page_with_presence(&re_minted, PAGE, Some(root.path()), 2_010)
        .await
        .unwrap();
    assert!(
        matches!(outcome, super::ReviewOutcome::Replayed(_)),
        "expected a replay, got {outcome:?}"
    );
}

/// T3. A different operation ID cannot launder a spent nonce — the nonce is
/// keyed on its own digest, so the second use is caught even though the receipt
/// lookup finds nothing.
#[tokio::test]
async fn one_nonce_cannot_be_spent_twice() {
    let (db, root) = db().await;
    db.review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();

    let same_nonce = mint(
        "review_page",
        &[PAGE],
        &body_digest(),
        "app",
        "op-2",
        &[0xa1; 16],
        1_000,
        SECRET,
    );
    let outcome = db
        .review_page_with_presence(&same_nonce, PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert!(
        matches!(
            outcome,
            super::ReviewOutcome::Refused(PresenceRefusal::Replayed)
        ),
        "expected a replayed-nonce refusal, got {outcome:?}"
    );
    assert_eq!(
        count(&db, "presence_receipts").await,
        1,
        "the second use stored no receipt"
    );
}

/// The crash-injection row of §8: consumption must share the mutation's
/// transaction.
///
/// Two things have to hold, and they are checked separately because only one of
/// them is still reachable through the public entry point.
///
/// *Statement order:* the nonce goes in before the mark, and a failing mark
/// propagates as an error rather than being swallowed. Driven directly here,
/// with an unknown page violating `page_truth_state`'s foreign key — the
/// entry point now reads the page inside the transaction and refuses a missing
/// one before any insert, so this failure cannot be reached through it.
///
/// *Rollback:* the entry point's only non-`Applied` path is a `ROLLBACK`, so an
/// error at that statement un-consumes the nonce. Checked below by rolling back
/// and showing the same capability still applies.
#[tokio::test]
async fn a_failed_mark_leaves_the_nonce_unspent() {
    let (db, root) = db().await;
    let missing = "page_00000000-0000-4000-8000-0000000000ff";
    let doomed = mint(
        "review_page",
        &[missing],
        "whatever digest",
        "app",
        "op-1",
        &[0xa1; 16],
        1_000,
        SECRET,
    );
    let verified = presence::verify(
        &doomed,
        SECRET,
        1_010,
        &presence::PresenceDemand {
            action: PresenceAction::ReviewPage,
            target_ids: &[missing.to_string()],
        },
    )
    .expect("the capability itself is well-formed");

    {
        let conn = db.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ()).await.unwrap();
        let failed = super::presence_review::apply_review_in_txn(
            &conn,
            &verified,
            missing,
            1,
            "whatever digest",
            1_010,
        )
        .await;
        assert!(failed.is_err(), "marking an unknown page must fail");
        conn.execute("ROLLBACK", ()).await.unwrap();
    }
    assert_eq!(
        count(&db, "presence_nonces").await,
        0,
        "a rolled-back mutation must not leave its nonce consumed"
    );
    assert_eq!(count(&db, "presence_receipts").await, 0);

    // And the proof that it is really unspent: the same nonce still works.
    let retry = gesture("op-1", 0xa1);
    assert!(matches!(
        db.review_page_with_presence(&retry, PAGE, Some(root.path()), 1_010)
            .await
            .unwrap(),
        super::ReviewOutcome::Applied(_)
    ));
}

/// T6, end to end: the page changed between the gesture and the submit, so the
/// human approved text that is no longer there.
///
/// A conflict rather than an invalid. The capability is genuine and its digest
/// is intact — what moved is the page — so the app's move is to show the human
/// what is there now and ask again, not to go looking for a binding bug.
#[tokio::test]
async fn a_gesture_against_stale_content_conflicts() {
    let (db, root) = db().await;
    let stale = mint(
        "review_page",
        &[PAGE],
        &crate::provenance::revision_content_digest("what the page used to say"),
        "app",
        "op-1",
        &[0xa1; 16],
        1_000,
        SECRET,
    );
    let outcome = db
        .review_page_with_presence(&stale, PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert!(
        matches!(
            outcome,
            super::ReviewOutcome::Refused(PresenceRefusal::Conflict)
        ),
        "expected a conflict, got {outcome:?}"
    );
    assert!(!reviewed(&db).await);
}

/// T2 at the mutation: a client that cannot read the secret produces something
/// well-formed that does not verify, and nothing is written.
#[tokio::test]
async fn a_forged_capability_writes_nothing() {
    let (db, root) = db().await;
    let forged = mint(
        "review_page",
        &[PAGE],
        &body_digest(),
        "mcp-client",
        "op-1",
        &[0xa1; 16],
        1_000,
        b"not the install secret, thirty-two",
    );
    let outcome = db
        .review_page_with_presence(&forged, PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert!(
        matches!(
            outcome,
            super::ReviewOutcome::Refused(PresenceRefusal::Invalid)
        ),
        "expected a refusal, got {outcome:?}"
    );
    assert!(!reviewed(&db).await);
    assert_eq!(count(&db, "presence_nonces").await, 0);
}

/// §7: what lands in the tables is the digest, never the nonce.
#[tokio::test]
async fn the_stored_nonce_is_only_ever_a_digest() {
    let (db, root) = db().await;
    let cap = gesture("op-1", 0xa1);
    db.review_page_with_presence(&cap, PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query("SELECT nonce_digest FROM presence_nonces", ())
        .await
        .unwrap();
    let stored: String = rows.next().await.unwrap().unwrap().get(0).unwrap();
    assert_ne!(stored, cap.nonce, "the raw nonce must never be stored");
    assert_eq!(
        stored,
        presence::nonce_digest(&hex::decode(&cap.nonce).unwrap())
    );
}

/// §6's reaper: expired rows go, and removing one cannot resurrect anything —
/// the capability it belonged to is refused by expiry alone.
#[tokio::test]
async fn expired_nonces_are_reaped_by_the_next_write() {
    let (db, root) = db().await;
    {
        let conn = db.conn.lock().await;
        conn.execute(
            "INSERT INTO presence_nonces
                 (nonce_digest, caller_id, operation_id, request_digest, consumed_at, expires_at)
             VALUES ('long-expired', 'app', 'op-0', 'd', 1, 500)",
            (),
        )
        .await
        .unwrap();
    }
    assert_eq!(count(&db, "presence_nonces").await, 1);

    db.review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert_eq!(
        count(&db, "presence_nonces").await,
        1,
        "the expired row is gone and only the fresh one remains"
    );
}

/// T10 / writer exclusivity. The machine axis moving must never move the human
/// one. Nothing but this mutation writes `human_reviewed = 1`, and the way that
/// stays true is that a support write cannot reach it.
#[tokio::test]
async fn a_machine_verdict_never_sets_the_human_axis() {
    let (db, _root) = db().await;
    {
        let conn = db.conn.lock().await;
        // Written as an insert rather than an update because a page distilled
        // after migration 99 has no truth row at all — the same gap the review
        // mutation upserts around.
        conn.execute(
            "INSERT INTO page_truth_state
                 (page_id, page_version, support_status, evaluated_at, human_reviewed, updated_at)
             VALUES (?1, 1, 'supported', 1700000000, 0, 1700000000)
             ON CONFLICT(page_id) DO UPDATE SET
                 support_status = 'supported', evaluated_at = 1700000000",
            libsql::params![PAGE],
        )
        .await
        .unwrap();
    }
    let states = db.page_truth_states(&[PAGE.to_string()]).await.unwrap();
    let truth = states.get(PAGE).unwrap();
    assert_eq!(truth.support, crate::truth_contract::Support::Supported);
    assert!(
        !truth.human_reviewed,
        "a machine verdict must not confer human review"
    );
}

/// T7, the case that made the ordering a finding rather than a preference: the
/// install secret was replaced between the write and the retry.
///
/// A rotated secret is worse than an expired capability. Expiry lets the client
/// re-mint; rotation means *nothing* it is holding can ever verify again. If the
/// secret is resolved before the receipt lookup, the retry of a mutation that
/// already succeeded is answered `presence_unavailable`, and the client
/// correctly reads that as "my write did not happen".
#[tokio::test]
async fn a_retry_after_the_secret_was_replaced_still_replays() {
    let (db, root) = db().await;
    let cap = gesture("op-1", 0xa1);
    let super::ReviewOutcome::Applied(applied) = db
        .review_page_with_presence(&cap, PAGE, Some(root.path()), 1_010)
        .await
        .unwrap()
    else {
        panic!("first execution must apply");
    };

    std::fs::remove_file(presence::secret_path(root.path())).unwrap();

    let retry = db
        .review_page_with_presence(&cap, PAGE, Some(root.path()), 1_020)
        .await
        .unwrap();
    let super::ReviewOutcome::Replayed(replayed) = retry else {
        panic!("a retry of an applied mutation must replay, not refuse: {retry:?}");
    };
    assert_eq!(replayed.nonce_digest, applied.nonce_digest);
    assert_eq!(replayed.verified_at, applied.verified_at);
}

/// The same, one step earlier: with no secret at all and no receipt on file,
/// the answer is still `presence_unavailable`. The reordering must not turn a
/// daemon that cannot check into one that waves the caller through.
#[tokio::test]
async fn a_first_attempt_without_a_secret_is_still_refused() {
    let (db, root) = db().await;
    std::fs::remove_file(presence::secret_path(root.path())).unwrap();
    let outcome = db
        .review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert!(
        matches!(
            outcome,
            super::ReviewOutcome::Refused(PresenceRefusal::Unavailable)
        ),
        "expected unavailable, got {outcome:?}"
    );
    assert!(!reviewed(&db).await);
    assert_eq!(count(&db, "presence_nonces").await, 0);
}

/// Idempotency has to be linearizable, not merely eventual.
///
/// Two deliveries of one request — the client retried, or a proxy duplicated it
/// — race. With the lookup and the insert in separate transactions both see "no
/// receipt", and the loser is told `presence_replayed` for a mutation it never
/// performed: a refusal that a client is entitled to read as "somebody replayed
/// my capability", over a write that succeeded. Every caller must come away
/// with the same receipt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_deliveries_of_one_request_all_end_with_the_same_receipt() {
    let (db, root) = db().await;
    let db = std::sync::Arc::new(db);
    let cap = std::sync::Arc::new(gesture("op-1", 0xa1));

    let mut handles = Vec::new();
    for _ in 0..4 {
        let db = db.clone();
        let cap = cap.clone();
        let path = root.path().to_path_buf();
        handles.push(tokio::spawn(async move {
            db.review_page_with_presence(&cap, PAGE, Some(&path), 1_010)
                .await
                .unwrap()
        }));
    }

    let mut applied = Vec::new();
    let mut replayed = Vec::new();
    for handle in handles {
        match handle.await.unwrap() {
            super::ReviewOutcome::Applied(receipt) => applied.push(receipt),
            super::ReviewOutcome::Replayed(receipt) => replayed.push(receipt),
            other => panic!("a concurrent delivery must never be refused: {other:?}"),
        }
    }

    assert_eq!(applied.len(), 1, "exactly one delivery may do the work");
    assert_eq!(replayed.len(), 3, "the rest must replay the stored answer");
    for receipt in &replayed {
        assert_eq!(receipt.nonce_digest, applied[0].nonce_digest);
        assert_eq!(receipt.verified_at, applied[0].verified_at);
    }
    assert_eq!(count(&db, "presence_receipts").await, 1);
    assert_eq!(count(&db, "presence_nonces").await, 1);
}

/// A caller who is not going to be authorized must not learn which pages exist.
///
/// The discriminator is an *expired* capability, because that is where the two
/// paths used to diverge: reading the page first, a missing one came back
/// `presence_invalid` before expiry was ever considered, while an existing one
/// got as far as the expiry check and came back `presence_expired`. Two answers,
/// no capability required, and the page table is walkable one id at a time. A
/// junk MAC alone never showed this — both cases answered `presence_invalid`
/// either way — which is why the probe here is a capability that is well-formed
/// and simply too old.
#[tokio::test]
async fn a_caller_who_will_be_refused_cannot_tell_which_pages_exist() {
    let (db, root) = db().await;
    let missing = "page_00000000-0000-4000-8000-0000000000ff";

    let code = |outcome: &super::ReviewOutcome| match outcome {
        super::ReviewOutcome::Refused(refusal) => refusal.code().to_string(),
        other => panic!("this probe must be refused: {other:?}"),
    };

    // Well-formed, correctly signed, and long past its window.
    let expired_here = gesture("op-1", 0xc3);
    let expired_there = mint(
        "review_page",
        &[missing],
        &body_digest(),
        "app",
        "op-2",
        &[0xc4; 16],
        1_000,
        SECRET,
    );
    let against_existing = db
        .review_page_with_presence(&expired_here, PAGE, Some(root.path()), 9_999)
        .await
        .unwrap();
    let against_missing = db
        .review_page_with_presence(&expired_there, missing, Some(root.path()), 9_999)
        .await
        .unwrap();
    assert_eq!(
        code(&against_existing),
        code(&against_missing),
        "an expired capability must not report whether the page exists"
    );
    assert_eq!(code(&against_existing), "presence_expired");

    // And the same for a forgery, which was already indistinguishable and has
    // to stay that way.
    let forge = |target: &str, op: &str| {
        mint(
            "review_page",
            &[target],
            &body_digest(),
            "prober",
            op,
            &[0xc5; 16],
            1_000,
            b"not the install secret, thirty-two",
        )
    };
    let forged_existing = db
        .review_page_with_presence(&forge(PAGE, "op-3"), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap();
    let forged_missing = db
        .review_page_with_presence(&forge(missing, "op-4"), missing, Some(root.path()), 1_010)
        .await
        .unwrap();
    assert_eq!(code(&forged_existing), code(&forged_missing));
    assert_eq!(code(&forged_existing), "presence_invalid");

    assert_eq!(count(&db, "presence_nonces").await, 0);
    assert_eq!(count(&db, "presence_receipts").await, 0);
}

/// F1(a): the receipt has to describe what was actually marked.
///
/// Reading the page's version and digest before opening the transaction leaves
/// a window for an edit to land in between, after which the stored receipt
/// attests to content that was never the content marked. Reading inside the
/// transaction closes it structurally — this pins the property the structure
/// provides, so a later refactor that hoists the read back out fails here.
#[tokio::test]
async fn the_receipt_records_the_version_and_digest_that_were_marked() {
    let (db, root) = db().await;
    let super::ReviewOutcome::Applied(receipt) = db
        .review_page_with_presence(&gesture("op-1", 0xa1), PAGE, Some(root.path()), 1_010)
        .await
        .unwrap()
    else {
        panic!("must apply");
    };

    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT t.reviewed_page_version, t.reviewed_page_digest, p.version, p.content
               FROM page_truth_state t JOIN pages p ON p.id = t.page_id
              WHERE t.page_id = ?1",
            libsql::params![PAGE],
        )
        .await
        .unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let marked_version: i64 = row.get(0).unwrap();
    let marked_digest: String = row.get(1).unwrap();
    let live_version: i64 = row.get(2).unwrap();
    let live_content: String = row.get(3).unwrap();

    assert_eq!(receipt.reviewed_page_version, marked_version);
    assert_eq!(receipt.reviewed_page_digest, marked_digest);
    assert_eq!(marked_version, live_version);
    assert_eq!(
        marked_digest,
        crate::provenance::revision_content_digest(&live_content),
        "the mark must describe the row the transaction actually saw"
    );
}
