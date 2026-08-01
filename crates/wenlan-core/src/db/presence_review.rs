// SPDX-License-Identifier: Apache-2.0
//! The human-review mutation, in the order the presence threat model requires.
//!
//! Binding spec: `docs/plans/2026-07-27-m5-presence-threat-model.md` §4 and §6.
//! The whole sequence lives in one function on purpose. Split across a handler
//! it can be reassembled in the wrong order by anybody adding a second
//! presence-carrying mutation later, and the wrong order is not a crash — it is
//! a retry that reports failure over a write that succeeded.

use super::MemoryDB;
use crate::error::WenlanError;
use crate::presence::{PresenceAction, PresenceDemand, PresenceRefusal, VerifiedPresence};
use std::path::Path;
use wenlan_types::requests::PresenceCapability;
use wenlan_types::responses::PageReviewReceipt;

/// What one presence-carrying review request did.
///
/// Database failures are still [`WenlanError`]; this enum carries only answers
/// the protocol itself has, which is why a refusal is a value here rather than
/// an error. A caller that treats `Refused` as success has to ignore a variant
/// rather than swallow an error type.
#[derive(Debug, Clone)]
pub enum ReviewOutcome {
    /// First execution: the nonce was consumed and the page was marked.
    Applied(PageReviewReceipt),
    /// A retry of a request that already ran. The stored response, unchanged,
    /// and no nonce was consumed.
    Replayed(PageReviewReceipt),
    Refused(PresenceRefusal),
}

impl MemoryDB {
    /// Mark one page human-reviewed on the strength of a presence capability.
    ///
    /// ```text
    /// BEGIN IMMEDIATE
    ///   receipt replay / collision lookup   <- FIRST; no secret, no page
    ///           |
    ///   load secret + verify capability     <- pure; still no page
    ///           |
    ///   read page binding, check T6, consume nonce, mark, store receipt
    /// COMMIT
    /// ```
    ///
    /// **One transaction, one connection, one lock acquisition**, because every
    /// adjacent pair of steps here is a race otherwise. Lookup and insert in
    /// separate transactions let two concurrent retries of one operation both
    /// see "no receipt", after which the loser is told `presence_replayed` for a
    /// mutation it never performed, or collides on the receipt key and surfaces
    /// a database error as a 500. Reading the page's version and digest outside
    /// the transaction lets an edit land in between, so the receipt records
    /// content that was never what got marked.
    ///
    /// **The lookup runs first** because validation can fail for reasons a retry
    /// cannot avoid. A capability lives sixty seconds, so a client retrying
    /// after a dropped response is very likely holding an expired one — and if
    /// the install secret was rotated in between, no capability it holds can
    /// ever verify again. Validating first answers `presence_expired` or
    /// `presence_unavailable` to a request that already succeeded, and the
    /// client correctly concludes its write failed. Looking first answers with
    /// the stored receipt.
    ///
    /// The lookup is keyed on fields taken from an unverified capability, which
    /// is safe: it only reads, and what it returns carries no capability
    /// material. Using the *submitted* action, targets, and digest rather than
    /// the demanded ones is what makes a collision detectable at all — compare
    /// the demanded values and every request under one operation ID hashes
    /// alike, and T8 can never fire.
    ///
    /// **The page is not touched until the MAC verifies**, so a caller holding a
    /// junk MAC cannot tell `presence_invalid` (no such page) from
    /// `presence_expired` (page exists) and walk the page table without ever
    /// holding a capability.
    pub async fn review_page_with_presence(
        &self,
        submitted: &PresenceCapability,
        page_id: &str,
        presence_root: Option<&Path>,
        now: i64,
    ) -> Result<ReviewOutcome, WenlanError> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("presence review begin: {e}")))?;

        let result = review_in_txn(&conn, submitted, page_id, presence_root, now).await;

        match result {
            Ok(ReviewOutcome::Applied(receipt)) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("presence review commit: {e}")))?;
                Ok(ReviewOutcome::Applied(receipt))
            }
            // Every other path — a replay, a refusal, or an error — has nothing
            // it wants to keep.
            other => {
                if let Err(e) = conn.execute("ROLLBACK", ()).await {
                    return Err(WenlanError::VectorDb(format!(
                        "presence review rollback failed; the nonce may be spent \
                         against no mutation: {e}"
                    )));
                }
                other
            }
        }
    }
}

/// The ordered body, inside the caller's transaction.
async fn review_in_txn(
    conn: &libsql::Connection,
    submitted: &PresenceCapability,
    page_id: &str,
    presence_root: Option<&Path>,
    now: i64,
) -> Result<ReviewOutcome, WenlanError> {
    // Step 1: has this caller run this operation before, and with what?
    //
    // Keyed on the caller and operation ID alone. Nothing about the submitted
    // request is inspected before the row on file is in hand — not the MAC, not
    // the window, not even whether the action parses. Anything that answers
    // ahead of this lookup answers for an operation ID whose mutation may
    // already have happened, and every refusal reads to a client as "your write
    // did not land".
    if let Some(stored) =
        stored_receipt(conn, &submitted.caller_id, &submitted.operation_id).await?
    {
        // Only now is the request itself looked at, and only to tell a replay
        // from a collision on the operation ID.
        let submitted_digest = parse_action(&submitted.action).map(|action| {
            crate::presence::request_digest(action, &submitted.target_ids, &submitted.base_digest)
        });
        return answer_for(stored, submitted_digest.as_deref());
    }

    // Step 2: the secret and the capability. Nothing here reads a row, so
    // nothing here can leak one — and `presence_unavailable` is reachable only
    // now, past the replay path, so a rotated secret cannot swallow the answer
    // to a mutation that already happened. An action the daemon does not know
    // is refused here too, by `verify` itself.
    let secret = match presence_root.map(crate::presence::load_secret) {
        Some(Ok(secret)) => secret,
        _ => return Ok(ReviewOutcome::Refused(PresenceRefusal::Unavailable)),
    };
    let targets = vec![page_id.to_string()];
    let verified = match crate::presence::verify(
        submitted,
        &secret,
        now.max(0) as u64,
        &PresenceDemand {
            action: PresenceAction::ReviewPage,
            target_ids: &targets,
        },
    ) {
        Ok(verified) => verified,
        Err(refusal) => return Ok(ReviewOutcome::Refused(refusal)),
    };

    // Step 3: now the page, read in the transaction that is about to mark it, so
    // the version and digest the receipt records are the ones actually marked.
    let Some((version, live_digest)) = page_binding(conn, page_id).await? else {
        return Ok(ReviewOutcome::Refused(PresenceRefusal::Invalid));
    };
    // T6: the human approved the text in front of them, and the page has since
    // moved on, so that approval is about content nobody is looking at any more.
    // A conflict rather than an invalid — the capability is genuine and the
    // world changed under it, so the app's move is to show what is there now and
    // ask again, not to hunt for a binding bug.
    if verified.base_digest != live_digest {
        return Ok(ReviewOutcome::Refused(PresenceRefusal::Conflict));
    }

    apply_review_in_txn(conn, &verified, page_id, version, &live_digest, now).await
}

/// Step 1 of §4: the answer already on file, if there is one.
///
/// `Some` means this request is finished with — either the stored response for a
/// retry, or a refusal because one operation ID has been used for two different
/// mutations. `None` means this is the first execution.
/// The row on file for this caller and operation, if there is one.
///
/// The key is the caller and operation ID and nothing else. Deliberately no
/// property of the submitted request takes part, so nothing about the request
/// can decide the answer before the row is in hand.
async fn stored_receipt(
    conn: &libsql::Connection,
    caller_id: &str,
    operation_id: &str,
) -> Result<Option<(String, String)>, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT request_digest, response_json FROM presence_receipts
              WHERE caller_id = ?1 AND operation_id = ?2",
            libsql::params![caller_id, operation_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("presence_receipt: {e}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("presence_receipt next: {e}")))?
    else {
        return Ok(None);
    };
    Ok(Some((
        row.get(0).unwrap_or_default(),
        row.get(1).unwrap_or_default(),
    )))
}

/// Replay or conflict, once a row is on file.
///
/// `submitted_digest` is `None` when the request cannot be hashed at all —
/// an action the daemon does not know. That is not the request on file, so it
/// is a second request under a spent operation ID: a conflict. Refusing it as
/// invalid instead would tell a client retrying a mutation that already
/// happened that its write never landed.
fn answer_for(
    stored: (String, String),
    submitted_digest: Option<&str>,
) -> Result<ReviewOutcome, WenlanError> {
    let (stored_digest, response_json) = stored;
    if submitted_digest != Some(stored_digest.as_str()) {
        return Ok(ReviewOutcome::Refused(PresenceRefusal::Conflict));
    }
    let receipt: PageReviewReceipt = serde_json::from_str(&response_json)
        .map_err(|e| WenlanError::VectorDb(format!("presence receipt is unreadable: {e}")))?;
    Ok(ReviewOutcome::Replayed(receipt))
}

/// The page's live version and the digest of its exact current content.
///
/// `None` means no such page. The digest is
/// [`crate::provenance::revision_content_digest`] — whitespace-INtolerant,
/// because the question is "is this the exact text the human was looking at",
/// not "is this the same content".
async fn page_binding(
    conn: &libsql::Connection,
    page_id: &str,
) -> Result<Option<(i64, String)>, WenlanError> {
    let mut rows = conn
        .query(
            "SELECT version, content FROM pages WHERE id = ?1",
            libsql::params![page_id],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("page_review_binding: {e}")))?;
    let Some(row) = rows
        .next()
        .await
        .map_err(|e| WenlanError::VectorDb(format!("page_review_binding next: {e}")))?
    else {
        return Ok(None);
    };
    let version: i64 = row.get(0).unwrap_or(1);
    let content: String = row.get(1).unwrap_or_default();
    Ok(Some((
        version,
        crate::provenance::revision_content_digest(&content),
    )))
}

/// Nonce consumption, the mark, and the receipt (§6). Statement order inside
/// matters as much as the transaction around it: the nonce goes in first, so
/// anything below it failing leaves the capability unspent once the caller rolls
/// back.
pub(super) async fn apply_review_in_txn(
    conn: &libsql::Connection,
    verified: &VerifiedPresence,
    page_id: &str,
    version: i64,
    digest: &str,
    now: i64,
) -> Result<ReviewOutcome, WenlanError> {
    let receipt = PageReviewReceipt {
        page_id: page_id.to_string(),
        human_reviewed: true,
        reviewed_page_version: version,
        reviewed_page_digest: digest.to_string(),
        protocol_version: verified.protocol_version,
        nonce_digest: verified.nonce_digest.clone(),
        verified_at: now,
        caller_id: verified.caller_id.clone(),
        operation_id: verified.operation_id.clone(),
    };
    let response_json = serde_json::to_string(&receipt)
        .map_err(|e| WenlanError::VectorDb(format!("presence receipt encode: {e}")))?;

    let consumed = conn
        .execute(
            "INSERT INTO presence_nonces
                 (nonce_digest, caller_id, operation_id, request_digest,
                  consumed_at, expires_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(nonce_digest) DO NOTHING",
            libsql::params![
                verified.nonce_digest.as_str(),
                verified.caller_id.as_str(),
                verified.operation_id.as_str(),
                verified.request_digest.as_str(),
                now,
                verified.expires_at as i64,
            ],
        )
        .await
        .map_err(|e| WenlanError::VectorDb(format!("presence nonce consume: {e}")))?;
    if consumed == 0 {
        // The digest is already in the table, so this capability has been spent.
        // Not a receipt replay — that was answered above and never reaches here
        // — so it is a second use of one nonce.
        return Ok(ReviewOutcome::Refused(PresenceRefusal::Replayed));
    }

    // Upsert rather than update: nothing creates a `page_truth_state` row for a
    // page distilled after migration 99, so a plain UPDATE would silently mark
    // nothing for exactly the newest pages.
    //
    // The inserted machine axis is `provisional` with a NULL `evaluated_at`,
    // which `page_truth_states` reads as *unevaluated*, not as *unsupported*.
    // The two axes are independent, and a human saying "I read this" is not
    // evidence about whether the machine found support — inventing `supported`
    // here would collapse the separation the whole rung exists to make.
    conn.execute(
        "INSERT INTO page_truth_state
             (page_id, page_version, support_status, provisional_reason,
              human_reviewed, reviewed_page_version, reviewed_page_digest,
              updated_at)
         VALUES (?1, ?2, 'provisional', 'never evaluated: no claim derivation has run',
                 1, ?2, ?3, ?4)
         ON CONFLICT(page_id) DO UPDATE SET
             human_reviewed = 1,
             reviewed_page_version = ?2,
             reviewed_page_digest = ?3,
             updated_at = ?4",
        libsql::params![page_id, version, digest, now],
    )
    .await
    .map_err(|e| WenlanError::VectorDb(format!("presence review mark: {e}")))?;

    if let Err(e) = conn
        .execute(
            "INSERT INTO presence_receipts
                 (caller_id, operation_id, request_digest, response_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            libsql::params![
                verified.caller_id.as_str(),
                verified.operation_id.as_str(),
                verified.request_digest.as_str(),
                response_json.as_str(),
                now,
            ],
        )
        .await
    {
        // Unreachable while this runs inside `BEGIN IMMEDIATE`: no other writer
        // can have landed a receipt since the lookup. If it ever becomes
        // reachable, the honest answer is the one already stored rather than a
        // 500 over a mutation somebody else completed — so look again before
        // giving up.
        return match stored_receipt(conn, &verified.caller_id, &verified.operation_id).await? {
            Some(stored) => answer_for(stored, Some(&verified.request_digest)),
            None => Err(WenlanError::VectorDb(format!(
                "presence receipt store: {e}"
            ))),
        };
    }

    // ponytail: reaping rides along on the write rather than on a scheduler
    // lane. Only rows already past `expires_at` go, and a capability that old is
    // refused by expiry alone, so removing its nonce cannot resurrect it. The
    // ceiling is that a daemon which never takes another review keeps its last
    // expired rows; add a scheduled sweep if that table is ever big enough to
    // notice.
    conn.execute(
        "DELETE FROM presence_nonces WHERE expires_at < ?1",
        libsql::params![now],
    )
    .await
    .map_err(|e| WenlanError::VectorDb(format!("presence nonce reap: {e}")))?;

    Ok(ReviewOutcome::Applied(receipt))
}

fn parse_action(action: &str) -> Option<PresenceAction> {
    match action {
        "attest_claim" => Some(PresenceAction::AttestClaim),
        "review_page" => Some(PresenceAction::ReviewPage),
        _ => None,
    }
}
