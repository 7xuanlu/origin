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
    /// The page's live version and the digest of its exact current content.
    ///
    /// `None` means no such page. The digest is
    /// [`crate::provenance::revision_content_digest`] — whitespace-INtolerant,
    /// because the question is "is this the exact text the human was looking
    /// at", not "is this the same content".
    pub async fn page_review_binding(
        &self,
        page_id: &str,
    ) -> Result<Option<(i64, String)>, WenlanError> {
        let conn = self.conn.lock().await;
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

    /// Mark one page human-reviewed on the strength of a presence capability.
    ///
    /// ```text
    /// receipt replay / collision lookup   <- FIRST
    ///         |
    /// capability validation + nonce consumption   <- only on first execution
    ///         |
    /// mutation
    /// ```
    ///
    /// The lookup runs first because validation can fail for reasons a retry
    /// cannot avoid. A capability lives 60 seconds; a client that retries after
    /// a dropped response is very likely holding an expired one. Validating
    /// first answers `presence_expired` to a request that already succeeded,
    /// and the client correctly concludes its write failed. Looking first
    /// answers with the stored receipt.
    ///
    /// It is keyed on fields taken from an unverified capability, which is
    /// safe: the lookup only reads, and what it returns carries no capability
    /// material. Using the *submitted* action, targets, and digest rather than
    /// the demanded ones is what makes a collision detectable at all — compare
    /// the demanded values and every request under one operation ID hashes
    /// alike, and T8 can never fire.
    pub async fn review_page_with_presence(
        &self,
        submitted: &wenlan_types::requests::PresenceCapability,
        page_id: &str,
        secret: &[u8],
        now: i64,
    ) -> Result<ReviewOutcome, WenlanError> {
        let Some(action) = parse_action(&submitted.action) else {
            return Ok(ReviewOutcome::Refused(PresenceRefusal::Invalid));
        };
        let submitted_digest =
            crate::presence::request_digest(action, &submitted.target_ids, &submitted.base_digest);

        if let Some((stored_digest, response_json)) = self
            .presence_receipt(&submitted.caller_id, &submitted.operation_id)
            .await?
        {
            if stored_digest != submitted_digest {
                return Ok(ReviewOutcome::Refused(PresenceRefusal::Conflict));
            }
            let receipt: PageReviewReceipt = serde_json::from_str(&response_json).map_err(|e| {
                WenlanError::VectorDb(format!("presence receipt is unreadable: {e}"))
            })?;
            return Ok(ReviewOutcome::Replayed(receipt));
        }

        let Some((version, live_digest)) = self.page_review_binding(page_id).await? else {
            return Ok(ReviewOutcome::Refused(PresenceRefusal::Invalid));
        };

        let targets = vec![page_id.to_string()];
        let verified = match crate::presence::verify(
            submitted,
            secret,
            now.max(0) as u64,
            &PresenceDemand {
                action: PresenceAction::ReviewPage,
                target_ids: &targets,
                base_digest: &live_digest,
            },
        ) {
            Ok(verified) => verified,
            Err(refusal) => return Ok(ReviewOutcome::Refused(refusal)),
        };

        self.commit_page_review(&verified, page_id, version, &live_digest, now)
            .await
    }

    /// Step 1 of §4: has this caller run this operation before, and with what?
    async fn presence_receipt(
        &self,
        caller_id: &str,
        operation_id: &str,
    ) -> Result<Option<(String, String)>, WenlanError> {
        let conn = self.conn.lock().await;
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

    /// Nonce consumption, the mark, and the receipt — one transaction, so there
    /// is no window where one of the three happened and the others did not
    /// (§6). A failed mark leaves the nonce unconsumed and the capability still
    /// usable; a conflicting nonce rolls the mark back.
    pub(super) async fn commit_page_review(
        &self,
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

        let conn = self.conn.lock().await;
        conn.execute("BEGIN IMMEDIATE", ())
            .await
            .map_err(|e| WenlanError::VectorDb(format!("presence review begin: {e}")))?;

        let result = async {
            // First, so that anything below failing un-consumes it.
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
                // The digest is already in the table, so this capability has
                // been spent. Not a receipt replay — that was answered above
                // and never reaches here — so it is a second use of one nonce.
                return Ok(ReviewOutcome::Refused(PresenceRefusal::Replayed));
            }

            // Upsert rather than update: nothing creates a `page_truth_state`
            // row for a page distilled after migration 99, so a plain UPDATE
            // would silently mark nothing for exactly the newest pages.
            //
            // The inserted machine axis is `provisional` with a NULL
            // `evaluated_at`, which `page_truth_states` reads as *unevaluated*,
            // not as *unsupported*. The two axes are independent, and a human
            // saying "I read this" is not evidence about whether the machine
            // found support — inventing `supported` here would collapse the
            // separation the whole rung exists to make.
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

            conn.execute(
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
            .map_err(|e| WenlanError::VectorDb(format!("presence receipt store: {e}")))?;

            // ponytail: reaping rides along on the write rather than on a
            // scheduler lane. Only rows already past `expires_at` go, and a
            // capability that old is refused by expiry alone, so removing its
            // nonce cannot resurrect it. The ceiling is that a daemon which
            // never takes another review keeps its last expired rows; add a
            // scheduled sweep if that table is ever big enough to notice.
            conn.execute(
                "DELETE FROM presence_nonces WHERE expires_at < ?1",
                libsql::params![now],
            )
            .await
            .map_err(|e| WenlanError::VectorDb(format!("presence nonce reap: {e}")))?;

            Ok(ReviewOutcome::Applied(receipt))
        }
        .await;

        match result {
            Ok(ReviewOutcome::Applied(receipt)) => {
                conn.execute("COMMIT", ())
                    .await
                    .map_err(|e| WenlanError::VectorDb(format!("presence review commit: {e}")))?;
                Ok(ReviewOutcome::Applied(receipt))
            }
            // Every other path — a refusal or an error — leaves nothing behind.
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

fn parse_action(action: &str) -> Option<PresenceAction> {
    match action {
        "attest_claim" => Some(PresenceAction::AttestClaim),
        "review_page" => Some(PresenceAction::ReviewPage),
        _ => None,
    }
}
