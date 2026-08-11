// SPDX-License-Identifier: AGPL-3.0-only
//! The page-review command — the only Tauri command that carries a UI-presence
//! capability.
//!
//! Binding spec: `docs/plans/2026-07-27-m5-presence-threat-model.md` in the
//! `wenlan` daemon repo. Its most load-bearing step is §3.3: **the Tauri
//! backend submits the mutation itself**, so frontend JavaScript never receives
//! a capability. That is why minting and submitting are one command here rather
//! than a `mint` command whose result a second command posts — the intermediate
//! value in that design is a capability sitting in JavaScript, which is a
//! capability sitting anywhere JavaScript reaches (T1).
//!
//! What crosses into JavaScript, in both directions:
//!
//! - in: a page ID and the content the detail view rendered;
//! - out: [`crate::api::PageReviewOutcome`], which carries a version number and
//!   a variant name and nothing else.
//!
//! This module is kept separate from `search.rs`, where the other page commands
//! live, for the same reason the daemon keeps `db/presence_review.rs` separate
//! from its other page routes: the ordering and binding rules only make sense
//! read together, and they are easy to weaken one edit at a time when they are
//! spread through a large file.

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

use crate::api::PageReviewOutcome;
use crate::search::parse_release_version;
use crate::state::AppState;

type State = Arc<RwLock<AppState>>;

/// SHA-256 over the **exact bytes** of a page's content, lowercase hex.
///
/// A byte-for-byte mirror of `wenlan_core::provenance::revision_content_digest`,
/// which is what the daemon computes over the `content` column before deciding
/// whether the capability still binds (T6). The daemon's other digest,
/// `canonical_content_digest`, is whitespace-tolerant and answers a different
/// question — "is this the same content?" rather than "is this the exact text
/// the human was looking at?" — and swapping the two here would make a reflow
/// invisible to a check whose entire job is to notice one.
fn content_digest(content: &str) -> String {
    let digest = Sha256::digest(content.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The idempotency key for reviewing one page at one exact content.
///
/// Derived rather than random, which is what makes a retry a retry. The daemon
/// looks a repeated `(caller_id, operation_id)` up before it validates
/// anything (§4), and answers with the stored receipt — so a review whose
/// response was dropped in transit succeeds again on the next click instead of
/// failing as a spent nonce. A fresh random ID each time would make every
/// retry a new mutation and put the replay path permanently out of reach.
///
/// Including the digest is what keeps distinct reviews distinct: once the page
/// changes, reviewing it is a different operation, and reusing the earlier ID
/// for it would be the operation-ID collision of T8.
fn operation_id(page_id: &str, base_digest: &str) -> String {
    format!("review:{page_id}:{base_digest}")
}

/// Marks one page human-reviewed, binding the mark to the exact text the human
/// had in front of them.
///
/// `content` is the body the detail view rendered, not a digest computed in
/// JavaScript and not something re-fetched here. Re-fetching would be the
/// quiet defect in this whole path: it would approve whatever the page says
/// *now*, which is precisely the content nobody looked at, and T6 exists to
/// refuse exactly that.
#[tauri::command]
pub async fn review_page(
    state: tauri::State<'_, State>,
    page_id: String,
    content: String,
) -> Result<PageReviewOutcome, String> {
    let client = state.read().await.client.clone();
    let base_digest = content_digest(&content);
    let operation_id = operation_id(&page_id, &base_digest);

    let body = match crate::presence::mint_page_review(&page_id, &base_digest, &operation_id) {
        Ok(body) => body,
        // The reason names a path on disk. It belongs in the log, where an
        // operator can act on it, and not in a UI string — the daemon's own
        // refusals are deliberately coarse for the same reason, and answering
        // more precisely from this side would undo that.
        Err(crate::presence::PresenceError::SecretUnavailable(reason)) => {
            log::warn!("[presence] cannot mint a review capability: {reason}");
            return Ok(PageReviewOutcome::Unavailable);
        }
    };

    client.review_page(&page_id, &body).await
}

/// The first released daemon containing `POST /api/pages/{id}/review`.
///
/// The route landed in wenlan `1c903bec` (#418). `version.txt` at that commit
/// reads 0.15.2 and `v0.15.2` is one of its ancestors, so #418 missed that
/// release and — the repo bumps patch on every release — lands in the next.
/// Release PR #419 publishes exactly that 0.15.3 from a main already
/// containing `1c903bec`, so any daemon reporting 0.15.3 or above necessarily
/// carries the route.
const REVIEW_DAEMON_FLOOR: &str = "0.15.3";

fn daemon_version_supports_review(version: &str) -> bool {
    let Some(candidate) = parse_release_version(version) else {
        return false;
    };
    let floor = parse_release_version(REVIEW_DAEMON_FLOOR)
        .expect("review daemon floor is a static valid release version");

    candidate >= floor
}

/// Why the Review action is or is not offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PageReviewAvailability {
    Ready,
    /// This daemon has no review route — too old, or unreachable.
    DaemonUnsupported,
    /// This build cannot mint a capability at all, on any daemon.
    PlatformUnsupported,
}

/// Whether a review can succeed right now, and if not, which fact stops it.
///
/// Two independent things must hold, and they fail for unrelated reasons, so
/// the UI is told which one gave way rather than being handed a bare false.
///
/// **Can this build mint?** Checked first because it is local, free, and
/// nothing remote can overturn it: where `owner_only_supported` refuses, every
/// mint fails and no daemon upgrade helps.
///
/// **Does this daemon have the route?** Two signals, because neither alone is
/// enough. `truth.is_some()` proves it — the daemon shipped its truth field
/// and the review route in one commit (wenlan `1c903bec`, #418) — but it does
/// NOT disprove it. `handle_status`
/// (`crates/wenlan-server/src/routes.rs:133-138`) filters `generation > 0`
/// before building the field, and the daemon pins that on purpose:
/// `truth_status_test.rs:74`, `status_omits_truth_until_the_cutover_is_live`,
/// asserts the key is absent at generation 0 — "reported the same way an old
/// daemon reports it: not at all". So `truth.is_some()` is exactly
/// `generation > 0`, and a daemon carrying the route but not yet its cutover
/// ceremony reports no truth whatsoever. Gating on the field alone disables
/// the action for a pre-ceremony operator, who is precisely the person whose
/// reviews the ceremony later consumes. The version floor covers that daemon.
///
/// The durable answer is a `page_review` capability string in the daemon's
/// `capabilities` list, which `StatusResponse` already tells clients to
/// negotiate against instead of inferring support from a version. #418 did not
/// publish one; when it does, this collapses back to one check.
///
/// Fails closed: an unreachable daemon answers neither probe and reads the
/// same as one too old.
pub(crate) async fn review_availability(
    client: &crate::api::WenlanClient,
) -> PageReviewAvailability {
    if !crate::presence::minting_supported() {
        return PageReviewAvailability::PlatformUnsupported;
    }

    if matches!(client.truth_status().await, Ok(Some(_))) {
        return PageReviewAvailability::Ready;
    }

    match client.health().await {
        Ok(health) if daemon_version_supports_review(&health.version) => {
            PageReviewAvailability::Ready
        }
        _ => PageReviewAvailability::DaemonUnsupported,
    }
}

/// Whether the Review action can be offered, and why not when it cannot.
///
/// Delegates without adding anything: the decision and both probes live in
/// [`review_availability`], which the tests drive directly against a fake
/// daemon. Nothing here is worth a second implementation to get wrong.
#[tauri::command]
pub async fn page_review_supported(
    state: tauri::State<'_, State>,
) -> Result<PageReviewAvailability, String> {
    let client = state.read().await.client.clone();
    Ok(review_availability(&client).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_content_digest_is_the_one_the_daemon_computes() {
        // Pinned against the published SHA-256 of "hello" rather than against
        // this module's own output, so a change of algorithm cannot rewrite
        // its own expectation. The daemon hashes the `content` column with
        // plain SHA-256 over its exact bytes; if these two ever disagree,
        // every review returns `presence_conflict` and nothing says why.
        assert_eq!(
            content_digest("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        );
    }

    #[test]
    fn the_content_digest_notices_whitespace() {
        // The whole point of binding to `revision_content_digest` rather than
        // the canonical one. A reflow changes every line boundary, so a
        // whitespace-tolerant digest would let a human's approval carry over
        // to text they never read.
        assert_ne!(content_digest("a b"), content_digest("a  b"));
        assert_ne!(content_digest("a"), content_digest("a\n"));
    }

    #[test]
    fn one_page_at_one_content_always_gets_the_same_operation_id() {
        // What makes a dropped response recoverable: the retry hits the
        // daemon's receipt lookup instead of minting a second mutation.
        assert_eq!(
            operation_id("p1", &content_digest("body")),
            operation_id("p1", &content_digest("body")),
        );
    }

    #[test]
    fn editing_the_page_makes_reviewing_it_a_different_operation() {
        assert_ne!(
            operation_id("p1", &content_digest("body")),
            operation_id("p1", &content_digest("body edited")),
        );
        assert_ne!(
            operation_id("p1", &content_digest("body")),
            operation_id("p2", &content_digest("body")),
        );
    }

    /// A daemon that answers `/api/status` and `/api/health` by path, as many
    /// times as asked. The readiness probe makes up to two requests and their
    /// order is its business, not the test's.
    async fn serve_daemon(
        status_body: &'static str,
        health_body: &'static str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = [0_u8; 2048];
                let Ok(n) = stream.read(&mut buf).await else {
                    continue;
                };
                let request = String::from_utf8_lossy(&buf[..n]).to_string();
                let body = if request.contains("/api/health") {
                    health_body
                } else {
                    status_body
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{}", addr), handle)
    }

    fn client_for(base_url: String) -> crate::api::WenlanClient {
        crate::api::WenlanClient::with_base_url(base_url)
    }

    const NO_TRUTH: &str =
        r#"{"is_running":true,"files_indexed":0,"files_total":0,"sources_connected":[]}"#;
    const WITH_TRUTH: &str = r#"{"is_running":true,"files_indexed":0,"files_total":0,"sources_connected":[],"truth":{"cutover_generation":3,"contract_version":1}}"#;

    #[tokio::test]
    #[cfg(unix)]
    async fn a_daemon_before_its_cutover_can_still_take_reviews() {
        // The regression this pair exists for. `handle_status` omits the truth
        // field below generation 1, so a daemon carrying the review route but
        // not yet its ceremony sends no truth at all — and gating on that field
        // alone disabled the action for the operator whose reviews the ceremony
        // is about to consume. The version is what says the route is there.
        let (base_url, _server) = serve_daemon(
            NO_TRUTH,
            r#"{"status":"ok","db_initialized":true,"version":"0.15.3"}"#,
        )
        .await;

        assert_eq!(
            review_availability(&client_for(base_url)).await,
            PageReviewAvailability::Ready,
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_daemon_that_reports_its_cutover_needs_no_version_check() {
        let (base_url, _server) = serve_daemon(
            WITH_TRUTH,
            r#"{"status":"ok","db_initialized":true,"version":"0.0.1"}"#,
        )
        .await;

        assert_eq!(
            review_availability(&client_for(base_url)).await,
            PageReviewAvailability::Ready,
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn a_daemon_below_the_floor_with_no_cutover_is_unsupported() {
        let (base_url, _server) = serve_daemon(
            NO_TRUTH,
            r#"{"status":"ok","db_initialized":true,"version":"0.15.2"}"#,
        )
        .await;

        assert_eq!(
            review_availability(&client_for(base_url)).await,
            PageReviewAvailability::DaemonUnsupported,
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn an_unreachable_daemon_is_unsupported_rather_than_ready() {
        // Fail closed. Nothing is listening on this port.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);

        assert_eq!(
            review_availability(&client_for(base_url)).await,
            PageReviewAvailability::DaemonUnsupported,
        );
    }

    #[tokio::test]
    #[cfg(not(unix))]
    async fn a_platform_that_cannot_mint_says_so_instead_of_blaming_the_daemon() {
        // On Windows every mint refuses (`presence::owner_only_supported`), so
        // a daemon new enough to take the review still cannot be offered one.
        // Telling the user to upgrade the daemon would send them after the
        // wrong thing.
        let (base_url, _server) = serve_daemon(
            WITH_TRUTH,
            r#"{"status":"ok","db_initialized":true,"version":"9.9.9"}"#,
        )
        .await;

        assert_eq!(
            review_availability(&client_for(base_url)).await,
            PageReviewAvailability::PlatformUnsupported,
        );
    }

    #[test]
    fn the_floor_is_the_first_release_carrying_the_route() {
        assert!(daemon_version_supports_review("0.15.3"));
        assert!(daemon_version_supports_review("0.16.0"));
        assert!(!daemon_version_supports_review("0.15.2"));
        // A pre-release of the floor is not the released artifact it names.
        assert!(!daemon_version_supports_review("0.15.3-dev"));
        assert!(!daemon_version_supports_review("not-a-version"));
    }
}
