// SPDX-License-Identifier: Apache-2.0
//! Daemon-side validation of UI-presence capabilities (M5 D7).
//!
//! Binding spec: `docs/plans/2026-07-27-m5-presence-threat-model.md`. Read §1
//! before describing anything here as security. This is a **cooperative
//! local-client provenance boundary**: the daemon binds loopback and is
//! unauthenticated, and a hostile process running as the same user can read the
//! install secret off disk (T11, explicitly out of scope). What it does make
//! true is that a `human_reviewed` write came from a gesture that reached the
//! app's Tauri backend in the normal case, that an ordinary MCP or HTTP client
//! cannot forge one, and that a replayed or reordered request cannot
//! double-consume presence.
//!
//! **This half only validates.** The app mints (its `app/src/presence.rs`), and
//! the app alone owns the secret file's lifecycle — creating it, repairing its
//! permissions, replacing a corrupt one. Two creators would be two chances to
//! win a first-use race with different bytes, after which every capability the
//! loser minted fails against the winner's secret and nothing says why. So this
//! module reads and never writes: no create, no repair, no rotate.

use std::path::{Path, PathBuf};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// D7 §3: "protocol version, so the format can change without ambiguity".
/// Must match the app's `PROTOCOL_VERSION`.
pub const PROTOCOL_VERSION: u32 = 1;

/// The app's `SECRET_FILE_NAME`, resolved under the same data directory the
/// app passes the sidecar as `WENLAN_DATA_DIR`.
const SECRET_FILE_NAME: &str = "presence_secret";

/// The app's `SECRET_LEN`. A file of any other length is not a secret this
/// protocol produced, so it is refused rather than stretched or truncated.
const SECRET_LEN: usize = 32;

/// The longest window this protocol recognises (D7 §3: "valid for 60 seconds").
/// The app mints exactly this; anything wider is refused rather than honoured,
/// because the signature proves who minted it and not that they meant to.
const MAX_WINDOW_SECS: u64 = 60;

/// How far ahead of the daemon's clock a capability may claim to have been
/// minted. Both processes run on one machine, so this absorbs the ordinary
/// disagreement between two reads of the same clock and nothing more.
const MAX_CLOCK_SKEW_SECS: u64 = 5;

/// What one capability authorizes. A capability for one action is invalid for
/// the other (D7 §3, T4), which is enforced by the action being a signed
/// component rather than by any check at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceAction {
    AttestClaim,
    ReviewPage,
}

impl PresenceAction {
    /// The exact strings the app signs. Changing one silently invalidates every
    /// capability for that action.
    pub const fn as_str(self) -> &'static str {
        match self {
            PresenceAction::AttestClaim => "attest_claim",
            PresenceAction::ReviewPage => "review_page",
        }
    }
}

/// Why a presence-carrying request was refused.
///
/// Deliberately coarse (D7 §7). In particular [`Self::Invalid`] covers a bad
/// HMAC, an unknown action, an unparseable nonce, a wrong protocol version, and
/// a target the capability does not name — all as one answer, so a caller
/// cannot use the distinction between them to probe the secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceRefusal {
    /// The capability does not verify, or does not authorize what was asked.
    Invalid,
    /// It verified, but its 60-second window has closed.
    Expired,
    /// Its nonce has already been consumed.
    Replayed,
    /// The same caller reused an operation ID for different content.
    Conflict,
    /// The install secret is absent or unreadable, so presence cannot be
    /// checked at all. **Not** a caller error, and not a licence to proceed:
    /// D7 §5 says every presence-requiring mutation is refused in this state.
    Unavailable,
}

impl PresenceRefusal {
    /// The wire code. Never carries a reason, a path, or any part of the
    /// submitted capability.
    pub const fn code(self) -> &'static str {
        match self {
            PresenceRefusal::Invalid => "presence_invalid",
            PresenceRefusal::Expired => "presence_expired",
            PresenceRefusal::Replayed => "presence_replayed",
            PresenceRefusal::Conflict => "presence_conflict",
            PresenceRefusal::Unavailable => "presence_unavailable",
        }
    }
}

impl std::fmt::Display for PresenceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

/// Where the install secret lives, given the daemon's data root.
pub fn secret_path(root: &Path) -> PathBuf {
    root.join(SECRET_FILE_NAME)
}

/// Read the per-install secret. Never creates one, never repairs one.
///
/// Every failure — missing file, unreadable file, wrong length, loose mode — is
/// the same [`PresenceRefusal::Unavailable`]. A daemon that started before the
/// app ever minted anything is in exactly this state, and the honest answer
/// there is "cannot check", never "trust the caller" (D7 §5).
///
/// On Unix a secret any other user can read is refused (T2). This side cannot
/// honestly say a capability proves a person was present while the key that
/// signs it is readable by every process on the machine — the whole claim rests
/// on that file. It refuses rather than tightening the mode itself: the app owns
/// the file's lifecycle, and silently repairing it would hide a state somebody
/// needs to see, since a secret that was world-readable may already have been
/// copied. On non-Unix the mode carries no comparable meaning and the app
/// refuses to mint there, so no capability exists for this side to check.
pub fn load_secret(root: &Path) -> Result<Vec<u8>, PresenceRefusal> {
    let path = secret_path(root);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path)
            .map_err(|_| PresenceRefusal::Unavailable)?
            .permissions()
            .mode();
        if mode & 0o077 != 0 {
            return Err(PresenceRefusal::Unavailable);
        }
    }
    let bytes = std::fs::read(&path).map_err(|_| PresenceRefusal::Unavailable)?;
    if bytes.len() != SECRET_LEN {
        return Err(PresenceRefusal::Unavailable);
    }
    Ok(bytes)
}

/// Appends one component as its big-endian `u64` byte length followed by its
/// bytes. Mirrors the app's `push_component` exactly.
///
/// The boundary is a count read before the bytes, not a value searched for
/// inside them, so nothing a component contains can be read as a separator. The
/// width has to be wide enough that no length can wrap it: a `u32` prefix would
/// sign a component of `2^32 + n` bytes identically to one of `n` bytes, which
/// puts the boundary confusion straight back.
fn push_component(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    out.extend_from_slice(bytes);
}

/// The canonical bytes a capability's HMAC is computed over.
///
/// This is a byte-for-byte mirror of the app's `signed_bytes`. It is not an
/// interpretation of it: the two run in different processes against the same
/// secret, so any disagreement about layout shows up only as "every capability
/// is invalid", with nothing pointing at which side moved. The layout is pinned
/// literally in this module's tests against the same hex the app pins.
#[allow(clippy::too_many_arguments)]
fn signed_bytes(
    protocol_version: u32,
    action: PresenceAction,
    target_ids: &[String],
    base_digest: &str,
    caller_id: &str,
    operation_id: &str,
    nonce: &[u8],
    minted_at: u64,
    expires_at: u64,
) -> Vec<u8> {
    let mut out = Vec::new();
    push_component(&mut out, &protocol_version.to_be_bytes());
    push_component(&mut out, action.as_str().as_bytes());
    out.extend_from_slice(&(target_ids.len() as u64).to_be_bytes());
    for target_id in target_ids {
        push_component(&mut out, target_id.as_bytes());
    }
    push_component(&mut out, base_digest.as_bytes());
    push_component(&mut out, caller_id.as_bytes());
    push_component(&mut out, operation_id.as_bytes());
    push_component(&mut out, nonce);
    push_component(&mut out, &minted_at.to_be_bytes());
    push_component(&mut out, &expires_at.to_be_bytes());
    out
}

/// SHA-256 of the raw nonce, lowercase hex. The **only** form of a nonce that
/// may be stored, logged, or returned (D7 §6, §7).
pub fn nonce_digest(nonce: &[u8]) -> String {
    hex::encode(Sha256::digest(nonce))
}

/// What a request means, independent of which capability carried it.
///
/// This is the `request_digest` D7 §6 uses to tell a retry (T7) from a
/// collision (T8): the same caller and operation ID arriving with this same
/// digest is the same mutation asked for twice, and arriving with a different
/// one is two different mutations wearing one operation ID.
///
/// The nonce is deliberately **not** in it. A client that re-mints before
/// retrying — new nonce, identical intent — is retrying, and reading that as a
/// collision would refuse a write that had already succeeded. Reuses
/// [`push_component`] so the digest inherits the same unambiguous framing
/// rather than inventing a second one.
pub fn request_digest(action: PresenceAction, target_ids: &[String], base_digest: &str) -> String {
    let mut out = Vec::new();
    push_component(&mut out, action.as_str().as_bytes());
    out.extend_from_slice(&(target_ids.len() as u64).to_be_bytes());
    for target_id in target_ids {
        push_component(&mut out, target_id.as_bytes());
    }
    push_component(&mut out, base_digest.as_bytes());
    hex::encode(Sha256::digest(&out))
}

/// A capability that has verified, with the fields a caller may act on.
///
/// Holds no HMAC and no raw nonce — only the digest — so nothing downstream can
/// log or serialize capability material even by accident (D7 §7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPresence {
    pub action: PresenceAction,
    pub target_ids: Vec<String>,
    pub base_digest: String,
    pub caller_id: String,
    pub operation_id: String,
    pub protocol_version: u32,
    pub nonce_digest: String,
    pub expires_at: u64,
    /// [`request_digest`] over this capability's action, targets, and digest.
    pub request_digest: String,
}

/// What the caller is asking the capability to authorize. Every field is
/// checked against the signed capability, so a mismatch is a refusal rather
/// than a downgrade.
pub struct PresenceDemand<'a> {
    pub action: PresenceAction,
    /// Exactly the targets, in order. A capability naming more is not a
    /// capability for fewer.
    pub target_ids: &'a [String],
    /// The digest of the content the daemon can actually see right now (T6).
    pub base_digest: &'a str,
}

/// Check a submitted capability against the secret and the demand.
///
/// Order matters only in that expiry is cheap and the MAC is not; both are
/// checked, and a capability that fails either is refused. It does **not**
/// touch the nonce store — replay is a database question, answered by the
/// caller inside the mutation transaction (D7 §4, §6).
pub fn verify(
    submitted: &wenlan_types::requests::PresenceCapability,
    secret: &[u8],
    now: u64,
    demand: &PresenceDemand<'_>,
) -> Result<VerifiedPresence, PresenceRefusal> {
    if submitted.protocol_version != PROTOCOL_VERSION {
        return Err(PresenceRefusal::Invalid);
    }
    let action = match submitted.action.as_str() {
        "attest_claim" => PresenceAction::AttestClaim,
        "review_page" => PresenceAction::ReviewPage,
        _ => return Err(PresenceRefusal::Invalid),
    };
    if action != demand.action
        || submitted.target_ids != demand.target_ids
        || submitted.base_digest != demand.base_digest
    {
        return Err(PresenceRefusal::Invalid);
    }
    // The window's shape, before its end. `now >= expires_at` can only refuse a
    // window that has closed; it has nothing to say about one that should never
    // have opened. A capability minted with a ten-year expiry is inside its
    // window for ten years and carries a MAC the app's own secret produced, so
    // every other check here passes it — a standing licence rather than a
    // gesture. A malformed window is `Invalid` rather than `Expired` because it
    // is a request nobody should have sent, not a real one that aged out.
    if submitted.expires_at <= submitted.minted_at
        || submitted.expires_at - submitted.minted_at > MAX_WINDOW_SECS
        // Minted in the daemon's future, by more than two clocks on one machine
        // can plausibly differ. Such a capability outlives a real one by exactly
        // the amount the caller chose.
        || submitted.minted_at > now.saturating_add(MAX_CLOCK_SKEW_SECS)
    {
        return Err(PresenceRefusal::Invalid);
    }
    // Checked before the MAC because an expired capability is a real one whose
    // window closed, and saying so is more useful than "invalid" — the app can
    // re-mint rather than hunt for a binding bug. It leaks nothing: the times
    // are the caller's own, already in the request it just sent.
    if now >= submitted.expires_at {
        return Err(PresenceRefusal::Expired);
    }
    let nonce = hex::decode(&submitted.nonce).map_err(|_| PresenceRefusal::Invalid)?;
    let mac = hex::decode(&submitted.mac).map_err(|_| PresenceRefusal::Invalid)?;
    let bytes = signed_bytes(
        submitted.protocol_version,
        action,
        &submitted.target_ids,
        &submitted.base_digest,
        &submitted.caller_id,
        &submitted.operation_id,
        &nonce,
        submitted.minted_at,
        submitted.expires_at,
    );
    let mut computed =
        HmacSha256::new_from_slice(secret).map_err(|_| PresenceRefusal::Unavailable)?;
    computed.update(&bytes);
    // `verify_slice` is the constant-time compare. A `==` on the two byte
    // arrays would leak the position of the first differing byte through timing
    // — the one channel that could actually chip at the MAC.
    computed
        .verify_slice(&mac)
        .map_err(|_| PresenceRefusal::Invalid)?;

    Ok(VerifiedPresence {
        action,
        target_ids: submitted.target_ids.clone(),
        base_digest: submitted.base_digest.clone(),
        caller_id: submitted.caller_id.clone(),
        operation_id: submitted.operation_id.clone(),
        protocol_version: submitted.protocol_version,
        nonce_digest: nonce_digest(&nonce),
        expires_at: submitted.expires_at,
        request_digest: request_digest(action, &submitted.target_ids, &submitted.base_digest),
    })
}

#[cfg(test)]
#[path = "presence_test.rs"]
mod presence_test;
