// SPDX-License-Identifier: AGPL-3.0-only
//! Tauri-owned UI-presence capability minting (M5 App PR, D7).
//!
//! This is a **cooperative local-client provenance boundary**, not hostile
//! same-user isolation — Wenlan's loopback daemon is unauthenticated, and a
//! hostile process running as the same macOS user can read this module's
//! secret because it can read anything that user can read. What this module
//! makes true: a mutation carrying a valid capability came from a concrete
//! user gesture that reached this Tauri backend, an ordinary MCP/HTTP client
//! cannot mint one, and the capability never reaches JavaScript. Binding
//! spec: `docs/plans/2026-07-27-m5-presence-threat-model.md` (in the
//! `wenlan` daemon repo).
//!
//! **Minting, plus the one wire shape a minted capability may take.** The
//! daemon half landed as `POST /api/pages/{id}/review` (daemon PR #418), so
//! the page-review action now has somewhere to go. [`mint_page_review`] mints
//! a capability and renders it as that route's request body in a single call,
//! and [`PresenceCapability`] itself never leaves this module — which is what
//! keeps T1 structural rather than a rule every future call site has to
//! remember. The claim-attest half still has no daemon route, so
//! [`PresenceAction::AttestClaim`] stays mint-only.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::identity_paths::app_data_dir;

type HmacSha256 = Hmac<Sha256>;

const SECRET_LEN: usize = 32;
const NONCE_LEN: usize = 16;
/// D7 §3: mint time + 60s.
const CAPABILITY_TTL_SECS: u64 = 60;
/// D7 §3: "protocol version, so the format can change without ambiguity".
pub const PROTOCOL_VERSION: u32 = 1;

const SECRET_FILE_NAME: &str = "presence_secret";
/// Sibling file whose only job is to be locked. It never holds secret bytes,
/// so its own permissions carry no weight beyond tidiness.
const SECRET_LOCK_FILE_NAME: &str = "presence_secret.lock";
/// Prefix every staging file carries, so a sweep can recognise the ones an
/// earlier run abandoned. Deliberately distinct from the lock file's name.
const STAGING_PREFIX: &str = "presence_secret.staging.";
/// What the sweep matches on, which is wider than what it writes — see
/// [`is_sweepable`]. Every sibling this module has ever created starts here.
const SWEEP_PREFIX: &str = "presence_secret.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceAction {
    AttestClaim,
    ReviewPage,
}

impl PresenceAction {
    fn as_str(self) -> &'static str {
        match self {
            PresenceAction::AttestClaim => "attest_claim",
            PresenceAction::ReviewPage => "review_page",
        }
    }
}

/// What one capability authorizes, before minting. `target_ids` and
/// `base_digest` are the exact-target/exact-version binding (D7 §3 T4, T6):
/// a capability minted against a stale page/revision digest is a capability
/// for content that no longer exists, so callers must re-mint against the
/// currently-viewed version rather than reusing an old request.
#[derive(Debug, Clone)]
pub struct PresenceRequest {
    pub action: PresenceAction,
    pub target_ids: Vec<String>,
    pub base_digest: String,
    pub caller_id: String,
    pub operation_id: String,
}

/// A minted, HMAC-signed capability.
///
/// Deliberately does **not** implement `serde::Serialize`. A `#[tauri::command]`
/// return type must implement `Serialize`, so this is a compile-time
/// guarantee that no command can hand a capability to JavaScript (T1) — the
/// alternative of implementing `Serialize` and trusting every future call
/// site to not use it is exactly the mistake the threat model calls out.
#[derive(Clone)]
pub struct PresenceCapability {
    pub action: PresenceAction,
    pub target_ids: Vec<String>,
    pub base_digest: String,
    pub caller_id: String,
    pub operation_id: String,
    pub minted_at: u64,
    pub expires_at: u64,
    pub protocol_version: u32,
    nonce: [u8; NONCE_LEN],
    mac: [u8; 32],
}

impl PresenceCapability {
    pub fn is_expired_at(&self, now: u64) -> bool {
        now >= self.expires_at
    }

    /// Safe to log, export, or return in an error (D7 §7): the nonce digest,
    /// never the nonce itself.
    pub fn nonce_digest(&self) -> String {
        hex(Sha256::digest(self.nonce))
    }
}

/// Redacted by construction: never prints `mac` or the raw nonce, only its
/// digest (D7 §7 — "never logged, exported, ... or returned in errors: the
/// HMAC, the raw capability, the install secret, the raw nonce").
impl std::fmt::Debug for PresenceCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PresenceCapability")
            .field("action", &self.action.as_str())
            .field("target_ids", &self.target_ids)
            .field("base_digest", &self.base_digest)
            .field("caller_id", &self.caller_id)
            .field("operation_id", &self.operation_id)
            .field("minted_at", &self.minted_at)
            .field("expires_at", &self.expires_at)
            .field("protocol_version", &self.protocol_version)
            .field("nonce_digest", &self.nonce_digest())
            .field("mac", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresenceError {
    /// The per-install secret could not be loaded or created. Presence
    /// minting is unavailable in this state — it must never degrade to
    /// trusting the caller instead (D7 §5).
    SecretUnavailable(String),
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}

fn random_bytes<const N: usize>() -> Result<[u8; N], PresenceError> {
    let mut buf = [0_u8; N];
    getrandom::getrandom(&mut buf)
        .map_err(|e| PresenceError::SecretUnavailable(format!("CSPRNG unavailable: {e}")))?;
    Ok(buf)
}

/// Creates a file that does not exist yet, owner-only from the moment it
/// does, and writes `bytes` into it. `create_new` is what makes this safe to
/// point at a shared directory: it fails rather than opening something
/// another writer is already using.
#[cfg(unix)]
fn create_owner_only(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    write_whole_file(path, file, bytes)
}

#[cfg(not(unix))]
fn create_owner_only(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    write_whole_file(path, file, bytes)
}

/// Fills a file that was just created, and deletes it if the write does not
/// complete. A half-written file is useless to every caller, and leaving it
/// behind would mean one orphan per failed attempt — the sweep would collect
/// it eventually, but only on the next successful load.
fn write_whole_file(path: &Path, mut file: std::fs::File, bytes: &[u8]) -> std::io::Result<()> {
    let written = std::io::Write::write_all(&mut file, bytes).and_then(|()| file.sync_all());
    if written.is_err() {
        let _ = std::fs::remove_file(path);
    }
    written
}

/// Opens (creating if needed) the file whose lock serialises access to the
/// secret. Owner-only on Unix for tidiness; it holds no secret bytes.
#[cfg(unix)]
fn open_lock_file(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_lock_file(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
}

/// Takes the exclusive lock covering the whole read-repair-replace-create
/// sequence, and returns the handle that holds it — dropping the handle
/// releases the lock.
///
/// Atomic single steps were not enough on their own. Two callers could both
/// read the same corrupt file, both stage a replacement, and both rename;
/// rename is atomic but it is not compare-and-swap, so the first caller
/// walked away with a secret that the second had already replaced on disk,
/// and every capability it went on to sign was unverifiable. Serialising the
/// sequence is what makes "read it, decide, act" a single decision.
fn lock_secret_dir(dir: &Path) -> Result<std::fs::File, PresenceError> {
    std::fs::create_dir_all(dir).map_err(|e| unavailable("create", dir, &e))?;
    let path = dir.join(SECRET_LOCK_FILE_NAME);
    let file = open_lock_file(&path).map_err(|e| unavailable("open lock", &path, &e))?;
    file.lock().map_err(|e| unavailable("lock", &path, &e))?;
    Ok(file)
}

/// Whether the sweep should collect `name`. Deliberately wider than the
/// current staging prefix: an earlier build staged as
/// `presence_secret.<pid>.<random>` with no `.staging.` segment, and those
/// orphans are already on installs that ran it. Matching the whole
/// `presence_secret.` family covers that scheme, the current one, and any
/// later rename, so this predicate never has to be revisited.
///
/// Two names are spared. The secret itself is `presence_secret` with no
/// trailing dot, so it cannot match in the first place; the lock file can,
/// and is excluded by name.
fn is_sweepable(name: &str) -> bool {
    name.starts_with(SWEEP_PREFIX) && name != SECRET_LOCK_FILE_NAME
}

/// Deletes staging files an earlier run abandoned. Only correct with the lock
/// held: the one staging file that might legitimately be in flight belongs to
/// whoever holds the lock, and that is this caller, which has not staged
/// anything yet.
///
/// Deletion failures are ignored on purpose. Every later load sweeps again
/// under the same lock, so a file that could not be removed this time is
/// simply retried next time — the condition heals itself, and failing the
/// load over it would take presence minting down for a leftover file that
/// harms nothing while it sits there.
fn sweep_staging_files(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if entry.file_name().to_str().is_some_and(is_sweepable) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

fn secret_path_in(dir: &Path) -> PathBuf {
    dir.join(SECRET_FILE_NAME)
}

fn unavailable(what: &str, path: &Path, error: &impl std::fmt::Display) -> PresenceError {
    PresenceError::SecretUnavailable(format!("{what} {}: {error}", path.display()))
}

/// Tightens an existing secret file to `0600` if an older build, an umask, or
/// a restore from backup left it looser. Refusing when the repair fails is
/// deliberate: a secret this process cannot make owner-only is one it cannot
/// promise anything about.
#[cfg(unix)]
fn repair_permissions(path: &Path) -> Result<(), PresenceError> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path)
        .map_err(|e| unavailable("stat", path, &e))?
        .permissions()
        .mode()
        & 0o777;
    if mode == 0o600 {
        return Ok(());
    }
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| unavailable("chmod", path, &e))
}

/// Non-Unix builds never reach this — [`owner_only_supported`] refuses first —
/// but it has to exist for the module to compile on those targets.
#[cfg(not(unix))]
fn repair_permissions(_path: &Path) -> Result<(), PresenceError> {
    Ok(())
}

/// Whether this platform can actually keep the secret file readable by its
/// owner alone. That guarantee is what the whole module rests on: without it
/// the secret is readable by anyone the parent directory's access control
/// admits, and a capability signed with it proves nothing about who asked.
///
/// Unix mode bits are the only implementation here, so every other platform
/// gets an honest refusal rather than a secret with inherited permissions and
/// a promise the module cannot keep. Windows needs real ACL work, which is
/// scheduled separately; until it lands, presence minting is unavailable
/// there. Refusing is the safe direction — no caller can mint a capability
/// that would be trusted.
#[cfg(unix)]
fn owner_only_supported() -> Result<(), PresenceError> {
    Ok(())
}

#[cfg(not(unix))]
fn owner_only_supported() -> Result<(), PresenceError> {
    Err(PresenceError::SecretUnavailable(
        "owner-only secret protection is not implemented on this platform; minting unavailable"
            .to_string(),
    ))
}

/// Whether this build can mint at all, answerable without touching the disk.
///
/// The UI needs this before offering a review: on a platform where
/// `owner_only_supported` refuses, every mint fails and no daemon upgrade
/// changes that, so an enabled control would be a promise nothing can keep.
/// Callers get a bool because the reason is a build-time fact about the
/// platform, not a diagnosis of this run — and the message inside the error
/// names an unimplemented feature, which is not something to put in front of
/// someone reading a page.
pub fn minting_supported() -> bool {
    owner_only_supported().is_ok()
}

/// Writes a fresh secret into its own new file in `dir` and returns that
/// file's path alongside the secret. The staging file is owner-only from
/// creation and belongs to this call alone, so the caller can put it in
/// place — by link or by rename — without the target path ever holding a
/// half-written secret or wearing looser permissions than `0600`.
fn stage_new_secret(dir: &Path) -> Result<(PathBuf, [u8; SECRET_LEN]), PresenceError> {
    let secret = random_bytes::<SECRET_LEN>()?;
    let path = dir.join(format!(
        "{STAGING_PREFIX}{}.{}",
        std::process::id(),
        hex(random_bytes::<8>()?)
    ));
    create_owner_only(&path, &secret).map_err(|e| unavailable("stage", &path, &e))?;
    Ok((path, secret))
}

/// Creates the secret file if and only if nothing is there yet, and returns
/// the new secret. `Ok(None)` means another caller got there first — its
/// secret is now the one every capability must verify against, so this one is
/// discarded rather than forced into place.
///
/// The link is what makes that decision the filesystem's rather than ours:
/// `hard_link` refuses an existing target, and it publishes the staged file
/// whole, so the secret path never appears half-written to anyone reading it.
fn create_secret_no_overwrite(dir: &Path) -> Result<Option<[u8; SECRET_LEN]>, PresenceError> {
    std::fs::create_dir_all(dir).map_err(|e| unavailable("create", dir, &e))?;
    let path = secret_path_in(dir);
    let (staged, secret) = stage_new_secret(dir)?;
    let outcome = match std::fs::hard_link(&staged, &path) {
        Ok(()) => Ok(Some(secret)),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
        Err(e) => Err(unavailable("link", &path, &e)),
    };
    let _ = std::fs::remove_file(&staged);
    outcome
}

/// How many times to re-read after losing a first-use race. Each pass can
/// only lose to a different caller creating the file, so a handful is far
/// more than the situation can produce.
const FIRST_USE_RETRIES: u32 = 4;

/// Loads the per-install presence secret, generating and persisting one
/// (owner-only, `0600`, never exported or synced — a plain file under the
/// app's own data directory) on first use.
fn load_or_create_secret_in(dir: &Path) -> Result<[u8; SECRET_LEN], PresenceError> {
    owner_only_supported()?;
    // Held for the whole sequence below, released when this handle drops.
    let _lock = lock_secret_dir(dir)?;
    sweep_staging_files(dir);
    load_or_create_secret_locked(dir)
}

/// The read-repair-replace-create sequence itself, with the caller
/// responsible for holding the lock. Split out only so the lock's scope is
/// impossible to misread.
fn load_or_create_secret_locked(dir: &Path) -> Result<[u8; SECRET_LEN], PresenceError> {
    let path = secret_path_in(dir);
    for _ in 0..FIRST_USE_RETRIES {
        match std::fs::read(&path) {
            Ok(bytes) if bytes.len() == SECRET_LEN => {
                repair_permissions(&path)?;
                let mut secret = [0_u8; SECRET_LEN];
                secret.copy_from_slice(&bytes);
                return Ok(secret);
            }
            Ok(_) => {
                // Wrong length means corrupt, not a partial write: the create
                // path publishes by link, so the secret path only ever
                // appears whole. Replacing it invalidates any outstanding
                // capability — acceptable per D7 §5 (rotation needs no drain;
                // capabilities live 60s) — but the replacement is staged and
                // renamed rather than truncated in place, so the new secret is
                // never for an instant sitting in a file that carries the old
                // file's looser permissions.
                let (staged, secret) = stage_new_secret(dir)?;
                std::fs::rename(&staged, &path).map_err(|e| {
                    let _ = std::fs::remove_file(&staged);
                    unavailable("replace", &path, &e)
                })?;
                return Ok(secret);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if let Some(secret) = create_secret_no_overwrite(dir)? {
                    return Ok(secret);
                }
                // Lost the race; read the winner's secret on the next pass.
            }
            // Permission denied, an I/O error, anything that is not "no such
            // file": refuse. Regenerating here would destroy a secret that is
            // still in use and that outstanding capabilities still verify
            // against, on nothing more than a transient failure to read it.
            Err(e) => return Err(unavailable("read", &path, &e)),
        }
    }
    Err(PresenceError::SecretUnavailable(format!(
        "{}: lost the first-use race {FIRST_USE_RETRIES} times",
        path.display()
    )))
}

/// Appends one component as its big-endian `u64` byte length followed by its
/// bytes. Nothing a component contains can be read as a boundary, because the
/// boundary is a count read before the bytes rather than a value looked for
/// inside them.
///
/// The width has to be wide enough that no length can wrap it. A `u32` prefix
/// would have signed a component of 2^32 + n bytes the same as one of n bytes,
/// putting the boundary confusion straight back. `usize as u64` cannot lose a
/// value on any target this app builds for, so the widening is exact.
fn push_component(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    out.extend_from_slice(bytes);
}

/// Canonical bytes a capability's HMAC is computed over: every component
/// length-prefixed, and the target list additionally prefixed with its
/// element count, so exactly one request maps to any given message.
///
/// A separator encoding could not promise that. Joining with a control
/// character signed `["a", "b"]` and `["a\u{1}b"]` identically, and let a
/// target ID borrow bytes from the base digest across their separator — so
/// one capability's MAC was also a valid MAC for a different request. Length
/// prefixes need no separators and no escaping.
///
/// Private helper called from exactly two places (`mint_in`, test-only
/// `verify_in`); a struct purely to satisfy the arg-count lint would be
/// unrequested abstraction for a one-shot serializer, hence the scoped allow.
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

/// Mints a capability for `request`, using the secret under `dir` and `now`
/// as the mint time. Split out from [`mint`] so tests can control both the
/// secret's home directory and the clock without touching process state.
fn mint_in(
    dir: &Path,
    now: u64,
    request: PresenceRequest,
) -> Result<PresenceCapability, PresenceError> {
    let secret = load_or_create_secret_in(dir)?;
    let nonce = random_bytes::<NONCE_LEN>()?;
    let expires_at = now.saturating_add(CAPABILITY_TTL_SECS);
    let bytes = signed_bytes(
        PROTOCOL_VERSION,
        request.action,
        &request.target_ids,
        &request.base_digest,
        &request.caller_id,
        &request.operation_id,
        &nonce,
        now,
        expires_at,
    );
    let mut mac = HmacSha256::new_from_slice(&secret)
        .expect("HMAC accepts a key of any length, including SECRET_LEN");
    mac.update(&bytes);
    let mac: [u8; 32] = mac.finalize().into_bytes().into();
    Ok(PresenceCapability {
        action: request.action,
        target_ids: request.target_ids,
        base_digest: request.base_digest,
        caller_id: request.caller_id,
        operation_id: request.operation_id,
        minted_at: now,
        expires_at,
        protocol_version: PROTOCOL_VERSION,
        nonce,
        mac,
    })
}

/// The identity this app signs with. It is half of the
/// `(caller_id, operation_id)` key the daemon looks a retry up by (D7 §6), so
/// it has to be the same string across restarts and installs of the app: a
/// value that changed per run would make every retry look like a first
/// execution, and the replay path could never fire.
const CALLER_ID: &str = "wenlan-app";

/// One minted capability in the shape the daemon deserializes
/// (`wenlan_types::requests::PresenceCapability`), with the nonce and the MAC
/// as lowercase hex because JSON has no bytes.
///
/// Private to this module, and the only `Serialize` thing here. The capability
/// it is rendered from still refuses `Serialize`, so the compile-time T1
/// guarantee is unchanged; this type exists to be handed straight to the HTTP
/// client by the same call that minted it, and nothing else can build one.
#[derive(serde::Serialize)]
struct CapabilityWire {
    protocol_version: u32,
    action: &'static str,
    target_ids: Vec<String>,
    base_digest: String,
    caller_id: String,
    operation_id: String,
    minted_at: u64,
    expires_at: u64,
    nonce: String,
    mac: String,
}

/// The body of `POST /api/pages/{id}/review`, mirroring
/// `wenlan_types::requests::ReviewPageRequest`.
///
/// Hand-mirrored rather than imported: the daemon's copy is `Deserialize`-only
/// on purpose, and this side is the one that has to serialize.
///
/// Its field is private, so the only way to obtain one is
/// [`mint_page_review`]. A body therefore cannot exist without a freshly
/// minted, correctly-bound capability behind it.
#[derive(serde::Serialize)]
pub struct ReviewPageBody {
    presence: CapabilityWire,
}

/// Mints a page-review capability against the real per-install secret and the
/// real clock, and renders it as the request body for that page's review
/// route.
///
/// Minting and rendering are one call because separating them would mean
/// handing a [`PresenceCapability`] to somebody, and the moment a capability
/// has a caller it has a call site that could return it. `target_ids` is
/// exactly `[page_id]`, which is what the daemon demands of a `review_page`
/// capability — a capability naming more targets is not a capability for
/// fewer.
pub fn mint_page_review(
    page_id: &str,
    base_digest: &str,
    operation_id: &str,
) -> Result<ReviewPageBody, PresenceError> {
    mint_page_review_in(
        &app_data_dir(),
        now_unix_secs(),
        page_id,
        base_digest,
        operation_id,
    )
}

/// A body minted against a throwaway secret directory. For tests in other
/// modules that need a well-formed body and do not care what it signs.
#[cfg(test)]
pub(crate) fn test_review_body(dir: &Path, page_id: &str) -> ReviewPageBody {
    mint_page_review_in(dir, 1_000, page_id, "digest-a", "op-1").expect("mint succeeds")
}

/// [`mint_page_review`] with the secret's directory and the clock injected, so
/// tests can exercise the wire shape without touching process state.
fn mint_page_review_in(
    dir: &Path,
    now: u64,
    page_id: &str,
    base_digest: &str,
    operation_id: &str,
) -> Result<ReviewPageBody, PresenceError> {
    let capability = mint_in(
        dir,
        now,
        PresenceRequest {
            action: PresenceAction::ReviewPage,
            target_ids: vec![page_id.to_string()],
            base_digest: base_digest.to_string(),
            caller_id: CALLER_ID.to_string(),
            operation_id: operation_id.to_string(),
        },
    )?;
    Ok(ReviewPageBody {
        presence: CapabilityWire {
            protocol_version: capability.protocol_version,
            action: capability.action.as_str(),
            target_ids: capability.target_ids,
            base_digest: capability.base_digest,
            caller_id: capability.caller_id,
            operation_id: capability.operation_id,
            minted_at: capability.minted_at,
            expires_at: capability.expires_at,
            nonce: hex(capability.nonce),
            mac: hex(capability.mac),
        },
    })
}

/// Recomputes a capability's HMAC against `secret` and reports whether it
/// matches and is unexpired at `now`. Exercises the same binding the daemon
/// would eventually need to check; kept `pub(crate)` since nothing outside
/// this module's tests calls it yet.
#[cfg(all(test, unix))]
fn verify_in(cap: &PresenceCapability, secret: &[u8], now: u64) -> bool {
    if cap.is_expired_at(now) {
        return false;
    }
    let bytes = signed_bytes(
        cap.protocol_version,
        cap.action,
        &cap.target_ids,
        &cap.base_digest,
        &cap.caller_id,
        &cap.operation_id,
        &cap.nonce,
        cap.minted_at,
        cap.expires_at,
    );
    let mut mac = match HmacSha256::new_from_slice(secret) {
        Ok(mac) => mac,
        Err(_) => return false,
    };
    mac.update(&bytes);
    mac.verify_slice(&cap.mac).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> PresenceRequest {
        PresenceRequest {
            action: PresenceAction::ReviewPage,
            target_ids: vec!["page-1".to_string()],
            base_digest: "digest-a".to_string(),
            caller_id: "app".to_string(),
            operation_id: "op-1".to_string(),
        }
    }

    /// The daemon reads the nonce and the MAC with `hex::decode`. This is the
    /// same operation, written out because the app does not depend on `hex`.
    #[cfg(unix)]
    fn unhex(s: &str) -> Vec<u8> {
        s.as_bytes()
            .chunks(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).expect("hex is ASCII");
                u8::from_str_radix(pair, 16).expect("hex digit")
            })
            .collect()
    }

    #[test]
    #[cfg(unix)]
    fn the_review_body_is_the_shape_the_daemon_deserializes() {
        let tmp = tempfile::tempdir().unwrap();
        let body = mint_page_review_in(tmp.path(), 1_000, "page-7", "digest-z", "op-9").unwrap();

        let wire = serde_json::to_value(&body).unwrap();
        let presence = &wire["presence"];
        assert_eq!(presence["protocol_version"], 1);
        // The exact string the daemon matches on; a capability for the other
        // action is invalid for this one (T4).
        assert_eq!(presence["action"], "review_page");
        // Exactly the one page, because that is the demand the daemon builds.
        assert_eq!(presence["target_ids"], serde_json::json!(["page-7"]));
        assert_eq!(presence["base_digest"], "digest-z");
        assert_eq!(presence["caller_id"], CALLER_ID);
        assert_eq!(presence["operation_id"], "op-9");
        assert_eq!(presence["minted_at"], 1_000);
        assert_eq!(presence["expires_at"], 1_060);
        // Lowercase hex of 16 nonce bytes and a 32-byte MAC.
        assert_eq!(presence["nonce"].as_str().unwrap().len(), NONCE_LEN * 2);
        assert_eq!(presence["mac"].as_str().unwrap().len(), 64);
    }

    #[test]
    #[cfg(unix)]
    fn a_review_body_verifies_against_the_secret_the_daemon_reads() {
        // The two processes never compare notes: if a signed field stopped
        // reaching the wire under the name the daemon reads, the only symptom
        // in production would be every review returning `presence_invalid`,
        // with nothing saying which side moved. So this rebuilds the signed
        // message out of the JSON the daemon will actually receive, and checks
        // that the MAC still verifies.
        //
        // It does not pin the component ORDER — it calls `signed_bytes` and so
        // moves with it. `the_signed_message_layout_is_pinned_byte_for_byte`
        // is the tooth that holds the layout against the daemon's copy.
        let tmp = tempfile::tempdir().unwrap();
        let body = mint_page_review_in(tmp.path(), 1_000, "page-7", "digest-z", "op-9").unwrap();
        let wire = serde_json::to_value(&body).unwrap();
        let presence = &wire["presence"];
        let text = |field: &str| {
            presence[field]
                .as_str()
                .unwrap_or_else(|| panic!("the daemon reads `{field}`; it is not on the wire"))
                .to_string()
        };
        let number = |field: &str| {
            presence[field]
                .as_u64()
                .unwrap_or_else(|| panic!("the daemon reads `{field}`; it is not on the wire"))
        };

        let secret = load_or_create_secret_in(tmp.path()).unwrap();
        let nonce = unhex(&text("nonce"));
        let expected = signed_bytes(
            number("protocol_version") as u32,
            PresenceAction::ReviewPage,
            &["page-7".to_string()],
            &text("base_digest"),
            &text("caller_id"),
            &text("operation_id"),
            &nonce,
            number("minted_at"),
            number("expires_at"),
        );
        let mut mac = HmacSha256::new_from_slice(&secret).unwrap();
        mac.update(&expected);
        assert!(
            mac.verify_slice(&unhex(&text("mac"))).is_ok(),
            "the daemon must be able to recompute this body's MAC",
        );
    }

    #[test]
    #[cfg(unix)]
    fn every_review_body_carries_its_own_nonce() {
        // The nonce is what makes a capability one-shot (T3). Two gestures that
        // happen to name the same page and content must still be two
        // capabilities, or the second is a replay of the first.
        let tmp = tempfile::tempdir().unwrap();
        let first = mint_page_review_in(tmp.path(), 1_000, "page-7", "digest-z", "op-9").unwrap();
        let second = mint_page_review_in(tmp.path(), 1_000, "page-7", "digest-z", "op-9").unwrap();

        assert_ne!(
            serde_json::to_value(&first).unwrap()["presence"]["nonce"],
            serde_json::to_value(&second).unwrap()["presence"]["nonce"],
        );
    }

    #[test]
    #[cfg(unix)]
    fn mint_creates_an_owner_only_secret_on_first_use() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = mint_in(tmp.path(), 1_000, request()).expect("mint succeeds");
        assert_eq!(cap.expires_at, 1_060);

        let secret_path = secret_path_in(tmp.path());
        assert!(secret_path.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&secret_path)
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    #[cfg(unix)]
    fn mint_reuses_the_same_secret_across_calls() {
        let tmp = tempfile::tempdir().unwrap();
        let first = mint_in(tmp.path(), 1_000, request()).unwrap();
        let secret = load_or_create_secret_in(tmp.path()).unwrap();
        assert!(verify_in(&first, &secret, 1_000));

        let second = mint_in(tmp.path(), 1_001, request()).unwrap();
        assert!(verify_in(&second, &secret, 1_001));
    }

    #[test]
    #[cfg(unix)]
    fn a_capability_is_valid_only_for_its_own_action_and_target_t4() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = mint_in(tmp.path(), 1_000, request()).unwrap();
        let secret = load_or_create_secret_in(tmp.path()).unwrap();
        assert!(verify_in(&cap, &secret, 1_000));

        let mut retargeted = cap.clone();
        retargeted.action = PresenceAction::AttestClaim;
        assert!(!verify_in(&retargeted, &secret, 1_000));

        let mut different_page = cap.clone();
        different_page.target_ids = vec!["page-2".to_string()];
        assert!(!verify_in(&different_page, &secret, 1_000));
    }

    #[test]
    #[cfg(unix)]
    fn a_capability_is_valid_only_for_its_own_base_digest_t6() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = mint_in(tmp.path(), 1_000, request()).unwrap();
        let secret = load_or_create_secret_in(tmp.path()).unwrap();

        let mut stale = cap.clone();
        stale.base_digest = "digest-b".to_string();
        assert!(!verify_in(&stale, &secret, 1_000));
    }

    #[test]
    #[cfg(unix)]
    fn a_capability_expires_sixty_seconds_after_minting_t5() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = mint_in(tmp.path(), 1_000, request()).unwrap();
        let secret = load_or_create_secret_in(tmp.path()).unwrap();

        assert!(verify_in(&cap, &secret, 1_059));
        assert!(!verify_in(&cap, &secret, 1_060));
    }

    /// One MAC over a fully fixed message, so two messages can be compared
    /// directly — `mint_in` draws a fresh nonce every call, which would make
    /// any two of its capabilities differ for the wrong reason.
    fn mac_over(target_ids: &[&str], base_digest: &str) -> [u8; 32] {
        let targets: Vec<String> = target_ids.iter().map(|id| (*id).to_string()).collect();
        let bytes = signed_bytes(
            PROTOCOL_VERSION,
            PresenceAction::ReviewPage,
            &targets,
            base_digest,
            "app",
            "op-1",
            &[0_u8; NONCE_LEN],
            1_000,
            1_060,
        );
        let mut mac = HmacSha256::new_from_slice(&[3_u8; SECRET_LEN]).unwrap();
        mac.update(&bytes);
        mac.finalize().into_bytes().into()
    }

    #[test]
    fn two_different_requests_never_share_a_mac_through_field_boundaries() {
        // A separator-joined message lets one field's content imitate the
        // separator and swallow the boundary. Both pairs below are distinct
        // requests that a separator encoding signs identically.
        assert_ne!(
            mac_over(&["a", "b"], "digest"),
            mac_over(&["a\u{1}b"], "digest"),
            "two target ids must not sign the same as one id containing the separator",
        );
        assert_ne!(
            mac_over(&["a\u{2}b"], "c"),
            mac_over(&["a"], "b\u{2}c"),
            "a target id must not borrow bytes from the base digest",
        );
    }

    #[test]
    fn the_signed_message_layout_is_pinned_byte_for_byte() {
        // Every length prefix is a big-endian u64, written out literally here
        // rather than recomputed, so narrowing a prefix cannot pass unnoticed.
        // A prefix narrow enough to wrap makes the serializer non-injective
        // again — the exact defect the length prefixes were added to close —
        // and no practical test can feed it a component large enough to prove
        // that directly, so the layout itself is what gets pinned.
        let bytes = signed_bytes(
            1,
            PresenceAction::ReviewPage,
            &["a".to_string()],
            "d",
            "c",
            "o",
            &[0xab],
            1,
            2,
        );

        let expected = concat!(
            "0000000000000004",
            "00000001", // protocol version 1, as a u32
            "000000000000000b",
            "7265766965775f70616765", // action "review_page"
            "0000000000000001",       // one target id follows
            "0000000000000001",
            "61", // target id "a"
            "0000000000000001",
            "64", // base digest "d"
            "0000000000000001",
            "63", // caller "c"
            "0000000000000001",
            "6f", // operation "o"
            "0000000000000001",
            "ab", // nonce
            "0000000000000008",
            "0000000000000001", // minted at 1
            "0000000000000008",
            "0000000000000002", // expires at 2
        );
        assert_eq!(hex(&bytes), expected);
    }

    #[test]
    fn the_same_request_always_signs_the_same_bytes() {
        assert_eq!(
            mac_over(&["a", "b"], "digest"),
            mac_over(&["a", "b"], "digest")
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_capability_from_a_rotated_secret_does_not_verify() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = mint_in(tmp.path(), 1_000, request()).unwrap();

        let other_secret = [7_u8; SECRET_LEN];
        assert!(!verify_in(&cap, &other_secret, 1_000));
    }

    #[test]
    #[cfg(unix)]
    fn debug_output_never_leaks_the_mac_or_the_raw_nonce_t9() {
        let tmp = tempfile::tempdir().unwrap();
        let cap = mint_in(tmp.path(), 1_000, request()).unwrap();
        let secret = load_or_create_secret_in(tmp.path()).unwrap();

        let debug_output = format!("{cap:?}");
        assert!(!debug_output.contains(&hex(cap.nonce)));
        assert!(!debug_output.contains(&hex(secret)));
        assert!(!debug_output.contains(&hex(cap.mac)));
        assert!(debug_output.contains("<redacted>"));
        assert!(debug_output.contains(&cap.nonce_digest()));
    }

    #[cfg(unix)]
    fn mode_of(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    /// True when the process can read `path` despite its mode bits saying
    /// otherwise — i.e. it is running as root, where there is no such thing
    /// as an unreadable file to test against.
    #[cfg(unix)]
    fn mode_bits_are_ignored(path: &Path) -> bool {
        std::fs::read(path).is_ok()
    }

    #[test]
    #[cfg(unix)]
    fn a_loosely_permissioned_secret_file_is_tightened_before_it_is_used() {
        let tmp = tempfile::tempdir().unwrap();
        let path = secret_path_in(tmp.path());
        std::fs::write(&path, [9_u8; SECRET_LEN]).unwrap();
        set_mode(&path, 0o644);

        let secret = load_or_create_secret_in(tmp.path()).expect("an existing secret is usable");

        assert_eq!(secret, [9_u8; SECRET_LEN], "the existing secret is kept");
        assert_eq!(mode_of(&path), 0o600, "and its permissions are repaired");
    }

    #[test]
    #[cfg(unix)]
    fn a_secret_that_cannot_be_read_refuses_instead_of_regenerating() {
        // Write-only is the case that separates the two behaviours. At mode
        // 0o000 a regenerating implementation fails anyway, because it cannot
        // open the file for writing either; at 0o200 it succeeds, silently
        // destroying a secret that outstanding capabilities still verify
        // against. A read error that is not "no such file" must refuse.
        for mode in [0o000, 0o200] {
            let tmp = tempfile::tempdir().unwrap();
            let path = secret_path_in(tmp.path());
            std::fs::write(&path, [9_u8; SECRET_LEN]).unwrap();
            set_mode(&path, mode);
            if mode_bits_are_ignored(&path) {
                return;
            }

            let result = mint_in(tmp.path(), 1_000, request());
            assert!(
                matches!(result, Err(PresenceError::SecretUnavailable(_))),
                "mode {mode:o} must make minting unavailable, not regenerate",
            );

            set_mode(&path, 0o600);
            assert_eq!(
                std::fs::read(&path).unwrap(),
                [9_u8; SECRET_LEN],
                "mode {mode:o} must leave the unreadable secret untouched",
            );
        }
    }

    /// Written out rather than taken from the implementation's constants on
    /// purpose. These are the on-disk naming contract: a staging file a
    /// crashed earlier build left behind carries whatever name that build
    /// used, and renaming a constant must not quietly stop those from being
    /// swept.
    #[cfg(unix)]
    const ORPHAN_STAGING_NAME: &str = "presence_secret.staging.999.00deadbeef00";
    /// The scheme the first version of this code staged under: no `.staging.`
    /// segment at all. Files with this shape are already on real installs.
    #[cfg(unix)]
    const LEGACY_ORPHAN_STAGING_NAME: &str = "presence_secret.999.00deadbeef00";

    /// Everything in `dir` the sweep is supposed to have collected: any
    /// `presence_secret.` sibling that is not the lock file.
    #[cfg(unix)]
    fn staging_files_in(dir: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("presence_secret.") && name != "presence_secret.lock"
                    })
            })
            .collect()
    }

    #[test]
    #[cfg(unix)]
    fn a_staging_file_left_by_an_earlier_run_is_swept_away() {
        // A run that dies between creating a staging file and putting it in
        // place leaves a file holding a whole, live secret. Nothing else ever
        // collects it, so they accumulate one per crash — each owner-only, but
        // each a secret sitting in the data directory for good.
        let tmp = tempfile::tempdir().unwrap();
        let orphan = tmp.path().join(ORPHAN_STAGING_NAME);
        std::fs::write(&orphan, [7_u8; SECRET_LEN]).unwrap();

        load_or_create_secret_in(tmp.path()).expect("a stale staging file must not block a load");

        assert!(
            !orphan.exists(),
            "the orphaned staging file must be swept away"
        );
        assert!(staging_files_in(tmp.path()).is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn a_staging_file_from_the_first_naming_scheme_is_swept_away_too() {
        // The first version of this code staged under `presence_secret.<pid>.
        // <random>`, with no `.staging.` segment. Orphans in that shape are
        // already sitting on installs that ran it, so a sweep that only knew
        // the current name would leave them there for good.
        let tmp = tempfile::tempdir().unwrap();
        let legacy = tmp.path().join(LEGACY_ORPHAN_STAGING_NAME);
        std::fs::write(&legacy, [7_u8; SECRET_LEN]).unwrap();

        let secret =
            load_or_create_secret_in(tmp.path()).expect("a legacy orphan must not block a load");

        assert!(!legacy.exists(), "the legacy orphan must be swept away");
        assert!(staging_files_in(tmp.path()).is_empty());

        // The two files the sweep must never touch.
        let secret_path = secret_path_in(tmp.path());
        assert!(secret_path.exists(), "the secret itself must survive");
        assert_eq!(std::fs::read(&secret_path).unwrap(), secret);
        assert!(
            tmp.path().join("presence_secret.lock").exists(),
            "the lock file must survive",
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_load_waits_for_whoever_holds_the_lock() {
        use std::sync::mpsc;
        use std::time::Duration;

        // Every decision about the secret — read it, repair it, replace a
        // corrupt one, create the first — happens inside this lock, so two
        // callers can never both act on the same observation.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().to_path_buf();
        let held = lock_secret_dir(&dir).expect("the lock is free to start with");

        let (finished, waiting) = mpsc::channel();
        let worker_dir = dir.clone();
        let worker = std::thread::spawn(move || {
            let secret = load_or_create_secret_in(&worker_dir).expect("the load succeeds");
            finished.send(()).unwrap();
            secret
        });

        assert!(
            waiting.recv_timeout(Duration::from_millis(250)).is_err(),
            "a load must not proceed while another caller holds the lock",
        );

        drop(held);

        // If the lock were never released this join would hang rather than
        // fail, which the test runner surfaces as a timeout.
        let secret = worker
            .join()
            .expect("the waiting load finishes once the lock frees");
        assert_eq!(secret.len(), SECRET_LEN);
        assert!(staging_files_in(&dir).is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn a_corrupt_secret_file_is_replaced_whole_rather_than_written_through() {
        use std::os::unix::fs::MetadataExt;

        let tmp = tempfile::tempdir().unwrap();
        let path = secret_path_in(tmp.path());
        std::fs::write(&path, b"too-short").unwrap();
        set_mode(&path, 0o644);
        let corrupt_inode = std::fs::metadata(&path).unwrap().ino();

        let secret = load_or_create_secret_in(tmp.path()).expect("a corrupt file is replaced");

        assert_ne!(
            std::fs::metadata(&path).unwrap().ino(),
            corrupt_inode,
            "the new secret must land in a fresh owner-only file, never in the loose one",
        );
        assert_eq!(std::fs::read(&path).unwrap(), secret);
        assert_eq!(mode_of(&path), 0o600);
        assert!(
            staging_files_in(tmp.path()).is_empty(),
            "no staging file is left behind",
        );
    }

    #[test]
    #[cfg(unix)]
    fn a_corrupt_secret_file_is_replaced_rather_than_trusted() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path()).unwrap();
        std::fs::write(secret_path_in(tmp.path()), b"too-short").unwrap();

        let secret = load_or_create_secret_in(tmp.path()).expect("regenerates instead of erroring");
        assert_eq!(secret.len(), SECRET_LEN);
    }

    #[test]
    #[cfg(unix)]
    fn a_first_use_race_never_replaces_the_secret_that_won_it() {
        // The loser of a first-use race arrives at the create path with a
        // secret already in place. It must hand back the winner's secret, not
        // its own: a caller holding a capability signed with a secret that no
        // longer exists holds a capability nothing can verify.
        let tmp = tempfile::tempdir().unwrap();
        let path = secret_path_in(tmp.path());
        std::fs::write(&path, [9_u8; SECRET_LEN]).unwrap();

        let outcome =
            create_secret_no_overwrite(tmp.path()).expect("losing a race is not an error");

        assert!(outcome.is_none(), "the existing secret must win");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            [9_u8; SECRET_LEN],
            "and must survive untouched",
        );
        assert_eq!(
            load_or_create_secret_in(tmp.path()).unwrap(),
            [9_u8; SECRET_LEN],
            "so every later mint signs with it",
        );
        assert!(
            staging_files_in(tmp.path()).is_empty(),
            "the discarded secret leaves no staging file behind",
        );
    }

    #[test]
    #[cfg(not(unix))]
    fn minting_refuses_where_owner_only_protection_is_not_implemented() {
        // Everything above this test is gated to Unix because minting itself
        // is. On any other platform the secret file would inherit whatever the
        // parent directory's access control grants, so the module cannot keep
        // the promise its threat model makes and says so instead.
        let tmp = tempfile::tempdir().unwrap();

        let result = mint_in(tmp.path(), 1_000, request());

        match result {
            Err(PresenceError::SecretUnavailable(message)) => assert!(
                message.contains("not implemented on this platform"),
                "the refusal must name its reason, got: {message}",
            ),
            _ => panic!("minting must refuse where the secret cannot be kept owner-only"),
        }
        assert!(
            !secret_path_in(tmp.path()).exists(),
            "and must not leave a secret file behind",
        );
    }

    #[test]
    #[cfg(unix)]
    fn secret_unavailable_surfaces_as_an_error_not_a_silent_bypass() {
        // A path segment that is a file, not a directory, makes
        // `create_dir_all` for anything under it fail — the closest portable
        // stand-in for "the secret's home directory cannot be created or
        // written", which must refuse rather than mint anyway (D7 §5).
        let tmp = tempfile::tempdir().unwrap();
        let blocked_file = tmp.path().join("not-a-directory");
        std::fs::write(&blocked_file, b"x").unwrap();
        let unusable_dir = blocked_file.join("presence");

        let result = mint_in(&unusable_dir, 1_000, request());
        assert!(matches!(result, Err(PresenceError::SecretUnavailable(_))));
    }
}
