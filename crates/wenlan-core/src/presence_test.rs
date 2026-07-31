// SPDX-License-Identifier: Apache-2.0
//! Teeth for the daemon half of the presence protocol.
//!
//! The rows of the threat model's §8 table that this file owns are the ones
//! about the capability itself: the layout it is signed over, the bindings it
//! carries, and its expiry. Replay, conflict, and transactional nonce
//! consumption are database questions and live beside the mutation.

use super::*;

use hmac::{Hmac, Mac};
use sha2::Sha256;

const SECRET: &[u8; 32] = b"01234567890123456789012345678901";

/// Mints the way the app mints, so a capability built here is one the app could
/// have produced. If this helper and the app's `mint_in` ever disagree, the
/// layout pin below is what says so.
#[allow(clippy::too_many_arguments)]
fn mint(
    action: PresenceAction,
    target_ids: &[&str],
    base_digest: &str,
    caller_id: &str,
    operation_id: &str,
    nonce: &[u8],
    minted_at: u64,
    secret: &[u8],
) -> wenlan_types::requests::PresenceCapability {
    mint_with_window(
        action,
        target_ids,
        base_digest,
        caller_id,
        operation_id,
        nonce,
        minted_at,
        minted_at + 60,
        secret,
    )
}

/// Mints with a window the caller chooses, correctly signed either way.
///
/// A capability whose window is absurd is not a forgery — the app's own secret
/// signed it — so the only thing that can refuse it is a daemon that checks the
/// window's shape rather than just its end.
#[allow(clippy::too_many_arguments)]
fn mint_with_window(
    action: PresenceAction,
    target_ids: &[&str],
    base_digest: &str,
    caller_id: &str,
    operation_id: &str,
    nonce: &[u8],
    minted_at: u64,
    expires_at: u64,
    secret: &[u8],
) -> wenlan_types::requests::PresenceCapability {
    let targets: Vec<String> = target_ids.iter().map(|t| (*t).to_string()).collect();
    let bytes = signed_bytes(
        PROTOCOL_VERSION,
        action,
        &targets,
        base_digest,
        caller_id,
        operation_id,
        nonce,
        minted_at,
        expires_at,
    );
    let mut mac = <Hmac<Sha256>>::new_from_slice(secret).unwrap();
    mac.update(&bytes);
    wenlan_types::requests::PresenceCapability {
        protocol_version: PROTOCOL_VERSION,
        action: action.as_str().to_string(),
        target_ids: targets,
        base_digest: base_digest.to_string(),
        caller_id: caller_id.to_string(),
        operation_id: operation_id.to_string(),
        minted_at,
        expires_at,
        nonce: hex::encode(nonce),
        mac: hex::encode(mac.finalize().into_bytes()),
    }
}

fn valid() -> wenlan_types::requests::PresenceCapability {
    mint(
        PresenceAction::ReviewPage,
        &["page-1"],
        "digest-a",
        "app",
        "op-1",
        &[0xab; 16],
        1_000,
        SECRET,
    )
}

fn demand<'a>(targets: &'a [String], base_digest: &'a str) -> PresenceDemand<'a> {
    PresenceDemand {
        action: PresenceAction::ReviewPage,
        target_ids: targets,
        base_digest,
    }
}

fn targets() -> Vec<String> {
    vec!["page-1".to_string()]
}

/// The one test that keeps two processes in agreement.
///
/// This hex is copied from the app's own `the_signed_message_layout_is_pinned_
/// byte_for_byte`, not recomputed here. The daemon and the app never exchange
/// this layout at runtime — they exchange a MAC computed over it — so a
/// disagreement about it surfaces only as "every capability is invalid," with
/// nothing in either process pointing at which side moved. Two independent
/// pins against the same literal is what turns that into a failing test.
///
/// Every length prefix is a big-endian `u64`, written out rather than
/// recomputed, so narrowing one cannot pass unnoticed. A prefix narrow enough
/// to wrap makes the encoder non-injective again — one capability's MAC
/// becomes a valid MAC for a different request — and no practical test can
/// feed it a component large enough to show that directly.
#[test]
fn the_signed_message_layout_matches_the_app_byte_for_byte() {
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
    assert_eq!(hex::encode(&bytes), expected);
}

/// The action strings are signed content, so they are pinned as literals. A
/// rename that looked like a tidy-up would invalidate every outstanding
/// capability for that action and read as a forgery attempt.
#[test]
fn the_action_strings_are_the_ones_the_app_signs() {
    assert_eq!(PresenceAction::AttestClaim.as_str(), "attest_claim");
    assert_eq!(PresenceAction::ReviewPage.as_str(), "review_page");
}

#[test]
fn a_capability_the_app_minted_verifies() {
    let targets = targets();
    let verified = verify(&valid(), SECRET, 1_010, &demand(&targets, "digest-a"))
        .expect("a well-formed capability inside its window verifies");
    assert_eq!(verified.caller_id, "app");
    assert_eq!(verified.operation_id, "op-1");
    assert_eq!(verified.action, PresenceAction::ReviewPage);
}

/// T4: a capability for one action is not a capability for the other. The
/// action is a signed component, so this cannot be defeated by editing the
/// submitted `action` field — that only breaks the MAC.
#[test]
fn a_capability_for_the_other_action_is_refused() {
    let attest = mint(
        PresenceAction::AttestClaim,
        &["page-1"],
        "digest-a",
        "app",
        "op-1",
        &[0xab; 16],
        1_000,
        SECRET,
    );
    let targets = targets();
    assert_eq!(
        verify(&attest, SECRET, 1_010, &demand(&targets, "digest-a")),
        Err(PresenceRefusal::Invalid)
    );
}

/// T4, the other half: a capability minted for one page cannot review another.
#[test]
fn a_capability_for_a_different_page_is_refused() {
    let elsewhere = vec!["page-2".to_string()];
    assert_eq!(
        verify(&valid(), SECRET, 1_010, &demand(&elsewhere, "digest-a")),
        Err(PresenceRefusal::Invalid)
    );
}

/// T6: the page changed between the gesture and the submit, so the human
/// approved text that is no longer there.
#[test]
fn a_capability_bound_to_content_that_has_since_changed_is_refused() {
    let targets = targets();
    assert_eq!(
        verify(&valid(), SECRET, 1_010, &demand(&targets, "digest-b")),
        Err(PresenceRefusal::Invalid)
    );
}

/// T5: 60 seconds, and the window is half-open — at exactly `expires_at` the
/// capability is already gone.
#[test]
fn a_capability_is_dead_the_instant_its_window_closes() {
    let targets = targets();
    assert!(verify(&valid(), SECRET, 1_059, &demand(&targets, "digest-a")).is_ok());
    assert_eq!(
        verify(&valid(), SECRET, 1_060, &demand(&targets, "digest-a")),
        Err(PresenceRefusal::Expired)
    );
}

/// T2, at the only point the daemon can enforce it. Minting needs the secret,
/// and a client that does not have it produces a capability that looks entirely
/// well-formed and does not verify.
#[test]
fn a_capability_minted_against_a_different_secret_is_refused() {
    let forged = mint(
        PresenceAction::ReviewPage,
        &["page-1"],
        "digest-a",
        "app",
        "op-1",
        &[0xab; 16],
        1_000,
        b"a different thirty-two byte secret",
    );
    let targets = targets();
    assert_eq!(
        verify(&forged, SECRET, 1_010, &demand(&targets, "digest-a")),
        Err(PresenceRefusal::Invalid)
    );
}

/// Flipping one byte of the MAC has to fail, and fail the same way an unknown
/// action fails. Nothing in the refusal tells the caller which check it lost.
#[test]
fn a_tampered_mac_is_refused_as_bluntly_as_anything_else() {
    let mut tampered = valid();
    tampered.mac.replace_range(0..1, "f");
    let mut unknown_action = valid();
    unknown_action.action = "delete_everything".to_string();
    let targets = targets();

    assert_eq!(
        verify(&tampered, SECRET, 1_010, &demand(&targets, "digest-a")),
        Err(PresenceRefusal::Invalid)
    );
    assert_eq!(
        verify(
            &unknown_action,
            SECRET,
            1_010,
            &demand(&targets, "digest-a")
        ),
        Err(PresenceRefusal::Invalid)
    );
}

/// A future protocol version is refused rather than validated under this
/// version's rules — the version exists so the format can change without
/// ambiguity, which only works if the old side declines to guess.
#[test]
fn a_capability_from_a_later_protocol_version_is_refused() {
    let mut newer = valid();
    newer.protocol_version = PROTOCOL_VERSION + 1;
    let targets = targets();
    assert_eq!(
        verify(&newer, SECRET, 1_010, &demand(&targets, "digest-a")),
        Err(PresenceRefusal::Invalid)
    );
}

/// §5: absent means unavailable, and unavailable is not permission.
#[test]
fn a_missing_secret_is_unavailable_rather_than_a_bypass() {
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(load_secret(tmp.path()), Err(PresenceRefusal::Unavailable));
}

/// A truncated or padded file is not a secret this protocol produced. Using it
/// anyway would compute MACs nothing can match, which reads as "every gesture
/// is a forgery" — a worse failure than refusing outright.
#[test]
fn a_secret_of_the_wrong_length_is_unavailable() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(secret_path(tmp.path()), b"too short").unwrap();
    assert_eq!(load_secret(tmp.path()), Err(PresenceRefusal::Unavailable));
}

/// The daemon reads the app's file and does not write one back. If this ever
/// creates a secret, two processes are racing to author it and the loser's
/// capabilities silently stop verifying.
#[test]
fn loading_a_secret_never_creates_one() {
    let tmp = tempfile::tempdir().unwrap();
    let _ = load_secret(tmp.path());
    assert!(
        !secret_path(tmp.path()).exists(),
        "the daemon must never author the install secret"
    );
}

#[test]
fn a_secret_the_app_wrote_loads() {
    let tmp = tempfile::tempdir().unwrap();
    write_secret(tmp.path(), 0o600);
    assert_eq!(load_secret(tmp.path()), Ok(SECRET.to_vec()));
}

/// Writes the secret with an explicit mode, because the default one depends on
/// the umask of whoever is running the tests.
fn write_secret(root: &std::path::Path, mode: u32) {
    let path = secret_path(root);
    std::fs::write(&path, SECRET).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
    }
    #[cfg(not(unix))]
    let _ = mode;
}

/// T2: the secret is the only thing standing between a local process and a
/// forged human gesture. Reading one that any process on the machine can also
/// read, and then treating capabilities signed with it as proof a person was
/// present, is a claim the file's own permissions contradict.
///
/// The daemon refuses rather than repairs. The app owns the file's lifecycle,
/// and a daemon that quietly tightened the mode would be hiding a state
/// somebody needs to see — the secret may already have been copied.
#[cfg(unix)]
#[test]
fn a_secret_other_processes_can_read_is_refused() {
    for mode in [0o644, 0o640, 0o604, 0o666] {
        let tmp = tempfile::tempdir().unwrap();
        write_secret(tmp.path(), mode);
        assert_eq!(
            load_secret(tmp.path()),
            Err(PresenceRefusal::Unavailable),
            "mode {mode:o} lets a process outside this user read the signing secret"
        );
    }
}

/// The other half of the same rule: owner-only modes load. A daemon that
/// refused these would take every correct install offline.
#[cfg(unix)]
#[test]
fn an_owner_only_secret_still_loads() {
    for mode in [0o600, 0o400] {
        let tmp = tempfile::tempdir().unwrap();
        write_secret(tmp.path(), mode);
        assert_eq!(
            load_secret(tmp.path()),
            Ok(SECRET.to_vec()),
            "mode {mode:o} is owner-only and must keep working"
        );
    }
}

/// T5 again, from the side expiry alone cannot see.
///
/// `now >= expires_at` refuses a capability whose window has closed. It says
/// nothing about a window that should never have been opened: a capability
/// minted with a ten-year expiry is inside its window for ten years, and the
/// app's own secret signed it, so every other check passes. The bound has to be
/// on the window's shape, not just on its end.
#[test]
fn a_capability_whose_window_is_far_longer_than_the_protocol_allows_is_refused() {
    let ten_years = 60 * 60 * 24 * 365 * 10;
    let submitted = mint_with_window(
        PresenceAction::ReviewPage,
        &["page-1"],
        "digest-a",
        "app",
        "op-1",
        &[0xab; 16],
        1_000,
        1_000 + ten_years,
        SECRET,
    );
    let targets = targets();
    assert_eq!(
        verify(&submitted, SECRET, 1_000, &demand(&targets, "digest-a")),
        Err(PresenceRefusal::Invalid),
        "a ten-year presence capability is a standing forgery licence, not a gesture"
    );
}

/// The window the app actually mints has to survive the bound above, including
/// at its exact edge. A check tight enough to refuse the real shape would take
/// every gesture offline.
#[test]
fn the_apps_own_sixty_second_window_verifies_at_both_ends() {
    let targets = targets();
    let submitted = valid();
    assert_eq!(submitted.expires_at - submitted.minted_at, 60);
    for now in [1_000, 1_059] {
        assert!(
            verify(&submitted, SECRET, now, &demand(&targets, "digest-a")).is_ok(),
            "the app's real 60-second window must verify at now={now}"
        );
    }
}

/// A window that ends before it begins is not a window. Nothing in the app can
/// mint one, so a capability carrying it is either a corrupted request or a
/// deliberately constructed one; both deserve the same answer.
#[test]
fn a_capability_that_expires_no_later_than_it_was_minted_is_refused() {
    let targets = targets();
    for expires_at in [1_000, 999] {
        let submitted = mint_with_window(
            PresenceAction::ReviewPage,
            &["page-1"],
            "digest-a",
            "app",
            "op-1",
            &[0xab; 16],
            1_000,
            expires_at,
            SECRET,
        );
        assert_eq!(
            verify(&submitted, SECRET, 900, &demand(&targets, "digest-a")),
            Err(PresenceRefusal::Invalid),
            "expires_at {expires_at} is not after minted_at 1000"
        );
    }
}

/// A capability minted in the future would still be inside its window long
/// after a real one had died — the clock the daemon trusts is its own. The
/// allowance is small on purpose: it absorbs the two machines being a few
/// seconds apart, which here is really one machine's clock moving under two
/// processes, and nothing more.
#[test]
fn a_capability_minted_further_ahead_than_clock_skew_explains_is_refused() {
    let targets = targets();
    let submitted = valid(); // minted_at 1_000
    assert_eq!(
        verify(&submitted, SECRET, 990, &demand(&targets, "digest-a")),
        Err(PresenceRefusal::Invalid),
        "minted ten seconds in the daemon's future is a clock nobody should trust"
    );
    assert!(
        verify(&submitted, SECRET, 996, &demand(&targets, "digest-a")).is_ok(),
        "four seconds of skew is the ordinary case and must still verify"
    );
}

/// T7 vs T8 turns on this digest, so what it does and does not cover is a
/// contract rather than an implementation detail. Same intent, different nonce
/// is the same request — that is what makes an honest re-mint-and-retry
/// idempotent instead of a conflict.
#[test]
fn the_request_digest_covers_intent_and_not_the_capability() {
    let targets = targets();
    let elsewhere = vec!["page-2".to_string()];
    let base = request_digest(PresenceAction::ReviewPage, &targets, "digest-a");

    assert_eq!(
        base,
        request_digest(PresenceAction::ReviewPage, &targets, "digest-a")
    );
    assert_ne!(
        base,
        request_digest(PresenceAction::AttestClaim, &targets, "digest-a"),
        "a different action is a different request"
    );
    assert_ne!(
        base,
        request_digest(PresenceAction::ReviewPage, &elsewhere, "digest-a"),
        "a different page is a different request"
    );
    assert_ne!(
        base,
        request_digest(PresenceAction::ReviewPage, &targets, "digest-b"),
        "different content is a different request"
    );
}

/// The framing has to be injective, or one capability's MAC is also a valid MAC
/// for a request nobody made. Two target lists whose bytes concatenate
/// identically must still sign differently.
#[test]
fn two_different_target_lists_never_sign_the_same_bytes() {
    let split = vec!["a".to_string(), "b".to_string()];
    let joined = vec!["ab".to_string()];
    assert_ne!(
        request_digest(PresenceAction::ReviewPage, &split, "d"),
        request_digest(PresenceAction::ReviewPage, &joined, "d")
    );
}

/// §7: `Debug` is the easiest accidental leak — one `tracing::error!("{req:?}")`
/// and the HMAC is in the log file forever.
#[test]
fn formatting_a_capability_prints_no_capability_material() {
    let cap = valid();
    let printed = format!("{cap:?}");
    assert!(
        !printed.contains(&cap.mac),
        "Debug must not print the HMAC: {printed}"
    );
    assert!(
        !printed.contains(&cap.nonce),
        "Debug must not print the raw nonce: {printed}"
    );
    assert!(printed.contains("op-1"), "identity is fine to print");
}

/// What crosses the wire, spelled out. The app has no serializer for its
/// capability yet, so this is the shape it must produce.
#[test]
fn the_wire_shape_is_what_the_app_has_to_send() {
    let json = serde_json::json!({
        "protocol_version": 1,
        "action": "review_page",
        "target_ids": ["page-1"],
        "base_digest": "digest-a",
        "caller_id": "app",
        "operation_id": "op-1",
        "minted_at": 1000,
        "expires_at": 1060,
        "nonce": "abababababababababababababababab",
        "mac": "00",
    });
    let parsed: wenlan_types::requests::PresenceCapability = serde_json::from_value(json).unwrap();
    assert_eq!(parsed.action, "review_page");
    assert_eq!(parsed.target_ids, vec!["page-1".to_string()]);
}
