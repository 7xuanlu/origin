//! Teeth for the M6 identity derivations (S0-29, S0-32, S0-34, S0-40, S0-161).

use super::constants::{DOMAIN_FINGERPRINT, PAGE_ID_PREFIX};
use super::digest::{int_part, m6_digest_owned, sorted_set, ABSENT_PART};
use super::identity::{
    active_root_digest, candidate_id, page_id, slot_id_community_overview,
    slot_id_evidence_cluster, slot_id_orphan_wikilink, slot_id_space_overview,
    CandidateFingerprint, SignalKind,
};

const SPACE: &str = "0f9c5a2e-1b3d-4c6f-8a90-2d4e6f8a0b1c";
const OTHER_SPACE: &str = "7e2b1c4d-5a6f-4b8c-9d0e-1f2a3b4c5d6e";

fn groups(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

// ---------------------------------------------------------------------------
// slot_id
// ---------------------------------------------------------------------------

#[test]
fn signal_tags_are_all_distinct() {
    let tags = [
        SignalKind::EvidenceCluster.tag(),
        SignalKind::OrphanWikilink.tag(),
        SignalKind::CommunityOverview.tag(),
        SignalKind::SpaceOverview.tag(),
    ];
    let mut sorted = tags;
    sorted.sort_unstable();
    let mut deduped = sorted.to_vec();
    deduped.dedup();
    assert_eq!(deduped.len(), tags.len(), "signal tags must be distinct");
}

#[test]
fn the_four_signals_never_share_a_slot_for_one_space() {
    // The space-overview slot has an empty per-signal vector, which is why the
    // signal tag must be its own part rather than folded into the domain.
    let ids = [
        slot_id_evidence_cluster(SPACE, &[]),
        slot_id_orphan_wikilink(SPACE, ""),
        slot_id_community_overview(SPACE, ""),
        slot_id_space_overview(SPACE),
    ];

    let mut deduped = ids.to_vec();
    deduped.sort_unstable();
    deduped.dedup();
    assert_eq!(
        deduped.len(),
        ids.len(),
        "two signals collided on one slot_id for the same space"
    );
}

#[test]
fn s0_161_slot_identity_follows_the_stable_space_id_not_a_name() {
    // A rename changes `spaces.name`; `spaces.id` is what we digest, so
    // re-derivation after a rename reproduces the same slot. Modelled here as
    // "same id in, same slot out" — the schema-level rename closure is tested
    // against the database in the migration teeth.
    let before = slot_id_evidence_cluster(SPACE, &groups(&["src:a", "turn:b"]));
    let after = slot_id_evidence_cluster(SPACE, &groups(&["src:a", "turn:b"]));
    assert_eq!(before, after);

    // Two different spaces must never share a slot.
    assert_ne!(
        slot_id_evidence_cluster(SPACE, &groups(&["src:a"])),
        slot_id_evidence_cluster(OTHER_SPACE, &groups(&["src:a"])),
    );
}

#[test]
fn s0_29_evidence_cluster_slot_ignores_group_order_and_duplicates() {
    let canonical = slot_id_evidence_cluster(SPACE, &groups(&["src:a", "turn:b", "batch:c"]));
    let shuffled = slot_id_evidence_cluster(SPACE, &groups(&["turn:b", "batch:c", "src:a"]));
    let duplicated =
        slot_id_evidence_cluster(SPACE, &groups(&["src:a", "turn:b", "batch:c", "src:a"]));

    assert_eq!(canonical, shuffled, "slot must not depend on group order");
    assert_eq!(canonical, duplicated, "a duplicate group must not re-key");
}

#[test]
fn evidence_cluster_slot_changes_when_the_initial_group_set_changes() {
    assert_ne!(
        slot_id_evidence_cluster(SPACE, &groups(&["src:a"])),
        slot_id_evidence_cluster(SPACE, &groups(&["src:a", "src:b"])),
    );
}

#[test]
fn orphan_wikilink_slot_is_keyed_on_the_normalized_label() {
    assert_ne!(
        slot_id_orphan_wikilink(SPACE, "rust ownership"),
        slot_id_orphan_wikilink(SPACE, "rust borrowing"),
    );
    assert_eq!(
        slot_id_orphan_wikilink(SPACE, "rust ownership"),
        slot_id_orphan_wikilink(SPACE, "rust ownership"),
    );
}

// ---------------------------------------------------------------------------
// page_id — S0-32
// ---------------------------------------------------------------------------

#[test]
fn s0_32_page_id_is_the_prefix_plus_a_full_64_hex_digest() {
    let slot = slot_id_space_overview(SPACE);
    let page = page_id(&slot);

    assert!(page.starts_with(PAGE_ID_PREFIX));
    let digest = page.strip_prefix(PAGE_ID_PREFIX).expect("prefix checked");

    // The refusal to truncate is the substantive half of S0-32: the
    // orphan-wikilink slot derives from an attacker-written label, so a 64-bit
    // page ID would be grindable at ~2^32 work.
    assert_eq!(
        digest.len(),
        64,
        "page_id digest must be full length, never truncated to 16 hex"
    );
    assert!(digest
        .chars()
        .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
}

#[test]
fn page_id_is_a_function_of_the_slot_alone() {
    let slot_a = slot_id_orphan_wikilink(SPACE, "alpha");
    let slot_b = slot_id_orphan_wikilink(SPACE, "beta");

    assert_eq!(page_id(&slot_a), page_id(&slot_a));
    assert_ne!(page_id(&slot_a), page_id(&slot_b));
}

#[test]
fn page_id_is_not_the_slot_id_wearing_a_prefix() {
    // The indirection is deliberate: it lets an `m6-page-v2` re-derive page IDs
    // without disturbing slot identity.
    let slot = slot_id_space_overview(SPACE);
    assert_ne!(page_id(&slot), format!("{PAGE_ID_PREFIX}{slot}"));
}

// ---------------------------------------------------------------------------
// candidate_id — S0-40
// ---------------------------------------------------------------------------

#[test]
fn s0_40_candidate_id_is_keyed_on_slot_and_coverage_epoch() {
    let slot = slot_id_space_overview(SPACE);

    assert_eq!(candidate_id(&slot, 1), candidate_id(&slot, 1));
    assert_ne!(
        candidate_id(&slot, 1),
        candidate_id(&slot, 2),
        "a coverage-epoch change must produce a new candidate"
    );
}

#[test]
fn candidate_id_re_derives_identically_after_a_crash_or_expiry() {
    // Nothing a crash or a lease expiry changes is an input, which is what
    // lets a re-prepare after staleness land on the same durable row.
    let slot = slot_id_evidence_cluster(SPACE, &groups(&["src:a", "src:b"]));
    let first_pass = candidate_id(&slot, 3);
    let after_restart = candidate_id(
        &slot_id_evidence_cluster(SPACE, &groups(&["src:b", "src:a"])),
        3,
    );
    assert_eq!(first_pass, after_restart);
}

// ---------------------------------------------------------------------------
// active-root digest — fingerprint field 8
// ---------------------------------------------------------------------------

fn roots(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
        .collect()
}

#[test]
fn field_8_notices_a_status_change_that_field_6_cannot() {
    let ids = groups(&["root:1", "root:2"]);
    let all_active = roots(&[("root:1", "active"), ("root:2", "active")]);
    let one_failed = roots(&[("root:1", "active"), ("root:2", "failed")]);

    // Control: field 6 (the root-ID set) is identical across the two, so
    // without field 8 the fingerprint would not move.
    assert_eq!(
        sorted_set(&ids),
        sorted_set(&ids),
        "control: the root-id set is unchanged by a status flip"
    );

    assert_ne!(
        active_root_digest(&all_active),
        active_root_digest(&one_failed),
        "a root going active -> failed must change the active-root digest"
    );
}

#[test]
fn field_8_is_order_independent_and_dedups() {
    let forward = roots(&[("root:1", "active"), ("root:2", "failed")]);
    let reverse = roots(&[("root:2", "failed"), ("root:1", "active")]);
    let duplicated = roots(&[
        ("root:1", "active"),
        ("root:2", "failed"),
        ("root:1", "active"),
    ]);

    assert_eq!(active_root_digest(&forward), active_root_digest(&reverse));
    assert_eq!(
        active_root_digest(&forward),
        active_root_digest(&duplicated)
    );
}

#[test]
fn field_8_does_not_alias_across_the_pair_boundary() {
    // ("ab", "c") and ("a", "bc") must not collide — the failure a separator
    // join would reintroduce.
    assert_ne!(
        active_root_digest(&roots(&[("ab", "c")])),
        active_root_digest(&roots(&[("a", "bc")])),
    );
}

// ---------------------------------------------------------------------------
// candidate fingerprint — S0-34 fixed arity
// ---------------------------------------------------------------------------

fn base_fingerprint<'a>(slot: &'a str, roots_digest: &'a str) -> CandidateFingerprint<'a> {
    CandidateFingerprint {
        slot_id: slot,
        signal_version: 1,
        community_id: None,
        published_generation: None,
        coverage_epoch: 1,
        root_ids: &[],
        input_generation: 1,
        active_root_digest: roots_digest,
        model_version: "qwen3-4b-instruct-2507",
        projection_version: "proj-1",
        prompt_version: "m6-genesis-prompt-v1",
    }
}

/// Mutant for S0-34: absent optionals are *omitted* rather than emitted as
/// zero-length parts, so the part vector changes arity.
fn mutant_omitting_absent_fields(fp: &CandidateFingerprint<'_>) -> String {
    let mut parts: Vec<Vec<u8>> = Vec::new();
    parts.push(fp.slot_id.as_bytes().to_vec());
    parts.push(int_part(fp.signal_version));
    if let Some(id) = fp.community_id {
        parts.push(id.as_bytes().to_vec());
    }
    if let Some(gen) = fp.published_generation {
        parts.push(int_part(gen));
    }
    parts.push(int_part(fp.coverage_epoch));
    parts.extend(sorted_set(fp.root_ids));
    parts.push(int_part(fp.input_generation));
    parts.push(fp.active_root_digest.as_bytes().to_vec());
    parts.push(fp.model_version.as_bytes().to_vec());
    parts.push(fp.projection_version.as_bytes().to_vec());
    parts.push(fp.prompt_version.as_bytes().to_vec());
    m6_digest_owned(DOMAIN_FINGERPRINT, &parts)
}

#[test]
fn s0_34_absent_optionals_are_emitted_not_omitted() {
    let slot = slot_id_space_overview(SPACE);
    let roots_digest = active_root_digest(&[]);

    // Two genuinely different candidates. Under omission both collapse to the
    // same part vector, because the text "5" and the integer 5 encode to the
    // same bytes and the omitted field shifts the other into its position.
    let mut absent_id_present_gen = base_fingerprint(&slot, &roots_digest);
    absent_id_present_gen.community_id = None;
    absent_id_present_gen.published_generation = Some(5);

    let mut present_id_absent_gen = base_fingerprint(&slot, &roots_digest);
    present_id_absent_gen.community_id = Some("5");
    present_id_absent_gen.published_generation = None;

    // Control: the forbidden encoding really does alias them.
    assert_eq!(
        mutant_omitting_absent_fields(&absent_id_present_gen),
        mutant_omitting_absent_fields(&present_id_absent_gen),
        "control failed: omitting absent parts was expected to alias these"
    );

    // Contract: fixed arity keeps them apart.
    assert_ne!(
        absent_id_present_gen.digest(),
        present_id_absent_gen.digest(),
        "fixed-arity encoding must keep these fingerprints distinct"
    );
}

#[test]
fn absent_optional_is_a_zero_length_part() {
    assert!(ABSENT_PART.is_empty());

    let slot = slot_id_space_overview(SPACE);
    let roots_digest = active_root_digest(&[]);

    let absent = base_fingerprint(&slot, &roots_digest);
    let mut empty_string = base_fingerprint(&slot, &roots_digest);
    empty_string.community_id = Some("");

    // A present empty string and an absent field encode to the same bytes at
    // the same position. That is acceptable precisely because arity is fixed —
    // asserted here so the equivalence is a recorded property, not a surprise.
    assert_eq!(absent.digest(), empty_string.digest());
}

#[test]
fn every_fingerprint_field_moves_the_digest() {
    let slot = slot_id_space_overview(SPACE);
    let other_slot = slot_id_orphan_wikilink(SPACE, "other");
    let roots_digest = active_root_digest(&[]);
    let other_roots = active_root_digest(&roots(&[("root:9", "active")]));
    let root_ids = groups(&["root:1"]);

    let base = base_fingerprint(&slot, &roots_digest);
    let baseline = base.digest();

    let mut variants: Vec<(&str, CandidateFingerprint<'_>)> = Vec::new();

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.slot_id = &other_slot;
    variants.push(("slot_id", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.signal_version = 2;
    variants.push(("signal_version", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.community_id = Some("c-1");
    variants.push(("community_id", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.published_generation = Some(7);
    variants.push(("published_generation", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.coverage_epoch = 2;
    variants.push(("coverage_epoch", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.root_ids = &root_ids;
    variants.push(("root_ids", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.input_generation = 2;
    variants.push(("input_generation", v));

    let mut v = base_fingerprint(&slot, &other_roots);
    variants.push(("active_root_digest", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.model_version = "qwen3.5-9b";
    variants.push(("model_version", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.projection_version = "proj-2";
    variants.push(("projection_version", v));

    let mut v = base_fingerprint(&slot, &roots_digest);
    v.prompt_version = "m6-genesis-prompt-v2";
    variants.push(("prompt_version", v));

    assert_eq!(variants.len(), 11, "all eleven fields must be covered");

    for (field, variant) in variants {
        assert_ne!(
            baseline,
            variant.digest(),
            "changing field {field} did not change the fingerprint"
        );
    }
}

#[test]
fn fingerprint_is_stable_for_identical_inputs() {
    let slot = slot_id_space_overview(SPACE);
    let roots_digest = active_root_digest(&[]);
    assert_eq!(
        base_fingerprint(&slot, &roots_digest).digest(),
        base_fingerprint(&slot, &roots_digest).digest(),
    );
}

#[test]
fn fingerprint_never_equals_the_slot_it_describes() {
    let slot = slot_id_space_overview(SPACE);
    let roots_digest = active_root_digest(&[]);
    assert_ne!(base_fingerprint(&slot, &roots_digest).digest(), slot);
}
