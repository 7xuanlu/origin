//! Teeth for the M6 digest primitive (S0-26 through S0-30).
//!
//! Each test follows the S0-135 shape: build the *mutant* the decision forbids,
//! assert the mutant really does exhibit the defect (the discriminating
//! positive control — without it a test can pass because the input was too weak
//! to distinguish anything), then assert the contract implementation does not.

use super::constants::{DOMAIN_PAGE, DOMAIN_SLOT};
use super::digest::{bool_part, int_part, m6_digest, m6_digest_owned, sorted_set, ABSENT_PART};
use sha2::{Digest, Sha256};

/// Mutant for S0-26: NUL-separator framing instead of length prefixes.
fn mutant_separator_framed(domain: &str, parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    for part in parts {
        hasher.update(b"\0");
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
}

/// Mutant for S0-28: the `source_page_id` shape — a raw domain prefix followed
/// by raw content, no length framing anywhere
/// (`document_enrichment.rs:763`). That construction is safe there only because
/// its tag is a single compile-time constant; M6 has several tags and a version
/// axis, so the ambiguity is reachable.
fn mutant_raw_prefix_domain(domain: &str, parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain.as_bytes());
    for part in parts {
        hasher.update(part);
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn s0_26_length_prefixing_separates_parts_that_separator_framing_aliases() {
    // Two genuinely different part vectors whose NUL-separated serializations
    // are byte-identical.
    let a: [&[u8]; 2] = [b"x\0", b"y"];
    let b: [&[u8]; 2] = [b"x", b"\0y"];

    // Control: the mutation really does collide. If this ever fails, the test
    // below is proving nothing.
    assert_eq!(
        mutant_separator_framed(DOMAIN_SLOT, &a),
        mutant_separator_framed(DOMAIN_SLOT, &b),
        "control failed: separator framing was expected to alias these inputs"
    );

    // Contract: length prefixing keeps them apart.
    assert_ne!(
        m6_digest(DOMAIN_SLOT, &a),
        m6_digest(DOMAIN_SLOT, &b),
        "length-prefixed framing must not alias distinct part vectors"
    );
}

#[test]
fn s0_28_domain_tag_is_length_prefixed_not_a_raw_prefix() {
    // `("m6-page-v1", "Xabc")` vs `("m6-page-v1X", "abc")` — same byte stream
    // under a raw prefix.
    let domain_a = "m6-page-v1";
    let parts_a: [&[u8]; 1] = [b"Xabc"];
    let domain_b = "m6-page-v1X";
    let parts_b: [&[u8]; 1] = [b"abc"];

    assert_eq!(
        mutant_raw_prefix_domain(domain_a, &parts_a),
        mutant_raw_prefix_domain(domain_b, &parts_b),
        "control failed: a raw domain prefix was expected to alias these"
    );

    assert_ne!(
        m6_digest(domain_a, &parts_a),
        m6_digest(domain_b, &parts_b),
        "the domain tag must be absorbed as its own length-prefixed part"
    );
}

#[test]
fn s0_28_same_parts_under_different_domains_never_collide() {
    let parts: [&[u8]; 1] = [b"the-same-bytes"];
    assert_ne!(
        m6_digest(DOMAIN_SLOT, &parts),
        m6_digest(DOMAIN_PAGE, &parts),
        "a slot_id and a page_id built from identical parts must differ"
    );
}

#[test]
fn s0_27_digest_is_full_64_lowercase_hex() {
    let digest = m6_digest(DOMAIN_SLOT, &[b"anything"]);

    assert_eq!(digest.len(), 64, "digest must not be truncated");
    assert!(
        digest
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "digest must be lowercase hex, got {digest}"
    );
}

#[test]
fn s0_30_set_encoding_dedups_before_counting() {
    let with_duplicate = [
        "src:alpha".to_string(),
        "src:beta".to_string(),
        "src:alpha".to_string(),
    ];
    let without = ["src:alpha".to_string(), "src:beta".to_string()];

    assert_eq!(
        sorted_set(&with_duplicate),
        sorted_set(&without),
        "a duplicated group must not inflate the set"
    );
    // The count part leads, and it counts the deduped set.
    assert_eq!(sorted_set(&with_duplicate)[0], int_part(2));
}

#[test]
fn s0_30_set_encoding_is_order_independent() {
    let forward = ["turn:a".to_string(), "batch:b".to_string()];
    let reverse = ["batch:b".to_string(), "turn:a".to_string()];

    // Control: insertion order really does differ, so sorting is doing work.
    assert_ne!(
        forward.as_slice(),
        reverse.as_slice(),
        "control failed: the two inputs were expected to differ in order"
    );

    assert_eq!(
        m6_digest_owned(DOMAIN_SLOT, &sorted_set(&forward)),
        m6_digest_owned(DOMAIN_SLOT, &sorted_set(&reverse)),
        "set encoding must be order-independent"
    );
}

#[test]
fn s0_30_set_encoding_sorts_by_bytes_not_by_length() {
    // A length-first ordering would put "z" before "aa"; byte-lexicographic
    // puts "aa" first. Assert the emitted element order directly.
    let items = ["z".to_string(), "aa".to_string()];
    let parts = sorted_set(&items);

    assert_eq!(parts[0], int_part(2));
    assert_eq!(
        parts[1],
        b"aa".to_vec(),
        "byte-lexicographic order expected"
    );
    assert_eq!(parts[2], b"z".to_vec());
}

#[test]
fn s0_30_disjoint_sets_do_not_alias_a_concatenated_element() {
    // {a, b} must not collide with {ab}.
    let two = ["a".to_string(), "b".to_string()];
    let one = ["ab".to_string()];

    assert_ne!(
        m6_digest_owned(DOMAIN_SLOT, &sorted_set(&two)),
        m6_digest_owned(DOMAIN_SLOT, &sorted_set(&one)),
        "{{a,b}} and {{ab}} must not alias"
    );
}

#[test]
fn absent_part_is_distinct_from_the_literal_string_none() {
    // The absent encoding must be a zero-length part, not a sentinel word a
    // real value could equal.
    assert!(ABSENT_PART.is_empty());
    assert_ne!(
        m6_digest(DOMAIN_SLOT, &[ABSENT_PART]),
        m6_digest(DOMAIN_SLOT, &[b"none"]),
        "an absent optional must not collide with the literal value \"none\""
    );
}

#[test]
fn integer_parts_are_ascii_decimal_without_leading_zeros() {
    assert_eq!(int_part(0), b"0".to_vec());
    assert_eq!(int_part(1), b"1".to_vec());
    assert_eq!(int_part(-7), b"-7".to_vec());
    assert_eq!(int_part(1_234_567), b"1234567".to_vec());
    // No `+`, and no zero padding that would make 01 and 1 distinct inputs.
    assert_ne!(int_part(1), b"01".to_vec());
}

#[test]
fn boolean_parts_are_the_literal_t_and_f() {
    assert_eq!(bool_part(true), b"t");
    assert_eq!(bool_part(false), b"f");
    assert_ne!(
        m6_digest(DOMAIN_SLOT, &[bool_part(true)]),
        m6_digest(DOMAIN_SLOT, &[bool_part(false)])
    );
}

#[test]
fn digest_is_stable_across_calls() {
    // Guards against any accidental nondeterminism (hasher state, iteration
    // order) creeping into the primitive.
    let once = m6_digest(DOMAIN_SLOT, &[b"alpha", b"beta"]);
    let twice = m6_digest(DOMAIN_SLOT, &[b"alpha", b"beta"]);
    assert_eq!(once, twice);
}
