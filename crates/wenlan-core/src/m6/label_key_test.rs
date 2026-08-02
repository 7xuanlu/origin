//! Teeth for D8 wikilink key normalization (artifact 4 §6).
//!
//! W1-W9 and A1-A9 are the artifact's own worked examples, named here by their
//! artifact IDs so a reviewer can diff the table against the tests one row at a
//! time.
//!
//! Note on W4/W5: `WIKILINK_RE` splits the fragment (group 3) and the alias
//! (group 4) off before normalization ever runs, so the input reaching
//! `normalize_label_key` is the target group alone. The tests assert that
//! contract at the function boundary — the parser's split is exercised by the
//! obsidian source tests, not re-tested here.

use super::label_key::{normalize_label_key, LabelRejection};

fn key(raw: &str) -> String {
    normalize_label_key(raw).expect("expected this label to normalize")
}

fn rejection(raw: &str) -> LabelRejection {
    normalize_label_key(raw).expect_err("expected this label to be rejected")
}

// ---------------------------------------------------------------------------
// W1-W9 — accepted
// ---------------------------------------------------------------------------

#[test]
fn w1_plain_label_lowercases() {
    assert_eq!(key("Rust Ownership"), "rust ownership");
}

#[test]
fn w2_internal_whitespace_run_collapses() {
    assert_eq!(key("Rust  Ownership"), "rust ownership");
}

#[test]
fn w3_leading_and_trailing_whitespace_is_trimmed() {
    assert_eq!(key(" Rust Ownership "), "rust ownership");
}

#[test]
fn w4_fragment_is_split_by_the_parser_before_normalization() {
    // The target group the regex hands over carries no fragment.
    assert_eq!(key("Rust Ownership"), "rust ownership");
    // And a `#` that somehow survived into the target is rejected, not folded.
    assert_eq!(
        rejection("Rust Ownership#borrowing"),
        LabelRejection::Structural
    );
}

#[test]
fn w5_alias_is_split_by_the_parser_before_normalization() {
    assert_eq!(key("Rust Ownership"), "rust ownership");
    assert_eq!(
        rejection("Rust Ownership|borrowing rules"),
        LabelRejection::Structural
    );
}

#[test]
fn w6_fullwidth_letters_fold_before_lowercasing() {
    // NFKC first gives `A B`, then lowercase gives `a b`. Lowercasing first
    // would yield fullwidth `a b` and a different key.
    assert_eq!(key("\u{FF21} \u{FF22}"), "a b");
}

#[test]
fn w7_single_scalar_expansion_normalizes() {
    // U+3392 SQUARE MHZ expands 1 -> 3 under NFKC.
    assert_eq!(key("\u{3392}"), "mhz");
}

#[test]
fn w8_sharp_s_is_not_case_folded_so_the_keys_stay_distinct() {
    // `to_lowercase` is not case folding: neither form becomes `ss`. This is an
    // accepted ceiling, asserted so a future switch to real case folding is a
    // deliberate change and not a silent one.
    assert_eq!(key("STRASSE"), "strasse");
    assert_eq!(key("Stra\u{00DF}e"), "stra\u{00DF}e");
    assert_ne!(key("STRASSE"), key("Stra\u{00DF}e"));
    // LATIN CAPITAL LETTER SHARP S lowercases to the same sharp s.
    assert_eq!(key("STRA\u{1E9E}E"), "stra\u{00DF}e");
}

#[test]
fn w9_nfc_and_nfd_spellings_produce_one_key() {
    let nfc = "Caf\u{00E9}"; // é precomposed
    let nfd = "Cafe\u{0301}"; // e + combining acute

    // Control: the two inputs really are different byte sequences.
    assert_ne!(nfc, nfd, "control failed: inputs were expected to differ");
    assert_eq!(key(nfc), key(nfd));
    assert_eq!(key(nfc), "caf\u{00E9}");
}

// ---------------------------------------------------------------------------
// A1-A9 — rejected
// ---------------------------------------------------------------------------

#[test]
fn a1_bidi_override_is_rejected() {
    assert_eq!(
        rejection("Rust\u{202E}Ownership"),
        LabelRejection::ControlOrFormat
    );
}

#[test]
fn a2_zero_width_space_is_rejected() {
    // An invisible split of one visible label into two keys.
    assert_eq!(
        rejection("Rust\u{200B}Ownership"),
        LabelRejection::ControlOrFormat
    );
}

#[test]
fn a3_fullwidth_number_sign_is_rejected_after_nfkc() {
    // Passes WIKILINK_RE (which excludes only ASCII `#`) and the bracket guard,
    // then NFKC turns it into a real `#`.
    let raw = "Rust\u{FF03}Ownership";

    // Control: the raw input carries no ASCII `#`, so the upstream guards that
    // run on pre-normalization bytes genuinely do not catch it.
    assert!(
        !raw.contains('#'),
        "control failed: the smuggled form already contained an ASCII #"
    );
    assert_eq!(rejection(raw), LabelRejection::Structural);
}

#[test]
fn a4_fullwidth_vertical_line_is_rejected_after_nfkc() {
    let raw = "Rust\u{FF5C}Ownership";
    assert!(
        !raw.contains('|'),
        "control failed: raw already contained |"
    );
    assert_eq!(rejection(raw), LabelRejection::Structural);
}

#[test]
fn a5_fullwidth_left_bracket_is_rejected_after_nfkc() {
    let raw = "Rust\u{FF3B}Ownership";
    assert!(
        !raw.contains('['),
        "control failed: raw already contained ["
    );
    assert_eq!(rejection(raw), LabelRejection::Structural);
}

#[test]
fn a5b_fullwidth_right_bracket_is_rejected_after_nfkc() {
    let raw = "Rust\u{FF3D}Ownership";
    assert!(
        !raw.contains(']'),
        "control failed: raw already contained ]"
    );
    assert_eq!(rejection(raw), LabelRejection::Structural);
}

#[test]
fn a6_expansion_over_the_post_normalization_cap_is_rejected() {
    // 44 x U+FDFA: 44 raw scalars (under the 1024 pre-cap) expanding to 792
    // normalized scalars (over the 128 cap). This is the case that proves the
    // length check is measured *after* normalization.
    let raw: String = "\u{FDFA}".repeat(44);

    assert!(
        raw.chars().count() <= super::constants::LABEL_RAW_PRECAP_SCALARS,
        "control failed: this input should pass the R0 pre-cap and fail on R3"
    );
    assert_eq!(rejection(&raw), LabelRejection::Length);
}

#[test]
fn a7_oversized_raw_target_is_rejected_by_the_precap() {
    let raw = "a".repeat(2000);
    assert_eq!(rejection(&raw), LabelRejection::TooLongRaw);
}

#[test]
fn a8_leading_bom_is_rejected_because_it_is_not_whitespace() {
    // The BOM is Cf, not White_Space, so the step-5 collapse does not remove it
    // and it reaches the step-6 rejection.
    assert!(
        !'\u{FEFF}'.is_whitespace(),
        "control failed: BOM was expected not to be White_Space"
    );
    assert_eq!(rejection("\u{FEFF}Rust"), LabelRejection::ControlOrFormat);
}

#[test]
fn a9_whitespace_only_target_is_rejected_on_length() {
    assert_eq!(rejection("   "), LabelRejection::Length);
    assert_eq!(rejection(""), LabelRejection::Length);
}

// ---------------------------------------------------------------------------
// Ordering and property teeth
// ---------------------------------------------------------------------------

#[test]
fn s0_38_zero_width_non_joiner_is_rejected_with_no_exceptions() {
    // Documented cost of the no-exceptions rule: ZWNJ is orthographically
    // required in Persian and some Indic spellings, so those labels cannot be
    // wikilink keys. Asserted so lifting the rule is a deliberate change.
    assert_eq!(
        rejection("Rust\u{200C}Ownership"),
        LabelRejection::ControlOrFormat
    );
    assert_eq!(
        rejection("Rust\u{200D}Ownership"),
        LabelRejection::ControlOrFormat
    );
}

#[test]
fn nbsp_and_ideographic_space_fold_to_a_plain_space() {
    // Both are Zs, both fold to U+0020 under NFKC, so both are accepted rather
    // than hitting the Cc/Cf rejection.
    assert_eq!(key("Rust\u{00A0}Ownership"), "rust ownership");
    assert_eq!(key("Rust\u{3000}Ownership"), "rust ownership");
}

#[test]
fn line_separator_collapses_rather_than_rejecting() {
    // U+2028 is Zl and White_Space, so step 5 consumes it before step 6 runs.
    assert!('\u{2028}'.is_whitespace());
    assert_eq!(key("Rust\u{2028}Ownership"), "rust ownership");
}

#[test]
fn kelvin_sign_lowercases_to_k() {
    assert_eq!(key("\u{212A}"), "k");
}

#[test]
fn structural_rejection_beats_silent_folding() {
    // `[[A#B]]` must not become the same key as `[[A]]`. The failure this
    // guards against is a *strip* implementation rather than a reject.
    let smuggled = normalize_label_key("A\u{FF03}B");
    assert!(smuggled.is_err());
    assert_eq!(key("A"), "a");
}

#[test]
fn normalization_output_is_a_fixed_point_of_normalization() {
    // The pipeline's own output must re-normalize to itself. This is the
    // property the second NFKC pass exists to preserve (S0-35): without a
    // re-normalization step somewhere, the pipeline can emit a string that is
    // not in NFKC form, and two inputs that should fold end up as different
    // keys depending on which case they arrived in.
    let corpus = [
        "Rust Ownership",
        " Rust  Ownership ",
        "\u{FF21} \u{FF22}",
        "\u{3392}",
        "Caf\u{00E9}",
        "Cafe\u{0301}",
        "\u{0130}stanbul",
        "STRA\u{1E9E}E",
        "\u{212A}elvin",
    ];

    for raw in corpus {
        let once = key(raw);
        let twice = key(&once);
        assert_eq!(once, twice, "normalization is not idempotent for {raw:?}");
    }
}

#[test]
fn rejection_codes_are_stable() {
    assert_eq!(LabelRejection::TooLongRaw.code(), "R0");
    assert_eq!(LabelRejection::Structural.code(), "R1");
    assert_eq!(LabelRejection::ControlOrFormat.code(), "R2");
    assert_eq!(LabelRejection::Length.code(), "R3");
}

#[test]
fn a_label_exactly_at_the_cap_is_accepted_and_one_over_is_not() {
    let at_cap = "a".repeat(super::constants::LABEL_KEY_MAX_SCALARS);
    let over_cap = "a".repeat(super::constants::LABEL_KEY_MAX_SCALARS + 1);

    assert_eq!(key(&at_cap).chars().count(), 128);
    assert_eq!(rejection(&over_cap), LabelRejection::Length);
}
