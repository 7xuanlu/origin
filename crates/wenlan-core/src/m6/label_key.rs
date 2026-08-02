//! D8 wikilink key normalization (artifact 4 §6).
//!
//! Turns the raw `target` capture of `WIKILINK_RE`
//! (`crates/wenlan-core/src/sources/obsidian.rs:16`-`:17`) into a `label_key`,
//! or rejects it. A rejected link is **dropped** — the page still writes, the
//! raw link text still survives in the body, and no `page_links` row appears
//! (S0-39). Rejection is never an ingest failure.
//!
//! The step order is not free-form. Two orderings are load-bearing:
//!
//! * NFKC runs **before** lowercasing, because `U+FF21` (fullwidth `A`)
//!   lowercases to fullwidth `ａ`; folding first gives `a`.
//! * The structural rejection runs **after** NFKC, because NFKC folds the
//!   fullwidth forms of `#`, `|`, `[`, `]` into their ASCII counterparts. The
//!   parser's own guards run on pre-normalization bytes, so `[[Rust＃Ownership]]`
//!   reaches normalization intact and only then becomes `rust#ownership` — a key
//!   containing a fragment separator the parser cannot itself produce (S0-36).

use unicode_normalization::UnicodeNormalization;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

use super::constants::{LABEL_KEY_MAX_SCALARS, LABEL_KEY_MIN_SCALARS, LABEL_RAW_PRECAP_SCALARS};

/// Why a raw wikilink target did not become a `label_key`.
///
/// The variants carry no payload on purpose: D13 forbids user content in logs,
/// and a wikilink label is user content by definition. Callers log the code and
/// the page ID, never the label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelRejection {
    /// R0 — raw target exceeded the pre-cap, before any normalization work.
    TooLongRaw,
    /// R1 — `#`, `|`, `[`, or `]` present after normalization.
    Structural,
    /// R2 — a scalar in general category `Cc` or `Cf`.
    ControlOrFormat,
    /// R3 — normalized length outside `1..=128` scalars.
    Length,
}

impl LabelRejection {
    /// Stable short code for counters and debug logs (`R0`..`R3`).
    pub fn code(self) -> &'static str {
        match self {
            LabelRejection::TooLongRaw => "R0",
            LabelRejection::Structural => "R1",
            LabelRejection::ControlOrFormat => "R2",
            LabelRejection::Length => "R3",
        }
    }
}

/// Characters rejected at step 4. All four are parser-structural.
const STRUCTURAL_CHARS: [char; 4] = ['#', '|', '[', ']'];

/// Normalize a raw wikilink target into its `label_key`.
pub fn normalize_label_key(raw: &str) -> Result<String, LabelRejection> {
    // 0. PRE-CAP — bound the work before doing any of it. Without this, a
    //    megabyte-long target pays for full normalization only to fail on
    //    length. 1024 is 8x the post-normalization cap (S0-37).
    if raw.chars().count() > LABEL_RAW_PRECAP_SCALARS {
        return Err(LabelRejection::TooLongRaw);
    }

    // 1-3. NFKC, lowercase, NFKC again. The second pass is not redundant:
    //      `U+0130`.to_lowercase() yields `U+0069 U+0307`, which is not in NFKC
    //      form, so without it the pipeline's output is not a fixed point of the
    //      pipeline (S0-35).
    let normalized: String = raw
        .nfkc()
        .collect::<String>()
        .to_lowercase()
        .nfkc()
        .collect();

    // 4. STRUCTURAL — reject, never strip. By this point the parser has already
    //    split off any legitimate fragment or alias, so a separator surviving
    //    here means the input was built to smuggle one. Stripping would make
    //    `[[A#B]]` and `[[A]]` the same key (S0-36).
    if normalized.contains(STRUCTURAL_CHARS) {
        return Err(LabelRejection::Structural);
    }

    // 5. WS COLLAPSE — every maximal White_Space run becomes one U+0020, both
    //    ends trimmed. `char::is_whitespace` is exactly the White_Space
    //    property. NBSP and IDEOGRAPHIC SPACE arrive here already folded to
    //    U+0020 by NFKC.
    let collapsed = collapse_whitespace(&normalized);

    // 6. CONTROL/BIDI — no exceptions (S0-38). This runs after the collapse, so
    //    a scalar that is both Cc and White_Space (tab, newline) has already
    //    become a space. What survives to here is the invisible-but-not-space
    //    class: RLO, ZWSP, BOM, ZWNJ.
    if collapsed.chars().any(is_control_or_format) {
        return Err(LabelRejection::ControlOrFormat);
    }

    // 7. LENGTH — measured last, on the normalized result. A single scalar can
    //    expand 1->18 under NFKC (`U+FDFA`), so measuring the raw input here
    //    would let 128 accepted scalars become 2304 stored ones (S0-37).
    let scalars = collapsed.chars().count();
    if !(LABEL_KEY_MIN_SCALARS..=LABEL_KEY_MAX_SCALARS).contains(&scalars) {
        return Err(LabelRejection::Length);
    }

    Ok(collapsed)
}

/// Collapse every maximal `White_Space` run to a single `U+0020` and trim.
fn collapse_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut pending_space = false;

    for ch in input.chars() {
        if ch.is_whitespace() {
            // Only emit once the run ends and something follows, which also
            // drops leading and trailing runs without a separate trim pass.
            pending_space = !out.is_empty();
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(ch);
        }
    }

    out
}

/// True for general category `Cc` (control) or `Cf` (format).
///
/// Rust's standard library offers `char::is_control()`, which is `Cc` only —
/// it would miss every `Cf` scalar, and `Cf` is where the whole attack lives
/// (RLO, ZWSP, BOM are all `Cf`).
fn is_control_or_format(ch: char) -> bool {
    matches!(
        ch.general_category(),
        GeneralCategory::Control | GeneralCategory::Format
    )
}
