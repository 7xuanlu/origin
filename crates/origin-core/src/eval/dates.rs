// SPDX-License-Identifier: AGPL-3.0-only
//! Date helpers shared across eval benchmark adapters.
//!
//! Centralises the three date utilities that were previously scattered across
//! `locomo`, `longmemeval`, and `shared`:
//!
//! - [`parse_locomo_date`] — LoCoMo session timestamps ("1:56 pm on 8 May, 2023")
//! - [`parse_lme_date`]    — LongMemEval session timestamps ("2023/04/10 (Mon) 23:07")
//! - [`format_ymd`]        — Unix-seconds → "YYYY-MM-DD" formatting
//! - [`seed_last_modified`] — resolve a benchmark chunk's `last_modified` field

/// Parse a LoCoMo session date like "1:56 pm on 8 May, 2023" into Unix seconds.
/// Returns `None` on parse failure (caller falls back to `now()`).
pub(crate) fn parse_locomo_date(s: &str) -> Option<i64> {
    use chrono::{NaiveDateTime, TimeZone, Utc};
    // The dataset uses "<h:mm am/pm> on <D Month, YYYY>". chrono's strftime
    // %p needs uppercase AM/PM; LoCoMo uses lowercase. Normalise first.
    let normalised = s.replace(" am ", " AM ").replace(" pm ", " PM ");
    NaiveDateTime::parse_from_str(&normalised, "%I:%M %p on %d %B, %Y")
        .ok()
        .and_then(|naive| Utc.from_local_datetime(&naive).single())
        .map(|dt| dt.timestamp())
}

/// Parse a LongMemEval `question_date` / `haystack_date` into Unix seconds.
/// Format example: "2023/04/10 (Mon) 23:07". Returns `None` on parse failure
/// (e.g. dataset variants with different formats -- caller falls back to `now()`).
pub(crate) fn parse_lme_date(s: &str) -> Option<i64> {
    use chrono::{NaiveDateTime, TimeZone, Utc};
    // Strip the weekday tag in parens: "2023/04/10 (Mon) 23:07" -> "2023/04/10 23:07"
    let cleaned: String = s
        .split_whitespace()
        .filter(|tok| !(tok.starts_with('(') && tok.ends_with(')')))
        .collect::<Vec<_>>()
        .join(" ");
    NaiveDateTime::parse_from_str(&cleaned, "%Y/%m/%d %H:%M")
        .ok()
        .and_then(|naive| Utc.from_local_datetime(&naive).single())
        .map(|dt| dt.timestamp())
}

/// Format a unix-seconds timestamp as ISO-8601 calendar date "YYYY-MM-DD" in UTC.
/// Returns "unknown date" on conversion failure (e.g. malformed timestamp).
pub fn format_ymd(ts: i64) -> String {
    use chrono::{TimeZone, Utc};
    Utc.timestamp_opt(ts, 0)
        .single()
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "unknown date".to_string())
}

/// Resolve `last_modified` for a benchmark-seeded chunk: parse the per-session
/// date string with `parser` if present, else fall back to `now()` (used for
/// noise / undated entries).
///
/// Logs a warning when a non-empty date string fails to parse, so silent
/// degradation to today's date is visible in eval logs.
pub(crate) fn seed_last_modified(date: Option<&str>, parser: fn(&str) -> Option<i64>) -> i64 {
    if let Some(s) = date {
        if let Some(ts) = parser(s) {
            return ts;
        }
        log::warn!(
            "[eval:dates] failed to parse date {s:?}; falling back to now() — temporal accuracy lost"
        );
    }
    chrono::Utc::now().timestamp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_locomo_date ────────────────────────────────────────────────────

    #[test]
    fn test_parse_locomo_date() {
        let ts = parse_locomo_date("1:56 pm on 8 May, 2023").expect("should parse");
        // 2023-05-08 13:56 UTC = 1683554160
        assert_eq!(ts, 1_683_554_160);
    }

    #[test]
    fn test_parse_locomo_date_garbage_returns_none() {
        assert!(parse_locomo_date("nonsense").is_none());
    }

    // ── parse_lme_date ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_lme_date_round_trip() {
        let ts = parse_lme_date("2023/04/10 (Mon) 23:07").expect("should parse");
        // 2023-04-10 23:07 UTC == 1681168020
        assert_eq!(ts, 1_681_168_020);
    }

    #[test]
    fn test_parse_lme_date_garbage_returns_none() {
        assert!(parse_lme_date("not a date").is_none());
        assert!(parse_lme_date("").is_none());
    }

    // ── format_ymd ───────────────────────────────────────────────────────────

    #[test]
    fn test_format_ymd_round_trip() {
        assert_eq!(format_ymd(1_681_168_020), "2023-04-10");
        assert_eq!(format_ymd(1_683_554_160), "2023-05-08");
        assert_eq!(format_ymd(0), "1970-01-01");
    }

    // ── seed_last_modified ───────────────────────────────────────────────────

    #[test]
    fn test_seed_last_modified_parses_when_date_present() {
        let ts = seed_last_modified(Some("2023/04/10 (Mon) 23:07"), parse_lme_date);
        assert_eq!(ts, 1_681_168_020);
    }

    #[test]
    fn test_seed_last_modified_falls_back_to_now_when_date_missing() {
        let before = chrono::Utc::now().timestamp();
        let ts = seed_last_modified(None, parse_lme_date);
        let after = chrono::Utc::now().timestamp();
        assert!(ts >= before && ts <= after);
    }

    #[test]
    fn test_seed_last_modified_falls_back_when_parser_rejects() {
        let before = chrono::Utc::now().timestamp();
        let ts = seed_last_modified(Some("malformed"), parse_lme_date);
        let after = chrono::Utc::now().timestamp();
        assert!(ts >= before && ts <= after);
    }
}
