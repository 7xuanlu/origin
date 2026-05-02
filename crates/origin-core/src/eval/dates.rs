// SPDX-License-Identifier: AGPL-3.0-only
//! Date helpers shared across eval benchmark adapters.
//!
//! Centralises the three date utilities that were previously scattered across
//! `locomo`, `longmemeval`, and `shared`:
//!
//! - [`parse_locomo_date`] — LoCoMo session timestamps ("1:56 pm on 8 May, 2023")
//! - [`parse_lme_date`]    — LongMemEval session timestamps ("2023/04/10 (Mon) 23:07")
//! - [`format_ymd`]        — Unix-seconds → "YYYY-MM-DD" formatting
//! - [`seed_event_date`]   — resolve a benchmark chunk's `event_date` field

/// Parse a LoCoMo session date like "1:56 pm on 8 May, 2023" into Unix seconds.
/// Returns `None` on parse failure (caller falls back to `now()`).
pub fn parse_locomo_date(s: &str) -> Option<i64> {
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
pub fn parse_lme_date(s: &str) -> Option<i64> {
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

/// Format a `SearchResult`'s display date for eval contexts: prefer
/// `event_date` (when the event happened) over `last_modified` (ingestion
/// time), but emit "unknown date" when `event_date` is None — *not*
/// today's date. Eval seeds set `last_modified = now()`, so falling back
/// to it would silently print today's date for old benchmark events,
/// actively misleading the LLM. Production code that wants "show last
/// edit time when event is unknown" should call `format_ymd` directly.
pub fn format_event_or_unknown(event_date: Option<i64>) -> String {
    match event_date {
        Some(ts) => format_ymd(ts),
        None => "unknown date".to_string(),
    }
}

/// Resolve `event_date` for a benchmark-seeded chunk: parse the per-session
/// date string with `parser`. Returns `None` if no date is provided, or if
/// parsing fails (with a warning so silent degradation is visible in logs).
///
/// Used at seed sites to populate `RawDocument.event_date` while
/// `last_modified` stays at `now()` — so search ranking treats benchmark
/// memories as fresh while LLM context still sees the original event date.
pub fn seed_event_date(date: Option<&str>, parser: fn(&str) -> Option<i64>) -> Option<i64> {
    let s = date?;
    if let Some(ts) = parser(s) {
        return Some(ts);
    }
    log::warn!(
        "[eval:dates] failed to parse date {s:?}; event_date set to None — display falls back to last_modified"
    );
    None
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

    // ── seed_event_date ──────────────────────────────────────────────────────

    #[test]
    fn test_seed_event_date_parses_when_date_present() {
        assert_eq!(
            seed_event_date(Some("2023/04/10 (Mon) 23:07"), parse_lme_date),
            Some(1_681_168_020)
        );
    }

    #[test]
    fn test_seed_event_date_returns_none_when_date_missing() {
        assert_eq!(seed_event_date(None, parse_lme_date), None);
    }

    #[test]
    fn test_seed_event_date_returns_none_when_parser_rejects() {
        // malformed string still yields None (with a logged warning) rather than
        // silently turning into "today" — that was the bug seed_last_modified had.
        assert_eq!(seed_event_date(Some("malformed"), parse_lme_date), None);
    }
}
