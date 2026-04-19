// SPDX-License-Identifier: AGPL-3.0-only
//! Context packager: chunks capture streams into session snapshots.
//!
//! This module holds the pure logic — session chunking, entertainment app
//! detection, activity log construction, session JSON parsing. The
//! state-coupled background timer (`run_packager_timer`) still lives in the
//! Tauri app crate because it consumes `AppState` directly; it will move into
//! origin-server during Task 4.

use crate::db::CaptureRefRow;
use crate::tuning::PackagerConfig;

/// Parse a session synthesis JSON response from the LLM.
/// Returns (summary, tags) or None if parsing fails.
pub fn parse_session_json(raw: &str) -> Option<(String, Vec<String>)> {
    let stripped = crate::llm_provider::strip_think_tags(raw);
    let json_str = crate::llm_provider::extract_json(&stripped)?;
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;

    let summary = val
        .get("summary")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default();
    if summary.is_empty() {
        return None;
    }

    let tags = val
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    Some((summary, tags))
}

/// Split a list of captures (sorted by timestamp) into sessions.
pub fn chunk_into_sessions<'a>(
    captures: &'a [CaptureRefRow],
    tuning: &PackagerConfig,
) -> Vec<Vec<&'a CaptureRefRow>> {
    if captures.is_empty() {
        return Vec::new();
    }

    let mut sessions: Vec<Vec<&CaptureRefRow>> = Vec::new();
    let mut current: Vec<&CaptureRefRow> = vec![&captures[0]];

    for cap in &captures[1..] {
        let last = current.last().unwrap();
        let time_gap = cap.timestamp - last.timestamp;
        let is_time_split = time_gap > tuning.session_gap_secs;

        let is_context_shift = last.app_name != cap.app_name && is_entertainment_app(&cap.app_name);

        let session_start = current.first().unwrap().timestamp;
        let is_too_long = cap.timestamp - session_start > tuning.max_session_duration_secs;

        if is_time_split || is_context_shift || is_too_long {
            sessions.push(current);
            current = Vec::new();
        }
        current.push(cap);
    }

    if !current.is_empty() {
        sessions.push(current);
    }

    // Filter out trivial sessions (but keep entertainment sessions as context-break markers)
    sessions
        .into_iter()
        .filter(|s: &Vec<&CaptureRefRow>| {
            s.len() >= tuning.min_session_captures
                || s.iter().any(|c| is_entertainment_app(&c.app_name))
        })
        .collect()
}

/// Check if an app is entertainment (context break signal).
pub fn is_entertainment_app(app_name: &str) -> bool {
    let lower = app_name.to_lowercase();
    matches!(
        lower.as_str(),
        "netflix"
            | "spotify"
            | "youtube"
            | "steam"
            | "tv"
            | "apple tv"
            | "disney+"
            | "hbo max"
            | "twitch"
    )
}

/// Extract deduplicated app names preserving first-appearance order.
pub fn extract_primary_apps(captures: &[CaptureRefRow]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    captures
        .iter()
        .filter_map(|c| {
            if seen.insert(c.app_name.clone()) {
                Some(c.app_name.clone())
            } else {
                None
            }
        })
        .collect()
}

/// Build the activity log text from captures for SLM input.
pub fn build_activity_log(captures: &[&CaptureRefRow]) -> String {
    let mut log = String::new();
    for cap in captures {
        let ts = chrono::DateTime::from_timestamp(cap.timestamp, 0)
            .map(|dt| dt.format("%H:%M").to_string())
            .unwrap_or_else(|| "??:??".to_string());
        log.push_str(&format!(
            "[{}] {}: {}\n",
            ts, cap.app_name, cap.window_title
        ));
    }
    log
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ref(source_id: &str, app: &str, ts: i64) -> CaptureRefRow {
        CaptureRefRow {
            source_id: source_id.to_string(),
            activity_id: "act-001".to_string(),
            snapshot_id: None,
            app_name: app.to_string(),
            window_title: "test".to_string(),
            timestamp: ts,
            source: "ambient".to_string(),
        }
    }

    #[test]
    fn test_no_split_continuous_work() {
        let tuning = PackagerConfig::default();
        let refs: Vec<_> = (0..10)
            .map(|i| make_ref(&format!("ctx_{}", i), "VS Code", 1000 + i * 30))
            .collect();
        let sessions = chunk_into_sessions(&refs, &tuning);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].len(), 10);
    }

    #[test]
    fn test_split_on_time_gap() {
        let tuning = PackagerConfig::default();
        let mut refs: Vec<_> = (0..5)
            .map(|i| make_ref(&format!("ctx_{}", i), "VS Code", 1000 + i * 30))
            .collect();
        refs.extend((5..10).map(|i| make_ref(&format!("ctx_{}", i), "VS Code", 2000 + i * 30)));
        let sessions = chunk_into_sessions(&refs, &tuning);
        assert_eq!(sessions.len(), 2);
    }

    #[test]
    fn test_split_on_entertainment() {
        let tuning = PackagerConfig::default();
        let refs = vec![
            make_ref("ctx_1", "VS Code", 1000),
            make_ref("ctx_2", "VS Code", 1030),
            make_ref("ctx_3", "VS Code", 1060),
            make_ref("ctx_4", "Netflix", 1090),
            make_ref("ctx_5", "Netflix", 1120),
        ];
        let sessions = chunk_into_sessions(&refs, &tuning);
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].len(), 3);
        assert_eq!(sessions[1].len(), 2);
    }

    #[test]
    fn test_skip_trivial_sessions() {
        let tuning = PackagerConfig::default();
        let refs = vec![
            make_ref("ctx_1", "VS Code", 1000),
            make_ref("ctx_2", "VS Code", 1030),
        ];
        let sessions = chunk_into_sessions(&refs, &tuning);
        assert_eq!(sessions.len(), 0);
    }

    #[test]
    fn test_force_split_long_session() {
        let tuning = PackagerConfig::default();
        let refs: Vec<_> = (0..360)
            .map(|i| make_ref(&format!("ctx_{}", i), "VS Code", 1000 + i * 30))
            .collect();
        let sessions = chunk_into_sessions(&refs, &tuning);
        assert!(
            sessions.len() >= 2,
            "should split sessions longer than 2 hours"
        );
    }

    #[test]
    fn test_is_entertainment_app() {
        assert!(is_entertainment_app("Netflix"));
        assert!(is_entertainment_app("Spotify"));
        assert!(is_entertainment_app("YouTube"));
        assert!(is_entertainment_app("Steam"));
        assert!(!is_entertainment_app("VS Code"));
        assert!(!is_entertainment_app("Chrome"));
    }

    #[test]
    fn test_extract_primary_apps() {
        let refs = vec![
            make_ref("ctx_1", "VS Code", 1000),
            make_ref("ctx_2", "Chrome", 1030),
            make_ref("ctx_3", "VS Code", 1060),
            make_ref("ctx_4", "Chrome", 1090),
            make_ref("ctx_5", "Terminal", 1120),
        ];
        let apps = extract_primary_apps(&refs);
        assert_eq!(apps, vec!["VS Code", "Chrome", "Terminal"]);
    }
}
