// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

pub static CARD_COUNTER: AtomicU64 = AtomicU64::new(0);

use crate::ambient::extract;
#[cfg(test)]
use crate::ambient::types::DismissalRecord;
use crate::ambient::types::{
    AmbientCard, AmbientCardKind, AmbientContext, AmbientMode, AmbientTriggerResult,
};
use crate::sensor::idle;
use crate::sensor::vision;
use crate::state::AppState;
use origin_types::requests::SearchMemoryRequest;
use origin_types::responses::SearchMemoryResponse;

/// Minimum RAW RRF score (before normalization) to consider results relevant.
/// Typical values: irrelevant noise = 0.003–0.006, weak match = 0.01–0.02,
/// genuine match = 0.03+. This gate prevents normalized scores (always 1.0
/// for top result) from passing the threshold on garbage results.
const RAW_SCORE_FLOOR: f32 = 0.02;

/// Minimum number of matching memories to surface a card.
const MIN_MEMORY_COUNT: usize = 1;

/// Don't interrupt if user has been typing/active for less than this many seconds
/// (they're in flow). Only surface cards when idle >= FLOW_GUARD_SECS or context just changed.
const FLOW_GUARD_SECS: f64 = 2.0;

/// How often to poll the focused window (milliseconds).
const POLL_INTERVAL_MS: u64 = 3000;

/// Skip apps that are Origin itself or system apps.
const SKIP_APPS: &[&str] = &["origin", "finder", "systempreferences", "system settings"];

/// Main ambient monitor loop.
/// Polls the focused window, detects context changes, triggers smart capture,
/// queries search, and emits ambient-card events.
pub async fn run_ambient_monitor(state: Arc<RwLock<AppState>>) {
    log::info!("[ambient] monitor started");

    let mut last_context: Option<AmbientContext> = None;

    loop {
        tokio::time::sleep(tokio::time::Duration::from_millis(POLL_INTERVAL_MS)).await;

        // Only run in Proactive mode with screen capture enabled
        {
            let app_state = state.read().await;
            if app_state.ambient_mode != AmbientMode::Proactive || !app_state.screen_capture_enabled
            {
                continue;
            }
        }

        // Skip if user is AFK
        if idle::user_idle_seconds() > 60.0 {
            continue;
        }

        // Fast metadata scan — no screenshots. ~5ms vs 200ms×N for full capture.
        let meta = match vision::focused_window_meta() {
            Some(m) => m,
            None => continue,
        };

        // Skip Origin's own windows and system apps (metadata check, free)
        if SKIP_APPS
            .iter()
            .any(|s| meta.app_name.to_lowercase().contains(s))
        {
            continue;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let current = AmbientContext {
            app_name: meta.app_name,
            window_title: meta.window_name,
            ocr_text: None,
            timestamp: now,
        };

        // Check if context actually changed (cheap string compare)
        if !context_changed(&last_context, &current) {
            continue;
        }

        // Flow guard: if user JUST switched (idle < 2s), wait for them to settle
        if idle::user_idle_seconds() < FLOW_GUARD_SECS {
            // Will catch it on the next poll after they settle
            continue;
        }

        log::debug!(
            "[ambient] context change: {} — {}",
            current.app_name,
            current.window_title
        );

        last_context = Some(current.clone());

        // Always capture + OCR the focused window, then combine with title for full-context search.
        let ocr_text = match vision::capture_focused_window() {
            Ok(Some(capture)) => ocr_focused_capture(&capture),
            _ => None,
        };

        // Build context query: title + OCR (truncated). This is what gets embedded for search.
        let context_query = build_context_query(
            &current.app_name,
            &current.window_title,
            ocr_text.as_deref(),
        );

        // Extract label for card title (best-effort person/topic name, falls back to title segment).
        let card_label = best_label(
            &current.app_name,
            &current.window_title,
            ocr_text.as_deref(),
        );

        if let Some(card) = search_with_context(&state, &context_query, &card_label).await {
            emit_card(&state, card).await;
        }
    }
}

/// Check if context meaningfully changed from last poll.
fn context_changed(last: &Option<AmbientContext>, current: &AmbientContext) -> bool {
    match last {
        None => true,
        Some(prev) => {
            prev.app_name != current.app_name || prev.window_title != current.window_title
        }
    }
}

/// Run OCR on a single pre-captured focused window. Returns text or None.
fn ocr_focused_capture(capture: &vision::WindowCapture) -> Option<String> {
    let results = vision::ocr_per_window(std::slice::from_ref(capture)).ok()?;
    let text: String = results
        .iter()
        .map(|r| r.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Build the search query from app name, window title, and OCR text.
/// Combines all three for full-context semantic search. Truncates OCR to stay
/// within the embedding model's ~512-token limit (~1500 chars of OCR is safe).
fn build_context_query(app: &str, title: &str, ocr: Option<&str>) -> String {
    let mut query = format!("{} — {}", app, title);
    if let Some(text) = ocr {
        let truncated: String = text.chars().take(1500).collect();
        if !truncated.trim().is_empty() {
            query.push('\n');
            query.push_str(truncated.trim());
        }
    }
    query
}

/// Pick the best human-readable label for the card title.
/// Tries extraction (person names first, then topics). Falls back to the first
/// meaningful segment of the window title.
fn best_label(app: &str, title: &str, ocr: Option<&str>) -> String {
    let mut extracted = extract::extract_from_window_title(app, title);
    if extracted.is_empty() {
        if let Some(text) = ocr {
            let ocr_e = extract::extract_from_ocr(text);
            extracted.persons.extend(ocr_e.persons);
            extracted.topics.extend(ocr_e.topics);
        }
    }
    extracted
        .to_queries()
        .into_iter()
        .next()
        .unwrap_or_else(|| extract::first_title_segment(title).to_string())
}

/// Search memories using the full context query, build a card if enough results exceed threshold.
async fn search_with_context(
    state: &Arc<RwLock<AppState>>,
    context_query: &str,
    card_label: &str,
) -> Option<AmbientCard> {
    let (client, dismissed_queries) = {
        let app_state = state.read().await;
        let dismissed: Vec<String> = app_state
            .ambient_dismissals
            .iter()
            .filter(|d| d.is_active())
            .map(|d| d.query.clone())
            .collect();
        (app_state.client.clone(), dismissed)
    };

    // Check if this context was recently dismissed
    if dismissed_queries.iter().any(|d| d == card_label) {
        log::warn!("[ambient] dismissed: {:?}", card_label);
        return None;
    }

    // Single search with full context via daemon HTTP API
    let t = std::time::Instant::now();
    let req = SearchMemoryRequest {
        query: context_query.to_string(),
        limit: 5,
        memory_type: None,
        domain: None,
        source_agent: None,
    };
    let resp: SearchMemoryResponse = match client.post_json("/api/memory/search", &req).await {
        Ok(r) => r,
        Err(e) => {
            log::warn!("[ambient] search_memory failed: {}", e);
            return None;
        }
    };
    let results = resp.results;
    log::warn!(
        "[ambient] search_memory: {}ms — {} results, top={:.3}",
        t.elapsed().as_millis(),
        results.len(),
        results.first().map(|r| r.score).unwrap_or(0.0)
    );

    // Gate on raw (pre-normalization) scores to reject genuinely irrelevant results.
    // Normalized scores are always 1.0 for the top result — useless for relevance gating.
    let raw_top = results.first().map(|r| r.raw_score).unwrap_or(0.0);
    let good: Vec<_> = results
        .iter()
        .filter(|r| r.raw_score >= RAW_SCORE_FLOOR)
        .collect();
    if good.len() < MIN_MEMORY_COUNT {
        log::warn!(
            "[ambient] raw_top={:.4}, {} above floor {:.3} (need {}) — skipping",
            raw_top,
            good.len(),
            RAW_SCORE_FLOOR,
            MIN_MEMORY_COUNT
        );
        return None;
    }

    let primary = &good[0];
    let kind = if primary.memory_type.as_deref() == Some("decision") {
        AmbientCardKind::DecisionReminder
    } else {
        AmbientCardKind::PersonContext
    };

    let body = primary.content.chars().take(200).collect::<String>();
    let topic = primary.domain.clone().unwrap_or_default();
    let sources: Vec<String> = good
        .iter()
        .filter_map(|r| r.source_agent.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Some(AmbientCard {
        card_id: format!(
            "ambient-{}-{}",
            now,
            CARD_COUNTER.fetch_add(1, Ordering::Relaxed)
        ),
        kind,
        title: card_label.to_string(),
        topic,
        body,
        sources,
        memory_count: good.len(),
        primary_source_id: primary.source_id.clone(),
        created_at: now,
        loading: false,
        snippets: vec![],
    })
}

/// Emit the card to the ambient overlay window via Tauri event.
async fn emit_card(state: &Arc<RwLock<AppState>>, card: AmbientCard) {
    let app_state = state.read().await;
    if let Some(handle) = &app_state.app_handle {
        log::warn!(
            "[ambient] EMITTING card: {} — body: {}",
            card.title,
            card.body.chars().take(60).collect::<String>()
        );
        use tauri::Emitter;
        let _ = handle.emit("ambient-card", &card);
    } else {
        log::warn!("[ambient] EMIT FAILED: no app_handle on state!");
    }
}

/// Emit a status card to the ambient overlay so user sees feedback.
async fn emit_status_card(state: &Arc<RwLock<AppState>>, result: &AmbientTriggerResult) {
    if let Some(reason) = &result.reason {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let card = AmbientCard {
            card_id: format!(
                "status-{}-{}",
                now,
                CARD_COUNTER.fetch_add(1, Ordering::Relaxed)
            ),
            kind: AmbientCardKind::PersonContext,
            title: if result.context_summary.is_empty() {
                "Ambient".to_string()
            } else {
                result.context_summary.clone()
            },
            topic: String::new(),
            body: reason.clone(),
            sources: vec![],
            memory_count: 0,
            primary_source_id: String::new(),
            created_at: now,
            loading: false,
            snippets: vec![],
        };
        emit_card(state, card).await;
    }
}

/// On-demand trigger: capture current context, search memories, emit cards.
pub async fn trigger_ambient_now(state: &Arc<RwLock<AppState>>) -> AmbientTriggerResult {
    let no_result = |ctx: &str, reason: &str| AmbientTriggerResult {
        cards_emitted: 0,
        context_summary: ctx.to_string(),
        reason: Some(reason.to_string()),
    };

    // Check mode — don't trigger if Off
    {
        let app_state = state.read().await;
        if app_state.ambient_mode == AmbientMode::Off {
            let r = no_result("", "Ambient overlay is turned off. Enable it in Settings.");
            emit_status_card(state, &r).await;
            return r;
        }
        if !app_state.screen_capture_enabled {
            let r = no_result(
                "",
                "Screen capture is disabled. Enable it in Settings > Capture.",
            );
            emit_status_card(state, &r).await;
            return r;
        }
    }

    // Step 1: fast metadata scan — no screenshots, ~5ms
    let t0 = std::time::Instant::now();
    let meta = match vision::focused_window_meta() {
        Some(m) => m,
        None => {
            let r = no_result("", "No focused window detected.");
            emit_status_card(state, &r).await;
            return r;
        }
    };
    log::warn!(
        "[ambient] timing: focused_meta={}ms",
        t0.elapsed().as_millis()
    );

    let now_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let current = AmbientContext {
        app_name: meta.app_name,
        window_title: meta.window_name,
        ocr_text: None,
        timestamp: now_ts,
    };
    let context_summary = format!("{} — {}", current.app_name, current.window_title);
    log::warn!("[ambient] trigger: focused on {}", context_summary);

    // Capture + OCR focused window
    let t1 = std::time::Instant::now();
    let ocr_text = match vision::capture_focused_window() {
        Ok(Some(capture)) => {
            log::warn!("[ambient] timing: capture={}ms", t1.elapsed().as_millis());
            let t_ocr = std::time::Instant::now();
            let text = ocr_focused_capture(&capture);
            log::warn!(
                "[ambient] timing: ocr={}ms len={}",
                t_ocr.elapsed().as_millis(),
                text.as_ref().map(|s| s.len()).unwrap_or(0)
            );
            if let Some(ref t) = text {
                log::warn!(
                    "[ambient] ocr (first 400 chars): {:?}",
                    &t.chars().take(400).collect::<String>()
                );
            }
            text
        }
        Ok(None) => {
            log::warn!(
                "[ambient] timing: capture={}ms (no window)",
                t1.elapsed().as_millis()
            );
            None
        }
        Err(e) => {
            log::warn!("[ambient] capture failed: {}", e);
            let r = no_result(
                "",
                "Screen capture failed. Check Screen Recording permission.",
            );
            emit_status_card(state, &r).await;
            return r;
        }
    };

    // Full-context search query: app + title + OCR text (truncated to ~1500 chars)
    let context_query = build_context_query(
        &current.app_name,
        &current.window_title,
        ocr_text.as_deref(),
    );
    log::warn!(
        "[ambient] context query ({} chars): {:?}",
        context_query.len(),
        &context_query.chars().take(200).collect::<String>()
    );

    // Card label: best-effort person/topic name from extraction, or first title segment
    let card_label = best_label(
        &current.app_name,
        &current.window_title,
        ocr_text.as_deref(),
    );
    log::warn!("[ambient] card label: {:?}", card_label);

    let t2 = std::time::Instant::now();
    if let Some(card) = search_with_context(state, &context_query, &card_label).await {
        log::warn!("[ambient] timing: search={}ms", t2.elapsed().as_millis());
        let count = card.memory_count;
        emit_card(state, card).await;
        return AmbientTriggerResult {
            cards_emitted: count,
            context_summary,
            reason: None,
        };
    }

    log::warn!(
        "[ambient] timing: search={}ms (no card)",
        t2.elapsed().as_millis()
    );
    let result = no_result(
        &context_summary,
        "No relevant memories found for this context.",
    );
    emit_status_card(state, &result).await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_changed_on_first_poll() {
        let ctx = AmbientContext {
            app_name: "Slack".into(),
            window_title: "Alice - Slack".into(),
            ocr_text: None,
            timestamp: 0,
        };
        assert!(context_changed(&None, &ctx));
    }

    #[test]
    fn context_unchanged_same_window() {
        let ctx = AmbientContext {
            app_name: "Slack".into(),
            window_title: "Alice - Slack".into(),
            ocr_text: None,
            timestamp: 0,
        };
        assert!(!context_changed(&Some(ctx.clone()), &ctx));
    }

    #[test]
    fn context_changed_different_title() {
        let old = AmbientContext {
            app_name: "Slack".into(),
            window_title: "Alice - Slack".into(),
            ocr_text: None,
            timestamp: 0,
        };
        let new_ctx = AmbientContext {
            app_name: "Slack".into(),
            window_title: "Bob - Slack".into(),
            ocr_text: None,
            timestamp: 1,
        };
        assert!(context_changed(&Some(old), &new_ctx));
    }

    #[test]
    fn skip_origin_app() {
        assert!(SKIP_APPS
            .iter()
            .any(|s| "Origin".to_lowercase().contains(s)));
    }

    #[test]
    fn dismissal_active_within_24h() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let d = DismissalRecord {
            query: "Alice".into(),
            dismissed_at: now - 3600, // 1 hour ago
        };
        assert!(d.is_active());
    }

    #[test]
    fn dismissal_expired_after_24h() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let d = DismissalRecord {
            query: "Alice".into(),
            dismissed_at: now - 90_000, // 25 hours ago
        };
        assert!(!d.is_active());
    }
}
