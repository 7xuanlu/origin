// SPDX-License-Identifier: AGPL-3.0-only
use crate::ambient::types::SelectionCardEvent;
use crate::privacy::redact_pii;
use crate::router::bundle::{assemble_bundle_with_intent, ContextBundle};
use crate::router::keywords;
use crate::sensor;
use crate::state::AppState;
use crate::trigger::types::TriggerEvent;

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::RwLock;

use origin_core::working_memory::{WorkingMemoryEntry, MAX_SNIPPET_CHARS};
use origin_types::requests::{IngestTextRequest, SearchMemoryRequest};
use origin_types::responses::{IngestResponse, SearchMemoryResponse};

/// Default keyword classification threshold (matches TuningConfig default).
const DEFAULT_KEYWORD_MIN_THRESHOLD: f64 = 0.005;
/// Default consumer dedup threshold (matches TuningConfig default).
const DEFAULT_CONSUMER_DEDUP_THRESHOLD: f64 = 0.85;
/// Default consumer dedup window in seconds (matches TuningConfig default).
const DEFAULT_CONSUMER_DEDUP_WINDOW_SECS: i64 = 60;

/// Stable hash of a window's OCR text for per-window dedup fast path.
#[allow(dead_code)]
fn content_hash(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Composite key that uniquely identifies a window across captures.
#[allow(dead_code)]
fn window_key(app_name: &str, window_title: &str) -> String {
    format!("{}\0{}", app_name, window_title)
}

/// Bigram Jaccard similarity between two strings.
///
/// Returns 0.0 (completely different) to 1.0 (identical).
fn text_similarity(a: &str, b: &str) -> f64 {
    fn bigrams(s: &str) -> HashSet<(char, char)> {
        s.chars().zip(s.chars().skip(1)).collect()
    }

    let ba = bigrams(a);
    let bb = bigrams(b);

    if ba.is_empty() && bb.is_empty() {
        return 1.0;
    }

    let intersection = ba.intersection(&bb).count();
    let union = ba.union(&bb).count();

    if union == 0 {
        return 1.0;
    }

    intersection as f64 / union as f64
}

/// The smart router — core of the unified trigger architecture.
///
/// Pattern-matches on TriggerEvent to apply the right optimization strategy:
/// - ManualHotkey: force OCR all windows, classify intent, send to consumer
/// - DragSnip: crop before OCR, classify intent, send to consumer
/// - QuickThought: bypass vision entirely, send to consumer
pub async fn run_router(
    mut event_rx: Receiver<TriggerEvent>,
    bundle_tx: Sender<ContextBundle>,
    state: Arc<RwLock<AppState>>,
) {
    // Wait for daemon to be reachable
    loop {
        let client = state.read().await.client.clone();
        if client.health().await.is_ok() {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    log::info!("[router] started, waiting for trigger events...");

    while let Some(event) = event_rx.recv().await {
        match event {
            // ── HOTKEY: force OCR all windows, classify intent, send to consumer ──
            TriggerEvent::ManualHotkey => {
                log::info!("[router] hotkey trigger — forcing OCR");

                // Immediately notify frontend
                let s = state.read().await;
                s.emit_capture_event(crate::state::CaptureEvent {
                    source: "hotkey".to_string(),
                    source_id: String::new(),
                    summary: String::new(),
                    chunks: 0,
                    processing: true,
                });
                drop(s);

                // Capture + OCR all windows (no filtering, no frame compare)
                let (skip_apps, skip_patterns, private_detect) = {
                    let cfg = crate::config::load_config();
                    if !cfg.screen_capture_enabled {
                        log::debug!("[router] hotkey: screen capture disabled, skipping");
                        continue;
                    }
                    (
                        cfg.skip_apps,
                        cfg.skip_title_patterns,
                        cfg.private_browsing_detection,
                    )
                };

                let captures = match sensor::vision::capture_windows(
                    &skip_apps,
                    &skip_patterns,
                    private_detect,
                ) {
                    Ok(c) if !c.is_empty() => c,
                    Ok(_) => {
                        log::debug!("[router] hotkey: no windows captured");
                        continue;
                    }
                    Err(e) => {
                        log::warn!("[router] hotkey capture failed: {}", e);
                        continue;
                    }
                };

                let ocr = match sensor::vision::ocr_per_window(&captures) {
                    Ok(results) if !results.is_empty() => results,
                    Ok(_) => {
                        log::debug!("[router] hotkey: OCR returned no text");
                        continue;
                    }
                    Err(e) => {
                        log::warn!("[router] hotkey OCR failed: {}", e);
                        continue;
                    }
                };

                let ocr = redact_all(ocr);

                // Classify + send to consumer
                let intent = keywords::classify(&ocr, DEFAULT_KEYWORD_MIN_THRESHOLD);
                let bundle = assemble_bundle_with_intent(ocr, &event, intent);
                let _ = bundle_tx.send(bundle).await;
            }

            // ── DRAGSNIP: native CG region capture + OCR ──
            TriggerEvent::DragSnip {
                x,
                y,
                width,
                height,
            } => {
                log::info!(
                    "[router] snip trigger: x={:.0} y={:.0} w={:.0} h={:.0}",
                    x,
                    y,
                    width,
                    height
                );

                let ocr = match sensor::vision::ocr_region(x, y, width, height) {
                    Ok(results) if !results.is_empty() => results,
                    Ok(_) => {
                        log::debug!("[router] snip: OCR returned no text");
                        continue;
                    }
                    Err(e) => {
                        log::warn!("[router] snip OCR failed: {}", e);
                        continue;
                    }
                };

                let ocr = redact_all(ocr);
                let intent = keywords::classify(&ocr, DEFAULT_KEYWORD_MIN_THRESHOLD);
                let bundle = assemble_bundle_with_intent(ocr, &event, intent);
                let _ = bundle_tx.send(bundle).await;
            }

            // ── QUICK THOUGHT: bypass vision entirely ──
            TriggerEvent::QuickThought { ref text } => {
                log::info!("[router] quick thought: {} chars", text.len());
                let bundle = ContextBundle::from_text(text.clone());
                let _ = bundle_tx.send(bundle).await;
            }

            // ── FOCUS CHANGE: ambient monitor handles this independently ──
            TriggerEvent::FocusChange { .. } => {
                // Intentionally no-op in the router. The ambient monitor runs its
                // own polling loop and processes focus changes directly.
            }

            // ── TEXT SELECTION: search with selected text via daemon, emit cursor-positioned card ──
            TriggerEvent::TextSelected(crate::trigger::types::SelectionEvent {
                text,
                cursor_x,
                cursor_y,
            }) => {
                log::info!(
                    "[router] text selected: {} chars at ({:.0},{:.0})",
                    text.len(),
                    cursor_x,
                    cursor_y
                );

                let client = state.read().await.client.clone();

                let req = SearchMemoryRequest {
                    query: text.clone(),
                    limit: 5,
                    memory_type: None,
                    domain: None,
                    source_agent: None,
                };
                let results = match client
                    .post_json::<SearchMemoryRequest, SearchMemoryResponse>(
                        "/api/memory/search",
                        &req,
                    )
                    .await
                {
                    Ok(resp) => resp.results,
                    Err(e) => {
                        log::warn!("[router] text selection search failed: {}", e);
                        continue;
                    }
                };

                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                use crate::ambient::types::{AmbientCard, AmbientCardKind};

                let good: Vec<_> = results.iter().filter(|r| r.score >= 0.8f32).collect();

                let card = if good.len() >= 2 {
                    let primary = &good[0];
                    let kind = if primary.memory_type.as_deref() == Some("decision") {
                        AmbientCardKind::DecisionReminder
                    } else {
                        AmbientCardKind::PersonContext
                    };
                    let sources: Vec<String> = good
                        .iter()
                        .filter_map(|r| r.source_agent.clone())
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    AmbientCard {
                        card_id: format!(
                            "sel-{}-{}",
                            now,
                            crate::ambient::monitor::CARD_COUNTER
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        ),
                        kind,
                        title: text.chars().take(40).collect(),
                        topic: primary.domain.clone().unwrap_or_default(),
                        body: primary.content.chars().take(200).collect(),
                        sources,
                        memory_count: good.len(),
                        primary_source_id: primary.source_id.clone(),
                        created_at: now,
                        loading: false,
                        snippets: vec![],
                    }
                } else {
                    // No-results card
                    AmbientCard {
                        card_id: format!("sel-none-{}", now),
                        kind: AmbientCardKind::PersonContext,
                        title: text.chars().take(40).collect(),
                        topic: String::new(),
                        body: "No memories found for this selection.".to_string(),
                        sources: vec![],
                        memory_count: 0,
                        primary_source_id: String::new(),
                        created_at: now,
                        loading: false,
                        snippets: vec![],
                    }
                };

                let payload = SelectionCardEvent {
                    card,
                    cursor_x,
                    cursor_y,
                };

                let s = state.read().await;
                if let Some(handle) = &s.app_handle {
                    use tauri::Emitter;
                    let _ = handle.emit("selection-card", &payload);
                }
            }

            // ── TEXT ICON: show icon overlay, card fires on click ──
            TriggerEvent::TextIcon { text, x, y } => {
                log::info!(
                    "[router] text icon: {} chars at ({:.0},{:.0})",
                    text.len(),
                    x,
                    y
                );
                let payload = crate::ambient::types::ShowIconPayload { text, x, y };
                let s = state.read().await;
                if let Some(handle) = &s.app_handle {
                    use tauri::Emitter;
                    let _ = handle.emit("show-icon", &payload);
                }
            }

            // ── HIDE ICON: user clicked/deselected ──
            TriggerEvent::HideIcon => {
                let s = state.read().await;
                if let Some(handle) = &s.app_handle {
                    use tauri::Emitter;
                    let _ = handle.emit("hide-icon", ());
                }
            }
        }
    }

    log::info!("[router] stopped (channel closed)");
}

/// Apply PII redaction to all OCR results.
fn redact_all(
    ocr: Vec<crate::sensor::vision::WindowOcrResult>,
) -> Vec<crate::sensor::vision::WindowOcrResult> {
    ocr.into_iter()
        .map(|mut r| {
            r.text = redact_pii(&r.text);
            r
        })
        .collect()
}

/// Context consumer task — receives bundles and sends them to the daemon
/// via HTTP for storage. LLM classification and space management are handled
/// by the daemon.
pub async fn run_context_consumer(
    mut bundle_rx: Receiver<ContextBundle>,
    state: Arc<RwLock<AppState>>,
) {
    // Wait for daemon to be reachable
    loop {
        let client = state.read().await.client.clone();
        if client.health().await.is_ok() {
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    log::info!("[consumer] started, waiting for context bundles...");

    // Consumer dedup: track last capture to skip near-duplicates
    let mut last_consumer: Option<(String, String, i64)> = None; // (source, text, timestamp)

    while let Some(bundle) = bundle_rx.recv().await {
        let trigger_type = bundle.trigger_type.clone();

        // Build title and content from the bundle (same logic as bundle_to_raw_document)
        let (source, source_id, title, content, url, metadata) = extract_ingest_fields(&bundle);

        // Snapshot working-memory metadata from the bundle before it's moved.
        // For QuickThought (raw_text Some), synthesize an "Origin / Quick Thought"
        // entry; otherwise use the focused (or first) window.
        let wm_entry_data: Option<(String, String, String)> =
            if let Some(ref text) = bundle.raw_text {
                Some((
                    "Origin".to_string(),
                    "Quick Thought".to_string(),
                    text.chars().take(MAX_SNIPPET_CHARS).collect(),
                ))
            } else {
                bundle
                    .windows
                    .iter()
                    .find(|w| w.focused)
                    .or(bundle.windows.first())
                    .map(|w| {
                        (
                            w.app_name.clone(),
                            w.window_title.clone(),
                            w.text.chars().take(MAX_SNIPPET_CHARS).collect(),
                        )
                    })
            };

        // Consumer dedup: if same content as last capture within time window, skip it
        let now_ts = chrono::Utc::now().timestamp();
        if let Some((ref prev_source, ref prev_text, prev_ts)) = last_consumer {
            let within_window = (now_ts - prev_ts) <= DEFAULT_CONSUMER_DEDUP_WINDOW_SECS;
            if within_window
                && prev_source == &source
                && text_similarity(prev_text, &content) >= DEFAULT_CONSUMER_DEDUP_THRESHOLD
            {
                log::info!("[consumer] dedup: skipping near-duplicate capture");
                continue;
            }
        }

        let client = state.read().await.client.clone();

        // Send to daemon via HTTP ingest
        let req = IngestTextRequest {
            source: source.clone(),
            source_id: source_id.clone(),
            title: title.clone(),
            content: content.clone(),
            url,
            metadata: if metadata.is_empty() {
                None
            } else {
                Some(metadata)
            },
        };

        match client
            .post_json::<IngestTextRequest, IngestResponse>("/api/ingest/text", &req)
            .await
        {
            Ok(resp) => {
                let now = chrono::Utc::now().timestamp();

                // Clone the working memory Arc out of the state guard so we
                // can push without holding the AppState write lock across
                // the Mutex acquisition.
                let wm_handle = {
                    let mut s = state.write().await;
                    s.last_ingestion_at = now;
                    s.touch_activity(bundle.timestamp.timestamp());

                    // Emit capture event to frontend (toast notification).
                    // LLM classification is handled by the daemon.
                    s.emit_capture_event(crate::state::CaptureEvent {
                        source: trigger_type.clone(),
                        source_id: source_id.clone(),
                        summary: title.clone(),
                        chunks: resp.chunks_created,
                        processing: false,
                    });

                    let handle = s.working_memory.clone();
                    // Store last bundle for retrieval via Tauri command
                    s.last_context_bundle = Some(bundle);
                    handle
                };

                // Push to working memory (rolling buffer, 15-min retention)
                if let Some((app_name, window_title, text_snippet)) = wm_entry_data {
                    let mut wm = wm_handle.lock().await;
                    wm.push(WorkingMemoryEntry {
                        timestamp: now,
                        source: trigger_type.clone(),
                        app_name,
                        window_title,
                        text_snippet,
                        source_id: source_id.clone(),
                    });
                }

                last_consumer = Some((source.clone(), content.clone(), now));

                log::info!(
                    "[consumer] ingested {}: \"{}\" ({} chunks)",
                    trigger_type,
                    title,
                    resp.chunks_created
                );
            }
            Err(e) => {
                log::warn!("[consumer] daemon ingest failed: {}", e);

                // Still store the bundle so it can be retried
                let mut s = state.write().await;
                s.last_context_bundle = Some(bundle);
            }
        }
    }

    log::info!("[consumer] stopped (channel closed)");
}

/// Extract fields needed for the ingest HTTP request from a ContextBundle.
fn extract_ingest_fields(
    bundle: &ContextBundle,
) -> (
    String,
    String,
    String,
    String,
    Option<String>,
    std::collections::HashMap<String, String>,
) {
    let source = match bundle.trigger_type.as_str() {
        "ambient" => "ambient",
        "focus" => "focus_capture",
        "hotkey" => "hotkey_capture",
        "snip" => "snip_capture",
        "thought" => "quick_thought",
        _ => "context",
    }
    .to_string();

    let source_id = format!("ctx_{}", bundle.timestamp.timestamp());

    let title = if let Some(ref text) = bundle.raw_text {
        let first_line = text.lines().next().unwrap_or("Quick Thought");
        if first_line.len() > 60 {
            format!("{}...", &first_line[..first_line.floor_char_boundary(60)])
        } else {
            first_line.to_string()
        }
    } else {
        bundle
            .windows
            .iter()
            .find(|w| w.focused)
            .or(bundle.windows.first())
            .map(|w| {
                let t = format!("{} — {}", w.app_name, w.window_title);
                if t.len() > 80 {
                    format!("{}...", &t[..t.floor_char_boundary(80)])
                } else {
                    t
                }
            })
            .unwrap_or_else(|| "Context Capture".to_string())
    };

    let content = if let Some(ref text) = bundle.raw_text {
        text.clone()
    } else {
        bundle
            .windows
            .iter()
            .map(|w| format!("## {} — {}\n{}", w.app_name, w.window_title, w.text))
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    let url = bundle.windows.iter().find_map(|w| w.url.clone());

    let mut metadata = std::collections::HashMap::new();
    if bundle.raw_text.is_none() {
        metadata.insert("screen_capture".to_string(), "true".to_string());
    }

    (source, source_id, title, content, url, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_selected_event_is_recognized() {
        use crate::trigger::types::{SelectionEvent, TriggerEvent};
        let event = TriggerEvent::TextSelected(SelectionEvent {
            text: "hello world test selection text".into(),
            cursor_x: 500.0,
            cursor_y: 300.0,
        });
        assert!(matches!(event, TriggerEvent::TextSelected(_)));
    }

    #[test]
    fn test_content_hash_deterministic() {
        let text = "The quick brown fox jumps over the lazy dog";
        assert_eq!(content_hash(text), content_hash(text));
        assert_ne!(content_hash(text), content_hash("different text"));
    }

    #[test]
    fn test_window_key_unique() {
        let k1 = window_key("Safari", "Google");
        let k2 = window_key("Chrome", "Google");
        let k3 = window_key("Safari", "GitHub");
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k2, k3);
        assert_eq!(k1, window_key("Safari", "Google"));
    }

    #[test]
    fn test_text_similarity_identical() {
        assert!((text_similarity("hello world", "hello world") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_text_similarity_completely_different() {
        let sim = text_similarity("abcdef", "xyz123");
        assert!(sim < 0.1, "expected <0.1, got {}", sim);
    }

    #[test]
    fn test_text_similarity_minor_ocr_noise() {
        // Simulates OCR noise: "Daylight Savi..." vs "8ight Savi..."
        let base = "The quick brown fox jumps over the lazy dog. This is a test of the OCR capture system.";
        let noisy = "The quick brown fox jumps over the lazy dog. This is a test of the OCR capture systen.";
        let sim = text_similarity(base, noisy);
        assert!(
            sim >= 0.85,
            "OCR noise should be above threshold, got {}",
            sim
        );
    }

    #[test]
    fn test_text_similarity_different_content() {
        let vscode = "fn main() { println!(\"hello\"); } // Rust code in VS Code editor with syntax highlighting";
        let chrome = "Google Search - Chrome browser tab showing search results for Rust programming language";
        let sim = text_similarity(vscode, chrome);
        assert!(
            sim < 0.85,
            "different content should be below threshold, got {}",
            sim
        );
    }
}
