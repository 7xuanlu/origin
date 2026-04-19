// SPDX-License-Identifier: AGPL-3.0-only

use serde::{Deserialize, Serialize};

/// Ambient overlay operating mode. Stored in config, switchable from Settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AmbientMode {
    /// Cards surface automatically on context change (focus polling + OCR).
    Proactive,
    /// Cards only surface when user hits the hotkey (Cmd+Shift+O).
    #[default]
    OnDemand,
    /// Ambient overlay disabled entirely.
    Off,
}

/// The kind of ambient card to surface.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AmbientCardKind {
    PersonContext,
    DecisionReminder,
}

/// A single memory excerpt shown inside a card's source list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnippet {
    /// Display name for the source (source_agent or domain).
    pub source: String,
    /// Short excerpt from the memory content (≤ 80 chars).
    pub text: String,
}

/// Payload emitted to the frontend via Tauri event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientCard {
    /// Unique ID for this card instance (for dismiss tracking).
    pub card_id: String,
    pub kind: AmbientCardKind,
    /// e.g. "Alice" or "CRM Selection"
    pub title: String,
    /// e.g. "Q3 Budget" or "Feb 2026"
    pub topic: String,
    /// 1-2 sentence insight.
    pub body: String,
    /// Source agent names that contributed.
    pub sources: Vec<String>,
    /// Number of memories that matched.
    pub memory_count: usize,
    /// source_id of the primary memory (for open-detail navigation).
    pub primary_source_id: String,
    /// Timestamp when card was created.
    pub created_at: u64,
    /// True while the card is a placeholder waiting for search results.
    #[serde(default)]
    pub loading: bool,
    /// Individual memory excerpts that contributed to the synthesis.
    #[serde(default)]
    pub snippets: Vec<MemorySnippet>,
}

/// Payload emitted as `"selection-card"` Tauri event.
/// Carries the card (or no-results card) + cursor position for overlay placement.
#[derive(Debug, Clone, Serialize)]
pub struct SelectionCardEvent {
    pub card: AmbientCard,
    /// Cursor X in macOS logical coordinates (0 = left of primary display).
    pub cursor_x: f64,
    /// Cursor Y in macOS logical coordinates (0 = bottom of primary display on macOS).
    pub cursor_y: f64,
}

/// Snapshot of what the user is currently focused on.
#[derive(Debug, Clone)]
pub struct AmbientContext {
    pub app_name: String,
    pub window_title: String,
    #[allow(dead_code)]
    pub ocr_text: Option<String>,
    #[allow(dead_code)]
    pub timestamp: u64,
}

/// Result of an on-demand ambient trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientTriggerResult {
    /// Number of cards emitted.
    pub cards_emitted: usize,
    /// What the system detected about current context.
    pub context_summary: String,
    /// Why no cards were shown (if cards_emitted == 0).
    pub reason: Option<String>,
}

/// Payload emitted as `"show-icon"` Tauri event.
/// Tells the icon overlay window where to appear and what text to search on click.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ShowIconPayload {
    /// The text to search when the icon is clicked (selected text).
    pub text: String,
    /// Icon X position in macOS logical coordinates (origin = bottom-left).
    pub x: f64,
    /// Icon Y position in macOS logical coordinates (origin = bottom-left).
    pub y: f64,
}

/// Tracks a dismissed card to avoid resurfacing.
#[derive(Debug, Clone)]
pub struct DismissalRecord {
    /// The search query that produced this card.
    pub query: String,
    /// When the user dismissed it.
    pub dismissed_at: u64,
}

impl DismissalRecord {
    /// Returns true if dismissal is still active (within 24 hours).
    pub fn is_active(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now.saturating_sub(self.dismissed_at) < 86_400
    }
}
