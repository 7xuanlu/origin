// SPDX-License-Identifier: AGPL-3.0-only
/// Unified trigger event enum. All input types flow through a single
/// `tokio::sync::mpsc::channel<TriggerEvent>` for serial processing.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum TriggerEvent {
    /// Global keyboard shortcut (Cmd+Shift+M).
    /// Forces OCR + intent classification.
    ManualHotkey,

    /// User selected a screen region via snip overlay.
    /// Pre-crops before OCR for compute savings.
    DragSnip {
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    },

    /// User typed a thought into quick-capture UI.
    /// Bypasses vision entirely — zero compute.
    QuickThought { text: String },

    /// Focus changed to a new window — ambient overlay may surface cards.
    FocusChange {
        app_name: String,
        window_title: String,
    },

    /// User highlighted text in any app — triggers memory search near cursor.
    TextSelected(SelectionEvent),

    /// Text selected — show icon near cursor, card fires on click.
    /// x, y in macOS logical coordinates (origin = bottom-left of primary display).
    TextIcon { text: String, x: f64, y: f64 },

    /// Hide the icon overlay (user clicked/deselected).
    HideIcon,
}

/// Payload for a text-selection trigger.
#[derive(Debug, Clone)]
pub struct SelectionEvent {
    /// The selected text (≥15 chars after debounce).
    pub text: String,
    /// Cursor X in macOS logical coordinates (0 = left edge of primary display).
    pub cursor_x: f64,
    /// Cursor Y in macOS logical coordinates (0 = bottom edge on macOS, top on Tauri).
    pub cursor_y: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_event_has_expected_fields() {
        let ev = TriggerEvent::TextSelected(SelectionEvent {
            text: "hello world this is a test".into(),
            cursor_x: 100.0,
            cursor_y: 200.0,
        });
        match ev {
            TriggerEvent::TextSelected(sel) => {
                assert_eq!(sel.text, "hello world this is a test");
                assert_eq!(sel.cursor_x, 100.0);
                assert_eq!(sel.cursor_y, 200.0);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn text_icon_event_has_expected_fields() {
        let ev = TriggerEvent::TextIcon {
            text: "hello world this is a test".into(),
            x: 500.0,
            y: 300.0,
        };
        match ev {
            TriggerEvent::TextIcon { text, x, y } => {
                assert_eq!(text, "hello world this is a test");
                assert_eq!(x, 500.0);
                assert_eq!(y, 300.0);
            }
            _ => panic!("wrong variant"),
        }
    }
}
