// SPDX-License-Identifier: AGPL-3.0-only
//! The global shortcuts the app asks the OS for, and what happens when the OS
//! says no.
//!
//! A global hotkey belongs to whichever process grabbed it first, desktop-wide.
//! So registration is a request that can be refused, and it is refused often:
//! another app already owns `Ctrl+K`, or a second Wenlan is running. The
//! single-instance plugin does NOT cover the second case — it keys on the
//! bundle identifier, so a build with a different identifier (a dev build, a
//! side-by-side install, a renamed bundle) is a different "instance" to the
//! plugin and reaches this code with the first instance's hotkeys still held.
//!
//! That refusal used to be fatal. `setup()` called `on_shortcuts([..])?`, which
//! stops at the FIRST failing shortcut — so one taken hotkey both aborted the
//! remaining registrations and propagated out of `setup()`, where Tauri turns
//! it into `Failed to setup app: HotKey already registered: HotKey { mods:
//! CONTROL, key: KeyK }` and panics. The window was already on screen by then:
//! the app appeared, showed "Starting the local runtime…", and vanished with no
//! error UI.
//!
//! A shortcut is a convenience; the app is not. So each shortcut is registered
//! on its own, every failure is a WARN naming the shortcut and the OS error,
//! and the outcome is recorded in [`GlobalShortcutStatus`] so a surface can
//! tell the user which keys are dead this session.
//!
//! ## Reading the status from the UI
//!
//! `global_shortcut_status` (registered in `lib.rs`) returns the recorded
//! outcome. Nothing renders it yet — deliberately, because inventing that
//! surface is a separate change: it needs a wrapper in `src/lib/tauri.ts`, a
//! block in `src/components/memory/settings/sections/DiagnosticsSection.tsx`
//! (Settings → Diagnostics is where the app already explains wiring that is
//! present but not working), and copy in all three locales in
//! `src/i18n/resources.ts`. Until then the WARN in the app log is the record.

use serde::{Deserialize, Serialize};
use tauri_plugin_global_shortcut::Shortcut;

/// Which of the app's shortcuts a spec is. An id rather than a comparison
/// against the parsed `Shortcut`, so a shortcut that failed to register (and
/// therefore has no parsed value to compare against) is still nameable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShortcutId {
    /// Summon the main window and focus the search field.
    ToggleSearch,
    /// Show/hide the main window.
    ShowMemory,
    /// Show/hide the quick-capture popup.
    QuickCapture,
}

impl ShortcutId {
    /// The stable wire name. Matches the serde representation; used in logs so
    /// a log line and a status payload name the same thing the same way.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToggleSearch => "toggle-search",
            Self::ShowMemory => "show-memory",
            Self::QuickCapture => "quick-capture",
        }
    }
}

impl std::fmt::Display for ShortcutId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One shortcut the app wants: what it is for, and the accelerator to ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortcutSpec {
    pub id: ShortcutId,
    /// The accelerator exactly as Tauri parses it.
    pub accelerator: &'static str,
}

/// Every global shortcut the app registers, in registration order.
///
/// The accelerators are load-bearing history: `toggle-spotlight` and
/// `show-memory` are still the event names the frontend listens for even though
/// spotlight mode is retired (see `src/App.tsx`), so changing these changes what
/// users' fingers already know.
pub const SHORTCUTS: [ShortcutSpec; 3] = [
    ShortcutSpec {
        id: ShortcutId::ToggleSearch,
        accelerator: "CmdOrCtrl+K",
    },
    ShortcutSpec {
        id: ShortcutId::ShowMemory,
        accelerator: "CmdOrCtrl+Shift+K",
    },
    ShortcutSpec {
        id: ShortcutId::QuickCapture,
        accelerator: "CmdOrCtrl+Shift+N",
    },
];

/// A shortcut the app holds this session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveShortcut {
    pub id: ShortcutId,
    pub accelerator: String,
}

/// A shortcut the app asked for and did not get, with the reason. `error` is
/// the OS/plugin message verbatim (`HotKey already registered: ...`) — a
/// summarised version would lose the only detail that distinguishes "someone
/// else holds it" from "this accelerator is not parseable".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnavailableShortcut {
    pub id: ShortcutId,
    pub accelerator: String,
    pub error: String,
}

/// What the app got when it asked the OS for its hotkeys. Recorded at startup
/// and never updated: registration is attempted once, in `setup()`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlobalShortcutStatus {
    pub active: Vec<ActiveShortcut>,
    pub unavailable: Vec<UnavailableShortcut>,
}

impl GlobalShortcutStatus {
    /// True when every shortcut the app asked for is live. False the moment one
    /// is not — including when nothing was asked for, which is not a state this
    /// app reaches but is not "all registered" either.
    pub fn all_registered(&self) -> bool {
        !self.active.is_empty() && self.unavailable.is_empty()
    }
}

/// Parse a spec's accelerator. Separated from registration so a malformed
/// accelerator is the same kind of recoverable failure as a taken one, rather
/// than the `unwrap()` panic it used to be.
pub fn parse(spec: &ShortcutSpec) -> Result<Shortcut, String> {
    spec.accelerator
        .parse::<Shortcut>()
        .map_err(|e| e.to_string())
}

/// The WARN line for a shortcut the app could not register. Explicit by
/// construction: it names the shortcut's id, the accelerator the user would
/// press, the fact that the key is dead for this session, and the OS error.
pub fn registration_warning(spec: &ShortcutSpec, error: &str) -> String {
    format!(
        "[shortcuts] global shortcut {} ({}) could not be registered and is unavailable this session; \
         another application or another Wenlan instance is likely holding it: {error}",
        spec.id, spec.accelerator
    )
}

/// Split per-shortcut registration outcomes into what the app holds and what it
/// lost. Order follows [`SHORTCUTS`] so both lists read the same way every run.
///
/// Takes outcomes rather than doing the registering, so the partition is
/// testable without a Tauri app or a real hotkey manager.
pub fn partition_registrations<I>(outcomes: I) -> GlobalShortcutStatus
where
    I: IntoIterator<Item = (ShortcutSpec, Result<(), String>)>,
{
    let mut status = GlobalShortcutStatus::default();
    for (spec, outcome) in outcomes {
        match outcome {
            Ok(()) => status.active.push(ActiveShortcut {
                id: spec.id,
                accelerator: spec.accelerator.to_string(),
            }),
            Err(error) => status.unavailable.push(UnavailableShortcut {
                id: spec.id,
                accelerator: spec.accelerator.to_string(),
                error,
            }),
        }
    }
    status
}

/// Which global shortcuts this session holds, and which it could not get.
///
/// Present so a taken hotkey is answerable rather than only logged; see the
/// module docs for the UI wiring this is waiting for.
#[tauri::command]
pub fn global_shortcut_status(
    status: tauri::State<'_, GlobalShortcutStatus>,
) -> GlobalShortcutStatus {
    status.inner().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: ShortcutId) -> ShortcutSpec {
        SHORTCUTS
            .iter()
            .copied()
            .find(|s| s.id == id)
            .expect("every id has a spec")
    }

    #[test]
    fn every_declared_shortcut_parses() {
        // The old code `unwrap()`ed these three parses in `setup()`, so a typo
        // in an accelerator was a startup panic. This is that guard, moved to
        // where it costs nothing.
        for spec in SHORTCUTS {
            assert!(
                parse(&spec).is_ok(),
                "{} accelerator {:?} does not parse",
                spec.id,
                spec.accelerator
            );
        }
    }

    #[test]
    fn shortcut_list_is_the_three_bindings_the_frontend_listens_for() {
        assert_eq!(
            SHORTCUTS.map(|s| (s.id, s.accelerator)),
            [
                (ShortcutId::ToggleSearch, "CmdOrCtrl+K"),
                (ShortcutId::ShowMemory, "CmdOrCtrl+Shift+K"),
                (ShortcutId::QuickCapture, "CmdOrCtrl+Shift+N"),
            ]
        );
    }

    #[test]
    fn shortcut_ids_are_distinct() {
        let mut ids: Vec<&str> = SHORTCUTS.iter().map(|s| s.id.as_str()).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "two specs share an id");
    }

    #[test]
    fn a_taken_shortcut_does_not_cost_the_others() {
        // The regression in one assertion: `Ctrl+K` refused, the other two
        // still held. `on_shortcuts` used to abort the whole batch instead.
        let status = partition_registrations([
            (
                spec(ShortcutId::ToggleSearch),
                Err("HotKey already registered: HotKey { mods: CONTROL, key: KeyK }".to_string()),
            ),
            (spec(ShortcutId::ShowMemory), Ok(())),
            (spec(ShortcutId::QuickCapture), Ok(())),
        ]);

        assert_eq!(
            status.active,
            vec![
                ActiveShortcut {
                    id: ShortcutId::ShowMemory,
                    accelerator: "CmdOrCtrl+Shift+K".to_string(),
                },
                ActiveShortcut {
                    id: ShortcutId::QuickCapture,
                    accelerator: "CmdOrCtrl+Shift+N".to_string(),
                },
            ]
        );
        assert_eq!(
            status.unavailable,
            vec![UnavailableShortcut {
                id: ShortcutId::ToggleSearch,
                accelerator: "CmdOrCtrl+K".to_string(),
                error: "HotKey already registered: HotKey { mods: CONTROL, key: KeyK }".to_string(),
            }]
        );
        assert!(!status.all_registered());
    }

    #[test]
    fn every_failure_keeps_its_own_error() {
        let status = partition_registrations([
            (
                spec(ShortcutId::ToggleSearch),
                Err("taken by A".to_string()),
            ),
            (spec(ShortcutId::ShowMemory), Err("taken by B".to_string())),
            (spec(ShortcutId::QuickCapture), Ok(())),
        ]);

        assert_eq!(
            status
                .unavailable
                .iter()
                .map(|u| u.error.as_str())
                .collect::<Vec<_>>(),
            vec!["taken by A", "taken by B"],
        );
        assert_eq!(status.active.len(), 1);
    }

    #[test]
    fn all_registered_only_when_nothing_failed() {
        let all_ok = partition_registrations(SHORTCUTS.map(|spec| (spec, Ok(()))));
        assert!(all_ok.all_registered());
        assert!(all_ok.unavailable.is_empty());

        // Nothing asked for is not the same claim as nothing refused.
        assert!(!GlobalShortcutStatus::default().all_registered());
    }

    #[test]
    fn warning_names_the_shortcut_the_key_and_the_error() {
        let warning = registration_warning(
            &spec(ShortcutId::ToggleSearch),
            "HotKey already registered: HotKey { mods: CONTROL, key: KeyK }",
        );

        assert!(warning.contains("toggle-search"), "{warning}");
        assert!(warning.contains("CmdOrCtrl+K"), "{warning}");
        assert!(warning.contains("unavailable this session"), "{warning}");
        assert!(
            warning.contains("HotKey already registered: HotKey { mods: CONTROL, key: KeyK }"),
            "{warning}"
        );
    }

    #[test]
    fn status_serializes_with_the_stable_kebab_ids() {
        let status =
            partition_registrations([(spec(ShortcutId::QuickCapture), Err("nope".to_string()))]);
        let json = serde_json::to_value(&status).expect("status serializes");

        assert_eq!(json["unavailable"][0]["id"], "quick-capture");
        assert_eq!(json["unavailable"][0]["accelerator"], "CmdOrCtrl+Shift+N");
        assert_eq!(json["unavailable"][0]["error"], "nope");
        assert_eq!(json["active"].as_array().map(Vec::len), Some(0));
    }
}
