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
//! A shortcut is a convenience; the app is not. So [`register_all`] attempts
//! every shortcut on its own, logs a WARN for each refusal, and returns a
//! [`GlobalShortcutStatus`] a surface can read to say which keys are dead this
//! session. It takes the registrar as a closure, so the "keep going" rule is
//! provable without a Tauri app: see the tests at the bottom of this file.
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

use serde::Serialize;
use tauri_plugin_global_shortcut::Shortcut;

/// Which of the app's shortcuts a spec is. An id rather than a comparison
/// against the parsed `Shortcut`, so a shortcut that failed to register (and
/// therefore has no parsed value to compare against) is still nameable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
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
#[derive(Debug, Clone, Copy)]
pub struct ShortcutSpec {
    pub id: ShortcutId,
    /// The accelerator exactly as Tauri parses it.
    pub accelerator: &'static str,
}

/// Every global shortcut the app registers, in registration order.
///
/// The accelerators are load-bearing: changing one changes what users' fingers
/// already know. They are not the same names as the Tauri events the handlers
/// emit in `lib.rs` — `toggle-spotlight` and `show-memory` are event names,
/// kept from the retired spotlight mode because `src/App.tsx` still listens for
/// them.
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ActiveShortcut {
    pub id: ShortcutId,
    pub accelerator: String,
}

/// A shortcut the app asked for and did not get, with the reason. `error` is
/// the OS/plugin message verbatim (`HotKey already registered: ...`) — a
/// summarised version would lose the only detail that distinguishes "someone
/// else holds it" from "this accelerator is not parseable".
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnavailableShortcut {
    pub id: ShortcutId,
    pub accelerator: String,
    pub error: String,
}

/// What the app got when it asked the OS for its hotkeys. Recorded at startup
/// and never updated: registration is attempted once, in `setup()`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GlobalShortcutStatus {
    pub active: Vec<ActiveShortcut>,
    pub unavailable: Vec<UnavailableShortcut>,
}

/// Parse a spec's accelerator. Separated from registration so a malformed
/// accelerator is the same kind of recoverable failure as a taken one, rather
/// than the `unwrap()` panic it used to be.
pub fn parse(spec: ShortcutSpec) -> Result<Shortcut, String> {
    spec.accelerator
        .parse::<Shortcut>()
        .map_err(|e| e.to_string())
}

/// The WARN line for a shortcut the app could not register. Explicit by
/// construction: it names the shortcut's id, the accelerator the user would
/// press, the fact that the key is dead for this session, and the error.
///
/// It offers no cause of its own. The same path carries both "another process
/// holds this hotkey" and "this accelerator does not parse", and a guess
/// appended to the second one would be a fabrication — the verbatim error
/// already says which happened.
pub fn registration_warning(spec: ShortcutSpec, error: &str) -> String {
    format!(
        "[shortcuts] global shortcut {} ({}) could not be registered and is unavailable this session: {error}",
        spec.id, spec.accelerator
    )
}

/// Ask for every shortcut in `specs`, one at a time, and report what came back.
///
/// `register` is the caller's registrar — in the app it parses the accelerator
/// and calls `on_shortcut`; in tests it is a closure. Injecting it is the point:
/// the rule this function exists to hold is "a refusal never skips the shortcuts
/// after it", and that rule is only observable from the sequence of calls made,
/// not from a status computed out of outcomes someone else already gathered.
///
/// So: no early return, no `?`, no `break`. Every spec is attempted, every
/// failure is logged, and the caller gets a status instead of an error.
pub fn register_all(
    specs: &[ShortcutSpec],
    mut register: impl FnMut(ShortcutSpec) -> Result<(), String>,
) -> GlobalShortcutStatus {
    let mut status = GlobalShortcutStatus::default();
    for &spec in specs {
        match register(spec) {
            Ok(()) => status.active.push(ActiveShortcut {
                id: spec.id,
                accelerator: spec.accelerator.to_string(),
            }),
            Err(error) => {
                // Explicit, never swallowed. Logged here rather than at the
                // call site so no caller can register without reporting.
                log::warn!("{}", registration_warning(spec, &error));
                status.unavailable.push(UnavailableShortcut {
                    id: spec.id,
                    accelerator: spec.accelerator.to_string(),
                    error,
                });
            }
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

    const TAKEN: &str = "HotKey already registered: HotKey { mods: CONTROL, key: KeyK }";

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
                parse(spec).is_ok(),
                "{} accelerator {:?} does not parse",
                spec.id,
                spec.accelerator
            );
        }
    }

    #[test]
    fn shortcut_list_is_the_three_bindings_the_app_registers() {
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
        // The regression itself. The FIRST shortcut is refused — the exact
        // shape of the crash, where `Ctrl+K` was already held — and the test
        // watches the registrar, not just the result: `on_shortcuts` used to
        // abandon the batch on the first refusal, so the other two were never
        // even attempted. An early `break` here fails this test.
        let mut attempted: Vec<ShortcutId> = Vec::new();

        let status = register_all(&SHORTCUTS, |spec| {
            attempted.push(spec.id);
            if spec.id == ShortcutId::ToggleSearch {
                Err(TAKEN.to_string())
            } else {
                Ok(())
            }
        });

        assert_eq!(
            attempted,
            vec![
                ShortcutId::ToggleSearch,
                ShortcutId::ShowMemory,
                ShortcutId::QuickCapture,
            ],
            "every shortcut must be attempted, including the ones after a refusal"
        );
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
                error: TAKEN.to_string(),
            }]
        );
    }

    #[test]
    fn every_failure_keeps_its_own_error() {
        let status = register_all(&SHORTCUTS, |spec| match spec.id {
            ShortcutId::ToggleSearch => Err("taken by A".to_string()),
            ShortcutId::ShowMemory => Err("taken by B".to_string()),
            ShortcutId::QuickCapture => Ok(()),
        });

        assert_eq!(
            status
                .unavailable
                .iter()
                .map(|u| (u.id, u.error.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (ShortcutId::ToggleSearch, "taken by A"),
                (ShortcutId::ShowMemory, "taken by B"),
            ],
        );
        assert_eq!(status.active.len(), 1);
    }

    #[test]
    fn nothing_is_unavailable_when_every_registration_succeeds() {
        let status = register_all(&SHORTCUTS, |_| Ok(()));

        assert_eq!(status.active.len(), SHORTCUTS.len());
        assert!(status.unavailable.is_empty());
    }

    #[test]
    fn warning_names_the_shortcut_the_key_and_the_error() {
        let warning = registration_warning(spec(ShortcutId::ToggleSearch), TAKEN);

        assert!(warning.contains("toggle-search"), "{warning}");
        assert!(warning.contains("CmdOrCtrl+K"), "{warning}");
        assert!(warning.contains("unavailable this session"), "{warning}");
        assert!(warning.contains(TAKEN), "{warning}");
        // No invented cause: the same line carries parse failures too, where
        // "something else is holding it" would be false.
        assert!(!warning.contains("likely"), "{warning}");
    }

    #[test]
    fn status_serializes_with_the_stable_kebab_ids() {
        let status = register_all(&[spec(ShortcutId::QuickCapture)], |_| {
            Err("nope".to_string())
        });
        let json = serde_json::to_value(&status).expect("status serializes");

        assert_eq!(json["unavailable"][0]["id"], "quick-capture");
        assert_eq!(json["unavailable"][0]["accelerator"], "CmdOrCtrl+Shift+N");
        assert_eq!(json["unavailable"][0]["error"], "nope");
        assert_eq!(json["active"].as_array().map(Vec::len), Some(0));
    }
}
