// SPDX-License-Identifier: AGPL-3.0-only
use std::path::PathBuf;

/// Test-only stand-in for `dirs::data_local_dir()` — a *base*, so the
/// `wenlan` / `origin` selection in [`app_data_dir_for_base`] still runs
/// against it. Consulted only after the real `WENLAN_DATA_DIR` /
/// `ORIGIN_DATA_DIR` overrides, so it changes nothing about their precedence;
/// it replaces exactly one leg — the developer's real profile.
///
/// Set it through `test_env::isolate_app_roots`, never by hand.
#[cfg(test)]
pub(crate) const TEST_DATA_LOCAL_DIR_ENV: &str = "WENLAN_TEST_DATA_LOCAL_DIR";

/// Test-only stand-in for `dirs::home_dir()` — also a *base*, so the
/// `.config/wenlan-mcp` join below still runs against it and the tests that
/// pin that layout keep testing the layout rather than the override.
///
/// Set it through `test_env::isolate_app_roots`, never by hand.
#[cfg(test)]
pub(crate) const TEST_HOME_DIR_ENV: &str = "WENLAN_TEST_HOME_DIR";

/// Under `cfg(test)` the real profile is not reachable at all.
///
/// `HOME` is not a data-root override on Windows: `dirs` resolves the known
/// folders (`FOLDERID_LocalAppData`, `FOLDERID_Profile`) and ignores `HOME`
/// entirely, so a test that sets `HOME` to a tempdir and then clears
/// `WENLAN_DATA_DIR` writes into `%LOCALAPPDATA%\wenlan` and
/// `%USERPROFILE%\.config` for real. `set_user_opted_out(true)` CREATES
/// `auto_start_disabled.flag`, which `path_has_app_state` then counts when
/// choosing between the `wenlan` and `origin` roots — so the unit suite could
/// opt the developer out of auto-start and move which data root the installed
/// app selects.
///
/// A comment on `lifecycle::opt_out_flag_round_trip` already said all of this
/// and three neighbouring tests did it anyway, plus five in `remote_access`
/// — one of which created a `relay_id` DIRECTORY in the real profile and
/// thereby broke the other four. So this is a panic, not a comment.
#[cfg(test)]
fn refuse_real_profile(what: &str, env_key: &str) -> ! {
    panic!(
        "{what}() reached the developer's real profile from a unit test. Setting HOME is not \
         enough — Windows resolves these roots from its own known folders. Wrap the test in \
         `crate::test_env::isolate_app_roots(tmpdir)` (or set {env_key}), and keep the returned \
         guard alive for the whole test."
    );
}

/// Which OS-provided profile root could not be resolved.
///
/// Carried as a value rather than a string so the message can name the exact
/// call that failed *and* the remedy that actually applies to that root:
/// `WENLAN_DATA_DIR` moves the data root and does nothing whatsoever for the
/// home root, so offering it there would send the user after the wrong fix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileRoot {
    DataLocal,
    Home,
}

/// What the user is told when the OS cannot name one of its own profile roots.
///
/// A plain function so it can be asserted on directly. The production half of
/// the measurement below is `#[cfg(not(test))]`, so no unit test can make
/// `dirs` itself return `None`; the decision and the message it produces are
/// what a test can reach, and they are the part that was wrong.
fn missing_profile_root_message(root: ProfileRoot) -> String {
    match root {
        ProfileRoot::DataLocal => concat!(
            "Wenlan could not find out where your application data belongs: ",
            "dirs::data_local_dir() returned nothing ",
            "(Windows: FOLDERID_LocalAppData; macOS: ~/Library/Application Support; ",
            "Linux: $XDG_DATA_HOME or ~/.local/share). ",
            "Wenlan will not guess a location. Guessing means writing your memory ",
            "database, MCP configuration and uploads into whichever directory Wenlan ",
            "happened to be launched from, where they are either lost or refused for ",
            "lack of permission with nothing naming the cause. ",
            "Set WENLAN_DATA_DIR to an absolute path you can write to, then start ",
            "Wenlan again.",
        )
        .to_string(),
        ProfileRoot::Home => concat!(
            "Wenlan could not find out where your home directory is: ",
            "dirs::home_dir() returned nothing ",
            "(Windows: FOLDERID_Profile; macOS and Linux: $HOME). ",
            "Wenlan will not guess a location. Guessing means writing your MCP ",
            "configuration and uploaded files into whichever directory Wenlan happened ",
            "to be launched from, where they are either lost or refused for lack of ",
            "permission with nothing naming the cause. ",
            "This root is the OS's own answer and no Wenlan setting overrides it: ",
            "repair the profile the OS reports (Windows: %USERPROFILE%; macOS and ",
            "Linux: $HOME), then start Wenlan again.",
        )
        .to_string(),
    }
}

/// Turn a *measurement* of an OS profile root into a path, or fail loudly.
///
/// `None` here does not mean "there is no such directory". It means the OS
/// could not tell us where the profile is — a failed measurement. Spending it
/// as a relative path (`.`) is writable, so the app would create `./wenlan/`
/// in whatever directory it was launched from with nothing naming the cause.
///
/// This aborts rather than returning an error because there is nothing to
/// return it to: `app_data_dir()` and `home_base()` return `PathBuf`, and every
/// caller (`config`, `activity`, `presence`, `search`, `sources::uploads`,
/// `remote_access`, `lifecycle`, `daemon_start`) is on its way to open or
/// create a file under the root it is asking for, with no recovery for "no
/// profile exists".
fn resolve_profile_root(measured: Option<PathBuf>, root: ProfileRoot) -> PathBuf {
    match measured {
        Some(path) => path,
        None => {
            let message = missing_profile_root_message(root);
            // Logged as well as panicked: a packaged desktop build has no
            // console for a panic message to land in, but it does have a log.
            log::error!("[identity] {message}");
            panic!("{message}");
        }
    }
}

/// The one place this module reads `dirs::data_local_dir()`, so the test guard
/// has a single seam to sit on instead of one per caller.
///
/// Only the *measurement* is `cfg`-split. What happens to a `None` is decided
/// once, in [`resolve_profile_root`], for both builds — so the production build
/// cannot quietly regrow a fallback that the unit suite never compiles.
fn data_local_base() -> PathBuf {
    resolve_profile_root(measured_data_local_dir(), ProfileRoot::DataLocal)
}

#[cfg(test)]
fn measured_data_local_dir() -> Option<PathBuf> {
    match std::env::var_os(TEST_DATA_LOCAL_DIR_ENV) {
        Some(base) => Some(PathBuf::from(base)),
        None => refuse_real_profile("data_local_dir", TEST_DATA_LOCAL_DIR_ENV),
    }
}

#[cfg(not(test))]
fn measured_data_local_dir() -> Option<PathBuf> {
    dirs::data_local_dir()
}

/// The one place the crate reads `dirs::home_dir()` for a directory it will
/// *write* into. Same seam, same reason. Read-only probes (client detection in
/// `mcp_config`, the Obsidian registry, the hf-hub cache) still call `dirs`
/// directly: they answer questions about the real machine and create nothing.
pub(crate) fn home_base() -> PathBuf {
    resolve_profile_root(measured_home_dir(), ProfileRoot::Home)
}

#[cfg(test)]
fn measured_home_dir() -> Option<PathBuf> {
    match std::env::var_os(TEST_HOME_DIR_ENV) {
        Some(base) => Some(PathBuf::from(base)),
        None => refuse_real_profile("home_dir", TEST_HOME_DIR_ENV),
    }
}

#[cfg(not(test))]
fn measured_home_dir() -> Option<PathBuf> {
    dirs::home_dir()
}

pub fn app_data_dir() -> PathBuf {
    if let Some(custom) = std::env::var_os("WENLAN_DATA_DIR") {
        log::info!("[identity] using WENLAN_DATA_DIR for app data");
        return PathBuf::from(custom);
    }
    if let Some(custom) = std::env::var_os("ORIGIN_DATA_DIR") {
        log::info!("[identity] using legacy ORIGIN_DATA_DIR for app data");
        return PathBuf::from(custom);
    }
    app_data_dir_for_base(&data_local_base())
}

pub(crate) fn app_data_dir_for_base(base: &std::path::Path) -> PathBuf {
    let current = base.join("wenlan");
    let legacy = base.join("origin");
    if path_has_app_state(&current) {
        return current;
    }
    if path_has_app_state(&legacy) {
        log::warn!(
            "[identity] using populated legacy Origin app data root for bridge release: {}",
            legacy.display()
        );
        return legacy;
    }
    current
}

pub fn legacy_app_data_dir() -> PathBuf {
    data_local_base().join("origin")
}

pub fn sidecar_data_dir_env() -> (&'static str, PathBuf) {
    ("WENLAN_DATA_DIR", app_data_dir())
}

fn path_has_app_state(path: &std::path::Path) -> bool {
    path.join("config.json").exists()
        || path.join("avatars").exists()
        || path.join("activities.json").exists()
        || path.join("auto_start_disabled.flag").exists()
}

#[allow(dead_code)]
pub fn legacy_mcp_config_dir() -> PathBuf {
    home_base().join(".config").join("origin-mcp")
}

#[allow(dead_code)]
pub fn mcp_config_dir() -> PathBuf {
    if let Some(state_dir) = isolated_dev_state_dir() {
        return state_dir.join("mcp-config");
    }
    home_base().join(".config").join("wenlan-mcp")
}

pub fn isolated_dev_state_dir() -> Option<PathBuf> {
    #[cfg(debug_assertions)]
    {
        std::env::var_os("WENLAN_DEV_STATE_DIR").map(PathBuf::from)
    }
    #[cfg(not(debug_assertions))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    use crate::test_env::EnvGuard;

    const IDENTITY_ENV_KEYS: &[&str] = &["HOME", "WENLAN_DATA_DIR", "ORIGIN_DATA_DIR"];

    #[test]
    #[serial_test::serial]
    fn app_data_dir_prefers_wenlan_env() {
        let _guard = env_lock();
        let _env = EnvGuard::capture(IDENTITY_ENV_KEYS);
        std::env::set_var("WENLAN_DATA_DIR", "/tmp/wenlan-app-test");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");
        assert_eq!(app_data_dir(), PathBuf::from("/tmp/wenlan-app-test"));
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_falls_back_to_origin_env() {
        let _guard = env_lock();
        let _env = EnvGuard::capture(IDENTITY_ENV_KEYS);
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");
        assert_eq!(app_data_dir(), PathBuf::from("/tmp/origin-app-test"));
    }

    #[test]
    #[serial_test::serial]
    fn sidecar_env_exports_selected_app_data_dir_as_wenlan_data_dir() {
        let _guard = env_lock();
        let _env = EnvGuard::capture(IDENTITY_ENV_KEYS);
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");

        let (key, value) = sidecar_data_dir_env();

        assert_eq!(key, "WENLAN_DATA_DIR");
        assert_eq!(value, PathBuf::from("/tmp/origin-app-test"));
    }

    /// Stage a root: `None` leaves the directory absent, `Some(entries)`
    /// creates it and then each entry — a trailing `/` makes it a directory,
    /// anything else an empty-ish file.
    fn stage_root(root: &std::path::Path, entries: Option<&[&str]>) {
        let Some(entries) = entries else { return };
        std::fs::create_dir_all(root).unwrap();
        for entry in entries {
            match entry.strip_suffix('/') {
                Some(dir) => std::fs::create_dir_all(root.join(dir)).unwrap(),
                None => std::fs::write(root.join(entry), "{}").unwrap(),
            }
        }
    }

    #[test]
    fn app_data_dir_picks_the_root_that_holds_app_state() {
        // (case, `wenlan` root, `origin` root, the root that must win)
        let cases: [(&str, Option<&[&str]>, Option<&[&str]>, &str); 6] = [
            (
                "current absent, legacy config",
                None,
                Some(&["config.json"]),
                "origin",
            ),
            (
                "current empty, legacy config",
                Some(&[]),
                Some(&["config.json"]),
                "origin",
            ),
            (
                "current empty, legacy activities",
                Some(&[]),
                Some(&["activities.json"]),
                "origin",
            ),
            (
                "current empty, legacy avatars dir",
                Some(&[]),
                Some(&["avatars/"]),
                "origin",
            ),
            (
                "both carry state — current wins",
                Some(&["config.json"]),
                Some(&["config.json"]),
                "wenlan",
            ),
            ("neither exists", None, None, "wenlan"),
        ];

        for (case, current, legacy, expected) in cases {
            let tmp = tempfile::tempdir().unwrap();
            stage_root(&tmp.path().join("wenlan"), current);
            stage_root(&tmp.path().join("origin"), legacy);
            assert_eq!(
                app_data_dir_for_base(tmp.path()),
                tmp.path().join(expected),
                "{case}"
            );
        }
    }

    // ── An unresolvable profile root ────────────────────────────────────
    //
    // `dirs` returning `None` means the OS could not tell us where the profile
    // is. That used to become `PathBuf::from(".")` — a writable relative path —
    // so a user launched from an arbitrary or protected directory got their
    // memory database, MCP config or uploads written there, or unexplained
    // permission errors, with nothing naming the cause.
    //
    // These test `resolve_profile_root`, which is where the decision now lives,
    // because the `dirs` call itself is `#[cfg(not(test))]` and no unit test can
    // make it return `None`.

    #[test]
    fn a_measured_profile_root_is_used_exactly_as_measured() {
        let measured = PathBuf::from("/tmp/measured-root");
        for root in [ProfileRoot::DataLocal, ProfileRoot::Home] {
            assert_eq!(
                resolve_profile_root(Some(measured.clone()), root),
                measured,
                "{root:?} must not rewrite a root the OS did give us"
            );
        }
    }

    #[test]
    #[should_panic(expected = "dirs::data_local_dir() returned nothing")]
    fn an_unresolvable_data_root_stops_instead_of_becoming_the_working_directory() {
        // Before the fix this returned `PathBuf::from(".")` and the test fails
        // with "did not panic" — which is the defect, stated as a test.
        let _ = resolve_profile_root(None, ProfileRoot::DataLocal);
    }

    #[test]
    #[should_panic(expected = "dirs::home_dir() returned nothing")]
    fn an_unresolvable_home_root_stops_instead_of_becoming_the_working_directory() {
        let _ = resolve_profile_root(None, ProfileRoot::Home);
    }

    #[test]
    fn the_failure_names_the_call_that_failed_and_a_remedy_that_applies_to_it() {
        let data = missing_profile_root_message(ProfileRoot::DataLocal);
        assert!(data.contains("dirs::data_local_dir()"), "{data}");
        assert!(data.contains("FOLDERID_LocalAppData"), "{data}");
        // The remedy has to be one the user can actually apply, and for this
        // root `app_data_dir()` really does consult it first.
        assert!(data.contains("WENLAN_DATA_DIR"), "{data}");

        let home = missing_profile_root_message(ProfileRoot::Home);
        assert!(home.contains("dirs::home_dir()"), "{home}");
        assert!(home.contains("FOLDERID_Profile"), "{home}");
        assert!(
            home.contains("USERPROFILE") && home.contains("$HOME"),
            "{home}"
        );
        // `home_base()` has no env override at all, so naming `WENLAN_DATA_DIR`
        // here would point the user at a setting that changes nothing.
        assert!(!home.contains("WENLAN_DATA_DIR"), "{home}");

        // Neither message may leave the reader thinking a relative location was
        // an acceptable answer.
        for message in [data, home] {
            assert!(message.contains("will not guess a location"), "{message}");
        }
    }
}

#[cfg(test)]
mod mcp_tests {
    use super::*;
    use std::ffi::OsString;

    struct HomeGuard {
        home: Option<OsString>,
        dev_state: Option<OsString>,
        /// `HOME` alone does not move `mcp_config_dir()` off the developer's
        /// real profile — on Windows `dirs::home_dir()` reads
        /// `FOLDERID_Profile` and never looks at `HOME`. The assertions below
        /// are still about the `.config/wenlan-mcp` join, because
        /// `isolate_app_roots` substitutes the *base* and leaves the join in
        /// place.
        _roots: crate::test_env::EnvGuard,
    }

    impl HomeGuard {
        fn set(path: &std::path::Path) -> Self {
            let home = std::env::var_os("HOME");
            let dev_state = std::env::var_os("WENLAN_DEV_STATE_DIR");
            let roots = crate::test_env::isolate_app_roots(path);
            std::env::set_var("HOME", path);
            std::env::remove_var("WENLAN_DEV_STATE_DIR");
            Self {
                home,
                dev_state,
                _roots: roots,
            }
        }
    }

    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match &self.dev_state {
                Some(value) => std::env::set_var("WENLAN_DEV_STATE_DIR", value),
                None => std::env::remove_var("WENLAN_DEV_STATE_DIR"),
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn mcp_config_dir_uses_wenlan_mcp() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());
        assert!(mcp_config_dir().ends_with(".config/wenlan-mcp"));
    }

    #[test]
    #[serial_test::serial]
    fn legacy_mcp_config_dir_uses_origin_mcp() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());
        assert!(legacy_mcp_config_dir().ends_with(".config/origin-mcp"));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[serial_test::serial]
    fn dev_mcp_config_dir_is_scoped_to_the_worktree_state() {
        let tmp = tempfile::tempdir().unwrap();
        let _home = HomeGuard::set(tmp.path());
        let state = tmp.path().join("worktree-state");
        std::env::set_var("WENLAN_DEV_STATE_DIR", &state);

        assert_eq!(mcp_config_dir(), state.join("mcp-config"));
    }
}
