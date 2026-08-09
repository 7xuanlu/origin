// SPDX-License-Identifier: AGPL-3.0-only
use std::path::PathBuf;

pub fn app_data_dir() -> PathBuf {
    if let Some(custom) = std::env::var_os("WENLAN_DATA_DIR") {
        log::info!("[identity] using WENLAN_DATA_DIR for app data");
        return PathBuf::from(custom);
    }
    if let Some(custom) = std::env::var_os("ORIGIN_DATA_DIR") {
        log::info!("[identity] using legacy ORIGIN_DATA_DIR for app data");
        return PathBuf::from(custom);
    }
    let current = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("wenlan");
    let legacy = legacy_app_data_dir();
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
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("origin")
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
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("origin-mcp")
}

#[allow(dead_code)]
pub fn mcp_config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("wenlan-mcp")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    struct EnvGuard {
        home: Option<std::ffi::OsString>,
        wenlan: Option<std::ffi::OsString>,
        origin: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn capture() -> Self {
            Self {
                home: std::env::var_os("HOME"),
                wenlan: std::env::var_os("WENLAN_DATA_DIR"),
                origin: std::env::var_os("ORIGIN_DATA_DIR"),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
            }
            match &self.wenlan {
                Some(value) => std::env::set_var("WENLAN_DATA_DIR", value),
                None => std::env::remove_var("WENLAN_DATA_DIR"),
            }
            match &self.origin {
                Some(value) => std::env::set_var("ORIGIN_DATA_DIR", value),
                None => std::env::remove_var("ORIGIN_DATA_DIR"),
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_prefers_wenlan_env() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        std::env::set_var("WENLAN_DATA_DIR", "/tmp/wenlan-app-test");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");
        assert_eq!(app_data_dir(), PathBuf::from("/tmp/wenlan-app-test"));
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_falls_back_to_origin_env() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");
        assert_eq!(app_data_dir(), PathBuf::from("/tmp/origin-app-test"));
    }

    #[test]
    #[serial_test::serial]
    fn sidecar_env_exports_selected_app_data_dir_as_wenlan_data_dir() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::set_var("ORIGIN_DATA_DIR", "/tmp/origin-app-test");

        let (key, value) = sidecar_data_dir_env();

        assert_eq!(key, "WENLAN_DATA_DIR");
        assert_eq!(value, PathBuf::from("/tmp/origin-app-test"));
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_uses_legacy_default_when_current_absent_and_legacy_has_config() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        let legacy = dirs::data_local_dir().unwrap().join("origin");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("config.json"), "{}").unwrap();
        assert_eq!(app_data_dir(), legacy);
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_uses_legacy_default_when_current_empty_and_legacy_has_config() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        let current = dirs::data_local_dir().unwrap().join("wenlan");
        let legacy = dirs::data_local_dir().unwrap().join("origin");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("config.json"), "{}").unwrap();
        assert_eq!(app_data_dir(), legacy);
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_uses_legacy_default_when_current_empty_and_legacy_has_activities() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        let current = dirs::data_local_dir().unwrap().join("wenlan");
        let legacy = dirs::data_local_dir().unwrap().join("origin");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("activities.json"), "[]").unwrap();
        assert_eq!(app_data_dir(), legacy);
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_uses_wenlan_default_when_current_has_app_state() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        let current = dirs::data_local_dir().unwrap().join("wenlan");
        let legacy = dirs::data_local_dir().unwrap().join("origin");
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join("config.json"), "{}").unwrap();
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("config.json"), "{}").unwrap();
        assert_eq!(app_data_dir(), current);
    }

    #[test]
    #[serial_test::serial]
    fn app_data_dir_uses_wenlan_default_when_neither_exists() {
        let _guard = env_lock();
        let _env = EnvGuard::capture();
        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HOME", tmp.path());
        std::env::remove_var("WENLAN_DATA_DIR");
        std::env::remove_var("ORIGIN_DATA_DIR");
        assert_eq!(
            app_data_dir(),
            dirs::data_local_dir().unwrap().join("wenlan")
        );
    }
}

#[cfg(test)]
mod mcp_tests {
    use super::*;
    use std::ffi::OsString;

    struct HomeGuard {
        home: Option<OsString>,
    }

    impl HomeGuard {
        fn set(path: &std::path::Path) -> Self {
            let home = std::env::var_os("HOME");
            std::env::set_var("HOME", path);
            Self { home }
        }
    }

    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match &self.home {
                Some(value) => std::env::set_var("HOME", value),
                None => std::env::remove_var("HOME"),
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
}
