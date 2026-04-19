// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SystemInfo {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub has_metal: bool,
    pub has_cuda: bool,
    pub os: String,
    pub arch: String,
    pub recommended_builtin: String,
}

pub fn detect_system_info() -> SystemInfo {
    use sysinfo::System;
    let mut sys = System::new_all();
    sys.refresh_memory();

    let total_ram_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_ram_gb = sys.available_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();

    let has_metal = os == "macos";
    let has_cuda = std::process::Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    // Single source of truth — defer to the on_device_models registry.
    let recommended_builtin = crate::on_device_models::recommend_for_ram(total_ram_gb)
        .id
        .to_string();

    SystemInfo {
        total_ram_gb,
        available_ram_gb,
        has_metal,
        has_cuda,
        os,
        arch,
        recommended_builtin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_valid_data() {
        let info = detect_system_info();
        assert!(info.total_ram_gb > 0.0);
        assert!(info.available_ram_gb > 0.0);
        assert!(info.available_ram_gb <= info.total_ram_gb);
        assert!(!info.os.is_empty());
        assert!(!info.arch.is_empty());
        // The recommendation must be a valid registry id.
        assert!(crate::on_device_models::get_model(&info.recommended_builtin).is_some());
    }
}
