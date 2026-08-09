// SPDX-License-Identifier: AGPL-3.0-only
//! System information detection for the Tauri app.
//! Inlined from origin-core::system_info in Phase 5-D PR2 to remove the dep.

use wenlan_types::system_info::SystemInfo;

/// Detect system capabilities and hardware information.
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

    // The app defers model recommendation to the daemon; return empty string here.
    let recommended_builtin = String::new();

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
