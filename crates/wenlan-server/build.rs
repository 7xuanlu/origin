// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("cargo:rerun-if-changed=src/macos_host_activity.m");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let mut build = cc::Build::new();
    build.file("src/macos_host_activity.m").flag("-fobjc-arc");
    let objects = build.compile_intermediates();

    // Xcode's BSD ar rejects the GNU `D` flag that cc probes before falling
    // back, which turns every otherwise-clean build into a Cargo warning. Use
    // cc's target-aware archiver/ranlib commands with Apple's deterministic
    // ZERO_AR_DATE contract, then publish normal native-library metadata so
    // downstream crates linking wenlan-server also receive this bridge.
    let out_dir = std::path::PathBuf::from(
        std::env::var_os("OUT_DIR").expect("Cargo must provide OUT_DIR to build scripts"),
    );
    let archive = out_dir.join("libwenlan_macos_host_activity.a");
    if archive.exists() {
        std::fs::remove_file(&archive).expect("failed to replace macOS host-activity archive");
    }
    let archive_status = build
        .get_archiver()
        .env("ZERO_AR_DATE", "1")
        .arg("cq")
        .arg(&archive)
        .args(&objects)
        .status()
        .expect("failed to run the macOS host-activity archiver");
    assert!(
        archive_status.success(),
        "macOS host-activity archiver failed"
    );
    let ranlib_status = build
        .get_ranlib()
        .env("ZERO_AR_DATE", "1")
        .arg(&archive)
        .status()
        .expect("failed to index the macOS host-activity archive");
    assert!(ranlib_status.success(), "macOS host-activity ranlib failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=wenlan_macos_host_activity");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
}
