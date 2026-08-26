// SPDX-License-Identifier: AGPL-3.0-only
//! Hand the running app over to a newer bundle that was just launched.
//!
//! The single-instance plugin makes every second launch exit right after it
//! hands its argv to the running instance. When that second launch is a newer
//! Wenlan — the installer replaced the bundle, or the user dragged a new one
//! into Applications — focusing the old window is wrong: the user asked for
//! the new version and nothing told them the old one is still on screen
//! (first-run gauntlet finding F2). The running app instead quits the way
//! "Quit Wenlan" does and reopens the newer bundle once its own process is
//! gone, so the single-instance socket is free by the time the newcomer
//! claims it.

use std::path::{Path, PathBuf};

/// A newer app bundle that just tried to launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewerBundle {
    pub bundle: PathBuf,
    pub version: semver::Version,
}

/// The newer bundle behind a second launch, if that is what `argv` describes.
///
/// `argv[0]` of a bundle launched through LaunchServices is the executable
/// inside `<name>.app/Contents/MacOS/`; the bundle's `Info.plist` carries the
/// version. Anything else — a dev binary, an unreadable plist, the same or an
/// older version — yields `None` and the caller keeps the focus behavior.
pub fn newer_bundle_from_launch(argv: &[String], running: &semver::Version) -> Option<NewerBundle> {
    let bundle = bundle_root(Path::new(argv.first()?))?;
    let version = bundle_version(&bundle)?;
    (version > *running).then_some(NewerBundle { bundle, version })
}

fn bundle_root(exe: &Path) -> Option<PathBuf> {
    let macos = exe.parent()?;
    let contents = macos.parent()?;
    let bundle = contents.parent()?;
    let is_bundle = macos.file_name()? == "MacOS"
        && contents.file_name()? == "Contents"
        && bundle.extension()? == "app";
    is_bundle.then(|| bundle.to_path_buf())
}

fn bundle_version(bundle: &Path) -> Option<semver::Version> {
    let info = plist::Value::from_file(bundle.join("Contents/Info.plist")).ok()?;
    let version = info
        .as_dictionary()?
        .get("CFBundleShortVersionString")?
        .as_string()?;
    semver::Version::parse(version).ok()
}

/// Shell script that reopens the bundle passed as `$0` once process `pid`
/// has exited.
///
/// The old app cannot `open` the newcomer itself: while it is alive,
/// LaunchServices would only activate it, and its single-instance socket
/// would send the newcomer straight back to exit. A detached shell waits the
/// old process out instead.
pub fn relaunch_script(pid: u32) -> String {
    format!("while kill -0 {pid} 2>/dev/null; do sleep 0.1; done; exec open \"$0\"")
}

/// Spawn the detached helper that reopens `bundle` after this process exits.
pub fn relaunch_after_exit(bundle: &Path) {
    let spawned = std::process::Command::new("/bin/sh")
        .arg("-c")
        .arg(relaunch_script(std::process::id()))
        .arg(bundle)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    match spawned {
        Ok(_) => log::info!(
            "[handover] will reopen {} once this process exits",
            bundle.display()
        ),
        Err(e) => log::error!(
            "[handover] could not schedule the reopen of {}: {e}",
            bundle.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_bundle(version: Option<&str>) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "wenlan-handover-{}-{}",
            std::process::id(),
            version.unwrap_or("none").replace('.', "_")
        ));
        let bundle = root.join("Wenlan.app");
        let macos = bundle.join("Contents/MacOS");
        std::fs::create_dir_all(&macos).unwrap();
        let exe = macos.join("wenlan-app");
        std::fs::write(&exe, b"").unwrap();
        if let Some(version) = version {
            std::fs::write(
                bundle.join("Contents/Info.plist"),
                format!(
                    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
                     <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
                     \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
                     <plist version=\"1.0\"><dict>\
                     <key>CFBundleIdentifier</key><string>com.wenlan.desktop</string>\
                     <key>CFBundleShortVersionString</key><string>{version}</string>\
                     </dict></plist>\n"
                ),
            )
            .unwrap();
        }
        (bundle, exe)
    }

    fn argv(exe: &Path) -> Vec<String> {
        vec![exe.to_string_lossy().into_owned()]
    }

    #[test]
    fn a_newer_bundle_is_recognized_and_an_equal_or_older_one_is_not() {
        let (bundle, exe) = scratch_bundle(Some("0.18.0"));
        let newer = newer_bundle_from_launch(&argv(&exe), &semver::Version::new(0, 17, 0));
        assert_eq!(
            newer,
            Some(NewerBundle {
                bundle: bundle.clone(),
                version: semver::Version::new(0, 18, 0),
            })
        );
        assert_eq!(
            newer_bundle_from_launch(&argv(&exe), &semver::Version::new(0, 18, 0)),
            None,
            "the same version must keep the focus behavior"
        );
        assert_eq!(
            newer_bundle_from_launch(&argv(&exe), &semver::Version::new(0, 19, 0)),
            None,
            "an older newcomer must never take over"
        );
        std::fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
    }

    #[test]
    fn anything_that_is_not_a_readable_app_bundle_keeps_the_focus_behavior() {
        let running = semver::Version::new(0, 17, 0);
        assert_eq!(newer_bundle_from_launch(&[], &running), None);
        assert_eq!(
            newer_bundle_from_launch(&["target/debug/wenlan-app".to_string()], &running),
            None,
            "a dev binary outside a bundle has no version to compare"
        );
        let (bundle, exe) = scratch_bundle(None);
        assert_eq!(
            newer_bundle_from_launch(&argv(&exe), &running),
            None,
            "a bundle without Info.plist has no version to compare"
        );
        std::fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
        let (bundle, exe) = scratch_bundle(Some("not-a-version"));
        assert_eq!(newer_bundle_from_launch(&argv(&exe), &running), None);
        std::fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
    }

    #[test]
    fn the_relaunch_script_waits_for_the_old_process_then_opens_the_bundle() {
        let script = relaunch_script(4242);
        assert_eq!(
            script,
            "while kill -0 4242 2>/dev/null; do sleep 0.1; done; exec open \"$0\""
        );
    }
}
