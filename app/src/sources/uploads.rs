// SPDX-License-Identifier: AGPL-3.0-only
//! Managed home for loose uploaded files (§2). The daemon indexes files in
//! place and stores no blobs, so the ONLY copies the app makes are loose
//! single-file uploads staged here for the daemon's 30s scheduler.

use std::fs;
use std::path::{Path, PathBuf};

/// `~/.wenlan/sources` — aligned under the daemon's knowledge home, next to
/// `pages/`.
/// ponytail: hardcoded to `~/.wenlan`; the daemon's knowledge_path can in
/// principle be customized but is not exposed via any API today. Verified
/// safe: this subdir is not a reserved ingest root and indexes normally.
pub fn sources_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".wenlan/sources")
}

/// Atomically place `src` into `sources_dir`: copy to a temp name on the same
/// filesystem, then `rename` into place, so the daemon's 30s scheduler never
/// sees a partially written file. Ensures the dir and its self-gitignore exist.
pub fn place_upload_file(sources_dir: &Path, src: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(sources_dir)?;
    let gitignore = sources_dir.join(".gitignore");
    if !gitignore.exists() {
        fs::write(&gitignore, "*\n")?;
    }
    let name = src.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "source has no file name")
    })?;
    let dest = sources_dir.join(name);
    let tmp = sources_dir.join(format!(".{}.tmp", name.to_string_lossy()));
    fs::copy(src, &tmp)?;
    fs::rename(&tmp, &dest)?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn place_upload_is_atomic_and_gitignored() {
        let tmp = tempfile::tempdir().unwrap();
        let sources = tmp.path().join("sources");
        let src = tmp.path().join("paper.pdf");
        std::fs::write(&src, b"%PDF-1.4 body").unwrap();

        let dest = place_upload_file(&sources, &src).unwrap();

        assert_eq!(dest, sources.join("paper.pdf"));
        assert_eq!(std::fs::read(&dest).unwrap(), b"%PDF-1.4 body");
        // self-gitignore keeps blobs out of the daemon's git home
        assert_eq!(
            std::fs::read_to_string(sources.join(".gitignore")).unwrap(),
            "*\n"
        );
        // no temp file left behind
        let leftover: Vec<_> = std::fs::read_dir(&sources)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftover.is_empty(), "temp file must be renamed away");
    }
}
