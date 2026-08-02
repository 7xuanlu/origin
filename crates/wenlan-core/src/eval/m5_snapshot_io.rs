// SPDX-License-Identifier: Apache-2.0
//! Capability-anchored I/O for the privacy-safe M5 snapshot exporter.
//!
//! The lexical output parent is resolved exactly once during preparation. All
//! temporary-file and publication operations after that are relative to the
//! held directory capability, never the lexical path. A lexical-parent handle
//! is retained only to fail visibly if the user-facing path is retargeted
//! before the first write.

use anyhow::{bail, Context, Result};
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};
use same_file::Handle;
use std::ffi::{OsStr, OsString};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub struct PreparedM5Snapshot {
    parent: Dir,
    parent_identity: Handle,
    lexical_parent: PathBuf,
    target_name: OsString,
}

/// Prepare one snapshot publication against a stable output-parent identity.
/// No output file or temporary file is created by this step.
pub fn prepare_m5_snapshot(output: &Path) -> Result<PreparedM5Snapshot> {
    let target_name = output
        .file_name()
        .filter(|name| !name.is_empty())
        .context("--output must include a file name")?
        .to_os_string();
    let valid_json_extension = target_name
        .to_str()
        .and_then(|name| Path::new(name).extension())
        .and_then(OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"));
    if !valid_json_extension {
        bail!("--output extension must be .json (ASCII case-insensitive)");
    }
    let lexical_parent = nonempty_parent(output);
    let parent = Dir::open_ambient_dir(&lexical_parent, ambient_authority())
        .context("open output-parent capability")?;
    let parent_identity = handle_for_dir(&parent).context("hold output-parent identity")?;

    let prepared = PreparedM5Snapshot {
        parent,
        parent_identity,
        lexical_parent,
        target_name,
    };
    prepared.ensure_lexical_parent_unchanged()?;
    prepared.ensure_target_absent()?;
    Ok(prepared)
}

impl PreparedM5Snapshot {
    /// Write and publish exact bytes relative to the prepared directory
    /// capability. A retargeted lexical parent is refused before temp creation.
    pub fn write(self, bytes: &[u8]) -> Result<()> {
        self.ensure_lexical_parent_unchanged()?;
        self.ensure_target_absent()?;

        let (temp_name, mut temp_file) = self.create_temp()?;
        let publication = (|| -> Result<()> {
            temp_file
                .write_all(bytes)
                .context("write capability-relative snapshot temp")?;
            temp_file
                .sync_all()
                .context("sync capability-relative snapshot temp")?;
            drop(temp_file);

            self.ensure_target_absent()?;
            self.parent
                .hard_link(&temp_name, &self.parent, &self.target_name)
                .context("output exists; choose a new --output path")?;
            self.parent
                .remove_file(&temp_name)
                .context("remove published snapshot temp link")?;
            Ok(())
        })();

        if let Err(error) = publication {
            match self.parent.remove_file(&temp_name) {
                Ok(()) => Err(error),
                Err(cleanup) if cleanup.kind() == ErrorKind::NotFound => Err(error),
                Err(cleanup) => Err(error).context(format!(
                    "also failed to clean capability-relative temp: {cleanup}"
                )),
            }
        } else {
            Ok(())
        }
    }

    fn ensure_lexical_parent_unchanged(&self) -> Result<()> {
        let current = Dir::open_ambient_dir(&self.lexical_parent, ambient_authority())
            .context("reopen lexical output parent for identity check")?;
        let current_identity =
            handle_for_dir(&current).context("read current lexical output-parent identity")?;
        if current_identity != self.parent_identity {
            bail!("lexical output parent changed after preparation");
        }
        Ok(())
    }

    fn ensure_target_absent(&self) -> Result<()> {
        match self.parent.symlink_metadata(&self.target_name) {
            Ok(_) => bail!("output exists; choose a new --output path"),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).context("inspect capability-relative output target"),
        }
    }

    fn create_temp(&self) -> Result<(OsString, cap_std::fs::File)> {
        for _ in 0..128 {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let name = OsString::from(format!(
                ".m5-snapshot-{}-{sequence}.tmp",
                std::process::id()
            ));
            let mut options = OpenOptions::new();
            options
                .write(true)
                .create_new(true)
                .follow(FollowSymlinks::No);
            match self.parent.open_with(&name, &options) {
                Ok(file) => return Ok((name, file)),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).context("create capability-relative snapshot temp")
                }
            }
        }
        bail!("could not allocate a unique capability-relative snapshot temp")
    }
}

fn handle_for_dir(dir: &Dir) -> std::io::Result<Handle> {
    Handle::from_file(dir.try_clone()?.into_std_file())
}

fn nonempty_parent(path: &Path) -> PathBuf {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}
