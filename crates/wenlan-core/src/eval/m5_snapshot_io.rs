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
    db_identity: Handle,
    db_name: OsString,
    shares_db_parent: bool,
    lexical_parent: PathBuf,
    target_name: OsString,
    overwrite: bool,
}

/// Prepare one snapshot publication against stable DB and output-parent file
/// identities. No output file or temporary file is created by this step.
pub fn prepare_m5_snapshot(
    db: &Path,
    output: &Path,
    overwrite: bool,
) -> Result<PreparedM5Snapshot> {
    let db_file = std::fs::File::open(db).context("open snapshot database identity")?;
    if !db_file
        .metadata()
        .context("inspect snapshot database identity")?
        .is_file()
    {
        bail!("--db must name a regular file");
    }
    let db_identity = Handle::from_file(db_file).context("hold snapshot database identity")?;
    let db_name = db
        .file_name()
        .filter(|name| !name.is_empty())
        .context("--db must include a file name")?
        .to_os_string();
    let db_parent = Dir::open_ambient_dir(nonempty_parent(db), ambient_authority())
        .context("open database-parent capability")?;
    let db_parent_identity = handle_for_dir(&db_parent).context("hold database-parent identity")?;

    let db_via_parent = db_parent
        .open(&db_name)
        .context("open database through held parent capability")?;
    let db_via_parent_identity =
        Handle::from_file(db_via_parent.into_std()).context("verify database-parent identity")?;
    if db_via_parent_identity != db_identity {
        bail!("database parent changed during snapshot preparation");
    }

    let target_name = output
        .file_name()
        .filter(|name| !name.is_empty())
        .context("--output must include a file name")?
        .to_os_string();
    let lexical_parent = nonempty_parent(output);
    let parent = Dir::open_ambient_dir(&lexical_parent, ambient_authority())
        .context("open output-parent capability")?;
    let parent_identity = handle_for_dir(&parent).context("hold output-parent identity")?;
    let shares_db_parent = parent_identity == db_parent_identity;

    let prepared = PreparedM5Snapshot {
        parent,
        parent_identity,
        db_identity,
        db_name,
        shares_db_parent,
        lexical_parent,
        target_name,
        overwrite,
    };
    prepared.ensure_lexical_parent_unchanged()?;
    prepared.validate_existing_target()?;
    Ok(prepared)
}

impl PreparedM5Snapshot {
    /// Write and publish exact bytes relative to the prepared directory
    /// capability. A retargeted lexical parent is refused before temp creation.
    pub fn write(self, bytes: &[u8]) -> Result<()> {
        self.ensure_lexical_parent_unchanged()?;
        self.validate_existing_target()?;

        let (temp_name, mut temp_file) = self.create_temp()?;
        let publication = (|| -> Result<()> {
            temp_file
                .write_all(bytes)
                .context("write capability-relative snapshot temp")?;
            temp_file
                .sync_all()
                .context("sync capability-relative snapshot temp")?;
            drop(temp_file);

            if self.overwrite {
                // This is the last target lookup before the capability-relative
                // rename. Symlinks, non-files, and the DB identity fail closed.
                self.validate_existing_target()?;
                self.parent
                    .rename(&temp_name, &self.parent, &self.target_name)
                    .context("capability-relative snapshot overwrite")?;
            } else {
                self.parent
                    .hard_link(&temp_name, &self.parent, &self.target_name)
                    .context("output exists; pass --overwrite to replace it")?;
                self.parent
                    .remove_file(&temp_name)
                    .context("remove published snapshot temp link")?;
            }
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

    fn validate_existing_target(&self) -> Result<()> {
        if self.shares_db_parent
            && is_sqlite_transaction_control_name(&self.db_name, &self.target_name)
        {
            bail!("--output must not name a SQLite transaction-control file for --db");
        }

        let metadata = match self.parent.symlink_metadata(&self.target_name) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error).context("inspect capability-relative output target"),
        };
        if metadata.file_type().is_symlink() {
            bail!("existing --output must not be a symlink");
        }
        if !metadata.is_file() {
            bail!("existing --output must be a regular file");
        }

        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let target = self
            .parent
            .open_with(&self.target_name, &options)
            .context("open existing output target no-follow")?;
        let target_identity =
            Handle::from_file(target.into_std()).context("hold existing output identity")?;
        if target_identity == self.db_identity {
            bail!("--output must not alias --db");
        }
        Ok(())
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

fn is_sqlite_transaction_control_name(db_name: &OsStr, target_name: &OsStr) -> bool {
    for suffix in ["-wal", "-shm", "-journal"] {
        let mut family_name = db_name.to_os_string();
        family_name.push(suffix);
        if target_name == family_name {
            return true;
        }
    }

    let mut super_journal_prefix = db_name.to_os_string();
    super_journal_prefix.push("-mj");
    target_name
        .as_encoded_bytes()
        .starts_with(super_journal_prefix.as_encoded_bytes())
}
