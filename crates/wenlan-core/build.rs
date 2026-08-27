// SPDX-License-Identifier: Apache-2.0
//! Build script for origin-core.
//!
//! Emits three compile-time env vars:
//!
//! - `WENLAN_MIGRATIONS_HASH`: 16-char hex prefix of SHA-256 over the
//!   migration-bearing source files (`src/db.rs` + every file under
//!   `src/migrations/`). Used by `ReportEnv` in the eval harness. Changes
//!   whenever a migration lands, which invalidates eval caches that were built
//!   against the old schema.
//!
//! - `WENLAN_GIT_SHA`: full 40-char git SHA of HEAD. Unset in tarball
//!   builds where `.git/` is absent (the `option_env!` call-sites handle
//!   the None case gracefully).
//!
//! - `WENLAN_VERSION_SUFFIX`: `+g<sha8>` for a local source build, empty for a
//!   release build — HEAD on a `v*` tag, or the `WENLAN_RELEASE_VERSION` env
//!   var set to exactly this crate's `CARGO_PKG_VERSION` (CI sets it on the
//!   release-please PR branch, where release binaries are built before the
//!   tag exists) — or a git-less tarball. Appended to the version by
//!   `wenlan_core::version()` so the drift nudges can tell a dev daemon from
//!   a released one. Always emitted (even empty) so `env!` resolves.

use sha2::{Digest, Sha256};
use std::path::Path;

fn main() {
    // --- migrations hash ---------------------------------------------------
    // Migrations live as inline Rust inside src/db.rs and as standalone
    // modules under src/migrations/. Hash both so the hash changes when
    // any migration source is added, edited, or removed.
    let sources: &[&str] = &["src/db.rs", "src/migrations"];
    let mut hasher = Sha256::new();
    let mut found_any = false;

    for src in sources {
        let p = Path::new(src);
        if p.is_file() {
            println!("cargo:rerun-if-changed={}", src);
            let bytes = std::fs::read(p).unwrap_or_default();
            hasher.update(src.as_bytes());
            hasher.update(b":");
            hasher.update(&bytes);
            hasher.update(b"\n");
            found_any = true;
        } else if p.is_dir() {
            println!("cargo:rerun-if-changed={}", src);
            let mut entries: Vec<_> = walkdir(p);
            entries.sort();
            for path_str in &entries {
                println!("cargo:rerun-if-changed={}", path_str);
                let bytes = std::fs::read(path_str).unwrap_or_default();
                hasher.update(path_str.as_bytes());
                hasher.update(b":");
                hasher.update(&bytes);
                hasher.update(b"\n");
                found_any = true;
            }
        }
    }

    if found_any {
        let hex = format!("{:x}", hasher.finalize());
        let short: String = hex.chars().take(16).collect();
        println!("cargo:rustc-env=WENLAN_MIGRATIONS_HASH={}", short);
    } else {
        println!("cargo:rustc-env=WENLAN_MIGRATIONS_HASH=missing");
    }

    // --- git SHA + dev version suffix --------------------------------------
    // Best-effort: silently skipped in tarball / CI checkouts without .git.
    //
    // Rerun when HEAD or the checked-out branch ref moves so the dev suffix
    // below tracks commits. The old `.git/HEAD` path was package-root-relative
    // (crates/wenlan-core/.git/HEAD — nonexistent, so it silently reran every
    // build by accident); resolve the real paths via `git rev-parse --git-path`,
    // which is worktree-safe. Committing on a branch moves the ref, not HEAD, so
    // watch the resolved ref too — that's what flips the suffix on/off.
    if let Some(path) = git_path("HEAD") {
        println!("cargo:rerun-if-changed={}", path);
    }
    if let Some(refname) = git_symbolic_head() {
        if let Some(path) = git_path(&refname) {
            println!("cargo:rerun-if-changed={}", path);
        }
    }

    let sha = git_head_sha();
    if let Some(sha) = &sha {
        println!("cargo:rustc-env=WENLAN_GIT_SHA={}", sha);
    }

    // Dev builds carry a `+g<sha8>` build-metadata suffix so the daemon↔plugin
    // and daemon↔mcp drift nudges can recognize a local source build (its
    // release-granular CARGO_PKG_VERSION is stale by construction) and stay
    // quiet. A release build — HEAD exactly on a `v*` tag, or
    // `WENLAN_RELEASE_VERSION` set to this crate's own `CARGO_PKG_VERSION` —
    // reports the bare version. Release binaries are built on the
    // release-please PR branch before the tag exists, so CI sets
    // `WENLAN_RELEASE_VERSION` there to mark the build as a release without a
    // tag to check; a marker that does not match the crate version is
    // untrusted and the dev suffix stays on. `+build` metadata is semver-legal
    // and ignored in ordering, so it never perturbs the mcp handshake
    // `compare()`. Always emit (even empty) so
    // `env!("WENLAN_VERSION_SUFFIX")` resolves at compile time.
    println!("cargo:rerun-if-env-changed=WENLAN_RELEASE_VERSION");
    let release_version = std::env::var("WENLAN_RELEASE_VERSION")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_default();
    if let Some(marker) = &release_version {
        if marker != &pkg_version {
            println!(
                "cargo:warning=WENLAN_RELEASE_VERSION={} does not match crate version {}; keeping the dev suffix",
                marker, pkg_version
            );
        }
    }
    let is_release_build =
        head_on_release_tag() || release_version.as_deref() == Some(pkg_version.as_str());
    let suffix = match (&sha, is_release_build) {
        (Some(sha), false) => format!("+g{}", sha.chars().take(8).collect::<String>()),
        _ => String::new(),
    };
    println!("cargo:rustc-env=WENLAN_VERSION_SUFFIX={}", suffix);
}

/// Run `git <args>` and return trimmed stdout, or None on any failure / empty.
fn git_output(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// Resolve a git path (worktree-safe) — e.g. the real filesystem path of HEAD.
fn git_path(p: &str) -> Option<String> {
    git_output(&["rev-parse", "--git-path", p])
}

/// The ref HEAD points at (e.g. `refs/heads/main`), or None when detached.
fn git_symbolic_head() -> Option<String> {
    git_output(&["symbolic-ref", "-q", "HEAD"])
}

fn git_head_sha() -> Option<String> {
    git_output(&["rev-parse", "HEAD"])
}

/// True when HEAD sits exactly on a `v*` version tag — i.e. a release build.
fn head_on_release_tag() -> bool {
    git_output(&[
        "describe",
        "--tags",
        "--exact-match",
        "--match",
        "v*",
        "HEAD",
    ])
    .is_some()
}

/// Recursively collect all file paths under `dir` as strings.
fn walkdir(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_file() {
            if let Some(s) = p.to_str() {
                out.push(s.to_owned());
            }
        } else if p.is_dir() {
            out.extend(walkdir(&p));
        }
    }
    out
}
