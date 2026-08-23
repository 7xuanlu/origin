// SPDX-License-Identifier: Apache-2.0
//! Output format helpers — JSON, table, quiet.
//!
//! `OutputFormat::Auto` resolves based on stdout TTY detection:
//! - TTY (humans)  -> `Table` (pretty)
//! - Pipe (scripts) -> `Json` (machine-readable)
//!
//! Uses `std::io::IsTerminal` (stable since Rust 1.70) so we don't pull in
//! the external `is-terminal` crate.

use anyhow::Result;
use serde::Serialize;
use std::io::IsTerminal;

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Pick automatically based on stdout TTY: terminal → Table, pipe → Json.
    Auto,
    Json,
    Table,
}

/// A format that has already been resolved from `OutputFormat::Auto`.
/// Commands take this, never `OutputFormat`, so the compiler proves
/// resolution happened instead of each command carrying a dead `Auto` arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedFormat {
    Json,
    Table,
}

impl OutputFormat {
    /// Resolve `Auto` to a concrete format based on whether stdout is a TTY.
    /// `Json` and `Table` pass through unchanged.
    pub fn resolve(self) -> ResolvedFormat {
        match self {
            OutputFormat::Auto => {
                if std::io::stdout().is_terminal() {
                    ResolvedFormat::Table
                } else {
                    ResolvedFormat::Json
                }
            }
            OutputFormat::Json => ResolvedFormat::Json,
            OutputFormat::Table => ResolvedFormat::Table,
        }
    }
}

/// Print a value as pretty-printed JSON to stdout.
pub fn print_json<T: Serialize>(value: &T) -> Result<()> {
    let s = serde_json::to_string_pretty(value)?;
    println!("{}", s);
    Ok(())
}
