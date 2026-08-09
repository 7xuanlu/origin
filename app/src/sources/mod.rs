// SPDX-License-Identifier: AGPL-3.0-only
//! App-level sources: wire types from wenlan-types + app-local trait/impls.
pub mod data_source;
pub mod local_files;
pub mod obsidian;
pub mod sync;
pub mod uploads;

// Wire types (Source, SourceStatus, RawDocument, MemoryType, etc.) from the
// shared types crate -- no heavy deps, safe for downstream consumers.
pub use wenlan_types::sources::*;

// App-local DataSource trait (moved from origin-core in Phase 5-D PR2).
pub use data_source::DataSource;
