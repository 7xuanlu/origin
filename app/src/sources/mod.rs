// SPDX-License-Identifier: AGPL-3.0-only
//! App-level sources — re-exports origin-core sources + app-specific sync logic.
pub mod sync;

// Re-export everything from origin-core::sources so `crate::sources::*` paths
// continue to work unchanged throughout the app.
pub use origin_core::sources::*;
