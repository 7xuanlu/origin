// SPDX-License-Identifier: AGPL-3.0-only
//! DataSource trait — app-local definition.
//! Moved from origin-core::sources; the trait and its impls are app-only
//! (origin-server never references DataSource directly).
use crate::error::AppError;
use async_trait::async_trait;
use std::any::Any;
use wenlan_types::sources::{RawDocument, SourceStatus};

/// Trait that all data source connectors must implement.
///
/// Intentionally duplicated from
/// `7xuanlu/origin: crates/origin-core/src/sources/mod.rs`. origin-app has
/// no origin-core dependency (Phase 5-D PR2 dropped it for license + bloat
/// reasons), so the trait declaration is mirrored here with `AppError` in
/// place of `OriginError`. The shared data shapes (`RawDocument`,
/// `SourceStatus`) live in wenlan-types so connectors can move freely
/// between crates.
#[async_trait]
pub trait DataSource: Send + Sync {
    /// Unique name for this source ("gmail", "notion", etc.)
    fn name(&self) -> &str;

    /// Whether this source requires OAuth authentication
    fn requires_auth(&self) -> bool;

    /// Check if the source is currently connected/authenticated
    async fn is_connected(&self) -> bool;

    /// Connect/authenticate the source (triggers OAuth if needed)
    async fn connect(&mut self) -> Result<(), AppError>;

    /// Disconnect the source (revoke tokens, cleanup)
    async fn disconnect(&mut self) -> Result<(), AppError>;

    /// Fetch new/updated content since last sync
    async fn fetch_updates(&mut self) -> Result<Vec<RawDocument>, AppError>;

    /// Initial full sync - fetches all available content
    async fn full_sync(&mut self) -> Result<Vec<RawDocument>, AppError>;

    /// Get the current status of this source
    async fn status(&self) -> SourceStatus;

    /// Downcast to concrete type
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
