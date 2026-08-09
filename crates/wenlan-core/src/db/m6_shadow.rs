// SPDX-License-Identifier: Apache-2.0
//! The daemon's seam onto the M6 shadow turn driver (spec §4.1).
//!
//! `m6::shadow::run_genesis_shadow_turn` takes a `&libsql::Connection`, and
//! `MemoryDB::conn` is private — deliberately so: `drift_guard`'s R4 census
//! treats any escape of the `.conn` field as a tracked test-support origin, and
//! `m6` never reaches for it. This module is the one place the connection
//! crosses, and it crosses inside an `impl MemoryDB` block exactly the way
//! `db/claim_derivation.rs` does for the M5 promoter.
//!
//! Holding the connection mutex across the await is not the `RwLock`-across-
//! `.await` hazard the root `AGENTS.md` forbids: that rule is about
//! `ServerState`'s `tokio::sync::RwLock`, which gates every HTTP handler. This
//! is `MemoryDB`'s own connection `Mutex`, which every database call in the
//! crate must hold for its duration because `libsql::Connection` is `Send` but
//! not `Sync`. A turn is bounded by §4.2 — one space's differential query, one
//! prepare, or one candidate's eight CAS gates — and makes no model call, so it
//! cannot hold it long.

use super::MemoryDB;
use crate::m6::shadow::{self, GenesisTurn, ShadowState};
use crate::WenlanError;

impl MemoryDB {
    /// Run one M6 genesis shadow turn (§4.1/§4.2).
    ///
    /// Writes only to `genesis_*`, the shared `grouping_leases` registry
    /// (M6 phases only), `m6_readiness`, `m6_counters`, and M6's own
    /// `app_metadata` frontier-cursor key. Never writes `genesis_enabled`;
    /// never touches `pages`, `page_*`, `entities`, `relations`,
    /// `observations`, `edges`, `memories`, or `chunks`; never consults an LLM.
    pub async fn run_genesis_shadow_turn(
        &self,
        state: &mut ShadowState,
    ) -> Result<GenesisTurn, WenlanError> {
        let conn = self.conn.lock().await;
        shadow::run_genesis_shadow_turn(&conn, state).await
    }
}
