// SPDX-License-Identifier: Apache-2.0
//! Retrieval module: pool builders, hard filters, and per-signal helpers
//! used by `db::search_memory*` paths.
//!
//! Moved out of `composite/` in PR-A so the legacy retrieval path can use
//! these helpers without importing the composite scoring orchestrator.

pub mod blend;
pub(crate) mod compress;
pub(crate) mod dedup;
pub(crate) mod document_cap;
pub(crate) mod fact_channel;
pub(crate) mod fts_query;
pub(crate) mod hard_filters;
pub(crate) mod integrity;
pub mod intent;
pub(crate) mod query_intent;
pub(crate) mod resolve;
pub(crate) mod session_diversity;
pub(crate) mod signals;
pub(crate) mod traversal;
