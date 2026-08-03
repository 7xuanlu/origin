//! M6 genesis substrate.
//!
//! PR-A lands the additive schema, the deterministic-identity primitives, and
//! the teeth that guard them. **Nothing in this module runs automatically.**
//! `genesis_coverage_state.genesis_enabled` defaults to `0` and a space with no
//! row is treated as genesis-disabled, so an empty table means nothing runs.
//!
//! The Stage-0 contract artifacts under `docs/plans/2026-08-01-m6-*.md` are
//! normative for everything here; `S0-NN` references in doc comments point at
//! their decision numbers.

pub mod constants;
pub mod digest;
pub mod identity;
pub mod independence;
pub mod label_key;
pub mod oracle;
pub mod signals;
// Migration 109 is schema-first: PR-B/PR-C wire these transaction-scoped
// writers. Keep the staged APIs crate-private without exposing speculative
// public surface solely to satisfy dead-code linting.
#[allow(dead_code)]
pub(crate) mod frontier_policy;
pub(crate) mod overview_subscriptions;
#[allow(dead_code)]
pub(crate) mod refresh_readiness;
#[allow(dead_code)]
pub(crate) mod relevance;
pub(crate) mod remaining_substrate;

#[cfg(test)]
mod digest_test;
#[cfg(test)]
mod frontier_policy_test;
#[cfg(test)]
mod identity_test;
#[cfg(test)]
mod independence_test;
#[cfg(test)]
mod label_key_test;
#[cfg(test)]
mod oracle_test;
#[cfg(test)]
mod overview_subscriptions_test;
#[cfg(test)]
mod refresh_readiness_test;
#[cfg(test)]
mod relevance_test;
#[cfg(test)]
mod remaining_substrate_test;
#[cfg(test)]
mod signals_test;
