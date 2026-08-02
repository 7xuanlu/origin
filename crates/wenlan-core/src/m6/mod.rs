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
pub mod label_key;

#[cfg(test)]
mod digest_test;
#[cfg(test)]
mod identity_test;
#[cfg(test)]
mod label_key_test;
