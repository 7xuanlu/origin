// SPDX-License-Identifier: AGPL-3.0-only
//! Memory eval system — quality measurement and feedback capture.

pub mod anthropic;
pub mod judge;
pub mod shared;

pub mod answer_quality;
pub mod distillation_quality;
pub mod fixtures;
pub mod gen;
pub mod lifebench;
pub mod lifecycle;
pub mod locomo;
pub mod locomo_plus;
pub mod longmemeval;
pub mod metrics;
pub mod pipeline;
pub mod report;
pub mod runner;
pub mod signals;
pub mod store_quality;
pub mod token_efficiency;

// Closed-core modules (proprietary, feature-gated)
#[cfg(feature = "closed-core-eval")]
pub mod auto_tune;
