// SPDX-License-Identifier: Apache-2.0
//! Memory eval system — quality measurement and feedback capture.

pub mod anthropic;
pub mod cli_batch;
pub mod judge;
pub mod shared;

pub mod answer_quality;
pub mod context_path;
pub mod cost;
pub mod engine_throughput;
pub mod entity_dedup;
mod eval_judge;
pub mod fixtures;
pub mod gen;
pub mod kg_faithfulness;
pub mod kg_faithfulness_llm;
pub mod latency;
pub mod layer;
pub mod lifecycle;
pub mod locomo;
pub mod longmemeval;
pub mod m5_bench_corpus;
#[cfg(feature = "eval-harness")]
pub mod m5_snapshot_io;
pub mod metrics;
pub mod page_faithfulness;
pub mod paired;
pub mod pipeline;
pub mod rank_overlap;
pub mod report;
pub mod retrieval;
pub mod retrieval_drift;
pub mod runner;
pub mod seed_contract;
pub mod signals;
pub mod wall_clock;
pub use layer::EvalLayer;

/// Backward-compat alias: old code using `eval::token_efficiency::*` still works.
pub use retrieval as token_efficiency;
