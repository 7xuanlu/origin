// SPDX-License-Identifier: Apache-2.0
//! Knowledge graph phase modules: per-phase logic for entity extraction, the
//! shared entity admission seam and its name-quality rules, and entity link
//! reweaving.

pub mod entity_admission;
pub mod entity_extraction;
pub mod entity_name_quality;
pub mod reweave;
