// SPDX-License-Identifier: Apache-2.0
//! Page export to an external vault, plus the shared `ExportResult`/`ExportStats` shapes.

pub mod knowledge;
pub mod obsidian;
pub mod provenance;

// ExportStats moved to wenlan-types in Phase 5-D PR2 so the Tauri app can
// deserialize it without pulling in the full wenlan-core dep.
pub use wenlan_types::ExportStats;

#[derive(Debug)]
pub struct ExportResult {
    pub concept_id: String,
    pub path: String,
}
