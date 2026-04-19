// SPDX-License-Identifier: AGPL-3.0-only
//! Generic concept export framework.

pub mod knowledge;
pub mod obsidian;

use crate::concepts::Concept;
use crate::error::OriginError;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ExportResult {
    pub concept_id: String,
    pub path: String,
}

#[derive(Debug, Default, Serialize)]
pub struct ExportStats {
    pub exported: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// Trait for exporting concepts to external formats/systems.
pub trait ConceptExporter {
    fn export(&self, concept: &Concept) -> Result<ExportResult, OriginError>;
    fn export_all(&self, concepts: &[Concept]) -> Result<ExportStats, OriginError>;
}
