pub use wenlan_types::lint::LintCheckGroup;
use wenlan_types::lint::{LintGateEffect, LintProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScopePolicy {
    ScopedRows,
    DbAnchoredProjection,
    GlobalAggregateOnly,
    GlobalOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeAxis {
    PagesWorkspace,
    MemoriesSpace,
    EntitiesSpace,
    IdentityGlobal,
    OperationsGlobal,
    RuntimeGlobal,
    ServingGlobal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintProfileMembership {
    General,
    DeepOnly,
}

impl LintProfileMembership {
    const fn applies_to(self, profile: LintProfile) -> bool {
        matches!(self, Self::General) || matches!(profile, LintProfile::Deep)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LintCatalogEntry {
    pub id: &'static str,
    pub scope_policy: ScopePolicy,
    pub scope_axis: ScopeAxis,
    pub group: LintCheckGroup,
    pub profile: LintProfileMembership,
    pub gate_effect: LintGateEffect,
}

const CATALOG: &[LintCatalogEntry] = &[
    deep_entry(
        "entities.alias_collision_inventory",
        ScopePolicy::ScopedRows,
        ScopeAxis::EntitiesSpace,
        LintGateEffect::Advisory,
    ),
    entity_entry("entities.partition_inventory", ScopePolicy::ScopedRows),
    entity_entry("entities.structural_integrity", ScopePolicy::ScopedRows),
    identity_entry("identity.cache_inventory", ScopePolicy::GlobalAggregateOnly),
    identity_entry("identity.memory_state_integrity", ScopePolicy::ScopedRows),
    identity_entry("identity.registry_integrity", ScopePolicy::GlobalOnly),
    identity_entry("identity.session_structure", ScopePolicy::GlobalOnly),
    identity_entry("identity.tag_integrity", ScopePolicy::GlobalOnly),
    entity_entry("kg.advisory_inventory", ScopePolicy::GlobalAggregateOnly),
    entity_entry("kg.aggregate_inventory", ScopePolicy::GlobalAggregateOnly),
    deep_entry(
        "kg.semantic.entity_relations",
        ScopePolicy::ScopedRows,
        ScopeAxis::EntitiesSpace,
        LintGateEffect::Advisory,
    ),
    deep_memory_kg_entry(
        "kg.semantic.memory_entity_links",
        ScopePolicy::ScopedRows,
        LintGateEffect::Advisory,
    ),
    memory_kg_entry("kg.substrate_liveness", ScopePolicy::ScopedRows),
    memory_entry("memories.derived.episode", ScopePolicy::ScopedRows),
    memory_entry("memories.derived.fact", ScopePolicy::ScopedRows),
    memory_entry("memories.derived.page_links", ScopePolicy::ScopedRows),
    memory_entry("memories.derived.summary", ScopePolicy::ScopedRows),
    memory_entry("memories.derived.temporal", ScopePolicy::ScopedRows),
    deep_entry(
        "memories.duplicate_inventory",
        ScopePolicy::ScopedRows,
        ScopeAxis::MemoriesSpace,
        LintGateEffect::Advisory,
    ),
    memory_entry("memories.embedding_integrity", ScopePolicy::ScopedRows),
    memory_entry("memories.enrichment_failures", ScopePolicy::ScopedRows),
    memory_entry("memories.lifecycle_integrity", ScopePolicy::ScopedRows),
    memory_entry("memories.partition_inventory", ScopePolicy::ScopedRows),
    deep_entry(
        "memories.retrieval_substrate_inventory",
        ScopePolicy::ScopedRows,
        ScopeAxis::MemoriesSpace,
        LintGateEffect::Advisory,
    ),
    deep_entry(
        "memories.semantic.classification",
        ScopePolicy::ScopedRows,
        ScopeAxis::MemoriesSpace,
        LintGateEffect::Advisory,
    ),
    deep_entry(
        "memories.semantic.contradiction",
        ScopePolicy::ScopedRows,
        ScopeAxis::MemoriesSpace,
        LintGateEffect::Advisory,
    ),
    deep_entry(
        "memories.semantic.staleness",
        ScopePolicy::ScopedRows,
        ScopeAxis::MemoriesSpace,
        LintGateEffect::Advisory,
    ),
    deep_entry(
        "memories.structured_conflict_inventory",
        ScopePolicy::ScopedRows,
        ScopeAxis::MemoriesSpace,
        LintGateEffect::Advisory,
    ),
    memory_entry("memories.supersession_integrity", ScopePolicy::ScopedRows),
    memory_kg_entry("memory_entities.integrity", ScopePolicy::ScopedRows),
    deep_entry(
        "observations.duplicate_inventory",
        ScopePolicy::ScopedRows,
        ScopeAxis::EntitiesSpace,
        LintGateEffect::Advisory,
    ),
    entity_entry("observations.integrity", ScopePolicy::ScopedRows),
    operations_entry("operations.document_queue"),
    operations_entry("operations.import_checkpoints"),
    operations_entry("operations.maintenance_backlogs"),
    operations_entry("operations.refinement_inventory"),
    operations_entry("operations.rejection_inventory"),
    operations_entry("operations.source_configuration"),
    deep_entry(
        "operations.source_lifecycle_residue",
        ScopePolicy::GlobalAggregateOnly,
        ScopeAxis::OperationsGlobal,
        LintGateEffect::Actionable,
    ),
    page_entry("pages.archive_inventory", ScopePolicy::ScopedRows),
    page_entry("pages.citations.partitions", ScopePolicy::ScopedRows),
    page_entry("pages.db.partitions", ScopePolicy::ScopedRows),
    page_entry("pages.duplicate_active_titles", ScopePolicy::ScopedRows),
    deep_entry(
        "pages.duplicate_body_inventory",
        ScopePolicy::ScopedRows,
        ScopeAxis::PagesWorkspace,
        LintGateEffect::Advisory,
    ),
    page_entry("pages.links.orphan_labels", ScopePolicy::ScopedRows),
    page_entry("pages.project.artifact_inventory", ScopePolicy::GlobalOnly),
    deep_entry(
        "pages.projection.body_alignment",
        ScopePolicy::DbAnchoredProjection,
        ScopeAxis::PagesWorkspace,
        LintGateEffect::Actionable,
    ),
    page_entry(
        "pages.projection.identity",
        ScopePolicy::DbAnchoredProjection,
    ),
    page_entry(
        "pages.projection.manifest_inventory",
        ScopePolicy::GlobalAggregateOnly,
    ),
    page_entry(
        "pages.projection.state_contract",
        ScopePolicy::DbAnchoredProjection,
    ),
    page_entry(
        "pages.projection.version_alignment",
        ScopePolicy::DbAnchoredProjection,
    ),
    page_entry(
        "pages.provenance.source_evidence_coverage",
        ScopePolicy::DbAnchoredProjection,
    ),
    page_entry("pages.review_status_inventory", ScopePolicy::ScopedRows),
    deep_entry(
        "pages.semantic.evidence_links",
        ScopePolicy::ScopedRows,
        ScopeAxis::PagesWorkspace,
        LintGateEffect::Advisory,
    ),
    deep_entry(
        "pages.semantic.faithfulness",
        ScopePolicy::ScopedRows,
        ScopeAxis::PagesWorkspace,
        LintGateEffect::Advisory,
    ),
    deep_entry(
        "pages.semantic.provenance_adequacy",
        ScopePolicy::ScopedRows,
        ScopeAxis::PagesWorkspace,
        LintGateEffect::Advisory,
    ),
    page_entry("pages.source_page_integrity", ScopePolicy::ScopedRows),
    entity_entry("relations.integrity", ScopePolicy::ScopedRows),
    deep_entry(
        "relations.vocabulary_integrity",
        ScopePolicy::ScopedRows,
        ScopeAxis::EntitiesSpace,
        LintGateEffect::Actionable,
    ),
    runtime_entry("runtime.ingest_worker_liveness", ScopePolicy::GlobalOnly),
    runtime_entry(
        "runtime.provider_inventory",
        ScopePolicy::GlobalAggregateOnly,
    ),
    runtime_entry("runtime.schema_contract", ScopePolicy::GlobalOnly),
    runtime_entry("runtime.search_index_contract", ScopePolicy::GlobalOnly),
    runtime_entry("runtime.status_parity", ScopePolicy::GlobalAggregateOnly),
    serving_entry("serving.channel.episode", ScopePolicy::ScopedRows),
    serving_entry("serving.channel.fact", ScopePolicy::ScopedRows),
    serving_entry("serving.channel.graph", ScopePolicy::ScopedRows),
    serving_entry("serving.channel.page", ScopePolicy::ScopedRows),
    serving_entry("serving.channel.summary", ScopePolicy::ScopedRows),
    serving_entry("serving.fact_scope_starvation", ScopePolicy::ScopedRows),
    serving_global_entry("serving.observability_inventory"),
    serving_global_entry("serving.reranker_fallback_inventory"),
    serving_global_entry("serving.route_scope_contracts"),
    deep_entry(
        "serving.semantic.retrieval_quality",
        ScopePolicy::ScopedRows,
        ScopeAxis::ServingGlobal,
        LintGateEffect::Advisory,
    ),
];

const fn deep_entry(
    id: &'static str,
    scope_policy: ScopePolicy,
    scope_axis: ScopeAxis,
    gate_effect: LintGateEffect,
) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis,
        group: match scope_axis {
            ScopeAxis::EntitiesSpace => LintCheckGroup::KnowledgeGraph,
            ScopeAxis::MemoriesSpace => LintCheckGroup::Memories,
            ScopeAxis::OperationsGlobal => LintCheckGroup::Operations,
            ScopeAxis::PagesWorkspace => LintCheckGroup::Pages,
            ScopeAxis::IdentityGlobal => LintCheckGroup::Identity,
            ScopeAxis::RuntimeGlobal => LintCheckGroup::Runtime,
            ScopeAxis::ServingGlobal => LintCheckGroup::Serving,
        },
        profile: LintProfileMembership::DeepOnly,
        gate_effect,
    }
}

const fn entity_entry(id: &'static str, scope_policy: ScopePolicy) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: ScopeAxis::EntitiesSpace,
        group: LintCheckGroup::KnowledgeGraph,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn memory_kg_entry(id: &'static str, scope_policy: ScopePolicy) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: ScopeAxis::MemoriesSpace,
        group: LintCheckGroup::KnowledgeGraph,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn deep_memory_kg_entry(
    id: &'static str,
    scope_policy: ScopePolicy,
    gate_effect: LintGateEffect,
) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: ScopeAxis::MemoriesSpace,
        group: LintCheckGroup::KnowledgeGraph,
        profile: LintProfileMembership::DeepOnly,
        gate_effect,
    }
}

const fn memory_entry(id: &'static str, scope_policy: ScopePolicy) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: ScopeAxis::MemoriesSpace,
        group: LintCheckGroup::Memories,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn page_entry(id: &'static str, scope_policy: ScopePolicy) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: ScopeAxis::PagesWorkspace,
        group: LintCheckGroup::Pages,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn operations_entry(id: &'static str) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy: ScopePolicy::GlobalAggregateOnly,
        scope_axis: ScopeAxis::OperationsGlobal,
        group: LintCheckGroup::Operations,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn identity_entry(id: &'static str, scope_policy: ScopePolicy) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: if matches!(scope_policy, ScopePolicy::ScopedRows) {
            ScopeAxis::MemoriesSpace
        } else {
            ScopeAxis::IdentityGlobal
        },
        group: LintCheckGroup::Identity,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn runtime_entry(id: &'static str, scope_policy: ScopePolicy) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: ScopeAxis::RuntimeGlobal,
        group: LintCheckGroup::Runtime,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn serving_entry(id: &'static str, scope_policy: ScopePolicy) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy,
        scope_axis: ScopeAxis::MemoriesSpace,
        group: LintCheckGroup::Serving,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

const fn serving_global_entry(id: &'static str) -> LintCatalogEntry {
    LintCatalogEntry {
        id,
        scope_policy: ScopePolicy::GlobalAggregateOnly,
        scope_axis: ScopeAxis::ServingGlobal,
        group: LintCheckGroup::Serving,
        profile: LintProfileMembership::General,
        gate_effect: LintGateEffect::Actionable,
    }
}

pub fn catalog() -> &'static [LintCatalogEntry] {
    CATALOG
}

pub fn catalog_for_profile(
    profile: LintProfile,
) -> impl Iterator<Item = &'static LintCatalogEntry> {
    CATALOG
        .iter()
        .filter(move |entry| entry.profile.applies_to(profile))
}

pub(crate) fn catalog_group(
    group: LintCheckGroup,
) -> impl Iterator<Item = &'static LintCatalogEntry> {
    CATALOG.iter().filter(move |entry| {
        entry.group == group && entry.profile == LintProfileMembership::General
    })
}

pub fn catalog_entry(check_id: &str) -> Option<&'static LintCatalogEntry> {
    CATALOG
        .binary_search_by_key(&check_id, |entry| entry.id)
        .ok()
        .map(|index| &CATALOG[index])
}

#[cfg(test)]
#[path = "catalog_validation.rs"]
mod validation;

#[cfg(test)]
pub(crate) use validation::has_valid_owner;
