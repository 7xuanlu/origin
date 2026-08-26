use super::ServingRunConfig;
use crate::lint::memories::MemoryFeatureConfig;

pub(crate) fn capture(
    memory: MemoryFeatureConfig,
    graph: bool,
    model_source_configured: bool,
) -> ServingRunConfig {
    let mode = crate::reranker::reranker_mode_resolved(&crate::config::load_config());
    let legacy = std::env::var("WENLAN_RERANKER_ENABLED").as_deref() == Ok("1");
    let plan = crate::reranker::resolve_reranker_plan(mode, legacy);
    ServingRunConfig::from_captured(
        memory,
        graph,
        crate::db::page_channel_enabled(),
        crate::retrieval::fact_channel::fact_channel_limit(),
        plan.light.is_some(),
        plan.deep.is_some(),
        model_source_configured,
    )
}
