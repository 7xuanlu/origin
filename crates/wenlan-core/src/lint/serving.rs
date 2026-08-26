use super::catalog::LintCheckGroup;
use super::context::LintContext;
use super::memories::MemoryFeatureConfig;
use wenlan_types::lint::LintCheckResult;

mod config;
mod fact_probe;
mod query;
mod result;
pub mod routes;
pub(super) use config::capture;

pub(crate) const ROUTE_SCOPE_ID: &str = "serving.route_scope_contracts";
pub(crate) const CHANNEL_EPISODE_ID: &str = "serving.channel.episode";
pub(crate) const CHANNEL_FACT_ID: &str = "serving.channel.fact";
pub(crate) const CHANNEL_GRAPH_ID: &str = "serving.channel.graph";
pub(crate) const CHANNEL_PAGE_ID: &str = "serving.channel.page";
pub(crate) const CHANNEL_SUMMARY_ID: &str = "serving.channel.summary";
pub(crate) const FACT_STARVATION_ID: &str = "serving.fact_scope_starvation";
pub(crate) const OBSERVABILITY_ID: &str = "serving.observability_inventory";
pub(crate) const RERANKER_ID: &str = "serving.reranker_fallback_inventory";
#[derive(Debug, Clone, Copy)]
pub(crate) struct ServingRunConfig {
    pub(crate) page: bool,
    episode: bool,
    fact: bool,
    summary: bool,
    graph: bool,
    pub(crate) reranker: bool,
    pub(crate) reranker_light: bool,
    pub(crate) reranker_deep: bool,
    pub(crate) reranker_paths: u64,
    pub(crate) fact_limit: usize,
    /// Whether the owner has chosen a model source. Threaded from
    /// [`crate::lint::kg::KgRunConfig`] so both this check and
    /// `kg.substrate_liveness` read one captured value.
    model_source_configured: bool,
}

impl ServingRunConfig {
    fn from_captured(
        memory: MemoryFeatureConfig,
        graph: bool,
        page: bool,
        fact_limit: usize,
        reranker_light: bool,
        reranker_deep: bool,
        model_source_configured: bool,
    ) -> Self {
        Self {
            page,
            episode: memory.episode,
            fact: memory.fact,
            summary: memory.summary,
            graph,
            reranker: reranker_light || reranker_deep,
            reranker_light,
            reranker_deep,
            reranker_paths: u64::from(reranker_light) + u64::from(reranker_deep),
            fact_limit,
            model_source_configured,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChannelAssessment {
    ExpectedEmpty {
        eligible: u64,
    },
    /// The channel is on, but the substrate it serves is built by background
    /// enrichment that has no model source to run on. Empty is the expected
    /// state, and the fix is a settings choice, not a repair.
    ModelSourceUnconfigured {
        eligible: u64,
    },
    Finding {
        eligible: u64,
    },
    Live {
        eligible: u64,
        observed: u64,
    },
}

pub(crate) const fn assess_channel(
    _id: &'static str,
    enabled: bool,
    eligible: u64,
    observed: u64,
) -> ChannelAssessment {
    if !enabled {
        ChannelAssessment::ExpectedEmpty { eligible }
    } else if eligible > 0 && observed == 0 {
        ChannelAssessment::Finding { eligible }
    } else {
        ChannelAssessment::Live { eligible, observed }
    }
}

pub(crate) async fn run(
    context: &LintContext<'_, '_>,
    config: ServingRunConfig,
) -> Vec<LintCheckResult> {
    let counts = match query::load(context).await {
        Ok(counts) => counts,
        Err(()) => return failed_results(context),
    };
    let bypasses = u64::try_from(routes::scope_contract_violations().count()).unwrap_or(u64::MAX);
    let mut results = vec![result::fixed(
        context,
        ROUTE_SCOPE_ID,
        bypasses,
        route_scope_finding(bypasses),
        false,
    )];
    // The graph channel is the one channel here whose substrate is built by
    // model-backed background enrichment. With no model source chosen, an empty
    // graph is the expected state, not a dead channel.
    for (id, enabled, count, needs_model_source) in [
        (CHANNEL_EPISODE_ID, config.episode, counts.episode, false),
        (CHANNEL_FACT_ID, config.fact, counts.fact, false),
        (CHANNEL_GRAPH_ID, config.graph, counts.graph, true),
        (CHANNEL_PAGE_ID, config.page, counts.page, false),
        (CHANNEL_SUMMARY_ID, config.summary, counts.summary, false),
    ] {
        let assessment = if enabled && needs_model_source && !config.model_source_configured {
            ChannelAssessment::ModelSourceUnconfigured {
                eligible: count.eligible,
            }
        } else {
            assess_channel(id, enabled, count.eligible, count.observed)
        };
        results.push(result::channel(context, id, assessment));
    }
    let fact_probe = if config.fact {
        match fact_probe::run(context, config.fact_limit).await {
            Ok(probe) => probe,
            Err(()) => return failed_results(context),
        }
    } else {
        fact_probe::FactProbe::default()
    };
    results.push(result::fact_probe(context, fact_probe));
    let telemetry = match query::load_telemetry(context).await {
        Ok(telemetry) => telemetry,
        Err(()) => return failed_results(context),
    };
    results.push(result::inventory(
        context,
        OBSERVABILITY_ID,
        vec![
            (
                wenlan_types::lint::LintMetricCode::AccessTelemetryRows,
                telemetry.access,
            ),
            (
                wenlan_types::lint::LintMetricCode::AgentActivityTelemetryRows,
                telemetry.activity,
            ),
            (
                wenlan_types::lint::LintMetricCode::UnattributedServingChannels,
                2,
            ),
        ],
    ));
    results.push(result::inventory(
        context,
        RERANKER_ID,
        vec![
            (
                wenlan_types::lint::LintMetricCode::RerankerConfiguredPaths,
                config.reranker_paths,
            ),
            (
                wenlan_types::lint::LintMetricCode::RerankerRuntimeReadinessUnavailable,
                1,
            ),
        ],
    ));
    results
}

const fn route_scope_finding(violations: u64) -> bool {
    violations > 0
}

fn failed_results(context: &LintContext<'_, '_>) -> Vec<LintCheckResult> {
    let selected = context.scope().filter().is_selected();
    for entry in crate::lint::catalog::catalog_group(LintCheckGroup::Serving) {
        let basis =
            if selected && entry.scope_policy == crate::lint::catalog::ScopePolicy::ScopedRows {
                crate::lint::context::PopulationBasis::SelectedScope
            } else {
                crate::lint::context::PopulationBasis::Global
            };
        let _ = context.record_population(entry.id, basis, 0);
    }
    super::runner::failed_results_for_group(context.clock(), LintCheckGroup::Serving)
}

#[cfg(test)]
pub(crate) fn channel_result(
    clock: &super::context::LintClock,
    id: &'static str,
    assessment: ChannelAssessment,
) -> LintCheckResult {
    result::channel_for_test(clock, id, assessment)
}

#[cfg(test)]
#[path = "serving_test.rs"]
mod tests;
