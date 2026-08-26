use wenlan_types::lint::{LintConfigSelection, LintConfigSetting, LintConfigValue};

#[derive(Debug, Clone, Copy)]
pub(crate) struct KgRunConfig {
    pub(crate) serving_enabled: bool,
    pub(crate) sweep_enabled: bool,
    pub(crate) provider_ready: bool,
    /// Whether the owner has chosen a model source for everyday background work.
    ///
    /// This is the config pin, not a runtime latch. `provider_ready` above is
    /// `llm_provider_ready()`, a sticky process-wide "some provider has served
    /// traffic" flag: it is false on a freshly started daemon even when a source
    /// is configured, so it cannot answer "has the owner picked a model source".
    /// The pin is what `refinery::resolve_everyday` reads to produce
    /// `RouteMode::Unconfigured`, which is the exact condition on which the
    /// capture path tells the user that "background enrichment is paused until
    /// you choose a model source". A source that is chosen but unavailable
    /// resolves to `PinnedUnavailable`, so it stays `true` here and is never
    /// told to choose a source it already chose.
    pub(crate) model_source_configured: bool,
    pub(crate) hub_cap: u64,
}

impl KgRunConfig {
    pub(crate) fn capture() -> Self {
        Self {
            serving_enabled: crate::db::graph_memory_stream_enabled(),
            sweep_enabled: crate::db::entity_sweep_enabled(),
            provider_ready: crate::llm_provider::llm_provider_ready(),
            model_source_configured: crate::refinery::EverydaySource::parse(
                crate::config::load_config().everyday_source.as_deref(),
            )
            .is_some(),
            hub_cap: u64::try_from(crate::db::graph_hub_cap()).unwrap_or(u64::MAX),
        }
    }

    #[cfg(test)]
    pub(crate) const fn for_test(
        serving_enabled: bool,
        sweep_enabled: bool,
        provider_ready: bool,
        model_source_configured: bool,
        hub_cap: u64,
    ) -> Self {
        Self {
            serving_enabled,
            sweep_enabled,
            provider_ready,
            model_source_configured,
            hub_cap,
        }
    }

    pub(crate) fn fingerprint_selections(self) -> [LintConfigSelection; 5] {
        [
            selection(
                LintConfigSetting::ModelSourceConfigured,
                self.model_source_configured,
            ),
            selection(
                LintConfigSetting::KnowledgeGraphServingEnabled,
                self.serving_enabled,
            ),
            selection(
                LintConfigSetting::KnowledgeGraphSweepEnabled,
                self.sweep_enabled,
            ),
            selection(
                LintConfigSetting::KnowledgeGraphProviderReady,
                self.provider_ready,
            ),
            LintConfigSelection::count(LintConfigSetting::KnowledgeGraphHubCap, self.hub_cap),
        ]
    }
}

const fn selection(setting: LintConfigSetting, enabled: bool) -> LintConfigSelection {
    LintConfigSelection::new(
        setting,
        if enabled {
            LintConfigValue::Enabled
        } else {
            LintConfigValue::Disabled
        },
    )
}
