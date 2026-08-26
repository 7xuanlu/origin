// SPDX-License-Identifier: Apache-2.0
use super::contract::LintDigest;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintConfigSetting {
    RerankerEnabled,
    PageProjectionEnabled,
    EpisodeChannelEnabled,
    FactChannelEnabled,
    SummaryPreludeEnabled,
    TemporalGroundingEnabled,
    KnowledgeGraphServingEnabled,
    KnowledgeGraphSweepEnabled,
    KnowledgeGraphProviderReady,
    KnowledgeGraphHubCap,
    SourceConfigurationCaptured,
    SourceConfigurationCount,
    SourceSnapshotIdentity,
    PageRetrievalChannelEnabled,
    FactChannelLimit,
    RerankerLightConfigured,
    RerankerDeepConfigured,
    ProviderSlotsRequested,
    ModelSlotsConfigured,
    RerankerPathsRequested,
    ModelConfigurationIdentity,
    RuntimeObservationCaptured,
    SemanticProviderReady,
    SemanticProviderOnDevice,
    SemanticExternalEgressEnabled,
    SemanticCallingAgentEnabled,
    ModelSourceConfigured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintConfigValue {
    Enabled,
    Disabled,
    Count(u64),
    Digest([u8; 32]),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LintConfigSelection {
    setting: LintConfigSetting,
    value: LintConfigValue,
}

impl LintConfigSelection {
    pub const fn new(setting: LintConfigSetting, value: LintConfigValue) -> Self {
        Self { setting, value }
    }

    pub const fn count(setting: LintConfigSetting, value: u64) -> Self {
        Self {
            setting,
            value: LintConfigValue::Count(value),
        }
    }

    pub const fn digest(setting: LintConfigSetting, value: [u8; 32]) -> Self {
        Self {
            setting,
            value: LintConfigValue::Digest(value),
        }
    }

    fn bytes(self) -> Vec<u8> {
        let setting = match self.setting {
            LintConfigSetting::RerankerEnabled => 1,
            LintConfigSetting::PageProjectionEnabled => 2,
            LintConfigSetting::EpisodeChannelEnabled => 3,
            LintConfigSetting::FactChannelEnabled => 4,
            LintConfigSetting::SummaryPreludeEnabled => 5,
            LintConfigSetting::TemporalGroundingEnabled => 6,
            LintConfigSetting::KnowledgeGraphServingEnabled => 7,
            LintConfigSetting::KnowledgeGraphSweepEnabled => 8,
            LintConfigSetting::KnowledgeGraphProviderReady => 9,
            LintConfigSetting::KnowledgeGraphHubCap => 10,
            LintConfigSetting::SourceConfigurationCaptured => 11,
            LintConfigSetting::SourceConfigurationCount => 12,
            LintConfigSetting::SourceSnapshotIdentity => 13,
            LintConfigSetting::PageRetrievalChannelEnabled => 14,
            LintConfigSetting::FactChannelLimit => 15,
            LintConfigSetting::RerankerLightConfigured => 16,
            LintConfigSetting::RerankerDeepConfigured => 17,
            LintConfigSetting::ProviderSlotsRequested => 18,
            LintConfigSetting::ModelSlotsConfigured => 19,
            LintConfigSetting::RerankerPathsRequested => 20,
            LintConfigSetting::ModelConfigurationIdentity => 21,
            LintConfigSetting::RuntimeObservationCaptured => 22,
            LintConfigSetting::SemanticProviderReady => 23,
            LintConfigSetting::SemanticProviderOnDevice => 24,
            LintConfigSetting::SemanticExternalEgressEnabled => 25,
            LintConfigSetting::SemanticCallingAgentEnabled => 26,
            LintConfigSetting::ModelSourceConfigured => 27,
        };
        let mut bytes = vec![setting];
        match self.value {
            LintConfigValue::Enabled => bytes.push(1),
            LintConfigValue::Disabled => bytes.push(2),
            LintConfigValue::Count(value) => {
                bytes.push(3);
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            LintConfigValue::Digest(value) => {
                bytes.push(4);
                bytes.extend_from_slice(&value);
            }
        }
        bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LintConfigFingerprint(LintDigest);

impl LintConfigFingerprint {
    pub fn from_effective_config(selections: &[LintConfigSelection]) -> Self {
        let mut sorted = selections.to_vec();
        sorted.sort_unstable();
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for selection in sorted {
            for byte in selection.bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Self(LintDigest::from_u64(hash))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}
