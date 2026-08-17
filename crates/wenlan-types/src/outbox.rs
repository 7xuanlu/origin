// SPDX-License-Identifier: Apache-2.0
//! Durable envelopes shared by the CLI outbox writer and daemon drainer.

use crate::{requests::StoreMemoryRequest, BriefUpdateRequest};
use serde::{Deserialize, Serialize};

pub const OUTBOX_SCHEMA: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OutboxEnvelope {
    pub schema: u32,
    pub created_at: String,
    pub caller_id: String,
    pub operation_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_name: Option<String>,
    #[serde(default)]
    pub reconcile_summary_version: bool,
    #[serde(flatten)]
    pub payload: OutboxPayload,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum OutboxPayload {
    BriefUpdate(BriefUpdateRequest),
    MemoryStore(StoreMemoryRequest),
}

impl OutboxPayload {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::BriefUpdate(_) => "brief_update",
            Self::MemoryStore(_) => "memory_store",
        }
    }
}

impl OutboxEnvelope {
    /// Return the canonical durable filename for this envelope.
    pub fn file_name(&self, unix_millis: u128) -> String {
        format!(
            "{unix_millis:013}-{}-{}.json",
            self.operation_id,
            self.payload.kind()
        )
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct OutboxDrainReport {
    pub applied: u32,
    pub duplicate: u32,
    pub failed: u32,
    pub remaining: u32,
    pub details: Vec<OutboxDrainDetail>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct OutboxDrainDetail {
    pub file: String,
    pub kind: String,
    pub outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BriefSummaryUpdate, WriteSpaceTarget};

    fn brief_envelope() -> OutboxEnvelope {
        OutboxEnvelope {
            schema: OUTBOX_SCHEMA,
            created_at: "2026-08-16T12:00:00.000Z".into(),
            caller_id: "wenlan-cli".into(),
            operation_id: "brief-op".into(),
            space: Some("demo".into()),
            agent_name: Some("codex".into()),
            reconcile_summary_version: true,
            payload: OutboxPayload::BriefUpdate(BriefUpdateRequest {
                space: "demo".into(),
                caller_id: "wenlan-cli".into(),
                operation_id: "brief-op".into(),
                summary: Some(BriefSummaryUpdate {
                    text: "queued summary".into(),
                    expected_version: 99,
                }),
                mutations: vec![],
            }),
        }
    }

    #[test]
    fn brief_envelope_roundtrips() {
        let envelope = brief_envelope();
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: OutboxEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
        assert_eq!(decoded.payload.kind(), "brief_update");
    }

    #[test]
    fn memory_envelope_roundtrips() {
        let envelope = OutboxEnvelope {
            schema: OUTBOX_SCHEMA,
            created_at: "2026-08-16T12:00:00.000Z".into(),
            caller_id: "wenlan-cli".into(),
            operation_id: "memory-op".into(),
            space: None,
            agent_name: None,
            reconcile_summary_version: false,
            payload: OutboxPayload::MemoryStore(StoreMemoryRequest {
                content: "a queued memory with enough content".into(),
                memory_type: Some("fact".into()),
                space: WriteSpaceTarget::Inherit,
                source_agent: None,
                title: None,
                confidence: None,
                supersedes: None,
                entity: None,
                entity_id: None,
                structured_fields: None,
                retrieval_cue: None,
            }),
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let decoded: OutboxEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, envelope);
        assert_eq!(decoded.payload.kind(), "memory_store");
    }

    #[test]
    fn canonical_file_name_is_zero_padded_and_kind_suffixed() {
        assert_eq!(
            brief_envelope().file_name(42),
            "0000000000042-brief-op-brief_update.json"
        );
    }
}
