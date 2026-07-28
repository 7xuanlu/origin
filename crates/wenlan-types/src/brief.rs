// SPDX-License-Identifier: Apache-2.0

use crate::memory::SearchResult;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BriefItemState {
    Active,
    Backlog,
}

impl BriefItemState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Backlog => "backlog",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BriefItem {
    pub id: String,
    pub text: String,
    pub state: BriefItemState,
    pub added_at: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate: Option<String>,
    pub version: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Brief {
    pub space_id: String,
    pub space: String,
    pub last_session_summary: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_handoff_at: Option<i64>,
    pub version: i64,
    pub active: Vec<BriefItem>,
    pub backlog: Vec<BriefItem>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BriefReadState {
    SpaceNotResolved,
    BriefNotCreated,
    Ready,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefRelatedContext {
    pub query: String,
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BriefReadRequest {
    #[serde(default)]
    pub topic: Option<String>,
    #[serde(default, alias = "domain")]
    pub space: Option<String>,
    // Internal compatibility marker: never serialized on the Brief wire.
    #[doc(hidden)]
    #[serde(skip)]
    pub legacy_context_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BriefReadResponse {
    pub state: BriefReadState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brief: Option<Brief>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub related_context: Option<BriefRelatedContext>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BriefSummaryUpdate {
    pub text: String,
    pub expected_version: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BriefMutation {
    Add {
        text: String,
        state: BriefItemState,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        added_at: Option<i64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gate: Option<String>,
    },
    Edit {
        item_id: String,
        expected_version: i64,
        text: String,
    },
    Move {
        item_id: String,
        expected_version: i64,
        state: BriefItemState,
    },
    SetGate {
        item_id: String,
        expected_version: i64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gate: Option<String>,
    },
    Complete {
        item_id: String,
        expected_version: i64,
    },
}

impl BriefMutation {
    pub fn kind(&self) -> BriefMutationKind {
        match self {
            Self::Add { .. } => BriefMutationKind::Add,
            Self::Edit { .. } => BriefMutationKind::Edit,
            Self::Move { .. } => BriefMutationKind::Move,
            Self::SetGate { .. } => BriefMutationKind::SetGate,
            Self::Complete { .. } => BriefMutationKind::Complete,
        }
    }

    pub fn item_id(&self) -> Option<&str> {
        match self {
            Self::Add { .. } => None,
            Self::Edit { item_id, .. }
            | Self::Move { item_id, .. }
            | Self::SetGate { item_id, .. }
            | Self::Complete { item_id, .. } => Some(item_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BriefUpdateRequest {
    pub space: String,
    pub caller_id: String,
    pub operation_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<BriefSummaryUpdate>,
    #[serde(default)]
    pub mutations: Vec<BriefMutation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BriefMutationKind {
    Summary,
    Add,
    Edit,
    Move,
    SetGate,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BriefAppliedMutation {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutation_index: Option<usize>,
    pub kind: BriefMutationKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BriefConflictReason {
    BriefVersionMismatch,
    VersionMismatch,
    ItemNotFound,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BriefMutationConflict {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mutation_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    pub reason: BriefConflictReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_version: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BriefUpdateReceipt {
    pub space: String,
    pub brief_version: i64,
    pub applied: Vec<BriefAppliedMutation>,
    pub conflicts: Vec<BriefMutationConflict>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_path: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn brief_read_states_are_explicit_on_the_wire() {
        assert_eq!(
            serde_json::to_value(BriefReadState::SpaceNotResolved).unwrap(),
            json!("space_not_resolved")
        );
        assert_eq!(
            serde_json::to_value(BriefReadState::BriefNotCreated).unwrap(),
            json!("brief_not_created")
        );
        assert_eq!(
            serde_json::to_value(BriefReadState::Ready).unwrap(),
            json!("ready")
        );
    }

    #[test]
    fn existing_item_mutations_carry_expected_versions() {
        let mutations = [
            BriefMutation::Edit {
                item_id: "item-1".into(),
                expected_version: 4,
                text: "ship the release".into(),
            },
            BriefMutation::Move {
                item_id: "item-1".into(),
                expected_version: 4,
                state: BriefItemState::Backlog,
            },
            BriefMutation::SetGate {
                item_id: "item-1".into(),
                expected_version: 4,
                gate: Some("CI green".into()),
            },
            BriefMutation::Complete {
                item_id: "item-1".into(),
                expected_version: 4,
            },
        ];

        for mutation in mutations {
            let value = serde_json::to_value(mutation).unwrap();
            assert_eq!(value["item_id"], "item-1");
            assert_eq!(value["expected_version"], 4);
        }
    }

    #[test]
    fn mutation_receipt_separates_applied_changes_from_conflicts() {
        let receipt = BriefUpdateReceipt {
            space: "wenlan".into(),
            brief_version: 3,
            applied: vec![BriefAppliedMutation {
                mutation_index: Some(0),
                kind: BriefMutationKind::Add,
                item_id: Some("item-new".into()),
                version: Some(1),
            }],
            conflicts: vec![BriefMutationConflict {
                mutation_index: Some(1),
                item_id: Some("item-old".into()),
                reason: BriefConflictReason::VersionMismatch,
                current_version: Some(7),
            }],
            projection_path: None,
            warnings: Vec::new(),
        };

        let value = serde_json::to_value(receipt).unwrap();
        assert_eq!(value["applied"][0]["kind"], "add");
        assert_eq!(value["conflicts"][0]["reason"], "version_mismatch");
    }
}
