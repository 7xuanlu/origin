// SPDX-License-Identifier: Apache-2.0
//! Versioned M4 community read and review surfaces.

use serde::{Deserialize, Serialize};

pub const COMMUNITY_READ_SCHEMA_VERSION: &str = "community-read-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CommunityReadScope {
    Global,
    Space { name: String },
    Uncategorized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunitySummary {
    pub community_id: String,
    pub space: String,
    pub display_name: Option<String>,
    pub member_count: u64,
    pub published_generation: i64,
    pub algo_version: String,
    pub projection_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunityMember {
    pub space: String,
    pub node_id: String,
    pub node_kind: String,
    pub community_id: String,
    pub published_generation: i64,
    pub attachment: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageCommunityAssignmentState {
    Assigned,
    Held,
    Dropped,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunityPageAssignment {
    pub page_id: String,
    pub space: String,
    pub community_id: Option<String>,
    pub state: PageCommunityAssignmentState,
    pub score: f64,
    pub page_version: i64,
    pub community_published_generation: i64,
    pub route_version: String,
    pub updated_at: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunityListResponse {
    pub schema_version: String,
    pub scope: CommunityReadScope,
    pub communities: Vec<CommunitySummary>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunityMemberCursor {
    pub space: String,
    pub node_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunityMembersResponse {
    pub schema_version: String,
    pub scope: CommunityReadScope,
    pub members: Vec<CommunityMember>,
    pub next_cursor: Option<CommunityMemberCursor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommunityPageAssignmentsResponse {
    pub schema_version: String,
    pub scope: CommunityReadScope,
    pub page_assignments: Vec<CommunityPageAssignment>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommunityProposalAction {
    CommunitySplit,
    CommunityMerge,
    CommunityRename,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum CommunityProposalPayload {
    CommunitySplit {
        space: String,
        source_generation: i64,
        subject_id: String,
        old_community_ids: Vec<String>,
        new_community_ids: Vec<String>,
        proposed_display_name: Option<String>,
    },
    CommunityMerge {
        space: String,
        source_generation: i64,
        subject_id: String,
        old_community_ids: Vec<String>,
        new_community_ids: Vec<String>,
        proposed_display_name: Option<String>,
    },
    CommunityRename {
        space: String,
        source_generation: i64,
        subject_id: String,
        proposed_display_name: String,
    },
}

impl CommunityProposalPayload {
    pub fn action(&self) -> CommunityProposalAction {
        match self {
            Self::CommunitySplit { .. } => CommunityProposalAction::CommunitySplit,
            Self::CommunityMerge { .. } => CommunityProposalAction::CommunityMerge,
            Self::CommunityRename { .. } => CommunityProposalAction::CommunityRename,
        }
    }

    pub fn space(&self) -> &str {
        match self {
            Self::CommunitySplit { space, .. }
            | Self::CommunityMerge { space, .. }
            | Self::CommunityRename { space, .. } => space,
        }
    }

    pub fn source_generation(&self) -> i64 {
        match self {
            Self::CommunitySplit {
                source_generation, ..
            }
            | Self::CommunityMerge {
                source_generation, ..
            }
            | Self::CommunityRename {
                source_generation, ..
            } => *source_generation,
        }
    }

    pub fn subject_id(&self) -> &str {
        match self {
            Self::CommunitySplit { subject_id, .. }
            | Self::CommunityMerge { subject_id, .. }
            | Self::CommunityRename { subject_id, .. } => subject_id,
        }
    }

    pub fn proposed_display_name(&self) -> Option<&str> {
        match self {
            Self::CommunitySplit {
                proposed_display_name,
                ..
            }
            | Self::CommunityMerge {
                proposed_display_name,
                ..
            } => proposed_display_name.as_deref(),
            Self::CommunityRename {
                proposed_display_name,
                ..
            } => Some(proposed_display_name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunityProposalSummary {
    pub id: String,
    pub action: CommunityProposalAction,
    pub payload: CommunityProposalPayload,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListCommunityProposalsResponse {
    pub proposals: Vec<CommunityProposalSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunityProposalAcceptResponse {
    pub id: String,
    pub action: CommunityProposalAction,
    pub display_name_updated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommunityProposalRejectResponse {
    pub id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn community_wire_is_versioned_and_proposal_payloads_are_closed() {
        let response = CommunityListResponse {
            schema_version: COMMUNITY_READ_SCHEMA_VERSION.to_string(),
            scope: CommunityReadScope::Space {
                name: "work".to_string(),
            },
            communities: Vec::new(),
            next_cursor: None,
        };
        assert_eq!(
            serde_json::to_value(response).unwrap()["schema_version"],
            COMMUNITY_READ_SCHEMA_VERSION
        );

        let unknown = serde_json::json!({
            "action": "community_rename",
            "space": "work",
            "source_generation": 1,
            "subject_id": "c1",
            "proposed_display_name": "Work",
            "unexpected": true
        });
        assert!(serde_json::from_value::<CommunityProposalPayload>(unknown).is_err());
    }
}
