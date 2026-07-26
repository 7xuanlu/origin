// SPDX-License-Identifier: Apache-2.0
//! M4 community grouping job: short DB phases around pure graph computation.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

use thiserror::Error;

use crate::{
    community_partition::{
        full_partition, incremental_partition, project_grounded_relates, rebind_durable_ids,
        IncrementalConfig, IncrementalPartitionState, PartitionConfig, ProjectionConfig,
        ProjectionInputEdge,
    },
    db::MemoryDB,
};

pub(crate) const COMMUNITY_ALGO_VERSION: &str = "leiden-m4-v1";
pub(crate) const COMMUNITY_PROJECTION_VERSION: &str = "grounded-relates-v1";
const MIN_COMMUNITY_PARTICIPANTS: usize = 10;

#[derive(Debug, Error)]
pub enum CommunityGroupingError {
    #[error(
        "community grouping lease already held for space {space} at generation {input_generation}"
    )]
    LeaseHeld {
        space: String,
        input_generation: i64,
    },
    #[error("community grouping database error: {0}")]
    Database(String),
    #[error("community grouping computation failed: {0}")]
    Compute(String),
}

#[derive(Debug)]
pub struct CommunityGroupingAttempt {
    pub(crate) space: String,
    pub(crate) input_generation: i64,
    pub(crate) published_generation: Option<i64>,
    pub(crate) token: String,
    pub(crate) edges: Vec<ProjectionInputEdge>,
    pub(crate) previous_ids: BTreeMap<String, String>,
    pub(crate) dirty_node_ids: BTreeSet<String>,
    pub(crate) db_mutex_hold: Duration,
}

impl CommunityGroupingAttempt {
    pub fn input_generation(&self) -> i64 {
        self.input_generation
    }
}

#[derive(Debug)]
pub struct CommunityGroupingComputed {
    pub(crate) members: Vec<ComputedCommunityMember>,
    pub(crate) projected_edge_count: usize,
    pub(crate) next_state: IncrementalPartitionState,
}

impl CommunityGroupingComputed {
    pub fn projected_edge_count(&self) -> usize {
        self.projected_edge_count
    }
}

#[derive(Debug)]
pub(crate) struct ComputedCommunityMember {
    pub(crate) node_id: String,
    pub(crate) community_id: String,
}

#[derive(Debug)]
pub struct CommunityGroupingReceipt {
    pub input_generation: i64,
    pub published_generation: i64,
    pub projected_edge_count: usize,
    pub member_count: usize,
    pub db_mutex_hold: Duration,
    pub(crate) next_state: Option<IncrementalPartitionState>,
}

#[derive(Debug)]
pub enum CommunityGroupingOutcome {
    Published(CommunityGroupingReceipt),
    Stale(CommunityGroupingReceipt),
}

#[derive(Debug, Clone)]
struct RuntimeCommunityState {
    published_generation: i64,
    partition: IncrementalPartitionState,
}

#[derive(Debug, Default, Clone)]
pub struct CommunityGroupingRuntime {
    spaces: BTreeMap<String, RuntimeCommunityState>,
}

impl CommunityGroupingRuntime {
    pub fn published_generation(&self, space: &str) -> Option<i64> {
        self.spaces
            .get(space)
            .map(|state| state.published_generation)
    }

    fn matching_partition(
        &self,
        space: &str,
        published_generation: Option<i64>,
    ) -> Option<IncrementalPartitionState> {
        let published_generation = published_generation?;
        self.spaces
            .get(space)
            .filter(|state| state.published_generation == published_generation)
            .map(|state| state.partition.clone())
    }

    fn install(
        &mut self,
        space: String,
        published_generation: i64,
        partition: IncrementalPartitionState,
    ) {
        self.spaces.insert(
            space,
            RuntimeCommunityState {
                published_generation,
                partition,
            },
        );
    }

    pub(crate) fn adopt_space_from(&mut self, other: &Self, space: &str) {
        if let Some(state) = other.spaces.get(space) {
            self.spaces.insert(space.to_owned(), state.clone());
        }
    }
}

/// Pure graph computation. This function deliberately has no database handle.
pub fn compute_community_grouping(
    attempt: &CommunityGroupingAttempt,
    runtime: &CommunityGroupingRuntime,
) -> Result<CommunityGroupingComputed, CommunityGroupingError> {
    let graph = project_grounded_relates(&attempt.edges, ProjectionConfig::default());
    let dirty_nodes = attempt
        .dirty_node_ids
        .iter()
        .filter_map(|node_id| graph.node_ids().binary_search(node_id).ok())
        .collect::<Vec<_>>();

    let (membership, next_state) = if let Some(prior) =
        runtime.matching_partition(&attempt.space, attempt.published_generation)
    {
        if dirty_nodes.is_empty() {
            full_partition_state(&graph)?
        } else {
            match incremental_partition(&graph, prior, &dirty_nodes, IncrementalConfig::default()) {
                Ok(output) => {
                    let membership = output.partition().membership().to_vec();
                    (membership, output.into_state())
                }
                Err(_) => full_partition_state(&graph)?,
            }
        }
    } else {
        full_partition_state(&graph)?
    };

    let members = if graph.node_ids().len() < MIN_COMMUNITY_PARTICIPANTS {
        Vec::new()
    } else {
        let previous_ids = graph
            .node_ids()
            .iter()
            .map(|node_id| {
                attempt
                    .previous_ids
                    .get(node_id)
                    .cloned()
                    .unwrap_or_else(|| format!("__m4-new-node-{node_id}"))
            })
            .collect::<Vec<_>>();
        let rebound = rebind_durable_ids(&previous_ids, &membership);
        let mut minted = BTreeMap::<String, String>::new();
        graph
            .node_ids()
            .iter()
            .zip(rebound)
            .map(|(node_id, rebound_id)| {
                let community_id = if rebound_id.starts_with("__m4-new-node-")
                    || rebound_id.starts_with("community-m4-new-")
                {
                    minted
                        .entry(rebound_id)
                        .or_insert_with(|| uuid::Uuid::new_v4().to_string())
                        .clone()
                } else {
                    rebound_id
                };
                ComputedCommunityMember {
                    node_id: node_id.clone(),
                    community_id,
                }
            })
            .collect()
    };

    Ok(CommunityGroupingComputed {
        members,
        projected_edge_count: attempt.edges.len(),
        next_state,
    })
}

fn full_partition_state(
    graph: &crate::community_partition::ProjectedGraph,
) -> Result<(Vec<usize>, IncrementalPartitionState), CommunityGroupingError> {
    let output = full_partition(graph, PartitionConfig::default())
        .map_err(|error| CommunityGroupingError::Compute(error.to_string()))?;
    let membership = output.membership().to_vec();
    let state = IncrementalPartitionState::new(graph, &membership)
        .map_err(|error| CommunityGroupingError::Compute(error.to_string()))?;
    Ok((membership, state))
}

/// Production composer. Runtime state is installed only after a successful
/// generation-CAS publication.
pub async fn run_community_grouping_cycle(
    db: &MemoryDB,
    runtime: &mut CommunityGroupingRuntime,
    space: &str,
) -> Result<CommunityGroupingOutcome, CommunityGroupingError> {
    let attempt = db.prepare_community_grouping(space).await?;
    let computed = compute_community_grouping(&attempt, runtime)?;
    let outcome = db.finalize_community_grouping(attempt, computed).await?;
    if let CommunityGroupingOutcome::Published(receipt) = outcome {
        let mut receipt = receipt;
        if let Some(next_state) = receipt.next_state.take() {
            runtime.install(space.to_owned(), receipt.published_generation, next_state);
        }
        Ok(CommunityGroupingOutcome::Published(receipt))
    } else {
        Ok(outcome)
    }
}

impl MemoryDB {
    /// Run at most one dirty space in the existing CommunityDetection phase
    /// slot. Runtime statistics are snapshotted before the async job and only
    /// the successfully published space is merged back afterward.
    pub async fn run_next_community_grouping_cycle(
        &self,
    ) -> Result<Option<CommunityGroupingOutcome>, CommunityGroupingError> {
        let space = {
            let conn = self.conn.lock().await;
            let mut rows = conn
                .query(
                    "SELECT space FROM space_graph_state \
                     WHERE dirty = 1 ORDER BY space LIMIT 1",
                    (),
                )
                .await
                .map_err(|error| {
                    CommunityGroupingError::Database(format!(
                        "select next dirty community space: {error}"
                    ))
                })?;
            rows.next()
                .await
                .map_err(|error| {
                    CommunityGroupingError::Database(format!(
                        "read next dirty community space: {error}"
                    ))
                })?
                .map(|row| {
                    row.get::<String>(0).map_err(|error| {
                        CommunityGroupingError::Database(format!(
                            "decode next dirty community space: {error}"
                        ))
                    })
                })
                .transpose()?
        };
        let Some(space) = space else {
            return Ok(None);
        };

        let mut runtime = self
            .community_grouping_runtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let outcome = match run_community_grouping_cycle(self, &mut runtime, &space).await {
            Ok(outcome) => outcome,
            Err(CommunityGroupingError::LeaseHeld { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        if matches!(outcome, CommunityGroupingOutcome::Published(_)) {
            self.community_grouping_runtime
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .adopt_space_from(&runtime, &space);
        }
        Ok(Some(outcome))
    }
}
