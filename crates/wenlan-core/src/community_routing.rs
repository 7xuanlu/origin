// SPDX-License-Identifier: Apache-2.0
//! M4 page-to-community routing with two-threshold hysteresis.

use std::collections::BTreeMap;

pub const COMMUNITY_ROUTE_VERSION: &str = "page-community-route-v1";
pub const COMMUNITY_ROUTE_ASSIGN_THRESHOLD: f64 = 0.50;
pub const COMMUNITY_ROUTE_DROP_THRESHOLD: f64 = 0.30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCommunityRoutingReceipt {
    pub pages_considered: usize,
    pub assignments_published: usize,
    pub stale_writes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PageCommunityRouteDecision {
    Assigned { community_id: String, score: f64 },
    Held { community_id: String, score: f64 },
    Dropped { score: f64 },
}

impl PageCommunityRouteDecision {
    pub fn state(&self) -> &'static str {
        match self {
            Self::Assigned { .. } => "assigned",
            Self::Held { .. } => "held",
            Self::Dropped { .. } => "dropped",
        }
    }

    pub fn community_id(&self) -> Option<&str> {
        match self {
            Self::Assigned { community_id, .. } | Self::Held { community_id, .. } => {
                Some(community_id)
            }
            Self::Dropped { .. } => None,
        }
    }

    pub fn score(&self) -> f64 {
        match self {
            Self::Assigned { score, .. } | Self::Held { score, .. } | Self::Dropped { score } => {
                *score
            }
        }
    }
}

/// Apply the authored M4 thresholds exactly:
/// - score >= 0.50 assigns the current arg-max;
/// - score in [0.30, 0.50) holds a prior assignment;
/// - score < 0.30, no candidate, or no prior to hold drops.
pub fn decide_page_community_route(
    prior_community_id: Option<&str>,
    best_candidate: Option<(&str, f64)>,
) -> PageCommunityRouteDecision {
    let Some((candidate_id, score)) = best_candidate else {
        return PageCommunityRouteDecision::Dropped { score: 0.0 };
    };
    if score >= COMMUNITY_ROUTE_ASSIGN_THRESHOLD {
        return PageCommunityRouteDecision::Assigned {
            community_id: candidate_id.to_owned(),
            score,
        };
    }
    if score >= COMMUNITY_ROUTE_DROP_THRESHOLD {
        return match prior_community_id {
            Some(community_id) => PageCommunityRouteDecision::Held {
                community_id: community_id.to_owned(),
                score,
            },
            None => PageCommunityRouteDecision::Dropped { score },
        };
    }
    PageCommunityRouteDecision::Dropped { score }
}

pub fn community_embedding_centroids(
    members: impl IntoIterator<Item = (String, String, Vec<f32>)>,
) -> BTreeMap<String, Vec<f32>> {
    let mut sums = BTreeMap::<String, (Vec<f64>, usize)>::new();
    for (_, community_id, embedding) in members {
        if embedding.is_empty() {
            continue;
        }
        let entry = sums
            .entry(community_id)
            .or_insert_with(|| (vec![0.0; embedding.len()], 0));
        if entry.0.len() != embedding.len() {
            continue;
        }
        for (sum, value) in entry.0.iter_mut().zip(embedding) {
            *sum += f64::from(value);
        }
        entry.1 += 1;
    }
    sums.into_iter()
        .filter_map(|(community_id, (sum, count))| {
            (count > 0).then(|| {
                (
                    community_id,
                    sum.into_iter()
                        .map(|value| (value / count as f64) as f32)
                        .collect(),
                )
            })
        })
        .collect()
}

pub fn nearest_community_centroid(
    embedding: &[f32],
    centroids: &BTreeMap<String, Vec<f32>>,
) -> Option<(String, f64)> {
    centroids
        .iter()
        .filter(|(_, centroid)| embedding.len() == centroid.len() && !embedding.is_empty())
        .map(|(community_id, centroid)| {
            (
                community_id.clone(),
                crate::db::cosine_similarity(embedding, centroid),
            )
        })
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_three_way_hysteresis_assigns_holds_and_drops() {
        assert_eq!(
            decide_page_community_route(None, Some(("c-new", 0.50))),
            PageCommunityRouteDecision::Assigned {
                community_id: "c-new".to_string(),
                score: 0.50,
            }
        );
        assert_eq!(
            decide_page_community_route(Some("c-old"), Some(("c-new", 0.30))),
            PageCommunityRouteDecision::Held {
                community_id: "c-old".to_string(),
                score: 0.30,
            }
        );
        assert_eq!(
            decide_page_community_route(Some("c-old"), Some(("c-new", 0.299_999))),
            PageCommunityRouteDecision::Dropped { score: 0.299_999 }
        );
    }
}
