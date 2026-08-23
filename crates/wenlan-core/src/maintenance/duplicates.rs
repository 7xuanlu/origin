// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use crate::db::{NearDuplicatePairRead, NearDuplicateSliceReader};
use crate::error::WenlanError;

const HIGH_SOURCE_OVERLAP_MIN: usize = 2;
const HIGH_SOURCE_OVERLAP_RATIO: f64 = 0.67;
pub(super) const AUTOMATIC_PAIR_BUDGET: usize = 128;
pub(super) const AUTOMATIC_SOURCE_CAP: usize = 256;

#[derive(Debug, Clone)]
pub(super) struct NearDuplicatePair {
    pub(super) left_id: String,
    pub(super) right_id: String,
    pub(super) similarity: Option<f64>,
    pub(super) source_overlap: usize,
    pub(super) source_overlap_ratio: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(super) struct NearDuplicateCursor {
    pub(super) left_id: String,
    pub(super) right_id: String,
}

#[derive(Debug)]
pub(super) struct NearDuplicateSlice {
    pub(super) candidate: Option<NearDuplicatePair>,
    pub(super) next_cursor: Option<NearDuplicateCursor>,
    pub(super) more: bool,
    pub(super) pairs_examined: usize,
    pub(super) pages_examined: usize,
    pub(super) source_rows_examined: usize,
    pub(super) truncated: bool,
}

/// Scan a stable keyset window of Page pairs. Distance is computed only after
/// the pair window has been bounded. Source evidence is independently capped
/// per Page; an overflow is never treated as partial overlap because that
/// could create false-positive merge cards.
pub(super) async fn evaluate_near_duplicate_slice(
    reader: &NearDuplicateSliceReader<'_>,
    pair_rows: Vec<NearDuplicatePairRead>,
    page_match_threshold: f64,
) -> Result<NearDuplicateSlice, WenlanError> {
    let mut fallback_sources = HashMap::<String, Vec<String>>::new();
    for pair in &pair_rows {
        if !pair.eligible {
            continue;
        }
        fallback_sources
            .entry(pair.left_id.clone())
            .or_insert_with(|| pair.left_fallback_sources.clone());
        fallback_sources
            .entry(pair.right_id.clone())
            .or_insert_with(|| pair.right_fallback_sources.clone());
    }

    let mut source_sets = HashMap::<String, (HashSet<String>, bool)>::new();
    let mut source_rows_examined = 0usize;
    let mut truncated = false;
    for (page_id, fallback) in fallback_sources {
        let mut source_ids = reader
            .load_bounded_page_source_ids(&page_id, AUTOMATIC_SOURCE_CAP + 1)
            .await?;
        if source_ids.is_empty() {
            source_ids.extend(fallback.into_iter().take(AUTOMATIC_SOURCE_CAP + 1));
        }
        source_rows_examined += source_ids.len();
        let page_truncated = source_ids.len() > AUTOMATIC_SOURCE_CAP;
        truncated |= page_truncated;
        source_ids.truncate(AUTOMATIC_SOURCE_CAP);
        source_sets.insert(page_id, (source_ids.into_iter().collect(), page_truncated));
    }

    let pages_examined = source_sets.len();
    let pairs_examined = pair_rows.len();
    let mut next_cursor = None;
    let mut candidate = None;
    let mut stopped_early = false;
    for (index, pair) in pair_rows.iter().enumerate() {
        next_cursor = Some(NearDuplicateCursor {
            left_id: pair.left_id.clone(),
            right_id: pair.right_id.clone(),
        });
        if !pair.eligible {
            continue;
        }
        let similarity = (!pair.left_embedding.is_empty() && !pair.right_embedding.is_empty())
            .then(|| crate::db::cosine_similarity(&pair.left_embedding, &pair.right_embedding));
        let (left_sources, left_truncated) = source_sets
            .get(&pair.left_id)
            .expect("every bounded pair has a left source set");
        let (right_sources, right_truncated) = source_sets
            .get(&pair.right_id)
            .expect("every bounded pair has a right source set");
        let (source_overlap, source_overlap_ratio) = if *left_truncated || *right_truncated {
            (0, 0.0)
        } else {
            let overlap = left_sources.intersection(right_sources).count();
            let smaller = left_sources.len().min(right_sources.len());
            let ratio = if smaller == 0 {
                0.0
            } else {
                overlap as f64 / smaller as f64
            };
            (overlap, ratio)
        };
        let embedding_match = similarity.is_some_and(|value| value >= page_match_threshold);
        let source_match = source_overlap >= HIGH_SOURCE_OVERLAP_MIN
            && source_overlap_ratio >= HIGH_SOURCE_OVERLAP_RATIO;
        if embedding_match || source_match {
            candidate = Some(NearDuplicatePair {
                left_id: pair.left_id.clone(),
                right_id: pair.right_id.clone(),
                similarity,
                source_overlap,
                source_overlap_ratio,
            });
            stopped_early = index + 1 < pair_rows.len();
            break;
        }
    }
    let more = stopped_early || pair_rows.len() == AUTOMATIC_PAIR_BUDGET;

    Ok(NearDuplicateSlice {
        candidate,
        next_cursor,
        more,
        pairs_examined,
        pages_examined,
        source_rows_examined,
        truncated,
    })
}
