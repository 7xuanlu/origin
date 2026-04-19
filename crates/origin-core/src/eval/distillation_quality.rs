// SPDX-License-Identifier: AGPL-3.0-only
//! Distillation quality metrics — evaluate how well the refinery distills memories.
//! Runs against real Origin DB distilled memories.

use crate::db::MemoryDB;
use crate::error::OriginError;
use serde::Serialize;

/// Aggregated distillation quality report.
#[derive(Debug, Clone, Serialize)]
pub struct DistillationQualityReport {
    pub total_distilled: u64,
    pub total_sources_archived: u64,
    pub avg_faithfulness: f64,
    pub avg_compression_ratio: f64,
    pub format_clean_rate: f64,
    pub completeness_rate: f64,
    pub per_memory: Vec<DistilledMemoryEval>,
}

/// Per-memory evaluation.
#[derive(Debug, Clone, Serialize)]
pub struct DistilledMemoryEval {
    pub source_id: String,
    pub content_preview: String,
    pub faithfulness: f64,
    pub compression_ratio: f64,
    pub format_clean: bool,
    pub complete: bool,
    pub issues: Vec<String>,
}

/// Run distillation quality evaluation against the current DB.
pub async fn evaluate_distillation_quality(
    db: &MemoryDB,
) -> Result<DistillationQualityReport, OriginError> {
    let conn = db.conn.lock().await;

    // Fetch all distilled memories with their superseded source
    let mut rows = conn
        .query(
            "SELECT m.source_id, m.content, m.supersedes \
         FROM memories m \
         WHERE m.source = 'memory' AND m.source_id LIKE 'merged_%' AND m.chunk_index = 0",
            (),
        )
        .await
        .map_err(|e| OriginError::VectorDb(format!("distill eval fetch: {}", e)))?;

    struct MergedRow {
        source_id: String,
        content: String,
        supersedes: Option<String>,
    }

    let mut merged: Vec<MergedRow> = Vec::new();
    while let Some(row) = rows
        .next()
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?
    {
        merged.push(MergedRow {
            source_id: row
                .get(0)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            content: row
                .get(1)
                .map_err(|e| OriginError::VectorDb(e.to_string()))?,
            supersedes: row.get(2).unwrap_or(None),
        });
    }
    drop(rows);

    // Count archived sources
    let mut arch_rows = conn.query(
        "SELECT COUNT(DISTINCT source_id) FROM memories WHERE source = 'memory' AND supersede_mode = 'archive' AND chunk_index = 0",
        (),
    ).await.map_err(|e| OriginError::VectorDb(format!("distill eval archive count: {}", e)))?;
    let total_sources_archived = arch_rows
        .next()
        .await
        .map_err(|e| OriginError::VectorDb(e.to_string()))?
        .map(|r| r.get::<u64>(0).unwrap_or(0))
        .unwrap_or(0);

    // For each merged memory, find its source content and evaluate
    let mut evals: Vec<DistilledMemoryEval> = Vec::new();

    for m in &merged {
        let mut issues: Vec<String> = Vec::new();

        // Get source content (the superseded memory)
        let source_content = if let Some(ref sid) = m.supersedes {
            let mut src_rows = conn
                .query(
                    "SELECT content FROM memories WHERE source_id = ?1 AND chunk_index = 0",
                    libsql::params![sid.to_string()],
                )
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?;
            src_rows
                .next()
                .await
                .map_err(|e| OriginError::VectorDb(e.to_string()))?
                .map(|r| r.get::<String>(0).unwrap_or_default())
                .unwrap_or_default()
        } else {
            String::new()
        };

        // Faithfulness: cosine similarity between output and source embeddings
        let faithfulness = if !source_content.is_empty() {
            let texts = vec![m.content.clone(), source_content.clone()];
            match db.generate_embeddings(&texts) {
                Ok(embs) if embs.len() == 2 => crate::db::cosine_similarity(&embs[0], &embs[1]),
                _ => 0.0,
            }
        } else {
            0.0
        };

        if faithfulness < 0.6 {
            issues.push(format!("low faithfulness: {:.2}", faithfulness));
        }

        // Compression ratio: output chars / source chars (lower = more compression)
        let compression_ratio = if !source_content.is_empty() {
            m.content.len() as f64 / source_content.len() as f64
        } else {
            1.0
        };

        // Format cleanliness checks
        let has_prefix = m.content.starts_with("Memory ")
            || m.content.starts_with("- [")
            || m.content.starts_with("---");
        let has_separator = m.content.contains("\n---\n");
        let has_repetition = {
            let chars: String = m.content.chars().take(40).collect();
            chars.len() >= 40 && m.content[chars.len()..].contains(&chars)
        };
        let format_clean = !has_prefix && !has_separator && !has_repetition;

        if has_prefix {
            issues.push("echoed input prefix".into());
        }
        if has_separator {
            issues.push("separator leak".into());
        }
        if has_repetition {
            issues.push("content repetition".into());
        }

        // Completeness: ends with punctuation, has 2+ sentences
        let ends_clean = m.content.trim().ends_with(['.', '!', '?', '"', ')', ']']);
        let sentence_count = m.content.matches(['.', '!', '?']).count();
        let complete = ends_clean && sentence_count >= 2;

        if !ends_clean {
            issues.push("truncated (no ending punctuation)".into());
        }
        if sentence_count < 2 {
            issues.push(format!("too short ({} sentences)", sentence_count));
        }

        evals.push(DistilledMemoryEval {
            source_id: m.source_id.clone(),
            content_preview: m.content.chars().take(100).collect(),
            faithfulness,
            compression_ratio,
            format_clean,
            complete,
            issues,
        });
    }
    drop(conn);

    let total = evals.len() as f64;
    let avg_faithfulness = if total > 0.0 {
        evals.iter().map(|e| e.faithfulness).sum::<f64>() / total
    } else {
        0.0
    };
    let avg_compression_ratio = if total > 0.0 {
        evals.iter().map(|e| e.compression_ratio).sum::<f64>() / total
    } else {
        0.0
    };
    let format_clean_rate = if total > 0.0 {
        evals.iter().filter(|e| e.format_clean).count() as f64 / total
    } else {
        0.0
    };
    let completeness_rate = if total > 0.0 {
        evals.iter().filter(|e| e.complete).count() as f64 / total
    } else {
        0.0
    };

    Ok(DistillationQualityReport {
        total_distilled: evals.len() as u64,
        total_sources_archived,
        avg_faithfulness,
        avg_compression_ratio,
        format_clean_rate,
        completeness_rate,
        per_memory: evals,
    })
}
