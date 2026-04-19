// SPDX-License-Identifier: AGPL-3.0-only
//! Store quality metrics — deterministic checks on how well memories are stored.
//! These run against a real Origin DB, not ephemeral test DBs.

use crate::db::MemoryDB;
use crate::error::OriginError;
use serde::Serialize;

/// Aggregated store quality report.
#[derive(Debug, Clone, Serialize)]
pub struct StoreQualityReport {
    pub total_memories: u64,
    pub schema_fill_rate: SchemaFillRate,
    pub type_coverage: TypeCoverage,
    pub entity_linkage_rate: f64,
    pub dedup_miss_rate: f64,
    pub confidence_calibration: f64,
    pub domain_consistency: DomainConsistency,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchemaFillRate {
    pub overall: f64,
    pub by_type: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeCoverage {
    pub entropy: f64,
    pub max_entropy: f64,
    pub normalized: f64,
    pub distribution: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DomainConsistency {
    pub suspicious_pairs: Vec<(String, String)>,
    pub total_domains: u64,
}

/// Run all store quality checks against a MemoryDB.
pub async fn run_store_quality(db: &MemoryDB) -> Result<StoreQualityReport, OriginError> {
    let total = total_memories(db).await?;
    if total == 0 {
        return Ok(StoreQualityReport {
            total_memories: 0,
            schema_fill_rate: SchemaFillRate {
                overall: 0.0,
                by_type: vec![],
            },
            type_coverage: TypeCoverage {
                entropy: 0.0,
                max_entropy: 1.609,
                normalized: 0.0,
                distribution: vec![],
            },
            entity_linkage_rate: 0.0,
            dedup_miss_rate: 0.0,
            confidence_calibration: 0.0,
            domain_consistency: DomainConsistency {
                suspicious_pairs: vec![],
                total_domains: 0,
            },
        });
    }

    Ok(StoreQualityReport {
        total_memories: total,
        schema_fill_rate: schema_fill_rate(db).await?,
        type_coverage: type_coverage(db).await?,
        entity_linkage_rate: entity_linkage_rate(db).await?,
        dedup_miss_rate: dedup_miss_rate(db).await?,
        confidence_calibration: confidence_calibration(db).await?,
        domain_consistency: domain_consistency(db).await?,
    })
}

async fn total_memories(db: &MemoryDB) -> Result<u64, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*) FROM memories WHERE source = 'memory' AND pending_revision = 0",
            libsql::params![],
        )
        .await
        .map_err(|e| OriginError::Generic(format!("count memories: {e}")))?;
    if let Ok(Some(row)) = rows.next().await {
        Ok(row.get::<u64>(0).unwrap_or(0))
    } else {
        Ok(0)
    }
}

async fn schema_fill_rate(db: &MemoryDB) -> Result<SchemaFillRate, OriginError> {
    let conn = db.conn.lock().await;

    let mut rows = conn
        .query(
            "SELECT COUNT(*), SUM(CASE WHEN structured_fields IS NOT NULL THEN 1 ELSE 0 END) \
         FROM memories WHERE source = 'memory' AND pending_revision = 0",
            libsql::params![],
        )
        .await
        .map_err(|e| OriginError::Generic(format!("schema fill overall: {e}")))?;

    let (total, filled) = if let Ok(Some(row)) = rows.next().await {
        (
            row.get::<u64>(0).unwrap_or(0),
            row.get::<u64>(1).unwrap_or(0),
        )
    } else {
        (0, 0)
    };
    let overall = if total > 0 {
        filled as f64 / total as f64
    } else {
        0.0
    };

    let mut rows = conn
        .query(
            "SELECT COALESCE(memory_type, 'null'), COUNT(*), \
                SUM(CASE WHEN structured_fields IS NOT NULL THEN 1 ELSE 0 END) \
         FROM memories WHERE source = 'memory' AND pending_revision = 0 \
         GROUP BY memory_type ORDER BY COUNT(*) DESC",
            libsql::params![],
        )
        .await
        .map_err(|e| OriginError::Generic(format!("schema fill by_type: {e}")))?;

    let mut by_type = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let mt: String = row.get(0).unwrap_or_default();
        let t: u64 = row.get(1).unwrap_or(0);
        let f: u64 = row.get(2).unwrap_or(0);
        let rate = if t > 0 { f as f64 / t as f64 } else { 0.0 };
        by_type.push((mt, rate));
    }

    Ok(SchemaFillRate { overall, by_type })
}

async fn type_coverage(db: &MemoryDB) -> Result<TypeCoverage, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COALESCE(memory_type, 'null'), COUNT(*) \
         FROM memories WHERE source = 'memory' AND pending_revision = 0 \
         GROUP BY memory_type ORDER BY COUNT(*) DESC",
            libsql::params![],
        )
        .await
        .map_err(|e| OriginError::Generic(format!("type coverage: {e}")))?;

    let mut distribution = Vec::new();
    let mut total = 0u64;
    while let Ok(Some(row)) = rows.next().await {
        let mt: String = row.get(0).unwrap_or_default();
        let count: u64 = row.get(1).unwrap_or(0);
        total += count;
        distribution.push((mt, count));
    }

    let max_entropy = 5.0f64.ln();

    if total == 0 {
        return Ok(TypeCoverage {
            entropy: 0.0,
            max_entropy,
            normalized: 0.0,
            distribution,
        });
    }

    let entropy: f64 = distribution
        .iter()
        .map(|(_, count)| {
            let p = *count as f64 / total as f64;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    let normalized = entropy / max_entropy;

    Ok(TypeCoverage {
        entropy,
        max_entropy,
        normalized,
        distribution,
    })
}

async fn entity_linkage_rate(db: &MemoryDB) -> Result<f64, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT COUNT(*), SUM(CASE WHEN entity_id IS NOT NULL THEN 1 ELSE 0 END) \
         FROM memories WHERE source = 'memory' AND pending_revision = 0",
            libsql::params![],
        )
        .await
        .map_err(|e| OriginError::Generic(format!("entity linkage: {e}")))?;

    if let Ok(Some(row)) = rows.next().await {
        let total: u64 = row.get(0).unwrap_or(0);
        let linked: u64 = row.get(1).unwrap_or(0);
        Ok(if total > 0 {
            linked as f64 / total as f64
        } else {
            0.0
        })
    } else {
        Ok(0.0)
    }
}

async fn dedup_miss_rate(db: &MemoryDB) -> Result<f64, OriginError> {
    let conn = db.conn.lock().await;

    let mut rows = conn.query(
        "SELECT COUNT(*) FROM memories WHERE source = 'memory' AND pending_revision = 0 AND embedding IS NOT NULL",
        libsql::params![],
    ).await.map_err(|e| OriginError::Generic(format!("dedup count: {e}")))?;
    let total: u64 = if let Ok(Some(row)) = rows.next().await {
        row.get(0).unwrap_or(0)
    } else {
        return Ok(0.0);
    };
    if total < 2 {
        return Ok(0.0);
    }

    let sample_size = total.min(50);
    let mut rows = conn.query(
        "SELECT id, source_id FROM memories WHERE source = 'memory' AND pending_revision = 0 AND embedding IS NOT NULL \
         ORDER BY last_modified DESC LIMIT ?1",
        libsql::params![sample_size as i64],
    ).await.map_err(|e| OriginError::Generic(format!("dedup sample: {e}")))?;

    let mut sample_ids: Vec<(String, String)> = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let id: String = row.get(0).unwrap_or_default();
        let source_id: String = row.get(1).unwrap_or_default();
        sample_ids.push((id, source_id));
    }

    let mut near_dup_count = 0u64;
    for (chunk_id, source_id) in &sample_ids {
        let mut nn_rows = conn.query(
            "SELECT c.source_id, vector_distance_cos(c.embedding, (SELECT embedding FROM memories WHERE id = ?1)) as dist \
             FROM memories c \
             WHERE c.source = 'memory' AND c.pending_revision = 0 AND c.source_id != ?2 AND c.embedding IS NOT NULL \
             ORDER BY dist ASC LIMIT 1",
            libsql::params![chunk_id.clone(), source_id.clone()],
        ).await.map_err(|e| OriginError::Generic(format!("dedup nn: {e}")))?;

        if let Ok(Some(row)) = nn_rows.next().await {
            let dist: f64 = row.get(1).unwrap_or(1.0);
            if dist < 0.15 {
                near_dup_count += 1;
            }
        }
    }

    Ok(near_dup_count as f64 / sample_size as f64)
}

async fn confidence_calibration(db: &MemoryDB) -> Result<f64, OriginError> {
    let conn = db.conn.lock().await;
    let mut rows = conn
        .query(
            "SELECT confidence, confirmed FROM memories \
         WHERE source = 'memory' AND pending_revision = 0 AND confidence IS NOT NULL",
            libsql::params![],
        )
        .await
        .map_err(|e| OriginError::Generic(format!("confidence cal: {e}")))?;

    let mut pairs: Vec<(f64, f64)> = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        let confidence: f64 = row.get(0).unwrap_or(0.0);
        let confirmed: f64 = if row.get::<i64>(1).unwrap_or(0) != 0 {
            1.0
        } else {
            0.0
        };
        pairs.push((confidence, confirmed));
    }

    if pairs.len() < 3 {
        return Ok(0.0);
    }

    Ok(spearman_correlation(&pairs))
}

fn spearman_correlation(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    if n < 3.0 {
        return 0.0;
    }

    let rank = |vals: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0; vals.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
                j += 1;
            }
            let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        ranks
    };

    let x_vals: Vec<f64> = pairs.iter().map(|(x, _)| *x).collect();
    let y_vals: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
    let x_ranks = rank(&x_vals);
    let y_ranks = rank(&y_vals);

    let d_sq_sum: f64 = x_ranks
        .iter()
        .zip(y_ranks.iter())
        .map(|(rx, ry)| (rx - ry).powi(2))
        .sum();

    1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0))
}

async fn domain_consistency(db: &MemoryDB) -> Result<DomainConsistency, OriginError> {
    let conn = db.conn.lock().await;

    let mut rows = conn.query(
        "SELECT DISTINCT domain FROM memories WHERE source = 'memory' AND domain IS NOT NULL AND pending_revision = 0",
        libsql::params![],
    ).await.map_err(|e| OriginError::Generic(format!("domain list: {e}")))?;

    let mut domains: Vec<String> = Vec::new();
    while let Ok(Some(row)) = rows.next().await {
        if let Ok(d) = row.get::<String>(0) {
            domains.push(d);
        }
    }
    let total_domains = domains.len() as u64;

    let mut suspicious_pairs = Vec::new();
    for i in 0..domains.len() {
        for j in (i + 1)..domains.len() {
            let a = &domains[i];
            let b = &domains[j];
            let a_lower = a.to_lowercase();
            let b_lower = b.to_lowercase();
            if a_lower.contains(&b_lower) || b_lower.contains(&a_lower) {
                suspicious_pairs.push((a.clone(), b.clone()));
            } else {
                let min_len = a_lower.len().min(b_lower.len());
                if min_len >= 4 {
                    let prefix_len = a_lower
                        .chars()
                        .zip(b_lower.chars())
                        .take_while(|(ca, cb)| ca == cb)
                        .count();
                    if prefix_len as f64 / min_len as f64 > 0.75 {
                        suspicious_pairs.push((a.clone(), b.clone()));
                    }
                }
            }
        }
    }

    Ok(DomainConsistency {
        suspicious_pairs,
        total_domains,
    })
}

impl StoreQualityReport {
    pub fn to_terminal(&self) -> String {
        let mut out = String::new();
        out.push_str("Store Quality Report\n");
        out.push_str("====================\n");
        out.push_str(&format!("Total memories: {}\n\n", self.total_memories));

        out.push_str(&format!(
            "  Schema fill rate:      {:.1}%\n",
            self.schema_fill_rate.overall * 100.0
        ));
        for (mt, rate) in &self.schema_fill_rate.by_type {
            out.push_str(&format!("    {:<20} {:.1}%\n", mt, rate * 100.0));
        }

        out.push_str(&format!(
            "\n  Type coverage:         {:.3} / {:.3} (normalized: {:.1}%)\n",
            self.type_coverage.entropy,
            self.type_coverage.max_entropy,
            self.type_coverage.normalized * 100.0
        ));
        for (mt, count) in &self.type_coverage.distribution {
            out.push_str(&format!("    {:<20} {}\n", mt, count));
        }

        out.push_str(&format!(
            "\n  Entity linkage rate:   {:.1}%\n",
            self.entity_linkage_rate * 100.0
        ));
        out.push_str(&format!(
            "  Dedup miss rate:       {:.1}%\n",
            self.dedup_miss_rate * 100.0
        ));
        out.push_str(&format!(
            "  Confidence calibration: {:.3} (Spearman rho)\n",
            self.confidence_calibration
        ));

        out.push_str(&format!(
            "\n  Domain consistency:    {} domains",
            self.domain_consistency.total_domains
        ));
        if self.domain_consistency.suspicious_pairs.is_empty() {
            out.push_str(" (no suspicious pairs)\n");
        } else {
            out.push_str(&format!(
                ", {} suspicious pairs:\n",
                self.domain_consistency.suspicious_pairs.len()
            ));
            for (a, b) in &self.domain_consistency.suspicious_pairs {
                out.push_str(&format!("    \"{}\" ~ \"{}\"\n", a, b));
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spearman_perfect_positive() {
        let pairs = vec![(0.5, 0.0), (0.7, 0.0), (0.9, 1.0), (1.0, 1.0)];
        let rho = spearman_correlation(&pairs);
        assert!(
            rho > 0.8,
            "Perfect positive should give rho > 0.8, got {}",
            rho
        );
    }

    #[test]
    fn test_spearman_negative() {
        let pairs = vec![(1.0, 0.0), (0.9, 0.0), (0.5, 1.0), (0.3, 1.0)];
        let rho = spearman_correlation(&pairs);
        assert!(
            rho < -0.5,
            "Negative correlation should give rho < -0.5, got {}",
            rho
        );
    }

    #[test]
    fn test_spearman_insufficient_data() {
        let pairs = vec![(0.5, 1.0)];
        assert_eq!(spearman_correlation(&pairs), 0.0);
    }

    #[tokio::test]
    async fn test_store_quality_empty_db() {
        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();
        let report = run_store_quality(&db).await.unwrap();
        assert_eq!(report.total_memories, 0);
        assert_eq!(report.schema_fill_rate.overall, 0.0);
        assert_eq!(report.type_coverage.normalized, 0.0);
    }

    #[tokio::test]
    async fn test_store_quality_with_memories() {
        use crate::sources::RawDocument;

        let tmp = tempfile::tempdir().unwrap();
        let db = MemoryDB::new(tmp.path(), std::sync::Arc::new(crate::events::NoopEmitter))
            .await
            .unwrap();

        let now = chrono::Utc::now().timestamp();
        let docs = vec![
            RawDocument {
                content: "I prefer dark mode in editors".into(),
                source_id: "mem_1".into(),
                source: "memory".into(),
                title: "Preference".into(),
                memory_type: Some("preference".into()),
                domain: Some("tools".into()),
                confirmed: Some(true),
                confidence: Some(0.9),
                structured_fields: Some(r#"{"preference":"dark mode"}"#.into()),
                last_modified: now,
                ..Default::default()
            },
            RawDocument {
                content: "Decided to use Rust for the backend".into(),
                source_id: "mem_2".into(),
                source: "memory".into(),
                title: "Decision".into(),
                memory_type: Some("decision".into()),
                domain: Some("origin".into()),
                confidence: Some(0.7),
                last_modified: now,
                ..Default::default()
            },
            RawDocument {
                content: "Full-stack developer specializing in Rust and TypeScript".into(),
                source_id: "mem_3".into(),
                source: "memory".into(),
                title: "Identity".into(),
                memory_type: Some("identity".into()),
                domain: Some("personal".into()),
                confidence: Some(0.9),
                confirmed: Some(true),
                last_modified: now,
                ..Default::default()
            },
        ];
        db.upsert_documents(docs).await.unwrap();

        let report = run_store_quality(&db).await.unwrap();

        assert_eq!(report.total_memories, 3);
        assert!((report.schema_fill_rate.overall - 1.0 / 3.0).abs() < 0.01);
        assert!(report.type_coverage.entropy > 0.0);
        assert!(report.type_coverage.normalized > 0.0);
        assert_eq!(report.entity_linkage_rate, 0.0);
        assert!(report.dedup_miss_rate < 0.5);

        let terminal = report.to_terminal();
        assert!(terminal.contains("Store Quality Report"));
        assert!(terminal.contains("Total memories: 3"));
    }
}
