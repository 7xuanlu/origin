// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use std::path::Path;

const AGGREGATE_QUERY: &str = r#"
SELECT
  COALESCE(SUM(CASE WHEN byte_len < 256 THEN 1 ELSE 0 END), 0),
  COALESCE(SUM(CASE WHEN byte_len >= 256 AND byte_len < 512 THEN 1 ELSE 0 END), 0),
  COALESCE(SUM(CASE WHEN byte_len >= 512 AND byte_len < 1024 THEN 1 ELSE 0 END), 0),
  COALESCE(SUM(CASE WHEN byte_len >= 1024 AND byte_len < 2048 THEN 1 ELSE 0 END), 0),
  COALESCE(SUM(CASE WHEN byte_len >= 2048 AND byte_len < 4096 THEN 1 ELSE 0 END), 0),
  COALESCE(SUM(CASE WHEN byte_len >= 4096 AND byte_len < 8192 THEN 1 ELSE 0 END), 0),
  COALESCE(SUM(CASE WHEN byte_len >= 8192 AND byte_len < 16384 THEN 1 ELSE 0 END), 0),
  COALESCE(SUM(CASE WHEN byte_len >= 16384 THEN 1 ELSE 0 END), 0)
FROM (
  SELECT length(CAST(content AS BLOB)) AS byte_len
  FROM pages
  WHERE status = 'active'
)
"#;

/// Opaque, read-only page-size source for the M5 aggregate snapshot.
pub struct M5PageSizeSnapshotDb {
    connection: libsql::Connection,
}

/// Mutation-refusal evidence without exposing the underlying libSQL handle.
pub struct M5MutationProbe {
    pub dml_refused: bool,
    pub ddl_refused: bool,
}

impl M5PageSizeSnapshotDb {
    pub async fn open(path: &Path) -> Result<Self> {
        let database = libsql::Builder::new_local(path)
            .flags(libsql::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .build()
            .await
            .context("open page-size database read-only")?;
        let connection = database
            .connect()
            .context("connect page-size database read-only")?;
        connection
            .execute("PRAGMA query_only = ON", ())
            .await
            .context("enforce query-only page-size connection")?;
        Ok(Self { connection })
    }

    pub async fn fixed_counts(&self) -> Result<[u64; 8]> {
        let mut rows = self
            .connection
            .query(AGGREGATE_QUERY, ())
            .await
            .context("query fixed page-size aggregate")?;
        let row = rows
            .next()
            .await
            .context("read aggregate result")?
            .context("aggregate query returned no row")?;

        let mut counts = [0_u64; 8];
        for (index, count) in counts.iter_mut().enumerate() {
            let value: i64 = row.get(index as i32).context("read aggregate count")?;
            *count = u64::try_from(value).context("aggregate count was negative")?;
        }
        Ok(counts)
    }

    pub async fn mutation_probe_for_test(&self) -> Result<M5MutationProbe> {
        let dml_refused = self
            .connection
            .execute("UPDATE pages SET content = content WHERE 0 = 1", ())
            .await
            .is_err();
        let ddl_refused = self
            .connection
            .execute("CREATE TABLE attack (secret TEXT)", ())
            .await
            .is_err();
        Ok(M5MutationProbe {
            dml_refused,
            ddl_refused,
        })
    }
}
