// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Context, Result};
use same_file::Handle;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

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
    source_identity: Arc<Handle>,
}

/// Mutation-refusal evidence without exposing the underlying libSQL handle.
pub struct M5MutationProbe {
    pub dml_refused: bool,
    pub ddl_refused: bool,
}

impl M5PageSizeSnapshotDb {
    pub async fn open(path: &Path) -> Result<Self> {
        let requested_file = File::open(path).context("open requested page-size database")?;
        if !requested_file
            .metadata()
            .context("inspect requested page-size database")?
            .is_file()
        {
            bail!("--db must name a regular file");
        }
        let requested_identity =
            Handle::from_file(requested_file).context("hold requested database identity")?;

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

        let mut databases = connection
            .query("PRAGMA database_list", ())
            .await
            .context("query connected database identity")?;
        let mut main_path = None;
        while let Some(row) = databases
            .next()
            .await
            .context("read connected database identity")?
        {
            let name: String = row.get(1).context("read connected database name")?;
            if name == "main" {
                let path: String = row.get(2).context("read connected database path")?;
                if path.is_empty() {
                    bail!("connected page-size database has no filesystem path");
                }
                main_path = Some(path);
                break;
            }
        }
        let connected_file = File::open(main_path.context("connected database has no main file")?)
            .context("open connected page-size database identity")?;
        let connected_identity = Handle::from_file(connected_file)
            .context("hold connected page-size database identity")?;
        if connected_identity != requested_identity {
            bail!("--db changed while opening the page-size database");
        }

        Ok(Self {
            connection,
            source_identity: Arc::new(connected_identity),
        })
    }

    pub(crate) fn source_identity(&self) -> Arc<Handle> {
        Arc::clone(&self.source_identity)
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
