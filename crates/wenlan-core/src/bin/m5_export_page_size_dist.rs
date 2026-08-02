// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use wenlan_core::eval::m5_bench_corpus::{
    distribution_from_fixed_counts, open_page_size_db_read_only,
};

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

#[derive(Debug)]
struct Args {
    db: PathBuf,
    output: PathBuf,
    overwrite: bool,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error:#}");
        std::process::exit(2);
    }
}

async fn run() -> Result<()> {
    let args = parse_args(std::env::args_os().skip(1))?;
    if !args.db.is_file() {
        bail!("--db must name an existing database file");
    }
    reject_database_output_alias(&args.db, &args.output)?;

    let connection = open_page_size_db_read_only(&args.db).await?;
    let mut rows = connection
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

    let distribution = distribution_from_fixed_counts(counts)?;
    let output = distribution.to_canonical_json_bytes()?;
    atomic_write(&args.output, &output, args.overwrite)?;
    println!(
        "wrote aggregate distribution: sample_size={} buckets={}",
        distribution.sample_size,
        distribution.buckets.len()
    );
    Ok(())
}

fn reject_database_output_alias(db: &Path, output: &Path) -> Result<()> {
    let canonical_db = std::fs::canonicalize(db).context("canonicalize --db")?;
    let canonical_output = match std::fs::symlink_metadata(output) {
        Ok(_) => std::fs::canonicalize(output).context("canonicalize existing --output")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let file_name = output
                .file_name()
                .context("--output must include a file name")?;
            let parent = output.parent().unwrap_or_else(|| Path::new("."));
            std::fs::canonicalize(parent)
                .context("canonicalize --output parent")?
                .join(file_name)
        }
        Err(error) => return Err(error).context("inspect --output"),
    };
    if canonical_db == canonical_output {
        bail!("--output must not alias --db");
    }
    Ok(())
}

fn parse_args(args: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Args> {
    let mut db = None;
    let mut output = None;
    let mut overwrite = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--db") => {
                if db.is_some() {
                    bail!("--db may be supplied only once");
                }
                db = Some(PathBuf::from(args.next().context("--db requires a path")?));
            }
            Some("--output") => {
                if output.is_some() {
                    bail!("--output may be supplied only once");
                }
                output = Some(PathBuf::from(
                    args.next().context("--output requires a path")?,
                ));
            }
            Some("--overwrite") => overwrite = true,
            _ => bail!("usage: m5_export_page_size_dist --db PATH --output PATH [--overwrite]"),
        }
    }
    Ok(Args {
        db: db.context("--db is required")?,
        output: output.context("--output is required")?,
        overwrite,
    })
}

fn atomic_write(path: &Path, bytes: &[u8], overwrite: bool) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        bail!("output parent directory does not exist");
    }
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .context("create same-directory temporary output")?;
    temporary
        .write_all(bytes)
        .context("write temporary aggregate output")?;
    temporary
        .as_file()
        .sync_all()
        .context("sync temporary aggregate output")?;
    if overwrite {
        temporary
            .persist(path)
            .context("atomically replace output")?;
    } else {
        temporary
            .persist_noclobber(path)
            .context("output exists; pass --overwrite to replace it")?;
    }
    Ok(())
}
