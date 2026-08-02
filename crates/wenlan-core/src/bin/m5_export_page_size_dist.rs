// SPDX-License-Identifier: Apache-2.0

use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use wenlan_core::db::M5PageSizeSnapshotDb;
use wenlan_core::eval::m5_bench_corpus::distribution_from_fixed_counts;
use wenlan_core::eval::m5_snapshot_io::prepare_m5_snapshot;

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
    let database = M5PageSizeSnapshotDb::open(&args.db).await?;
    let snapshot = prepare_m5_snapshot(&args.output, args.overwrite)?;
    let counts = database.fixed_counts().await?;

    let distribution = distribution_from_fixed_counts(counts)?;
    let output = distribution.to_canonical_json_bytes()?;
    snapshot.write(&output)?;
    println!(
        "wrote aggregate distribution: sample_size={} buckets={}",
        distribution.sample_size,
        distribution.buckets.len()
    );
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
