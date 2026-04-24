// SPDX-License-Identifier: AGPL-3.0-only
//! fixture-gen: Generate eval fixtures and run store quality checks.
//!
//! Dev-only. Gated behind the `fixture-gen` feature so release builds
//! skip it (otherwise Tauri bundles every crate binary into Origin.app).
//!
//! Usage:
//!   cargo run --features fixture-gen --bin fixture_gen -- --mode regression --count 6 --out eval/fixtures/gen/regression
//!   cargo run --features fixture-gen --bin fixture_gen -- --mode blind --count 10 --out eval/fixtures/gen/blind
//!   cargo run --features fixture-gen --bin fixture_gen -- --mode store-quality --db-path ~/Library/Application\ Support/origin/memorydb/
//!   cargo run --features fixture-gen --bin fixture_gen -- --help

use origin_lib::eval::gen;
use origin_lib::llm_provider::OnDeviceProvider;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug)]
struct Args {
    mode: String,
    count: usize,
    out_dir: PathBuf,
    db_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("fixture-gen: Generate eval fixtures and run store quality checks\n");
        eprintln!("Usage:");
        eprintln!(
            "  cargo run --bin fixture_gen -- --mode <regression|blind|store-quality> [OPTIONS]\n"
        );
        eprintln!("Modes:");
        eprintln!("  regression     Pipeline-aware adversarial fixtures (requires on-device LLM)");
        eprintln!("  blind          Pipeline-ignorant fixtures (requires on-device LLM)");
        eprintln!("  store-quality  Run store quality metrics against a real Origin DB\n");
        eprintln!("Options:");
        eprintln!("  --count N      Number of fixtures to generate (default: 6 for regression, 10 for blind)");
        eprintln!("  --out DIR      Output directory (default: eval/fixtures/gen/<mode>)");
        eprintln!("  --db-path DIR  Path to Origin DB directory (required for store-quality)");
        std::process::exit(0);
    }

    let mode = get_flag(&args, "--mode")
        .ok_or("--mode is required (regression, blind, or store-quality)")?;

    if mode != "regression" && mode != "blind" && mode != "store-quality" {
        return Err(format!(
            "unknown mode '{mode}' — expected 'regression', 'blind', or 'store-quality'"
        ));
    }

    let default_count = if mode == "regression" { 6 } else { 10 };
    let count = get_flag(&args, "--count")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_count);

    let default_out = format!("eval/fixtures/gen/{}", mode);
    let out_dir = PathBuf::from(get_flag(&args, "--out").unwrap_or(default_out));

    let db_path = get_flag(&args, "--db-path").map(PathBuf::from);

    Ok(Args {
        mode,
        count,
        out_dir,
        db_path,
    })
}

fn get_flag(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!("Run with --help for usage.");
            std::process::exit(1);
        }
    };

    eprintln!("fixture-gen: mode={}", args.mode);

    match args.mode.as_str() {
        "regression" | "blind" => {
            eprintln!("count={}, out={}", args.count, args.out_dir.display());
            eprintln!("Loading on-device LLM (this may download the model on first run)...");
            let provider = match OnDeviceProvider::new() {
                Ok(p) => Arc::new(p) as Arc<dyn origin_lib::llm_provider::LlmProvider>,
                Err(e) => {
                    eprintln!("Failed to initialize LLM: {e}");
                    eprintln!("The on-device model (Qwen3-4B) is required for fixture generation.");
                    std::process::exit(1);
                }
            };

            let result = match args.mode.as_str() {
                "regression" => {
                    gen::generate_regression(&provider, args.count, &args.out_dir).await
                }
                "blind" => gen::generate_blind(&provider, args.count, &args.out_dir).await,
                _ => unreachable!(),
            };

            match result {
                Ok(n) => eprintln!("Generated {n} fixtures in {}", args.out_dir.display()),
                Err(e) => {
                    eprintln!("Generation failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        "store-quality" => {
            let db_path = match &args.db_path {
                Some(p) => p.clone(),
                None => {
                    eprintln!("Error: --db-path is required for store-quality mode");
                    eprintln!(
                        "Example: --db-path ~/Library/Application\\ Support/origin/memorydb/"
                    );
                    std::process::exit(1);
                }
            };

            eprintln!("Opening DB at {}...", db_path.display());
            let emitter: std::sync::Arc<dyn origin_core::events::EventEmitter> =
                std::sync::Arc::new(origin_core::events::NoopEmitter);
            let db = match origin_lib::memory_db::MemoryDB::new(&db_path, emitter).await {
                Ok(db) => db,
                Err(e) => {
                    eprintln!("Failed to open DB at {}: {e}", db_path.display());
                    std::process::exit(1);
                }
            };

            match origin_lib::eval::store_quality::run_store_quality(&db).await {
                Ok(report) => {
                    eprintln!("{}", report.to_terminal());
                }
                Err(e) => {
                    eprintln!("Store quality check failed: {e}");
                    std::process::exit(1);
                }
            }
        }
        _ => unreachable!(),
    }
}
