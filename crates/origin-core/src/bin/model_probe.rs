// SPDX-License-Identifier: Apache-2.0
//! Quick probe: run a single CLASSIFY_MEMORY prompt against a model and print RAW output.

use origin_core::engine::LlmEngine;
use origin_core::prompts::PromptRegistry;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path/to/model.gguf>", args[0]);
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let prompts = PromptRegistry::default();
    let engine = LlmEngine::new(&model_path, prompts.clone())?;

    let system = prompts.classify_memory.clone();
    let user = "I prefer tabs over spaces for indentation.";

    let prompt = format!(
        "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        system, user
    );

    println!("=== Probe: CLASSIFY_MEMORY ===");
    println!("Input: {}", user);
    println!();
    println!("--- Running inference ---");

    let start = Instant::now();
    let raw = engine.run_inference_raw(&prompt, 256, 0.0, 30, 4096);
    let elapsed = start.elapsed();

    println!();
    println!("--- Raw output ({:?}) ---", elapsed);
    match raw {
        Some(s) => {
            println!("{}", s);
            println!();
            println!("--- Length: {} chars ---", s.len());
        }
        None => println!("(empty/None)"),
    }

    Ok(())
}
