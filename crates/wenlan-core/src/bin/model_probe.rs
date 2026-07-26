// SPDX-License-Identifier: Apache-2.0
//! Quick probe: run a single CLASSIFY_MEMORY prompt against a model and print RAW output.

use std::path::PathBuf;
use std::time::Instant;
use wenlan_core::engine::LlmEngine;
use wenlan_core::prompts::PromptRegistry;

fn validate_probe_classification(raw: &str) -> Result<&'static str, String> {
    let result = wenlan_core::llm_provider::parse_classify_response(raw)
        .ok_or_else(|| "expected preference classification in a valid JSON object".to_string())?;

    if result.memory_type == "preference" {
        Ok("preference")
    } else {
        Err(format!(
            "expected preference classification, got {}",
            result.memory_type
        ))
    }
}

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
    let runtime = engine.inference_runtime_info();

    println!("--- Inference backend: {} ---", runtime.backend);
    println!(
        "--- Inference device index: {} ---",
        runtime
            .device_index
            .map_or_else(|| "none".to_string(), |index| index.to_string())
    );
    println!(
        "--- Inference device: {} ---",
        runtime.device.as_deref().unwrap_or("none")
    );
    if let Some(reason) = runtime.fallback_reason.as_deref() {
        println!("--- Inference fallback: {reason} ---");
    }

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
    let raw = raw.ok_or_else(|| std::io::Error::other("model returned no inference output"))?;
    println!("{}", raw);
    println!();
    println!("--- Length: {} chars ---", raw.len());

    let classification = validate_probe_classification(&raw).map_err(std::io::Error::other)?;
    println!("--- Verified classification: {classification} ---");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_preference_classification_passes() {
        let raw = r#"{"memory_type":"preference","domain":"tools","tags":["indentation","tabs"]}"#;

        assert_eq!(validate_probe_classification(raw).unwrap(), "preference");
    }

    #[test]
    fn partial_or_invalid_output_fails() {
        let err = validate_probe_classification(r#"{"memory_type":"pref"#).unwrap_err();

        assert!(err.contains("expected preference"));
    }

    #[test]
    fn wrong_classification_fails() {
        let raw = r#"{"memory_type":"fact","domain":"tools","tags":[]}"#;

        assert!(validate_probe_classification(raw).is_err());
    }
}
