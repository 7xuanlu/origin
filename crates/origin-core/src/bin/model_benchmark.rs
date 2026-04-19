// SPDX-License-Identifier: Apache-2.0
#![allow(clippy::type_complexity)]
//! model-benchmark: Evaluate a GGUF model against Origin's P0 prompts.
//!
//! Usage:
//!   cargo run --release -p origin-core --bin model_benchmark -- <path/to/model.gguf> [<output.json>]
//!
//! Scores structural quality and latency for:
//!   - CLASSIFY_MEMORY (JSON parse, correct enum, domain present)
//!   - EXTRACT_KNOWLEDGE_GRAPH (JSON parse, entity structure)
//!   - DETECT_CONTRADICTION (correct label accuracy)
//!   - DISTILL_CONCEPT (structural: has TLDR, headers, sources)

use origin_core::engine::LlmEngine;
use origin_core::prompts::PromptRegistry;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct PromptResult {
    name: String,
    passed: usize,
    total: usize,
    avg_latency_ms: u128,
    samples: Vec<SampleResult>,
}

#[derive(Debug, Serialize)]
struct SampleResult {
    input: String,
    output: String,
    raw_output: String,
    passed: bool,
    reasons: Vec<String>,
    latency_ms: u128,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    model_path: String,
    model_size_bytes: u64,
    total_score: f64,
    prompts: Vec<PromptResult>,
}

// ---- Test fixtures ----

const CLASSIFY_CASES: &[&str] = &[
    "I'm a senior Rust engineer at Meta working on Origin, a personal knowledge wiki app.",
    "I prefer tabs over spaces for indentation because it's more configurable.",
    "Switched from PostgreSQL to libSQL for the embedded use case and better vector support.",
    "The libSQL database schema uses F32_BLOB(384) for vector columns.",
    "Goal: ship BYOK model routing in Origin by end of week.",
    "My favorite coffee is oat milk flat white from Blue Bottle.",
    "Chose Tauri over Electron because it uses the system webview and has smaller bundle size.",
    "React Query v5 uses isPending instead of isLoading for new queries.",
    "Need to finish the eval harness before running benchmarks on Qwen3.5.",
    "Born in Taiwan, moved to the US in 2018, now based in San Francisco.",
];

const CLASSIFY_EXPECTED_TYPES: &[&str] = &[
    "identity",   // 0
    "preference", // 1
    "decision",   // 2
    "fact",       // 3
    "goal",       // 4
    "preference", // 5
    "decision",   // 6
    "fact",       // 7
    "goal",       // 8
    "identity",   // 9
];

const EXTRACT_CASES: &[&str] = &[
    "Alice from the ML team helped debug the tokenizer bug in Qwen3-4B. She found that the BOS token wasn't being added properly.",
    "Installed Ollama on my MacBook to test Gemma 4 E4B locally. Runs at about 30 tok/s on M2 Pro.",
    "Had a call with Sarah and Tom about the Origin roadmap. Decided to prioritize BYOK over cloud sync for Q2.",
    "Reading Karpathy's blog post about LLM knowledge bases. He recommends storing notes as markdown files and using Claude Code to query them.",
    "Used Rust's tokio runtime for the HTTP server. Axum 0.8 made the routing much cleaner than manual Hyper.",
];

const EXTRACT_EXPECTED_ENTITIES: &[&[&str]] = &[
    &["Alice", "Qwen3-4B", "ML team"],
    &["Ollama", "MacBook", "Gemma 4", "M2"],
    &["Sarah", "Tom", "Origin", "BYOK"],
    &["Karpathy", "Claude Code"],
    &["Rust", "tokio", "Axum"],
];

const CONTRADICTION_PAIRS: &[(&str, &str, &str)] = &[
    (
        "I prefer Rust for all backend work because of memory safety.",
        "I've been writing Go for all my new backend services.",
        "CONTRADICTS",
    ),
    (
        "The Origin database is stored in libSQL at ~/Library/Application Support/origin/",
        "Origin uses libSQL for its local database.",
        "CONSISTENT",
    ),
    (
        "I live in San Francisco.",
        "I moved from San Francisco to Seattle last month.",
        "CONTRADICTS",
    ),
    (
        "Working on BYOK feature this week.",
        "The weather in Taipei is hot in summer.",
        "CONSISTENT",
    ),
    (
        "Chose Haiku for routine tasks because it's cheap.",
        "Chose Sonnet for routine tasks because it's more accurate.",
        "CONTRADICTS",
    ),
];

const DISTILL_CLUSTER: &[&str] = &[
    "[mem_1] Origin is a personal knowledge wiki app built with Tauri 2 and Rust.",
    "[mem_2] Origin uses libSQL (Turso's SQLite fork) as its database with F32_BLOB(384) vector columns.",
    "[mem_3] The on-device LLM in Origin is Qwen3-4B-Instruct-2507 running on Metal GPU via llama-cpp-2.",
    "[mem_4] Origin ingests memories from file watching, clipboard, and quick capture, then distills them into concept pages.",
    "[mem_5] The app is AGPL-3.0 licensed and targets macOS as the primary platform.",
    "[mem_6] Origin uses FastEmbed (BGE-Small-EN-v1.5, 384-dim) for semantic search embeddings.",
];

// ---- Scoring logic ----

fn score_classify(output: &str, expected_type: &str) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    let trimmed = extract_json(output);
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(trimmed);
    let Ok(json) = parsed else {
        reasons.push("JSON parse failed".into());
        return (false, reasons);
    };
    let Some(mt) = json.get("memory_type").and_then(|v| v.as_str()) else {
        reasons.push("missing memory_type".into());
        return (false, reasons);
    };
    if mt != expected_type {
        reasons.push(format!("expected {}, got {}", expected_type, mt));
    }
    let has_domain = json
        .get("domain")
        .and_then(|v| v.as_str())
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let has_tags = json
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    if !has_domain {
        reasons.push("missing/empty domain".into());
    }
    if !has_tags {
        reasons.push("missing/empty tags".into());
    }
    let passed = mt == expected_type && has_domain && has_tags;
    (passed, reasons)
}

fn score_extract(output: &str, expected: &[&str]) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    let trimmed = extract_json(output);
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(trimmed);
    let Ok(json) = parsed else {
        reasons.push("JSON parse failed".into());
        return (false, reasons);
    };
    let entities: Vec<String> = if let Some(arr) = json.as_array() {
        arr.iter()
            .filter_map(|item| item.get("entities").and_then(|e| e.as_array()))
            .flatten()
            .filter_map(|e| e.get("name").and_then(|n| n.as_str()).map(String::from))
            .collect()
    } else if let Some(ents) = json.get("entities").and_then(|e| e.as_array()) {
        ents.iter()
            .filter_map(|e| e.get("name").and_then(|n| n.as_str()).map(String::from))
            .collect()
    } else {
        vec![]
    };

    if entities.is_empty() {
        reasons.push("no entities extracted".into());
        return (false, reasons);
    }

    let entities_lower: Vec<String> = entities.iter().map(|e| e.to_lowercase()).collect();
    let found = expected
        .iter()
        .filter(|e| {
            let el = e.to_lowercase();
            entities_lower
                .iter()
                .any(|got| got.contains(&el) || el.contains(got.as_str()))
        })
        .count();
    let recall = found as f64 / expected.len() as f64;
    reasons.push(format!(
        "recall: {}/{} ({:.0}%)",
        found,
        expected.len(),
        recall * 100.0
    ));
    let passed = recall >= 0.5;
    (passed, reasons)
}

fn score_contradiction(output: &str, expected: &str) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    let upper = output.to_uppercase();
    let detected = if upper.contains("CONTRADICTS") {
        "CONTRADICTS"
    } else if upper.contains("SUPERSEDES") {
        "SUPERSEDES"
    } else if upper.contains("CONSISTENT") {
        "CONSISTENT"
    } else {
        "UNKNOWN"
    };
    reasons.push(format!("got {}, expected {}", detected, expected));
    let passed = match expected {
        "CONTRADICTS" => detected == "CONTRADICTS" || detected == "SUPERSEDES",
        "CONSISTENT" => detected == "CONSISTENT",
        _ => false,
    };
    (passed, reasons)
}

fn score_distill(output: &str) -> (bool, Vec<String>) {
    let mut reasons = vec![];
    let has_tldr = output.to_lowercase().contains("tldr")
        || output
            .lines()
            .next()
            .map(|l| l.len() > 20 && l.len() < 300)
            .unwrap_or(false);
    let has_header = output.lines().any(|l| l.trim_start().starts_with("##"));
    let has_sources = output.to_lowercase().contains("sources")
        || output.contains("[mem_") && output.matches("[mem_").count() >= 2;
    let has_wikilinks = output.contains("[[") && output.contains("]]");
    let word_count = output.split_whitespace().count();
    let reasonable_length = (100..2000).contains(&word_count);

    if !has_tldr {
        reasons.push("missing TLDR/opener".into());
    }
    if !has_header {
        reasons.push("missing ## headers".into());
    }
    if !has_sources {
        reasons.push("missing source attribution".into());
    }
    if !has_wikilinks {
        reasons.push("missing [[wikilinks]]".into());
    }
    if !reasonable_length {
        reasons.push(format!("unusual word count: {}", word_count));
    }
    let score = [
        has_tldr,
        has_header,
        has_sources,
        has_wikilinks,
        reasonable_length,
    ]
    .iter()
    .filter(|&&b| b)
    .count();
    reasons.push(format!("structural score: {}/5", score));
    (score >= 3, reasons)
}

fn extract_json(s: &str) -> &str {
    let s = s.trim();
    let start = s.find(['{', '[']).unwrap_or(0);
    let rest = &s[start..];
    let last = rest.rfind(['}', ']']).map(|i| i + 1).unwrap_or(rest.len());
    &rest[..last]
}

fn build_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{}\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        system, user
    )
}

fn strip_thinking(output: &str) -> String {
    let mut result = output.to_string();
    while let Some(start) = result.find("<think>") {
        if let Some(end_offset) = result[start..].find("</think>") {
            let end = start + end_offset + "</think>".len();
            result = format!("{}{}", &result[..start], &result[end..]);
        } else {
            result.truncate(start);
            break;
        }
    }
    result.trim().to_string()
}

// ---- Main benchmark driver ----

#[allow(clippy::type_complexity)]
fn run_benchmark(
    engine: &LlmEngine,
    name: &str,
    samples: Vec<(String, String, Box<dyn Fn(&str) -> (bool, Vec<String>)>)>,
    max_tokens: i32,
    temperature: f32,
) -> PromptResult {
    println!("\n=== {} ===", name);
    let mut results = vec![];
    let mut total_latency = 0u128;
    let mut passed_count = 0;

    let raw_max = max_tokens;
    let timeout = 30u64;
    let ctx = 4096u32;

    for (i, (input, prompt, scorer)) in samples.iter().enumerate() {
        let start = Instant::now();
        let raw = engine
            .run_inference_raw(prompt, raw_max, temperature, timeout, ctx)
            .unwrap_or_default();
        let output = strip_thinking(&raw);
        let latency = start.elapsed().as_millis();
        total_latency += latency;

        let (passed, reasons) = scorer(&output);
        if passed {
            passed_count += 1;
        }

        let status = if passed { "PASS" } else { "FAIL" };
        println!(
            "  [{}/{}] {} {} ({}ms)",
            i + 1,
            samples.len(),
            status,
            reasons.join("; "),
            latency
        );

        results.push(SampleResult {
            input: input.clone(),
            output: output.chars().take(500).collect(),
            raw_output: raw.chars().take(2000).collect(),
            passed,
            reasons,
            latency_ms: latency,
        });
    }

    let avg_latency = total_latency / samples.len() as u128;
    println!(
        "  -> {}/{} passed, avg {}ms",
        passed_count,
        samples.len(),
        avg_latency
    );

    PromptResult {
        name: name.into(),
        passed: passed_count,
        total: samples.len(),
        avg_latency_ms: avg_latency,
        samples: results,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path/to/model.gguf> [output.json]", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!(
            "  cargo run --release -p origin-core --bin model_benchmark -- ~/.cache/huggingface/models/qwen3.5/Qwen3.5-4B-Q4_K_M.gguf"
        );
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    if !model_path.exists() {
        eprintln!("Error: model file not found: {}", model_path.display());
        std::process::exit(1);
    }

    let output_path = args.get(2).cloned();
    let model_size = std::fs::metadata(&model_path)?.len();

    println!("Loading model: {}", model_path.display());
    println!("Size: {:.2} GB", model_size as f64 / 1e9);

    let prompts = PromptRegistry::default();
    let engine = LlmEngine::new(&model_path, prompts.clone())?;
    println!("Model loaded.\n");

    let mut all_results = vec![];

    // 1. CLASSIFY_MEMORY
    let classify_samples: Vec<_> = CLASSIFY_CASES
        .iter()
        .zip(CLASSIFY_EXPECTED_TYPES.iter())
        .map(|(input, expected)| {
            let prompt = build_prompt(&prompts.classify_memory, input);
            let expected = expected.to_string();
            let scorer: Box<dyn Fn(&str) -> (bool, Vec<String>)> =
                Box::new(move |output: &str| score_classify(output, &expected));
            (input.to_string(), prompt, scorer)
        })
        .collect();
    all_results.push(run_benchmark(
        &engine,
        "CLASSIFY_MEMORY",
        classify_samples,
        256,
        0.0,
    ));

    // 2. EXTRACT_KNOWLEDGE_GRAPH
    let extract_samples: Vec<_> = EXTRACT_CASES
        .iter()
        .zip(EXTRACT_EXPECTED_ENTITIES.iter())
        .enumerate()
        .map(|(i, (input, expected))| {
            let user = format!("Memory {}: {}", i + 1, input);
            let prompt = build_prompt(&prompts.extract_knowledge_graph, &user);
            let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
            let scorer: Box<dyn Fn(&str) -> (bool, Vec<String>)> = Box::new(move |output: &str| {
                let refs: Vec<&str> = expected.iter().map(|s| s.as_str()).collect();
                score_extract(output, &refs)
            });
            (input.to_string(), prompt, scorer)
        })
        .collect();
    all_results.push(run_benchmark(
        &engine,
        "EXTRACT_KNOWLEDGE_GRAPH",
        extract_samples,
        512,
        0.0,
    ));

    // 3. DETECT_CONTRADICTION
    let contradiction_samples: Vec<_> = CONTRADICTION_PAIRS
        .iter()
        .map(|(a, b, expected)| {
            let user = format!("Memory A: {}\n\nMemory B: {}", a, b);
            let prompt = build_prompt(&prompts.detect_contradiction, &user);
            let expected = expected.to_string();
            let scorer: Box<dyn Fn(&str) -> (bool, Vec<String>)> =
                Box::new(move |output: &str| score_contradiction(output, &expected));
            (user, prompt, scorer)
        })
        .collect();
    all_results.push(run_benchmark(
        &engine,
        "DETECT_CONTRADICTION",
        contradiction_samples,
        128,
        0.0,
    ));

    // 4. DISTILL_CONCEPT (single cluster)
    let cluster_text = DISTILL_CLUSTER.join("\n\n");
    let distill_user = format!("Topic: Origin Architecture\n\n{}", cluster_text);
    let distill_prompt = build_prompt(&prompts.distill_concept, &distill_user);
    let distill_samples: Vec<_> = vec![(
        cluster_text,
        distill_prompt,
        Box::new(score_distill) as Box<dyn Fn(&str) -> (bool, Vec<String>)>,
    )];
    all_results.push(run_benchmark(
        &engine,
        "DISTILL_CONCEPT",
        distill_samples,
        1024,
        0.1,
    ));

    // ---- Aggregate score ----
    let total_passed: usize = all_results.iter().map(|r| r.passed).sum();
    let total_samples: usize = all_results.iter().map(|r| r.total).sum();
    let total_score = total_passed as f64 / total_samples as f64 * 100.0;

    println!("\n================================");
    println!("SUMMARY");
    println!("================================");
    println!("Model: {}", model_path.display());
    println!(
        "Overall: {}/{} ({:.1}%)",
        total_passed, total_samples, total_score
    );
    for r in &all_results {
        println!(
            "  {:<28} {:>2}/{:<2} {:>6}ms avg",
            r.name, r.passed, r.total, r.avg_latency_ms
        );
    }

    // Write JSON report
    let report = BenchmarkReport {
        model_path: model_path.display().to_string(),
        model_size_bytes: model_size,
        total_score,
        prompts: all_results,
    };

    if let Some(out) = output_path {
        let json = serde_json::to_string_pretty(&report)?;
        std::fs::write(&out, json)?;
        println!("\nReport saved to: {}", out);
    }

    Ok(())
}
