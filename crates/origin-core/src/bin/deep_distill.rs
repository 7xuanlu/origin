// SPDX-License-Identifier: Apache-2.0
//! deep_distill: Test on-device model quality for the deep distillation scenario.
//! Feeds 28 memories into DISTILL_CONCEPT and measures quality + latency.

use origin_core::engine::LlmEngine;
use origin_core::prompts::PromptRegistry;
use std::path::PathBuf;
use std::time::Instant;

const MEMORIES: &[&str] = &[
    "[mem_1] Origin is a personal knowledge wiki app built with Tauri 2 and Rust.",
    "[mem_2] Origin uses libSQL (Turso's SQLite fork) for storage with F32_BLOB(384) vector columns.",
    "[mem_3] The on-device LLM is Qwen3-4B-Instruct-2507 running on Metal GPU via llama-cpp-2.",
    "[mem_4] Origin ingests memories from file watching, clipboard, and quick capture.",
    "[mem_5] The app is AGPL-3.0 licensed and targets macOS as the primary platform.",
    "[mem_6] FastEmbed provides BGE-Small-EN-v1.5 embeddings at 384 dimensions.",
    "[mem_7] Hybrid search combines vector similarity with FTS5 via Reciprocal Rank Fusion (RRF).",
    "[mem_8] The smart router consumes a TriggerEvent mpsc channel with capacity 32.",
    "[mem_9] PII redaction runs before storage — covers credit cards, SSN, API keys, emails.",
    "[mem_10] Four Tauri windows: main, toast, snip, and quick-capture — routed by URL hash.",
    "[mem_11] The knowledge graph uses entities, relations, and observations tables with FK cascades.",
    "[mem_12] Frame comparison uses a 3-tier algorithm: downscale, hash, Hellinger distance.",
    "[mem_13] AFK detection triggers after 60 seconds of idle via CGEventSource.",
    "[mem_14] Two-pass capture pipeline: immediate OCR/chunk/upsert, then async LLM reformat.",
    "[mem_15] The refinery steep cycle runs every 2 hours — decay, recaps, entity extraction, distillation.",
    "[mem_16] Origin stores all data locally in ~/Library/Application Support/origin/.",
    "[mem_17] The HTTP server runs on 127.0.0.1:7878 using Axum 0.8 — plus a Unix socket.",
    "[mem_18] Entity extraction uses the EXTRACT_KNOWLEDGE_GRAPH prompt with JSON output.",
    "[mem_19] Concept distillation produces wiki-style pages with TLDR, headers, and [[wikilinks]].",
    "[mem_20] The contradiction detector returns CONSISTENT, CONTRADICTS, or SUPERSEDES labels.",
    "[mem_21] Qwen3-4B uses a /no_think suffix to disable thinking mode for JSON tasks.",
    "[mem_22] DiskANN vector indexing is built into libSQL for fast semantic search at scale.",
    "[mem_23] React Query (TanStack v5) manages frontend state; module stores for cross-unmount state.",
    "[mem_24] The focus sensor polls cursor position at 10Hz from a dedicated std::thread.",
    "[mem_25] Origin exposes ~92 Tauri commands from search.rs called via invoke() wrappers.",
    "[mem_26] Community detection runs before distillation to inform cluster formation.",
    "[mem_27] Recap generation uses the DETECT_PATTERN prompt on 30-minute memory bursts.",
    "[mem_28] BYOK model routing: synthesis_llm -> api_llm -> external_llm -> on_device.",
];

const EXPECTED_ENTITIES: &[&str] = &[
    "Tauri",
    "Rust",
    "libSQL",
    "Qwen3",
    "Metal GPU",
    "FastEmbed",
    "BGE",
    "FTS5",
    "Axum",
    "AGPL",
    "macOS",
    "React Query",
];

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

struct QualityScore {
    factual_density: usize,
    total_expected: usize,
    has_wikilinks: bool,
    wikilink_count: usize,
    has_headers: bool,
    header_count: usize,
    has_tldr: bool,
    has_sources: bool,
    source_citation_count: usize,
    word_count: usize,
    paragraph_count: usize,
}

fn score_distillation(output: &str) -> QualityScore {
    let lower = output.to_lowercase();
    let factual_density = EXPECTED_ENTITIES
        .iter()
        .filter(|e| lower.contains(&e.to_lowercase()))
        .count();

    let wikilink_count = output.matches("[[").count();
    let has_wikilinks = wikilink_count > 0;

    let header_count = output
        .lines()
        .filter(|l| l.trim_start().starts_with("##"))
        .count();
    let has_headers = header_count > 0;

    let has_tldr = lower.contains("tldr");
    let has_sources = lower.contains("sources") || lower.contains("source");
    let source_citation_count = output.matches("[mem_").count();

    let word_count = output.split_whitespace().count();
    let paragraph_count = output
        .split("\n\n")
        .filter(|p| !p.trim().is_empty())
        .count();

    QualityScore {
        factual_density,
        total_expected: EXPECTED_ENTITIES.len(),
        has_wikilinks,
        wikilink_count,
        has_headers,
        header_count,
        has_tldr,
        has_sources,
        source_citation_count,
        word_count,
        paragraph_count,
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

    println!("Loading model: {}", model_path.display());
    let model_name = model_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    let engine = LlmEngine::new(&model_path, prompts.clone())?;
    println!("Model loaded.\n");

    let memories_block = MEMORIES.join("\n\n");
    let user = format!("Topic: Origin Architecture\n\n{}", memories_block);
    let prompt = build_prompt(&prompts.distill_concept, &user);
    let prompt_chars = prompt.len();

    let max_tokens = 2048;
    let timeout = 180;
    let ctx_size = 8192;

    println!("=== Deep Distillation Test ===");
    println!("Memories: {}", MEMORIES.len());
    println!("Prompt chars: {}", prompt_chars);
    println!(
        "Max tokens: {}, ctx: {}, timeout: {}s",
        max_tokens, ctx_size, timeout
    );
    println!();
    println!("Running inference...");

    let start = Instant::now();
    let raw = engine
        .run_inference_raw(&prompt, max_tokens, 0.1, timeout, ctx_size)
        .unwrap_or_default();
    let elapsed = start.elapsed();

    let output = strip_thinking(&raw);
    let score = score_distillation(&output);

    println!();
    println!("=== {} ===", model_name);
    println!("Latency: {:.1}s", elapsed.as_secs_f64());
    println!(
        "Output length: {} words, {} paragraphs",
        score.word_count, score.paragraph_count
    );
    println!();
    println!("Quality metrics:");
    println!(
        "  Factual density:   {}/{} expected entities ({:.0}%)",
        score.factual_density,
        score.total_expected,
        score.factual_density as f64 / score.total_expected as f64 * 100.0
    );
    println!(
        "  Wikilinks:         {}  ({} occurrences)",
        if score.has_wikilinks { "yes" } else { "no" },
        score.wikilink_count
    );
    println!(
        "  Headers (##):      {}  ({} sections)",
        if score.has_headers { "yes" } else { "no" },
        score.header_count
    );
    println!(
        "  TLDR:              {}",
        if score.has_tldr { "yes" } else { "no" }
    );
    println!(
        "  Source citations:  {}  ({} [mem_X] references)",
        if score.has_sources { "yes" } else { "no" },
        score.source_citation_count
    );

    println!();
    println!("--- Full output ---");
    println!("{}", output);
    println!("--- End ---");

    let summary_path = format!(
        "/tmp/origin-bench/deep-distill-{}.txt",
        model_name.replace('.', "-")
    );
    std::fs::create_dir_all("/tmp/origin-bench").ok();
    let summary = format!(
        "Model: {}\nLatency: {:.1}s\nWords: {}\nFactual density: {}/{}\nWikilinks: {}\nHeaders: {}\nSource citations: {}\n\n--- Output ---\n{}\n",
        model_name,
        elapsed.as_secs_f64(),
        score.word_count,
        score.factual_density,
        score.total_expected,
        score.wikilink_count,
        score.header_count,
        score.source_citation_count,
        output
    );
    std::fs::write(&summary_path, summary)?;
    println!("\nSaved: {}", summary_path);

    Ok(())
}
