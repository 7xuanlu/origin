// SPDX-License-Identifier: Apache-2.0
//! deep_distill_cloud: Run the same deep distillation prompt through Anthropic models.
//! Produces output directly comparable to deep_distill.rs for on-device models.

use origin_core::prompts::PromptRegistry;
use serde_json::json;
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

struct QualityScore {
    factual_density: usize,
    total_expected: usize,
    wikilink_count: usize,
    header_count: usize,
    has_tldr: bool,
    source_citation_count: usize,
    word_count: usize,
}

fn score(output: &str) -> QualityScore {
    let lower = output.to_lowercase();
    let factual_density = EXPECTED_ENTITIES
        .iter()
        .filter(|e| lower.contains(&e.to_lowercase()))
        .count();
    QualityScore {
        factual_density,
        total_expected: EXPECTED_ENTITIES.len(),
        wikilink_count: output.matches("[[").count(),
        header_count: output
            .lines()
            .filter(|l| l.trim_start().starts_with("##"))
            .count(),
        has_tldr: lower.contains("tldr"),
        source_citation_count: output.matches("[mem_").count(),
        word_count: output.split_whitespace().count(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_id>", args[0]);
        eprintln!("Example: {} claude-haiku-4-5-20251001", args[0]);
        eprintln!("Example: {} claude-sonnet-4-6", args[0]);
        eprintln!("Example: {} claude-opus-4-6", args[0]);
        std::process::exit(1);
    }
    let model = &args[1];

    let config = origin_core::config::load_config();
    let api_key = config
        .anthropic_api_key
        .ok_or("No anthropic_api_key in config")?;

    let prompts = PromptRegistry::default();
    let system = prompts.distill_concept.clone();
    let memories_block = MEMORIES.join("\n\n");
    let user = format!("Topic: Origin Architecture\n\n{}", memories_block);

    println!("=== Deep Distillation (Cloud) ===");
    println!("Model: {}", model);
    println!("Memories: {}", MEMORIES.len());
    println!();
    println!("Running inference...");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()?;

    let body = json!({
        "model": model,
        "max_tokens": 2048,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    });

    let start = Instant::now();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    let elapsed = start.elapsed();
    let status = resp.status();
    let text = resp.text().await?;

    if !status.is_success() {
        eprintln!("API error {}: {}", status, text);
        std::process::exit(1);
    }

    let json_resp: serde_json::Value = serde_json::from_str(&text)?;
    let output = json_resp["content"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let input_tokens = json_resp["usage"]["input_tokens"].as_u64().unwrap_or(0);
    let output_tokens = json_resp["usage"]["output_tokens"].as_u64().unwrap_or(0);
    let s = score(&output);

    println!();
    println!("=== {} ===", model);
    println!("Latency: {:.1}s", elapsed.as_secs_f64());
    println!("Tokens: {} in, {} out", input_tokens, output_tokens);
    println!("Output length: {} words", s.word_count);
    println!();
    println!("Quality metrics:");
    println!(
        "  Factual density:   {}/{} ({:.0}%)",
        s.factual_density,
        s.total_expected,
        s.factual_density as f64 / s.total_expected as f64 * 100.0
    );
    println!("  Wikilinks:         {}", s.wikilink_count);
    println!("  Headers:           {}", s.header_count);
    println!(
        "  TLDR:              {}",
        if s.has_tldr { "yes" } else { "no" }
    );
    println!("  Source citations:  {}", s.source_citation_count);
    println!();
    println!("--- Full output ---");
    println!("{}", output);
    println!("--- End ---");

    let clean_name = model.replace([':', '/'], "-");
    let summary_path = format!("/tmp/origin-bench/deep-distill-{}.txt", clean_name);
    std::fs::create_dir_all("/tmp/origin-bench").ok();
    let summary = format!(
        "Model: {}\nLatency: {:.1}s\nTokens: {} in, {} out\nWords: {}\nFactual density: {}/{}\nWikilinks: {}\nHeaders: {}\nSource citations: {}\n\n--- Output ---\n{}\n",
        model, elapsed.as_secs_f64(), input_tokens, output_tokens,
        s.word_count, s.factual_density, s.total_expected,
        s.wikilink_count, s.header_count, s.source_citation_count, output
    );
    std::fs::write(&summary_path, summary)?;
    println!("\nSaved: {}", summary_path);

    Ok(())
}
