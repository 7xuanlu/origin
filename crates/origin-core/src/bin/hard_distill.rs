// SPDX-License-Identifier: Apache-2.0
//! hard_distill: Stress test for distillation quality.
//! Memories include supersession, fragmentary inference, open questions,
//! partial fixes, hindsight notes, and cross-domain noise.
//!
//! Topic: "Aperture Realtime Sync" (fictional canvas app, no external knowledge)

use origin_core::engine::LlmEngine;
use origin_core::prompts::PromptRegistry;
use std::path::PathBuf;
use std::time::Instant;

const TOPIC: &str = "Aperture Realtime Sync";

const MEMORIES: &[&str] = &[
    "[mem_1] Started with WebSockets for Aperture realtime — seemed obvious for cursor broadcasts.",
    "[mem_2] Hit issues with WebSockets reconnecting after laptop sleep — cursors go stale.",
    "[mem_3] Sarah suggested looking at WebTransport as alternative.",
    "[mem_4] WebTransport only works in Chrome right now, not Safari — ruled out for now.",
    "[mem_5] Decided to stick with WebSockets and add a heartbeat protocol.",
    "[mem_6] Added 15-second heartbeat, still seeing 5% drop rate in prod.",
    "[mem_7] Turned out the issue was Cloudflare killing idle WebSocket connections after ~100s.",
    "[mem_8] Switched heartbeat to 30s to stay under Cloudflare's window.",
    "[mem_9] That didn't fix it — Cloudflare kills based on activity not age.",
    "[mem_10] Added a WS ping frame every 20s on the transport layer, drop rate fell to 0.3%.",
    "[mem_11] Users report occasional 'ghost cursors' — other users' cursors appear in wrong position.",
    "[mem_12] Reproduced ghost cursor with three laptops on the same wifi.",
    "[mem_13] Noticed ghost cursors only happen after someone leaves the document.",
    "[mem_14] Cleanup code removes user from participants list but doesn't clear their last cursor position.",
    "[mem_15] Added cleanup hook to zero out cursor on disconnect.",
    "[mem_16] Still seeing ghost cursors in prod but can't reproduce locally — not fixed.",
    "[mem_17] Stroke sync uses CRDT via the Yjs library.",
    "[mem_18] Cursor position sync doesn't use CRDT — just last-write-wins over WS.",
    "[mem_19] This mismatch means cursors briefly appear on objects that no longer exist after undo.",
    "[mem_20] Fixed the 'cursor-on-deleted-object' case by sending cursor position with a Yjs document version vector.",
    "[mem_21] Yjs ships an Awareness API for ephemeral state like cursors and presence.",
    "[mem_22] We didn't use Awareness because we didn't want to bundle more Yjs code.",
    "[mem_23] In retrospect, implementing our own awareness was ~200 lines we could have avoided.",
    "[mem_24] Kyle argued at standup we should've used Awareness from the start.",
    "[mem_25] Performance is fine up to 30 concurrent users per document.",
    "[mem_26] At 50 users, stroke sync starts lagging — strokes arrive 2-3 seconds late.",
    "[mem_27] Root cause: we broadcast every stroke to every user regardless of viewport.",
    "[mem_28] No decision yet on whether to add interest-region broadcasting or just scale the server.",
    "[mem_29] Sarah thinks interest regions are too complex, prefers vertical scaling.",
    "[mem_30] Kyle wants to try Signal Pattern library before committing either way.",
    "[mem_31] Added presence indicators — colored dots in corner showing active users.",
    "[mem_32] Presence indicators reuse the same WS channel as cursor updates.",
    "[mem_33] Sarah was promoted to tech lead last month.",
    "[mem_34] Aperture team size is 4 engineers plus 1 PM.",
];

const EXPECTED_ENTITIES: &[&str] = &[
    "WebSocket",
    "WebTransport",
    "Cloudflare",
    "heartbeat",
    "Yjs",
    "CRDT",
    "Awareness",
    "cursor",
    "stroke",
    "version vector",
];

struct HardChecks {
    mentions_websocket_as_current: bool,
    mentions_webtransport_as_rejected_or_not_current: bool,
    mentions_cloudflare: bool,
    mentions_ping_frame_or_0_3: bool,
    mentions_ghost_cursors: bool,
    flags_ghost_cursors_unresolved: bool,
    falsely_claims_ghost_cursors_fixed: bool,
    mentions_crdt_lww_mismatch: bool,
    mentions_version_vector_fix: bool,
    mentions_awareness_missed: bool,
    mentions_50_user_ceiling: bool,
    flags_scaling_as_open_question: bool,
    leaks_team_noise: bool,
}

fn analyze(output: &str) -> HardChecks {
    let lower = output.to_lowercase();
    let has = |s: &str| lower.contains(s);

    let mentions_websocket_as_current = (has("websocket") || has("ws "))
        && !has("webtransport was chosen")
        && !has("switched to webtransport");
    let mentions_webtransport_as_rejected_or_not_current = has("webtransport")
        && (has("ruled out")
            || has("not chosen")
            || has("rejected")
            || has("only works in chrome")
            || has("only chrome")
            || has("safari")
            || has("considered but")
            || has("not used")
            || has("declined"));
    let mentions_cloudflare = has("cloudflare");
    let mentions_ping_frame_or_0_3 =
        has("ping frame") || has("0.3%") || has("0.3 %") || has("0.3 percent");

    let mentions_ghost_cursors = has("ghost cursor");
    let flags_ghost_cursors_unresolved = mentions_ghost_cursors
        && (has("still")
            || has("unresolved")
            || has("not fixed")
            || has("not fully")
            || has("remain")
            || has("persist")
            || has("open question")
            || has("partial"));
    let falsely_claims_ghost_cursors_fixed = mentions_ghost_cursors
        && (has("ghost cursors were fixed")
            || has("ghost cursor was fixed")
            || has("ghost cursors are fixed")
            || has("ghost cursor bug was resolved"))
        && !flags_ghost_cursors_unresolved;

    let mentions_crdt_lww_mismatch = (has("crdt") || has("yjs"))
        && (has("last-write-wins") || has("lww") || has("mismatch") || has("different sync"));
    let mentions_version_vector_fix = has("version vector");

    let mentions_awareness_missed = has("awareness")
        && (has("in hindsight")
            || has("should have")
            || has("should've")
            || has("missed")
            || has("retrospect")
            || has("could have"));

    let mentions_50_user_ceiling =
        has("50 user") || has("50 concurrent") || has("50 users") || has("users per document");
    let flags_scaling_as_open_question = has("interest region")
        && (has("undecided")
            || has("open question")
            || has("not decided")
            || has("no decision")
            || has("versus")
            || has("vs ")
            || has("or"));

    let leaks_team_noise = has("promoted") || has("tech lead") || has("5 pm") || has("4 engineers");

    HardChecks {
        mentions_websocket_as_current,
        mentions_webtransport_as_rejected_or_not_current,
        mentions_cloudflare,
        mentions_ping_frame_or_0_3,
        mentions_ghost_cursors,
        flags_ghost_cursors_unresolved,
        falsely_claims_ghost_cursors_fixed,
        mentions_crdt_lww_mismatch,
        mentions_version_vector_fix,
        mentions_awareness_missed,
        mentions_50_user_ceiling,
        flags_scaling_as_open_question,
        leaks_team_noise,
    }
}

fn factual_density(output: &str) -> (usize, usize) {
    let lower = output.to_lowercase();
    let found = EXPECTED_ENTITIES
        .iter()
        .filter(|e| lower.contains(&e.to_lowercase()))
        .count();
    (found, EXPECTED_ENTITIES.len())
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
    let user = format!("Topic: {}\n\n{}", TOPIC, memories_block);
    let prompt = build_prompt(&prompts.distill_concept, &user);

    println!("=== Hard Distillation Test ===");
    println!("Topic:    {}", TOPIC);
    println!(
        "Memories: {} (incl. supersession, partial fixes, cross-domain noise)",
        MEMORIES.len()
    );
    println!();
    println!("Running inference...");

    let start = Instant::now();
    let raw = engine
        .run_inference_raw(&prompt, 2048, 0.1, 180, 8192)
        .unwrap_or_default();
    let elapsed = start.elapsed();
    let output = strip_thinking(&raw);

    let (fd_found, fd_total) = factual_density(&output);
    let checks = analyze(&output);
    let word_count = output.split_whitespace().count();

    println!();
    println!("=== {} ===", model_name);
    println!("Latency:      {:.1}s", elapsed.as_secs_f64());
    println!("Word count:   {}", word_count);
    println!(
        "Factual:      {}/{} expected entities ({:.0}%)",
        fd_found,
        fd_total,
        fd_found as f64 / fd_total as f64 * 100.0
    );
    println!();
    println!("Hard checks:");
    let mark = |b: bool| if b { "PASS" } else { "FAIL" };
    println!(
        "  Supersession -- WebSocket is current:         {}",
        mark(checks.mentions_websocket_as_current)
    );
    println!(
        "  Supersession -- WebTransport noted as not-now: {}",
        mark(checks.mentions_webtransport_as_rejected_or_not_current)
    );
    println!(
        "  Debugging -- Cloudflare root cause:           {}",
        mark(checks.mentions_cloudflare)
    );
    println!(
        "  Debugging -- ping frame / 0.3% drop:          {}",
        mark(checks.mentions_ping_frame_or_0_3)
    );
    println!(
        "  Partial fix -- ghost cursors mentioned:        {}",
        mark(checks.mentions_ghost_cursors)
    );
    println!(
        "  Partial fix -- flagged as unresolved:          {}",
        mark(checks.flags_ghost_cursors_unresolved)
    );
    println!(
        "  Partial fix -- NOT falsely claimed fixed:      {}",
        mark(!checks.falsely_claims_ghost_cursors_fixed)
    );
    println!(
        "  Inference -- CRDT/LWW mismatch noted:          {}",
        mark(checks.mentions_crdt_lww_mismatch)
    );
    println!(
        "  Inference -- version vector fix named:         {}",
        mark(checks.mentions_version_vector_fix)
    );
    println!(
        "  Hindsight -- Awareness API missed opportunity: {}",
        mark(checks.mentions_awareness_missed)
    );
    println!(
        "  Open Q -- 50-user ceiling noted:               {}",
        mark(checks.mentions_50_user_ceiling)
    );
    println!(
        "  Open Q -- scaling approach is undecided:       {}",
        mark(checks.flags_scaling_as_open_question)
    );
    println!(
        "  Noise filter -- team/headcount NOT leaked:     {}",
        mark(!checks.leaks_team_noise)
    );

    let passed = [
        checks.mentions_websocket_as_current,
        checks.mentions_webtransport_as_rejected_or_not_current,
        checks.mentions_cloudflare,
        checks.mentions_ping_frame_or_0_3,
        checks.mentions_ghost_cursors,
        checks.flags_ghost_cursors_unresolved,
        !checks.falsely_claims_ghost_cursors_fixed,
        checks.mentions_crdt_lww_mismatch,
        checks.mentions_version_vector_fix,
        checks.mentions_awareness_missed,
        checks.mentions_50_user_ceiling,
        checks.flags_scaling_as_open_question,
        !checks.leaks_team_noise,
    ]
    .iter()
    .filter(|&&b| b)
    .count();
    println!();
    println!(
        "Hard score:   {}/13 ({:.0}%)",
        passed,
        passed as f64 / 13.0 * 100.0
    );

    println!();
    println!("--- Full output ---");
    println!("{}", output);
    println!("--- End ---");

    let summary_path = format!(
        "/tmp/origin-bench/hard-distill-{}.txt",
        model_name.replace('.', "-")
    );
    std::fs::create_dir_all("/tmp/origin-bench").ok();
    let summary = format!(
        "Model: {}\nLatency: {:.1}s\nWords: {}\nFactual: {}/{}\nHard score: {}/13\n\n--- Output ---\n{}\n",
        model_name,
        elapsed.as_secs_f64(),
        word_count,
        fd_found,
        fd_total,
        passed,
        output
    );
    std::fs::write(&summary_path, summary)?;
    println!("\nSaved: {}", summary_path);

    Ok(())
}
