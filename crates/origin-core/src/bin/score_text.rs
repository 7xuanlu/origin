// SPDX-License-Identifier: Apache-2.0
//! score_text: Apply hard_distill scoring to a text file (for scoring inline Opus output).

use std::path::PathBuf;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path/to/output.txt>", args[0]);
        std::process::exit(1);
    }

    let output = std::fs::read_to_string(PathBuf::from(&args[1]))?;
    let lower = output.to_lowercase();
    let has = |s: &str| lower.contains(s);

    let fd_found = EXPECTED_ENTITIES
        .iter()
        .filter(|e| lower.contains(&e.to_lowercase()))
        .count();

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
        && (has("last-write-wins")
            || has("lww")
            || has("mismatch")
            || has("different sync")
            || has("different model"));
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
    let flags_scaling_as_open_question = (has("interest region") || has("interest-region"))
        && (has("undecided")
            || has("open question")
            || has("not decided")
            || has("no decision")
            || has("versus")
            || has("vs ")
            || has(" or "));

    let leaks_team_noise = has("promoted") || has("tech lead") || has("5 pm") || has("4 engineers");

    let mark = |b: bool| if b { "PASS" } else { "FAIL" };
    println!("Factual:      {}/{}", fd_found, EXPECTED_ENTITIES.len());
    println!();
    println!(
        "Supersession -- WebSocket is current:         {}",
        mark(mentions_websocket_as_current)
    );
    println!(
        "Supersession -- WebTransport noted as not-now: {}",
        mark(mentions_webtransport_as_rejected_or_not_current)
    );
    println!(
        "Debugging -- Cloudflare root cause:           {}",
        mark(mentions_cloudflare)
    );
    println!(
        "Debugging -- ping frame / 0.3% drop:          {}",
        mark(mentions_ping_frame_or_0_3)
    );
    println!(
        "Partial fix -- ghost cursors mentioned:        {}",
        mark(mentions_ghost_cursors)
    );
    println!(
        "Partial fix -- flagged as unresolved:          {}",
        mark(flags_ghost_cursors_unresolved)
    );
    println!(
        "Partial fix -- NOT falsely claimed fixed:      {}",
        mark(!falsely_claims_ghost_cursors_fixed)
    );
    println!(
        "Inference -- CRDT/LWW mismatch noted:          {}",
        mark(mentions_crdt_lww_mismatch)
    );
    println!(
        "Inference -- version vector fix named:         {}",
        mark(mentions_version_vector_fix)
    );
    println!(
        "Hindsight -- Awareness missed opportunity:     {}",
        mark(mentions_awareness_missed)
    );
    println!(
        "Open Q -- 50-user ceiling noted:               {}",
        mark(mentions_50_user_ceiling)
    );
    println!(
        "Open Q -- scaling approach is undecided:       {}",
        mark(flags_scaling_as_open_question)
    );
    println!(
        "Noise filter -- team/headcount NOT leaked:     {}",
        mark(!leaks_team_noise)
    );

    let passed = [
        mentions_websocket_as_current,
        mentions_webtransport_as_rejected_or_not_current,
        mentions_cloudflare,
        mentions_ping_frame_or_0_3,
        mentions_ghost_cursors,
        flags_ghost_cursors_unresolved,
        !falsely_claims_ghost_cursors_fixed,
        mentions_crdt_lww_mismatch,
        mentions_version_vector_fix,
        mentions_awareness_missed,
        mentions_50_user_ceiling,
        flags_scaling_as_open_question,
        !leaks_team_noise,
    ]
    .iter()
    .filter(|&&b| b)
    .count();
    println!();
    println!("Hard score:   {}/13", passed);
    println!("Word count:   {}", output.split_whitespace().count());

    Ok(())
}
