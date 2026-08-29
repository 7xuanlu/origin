//! Live smoke (ignored): the default on-device model must answer the title
//! job's prompt with real text.
//!
//! Regression for the empty-title bug: the prompt used to open the assistant
//! turn with an empty `<think></think>` block for every model. The default
//! model (Qwen3-4B-Instruct-2507) has no thinking mode, read the block as an
//! answer that had already ended, and stopped before its first word, so every
//! memory and page title fell back to the raw topic key.
//!
//! Needs the default model in the hf-hub cache and a GPU-capable process:
//! `cargo test -p wenlan-core --release --test on_device_prompt_smoke -- --ignored --nocapture`
use wenlan_core::engine::LlmEngine;
use wenlan_core::on_device_models::get_default_model;

const TITLE_SYSTEM: &str = "Given a note, write a 3-5 word title. Output ONLY the title.\n\nExample: 'The system uses libsql for vector storage with DiskANN indexing' → libsql Vector Storage\nExample: 'Google Sign-In fails with developer_error status 10' → Google Sign-In SHA Fix";
const NOTE: &str = "The Tally reminder email goes out at 9am local time. The copy is being shortened to one sentence so it reads well on a phone lock screen, and the subject line now names the amount owed.";

#[test]
#[ignore = "live GPU smoke: needs the default GGUF in the hf-hub cache"]
fn default_model_answers_the_title_prompt() {
    let spec = get_default_model();
    let path = LlmEngine::download_model().expect("default model cached");
    let engine = LlmEngine::new(&path, Default::default())
        .expect("engine")
        .with_thinking_mode(spec.thinking_mode);
    let mut ctx = engine
        .build_persistent_context_with_seq_max(spec.context_size, 1)
        .expect("persistent context");

    // Same budget and temperature as `refinery::generate_short_title`.
    let prompt = engine.format_prompt(Some(TITLE_SYSTEM), NOTE);
    let out = engine.run_inference_persistent(&mut ctx, &prompt, 16, 0.3, 30, true, Some("title"));
    eprintln!("title answer: {out:?}");
    let out = out.expect("title prompt produced no text");
    assert!(!out.trim().is_empty());
}
