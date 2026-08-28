// SPDX-License-Identifier: AGPL-3.0-only
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { REVIEW_COMMAND_CAPABILITIES } from "../../review/commandCapabilities";

/**
 * Every `invoke("...")` literal in the production frontend has to be a
 * decision: either the Review fixture surface answers it, or somebody said out
 * loud that it does not.
 *
 * The failure this exists to stop is silent. `review/tauri-core.ts` throws
 * `UnknownTauriCommandError` for anything outside the contract, so a new
 * command that nobody registered does not break a type check or a unit test —
 * it breaks the Review build at runtime, on whichever screen happens to call
 * it, which is the last place anyone looks. Modelled on the daemon repo's
 * `drift_guard` teeth #2: a known list, a grandfathered baseline, and a new
 * entry that fails the build rather than being discovered later.
 */
const INVOKE_LITERAL = /invoke[A-Za-z]*(?:<[^>]*>)?\(\s*"([a-z0-9_]+)"/g;

/**
 * Commands invoked by the production app that the Review fixture surface does
 * not answer, grandfathered at the point this guard was written.
 *
 * They share one reason, which is why it is stated once here rather than
 * copied a hundred times: Review is a fixture-only product surface for QA and
 * design, not a mirror of every production IPC command
 * (`review/commandCapabilities.ts` says as much). Setup wizards, sidecar and
 * daemon lifecycle, remote access, OS integration, and the capture window all
 * sit outside what a fixture can meaningfully stand in for.
 *
 * This is a burn-down list, not a dumping ground: an entry that is no longer
 * invoked anywhere fails below, so the list can only shrink on its own.
 * Adding to it is a deliberate act, and reviewable as one.
 */
const OUTSIDE_THE_REVIEW_SURFACE: readonly string[] = [
  "accept_pending_revision",
  "acknowledge_guarded_quit_request",
  "add_source",
  "add_watch_path",
  "archive_page",
  "cancel_guarded_quit_request",
  "connect_source",
  "correct_memory_cmd",
  "count_knowledge_files",
  "create_entity_cmd",
  "delete_agent",
  "delete_bulk",
  "delete_by_time_range",
  "delete_snapshot",
  "delete_tag",
  "detect_mcp_clients_cmd",
  "detect_obsidian_vaults",
  "disconnect_source",
  "dismiss_contradiction",
  "dismiss_pending_revision",
  "dismiss_quick_capture",
  "download_on_device_model",
  "export_page_to_obsidian",
  "export_pages_to_obsidian",
  "get_agent",
  "get_api_key",
  "get_avatar_data_url",
  "get_capture_stats",
  "get_chunks",
  "get_external_llm",
  "get_external_llm_key_configured",
  "get_index_status",
  "get_knowledge_path",
  "get_model_choice",
  "get_nurture_cards_cmd",
  "get_on_device_model",
  "get_pipeline_status",
  "get_profile_narrative",
  "get_remote_access_status",
  "get_resolved_routing",
  "get_session_snapshots",
  "get_setup_completed",
  "get_snapshot_captures",
  "get_snapshot_captures_with_content",
  "get_wenlan_mcp_entry",
  "import_chat_export",
  "import_memories_cmd",
  "ingest_webpage",
  "install_client_plugin",
  "is_run_at_login_enabled",
  "list_activities",
  "list_decision_domains_cmd",
  "list_decisions_cmd",
  "list_external_models",
  "list_orphan_links",
  "list_pending_imports",
  "list_pinned_memories",
  "list_sources",
  "list_watch_paths",
  "move_space",
  "on_device_model_download_bytes",
  "open_file",
  "open_search_result",
  "position_quick_capture",
  "quick_capture",
  "quit_wenlan_full",
  "read_source_dir",
  "read_text_file",
  "rebuild_activities",
  "regenerate_narrative",
  "reindex",
  "remove_avatar",
  "remove_legacy_mcp_entry",
  "remove_raw_mcp_entry",
  "remove_source",
  "remove_watch_path",
  "reset_onboarding_milestones",
  "save_temp_file",
  "set_api_key",
  "set_avatar",
  "set_document_space",
  "set_document_tags",
  "set_external_llm",
  "set_model_choice",
  "set_run_at_login",
  "set_source_pin",
  "start_daemon_sidecar",
  "store_memory",
  "suggest_tags",
  "sync_registered_source",
  "sync_source",
  "test_external_llm",
  "test_remote_mcp_connection",
  "toggle_remote_access",
  "update_agent",
  "update_chunk",
  "update_profile",
  "upload_source_file",
  "wire_state",
  "write_mcp_config",
];

function productionSources(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      // `src/test` is the harness, not the product.
      if (entry !== "test") productionSources(path, out);
      continue;
    }
    if (!/\.tsx?$/.test(entry) || entry.includes(".test.")) continue;
    out.push(path);
  }
  return out;
}

function invokedCommands(): Map<string, string> {
  const origins = new Map<string, string>();
  for (const path of productionSources("src")) {
    const source = readFileSync(path, "utf8");
    for (const match of source.matchAll(INVOKE_LITERAL)) {
      if (!origins.has(match[1])) origins.set(match[1], path);
    }
  }
  return origins;
}

const contract = new Set<string>(Object.values(REVIEW_COMMAND_CAPABILITIES).flat());

describe("review command contract", () => {
  it("accounts for every command the production frontend invokes", () => {
    const unaccounted = [...invokedCommands()]
      .filter(([command]) => !contract.has(command))
      .filter(([command]) => !OUTSIDE_THE_REVIEW_SURFACE.includes(command))
      .map(([command, path]) => `${command} (${path})`);

    expect(
      unaccounted,
      "Add each command to REVIEW_COMMAND_CAPABILITIES in review/commandCapabilities.ts "
        + "and give it an answer in e2e/tauriMock/runtime.ts, or list it in "
        + "OUTSIDE_THE_REVIEW_SURFACE if the Review fixture surface should not answer it.",
    ).toEqual([]);
  });

  it("keeps the exemption list free of commands the contract already covers", () => {
    // Both lists claiming one command is how an exemption outlives the reason
    // it was written for.
    const doubleBooked = OUTSIDE_THE_REVIEW_SURFACE.filter((command) => contract.has(command));

    expect(doubleBooked).toEqual([]);
  });

  it("keeps the exemption list free of commands nothing invokes any more", () => {
    const invoked = invokedCommands();
    const stale = OUTSIDE_THE_REVIEW_SURFACE.filter((command) => !invoked.has(command));

    expect(stale, "Remove these from OUTSIDE_THE_REVIEW_SURFACE; nothing invokes them.").toEqual([]);
  });

  it("covers the page-review commands, which the Review surface does answer", () => {
    // The reason this guard exists at all. Both are new, both are reachable
    // from the page detail screen the Review surface renders.
    expect(contract.has("review_page")).toBe(true);
    expect(contract.has("page_review_supported")).toBe(true);
  });
});
