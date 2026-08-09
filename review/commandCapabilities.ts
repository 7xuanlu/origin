// SPDX-License-Identifier: AGPL-3.0-only

/**
 * Commands exercised by the fixture-only Review product surface.
 *
 * This is an allowlist, not a claim that Review mirrors every production IPC
 * command. `review/tauri-core.ts` rejects commands outside this contract.
 */
export const REVIEW_COMMAND_CAPABILITIES = {
  navigationAndHome: [
    "should_show_wizard",
    "set_traffic_lights_visible",
    "set_setup_completed",
    "search",
    "search_pages",
    "search_entities_cmd",
    "list_spaces",
    "list_memories_cmd",
    "list_memories_by_ids",
    "get_memory_stats_cmd",
    "get_home_stats",
    "list_recent_retrievals",
    "list_recent_changes",
    "list_recent_memories",
    "list_recent_pages",
    "list_recent_relations",
    "list_entities_cmd",
    "get_entity_suggestions_cmd",
    "get_profile",
    "list_agents",
    "list_agent_activity",
    "list_onboarding_milestones",
    "acknowledge_onboarding_milestone",
    "get_briefing",
    "get_pending_contradictions",
  ],
  wikiAndPages: [
    "list_pages",
    "get_page",
    "create_page",
    "update_page",
    "delete_page",
    "redistill_page",
    "get_page_links",
    "get_page_revisions",
    "get_page_sources",
    "list_registered_sources",
  ],
  pageDrafts: [
    "create_page_draft",
    "update_page_draft",
    "publish_page_draft",
    "discard_page_draft",
  ],
  refinement: [
    "distill_review",
    "list_refinements",
    "accept_refinement",
    "reject_refinement",
    "list_pending_revisions",
    "list_unconfirmed_memories",
    "confirm_memory",
    "delete_memory",
    "get_memory_detail",
    "get_pending_revision",
    "get_enrichment_status",
    "get_version_chain_cmd",
    "get_memory_revisions",
  ],
  memoryActions: [
    "update_memory_cmd",
    "reclassify_memory_cmd",
    "set_stability_cmd",
    "delete_file_chunks",
    "pin_memory",
    "unpin_memory",
  ],
  spaces: [
    "get_space",
    "create_space",
    "update_space",
    "delete_space",
    "confirm_space",
    "toggle_space_starred",
    "reorder_space",
  ],
  entityDetail: [
    "get_entity_detail_cmd",
    "add_observation_cmd",
    "update_observation_cmd",
    "delete_observation_cmd",
    "confirm_observation_cmd",
    "confirm_entity_cmd",
    "delete_entity_cmd",
  ],
  existingUtilityReads: [
    "get_clipboard_enabled",
    "list_all_tags",
    "list_indexed_files",
  ],
} as const;

export type ReviewCommandArea = keyof typeof REVIEW_COMMAND_CAPABILITIES;
export type ReviewCommand =
  (typeof REVIEW_COMMAND_CAPABILITIES)[ReviewCommandArea][number];

const REVIEW_COMMAND_ALLOWLIST: ReadonlySet<string> = new Set(
  Object.values(REVIEW_COMMAND_CAPABILITIES).flat(),
);

export function isReviewCommand(command: string): command is ReviewCommand {
  return REVIEW_COMMAND_ALLOWLIST.has(command);
}
