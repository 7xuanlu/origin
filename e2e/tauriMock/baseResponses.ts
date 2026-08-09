// SPDX-License-Identifier: AGPL-3.0-only
import { UnknownTauriCommandError } from "./errors";

type BaseResponseContext = {
  readonly activityRows: readonly Record<string, unknown>[];
  readonly memoryCount: number;
};

function optionalString(args: unknown, key: string): string | null {
  const value = typeof args === "object" && args !== null ? Reflect.get(args, key) : undefined;
  return typeof value === "string" ? value : null;
}

export function baseResponse(command: string, args: unknown, context: BaseResponseContext): unknown {
  switch (command) {
    case "should_show_wizard": case "get_clipboard_enabled": return false;
    case "set_traffic_lights_visible": case "set_setup_completed": return null;
    case "list_agent_activity": return context.activityRows;
    case "list_agents": return [{ id: "agent-claude-code", name: "claude-code", display_name: "Claude Code", agent_type: "claude-code", description: null, enabled: true, trust_level: "full", last_seen_at: 1_783_728_000, memory_count: context.memoryCount, created_at: 1_783_728_000, updated_at: 1_783_728_000 }];
    case "search_pages": case "search_entities_cmd": case "list_recent_retrievals": case "list_recent_changes": case "list_recent_memories": case "list_unconfirmed_memories": case "list_recent_pages": case "list_recent_relations": case "list_onboarding_milestones": case "list_pending_revisions": case "get_entity_suggestions_cmd": case "get_pending_contradictions": return [];
    case "get_memory_stats_cmd": return { total: context.memoryCount, new_today: 0, confirmed: context.memoryCount, domains: [], by_type: [] };
    case "get_home_stats": return { total: context.memoryCount, new_today: 0, confirmed: context.memoryCount, total_ingested: 0, active_insights: 0, distilled_today: 0, distilled_all: 0, sources_archived: 0, times_served_today: 0, words_saved_today: 0, times_served_week: 0, words_saved_week: 0, times_served_all: 0, words_saved_all: 0, corrections_active: 0, top_memories: [] };
    case "get_profile": case "get_pending_revision": return null;
    case "get_briefing": return { content: "", new_today: 0, primary_agent: null, generated_at: 1_783_728_000, is_stale: false };
    case "get_enrichment_status": return { source_id: optionalString(args, "sourceId") ?? "", summary: "", steps: [] };
    case "get_version_chain_cmd": return [];
    case "get_memory_revisions": return { current_source_id: optionalString(args, "sourceId") ?? "", chain_depth: 0, entries: [] };
    case "list_all_tags": return { tags: [], document_tags: {}, categories: [], document_categories: {} };
    case "get_page_links": return { outbound: [], inbound: [] };
    case "get_page_sources": case "list_registered_sources": return [];
    case "get_page_revisions": return { page_id: optionalString(args, "pageId") ?? "", current_version: 1, user_edited: false, entries: [] };
    default: throw new UnknownTauriCommandError(command);
  }
}
