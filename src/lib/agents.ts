// SPDX-License-Identifier: AGPL-3.0-only
//
// Agent display-name resolution.
//
// Canonical technical IDs are strings like `claude-code` or `openai-mcp`.
// Users see friendly display names like `Claude Code` or `ChatGPT`.
//
// Resolution order:
//   1. User registration (`agent_connections.display_name` via listAgents).
//   2. Built-in `KNOWN_CLIENT_DISPLAY_NAMES` registry (mirrors origin-core's `KNOWN_CLIENTS`).
//   3. The raw canonical ID, unchanged.
//
// **Keep in sync with `crates/origin-core/src/db.rs::KNOWN_CLIENTS`.**

import type { AgentConnection } from "./tauri";

export const KNOWN_CLIENT_DISPLAY_NAMES: Record<string, string> = {
  "chatgpt": "ChatGPT",
  "claude-code": "Claude Code",
  "claude-desktop": "Claude Desktop",
  "codex": "Codex",
  "continue": "Continue",
  "cursor": "Cursor",
  "gemini-cli": "Gemini CLI",
  "obsidian": "Obsidian",
  "openai-mcp": "ChatGPT",
  "raycast": "Raycast",
  "vscode": "VS Code",
  "windsurf": "Windsurf",
  "zed": "Zed",
};

/**
 * Resolve a canonical agent ID to a friendly display name.
 *
 * @param canonicalId The raw value from `x-agent-name` / `activity.agent_name` /
 *                    `agent_connections.name`. Lowercased hyphen-case.
 * @param connections Optional list of registered agents, used to honor user
 *                    overrides (a user-set `display_name` always wins).
 * @returns The best available display label. Never throws. Never returns empty
 *          for a non-empty input — worst case, returns the canonical ID itself.
 */
export function resolveAgentDisplayName(
  canonicalId: string,
  connections?: AgentConnection[],
): string {
  if (!canonicalId) return canonicalId;

  // 1. User override via registration.
  if (connections) {
    const match = connections.find((c) => c.name === canonicalId);
    if (match?.display_name) {
      return match.display_name;
    }
  }

  // 2. Built-in registry.
  const known = KNOWN_CLIENT_DISPLAY_NAMES[canonicalId];
  if (known) return known;

  // 3. Fall through to the raw ID. Honest.
  return canonicalId;
}

// ── Trust level presentation ───────────────────────────────────────────
// Mirrors the tier gating in `crates/origin-core/src/router/classify.rs`
// (`tier_allowed`) and `crates/origin-server/src/routes.rs` (chat-context).

/** Known trust levels the backend recognizes. Keep in sync with `register_agent`. */
export type TrustLevel = "full" | "review" | "unknown";

export interface TrustLevelDescriptor {
  label: string;
  /** One-line explanation shown as tooltip / badge caption. */
  summary: string;
  /** Longer explanation for the section explainer. */
  detail: string;
  /** CSS accent var for the badge. */
  accent: string;
}

export const TRUST_LEVELS: Record<TrustLevel, TrustLevelDescriptor> = {
  full: {
    label: "Full",
    summary:
      "Sees everything: identity, preferences, goals, decisions, concepts, and search results.",
    detail:
      "This is the default for agents you've explicitly registered — if you walked through setup for it, Origin treats it as trusted and gives it access to every tier of context.",
    accent: "var(--mem-accent-indigo)",
  },
  review: {
    label: "Review",
    summary:
      "Sees goals, decisions, corrections, concepts, and search — but not identity or preferences.",
    detail:
      "Useful when you want an agent to help with work but not see your personal identity or preferences. Rarely needed for single-user setups.",
    accent: "var(--mem-accent-amber)",
  },
  unknown: {
    label: "Unknown",
    summary: "Sees search results only. No identity, preferences, or goals.",
    detail:
      "Unregistered callers (direct curl, third-party MCP clients you didn't register) land here automatically. Safe fallback for untrusted sources.",
    accent: "var(--mem-accent-warm)",
  },
};

export function describeTrustLevel(level: string): TrustLevelDescriptor {
  return TRUST_LEVELS[level as TrustLevel] ?? TRUST_LEVELS.unknown;
}
