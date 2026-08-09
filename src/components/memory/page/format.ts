// SPDX-License-Identifier: AGPL-3.0-only
// Shared display helpers for the page-detail subcomponents.
import type { MemoryItem } from "../../../lib/tauri";

const KNOWN_AGENTS: Record<string, string> = {
  "claude-code": "Claude Code",
  "claude-desktop": "Claude Desktop",
  cursor: "Cursor",
  "chatgpt-mcp": "ChatGPT",
  chatgpt: "ChatGPT",
  "gemini-cli": "Gemini CLI",
  windsurf: "Windsurf",
  zed: "Zed",
};

export function prettyAgent(name: string | null | undefined): string {
  if (!name) return "unknown agent";
  const key = name.trim().toLowerCase();
  return KNOWN_AGENTS[key] ?? name;
}

const SOURCE_KIND_LABEL: Record<string, string> = {
  memory: "memory",
  chat: "chat",
  file: "file",
  obsidian: "obsidian",
  web: "web",
};

export function sourceKindLabel(mem: MemoryItem): string {
  const mt = mem.memory_type?.toLowerCase() ?? "";
  return SOURCE_KIND_LABEL[mt] ?? (mt || "memory");
}

export function relativeMs(ms: number): string {
  const delta = Date.now() - ms;
  if (delta < 60_000) return "just now";
  if (delta < 3_600_000) return `${Math.floor(delta / 60_000)}m ago`;
  if (delta < 86_400_000) return `${Math.floor(delta / 3_600_000)}h ago`;
  return `${Math.floor(delta / 86_400_000)}d ago`;
}
