// SPDX-License-Identifier: AGPL-3.0-only
import { describe, expect, it } from "vitest";
import {
  clientTypeFamily,
  familyDisplayName,
  resolveAgentDisplayName,
  toolFamilyOf,
} from "./agents";
import type { AgentConnection } from "./tauri";

describe("resolveAgentDisplayName", () => {
  it("uses the built-in registry for known IDs", () => {
    expect(resolveAgentDisplayName("claude-code")).toBe("Claude Code");
    expect(resolveAgentDisplayName("codex")).toBe("Codex");
  });

  it("prefers a user's registered display_name over the registry", () => {
    const connections: AgentConnection[] = [
      { name: "claude-code", display_name: "My Laptop" } as AgentConnection,
    ];
    expect(resolveAgentDisplayName("claude-code", connections)).toBe("My Laptop");
  });

  it("prettifies an unrecognized slug instead of passing it through raw", () => {
    expect(resolveAgentDisplayName("codex-mcp-client")).toBe("Codex MCP Client");
  });

  it("applies the acronym map for cli/api/ai/vscode", () => {
    expect(resolveAgentDisplayName("codex-ulw-loop")).toBe("Codex Ulw Loop");
    expect(resolveAgentDisplayName("cursor-vscode")).toBe("Cursor VS Code");
    expect(resolveAgentDisplayName("some-ai-cli")).toBe("Some AI CLI");
  });

  it("handles underscores the same as hyphens", () => {
    expect(resolveAgentDisplayName("foo_bar_baz")).toBe("Foo Bar Baz");
  });

  it("never returns empty for a non-empty input", () => {
    expect(resolveAgentDisplayName("x")).toBe("X");
  });
});

describe("clientTypeFamily", () => {
  it("maps each known client_type to its family", () => {
    expect(clientTypeFamily("claude_code")).toBe("claude-code");
    expect(clientTypeFamily("codex_cli")).toBe("codex");
    expect(clientTypeFamily("claude_desktop")).toBe("claude-desktop");
    expect(clientTypeFamily("cursor")).toBe("cursor");
    expect(clientTypeFamily("gemini_cli")).toBe("gemini-cli");
  });

  it("returns empty string for an unknown client_type", () => {
    expect(clientTypeFamily("obsidian")).toBe("");
    expect(clientTypeFamily("")).toBe("");
  });
});

describe("toolFamilyOf", () => {
  it("(a) trusts agent_type when it names a known client", () => {
    expect(toolFamilyOf({ name: "whatever", agent_type: "codex_cli" })).toBe("codex");
    expect(toolFamilyOf({ name: "x", agent_type: "claude_code" })).toBe("claude-code");
  });

  it("(b) matches an exact registry key when agent_type is unhelpful", () => {
    expect(toolFamilyOf({ name: "cursor", agent_type: "" })).toBe("cursor");
    expect(toolFamilyOf({ name: "codex", agent_type: "mcp" })).toBe("codex");
  });

  it("(b) aliases openai-mcp into the chatgpt family", () => {
    expect(toolFamilyOf({ name: "openai-mcp", agent_type: "" })).toBe("chatgpt");
  });

  it("(c) coalesces suffixed identities by longest known-key prefix", () => {
    expect(toolFamilyOf({ name: "codex-ulw-loop", agent_type: "" })).toBe("codex");
    expect(toolFamilyOf({ name: "codex-mcp-client", agent_type: "" })).toBe("codex");
    // `claude-code` must win over `claude-desktop`'s shorter shared root.
    expect(toolFamilyOf({ name: "claude-code-x", agent_type: "" })).toBe("claude-code");
  });

  it("(d) keeps the wizard probe as its own family, then passes names through", () => {
    expect(toolFamilyOf({ name: "wenlan-setup", agent_type: "" })).toBe("wenlan-setup");
    expect(toolFamilyOf({ name: "my-custom-bot", agent_type: "" })).toBe("my-custom-bot");
  });
});

describe("familyDisplayName", () => {
  it("uses the registry for a known family, prettifies an unknown one", () => {
    expect(familyDisplayName("codex")).toBe("Codex");
    expect(familyDisplayName("claude-code")).toBe("Claude Code");
    expect(familyDisplayName("my-custom-bot")).toBe("My Custom Bot");
  });
});
