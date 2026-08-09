// SPDX-License-Identifier: AGPL-3.0-only
import type { PluginInstallClientType } from "../../lib/tauri";

/** Claude Code and Codex both ship a Wenlan plugin that declares its own
 *  `mcpServers`, so installing the plugin already registers the MCP server.
 *  Writing `~/.claude.json` / `[mcp_servers.wenlan]` for them as well would
 *  register Wenlan twice. These two clients therefore go through
 *  `installClientPlugin` and NEVER through `writeMcpConfig` — the one home
 *  for that rule, shared by the wizard and Settings so the two surfaces
 *  cannot drift apart on it. */
const PLUGIN_CLIENTS = new Set<string>(["claude_code", "codex_cli"]);

export function isPluginClient(
  clientType: string,
): clientType is PluginInstallClientType {
  return PLUGIN_CLIENTS.has(clientType);
}
