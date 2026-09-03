// SPDX-License-Identifier: AGPL-3.0-only
import type { McpClient } from "../../lib/tauri";
import { isPluginClient } from "./pluginClients";

/** Whether setting this client up NOW would be a duplicate-producing write
 *  taken on the strength of a reading that FAILED — and if so, why the reading
 *  failed.
 *
 *  ROUND 6, D6a. The tri-state was carried honestly all the way to the chip:
 *  `has_plugin = unreadable` renders "Setup state unknown" rather than nothing.
 *  But the ACTION beside the chip was the same unqualified "Set up" a measured
 *  `no` gets, and for a client whose route is a raw config write that is the
 *  double registration this app already has a warning box for:
 *
 *    - `claude_desktop_config.json` is readable and holds no raw entry;
 *    - the chat-side plugin manifest scan FAILS (a denied session directory, a
 *      `config.json` that would not parse);
 *    - detection correctly reports `has_plugin = unreadable`;
 *    - "Set up" writes `mcpServers.wenlan` anyway;
 *    - the plugin was in fact enabled, and Wenlan is now registered twice.
 *
 *  Withholding the button entirely would be the same false negative wearing a
 *  different coat — the user can still be right to write. What must not happen
 *  is the write going through on a reading nobody took. So the action is
 *  QUALIFIED: the row says what could not be read and what writing anyway will
 *  do, and the button says "Set up anyway". The choice is then the user's,
 *  made from a stated unknown, instead of the app's, made from a fiction.
 *
 *  Returns `null` when there is no such risk:
 *   - `claude_code` / `codex_cli` go through `installClientPlugin`, which
 *     installs the plugin rather than writing a raw entry, so an unread plugin
 *     state cannot produce a raw+plugin pair through this button;
 *   - `cursor` / `gemini_cli` have no plugin surface at all, so their
 *     `has_plugin` is a MEASURED `no` from the Rust side and always will be. */
export function unreadPluginWriteRisk(client: McpClient): string | null {
  // A plugin-install route never writes a raw entry — see pluginClients.ts.
  if (isPluginClient(client.client_type)) return null;
  if (client.has_plugin.kind !== "unreadable") return null;
  return client.has_plugin.error;
}
