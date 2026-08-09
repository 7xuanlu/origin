// SPDX-License-Identifier: AGPL-3.0-only
// Browser stand-in for @tauri-apps/plugin-shell.
export async function open(target: string): Promise<void> {
  console.log("[preview] shell.open:", target);
  if (target.startsWith("http")) window.open(target, "_blank");
}
