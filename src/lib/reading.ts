// SPDX-License-Identifier: AGPL-3.0-only
/** Three-valued measurements from the Rust side (`mcp_config::Reading`).
 *
 *  This lives in its own module, apart from `tauri.ts`, for one practical
 *  reason: component tests mock `lib/tauri` wholesale, and a predicate that
 *  only exists on that module disappears with it. A rule this important must
 *  not be reachable only when the IPC module is real.
 */

/** A measurement with THREE values.
 *
 *  `no` is a measured negative: the look succeeded and the answer was no.
 *  `unreadable` is a look that FAILED — the OS refused the file, or the
 *  directory the path hangs off could not be determined, so nothing was
 *  measured at all. These used to be the same `false`, which is how "we could
 *  not read your config" reached the screen as "this tool is not installed".
 *  There is deliberately no boolean beside it: every reader must choose. */
export type Reading =
  | { kind: "yes" }
  | { kind: "no" }
  | { kind: "unreadable"; error: string };

/** True only for a measured yes. An unreadable reading is NOT a yes. */
export function readingIsYes(reading: Reading): boolean {
  return reading.kind === "yes";
}

/** True only for a measured no. An unreadable reading is NOT a no — this is
 *  the half callers forget, and the half that turns a failed look into
 *  "not installed". */
export function readingIsNo(reading: Reading): boolean {
  return reading.kind === "no";
}

/** True when the look failed. Anything that renders a negative must check
 *  this FIRST and say so differently. */
export function readingFailed(reading: Reading): boolean {
  return reading.kind === "unreadable";
}
