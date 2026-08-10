import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * The page map's TypeScript interfaces are a hand copy of the daemon's wire
 * structs in `crates/wenlan-types/src/page_map.rs`, and the Rust bridge passes
 * the daemon's JSON straight through without typing it. Nothing else notices
 * if a field is renamed on one side: the canvas would simply read `undefined`
 * and draw boxes with no labels or no positions.
 *
 * So: every field the TypeScript side declares must exist on the Rust struct it
 * mirrors. The check is one-directional on purpose — the daemon adding a field
 * the app has not adopted yet is normal, while the app naming a field the
 * daemon does not have is always a bug.
 *
 * Both sides are read as source text rather than imported, because the
 * TypeScript copy lives in the `lib/tauri` barrel and importing that pulls in
 * the Tauri runtime.
 */
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

const rust = readFileSync(resolve(root, "crates/wenlan-types/src/page_map.rs"), "utf8");
const ts = readFileSync(resolve(root, "src/lib/tauri.ts"), "utf8");

function rustFields(name: string): string[] {
  const body = rust.match(new RegExp(`pub struct ${name} \\{([\\s\\S]*?)\\n\\}`));
  expect(body, `crates/wenlan-types/src/page_map.rs no longer declares ${name}`).not.toBeNull();
  return [...body![1].matchAll(/^\s*pub ([a-z_0-9]+):/gm)].map((m) => m[1]);
}

function tsFields(name: string): string[] {
  const body = ts.match(new RegExp(`export interface ${name} \\{([\\s\\S]*?)\\n\\}`));
  expect(body, `src/lib/tauri.ts no longer declares ${name}`).not.toBeNull();
  return [...body![1].matchAll(/^\s*([a-z_0-9]+)\??:/gm)].map((m) => m[1]);
}

describe("page map wire types mirror the daemon", () => {
  it.each([
    ["PageMap", "PageMapResponse"],
    ["PageMapNode", "PageMapNode"],
    ["PageMapEdge", "PageMapEdge"],
    ["PageMapViewport", "PageMapViewport"],
    ["PageMapNodeMutation", "NodeMutationResponse"],
    ["PageMapLayoutPosition", "PageMapNodeLayout"],
  ])("%s carries only fields %s declares", (tsName, rustName) => {
    const declared = rustFields(rustName);
    expect(declared.length).toBeGreaterThan(0);

    const mirrored = tsFields(tsName);
    expect(mirrored.length).toBeGreaterThan(0);

    expect(mirrored.filter((field) => !declared.includes(field))).toEqual([]);
  });
});
