import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * `community-read-v1` is written out in three places that must agree, and
 * nothing else makes them. The daemon's copy is the contract; the app's two
 * copies decide whether a space is allowed to count as durably mapped, because
 * `fetchSpaceCartography` refuses any response whose `schema_version` does not
 * match and reports the space as `partial-error` instead.
 *
 * A daemon that moves to v2 while the app still says v1 is the failure this
 * guards: every space would silently fall back to the client-side climb, which
 * is exactly the state the community layer is waiting to leave.
 *
 * This reads the three source files rather than importing the values, because
 * the app's own copy lives in the `lib/tauri` barrel, and importing that pulls
 * in the Tauri runtime. Matching the constant's text is enough to catch a
 * one-sided bump, which is the drift that actually happens.
 */
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

function declaredVersion(file: string, pattern: RegExp): string {
  const source = readFileSync(resolve(root, file), "utf8");
  const match = source.match(pattern);
  expect(match, `${file} no longer declares the community read schema version`).not.toBeNull();
  return match![1];
}

describe("community read schema version", () => {
  it("is the same string the daemon publishes", () => {
    const daemon = declaredVersion(
      "crates/wenlan-types/src/communities.rs",
      /pub const COMMUNITY_READ_SCHEMA_VERSION: &str = "([^"]+)";/,
    );

    expect(
      declaredVersion(
        "src/lib/tauri.ts",
        /export const COMMUNITY_READ_SCHEMA_VERSION = "([^"]+)";/,
      ),
    ).toBe(daemon);

    expect(
      declaredVersion(
        "e2e/tauriMock/baseResponses.ts",
        /const COMMUNITY_READ_SCHEMA_VERSION = "([^"]+)";/,
      ),
    ).toBe(daemon);
  });
});
