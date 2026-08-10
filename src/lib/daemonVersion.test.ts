// SPDX-License-Identifier: AGPL-3.0-only
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { daemonMeetsFloor, PAGE_EDIT_DAEMON_FLOOR } from "./daemonVersion";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

describe("page-edit daemon floor", () => {
  // The floor is duplicated across the language boundary: this module gates the
  // UI, and app/src/search.rs gates the command the UI calls. If they drift, the
  // editor opens for a daemon that will then refuse every save.
  it("matches the floor the Rust save command enforces", () => {
    const search = readFileSync(resolve(root, "app/src/search.rs"), "utf8");
    const declaration = search.match(
      /const PAGE_EDIT_DAEMON_FLOOR: &str = "([^"]+)";/,
    );

    expect(
      declaration,
      "app/src/search.rs no longer declares PAGE_EDIT_DAEMON_FLOOR as a string constant",
    ).not.toBeNull();
    expect(declaration?.[1]).toBe(PAGE_EDIT_DAEMON_FLOOR);
  });

  it("is a stable release the gate accepts", () => {
    expect(daemonMeetsFloor(PAGE_EDIT_DAEMON_FLOOR, PAGE_EDIT_DAEMON_FLOOR)).toBe(
      true,
    );
  });
});
