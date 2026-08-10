// SPDX-License-Identifier: AGPL-3.0-only
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  bashExecutable,
  canonicalBashPath,
  canonicalizePathEnvironment,
  minimalBashPath,
  prependNativePath,
} from "./test-platform";

const tempRoots: string[] = [];

afterEach(() => {
  for (const dir of tempRoots.splice(0)) {
    rmSync(dir, { force: true, recursive: true, maxRetries: 20, retryDelay: 250 });
  }
});

describe("cross-platform shell test helpers", () => {
  it("prepends PATH entries with the host-native delimiter", () => {
    expect(prependNativePath("first", "second")).toBe(`first${delimiter}second`);
    expect(prependNativePath("first", "")).toBe("first");
  });

  it("resolves the Bash launcher independently of a child PATH override", () => {
    expect(bashExecutable()).toMatch(process.platform === "win32" ? /bash\.exe$/i : /bash$/);
    if (process.platform === "win32") {
      expect(bashExecutable()).not.toMatch(/[\\/]Windows[\\/]System32[\\/]bash\.exe$/i);
    }
  });

  it("provides a minimal native PATH that can launch Bash scripts", () => {
    expect(minimalBashPath().split(delimiter)).not.toContain("");
  });

  it("collapses Windows Path/PATH aliases so an override wins deterministically", () => {
    expect(canonicalizePathEnvironment({
      Path: "inherited",
      HOME: "home",
      PATH: "override",
    }, "win32")).toEqual({
      HOME: "home",
      PATH: "override",
    });
  });

  it("preserves case-sensitive environment keys on non-Windows hosts", () => {
    expect(canonicalizePathEnvironment({
      Path: "mixed-case",
      PATH: "uppercase",
    }, "linux")).toEqual({
      Path: "mixed-case",
      PATH: "uppercase",
    });
  });

  it("canonicalizes a host directory to the spelling Bash reports", () => {
    const base = realpathSync(mkdtempSync(resolve(tmpdir(), "wenlan-bash-path-")));
    tempRoots.push(base);
    const nested = resolve(base, "nested");
    mkdirSync(nested);

    const path = canonicalBashPath(nested);

    expect(path).toMatch(/\/nested$/);
    expect(path).not.toContain("\\");
  });
});
