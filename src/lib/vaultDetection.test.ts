// SPDX-License-Identifier: AGPL-3.0-only
import { describe, it, expect, vi, beforeEach } from "vitest";

const mocks = vi.hoisted(() => ({ readDir: vi.fn() }));
vi.mock("@tauri-apps/plugin-fs", () => ({ readDir: mocks.readDir }));

import { detectVault, MAX_ENTRIES } from "./vaultDetection";

type Entry = { name: string; isDirectory: boolean };
const file = (name: string): Entry => ({ name, isDirectory: false });
const dir = (name: string): Entry => ({ name, isDirectory: true });

/** Wire a fake filesystem: map of absolute path → entries. */
function fakeFs(tree: Record<string, Entry[]>) {
  mocks.readDir.mockImplementation(async (p: string) => {
    if (!(p in tree)) throw new Error(`ENOENT: ${p}`);
    return tree[p];
  });
}

describe("detectVault", () => {
  // Block body: an expression body would return mockReset()'s chainable
  // result (the mock function itself), and Vitest treats a function
  // returned from beforeEach as a teardown callback — invoking readDir()
  // a second time with no args after every test.
  beforeEach(() => {
    mocks.readDir.mockReset();
  });

  it("counts markdown at the top level of an Obsidian vault", async () => {
    fakeFs({ "/v": [dir(".obsidian"), file("a.md"), file("b.md"), file("c.pdf")] });
    const d = await detectVault("/v");
    expect(d.isVault).toBe(true);
    expect(d.sourceType).toBe("obsidian");
    // obsidian sources are markdown-only (daemon has_any_markdown) — c.pdf doesn't count
    expect(d.docCount).toBe(2);
    expect(d.hasValidDoc).toBe(true);
  });

  it("finds notes only in subfolders (the shallow-scan bug)", async () => {
    fakeFs({
      "/v": [dir(".obsidian"), dir("daily")],
      "/v/daily": [file("2026-07-10.md")],
    });
    const d = await detectVault("/v");
    expect(d.docCount).toBe(1);
    expect(d.hasValidDoc).toBe(true);
  });

  it("plain directory counts md/txt/pdf", async () => {
    fakeFs({ "/n": [file("a.md"), file("b.txt"), file("c.pdf"), file("d.docx")] });
    const d = await detectVault("/n");
    expect(d.isVault).toBe(false);
    expect(d.sourceType).toBe("directory");
    expect(d.docCount).toBe(3);
  });

  it("obsidian vault with only txt files has no valid doc", async () => {
    fakeFs({ "/v": [dir(".obsidian"), file("notes.txt")] });
    const d = await detectVault("/v");
    expect(d.sourceType).toBe("obsidian");
    expect(d.hasValidDoc).toBe(false);
    expect(d.docCount).toBe(0);
  });

  it("skips dot-directories and dot-files", async () => {
    fakeFs({
      "/n": [dir(".git"), file(".hidden.md"), file("real.md")],
      // /n/.git is never listed — walking into it would throw
    });
    const d = await detectVault("/n");
    expect(d.docCount).toBe(1);
  });

  it("stops descending beyond depth 6", async () => {
    const tree: Record<string, Entry[]> = {};
    let p = "/r";
    // depth 1 = root; build dirs to depth 8, each with one md
    tree[p] = [dir("d"), file("f1.md")];
    for (let i = 2; i <= 8; i++) {
      p = `${p}/d`;
      tree[p] = [dir("d"), file(`f${i}.md`)];
    }
    fakeFs(tree);
    const d = await detectVault("/r");
    // files at depth 1..6 counted; deeper dirs never entered
    expect(d.docCount).toBe(6);
  });

  it("caps at MAX_ENTRIES and reports countCapped", async () => {
    const many = Array.from({ length: MAX_ENTRIES + 100 }, (_, i) => file(`f${i}.md`));
    fakeFs({ "/big": many });
    const d = await detectVault("/big");
    expect(d.countCapped).toBe(true);
    expect(d.docCount).toBeLessThanOrEqual(MAX_ENTRIES);
  });

  it("unreadable root allows submit (daemon is the authority)", async () => {
    mocks.readDir.mockRejectedValue(new Error("EACCES"));
    const d = await detectVault("/locked");
    expect(d.unreadable).toBe(true);
    expect(d.hasValidDoc).toBe(false);
  });

  it("unreadable subdirectory is skipped, walk continues", async () => {
    fakeFs({ "/n": [dir("locked"), file("ok.md")] }); // /n/locked missing → throws
    const d = await detectVault("/n");
    expect(d.docCount).toBe(1);
    expect(d.unreadable).toBe(false);
  });
});
