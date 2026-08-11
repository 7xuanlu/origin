import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scriptPath = resolve(root, "scripts/run-tauri.mjs");

describe("run-tauri wrapper", () => {
  // The regression this guards: `TAURI_DIR=app tauri` is POSIX shell syntax,
  // and pnpm hands package.json scripts to cmd.exe on Windows, which reads the
  // assignment as a command name. Any env prefix here breaks the Windows build.
  it("is how package.json invokes Tauri, with no shell env prefix", () => {
    const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));

    expect(pkg.scripts.tauri).toBe("node scripts/run-tauri.mjs");
    expect(pkg.scripts.tauri).not.toMatch(/^\w+=/);
  });

  it("still points the CLI at the app directory", () => {
    const source = readFileSync(scriptPath, "utf8");

    expect(source).toContain('TAURI_DIR: "app"');
  });

  it("forwards arguments and the CLI's exit code", () => {
    const ok = spawnSync(process.execPath, [scriptPath, "--version"], {
      cwd: root,
      encoding: "utf8",
    });

    expect(ok.status).toBe(0);
    expect(ok.stdout).toContain("tauri-cli");

    const bad = spawnSync(process.execPath, [scriptPath, "not-a-subcommand"], {
      cwd: root,
      encoding: "utf8",
    });

    expect(bad.status).not.toBe(0);
  });
});
