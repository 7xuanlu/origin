import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const script = readFileSync(resolve(root, "scripts/refactor/inventory.sh"), "utf8");
const tempRoots: string[] = [];

afterEach(() => {
  for (const dir of tempRoots.splice(0)) {
    rmSync(dir, { force: true, recursive: true });
  }
});

function makeTempRoot(): string {
  const dir = mkdtempSync(resolve(tmpdir(), "wenlan-app-inventory-"));
  tempRoots.push(dir);
  return dir;
}

function writeFixtureRepo(path: string): void {
  mkdirSync(resolve(path, "scripts/refactor"), { recursive: true });
  mkdirSync(resolve(path, "docs/superpowers/refactor"), { recursive: true });
  mkdirSync(resolve(path, "src/lib"), { recursive: true });
  mkdirSync(resolve(path, "app/src"), { recursive: true });
  mkdirSync(resolve(path, "app/tests"), { recursive: true });

  writeFileSync(resolve(path, "scripts/refactor/inventory.sh"), script, { mode: 0o755 });
  writeFileSync(resolve(path, "src/lib/tauri.ts"), `import { invoke } from "@tauri-apps/api/core";
export function ping() { return invoke("ping"); }
`);
  writeFileSync(resolve(path, "app/src/lib.rs"), `tauri::generate_handler![search::ping];
`);
  writeFileSync(resolve(path, "app/src/api.rs"), `pub struct HealthResponse { pub status: String }
`);
  writeFileSync(resolve(path, "app/src/search.rs"), `pub struct PingResponse { pub ok: bool }
`);
  writeFileSync(resolve(path, "package.json"), `{"name":"wenlan-app"}
`);
  writeFileSync(resolve(path, "README.md"), `# wenlan-app
`);
  writeFileSync(resolve(path, "Cargo.toml"), `[workspace]
`);

  execFileSync("git", ["init", "-q"], { cwd: path });
}

function writeFakeTools(binRoot: string): void {
  mkdirSync(binRoot, { recursive: true });
  const fake = `#!/usr/bin/env bash
last=""
for arg in "$@"; do
  last="$arg"
done
if [[ -d "$last" ]]; then
  echo "$last/generated.rs:1: pub struct Generated {}"
else
  echo "$last:1: generated outline"
fi
`;
  const npxPath = resolve(binRoot, "npx");
  writeFileSync(npxPath, fake);
  chmodSync(npxPath, 0o755);

  const rg = `#!/usr/bin/env bash
if [[ "\${1:-}" == "--files" ]]; then
  shift
  for dir in "$@"; do
    if [[ -d "$dir" ]]; then
      find "$dir" -type f
    fi
  done
  exit 0
fi
exit 1
`;
  const rgPath = resolve(binRoot, "rg");
  writeFileSync(rgPath, rg);
  chmodSync(rgPath, 0o755);
}

describe("refactor inventory", () => {
  // 30s, not the 5s default: this shells out to the whole inventory pipeline via
  // spawnSync, which blocks the worker's event loop for the subprocess's entire
  // life. It runs in ~0.7s alone, but under the full suite's parallel workers it
  // crossed 5s about one run in six — a red CI that had nothing to do with the
  // code under test.
  it("writes repo-relative artifact paths so worktree paths do not churn", { timeout: 30_000 }, () => {
    const appRoot = resolve(makeTempRoot(), "wenlan-app");
    const binRoot = resolve(appRoot, "bin");
    writeFixtureRepo(appRoot);
    writeFakeTools(binRoot);

    const result = spawnSync("bash", ["scripts/refactor/inventory.sh"], {
      cwd: appRoot,
      encoding: "utf8",
      env: { ...process.env, PATH: `${binRoot}:${process.env.PATH ?? ""}` },
    });

    expect(result.status, result.stderr).toBe(0);
    const outDir = resolve(appRoot, "docs/superpowers/refactor/wenlan-app-inventory");
    const artifacts = [
      "summary.md",
      "tauri-ts-outline.txt",
      "api-rs-outline.txt",
      "search-rs-outline.txt",
      "frontend-invokes.txt",
      "rust-structs.txt",
    ];
    for (const artifact of artifacts) {
      const content = readFileSync(resolve(outDir, artifact), "utf8");
      expect(content).not.toContain(appRoot);
    }
    expect(readFileSync(resolve(outDir, "tauri-ts-outline.txt"), "utf8")).toContain("src/lib/tauri.ts");
    expect(readFileSync(resolve(outDir, "frontend-invokes.txt"), "utf8")).toContain("src/generated.rs");
    expect(readFileSync(resolve(outDir, "api-rs-outline.txt"), "utf8")).toContain("app/src/api.rs");
  });
});
