import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scriptPath = resolve(root, "scripts/prepare-sidecars.sh");
const tauriBuildScriptPath = resolve(root, "scripts/prepare-tauri-build-sidecars.sh");
const resolverScriptPath = resolve(root, "scripts/resolve-backend-dir.sh");
const devRuntimeScriptPath = resolve(root, "scripts/dev-runtime.sh");
const tempRoots: string[] = [];
const pathOverrideEnvKeys = new Set([
  "WENLAN_BACKEND_DIR",
  "CARGO_TARGET_DIR",
  "TARGET_TRIPLE",
  "TAURI_ENV_DEBUG",
  "TAURI_ENV_TARGET_TRIPLE",
]);

afterEach(() => {
  for (const dir of tempRoots.splice(0)) {
    rmSync(dir, { force: true, recursive: true });
  }
});

function makeTempRoot(): string {
  const dir = realpathSync(mkdtempSync(resolve(tmpdir(), "wenlan-app-sidecars-")));
  tempRoots.push(dir);
  return dir;
}

function writeBackendRepo(path: string): void {
  mkdirSync(resolve(path, "crates/wenlan-server"), { recursive: true });
  mkdirSync(resolve(path, "crates/wenlan-mcp"), { recursive: true });
  mkdirSync(resolve(path, "crates/wenlan-cli"), { recursive: true });
  writeFileSync(resolve(path, "Cargo.toml"), "[workspace]\n");
}

function writeExecutable(path: string, content = "#!/usr/bin/env bash\nexit 0\n"): void {
  writeFileSync(path, content, { mode: 0o755 });
}

function writeAppScripts(appRoot: string): void {
  mkdirSync(resolve(appRoot, "scripts"), { recursive: true });
  copyFileSync(scriptPath, resolve(appRoot, "scripts/prepare-sidecars.sh"));
  copyFileSync(tauriBuildScriptPath, resolve(appRoot, "scripts/prepare-tauri-build-sidecars.sh"));
  copyFileSync(resolverScriptPath, resolve(appRoot, "scripts/resolve-backend-dir.sh"));
  copyFileSync(devRuntimeScriptPath, resolve(appRoot, "scripts/dev-runtime.sh"));
}

function childEnv(overrides: Record<string, string> = {}): Record<string, string> {
  const childEnv: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value === undefined) continue;
    if (pathOverrideEnvKeys.has(key)) {
      continue;
    }
    childEnv[key] = value;
  }
  return { ...childEnv, ...overrides };
}

function printPaths(appRoot: string, env: Record<string, string> = {}): string {
  return execFileSync("bash", ["scripts/prepare-sidecars.sh", "--print-paths"], {
    cwd: appRoot,
    encoding: "utf8",
    env: childEnv(env),
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function printTauriBuildPaths(appRoot: string, env: Record<string, string> = {}): string {
  return execFileSync("bash", ["scripts/prepare-tauri-build-sidecars.sh", "--print-paths"], {
    cwd: appRoot,
    encoding: "utf8",
    env: childEnv(env),
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function resolveBackend(appRoot: string, env: Record<string, string> = {}): string {
  return execFileSync("bash", ["scripts/resolve-backend-dir.sh"], {
    cwd: appRoot,
    encoding: "utf8",
    env: childEnv(env),
    stdio: ["ignore", "pipe", "pipe"],
  }).trim();
}

function restoreEnv(name: string, value: string | undefined): void {
  if (value === undefined) {
    delete process.env[name];
  } else {
    process.env[name] = value;
  }
}

describe("prepare-sidecars backend discovery", () => {
  it("discovers a sibling wenlan backend from a standalone wenlan-app checkout", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const output = printPaths(appRoot);

    expect(output).toContain(`server_src=${backendRoot}/target/debug/wenlan-server`);
    expect(output).toContain(`mcp_src=${backendRoot}/target/debug/wenlan-mcp`);
    expect(output).toContain(`cli_src=${backendRoot}/target/debug/wenlan`);
  });

  it("discovers a sibling wenlan backend from a project-local worktree checkout", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app/.worktrees/launch-smoke");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const output = printPaths(appRoot);

    expect(output).toContain(`server_src=${backendRoot}/target/debug/wenlan-server`);
    expect(output).toContain(`mcp_src=${backendRoot}/target/debug/wenlan-mcp`);
    expect(output).toContain(`cli_src=${backendRoot}/target/debug/wenlan`);
  });

  it("defaults to the monorepo root when no backend override is set", () => {
    const base = makeTempRoot();
    // Deliberately not named "wenlan": the old resolver's sibling probe
    // ($REPO_ROOT/../wenlan) would otherwise resolve back to this same
    // directory whenever the checkout happens to be named "wenlan" (true
    // locally and on GitHub runners), making this case pass vacuously.
    const monoRoot = resolve(base, "monorepo");
    writeAppScripts(monoRoot);
    writeBackendRepo(monoRoot);

    expect(resolveBackend(monoRoot)).toBe(monoRoot);
  });

  it("keeps relative WENLAN_BACKEND_DIR overrides relative to the app checkout", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(appRoot, "local-backend");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const output = printPaths(appRoot, { WENLAN_BACKEND_DIR: "local-backend" });

    expect(output).toContain(`server_src=${backendRoot}/target/debug/wenlan-server`);
    expect(output).toContain(`mcp_src=${backendRoot}/target/debug/wenlan-mcp`);
    expect(output).toContain(`cli_src=${backendRoot}/target/debug/wenlan`);
  });

  it("uses Tauri target triples when Tauri runs sidecar prep for target builds", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const output = printPaths(appRoot, {
      TAURI_ENV_TARGET_TRIPLE: "x86_64-apple-darwin",
    });

    expect(output).toContain(`server_src=${backendRoot}/target/x86_64-apple-darwin/debug/wenlan-server`);
    expect(output).toContain(`mcp_src=${backendRoot}/target/x86_64-apple-darwin/debug/wenlan-mcp`);
    expect(output).toContain(`cli_src=${backendRoot}/target/x86_64-apple-darwin/debug/wenlan`);
    expect(output).toContain(`server_dest=${appRoot}/app/binaries/wenlan-server-x86_64-apple-darwin`);
    expect(output).toContain(`mcp_dest=${appRoot}/app/binaries/wenlan-mcp-x86_64-apple-darwin`);
    expect(output).toContain(`cli_dest=${appRoot}/app/binaries/wenlan-x86_64-apple-darwin`);
  });

  it("uses release sidecars when Tauri runs sidecar prep for release builds", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const output = printPaths(appRoot, {
      TAURI_ENV_DEBUG: "false",
    });

    expect(output).toContain(`server_src=${backendRoot}/target/release/wenlan-server`);
    expect(output).toContain(`mcp_src=${backendRoot}/target/release/wenlan-mcp`);
    expect(output).toContain(`cli_src=${backendRoot}/target/release/wenlan`);
  });

  it("uses release sidecars by default from the Tauri build hook", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const output = printTauriBuildPaths(appRoot);

    expect(output).toContain(`server_src=${backendRoot}/target/release/wenlan-server`);
    expect(output).toContain(`mcp_src=${backendRoot}/target/release/wenlan-mcp`);
    expect(output).toContain(`cli_src=${backendRoot}/target/release/wenlan`);
  });

  it("uses debug sidecars from the Tauri build hook when Tauri is building debug", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const output = printTauriBuildPaths(appRoot, {
      TAURI_ENV_DEBUG: "true",
    });

    expect(output).toContain(`server_src=${backendRoot}/target/debug/wenlan-server`);
    expect(output).toContain(`mcp_src=${backendRoot}/target/debug/wenlan-mcp`);
    expect(output).toContain(`cli_src=${backendRoot}/target/debug/wenlan`);
  });

  it("ignores inherited path overrides while testing default discovery", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    const originalBackendDir = process.env.WENLAN_BACKEND_DIR;
    const originalCargoTargetDir = process.env.CARGO_TARGET_DIR;
    const originalTargetTriple = process.env.TARGET_TRIPLE;
    process.env.WENLAN_BACKEND_DIR = resolve(base, "not-a-backend");
    process.env.CARGO_TARGET_DIR = resolve(base, "custom-target");
    process.env.TARGET_TRIPLE = "x86_64-unknown-linux-gnu";
    try {
      const output = printPaths(appRoot);

      expect(output).toContain(`server_src=${backendRoot}/target/debug/wenlan-server`);
      expect(output).toContain(`mcp_src=${backendRoot}/target/debug/wenlan-mcp`);
      expect(output).toContain(`cli_src=${backendRoot}/target/debug/wenlan`);
    } finally {
      restoreEnv("WENLAN_BACKEND_DIR", originalBackendDir);
      restoreEnv("CARGO_TARGET_DIR", originalCargoTargetDir);
      restoreEnv("TARGET_TRIPLE", originalTargetTriple);
    }
  });

  it("fails loud for invalid WENLAN_BACKEND_DIR overrides", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    writeAppScripts(appRoot);

    let error: unknown;
    try {
      printPaths(appRoot, { WENLAN_BACKEND_DIR: "not-a-backend" });
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeTruthy();
    const stderr = String((error as { stderr?: unknown }).stderr ?? "");
    expect(stderr).toContain("WENLAN_BACKEND_DIR is not a Wenlan backend checkout");
  });

  it("exposes the same backend resolver for dev scripts", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);

    expect(resolveBackend(appRoot)).toBe(backendRoot);
  });

  it("uses the shared backend resolver from dev:daemon", () => {
    const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const devDaemon = String(packageJson.scripts["dev:daemon"]);
    const devRuntime = readFileSync(devRuntimeScriptPath, "utf8");

    expect(devDaemon).toBe("bash scripts/dev-runtime.sh start");
    expect(devRuntime).toContain('resolve-backend-dir.sh" "$REPO_ROOT"');
    expect(devRuntime).not.toContain("${WENLAN_BACKEND_DIR:-../..}");
  });

  it("uses the release-aware sidecar prep wrapper from Tauri build config", () => {
    const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const tauri = JSON.parse(readFileSync(resolve(root, "app/tauri.conf.json"), "utf8"));

    expect(packageJson.scripts["prepare:sidecars:tauri-build"]).toBe(
      "bash scripts/prepare-tauri-build-sidecars.sh",
    );
    expect(tauri.build.beforeDevCommand).toContain("pnpm prepare:sidecars");
    expect(tauri.build.beforeBuildCommand).toContain("pnpm prepare:sidecars:tauri-build");
  });

  it("does not reach cargo when dev:daemon backend resolution fails", () => {
    const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const devDaemon = String(packageJson.scripts["dev:daemon"]);
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const binRoot = resolve(base, "bin");
    writeAppScripts(appRoot);
    mkdirSync(binRoot, { recursive: true });
    writeFileSync(resolve(binRoot, "cargo"), "#!/usr/bin/env bash\necho cargo should not run >&2\nexit 99\n", { mode: 0o755 });

    const result = spawnSync("bash", ["-lc", devDaemon], {
      cwd: appRoot,
      encoding: "utf8",
      env: childEnv({
        WENLAN_BACKEND_DIR: "not-a-backend",
        PATH: `${binRoot}:${process.env.PATH ?? ""}`,
      }),
    });

    expect(result.status).toBe(1);
    expect(result.stderr).toContain("WENLAN_BACKEND_DIR is not a Wenlan backend checkout");
    expect(result.stderr).not.toContain("cargo should not run");
  });

  it("fails loud when cloudflared is required but missing", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    const backendRoot = resolve(base, "wenlan");
    const binRoot = resolve(base, "bin");
    writeAppScripts(appRoot);
    writeBackendRepo(backendRoot);
    mkdirSync(resolve(backendRoot, "target/debug"), { recursive: true });
    writeExecutable(resolve(backendRoot, "target/debug/wenlan-server"));
    writeExecutable(resolve(backendRoot, "target/debug/wenlan-mcp"));
    writeExecutable(resolve(backendRoot, "target/debug/wenlan"));
    mkdirSync(binRoot, { recursive: true });
    writeExecutable(
      resolve(binRoot, "rustc"),
      "#!/usr/bin/env bash\nprintf 'host: aarch64-apple-darwin\\n'\n",
    );

    const result = spawnSync("bash", ["scripts/prepare-sidecars.sh"], {
      cwd: appRoot,
      encoding: "utf8",
      env: childEnv({
        PATH: `${binRoot}:/usr/bin:/bin`,
      }),
    });

    expect(result.status).toBe(1);
    expect(result.stderr).toContain("cloudflared not found in PATH");
    expect(result.stderr).toContain("Required by Tauri externalBin");
  });
});

