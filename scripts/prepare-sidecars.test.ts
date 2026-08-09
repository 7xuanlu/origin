import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { execFileSync, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const scriptPath = resolve(root, "scripts/prepare-sidecars.sh");
const tauriBuildScriptPath = resolve(root, "scripts/prepare-tauri-build-sidecars.sh");
const resolverScriptPath = resolve(root, "scripts/resolve-backend-dir.sh");
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

    expect(devDaemon).toContain("scripts/resolve-backend-dir.sh");
    expect(devDaemon).toContain("BACKEND=$(bash scripts/resolve-backend-dir.sh) && cargo build");
    expect(devDaemon).not.toContain("${WENLAN_BACKEND_DIR:-../..}");
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

const DAEMON_DEST_NAMES = [
  "wenlan-aarch64-apple-darwin",
  "wenlan-server-aarch64-apple-darwin",
  "wenlan-mcp-aarch64-apple-darwin",
];

function fakeBinaryContents(name: string): string {
  return `#!/usr/bin/env bash\necho fake-${name}\n`;
}

function buildFakeTarball(dir: string): { tarballPath: string; sha256: string } {
  const contentsDir = resolve(dir, "contents");
  mkdirSync(contentsDir, { recursive: true });
  for (const name of ["wenlan", "wenlan-server", "wenlan-mcp"]) {
    writeFileSync(resolve(contentsDir, name), fakeBinaryContents(name), { mode: 0o755 });
  }
  const tarballPath = resolve(dir, "wenlan-darwin-arm64.tar.gz");
  execFileSync("tar", ["czf", tarballPath, "-C", contentsDir, "wenlan", "wenlan-server", "wenlan-mcp"]);
  const sha256 = createHash("sha256").update(readFileSync(tarballPath)).digest("hex");
  return { tarballPath, sha256 };
}

function writeVersionPin(appRoot: string, tag: string, sha256: string): void {
  writeFileSync(resolve(appRoot, ".wenlan-backend-version"), `${tag}\n${sha256}\n`);
}

function writeFakeGh(binDir: string, opts: { tarballPath?: string; exitCode?: number }): void {
  mkdirSync(binDir, { recursive: true });
  const script = opts.tarballPath
    ? [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'DIR=""',
        'prev=""',
        'for arg in "$@"; do',
        '  if [[ "$prev" == "--dir" ]]; then',
        '    DIR="$arg"',
        "  fi",
        '  prev="$arg"',
        "done",
        'if [[ -z "$DIR" ]]; then',
        '  echo "fake-gh: missing --dir" >&2',
        "  exit 1",
        "fi",
        'mkdir -p "$DIR"',
        `cp "${opts.tarballPath}" "$DIR/wenlan-darwin-arm64.tar.gz"`,
        "",
      ].join("\n")
    : `#!/usr/bin/env bash\nexit ${opts.exitCode ?? 1}\n`;
  writeFileSync(resolve(binDir, "gh"), script, { mode: 0o755 });
}

function writeFakeXattr(binDir: string, logPath: string): void {
  mkdirSync(binDir, { recursive: true });
  writeFileSync(resolve(binDir, "xattr"), `#!/usr/bin/env bash\necho "$@" >> "${logPath}"\n`, {
    mode: 0o755,
  });
}

function runDownload(appRoot: string, env: Record<string, string> = {}) {
  return spawnSync("bash", ["scripts/prepare-sidecars.sh", "--download"], {
    cwd: appRoot,
    encoding: "utf8",
    env: childEnv(env),
  });
}

function withFakeBin(fakeBinDir: string): Record<string, string> {
  return { PATH: `${fakeBinDir}:${process.env.PATH ?? ""}` };
}

describe("prepare-sidecars --download mode", () => {
  it("installs the three daemon sidecars from the pinned release asset", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    writeAppScripts(appRoot);

    const { tarballPath, sha256 } = buildFakeTarball(base);
    writeVersionPin(appRoot, "v0.9.5", sha256);

    const fakeBinDir = resolve(base, "fake-bin");
    writeFakeGh(fakeBinDir, { tarballPath });

    const result = runDownload(appRoot, withFakeBin(fakeBinDir));

    expect(result.status, `stderr: ${result.stderr}`).toBe(0);

    const originalNames = ["wenlan", "wenlan-server", "wenlan-mcp"];
    for (const [destName, originalName] of DAEMON_DEST_NAMES.map(
      (destName, i) => [destName, originalNames[i]] as const,
    )) {
      const dest = resolve(appRoot, "app/binaries", destName);
      expect(existsSync(dest)).toBe(true);
      expect(statSync(dest).mode & 0o777).toBe(0o755);
      expect(readFileSync(dest, "utf8")).toBe(fakeBinaryContents(originalName));
    }
  });

  it("fails loud and installs nothing when the release asset download fails", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    writeAppScripts(appRoot);

    const { sha256 } = buildFakeTarball(base);
    writeVersionPin(appRoot, "v0.9.5", sha256);

    const fakeBinDir = resolve(base, "fake-bin");
    writeFakeGh(fakeBinDir, { exitCode: 17 });

    const result = runDownload(appRoot, withFakeBin(fakeBinDir));

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("failed to download");
    for (const destName of DAEMON_DEST_NAMES) {
      expect(existsSync(resolve(appRoot, "app/binaries", destName))).toBe(false);
    }
  });

  it("fails loud on sha256 mismatch and installs nothing", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    writeAppScripts(appRoot);

    const { tarballPath, sha256 } = buildFakeTarball(base);
    const wrongSha = sha256.slice(0, -1) + (sha256.endsWith("0") ? "1" : "0");
    writeVersionPin(appRoot, "v0.9.5", wrongSha);

    const fakeBinDir = resolve(base, "fake-bin");
    writeFakeGh(fakeBinDir, { tarballPath });

    const result = runDownload(appRoot, withFakeBin(fakeBinDir));

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("sha256 mismatch");
    for (const destName of DAEMON_DEST_NAMES) {
      expect(existsSync(resolve(appRoot, "app/binaries", destName))).toBe(false);
    }
  });

  it("downloads sidecars from the tauri build hook when WENLAN_DOWNLOAD_SIDECARS=1", () => {
    // The release flow has NO backend checkout: the tauri beforeBuildCommand
    // wrapper must take the --download path, not fall through to --release
    // (compile), which would die on resolve-backend-dir.
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    writeAppScripts(appRoot);

    const { tarballPath, sha256 } = buildFakeTarball(base);
    writeVersionPin(appRoot, "v0.9.5", sha256);

    const fakeBinDir = resolve(base, "fake-bin");
    writeFakeGh(fakeBinDir, { tarballPath });

    const result = spawnSync("bash", ["scripts/prepare-tauri-build-sidecars.sh"], {
      cwd: appRoot,
      encoding: "utf8",
      env: childEnv({ ...withFakeBin(fakeBinDir), WENLAN_DOWNLOAD_SIDECARS: "1" }),
    });

    expect(result.status, `stderr: ${result.stderr}`).toBe(0);
    for (const destName of DAEMON_DEST_NAMES) {
      expect(existsSync(resolve(appRoot, "app/binaries", destName))).toBe(true);
    }
  });

  it("strips quarantine via xattr -cr on each installed daemon binary", () => {
    const base = makeTempRoot();
    const appRoot = resolve(base, "wenlan-app");
    writeAppScripts(appRoot);

    const { tarballPath, sha256 } = buildFakeTarball(base);
    writeVersionPin(appRoot, "v0.9.5", sha256);

    const fakeBinDir = resolve(base, "fake-bin");
    writeFakeGh(fakeBinDir, { tarballPath });
    const xattrLog = resolve(base, "xattr.log");
    writeFakeXattr(fakeBinDir, xattrLog);

    const result = runDownload(appRoot, withFakeBin(fakeBinDir));

    expect(result.status, `stderr: ${result.stderr}`).toBe(0);
    const log = readFileSync(xattrLog, "utf8");
    for (const destName of DAEMON_DEST_NAMES) {
      const dest = resolve(appRoot, "app/binaries", destName);
      expect(log).toContain(`-cr ${dest}`);
    }
  });
});
