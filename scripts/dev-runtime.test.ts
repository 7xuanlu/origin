import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const tempRoots: string[] = [];

afterEach(() => {
  for (const path of tempRoots.splice(0)) {
    rmSync(path, { recursive: true, force: true });
  }
});

describe("scoped dev runtime", () => {
  it("routes dev lifecycle commands through worktree-owned scripts", () => {
    const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const scripts = packageJson.scripts as Record<string, string>;

    expect(scripts["dev:daemon"]).toBe("bash scripts/dev-runtime.sh start");
    expect(scripts["clean:dev"]).toBe("bash scripts/dev-runtime.sh stop");
    expect(scripts["dev:all"]).toBe("bash scripts/dev-all.sh");

    const lifecycleCommands = [
      scripts["dev:daemon"],
      scripts["clean:dev"],
      scripts["dev:all"],
    ].join("\n");
    expect(lifecycleCommands).not.toContain("pkill");
    expect(lifecycleCommands).not.toContain("lsof -ti :7878");
    expect(lifecycleCommands).not.toContain("kill -9");
  });

  it("defaults to an isolated non-production port and data directory", () => {
    const tempRoot = mkdtempSync(resolve(tmpdir(), "wenlan-app-dev-test-"));
    tempRoots.push(tempRoot);

    const result = spawnSync("bash", ["scripts/dev-runtime.sh", "print-config"], {
      cwd: root,
      encoding: "utf8",
      env: {
        ...process.env,
        TMPDIR: `${tempRoot}/`,
      },
    });

    expect(result.status, result.stderr).toBe(0);
    const config = Object.fromEntries(
      result.stdout
        .trim()
        .split("\n")
        .map((line) => line.split("=", 2)),
    );
    expect(config.WENLAN_PORT).toMatch(/^\d+$/);
    expect(config.WENLAN_PORT).not.toBe("7878");
    expect(config.WENLAN_DEV_UI_PORT).toMatch(/^\d+$/);
    expect(config.WENLAN_DEV_UI_PORT).not.toBe("1420");
    expect(config.WENLAN_DEV_REMOTE_PORT_START).toMatch(/^\d+$/);
    expect(Number(config.WENLAN_DEV_REMOTE_PORT_START)).toBeGreaterThanOrEqual(20_000);
    expect(config.WENLAN_DEV_APP_ID).toMatch(/^com\.wenlan\.desktop\.dev\.\d+$/);
    expect(config.WENLAN_DEV_TAURI_MCP_SOCKET).toContain(tempRoot);
    expect(config.WENLAN_DEV_TAURI_MCP_SOCKET).toMatch(/tauri-mcp\.sock$/);
    expect(config.WENLAN_DATA_DIR).toContain(tempRoot);
    expect(config.WENLAN_DATA_DIR).toContain("wenlan-app-dev");
  });

  it.each([
    ["WENLAN_DEV_PORT", "7878"],
    ["WENLAN_DEV_UI_PORT", "1420"],
    ["WENLAN_DEV_APP_ID", "com.wenlan.desktop"],
    ["WENLAN_DEV_TAURI_MCP_SOCKET", "/tmp/tauri-mcp.sock"],
    ["WENLAN_DEV_REMOTE_PORT_START", "18080"],
  ])("rejects production identity override %s=%s", (key, value) => {
    const result = spawnSync("bash", ["scripts/dev-runtime.sh", "print-config"], {
      cwd: root,
      encoding: "utf8",
      env: {
        ...process.env,
        [key]: value,
      },
    });

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("refusing production");
  });

  it.each([
    ["Library/Application Support/wenlan"],
    ["Library/Application Support/origin"],
    [".origin"],
    [".config/origin-mcp"],
  ])("rejects the production data directory override %s", (suffix) => {
    const home = process.env.HOME;
    expect(home).toBeTruthy();
    const result = spawnSync("bash", ["scripts/dev-runtime.sh", "print-config"], {
      cwd: root,
      encoding: "utf8",
      env: {
        ...process.env,
        WENLAN_DEV_DATA_DIR: resolve(home!, suffix),
      },
    });

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("refusing production");
  });

  it.each([
    ["Library/LaunchAgents"],
    ["Library/Logs/com.wenlan.desktop"],
    ["Library/Logs/com.origin.desktop"],
  ])("rejects a dev state directory under the production root %s", (suffix) => {
    const home = process.env.HOME;
    expect(home).toBeTruthy();
    const result = spawnSync("bash", ["scripts/dev-runtime.sh", "print-config"], {
      cwd: root,
      encoding: "utf8",
      env: {
        ...process.env,
        WENLAN_DEV_STATE_DIR: resolve(home!, suffix),
      },
    });

    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("refusing production");
  });

  it("detaches the daemon from the lifecycle command", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");

    expect(script).toContain("nohup env");
    expect(script).toContain("</dev/null");
  });

  it("claims a daemon only after the spawned PID owns the selected listener", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");

    expect(script).toContain("listener_pid_for_port");
    expect(script).toContain('[[ "$listener_pid" == "$pid" ]]');
    expect(script).toContain("has_owned_command_identity");
    expect(script).toContain("acquire_runtime_lock");
    expect(script).toContain("wenlan-server.data-dir");
  });

  it("refuses to reuse a worktree daemon opened on a different data directory", () => {
    const tempRoot = mkdtempSync(resolve(tmpdir(), "wenlan-dev-data-identity-test-"));
    tempRoots.push(tempRoot);
    const backend = resolve(tempRoot, "wenlan");
    const server = resolve(backend, "target/debug/wenlan-server");
    const stateDir = resolve(tempRoot, "state");
    const originalDataDir = resolve(stateDir, "data-original");
    const changedDataDir = resolve(stateDir, "data-changed");
    const fakeBin = resolve(tempRoot, "bin");

    mkdirSync(resolve(backend, "crates/wenlan-server"), { recursive: true });
    mkdirSync(resolve(backend, "crates/wenlan-mcp"), { recursive: true });
    mkdirSync(resolve(backend, "crates/wenlan-cli"), { recursive: true });
    mkdirSync(resolve(backend, "target/debug"), { recursive: true });
    mkdirSync(originalDataDir, { recursive: true });
    mkdirSync(changedDataDir, { recursive: true });
    mkdirSync(fakeBin, { recursive: true });
    writeFileSync(resolve(backend, "Cargo.toml"), "[workspace]\n");
    symlinkSync("/bin/sleep", server);
    writeFileSync(
      resolve(fakeBin, "lsof"),
      '#!/usr/bin/env bash\nprintf \'%s\\n\' "$FAKE_DAEMON_PID"\n',
    );
    chmodSync(resolve(fakeBin, "lsof"), 0o755);

    const daemon = spawn(server, ["60"], { stdio: "ignore" });
    expect(daemon.pid).toBeDefined();
    writeFileSync(resolve(stateDir, "wenlan-server.pid"), `${daemon.pid}\n`);
    writeFileSync(resolve(stateDir, "wenlan-server.path"), `${server}\n`);
    writeFileSync(resolve(stateDir, "wenlan-server.port"), "27992\n");
    writeFileSync(
      resolve(stateDir, "wenlan-server.data-dir"),
      `${realpathSync(originalDataDir)}\n`,
    );

    try {
      const result = spawnSync("bash", ["scripts/dev-runtime.sh", "start"], {
        cwd: root,
        encoding: "utf8",
        env: {
          ...process.env,
          PATH: `${fakeBin}:${process.env.PATH}`,
          WENLAN_BACKEND_DIR: backend,
          WENLAN_DEV_STATE_DIR: stateDir,
          WENLAN_DEV_DATA_DIR: changedDataDir,
          WENLAN_DEV_PORT: "27992",
          WENLAN_DEV_UI_PORT: "28992",
          FAKE_DAEMON_PID: `${daemon.pid}`,
        },
      });

      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("identity does not match");
    } finally {
      daemon.kill("SIGKILL");
    }
  });

  it("passes sidecar flags through pnpm without a literal separator", () => {
    const script = readFileSync(resolve(root, "scripts/dev-all.sh"), "utf8");

    expect(script).toContain("pnpm prepare:sidecars --force-build");
    expect(script).not.toContain("pnpm prepare:sidecars -- --force-build");
  });

  it("dev:all leaves a pre-existing worktree daemon running", () => {
    const tempRoot = mkdtempSync(resolve(tmpdir(), "wenlan-dev-owner-test-"));
    tempRoots.push(tempRoot);
    const backend = resolve(tempRoot, "wenlan");
    const server = resolve(backend, "target/debug/wenlan-server");
    const stateDir = resolve(tempRoot, "state");
    const fakeBin = resolve(tempRoot, "bin");
    const pnpmEnvLog = resolve(tempRoot, "pnpm-env.log");

    mkdirSync(resolve(backend, "crates/wenlan-server"), { recursive: true });
    mkdirSync(resolve(backend, "crates/wenlan-mcp"), { recursive: true });
    mkdirSync(resolve(backend, "crates/wenlan-cli"), { recursive: true });
    mkdirSync(resolve(backend, "target/debug"), { recursive: true });
    mkdirSync(stateDir, { recursive: true });
    mkdirSync(fakeBin, { recursive: true });
    writeFileSync(resolve(backend, "Cargo.toml"), "[workspace]\n");
    symlinkSync("/bin/sleep", server);
    writeFileSync(
      resolve(fakeBin, "pnpm"),
      '#!/usr/bin/env bash\nprintf \'%s\\n\' "${WENLAN_DEV_PRESERVE_DAEMON_ON_QUIT:-unset}" >> "$FAKE_PNPM_ENV_LOG"\nexit 0\n',
    );
    writeFileSync(
      resolve(fakeBin, "lsof"),
      '#!/usr/bin/env bash\nprintf \'%s\\n\' "$FAKE_DAEMON_PID"\n',
    );
    chmodSync(resolve(fakeBin, "pnpm"), 0o755);
    chmodSync(resolve(fakeBin, "lsof"), 0o755);

    const daemon = spawn(server, ["60"], { stdio: "ignore" });
    expect(daemon.pid).toBeDefined();
    writeFileSync(resolve(stateDir, "wenlan-server.pid"), `${daemon.pid}\n`);
    writeFileSync(resolve(stateDir, "wenlan-server.path"), `${server}\n`);
    writeFileSync(resolve(stateDir, "wenlan-server.port"), "27991\n");
    writeFileSync(
      resolve(stateDir, "wenlan-server.data-dir"),
      `${resolve(realpathSync(stateDir), "data")}\n`,
    );

    try {
      const result = spawnSync("bash", ["scripts/dev-all.sh"], {
        cwd: root,
        encoding: "utf8",
        env: {
          ...process.env,
          PATH: `${fakeBin}:${process.env.PATH}`,
          WENLAN_BACKEND_DIR: backend,
          WENLAN_DEV_STATE_DIR: stateDir,
          WENLAN_DEV_PORT: "27991",
          WENLAN_DEV_UI_PORT: "28991",
          FAKE_DAEMON_PID: `${daemon.pid}`,
          FAKE_PNPM_ENV_LOG: pnpmEnvLog,
        },
      });

      expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
      expect(() => process.kill(daemon.pid!, 0)).not.toThrow();
      expect(readFileSync(pnpmEnvLog, "utf8").trim().split("\n")).toContain("1");
    } finally {
      daemon.kill("SIGKILL");
    }
  }, 10_000);

  it("routes Vite and Tauri through the worktree-owned UI port", () => {
    const devAll = readFileSync(resolve(root, "scripts/dev-all.sh"), "utf8");
    const viteConfig = readFileSync(resolve(root, "vite.config.ts"), "utf8");

    expect(devAll).toContain("WENLAN_PORT|WENLAN_DEV_UI_PORT|");
    expect(devAll).toContain("WENLAN_DEV_APP_ID|");
    expect(devAll).toContain("WENLAN_DEV_TAURI_MCP_SOCKET|");
    expect(devAll).toContain("WENLAN_DEV_REMOTE_PORT_START|");
    expect(devAll).toContain('[[ -S "$WENLAN_DEV_TAURI_MCP_SOCKET" ]]');
    expect(devAll).toContain('rm -f "$WENLAN_DEV_TAURI_MCP_SOCKET"');
    expect(devAll).toContain('identifier\\":\\"$WENLAN_DEV_APP_ID');
    expect(devAll).toContain('devUrl\\":\\"http://localhost:$WENLAN_DEV_UI_PORT');
    expect(viteConfig).toContain("process.env.WENLAN_DEV_UI_PORT");
  });

  it("remote access passes the selected dev daemon URL to wenlan-mcp", () => {
    const remoteAccess = readFileSync(resolve(root, "app/src/remote_access.rs"), "utf8");

    expect(remoteAccess).toContain('"--origin-url"');
    expect(remoteAccess).toContain("crate::api::WenlanClient::new().base_url()");
    expect(remoteAccess).toContain("mcp_child.pid()");
    expect(remoteAccess).toContain("listener_pid_for_port");
    expect(remoteAccess).toContain("wait_for_generation_change");
    expect(remoteAccess).toContain("Remote access start cancelled.");
  });

  it("remote cleanup verifies listener identity before sending signals", () => {
    const remoteAccess = readFileSync(resolve(root, "app/src/remote_access.rs"), "utf8");

    expect(remoteAccess).toContain("wenlan_mcp_process_identity");
    expect(remoteAccess).toContain("cloudflared_process_identity");
    expect(remoteAccess).toContain("refusing to kill non-wenlan-mcp listener");
  });

  it("documents dev:all as the supported isolated app entry point", () => {
    const readme = readFileSync(resolve(root, "README.md"), "utf8");

    expect(readme).toContain("pnpm dev:all");
    expect(readme).not.toContain("pnpm tauri dev");
  });
});
