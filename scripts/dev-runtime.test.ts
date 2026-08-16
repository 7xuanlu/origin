import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve, win32 } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { afterEach, describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const tempRoots: string[] = [];

// The dev runtime is a POSIX shell workflow: these cases spawn `bash`, symlink
// /bin/sleep, chmod a fake `lsof`, and join PATH with ':'. None of that has a
// Windows meaning — and `bash` there is not even guaranteed to be Git Bash: on a
// box with WSL installed it resolves to the Linux distro, which has no Windows
// node or rustc on its PATH. The assertions still guard macOS/Linux; the rest of
// this file (package.json and script-text contracts) stays platform-neutral.
const itPosix = it.skipIf(process.platform === "win32");
// The mirror of the above: cases whose subject is Windows path resolution, run
// through `run-bash.mjs` for the same reason package.json does.
const itWindows = it.skipIf(process.platform !== "win32");
// 8.3 alias generation is off by default on modern volumes, so a machine may
// have no alias to test with. `C:\PROGRA~1` predates that default on most
// installs; where it is absent the alias case has nothing to assert against.
const dosAliasRoot = "C:\\PROGRA~1";
const itWindowsAlias = it.skipIf(
  process.platform !== "win32" || !existsSync(dosAliasRoot),
);

afterEach(() => {
  for (const path of tempRoots.splice(0)) {
    rmSync(path, { recursive: true, force: true });
  }
});

describe("scoped dev runtime", () => {
  it("routes dev lifecycle commands through worktree-owned scripts", () => {
    const packageJson = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const scripts = packageJson.scripts as Record<string, string>;

    // `node scripts/run-bash.mjs`, never a bare `bash`: on a Windows machine
    // with WSL installed the first `bash` on PATH is the Linux distro, which
    // cannot see the Windows toolchain these scripts need.
    expect(scripts["dev:daemon"]).toBe("node scripts/run-bash.mjs scripts/dev-runtime.sh start");
    expect(scripts["clean:dev"]).toBe("node scripts/run-bash.mjs scripts/dev-runtime.sh stop");
    expect(scripts["dev:all"]).toBe("node scripts/run-bash.mjs scripts/dev-all.sh");

    const lifecycleCommands = [
      scripts["dev:daemon"],
      scripts["clean:dev"],
      scripts["dev:all"],
    ].join("\n");
    expect(lifecycleCommands).not.toContain("pkill");
    expect(lifecycleCommands).not.toContain("lsof -ti :7878");
    expect(lifecycleCommands).not.toContain("kill -9");
  });

  itPosix("defaults to an isolated non-production port and data directory", () => {
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

  itPosix.each([
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

  itPosix.each([
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

  itPosix.each([
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

  itWindows.each([
    // Case, a trailing dot, and the \\?\ prefix are all the same directory to
    // Win32 and three different strings to a plain comparison.
    ["WENLAN_DEV_DATA_DIR", "WENLAN"],
    ["WENLAN_DEV_DATA_DIR", "OrIgIn"],
    ["WENLAN_DEV_DATA_DIR", "wenlan."],
    ["WENLAN_DEV_DATA_DIR", "wenlan\\sub\\.."],
    ["WENLAN_DEV_STATE_DIR", "WENLAN"],
  ])(
    "rejects %s pointed at another spelling of a LOCALAPPDATA production root %s",
    (key, name) => {
      const localAppData = process.env.LOCALAPPDATA;
      expect(localAppData).toBeTruthy();
      const result = spawnSync(
        process.execPath,
        ["scripts/run-bash.mjs", "scripts/dev-runtime.sh", "print-config"],
        {
          cwd: root,
          encoding: "utf8",
          env: {
            ...process.env,
            [key]: `${localAppData}\\${name}`,
          },
        },
      );

      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("refusing production");
    },
    // Node, then Git Bash, then a `node -e` per path this canonicalizes. Six
    // seconds is normal on Windows; the 5s default is not enough.
    30_000,
  );

  // `\\?\` and `\\.\` mean "pass this through unchanged", and nothing below can
  // honour that: `realpathSync.native` answers without the prefix, and MSYS and
  // the daemon both go through Win32. So the guard would compare a path whose
  // trailing dots have stopped being literal, and `\\?\%LOCALAPPDATA%\wenlan.`
  // would read as a sibling of production and then be opened as production.
  //
  // Win32 accepts either separator in all three positions and `path.resolve`
  // folds every combination into the same two prefixes, so all sixteen are
  // enumerated here rather than the tidy two. A `wenlan.` tail is used because
  // that is the spelling the guard is standing in front of: dropped by Win32,
  // literal under the prefix, and a sibling of production either way.
  itWindows.each(
    ["\\", "/"].flatMap((a) =>
      ["\\", "/"].flatMap((b) =>
        ["?", "."].flatMap((mark) =>
          ["\\", "/"].map((c) => [`${a}${b}${mark}${c}`] as [string]),
        ),
      ),
    ),
  )(
    "refuses the %s device prefix",
    (prefix) => {
      const localAppData = process.env.LOCALAPPDATA;
      expect(localAppData).toBeTruthy();
      const result = spawnSync(
        process.execPath,
        ["scripts/run-bash.mjs", "scripts/dev-runtime.sh", "print-config"],
        {
          cwd: root,
          encoding: "utf8",
          env: {
            ...process.env,
            WENLAN_DEV_DATA_DIR: `${prefix}${localAppData}\\wenlan.\\dev`,
          },
        },
      );

      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("extended-length or device path");
    },
    30_000,
  );


  itWindows(
    "refuses a verbatim path that has nothing to do with production",
    () => {
      const tempRoot = mkdtempSync(resolve(tmpdir(), "wenlan-verbatim-test-"));
      tempRoots.push(tempRoot);
      // Rewriting this one would silently send the daemon to `dev\data`
      // instead; refusing is the other half of the same contract.
      const result = spawnSync(
        process.execPath,
        ["scripts/run-bash.mjs", "scripts/dev-runtime.sh", "print-config"],
        {
          cwd: root,
          encoding: "utf8",
          env: {
            ...process.env,
            WENLAN_DEV_DATA_DIR: `\\\\?\\${tempRoot}\\dev.\\data`,
          },
        },
      );

      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("extended-length or device path");
    },
    30_000,
  );

  itWindows(
    "allows the same directory written in ordinary form",
    () => {
      const tempRoot = mkdtempSync(resolve(tmpdir(), "wenlan-ordinary-test-"));
      tempRoots.push(tempRoot);
      const result = spawnSync(
        process.execPath,
        ["scripts/run-bash.mjs", "scripts/dev-runtime.sh", "print-config"],
        {
          cwd: root,
          encoding: "utf8",
          env: {
            ...process.env,
            WENLAN_DEV_DATA_DIR: `${tempRoot}\\dev\\data`,
          },
        },
      );

      expect(result.status, result.stderr).toBe(0);
    },
    30_000,
  );

  itWindowsAlias(
    "rejects a production root reached through a DOS 8.3 alias",
    () => {
      // An alias and its long form are two names for one directory, and
      // `realpath` keeps whichever one it was handed. LOCALAPPDATA is
      // overridden here because that is what the guard builds its Windows
      // roots from, and this machine's real one has no alias to exercise.
      const result = spawnSync(
        process.execPath,
        ["scripts/run-bash.mjs", "scripts/dev-runtime.sh", "print-config"],
        {
          cwd: root,
          encoding: "utf8",
          env: {
            ...process.env,
            LOCALAPPDATA: "C:\\Program Files",
            WENLAN_DEV_DATA_DIR: `${dosAliasRoot}\\wenlan`,
          },
        },
      );

      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain("refusing production");
    },
    30_000,
  );

  it("canonicalizes by asking the OS for the real on-disk path", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const start = script.indexOf("canonicalize_path() {");
    expect(start).toBeGreaterThan(-1);
    const body = script.slice(start, script.indexOf("\n}\n", start));

    // Windows spells one directory many ways - a different case, a DOS 8.3
    // alias, a \\?\ prefix, a trailing dot, a junction - and the production
    // guard compares strings. String normalization cannot reduce all of them,
    // so this has to stay a GetFinalPathNameByHandle call.
    expect(body).toContain("realpathSync.native");
  });

  it("compares production roots case-insensitively on Windows only", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const start = script.indexOf("path_is_within() {");
    expect(start).toBeGreaterThan(-1);
    const guard = script.slice(start, script.indexOf("\n}\n", start));

    // Windows resolves %LOCALAPPDATA%\WENLAN and %LOCALAPPDATA%\wenlan to one
    // directory, so an unfolded comparison lets the second spelling walk past
    // the guard and point the dev daemon at production. app-check runs on
    // macOS, so this text contract is what keeps the Windows-only branch under
    // a required lane; the case above proves the behaviour on Windows itself.
    expect(guard).toContain("HOST_IS_WINDOWS == 1");
    expect(guard).toContain("tr '[:upper:]' '[:lower:]'");
  });

  it("refuses extended-length and device paths instead of rewriting them", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const start = script.indexOf("reject_verbatim_path() {");
    expect(start).toBeGreaterThan(-1);
    const guard = script.slice(start, script.indexOf("\n}\n", start));

    // Matched by shape, not by spelling: either separator in all three
    // positions, because path.resolve folds all sixteen combinations into the
    // same two prefixes. Every one of the three dev inputs goes through it, and
    // it is Windows-only because a leading \\ is an ordinary relative filename
    // on POSIX. app-check runs on macOS, so this text contract is what keeps
    // the branch under a required lane; the cases above prove the behaviour.
    expect(guard).toContain("HOST_IS_WINDOWS == 1");
    expect(guard).toContain("'^[\\\\/][\\\\/][?.][\\\\/]'");
    for (const label of [
      "WENLAN_DEV_STATE_DIR",
      "WENLAN_DEV_DATA_DIR",
      "WENLAN_DEV_TAURI_MCP_SOCKET",
    ]) {
      expect(script).toContain(`reject_verbatim_path "${label}"`);
    }
  });

  it("refuses exactly the paths win32 resolves to a device prefix", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const match = script.match(/^\s*local verbatim='(.+)'$/m);
    expect(match).toBeTruthy();
    // The script writes the class as `[\\/]`, which is the same set in a POSIX
    // bracket expression and in a JS regex, so the text can be handed straight
    // to RegExp. Reading it out of the script rather than restating it is what
    // makes this a test of the guard.
    const guard = new RegExp(match![1]);

    // What win32 treats as a device path, taken from path.win32.resolve rather
    // than asserted by hand, so a Node change shows up here instead of in the
    // production guard. Runs on every platform: path.win32 does not need to be
    // on win32, which is the point - app-check is macOS.
    const isDevice = (input: string) =>
      /^\\\\[?.]\\/.test(win32.resolve("C:\\anchor", input));

    const separators = ["\\", "/"];
    const inputs: string[] = [];
    for (const a of separators)
      for (const b of separators)
        for (const mark of ["?", "."])
          for (const c of separators)
            inputs.push(`${a}${b}${mark}${c}C:/scratch/dev./data`);
    inputs.push(
      "\\\\server\\share\\dev.\\data",
      "//server/share/dev./data",
      "C:\\scratch\\dev.\\data",
      "\\\\?x\\C:\\dev",
      "\\\\.x\\C:\\dev",
      "\\?\\C:\\dev",
      "/tmp/wenlan-dev",
    );

    for (const input of inputs) {
      expect(guard.test(input), input).toBe(isDevice(input));
    }
    // Sanity: the sixteen device spellings are actually device spellings, so a
    // guard that matched nothing could not pass the loop above.
    expect(inputs.filter(isDevice)).toHaveLength(16);
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

  itPosix("refuses to reuse a worktree daemon opened on a different data directory", () => {
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

  itPosix("dev:all leaves a pre-existing worktree daemon running", () => {
    const tempRoot = mkdtempSync(resolve(tmpdir(), "wenlan-dev-owner-test-"));
    tempRoots.push(tempRoot);
    const backend = resolve(tempRoot, "wenlan");
    const server = resolve(backend, "target/debug/wenlan-server");
    const stateDir = resolve(tempRoot, "state");
    const fakeBin = resolve(tempRoot, "bin");

    mkdirSync(resolve(backend, "crates/wenlan-server"), { recursive: true });
    mkdirSync(resolve(backend, "crates/wenlan-mcp"), { recursive: true });
    mkdirSync(resolve(backend, "crates/wenlan-cli"), { recursive: true });
    mkdirSync(resolve(backend, "target/debug"), { recursive: true });
    mkdirSync(stateDir, { recursive: true });
    mkdirSync(fakeBin, { recursive: true });
    writeFileSync(resolve(backend, "Cargo.toml"), "[workspace]\n");
    symlinkSync("/bin/sleep", server);
    writeFileSync(resolve(fakeBin, "pnpm"), "#!/usr/bin/env bash\nexit 0\n");
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
        },
      });

      expect(result.status, `${result.stdout}\n${result.stderr}`).toBe(0);
      expect(() => process.kill(daemon.pid!, 0)).not.toThrow();
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
