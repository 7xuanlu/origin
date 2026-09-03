// @vitest-environment node
// Nothing here touches the DOM: every case spawns bash or reads the script.
// jsdom costs seconds of setup per run for a document nobody opens.
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { resolve, win32 } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";
import { branchesOnUnmeasured, probeCallSites } from "./lib/probe-call-sites";

// The same reason as scripts/host-process.test.ts: these cases spawn node and
// Git Bash, and dev-runtime.sh's file-scope production-root guard canonicalizes
// eleven roots through `node` before any command reaches a body. At the 5s
// default a green run is a statement about machine load. Cases that start a
// daemon or wait on a signal raise it further at their own site.
vi.setConfig({ testTimeout: 30_000 });

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
    // `canonicalize_paths`, plural: the resolution is one `node` for a LIST of
    // paths and `canonicalize_path` is a one-argument wrapper on it.
    const start = script.indexOf("canonicalize_paths() {");
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

    expect(script).toContain("probe_listener_port");
    // Ownership is claimed only on a MEASURED match: the liveness probe says
    // `alive`, the listener probe says `found`, and the found pid is ours.
    expect(script).toContain('[[ "$PROCESS_ALIVE_STATE" == alive && "$LISTENER_PROBE_STATE" == found ]]');
    expect(script).toContain('[[ "$LISTENER_PROBE_PID" == "$pid" ]]');
    expect(script).toContain("has_owned_command_identity");
    expect(script).toContain("acquire_runtime_lock");
    expect(script).toContain("wenlan-server.data-dir");
  });

  // The tri-state invariant, asserted on the text of the script: an unmeasured
  // probe must reach a branch of its own. If `unmeasured` ever falls through to
  // the negative branch, a daemon that could not be measured reads as absent —
  // the port is then treated as free, and the ownership record is deleted.
  it("branches on 'could not measure' everywhere it probes a port or a process", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const code = script
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");

    // Every probe call site is followed by a branch that names `unmeasured` —
    // asserted ONE CALL SITE AT A TIME.
    //
    // Round 13h: what stood here was `probeCalls.length >= 6` and then
    // `code.match(/\bunmeasured\b/g).length >= probeCalls.length`, and both
    // halves are counts over the WHOLE FILE. Nine call sites that each spell
    // `unmeasured` twice cover a tenth that has stopped handling it, and the
    // file-global total never moves. That is the same collapse the script
    // itself is about, one level up: a caller that cannot be measured
    // individually is indistinguishable from a caller that is correct.
    const sites = probeCallSites(code);
    expect(sites.length, "no probe call sites found in dev-runtime.sh").toBeGreaterThanOrEqual(
      10,
    );
    for (const site of sites) {
      const where = `dev-runtime.sh (comments stripped) line ${site.line}: ${site.probe}`;
      // A probe whose answer nothing reads is the same defect one step earlier.
      expect(site.read, `${where}: nothing branches on ${site.state}`).toBe(true);
      expect(
        branchesOnUnmeasured(site),
        `${where}: no branch for the third state`,
      ).toBe(true);
    }

    // The start gate refuses on an unmeasured port rather than starting.
    expect(code).toContain("refusing to start: an unmeasured port is not a free port");
    // Teardown keeps the ownership record when liveness could not be measured.
    expect(code).toContain("keeping the ownership record so the process stays attributable");
    // The old fail-open shapes are gone: no bare local copies of the probes,
    // and no `|| true` swallowing a terminate status.
    expect(code).not.toMatch(/^listener_pid_for_port\(\)/m);
    expect(code).not.toMatch(/^process_is_alive\(\)/m);
    expect(code).not.toMatch(/^process_image_path\(\)/m);
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

// Round 13h. Everything above this line is prose: every top-level guard exits 2
// and everything `start_runtime` returns exits 1, so the ONLY thing separating
// "this runtime refused to touch the production port" from "cargo did not
// build" was the wording of a message. A consumer that classifies by
// string-matching those messages reclassifies a SAFETY REFUSAL the day one of
// them is reworded, combined, or translated — silently, and in the direction
// that hides it.
//
// So there is a machine-readable line, it is additive, and it is last.
describe("dev runtime outcome marker", () => {
  const KINDS = [
    "ok",
    "safety-refusal",
    "build-failure",
    // Split off `build-failure` in round 4: cargo has already succeeded by the
    // staging call, so a caller that reads `build-failure` retries a build that
    // was never the problem and never sees the held stage that is.
    "staging-failure",
    "health-failure",
    "port-conflict",
    // A signal ended the run. Not `unknown` — `unknown` is a refusal a consumer
    // must not act through, and this is the one failing kind it may simply
    // re-run. Before it existed, `$?` inside a signal trap was the PREVIOUS
    // command's status, so an interrupted run could exit ZERO and print `ok`.
    "interrupted",
    "unknown",
  ];

  const script = () => readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");

  it("documents the contract, and names exactly the kinds it can print", () => {
    const header = script().slice(0, script().indexOf("set -euo pipefail"));
    expect(header).toContain("DEV_RUNTIME_RESULT: <kind>");
    for (const kind of KINDS) {
      expect(header, `the header does not document ${kind}`).toMatch(
        new RegExp(String.raw`^#\s+${kind}\s`, "m"),
      );
    }
    // `unknown` is a REFUSAL, not a pass — the same rule as every probe in this
    // file, carried out to the process boundary.
    expect(header).toContain("`unknown` is a REFUSAL");
  });

  it("assigns no kind the contract does not name", () => {
    // A typo here is invisible: the marker still prints, the consumer still
    // reads a word, and the word means nothing. `health_kind` is the same
    // variable one indirection back, so it is held to the same list.
    const assignments = [...script().matchAll(/^\s*(?:RESULT_KIND|health_kind)=([^\s;"]+)/gm)].map(
      (m) => m[1],
    );
    expect(assignments.length).toBeGreaterThan(20);
    for (const kind of assignments) {
      expect(KINDS, `RESULT_KIND=${kind} is not a kind the contract names`).toContain(kind);
    }
    // The one indirect assignment, which is `health_kind` and nothing else.
    expect(script()).toContain('RESULT_KIND="$health_kind"');
  });

  it("prints the marker exactly once, on stderr, from a trap on every way out", () => {
    const code = script()
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");
    // One printf, so two consumers reading opposite ends of the stream cannot
    // disagree about the same run.
    const prints = code.match(/printf 'DEV_RUNTIME_RESULT: %s\\n'/g) ?? [];
    expect(prints, "the marker is printed in more than one place").toHaveLength(1);
    expect(code).toMatch(/printf 'DEV_RUNTIME_RESULT: %s\\n' "\$1" >&2/);
    // From a trap, because the ways out are not all `return`s: a file-scope
    // guard `exit 2`s before `start` reaches its body, and a `set -e` abort
    // leaves through a command nothing captured.
    //
    // FOUR traps and not one. `trap on_runtime_exit EXIT HUP INT TERM` cannot
    // tell the handler which signal arrived, and `$?` inside a signal trap is
    // the PREVIOUS command's status rather than `128+n` — so the one-trap form
    // reported whatever ran last, and an interrupt after a successful command
    // exited 0. Each signal names itself here.
    expect(code).toMatch(/^trap on_runtime_exit EXIT$/m);
    for (const [signal, status] of [
      ["HUP", 129],
      ["INT", 130],
      ["TERM", 143],
    ] as const) {
      expect(code, `${signal} has no trap of its own`).toMatch(
        new RegExp(String.raw`^trap 'on_runtime_exit ${signal}' ${signal}$`, "m"),
      );
      // The status is derived from the signal, not read out of `$?`.
      expect(code, `${signal} does not exit 128+n`).toMatch(
        new RegExp(String.raw`${signal}\)\s*status=${status};\s*RESULT_KIND=interrupted`),
      );
    }
    // And the handler runs its body at most once. `exit` below re-enters it
    // through the EXIT trap, which is how the lock release used to happen a
    // second time — after the marker, putting its `rm` diagnostic below the
    // line this contract promises is last.
    expect(code, "the outcome handler is not idempotent").toMatch(
      /if \(\( RUNTIME_EXIT_RAN == 1 \)\); then\n\s*exit "\$status"\n\s*fi\n\s*RUNTIME_EXIT_RAN=1/,
    );
    // Installed before the guards can fire, and the guards are many.
    expect(code.indexOf("trap on_runtime_exit")).toBeLessThan(code.indexOf("refuse_unsafe()"));
    // And before the script locates itself or sources anything. Each of those
    // is a way out under `set -e` — an unreadable library, a `cd` into a
    // directory that moved — and a way out installed before the trap is a way
    // out that prints no marker at all.
    const trapAt = code.indexOf("trap on_runtime_exit");
    for (const earlier of [
      'SCRIPT_DIR="$(cd',
      'REPO_ROOT="$(cd',
      '. "$SCRIPT_DIR/lib/host-process.sh"',
    ]) {
      const at = code.indexOf(earlier);
      expect(at, `${earlier} is not in the script any more`).toBeGreaterThan(-1);
      expect(at, `${earlier} runs before the trap that would report its failure`)
        .toBeGreaterThan(trapAt);
    }
    // The exit status is the script's own: the trap must not turn a refusal
    // into a success by returning its own.
    expect(code).toMatch(/on_runtime_exit\(\) \{\s*\n\s*local status=\$\?/);
    expect(code).toContain('exit "$status"');
    // ROUND 4: this used to assert `release_runtime_lock || true`, and called it
    // the one `|| true` the file allows. The reasoning was right about the trap
    // — errexit is NOT suspended inside a handler, so a bare non-zero release
    // would abort it before `emit_result` ran, and a run with no verdict is the
    // one outcome this contract cannot spell — and wrong about where the status
    // went. `|| true` DISCARDS it. So: an `if !`, which suspends errexit for the
    // same reason and keeps the answer, still reaching `emit_result`.
    expect(code, "a discarded release status is back").not.toMatch(
      /release_runtime_lock \|\| true/,
    );
    expect(code).toMatch(
      /if ! release_runtime_lock; then\n(?:.*\n)*?\s*fi\n\s*fi\n\s*emit_result "\$RESULT_KIND"/,
    );
    // And it must not report success over a lock that is still standing.
    expect(code).toMatch(/RESULT_KIND=unknown/);
  });

  const runtime = (args: string[], env: Record<string, string> = {}) => {
    const stateDir = mkdtempSync(resolve(tmpdir(), "wenlan-dev-marker-"));
    tempRoots.push(stateDir);
    const argv = ["scripts/dev-runtime.sh", ...args];
    const options = {
      cwd: root,
      encoding: "utf8" as const,
      env: { ...process.env, WENLAN_DEV_STATE_DIR: stateDir, ...env },
    };
    return process.platform === "win32"
      ? spawnSync(process.execPath, ["scripts/run-bash.mjs", ...argv], options)
      : spawnSync("bash", argv, options);
  };
  const marker = (stderr: string) => {
    const lines = stderr.replace(/\r/g, "").split("\n").filter((line) => line !== "");
    return lines[lines.length - 1];
  };

  it.each([
    // A command that did what it was asked to do.
    [["print-config"], {}, 0, "ok"],
    // Nothing to stop is still a successful stop.
    [["stop"], {}, 0, "ok"],
    // The production daemon port, refused at file scope — before `start` has a
    // body to run, which is why the marker cannot live at the end of one.
    [["start"], { WENLAN_DEV_PORT: "7878" }, 2, "safety-refusal"],
    // Not a kind this can name, so it says so rather than guessing one.
    [["frobnicate"], {}, 2, "unknown"],
    // Round 4: the suite covered `ok`, `safety-refusal` and a usage `unknown`,
    // and nothing else — so four of the seven kinds were asserted only as
    // strings in the header. This one is reachable without a compiler, because
    // `resolve-backend-dir.sh` refuses a WENLAN_BACKEND_DIR that is not a
    // backend checkout, and that refusal is the first thing `start_runtime`
    // classifies.
    [["start"], { WENLAN_BACKEND_DIR: "__not_a_backend__" }, 1, "build-failure"],
  ])(
    "%s exits with the marker as its last stderr line",
    (args, env, status, kind) => {
      const result = runtime(args as string[], env as Record<string, string>);
      expect(result.status, result.stderr).toBe(status);
      expect(marker(result.stderr)).toBe(`DEV_RUNTIME_RESULT: ${kind}`);
    },
    // Generous, and it is not the script being slow at what it does: the
    // file-scope production-root guard canonicalizes eleven roots through
    // `node`, which is ~23s of process creation on Windows before any of these
    // commands reaches a body.
    90_000,
  );

  it("leaves the human prose above the marker untouched", () => {
    // Additive. A consumer that ignores the last line sees exactly what it saw
    // before, which is the only reason this could be added to a script other
    // lanes already parse.
    const refusal = runtime(["start"], { WENLAN_DEV_PORT: "7878" });
    expect(refusal.stderr).toContain("error: refusing production daemon port 7878");
    const empty = runtime(["stop"]);
    expect(empty.stdout).toContain("No worktree-owned Wenlan dev daemon is recorded.");
  }, 90_000);

  // A lock directory with no owner file in it. That is the state a run killed
  // between `mkdir "$LOCK_DIR"` and the very next line leaves — and it is ALSO
  // the state a run that is very much alive is in for the microseconds between
  // those two lines. The old code recovered it on sight; this waits for an
  // owner and then refuses, which is the only honest answer for a lock nobody
  // can attribute. It doubles as a second reachable `unknown`.
  it("refuses a lock whose owner cannot be established, and says so", () => {
    const stateDir = mkdtempSync(resolve(tmpdir(), "wenlan-dev-lockmarker-"));
    tempRoots.push(stateDir);
    mkdirSync(resolve(stateDir, "runtime.lock"));
    const argv = ["scripts/dev-runtime.sh", "stop"];
    const options = {
      cwd: root,
      encoding: "utf8" as const,
      env: { ...process.env, WENLAN_DEV_STATE_DIR: stateDir },
    };
    const result =
      process.platform === "win32"
        ? spawnSync(process.execPath, ["scripts/run-bash.mjs", ...argv], options)
        : spawnSync("bash", argv, options);
    expect(result.stderr).toContain("error: the dev runtime lock names no owner");
    expect(
      result.stderr,
      "an unattributable lock was recovered instead of refused",
    ).toContain("is not an absent owner");
    expect(result.status, result.stderr).toBe(1);
    expect(marker(result.stderr)).toBe("DEV_RUNTIME_RESULT: unknown");
    // And it did not take the lock: the owner file it would have written is
    // not there.
    expect(existsSync(resolve(stateDir, "runtime.lock", "pid"))).toBe(false);
  }, 90_000);

  // THE SIGNAL PATH, which had no case at all and is the one way out that could
  // report success for a run that did not finish. `$?` inside a signal trap is
  // the status of the PREVIOUS command, so a TERM delivered after anything that
  // succeeded exited 0 with whatever kind RESULT_KIND had reached.
  //
  // The signal is sent from INSIDE bash rather than with `child.kill`: on
  // Windows a Node `SIGTERM` is TerminateProcess, which delivers no signal at
  // all and would make this assert on a process that never ran its trap. The
  // bounded wait `acquire_runtime_lock` does for a missing owner file is the
  // window it is signalled in — five seconds, deterministic, and it needs no
  // compiler, no port and no daemon.
  // Round 4, S1 and S3. Two statuses this file dropped, in the two functions
  // whose whole job is to keep one run's state away from another's — and both
  // are reachable only through a failing `rm`/`>` on the developer's own state
  // directory, which is not a condition a test may manufacture on this host
  // without holding a real file open or rewriting real ACLs. So they are held
  // to the SOURCE SHAPE, which is a weaker witness than a driven case and is
  // said to be one. Each assertion below fails if its check is deleted.
  it("reads the status of every write that makes a lock or a record real", () => {
    // Comments stripped: this file DESCRIBES the shapes it no longer uses, at
    // length and on purpose, so a search of the raw text finds the defect in
    // the prose that records it.
    const code = script()
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");

    // S1. `acquire_runtime_lock` is called on the LEFT OF `||` at every one of
    // its call sites, so `set -e` is suspended through its entire body: a
    // failed owner-file write did not abort, and the function went on to
    // return the status of its last command, which was zero. The caller then
    // believed it held a lock whose owner nothing had recorded — and the next
    // command reads an owner-less lock, which is precisely the state S4 above
    // now refuses. One dropped status manufactures the other finding's input.
    // And it is a CREATE-OR-FAIL write: `set -C` is what stops a run whose
    // directory was broken and retaken from stamping its name over the new
    // holder's record and returning "acquired" to both of them.
    expect(code, "the lock owner write is not checked").toMatch(
      /if ! \( set -C; printf '%s\\n' "\$token" >"\$LOCK_OWNER_FILE" \) 2>\/dev\/null; then/,
    );
    // And the flag the EXIT trap releases on is set only after that write has
    // been shown to have happened, not before it is attempted.
    const acquire = code.slice(
      code.indexOf("acquire_runtime_lock() {"),
      code.indexOf("\n}\n", code.indexOf("acquire_runtime_lock() {")),
    );
    expect(acquire.indexOf("RUNTIME_LOCK_HELD=1"), "the lock is claimed before it is written")
      .toBeGreaterThan(acquire.indexOf('>"$LOCK_OWNER_FILE"'));

    // The audit's other write. The four ownership-record files are what every
    // later `stop`, reap and port check reads; a record that was not written
    // leaves a daemon running that nothing in this worktree can find again.
    // Four bare `printf … >FILE` lines stood here, inside `start_runtime`,
    // which the dispatch also calls on the left of nothing at all — but the
    // status still went nowhere.
    for (const file of ["$PID_FILE", "$SERVER_PATH_FILE", "$PORT_FILE", "$DATA_DIR_FILE"]) {
      expect(code, `the write of ${file} is not checked`).toContain(
        `>"${file}" || record_rc=1`,
      );
    }
    expect(code).toContain("if (( record_rc != 0 )); then");

    // S3. `clear_owned_state` returns 0/1 and is the only thing that deletes an
    // ownership record. A bare call reads as a statement; every one of them is
    // a branch.
    const bare = code
      .split("\n")
      .filter(
        (line) =>
          /^\s*clear_owned_state\b/.test(line) &&
          !/^\s*#/.test(line) &&
          // its own definition, which is the one line that is not a call
          !/^clear_owned_state\(\) \{$/.test(line),
      );
    expect(bare, `clear_owned_state is called without reading its status:\n${bare.join("\n")}`)
      .toHaveLength(0);
    expect(code.match(/if ! clear_owned_state; then/g) ?? []).not.toHaveLength(0);

    // S3, the outer half. The health path called `stop_runtime || true`, which
    // discards the status of the cleanup AND lets the line below overwrite the
    // kind that cleanup just set — so a daemon that could not be stopped was
    // reported as a health failure, and the thing still running on the port
    // was never mentioned.
    expect(code, "the health path swallows its cleanup status").not.toContain(
      "stop_runtime || true",
    );
    expect(code).toContain("stop_runtime || cleanup_rc=$?");
    const health = code.slice(code.indexOf("stop_runtime || cleanup_rc=$?"));
    expect(
      health.indexOf("if (( cleanup_rc != 0 )); then"),
      "the health kind is published without looking at the cleanup status",
    ).toBeLessThan(health.indexOf('RESULT_KIND="$health_kind"'));
  });

  // Round 4, the audit. `print_config` is what `dev-all.sh` EVALS, so a line
  // that did not arrive is not a missing line — it is the production default
  // for that variable, silently substituted into a run that believes it is
  // isolated. Seven bare `printf`s stood here with nothing reading their
  // status, in a function whose caller is `print_config || …`, so `set -e`
  // was suspended and even the abort did not happen.
  //
  // Driven with stdout CLOSED, which is a real way for a consumer's pipe to
  // end and the only one a test can arrange without a full disk.
  // ROUND 4, the limitation this case has to state rather than imply. Stdout is
  // CLOSED before the first `printf`, so EVERY write fails, and what is proved
  // is "no line arrived and the status said so". `print_config` is NOT
  // TRANSACTIONAL: the seven `printf`s go straight to stdout and `rc` is only
  // consulted after the last one, so a stream that dies at line four leaves
  // three lines already written and then returns 1. That case is not driven
  // here — arranging a pipe that accepts three writes and refuses the fourth is
  // beyond what this harness can set up — and it is the case that matters most,
  // because a consumer which evals what it received and ignores the status gets
  // production defaults for the four lines that never came. The remedy for it
  // is entirely the non-zero status plus the message below; nothing withholds
  // the partial output. `dev-all.sh` is the consumer, and it is the one that
  // has to honour it.
  it("refuses a configuration it could not write in full", () => {
    const stateDir = mkdtempSync(resolve(tmpdir(), "wenlan-dev-config-"));
    tempRoots.push(stateDir);
    const driver = resolve(stateDir, "closed.sh");
    writeFileSync(
      driver,
      [
        "#!/usr/bin/env bash",
        "set -u",
        'WENLAN_DEV_STATE_DIR="$1" bash "$2/scripts/dev-runtime.sh" print-config >&- 2>"$1/err.txt"',
        'printf "exit=%s\\n" "$?"',
      ].join("\n"),
      { mode: 0o755 },
    );
    const argv = [driver, stateDir, root];
    const options = { cwd: root, encoding: "utf8" as const, env: { ...process.env } };
    const result =
      process.platform === "win32"
        ? spawnSync(process.execPath, ["scripts/run-bash.mjs", ...argv], options)
        : spawnSync("bash", argv, options);
    expect(result.stdout.trim(), result.stderr).toBe("exit=1");
    const stderr = readFileSync(resolve(stateDir, "err.txt"), "utf8");
    expect(stderr).toContain("the dev runtime configuration could not be written in full");
    // `unknown`, not `ok`: a config nobody could write is a measurement that
    // did not happen, and the consumer must refuse rather than eval what came.
    expect(marker(stderr), stderr).toBe("DEV_RUNTIME_RESULT: unknown");
  }, 90_000);

  // ROUND 4, the limitation this case has to state rather than imply. ONE of
  // the three signals is driven. `kill -TERM` is delivered here and the exit
  // status and marker are measured; HUP and INT are asserted only against the
  // SOURCE, in the case above, which proves the traps exist and the arms map to
  // 129 and 130 — not that a delivered SIGHUP or SIGINT reaches them. On this
  // platform they are also the two that are hardest to deliver honestly: MSYS
  // synthesises signals for a native child, and Ctrl-C in particular arrives
  // through a console control handler rather than as a POSIX signal, so a
  // passing case here would be a statement about the harness. What is proved is
  // that the ONE-TRAP form's defect — `$?` inside a signal handler being the
  // previous command's status — is fixed for the signal that was driven, and
  // that the other two are wired the same way in the text.
  it("reports an interrupted run as interrupted, and exits 128+n", () => {
    const stateDir = mkdtempSync(resolve(tmpdir(), "wenlan-dev-signal-"));
    tempRoots.push(stateDir);
    mkdirSync(resolve(stateDir, "runtime.lock"));
    const driver = resolve(stateDir, "signal.sh");
    writeFileSync(
      driver,
      [
        "#!/usr/bin/env bash",
        "set -u",
        'WENLAN_DEV_STATE_DIR="$1" bash "$2/scripts/dev-runtime.sh" stop >/dev/null 2>"$1/err.txt" &',
        "bg=$!",
        "sleep 1.5",
        'kill -TERM "$bg" 2>/dev/null',
        'wait "$bg"',
        'printf "exit=%s\\n" "$?"',
      ].join("\n"),
      { mode: 0o755 },
    );
    const argv = [driver, stateDir, root];
    const options = { cwd: root, encoding: "utf8" as const, env: { ...process.env } };
    const result =
      process.platform === "win32"
        ? spawnSync(process.execPath, ["scripts/run-bash.mjs", ...argv], options)
        : spawnSync("bash", argv, options);
    // 128 + SIGTERM(15). Not `$?`, which at the moment of the signal is the
    // status of the last `sleep` — zero.
    expect(result.stdout.trim(), result.stderr).toBe("exit=143");
    const stderr = readFileSync(resolve(stateDir, "err.txt"), "utf8");
    expect(marker(stderr), stderr).toBe("DEV_RUNTIME_RESULT: interrupted");
    // Exactly one marker: the handler runs twice on a signal (its own trap,
    // then EXIT), and only the idempotence guard keeps the second run from
    // releasing the lock again AFTER this line.
    expect(stderr.match(/DEV_RUNTIME_RESULT:/g) ?? []).toHaveLength(1);
  }, 90_000);
});

// Round 13g. Staging was `cp -u`, which compares TIMESTAMPS, and the claim the
// dev runtime makes about what it staged is about BYTES. They come apart in
// ordinary ways — a staged executable restored from a backup, a clock that went
// backwards, an artifact copied rather than built — and when they do, cargo
// succeeds, `cp -u` keeps the old file, `/api/health` is answered by a daemon
// built from other source, and the run reports current-source provenance.
// Nothing downstream can notice: the recorded path, the pid and the port are
// all correct. The DLL copies had the same defect with a second one on top,
// `|| true`, which is os error 32's own failure mode — a library held open by a
// process running out of the stage directory, left at its old version beside a
// new daemon.
//
// `dev-runtime.sh` dispatches on `$1` at top level and cannot be sourced, so
// the staging functions are extracted by brace matching and driven directly —
// the same idiom `scripts/negative-controls/dev-runtime-*-controls.sh` use, and
// the only alternative to a `cargo build` per case.
describe("dev runtime daemon staging", () => {
  function extractFunction(script: string, name: string): string {
    const lines = script.split("\n");
    const start = lines.indexOf(`${name}() {`);
    expect(start, `${name} is not a function in dev-runtime.sh`).toBeGreaterThan(-1);
    let depth = 0;
    for (let i = start; i < lines.length; i += 1) {
      depth += (lines[i].match(/\{/g) ?? []).length;
      depth -= (lines[i].match(/\}/g) ?? []).length;
      if (depth === 0) return lines.slice(start, i + 1).join("\n");
    }
    throw new Error(`${name} is never closed in dev-runtime.sh`);
  }

  // Paths travel in the ENVIRONMENT and are spelled with forward slashes: MSYS
  // bash strips the backslashes out of a `C:\Users\...` argument handed to it
  // from node, which would leave every fixture below pointing at a path that
  // never existed and every case green for the wrong reason.
  const posix = (path: string) => path.replace(/\\/g, "/");

  const DRIVER_PREAMBLE = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    "# Git for Windows puts /usr/bin at the front of PATH before this script",
    "# runs, so a shim directory handed in from outside loses to /usr/bin/cp.",
    'if [ -n "${WENLAN_TEST_SHIM_DIR:-}" ]; then',
    '  shim_dir="$WENLAN_TEST_SHIM_DIR"',
    '  if command -v cygpath >/dev/null 2>&1; then shim_dir="$(cygpath -u "$shim_dir")"; fi',
    '  PATH="$shim_dir:$PATH"',
    "  export PATH",
    "fi",
    'REPO_ROOT="$WENLAN_TEST_REPO_ROOT"',
    'DAEMON_STAGE_DIR="$WENLAN_TEST_STAGE_DIR"',
  ].join("\n");

  type Fixture = {
    root: string;
    repo: string;
    stage: string;
    build: string;
    source: string;
  };

  function makeFixture(): Fixture {
    const dir = mkdtempSync(resolve(tmpdir(), "wenlan-stage-"));
    tempRoots.push(dir);
    const repo = resolve(dir, "repo");
    const stage = resolve(dir, "stage");
    const build = resolve(dir, "build");
    // The app/binaries staging area exists and is empty on a checkout that has
    // not built the app, which is the ordinary case and must not be an error.
    mkdirSync(resolve(repo, "app/binaries"), { recursive: true });
    mkdirSync(stage, { recursive: true });
    mkdirSync(build, { recursive: true });
    return { root: dir, repo, stage, build, source: resolve(build, "wenlan-server.exe") };
  }

  // A staged file with a NEWER mtime than the source, which is the whole point:
  // under `cp -u` this file wins, and it is the wrong file.
  function stageStale(fixture: Fixture, name: string, contents: string) {
    const path = resolve(fixture.stage, name);
    writeFileSync(path, contents);
    const later = new Date(Date.now() + 60_000);
    utimesSync(path, later, later);
    return path;
  }

  function runStaging(
    fixture: Fixture,
    shims: Record<string, string> = {},
  ): { status: number | null; stdout: string; stderr: string } {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const driver = [
      DRIVER_PREAMBLE,
      extractFunction(script, "file_sha256"),
      extractFunction(script, "stage_file_by_identity"),
      // The directory measurement `stage_windows_daemon` now stands on. It is
      // the same tri-state listing `read_owned_pid` uses, which is the point:
      // a glob has two answers where the question has three.
      extractFunction(script, "listing_has_name"),
      extractFunction(script, "list_dir_tristate"),
      extractFunction(script, "stage_windows_daemon"),
      "rc=0",
      'stage_windows_daemon "$WENLAN_TEST_SOURCE" || rc=$?',
      `printf 'rc=%s\\n' "$rc"`,
    ].join("\n");
    const driverPath = resolve(fixture.root, "driver.sh");
    writeFileSync(driverPath, driver, { mode: 0o755 });

    const env: Record<string, string> = {};
    for (const [key, value] of Object.entries(process.env)) {
      if (value !== undefined) env[key] = value;
    }
    env.WENLAN_TEST_REPO_ROOT = posix(fixture.repo);
    env.WENLAN_TEST_STAGE_DIR = posix(fixture.stage);
    env.WENLAN_TEST_SOURCE = posix(fixture.source);
    const names = Object.keys(shims);
    if (names.length > 0) {
      const shimDir = resolve(fixture.root, "shim");
      mkdirSync(shimDir, { recursive: true });
      for (const name of names) {
        const path = resolve(shimDir, name);
        writeFileSync(path, `#!/usr/bin/env bash\n${shims[name]}\n`, { mode: 0o755 });
        chmodSync(path, 0o755);
      }
      env.WENLAN_TEST_SHIM_DIR = shimDir;
    }

    const result =
      process.platform === "win32"
        ? spawnSync(process.execPath, ["scripts/run-bash.mjs", posix(driverPath)], {
            cwd: root,
            encoding: "utf8",
            env,
          })
        : spawnSync("bash", [driverPath], { cwd: root, encoding: "utf8", env });
    return { status: result.status, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
  }

  // A `cp` that hands the real one everything it is not asked to sabotage, so
  // each case below breaks exactly one copy and the rest of the staging runs.
  const REAL_CP = [
    "for real in /usr/bin/cp /bin/cp; do",
    '  if [ -x "$real" ]; then exec "$real" "$@"; fi',
    "done",
    'echo "no real cp on this host" >&2',
    "exit 127",
  ].join("\n");
  const LAST_ARG = ['last=""', 'for a in "$@"; do last="$a"; done'].join("\n");
  // Same idea for `ls`: the directory listing is a MEASUREMENT now, so a case
  // has to be able to break exactly one of them.
  const REAL_LS = [
    "for real in /usr/bin/ls /bin/ls; do",
    '  if [ -x "$real" ]; then exec "$real" "$@"; fi',
    "done",
    'echo "no real ls on this host" >&2',
    "exit 127",
  ].join("\n");

  it("replaces a staged daemon whose mtime is newer and whose bytes are older", () => {
    // THE case. `cp -u` keeps the file on the left of this comparison; identity
    // keeps the one on the right, which is the one that was just built.
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    const staged = stageStale(fixture, "wenlan-server.exe", "a daemon from some other build\n");

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=0");
    expect(readFileSync(staged, "utf8")).toBe("the daemon this run built\n");
  });

  it("replaces a stale runtime library beside the daemon", () => {
    // The DLLs carried the same `cp -u`, and a loader resolves them from the
    // executable's own directory: an old library beside a new daemon is a
    // mismatch nothing downstream looks for.
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    writeFileSync(resolve(fixture.build, "onnxruntime.dll"), "the library it was built against\n");
    const staged = stageStale(fixture, "onnxruntime.dll", "the library from some other build\n");

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=0");
    expect(readFileSync(staged, "utf8")).toBe("the library it was built against\n");
  });

  it("does not copy over a staged daemon that is already the built bytes", () => {
    // The identity check has to decide the copy, not merely follow it: a `cp`
    // that runs every time would pass every other case in this block. The
    // staged file is also the NEWER one here, so `cp -u` skips it too — this
    // case is about the digest, and it must stay green under the control that
    // reverts to timestamps.
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    stageStale(fixture, "wenlan-server.exe", "the daemon this run built\n");

    const result = runStaging(fixture, {
      cp: [LAST_ARG, `printf '%s\\n' "$last" >>"$(dirname "$0")/cp.log"`, REAL_CP].join("\n"),
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=0");
    expect(existsSync(resolve(fixture.root, "shim/cp.log"))).toBe(false);
  });

  it("refuses when the copy reported success and the staged bytes are different", () => {
    // The other half of "stage by identity": a copy that EXITS 0 and leaves
    // different bytes — a partial write, a concurrent writer, a filesystem that
    // lied — is invisible unless what is on disk is re-read afterwards. cp's
    // own status cannot see it, and neither can the pre-copy comparison.
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    stageStale(fixture, "wenlan-server.exe", "a daemon from some other build\n");

    const result = runStaging(fixture, {
      cp: [LAST_ARG, `printf '%s\\n' 'not what was asked for' >"$last"`, "exit 0"].join("\n"),
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain("the staged dev daemon is not the one that was just built");
  });

  it("fails loudly when a runtime library cannot be copied", () => {
    // os error 32: the DLL is held open by a process running out of the stage
    // directory. This is the failure `|| true` was swallowing, and the one that
    // actually happens. The daemon copy is left alone so the error names the
    // library and not the executable.
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    writeFileSync(resolve(fixture.build, "onnxruntime.dll"), "the library it was built against\n");
    stageStale(fixture, "onnxruntime.dll", "the library from some other build\n");

    const result = runStaging(fixture, {
      cp: [
        LAST_ARG,
        "case $last in",
        `  *.dll) echo "cp: cannot create regular file '$last': Device or resource busy" >&2; exit 1 ;;`,
        "esac",
        REAL_CP,
      ].join("\n"),
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain("could not stage runtime library onnxruntime.dll");
  });

  it("refuses when the built daemon cannot be hashed at all", () => {
    // An unmeasured source digest can never conclude "the staged copy matches",
    // so it must not be allowed to conclude anything: staging by timestamp
    // instead is how the stale bytes got to answer /api/health in the first
    // place.
    const fixture = makeFixture();
    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain("could not hash the built dev daemon");
  });

  // --- the library directory is measured, not globbed ------------------------
  //
  // `for dll in "$dir"/*.dll; do [[ -f "$dll" ]] || continue` has TWO answers
  // and the question has three. A directory that cannot be listed leaves the
  // pattern unexpanded, `-f` rejects the literal `*.dll`, the body never runs,
  // and staging returns SUCCESS having copied no libraries — identical, from
  // outside, to the fresh checkout where there genuinely are none. The daemon
  // then starts without onnxruntime.dll beside it and fails at the first embed,
  // a long way from here.

  it("refuses when the runtime library directory cannot be listed", () => {
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");

    const result = runStaging(fixture, {
      // Not a missing directory — an unreadable one. `list_dir_tristate` climbs
      // to the parent, finds `binaries` in ITS listing, and so answers 2: the
      // directory is there and could not be read.
      ls: [
        "case \"$*\" in",
        '  *app/binaries*) echo "ls: cannot open directory: Permission denied" >&2; exit 2 ;;',
        "esac",
        REAL_LS,
      ].join("\n"),
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain("could not list the runtime libraries in");
    expect(result.stderr).toContain("an unreadable directory is not an empty one");
    // And nothing was staged beside the daemon, which is the outcome the old
    // shape reported as success.
    expect(existsSync(resolve(fixture.stage, "onnxruntime.dll"))).toBe(false);
  });

  it("refuses when a prepared app/binaries has lost the libraries it must carry", () => {
    // prepare-sidecars.sh installs the sidecars under triple-qualified names and
    // refuses to finish on a Windows triple unless onnxruntime.dll and
    // vulkan-1.dll are beside them. So a binaries directory holding a prepared
    // sidecar and NOT holding those is a broken layout, and a zero-library stage
    // out of it is a failure — which is exactly what the glob could not say.
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    writeFileSync(
      resolve(fixture.repo, "app/binaries/wenlan-server-x86_64-pc-windows-msvc.exe"),
      "a prepared sidecar\n",
    );

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain(
      "the dev daemon was staged without onnxruntime.dll vulkan-1.dll",
    );
  });

  it("stages every expected runtime library out of a prepared app/binaries", () => {
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    writeFileSync(
      resolve(fixture.repo, "app/binaries/wenlan-server-x86_64-pc-windows-msvc.exe"),
      "a prepared sidecar\n",
    );
    for (const name of ["onnxruntime.dll", "vulkan-1.dll"]) {
      writeFileSync(resolve(fixture.repo, "app/binaries", name), `${name} as prepared\n`);
    }

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=0");
    for (const name of ["onnxruntime.dll", "vulkan-1.dll"]) {
      expect(readFileSync(resolve(fixture.stage, name), "utf8")).toBe(`${name} as prepared\n`);
    }
    // The positive count was reached, so the zero-library note must NOT appear.
    expect(result.stderr).not.toContain("no runtime libraries beside it");
  });

  it("names a zero-library stage instead of passing it off as an ordinary one", () => {
    // The other side of the same coin. An empty app/binaries on a checkout that
    // has not run prepare-sidecars.sh is a real and ordinary outcome, so it
    // stays rc=0 — but it is SAID, because it used to be indistinguishable from
    // the directory nobody could read.
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=0");
    expect(result.stderr).toContain("staged the dev daemon with no runtime libraries beside it");
  });

  // Round 3, S8a. Every case above asks "did this file get there". The daemon
  // does not load a series of files — it loads whatever is in its own
  // directory, and nothing had ever looked at that directory as a whole. A
  // leftover library from an earlier layout sat in the stage, no copy touched
  // it, and this function returned SUCCESS while printing "no runtime
  // libraries beside the daemon". The two claims were about different
  // directories, and the stale one is the one that gets loaded.
  it("refuses a stage holding a library that neither source directory has", () => {
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    // Neither app/binaries nor the build directory has this. It is in the
    // stage because some earlier run put it there.
    writeFileSync(resolve(fixture.stage, "onnxruntime.dll"), "an onnxruntime nobody built\n");

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain("which this run did not put there");
    expect(result.stderr).toContain("onnxruntime.dll");
    // The old shape said this instead, about a directory it had not read.
    expect(result.stderr).not.toContain("staged the dev daemon with no runtime libraries");
  });

  // Round 3, S8b. TWO SOURCES, ONE NAME. Both directories are walked and each
  // copy is verified on its own, so a different onnxruntime.dll in the second
  // was written over the verified first and every per-file check stayed green:
  // the file the daemon loads is whichever the loop reached last.
  it("refuses when the two source directories disagree about a library name", () => {
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    writeFileSync(resolve(fixture.repo, "app/binaries/onnxruntime.dll"), "onnxruntime A\n");
    // fixture.source lives in fixture.build, so this is the second directory
    // the loop walks.
    writeFileSync(resolve(fixture.build, "onnxruntime.dll"), "onnxruntime B\n");

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain("two different runtime libraries are both called onnxruntime.dll");
  });

  // Round 3, S2. A bare `mkdir -p` whose status nobody read, in a function
  // called on the left of `||` — so `set -e` was suspended through its whole
  // body and the failure did not even abort. Every check below it would then
  // be asking about a directory that is not there.
  it("refuses when the stage directory cannot be created", () => {
    const fixture = makeFixture();
    writeFileSync(fixture.source, "the daemon this run built\n");
    // A regular file where the stage's parent has to be a directory. `mkdir -p`
    // cannot make this and says so; nothing else about the fixture is broken.
    const blocker = resolve(fixture.root, "blocker");
    writeFileSync(blocker, "not a directory\n");
    fixture.stage = resolve(blocker, "stage");

    const result = runStaging(fixture);
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
    expect(result.stderr).toContain("could not create the dev daemon stage directory");
  });

  it("expects exactly the runtime libraries the rest of the repository stages", () => {
    // The expected set is not a guess and must not become one. Three files
    // already say what the daemon loads from its own directory; this reads all
    // three and fails when the list in dev-runtime.sh stops agreeing with them.
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const declared = script.match(/local -a expected_libraries=\(([^)]*)\)/);
    expect(declared, "stage_windows_daemon no longer declares expected_libraries").not.toBeNull();
    const expected = declared![1].trim().split(/\s+/).sort();

    // 1. prepare-sidecars.sh is what PUTS them in app/binaries, and refuses to
    //    finish for a Windows triple without them. Its list carries a licence
    //    text file too, which is not loaded and so is not staged here.
    const prepare = readFileSync(resolve(root, "scripts/prepare-sidecars.sh"), "utf8");
    const staged = prepare.match(/WINDOWS_RUNTIME_FILES=\(([^)]*)\)/);
    expect(staged, "prepare-sidecars.sh no longer declares WINDOWS_RUNTIME_FILES").not.toBeNull();
    expect(
      staged![1]
        .trim()
        .split(/\s+/)
        .filter((name) => name.endsWith(".dll"))
        .sort(),
    ).toEqual(expected);

    // 2. The Windows bundle ships exactly these out of app/binaries, which is
    //    the release-time statement of the same fact.
    const bundled = JSON.parse(
      readFileSync(resolve(root, "app/tauri.windows.conf.json"), "utf8"),
    ) as { bundle?: { resources?: string[] | Record<string, string> } };
    // Tauri takes `resources` either as a list or as a source -> destination
    // map, and this file uses the map.
    const resources = bundled.bundle?.resources ?? [];
    const sources = Array.isArray(resources) ? resources : Object.keys(resources);
    expect(sources.length, "app/tauri.windows.conf.json bundles no resources").toBeGreaterThan(0);
    expect(
      sources
        .filter((entry) => entry.endsWith(".dll"))
        .map((entry) => entry.replace(/^.*\//, ""))
        .sort(),
    ).toEqual(expected);

    // 3. And the REASON: neither library is inside the executable. fastembed is
    //    built with `ort-load-dynamic`, so ONNX Runtime is dlopen'd at runtime,
    //    and llama-cpp-2 is built with `vulkan`, which links the loader.
    const cargo = readFileSync(resolve(root, "crates/wenlan-core/Cargo.toml"), "utf8");
    expect(cargo).toContain("ort-load-dynamic");
    expect(cargo).toMatch(/llama-cpp-2[\s\S]{0,200}vulkan/);
  });

  it("start_runtime stages the Windows daemon and propagates the failure", () => {
    // Every case above drives `stage_windows_daemon` directly, which is how a
    // fix can be complete and unreached at the same time: none of them would
    // notice the call being deleted from `start_runtime`, or its status being
    // swallowed with `|| true`. So the SHIPPED call site is read here.
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const code = script
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");

    const start = code.indexOf("start_runtime() {");
    expect(start, "start_runtime is not a function in dev-runtime.sh").toBeGreaterThan(-1);
    const body = code.slice(start, code.indexOf("\n}\n", start));

    // Called, and only on the Windows path.
    const calls = body.match(/^\s*stage_windows_daemon\b.*$/gm) ?? [];
    expect(calls, "start_runtime does not call stage_windows_daemon").toHaveLength(1);
    const call = calls[0]!;
    const guard = body.slice(0, body.indexOf(call));
    expect(
      guard.slice(guard.lastIndexOf("if ")),
      "the staging call is not guarded by the Windows branch",
    ).toContain("HOST_IS_WINDOWS == 1");

    // And its failure ends the start. `|| true` is the shape that was already
    // found once inside this function, on the DLL copy.
    expect(call, "the staging call swallows its own status").not.toMatch(/\|\|\s*true/);
    const after = body.slice(body.indexOf(call) + call.length);
    // To the end of the Windows branch, and no further — a `return 1` belonging
    // to some later block is not this call's propagation. An unclosed branch is
    // a failure here rather than a slice that silently runs to the end.
    const closes = after.indexOf("\n  fi");
    expect(closes, "the Windows branch around the staging call never closes").toBeGreaterThan(-1);
    const propagation = after.slice(0, closes);
    expect(call + propagation, "a staging failure does not stop the start").toMatch(
      /\|\|\s*\{[\s\S]*?return 1/,
    );
    // And classified as its OWN kind. `build-failure` was wrong in the way that
    // costs a caller a retry loop: cargo succeeded three lines up, so the
    // remediation a build failure calls for — fix the source, build again — is
    // the one thing that cannot help with the failure that actually happens
    // here, a DLL held open by a daemon running out of the stage directory.
    expect(call + propagation, "a staging failure is not classified").toContain(
      "RESULT_KIND=staging-failure",
    );
    expect(
      call + propagation,
      "a staging failure is still reported as a build failure",
    ).not.toContain("RESULT_KIND=build-failure");
    // The cargo build above it keeps its own kind, so the two are genuinely
    // distinguishable rather than renamed.
    const buildStep = body.slice(0, body.indexOf(call));
    expect(buildStep, "the cargo build no longer reports build-failure").toContain(
      "RESULT_KIND=build-failure",
    );
  });

  it("stages by content identity and lets no copy failure pass", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const start = script.indexOf("stage_windows_daemon() {");
    expect(start).toBeGreaterThan(-1);
    // The prose above the loop explains both traps by name, so this reads the
    // code and not the commentary about it — the same rule the awk shape
    // contracts in scripts/host-process.test.ts follow.
    const body = script
      .slice(start, script.indexOf("\n}\n", start))
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");

    // `cp -u` is a timestamp comparison wearing a provenance claim, and
    // `|| true` is the DLL half of the same thing. Pinned as shapes because the
    // behavioural cases above cannot prove the absence of a second copy site.
    expect(body).not.toContain("cp -u");
    expect(body).not.toContain("|| true");
    expect(body).toContain("stage_file_by_identity");
  });
});

// Round 4, the audit half of S1–S3. Every function this file calls on the LEFT
// OF `||` runs with errexit suspended through its whole body, so a dropped
// status inside it is not merely unread — it does not even abort.
// `refuse_production_path` calls this one that way, and it is the guard that
// keeps a dev daemon off the real data directory.
//
// `child="$(printf … | tr …)"` had nowhere to put a failure: a `tr` that could
// not run yields the empty string, the empty string is neither `$parent` nor
// `$parent/`-prefixed, and the comparison answers "outside production" — the
// one answer that lets the run continue.
describe("dev runtime production-root comparison", () => {
  function extract(name: string): string {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const lines = script.split("\n");
    const start = lines.indexOf(`${name}() {`);
    expect(start, `${name} is not a function in dev-runtime.sh`).toBeGreaterThan(-1);
    let depth = 0;
    for (let i = start; i < lines.length; i += 1) {
      depth += (lines[i].match(/\{/g) ?? []).length;
      depth -= (lines[i].match(/\}/g) ?? []).length;
      if (depth === 0) return lines.slice(start, i + 1).join("\n");
    }
    throw new Error(`${name} is never closed in dev-runtime.sh`);
  }

  const slash = (path: string) => path.split("\\").join("/");

  function within(child: string, parent: string, trShim?: string) {
    const dir = mkdtempSync(resolve(tmpdir(), "wenlan-within-"));
    tempRoots.push(dir);
    const preamble = ["#!/usr/bin/env bash", "set -euo pipefail"];
    if (trShim !== undefined) {
      const shimDir = resolve(dir, "shim");
      mkdirSync(shimDir, { recursive: true });
      const shimPath = resolve(shimDir, "tr");
      writeFileSync(shimPath, `#!/usr/bin/env bash\n${trShim}\n`, { mode: 0o755 });
      chmodSync(shimPath, 0o755);
      // Git for Windows puts /usr/bin ahead of an inherited PATH, so the shim
      // is prepended from inside the driver rather than from the environment.
      preamble.push(
        `shim_dir="${slash(shimDir)}"`,
        'if command -v cygpath >/dev/null 2>&1; then shim_dir="$(cygpath -u "$shim_dir")"; fi',
        'PATH="$shim_dir:$PATH"',
        "export PATH",
      );
    }
    const driver = [
      ...preamble,
      "HOST_IS_WINDOWS=1",
      extract("path_is_within"),
      "rc=0",
      'path_is_within "$1" "$2" || rc=$?',
      "printf 'rc=%s\\n' \"$rc\"",
    ].join("\n");
    const driverPath = resolve(dir, "driver.sh");
    writeFileSync(driverPath, driver, { mode: 0o755 });
    const options = { cwd: root, encoding: "utf8" as const, env: { ...process.env } };
    const result =
      process.platform === "win32"
        ? spawnSync(
            process.execPath,
            ["scripts/run-bash.mjs", slash(driverPath), child, parent],
            options,
          )
        : spawnSync("bash", [driverPath, child, parent], options);
    return { stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
  }

  const INSIDE = "c:/users/x/AppData/Local/WENLAN/memorydb";
  const ROOT = "C:/Users/x/AppData/Local/wenlan";

  it("folds case before comparing, so a second spelling is still production", () => {
    expect(within(INSIDE, ROOT).stdout.trim()).toBe("rc=0");
  });

  it("reports a MEASURED negative for a path that really is outside", () => {
    expect(within("C:/Users/x/AppData/Local/wenlan-dev", ROOT).stdout.trim()).toBe("rc=1");
  });

  // The same inputs as the first, and the fold cannot run. Measured on the
  // pre-fix code: BOTH operands collapse to the empty string, the empty string
  // equals itself, and the answer is 0 — "inside production", a refusal that
  // is correct BY ACCIDENT. Nothing chose that direction, which is the point:
  // the answer is decided by how the failure happens to be shaped.
  it("reports COULD NOT MEASURE when the case fold cannot run", () => {
    const result = within(INSIDE, ROOT, "exit 1");
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  it("reports COULD NOT MEASURE when the case fold exits 0 saying nothing", () => {
    const result = within(INSIDE, ROOT, "exit 0");
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  // THE case, and the reason the one above is not enough. When the fold fails
  // for ONE operand — the second call, the second read of a device that is
  // going, a `tr` killed by a resource limit part-way through the pair — the
  // pre-fix code compared a folded child against an EMPTY parent, answered 1,
  // and `refuse_production_path` read that as "outside production" and let the
  // dev runtime point at %LOCALAPPDATA%\wenlan. Measured: rc=1 without the
  // status checks, rc=2 with them.
  it("reports COULD NOT MEASURE when the case fold fails for one operand only", () => {
    const result = within(
      INSIDE,
      ROOT,
      [
        'count="$(dirname "$0")/tr.count"',
        "n=0",
        'if [ -f "$count" ]; then n="$(cat "$count")"; fi',
        "n=$((n + 1))",
        `printf '%s' "$n" >"$count"`,
        'if [ "$n" -ge 2 ]; then exit 1; fi',
        "exec /usr/bin/tr \"$@\"",
      ].join("\n"),
    );
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  // And the only caller acts on all three: a 2 folded into 1 would undo every
  // case above without changing one of them.
  it("refuses the production check it could not make", () => {
    const script = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const start = script.indexOf("refuse_production_path() {");
    expect(start).toBeGreaterThan(-1);
    const body = script
      .slice(start, script.indexOf("\n}\n", start))
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");
    expect(body).toContain('path_is_within "$canonical" "$resolved" || within_rc=$?');
    expect(body, "the unmeasured comparison is not refused").toContain(
      "could not compare $label against the production root",
    );
    // The refusal has to be the catch-all arm, not a fourth named status.
    expect(body).toMatch(/\*\)\s*\n\s*refuse_unsafe/);
  });
});

// ROUND 4 (Codex Sol). Three findings meet in this file's lock, and all three
// are the same shape as everything above them: a status that existed, that
// nothing read, and whose loss is spelled the same way as an ordinary answer.
//
//   2. `release_runtime_lock` returned 0 -- "released" -- from three tests it
//      had not passed. The main operation could therefore exit 0 with
//      `DEV_RUNTIME_RESULT: ok` while its lock was still on disk, and the next
//      user-visible command refused on a lock the previous successful one said
//      it had released. The `|| true` at the trap was not the collapse: the
//      collapse was INSIDE the function, before `|| true` ever saw a status.
//   3. `for _ in $(seq 1 50)` performs ZERO measurements when `seq` cannot run,
//      and then returns its ordinary negative -- "still alive", "nothing
//      reaped", "never appeared". The loops are arithmetic now, so there is no
//      status to drop.
//   4. `lock_owner_file_appeared` returned 1 both for "the lock is gone" and
//      for "the lock is here and still names nobody". Its own comment promised
//      the first would go round again and let `mkdir` arbitrate; the caller
//      refused on both.
describe("dev runtime lock release and recovery", () => {
  function extract(script: string, name: string): string {
    const lines = script.split("\n");
    const start = lines.indexOf(`${name}() {`);
    expect(start, `${name} is not a function in dev-runtime.sh`).toBeGreaterThan(-1);
    let depth = 0;
    for (let i = start; i < lines.length; i += 1) {
      depth += (lines[i].match(/\{/g) ?? []).length;
      depth -= (lines[i].match(/\}/g) ?? []).length;
      if (depth === 0) return lines.slice(start, i + 1).join("\n");
    }
    throw new Error(`${name} is never closed in dev-runtime.sh`);
  }
  const posix = (path: string) => path.replace(/\\/g, "/");
  const source = () => readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
  const libPath = posix(resolve(root, "scripts/lib/host-process.sh"));

  type Run = { status: number | null; stdout: string; stderr: string };

  // A driver is the shipped functions, verbatim, over a real temp state
  // directory. dev-runtime.sh dispatches on "$1" at top level and cannot be
  // sourced, so the functions are extracted; nothing about them is rewritten.
  function drive(
    body: string,
    fns: string[],
    options: { shims?: Record<string, string>; sourceLib?: boolean } = {},
  ): Run & { dir: string } {
    const dir = mkdtempSync(resolve(tmpdir(), "wenlan-dev-lock-"));
    tempRoots.push(dir);
    const state = resolve(dir, "state");
    mkdirSync(state, { recursive: true });
    const script = source();
    const head = [
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      // Git for Windows puts /usr/bin at the front of PATH before this runs, so
      // a shim handed in from outside would lose to /usr/bin/sed.
      'if [ -n "${WENLAN_TEST_SHIM_DIR:-}" ]; then',
      '  shim_dir="$WENLAN_TEST_SHIM_DIR"',
      '  if command -v cygpath >/dev/null 2>&1; then shim_dir="$(cygpath -u "$shim_dir")"; fi',
      '  PATH="$shim_dir:$PATH"',
      "  export PATH",
      "fi",
      'STATE_DIR="$WENLAN_TEST_STATE_DIR"',
      'LOCK_DIR="$STATE_DIR/runtime.lock"',
      'LOCK_OWNER_FILE="$LOCK_DIR/pid"',
      "RUNTIME_LOCK_HELD=0",
      // Mirrors the file-scope initialisation in dev-runtime.sh. The driver
      // runs under `set -u`, and `on_runtime_exit` reads this, so a driver that
      // did not declare it would abort inside the trap — which is the one place
      // this file's contract cannot express a failure.
      "RUNTIME_LOCK_STOLEN=0",
      // The release compares the owner file against this run's ACQUISITION
      // token rather than against `$$`, and `acquire_runtime_lock` mints one.
      'RUNTIME_LOCK_TOKEN=""',
      "RUNTIME_LOCK_GEN=0",
      "RUNTIME_EXIT_RAN=0",
      "RESULT_EMITTED=0",
      "RESULT_KIND=ok",
    ];
    if (options.sourceLib) head.push(`. "${libPath}"`);
    const driver = [...head, ...fns.map((name) => extract(script, name)), body].join("\n");
    const driverPath = resolve(dir, "driver.sh");
    writeFileSync(driverPath, driver, { mode: 0o755 });

    const env: Record<string, string> = {};
    for (const [key, value] of Object.entries(process.env)) {
      if (value !== undefined) env[key] = value;
    }
    env.WENLAN_TEST_STATE_DIR = posix(state);
    env.WENLAN_HOST_PROCESS_PLATFORM = "windows";
    const shims = options.shims ?? {};
    if (Object.keys(shims).length > 0) {
      const shimDir = resolve(dir, "shim");
      mkdirSync(shimDir, { recursive: true });
      for (const [name, shimBody] of Object.entries(shims)) {
        const path = resolve(shimDir, name);
        writeFileSync(path, `#!/usr/bin/env bash\n${shimBody}\n`, { mode: 0o755 });
        chmodSync(path, 0o755);
      }
      env.WENLAN_TEST_SHIM_DIR = shimDir;
    }
    const result =
      process.platform === "win32"
        ? spawnSync(process.execPath, ["scripts/run-bash.mjs", posix(driverPath)], {
            cwd: root,
            encoding: "utf8",
            env,
          })
        : spawnSync("bash", [driverPath], { cwd: root, encoding: "utf8", env });
    return {
      status: result.status,
      stdout: result.stdout ?? "",
      stderr: result.stderr ?? "",
      dir: state,
    };
  }

  const lastLine = (text: string) => {
    const lines = text.replace(/\r/g, "").split("\n").filter((line) => line !== "");
    return lines[lines.length - 1];
  };

  // --- DEFECT 2: a release that did not happen ------------------------------

  // The whole outcome path, driven: the trap is installed, the lock is declared
  // held, and the script exits 0 with RESULT_KIND=ok. What the marker says
  // afterwards is the entire question.
  const RELEASE_DRIVER = [
    "trap on_runtime_exit EXIT",
    "RUNTIME_LOCK_HELD=1",
    "exit 0",
  ].join("\n");
  // `list_dir_tristate` and `listing_has_name` are in here because round 5 made
  // the release ASK the lock directory a three-answer question instead of
  // testing `-f`/`-d`: two two-answer tests that both answer "absent" for a
  // path that could not be examined, which is how a release nobody could
  // measure reported itself as a clean one.
  const releaseFns = [
    "emit_result",
    "listing_has_name",
    "list_dir_tristate",
    "release_runtime_lock",
    "on_runtime_exit",
  ];

  // The fixture is built by the driver itself, out of the environment, so there
  // is exactly one temp directory in play and no ordering between node and bash
  // to get wrong — and, more to the point, the owner file has to be written with
  // the DRIVER's `$$`, which only the driver knows.
  const releaseRun = (shims: Record<string, string> = {}): Run & { dir: string } =>
    drive(['eval "$WENLAN_TEST_FIXTURE"', RELEASE_DRIVER].join("\n"), releaseFns, { shims });

  // The owner file carries this run's acquisition token, which is what the
  // release compares against; a bare `$$` is somebody else's record now.
  const FIXTURE_OWNED =
    'mkdir -p "$LOCK_DIR"; RUNTIME_LOCK_TOKEN="$$ 1.1.1"; ' +
    'printf "%s\\n" "$RUNTIME_LOCK_TOKEN" >"$LOCK_OWNER_FILE"';
  const FIXTURE_DIR_ONLY = 'mkdir -p "$LOCK_DIR"';
  const FIXTURE_UNREMOVABLE = 'mkdir -p "$LOCK_DIR/squatter"';
  const FIXTURE_OTHER_OWNER = 'mkdir -p "$LOCK_DIR"; printf "999999\\n" >"$LOCK_OWNER_FILE"';
  // ROUND 6. The lock this run took is GONE — no directory, no owner file —
  // while `RUNTIME_LOCK_HELD` is 1. `mkdir -p "$STATE_DIR"` and not a bare `:`
  // so the absence is one `list_dir_tristate` can MEASURE (it walks to the
  // parent and finds `runtime.lock` missing from a listing it really read)
  // rather than one it merely failed to examine, which is a different arm.
  const FIXTURE_STOLEN = 'mkdir -p "$STATE_DIR"; rm -rf "$LOCK_DIR"';

  const withFixture = (fixture: string, shims: Record<string, string> = {}) => {
    const saved = process.env.WENLAN_TEST_FIXTURE;
    process.env.WENLAN_TEST_FIXTURE = fixture;
    try {
      return releaseRun(shims);
    } finally {
      if (saved === undefined) delete process.env.WENLAN_TEST_FIXTURE;
      else process.env.WENLAN_TEST_FIXTURE = saved;
    }
  };

  // The control on the three cases below: a lock this run really does own is
  // released, and the run reports `ok`. Without it, "always say unknown" would
  // pass every case that follows.
  it("releases its own lock and reports ok", () => {
    const run = withFixture(FIXTURE_OWNED);
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: ok");
    expect(run.status).toBe(0);
  });

  // THE case. The lock directory outlived its owner file -- which is precisely
  // the state `acquire_runtime_lock` refuses on, so a run that leaves it
  // silently has armed the next command's refusal while telling its own caller
  // everything went fine.
  //
  // ROUND 6 changed one assertion here and the reason is not about this fixture.
  // The arm this drives no longer says "did not come off", because that wording
  // describes a leftover this run should have tidied, and an owner file this run
  // WROTE and that is now gone is not a leftover -- it is a claim broken while
  // the run still held it. Same event as a vanished directory, same words.
  it("refuses to report ok when the lock directory outlives the release", () => {
    const run = withFixture(FIXTURE_UNREMOVABLE);
    expect(run.stderr).toContain("outlived its owner file");
    expect(run.stderr).toContain("broken while it still held it");
    expect(run.stderr).not.toContain("its dev runtime lock did not come");
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: unknown");
    expect(run.status).not.toBe(0);
  });

  // The `sed` Codex named. A read that FAILED and a read that returned nothing
  // were the same empty string, and the empty string is not `$$`, so this used
  // to fall into the "not mine, leave it" branch and return 0 anyway.
  it("refuses to report ok when the owner file could not be read", () => {
    const run = withFixture(FIXTURE_OWNED, { sed: "exit 1" });
    expect(run.stderr).toContain("could not be read at release");
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: unknown");
    expect(run.status).not.toBe(0);
  });

  // A lock recorded to somebody else, while this run believed it held it.
  // Leaving it alone is right; calling that a release is not.
  it("refuses to report ok when the lock is recorded to another run", () => {
    const run = withFixture(FIXTURE_OTHER_OWNER);
    expect(run.stderr).toContain("recorded to another acquisition");
    expect(run.stderr).toContain("[999999]");
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: unknown");
    expect(run.status).not.toBe(0);
    // ROUND 6, and it is about the WORDS, because the status was already
    // right. This is the second shape of a broken claim -- the lock came off
    // and was retaken -- so the outcome line must not send a reader looking
    // for a leftover directory this run failed to tidy.
    expect(run.stderr).toContain("broken while it still held it");
    expect(run.stderr).not.toContain("its dev runtime lock did not come");
  });

  // ROUND 6. THIS CASE IS REVERSED, AND THE REVERSAL IS THE POINT.
  //
  // It read: "still reports ok when an empty lock directory is tidied away",
  // asserting `DEV_RUNTIME_RESULT: ok` and status 0, over the rationale "a lock
  // whose directory is gone WITH its owner file really was released -- a
  // recovery that found this run dead is entitled to do that -- so an empty
  // removable directory is not an error. This is what keeps the three cases
  // above from being 'any anomaly is unknown'." Three things are wrong with
  // that, and the third is the one that forced the change.
  //
  // ONE: the rationale does not describe this fixture. It describes a directory
  // that is GONE; `FIXTURE_DIR_ONLY` is a directory that is PRESENT with its
  // owner file gone. The state the rationale actually describes is
  // `FIXTURE_STOLEN`, and round 6 already reclassified that one as theft --
  // including the "entitled to" clause, which asserts a recovery nothing on the
  // path measured, about a run that is by construction not dead.
  //
  // TWO: the anti-over-refusal duty this case claimed is not its own. It
  // belongs to "releases its own lock and reports ok" above, whose comment says
  // so in as many words: "Without it, 'always say unknown' would pass every
  // case that follows." That control is untouched and still green, and "does
  // not call an unexaminable lock a stolen one" below pins the theft verdict
  // specifically. Nothing is lost by moving this one.
  //
  // THREE, and decisive: what made this case `ok` and `FIXTURE_UNREMOVABLE`
  // `unknown` was nothing but whether `rmdir` SUCCEEDED. Once the release stops
  // performing a removal it cannot license -- see the arm in dev-runtime.sh --
  // the two fixtures are one indistinguishable state and cannot carry opposite
  // verdicts. An empty ownerless lock directory is not the safe end of that
  // pair; it is the DANGEROUS end, because it is exactly what a fresh holder
  // looks like between its `mkdir` and its owner write. The old code was most
  // willing to destroy precisely where destruction was most likely to hit a
  // live peer.
  it("refuses to report ok when its owner file was removed under it", () => {
    const run = withFixture(FIXTURE_DIR_ONLY);
    expect(run.stderr).toContain("outlived its owner file");
    expect(run.stderr).toContain("broken while it still held it");
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: unknown");
    expect(run.status).not.toBe(0);
  });

  // THE ABA, and the only assertion in this file that is about an ACT rather
  // than a verdict. Every other case here reads stderr and the marker, and a
  // release that still ran `rmdir` on a directory it cannot prove is its own
  // would satisfy all of them as long as it reported the refusal afterwards.
  // This is the witness that the directory is still on disk when the run ends.
  //
  // `FIXTURE_DIR_ONLY` is the world-2 shape -- an empty lock directory with no
  // owner file, indistinguishable from a peer that has just `mkdir`'d and not
  // yet written its pid -- so removing it here is removing a live peer's lock,
  // after which two runs share this worktree's port and data directory. Left
  // alone, it is `acquire_runtime_lock`'s to arbitrate, which is where the
  // four-state `lock_owner_file_appeared` waiter already lives.
  it("does not remove a lock directory it cannot prove is its own", () => {
    const run = withFixture(FIXTURE_DIR_ONLY);
    expect(existsSync(resolve(run.dir, "runtime.lock")), run.stderr).toBe(true);
    expect(run.stderr).toContain("cannot be shown to be the");
  });

  // And the boundary THAT must not overrun: refusing to remove what is not ours
  // is not "never remove anything". A lock this run does own is still released,
  // owner file and directory both, or the fix above would have turned every run
  // into one that leaves a lock behind for the next command to refuse -- the
  // exact failure `release_runtime_lock` exists to prevent, reintroduced from
  // the other side.
  it("still removes the lock directory when it is provably this run's", () => {
    const run = withFixture(FIXTURE_OWNED);
    expect(existsSync(resolve(run.dir, "runtime.lock")), run.stderr).toBe(false);
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: ok");
    expect(run.status).toBe(0);
  });

  // ROUND 6. THE ARM THAT REPORTED A STOLEN LOCK AS A CLEAN RELEASE.
  //
  // `list_dir_tristate` answering 1 -- the lock directory is MEASURED absent --
  // used to `return 0`, on the stated reasoning that "a recovery that found
  // this run dead is entitled to have done that". Nothing on that path
  // establishes any such recovery, and the run reaching it is by construction
  // not dead: `RUNTIME_LOCK_HELD` is set in exactly one place, last in
  // `acquire_runtime_lock` after every step succeeded, and `release_runtime_lock`
  // only runs when it is 1. So this state is "this run took the lock, and the
  // lock is gone" -- the exclusive claim broken while the run was still using
  // it, with the isolated port and data directory unfenced for that window --
  // and it exited 0 with `DEV_RUNTIME_RESULT: ok`.
  //
  // It is the same condition `attest.sh` reports through `LOCK_STOLEN`, and
  // the two files must not disagree about what a vanished lock means.
  it("refuses to report ok when the lock it held vanished under it", () => {
    const run = withFixture(FIXTURE_STOLEN);
    expect(run.stderr).toContain("the dev runtime lock this run took is gone");
    expect(run.stderr).toContain("broken while it still held it");
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: unknown");
    expect(run.status).not.toBe(0);
  });

  // The other half of that verdict, and the half a status assertion cannot
  // reach: a stolen lock is not a lock left behind, and the line a human reads
  // must not say it is. Without this, replacing the whole branch with the
  // generic "did not come off" message would keep every assertion above green
  // while telling the reader to go looking for a directory that is not there.
  it("does not describe a stolen lock as one that failed to come off", () => {
    const run = withFixture(FIXTURE_STOLEN);
    expect(run.stderr).not.toContain("its dev runtime lock did not come");
    expect(run.stderr).toContain("port and data directory at the same time");
  });

  // And the boundary this must not overrun, stated as its own case: a lock
  // that is gone is only reportable as stolen when its absence was MEASURED.
  // An `ls` that cannot list anything must still reach the could-not-examine
  // refusal above, not the theft one -- "somebody took it" is a claim, and a
  // failed measurement is not evidence for a claim.
  it("does not call an unexaminable lock a stolen one", () => {
    const run = withFixture([FIXTURE_STOLEN, "ls() { return 2; }"].join("\n"));
    expect(run.stderr).toContain("could not be examined at release");
    expect(run.stderr).not.toContain("is gone from");
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: unknown");
    expect(run.status).not.toBe(0);
  });

  // ROUND 5, and it is the case above with the one thing it cannot distinguish
  // itself from. The release opened with `[[ ! -f "$LOCK_OWNER_FILE" ]]` and
  // then `[[ -d "$LOCK_DIR" ]] || return 0` -- and BOTH are false when the path
  // cannot be examined at all, not only when it is not there. Remove search
  // permission from an ancestor, lose the mount, hit an ACL: `-f` says "no owner
  // file", `-d` says "no directory", and the function returned 0 -- RELEASED --
  // for a lock it never managed to look at, keeping RESULT_KIND=ok and exiting
  // 0 with the lock still on disk. The two-answer test was the whole of it.
  //
  // An `ls` that cannot list anything is the same state from in here, and the
  // owner file IS there: the answer has to be the refusal, not the tidy-away
  // above it.
  it("refuses to report ok when the lock cannot be examined at all", () => {
    const run = withFixture([FIXTURE_OWNED, "ls() { return 2; }"].join("\n"));
    expect(run.stderr).toContain("could not be examined at release");
    expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: unknown");
    expect(run.status).not.toBe(0);
  });

  // The kind is downgraded only from `ok`. A run that already measured a
  // specific failure keeps its diagnosis: `unknown` is a refusal, and replacing
  // `health-failure` with it would throw away what the consumer acts on.
  it("keeps a measured failure kind rather than replacing it with unknown", () => {
    const saved = process.env.WENLAN_TEST_FIXTURE;
    process.env.WENLAN_TEST_FIXTURE = FIXTURE_UNREMOVABLE;
    try {
      const run = drive(
        [
          'eval "$WENLAN_TEST_FIXTURE"',
          "trap on_runtime_exit EXIT",
          "RUNTIME_LOCK_HELD=1",
          "RESULT_KIND=health-failure",
          "exit 1",
        ].join("\n"),
        releaseFns,
      );
      expect(lastLine(run.stderr), run.stderr).toBe("DEV_RUNTIME_RESULT: health-failure");
      expect(run.status).toBe(1);
    } finally {
      if (saved === undefined) delete process.env.WENLAN_TEST_FIXTURE;
      else process.env.WENLAN_TEST_FIXTURE = saved;
    }
  });

  // --- DEFECT 3: a loop that measured nothing -------------------------------

  // `wait_for_owned_exit` polled `for _ in $(seq 1 50)`. A `seq` that cannot
  // run yields the empty word list, the loop body never executes, and the
  // function falls straight through to `return 1` -- "the process is still
  // alive after the window", from fifty measurements that never happened. Its
  // caller's response to 1 is to keep the ownership record and refuse; its
  // response after a real kill is to report the daemon unstoppable. Both are
  // wrong, and neither is distinguishable from the honest answer.
  //
  // The loops are arithmetic now, so `seq` is not on the path at all. This case
  // puts a `seq` that exits 127 on PATH and requires the answer to be the
  // measurement rather than the tool.
  it("measures liveness even when `seq` cannot run", () => {
    // A complete tasklist table with our pid absent: the measured "gone".
    const gone = [
      '"System","4","Console","1","9,000 K"',
      '"Registry","132","Console","1","9,000 K"',
      '"smss.exe","608","Console","1","9,000 K"',
      '"csrss.exe","888","Console","1","9,000 K"',
      '"wininit.exe","964","Console","1","9,000 K"',
      '"services.exe","1048","Console","1","9,000 K"',
      '"lsass.exe","1072","Console","1","9,000 K"',
      '"svchost.exe","1576","Console","1","9,000 K"',
      '"svchost.exe","1704","Console","1","9,000 K"',
      '"explorer.exe","5320","Console","1","9,000 K"',
      '"bash.exe","9012","Console","1","9,000 K"',
    ];
    const run = drive(
      [
        "OWNED_PID=4242",
        "rc=0",
        "wait_for_owned_exit || rc=$?",
        `printf 'rc=%s\\n' "$rc"`,
      ].join("\n"),
      ["wait_for_owned_exit"],
      {
        sourceLib: true,
        shims: {
          tasklist: `printf '%s\\n' ${gone.map((row) => `'${row}'`).join(" ")}`,
          // The tool the loop must not depend on.
          seq: "exit 127",
        },
      },
    );
    expect(run.stdout.trim(), run.stderr).toBe("rc=0");
  }, 60_000);

  // ROUND 5. The round count came off `seq`; the DELAY was still a bare `sleep
  // 0.1` with its status dropped, and this function is called from the left of
  // a `||` at every site -- errexit off through the whole body -- so a `sleep`
  // that fails runs all fifty probes in microseconds and returns 1: "still
  // alive when the window closed". `stop_runtime` reads that 1 as licence to
  // escalate to a FORCE KILL, five seconds early, on a window nobody waited
  // through. The pid is ALIVE in this table, so the honest answers are 1 after
  // five real seconds or 2 -- and only 2 is available when the wait did not
  // happen.
  it("reports could-not-measure when the wait between liveness probes cannot happen", () => {
    const alive = [
      '"System","4","Console","1","9,000 K"',
      '"Registry","132","Console","1","9,000 K"',
      '"smss.exe","608","Console","1","9,000 K"',
      '"csrss.exe","888","Console","1","9,000 K"',
      '"wininit.exe","964","Console","1","9,000 K"',
      '"services.exe","1048","Console","1","9,000 K"',
      '"lsass.exe","1072","Console","1","9,000 K"',
      '"svchost.exe","1576","Console","1","9,000 K"',
      '"svchost.exe","1704","Console","1","9,000 K"',
      '"explorer.exe","5320","Console","1","9,000 K"',
      '"bash.exe","9012","Console","1","9,000 K"',
      '"wenlan-server.exe","4242","Console","1","9,000 K"',
    ];
    const run = drive(
      [
        "OWNED_PID=4242",
        "rc=0",
        "wait_for_owned_exit || rc=$?",
        `printf 'rc=%s\\n' "$rc"`,
      ].join("\n"),
      ["wait_for_owned_exit"],
      {
        sourceLib: true,
        shims: {
          tasklist: `printf '%s\\n' ${alive.map((row) => `'${row}'`).join(" ")}`,
          sleep: "exit 1",
        },
      },
    );
    expect(run.stdout.trim(), run.stderr).toBe("rc=2");
  }, 60_000);

  // The shape assertion behind it, across both shell files: a `for … in $(…)`
  // has nowhere to put the substitution's status, so the class is removed
  // rather than checked. Codex named three sites; there are five.
  it("has no loop whose round count comes from an unchecked command", () => {
    for (const path of ["scripts/dev-runtime.sh", "scripts/lib/host-process.sh"]) {
      const text = readFileSync(resolve(root, path), "utf8")
        .split("\n")
        .filter((line) => !/^\s*#/.test(line))
        .join("\n");
      expect(text, `${path} has a for-loop bounded by a command substitution`).not.toMatch(
        /for\s+\S+\s+in\s+\$\(/,
      );
    }
  });

  // --- DEFECT 4: two answers spelled the same -------------------------------

  const acquireFns = [
    "list_dir_tristate",
    "listing_has_name",
    "lock_owner_file_appeared",
    "lock_new_token",
    "acquire_runtime_lock",
  ];
  const ACQUIRE_DRIVER = [
    'eval "$WENLAN_TEST_FIXTURE"',
    "rc=0",
    "acquire_runtime_lock || rc=$?",
    `printf 'rc=%s held=%s owner=%s\\n' "$rc" "$RUNTIME_LOCK_HELD" "$(cat "$LOCK_OWNER_FILE" 2>/dev/null || printf none)"`,
  ].join("\n");

  const acquireRun = (fixture: string) => {
    const saved = process.env.WENLAN_TEST_FIXTURE;
    process.env.WENLAN_TEST_FIXTURE = fixture;
    try {
      return drive(ACQUIRE_DRIVER, acquireFns, { sourceLib: true });
    } finally {
      if (saved === undefined) delete process.env.WENLAN_TEST_FIXTURE;
      else process.env.WENLAN_TEST_FIXTURE = saved;
    }
  };

  // THE case for finding 4. An ownerless lock that DISAPPEARS during the wait
  // is not "a lock that names nobody" -- it is no lock at all, and `mkdir` is
  // the arbiter, exactly as it is at the top of the function. The source
  // comment said so; the code returned the same 1 as the timeout and refused.
  //
  // The lock is removed by a background shell a moment after the wait starts,
  // which is what a holder releasing normally looks like from in here.
  it("retakes a lock that is released while it waits for an owner", () => {
    const run = acquireRun(
      [
        'mkdir -p "$LOCK_DIR"',
        // Released after the wait has begun. 0.5s is five rounds of the poll
        // and a tenth of its five-second deadline.
        '( sleep 0.5; rmdir "$LOCK_DIR" ) &',
      ].join("\n"),
    );
    // `PID gen.nonce.nonce`: the owner record names an acquisition, so two
    // generations of the directory cannot compare equal.
    expect(run.stdout.trim(), run.stderr).toMatch(/^rc=0 held=1 owner=\d+ \d+\.\d+\.\d+$/);
  }, 60_000);

  // And the answer that must NOT change: a lock that is there throughout and
  // still names nobody after the deadline is unattributable, and recovering it
  // is how two commands come to share a lock directory. Refusing costs a manual
  // `rm -rf` after the one thing that produces this state.
  it("refuses a lock that stays ownerless through the whole wait", () => {
    const run = acquireRun('mkdir -p "$LOCK_DIR"');
    expect(run.stdout.trim(), run.stderr).toBe("rc=1 held=0 owner=none");
    expect(run.stderr).toContain("names no owner");
  }, 60_000);

  // The third answer, unchanged: a lock directory that cannot be listed at all
  // is neither owned nor free, and nothing may be recovered from it.
  it("refuses a lock it cannot look inside", () => {
    const run = acquireRun(['mkdir -p "$LOCK_DIR"', 'ls() { return 2; }'].join("\n"));
    expect(run.stdout.trim(), run.stderr).toBe("rc=1 held=0 owner=none");
  }, 60_000);

  // --- THE STALE BREAK IS ONE ATOMIC RENAME ---------------------------------
  //
  // The ABA the re-read could not close. `rm -f owner` then `rmdir` is two
  // destructive steps, and the owner comparison sits BEFORE both of them, so a
  // second breaker could remove the same stale lock, `mkdir` a fresh one, write
  // its own token and start work inside the interval -- and the first breaker's
  // removals then destroyed that live generation, its own `mkdir` succeeded, and
  // both ran against one isolated port and data directory. The first breaker's
  // release compares its own token, finds it, and reports `ok`; only the victim
  // ever notices.
  //
  // `mv` is the removal and the test in one step: at most one process can move a
  // given directory away. The three cases below are the control, the ABA itself,
  // and the shape assertion that the two destructive steps are gone.
  const BREAK_DRIVER = [
    'eval "$WENLAN_TEST_FIXTURE"',
    "rc=0",
    "acquire_runtime_lock || rc=$?",
    "aside=0",
    // Anything left under the break-aside name is litter this run created. The
    // glob is checked with `-e` because an unmatched glob stays literal.
    'for d in "$STATE_DIR"/runtime.lock.breaking.*; do',
    '  if [[ -e "$d" ]]; then aside=$(( aside + 1 )); fi',
    "done",
    `printf 'rc=%s held=%s owner=%s aside=%s\\n' "$rc" "$RUNTIME_LOCK_HELD" ` +
      `"$(cat "$LOCK_OWNER_FILE" 2>/dev/null || printf none)" "$aside"`,
  ].join("\n");

  const breakRun = (fixture: string, shims?: Record<string, string>) => {
    const saved = process.env.WENLAN_TEST_FIXTURE;
    process.env.WENLAN_TEST_FIXTURE = fixture;
    try {
      return drive(BREAK_DRIVER, acquireFns, { sourceLib: true, shims });
    } finally {
      if (saved === undefined) delete process.env.WENLAN_TEST_FIXTURE;
      else process.env.WENLAN_TEST_FIXTURE = saved;
    }
  };

  // The control the two below are read against: a lock whose owner is measurably
  // gone is still broken and still taken, and the directory it was moved aside
  // into is not left behind.
  it("breaks a genuinely stale lock and leaves nothing moved aside", () => {
    const run = breakRun(['mkdir -p "$LOCK_DIR"', `printf '999999\\n' >"$LOCK_OWNER_FILE"`].join("\n"));
    expect(run.stdout.trim(), run.stderr).toMatch(
      /^rc=0 held=1 owner=\d+ \d+\.\d+\.\d+ aside=0$/,
    );
  }, 60_000);

  // THE ABA, driven. The owner re-read happens first and matches; the lock is
  // then replaced by a whole new generation inside the second test hook's
  // window, which is exactly the interleaving the re-read cannot see. Under
  // `rm`+`rmdir` this run deleted the new generation's owner file and its
  // directory and took the lock: `rc=0 held=1` beside a live holder. Under the
  // rename it moves that generation aside, reads a token that is not the one it
  // measured, puts it straight back and goes round -- where the new owner is
  // unparsable as a pid and is therefore refused rather than broken.
  //
  // The proof that it was put back INTACT is `owner=LIVE-GENERATION`: the
  // victim's own record, still in the lock directory, still readable.
  //
  // ARRANGED, NOT WAITED FOR. The interleaving is placed by SHIMMING the hook's
  // `sleep`, so the swap happens at the hook and nowhere else. A background
  // shell on a wall-clock delay would race the liveness probe -- slow enough and
  // the swap lands before the re-read instead of after it, and the run refuses
  // for the OTHER reason with the same stdout. The hook's `sleep` is the only
  // one this path reaches (`poll_delay` belongs to the ownerless wait, which an
  // owner file present from the start never enters).
  it("puts back a live generation it renamed away, and takes nothing", () => {
    const run = breakRun(
      ['mkdir -p "$LOCK_DIR"', `printf '999999\\n' >"$LOCK_OWNER_FILE"`, "DEV_RUNTIME_RACE_SLEEP_BREAK=1"].join(
        "\n",
      ),
      {
        sleep: [
          'lock="$WENLAN_TEST_STATE_DIR/runtime.lock"',
          'rm -f "$lock/pid"',
          'rmdir "$lock"',
          'mkdir "$lock"',
          `printf 'LIVE-GENERATION\\n' >"$lock/pid"`,
        ].join("\n"),
      },
    );
    expect(run.stdout.trim(), run.stderr).toBe("rc=1 held=0 owner=LIVE-GENERATION aside=0");
    expect(run.stderr).toContain("the dev runtime lock owner is not a pid");
  }, 60_000);

  // And the shape, because the behaviour above is only reachable while the two
  // destructive steps are actually gone. A `rmdir "$LOCK_DIR"` reintroduced
  // anywhere in this function is the defect coming back.
  it("breaks the stale lock by rename, with no two-step removal left in it", () => {
    const text = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const start = text.indexOf("acquire_runtime_lock() {");
    expect(start).toBeGreaterThan(-1);
    const body = text.slice(start, text.indexOf("\n}\n", start));
    const code = body
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");
    // Only the CONTENDED half — everything before the owner write. The
    // give-back after a failed owner write legitimately removes a directory
    // this run created moments earlier and is not a break of anybody's lock.
    const contended = code.slice(0, code.indexOf('token="$(lock_new_token)"'));
    expect(contended, "the stale break is not an atomic rename").toContain(
      'mv "$LOCK_DIR" "$breaking"',
    );
    expect(contended, "the stale break still removes the lock in two steps").not.toMatch(
      /rmdir "\$LOCK_DIR"/,
    );
    expect(contended, "the stale break still deletes the owner file in place").not.toMatch(
      /rm -f "\$LOCK_OWNER_FILE"\s*;?\s*then/,
    );
    // The renamed directory is read before it is destroyed, and destroyed only
    // when it is the generation that was measured.
    const rename = contended.indexOf('mv "$LOCK_DIR" "$breaking"');
    const verify = contended.indexOf('"$breaking/${LOCK_OWNER_FILE##*/}"');
    const destroy = contended.indexOf('rm -rf "$breaking"');
    expect(verify, "the renamed lock is destroyed without being read").toBeGreaterThan(rename);
    expect(destroy, "the renamed lock is destroyed before it is identified").toBeGreaterThan(
      verify,
    );
    // Both hooks exist, and the second one sits in the interval the first
    // cannot reach: after the re-read, before the rename.
    const reread = contended.indexOf('owner_again="$(sed');
    const hook2 = contended.indexOf("DEV_RUNTIME_RACE_SLEEP_BREAK");
    expect(contended).toContain("DEV_RUNTIME_RACE_SLEEP:-0");
    expect(hook2, "the second race hook does not follow the re-read").toBeGreaterThan(reread);
    expect(rename, "the second race hook does not precede the rename").toBeGreaterThan(hook2);
  });

  // --- ROUND 5: a LIVE daemon's record is not a stale one --------------------
  //
  // `is_owned_process` answered 1 -- "no" -- for two states that call for
  // opposite handling: the recorded pid is GONE (or is some other image), which
  // is a stale record and may be deleted, and the recorded pid is ALIVE and IS
  // the recorded executable but is not yet the listener on the recorded port,
  // which is a running daemon. Binding a socket happens milliseconds after the
  // process exists, so a launcher killed in that window leaves exactly the
  // second state -- and `start_runtime` handled 0 and 2, then fell through to
  // `clear_owned_state` and started a second daemon against the same data
  // directory, with nothing able to stop the first.
  //
  // The dependencies are stubbed rather than shimmed because the subject is the
  // STATE MACHINE: which of the four answers each combination produces.
  const ownedDriver = (identity: string, listener: string) =>
    [
      `has_owned_command_identity() { ${identity} }`,
      `probe_listener_port() { ${listener} }`,
      "OWNED_PID=4242",
      "OWNED_PORT=17931",
      "rc=0",
      "is_owned_process || rc=$?",
      `printf 'rc=%s\\n' "$rc"`,
    ].join("\n");
  const owned = (identity: string, listener: string) =>
    drive(ownedDriver(identity, listener), ["is_owned_process"]);

  const LISTENER_OURS = 'LISTENER_PROBE_STATE=found; LISTENER_PROBE_PID=4242;';
  const LISTENER_OTHER = 'LISTENER_PROBE_STATE=found; LISTENER_PROBE_PID=9999;';
  const LISTENER_NONE = 'LISTENER_PROBE_STATE=none; LISTENER_PROBE_PID="";';
  const LISTENER_UNMEASURED = 'LISTENER_PROBE_STATE=unmeasured; LISTENER_PROBE_PID="";';

  it.each([
    ["ours and serving the port", "return 0;", LISTENER_OURS, "rc=0"],
    // The two that used to share a 1. The first is the stale record; the second
    // is the live daemon whose record the first one's handling deleted.
    ["gone or a different image", "return 1;", LISTENER_NONE, "rc=1"],
    ["alive, ours, and not listening yet", "return 0;", LISTENER_NONE, "rc=3"],
    ["alive, ours, and the port held by another pid", "return 0;", LISTENER_OTHER, "rc=3"],
    ["identity unmeasurable", "return 2;", LISTENER_NONE, "rc=2"],
    ["listener unmeasurable", "return 0;", LISTENER_UNMEASURED, "rc=2"],
  ])("answers %s with %s", (_name, identity, listener, want) => {
    const run = owned(identity, listener);
    expect(run.stdout.trim(), run.stderr).toBe(want);
  }, 60_000);

  // And the consumer half, read from the source rather than run: reaching
  // `start_runtime`'s fall-through needs a cargo build and a daemon, which no
  // case here can pay for. What it pins is the ORDER -- the 3 has to be answered
  // before the deletion, because being answered afterwards is the same as not
  // being answered at all. It does not prove the message or the marker; the
  // state machine above is what proves the 3 exists to be branched on.
  it("start_runtime refuses on the live-but-not-listening state before clearing anything", () => {
    const text = readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8");
    const start = text.indexOf("start_runtime() {");
    expect(start).toBeGreaterThan(-1);
    const body = text.slice(start);
    const probe = body.indexOf("is_owned_process || owned_rc=$?");
    const guard = body.indexOf("(( owned_rc == 3 ))");
    const clear = body.indexOf("clear_owned_state");
    expect(probe, "start_runtime no longer consults is_owned_process").toBeGreaterThan(-1);
    expect(guard, "start_runtime does not branch on the live-daemon state").toBeGreaterThan(
      probe,
    );
    expect(
      clear,
      "start_runtime clears the ownership record before it has ruled out a live daemon",
    ).toBeGreaterThan(guard);
  });
});
