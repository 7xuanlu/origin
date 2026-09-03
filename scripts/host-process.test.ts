// @vitest-environment node
//
// These cases spawn processes and read files; nothing here touches a DOM, and
// building jsdom for each file costs more than every assertion in it.
import {
  chmodSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { branchesOnUnmeasured, probeCallSites } from "./lib/probe-call-sites";
import { resolveTestBash } from "./lib/test-bash";

// Resolved once for the file rather than per case: on Windows this is Git Bash,
// never the WSL `bash` that PATH offers first.
const TEST_BASH = resolveTestBash();

// Every case spawns node, then Git Bash, then a shim or two: a few hundred
// milliseconds idle, seconds under load. At the 5s default the assertion being
// reported would be about the HOST, not the library. Cases that poll or drive
// twice raise it further at their own site.
vi.setConfig({ testTimeout: 30_000 });

// scripts/lib/host-process.sh carries ONE invariant, and it is the reason this
// file exists:
//
//   A FAILED MEASUREMENT MUST NEVER BE INDISTINGUISHABLE FROM A NEGATIVE
//   MEASUREMENT.
//
// Every helper is tri-state — measured / negative / could not measure — so the
// interesting case is always the THIRD one: a suite covering only "found" and
// "not found" is how the two-state versions survived, where a missing tool
// produced empty output and empty output read as "port free" and as "process
// dead". Every helper is exercised in all three states, and the wrappers are
// additionally checked for the inverse mistake, a genuine negative reported as
// unmeasured.
//
// The Windows branches run on every platform: a POSIX host opts INTO them with
// WENLAN_HOST_PROCESS_PLATFORM=windows (one-way — a real Windows host can never
// be talked out of its identity-checked paths) and the tools they call are
// shimmed on PATH. Without that none of this would ever run, because `pnpm
// test` only runs in the macOS `app-check` lane.

const root = resolve(import.meta.dirname, "..");
// WENLAN_HOST_PROCESS_LIB names the library under test, defaulting to the
// shipped one, so the negative-control harness
// (scripts/negative-controls/posix-probes-negative-controls.py) can revert a
// fix and re-run this suite without patching the shipped file in place, which
// any concurrently running suite would then read.
//
// The override is locked to the exact BYTES the harness meant to test.
// WENLAN_HOST_PROCESS_LIB_CONTROL is the sha256 of the override file's
// contents: only something that has just written that file can know it, and a
// stale value stops matching the moment the file changes or is deleted. A
// boolean flag would not do — two strings inherited from a shell, or exported
// by a wrapper that outlived the harness, aim an ordinary `pnpm test` at a
// valid old copy while the shipped library is broken. For the same reason the
// identity test below ASSERTS rather than returning early when the flag is
// set: a row that switches itself off is the row that stops watching.
const shippedLibPath = resolve(root, "scripts/lib/host-process.sh");
const libOverride = process.env.WENLAN_HOST_PROCESS_LIB;
const libOverrideDigest = process.env.WENLAN_HOST_PROCESS_LIB_CONTROL ?? "";
const sha256 = (text: string) => createHash("sha256").update(text, "utf8").digest("hex");
const overrideText = libOverride ? readFileSync(resolve(libOverride), "utf8") : "";
if (libOverride && sha256(overrideText) !== libOverrideDigest) {
  throw new Error(
    `WENLAN_HOST_PROCESS_LIB is set to ${libOverride} but ` +
      "WENLAN_HOST_PROCESS_LIB_CONTROL is not the sha256 of its contents " +
      `(file hashes ${sha256(overrideText)}, flag says ${libOverrideDigest || "nothing"}). ` +
      "Refusing to test a library other than the shipped one by accident: a " +
      "green run against a stale copy proves nothing about what ships. Unset " +
      "both, or have the negative-control harness hash the file it just wrote.",
  );
}
// The digest is a claim about CONTENT, not about AUTHORSHIP: a harness that
// hashes the file by re-reading it hashes whatever is at that path, so a write
// that silently did not happen hashes the copy the LAST run left there, the
// flag matches, and this suite runs the previous mutation while reporting on
// the current one. Two stale things agreeing is not a measurement.
//
// Anything reasoning from TIME has a window and the window is the hole — mtime
// is stored to two seconds on some filesystems, so an age rule must allow two
// seconds of slack and a leftover from one second earlier passes. Authorship is
// established by a secret instead: the harness generates a nonce, writes it
// into the bytes and names it in the environment, and the override must carry
// exactly that nonce on a line of its own. A leftover from an earlier write
// carries an earlier nonce and is refused however old or new it is.
//
// The line is control scaffolding rather than library text, so every comparison
// against the shipped library strips it first.
const OVERRIDE_NONCE_LINE = /^# wenlan-control-nonce: ([0-9a-f]{32,})$/gm;
const withoutNonce = (text: string) => text.replace(OVERRIDE_NONCE_LINE, "").trimEnd();
//
// Not applied when the override IS the shipped library: the shipped file
// carries no nonce and never should, and the identity row below has the
// specific refusal for that case. Pre-empting it here would answer "stale copy"
// to a question about the wrong file entirely.
const overrideNonce = process.env.WENLAN_HOST_PROCESS_LIB_NONCE ?? "";
if (libOverride && resolve(libOverride) !== shippedLibPath) {
  if (!/^[0-9a-f]{32,}$/.test(overrideNonce)) {
    throw new Error(
      `WENLAN_HOST_PROCESS_LIB is set to ${libOverride} but ` +
        "WENLAN_HOST_PROCESS_LIB_NONCE is not a nonce this run generated " +
        `(it says ${JSON.stringify(process.env.WENLAN_HOST_PROCESS_LIB_NONCE ?? null)}). ` +
        "The digest proves the bytes; only this proves THIS run put them there. " +
        "Without it a harness whose write silently failed hashes the copy a previous " +
        "run left at the path, gets a flag that matches, and this suite reports on a " +
        "mutation it never loaded.",
    );
  }
  const carried = [...overrideText.matchAll(OVERRIDE_NONCE_LINE)].map((m) => m[1]);
  if (carried.length !== 1 || carried[0] !== overrideNonce) {
    throw new Error(
      `WENLAN_HOST_PROCESS_LIB is set to ${libOverride}, whose contents match ` +
        "WENLAN_HOST_PROCESS_LIB_CONTROL, but the file carries " +
        `${carried.length === 0 ? "no nonce" : JSON.stringify(carried)} and this run ` +
        `declared ${JSON.stringify(overrideNonce)} — so it is a copy left behind by an ` +
        "earlier write, and the digest cannot tell that apart because a harness whose " +
        "write failed hashes the stale file and gets a flag that matches. Re-run the " +
        "harness so the mutation under test is the one on disk.",
    );
  }
}
const libPath = libOverride ? resolve(libOverride) : shippedLibPath;
const tempRoots: string[] = [];

afterEach(() => {
  for (const path of tempRoots.splice(0)) {
    rmSync(path, { recursive: true, force: true });
  }
});

function makeTempRoot(): string {
  const dir = mkdtempSync(resolve(tmpdir(), "wenlan-host-process-"));
  tempRoots.push(dir);
  return dir;
}

// The driver sources the library and prints one machine-readable line per call,
// so the test asserts on the helper's own status and state rather than on any
// side effect.
// PATH is re-prepended INSIDE the shell rather than by the caller: Git for
// Windows' bash.exe puts /mingw64/bin and /usr/bin at the front before the
// script runs, so a shim dir handed in from outside loses to /usr/bin/ps — but
// wins for netstat and tasklist, which live nowhere on that path. That
// asymmetry silently exercises the real `ps` against a stub's expectations.
const DRIVER = `#!/usr/bin/env bash
set -uo pipefail
if [ -n "\${WENLAN_TEST_SHIM_DIR:-}" ]; then
  shim_dir="$WENLAN_TEST_SHIM_DIR"
  # A Windows-spelled directory is not a usable PATH entry inside MSYS.
  if command -v cygpath >/dev/null 2>&1; then shim_dir="$(cygpath -u "$shim_dir")"; fi
  PATH="$shim_dir:$PATH"
  export PATH
fi
# PATH is REPLACED, not prepended, for the "this tool does not exist" cases:
# \`netstat\` and \`tasklist\` are always on PATH on Windows, so skipping the case
# there leaves it with no assertion in it, counted among the passes. cygpath is
# resolved above, before this.
if [ "\${WENLAN_TEST_ISOLATE_PATH:-}" = "1" ]; then
  PATH="\${shim_dir:-}"
  export PATH
fi
# The source is CHECKED. The library refuses to define anything when it cannot
# tell which platform this is, and a driver that read on past that would run
# every probe below through the POSIX branch — which is the exact fail-open the
# refusal exists to prevent, reproduced in the harness that tests it.
# shellcheck source=/dev/null
if ! . "$WENLAN_HOST_PROCESS_LIB"; then
  printf 'source-refused\\n'
  exit 3
fi
# The library's own override is one-way on purpose: a real Windows host can
# never be talked out of its identity-checked kill path. That leaves the POSIX
# listener branch unrunnable on Windows, and it is the branch that runs in CI on
# ubuntu and macos, so it is the one that most needs covering. Forcing it HERE,
# in the test driver, exercises it on every host without adding a way for a real
# caller to reach the unsafe direction.
if [ "\${WENLAN_TEST_POSIX_BRANCH:-}" = "1" ]; then HOST_IS_WINDOWS=0; fi
op="$1"; arg="\${2:-}"; arg2="\${3:-}"
rc=0
case "$op" in
  listener) out="$(listener_pid_for_port "$arg")" || rc=$?; printf 'rc=%s out=%s\\n' "$rc" "$out" ;;
  alive)    process_is_alive "$arg" || rc=$?;               printf 'rc=%s\\n' "$rc" ;;
  image)    out="$(process_image_path "$arg")" || rc=$?;    printf 'rc=%s out=%s\\n' "$rc" "$out" ;;
  jobpid)   out="$(windows_pid_for_job "$arg" "$arg2")" || rc=$?; printf 'rc=%s out=%s\\n' "$rc" "$out" ;;
  probe-listener) probe_listener_port "$arg"; printf 'state=%s pid=%s\\n' "$LISTENER_PROBE_STATE" "$LISTENER_PROBE_PID" ;;
  probe-alive)    probe_process_alive "$arg"; printf 'state=%s\\n' "$PROCESS_ALIVE_STATE" ;;
  probe-image)    probe_process_image "$arg"; printf 'state=%s value=%s\\n' "$PROCESS_IMAGE_STATE" "$PROCESS_IMAGE_VALUE" ;;
  *) echo "unknown op $op" >&2; exit 99 ;;
esac
`;

type ShimSpec = { name: string; body: string };

function writeShims(dir: string, shims: ShimSpec[]): string {
  const shimDir = resolve(dir, "shim");
  mkdirSync(shimDir, { recursive: true });
  for (const shim of shims) {
    const path = resolve(shimDir, shim.name);
    writeFileSync(path, shim.body, { mode: 0o755 });
    chmodSync(path, 0o755);
  }
  return shimDir;
}

function runDriver(
  args: string[],
  options: {
    shims?: ShimSpec[];
    forceWindows?: boolean;
    posixBranch?: boolean;
    /** Replace PATH with the shim directory, so an absent tool is really absent. */
    isolatePath?: boolean;
    env?: Record<string, string>;
  } = {},
): { status: number | null; stdout: string; stderr: string } {
  const dir = makeTempRoot();
  const driverPath = resolve(dir, "driver.sh");
  writeFileSync(driverPath, DRIVER, { mode: 0o755 });

  const env: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined) env[key] = value;
  }
  env.WENLAN_HOST_PROCESS_LIB = libPath;
  if (options.posixBranch) {
    env.WENLAN_TEST_POSIX_BRANCH = "1";
  } else if (options.forceWindows !== false) {
    env.WENLAN_HOST_PROCESS_PLATFORM = "windows";
  }
  if (options.shims?.length || options.isolatePath) {
    const shimDir = writeShims(dir, options.shims ?? []);
    env.WENLAN_TEST_SHIM_DIR = shimDir;
    env.PATH = `${shimDir}${delimiter}${env.PATH ?? ""}`;
  }
  if (options.isolatePath) env.WENLAN_TEST_ISOLATE_PATH = "1";
  for (const [key, value] of Object.entries(options.env ?? {})) env[key] = value;

  // TEST_BASH is Git Bash on Windows, resolved at module scope by the same
  // candidate search scripts/run-bash.mjs uses, so a machine with WSL installed
  // never runs these fixtures on the Linux distro PATH offers first.
  const result = spawnSync(TEST_BASH, [driverPath, ...args], {
    cwd: root,
    encoding: "utf8",
    env,
  });

  return { status: result.status, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
}

const shim = (name: string, body: string): ShimSpec => ({
  name,
  body: `#!/usr/bin/env bash\n${body}\n`,
});

// One `printf` argument per line — an embedded `\n` inside a single `%s`
// argument is not interpreted, which would collapse the fixture to one line and
// quietly turn every "found" case into a "none".
const emitLines = (lines: string[]) =>
  `printf '%s\\n' ${lines.map((line) => `'${line}'`).join(" ")}`;

// `netstat -ano` prints the whole TCP table and then the whole UDP table, so a
// UDP row is the witness that the TCP section ENDED rather than stopped —
// measured on the host this was written on: 198 lines, TCP rows 5..97, UDP rows
// 98..198, and no header between them. The probe requires one, which is what
// lets it tell a complete table from one truncated after a well-formed row, so
// every fixture that stands for a WHOLE table carries one. Drop it and the
// fixture stands for a truncated table instead, which is a different case (see
// "a table that stopped inside the TCP section" below).
const udpRow = "  UDP    0.0.0.0:5353           *:*                                    9528";

// A netstat table with the header Windows really prints, so the parser is
// exercised against the shape it will meet.
const netstatTable = (rows: string[]) =>
  emitLines([
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    ...rows,
    udpRow,
  ]);

const listeningRow = (port: number, pid: string) =>
  `  TCP    127.0.0.1:${port}         0.0.0.0:0              LISTENING       ${pid}`;

// The same row as a localised Windows would print it. netstat's State column is
// translated; `TCP` and the wildcard foreign address are not. This is the row
// that made a busy port read as free.
const localisedListeningRow = (port: number, pid: string) =>
  `  TCP    127.0.0.1:${port}         0.0.0.0:0              ABHOEREN        ${pid}`;

// `tasklist //NH //FO CSV` prints one CSV row per process — 268 of them on the
// machine this was measured on. The library asks for the WHOLE table because a
// filtered query cannot tell a real miss from a broken tool: measured here,
// both print no CSV row and exit 0. So a fixture that is a table has to look
// like one, and these two rows are the "everything else on the machine" part.
const tasklistRow = (image: string, pid: string) =>
  `"${image}","${pid}","Console","1","9,000 K"`;

// The probe asks for three things and this fixture supplies all three: every
// line a CSV row, the System process (pid 4, present on every Windows NT
// kernel), and at least ten rows. A two-row fixture is the well-formed fragment
// the probe is supposed to refuse, so it cannot stand for a whole table. These
// are the "everything else on the machine" part; the real table has 268.
const TASKLIST_BACKGROUND = [
  tasklistRow("System", "4"),
  tasklistRow("Registry", "132"),
  tasklistRow("smss.exe", "608"),
  tasklistRow("csrss.exe", "888"),
  tasklistRow("wininit.exe", "964"),
  tasklistRow("services.exe", "1048"),
  tasklistRow("lsass.exe", "1072"),
  tasklistRow("svchost.exe", "1576"),
  tasklistRow("svchost.exe", "1704"),
  tasklistRow("explorer.exe", "5320"),
  tasklistRow("bash.exe", "9012"),
];
const tasklistTable = (rows: string[]) => emitLines([...TASKLIST_BACKGROUND, ...rows]);

// `ps -W` aligns COMMAND at a fixed offset and prints STIME as either "10:23:45"
// (one field) or "Aug 27" (two). The header offset is the only parse that reads
// both, which is why the library stopped counting fields from $8.
const psHeader = "      PID    PPID    PGID     WINPID   TTY         UID    STIME COMMAND";
const psRow = (pid: string, winpid: string, stime: string, command: string) =>
  `${pid.padStart(9)}${"0".padStart(8)}${"0".padStart(8)}${winpid.padStart(11)}  ?${" ".repeat(14)}0${stime.padStart(9)} ${command}`;

// `process_image_path` asks `ps -W` for the same two completeness witnesses
// `process_is_alive` asks of `tasklist`: WINPID 4, the System process, which
// exists on every Windows NT kernel from boot to shutdown, and a floor of ten
// rows. A header plus ONE row is the fragment that fix forbids, so it cannot
// stand for a whole table here either. These rows are the "everything else on
// the machine" part; the real table has 277 of them.
//
// A `ps -W` row is PID PPID PGID WINPID, four numbers, and the library's row
// shape reads only those, so the MSYS pid here is arbitrary and the WINPID is
// what the parse keys on. None of these commands may contain "wenlan-server":
// the same background feeds the scan in dev-runtime.sh's harness, where a
// match is what gets killed.
const PS_BACKGROUND = [
  psRow("4", "4", "Aug 27", "System"),
  psRow("140", "140", "Aug 27", "C:\\Windows\\System32\\smss.exe"),
  psRow("680", "680", "Aug 27", "C:\\Windows\\System32\\csrss.exe"),
  psRow("708", "708", "Aug 27", "C:\\Windows\\System32\\wininit.exe"),
  psRow("796", "796", "Aug 27", "C:\\Windows\\System32\\services.exe"),
  psRow("872", "872", "Aug 27", "C:\\Windows\\System32\\lsass.exe"),
  psRow("920", "920", "Aug 27", "C:\\Windows\\System32\\svchost.exe"),
  psRow("1704", "1704", "Aug 27", "C:\\Windows\\System32\\svchost.exe"),
  psRow("5320", "5320", "Aug 27", "C:\\Windows\\explorer.exe"),
  psRow("9012", "9012", "10:23:45", "C:\\Program Files\\Git\\usr\\bin\\bash.exe"),
  psRow("9104", "9104", "10:23:45", "C:\\Windows\\System32\\conhost.exe"),
];
// The header, the rest of the machine, then whatever the case is about.
const psTable = (rows: string[], header: string = psHeader) =>
  emitLines([header, ...PS_BACKGROUND, ...rows]);
// Ten rows, none of them WINPID 4. Long enough to clear the floor, and still
// not a Windows process table: nothing runs without the System process.
const psTableNoSystem = (rows: string[]) =>
  emitLines([psHeader, ...PS_BACKGROUND.slice(1), ...rows]);
// Three rows, WINPID 4 among them. Well-formed, carries the System process,
// and is not this machine — a Windows session with a login shell in it cannot
// have three processes.
const psTableTooShort = (rows: string[]) =>
  emitLines([psHeader, PS_BACKGROUND[0], PS_BACKGROUND[6], ...rows]);

describe("host-process.sh: the suite is reading the shipped library", () => {
  // Every other assertion in this file is about a library whose path is decided
  // above. If that decision is wrong, all of them are green about the wrong
  // file — the failure mode is silent and total, so it gets its own row in the
  // ledger rather than living in a comment.
  it("tests scripts/lib/host-process.sh unless a control explicitly says otherwise", () => {
    const shipped = readFileSync(shippedLibPath, "utf8");
    if (!libOverride) {
      expect(libPath).toBe(shippedLibPath);
      expect(shipped).toContain("listener_pid_for_port");
      return;
    }
    // An override is only legitimate as a DELIBERATE MUTATION of the shipped
    // library, written moments ago by the harness whose digest is in the flag.
    // Asserted, never returned past: an override that is the shipped path, or
    // that is byte-identical to it, or whose bytes have moved on from the
    // digest, is a suite testing something nobody chose.
    const actual = readFileSync(libPath, "utf8");
    expect(sha256(actual), "the override's bytes changed after the flag was set").toBe(
      libOverrideDigest,
    );
    expect(libPath, "a control must be a copy, never the shipped file").not.toBe(
      shippedLibPath,
    );
    // The nonce line is scaffolding, not library text, so the comparison that
    // asks "is this actually a mutation?" has to strip it — otherwise every
    // copy differs from the shipped file by construction and this row is
    // satisfied by a control that mutated nothing.
    expect(
      withoutNonce(actual),
      "an override identical to the shipped library is not a control",
    ).not.toBe(withoutNonce(shipped));
    expect(actual, "the override is not a copy of this library at all").toContain(
      "listener_pid_for_port",
    );
    // Written by THIS run, not merely consistent with it. The digest is a claim
    // about bytes and two stale things can agree about bytes indefinitely; no
    // window over mtime separates them, because mtime granularity is seconds.
    // The nonce is unguessable, so carrying it is authorship, not recency.
    const carried = [...actual.matchAll(OVERRIDE_NONCE_LINE)].map((m) => m[1]);
    expect(carried, "the override does not carry exactly one control nonce").toHaveLength(
      1,
    );
    expect(
      carried[0],
      "the override carries a nonce from an earlier write, not this one",
    ).toBe(overrideNonce);
  });
});

describe("host-process.sh: listener_pid_for_port is tri-state", () => {
  it("reports the listener pid when the port is held", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [shim("netstat", netstatTable([listeningRow(17931, "4242")]))],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
  });

  it("reports a NEGATIVE, not a failure, when nothing is listening", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [shim("netstat", netstatTable([listeningRow(17999, "4242")]))],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1 out=");
  });

  // THE case. A probe that cannot run must never be indistinguishable from
  // "port free" — that is what the old `command -v lsof || fail` gate existed
  // to prevent, and collapsing it is how a port-conflict bug ships.
  // An argument that is not a port reached the ledger as a measured negative:
  // `port-precheck.sh notaport` wrote `PASS ... measured free` and exited 0
  // about a port nothing looked at. The probe's three states were all intact;
  // the input was never checked to be a question it could answer.
  it.each([
    ["a word", "notaport"],
    ["empty", ""],
    ["zero", "0"],
    ["above the 16-bit range", "65536"],
    ["negative", "-1"],
    ["a port with trailing text", "17931x"],
    ["a float", "17931.0"],
    // The first cut of the guard accepted these two and then looked the port
    // up with the raw argument, so a BUSY port came back "measured free":
    //     135 -> rc=0 found pid 1576, 0000000135 -> rc=1 measured free.
    // A normalisation applied to the range check and not to the lookup.
    ["a zero-padded port", "0000000135"],
    ["a single leading zero", "017931"],
    // Long enough to overflow 64-bit arithmetic, which is what the `10#`
    // normalisation this replaced would have had to survive.
    ["a 23-digit number", "99999999999999999999999"],
    ["hex", "0x50"],
    ["scientific notation", "1e3"],
    ["a leading space", " 17931"],
    ["a trailing space", "17931 "],
  ])("reports COULD NOT MEASURE for %s, never a free port", (_label, arg) => {
    const result = runDriver(["listener", arg], {
      // A table in which the port genuinely IS free, so "rc=1" would be the
      // answer if the argument were merely absent from it. The point is that
      // the probe must refuse the question rather than answer it.
      shims: [shim("netstat", netstatTable([listeningRow(17999, "4242")]))],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // Not "it returns 2" but "it does not answer FREE about a port that is held":
  // a range check that normalises and a lookup that does not answer rc=1 for
  // the padded spelling of a port this table says is busy.
  it("never reports a HELD port as free, however the port is spelled", () => {
    for (const arg of ["0000000135", "000135", "0135"]) {
      const result = runDriver(["listener", arg], {
        shims: [shim("netstat", netstatTable([listeningRow(135, "1576")]))],
      });
      expect(result.stdout.trim(), `${arg}: ${result.stderr}`).not.toBe("rc=1 out=");
      expect(result.stdout.trim(), `${arg}: ${result.stderr}`).toBe("rc=2 out=");
    }
    // The control: the same table, the plain spelling, genuinely found.
    const plain = runDriver(["listener", "135"], {
      shims: [shim("netstat", netstatTable([listeningRow(135, "1576")]))],
    });
    expect(plain.stdout.trim(), plain.stderr).toBe("rc=0 out=1576");
  });

  it("still measures a well-formed port at both ends of the range", () => {
    for (const port of ["1", "65535"]) {
      const result = runDriver(["listener", port], {
        shims: [shim("netstat", netstatTable([listeningRow(Number(port), "4242")]))],
      });
      expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
    }
  });

  it.each([
    ["reports COULD NOT MEASURE when the listener table command fails", "exit 1"],
    ["reports COULD NOT MEASURE when the listener table comes back empty", "exit 0"],
  ])("%s", (_title, body) => {
    const result = runDriver(["listener", "17931"], { shims: [shim("netstat", body)] });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when the table is not the table we think it is", () => {
    // A non-numeric pid means the columns moved. Printing it as a pid, or
    // treating the miss as "no listener", would both be lies.
    const result = runDriver(["listener", "17931"], {
      shims: [
        shim("netstat", netstatTable([listeningRow(17931, "not-a-pid")])),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // A schema gate of "any line whose first token is TCP or UDP" is satisfied by
  // `TCP diagnostic text`, and the numeric-pid check downstream never sees it
  // because no row matched the port at all — so output that is not a table
  // passes the gate and the miss comes back as a measured negative.
  it("reports COULD NOT MEASURE for output that starts with TCP but is not a table", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [
        shim(
          "netstat",
          emitLines(["TCP diagnostic text", "UDP another line of it", "TCP and a third"]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // The gate cannot demand the word LISTENING: it is localised, so a German
  // Windows reads every busy port as free. `TCP` and the wildcard foreign
  // address are not localised, and measured against this host's real table the
  // wildcard-foreign shape selects exactly the 34 LISTENING rows and no others.
  it("finds the listener in a table whose State column is translated", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [shim("netstat", netstatTable([localisedListeningRow(17931, "4242")]))],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
  });

  it("still reports a NEGATIVE for a translated table without this port", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [shim("netstat", netstatTable([localisedListeningRow(17999, "4242")]))],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1 out=");
  });

  // A rule that only rejects a malformed line whose FIRST TOKEN is a protocol
  // name does not see a status-0 `WARNING: partial results` or `Access denied`
  // merged by the `2>&1` beside perfectly valid rows: the validator passes on
  // the valid rows, the port query finds nothing in the truncated remainder,
  // and the function returns MEASURED FREE about a table netstat complained
  // about — which is what deletes an ownership record and starts a second
  // daemon on an occupied port.
  //
  // The port asked for is deliberately absent from the rows that ARE there, so
  // this case is exactly the one that would come back "free".
  it("reports COULD NOT MEASURE for a diagnostic merged after valid rows", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [
        shim(
          "netstat",
          netstatTable([
            listeningRow(17999, "4242"),
            listeningRow(18000, "4243"),
            "WARNING: partial results",
          ]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // The same diagnostic, landing BEFORE the first row instead. `2>&1` does not
  // order the two streams, so the rule cannot assume where a warning appears.
  // The real preamble is exactly two lines — the `Active Connections` banner
  // and the column header — and both are localised, so they can only be
  // counted, never matched by text. One merged diagnostic makes it three.
  it("reports COULD NOT MEASURE for a diagnostic merged into the preamble", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [
        shim(
          "netstat",
          emitLines([
            "",
            "Active Connections",
            "",
            "  Proto  Local Address          Foreign Address        State           PID",
            "Access denied",
            listeningRow(17999, "4242"),
            // The UDP witness belongs here even though this table is built by
            // hand: without it the table is ALSO truncated, and the truncation
            // rule refuses it whatever the preamble rule does. The negative
            // control that reverts the preamble rule then finds this case still
            // green and reports that it is measuring nothing -- which is
            // exactly what `nc-netstat-preamble-uncounted` did the first time
            // this suite ran after the witness landed. One fixture must carry
            // one defect.
            udpRow,
          ]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // Fail-closed in the other direction too: a contaminated table whose rows DO
  // contain the port is still unmeasurable. Being able to find one answer in a
  // stream the tool complained about is not evidence the rest of it arrived,
  // and the caller's question ("is this port free") is answered by the ABSENCE
  // of rows, which is precisely what a truncated table cannot support.
  it("refuses a contaminated table even when the port IS in it", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [
        shim(
          "netstat",
          netstatTable([listeningRow(17931, "4242"), "WARNING: partial results"]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // And the control on the two rules above: the shape they were measured
  // against still parses. A blank line inside the row block is not a
  // diagnostic (the real table has two), and the two-line preamble is exactly
  // the budget, not one under it — so this must still be a measured answer,
  // not a refusal. Without this, "refuse everything" would pass the three
  // cases above.
  it("still parses the real table shape, blank lines and all", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [
        shim(
          "netstat",
          emitLines([
            "",
            "Active Connections",
            "",
            "  Proto  Local Address          Foreign Address        State           PID",
            listeningRow(17931, "4242"),
            "",
            listeningRow(18000, "4243"),
            udpRow,
          ]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
  });

  // The two rules above are a GRAMMAR, and a grammar cannot see a table cut
  // after a well-formed row: every surviving row validates, the port query
  // finds nothing in what arrived, and the answer is MEASURED FREE from a table
  // that stopped early. The witness is the UDP section — netstat prints all of
  // TCP and then all of UDP, so a UDP row means the stream got past every TCP
  // row there was. This fixture is the real shape with that section missing:
  // well-formed, and not whole.
  it("reports COULD NOT MEASURE for a table that stopped inside the TCP section", () => {
    const result = runDriver(["listener", "17931"], {
      shims: [
        shim(
          "netstat",
          emitLines([
            "",
            "Active Connections",
            "",
            "  Proto  Local Address          Foreign Address        State           PID",
            listeningRow(17999, "4242"),
            listeningRow(18000, "4243"),
          ]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when no listener table command exists at all", () => {
    // PATH is emptied rather than the case skipped on Windows, where `netstat`
    // is always present: a skip on the one platform this branch exists for is a
    // test with no assertion in it. Absence must be 2; a 1 here would mean "no
    // listener", which is the defect.
    const result = runDriver(["listener", "17931"], { isolatePath: true });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });
});

// THE PLATFORM ITSELF, which every branch in this file rests on.
//
// `case "$(uname -s)" in … *) HOST_IS_WINDOWS=0` has nowhere to put a status: a
// `uname` that is missing, or that fails, or that prints nothing, yields the
// empty string, the empty string matches no Windows pattern, and it falls into
// the POSIX arm. On a Windows host that is `lsof` (which is not there), a POSIX
// `kill` (which cannot reach a native process by its Windows pid) and the
// UNSTAGED server path -- three wrong measurements, every one of them in the
// fail-open direction, from one dropped status.
describe("host-process.sh: the platform is measured, not assumed", () => {
  const brokenUname = (body: string) =>
    runDriver(["listener", "17931"], { forceWindows: false, shims: [shim("uname", body)] });

  it("refuses to define anything when uname fails", () => {
    const result = brokenUname("exit 127");
    expect(result.stdout.trim(), result.stderr).toBe("source-refused");
    expect(result.stderr).toContain("could not measure which platform this is");
  });

  it("refuses to define anything when uname exits 0 saying nothing", () => {
    // Status 0 and silence is the shape that used to be indistinguishable from
    // "this is Linux": the empty string falls into the catch-all arm.
    const result = brokenUname("exit 0");
    expect(result.stdout.trim(), result.stderr).toBe("source-refused");
  });

  it("still loads when uname is broken and the platform is declared", () => {
    // The refusal must not be reachable when somebody HAS measured the
    // platform: the test override is a declaration, which is exactly what the
    // empty string is not. Without this the case above would pass for a library
    // that refuses unconditionally.
    const result = runDriver(["listener", "17931"], {
      shims: [shim("uname", "exit 127")],
    });
    expect(result.stdout.trim(), result.stderr).not.toBe("source-refused");
    // It answered the probe. WHICH of the three answers depends on whether this
    // host has a `netstat` -- on Windows it does and says nothing is on 17931,
    // on a POSIX runner it does not and the answer is 2 -- and the claim here is
    // only that the library loaded and measured something.
    expect(result.stdout.trim(), result.stderr).toMatch(/^rc=[0-2] out=/);
  });

  it("names a real host when uname works", () => {
    const result = runDriver(["listener", "17931"], { forceWindows: false });
    expect(result.stdout.trim(), result.stderr).not.toBe("source-refused");
  });
});

// The POSIX branch is the one that runs on the ubuntu and macos CI lanes, so it
// needs behavioural coverage of its own: every case above forces the Windows
// branch.
//
// lsof returns 1 for "nothing matched" AND for "an error was detected", so the
// status alone cannot separate them. With -t the only thing on stdout is pids,
// so stderr merged in is the discriminator.
describe("host-process.sh: listener_pid_for_port, POSIX branch", () => {
  // TWO lsof calls, not one, and the shim has to tell them apart the way the
  // library does: the targeted read carries `-tiTCP:<port>`, and the witness
  // that ratifies its negative carries a bare `-tiTCP`. So the PORT COLON is
  // the discriminator — `-sTCP:LISTEN` is on both command lines, and matching
  // on `TCP:` alone would put every call down the same arm and make every
  // negative case below pass for a reason nobody chose.
  const lsofShim = (targeted: string, witness = `    printf '%s\\n' '999'`) =>
    shim(
      "lsof",
      [
        "case \"$*\" in",
        "  *iTCP:*)",
        targeted,
        "    ;;",
        "  *)",
        witness,
        "    ;;",
        "esac",
      ].join("\n"),
    );
  const posix = (targeted: string, witness?: string) =>
    runDriver(["listener", "17931"], {
      posixBranch: true,
      shims: [lsofShim(targeted, witness)],
    });

  it("reports the listener pid when lsof matches", () => {
    expect(posix(`    printf '%s\\n' '4242'`).stdout.trim()).toBe("rc=0 out=4242");
  });

  it("reports a NEGATIVE when lsof exits 1 with nothing to say and can still enumerate", () => {
    expect(posix("    exit 1").stdout.trim()).toBe("rc=1 out=");
  });

  // THE case `|| hit=""` gets wrong: lsof ran, lsof failed, answer "port free".
  //
  // What this shim does NOT prove on its own: it writes to stderr while the
  // library passes `-w`, which real lsof honours, so it cannot reproduce the
  // failure it is named after. It is kept because "text alongside status 1"
  // must stay unmeasured; the case BELOW is the real shape.
  it("reports COULD NOT MEASURE when lsof exits 1 with an error on stderr", () => {
    const result = posix(
      `    echo "lsof: status error on /proc/1/fd: Permission denied" >&2\n    exit 1`,
    );
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // THE REAL SHAPE, and the reason the case above cannot stand alone. `-w` is
  // in the command line specifically to SUPPRESS lsof's warnings, so an lsof
  // that hit an unreadable /proc, mount or device exits 1 having said NOTHING —
  // byte for byte the shape of "nothing matched", and read as "port free".
  //
  // It is separated from a real absence by a second read that covers the same
  // scan: lsof exits 0 only when it detected no error anywhere, so a status-0
  // enumeration of the listening set proves the scan ran. Here that second read
  // fails the same silent way, and the answer is 2.
  it("reports COULD NOT MEASURE when a silent lsof cannot ratify its own negative", () => {
    const result = posix("    exit 1", "    exit 1");
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // The witness has to be a MEASUREMENT and not a formality: an enumeration
  // that exits 0 while printing something that is not a pid is lsof telling us
  // this is not the tool we think it is, and it cannot ratify anything.
  it("reports COULD NOT MEASURE when the enumeration prints something that is not a pid", () => {
    const result = posix("    exit 1", `    printf '%s\\n' 'lsof: WARNING: /proc'`);
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when the enumeration exits 0 saying nothing", () => {
    const result = posix("    exit 1", "    exit 0");
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // And the witness must not be consulted on the paths that do not need it: a
  // FOUND listener is its own evidence, so a broken enumeration cannot turn a
  // measured pid into a refusal.
  it("still reports the listener pid when the enumeration is broken", () => {
    const result = posix(`    printf '%s\\n' '4242'`, "    exit 1");
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
  });

  it("reports COULD NOT MEASURE for any status other than 0 or 1", () => {
    expect(posix("    exit 3").stdout.trim()).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when lsof exits 0 saying nothing", () => {
    // A -t query that matched exits 0 AND prints a pid; one that did not exits
    // 1. Silence with 0 means this is not the lsof we think it is, so calling
    // it "port free" would be a guess dressed as a measurement.
    expect(posix("    exit 0").stdout.trim()).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when lsof prints something that is not a pid", () => {
    expect(posix(`    printf '%s\\n' 'wenlan-server'`).stdout.trim()).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when lsof is not installed", () => {
    // Git Bash has no lsof at all, which is why a `command -v lsof || fail`
    // gate stops the smokes dead on this platform.
    const result = runDriver(["listener", "17931"], { posixBranch: true });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("passes the failure through probe_listener_port as unmeasured, not none", () => {
    const result = runDriver(["probe-listener", "17931"], {
      posixBranch: true,
      shims: [lsofShim(`    echo "lsof: no pwd entry for UID 1000" >&2\n    exit 1`)],
    });
    expect(result.stdout.trim(), result.stderr).toBe("state=unmeasured pid=");
  });

  // And the ratified negative still arrives as `none` rather than `unmeasured`:
  // a fix that made every negative unmeasurable would refuse every start, which
  // is a correctness fix that breaks the ordinary case.
  it("passes a ratified negative through probe_listener_port as none", () => {
    const result = runDriver(["probe-listener", "17931"], {
      posixBranch: true,
      shims: [lsofShim("    exit 1")],
    });
    expect(result.stdout.trim(), result.stderr).toBe("state=none pid=");
  });
});

describe("host-process.sh: process_is_alive is tri-state", () => {
  it("reports alive for a pid the process table lists", () => {
    const result = runDriver(["alive", "4242"], {
      shims: [shim("tasklist", tasklistTable([tasklistRow("wenlan-server.exe", "4242")]))],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=0");
  });

  it("reports a NEGATIVE for a table that lists other processes but not this pid", () => {
    // THE negative: the table is demonstrably a table, and the pid is not in
    // it. That is the only shape from which absence can honestly be read.
    const result = runDriver(["alive", "4242"], {
      shims: [shim("tasklist", tasklistTable([]))],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1");
  });

  // `tasklist //FI "PID eq N"` exits 0 and prints this notice for an absent pid
  // — and exits 0 and prints one line of prose for anything else that goes
  // wrong, too. Under "anything not starting with a quote is absence" every
  // status-0 diagnostic is a dead process, and the caller's next move is to
  // delete a live daemon's ownership record.
  it("reports COULD NOT MEASURE for a status-0 line of prose, not a negative", () => {
    const notice = `printf '%s\\n' 'INFO: No tasks are running which match the specified criteria.'`;
    expect(runDriver(["alive", "4242"], { shims: [shim("tasklist", notice)] }).stdout.trim()).toBe(
      "rc=2",
    );
    const denied = `printf '%s\\n' 'ERROR: Access denied'`;
    expect(runDriver(["alive", "4242"], { shims: [shim("tasklist", denied)] }).stdout.trim()).toBe(
      "rc=2",
    );
  });

  // One fixture per rule the probe applies, each one caught by that rule alone:
  //   too short          one CSV row is a fragment, not a machine's process list
  //   pid-4 fragment     well-formed CSV carrying the System process, so only
  //                      the ten-row floor catches it
  //   no System process  full length and well-formed, so only pid 4 catches it
  //   merged warning     stderr is merged in, so a status-0 warning is a line
  //                      that is not a row; discarded, it would be invisible
  //                      beside a table missing the process being asked about
  //   exit 1             `tasklist … | grep -q '^"'` finds nothing and answers
  //                      "dead", after which the caller deletes the ownership
  //                      record of a running daemon
  //   exit 0             tasklist always says something, so silence is an
  //                      anomaly and not the "no tasks" notice
  it.each([
    [
      "reports COULD NOT MEASURE for a table too short to be a process table",
      () => emitLines([tasklistRow("System", "4")]),
    ],
    [
      "reports COULD NOT MEASURE for a well-formed fragment that contains pid 4",
      () => emitLines([tasklistRow("System", "4"), tasklistRow("svchost.exe", "1576")]),
    ],
    [
      "reports COULD NOT MEASURE for a full table with no System process in it",
      () => emitLines(TASKLIST_BACKGROUND.filter((row) => !row.includes('","4","'))),
    ],
    [
      "reports COULD NOT MEASURE for a table with a warning merged into it",
      () => `${tasklistTable([])}\necho 'WARNING: the RPC server is unavailable.' >&2`,
    ],
    ["reports COULD NOT MEASURE when the process table command fails", () => "exit 1"],
    ["reports COULD NOT MEASURE when the process table says nothing at all", () => "exit 0"],
  ])("%s", (_title, body) => {
    const result = runDriver(["alive", "4242"], { shims: [shim("tasklist", body())] });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  it("reports COULD NOT MEASURE when no process table command exists at all", () => {
    // `tasklist` is always on PATH on Windows, so skipping this case there
    // makes it an assertion-free pass on the one platform its branch covers.
    const result = runDriver(["alive", "4242"], { isolatePath: true });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  // A pid that is not a pid is a question that was never asked. Both branches
  // would otherwise answer it: tasklist prints the no-such-task notice for
  // `PID eq abc`, and `kill -0 abc` fails, and both of those read as "dead".
  it("reports COULD NOT MEASURE for an argument that is not a pid", () => {
    expect(runDriver(["alive", "not-a-pid"]).stdout.trim()).toBe("rc=2");
    expect(runDriver(["alive", ""]).stdout.trim()).toBe("rc=2");
  });
});

// `kill -0` fails with ESRCH for "no such process" and with EPERM for "there it
// is, but it is not yours", and the shell shows status 1 for both. A daemon
// running as another user therefore read as dead — and the caller's response to
// "dead" is to delete its ownership record, after which nothing can ever
// identity-check that process again.
//
// `kill` is a bash builtin and cannot be shimmed, so these drive the branch with
// a pid that certainly does not exist and shim the second witness, `ps`, which
// is what the fix consults whenever kill declines to confirm.
describe("host-process.sh: process_is_alive, POSIX branch", () => {
  // A pid inside every plausible pid_max (Linux defaults to 4194304) and not in
  // use. 4294967290 is PAST bash's own integer range: measured here, `kill -0`
  // answers "arguments must be process or job IDs", which is the question being
  // REJECTED, not answered — pinned below.
  const GONE = "999999";
  const posixAlive = (body: string, env?: Record<string, string>) =>
    runDriver(["alive", GONE], { posixBranch: true, shims: [shim("ps", body)], env });

  it("reports ALIVE when kill says no but the process table lists the pid", () => {
    // The EPERM case. This is the whole finding: `kill -0` alone called it dead.
    expect(posixAlive(`printf '%s\\n' ' ${GONE}'`).stdout.trim()).toBe("rc=0");
  });

  it("reports a NEGATIVE when the process table quietly has no such pid", () => {
    expect(posixAlive("exit 1").stdout.trim()).toBe("rc=1");
  });

  // Under Linux `hidepid=invisible` another user's process is absent from /proc
  // entirely, so `ps -p` is silent with status 1 — the exact shape of a real
  // absence — while `kill -0` fails with EPERM, which is the kernel confirming
  // the process EXISTS. Reading only kill's status discards that proof and
  // answers "dead", after which the caller deletes the ownership record.
  it("reports COULD NOT MEASURE for a silent ps when kill says EPERM, not ESRCH", () => {
    const result = posixAlive("exit 1", { WENLAN_HOST_PROCESS_FORCE_EPERM: "1" });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  // The same branch, driven with no override at all: a pid bash cannot even
  // parse is a question that was rejected. "Not ESRCH" is the only safe read.
  it("reports COULD NOT MEASURE for a pid past the shell's own integer range", () => {
    const result = runDriver(["alive", "4294967290"], {
      posixBranch: true,
      shims: [shim("ps", "exit 1")],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  it("reports COULD NOT MEASURE when the process table errors", () => {
    // A busybox `ps` without -p, a denied /proc: the question was not answered,
    // and answering "dead" would delete a live daemon's ownership record.
    const result = posixAlive(`echo "ps: unrecognized option: p" >&2\nexit 1`);
    expect(result.stdout.trim(), result.stderr).toBe("rc=2");
  });

  it("reports COULD NOT MEASURE when the process table prints something else", () => {
    expect(posixAlive(`printf '%s\\n' 'PID TTY'`).stdout.trim()).toBe("rc=2");
  });

  it("passes the failure through probe_process_alive as unmeasured, not dead", () => {
    const result = runDriver(["probe-alive", GONE], {
      posixBranch: true,
      shims: [shim("ps", `echo "ps: permission denied" >&2\nexit 1`)],
    });
    expect(result.stdout.trim(), result.stderr).toBe("state=unmeasured");
  });
});

describe("host-process.sh: process_image_path is tri-state", () => {
  it("reports the image path for a pid in the snapshot", () => {
    const result = runDriver(["image", "4242"], {
      shims: [
        shim(
          "ps",
          psTable([psRow("1000", "4242", "10:23:45", "C:\\wl-target\\debug\\wenlan-server.exe")]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe(
      "rc=0 out=C:\\wl-target\\debug\\wenlan-server.exe",
    );
  });

  // Measured on a real host: 214 of 246 `ps -W` rows carried a two-token STIME
  // ("Aug 27"), so a `for (i = 8; i <= NF; i++)` parse prefixes the image with a
  // stray day number. A daemon that outlives midnight then stops matching its
  // own recorded path and `stop` refuses to stop it — a WRONG NEGATIVE, the
  // same family of lie as an unmeasured one.
  it("reads the image column by offset, so a two-token STIME cannot corrupt it", () => {
    const result = runDriver(["image", "4242"], {
      shims: [
        shim(
          "ps",
          psTable([psRow("1000", "4242", "Aug 27", "C:\\wl-target\\debug\\wenlan-server.exe")]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe(
      "rc=0 out=C:\\wl-target\\debug\\wenlan-server.exe",
    );
  });

  it("reports a NEGATIVE when the snapshot has no such pid", () => {
    const result = runDriver(["image", "4242"], {
      shims: [
        shim("ps", psTable([psRow("1000", "9999", "10:23:45", "C:\\other.exe")])),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1 out=");
  });

  it.each([
    ["reports COULD NOT MEASURE when the snapshot command fails", "exit 1"],
    ["reports COULD NOT MEASURE when the snapshot comes back empty", "exit 0"],
  ])("%s", (_title, body) => {
    const result = runDriver(["image", "4242"], { shims: [shim("ps", body)] });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // The parse takes the image from the COMMAND column's offset and the pid from
  // WINPID's field index. Under a `col > 0 &&` guard a header carrying neither
  // skips every row, prints nothing, and falls through to the "no such pid"
  // return — a failed parse wearing a measured negative, for a pid that was in
  // the table all along.
  it("reports COULD NOT MEASURE when the snapshot header is not the one it parses", () => {
    const noCommand = psHeader.replace("COMMAND", "CMDLINE");
    const noWinpid = psHeader.replace("WINPID", "WPID  ");
    for (const header of [noCommand, noWinpid]) {
      const result = runDriver(["image", "4242"], {
        shims: [
          shim("ps", psTable([psRow("1000", "4242", "10:23:45", "C:\\wl.exe")], header)),
        ],
      });
      expect(result.stdout.trim(), `${header}\n${result.stderr}`).toBe("rc=2 out=");
    }
    // Two full driver spawns in one case. On a loaded Windows host each is a
    // second or two, so the 5s default measures the machine, not the parse.
  }, 60_000);

  // One case per completeness witness, both about the difference between "this
  // pid is not running" and "this is not the whole table". The pid asked about
  // is in NEITHER fixture, so a parse that accepts them answers rc=1 — a
  // measured negative — to a question nothing measured, and the caller acts on
  // it by deleting a live daemon's ownership record. Neither witness proves the
  // table is COMPLETE; that residual is stated in lib/host-process.sh and in
  // scripts/AGENTS.md.
  it.each([
    ["reports COULD NOT MEASURE for a ps -W table with no System process in it", psTableNoSystem],
    ["reports COULD NOT MEASURE for a ps -W table too short to be a process table", psTableTooShort],
  ])("%s", (_title, table) => {
    const result = runDriver(["image", "4242"], {
      shims: [
        shim(
          "ps",
          table([psRow("7100", "7100", "10:23:45", "C:\\Windows\\System32\\spoolsv.exe")]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });
});

// `windows_pid_for_job` asks `ps -W` the SAME question `process_image_path`
// asks it — which row is mine, and what is it running — keyed on the MSYS pid
// instead of the Windows one. A SECOND COPY of the parse is the defect: every
// hardening lands on one of them, and the other goes on counting words, so a
// header plus an eight-word warning or a short partial table spins the bounded
// poll for ten seconds and returns 1, "the process never appeared", for a table
// nothing managed to read. It fails SAFE — the caller refuses to start either
// way — and reports the WRONG STATE, which all three callers branch on.
//
// The pid asked for is ABSENT from every fixture below, which is what makes
// these cases able to fail: a parse that accepts the table answers "never
// appeared" rather than "could not measure", and the assertion is exactly that
// distinction.
describe("host-process.sh: windows_pid_for_job is tri-state", () => {
  // Both the recorded path and the table's spelling of it go through
  // `normalize_program_path`, which is `cygpath` on the Windows branch — and
  // `cygpath` does not exist on the ubuntu and macOS lanes, where the forced
  // Windows branch would otherwise answer 2 before `ps` was ever called, for a
  // reason that has nothing to do with the parse. An identity shim makes the
  // spelling a no-op on every host, so what these cases measure is the table.
  const cygpath = shim("cygpath", `printf '%s' "$2"`);
  // Forward slashes, and not because Windows minds: `cygpath -m` is what
  // dev-runtime.sh records, so this is the real recorded shape — and MSYS bash
  // strips the backslashes out of a `C:\...` argument handed to it from node,
  // which would leave this case comparing two strings that were never equal.
  const SERVER = "C:/wl-target/stage/wenlan-server.exe";
  // The MSYS pid of the backgrounded job. Not one of the background rows'.
  const JOB = "1000";
  const jobpid = (psBody: string) =>
    runDriver(["jobpid", JOB, SERVER], { shims: [shim("ps", psBody), cygpath] });
  // A row for some other process, so every "could not measure" fixture below
  // is a table that a parse which accepts it would read as a clean negative.
  const strangerRow = psRow("7100", "7100", "10:23:45", "C:/Windows/System32/spoolsv.exe");

  it("reports the Windows pid when the row names the program that was launched", () => {
    const result = jobpid(psTable([psRow(JOB, "4242", "10:23:45", SERVER)]));
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
  });


  // `for _ in $(seq 1 100)` with a `seq` that cannot run — absent, broken, a
  // PATH that lost /usr/bin — yields the empty word list, the body never
  // executes, and the function falls through to `return 1`: "the process never
  // appeared", from a hundred measurements that did not happen, and the caller
  // gives up on a daemon it has just spawned, leaving it running with no
  // ownership record. The loop is arithmetic, so a `seq` exiting 127 sits on
  // PATH here and the answer still has to come from the table.
  it("finds the pid even when `seq` cannot run", () => {
    const result = runDriver(["jobpid", JOB, SERVER], {
      shims: [
        shim("ps", psTable([psRow(JOB, "4242", "10:23:45", SERVER)])),
        cygpath,
        shim("seq", "exit 127"),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
  }, 60_000);
  // The measured negative is the reason this helper polls at all: `nohup env …
  // server &` is a chain of MSYS processes ending in an exec, and until that
  // exec lands the row is legitimately not there. So 1 from the parse must NOT
  // be escalated to "could not measure" — a helper that returned 2 on the first
  // empty snapshot would refuse every start that was merely slow.
  it("keeps polling while the job row is measurably absent, then reports the pid", () => {
    const counted = [
      `count="$(dirname "$0")/ps.count"`,
      "n=0",
      `if [ -f "$count" ]; then n="$(cat "$count")"; fi`,
      "n=$((n + 1))",
      `printf '%s' "$n" >"$count"`,
      `if [ "$n" -lt 2 ]; then`,
      psTable([strangerRow]),
      "else",
      psTable([psRow(JOB, "4242", "10:23:45", SERVER)]),
      "fi",
    ].join("\n");
    const result = jobpid(counted);
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=4242");
    // Polls, so it sleeps and re-spawns `ps`; the 5s default measures the
    // host's load rather than the poll.
  }, 60_000);

  // THE TERMINAL NEGATIVE: the one answer this helper may give after a window
  // of readable tables that did not contain the row, and the one the `rc == 2`
  // early return above must never become — the caller's responses to 1 and to 2
  // differ in what they print and in whether they call the reap a measurement.
  //
  // Run at FIVE rounds rather than the shipped hundred. At 0.1s a round the
  // shipped window is ten seconds of sleeping plus a hundred `ps` spawns, which
  // measured 70.6s here; what this case is about is the answer at the end of a
  // window, not how long the window is. The count file makes the shortening
  // observable rather than assumed: the shim is consulted exactly `_rounds`
  // times, so a helper that ignored the override, or ran a different number of
  // rounds, fails here. The shipped 100 is pinned by the two rows below.
  it("reports the MEASURED NEGATIVE after a full window of readable tables", () => {
    const dir = makeTempRoot();
    const countPath = resolve(dir, "ps.count");
    const counted = [
      `count="$WENLAN_TEST_PS_COUNT"`,
      "n=0",
      `if [ -f "$count" ]; then n="$(cat "$count")"; fi`,
      "n=$((n + 1))",
      `printf '%s' "$n" >"$count"`,
      psTable([strangerRow]),
    ].join("\n");
    const result = runDriver(["jobpid", JOB, SERVER], {
      shims: [shim("ps", counted), cygpath],
      env: {
        WENLAN_HOST_PROCESS_POLL_ROUNDS: "5",
        WENLAN_TEST_PS_COUNT: countPath.split("\\").join("/"),
      },
    });
    // 1, and emphatically not 2: every table it read was whole, well-formed,
    // carried the System process and cleared the row floor. The row simply was
    // not in any of them.
    expect(result.stdout.trim(), result.stderr).toBe("rc=1 out=");
    expect(readFileSync(countPath, "utf8"), "the window was not five rounds").toBe("5");
  }, 60_000);

  // The override is a TEST convenience, so the shipped default is pinned in the
  // source: a library that shortened its own window would still satisfy the
  // case above, which supplies its own count.
  it("ships a hundred-round window as its default", () => {
    // `${VAR-100}`, not `${VAR:-100}`: an empty value is something a caller set,
    // and the row below requires it to be refused rather than defaulted.
    expect(readFileSync(libPath, "utf8")).toContain(
      '"${WENLAN_HOST_PROCESS_POLL_ROUNDS-100}"',
    );
  });

  // And a window that was asked for and not understood is not a window. Falling
  // back to the default here would report a measurement about a window nobody
  // chose — the same substitution as reading a failed probe as a negative, one
  // level up.
  it.each([
    ["reports COULD NOT MEASURE for a poll window that is not a number", "five"],
    ["reports COULD NOT MEASURE for a poll window of zero rounds", "0"],
    ["reports COULD NOT MEASURE for a negative poll window", "-1"],
    ["reports COULD NOT MEASURE for an empty poll window", ""],
  ])("%s", (_title, rounds) => {
    const result = runDriver(["jobpid", JOB, SERVER], {
      shims: [shim("ps", psTable([psRow(JOB, "4242", "10:23:45", SERVER)])), cygpath],
      env: { WENLAN_HOST_PROCESS_POLL_ROUNDS: rounds },
    });
    // The table CONTAINS the row, so rc=0 is what a fallback to the default
    // would produce: this refuses before it ever looks.
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // The case above with its WINDOW taken away. Moving the round COUNT off `seq`
  // is not enough: a bare `sleep 0.1` whose status nothing reads, invoked from
  // the left of a `||` (which disables errexit through the whole body), lets a
  // failing `sleep` run all hundred polls back to back in microseconds and
  // return the same terminal 1 the case above earns with ten real seconds. A
  // ten-second window that took ten milliseconds is not a window.
  //
  // The table is the same measured-absent one, so the ONLY difference between
  // this case and the one above is whether the delay happened.
  it("reports COULD NOT MEASURE when the polling delay cannot be performed", () => {
    const result = runDriver(["jobpid", JOB, SERVER], {
      shims: [
        shim("ps", psTable([strangerRow])),
        cygpath,
        // Both spellings: the helper falls back to a whole second when a
        // fractional argument is refused, so only a `sleep` that cannot delay
        // AT ALL is unmeasurable.
        shim("sleep", "exit 1"),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  }, 60_000);

  // The warning is EIGHT WORDS, which an `NF < 8` word count reads as a row —
  // the nine-word one next door is the same point for the other copy of the
  // parse, and a four-word one is what both rules already catch.
  it.each([
    [
      "reports COULD NOT MEASURE for a job snapshot with an eight-word warning in it",
      () => psTable(["ps: warning: the process table was read late", strangerRow]),
    ],
    [
      "reports COULD NOT MEASURE for a job snapshot too short to be a process table",
      () => psTableTooShort([strangerRow]),
    ],
    [
      "reports COULD NOT MEASURE for a job snapshot with no System process in it",
      () => psTableNoSystem([strangerRow]),
    ],
  ])("%s", (_title, table) => {
    const result = jobpid(table());
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when the job snapshot header is not the one it parses", () => {
    const noCommand = psHeader.replace("COMMAND", "CMDLINE");
    const noPid = psHeader.replace("    PID ", "    XID ");
    for (const header of [noCommand, noPid]) {
      const result = jobpid(psTable([psRow(JOB, "4242", "10:23:45", SERVER)], header));
      expect(result.stdout.trim(), `${header}\n${result.stderr}`).toBe("rc=2 out=");
    }
    // Two full driver spawns in one case. On a loaded Windows host each is a
    // second or two, so the 5s default measures the machine, not the parse.
  }, 60_000);
});

// `2>/dev/null` on the Windows snapshots throws away the one signal that says
// the tool had something to complain about, so a status-0 warning can sit
// beside a table missing the very process being asked about. stderr is merged
// instead, and "every line is a row" is what makes the merge fatal rather than
// decorative.
describe("host-process.sh: a contaminated Windows snapshot is unmeasured", () => {
  const warned = (body: string) => `${body}\necho 'ps: warning: /proc unavailable' >&2`;

  it("refuses a ps -W snapshot with a warning merged into it", () => {
    const table = psTable([psRow("1000", "4242", "10:23:45", "C:\\x\\wenlan-server.exe")]);
    const result = runDriver(["image", "4242"], { shims: [shim("ps", warned(table))] });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // An `NF < 8` WORD COUNT catches the four-word warning above and counts this
  // nine-word one as a row, which makes the stderr merge decorative for exactly
  // the lines it was added to catch. The rule is structural instead: PID PPID
  // PGID WINPID, four numbers, which every one of this host's 277 rows begins
  // with and no diagnostic does. Field counts run 8 to 15, so no word count is
  // a ceiling either.
  //
  // The line sits in the stream at a fixed position rather than arriving on
  // stderr, deliberately: `2>&1` gives no ordering guarantee between a buffered
  // stdout and an unbuffered stderr, and a diagnostic that lands FIRST is read
  // as the header instead — rc=2 for a different reason, which would make this
  // case pass whatever the row rule said.
  it("refuses a ps -W snapshot with a nine-word diagnostic in the table", () => {
    const result = runDriver(["image", "4242"], {
      shims: [
        shim(
          "ps",
          psTable([
            "ps: could not read the process table for you",
            psRow("1000", "4242", "10:23:45", "C:\\x\\wenlan-server.exe"),
          ]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("refuses a netstat table with a torn protocol row in it", () => {
    // Not a diagnostic — a row that claims to be TCP and is not one. A table
    // that can lose half a row can lose a whole one, including ours.
    const result = runDriver(["listener", "17878"], {
      shims: [
        shim(
          "netstat",
          netstatTable([
            "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576",
            "  TCP    0.0.0.0:445",
          ]),
        ),
      ],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });
});

describe("host-process.sh: process_image_path, POSIX branch", () => {
  it("reports the image path for a pid ps lists", () => {
    const result = runDriver(["image", "4242"], {
      posixBranch: true,
      shims: [shim("ps", `printf '%s\\n' '/usr/local/bin/wenlan-server'`)],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=0 out=/usr/local/bin/wenlan-server");
  });

  // The pid has to be one the kernel really does not have: the negative now
  // needs ESRCH from `kill -0` as its third witness, and `kill` is a builtin
  // that no shim can intercept. 4242 is a plausible live pid on the host
  // running this suite; 999999 is not.
  it("reports a NEGATIVE only for status 1 with nothing on either stream", () => {
    const result = runDriver(["image", "999999"], {
      posixBranch: true,
      shims: [shim("ps", "exit 1")],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=1 out=");
  });

  // The same hidepid hole as in process_is_alive: silent ps, status 1, and a
  // kill that failed with EPERM rather than ESRCH — two failed measurements
  // agreeing on a negative neither of them made. The caller reads "none" as "a
  // different binary" and kills nothing, or deletes the record.
  //
  // The title must stay distinct from the process_is_alive case above it: the
  // negative-control harness keys per-test states by title, so two tests with
  // one name collapse to one entry and a must_survive check silently scores
  // the wrong probe.
  it("reports COULD NOT MEASURE for a silent image ps when kill says EPERM, not ESRCH", () => {
    const result = runDriver(["image", "999999"], {
      posixBranch: true,
      shims: [shim("ps", "exit 1")],
      env: { WENLAN_HOST_PROCESS_FORCE_EPERM: "1" },
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // `ps` exiting 0 with nothing to say. A pid ps lists always has a command, so
  // silence on the success path is a broken probe — and a trailing
  // `[[ -n "$out" ]] || return 1` reports it as "no such process".
  it("reports COULD NOT MEASURE when ps succeeds and prints nothing", () => {
    const result = runDriver(["image", "999999"], {
      posixBranch: true,
      shims: [shim("ps", "exit 0")],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  // The pid has to be one this host really does not have. The shipped library
  // answers 2 here before it ever consults errno — the message on stderr is
  // enough — but the negative control that drops stderr reaches the errno
  // probe, and with a pid that happens to exist the control would score a
  // reverted fix as still-defended.
  it("reports COULD NOT MEASURE when ps fails with status 1 AND a message", () => {
    const result = runDriver(["image", "999999"], {
      posixBranch: true,
      shims: [shim("ps", `echo "ps: /proc/999999: Permission denied" >&2\nexit 1`)],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE for a status that is not ps's own no-match", () => {
    const result = runDriver(["image", "4242"], {
      posixBranch: true,
      shims: [shim("ps", "exit 2")],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });

  it("reports COULD NOT MEASURE when a warning is merged into the image path", () => {
    // stderr has to be merged in to see the message above; that means the
    // success path can be handed a warning plus the command. An image path this
    // cannot trust must not be compared, because the comparison decides a kill.
    const result = runDriver(["image", "4242"], {
      posixBranch: true,
      shims: [shim("ps", `echo "ps: warning: bad ps syntax" >&2\nprintf '%s\\n' '/usr/bin/x'`)],
    });
    expect(result.stdout.trim(), result.stderr).toBe("rc=2 out=");
  });
});

describe("host-process.sh: the probe_* wrappers map all three states", () => {
  // `out="$(f)"; rc=$?` aborts at the assignment under `set -e`, and
  // `if out="$(f)"; then … fi; rc=$?` reads the compound's own status, which is
  // 0 — that second form silently reported every NEGATIVE as unmeasured. It was
  // caught by exercising the third state; these cases keep it caught.
  it.each([
    ["found", netstatTable([listeningRow(17931, "4242")]), "state=found pid=4242"],
    ["none", netstatTable([listeningRow(17999, "4242")]), "state=none pid="],
    ["unmeasured", "exit 1", "state=unmeasured pid="],
  ])("probe_listener_port reports %s", (_label, body, expected) => {
    const result = runDriver(["probe-listener", "17931"], { shims: [shim("netstat", body)] });
    expect(result.stdout.trim(), result.stderr).toBe(expected);
  });

  it.each([
    ["alive", tasklistTable([tasklistRow("x", "4242")]), "state=alive"],
    ["gone", tasklistTable([]), "state=gone"],
    ["unmeasured", "exit 1", "state=unmeasured"],
  ])("probe_process_alive reports %s", (_label, body, expected) => {
    const result = runDriver(["probe-alive", "4242"], { shims: [shim("tasklist", body)] });
    expect(result.stdout.trim(), result.stderr).toBe(expected);
  });

  it.each([
    [
      "found",
      psTable([psRow("1000", "4242", "10:23:45", "C:\\a.exe")]),
      "state=found value=C:\\a.exe",
    ],
    [
      "none",
      psTable([psRow("1000", "9999", "10:23:45", "C:\\a.exe")]),
      "state=none value=",
    ],
    ["unmeasured", "exit 1", "state=unmeasured value="],
  ])("probe_process_image reports %s", (_label, body, expected) => {
    const result = runDriver(["probe-image", "4242"], { shims: [shim("ps", body)] });
    expect(result.stdout.trim(), result.stderr).toBe(expected);
  });
});

describe("host-process.sh: shape contracts the behaviour above cannot reach", () => {
  const source = () => readFileSync(libPath, "utf8");
  // The prose explains the traps by name, so the shape contracts read the code
  // and not the commentary about it.
  const code = () =>
    source()
      .split("\n")
      .filter((line) => !/^\s*#/.test(line))
      .join("\n");

  it("has no early `exit` in any measurement awk program", () => {
    // Adding `|| return N` to a pipeline whose consumer exits on its first match
    // CAN make the producer take SIGPIPE, and under `pipefail` that 141 is
    // indistinguishable from a real parser failure — a correctness fix turned
    // into a spurious failure. The awk programs set a flag and print at END.
    const awkPrograms = code().match(/awk[^']*'[^']*'/gs) ?? [];
    expect(awkPrograms.length).toBeGreaterThan(0);
    for (const program of awkPrograms) {
      expect(program, program).toContain("END");
      // POSITION, not spelling. The rule used to be the shape `exit}`, `exit;`
      // or `exit` at end of line, which `exit 3` walks straight past — and
      // `exit 3` happens to be correct, but for a reason the rule was not
      // checking. What actually makes an exit safe is that all input has
      // already been read, which is what END means, so that is what is pinned.
      const before = program.slice(0, program.indexOf("END"));
      expect(before, program).not.toMatch(/(^|[\s;{(])exit\b/);
    }
  });

  it("reads the `ps -W` table in exactly one place", () => {
    // The behavioural cases above cannot see this. `process_image_path` and
    // `windows_pid_for_job` ask the same table the same question, and while
    // there are two parses every hardening lands on one of them. A second call
    // site is fine, a second PARSE is the defect, so what is pinned is the
    // number of places that run the command at all.
    //
    // Across the REPOSITORY, not across this library: a copy has also lived in
    // `scripts/dev-runtime.sh`'s `reap_staged_daemon` — the same table, the same
    // four witnesses, maintained separately — which a check reading only the
    // library cannot see. Every shell file that sources this library, and the
    // scaffolding they share, is counted too, comments stripped from each.
    const strip = (text: string) =>
      text
        .split("\n")
        .filter((line) => !/^\s*#/.test(line))
        .join("\n");
    // The library under test (which a control may replace), then the callers,
    // which are always the shipped ones.
    const searched: [string, string][] = [
      [libPath, code()],
      ...[
        "scripts/dev-runtime.sh",
        "scripts/smoke-cli.sh",
        "scripts/smoke-mcp.sh",
        "scripts/lib/smoke-common.sh",
      ].map(
        (relative) =>
          [relative, strip(readFileSync(resolve(root, relative), "utf8"))] as [string, string],
      ),
    ];
    const places = searched.flatMap(([where, text]) =>
      (text.match(/ps -W\b/g) ?? []).map(() => where),
    );
    expect(places, "ps -W is parsed in more than one place again").toHaveLength(1);
    expect(places[0], "the one parse must be the library's").toBe(libPath);
    expect(code()).toContain("ps_w_row_for WINPID");
    expect(code()).toContain("ps_w_row_for PID");
    // The all-rows entry point, which is what let the third copy be deleted
    // rather than merely hardened for a fourth time.
    expect(code()).toContain("ps_w_rows_matching");
    expect(
      strip(readFileSync(resolve(root, "scripts/dev-runtime.sh"), "utf8")),
      "dev-runtime.sh must ask the library for its scan, not re-implement it",
    ).toContain("ps_w_rows_matching");
  });

  it("never pipes a parser into `head`", () => {
    // `sed … | head -1` CAN SIGPIPE sed for the same reason.
    expect(code()).not.toMatch(/\|\s*head\b/);
  });

  it("checks the status of every stage of the Windows listener pipeline", () => {
    const start = source().indexOf("listener_pid_for_port() {");
    expect(start).toBeGreaterThan(-1);
    const body = source().slice(start, source().indexOf("\n}\n", start));
    // netstat, awk and tr each get their own `|| return 2`: checking only the
    // first stage let a failing parser read as "no listener".
    expect(body.match(/\|\| return 2/g)?.length ?? 0).toBeGreaterThanOrEqual(4);
    expect(body).toContain("command -v netstat");
    expect(body).toContain("command -v lsof");
    // `|| hit=""` is the shape that shipped here first: it folds a failing
    // lsof into an empty result, which the tail below then reports as "no
    // listener". The behavioural cases cover it, and this pins the shape so a
    // future edit cannot reintroduce it while the shims keep passing.
    expect(body).not.toContain('|| hit=""');
    // stderr must be captured, not discarded: with -t it is the only thing
    // that separates "nothing matched" from "the probe broke".
    expect(body).toContain("2>&1");
  });

  it("lets a POSIX host opt into the Windows branch but never the reverse", () => {
    const text = source();
    expect(text).toContain('WENLAN_HOST_PROCESS_PLATFORM:-}" == "windows"');
    // Only the assignment to 1 may be reachable from the override, so a real
    // Windows host cannot be talked out of its identity-checked kill path.
    const overrideBlock = text.slice(text.indexOf("WENLAN_HOST_PROCESS_PLATFORM"));
    expect(overrideBlock.slice(0, 200)).toContain("HOST_IS_WINDOWS=1");
    expect(overrideBlock.slice(0, 200)).not.toContain("HOST_IS_WINDOWS=0");
  });
});

describe("the tri-state library is the one every caller uses", () => {
  // The script itself plus every `scripts/lib/*.sh` it names in a shellcheck
  // source= annotation, except host-process.sh: that one DEFINES the probes,
  // so its internals are not call sites of the kind this contract governs.
  const callerFiles = (relative: string): string[] => {
    const text = readFileSync(resolve(root, relative), "utf8");
    const libs = [
      ...text.matchAll(/# shellcheck source=(scripts\/lib\/[A-Za-z0-9._-]+\.sh)/g),
    ]
      .map((m) => m[1])
      .filter((p) => p !== "scripts/lib/host-process.sh");
    return [relative, ...new Set(libs)];
  };

  it.each([
    ["scripts/dev-runtime.sh"],
    ["scripts/smoke-cli.sh"],
    ["scripts/smoke-mcp.sh"],
  ])("%s sources it with a shellcheck source= annotation", (relative) => {
    const text = readFileSync(resolve(root, relative), "utf8");
    expect(text).toContain("# shellcheck source=scripts/lib/host-process.sh");
    expect(text).toMatch(/\.\s+"\$(SCRIPT_DIR|ROOT)(\/scripts)?\/lib\/host-process\.sh"/);
  });

  it.each([
    ["scripts/dev-runtime.sh"],
    ["scripts/smoke-cli.sh"],
    ["scripts/smoke-mcp.sh"],
  ])("%s defines no local copy of the moved primitives", (relative) => {
    const text = readFileSync(resolve(root, relative), "utf8");
    for (const name of [
      "listener_pid_for_port",
      "process_is_alive",
      "process_image_path",
      "kill_by_image_path",
      "terminate_process",
      "force_terminate_process",
      "windows_pid_for_job",
      "native_path",
      "normalize_program_path",
    ]) {
      expect(text, `${relative} redefines ${name}`).not.toMatch(
        new RegExp(`^${name}\\(\\) \\{`, "m"),
      );
    }
  });

  it.each([
    ["scripts/dev-runtime.sh", 10],
    ["scripts/smoke-cli.sh", 5],
    ["scripts/smoke-mcp.sh", 5],
  ])("%s branches on the unmeasured state at EVERY call site", (relative, atLeast) => {
    // Counting probe calls in the WHOLE FILE and then looking for the word
    // "unmeasured" anywhere in the WHOLE FILE passes a file with nine correct
    // call sites and one that has lost its third branch — the other nine pay
    // for it. The contract is per-caller ("every caller branches on all
    // three"), so enumerate the sites, take the branch that consumes each
    // one's answer, and judge that window on its own.
    //
    // A caller is the script PLUS the libraries it sources, minus
    // host-process.sh, which defines the probes rather than calling them: the
    // smokes keep their shared scaffolding in lib/smoke-common.sh, and a site
    // moved there is still a site this row has to judge. Each is reported
    // against the file it is written in.
    const sites = callerFiles(relative).flatMap((file) =>
      probeCallSites(readFileSync(resolve(root, file), "utf8")).map((site) => ({
        ...site,
        file,
      })),
    );
    expect(sites.length, `${relative}: no probe call sites found`).toBeGreaterThanOrEqual(
      atLeast,
    );
    for (const site of sites) {
      const where = `${site.file}:${site.line} (${site.probe})`;
      // A call whose answer nothing reads is the same defect one step earlier.
      expect(site.read, `${where}: nothing branches on ${site.state}`).toBe(true);
      expect(
        branchesOnUnmeasured(site),
        `${where}: does not branch on the unmeasured state`,
      ).toBe(true);
    }
  });
});

// A skill is instructions to a future agent, so a fenced block in one is code:
// it gets pasted and run as written, and the prose two paragraphs below telling
// the reader to use the library instead does not. A block such as
//
//   netstat -ano | awk '$1=="TCP" && $2 ~ /:17878$/ && $4=="LISTENING" {print $5 ...
//
// is two-state where the question has three: a netstat that is missing or
// killed, a table whose columns moved, and a non-English Windows whose State
// column reads ABHOEREN all produce the same no-match as a genuinely free port
// — the exact substitution `listener_pid_for_port` exists to remove, published
// as the recommended recipe.
describe("the run-wenlan skill's runnable blocks use the tri-state library", () => {
  const skillPath = ".claude/skills/run-wenlan/SKILL.md";
  // Comments stripped, like the shape contracts above: what a future agent RUNS
  // is the code, and a comment naming a tool is prose that happens to sit
  // inside the fence ("NOT the pid netstat and tasklist report").
  const fencedBash = (): string[] => {
    const text = readFileSync(resolve(root, skillPath), "utf8");
    return [...text.matchAll(/^```(?:bash|sh)\n([\s\S]*?)^```/gm)].map((m) =>
      m[1]
        .split("\n")
        .map((line) => line.replace(/(^|\s)#.*$/, "$1"))
        .join("\n"),
    );
  };

  it("has fenced bash blocks to judge", () => {
    expect(fencedBash().length).toBeGreaterThan(0);
  });

  it.each([
    ["netstat", /\bnetstat\b/],
    ["ps -W", /\bps\s+-W\b/],
    ["tasklist", /\btasklist\b/],
  ])("no fenced block parses the %s table itself", (tool, pattern) => {
    for (const block of fencedBash()) {
      expect(
        pattern.test(block),
        `${skillPath} fences a runnable ${tool} parse:\n${block}\n` +
          "Every one of these tables has a failure mode that looks exactly like " +
          "the negative answer. scripts/lib/host-process.sh parses each of them " +
          "once, with the witnesses that tell an unreadable table from an absent " +
          "row; a second parse in a skill is a second parse no hardening reaches.",
      ).toBe(false);
    }
  });

  it("shows the listener probe with all three states branched on", () => {
    const blocks = fencedBash().filter((block) => block.includes("probe_listener_port"));
    expect(blocks.length, `${skillPath} never demonstrates probe_listener_port`).toBeGreaterThan(
      0,
    );
    for (const block of blocks) {
      // A two-branch example is the defect back again with the library's name
      // on it, so the third state has to be visible in the recipe itself.
      for (const state of ["found", "none", "unmeasured"]) {
        expect(block, `${skillPath}: the probe example omits the ${state} state`).toContain(
          state,
        );
      }
    }
  });

  it("resolves the Windows pid through the single validated ps -W parse", () => {
    const blocks = fencedBash();
    expect(
      blocks.some(
        (block) => block.includes("ps_w_row_for") || block.includes("windows_pid_for_job"),
      ),
      `${skillPath} shows no way to resolve a Windows pid through the library`,
    ).toBe(true);
  });

  it("names only helpers the library actually defines", () => {
    // A recipe that calls a helper by the wrong name fails at run time with
    // "command not found", which a future agent reads as "the library is
    // broken" rather than "the skill is stale".
    const lib = readFileSync(resolve(root, "scripts/lib/host-process.sh"), "utf8");
    const referenced = new Set<string>();
    for (const block of fencedBash()) {
      for (const match of block.matchAll(
        /\b(probe_[a-z_]+|listener_pid_for_port|ps_w_row_for|ps_w_rows_matching|windows_pid_for_job|process_is_alive|process_image_path|kill_by_image_path|native_path|normalize_program_path)\b/g,
      )) {
        referenced.add(match[1]);
      }
    }
    expect(referenced.size).toBeGreaterThan(0);
    for (const name of referenced) {
      expect(lib, `${skillPath} calls ${name}, which host-process.sh does not define`).toMatch(
        new RegExp(`^${name}\\(\\) \\{`, "m"),
      );
    }
  });
});
