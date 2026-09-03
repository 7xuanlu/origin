import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { afterEach, describe, expect, it, vi } from "vitest";

// `scripts/attest.sh` is the portable replacement for `~/.claude/bin/attest.sh`,
// a personal macOS helper that does not exist in a fresh checkout and does not
// exist on Windows at all — which left `/verify` and `/prove` mandating a wrapper
// no Windows worker could run.
//
// It carries the ledger form of the tri-state invariant: AN UNRECORDED RUN MUST
// NEVER BE INDISTINGUISHABLE FROM A RECORDED ONE. The weekly sweep reads a
// missing row as "the smoke never ran", so a wrapper that exited 0 after failing
// to write would manufacture that finding out of a green run. The third case
// below — green command, unwritable ledger — is the one that matters.

// Every case here spawns node -> Git Bash -> attest.sh, which costs a few
// hundred milliseconds on an idle box and several seconds when other agents are
// building on the same machine: the lock case below measured 2.9s alone and
// over 5s under a full parallel run of scripts/. At vitest's 5000ms default a
// red here is therefore a statement about the HOST, not about attest.sh -- an
// unmeasured run wearing the same red as a real behavioural failure, which is
// the exact confusion this wrapper exists to prevent in the ledger. Raised at
// file scope (it touches no other suite) so a slow host reports slowly instead
// of falsely; the cases that poll, or spawn twelve writers, keep their own
// larger budgets at the call site, and those are the ones a genuine hang would
// still be caught by.
vi.setConfig({ testTimeout: 30_000 });

const root = resolve(import.meta.dirname, "..");
const attest = "scripts/attest.sh";
const tempRoots: string[] = [];

afterEach(() => {
  for (const path of tempRoots.splice(0)) {
    rmSync(path, { recursive: true, force: true });
  }
});

function makeTempRoot(): string {
  const dir = mkdtempSync(resolve(tmpdir(), "wenlan-attest-"));
  tempRoots.push(dir);
  return dir;
}

function runAttest(
  args: string[],
  env: Record<string, string> = {},
): { status: number | null; stdout: string; stderr: string } {
  const merged: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (value !== undefined) merged[key] = value;
  }
  Object.assign(merged, env);

  // Git Bash explicitly on Windows, for the same reason package.json uses it: a
  // bare `bash` on a machine with WSL is the Linux distro.
  const result =
    process.platform === "win32"
      ? spawnSync(process.execPath, ["scripts/run-bash.mjs", attest, ...args], {
          cwd: root,
          encoding: "utf8",
          env: merged,
        })
      : spawnSync("bash", [attest, ...args], { cwd: root, encoding: "utf8", env: merged });

  return { status: result.status, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
}

function readLedger(path: string): Record<string, unknown>[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

describe("attest.sh: the run is recorded, or the wrapper fails", () => {
  it("appends one JSON row and preserves a passing status", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttest(["echo", "attested hello"], {
      WENLAN_ATTEST_LEDGER: ledger,
      WENLAN_ATTEST_SURFACE: "cli",
    });

    expect(result.status, result.stderr).toBe(0);
    // The command owns the terminal: its output is passed through, never
    // captured, so a caller watching the smoke still sees the smoke.
    expect(result.stdout).toContain("attested hello");

    const rows = readLedger(ledger);
    expect(rows).toHaveLength(1);
    expect(rows[0].status).toBe(0);
    expect(rows[0].surface).toBe("cli");
    expect(rows[0].command).toBe("echo 'attested hello'");
    expect(rows[0].ts).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/);
    expect(["windows", "macos", "linux"]).toContain(rows[0].platform);
  });

  it("records a failing command's exact status and exits with it", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttest(["bash", "-c", "exit 7"], { WENLAN_ATTEST_LEDGER: ledger });

    expect(result.status).toBe(7);
    const rows = readLedger(ledger);
    expect(rows).toHaveLength(1);
    expect(rows[0].status).toBe(7);
  });

  // THE case. A green command whose evidence was lost is not a green check: the
  // sweep would see no row and conclude the smoke never ran, with nothing
  // anywhere saying otherwise.
  it("fails when the command passed but the ledger could not be written", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");
    // A directory where the ledger file belongs: the append cannot succeed, and
    // no amount of retrying will change that.
    mkdirSync(ledger, { recursive: true });

    const result = runAttest(["echo", "green"], { WENLAN_ATTEST_LEDGER: ledger });

    expect(result.stdout).toContain("green");
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("was NOT recorded");
    expect(result.stderr).toContain("command exited 0");
  });

  it("keeps the command's status when both the command and the ledger fail", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");
    mkdirSync(ledger, { recursive: true });

    const result = runAttest(["bash", "-c", "exit 9"], { WENLAN_ATTEST_LEDGER: ledger });

    // The command's own failure is the more informative one, and it is what a
    // caller's `|| exit $?` is looking for.
    expect(result.status).toBe(9);
    expect(result.stderr).toContain("was NOT recorded");
  });

  it("appends rather than truncating, so a week of evidence accumulates", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");
    writeFileSync(ledger, '{"ts":"2026-01-01T00:00:00Z","command":"earlier"}\n');

    runAttest(["true"], { WENLAN_ATTEST_LEDGER: ledger });
    runAttest(["true"], { WENLAN_ATTEST_LEDGER: ledger });

    const rows = readLedger(ledger);
    expect(rows).toHaveLength(3);
    expect(rows[0].command).toBe("earlier");
  });

  it("refuses an empty invocation instead of recording a run that never happened", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttest([], { WENLAN_ATTEST_LEDGER: ledger });

    expect(result.status).toBe(2);
    expect(result.stderr).toContain("usage:");
    expect(() => readFileSync(ledger, "utf8")).toThrow();
  });

  // The two commands that used to walk out of the wrapper before it could write
  // anything. The arguments were executed by the wrapper's own shell, so a
  // builtin able to end that shell ended the WRAPPER: `exit` returned from it at
  // the call site, and `exec` replaced it outright. Both left status 0 and no
  // ledger row — and the sweep reads a missing row as "the smoke never ran", so
  // silence behind a green exit is the one answer this file may not give.
  // Measured before the fix: `bash scripts/attest.sh exit 0` and
  // `bash scripts/attest.sh exec true` each exited 0 with no ledger file at all.
  it("records `exit 0`, which used to return from the wrapper itself", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttest(["exit", "0"], { WENLAN_ATTEST_LEDGER: ledger });

    expect(result.status, result.stderr).toBe(0);
    const rows = readLedger(ledger);
    expect(rows).toHaveLength(1);
    expect(rows[0].command).toBe("exit 0");
    expect(rows[0].status).toBe(0);
  });

  it("records `exec`, which used to replace the wrapper outright", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttest(["exec", "true"], { WENLAN_ATTEST_LEDGER: ledger });

    expect(result.status, result.stderr).toBe(0);
    const rows = readLedger(ledger);
    expect(rows).toHaveLength(1);
    expect(rows[0].command).toBe("exec true");
  });

  // A non-zero `exit` must still reach the caller through the subshell, or the
  // containment above would have been bought by losing the status.
  it("still reports a non-zero status through the subshell", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttest(["exit", "5"], { WENLAN_ATTEST_LEDGER: ledger });

    expect(result.status).toBe(5);
    expect(readLedger(ledger)[0].status).toBe(5);
  });

  it("escapes a command containing quotes and backslashes into valid JSON", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    // A Windows path and a quoted argument in one command line: the row must
    // still parse, or the sweep's `tail .claude/attest.jsonl` reads garbage.
    const result = runAttest(["echo", 'C:\\wl-target\\debug "quoted"'], {
      WENLAN_ATTEST_LEDGER: ledger,
    });

    expect(result.status, result.stderr).toBe(0);
    const rows = readLedger(ledger);
    expect(rows[0].command).toContain("C:\\wl-target\\debug");
    expect(rows[0].command).toContain('"quoted"');
  });
});

// The append is three operations — write, read the last line back, compare —
// and unlocked, a neighbour's row landing between the write and the read makes
// a writer whose own row went down perfectly report that its evidence was lost.
// Measured on this host before the lock: twelve concurrent `attest.sh true`
// invocations produced twelve good rows and THREE workers exiting 1 with "the
// append was truncated or interleaved". A green command turned red, and the
// sweep told its evidence is missing, by another process doing nothing wrong.
describe("attest.sh: concurrent writers", () => {
  function runAttestAsync(
    args: string[],
    env: Record<string, string>,
  ): Promise<{ status: number | null; stderr: string }> {
    const merged: Record<string, string> = {};
    for (const [key, value] of Object.entries(process.env)) {
      if (value !== undefined) merged[key] = value;
    }
    Object.assign(merged, env);
    const [command, commandArgs] =
      process.platform === "win32"
        ? [process.execPath, ["scripts/run-bash.mjs", attest, ...args]]
        : ["bash", [attest, ...args]];
    return new Promise((done) => {
      const child = spawn(command, commandArgs, { cwd: root, env: merged });
      let stderr = "";
      child.stderr.on("data", (chunk) => {
        stderr += String(chunk);
      });
      child.on("close", (status) => done({ status, stderr }));
    });
  }

  it("serializes twelve simultaneous appends: twelve rows, none reported lost", async () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const results = await Promise.all(
      Array.from({ length: 12 }, () => runAttestAsync(["true"], { WENLAN_ATTEST_LEDGER: ledger })),
    );

    // Every worker's command passed, so every worker must exit 0. Before the
    // lock this is where the failures showed up, and they said the run was not
    // recorded while the row sat in the file.
    const bad = results.filter((r) => r.status !== 0);
    expect(bad.map((r) => r.stderr).join("\n")).toBe("");
    expect(bad).toHaveLength(0);
    // And the ledger holds twelve rows that all parse. A row torn in half by an
    // interleaved write is the quieter half of the same defect: it appends
    // cleanly, exits 0, and no reader of the ledger can read it.
    expect(readLedger(ledger)).toHaveLength(12);
    // The lock is a directory, and a run that finished must not leave one
    // behind: the next writer would wait its whole timeout and then fail.
    expect(existsSync(`${ledger}.lock`)).toBe(false);
  }, 120_000);

  // A lock that cannot be taken is the one case where refusing to write is
  // right. Appending anyway is the interleaving; skipping the lock silently is
  // the same thing wearing the remedy's name.
  it("refuses to append at all when the lock cannot be taken", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");
    // A live holder: the directory exists and its stamp is now, so the stale
    // path cannot reclaim it.
    mkdirSync(`${ledger}.lock`, { recursive: true });
    writeFileSync(`${ledger}.lock/owner`, `999999 ${Math.floor(Date.now() / 1000)}\n`);

    const result = runAttest(["echo", "green"], {
      WENLAN_ATTEST_LEDGER: ledger,
      WENLAN_ATTEST_LOCK_WAIT_S: "1",
    });

    expect(result.stdout).toContain("green");
    expect(result.status).not.toBe(0);
    expect(result.stderr).toContain("could not take the ledger lock");
    expect(result.stderr).toContain("was NOT recorded");
    // Nothing was written, so nothing could have been interleaved.
    expect(existsSync(ledger)).toBe(false);
    // An explicit bound, because the default 5s is not one this test can meet.
    // It spends a second waiting for the lock ON PURPOSE, plus a bash start on
    // Windows, and it shares the machine with the rest of the suite: measured
    // at 2.9s alone and over 5s under the full parallel run, where it timed out
    // and reported as a defect in the lock rather than in the budget.
  }, 60_000);

  // The other half: a holder that was killed leaves a directory nothing
  // releases, and a lock nobody can ever take would wedge every later run.
  it("reclaims a lock whose holder is long gone", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");
    mkdirSync(`${ledger}.lock`, { recursive: true });
    writeFileSync(`${ledger}.lock/owner`, `999999 ${Math.floor(Date.now() / 1000) - 9999}\n`);

    const result = runAttest(["true"], {
      WENLAN_ATTEST_LEDGER: ledger,
      WENLAN_ATTEST_LOCK_WAIT_S: "5",
      WENLAN_ATTEST_LOCK_STALE_S: "300",
    });

    expect(result.status, result.stderr).toBe(0);
    expect(result.stderr).toContain("breaking a stale lock");
    expect(readLedger(ledger)).toHaveLength(1);
  }, 30_000);

  // ---- the stamp is what makes the lock a lock ----
  //
  // Round 15: `LOCK_HELD=1` followed the owner write unconditionally, and the
  // stamp was `$(date +%s || printf 0)`. A holder that cannot stamp is
  // indistinguishable from a crashed one -- the unstamped-lock breaker removes
  // it after ~2s -- and a stamp of 0 makes a LIVE holder look about 1.7 billion
  // seconds old, so any waiter reclaims it immediately. Both hand the lock to a
  // second writer while the first still believes it holds it, which is the
  // interleaving the lock exists to remove.
  //
  // Driven through PATH, never by editing the script: the stub goes in a
  // directory prepended INSIDE the shell, because the MSYS runtime puts its own
  // /usr/bin ahead of any PATH it inherits (measured when a stub curl lost
  // exactly that way in the SignPath status harness), and the wrapper refuses
  // to run at all unless the stub won the lookup.
  function runAttestWithStubs(
    args: string[],
    env: Record<string, string>,
    stubs: Record<string, string>,
  ): { status: number | null; stdout: string; stderr: string } {
    const dir = makeTempRoot();
    const bin = resolve(dir, "bin");
    mkdirSync(bin, { recursive: true });
    for (const [name, body] of Object.entries(stubs)) {
      writeFileSync(resolve(bin, name), body, { mode: 0o755 });
    }
    const wrapper = resolve(dir, "with-stubs.sh");
    writeFileSync(
      wrapper,
      [
        "#!/usr/bin/env bash",
        'if command -v cygpath >/dev/null 2>&1; then',
        '  _bin="$(cygpath -u "$WENLAN_STUB_BIN")"',
        "else",
        '  _bin="$WENLAN_STUB_BIN"',
        "fi",
        'PATH="$_bin:$PATH"',
        "export PATH",
        'for n in $WENLAN_STUB_NAMES; do',
        '  case "$(command -v "$n")" in',
        '    "$_bin"/*) : ;;',
        '    *) echo "STUB $n NOT IN EFFECT: $(command -v "$n")" >&2; exit 97 ;;',
        "  esac",
        "done",
        'exec bash scripts/attest.sh "$@"',
        "",
      ].join("\n"),
      { mode: 0o755 },
    );

    const merged: Record<string, string> = {};
    for (const [key, value] of Object.entries(process.env)) {
      if (value !== undefined) merged[key] = value;
    }
    Object.assign(merged, env, {
      WENLAN_STUB_BIN: bin,
      WENLAN_STUB_NAMES: Object.keys(stubs).join(" "),
    });
    const result =
      process.platform === "win32"
        ? spawnSync(process.execPath, ["scripts/run-bash.mjs", wrapper, ...args], {
            cwd: root,
            encoding: "utf8",
            env: merged,
          })
        : spawnSync("bash", [wrapper, ...args], { cwd: root, encoding: "utf8", env: merged });
    return { status: result.status, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
  }

  it("refuses the lock when the clock cannot be read, instead of stamping it 0", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttestWithStubs(
      ["echo", "green"],
      { WENLAN_ATTEST_LEDGER: ledger, WENLAN_ATTEST_LOCK_WAIT_S: "1" },
      { date: "#!/usr/bin/env bash\nexit 1\n" },
    );

    expect(result.stderr).not.toContain("STUB date NOT IN EFFECT");
    // A stamp of 0 would have been accepted by the old code and would make this
    // live lock reclaimable by the next writer that looked at it.
    expect(result.stderr).toContain("cannot read a usable clock");
    expect(result.stderr).toContain("was NOT recorded");
    expect(result.status).toBe(1);
    // Nothing recorded, and nothing left holding the ledger either.
    expect(existsSync(ledger)).toBe(false);
    expect(existsSync(`${ledger}.lock`)).toBe(false);
  }, 60_000);

  it("refuses the lock when the clock answers something that is not a time", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttestWithStubs(
      ["echo", "green"],
      { WENLAN_ATTEST_LEDGER: ledger, WENLAN_ATTEST_LOCK_WAIT_S: "1" },
      { date: '#!/usr/bin/env bash\nprintf %s "not-a-time"\n' },
    );

    expect(result.stderr).not.toContain("STUB date NOT IN EFFECT");
    // `lock_age_s` rejects a non-numeric stamp, so writing one produces a lock
    // that every waiter reads as unstamped and breaks -- the same outcome as
    // never stamping it.
    //
    // This case also pins the duration guard: with `[ -n ... ]` in place of the
    // numeric test, the wrapper died at `$(( END_EPOCH - START_EPOCH ))` with
    // `not: unbound variable` -- after the command ran, before any row -- and
    // never reached the lock at all. Measured here before the guard was added.
    expect(result.stderr).not.toContain("unbound variable");
    expect(result.stderr).toContain("cannot read a usable clock");
    expect(result.status).toBe(1);
    expect(existsSync(ledger)).toBe(false);
    expect(existsSync(`${ledger}.lock`)).toBe(false);
  }, 60_000);

  it("refuses the lock when the owner stamp cannot be written", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    // A `mkdir` that leaves `owner` as a DIRECTORY inside every lock it
    // creates, so the shipped `printf > "$LOCK_DIR/owner"` fails. This is the
    // full-disk / read-only-parent case made deterministic on any host.
    const result = runAttestWithStubs(
      ["echo", "green"],
      { WENLAN_ATTEST_LEDGER: ledger, WENLAN_ATTEST_LOCK_WAIT_S: "1" },
      {
        mkdir: [
          "#!/usr/bin/env bash",
          "real=/usr/bin/mkdir",
          '[ -x "$real" ] || real=/bin/mkdir',
          '"$real" "$@" || exit $?',
          'for a in "$@"; do',
          '  case "$a" in',
          '    *.lock) "$real" -p "$a/owner" ;;',
          "  esac",
          "done",
          "",
        ].join("\n"),
      },
    );

    expect(result.stderr).not.toContain("STUB mkdir NOT IN EFFECT");
    expect(result.stderr).toContain("cannot stamp the lock");
    expect(result.stderr).toContain("was NOT recorded");
    expect(result.status).toBe(1);
    // The whole point: no row. An unstamped lock is one another writer breaks,
    // so appending under it is an unserialised append.
    expect(existsSync(ledger)).toBe(false);
  }, 60_000);

  // ROUND 5. The write's status was read; the READ-BACK's was thrown away. The
  // check was one `[ "$(cat "$LOCK_DIR/owner" 2>/dev/null)" != "$$ $stamp" ]`,
  // which compares TEXT: a `cat` that prints the expected `PID timestamp` and
  // then exits non-zero satisfies "not unequal", and LOCK_HELD became 1 on a
  // read that failed. Reading the stamp back exists to tell a stamp that is on
  // disk from one that only appeared to be written, and half of that question
  // is the status.
  //
  // The stub prints the real bytes and then fails, which is the shape of a read
  // error after the buffer was already flushed -- the one shape that cannot be
  // caught by looking at the text.
  it("refuses the lock when the stamp read-back fails despite printing the stamp", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttestWithStubs(
      ["echo", "green"],
      { WENLAN_ATTEST_LEDGER: ledger, WENLAN_ATTEST_LOCK_WAIT_S: "1" },
      {
        cat: [
          "#!/usr/bin/env bash",
          "real=/usr/bin/cat",
          '[ -x "$real" ] || real=/bin/cat',
          '"$real" "$@"',
          "exit 5",
          "",
        ].join("\n"),
      },
    );

    expect(result.stderr).not.toContain("STUB cat NOT IN EFFECT");
    expect(result.stderr).toContain("cannot stamp the lock");
    expect(result.stderr).toContain("reading it back exited 5");
    expect(result.stderr).toContain("was NOT recorded");
    expect(result.status).toBe(1);
    // No row, for the same reason as the case above: the lock was never shown
    // to be stamped, and an unstamped lock is one any other writer may break.
    expect(existsSync(ledger)).toBe(false);
  }, 60_000);

  // ROUND 5, the other end of the same lock. `rm -f`/`rmdir` ran with their
  // statuses dropped and `LOCK_HELD=0` underneath them, so a lock that could
  // not be removed was spelled exactly like one that was: the row is written,
  // the wrapper exits 0, and the lock it says it released is still on disk --
  // where it makes every later writer wait LOCK_WAIT_S and then refuse to
  // record ITS run, with nothing in this run's output to explain it.
  it("does not exit 0 claiming a release that did not happen", () => {
    const dir = makeTempRoot();
    const ledger = resolve(dir, "attest.jsonl");

    const result = runAttestWithStubs(
      ["echo", "green"],
      { WENLAN_ATTEST_LEDGER: ledger, WENLAN_ATTEST_LOCK_WAIT_S: "1" },
      { rmdir: "#!/usr/bin/env bash\nexit 1\n" },
    );

    expect(result.stderr).not.toContain("STUB rmdir NOT IN EFFECT");
    expect(result.stderr).toContain("could not be released");
    expect(result.status).not.toBe(0);
    // The row IS there: this is not the unrecorded case, and saying so would
    // send a reader looking for the wrong thing. The lock is what is wrong.
    const rows = readFileSync(ledger, "utf8").trim().split("\n");
    expect(rows).toHaveLength(1);
    expect(JSON.parse(rows[0]).status).toBe(0);
    expect(existsSync(`${ledger}.lock`)).toBe(true);
  }, 60_000);
});

describe("attest.sh: the skills point at the portable wrapper", () => {
  it("verify names scripts/attest.sh, not the personal macOS helper alone", () => {
    const skill = readFileSync(resolve(root, ".claude/skills/verify/SKILL.md"), "utf8");

    expect(skill).toContain("scripts/attest.sh");
    // The old line mandated a path that exists on exactly one machine. If it is
    // still mentioned it must be as a fallback, never as the requirement.
    const lines = skill.split("\n").filter((line) => line.includes("~/.claude/bin/attest.sh"));
    for (const line of lines) {
      expect(line.toLowerCase()).toMatch(/optional|personal|macos|if you have/);
    }
  });
});
