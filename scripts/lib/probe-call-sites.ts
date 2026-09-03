// The probe call sites in a shell script, each paired with the code that reads
// its answer.
//
// Every helper in `scripts/lib/host-process.sh` is TRI-STATE — measured /
// negative / could not measure — and the contract is that EVERY CALLER branches
// on all three. The shape tests that defended that contract used to count probe
// calls in a whole file and then look for the word `unmeasured` anywhere in the
// same file. Both halves are file-global, so a file with nine correct call sites
// and one that had lost its third branch passed: the nine paid for the tenth.
// A per-caller contract needs a per-caller assertion, which needs the call sites
// enumerated, which is what this is.
//
// It lives here, once, rather than in each suite that needs it, for the same
// reason `ps -W` is parsed in one place: a second copy is a second place for the
// next hardening to miss.
//
// The window is deliberately NOT "the rest of the enclosing block". At top level
// that is the rest of the file, so one site's third branch would excuse the next
// site's missing one — the file-global bug again, wearing a loop. It is exactly
// the run of constructs that mention this probe's own state variable: the `case`
// or `if` chain immediately after the call, stopping at the first statement that
// has moved on.

export const PROBE_STATES: Record<string, { state: string; measured: [string, string] }> = {
  probe_listener_port: { state: "LISTENER_PROBE_STATE", measured: ["found", "none"] },
  probe_process_alive: { state: "PROCESS_ALIVE_STATE", measured: ["alive", "gone"] },
  probe_process_image: { state: "PROCESS_IMAGE_STATE", measured: ["found", "none"] },
};

const PROBE_CALL = /^\s*(probe_(?:listener_port|process_alive|process_image))\b/;

export interface ProbeCallSite {
  probe: string;
  state: string;
  measured: [string, string];
  /** 1-based line number of the call itself. */
  line: number;
  /** Whether anything at all read this call's state variable. */
  read: boolean;
  window: string[];
}

export function probeCallSites(text: string): ProbeCallSite[] {
  const lines = text.split("\n");
  const indentOf = (line: string) => line.length - line.replace(/^\s*/, "").length;
  const sites: ProbeCallSite[] = [];

  for (let i = 0; i < lines.length; i += 1) {
    const call = lines[i].match(PROBE_CALL);
    if (!call) continue;
    const probe = call[1];
    const { state, measured } = PROBE_STATES[probe];
    const window = [lines[i]];
    let read = false;
    let j = i + 1;

    while (j < lines.length) {
      const line = lines[j];
      const trimmed = line.trim();
      if (trimmed === "" || trimmed.startsWith("#")) {
        window.push(line);
        j += 1;
        continue;
      }
      // Two probes back to back, their answers weighed together afterwards.
      // Only before this call's own answer has been read: once it has been, the
      // next probe call is the next SITE, and its branches are not this one's.
      if (PROBE_CALL.test(line) && !read) {
        window.push(line);
        j += 1;
        continue;
      }
      if (!line.includes(state)) break;
      read = true;
      if (/^\s*case\b/.test(line)) {
        const indent = indentOf(line);
        while (j < lines.length) {
          window.push(lines[j]);
          const done = indentOf(lines[j]) === indent && /^\s*esac\b/.test(lines[j]);
          j += 1;
          if (done) break;
        }
        continue;
      }
      if (/^\s*(?:el)?if\b/.test(line)) {
        let depth = 0;
        while (j < lines.length) {
          window.push(lines[j]);
          if (/^\s*if\b/.test(lines[j])) depth += 1;
          if (/^\s*fi\b/.test(lines[j])) depth -= 1;
          j += 1;
          if (depth === 0) break;
        }
        continue;
      }
      // Something else reads it — a bare `[[ "$STATE" == … ]]`, an assignment.
      // One line, then re-judge.
      window.push(line);
      j += 1;
    }

    sites.push({ probe, state, measured, line: i + 1, read, window });
  }
  return sites;
}

/**
 * Whether a call site's window branches on the third state.
 *
 * An explicit `unmeasured` arm or comparison counts. A `*)` catch-all counts
 * too, but ONLY when both measured outcomes are spelled out above it —
 * otherwise it is the two-state shape again, with the negative folded in with
 * the failure. Prose does not count: `none) fail "… no image — unmeasured, not
 * a match"` is a message, and a message in the negative arm must never be able
 * to stand in for the arm that is missing.
 */
export function branchesOnUnmeasured(site: ProbeCallSite): boolean {
  const body = site.window.join("\n");
  const arm = (label: string) =>
    new RegExp(String.raw`^\s*(?:[^)\n]*\|\s*)?${label}\s*\)`, "m").test(body);
  const compared = (label: string) => new RegExp(String.raw`(?:==|!=|=)\s*"?${label}"?`).test(body);
  if (arm("unmeasured") || compared("unmeasured")) return true;
  return (
    /^\s*\*\)/m.test(body) && site.measured.every((label) => arm(label) || compared(label))
  );
}
