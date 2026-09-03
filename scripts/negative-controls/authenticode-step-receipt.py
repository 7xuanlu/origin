"""Receipt for the release Authenticode gate's BEHAVIOURAL contract.

Round 13 finding 1: `Verify the installer carries a valid SignPath signature`
was held by substring markers only. `exit 0` at the top of its body leaves
every marker in place, satisfies the static contract, and publishes an unsigned
installer -- the same failed-measurement-reads-as-a-pass shape the whole
workstream is about, sitting in the gate meant to catch it.

Round 13b reopened it. The fix above was itself the same defect one level up:
when the signed-binary fixture was unavailable -- which is the state of the
Ubuntu `docs` job that actually runs this suite in CI, via
`validate-versions.test.sh` at ci.yml:2169 -- the harness printed UNCHECKED,
skipped EVERY mutation, and exited 0. A gate that cannot run reported as a gate
that ran.

Round 13e reopened it a third time, one level further out again: the arm that
decides when an unmeasured row may be EXCUSED was itself read off an exit
status alone, so a probe that crashed with status 2 was classified as a host
that cannot do Authenticode -- and the receipt could not have seen it, because
the receipt produced the arms by substituting the classification's own answer.

So this receipt now measures seven things, not one:

  1. the truth table, run for real against produced signature states;
  2. every mutation a row on THIS host can catch, caught;
  3. the degraded host: with the fixture removed, the `missing` row still runs,
     still catches the top-of-body `exit 0`, and the mutations it cannot catch
     are named UNCHECKED rather than skipped in silence;
  4. the metadata gate: `continue-on-error: true` on any signing step -- as a
     token AND as a decoded YAML property, which is the only form that can see
     a merge key -- and the removal of `shell: pwsh` or the SIGNPATH_CONFIGURED
     guard, all rejected: none of which any body-level assertion can see; plus
     the gate's own no-parser fallback, driven with PyYAML forced absent,
     because the CI lane that runs this suite installs nothing and a fallback
     that reports "clean" about an input it cannot read is this workstream's
     defect wearing a dependency's clothes;
  5. environment invariance: the same row run under the release's environment,
     under none of it and under all of it wrong, which is the only form of this
     property that a drive name assembled at runtime cannot walk past;
  6. the capability probe's own classification, driven by a stub shell rather
     than by substituting its answer, including every way it can fail to
     answer at all;
  7. the suite's refusal on all four arms of that classification.

Run: python3 scripts/negative-controls/authenticode-step-receipt.py
"""

from __future__ import annotations

import atexit
import importlib.util
import re
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "scripts" / "release-workflow-contract.test.py"

#: The terminal completion marker. Last line, unconditionally. Every early
#: `sys.exit` above it is a refusal that this receipt could not be produced --
#: which is what the aggregate runner needs to be able to tell apart from a
#: receipt that finished and found nothing.
MARKER = "NEGATIVE-CONTROL COMPLETE"
HARNESS = "authenticode-step-receipt.py"
STARTED = time.time()
_ABORT_STARTED = time.time()


#: ROUND 3 (Codex Sol), FINDING N7. Set immediately before the completion
#: marker is printed. Every OTHER way out of this file -- an early
#: `sys.exit("...")` refusal, an unhandled exception, a signal, a watchdog kill
#: -- leaves it False, and the handler below says so. A transcript that simply
#: stops is the one thing a reader cannot tell from a transcript that finished.
_COMPLETED = False


@atexit.register
def _abort_marker():
    if _COMPLETED:
        return
    # stderr first: an early `sys.exit("...")` writes its message there, and the
    # aggregate runner reads the LAST non-empty line of the two merged.
    sys.stderr.flush()
    print('NEGATIVE-CONTROL ABORTED %s elapsed=%.1fs'
          % (HARNESS, time.time() - _ABORT_STARTED))
    print('  This run did not reach its own summary. Nothing above it is a '
          'result about this harness.')
    sys.stdout.flush()


spec = importlib.util.spec_from_file_location("release_contract_test", TEST)
if spec is None or spec.loader is None:
    sys.exit(f"cannot load {TEST}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
failures: list[str] = []

#: The two subjects, as they were when this run started. This takes about six
#: minutes, which is long enough for another lane to land an edit halfway
#: through -- and then the arms scored before it and the arms scored after it
#: are about different files, with nothing in the transcript to say so. That is
#: not hypothetical here: the sibling POSIX harness aborted twice for exactly
#: this reason while these were being repaired. Checked once at the end, because
#: a change anywhere in the run invalidates the whole run, not part of it.
#:
#: ROUND 4. These are compared as BYTES, and that is not fastidiousness. The
#: obvious spelling -- `read_text()` on both sides -- reads in text mode, and
#: text mode applies universal newlines: it turns \r\n into \n before the
#: comparison ever happens. A guard written that way cannot see a line-ending
#: conversion at all, and it reports "unchanged" over a file whose bytes on
#: disk are different. That is this directory's own defect, a failed
#: measurement indistinguishable from a negative one, inside the guard written
#: to catch exactly that.
#:
#: It is also not theoretical for these two files. Another lane rewrote six
#: files today through Python's text-mode write, which performs the same
#: translation in reverse, and `scripts/release-workflow-contract.test.py` --
#: one of the two subjects below -- went to 4695 CRLF line endings and back.
#: A text-mode guard would have called that run clean.
_SUBJECTS = {
    ".github/workflows/release.yml": (ROOT / ".github/workflows/release.yml").read_bytes(),
    "scripts/release-workflow-contract.test.py": TEST.read_bytes(),
}


def _subjects_unchanged() -> None:
    for rel, before in _SUBJECTS.items():
        now = (ROOT / rel).read_bytes()
        if now != before:
            detail = ""
            if now.replace(b"\r\n", b"\n") == before.replace(b"\r\n", b"\n"):
                crlf = b"\r\n"
                detail = (
                    f" Only the line endings moved: {before.count(crlf)} CRLF "
                    f"before, {now.count(crlf)} now."
                )
            sys.exit(
                f"FATAL: {rel} changed during the run. Arms scored before the "
                "edit and after it are about different files; nothing above "
                f"this line is a result about either.{detail}"
            )

shell = mod._powershell()
print(f"powershell: {shell or 'NONE'}")
if shell is None:
    sys.exit("UNCHECKED: no PowerShell on this host; the table cannot run here")

# ---------------------------------------------------------------- 1. the table
run = mod.authenticode_behaviour_violations(release)
rows = mod.AUTHENTICODE_TRUTH_TABLE
print(f"\ntruth table ({len(rows)} rows), against real signature states:")
for description, kind, status, required in rows:
    bad = [v for v in run.violations if description in v]
    skipped = [u for u in run.unchecked if description in u]
    mark = "UNCHECKED" if skipped else ("FAIL" if bad else "ok")
    print(f"  {mark:9} exit {status}  {kind:9} {description}")
    for line in bad + skipped:
        print(f"            {line}")
if run.violations:
    failures.append(f"{len(run.violations)} truth-table violation(s)")
print(f"  rows that ran: {sorted(run.ran) or 'none'}")

# ------------------------------------------------------------ 2. the mutations
print(f"\nmutations ({len(mod.AUTHENTICODE_MUTATIONS)}), on this host:")
for old, new, catchers, why in mod.AUTHENTICODE_MUTATIONS:
    live = catchers & run.ran
    if release.count(old) != 1:
        failures.append(f"stale mutation fixture: {why}")
        print(f"  STALE     {why}: matched {release.count(old)} times")
        continue
    if not live:
        print(f"  UNCHECKED {why}: needs one of {sorted(catchers)}")
        continue
    mutant = mod.authenticode_behaviour_violations(release.replace(old, new, 1))
    if not live <= mutant.ran:
        failures.append(f"mutation of {why} stopped rows from running")
        print(f"  BROKEN    {why}: {sorted(live - mutant.ran)} never ran")
    elif live <= mutant.failed:
        # EVERY claimed catcher failed, not merely one. The catcher set is what
        # the UNCHECKED accounting rests on, so it has to be exact: too
        # generous and a fixture-free host reports "enforced" over a mutation
        # nothing there can see.
        print(f"  caught    {why}  <- {sorted(live)}")
    else:
        failures.append(f"claimed catcher did not catch: {why}")
        print(f"  MISSED    {why}: {sorted(live - mutant.failed)} claim to catch it "
              f"and did not")

# ------------------------------------------------- 3. the degraded (Linux) host
#
# The Ubuntu lane has no Authenticode at all, so `_signed_fixture` answers
# "no-support" there -- the incapacity arm, not the broken-probe arm (round 13d
# replaced the OS proxy with that distinction). Reproduce that state here by
# stubbing it, and require that what CAN still be measured IS still measured.
# This is the round-13b finding: before the fix, this whole section was one
# silent `else`.
print("\ndegraded host (no Authenticode support, as on the Ubuntu `docs` lane):")
real_fixture = mod._signed_fixture
mod._signed_fixture = lambda shell, work: ("no-support", None, None)
try:
    bare = mod.authenticode_behaviour_violations(release)
    if bare.ran != {"missing"}:
        failures.append(f"degraded host ran {sorted(bare.ran)}, expected ['missing']")
    print(f"  rows that ran:       {sorted(bare.ran) or 'none'}")
    print(f"  rows unchecked:      {len(bare.unchecked)}")
    if bare.violations:
        failures.append("degraded host reported violations on unmutated release.yml")
        for line in bare.violations:
            print(f"  FAIL      {line}")
    checked = 0
    # ROUND 3 (Codex Sol), FINDING N8. The four refusal arms below used to
    # compare the number of mutations the suite named as unchecked against the
    # literal 3. That literal was written when the mutation table had four rows;
    # a fifth landed (the case-SENSITIVE publisher comparison) and the arms
    # started failing with "named 4, expected 3" -- a stale constant reporting
    # as a defect. The number is not a constant: it is exactly the set of
    # mutations no still-running row can catch, which is measured right here.
    UNCHECKABLE_WITHOUT_A_FIXTURE: list[str] = []
    for old, new, catchers, why in mod.AUTHENTICODE_MUTATIONS:
        live = catchers & bare.ran
        if not live:
            UNCHECKABLE_WITHOUT_A_FIXTURE.append(why)
            print(f"  UNCHECKED {why}")
            continue
        mutant = mod.authenticode_behaviour_violations(release.replace(old, new, 1))
        if live <= mutant.failed:
            checked += 1
            print(f"  caught    {why}  <- {sorted(live)}")
        else:
            failures.append(f"degraded host missed: {why}")
            print(f"  MISSED    {why}")
    if checked != 1:
        failures.append(
            f"degraded host enforced {checked} mutation(s); it must enforce exactly "
            "the one the `missing` row can catch, or CI's Ubuntu lane is a free pass"
        )
finally:
    mod._signed_fixture = real_fixture
if not UNCHECKABLE_WITHOUT_A_FIXTURE:
    # If nothing were unchecked on a fixture-free host, the refusal arms below
    # would be comparing two empty lists and could not fail. Say so rather than
    # print four `ok`s that mean nothing.
    failures.append(
        "a fixture-free host reported nothing unchecked; the four refusal arms "
        "below would then assert an empty set against an empty set"
    )

# ------------------------------------------------------------- 4. the metadata
print("\nstep metadata (invisible to every body-level assertion):")
if mod.authenticode_step_metadata_violations(release):
    failures.append("metadata gate rejects the shipped release.yml")
    for line in mod.authenticode_step_metadata_violations(release):
        print(f"  FAIL      {line}")
else:
    print("  ok        shipped release.yml is clean")
# Spliced inside the job, not into the whole file: several of these names also
# exist in the macOS and Linux jobs, and a whole-file replace would mutate
# whichever came first -- a mutation applied somewhere the contract does not
# look, reported as a contract that did not notice.
job_text = mod.job_body(release, "app-bundle-windows")
job_at = release.index(job_text)
job_end = job_at + len(job_text)

# ROUND 3 (Codex Sol), FINDING N8. This used to be `mod.SIGNING_STEPS` plus a
# hand-written UNLISTED_STEPS tuple whose own comment claimed it covered "every
# step in the job, not a hand-listed four". It was a hand-list of five, and it
# went stale the moment someone renamed a step: the concurrent SIGNPATH_REQUIRED
# work renamed "Check SignPath configuration is all-or-nothing" to "... , and
# present when required", and this receipt reported STALE for it -- which is the
# right noise, but only because the mismatch happened to be a rename rather than
# an ADDITION. A step added to the job would simply not have appeared here, and
# nothing would have said so.
#
# So the job is the authority for which steps exist. Every `- name:` in it gets
# the mutation, the list cannot fall behind the workflow, and a new step is
# covered on the day it lands.
STEP_NAME = re.compile(r"(?m)^      - name: (.+)$")
JOB_STEPS = STEP_NAME.findall(job_text)
if not JOB_STEPS:
    failures.append(
        "no steps parsed out of app-bundle-windows; the metadata section would "
        "have run zero mutations and reported a clean sweep"
    )
    print("  BROKEN    no `- name:` steps found in the job")
_dupes = sorted({n for n in JOB_STEPS if JOB_STEPS.count(n) > 1})
if _dupes:
    # Two steps with one name make `replace(..., 1)` mutate the first and leave
    # the second, so the row would report on a workflow nobody wrote.
    failures.append(f"duplicate step names in app-bundle-windows: {_dupes}")
    print(f"  BROKEN    duplicate step name(s): {_dupes}")
    JOB_STEPS = [n for n in JOB_STEPS if n not in _dupes]
# The suite keeps its own list of the steps whose failure must never be
# discarded. If one of those names is not a step in the job, the suite is
# guarding a step that no longer exists -- report it here rather than let the
# suite's rule quietly cover nothing.
_absent = [s for s in mod.SIGNING_STEPS if s not in JOB_STEPS]
if _absent:
    failures.append(
        f"the suite's SIGNING_STEPS names {_absent}, which are not steps in "
        "app-bundle-windows; its metadata rule guards nothing for them"
    )
    print(f"  STALE     SIGNING_STEPS entries absent from the job: {_absent}")
metadata_caught = 0
fallback_cases = 0
env_caught = 0
for name in JOB_STEPS:
    marker = f"- name: {name}\n"
    if job_text.count(marker) != 1:
        failures.append(f"stale continue-on-error fixture for {name!r}")
        print(f"  STALE     {name!r} matched {job_text.count(marker)} times in the job")
        continue
    # Round 13d: the rule recognised one YAML spelling of the key, so a quoted
    # key -- identical to YAML, invisible to the regex -- discarded the failure
    # of a signing step with every control still green.
    #
    # Round 13e, reopened finding 1, third bullet: `"continue-on-error"`
    # decodes to the same key and contains no matching token at all, because
    # YAML processes escapes inside a double-quoted scalar (YAML 1.2.2 5.7).
    # `lexical` records which rule is expected to do the work, so this row
    # cannot quietly be credited to the token scan.
    for spelling, label, lexical in (
        ("        continue-on-error: true\n", "", True),
        ('        "continue-on-error": true\n', "quoted ", True),
        ('        "continue-on-\\u0065rror": true\n', "escaped ", False),
    ):
        mutated = (
            release[:job_at]
            + job_text.replace(marker, marker + spelling, 1)
            + release[job_end:]
        )
        listed = "" if name in mod.SIGNING_STEPS else "  (not on the suite's own list)"
        seen_by_token = "continue-on-error" in mod.job_body(
            mutated, "app-bundle-windows"
        )
        if seen_by_token != lexical:
            failures.append(
                f"{label}continue-on-error on {name!r}: the token scan "
                f"{'does' if seen_by_token else 'does not'} see it, which is not "
                "what this row claims"
            )
            print(f"  BROKEN    {label}continue-on-error on {name!r}: wrong rule")
        elif mod.authenticode_step_metadata_violations(mutated):
            metadata_caught += 1
            print(f"  caught    {label}continue-on-error on {name!r}{listed}")
        else:
            failures.append(f"{label}continue-on-error accepted on {name!r}")
            print(f"  MISSED    {label}continue-on-error on {name!r}")

header = f"      - name: {mod.AUTHENTICODE_STEP}\n"
start = release.index(header) + len(header)
body = mod.named_step_body(mod.job_body(release, "app-bundle-windows"), mod.AUTHENTICODE_STEP)
end = release.index(body, start) + len(body)
region = release[start:end]
for line, why in (
    ("        shell: pwsh\n", "shell: pwsh removed"),
    ("        if: env.SIGNPATH_CONFIGURED == 'true'\n", "SIGNPATH_CONFIGURED guard removed"),
):
    mutated = release[:start] + region.replace(line, "", 1) + release[end:]
    if mod.authenticode_step_metadata_violations(mutated):
        metadata_caught += 1
        print(f"  caught    {why}")
    else:
        failures.append(f"accepted: {why}")
        print(f"  MISSED    {why}")

# Round 13e, reopened finding 1: the rule above is a scan for the TOKEN inside
# the job's text, and a YAML merge key puts the property on a step without the
# token appearing there. Both halves are measured here -- that the token scan
# is blind to it, which is why the parsed rule had to exist, and that the
# parsed rule catches it.
print("\ncontinue-on-error carried in by a merge key:")
merge_anchor = f"      - name: {mod.AUTHENTICODE_STEP}\n"
if release.count(merge_anchor) != 1:
    failures.append(f"stale merge-key anchor: matched {release.count(merge_anchor)}")
    print(f"  STALE     anchor matched {release.count(merge_anchor)} times")
else:
    for where, injected, prelude in (
        ("the verification step", merge_anchor + "        <<: *swallow\n",
         "x-swallow: &swallow\n  continue-on-error: true\n"),
        ("the app-bundle-windows job itself", merge_anchor,
         "x-swallow: &swallow\n  continue-on-error: true\n"),
    ):
        merged = prelude + release.replace(merge_anchor, injected, 1)
        if where.endswith("itself"):
            job_anchor = "  app-bundle-windows:\n"
            if merged.count(job_anchor) != 1:
                failures.append("stale job anchor for the merge-key control")
                print("  STALE     job anchor")
                continue
            merged = merged.replace(job_anchor, job_anchor + "    <<: *swallow\n", 1)
        blind = "continue-on-error" not in mod.job_body(merged, "app-bundle-windows")
        caught = [
            v for v in mod.authenticode_step_metadata_violations(merged)
            if "decodes with continue-on-error" in v
        ]
        if not blind:
            failures.append(f"merge-key control on {where} is not the case it claims")
            print(f"  BROKEN    {where}: the token IS in the job; this proves nothing")
        elif not caught:
            failures.append(f"a merge key put continue-on-error on {where} unnoticed")
            print(f"  MISSED    {where}")
        else:
            metadata_caught += 1
            print(f"  caught    {where}, with the token nowhere in the job text")
    # The parsed rule is the one that can be switched off by an absent
    # dependency, so say out loud which one ran.
    try:
        import yaml as _yaml  # noqa: F401

        print("  note      the parsed rule ran here; PyYAML is installed")
    except ImportError:  # pragma: no cover - reported, never silent
        failures.append("PyYAML is absent, so the parsed rule did not run at all")
        print("  UNCHECKED PyYAML is absent; only the token scan ran")

    # The parsed rule can be switched off by an absent dependency, and the CI
    # lane that runs this suite -- the Ubuntu `docs` job, through
    # validate-versions.test.sh -- installs nothing. So the fallback is itself a
    # measurement that has to be measured. Its failure mode is the whole
    # workstream's: reporting a clean job it could not read. It names the two
    # constructs the token scan is blind to, and this section drives each of
    # them with the parser forced absent.
    #
    # `sys.modules["yaml"] = None` makes `import yaml` raise ImportError inside
    # the function without unloading the real module from anywhere else.
    print("\nthe no-parser fallback (PyYAML forced absent):")
    escaped_job = release.replace(
        merge_anchor,
        merge_anchor + '        "continue-on-\\u0065rror": true\n',
        1,
    )
    alias_job = "x-swallow: &swallow\n  continue-on-error: true\n" + release.replace(
        merge_anchor, merge_anchor + "        <<: *swallow\n", 1
    )
    _real_yaml = sys.modules.get("yaml", "absent")
    sys.modules["yaml"] = None  # type: ignore[assignment]
    try:
        for label, text, want_report in (
            ("the shipped file", release, False),
            ("a job carrying a merge key", alias_job, True),
            ("a job with a backslash-escaped key", escaped_job, True),
        ):
            body = mod.job_body(text, "app-bundle-windows")
            reported = [
                v
                for v in mod.continue_on_error_violations(text, body)
                if "unmeasured, not clean" in v
            ]
            if bool(reported) != want_report:
                failures.append(
                    f"the no-parser fallback {'stayed silent about' if want_report else 'complained about'} "
                    f"{label}"
                )
                print(
                    f"  {'MISSED' if want_report else 'FALSE+':9s} {label}: "
                    f"{len(reported)} report(s), wanted {'one' if want_report else 'none'}"
                )
            else:
                # Counted apart from metadata_caught: these are not mutations of
                # the workflow that the gate caught, they are the fallback
                # refusing to report on an input it cannot read. Folding them in
                # would make the summary's arithmetic a total nobody can check.
                fallback_cases += 1
                print(
                    f"  {'caught' if want_report else 'silent':9s} {label}"
                    + ("" if want_report else "  <- nothing here needs a parser")
                )
    finally:
        if _real_yaml == "absent":
            sys.modules.pop("yaml", None)
        else:
            sys.modules["yaml"] = _real_yaml  # type: ignore[assignment]

# The one mutation the truth table structurally CANNOT see, because the table
# is what supplies the environment: a branch on a variable that is unset here
# and set in the release. Every row green here, and a gate that never runs where
# it matters. Caught statically, so a fixture-free host catches it too.
print("\nenvironment-keyed early returns (invisible to a table that sets the environment):")
anchor = '          $ErrorActionPreference = "Stop"\n'
if region.count(anchor) != 1:
    failures.append(f"stale insertion anchor: matched {region.count(anchor)} times")
    print(f"  STALE     anchor matched {region.count(anchor)} times")
else:
    for inserted, why in (
        ("          if ($env:SIGNPATH_CONFIGURED -eq 'true') { exit 0 }\n",
         "exit 0 under the release's own guard variable"),
        ("          if ($env:CI -eq 'true') { exit 0 }\n",
         "exit 0 on a CI runner"),
        ("          if ($env:RUNNER_OS -eq 'Windows') { exit 0 }\n",
         "exit 0 on the only OS this job runs on"),
        ("          if ($env:GITHUB_JOB -eq 'app-bundle-windows') { exit 0 }\n",
         "exit 0 in the only job that runs it"),
        ("          Write-Host \"ref $env:GITHUB_REF\"\n",
         "any other read of the environment at all"),
        # Round 13d: the rule was a case-sensitive regex for ONE spelling, so
        # every one of these walked past it. They are the same read.
        ("          if ($Env:SIGNPATH_CONFIGURED -eq 'true') { exit 0 }\n",
         "the same read spelled $Env:"),
        ("          if (${env:SIGNPATH_CONFIGURED} -eq 'true') { exit 0 }\n",
         "the same read spelled ${env:...}"),
        ("          if ((Get-Item Env:SIGNPATH_CONFIGURED).Value -eq 'true') { exit 0 }\n",
         "the same read through the Env: drive"),
        ("          if ([System.Environment]::GetEnvironmentVariable("
         "'SIGNPATH_CONFIGURED') -eq 'true') { exit 0 }\n",
         "the same read through [System.Environment]"),
    ):
        mutated = (
            release[:start] + region.replace(anchor, anchor + inserted, 1) + release[end:]
        )
        if mod.authenticode_step_metadata_violations(mutated):
            env_caught += 1
            print(f"  caught    {why}")
        else:
            failures.append(f"accepted: {why}")
            print(f"  MISSED    {why}")
    # And the environment the rows DO run in is the release's, so even without
    # the static scan the first of those would now be caught by every row.
    guarded = (
        release[:start]
        + region.replace(
            anchor,
            anchor + "          if ($env:SIGNPATH_CONFIGURED -eq 'true') { exit 0 }\n",
            1,
        )
        + release[end:]
    )
    run_guarded = mod.authenticode_behaviour_violations(guarded)
    if run.ran and run.ran <= run_guarded.failed:
        print(f"  caught    the same mutation at runtime  <- {sorted(run.ran)}")
    else:
        failures.append(
            "the truth table ran the body without the release job's environment; "
            f"rows that failed: {sorted(run_guarded.failed) or 'none'}"
        )
        print("  MISSED    the same mutation at runtime")

# ------------------------- 5. the property no reading of the body can settle
#
# Round 13e, reopened finding 1 and new finding 3: every guard on the body was
# lexical, and PowerShell can assemble the drive name at runtime. Nothing that
# reads text catches `('E' + 'nv:GITHUB_WORKFLOW_REF')`. Running the step three
# times under three environments does, and `lexical` below records which rule is
# expected to do the work so the semantic one cannot be credited for a catch the
# scan made anyway.
print("\nenvironment invariance, by running the step (not by reading it):")
base_invariance = mod.authenticode_environment_invariance(release)
if base_invariance.violations:
    failures.append("the shipped step's outcome already depends on the environment")
    for line in base_invariance.violations:
        print(f"  FAIL      {line}")
else:
    print(f"  ok        shipped release.yml is invariant over {base_invariance.ran}")
for line in base_invariance.unchecked:
    print(f"  UNCHECKED {line}")
if not base_invariance.ran:
    failures.append("environment invariance ran no rows at all")
for inserted, why, lexical in (
    ("          if ($env:SIGNPATH_CONFIGURED -eq 'true') { exit 0 }\n",
     "the plain spelling, which the scan also catches", True),
    ("          if ((Get-Item ('E' + 'nv:GITHUB_WORKFLOW_REF')).Value) { exit 0 }\n",
     "a drive name assembled at runtime", False),
    ("          $n = [char]69 + 'nv:CI'; if ((Get-Item $n).Value) { exit 0 }\n",
     "a drive name built from a character code", False),
    ("          $d = 'E'; $d += 'nv:GITHUB_JOB'; if (Test-Path $d) { exit 0 }\n",
     "a drive path accumulated into a variable and tested, never read", False),
):
    if region.count(anchor) != 1:
        break
    mutated = release[:start] + region.replace(anchor, anchor + inserted, 1) + release[end:]
    seen_by_scan = bool(mod.authenticode_body_env_violations(mutated))
    caught = bool(mod.authenticode_environment_invariance(mutated).violations)
    if seen_by_scan != lexical:
        failures.append(
            f"{why}: the lexical scan {'does' if seen_by_scan else 'does not'} "
            "see it, which is not what this row claims"
        )
        print(f"  BROKEN    {why}: wrong rule")
    elif not caught:
        failures.append(f"environment invariance did not notice {why}")
        print(f"  MISSED    {why}")
    else:
        env_caught += 1
        print(f"  caught    {why}"
              + ("" if lexical else "  <- only the running property sees this"))


def _stub_shell(directory: Path, name: str, lines: tuple[str, ...], status: int) -> str:
    """A stand-in for PowerShell that answers with exactly these bytes.

    Round 13e, new finding 4. Every arm below used to be produced by replacing
    `_signed_fixture` outright with `return ("no-support", None, None)`, which
    measures the CALLER's handling of an answer nobody computed. The thing that
    was wrong lived inside the replaced function -- it read the arm off an exit
    status alone, so a probe that crashed with status 2 arrived spelled
    "no-support", the one answer allowed to excuse an unmeasured row -- and a
    receipt that substitutes the answer could not have seen it.

    What a host actually supplies is a PowerShell. That is the input, so that
    is what is stubbed, and the classification under test is the shipped one.
    """
    path = directory / f"{name}.cmd"
    body = "@echo off\r\n" + "".join(f"{line}\r\n" for line in lines)
    # write_BYTES, not write_text. On Windows, text mode translates every "\n"
    # on its way to disk, and `body` already carries CRLF, so the obvious
    # spelling wrote "\r\r\n" on every line: measured on this host, 56 bytes
    # where 53 were meant, three doubled terminators in a four-line file. The
    # docstring above says this stub "answers with exactly these bytes"; text
    # mode made that sentence false. cmd.exe tolerates the doubling, which is
    # exactly why it survived -- the harness stayed green while the file on
    # disk was not the file this function claims to write. The same reasoning
    # as the byte-comparison guard at the top: where the bytes are the point,
    # do not let the runtime edit them.
    path.write_bytes((body + f"exit /b {status}\r\n").encode("ascii"))
    return str(path)


# (description, what the stub prints, its exit status, the capability it must
# be classified as). The four rows that must come out "probe-failed" are the
# finding: every one of them was "no-support" under the status-only rule.
CLASSIFICATION = (
    ("the cmdlet is absent", ("echo NO-SUPPORT cmdlet-absent",), 2, "no-support"),
    ("the cmdlet throws on this platform",
     ("echo NO-SUPPORT cmdlet-unusable",), 2, "no-support"),
    ("the search found nothing signed", ("echo NONE-FOUND",), 1, "none-found"),
    ("a signed binary was found",
     ("echo FIXTURE\tC:\\w\\stub.exe\tContoso Ltd",), 0, "fixture"),
    ("the probe died saying nothing", (), 2, "probe-failed"),
    ("the probe died with a message",
     ("echo unrecognized parameter 1>&2",), 2, "probe-failed"),
    ("the shell refused the command line", (), 64, "probe-failed"),
    ("the status and the token disagree",
     ("echo NO-SUPPORT cmdlet-absent",), 0, "probe-failed"),
    ("a truncated fixture line", ("echo FIXTURE\tC:\\w\\stub.exe",), 0, "probe-failed"),
)

print("\nthe capability probe's own classification, against a stub shell:")
with tempfile.TemporaryDirectory() as tmp:
    stub_dir = Path(tmp) / "shells"
    stub_dir.mkdir()
    for index, (why, lines, status, want) in enumerate(CLASSIFICATION):
        stub = _stub_shell(stub_dir, f"stub{index}", lines, status)
        with tempfile.TemporaryDirectory() as work:
            capability, path, publisher = mod._signed_fixture(stub, work)
        if capability != want:
            failures.append(
                f"{why}: classified {capability!r}, expected {want!r}"
            )
            print(f"  MISSED    {why}: {capability!r}, expected {want!r}")
        elif want == "fixture" and (path != "C:\\w\\stub.exe" or publisher != "Contoso Ltd"):
            failures.append(f"{why}: parsed {path!r}/{publisher!r}")
            print(f"  MISSED    {why}: parsed {path!r} / {publisher!r}")
        else:
            print(f"  ok        exit {status:<3} -> {capability:<12} {why}")
    # And the shell that is not there at all: `subprocess` raises rather than
    # returning a status, and an exception is not an incapacity either.
    missing_shell = str(Path(tmp) / "no-such-shell.cmd")
    with tempfile.TemporaryDirectory() as work:
        capability, _, reason = mod._signed_fixture(missing_shell, work)
    if capability != "probe-failed":
        failures.append(f"an unlaunchable shell classified {capability!r}")
        print(f"  MISSED    the shell could not be started: {capability!r}")
    else:
        print(f"  ok        unlaunchable -> probe-failed  ({str(reason)[:60]}...)")

# ------------------------------- 6. UNCHECKED is fatal where it has no excuse
#
# In-process classification proves the measurement; this proves the SUITE
# refuses. A copy of the shipped test pointed at a stub shell must exit
# non-zero on every arm that is not a genuine incapacity -- printing UNCHECKED
# and exiting 0 is the defect, not the report of it.
#
# The predicate's arms are the answers a real host can give, and here they are
# produced the way a real host produces them: by what its PowerShell does. Only
# the SHELL is substituted; `_signed_fixture` classifies it for itself, and the
# truth table's rows keep the real PowerShell so the `missing` row still runs.
#
#   broken probe     -- the cmdlet works and a search of the system binary
#                       directories still found nothing signed. That is a broken
#                       probe or trust store on ANY OS, so unchecked rows are a
#                       failure and the refusal has to say so;
#   probe blew up    -- the probe did not answer at all. Nothing here knows
#                       whether this host can measure Authenticode, so it may
#                       not be treated as a host that cannot;
#   opted-in lane    -- a real incapacity, but with WENLAN_REQUIRE_AUTHENTICODE=1
#                       the lane declared it arranged a fixture, so unchecked
#                       fails too;
#   genuinely unable -- a real incapacity and no such declaration: Authenticode
#                       does not exist on this platform, so the rows must be
#                       REPORTED and the run must exit 0. Hard-failing it would
#                       make the gate unrunnable rather than honest.
print("\nthe suite's refusal, all four arms:")
with tempfile.TemporaryDirectory() as tmp:
    src = TEST.read_text(encoding="utf-8")
    anchor = "def _signed_fixture(shell: str, work: str) -> FixtureResult:\n"
    # The copy lives outside the repo, so its __file__-relative REPO_ROOT would
    # point at the temp dir and every other contract in the suite would fail on
    # a missing file -- a failure that looks exactly like the one being tested.
    rooted = "REPO_ROOT = Path(__file__).resolve().parent.parent\n"
    if src.count(anchor) != 1 or src.count(rooted) != 1:
        failures.append("cannot build the stub-shell copy: an anchor is stale")
    else:
        rebased = src.replace(rooted, f"REPO_ROOT = Path(r{str(ROOT)!r})\n", 1)
        stub_dir = Path(tmp) / "shells"
        stub_dir.mkdir()

        def with_shell(lines: tuple[str, ...], status: int, name: str) -> str:
            """The shipped `_signed_fixture`, handed a shell that answers thus."""
            stub = _stub_shell(stub_dir, name, lines, status)
            return rebased.replace(anchor, anchor + f"    shell = {stub!r}\n", 1)

        def run_copy(name: str, text: str, env: dict[str, str]):
            copy = Path(tmp) / name
            # Bytes for the same reason as the stub above. This copy IS the
            # subject of the four arms below, and text mode would rewrite its
            # line endings between `text` and the file the child runs. Nothing
            # here compares it byte for byte today -- Python parses CRLF and LF
            # alike, so this is a latent hole rather than a live defect -- but a
            # harness whose subject is silently re-encoded on the way to disk
            # has no business calling itself a receipt.
            copy.write_bytes(text.encode("utf-8"))
            return subprocess.run(
                [sys.executable, str(copy)],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(ROOT),
                env={**os.environ, **env},
            )

        # ROUND 3 (Codex Sol), FINDING N6, first half. A GREEN BASELINE for the
        # copy machinery, in this same run. Without it, "the copy exited 1" is
        # satisfied by a rebased path that broke some other contract, by a
        # concurrent edit to release.yml, by anything at all -- and every arm
        # below would be credited for a red it did not cause.
        baseline_proc = run_copy(
            "baseline-unsubstituted.py", rebased, {"WENLAN_REQUIRE_AUTHENTICODE": ""}
        )
        baseline_out = baseline_proc.stdout + baseline_proc.stderr
        if baseline_proc.returncode != 0:
            failures.append(
                "the UNSUBSTITUTED copy of the suite did not pass "
                f"(exit {baseline_proc.returncode}); the arms below would be "
                "scored against a copy that was already red"
            )
            print(
                f"  MISSED    baseline: the copy exited {baseline_proc.returncode} "
                "with no shell substituted"
            )
            print(f"            {baseline_out.strip()[-400:]!r}")
        elif "PASS: release promotion" not in baseline_out:
            failures.append(
                "the unsubstituted copy exited 0 without its terminal PASS line; "
                "it did not run to the end"
            )
            print("  BROKEN    baseline: exited 0 with no terminal PASS line")
        else:
            print("  ok        baseline: the unsubstituted copy passes in this run")

        def terminating_assertion(text: str) -> str | None:
            """The message of the AssertionError the process DIED on.

            FINDING N6, second half. The arms used to accept "the copy exited
            non-zero AND the expected words appear somewhere in its output".
            The suite prints `UNCHECKED: ...` lines on its way past a row it
            could not build, so the words can be present while the exit came
            from an entirely different contract. The exception that terminated
            the process is the one thing that cannot be true of an unrelated
            failure.
            """
            marker = "\nAssertionError: "
            if marker not in text:
                return None
            return text.rsplit(marker, 1)[-1]

        arms = (
            (
                "broken probe",
                "no-fixture-broken.py",
                with_shell(("echo NONE-FOUND",), 1, "arm-none-found"),
                {"WENLAN_REQUIRE_AUTHENTICODE": ""},
                1,
                "probe is broken",
            ),
            (
                "probe blew up",
                "no-fixture-crashed.py",
                with_shell((), 2, "arm-crashed"),
                {"WENLAN_REQUIRE_AUTHENTICODE": ""},
                1,
                "did not answer at all",
            ),
            (
                "opted-in lane",
                "no-fixture-required.py",
                with_shell(("echo NO-SUPPORT cmdlet-absent",), 2, "arm-required"),
                {"WENLAN_REQUIRE_AUTHENTICODE": "1"},
                1,
                "only partly measured",
            ),
            (
                "genuinely unable",
                "no-fixture-unable.py",
                with_shell(("echo NO-SUPPORT cmdlet-absent",), 2, "arm-unable"),
                {"WENLAN_REQUIRE_AUTHENTICODE": ""},
                0,
                "UNCHECKED: Authenticode step:",
            ),
        )
        for arm, name, text, env, want_rc, needle in arms:
            proc = run_copy(name, text, env)
            out = proc.stdout + proc.stderr
            cause = terminating_assertion(out)
            if proc.returncode != want_rc:
                failures.append(
                    f"{arm}: exited {proc.returncode}, expected {want_rc}"
                )
                print(f"  MISSED    {arm}: exited {proc.returncode}, expected {want_rc}")
                print(f"            {out.strip()[-400:]!r}")
                continue
            if want_rc != 0:
                # The refusal under examination has to be the thing that ENDED
                # the process, not a phrase that happened to be printed on the
                # way to some other failure.
                if cause is None:
                    failures.append(
                        f"{arm}: exited {proc.returncode} without an AssertionError; "
                        "it died rather than refused, and nothing ties the status "
                        "to the refusal under test"
                    )
                    print(f"  BROKEN    {arm}: no terminating AssertionError")
                    print(f"            {out.strip()[-400:]!r}")
                    continue
                if needle not in cause:
                    failures.append(
                        f"{arm}: the AssertionError that ended the run does not "
                        f"mention {needle!r}; the arm was credited for an "
                        "unrelated failure"
                    )
                    print(
                        f"  BROKEN    {arm}: died on {cause.strip()[:200]!r}, "
                        f"which is not {needle!r}"
                    )
                    continue
            else:
                # The one arm that must SUCCEED: it has to reach the end and
                # print the terminal PASS line, and it must not have refused.
                if cause is not None:
                    failures.append(
                        f"{arm}: exited 0 but an AssertionError was raised on the way"
                    )
                    print(f"  BROKEN    {arm}: {cause.strip()[:200]!r}")
                    continue
                if "PASS: release promotion" not in out:
                    failures.append(
                        f"{arm}: exited 0 without its terminal PASS line; it did "
                        "not run to the end"
                    )
                    print(f"  BROKEN    {arm}: no terminal PASS line")
                    continue
                if needle not in out:
                    failures.append(
                        f"{arm}: exited 0 without reporting the rows it could not "
                        f"measure ({needle!r} absent)"
                    )
                    print(f"  BROKEN    {arm}: {needle!r} not in the transcript")
                    continue
            named = sorted(m[3] for m in mod.AUTHENTICODE_MUTATIONS if m[3] in out)
            where = "the terminating AssertionError" if want_rc else "the transcript"
            print(
                f"  ok        {arm}: exited {proc.returncode}, {where} said "
                f"{needle!r}, named {len(named)} unchecked mutation(s)"
            )
            # Every one of these arms is a host with no signed fixture, which is
            # the state measured in section 3 -- so the mutations it must name
            # are exactly the ones that section found no running row can catch.
            # Comparing the NAMES, not just the count, means an arm that names
            # the right number of the wrong mutations is still a failure.
            if named != sorted(UNCHECKABLE_WITHOUT_A_FIXTURE):
                failures.append(
                    f"{arm}: named {named} as unchecked; without a fixture the "
                    f"unchecked set is {sorted(UNCHECKABLE_WITHOUT_A_FIXTURE)}"
                )

_subjects_unchanged()
if failures:
    for line in failures:
        print(f"\nFAILURE: {line}")
    print(f"\nRECEIPT: {len(failures)} failure(s)")
    globals()["_COMPLETED"] = True
    print(f"{MARKER} {HARNESS} failures={len(failures)} "
          f"elapsed={time.time() - STARTED:.1f}s")
    sys.exit(1)
# Round 13e, new finding 5: the old summary claimed "no host here has a
# SignPath-signed file", which is not something this receipt measured. The
# probe does not look for a SignPath-signed file and would not know one; it
# takes the FIRST validly signed binary it finds in the system directories and
# reports whatever publisher that binary carries. The substitution is made
# because that publisher is not SignPath Foundation, which is a different and
# much weaker statement than the one the summary used to make.
steps_checked = len(JOB_STEPS)
# Round 13f: derived, not restated. The count of substituted rows was written as
# a literal `1` when `accepted` was the only one; adding the `casefolded` row
# made the receipt claim one more verbatim row than actually ran verbatim --
# the same "a claim that outlives what was measured" defect this directory
# exists to catch, in the receipt that reports the catching. The suite owns the
# list; read it from there so a new substituted row cannot go unreported.
substituted = sum(1 for row in rows if row[1] in mod.AUTHENTICODE_SUBSTITUTED_KINDS)
print(
    f"\nRECEIPT: {len(rows) - substituted} rows verbatim + {substituted} with the "
    "expected publisher substituted (the probe takes the first validly signed "
    f"binary it finds, and that binary's publisher is not SignPath Foundation), "
    f"{len(mod.AUTHENTICODE_MUTATIONS)} mutations, degraded host still enforcing, "
    f"{metadata_caught} metadata mutations caught (every one of the "
    f"{steps_checked} steps the job declares, enumerated from it rather than "
    f"hand-listed, x 3 "
    f"spellings of continue-on-error = {steps_checked * 3} (plain, quoted, and "
    "a unicode escape only the decoded document can see), plus 2 carried in "
    "by a merge key, plus `shell: pwsh` and the SIGNPATH_CONFIGURED guard), "
    f"{fallback_cases} no-parser fallback cases (it reports "
    "'unmeasured, not clean' for an alias and for an escaped key, and stays "
    "silent on the shipped file), "
    f"{env_caught} environment-keyed early returns "
    f"caught, {len(CLASSIFICATION) + 1} capability classifications and all 4 "
    "arms of the suite's refusal correct -- each tied to the AssertionError "
    "that ended its run, off an unsubstituted copy that passed in this same run"
)
_COMPLETED = True
print(f"{MARKER} {HARNESS} failures=0 elapsed={time.time() - STARTED:.1f}s")
