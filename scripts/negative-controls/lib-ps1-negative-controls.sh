#!/usr/bin/env bash
# Negative controls for scripts/first-run/lib.test.ps1.
#
# Green cases prove nothing on their own: a suite that would pass with the fix
# reverted is not measuring the fix. Each control below copies the shipped
# scripts/first-run tree twice, reverts exactly ONE property in one copy, and
# requires four things of the run:
#
#   1. the UNMUTATED copy passes, in this same harness (otherwise "it went red"
#      only proves that something broke);
#   2. the mutated copy exits non-zero;
#   3. it printed its terminal summary line, so it FAILED rather than DIED --
#      the first runs of nc2 and nc3 died on a NativeCommandError and would
#      otherwise have been reported as controls that fired;
#   4. exactly the named cases failed, no more, so the control is pinned to the
#      property it defends rather than to collateral damage.
#
# A control that passes is itself a failure.
#
# Run: bash scripts/negative-controls/lib-ps1-negative-controls.sh
set -u

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
src="$root/scripts/first-run"
# The subject must not move under the run. Each control re-copies $src, so an
# edit landing halfway through means the first controls scored one lib.ps1 and
# the rest scored another, and the transcript says nothing about which. That is
# not hypothetical on this track: another lane edited scripts/lib/host-process.sh
# twice during a twenty-five-minute POSIX run, and only that harness's identical
# guard kept the result from being reported as a clean sweep. Only the two files
# this harness actually drives are pinned -- an unrelated edit to windows-zip.ps1
# in the same directory must not make this unrunnable.
subject="$src/lib.ps1"
suite="$src/lib.test.ps1"
subject_before="$(cat "$subject" "$suite")" || exit 1
assert_subject_unchanged() {
    if [ "$(cat "$subject" "$suite")" != "$subject_before" ]; then
        echo "FATAL: scripts/first-run/lib.ps1 or lib.test.ps1 changed during the run${1:+ (before $1)}"
        echo "  Controls scored before the edit and after it are about different"
        echo "  files. Nothing above this line is a result about either."
        exit 1
    fi
}
# Transcripts go under the gitignored target/, not next to this file: the
# harness now lives in the tracked tree (round 13c, new finding 4) and a control
# that litters the checkout it controls is one people delete rather than run.
logs="$root/target/negative-control-logs"
mkdir -p "$logs" || exit 1
work="$(mktemp -d "${TMPDIR:-/tmp}/libps1-nc.XXXXXX")" || exit 1

failures=0
reached_end=0
# ROUND 3 (Codex Sol), FINDING N7. reached_end already made an early death
# non-zero; it did not make one DISTINGUISHABLE in the transcript. The marker
# below is the last line of a completed run and appears nowhere else, so the
# aggregate runner can tell "scored every control, found none broken" from
# "was killed before it got there" -- which no exit status alone survives, since
# a watchdog's kill and a control's failure are both non-zero.
MARKER="NEGATIVE-CONTROL COMPLETE"
HARNESS="lib-ps1-negative-controls.sh"
started=$SECONDS
finish() {
    rc=$?
    rm -rf -- "$work"
    if [ "$reached_end" = 1 ]; then
        echo "$MARKER $HARNESS failures=$failures elapsed=$((SECONDS - started))s"
    else
        # Keep the status that killed the run -- 137 for a SIGKILL, 9 for a
        # watchdog exit -- so a supervisor reads the same number the transcript
        # does. Only an abort that somehow ended 0 is rewritten, because a run
        # that did not score every control must never exit 0.
        #
        # ROUND 4: the rewrite used to happen AFTER the line was printed, so on
        # the one path where it matters the transcript said `rc=0` and the
        # process exited 1 -- the exact confusion the line exists to prevent,
        # and it is not theoretical: `timeout` SIGTERMs this harness and bash
        # enters the trap with $? = 0, which is how it was found. The number is
        # settled first and the line reports the status actually exited with;
        # when it was rewritten, the line says so rather than leaving a reader
        # to reconcile two numbers.
        local inherited="$rc"
        if [ "$rc" = 0 ]; then rc=1; fi
        echo "NEGATIVE-CONTROL ABORTED $HARNESS rc=$rc elapsed=$((SECONDS - started))s"
        if [ "$inherited" != "$rc" ]; then
            echo "  (the run ended with status $inherited; a run that did not score"
            echo "   every control must not exit 0, so this exits $rc)"
        fi
    fi
    exit "$rc"
}
trap finish EXIT

fail() { echo "  FAIL $*"; failures=$((failures + 1)); }

# apply NAME PYTHON_SNIPPET  -- patches the copied lib.ps1 via an exact-match
# replacement that must hit exactly once, so a control can never silently
# inject nothing and then "pass" because the suite failed for another reason.
run_control() {
    local name="$1"; shift
    local patch="$1"; shift
    # remaining args: case names the suite MUST report as FAIL
    # ROUND 4. This used to be checked once, at the bottom of the file. A run
    # that never reaches the bottom therefore never checks at all -- and the
    # runs that do not reach it are exactly the long ones a watchdog kills, in
    # other words the ones with the most time for another lane to land an edit.
    # Found for real: a `timeout` killed this harness at nc9 while another lane
    # was rewriting lib.ps1's line endings mid-run, and the end-of-file check
    # never executed. Checked per control now, so the refusal names the control
    # it was about to score.
    assert_subject_unchanged "$name"
    local d="$work/$name"
    mkdir -p "$d" "$d.baseline" || { fail "$name: cannot create $d"; return; }
    cp -r "$src"/. "$d"/ || { fail "$name: copy failed"; return; }
    cp -r "$src"/. "$d.baseline"/ || { fail "$name: baseline copy failed"; return; }
    rm -rf "$d/__pycache__" "$d.baseline/__pycache__"

    # ROUND 4 (Codex Sol), FINDING C5.1. Each patch script checks its anchor
    # occurs exactly once and then writes `s.replace(old, new)`. Matching once
    # is not the same as changing something: if `old` and `new` are equal --
    # which is what a control written by copying the block above and editing
    # neither string looks like -- the anchor check passes, the write succeeds,
    # and the "mutated" copy is byte-for-byte the original. The suite then
    # passes and the control is reported as "the suite PASSED with the fix
    # removed", which names the suite for a defect in the control. Hashed here,
    # once, rather than in fifteen inline patch scripts that would each have to
    # remember.
    local before_hash after_hash
    before_hash="$(sha256sum "$d/lib.ps1" | cut -c1-64)"
    if ! python3 "$patch" "$d/lib.ps1"; then
        fail "$name: injection did not apply - the control tested nothing"
        return
    fi
    after_hash="$(sha256sum "$d/lib.ps1" | cut -c1-64)"
    if [ -z "$before_hash" ] || [ -z "$after_hash" ]; then
        fail "$name: could not hash the copy; a control that cannot be shown to mutate is not a control"
        return
    fi
    if [ "$before_hash" = "$after_hash" ]; then
        fail "$name: the injected copy is identical to lib.ps1 - this control reverts nothing"
        return
    fi

    # A green BASELINE first, from the same copied tree and the same harness.
    # Without it, "the suite went red" proves only that something broke -- a
    # stale copy, a missing file, a harness that died on line one would all
    # produce the same red.
    local base_log="$logs/$name.baseline.log"
    powershell -NoProfile -ExecutionPolicy Bypass -File "$d.baseline/lib.test.ps1" \
        >"$base_log" 2>&1
    local base_rc=$?
    if [ "$base_rc" -ne 0 ]; then
        fail "$name: the UNMUTATED copy did not pass (exit $base_rc); see $base_log"
        return
    fi
    if ! grep -q '^PASS: lib.test.ps1$' "$base_log"; then
        fail "$name: the unmutated copy never printed its terminal PASS line"
        return
    fi
    rm -f "$base_log"

    local log="$logs/$name.log"
    powershell -NoProfile -ExecutionPolicy Bypass -File "$d/lib.test.ps1" >"$log" 2>&1
    local rc=$?

    if [ "$rc" -eq 0 ]; then
        fail "$name: the suite PASSED with the fix removed - it is not measuring the fix"
        return
    fi

    # The suite must have RUN TO THE END. A harness that dies partway also exits
    # non-zero, and would otherwise be reported as a control that fired -- which
    # is what actually happened on the first run of nc2 and nc3.
    if ! grep -q '^FAIL: lib.test.ps1' "$log"; then
        fail "$name: the suite exited $rc without its terminal summary line; it died rather than failed"
        return
    fi

    local missing=0
    for case_name in "$@"; do
        if ! grep -Eq "^  FAIL $case_name\$" "$log"; then
            fail "$name: expected case '$case_name' to fail, it did not"
            missing=1
        fi
    done

    # And ONLY the named cases. A mutation that reddens the whole suite is not
    # pinned to the behaviour the control is defending.
    local unexpected
    unexpected="$(grep -E '^  FAIL ' "$log" | sed 's/^  FAIL //')"
    local got
    while IFS= read -r got; do
        [ -n "$got" ] || continue
        local wanted=0
        for case_name in "$@"; do
            [ "$got" = "$case_name" ] && wanted=1
        done
        if [ "$wanted" = 0 ]; then
            fail "$name: case '$got' also failed; the control is not pinned to the fix"
            missing=1
        fi
    done <<EOF
$unexpected
EOF

    [ "$missing" = 0 ] && echo "  ok   $name (baseline green; suite exited $rc; exactly $* failed)"
}

echo "lib-ps1-negative-controls"

# --- NC1: Evaluate stops asserting the declared row multiset ----------------
cat > "$work/nc1.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '    return (($fails -eq 0) -and ($missing -eq 0) -and ($surplus -eq 0))'
new = '    return (($fails -eq 0) -and ($surplus -eq 0))   # INJECTED nc1: missing rows tolerated'
if s.count(old) != 1:
    sys.exit('nc1 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
# carried-row-declared-the-old-way is on the roster because it is defended by
# this same return: it declares a row that is recorded above the mark and can
# therefore never be balanced, so its whole assertion is that the suite exits
# 1 for a MISSING row. Tolerate missing rows and it goes green -- correct
# blast radius, and naming it is what keeps the pin honest.
run_control nc1-evaluate-ignores-missing "$work/nc1.py" \
    one-missing only-info-rows shadowing-row-names \
    carried-row-declared-the-old-way

# --- NC2: the no-declaration guard is removed -------------------------------
cat > "$work/nc2.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '''    if ($expected.PSBase.Count -eq 0) {
        Write-Host "==> no contracted rows declared for $script:GauntletChannel in $script:GauntletExpectedFile : unchecked, not a pass"
        return $false
    }'''
new = '    # INJECTED nc2: an undeclared run is allowed through'
if s.count(old) != 1:
    sys.exit('nc2 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc2-no-declaration-guard "$work/nc2.py" no-declaration declared-by-another-channel-only

# --- NC3: Check-Helper calls the interpreter bare, as the channels used to ---
# Cut between two boundary lines rather than matching the whole block: the
# body of Check-Helper's Check keeps growing (-MustDeclare added a branch to
# it), and a whole-block anchor silently stops applying every time. The harness
# does refuse a control whose injection did not apply, so a stale anchor is
# reported rather than counted as a pass -- but it is still a control that
# tested nothing until someone retargets it.
cat > "$work/nc3.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
head = '    Check -Name $Name -Script {\n'
tail = '        Write-Output "$Interpreter $Path ran; output in $helperLog"\n    }\n'
if s.count(head) != 1 or s.count(tail) != 1:
    sys.exit('nc3 anchors matched %d/%d times' % (s.count(head), s.count(tail)))
start = s.index(head)
end = s.index(tail) + len(tail)
if end <= start:
    sys.exit('nc3 anchors are out of order')
new = ('    # INJECTED nc3: the pre-remedy shape - bare interpreter, laundered status\n'
       '    & $Interpreter @InterpreterArgs $Path\n'
       '    $global:LASTEXITCODE = 0\n')
io.open(p, 'w', encoding='utf-8', newline='').write(s[:start] + new + s[end:])
PY
# Blast radius is every helper case, by construction: nc3 deletes the whole
# Check wrapper. Each new helper case has to be added here, and the harness
# says so out loud when one is missing rather than passing quietly.
run_control nc3-bare-interpreter "$work/nc3.py" \
    helper-interpreter-missing helper-nonzero-exit helper-ok \
    helper-exits-0-declaring-nothing helper-declares-required-prefix \
    helper-reuses-stale-declaration helper-declaration-count-unreadable \
    declaration-count-lookup-answers-absent

# --- NC4: Evaluate goes back to $hash.Keys, which a row named Keys shadows ---
cat > "$work/nc4.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
subs = [
    ('foreach ($name in ($expected.PSBase.Keys | Sort-Object)) {',
     'foreach ($name in ($expected.Keys | Sort-Object)) {   # INJECTED nc4'),
    ('foreach ($name in ($actual.PSBase.Keys | Sort-Object)) {',
     'foreach ($name in ($actual.Keys | Sort-Object)) {   # INJECTED nc4'),
    ('if ($expected.PSBase.Count -eq 0) {',
     'if ($expected.Count -eq 0) {   # INJECTED nc4'),
]
for old, new in subs:
    if s.count(old) != 1:
        sys.exit('nc4 anchor matched %d times: %r' % (s.count(old), old[:50]))
    s = s.replace(old, new)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
PY
run_control nc4-hashtable-member-shadowing "$work/nc4.py" shadowing-row-names

# --- NC5: the declaration guard goes back to asking about the FILE ----------
# Found by Codex Sol in the Phase 4 review: GAUNTLET_OUT is one directory and
# the workflow header tells a human to reuse it channel after channel, so a
# file-level "is it non-empty" guard lets a channel that declared nothing ride
# on the previous channel's rows.
cat > "$work/nc5.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '    if ($expected.PSBase.Count -eq 0) {'
new = ('    if (-not (Test-Path $script:GauntletExpectedFile) -or '
       '-not (Get-Content $script:GauntletExpectedFile -ErrorAction SilentlyContinue)) {'
       '   # INJECTED nc5: back to asking the file, not this channel')
if s.count(old) != 1:
    sys.exit('nc5 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc5-declaration-guard-is-file-wide "$work/nc5.py" declared-by-another-channel-only

# --- NC6: surplus rows go back to being reported and tolerated --------------
# Also from the Phase 4 review: an undeclared row leaves a hole its own size,
# because nothing declared it and so nothing notices it leaving.
cat > "$work/nc6.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '    return (($fails -eq 0) -and ($missing -eq 0) -and ($surplus -eq 0))'
new = '    return (($fails -eq 0) -and ($missing -eq 0))   # INJECTED nc6: drift tolerated'
if s.count(old) != 1:
    sys.exit('nc6 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc6-surplus-rows-tolerated "$work/nc6.py" drift-row-only recorded-more-often-than-declared

# --- NC7: the row maps go back to PowerShell's case-insensitive default -----
cat > "$work/nc7.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = 'New-Object System.Collections.Hashtable([System.StringComparer]::Ordinal)'
new = '@{}   # INJECTED nc7: back to the case-insensitive default'
if s.count(old) != 2:
    sys.exit('nc7 anchor matched %d times, expected 2' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc7-case-insensitive-row-names "$work/nc7.py" case-differs-from-declaration

# --- NC8: Check-Helper stops requiring the helper to have declared anything -
# The helper declares its own rows from its own inputs, so the producer is the
# only authority that the producer exists. Remove the caller's half and a
# helper that returns before its own Expect-Rows is a PASS with nothing owed.
# Only the assertion is reverted, not the count message, so the control stays
# pinned to the assertion rather than to the log line beside it.
# Cut between boundary lines rather than quoting the throw: the message inside
# this block carries the diagnosis and gets rewritten (it has been reordered
# once already, to put the verdict ahead of two absolute paths that Record-Row
# truncates), and a whole-block anchor silently stops applying each time.
cat > "$work/nc8.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
head = '            if ($added -le 0) {\n'
close = '\n            }\n'
if s.count(head) != 1:
    sys.exit('nc8 anchor matched %d times' % s.count(head))
start = s.index(head)
end = s.index(close, start) + len(close)
new = '            # INJECTED nc8: a helper that declared nothing is accepted\n'
io.open(p, 'w', encoding='utf-8', newline='').write(s[:start] + new + s[end:])
PY
# declaration-count-lookup-answers-absent is on the roster for the same reason
# it is on nc9's: its whole assertion is the throw this control deletes, so
# accepting a helper that declared nothing turns its [FAIL] driver row green.
run_control nc8-helper-need-not-declare "$work/nc8.py" \
    helper-exits-0-declaring-nothing helper-reuses-stale-declaration \
    declaration-count-lookup-answers-absent

# --- nc9: the declaration may be the PREVIOUS run's -------------------------
# Keeps the whole "must declare" assertion and reverts only the before/after
# delta, so the count is taken against zero and any historical row satisfies it.
# Pinned to one case: the plain never-declared shape still fails, which is what
# made this hole survive round 12c.
cat > "$work/nc9.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = ('        try { $before = Count-Declared $MustDeclare }\n'
       '        catch { $beforeError = $_.Exception.Message }\n')
new = '        $before = 0  # INJECTED nc9: last run\'s rows count as this run\'s\n'
if s.count(old) != 1:
    sys.exit('nc9 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
# helper-declaration-count-unreadable is in the blast radius on purpose: with
# the before-count gone there is nothing to fail EARLY on an unreadable
# expected.tsv, so the row still fails but for the after-count's reason and with
# the wrong diagnosis. declaration-count-lookup-answers-absent is there for the
# same reason from the other side: it asserts the before-count reads 1 for a
# lookup that answered, and a hard-coded 0 is exactly the stale-declaration
# defect this control reverts. The named-cases pin says so out loud rather than
# letting the controls quietly overlap.
run_control nc9-stale-declaration-accepted "$work/nc9.py" \
    helper-reuses-stale-declaration helper-declaration-count-unreadable \
    declaration-count-lookup-answers-absent

# --- nc10: a row name may carry a tab ---------------------------------------
# Both ledgers are unquoted TSV, so a tab in a name writes its own status
# column. With the guard gone the failing check lands as a PASS row and the
# channel exits 0 -- the ledger reporting the opposite of what happened.
cat > "$work/nc10.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = 'function Record-Row([string]$status, [string]$name, [int]$rc, [string]$detail) {\n    Assert-RowName $name\n'
new = 'function Record-Row([string]$status, [string]$name, [int]$rc, [string]$detail) {\n    # INJECTED nc10: no name validation\n'
if s.count(old) != 1:
    sys.exit('nc10 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc10-tab-in-row-name "$work/nc10.py" row-name-with-a-tab

# --- nc11: Evaluate reads the files and nothing else -------------------------
# Both ledgers are append-only in a reused directory. With this guard gone, a
# run whose every write failed is judged on the PREVIOUS run's rows: balanced,
# no FAIL, no GONE, exit 0. The evidence on disk is real - it is just not this
# run's.
cat > "$work/nc11.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '''    if ($script:GauntletLedgerBroken) {
        Write-Host "==> the ledger could not be written ($script:GauntletLedgerBroken):"
        Write-Host "    this run recorded nothing, and $script:GauntletTsv holds whatever"
        Write-Host "    the last run left. Unchecked, not a pass."
        return $false
    }
'''
new = '    # INJECTED nc11: a run that could not write is judged on what it can read\n'
if s.count(old) != 1:
    sys.exit('nc11 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc11-ledger-failure-ignored "$work/nc11.py" ledger-unwritable

# --- nc12: Evaluate reads both ledgers non-terminatingly, as it used to ------
# The read half of nc11. `Get-Content` hands back the lines it managed to read
# and then writes a NON-TERMINATING error, so a file that failed partway
# through arrives as a prefix. Both ledgers are append-only in a reused
# directory, so that prefix is the PREVIOUS run's balanced pair, and this run's
# FAIL sits below the cut: zero fails, zero missing, zero surplus, exit 0.
#
# This control reverts the READ half only. Evaluate now also windows both
# ledgers to the rows THIS run wrote, and that is a different remedy with its
# own cases (clean-run-after-a-historical-failure and
# historical-pass-does-not-cover-a-check-that-did-not-run). Deleting the window
# here as well would make three cases fail and this control would no longer say
# which defect it caught, so every injection below keeps the base counts and the
# `Select-Object -Skip` intact and changes only how the file is read: a
# terminating `Read-Ledger` back to a bare non-terminating `Get-Content`.
cat > "$work/nc12.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()

# 1. findings.tsv: the Test-Path guard plus the terminating read and its refusal
#    become one non-terminating read, which is what a truncated file used to be
#    judged from.
head1 = '    if (-not (Test-Path $script:GauntletTsv)) {\n'
tail1 = ('    } catch {\n'
         '        Write-Host "==> cannot read $script:GauntletTsv '
         '($($_.Exception.Message)):"\n'
         '        Write-Host "    the rows this run recorded cannot be counted, so a '
         'FAIL row that is"\n'
         '        Write-Host "    there cannot be seen. Unchecked, not a pass."\n'
         '        return $false\n'
         '    }\n')
new1 = ('    $recorded = @(Get-Content $script:GauntletTsv -ErrorAction '
        'SilentlyContinue)   # INJECTED nc12\n')
if s.count(head1) != 1 or s.count(tail1) != 1:
    sys.exit('nc12 anchor 1 matched %d/%d times' % (s.count(head1), s.count(tail1)))
start = s.index(head1)
end = s.index(tail1) + len(tail1)
if end <= start:
    sys.exit('nc12 anchor 1 is out of order')
s = s[:start] + new1 + s[end:]

# 2. expected.tsv, the same way. The shrink refusal and the skip that follow are
#    left alone.
head2 = ('        try {\n'
         '            $declared = Read-Ledger $script:GauntletExpectedFile\n')
tail2 = ('            Write-Host "==> cannot read $script:GauntletExpectedFile '
         '($($_.Exception.Message)):"\n'
         '            Write-Host "    the contract this run declared cannot be read, '
         'so a declared row"\n'
         '            Write-Host "    that never arrived cannot be reported. '
         'Unchecked, not a pass."\n'
         '            return $false\n'
         '        }\n')
new2 = ('        $declared = @(Get-Content $script:GauntletExpectedFile -ErrorAction '
        'SilentlyContinue)   # INJECTED nc12\n')
if s.count(head2) != 1 or s.count(tail2) != 1:
    sys.exit('nc12 anchor 2 matched %d/%d times' % (s.count(head2), s.count(tail2)))
start = s.index(head2)
end = s.index(tail2) + len(tail2)
if end <= start:
    sys.exit('nc12 anchor 2 is out of order')
s = s[:start] + new2 + s[end:]

# 3. And the second read at the bottom, which let the two reads of one file
#    disagree. Windowed exactly as the single read is, so this reintroduces the
#    re-read and nothing else.
old3 = '    foreach ($line in $recorded) {\n'
new3 = ('    foreach ($line in @(Get-Content $script:GauntletTsv -ErrorAction '
        'SilentlyContinue | Select-Object -Skip $script:GauntletFindingsBase)) {   '
        '# INJECTED nc12\n')
if s.count(old3) != 1:
    sys.exit('nc12 anchor 3 matched %d times' % s.count(old3))
s = s.replace(old3, new3)

io.open(p, 'w', encoding='utf-8', newline='').write(s)
PY
run_control nc12-evaluate-reads-non-terminating "$work/nc12.py" ledger-read-truncated

# --- nc13: Count-Declared's throw escapes Check-Helper again -----------------
# Count-Declared uses -ErrorAction Stop so a failed read cannot return 0, but
# the before-count is taken OUTSIDE Check. Put it back and the throw skips the
# row, skips every later statement, and leaves the script exiting 0 -- the very
# bare-interpreter shape nc3 defends against, reintroduced by the guard.
cat > "$work/nc13.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '''    $before = 0
    $beforeError = ""
    if ($MustDeclare) {
        try { $before = Count-Declared $MustDeclare }
        catch { $beforeError = $_.Exception.Message }
    }
'''
new = ('    # INJECTED nc13: the throw escapes Check-Helper again\n'
       '    $before = if ($MustDeclare) { Count-Declared $MustDeclare } else { 0 }\n')
if s.count(old) != 1:
    sys.exit('nc13 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc13-declaration-count-throw-escapes "$work/nc13.py" \
    helper-declaration-count-unreadable

# --- nc14: the carried-findings window is only a COUNT -----------------------
# Round 4 finding C3.6. Evaluate separates rows carried in from earlier runs
# from rows this run wrote by COUNTING: the first N lines are somebody else's.
# A count is not an identity. If a carried row is deleted and one of this run's
# rows lands in its place, the ledger is the same length, the mark still says
# N, and it now cuts through this run's own output -- so a check this run
# measured as FAILING is read as carried-in and never evaluated. `lib.ps1`
# answers that by digesting the carried prefix at dot-source and comparing it
# before it trusts the count. Revert the comparison and the window goes back to
# being a length.
cat > "$work/nc14.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '    if ((Get-LinesDigest $carriedRows) -ne $script:GauntletFindingsDigest) {\n'
new = '    if ($false) {   # INJECTED nc14: the carried window is a count again\n'
if s.count(old) != 1:
    sys.exit('nc14 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc14-carried-window-is-only-a-count "$work/nc14.py" \
    carried-prefix-rewritten

# --- nc15: the declaration window is only a COUNT ----------------------------
# The same defect on the other ledger. The declared-rows file carries the same
# prefix mark for the same reason, and it is read by Count-Declared, whose
# before/after delta decides whether a helper declared anything this run. A
# rewritten declared prefix of unchanged length makes a stale declaration look
# like a fresh one.
cat > "$work/nc15.py" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8', newline='').read()
old = '        if ((Get-LinesDigest $declaredCarried) -ne $script:GauntletExpectedDigest) {\n'
new = '        if ($false) {   # INJECTED nc15: the declaration window is a count again\n'
if s.count(old) != 1:
    sys.exit('nc15 anchor matched %d times' % s.count(old))
io.open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))
PY
run_control nc15-declaration-window-is-only-a-count "$work/nc15.py" \
    declared-prefix-rewritten

# ---------------------------------------------------------------------------
assert_subject_unchanged
if [ "$failures" -gt 0 ]; then
    echo "CONTROL FAILURES: $failures"
    # A completed run that found broken controls is still a completed run: it
    # gets the marker, and the non-zero status carries the verdict.
    reached_end=1
    exit 1
fi
echo "CONTROL FAILURES: 0"
echo "RUN OK: every control removed one half of the remedy and the shipped suite caught it."
reached_end=1
exit 0
