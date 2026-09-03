#!/usr/bin/env python3
"""Behaviour cases and negative controls for the Windows first-run gauntlet probes.

Two things, in the order that matters:

  1. CASES -- drive the shipped probes and the shipped `Check` blocks that call
     them, over stubbed cmdlets, and assert the row each records.
  2. CONTROLS -- revert ONE property of one remedy in a copy of the source,
     re-run the cases, and fail when the case that defends it stays green.

THE SUBJECTS are scripts/first-run/windows-zip.ps1, windows-nsis.ps1 and
lib.ps1. lib.ps1 is driven exactly like the channels: `Get-HealthReachability`
does not classify its exception itself but asks lib.ps1, and the CIM witnesses
and `Get-DirPresence` are defined there once instead of once per channel.
Unstubbed they would trip the stub-escape guard; unmutatable, the rules inside
them would have no control defending them anywhere. A control whose subject is
lib.ps1 still reddens the cases of a CHANNEL, because lib.ps1 has no rows of its
own, and which channel is DERIVED from the control's own must_fail list.

THE DEFECT EVERY CASE HERE IS ABOUT is a FAILED measurement that reads as a
NEGATIVE one. The shapes it takes, as rules rather than as history:

  A PROVIDER THAT FAILED IS NOT AN EMPTY TABLE. `-ErrorAction SilentlyContinue`
      on Get-NetTCPConnection or Get-Process returns, when the provider fails,
      exactly what a closed port and a dead process return -- so ONE broken read
      passes two rows meant to be independent evidence.
  AN EXCEPTION WITH NO Response IS NOT A REFUSAL: a DNS failure, a TLS failure,
      a timeout and a malformed URI all reach `catch` without one, as a refused
      connection does.
  A RESET IS NOT A REFUSAL EITHER, and this one is MEASURED: a TcpListener that
      accepts and closes with LingerState(true, 0) -- indisputably UP -- produces
      WebException/ReceiveFailure wrapping IOException wrapping
      SocketException/ConnectionReset. Only ConnectionRefused is the negative.
  A TRI-STATE THAT DIES AT A CALLER BOUNDARY IS NOT A FIX: three outcomes in,
      one printed line and $LASTEXITCODE 0 out. The probe returns its state and
      every call site records a row.
  A TRI-STATE THAT DIES INSIDE THE CALLEE IS NOT ONE EITHER, by three routes.
      NATIVE STDERR, measured on the reference host: `& cmd.exe /c "echo x 1>&2
      & exit /b 0" 2>&1 | Out-Null` sets `$?` False under `Continue` and THROWS
      System.Management.Automation.RemoteException under `Stop`, and `schtasks
      /end` writes a benign stderr line when the task is not running -- so a
      probe can throw on its first statement and never reach `return`. Hence
      `Invoke-Native`, which pins the preference in its own dynamic scope, keeps
      stderr as text and hands the exit code back as data. A BARE CAST:
      `[int]"unknown"` on a malformed listener row throws rather than returning.
      AN UNBOUNDED OVERRIDE: a health timeout with a floor but no ceiling lets an
      inherited GAUNTLET_HEALTH_TIMEOUT_SEC=86400 make the twenty-attempt stop
      loop effectively non-terminating.
  A WITNESS THAT DOES NOT CO-VARY WITH WHAT IT RATIFIES IS NOT A WITNESS: "is
      this a process table" (pid 4 present, ten rows) cannot ratify "wenlan-
      server is not in it", because the targeted read can throw its absence error
      while the whole-table read CONTAINS the process.
  A WITNESS MUST NOT QUERY THE PROVIDER WHOSE FAILURE IT CLAIMS TO DETECT. The
      port witness is `netstat -ano` (iphlpapi, no CIM, no WMI); the process
      witness adds Win32_Process (WMI's winmgmt service). Neither witnesses
      against the KERNEL, and both say so in the source.
  A CONTROL THAT CANNOT FAIL IS NOT A CONTROL: a process-table fixture that is
      BOTH too short AND missing pid 4 tests neither witness alone.
  A CONSTANT A NEGATIVE DEPENDS ON IS A MEASUREMENT, AND SOMETHING MUST OBSERVE
      IT. -TimeoutSec is 5 rather than 2 because a refused loopback connect takes
      ~2.05 s on the reference host (Windows retries the SYN), so `-TimeoutSec 2`
      turns every genuine refusal into a Timeout and the negative becomes
      unreachable in principle. The slow-refusal stub below observes the
      constant; a stub that ignores TimeoutSec leaves it unpinned.
  OWNERSHIP IS MEASURED, NEVER ASSUMED. `Get-Process -Name wenlan-server |
      Stop-Process -Force` selects EVERY process with that name, and `schtasks
      /delete /tn WenlanServer /f` removes a registration this run may never have
      made. Ownership is by image path AND by "was it already running", and the
      task is ended and deleted only when the name was free before the run
      started. Likewise "the daemon stopped" is the owned-process poll, not one
      ConnectionRefused: a failed kill, or a survivor that merely unbound 7878,
      refuses too.

WHY THE SUBJECT IS EXTRACTED RATHER THAN RUN. Neither channel script can be
executed here at any cost: each calls `Remove-Item -Recurse -Force` on
%LOCALAPPDATA%\\wenlan, which on a developer machine is the real memorydb, config
and logs. So the probe functions and the `Check` blocks that call them are
extracted by brace matching and evaluated in isolation -- the same reason and
technique as `dev-runtime-scan-controls.sh`, which extracts `reap_staged_daemon`
because `dev-runtime.sh` dispatches on "$1" and cannot be sourced. Nothing here
installs, uninstalls, binds a port, kills a process, or writes outside a
temporary directory.

WHAT MAKES THAT A GUARANTEE RATHER THAN A HABIT is that the extraction guard is
an ALLOW-LIST per invocation construct, not a list of forbidden names. A guard
recognising a command held in a variable and nothing else accepts `& ("Remove-" +
"Item") -LiteralPath "$env:LOCALAPPDATA\\wenlan" -Recurse -Force` -- which
contains no banned token anywhere -- and accepts
`[System.IO.Directory]::Delete(...)` and `(Get-Item $p).Delete($true)` as well.
The call operator, the `::` static member and the `.X()` instance call are the
only ways extracted PowerShell can invoke anything, and each is pinned to the set
the shipped text actually uses. See the block above CALL_TARGET_ALLOWED for what
that does and does not cover.

WHAT IS STUBBED, NAMED SO NO READER HAS TO INFER IT:
  Get-NetTCPConnection, Get-Process, Invoke-WebRequest, Get-CimInstance -- the
      measuring cmdlets. Each case says which state it makes them produce.
  netstat.exe and schtasks.exe -- the measuring NATIVE commands, reached through
      the shipped `Invoke-Native`.
  Start-Sleep -- a no-op, so the shipped ten-second poll loops are measured for
      their BRANCHES rather than for the wall clock.
  Stop-OwnedServerProcess and Stop-ProcessByImage -- the shipped KILLERS, and a
      SAFETY REQUIREMENT rather than a convenience: both end in
      `[System.Diagnostics.Process]...Kill()`, so a driver that reached an
      unstubbed one would kill a real process on this machine. They are stubbed
      as RECORDING no-ops so a case can assert what the shipped code decided to
      kill, and `GetProcessById`/`.Kill()` are refused in extracted text so a
      future edit cannot move a real kill into an extracted region.
  Stop-Process -- still stubbed and still in MUST_BE_STUBBED although neither
      channel calls it any more, so a regression that brings it back cannot reach
      a real process on its first run.
  Check, Record-Row, Reached and Test-SingleStatementBlock -- lib.ps1's own
      judging rule, MODELLED rather than summarised, including `-ExpectFail`, the
      reach witness, the single-pipeline AST fallback, and rc=2 for the third
      state. A replica that declares `-ExpectFail` and never reads it judges the
      one shipped row asserting a REFUSAL by the rule for rows asserting a
      success, so the replica's parameter set and rule anchors are compared
      against lib.ps1 on every run (SHIPPED_CHECK_RULES) and its outcomes are
      driven in real processes and then made to come out wrong
      (REPLICA_MUTATIONS).
  wenlan.exe -- the product CLI, including the `search` verb the -ExpectFail row
      runs, so a refusal after `background off` can be told from an answer.

THE STUBS ARE CHECKED TO STILL BE THE STUBS, by the guard emitted into every
driver. Measured on the reference host: auto-importing a module that exports a
same-named function REPLACES a script-scope stub, so after the first
`NetTCPIP\\Get-NetTCPConnection` call the name resolves to the real provider and
the driver silently measures the developer's machine instead of the fixture. The
TCP stubs therefore capture their one real error BEFORE they define themselves,
and the guard refuses to run a `Check` block if any stubbed name has escaped.

The ABSENCE arms do not fabricate an exception: the stub replays a real one --
for processes by calling the real cmdlet for a process that genuinely does not
exist, for ports by re-throwing an ErrorRecord captured from the real provider --
so the id the fix keys on is PowerShell's, not this file's.

ONE STUB RUNS A REAL NATIVE COMMAND, named here rather than buried:
`schtasks-native-stderr-under-stop` invokes `cmd.exe /c "echo ... 1>&2"`, because
a PowerShell FUNCTION cannot produce a NativeCommandError and the native-stderr
rule above is about nothing else. It echoes; it changes nothing.

WHAT THIS DOES NOT PROVE: that the gauntlet installs, uninstalls, or reaches a
daemon. No product code runs here at all.

Run: python3 scripts/negative-controls/windows-probes-negative-controls.py
"""

import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "scripts" / "first-run" / "windows-zip.ps1"
NSIS = ROOT / "scripts" / "first-run" / "windows-nsis.ps1"
# The shared gauntlet library, READ ONLY and never mutated by a control: it
# belongs to the shell/release lane, and this file only ever compares its own
# replica of `Check` against it. See SHIPPED_CHECK_RULES below.
LIB = ROOT / "scripts" / "first-run" / "lib.ps1"
LOGS = ROOT / "target" / "negative-control-logs"

# Each case is one powershell process, and every control re-runs every case of
# its subject, so this is a few hundred short processes. They are independent by
# construction -- separate processes, separate driver files, separate logs, no
# shared state but the read-only source text -- so they are run a few at a time.
# Reporting order is still the order of CASES: results are collected, then
# printed.
WORKERS = 6

failures = 0


def fail(msg):
    global failures
    failures += 1
    print("    FAIL %s" % msg)


# --------------------------------------------------------------------------
# Extraction. Brace matching over the shipped text.
# --------------------------------------------------------------------------
def _match_braces(text, start):
    """From the first '{' at or after `start`, return the index just past its
    match. Naive counting, exactly like dev-runtime-scan-controls.sh's gsub
    pair; every construct extracted here is verified brace-balanced by the
    pre-check below, and an imbalance is a hard error rather than a short read.
    """
    i = text.index("{", start)
    depth = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


def extract_function(text, name):
    m = re.search(r"^function %s\b" % re.escape(name), text, re.M)
    if not m:
        raise ValueError("function %s not found" % name)
    return text[m.start():_match_braces(text, m.start())]


def extract_check_block(text, row):
    # `[^\n]*?` because -Expect / -ExpectFail sit between the name and -Script
    # on the same line. Requiring them to be adjacent silently failed to find
    # the three `background-*` rows, which is the pre-check working: a Check
    # block that cannot be extracted is reported, never skipped.
    m = re.search(r"^[ \t]*Check -Name \"%s\"[^\n]*? -Script " % re.escape(row),
                  text, re.M)
    if not m:
        raise ValueError("Check block for row %r not found" % row)
    return text[m.start():_match_braces(text, m.end())]


# --------------------------------------------------------------------------
# The PowerShell driver.
# --------------------------------------------------------------------------
PREAMBLE = r"""
# Generated by scripts/negative-controls/windows-probes-negative-controls.py.
# Never run a channel script; only the extracted probe and its caller.
$ErrorActionPreference = "Continue"

# No-op: the shipped poll loops are measured for their branches, not their
# wall clock.
function Start-Sleep { param([int]$Seconds, [int]$Milliseconds) }

# --- SAFETY, NOT CONVENIENCE ----------------------------------------------
#
# The two shipped killers end in [System.Diagnostics.Process]...Kill(). A
# driver that reached either would kill a real process on this machine, so both
# are stubbed here as RECORDING no-ops before any extracted code runs, and both
# are in MUST_BE_STUBBED so a driver that lost one dies rather than kills. The
# extractor additionally refuses any region containing GetProcessById or
# .Kill(), so the real ones can never be lifted into a driver at all.
$script:Killed = New-Object System.Collections.Generic.List[string]
function Stop-OwnedServerProcess {
    param([int]$ProcessId, [string]$ImagePath)
    $script:Killed.Add("$ProcessId")
    [pscustomobject]@{ State = "killed"; Detail = "STUB killed pid $ProcessId ('$ImagePath')" }
}
function Stop-ProcessByImage {
    param([int]$ProcessId, [string]$ImagePath)
    $script:Killed.Add("$ProcessId")
    [pscustomobject]@{ State = "killed"; Detail = "STUB killed pid $ProcessId ('$ImagePath')" }
}
# Neither channel calls this any more. It stays stubbed so a regression that
# brings the bare-name pipeline back cannot reach a real process on its way to
# being caught.
function Stop-Process { param($InputObject, [int]$Id, [switch]$Force, [string]$ErrorAction) $null = $input }

# --- default measuring stubs ----------------------------------------------
#
# Every name that can reach the machine is a function before any extracted code
# runs. A case that needs a different answer layers its own definition after
# these; a case that does not, cannot accidentally reach the real one.
function Get-Process { param([int[]]$Id, [string[]]$Name, [string]$ErrorAction) @() }
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    throw (New-Object System.InvalidOperationException "this case did not stub Invoke-WebRequest; refusing to make a real request")
}
function Get-NetTCPConnection {
    param([string]$State, [int]$LocalPort, [string]$ErrorAction)
    throw (New-Object System.InvalidOperationException "this case did not stub Get-NetTCPConnection; refusing to query the real provider")
}
# Get-DirPresence's provider. Read-only, but stubbed all the same: a driver
# that read the developer's real filesystem would report on a machine nobody
# meant to measure, and %LOCALAPPDATA%\wenlan is one of the paths it would read.
# The default is ABSENT, and the error is REPLAYED from the real cmdlet so the
# ItemNotFoundException the shipped catch keys on is PowerShell's own rather
# than one this file made up. Measured on this host: a path under a directory
# that does not exist arrives as System.Management.Automation.ItemNotFoundException.
function Get-Item {
    param([string]$LiteralPath, [switch]$Force, [string]$ErrorAction)
    Microsoft.PowerShell.Management\Get-Item -LiteralPath "$env:TEMP\wenlan-nc-no-such-path-xyz\nope" -Force -ErrorAction Stop
}
# The two WITNESS providers default to a believable table that AGREES with the
# absence, rather than to a refusal. That is deliberate: they are witnesses, so
# most cases are about the other provider and want the witness out of the way.
# The cases that are about the witness override them below, and a control that
# deletes a witness is caught by those.
# Pids the wenlan-server fixtures last reported. The default WMI table mirrors
# them, so the two providers agree by construction and a case only diverges when
# it means to.
$script:SrvCimPids = @()
function Get-CimInstance {
    param([string]$ClassName, [string]$Filter, [string]$ErrorAction)
    @(@(0..288 | ForEach-Object { [pscustomobject]@{ ProcessId = $_; Name = "p$_.exe"; ExecutablePath = "C:\Windows\System32\p$_.exe" } }) +
      @($script:SrvCimPids | ForEach-Object { [pscustomobject]@{ ProcessId = $_; Name = "wenlan-server.exe"; ExecutablePath = "C:\x\wenlan-server.exe" } }))
}
# The user's files under $DataDir, as the row's TWO reads see them: once before
# the uninstaller runs and once after. The pair is the point -- a fixture that
# answered the same thing every time could not express a file going missing,
# which is the only thing this row exists to notice. The default is a clean
# uninstall: the same three files, same digests, both times.
$script:TreeSnapshots = @(
    @{ State = "taken"; Files = @{ "memorydb.sqlite" = "AAA"; "config.json" = "BBB"; "logs\wenlan.log" = "CCC" } },
    @{ State = "taken"; Files = @{ "memorydb.sqlite" = "AAA"; "config.json" = "BBB"; "logs\wenlan.log" = "CCC" } }
)
$script:TreeReads = 0
function Get-TreeFileDigests {
    param([string]$Root)
    $i = $script:TreeReads
    if ($i -ge $script:TreeSnapshots.Count) { $i = $script:TreeSnapshots.Count - 1 }
    $script:TreeReads++
    $s = $script:TreeSnapshots[$i]
    [pscustomobject]@{ State = $s.State; Files = $s.Files
        Detail = "driver fixture read $($i + 1) of ${Root}: $($s.State), $($s.Files.Count) files" }
}
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    # The shape `netstat -ano` really prints, including the header line (which
    # is not a protocol row), one ESTABLISHED row (five fields, NOT a wildcard
    # foreign address) and one UDP row (four fields).
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576",
    "  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4",
    "  TCP    0.0.0.0:5985           0.0.0.0:0              LISTENING       4",
    "  TCP    127.0.0.1:49670        127.0.0.1:49671        ESTABLISHED     8888",
    "  TCP    [::]:135               [::]:0                 LISTENING       1576",
    "  UDP    0.0.0.0:5353           *:*                                    2340"
}
# --- the scheduled-task pair -------------------------------------------
# Get-TaskPresence reads TWO providers per call: schtasks.exe, then
# Get-ScheduledTask. They must agree, so one counter drives both.
#
# $TaskSeq is what the table says on successive reads, in the order the shipped
# script takes them: BEFORE the run, AFTER `background on`, and AFTER the
# delete. That sequence is the point -- ownership is now "free before AND
# present after", which a single fixed answer cannot express.
$script:TaskSeq = @($false, $true, $false)
$script:TaskReads = 0
$script:TaskPresentNow = $false
$script:TaskRowCount = 30      # DISTINCT task names schtasks prints
$script:TaskRowsPerTask = 1    # rows per task: schtasks prints one per TRIGGER
$script:TaskCimCount = 25      # independent task objects (one per task)
$script:TaskNameCase = "normal"  # normal | upper -- the case WenlanServer is registered under
$script:TaskCsvExit = 0
$script:TaskCsvShape = "good"  # good | torn
$script:TaskRowsAreWindows = $true
$script:TaskDeleteExit = 0
$script:TaskCimThrows = $false
$script:TaskCimPresentOverride = "follow"   # follow | yes | no

function schtasks.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    $joined = ($Rest -join " ")
    if ($joined -like "*/fo CSV*") {
        $i = $script:TaskReads
        $script:TaskReads = $i + 1
        $script:TaskPresentNow = if ($i -lt $script:TaskSeq.Count) { $script:TaskSeq[$i] } else { $script:TaskSeq[$script:TaskSeq.Count - 1] }
        $global:LASTEXITCODE = $script:TaskCsvExit
        if ($script:TaskCsvExit -ne 0 -and $script:TaskRowCount -eq 0) { return "ERROR: Access is denied." }
        $prefix = if ($script:TaskRowsAreWindows) { '"\Microsoft\Windows\Fixture\Task' } else { '"\Vendor\Task' }
        # ONE ROW PER TRIGGER, which is what schtasks really prints and what
        # makes a row COUNT useless as a completeness test: $TaskRowsPerTask
        # rows can carry the same task name, so a table can have more rows than
        # the independent enumeration has tasks while still omitting tasks.
        $rows = @(0..($script:TaskRowCount - 1) | ForEach-Object {
            $n = $_
            1..$script:TaskRowsPerTask | ForEach-Object { $prefix + $n + '","N/A","Ready"' } })
        $wenlanRow = if ($script:TaskNameCase -eq "upper") { '"\WENLANSERVER","N/A","Ready"' } else { '"\WenlanServer","N/A","Ready"' }
        if ($script:TaskPresentNow) { $rows = @($wenlanRow) + $rows }
        if ($script:TaskCsvShape -eq "torn") { $rows[$rows.Count - 1] = '"\Microsoft\Windows\Fixture\Torn","N/A"' }
        return $rows
    }
    if ($joined -like "*/delete*") {
        $global:LASTEXITCODE = $script:TaskDeleteExit
        if ($script:TaskDeleteExit -ne 0) { return "ERROR: Access is denied." }
        return "SUCCESS: The scheduled task was successfully deleted."
    }
    if ($joined -like "*/query*") {
        $global:LASTEXITCODE = 1
        return "ERROR: The system cannot find the file specified."
    }
    # /end: silent success.
}

# The INDEPENDENT provider. A cmdletized Function in real life (measured:
# CommandType=Function, Module=ScheduledTasks), which is why the stub-escape
# guard's empty-ModuleName rule matters for it.
function Get-ScheduledTask {
    param([string]$TaskName, [string]$TaskPath, [string]$ErrorAction)
    if ($script:TaskCimThrows) { throw (New-Object System.Management.Automation.RuntimeException "the task scheduler service is not available") }
    $all = @(0..($script:TaskCimCount - 1) | ForEach-Object {
        [pscustomobject]@{ TaskPath = "\Microsoft\Windows\Fixture\"; TaskName = "Task$_" } })
    $present = switch ($script:TaskCimPresentOverride) {
        "yes"  { $true }
        "no"   { $false }
        default { $script:TaskPresentNow }
    }
    $wenlanName = if ($script:TaskNameCase -eq "upper") { "WENLANSERVER" } else { "WenlanServer" }
    if ($present) { $all = @([pscustomobject]@{ TaskPath = "\"; TaskName = $wenlanName }) + $all }
    return $all
}

# The product CLI. `background on` / `background off` reach the SAME scheduled
# task the scheduler binary does, which is the door round-4 review found open.
function wenlan.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    $joined = ($Rest -join " ")
    if ($joined -like "background on*") { $script:CliDroveTask = "on"; return "Installed and started Windows scheduled task" }
    if ($joined -like "background off*") { $script:CliDroveTask = "off"; return "Background registration kept" }
    # `search` is what the one shipped -ExpectFail row runs. After `background
    # off` the CLI must REFUSE with the stopped marker instead of autostarting
    # the daemon, and the three things it can do instead are the three modes
    # here. A refusal is a NONZERO EXIT plus the marker text, which is why the
    # exit code is set the way a native command sets it -- `& wenlan.exe` is a
    # bare call in the shipped block, not one routed through Invoke-Native, so
    # its $LASTEXITCODE is exactly what Check reads.
    if ($joined -like "search*") {
        if ($script:CliSearchMode -eq "refuses") {
            Write-Output "error: daemon stopped by 'wenlan background off'; run 'wenlan background on' to start it again"
            $global:LASTEXITCODE = 1
            return
        }
        if ($script:CliSearchMode -eq "fails-otherwise") {
            Write-Output "error: could not connect to 127.0.0.1:7878 after 3 attempts"
            $global:LASTEXITCODE = 1
            return
        }
    }
    return "wenlan.exe $joined"
}
$script:CliDroveTask = "none"
$script:CliSearchMode = "succeeds"   # succeeds | refuses | fails-otherwise

# --- lib.ps1's Check, MODELLED, not summarised -------------------------------
#
# ROUND 6, found by the shell/release lane in THIS file. The block below used to
# say "reduced to its PASS/FAIL rule", and the reduction had quietly dropped a
# rule: `-ExpectFail` was declared and never read. `$status` consulted `$Expect`
# alone, so the one shipped row that asserts a REFUSAL
# -- `stopped-marker-error (wenlan.exe search x)` -- was judged by the rule for
# rows that assert a SUCCESS, which fails every nonzero block, including the
# refusal that row exists to require. A parameter accepted and ignored is worse
# than one not accepted at all: the call site reads as measured.
#
# The authority is scripts/first-run/lib.ps1:330-437 -- the shell/release lane's
# file, read here and never edited. What is modelled, and must stay modelled:
#
#   * `$script:CheckReached` and `Reached`, the witness that execution actually
#     reached the construct under test before the expected failure happened;
#   * `Test-SingleStatementBlock`, the AST fallback that lets a block which is
#     one simple pipeline witness itself, because there is no earlier statement
#     that could have thrown the expected text first;
#   * the `$phase` / `$faultedIn` split, so a fault raised by the harness's OWN
#     output capture is not read as the block under test failing;
#   * rc = 2, THE THIRD STATE. lib.ps1 has three outcomes and only two status
#     words available, because `first-run-gauntlet.yml` asserts the PASS|FAIL|INFO
#     vocabulary and `summary.py` reads it; so "unmeasured" rides in the rc
#     column as 2, with an `unmeasured:` prefix on the detail. The replica
#     carries it the same way and publishes it as `RC[n]` on the ROW so a case
#     can assert on it -- otherwise the third state would arrive here as a
#     plain FAIL and this harness would have reproduced, in its own model of
#     Check, the exact confusion Check was changed to prevent.
#
# DOCUMENTED RESIDUAL -- the parts of the shipped Check the replica does NOT
# model, and why each is a deliberate omission rather than an oversight:
#
#   * The LEDGER. lib.ps1's `Record-Row` appends `channel<TAB>name<TAB>status<TAB>rc<TAB>detail`
#     to a file the workflow later parses. The replica's `Record-Row` prints one
#     ROW line to stdout instead, because that is what this harness's
#     `classify_driver_output` reads. All four facts survive; the transport does
#     not, and nothing in any case turns on the transport.
#   * The PER-CHECK LOG. `Set-Content -Path (Join-Path $script:GauntletOut ...)`
#     needs `$GauntletOut`, a directory the shipped script creates on the machine
#     under test. A harness whose entire premise is that the channel scripts are
#     never executed must not start creating their output tree, so this one line
#     is dropped. Consequence, stated rather than hidden: a defect that broke
#     only the log write would not be visible here.
#   * `Assert-RowName` and the row-name registry, which live outside Check and
#     govern which names may be recorded, not how a check is judged.
#
# AND ONE THING THAT LOOKS LIKE A RESIDUAL AND IS NOT. `$rc` now also comes from
# `$LASTEXITCODE`, exactly as upstream, which is what makes `-ExpectFail` mean
# anything at all: a native command that exits nonzero is the only way a shipped
# block reports a refusal without throwing. That could have turned every row red
# here, because several stubs below leave `$LASTEXITCODE` nonzero on purpose --
# and it does not, for a reason worth writing down: every shipped native call
# goes through `Invoke-Native`, which sets `$global:LASTEXITCODE = 0` after
# capturing the code (windows-zip.ps1:138-140, commented there as "the stale rc must
# not leak into the next Check block"). The one bare native call in either
# channel is `& wenlan.exe`, which is the call this rule is for.
#
# And the drift itself is now measured rather than reviewed: `check_param_drift`
# below compares this replica's parameter set against the shipped Check's on
# every run, and `SHIPPED_CHECK_RULES` requires each rule named above to still
# be present in BOTH. The next parameter added upstream is a red run here, not a
# reviewer's lucky catch.
$script:CheckReached = $null

function Reached {
    param([string]$What = "")
    if ($null -eq $script:CheckReached) {
        throw "Reached was called outside a Check block, where it witnesses nothing"
    }
    $script:CheckReached = if ($What) { $What } else { "yes" }
}

function Test-SingleStatementBlock([scriptblock]$Script) {
    try {
        $ast = $Script.Ast
        if ($null -eq $ast) { return $false }
        if ($null -ne $ast.ParamBlock) { return $false }
        if ($null -ne $ast.BeginBlock -or $null -ne $ast.ProcessBlock) { return $false }
        $end = $ast.EndBlock
        if ($null -eq $end) { return $false }
        $statements = $end.Statements
        if ($null -eq $statements) { return $false }
        if ($statements.Count -ne 1) { return $false }
        return ($statements[0].GetType().Name -eq "PipelineAst")
    } catch {
        return $false
    }
}

# lib.ps1 writes a ledger line here; this writes the ROW line the classifier
# reads. The two probe markers this harness adds -- what got killed, and whether
# the product CLI drove the scheduled task -- and the rc the third state rides
# in are appended to the detail, AFTER the decision has been made on the shipped
# text alone. That order matters: a marker inside the text a `-Expect` is tested
# against would let a row pass on this harness's own bookkeeping.
function Record-Row([string]$status, [string]$name, [int]$rc, [string]$detail) {
    $markers = " | KILLED[" + ($script:Killed -join ",") + "]" +
               " | CLI[" + $script:CliDroveTask + "]" +
               " | RC[" + $rc + "]"
    Write-Host ("ROW`t" + $name + "`t" + $status + "`t" +
                (($detail + $markers) -replace "`r?`n", " | "))
}

function Check {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Script,
        [string]$Expect,
        [string]$ExpectFail
    )
    $global:LASTEXITCODE = 0
    $rc = 0
    $phase = @{ Value = "block" }
    $faultedIn = ""
    $outerReached = $script:CheckReached
    $script:CheckReached = ""
    $lines = New-Object System.Collections.Generic.List[string]
    try {
        & $Script 2>&1 | ForEach-Object {
            $phase.Value = "capture"
            $lines.Add(($_ | Out-String).TrimEnd())
            $phase.Value = "block"
        }
        if ($LASTEXITCODE -is [int] -and $LASTEXITCODE -ne 0) { $rc = $LASTEXITCODE }
    } catch {
        $lines.Add($_.ToString())
        $rc = 1
        $faultedIn = $phase.Value
    }
    $reached = $script:CheckReached
    $script:CheckReached = $outerReached
    $out = ($lines -join "`n")
    if ($faultedIn -eq "capture") {
        Record-Row FAIL $Name 2 ("unmeasured: the fault came from this harness's own output capture " +
            "rather than from the block, so nothing was learned about the check; got: " + $out)
        return
    }
    if ($ExpectFail) {
        if ($rc -ne 0 -and -not $reached -and -not (Test-SingleStatementBlock $Script)) {
            Record-Row FAIL $Name 2 ("unmeasured: the block failed with the expected text but nothing " +
                "witnesses that execution reached the construct under test -- a setup statement can throw " +
                "'" + $ExpectFail + "' before the construct runs. Call Reached immediately before it, or " +
                "make the block a single pipeline; got: " + $out)
            return
        }
        if ($rc -ne 0 -and $out.Contains($ExpectFail)) { Record-Row PASS $Name $rc $out }
        else { Record-Row FAIL $Name $rc ("expected nonzero exit with substring: " + $ExpectFail + "; got: " + $out) }
        return
    }
    if ($rc -eq 0 -and (-not $Expect -or $out.Contains($Expect))) { Record-Row PASS $Name $rc $out }
    elseif ($Expect) { Record-Row FAIL $Name $rc ("expected substring: " + $Expect + "; got: " + $out) }
    else { Record-Row FAIL $Name $rc $out }
}

# Values the extracted blocks and probes close over in their own script.
$Health = "http://127.0.0.1:7878/api/health"
$App = [pscustomobject]@{ Id = 4242 }
$AppExe = "C:\Users\ci\AppData\Local\Programs\Wenlan\wenlan-app.exe"
# The zip channel's ownership state. The shipped script sets these from
# measurements taken before anything is installed; the driver sets them
# directly, and the cases that are ABOUT them drive Get-TaskPresence through a
# setup line instead.
$TaskName = "WenlanServer"
$DataDir = "C:\Users\ci\AppData\Local\wenlan"
$OwnedServerImage = "C:\Users\ci\AppData\Local\Programs\wenlan\wenlan-server.exe"
$PreexistingServerPids = @()
$TaskOwned = $true
$preTask = [pscustomobject]@{ State = "absent"; Detail = "driver default" }
$postTask = [pscustomobject]@{ State = "present"; Detail = "driver default" }
$MayDriveTask = $true
$TaskDeleteResult = [pscustomobject]@{ Ran = $true; ExitCode = 0; Output = "driver default" }
# The teardown's per-tree record, as the shipped cleanup leaves it on a clean
# run: both trees created by this run, both still bound to it, both deleted and
# the delete's own report saying so. Every fact `no-leftover-dirs` branches on
# is one entry in here, and a case that is about one of them overrides that
# entry alone. BOTH channels record this row now -- the nsis one used to read
# only the data dir, which is the false green round-5 review constructed.
$CleanupDirs = @(
    @{ Name = "install dir"; Path = "C:\Users\ci\AppData\Local\Programs\wenlan"; Owned = $true
       Pre = [pscustomobject]@{ State = "absent"; Detail = "driver default" }
       Licence = [pscustomobject]@{ State = "granted"; Detail = "driver default: marked by this run and verified" }
       Removal = [pscustomobject]@{ State = "removed"; Attempts = 1; Detail = "driver default: measured gone after attempt 1" } },
    @{ Name = "data dir"; Path = "C:\Users\ci\AppData\Local\wenlan"; Owned = $true
       Pre = [pscustomobject]@{ State = "absent"; Detail = "driver default" }
       Licence = [pscustomobject]@{ State = "granted"; Detail = "driver default: marked by this run and verified" }
       Removal = [pscustomobject]@{ State = "removed"; Attempts = 1; Detail = "driver default: measured gone after attempt 1" } }
)

# A WebException carrying a non-null Response, built without binding a port.
function New-HttpErrorException {
    Add-Type -TypeDefinition @"
using System.Net;
public class WenlanFakeWebResponse : WebResponse { public int Marker { get { return 500; } } }
"@ -ErrorAction Stop -WarningAction SilentlyContinue
    $resp = New-Object WenlanFakeWebResponse
    return (New-Object System.Net.WebException "The remote server returned an error: (500).", $null, ([System.Net.WebExceptionStatus]::ProtocolError), $resp)
}
function New-SocketWebException([string]$Status, [int]$SocketCode) {
    $inner = $null
    if ($SocketCode -gt 0) { $inner = New-Object System.Net.Sockets.SocketException $SocketCode }
    return (New-Object System.Net.WebException "stubbed $Status", $inner, ([System.Net.WebExceptionStatus]$Status), $null)
}
"""

# Prepended to every Get-NetTCPConnection stub, and the ORDER IS THE POINT.
#
# MEASURED ON THE REFERENCE HOST: auto-importing a module that exports a
# same-named function REPLACES a script-scope stub. Before the import,
# `Get-Command Get-NetTCPConnection` is a Function with no module and the stub
# answers; after one `NetTCPIP\Get-NetTCPConnection` call it is a Function from
# module NetTCPIP and the driver reads the developer's real listener table (39
# rows here) while believing it is reading a fixture. A stub that replays a real
# error by calling the real cmdlet therefore destroys itself on first use.
#
# So the real error is captured ONCE, HERE, BEFORE the stub is defined: the
# import happens first, the stub is defined after it and wins for good.
#
# It is kept even though the shipped probe no longer makes a targeted
# Get-NetTCPConnection call, because the capture is what proves the import
# hazard is still handled the same way, and because a future stub that needs a
# real absence error must not reintroduce the ordering bug.
#
# 63999 is a port this repository never binds; the call is a read-only CIM query.
# If something IS listening there the driver dies here with a named reason
# rather than quietly changing what the fixtures mean.
TCP_ABSENCE_CAPTURE = r"""
$script:RealPortAbsence = $null
try { $null = NetTCPIP\Get-NetTCPConnection -LocalPort 63999 -State Listen -ErrorAction Stop }
catch { $script:RealPortAbsence = $_ }
if ($null -eq $script:RealPortAbsence) {
    throw "port 63999 answered on this host, so the TCP stub cannot replay a real absence error"
}
"""

# The absence arms of the process stubs replay a REAL error from the real
# cmdlet, so the type and the FullyQualifiedErrorId the fix keys on are
# PowerShell's, not this file's. Microsoft.PowerShell.Management is loaded before
# this driver starts, so -- unlike NetTCPIP -- naming it module-qualified imports
# nothing and cannot displace the stub. The guard proves it.
REAL_PROC_ABSENCE = r"""
    if ($PSBoundParameters.ContainsKey('Name')) {
        Microsoft.PowerShell.Management\Get-Process -Name "definitely-not-a-real-process-name-xyz" -ErrorAction Stop
    } else {
        Microsoft.PowerShell.Management\Get-Process -Id 999999 -ErrorAction Stop
    }
"""

# --- cmdlet stubs, one per state a case needs ------------------------------
STUBS = {
    # --- Get-NetTCPConnection: the PRIMARY listener read ------------------
    "tcp-table-without-7878": TCP_ABSENCE_CAPTURE + r"""
function Get-NetTCPConnection {
    param([string]$State, [int]$LocalPort, [string]$ErrorAction)
    @(
        [pscustomobject]@{ LocalPort = 135;  OwningProcess = 1576 }
        [pscustomobject]@{ LocalPort = 445;  OwningProcess = 4 }
        [pscustomobject]@{ LocalPort = 5985; OwningProcess = 4 }
        [pscustomobject]@{ LocalPort = 49664; OwningProcess = 900 }
    )
}
""",
    "tcp-table-with-7878": TCP_ABSENCE_CAPTURE + r"""
function Get-NetTCPConnection {
    param([string]$State, [int]$LocalPort, [string]$ErrorAction)
    @(
        [pscustomobject]@{ LocalPort = 135;  OwningProcess = 1576 }
        [pscustomobject]@{ LocalPort = 445;  OwningProcess = 4 }
        [pscustomobject]@{ LocalPort = 7878; OwningProcess = 4242 }
    )
}
""",
    "tcp-provider-throws": TCP_ABSENCE_CAPTURE + r"""
function Get-NetTCPConnection {
    param([string]$State, [int]$LocalPort, [string]$ErrorAction)
    throw (New-Object System.Management.Automation.RuntimeException "the CIM server is unavailable")
}
""",
    "tcp-table-empty": TCP_ABSENCE_CAPTURE + r"""
function Get-NetTCPConnection {
    param([string]$State, [int]$LocalPort, [string]$ErrorAction)
    @()
}
""",
    "tcp-rows-unusable": TCP_ABSENCE_CAPTURE + r"""
function Get-NetTCPConnection {
    param([string]$State, [int]$LocalPort, [string]$ErrorAction)
    @(
        [pscustomobject]@{ LocalPort = $null; OwningProcess = $null }
        [pscustomobject]@{ LocalPort = $null; OwningProcess = $null }
    )
}
""",
    # C6. A row whose LocalPort is a STRING THAT IS NOT A NUMBER. `[int]"unknown"`
    # does not return a value, it THROWS -- and in the form this replaces the
    # cast sat outside the try that read the table, so the throw left
    # Get-PortListenerState without returning any of its three states at all.
    # The function coped with NULL and not with UNPARSEABLE, which is the same
    # failure wearing a different value.
    "tcp-rows-unparseable": TCP_ABSENCE_CAPTURE + r"""
function Get-NetTCPConnection {
    param([string]$State, [int]$LocalPort, [string]$ErrorAction)
    @(
        [pscustomobject]@{ LocalPort = 135;       OwningProcess = 1576 }
        [pscustomobject]@{ LocalPort = "unknown"; OwningProcess = $null }
        [pscustomobject]@{ LocalPort = 445;       OwningProcess = 4 }
    )
}
""",

    # --- netstat.exe: the INDEPENDENT listener witness --------------------
    # C4. The previous witness was Get-NetTCPConnection ratifying
    # Get-NetTCPConnection. These fixtures exist to make the two reads disagree,
    # which only means anything now that they are different providers.
    "netstat-shows-7878": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576",
    "  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4",
    "  TCP    127.0.0.1:7878         0.0.0.0:0              LISTENING       4242",
    "  UDP    0.0.0.0:5353           *:*                                    2340"
}
""",
    "netstat-fails": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    $global:LASTEXITCODE = 1
    "netstat: unable to open the TCP table"
}
""",
    # A table whose protocol rows are not the rows this parse understands -- a
    # torn line, a merged warning, a column layout that moved. A parse that
    # matched no row for our port would otherwise report a busy port as free.
    "netstat-garbled": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING",
    "  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING",
    "  UDP    0.0.0.0:5353           *:*"
}
""",
    # Well-formed rows, none of them LISTENING. The netstat equivalent of the
    # empty listener table: a host with no listening socket at all is not a
    # measurement of an idle machine.
    #
    # It carries a UDP row so that this fixture isolates the LISTENING floor:
    # without one the end witness below refuses it first, and a control that
    # deleted the floor would still see a red case for the wrong reason.
    "netstat-no-listeners": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    127.0.0.1:49670        127.0.0.1:49671        ESTABLISHED     8888",
    "  TCP    127.0.0.1:49671        127.0.0.1:49670        ESTABLISHED     8889",
    "  UDP    0.0.0.0:5353           *:*                                    2340"
}
""",
    # ROUND 4 of the parse, ported from scripts/lib/host-process.sh. A status-0
    # diagnostic merged BESIDE real rows. Its first token is neither TCP nor
    # UDP, so a parse that skipped every line not already claiming to be a
    # protocol row never looked at it: the rows that survived validated, none of
    # them was 7878, and an incomplete read became a measured negative.
    "netstat-warning-beside-rows": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576",
    "WARNING: provider returned partial results",
    "  UDP    0.0.0.0:5353           *:*                                    2340"
}
""",
    # The SAME table with the warning line removed, and nothing else changed.
    # It is what makes the case above a measurement of the warning rather than
    # of the fixture: this one still reads as a measured negative.
    "netstat-warning-free": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576",
    "  UDP    0.0.0.0:5353           *:*                                    2340"
}
""",
    # ROUND 5. A believable PREFIX: every line is a well-formed row and the
    # preamble is intact, so the grammar has nothing to object to. Only the
    # missing UDP section says the stream stopped before the TCP table ended.
    "netstat-truncated-before-udp": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576",
    "  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4"
}
""",
    # ROUND 6. A UDP row is present, so the end witness is satisfied -- but a
    # TCP row follows it, so this stream is not "all of TCP, then all of UDP"
    # and the witness licenses nothing about the TCP section.
    "netstat-tcp-after-udp": r"""
function netstat.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    "",
    "Active Connections",
    "",
    "  Proto  Local Address          Foreign Address        State           PID",
    "  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1576",
    "  UDP    0.0.0.0:5353           *:*                                    2340",
    "  TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4"
}
""",

    # --- the user's data across the uninstall ------------------------------
    # THE FIXTURE THE ROW WAS BLIND TO: the data ROOT is still there -- this run
    # holds a DeleteOnClose handle inside it, so it cannot not be -- and one of
    # the user's files under it is gone.
    "user-data-file-erased": r"""
$script:TreeSnapshots = @(
    @{ State = "taken"; Files = @{ "memorydb.sqlite" = "AAA"; "config.json" = "BBB"; "logs\wenlan.log" = "CCC" } },
    @{ State = "taken"; Files = @{ "config.json" = "BBB"; "logs\wenlan.log" = "CCC" } }
)
""",
    # Present, same path, different bytes.
    "user-data-file-rewritten": r"""
$script:TreeSnapshots = @(
    @{ State = "taken"; Files = @{ "memorydb.sqlite" = "AAA"; "config.json" = "BBB" } },
    @{ State = "taken"; Files = @{ "memorydb.sqlite" = "ZZZ"; "config.json" = "BBB" } }
)
""",
    "user-data-post-read-failed": r"""
$script:TreeSnapshots = @(
    @{ State = "taken"; Files = @{ "memorydb.sqlite" = "AAA"; "config.json" = "BBB" } },
    @{ State = "unmeasurable"; Files = @{} }
)
""",
    "user-data-pre-read-failed": r"""
$script:TreeSnapshots = @(
    @{ State = "unmeasurable"; Files = @{} },
    @{ State = "taken"; Files = @{ "memorydb.sqlite" = "AAA" } }
)
""",
    # Both reads succeeded and there was nothing to lose. Not a clean uninstall
    # -- an observation with nothing in it.
    "user-data-empty-before": r"""
$script:TreeSnapshots = @(
    @{ State = "taken"; Files = @{} },
    @{ State = "taken"; Files = @{} }
)
""",

    # --- Invoke-WebRequest ------------------------------------------------
    "http-200": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    [pscustomobject]@{ StatusCode = 200; Content = '{"version":"0.0.0"}' }
}
""",
    "http-refused": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    throw (New-SocketWebException "ConnectFailure" 10061)
}
""",
    # THE SHAPE A LIVE LISTENER PRODUCES, measured on the reference host against
    # a TcpListener that accepts and closes with LingerState(true, 0):
    # WebException/ReceiveFailure -> IOException -> SocketException 10054. Note
    # the socket error is NOT $we.InnerException; a probe that reads only the
    # first layer cannot even name it.
    "http-reset-live-listener": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    $sock = New-Object System.Net.Sockets.SocketException 10054
    $io = New-Object System.IO.IOException "Unable to read data from the transport connection: An existing connection was forcibly closed by the remote host.", $sock
    throw (New-Object System.Net.WebException "The underlying connection was closed: An unexpected error occurred on a receive.", $io, ([System.Net.WebExceptionStatus]::ReceiveFailure), $null)
}
""",
    # The exact shape the old classifier called `down`: a connect that failed
    # with a RESET rather than a refusal, which is what a proxy or a filtering
    # middlebox in front of a LIVE service produces.
    "http-connectfailure-reset": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    throw (New-SocketWebException "ConnectFailure" 10054)
}
""",
    # THE STUB THAT OBSERVES TimeoutSec FROM BELOW, and the reason the floor is
    # pinned. A refusal on the reference host arrives at ~2050 ms (measured
    # 2011-2090 ms: Windows retries the SYN), so a client timeout below that
    # wins the race and the caller is handed Status=Timeout -- indistinguishable
    # from a wedged daemon, and therefore unmeasurable.
    "http-refused-after-2050ms": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('TimeoutSec')) {
        throw (New-Object System.InvalidOperationException "the health probe passed no TimeoutSec at all; the .NET default is 100 s and nothing pins it")
    }
    if (($TimeoutSec * 1000) -lt 2050) { throw (New-SocketWebException "Timeout" 0) }
    throw (New-SocketWebException "ConnectFailure" 10061)
}
""",
    # C7, and it observes TimeoutSec FROM ABOVE. The floor stops a value that
    # makes the negative unreachable; nothing stopped a value that makes the run
    # unreachable. Twenty of these per stop-wait loop at 86400 s is not a slow
    # test, it is a gauntlet that never records a row -- and no row is the one
    # outcome the ledger cannot tell from a run that never started.
    "http-refused-unless-timeout-absurd": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('TimeoutSec')) {
        throw (New-Object System.InvalidOperationException "the health probe passed no TimeoutSec at all")
    }
    if ($TimeoutSec -gt 60) {
        throw (New-Object System.InvalidOperationException ("the health probe asked for a " + $TimeoutSec + "s client timeout; the stop-wait loop runs it twenty times, so nothing bounds this run"))
    }
    if (($TimeoutSec * 1000) -lt 2050) { throw (New-SocketWebException "Timeout" 0) }
    throw (New-SocketWebException "ConnectFailure" 10061)
}
""",
    "http-timeout": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    throw (New-SocketWebException "Timeout" 0)
}
""",
    "http-dns-failure": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    throw (New-SocketWebException "NameResolutionFailure" 0)
}
""",
    "http-bad-uri": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    throw (New-Object System.UriFormatException "Invalid URI: The hostname could not be parsed.")
}
""",
    "http-500": r"""
function Invoke-WebRequest {
    param([string]$Uri, [switch]$UseBasicParsing, [int]$TimeoutSec, [string]$ErrorAction)
    throw (New-HttpErrorException)
}
""",

    # --- Get-Process ------------------------------------------------------
    # The witness arm (no -Id, no -Name) and the target arm are chosen
    # independently, because the fix's positive witness only matters when the
    # target says "absent".
    "proc-gone-table-good": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
""" + REAL_PROC_ABSENCE + r"""}
""",
    # Three entries, none of them pid 4: caught by EITHER witness, which is
    # precisely why it cannot isolate one. The two fixtures after it can.
    "proc-gone-table-fragment": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(101, 102, 103 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
""" + REAL_PROC_ABSENCE + r"""}
""",
    # 288 rows -- a table long enough to look complete -- and no pid 4. Only the
    # pid-4 witness can reject this, so only this fixture can prove that witness
    # is still there.
    "proc-gone-long-table-without-pid4": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(100..387 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
""" + REAL_PROC_ABSENCE + r"""}
""",
    # Six rows, pid 4 among them. Only the row floor can reject this.
    "proc-gone-short-table-with-pid4": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0, 4, 101, 102, 103, 104 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
""" + REAL_PROC_ABSENCE + r"""}
""",
    # A perfectly believable table -- 289 rows, pid 4 present -- that CONTAINS
    # the very process the targeted read just said was absent, under both the
    # pid the app case asks for and the name the sidecar case asks for. No
    # witness of the table's SHAPE can catch this; only one that covers the
    # claim can.
    "proc-gone-table-contains-target": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(@(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } }) +
                 @([pscustomobject]@{ Id = 4242; ProcessName = "wenlan-server" }))
    }
""" + REAL_PROC_ABSENCE + r"""}
""",
    "proc-alive": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    return @([pscustomobject]@{ Id = 4242; ProcessName = "wenlan-server" })
}
""",
    "proc-probe-throws": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    throw (New-Object System.Management.Automation.RuntimeException "access is denied reading the process table")
}
""",

    # --- Get-CimInstance: the INDEPENDENT process witness -----------------
    # C5. Get-Process asked twice is one provider agreeing with itself. These
    # fixtures are the only thing in this file that can tell the difference.
    "cim-contains-target": r"""
function Get-CimInstance {
    param([string]$ClassName, [string]$Filter, [string]$ErrorAction)
    @(@(0..288 | ForEach-Object { [pscustomobject]@{ ProcessId = $_; Name = "p$_.exe" } }) +
      @([pscustomobject]@{ ProcessId = 4242; Name = "wenlan-server.exe" }))
}
""",
    "cim-throws": r"""
function Get-CimInstance {
    param([string]$ClassName, [string]$Filter, [string]$ErrorAction)
    throw (New-Object System.Management.Automation.RuntimeException "the WMI repository is not available")
}
""",
    "cim-long-table-without-pid4": r"""
function Get-CimInstance {
    param([string]$ClassName, [string]$Filter, [string]$ErrorAction)
    @(100..387 | ForEach-Object { [pscustomobject]@{ ProcessId = $_; Name = "p$_.exe" } })
}
""",
    "cim-short-table-with-pid4": r"""
function Get-CimInstance {
    param([string]$ClassName, [string]$Filter, [string]$ErrorAction)
    @(0, 4, 101, 102, 103, 104 | ForEach-Object { [pscustomobject]@{ ProcessId = $_; Name = "p$_.exe" } })
}
""",

    # --- the zip channel's wenlan-server inventory ------------------------
    # `Get-Process -Name wenlan-server` with a Path property, which is what
    # ownership is decided on.
    "srv-none": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
""" + REAL_PROC_ABSENCE + r"""}
""",
    # Ours, and it stays up through the whole poll.
    "srv-ours-alive": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    $script:SrvCimPids = @(4242)
    return @([pscustomobject]@{ Id = 4242; ProcessName = "wenlan-server"
        Path = "C:\Users\ci\AppData\Local\Programs\wenlan\wenlan-server.exe" })
}
""",
    # Ours, and it goes away after the kill. The happy path.
    "srv-ours-then-gone": r"""
$script:SrvReads = 0
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    $script:SrvReads++
    if ($script:SrvReads -le 1) {
        $script:SrvCimPids = @(4242)
        return @([pscustomobject]@{ Id = 4242; ProcessName = "wenlan-server"
            Path = "C:\Users\ci\AppData\Local\Programs\wenlan\wenlan-server.exe" })
    }
    $script:SrvCimPids = @()
""" + REAL_PROC_ABSENCE + r"""}
""",
    # C1, and the fixture the whole finding turns on: a wenlan-server that is
    # running from SOMEWHERE ELSE. Another worktree, another install, a
    # hand-built binary. The bare-name pipeline killed it; nothing here may.
    "srv-foreign-only": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    $script:SrvCimPids = @(5555)
    return @([pscustomobject]@{ Id = 5555; ProcessName = "wenlan-server"
        Path = "D:\worktrees\other\target\debug\wenlan-server.exe" })
}
""",
    # C1's harder half, and the reason the image test alone is not enough: the
    # zip channel installs to the DOCUMENTED per-user location, so a developer's
    # PRODUCTION daemon has the identical image path. Only "it was already
    # running when this script started" separates the two.
    "srv-preexisting-same-image": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    $script:SrvCimPids = @(3131)
    return @([pscustomobject]@{ Id = 3131; ProcessName = "wenlan-server"
        Path = "C:\Users\ci\AppData\Local\Programs\wenlan\wenlan-server.exe" })
}
$PreexistingServerPids = @(3131)
""",
    # An image that cannot be read is an identity that cannot be proved, and a
    # process this run cannot identify is neither proven its own nor proven
    # someone else's. Not killed, and not counted as stopped either.
    "srv-image-unreadable": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    $script:SrvCimPids = @(4242)
    return @([pscustomobject]@{ Id = 4242; ProcessName = "wenlan-server"; Path = $null })
}
""",
    "srv-read-throws": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    throw (New-Object System.Management.Automation.RuntimeException "access is denied reading the process table")
}
""",
    # The set of processes running at the start was never measured, so nothing
    # can be shown to belong to this run. Refusal, not a free hand.
    "srv-ownership-unmeasured": r"""
$PreexistingServerPids = $null
""",
    # D3: Get-Process SUCCEEDS and returns a NON-EMPTY set that is short by one.
    # This is the dangerous shape, not the loud one: 3131 is a pre-existing,
    # same-image daemon, and a snapshot that omits it hands this run a licence
    # to kill a stranger's process as though it had started it. The old witness
    # was reachable only from the absence exception, so nothing looked at this.
    "srv-partial-read": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    @([pscustomobject]@{ Id = 4242; ProcessName = "wenlan-server"
                         Path = "C:\Users\ci\AppData\Local\Programs\wenlan\wenlan-server.exe" })
}
function Get-CimInstance {
    param([string]$ClassName, [string]$Filter, [string]$ErrorAction)
    @(@(0..288 | ForEach-Object { [pscustomobject]@{ ProcessId = $_; Name = "p$_.exe" } }) +
      @([pscustomobject]@{ ProcessId = 4242; Name = "wenlan-server.exe" }) +
      @([pscustomobject]@{ ProcessId = 3131; Name = "wenlan-server.exe" }))
}
""",

    # --- the scheduled-task pair ------------------------------------------
    # Each of these changes ONE knob of the fixture in the PREAMBLE. The
    # sequence semantics (before / after registering / after deleting) are the
    # same everywhere, so a case says only how it differs.
    "task-already-registered": r"""
$script:TaskSeq = @($true, $true, $true)
""",
    "task-table-unreadable": r"""
$script:TaskCsvExit = 1
$script:TaskRowCount = 0
""",
    # Thirty rows and not one of them a Windows task. It is not a task table,
    # and "our task is not in it" is not a measurement of a table like that.
    "task-table-not-whole": r"""
$script:TaskRowsAreWindows = $false
""",
    # THE ISOLATE FOR THE EXIT CODE: schtasks enumerates most of the tree, hits
    # a folder it may not read, prints a table that passes every STRUCTURAL test
    # -- thirty rows, Windows tasks throughout -- and exits non-zero. Nothing but
    # the status says the enumeration did not finish.
    "task-query-exits-nonzero-with-full-table": r"""
$script:TaskCsvExit = 1
""",
    # THE ISOLATE FOR THE ROW FLOOR: exit 0, Windows tasks present, five rows,
    # and an independent enumeration just as small so the count rule cannot be
    # what rejects it.
    "task-table-tiny": r"""
$script:TaskRowCount = 5
$script:TaskCimCount = 4
""",
    # D4, AND THE ONE A ROW FLOOR CANNOT CATCH. Thirty well-formed rows, all of
    # them real Windows tasks, exit 0 -- a believable table by every structural
    # test there is. It is a PREFIX: the independent provider counts 207 tasks,
    # and schtasks prints one row per TRIGGER, so a whole table can never have
    # fewer rows than that. WenlanServer is below the cut.
    "task-table-truncated-prefix": r"""
$script:TaskSeq = @($false, $true, $false)
$script:TaskRowCount = 30
$script:TaskCimCount = 207
$script:TaskCimPresentOverride = "yes"
""",
    # D4: the table stops mid-record. The last row has two fields, not three,
    # which is the shape a stream cut short actually has.
    "task-table-torn-row": r"""
$script:TaskCsvShape = "torn"
""",
    # D4: the two providers answer the same question differently. Neither is a
    # measurement; a run does not get to pick the convenient one.
    "task-providers-disagree": r"""
$script:TaskCimPresentOverride = "yes"
$script:TaskSeq = @($false, $false, $false)
""",
    # D4: the independent read cannot be taken at all.
    "task-cim-unreadable": r"""
$script:TaskCimThrows = $true
""",
    # D1: `background on` did not take. Free before, still absent after, so
    # nothing at that name is this run's to end, switch off or delete.
    "task-registration-did-not-take": r"""
$script:TaskSeq = @($false, $false, $false)
""",
    # The task survived the delete: still present on the read after it.
    "task-still-registered-after-delete": r"""
$script:TaskSeq = @($false, $true, $true)
""",
    # D2: the delete FAILED and the post-delete read says absent anyway. The
    # previous `no-leftover-task` logged the delete and believed the query, so
    # this combination passed.
    "task-delete-failed": r"""
$script:TaskDeleteExit = 1
""",
    # D2: the delete succeeded, but the read that is supposed to confirm it
    # could not be taken. Unproven, not removed.
    "task-post-delete-query-unmeasurable": r"""
$script:TaskSeq = @($false, $true, $false)
$script:TaskCimThrows = $false
$script:TaskPostDeleteBreaks = $true
$script:TaskReadsBeforeBreak = 2
$__origSchtasks = ${function:schtasks.exe}
function schtasks.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    $joined = ($Rest -join " ")
    if ($joined -like "*/fo CSV*" -and $script:TaskReads -ge $script:TaskReadsBeforeBreak) {
        $script:TaskReads = $script:TaskReads + 1
        $global:LASTEXITCODE = 1
        return "ERROR: Access is denied."
    }
    return (& $__origSchtasks @Rest)
}
""",

    # C2. $ErrorActionPreference is set to Stop FOR THIS CASE ONLY, and schtasks
    # writes to stderr the way the real one does when the task is not running.
    #
    # MEASURED ON THE REFERENCE HOST (Windows PowerShell 5.1.26100.9278):
    #
    #   $ErrorActionPreference = 'Continue'
    #     & cmd.exe /c "echo benign 1>&2 & exit /b 0" 2>&1 | Out-Null
    #     -> $? = False, $Error.Count = 1, LASTEXITCODE = 0, the text discarded
    #   $ErrorActionPreference = 'Stop', the same call inside a function
    #     -> THREW System.Management.Automation.RemoteException, and the
    #        function NEVER REACHED ITS `return`
    #
    # A PowerShell FUNCTION cannot produce a NativeCommandError, so this stub
    # shells out to cmd.exe to make one. That is the single real native command
    # in this file; it echoes, and changes nothing.
    # ROUND 5, C6. The believable prefix a row FLOOR cannot see: forty rows,
    # every one well-formed and a real Windows task, exit 0 -- and only twenty
    # DISTINCT names in them, because schtasks prints one row per trigger. The
    # independent enumeration lists twenty-five tasks, so five are missing while
    # `40 -ge 25` holds comfortably. This is round-5 review's 220-vs-207 input,
    # scaled to the fixture.
    "task-table-prefix-longer-than-the-tasks": r"""
$script:TaskRowCount = 20
$script:TaskRowsPerTask = 2
$script:TaskCimCount = 25
""",
    # ROUND 5, C7. The task is registered, and registered under a different
    # CASE -- which is a name Windows resolves to the same task. The schtasks
    # side used a case-SENSITIVE StartsWith, so it alone reported absence.
    "task-registered-under-another-case": r"""
$script:TaskSeq = @($true, $true, $true)
$script:TaskNameCase = "upper"
""",
    # ROUND 5. Get-Process SUCCEEDS on the targeted read and hands back nothing.
    # A query that matched returns objects and one that did not throws, so this
    # is neither -- and the teardown sweep used to read it as an empty machine.
    "proc-name-silent-empty": r"""
function Get-Process {
    param([int[]]$Id, [string[]]$Name, [string]$ErrorAction)
    if (-not $PSBoundParameters.ContainsKey('Id') -and -not $PSBoundParameters.ContainsKey('Name')) {
        return @(0..287 | ForEach-Object { [pscustomobject]@{ Id = $_; ProcessName = "p$_" } })
    }
    return @()
}
""",
    # --- the teardown record the no-leftover-dirs rows read ---------------
    # ROUND 5, C4. The delete ran ten times and the tree is still there.
    # `Remove-Retry` returned nothing at all before this round, so this state
    # and a clean removal were the same silence.
    "dirs-install-delete-failed": r"""
$CleanupDirs[0].Removal = [pscustomobject]@{ State = "failed"; Attempts = 10
    Detail = "C:\Users\ci\AppData\Local\Programs\wenlan is still there after 10 delete attempts over about 5s; last error: System.IO.IOException: the process cannot access the file because it is being used by another process" }
""",
    # ROUND 5, C4. The INSTALL tree was not this run's to delete. The nsis row
    # examined only the data dir, so this passed while the install dir stood.
    "dirs-install-licence-refused": r"""
$CleanupDirs[0].Licence = [pscustomobject]@{ State = "refused"
    Detail = "this run did not create the install dir C:\Users\ci\AppData\Local\Programs\wenlan (before this run: present), so it is not this run's to delete" }
$CleanupDirs[0].Removal = $null
""",
    # ROUND 5, C1/C2. The tree was created by this run and REPLACED before the
    # cleanup: the marker this run wrote is not the one that is there now.
    "dirs-data-licence-refused": r"""
$CleanupDirs[1].Licence = [pscustomobject]@{ State = "refused"
    Detail = "the data dir C:\Users\ci\AppData\Local\wenlan is not bound to this run (not-this-run): the marker this run wrote is gone, so whatever is at this path now is not the tree this run created" }
$CleanupDirs[1].Removal = $null
""",
    # The delete reported success and the tree is still there.
    "dir-after-present": r"""
function Get-Item {
    param([string]$LiteralPath, [switch]$Force, [string]$ErrorAction)
    [pscustomobject]@{ FullName = $LiteralPath
        CreationTimeUtc = [datetime]::SpecifyKind([datetime]::Parse("2026-01-01T00:00:00"), [System.DateTimeKind]::Utc) }
}
""",
    # The confirming read could not be taken. Unproven, never "gone".
    "dir-after-unreadable": r"""
function Get-Item {
    param([string]$LiteralPath, [switch]$Force, [string]$ErrorAction)
    throw (New-Object System.UnauthorizedAccessException "access to the path is denied")
}
""",
    "schtasks-native-stderr-under-stop": r"""
$ErrorActionPreference = 'Stop'
function schtasks.exe {
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    $joined = ($Rest -join " ")
    if ($joined -like "*/fo CSV*") {
        return @(0..29 | ForEach-Object { '"\Microsoft\Windows\Fixture\Task' + $_ + '","N/A","Ready"' })
    }
    & cmd.exe /c "echo ERROR: The system cannot find the file specified. 1>&2 & exit /b 1" 2>&1
    $global:LASTEXITCODE = 1
}
""",

    # --- the product CLI after `background off` ---------------------------
    # The shipped `stopped-marker-error` row is the only `-ExpectFail` call site
    # in either channel, and its three outcomes are the three things the CLI can
    # do: refuse with the stopped marker (the measured negative the row wants),
    # answer normally (the daemon is still up, or autostart fired), or fail for
    # an unrelated reason (which proves nothing about the marker). The default
    # mode is `succeeds`, so a case that wants a refusal has to ask for one.
    "cli-search-refuses": '$script:CliSearchMode = "refuses"\n',
    "cli-search-fails-otherwise": '$script:CliSearchMode = "fails-otherwise"\n',
}

# Emitted into every driver, immediately before the block under test.
#
# A stub that is no longer in effect does not fail loudly: it answers from the
# real machine, and the case reports on a host nobody meant to measure.
# `$STUBBED$` is filled in with the names this particular driver defines, PLUS
# the ones in MUST_BE_STUBBED, which are listed rather than derived: a guard
# built only from what the driver defines cannot notice a stub that was deleted.
GUARD = r"""
# --- the stubs must still BE the stubs ---------------------------------
foreach ($n in @($STUBBED$)) {
    $c = @(Get-Command $n -CommandType Function -ErrorAction SilentlyContinue)
    if ($c.Count -ne 1 -or $c[0].ModuleName) {
        $m = if ($c.Count) { $c[0].ModuleName } else { "<not a function>" }
        Write-Host ("STUB-ESCAPED " + $n + " resolves to module '" + $m + "'; this driver would measure the real machine")
        exit 9
    }
}
"""

# --- THE SETUP LINE IS PART OF THE MEASURED SURFACE -------------------------
# ROUND 5 follow-up. The setup statements below run OUTSIDE `Check`, because
# that is where the shipped script runs them. Under an inherited
# `$ErrorActionPreference = 'Stop'` a setup line can THROW -- which is exactly
# how the C2 defect manifests when a control puts it back -- and the driver
# then exits with no ROW at all. `run_case` correctly refuses to call that a
# case failure, because "no row" is also what a driver that never started
# produces, and crediting one as the other is this directory's whole subject.
# The result was a real control stuck at COULD NOT MEASURE forever: the
# mutation broke something, and nothing could say what.
#
# So the setup gets a diagnosis instead of a silence. A throw is caught HERE,
# where it can still be attributed -- the exception type, its message, and the
# source line that raised it -- and printed as one tab-separated record before
# the driver exits. That is the difference the tri-state actually turns on: a
# driver that dies with this line is a case that FAILED for a named reason; a
# driver that dies without one is still a measurement that did not happen.
#
# This is deliberately general and not a fix for one control. Any mutation that
# kills a setup statement is a legitimate way for a defect to show itself, and
# every control gets to score it.
#
# `try`/`catch` is not a new scope in PowerShell, so every variable the setup
# assigns is still in scope for the Check block below it -- the wrapper changes
# what happens when the setup throws and nothing at all when it does not.
SETUP_OPEN = "try {"
SETUP_CLOSE = r"""} catch {
    $__why = $_.Exception.GetType().Name + ": " + $_.Exception.Message
    if ($_.InvocationInfo -and $_.InvocationInfo.Line) {
        $__why = $__why + " -- raised at driver line " + $_.InvocationInfo.ScriptLineNumber +
                 ": " + $_.InvocationInfo.Line.Trim()
    }
    Write-Host ("SETUP-THREW`t" + ($__why -replace "`r?`n", " "))
    exit 3
}"""

# Statements a Check block needs in front of it, because the shipped script sets
# the variable it closes over. Keyed by row; the value is (variable, statement).
# The pre-check refuses to run if a block closes over a variable no driver
# defines -- silence there would be $null.State, which reads as "could not
# measure" and would make an unmeasurable case pass for the wrong reason.
def _setup_vars(declared):
    """A setup entry's declared variable(s), always as a tuple.

    A setup may define more than one -- the two data-dir snapshots are two
    shipped statements and the row branches on both -- and a name left out here
    is a name the free-variable check would not know a driver defines.
    """
    return (declared,) if isinstance(declared, str) else tuple(declared)


def _grab(src, needle):
    """The one source line starting with `needle`, verbatim, left-stripped.

    Raises rather than returning a default: a setup line that quietly went
    missing would leave the variable undefined, which reads as $null, which
    reads as "could not measure" -- a case passing for the wrong reason, which
    is the exact failure this whole file is about.
    """
    # The file-scope `$X = $false` initialisers are skipped explicitly rather
    # than by taking "the last match": both of these names are declared at file
    # scope precisely so `finally` can read them, and a positional rule would
    # quietly pick the wrong line the day a third assignment appears.
    hits = [ln.strip() for ln in src.splitlines()
            if ln.strip().startswith(needle) and ln.strip() != needle.strip() + " $false"]
    if len(hits) != 1:
        raise ValueError("setup anchor %r matched %d source lines, wanted 1"
                         % (needle, len(hits)))
    return hits[0]


def ownership_setup(src, through="owned"):
    """The shipped ownership decision, EXTRACTED so a control can revert it.

    The sequence is the point and it is the shipped one: read the task table
    before anything runs, read it again after `background on`, and only then
    decide. `through="free"` stops after the first half, for the rows that only
    need the licence to CREATE.
    """
    lines = ["$preTask = Get-TaskPresence $TaskName",
             _grab(src, "$MayDriveTask = ")]
    if through != "free":
        lines += ["$postTask = Get-TaskPresence $TaskName",
                  _grab(src, "$TaskOwned = ")]
    return "\n".join(lines) + "\n"


# KEYED BY (channel, row), not by row alone. Both channels record
# `no-leftover-dirs` now, about different trees and from different code, and a
# row-keyed map could not tell the two apart -- it had to call the second one a
# collision and refuse, which would have left a row undriven.
SETUP = {
    ("zip", "daemon-stopped-before-recovery"): ("stopped", "$stopped = Stop-Daemon\n"),
    ("zip", "daemon-stopped-at-cleanup"): ("stoppedAtCleanup", "$stoppedAtCleanup = Stop-Daemon\n"),
    # The delete is the shipped one too, and its RESULT is what
    # `no-leftover-task` now requires, so it has to be a real value here.
    ("zip", "no-leftover-task"): (
        "TaskOwned",
        lambda src: ownership_setup(src) +
        '$TaskDeleteResult = $(if ($TaskOwned) '
        '{ Invoke-Native "schtasks.exe" @("/delete", "/tn", $TaskName, "/f") } '
        'else { $null })\n'),
    # The three CLI rows close over the same decision, taken at the same points.
    ("zip", "background-on (wenlan.exe background on)"):
        ("MayDriveTask", lambda src: ownership_setup(src, through="free")),
    ("zip", "background-off (wenlan.exe background off)"):
        ("TaskOwned", ownership_setup),
    ("zip", "background-on-again"):
        ("TaskOwned", ownership_setup),
    # ROUND 5. The teardown sweep, driven through the SHIPPED call rather than a
    # copy of it -- so a control that reverts Get-OwnedProcessesByImage's
    # tri-state is exercised here instead of being reverted under a driver that
    # kept running its own copy.
    ("nsis", "sidecar-sweep-measured"):
        ("ownedLeft", lambda src: _grab(src, "$ownedLeft = ") + "\n"),
    # BOTH snapshots, taken from the shipped statements rather than copied: the
    # pre one runs where the shipped script runs it (before the uninstaller),
    # the post one where it runs it (after). A copy here would keep the row
    # green under a control that deleted one of them from the source.
    ("nsis", "user-data-survives-uninstall"):
        (("preDataSnapshot", "postDataSnapshot"),
         lambda src: _grab(src, "$preDataSnapshot = Get-TreeFileDigests") + "\n"
                     + _grab(src, "$postDataSnapshot = ") + "\n"),
}

# ROUND 6 (Codex Sol), B1 + RANKED #2. THE `SETUP-THREW` MARKER WAS CREDITED
# FAR TOO FREELY, and what it bought was a green verdict.
#
# `classify_driver_output` used to score ANY line beginning `SETUP-THREW\t` as
# the case having FAILED. It required no exit status, no uniqueness, no absence
# of a ROW, no raising line and no anchor to a particular exception. So this
# driver output, with exit 0:
#
#     SETUP-THREW\tRuntimeException: unrelated harness failure
#     ROW\tprobe-row\tPASS\tthe expected detail
#
# was `fail` -- and inside a control, a `fail` on a must_fail case is a control
# that FIRED. `SETUP_CLOSE` wraps the WHOLE transitive setup operation, so any
# mutation anywhere under it produces the marker, and every control whose
# must_fail list mentions a row with a setup line could be satisfied by an
# unrelated setup regression. That is scripts/AGENTS.md's rule about a witness
# reached only from the exception path, in this harness's own scorer: it
# ratified "something outside Check declined", never "the reverted construct
# declined".
#
# So a setup death is now a case failure only where a CONTROL claims it, and
# only when the diagnosis says what that control's rule emits. Keyed by
# (control, case) rather than by case, because the marker is generic and the
# claim is not: the same case under a different mutation must not inherit it.
# Everything else -- an undeclared marker, a marker with a ROW beside it, a
# marker without exit 3, two markers, a marker naming a different exception --
# is COULD NOT MEASURE, which score_control already refuses to credit.
#
# MEASURED, on this host, with a standalone reproduction of the C2 shape (a
# `function schtasks.exe` whose body runs `& cmd.exe ... 1>&2 ... 2>&1` under
# an inherited `$ErrorActionPreference = 'Stop'`):
#
#   SETUP-THREW<TAB>RemoteException: ERROR: The system cannot find the file
#   specified.   -- raised at driver line 6: & cmd.exe /c "echo ERROR: ..."
#
# exit 3. `RemoteException` is the type PowerShell raises when a NATIVE command
# writes to stderr under `Stop` -- which is the entire mechanism C2 is about,
# and which no other mutation in this file can produce, because no other
# mutation injects a bare native call. The anchor is that type together with
# the stub's own message, so it names WHICH exception and WHICH stub raised it.
SETUP_THROW_WITNESS = {
    ("nc-stop-daemon-native-stderr-into-the-error-stream", "stop-daemon-native-stderr"):
        "RemoteException: ERROR: The system cannot find the file specified.",
}

# Variables the PREAMBLE defines for the extracted blocks.
PREAMBLE_VARS = {"Health", "App", "AppExe", "TaskName", "DataDir",
                 "OwnedServerImage", "PreexistingServerPids", "preTask",
                 "postTask", "MayDriveTask", "TaskDeleteResult", "CleanupDirs"}

# Environment a case runs with. GAUNTLET_HEALTH_TIMEOUT_SEC is removed for every
# other case, so a value in the developer's shell cannot change what a case
# measures.
CASE_ENV = {
    "health-timeout-env-below-floor": {"GAUNTLET_HEALTH_TIMEOUT_SEC": "2"},
    "health-timeout-env-above-ceiling": {"GAUNTLET_HEALTH_TIMEOUT_SEC": "86400"},
}

# --------------------------------------------------------------------------
# Cases: (name, subject, stubs, row, want_status, want_detail[, want_absent])
#
# `stubs` is a tuple of STUBS keys, layered in order over the PREAMBLE, because
# a case now chooses independently along four axes: which HTTP answer, which
# listener table, which process table, and which native command behaviour.
#
# `want_absent`, when given, is a substring the row detail must NOT contain.
# That is what makes "nothing else was killed" a real assertion rather than an
# absence nobody looked for: every driver's Check appends KILLED[...] to the
# row, so a case can require a pid to be in it and another to be out of it.
#
# CLI[...] is the same idea for the OTHER way a task gets driven, and it is why
# round 4 needed it: the refusal that matters is not "the row said Refusing", it
# is "wenlan.exe was never asked to touch the task". The stub records which verb
# it was given -- none / on / off -- and the refusal cases require the verb they
# refused to be ABSENT from the row. The positive direction needs no marker
# because only the stub emits "Installed and started Windows scheduled task";
# if that string is in the row, the stub produced it.
# --------------------------------------------------------------------------
ZIP_FUNCS = ["Invoke-Native", "Get-HealthTimeoutSec", "Get-HealthReachability",
             "Get-PortListenerWitness", "Get-PortListenerState",
             "Get-ServerProcessInventory", "Get-OwnedServerProcesses",
             "Get-ScheduledTaskWitness", "Get-TaskPresence",
             "Stop-Daemon"]
NSIS_FUNCS = ["Get-ProcessTableWitness", "Get-ProcessLiveness",
              "Get-OwnedProcessesByImage"]
# lib.ps1 is a THIRD subject, and it has to be. `Get-HealthReachability` no
# longer classifies the exception itself, it asks the three classifiers; and
# the CIM witnesses and `Get-DirPresence` are defined there once instead of
# once per channel, so this is where they are extracted from now. Driven
# exactly like the channel functions -- unstubbed, they would trip the
# stub-escape guard, and unmutatable, the reset/refusal discrimination and the
# witness co-variance rule would have no control defending them anywhere.
LIB_FUNCS = ["Get-WebExceptionShape", "Test-ConnectionRefused",
             "Test-HttpErrorResponse", "Get-CimProcessWitness",
             "Get-CimProcessSet", "Get-DirPresence"]
# WHICH CHANNEL'S CASES A CONTROL RE-RUNS. A control whose subject is a
# channel runs that channel's cases. lib.ps1 has no rows of its own, so a
# control on it runs the channels ITS OWN must_fail cases live in -- DERIVED,
# not listed. A hard-coded {"lib": "zip"} was right while every lib control
# was a health classifier; the moment a shared helper moved into lib.ps1 and a
# control on it named nsis cases, that mapping sent the control at the wrong
# channel and every one of its required red outcomes would have been reported
# as "survived" -- a control failure describing a defect that is not there.
def control_case_channels(subject, must_fail):
    """The set of channels whose cases this control re-runs."""
    if subject != "lib":
        return {subject}
    # Read off CASES itself -- defined below this point, so it is looked up
    # at call time rather than at import time.
    channel_of = {c[0]: c[1] for c in CASES}
    # An empty or wholly unknown must_fail runs EVERYTHING rather than
    # nothing: a control that reddens no case must be REPORTED, and a filter
    # that selected no cases at all would report nothing.
    return ({channel_of[m] for m in must_fail if m in channel_of}
            or set(channel_of.values()))

CASES = [
    # --- finding 1: the port probe -------------------------------------
    ("port-closed-measured", "zip", ("tcp-table-without-7878",),
     "port-7878-closed", "PASS", "measured closed"),
    ("port-busy", "zip", ("tcp-table-with-7878",),
     "port-7878-closed", "FAIL", "still listening (pid 4242)"),
    ("port-provider-fails", "zip", ("tcp-provider-throws",),
     "port-7878-closed", "FAIL", "could not measure"),
    ("port-table-empty", "zip", ("tcp-table-empty",),
     "port-7878-closed", "FAIL", "could not measure"),
    ("port-rows-unusable", "zip", ("tcp-rows-unusable",),
     "port-7878-closed", "FAIL", "carry no usable LocalPort"),
    # C6: an UNPARSEABLE port, not a null one. Without the defensive parse the
    # cast throws and the function never returns any of its three states.
    ("port-rows-unparseable", "zip", ("tcp-rows-unparseable",),
     "port-7878-closed", "FAIL", "carry no usable LocalPort"),
    # C4: the primary provider's table is believable and simply SHORT of our
    # row, and an INDEPENDENT provider finds what it missed.
    ("port-table-hides-7878-netstat-finds-it", "zip",
     ("tcp-table-without-7878", "netstat-shows-7878"),
     "port-7878-closed", "FAIL", "the two reads contradict each other"),
    ("port-witness-cannot-run", "zip", ("tcp-table-without-7878", "netstat-fails"),
     "port-7878-closed", "FAIL", "could not be corroborated"),
    ("port-witness-table-garbled", "zip", ("tcp-table-without-7878", "netstat-garbled"),
     "port-7878-closed", "FAIL", "not rows this parse understands"),
    ("port-witness-no-listeners", "zip", ("tcp-table-without-7878", "netstat-no-listeners"),
     "port-7878-closed", "FAIL", "no listening TCP row at all"),
    # The three rounds scripts/lib/host-process.sh had already taken and this
    # parse had not. Each pairs the primary table with a netstat that is
    # incomplete in a different way, and each must be UNMEASURABLE.
    ("port-witness-warning-beside-rows", "zip",
     ("tcp-table-without-7878", "netstat-warning-beside-rows"),
     "port-7878-closed", "FAIL", "not rows this parse understands"),
    # ...and the same table without the warning, which must still measure.
    ("port-witness-warning-free-still-measures", "zip",
     ("tcp-table-without-7878", "netstat-warning-free"),
     "port-7878-closed", "PASS", "measured closed"),
    ("port-witness-truncated-before-udp", "zip",
     ("tcp-table-without-7878", "netstat-truncated-before-udp"),
     "port-7878-closed", "FAIL", "no UDP row"),
    ("port-witness-tcp-after-udp", "zip",
     ("tcp-table-without-7878", "netstat-tcp-after-udp"),
     "port-7878-closed", "FAIL", "the sections are interleaved"),

    # --- finding 2: the health probe -----------------------------------
    ("health-off-refused", "zip", ("http-refused",),
     "health-unreachable-after-off", "PASS", "ConnectionRefused"),
    ("health-off-reachable", "zip", ("http-200",),
     "health-unreachable-after-off", "FAIL", "reachable"),
    ("health-off-timeout", "zip", ("http-timeout",),
     "health-unreachable-after-off", "FAIL", "could not measure"),
    ("health-off-dns", "zip", ("http-dns-failure",),
     "health-unreachable-after-off", "FAIL", "could not measure"),
    ("health-off-bad-uri", "zip", ("http-bad-uri",),
     "health-unreachable-after-off", "FAIL", "did not reach the network"),
    ("health-off-http-500", "zip", ("http-500",),
     "health-unreachable-after-off", "FAIL", "an HTTP error still proves reachable"),
    # B1: a reset is not a refusal.
    ("health-off-reset-by-live-listener", "zip", ("http-reset-live-listener",),
     "health-unreachable-after-off", "FAIL", "status ReceiveFailure, socket ConnectionReset"),
    ("health-off-connectfailure-reset", "zip", ("http-connectfailure-reset",),
     "health-unreachable-after-off", "FAIL", "status ConnectFailure, socket ConnectionReset"),
    # B5 and C7: the timeout has to outlast a ~2.05 s refusal, the floor has to
    # hold when the environment asks for less, and the CEILING has to hold when
    # it asks for a value that makes the run non-terminating.
    ("health-off-refused-slow", "zip", ("http-refused-after-2050ms",),
     "health-unreachable-after-off", "PASS", "ConnectionRefused"),
    ("health-timeout-env-below-floor", "zip", ("http-refused-after-2050ms",),
     "health-unreachable-after-off", "PASS", "ConnectionRefused"),
    ("health-timeout-env-above-ceiling", "zip", ("http-refused-unless-timeout-absurd",),
     "health-unreachable-after-off", "PASS", "ConnectionRefused"),
    ("health-recovery-ok", "zip", ("http-200",),
     "healthy-after-recovery", "PASS", "recovered"),
    ("health-recovery-refused", "zip", ("http-refused",),
     "healthy-after-recovery", "FAIL", "health unreachable after recovery"),
    ("health-recovery-timeout", "zip", ("http-timeout",),
     "healthy-after-recovery", "FAIL", "could not measure"),

    # --- B2 and C3: Stop-Daemon's state, and what it is a state ABOUT ----
    #
    # Every one of these pairs a PROCESS answer with a HEALTH answer that points
    # the other way, which is the whole of C3: the row is about the process, so
    # the health probe must not be able to decide it in either direction.
    ("stop-daemon-process-gone", "zip", ("srv-ours-then-gone", "http-200"),
     "daemon-stopped-before-recovery", "PASS", "daemon stopped before the recovery check", "KILLED[]"),
    ("stop-daemon-process-alive-port-refuses", "zip", ("srv-ours-alive", "http-refused"),
     "daemon-stopped-before-recovery", "FAIL", "is still running after Stop-Daemon"),
    ("stop-daemon-process-unmeasurable", "zip", ("srv-read-throws", "http-refused"),
     "daemon-stopped-before-recovery", "FAIL", "could not measure"),
    ("stop-daemon-image-unreadable", "zip", ("srv-image-unreadable", "http-refused"),
     "daemon-stopped-before-recovery", "FAIL", "could not measure", "KILLED[4242]"),
    # The absence of a wenlan-server is a NEGATIVE, so it is not taken from the
    # provider that would produce it by breaking.
    ("stop-daemon-absence-not-ratified", "zip", ("srv-none", "cim-contains-target", "http-refused"),
     "daemon-stopped-before-recovery", "FAIL", "not ratified"),
    # C1. The two fixtures the whole finding turns on. Both PASS -- none of this
    # run's processes is running -- and both must leave the other daemon alone.
    ("stop-daemon-spares-foreign-image", "zip", ("srv-foreign-only", "http-200"),
     "daemon-stopped-before-recovery", "PASS", "daemon stopped before the recovery check", "KILLED[5555]"),
    ("stop-daemon-spares-preexisting-pid", "zip", ("srv-preexisting-same-image", "http-200"),
     "daemon-stopped-before-recovery", "PASS", "daemon stopped before the recovery check", "KILLED[3131]"),
    # D3: a partial success is not a measurement of the set.
    ("stop-daemon-partial-process-read", "zip", ("srv-partial-read", "http-refused"),
     "daemon-stopped-before-recovery", "FAIL", "do not agree on which wenlan-server processes exist"),
    ("stop-daemon-ownership-unmeasured-kills-nothing", "zip",
     ("srv-ours-alive", "srv-ownership-unmeasured", "http-refused"),
     "daemon-stopped-before-recovery", "FAIL", "were never measured", "KILLED[4242]"),
    # C2. The native call writes to stderr under an inherited `Stop`. With the
    # fix the text is captured and Stop-Daemon returns; without it the function
    # throws before its `return` and the driver dies with no row at all.
    ("stop-daemon-native-stderr", "zip",
     ("srv-ours-then-gone", "http-refused", "schtasks-native-stderr-under-stop"),
     "daemon-stopped-before-recovery", "PASS", "cannot find the file specified"),
    # The SECOND caller. A tri-state that survives one boundary and not the
    # other is still a dropped measurement.
    ("stop-daemon-cleanup-unmeasurable", "zip", ("srv-read-throws", "http-refused"),
     "daemon-stopped-at-cleanup", "FAIL", "could not measure"),
    ("stop-daemon-cleanup-gone", "zip", ("srv-ours-then-gone", "http-200"),
     "daemon-stopped-at-cleanup", "PASS", "daemon stopped during cleanup"),

    # --- C1's other half: the scheduled task ----------------------------
    # --- D1/D2/D4: the scheduled task -------------------------------------
    # Ownership is now a SEQUENCE -- free before, present after -- so these
    # cases drive Get-TaskPresence twice through the shipped decision.
    ("task-owned-and-deleted", "zip", (),
     "no-leftover-task", "PASS", "the registration this run made is gone"),
    ("task-preexisting-is-not-ours", "zip", ("task-already-registered",),
     "no-leftover-task", "FAIL", "does not own WenlanServer"),
    ("task-table-unreadable-is-not-ours", "zip", ("task-table-unreadable",),
     "no-leftover-task", "FAIL", "does not own WenlanServer"),
    # The want_detail is the PROVENANCE message, not the generic ownership
    # line: the \Vendor\ table is refused by the provenance rule first and by
    # the name-set completeness rule second, so a case pinned to the generic
    # line stays green when provenance is deleted and its control catches
    # nothing. Pinned to the reason, it goes red.
    ("task-table-not-whole-is-not-ours", "zip", ("task-table-not-whole",),
     "no-leftover-task", "FAIL", "did not come from a real task table"),
    ("task-table-query-failed-is-not-ours", "zip", ("task-query-exits-nonzero-with-full-table",),
     "no-leftover-task", "FAIL", "does not own WenlanServer"),
    ("task-table-tiny-is-not-ours", "zip", ("task-table-tiny",),
     "no-leftover-task", "FAIL", "does not own WenlanServer"),
    # D4: the believable prefix. Nothing structural rejects it; only the count
    # against the independent enumeration does.
    ("task-truncated-prefix-is-not-ours", "zip", ("task-table-truncated-prefix",),
     "no-leftover-task", "FAIL", "truncated prefix"),
    ("task-torn-row-is-not-ours", "zip", ("task-table-torn-row",),
     "no-leftover-task", "FAIL", "not well-formed CSV records"),
    ("task-providers-disagree-is-not-ours", "zip", ("task-providers-disagree",),
     "no-leftover-task", "FAIL", "two providers contradict each other"),
    ("task-cim-unreadable-is-not-ours", "zip", ("task-cim-unreadable",),
     "no-leftover-task", "FAIL", "cannot be corroborated"),
    # D1's second half: the name was free, but the registration did not take,
    # so there is nothing here this run created.
    ("task-registration-did-not-take-is-not-ours", "zip", ("task-registration-did-not-take",),
     "no-leftover-task", "FAIL", "does not own WenlanServer"),
    ("task-survived-the-delete", "zip", ("task-still-registered-after-delete",),
     "no-leftover-task", "FAIL", "still registered after this run deleted it"),
    # D2: a FAILED delete may not be certified by a query, however that query
    # answers.
    ("task-delete-failed-is-not-clean", "zip", ("task-delete-failed",),
     "no-leftover-task", "FAIL", "the delete FAILED"),
    # D2: the confirming read could not be taken. Unproven, not removed.
    ("task-post-delete-query-unmeasurable", "zip", ("task-post-delete-query-unmeasurable",),
     "no-leftover-task", "FAIL", "could not measure whether WenlanServer is still registered"),

    # --- D1: the CLI reaches the same task --------------------------------
    ("background-on-runs-when-the-name-is-free", "zip", (),
     "background-on (wenlan.exe background on)", "PASS", "Installed and started Windows scheduled task"),
    ("background-on-refused-when-task-preexists", "zip", ("task-already-registered",),
     "background-on (wenlan.exe background on)", "FAIL", "Refusing to run 'wenlan background on'", "CLI[on]"),
    ("background-off-runs-when-owned", "zip", (),
     "background-off (wenlan.exe background off)", "PASS", "Background registration kept"),
    ("background-off-refused-when-not-owned", "zip", ("task-already-registered",),
     "background-off (wenlan.exe background off)", "FAIL", "would change the autostart state of a task belonging to someone else", "CLI[off]"),
    ("background-off-refused-when-registration-did-not-take", "zip", ("task-registration-did-not-take",),
     "background-off (wenlan.exe background off)", "FAIL", "would change the autostart state of a task belonging to someone else", "CLI[off]"),
    ("background-on-again-refused-when-not-owned", "zip", ("task-already-registered",),
     "background-on-again", "FAIL", "may not re-register at that name", "CLI[on]"),

    # --- the one shipped -ExpectFail row ---------------------------------
    # ROUND 6 (shell/release lane). Until now nothing drove a `-ExpectFail` row,
    # which is why the replica could declare the parameter and ignore it for a
    # whole review cycle without a single case noticing. These three are the
    # rule's three answers, and each of them is a different verdict: only the
    # first is the measured refusal the row claims.
    ("stopped-marker-refusal-measured", "zip", ("cli-search-refuses",),
     "stopped-marker-error (wenlan.exe search x)", "PASS",
     "daemon stopped by 'wenlan background off'", "RC[0]"),
    # The CLI answered the search instead of refusing -- the daemon is still
    # reachable, or autostart brought it back. Nonzero-exit-and-the-marker is
    # the whole rule, and this is the half that fails on the exit code.
    ("stopped-marker-cli-answered-instead", "zip", (),
     "stopped-marker-error (wenlan.exe search x)", "FAIL",
     "expected nonzero exit with substring: daemon stopped by; got: wenlan.exe search x"),
    # ...and the half that fails on the text: the CLI did fail, for a reason
    # that says nothing about whether `background off` left a stopped marker.
    ("stopped-marker-failed-for-another-reason", "zip", ("cli-search-fails-otherwise",),
     "stopped-marker-error (wenlan.exe search x)", "FAIL",
     "expected nonzero exit with substring: daemon stopped by; got: error: could not connect"),

    # --- finding 3: the process liveness probe -------------------------
    ("app-exited", "nsis", ("proc-gone-table-good",),
     "app-exited-after-kill", "PASS", "exited"),
    ("app-alive", "nsis", ("proc-alive",),
     "app-exited-after-kill", "FAIL", "still alive 10s after the identity-checked kill"),
    ("app-probe-fails", "nsis", ("proc-probe-throws",),
     "app-exited-after-kill", "FAIL", "could not measure"),
    ("app-absent-but-table-is-a-fragment", "nsis", ("proc-gone-table-fragment",),
     "app-exited-after-kill", "FAIL", "not ratified"),
    # B4: one fixture per witness, so each can be reverted alone.
    ("app-absent-but-long-table-lacks-pid4", "nsis", ("proc-gone-long-table-without-pid4",),
     "app-exited-after-kill", "FAIL", "no pid 4 (System)"),
    ("app-absent-but-short-table-has-pid4", "nsis", ("proc-gone-short-table-with-pid4",),
     "app-exited-after-kill", "FAIL", "a running Windows session has far more"),
    # B3: the witness table contains the process the probe called absent.
    ("app-absent-but-table-contains-it", "nsis", ("proc-gone-table-contains-target",),
     "app-exited-after-kill", "FAIL", "CONTAINS it (pid 4242)"),
    # C5: the Get-Process reads AGREE, believably, and the INDEPENDENT provider
    # is the only thing in the room that can say otherwise. This is the case
    # that the previous witness could not have.
    ("app-absent-but-wmi-has-it", "nsis", ("proc-gone-table-good", "cim-contains-target"),
     "app-exited-after-kill", "FAIL", "two INDEPENDENT providers contradict each other"),
    ("app-absent-but-wmi-unreadable", "nsis", ("proc-gone-table-good", "cim-throws"),
     "app-exited-after-kill", "FAIL", "the independent Win32_Process table could not be read"),
    ("app-absent-but-wmi-table-lacks-pid4", "nsis",
     ("proc-gone-table-good", "cim-long-table-without-pid4"),
     "app-exited-after-kill", "FAIL", "no ProcessId 4 (System)"),
    ("app-absent-but-wmi-table-is-short", "nsis",
     ("proc-gone-table-good", "cim-short-table-with-pid4"),
     "app-exited-after-kill", "FAIL", "Win32_Process table has only 6 rows"),
    ("sidecar-gone", "nsis", ("proc-gone-table-good",),
     "sidecar-exits-after-app", "PASS", "no wenlan-server process"),
    ("sidecar-orphan", "nsis", ("proc-alive",),
     "sidecar-exits-after-app", "FAIL", "orphan wenlan-server"),
    ("sidecar-probe-fails", "nsis", ("proc-probe-throws",),
     "sidecar-exits-after-app", "FAIL", "could not measure"),
    ("sidecar-absent-but-table-contains-it", "nsis", ("proc-gone-table-contains-target",),
     "sidecar-exits-after-app", "FAIL", "CONTAINS it (pid 4242)"),
    ("sidecar-absent-but-wmi-has-it", "nsis", ("proc-gone-table-good", "cim-contains-target"),
     "sidecar-exits-after-app", "FAIL", "two INDEPENDENT providers contradict each other"),

    # --- ROUND 5, C6: completeness is a comparison of NAMES ---------------
    # The row floor is a NECESSARY condition and nothing more. This table has
    # more rows than the independent enumeration has tasks and is still missing
    # five of them, because schtasks prints one row per TRIGGER.
    ("zip-task-prefix-longer-than-the-tasks", "zip", ("task-table-prefix-longer-than-the-tasks",),
     "no-leftover-task", "FAIL", "are not among them"),
    # --- ROUND 5, C7: the one unfolded name test --------------------------
    # A task really registered as WENLANSERVER is the same task. With the fold
    # both providers see it and the run refuses to touch it; without, only the
    # independent one does and the refusal comes from a DISAGREEMENT instead --
    # safe by accident, and about the wrong fact.
    ("zip-task-registered-under-another-case", "zip", ("task-registered-under-another-case",),
     "no-leftover-task", "FAIL", "is already registered"),

    # --- ROUND 5: the nsis teardown sweep is a MEASUREMENT ----------------
    # It collapsed every failure into an empty result, so access denied and a
    # broken provider read exactly like "no such process" -- and the directory
    # deletes ran afterwards on the strength of it.
    ("nsis-sweep-measured-none", "nsis", ("proc-gone-table-good",),
     "sidecar-sweep-measured", "PASS", "no 'wenlan-server' process is running"),
    ("nsis-sweep-measured-owned", "nsis", ("srv-ours-alive",),
     "sidecar-sweep-measured", "PASS", "owned: pid 4242"),
    ("nsis-sweep-read-fails", "nsis", ("proc-probe-throws",),
     "sidecar-sweep-measured", "FAIL", "could not read the wenlan-server processes"),
    ("nsis-sweep-absence-not-ratified", "nsis", ("proc-gone-table-good", "cim-contains-target"),
     "sidecar-sweep-measured", "FAIL", "not ratified"),
    ("nsis-sweep-silent-read", "nsis", ("proc-name-silent-empty",),
     "sidecar-sweep-measured", "FAIL", "silence is not absence"),
    ("nsis-sweep-image-unreadable", "nsis", ("srv-image-unreadable",),
     "sidecar-sweep-measured", "FAIL", "whose image cannot be read"),
    ("nsis-sweep-ownership-unmeasured", "nsis", ("srv-ownership-unmeasured",),
     "sidecar-sweep-measured", "FAIL", "were never measured"),

    # --- the user's data across the uninstall, in the nsis channel ---------
    # The claim is about the FILES. The data root's survival is guaranteed by
    # this run's own open handle, so every one of these fixtures has the root
    # standing and differs only underneath it.
    ("nsis-user-data-intact", "nsis", (),
     "user-data-survives-uninstall", "PASS", "byte-for-byte what they were"),
    ("nsis-user-data-file-erased", "nsis", ("user-data-file-erased",),
     "user-data-survives-uninstall", "FAIL", "took 1 and rewrote 0"),
    ("nsis-user-data-file-rewritten", "nsis", ("user-data-file-rewritten",),
     "user-data-survives-uninstall", "FAIL", "took 0 and rewrote 1"),
    ("nsis-user-data-post-read-failed", "nsis", ("user-data-post-read-failed",),
     "user-data-survives-uninstall", "FAIL", "could not measure"),
    ("nsis-user-data-pre-read-failed", "nsis", ("user-data-pre-read-failed",),
     "user-data-survives-uninstall", "FAIL", "could not measure"),
    ("nsis-user-data-empty-before", "nsis", ("user-data-empty-before",),
     "user-data-survives-uninstall", "FAIL", "there were no files under"),

    # --- ROUND 5, C4: the row that grades the teardown, in BOTH channels ---
    # The nsis row read only $DataDir. Its false green: the uninstaller removes
    # the install dir so `uninstall-removes-dir` passes, another installer
    # recreates the documented path with a file locked open, the install-dir
    # delete exhausts its retries in silence, the data-dir delete succeeds, and
    # the row certifies a clean machine with the install dir still standing.
    ("nsis-dirs-all-clean", "nsis", (),
     "no-leftover-dirs", "PASS", "every tree this run created is gone"),
    ("nsis-dirs-install-delete-failed", "nsis", ("dirs-install-delete-failed",),
     "no-leftover-dirs", "FAIL", "was not removed (failed)"),
    ("nsis-dirs-install-licence-refused", "nsis", ("dirs-install-licence-refused",),
     "no-leftover-dirs", "FAIL", "did not create the install dir"),
    ("nsis-dirs-data-licence-refused", "nsis", ("dirs-data-licence-refused",),
     "no-leftover-dirs", "FAIL", "is not bound to this run"),
    ("nsis-dirs-still-there", "nsis", ("dir-after-present",),
     "no-leftover-dirs", "FAIL", "is still there after this run deleted it"),
    ("nsis-dirs-post-read-unmeasurable", "nsis", ("dir-after-unreadable",),
     "no-leftover-dirs", "FAIL", "could not measure whether"),
    ("zip-dirs-all-clean", "zip", (),
     "no-leftover-dirs", "PASS", "every tree this run created is gone"),
    ("zip-dirs-install-delete-failed", "zip", ("dirs-install-delete-failed",),
     "no-leftover-dirs", "FAIL", "was not removed (failed)"),
    ("zip-dirs-install-licence-refused", "zip", ("dirs-install-licence-refused",),
     "no-leftover-dirs", "FAIL", "did not create the install dir"),
    ("zip-dirs-data-licence-refused", "zip", ("dirs-data-licence-refused",),
     "no-leftover-dirs", "FAIL", "is not bound to this run"),
    ("zip-dirs-still-there", "zip", ("dir-after-present",),
     "no-leftover-dirs", "FAIL", "is still there after this run deleted it"),
    ("zip-dirs-post-read-unmeasurable", "zip", ("dir-after-unreadable",),
     "no-leftover-dirs", "FAIL", "could not measure whether"),
]

# --------------------------------------------------------------------------
# Controls: (name, why, subject, old, new, must_fail)
# Every case of that subject not named in must_fail must SURVIVE.
# --------------------------------------------------------------------------
CONTROLS = [
    ("nc-port-provider-failure-is-closed",
     "finding 1: a TCP provider that could not run is recorded as a closed port",
     "zip",
     """    } catch {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the TCP listener table could not be read ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }""",
     """    } catch {
        return [pscustomobject]@{ State = "none"; OwningProcess = $null
            Detail = "INJECTED: a provider failure read as a closed port" }
    }""",
     ["port-provider-fails"]),

    ("nc-port-empty-table-is-closed",
     "finding 1's witness: an empty listener table is accepted as an idle machine",
     "zip",
     """    if ($table.Count -lt 1) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the TCP listener table came back empty; a Windows host always has listening sockets, so this is a failed read, not an idle machine" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "INJECTED: the empty-table witness is gone" }
    }""",
     ["port-table-empty"]),

    ("nc-port-unusable-rows-are-a-closed-port",
     "finding 1's other witness: rows with no usable LocalPort are compared "
     "against anyway, so an empty comparison over rows that exist reads as a "
     "closed port",
     "zip",
     """    if ($unusable -ne 0) {""",
     """    if ($false) {""",
     ["port-rows-unusable", "port-rows-unparseable"]),

    ("nc-port-row-parse-is-a-bare-cast-outside-the-handler",
     "C6: the defensive TryParse goes back to `[int]$_.LocalPort` in a "
     "Where-Object, so an UNPARSEABLE port throws instead of returning a state",
     "zip",
     """    $parsed = New-Object System.Collections.Generic.List[object]
    $unusable = 0
    try {
        foreach ($row in $table) {
            $n = 0
            $raw = if ($null -eq $row.LocalPort) { "" } else { "$($row.LocalPort)" }
            if ([int]::TryParse($raw, [ref]$n) -and $n -gt 0) {
                $parsed.Add([pscustomobject]@{ Port = $n; OwningProcess = $row.OwningProcess })
            } else { $unusable++ }
        }
    } catch {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the listener table's rows could not be read ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }""",
     """    $parsed = @($table | ForEach-Object { [pscustomobject]@{ Port = [int]$_.LocalPort; OwningProcess = $_.OwningProcess } })
    $unusable = @($table | Where-Object { $null -eq $_.LocalPort -or [int]$_.LocalPort -le 0 }).Count""",
     ["port-rows-unparseable"]),

    ("nc-port-independent-witness-dropped",
     "C4: the port's negative is taken from the whole-table read alone, so a "
     "table that is merely SHORT of our row reads as a closed port",
     "zip",
     """    $witness = Get-PortListenerWitness $Port
    if ($witness.State -eq "found") {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $witness.OwningProcess
            Detail = "the listener table ($($table.Count) rows) has no row for $Port, but $($witness.Detail); the two reads contradict each other, so neither is a measurement" }
    }
    if ($witness.State -ne "none") {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the listener table ($($table.Count) rows) has no row for $Port, but that absence could not be corroborated: $($witness.Detail)" }
    }
    return [pscustomobject]@{ State = "none"; OwningProcess = $null
        Detail = "measured closed: $($table.Count) listening sockets read, none on $Port, and $($witness.Detail)" }""",
     """    return [pscustomobject]@{ State = "none"; OwningProcess = $null
        Detail = "INJECTED: measured closed on the whole-table read alone: $($table.Count) rows, none on $Port" }""",
     ["port-table-hides-7878-netstat-finds-it", "port-witness-cannot-run",
      "port-witness-table-garbled", "port-witness-no-listeners",
      "port-witness-warning-beside-rows", "port-witness-truncated-before-udp",
      "port-witness-tcp-after-udp"]),

    ("nc-port-witness-row-shape-not-checked",
     "C4's parse: every line claiming to be a protocol row no longer has to BE "
     "one, so a torn or localised table matches nothing and reads as free",
     "zip",
     """        $wellFormed = (($f[0] -eq "TCP" -and $f.Count -eq 5 -and $f[4] -match '^\\d+$') -or
                       ($f[0] -eq "UDP" -and $f.Count -eq 4 -and $f[3] -match '^\\d+$'))""",
     """        $wellFormed = ($f[0] -eq "TCP" -or $f[0] -eq "UDP")""",
     ["port-witness-table-garbled"]),

    # ROUND 4 of scripts/lib/host-process.sh's parse, which this copy was three
    # rounds behind. The mutation is verbatim the shape it shipped with.
    ("nc-port-witness-skips-non-protocol-lines",
     "round 4: a line whose first token is neither TCP nor UDP is skipped "
     "instead of counted, so a status-0 warning merged beside the rows leaves "
     "an incomplete table reading as a measured negative",
     "zip",
     """        if (-not $wellFormed) {
            if ($rows -ne 0) { $notARow++ } else { $preamble++ }
            continue
        }""",
     """        if ($f[0] -ne "TCP" -and $f[0] -ne "UDP") { continue }
        if (-not $wellFormed) { $notARow++; continue }""",
     ["port-witness-warning-beside-rows"]),

    ("nc-port-witness-end-witness-dropped",
     "round 5: nothing requires the UDP section, so a table truncated after a "
     "well-formed prefix -- every remaining row valid -- reads as a closed port",
     "zip",
     """    if ($udp -lt 1) {""",
     """    if ($false) {""",
     ["port-witness-truncated-before-udp"]),

    ("nc-port-witness-section-order-assumed",
     "round 6: the TCP-before-UDP ordering the end witness rests on is assumed "
     "rather than checked, so an interleaved stream ratifies its own TCP half",
     "zip",
     """    if ($tcpAfterUdp -ne 0) {""",
     """    if ($false) {""",
     ["port-witness-tcp-after-udp"]),

    ("nc-port-witness-empty-is-free",
     "C4's other parse guard: a netstat with no listening row at all is read as "
     "an idle machine rather than a failed read",
     "zip",
     """    if ($listening -lt 1) {""",
     """    if ($false) {""",
     ["port-witness-no-listeners"]),

    ("nc-health-timeout-is-down",
     "finding 2: any web failure without a verdict -- timeout, DNS, a reset -- "
     "is recorded as a stopped daemon",
     "zip",
     """        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "the health probe failed without reaching a verdict ($($shape.Type)$(if ($shape.Status) { ", status $($shape.Status)" })$(if ($shape.SocketError) { ", socket $($shape.SocketError) at depth $($shape.SocketDepth)" }): $($ex.Message))" }""",
     """        return [pscustomobject]@{ State = "down"
            Detail = "INJECTED: any web failure read as a stopped daemon" }""",
     ["health-off-timeout", "health-off-dns", "health-recovery-timeout",
      "health-off-reset-by-live-listener", "health-off-connectfailure-reset"]),

    ("nc-health-non-web-error-is-down",
     "finding 2's other half: a probe that never reached the network is recorded as a stopped daemon",
     "zip",
     """            return [pscustomobject]@{ State = "unmeasurable"
                Detail = "the health probe did not reach the network ($($shape.Type): $($ex.Message))" }""",
     """            return [pscustomobject]@{ State = "down"
                Detail = "INJECTED: a probe that never ran read as a stopped daemon" }""",
     ["health-off-bad-uri"]),

    ("nc-health-reset-is-down",
     "B1: a connect that failed with a RESET is put back in the negative, so a "
     "live peer slamming the door certifies a stopped daemon",
     "zip",
     """        if (Test-ConnectionRefused $shape) {""",
     """        if ((Test-ConnectionRefused $shape) -or
            ($shape.Status -eq "ConnectFailure" -and $shape.SocketError -eq "ConnectionReset")) {""",
     ["health-off-connectfailure-reset"]),

    # The classifier moved into lib.ps1, so the control that defends it follows
    # it: without the walk the SocketException an IOException hides is never
    # reached, and the reset a live listener produces cannot be named at all.
    ("nc-health-inner-chain-not-walked",
     "B1's evidence: only the direct InnerException is read, so the reset a "
     "live listener produces cannot even be named in the row",
     "lib",
     """    $depth = 0
    $cur = $Exception.InnerException
    while ($cur) {
        $depth++
        if ($cur.GetType().FullName -eq "System.Net.Sockets.SocketException") {
            $shape.SocketError = "" + $cur.SocketErrorCode
            $shape.SocketDepth = $depth
            break
        }
        $cur = $cur.InnerException
    }""",
     """    $cur = $Exception.InnerException
    if ($cur -and $cur.GetType().FullName -eq "System.Net.Sockets.SocketException") {
        $shape.SocketError = "" + $cur.SocketErrorCode
        $shape.SocketDepth = 1
    }""",
     ["health-off-reset-by-live-listener"]),

    ("nc-health-timeout-below-the-refusal",
     "B5: the client timeout goes back under the ~2.05 s a refusal takes, so "
     "every genuine refusal arrives as an unmeasurable Timeout",
     "zip",
     "    $floor = 5\n",
     "    $floor = 2\n",
     ["health-off-refused-slow", "health-timeout-env-below-floor"]),

    ("nc-health-timeout-floor-not-enforced",
     "B5's other half: the floor stops holding, so an environment variable can "
     "put the timeout back under the refusal",
     "zip",
     "    if ($n -lt $floor) {\n",
     "    if ($false) {\n",
     ["health-timeout-env-below-floor"]),

    ("nc-health-timeout-ceiling-not-enforced",
     "C7: the ceiling stops holding, so an inherited GAUNTLET_HEALTH_TIMEOUT_SEC "
     "makes each probe -- and twenty of them per stop-wait loop -- unbounded",
     "zip",
     "    if ($n -gt $ceiling) {\n",
     "    if ($false) {\n",
     ["health-timeout-env-above-ceiling"]),

    ("nc-stop-daemon-state-dropped",
     "B2: Stop-Daemon goes back to printing its tri-state and returning nothing",
     "zip",
     "    return [pscustomobject]@{ State = $state; Detail = $detail; Health = $reach.State }\n}",
     "    $null = $state\n}",
     ["stop-daemon-process-gone", "stop-daemon-process-alive-port-refuses",
      "stop-daemon-absence-not-ratified", "stop-daemon-spares-foreign-image",
      "stop-daemon-spares-preexisting-pid",
      "stop-daemon-ownership-unmeasured-kills-nothing",
      "stop-daemon-partial-process-read",
      "stop-daemon-native-stderr", "stop-daemon-cleanup-gone"]),

    ("nc-stop-daemon-caller-ignores-state",
     "B2 at the caller: the row goes back to reporting the state instead of "
     "branching on it",
     "zip",
     """    Check -Name "daemon-stopped-before-recovery" -Script {
        if ($stopped.State -eq "stopped") { Write-Output "daemon stopped before the recovery check: $($stopped.Detail)"; return }
        if ($stopped.State -eq "alive") { throw "the daemon this run started is still running after Stop-Daemon: $($stopped.Detail); the recovery check below would be testing nothing" }
        throw "could not measure whether the daemon stopped before the recovery check: $($stopped.Detail); recorded as unproven, not as stopped"
    }""",
     """    Check -Name "daemon-stopped-before-recovery" -Script {
        Write-Output "INJECTED: Stop-Daemon settled at '$($stopped.State)' -- $($stopped.Detail)"
    }""",
     # NOT stop-daemon-native-stderr: that case is about the tri-state surviving
     # the NATIVE call, and a caller that merely prints the state still prints
     # the schtasks text the case asserts on. A control has to be pinned to the
     # fix it reverts, and listing a case it does not redden would be the same
     # untested green this file exists to refuse.
     ["stop-daemon-process-gone", "stop-daemon-process-alive-port-refuses",
      "stop-daemon-process-unmeasurable", "stop-daemon-image-unreadable",
      "stop-daemon-absence-not-ratified", "stop-daemon-spares-foreign-image",
      "stop-daemon-spares-preexisting-pid",
      "stop-daemon-ownership-unmeasured-kills-nothing",
      "stop-daemon-partial-process-read"]),

    ("nc-stop-daemon-verdict-is-reachability",
     "C3: Stop-Daemon goes back to returning the HEALTH tri-state, so a refused "
     "socket certifies a stop over a process that is still running",
     "zip",
     """    $state = if ($last.State -eq "none") { "stopped" }
             elseif ($last.State -eq "owned") { "alive" }
             else { "unmeasurable" }""",
     """    $state = if ($reach.State -eq "down") { "stopped" }
             elseif ($reach.State -eq "reachable") { "alive" }
             else { "unmeasurable" }""",
     ["stop-daemon-process-gone", "stop-daemon-process-alive-port-refuses",
      "stop-daemon-process-unmeasurable", "stop-daemon-image-unreadable",
      "stop-daemon-absence-not-ratified", "stop-daemon-spares-foreign-image",
      "stop-daemon-spares-preexisting-pid",
      "stop-daemon-ownership-unmeasured-kills-nothing",
      "stop-daemon-partial-process-read",
      "stop-daemon-cleanup-unmeasurable", "stop-daemon-cleanup-gone"]),

    ("nc-stop-daemon-kills-by-name",
     "C1: the ownership filter is dropped, so Stop-Daemon force-kills every "
     "wenlan-server it can see -- a developer's production daemon, another "
     "worktree's, a hand-started one",
     "zip",
     """    $ours = @($inv.Processes | Where-Object {
        [string]::Equals($_.Path, $script:OwnedServerImage, [System.StringComparison]::OrdinalIgnoreCase) -and
        ($script:PreexistingServerPids -notcontains $_.Id) })""",
     """    $ours = @($inv.Processes)""",
     ["stop-daemon-spares-foreign-image", "stop-daemon-spares-preexisting-pid"]),

    ("nc-stop-daemon-image-check-dropped",
     "C1's first half alone: the image is no longer compared, so a wenlan-server "
     "from another worktree is this run's to kill",
     "zip",
     """        [string]::Equals($_.Path, $script:OwnedServerImage, [System.StringComparison]::OrdinalIgnoreCase) -and
        ($script:PreexistingServerPids -notcontains $_.Id) })""",
     """        ($script:PreexistingServerPids -notcontains $_.Id) })""",
     ["stop-daemon-spares-foreign-image"]),

    ("nc-stop-daemon-preexisting-check-dropped",
     "C1's second half alone, and the one the image test cannot cover: a "
     "production daemon installed to the DOCUMENTED location has the identical "
     "image path, so only 'it was already running' separates it from ours",
     "zip",
     """        ($script:PreexistingServerPids -notcontains $_.Id) })""",
     """        $true })""",
     ["stop-daemon-spares-preexisting-pid"]),

    ("nc-stop-daemon-unmeasured-ownership-kills",
     "C1: an ownership set that was never measured stops being a refusal, so a "
     "run that could not tell whose daemon is whose kills anyway",
     "zip",
     """    if ($null -eq $script:PreexistingServerPids) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "the wenlan-server processes already running when this script started were never measured, so no process can be shown to belong to this run and none may be killed" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "INJECTED: an unmeasured ownership set is not a refusal" }
    }""",
     ["stop-daemon-ownership-unmeasured-kills-nothing"]),

    ("nc-stop-daemon-unreadable-image-is-not-ours",
     "C1: a process whose image cannot be read stops being a refusal, so an "
     "identity this run could not prove is treated as an absence",
     "zip",
     """    if ($unreadable.Count -ne 0) {""",
     """    if ($false) {""",
     ["stop-daemon-image-unreadable"]),

    ("nc-stop-daemon-absence-not-ratified",
     "C5 in the zip channel: 'there is no wenlan-server' is believed from the "
     "provider that would say exactly that by breaking",
     "zip",
     """        $w = Get-CimProcessWitness -Name $name
        if (-not $w.Ok) {""",
     """        $w = Get-CimProcessWitness -Name $name
        if ($false) {""",
     ["stop-daemon-absence-not-ratified"]),

    ("nc-stop-daemon-native-stderr-into-the-error-stream",
     "C2: the schtasks call goes back to `2>&1 | Out-Null`, so a benign stderr "
     "line throws under an inherited Stop and Stop-Daemon never returns its "
     "tri-state at all",
     "zip",
     """        $r = Invoke-Native "schtasks.exe" @("/end", "/tn", $script:TaskName)
        $notes.Add("schtasks /end /tn $($script:TaskName): ran=$($r.Ran) exit=$($r.ExitCode) $($r.Output -replace "`r?`n", ' ')")""",
     """        & schtasks.exe /end /tn $script:TaskName 2>&1 | Out-Null
        $notes.Add("INJECTED: schtasks /end status discarded")""",
     ["stop-daemon-native-stderr"]),

    ("nc-task-ownership-dropped",
     "C1: `no-leftover-task` stops requiring that this run registered the task, "
     "so a run that deleted nothing certifies a clean machine -- and the "
     "cleanup that pairs with it deletes a registration it never made",
     "zip",
     """        if (-not $TaskOwned) { throw "this run does not own $TaskName (free before: $MayDriveTask -- $($preTask.State): $($preTask.Detail)) (present after registering: $($postTask.State): $($postTask.Detail)), so it did not delete it and cannot claim the machine was left as it was found; recorded as unproven" }""",
     """        if ($false) { throw "INJECTED: ownership no longer required" }""",
     ["task-preexisting-is-not-ours", "task-table-unreadable-is-not-ours",
      "task-table-not-whole-is-not-ours", "task-table-query-failed-is-not-ours",
      "task-table-tiny-is-not-ours", "task-truncated-prefix-is-not-ours",
      "task-torn-row-is-not-ours", "task-providers-disagree-is-not-ours",
      "task-cim-unreadable-is-not-ours",
      "task-registration-did-not-take-is-not-ours",
      # ROUND 5: the two new refusals reach this row through the same throw.
      "zip-task-prefix-longer-than-the-tasks",
      "zip-task-registered-under-another-case"]),

    ("nc-stopped-marker-row-stops-asserting-a-refusal",
     # ROUND 6 (shell/release lane). The only row in either channel that asserts
     # a REFUSAL stops asserting one: `-ExpectFail` becomes `-Expect`, and the
     # row now certifies the stopped marker whenever the CLI answers a search
     # normally after `background off` -- which is the state the marker exists to
     # rule out. What defends the REPLICA is one level up, and it is worth being
     # exact about which: with `-ExpectFail` declared and ignored, all three
     # cases below go red against the SHIPPED source -- the refusal is scored
     # FAIL, the normal answer is scored PASS, and the unrelated failure is
     # scored with the wrong detail -- so the run is red before any control is
     # reached. REPLICA_MUTATIONS reverts that branch and requires the driven
     # probes to notice it. THIS control's job is the ordinary one: to show the
     # three cases are pinned to the row still asserting a refusal, and are not
     # passing for some reason the row does not control.
     "ROUND 6: the shipped -ExpectFail row is downgraded to -Expect, so a CLI "
     "that answered the search after `background off` certifies the stopped "
     "marker instead of a refusal proving it",
     "zip",
     ''' -ExpectFail "daemon stopped by" -Script { & wenlan.exe search x }''',
     ''' -Expect "daemon stopped by" -Script { & wenlan.exe search x }''',
     ["stopped-marker-refusal-measured", "stopped-marker-cli-answered-instead",
      "stopped-marker-failed-for-another-reason"]),

    ("nc-task-table-completeness-dropped",
     # ROUND 6 (Codex Sol) NARROWED this, and the correction is in the WHY,
     # not in the control: the case is falsifiable and stays, the sentence that
     # described it was claiming more than the mutation shows. It used to say
     # deleting this branch "licenses this run to delete" the task. It does not.
     # The \Vendor\ fixture is refused twice over -- by this PROVENANCE branch
     # first, and by the later `$missing.Count -ne 0` name-set rule second,
     # because 25 CIM task names are absent from 30 \Vendor\ rows. With the
     # branch gone the fixture is still refused, with "are not among them".
     # What goes red is the case's want_detail, which is pinned to this rule's
     # own message, so what this control proves is exactly that: the provenance
     # branch and its wording are still present and still the reason a table
     # from nowhere is refused. That is worth a control -- a refusal arriving
     # from a different rule is a different measurement, and the two are worth
     # telling apart -- but it is not "the licence to delete".
     "C1's witness: the PROVENANCE branch goes, so a table with no "
     "\\Microsoft\\Windows\\ row in it is no longer refused for having come from "
     "nowhere. The later name-set completeness rule still refuses this fixture, "
     "so what this pins is the branch and its message, not the licence to delete",
     "zip",
     """    if (-not @($rows | Where-Object { $_ -like '*\\Microsoft\\Windows\\*' }).Count) {""",
     """    if ($false) {""",
     ["task-table-not-whole-is-not-ours"]),

    ("nc-task-query-failure-is-absence",
     "C1: a task query that FAILED is read as a task that is not there, which "
     "is this whole file's defect in the scheduler namespace",
     "zip",
     """    if ($r.ExitCode -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "schtasks /query exited $($r.ExitCode), so the task table was not read: $($r.Output -replace "`r?`n", ' ')" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ State = "unmeasurable"; Detail = "INJECTED" }
    }""",
     # Only the fixture whose table passes every STRUCTURAL test can isolate
     # this. `task-table-unreadable` is caught by the row floor as well, so it
     # would have made this control look pinned when it was not.
     ["task-table-query-failed-is-not-ours"]),

    ("nc-task-table-row-floor-dropped",
     "C1: the independent task enumeration stops having to be BIG, so a "
     "five-task reply licenses this run to delete a task it never registered",
     "zip",
     """    if ($all.Count -lt 10) {
        return [pscustomobject]@{ Ok = $false; Present = $false; Count = $all.Count; Names = @()
            Detail = "the independent task enumeration returned only $($all.Count) tasks; a Windows install has far more, so it is a failed read" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ Ok = $false; Present = $false; Count = 0; Names = @(); Detail = "INJECTED" }
    }""",
     ["task-table-tiny-is-not-ours"]),

    # ---- ROUND 4 ---------------------------------------------------------
    ("nc-background-on-not-gated",
     "D1: `wenlan background on` stops being gated, so a pre-existing user task "
     "is REWRITTEN AND STARTED at the fixed product name -- the same task the "
     "schtasks gate refuses to touch, reached through the CLI instead",
     "zip",
     """        if (-not $MayDriveTask) {
            throw ("$TaskName was '$($preTask.State)' before this run, so registering at that name would take over a task this run did not create. " +
                   "Refusing to run 'wenlan background on'; the documented flow is UNTESTED here, which is not the same as broken.")
        }""",
     """        # INJECTED: no gate""",
     ["background-on-refused-when-task-preexists"]),

    ("nc-background-off-not-gated",
     "D1: `wenlan background off` stops being gated, so this run switches off "
     "the autostart of a scheduled task belonging to the developer",
     "zip",
     """        if (-not $TaskOwned) {
            throw ("this run does not own $TaskName (free before: $MayDriveTask, present after registering: '$($postTask.State)'), " +
                   "so switching it off would change the autostart state of a task belonging to someone else. Refusing.")
        }""",
     """        # INJECTED: no gate""",
     ["background-off-refused-when-not-owned",
      "background-off-refused-when-registration-did-not-take"]),

    ("nc-background-on-again-not-gated",
     "D1: the re-registration stops being gated",
     "zip",
     """        if (-not $TaskOwned) { throw "not attempted: this run does not own $TaskName, so it may not re-register at that name" }""",
     """        # INJECTED: no gate""",
     ["background-on-again-refused-when-not-owned"]),

    ("nc-task-ownership-ignores-registration-result",
     "D1's second half: ownership goes back to being inferred from the "
     "pre-state alone, so a `background on` that never registered still leaves "
     "the run believing the task at that name is its own to delete",
     "zip",
     """    $TaskOwned = ($MayDriveTask -and $postTask.State -eq "present")""",
     """    $TaskOwned = $MayDriveTask""",
     ["task-registration-did-not-take-is-not-ours",
      "background-off-refused-when-registration-did-not-take"]),

    ("nc-leftover-task-query-failure-is-absence",
     "D2: `no-leftover-task` goes back to reading a non-zero `schtasks /query "
     "/tn` as absence, so access denied, a scheduler fault and a genuine "
     "removal are one answer -- the defect this file exists to disprove, in the "
     "row that certifies the machine was left as it was found",
     "zip",
     """        $t = Get-TaskPresence $TaskName
        if ($t.State -eq "absent") { Write-Output "the registration this run made is gone: $($t.Detail)"; return }
        if ($t.State -eq "present") { throw "$TaskName is still registered after this run deleted it: $($t.Detail)" }
        throw "could not measure whether $TaskName is still registered: $($t.Detail); recorded as unproven, not as removed"
""",
     """        $q = Invoke-Native "schtasks.exe" @("/query", "/tn", $TaskName)
        if ($q.ExitCode -eq 0) { throw "$TaskName task still registered: $($q.Output)" }
        Write-Output "INJECTED: a failed query read as absence: exit $($q.ExitCode)"
""",
     # task-owned-and-deleted too: the injected version reports the exit code
     # instead of the removal, so even the row that SHOULD pass no longer says
     # anything that was measured.
     ["task-post-delete-query-unmeasurable", "task-survived-the-delete",
      "task-owned-and-deleted"]),

    ("nc-leftover-task-delete-result-ignored",
     "D2's other half: the delete's own status stops being required, so a "
     "delete that FAILED followed by a query that also failed passes",
     "zip",
     """        if ($null -eq $TaskDeleteResult) { throw "this run owned $TaskName but no delete was attempted, so nothing establishes the registration is gone; recorded as unproven" }
        if (-not $TaskDeleteResult.Ran) { throw "the $TaskName delete could not be run ($($TaskDeleteResult.Output)), so the registration this run made may still be there; recorded as unproven" }
        if ($TaskDeleteResult.ExitCode -ne 0) { throw "schtasks /delete /tn $TaskName exited $($TaskDeleteResult.ExitCode) ($($TaskDeleteResult.Output -replace "`r?`n", ' ')); the delete FAILED, so this run left its own registration behind" }""",
     """        # INJECTED: the delete result is only logged, never required""",
     ["task-delete-failed-is-not-clean"]),

    ("nc-task-row-shape-not-validated",
     "D4: rows stop having to be well-formed records, so a table cut off "
     "mid-record still answers about WenlanServer",
     "zip",
     """    $malformed = @($rows | Where-Object { $_ -notmatch $shape })""",
     """    $malformed = @()""",
     ["task-torn-row-is-not-ours"]),

    ("nc-task-completeness-dropped",
     "D4/C6: the completeness test goes entirely, so a believable truncated "
     "PREFIX -- well-formed Windows tasks, exit 0 -- answers about WenlanServer "
     "from a table nobody established was whole",
     "zip",
     """    if ($missing.Count -ne 0) {""",
     """    if ($false) {""",
     ["task-truncated-prefix-is-not-ours", "zip-task-prefix-longer-than-the-tasks"]),

    ("nc-task-completeness-is-a-row-floor",
     "ROUND 5, C6, and the finding stated exactly: `$rows.Count -lt $w.Count` "
     "is a NECESSARY condition, not completeness. schtasks prints one row per "
     "TRIGGER and CIM one per TASK, so the counts are not comparable: a "
     "well-formed prefix with MORE rows than there are tasks still passes it "
     "while omitting tasks, and an omitted WenlanServer then reads as absent "
     "from both providers and authorises registering over it",
     "zip",
     """    $missing = @($w.Names | Where-Object {
        -not [string]::Equals($_, $target, [System.StringComparison]::OrdinalIgnoreCase) -and
        -not $rowNames.Contains($_) })
    if ($missing.Count -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "schtasks printed $($rows.Count) rows naming $($rowNames.Count) tasks, but $($missing.Count) of the $($w.Count) tasks the independent enumeration lists are not among them (first: '$($missing[0])'); the table is a truncated prefix, not the whole table" }
    }""",
     """    $missing = @()
    if ($rows.Count -lt $w.Count) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "INJECTED row floor: schtasks returned $($rows.Count) rows but the independent enumeration reports $($w.Count) tasks, so the table is a truncated prefix" }
    }""",
     ["zip-task-prefix-longer-than-the-tasks"]),

    ("nc-task-name-match-is-case-sensitive",
     "ROUND 5, C7: the schtasks-side name test goes back to a case-SENSITIVE "
     "StartsWith, so a task really registered as WENLANSERVER is invisible to "
     "one of the two providers and the refusal comes from a disagreement "
     "instead of from the task being there",
     "zip",
     """$_.StartsWith($want + ",", [System.StringComparison]::OrdinalIgnoreCase)""",
     """$_.StartsWith($want + ",")""",
     ["zip-task-registered-under-another-case"]),

    ("nc-task-providers-not-cross-checked",
     "D4: the two providers stop having to agree, so whichever answered first "
     "decides",
     "zip",
     """    if ($present -ne $w.Present) {""",
     """    if ($false) {""",
     # NOT task-truncated-prefix-is-not-ours: the count rule catches that one
     # first, so it is not this control's to claim.
     ["task-providers-disagree-is-not-ours"]),

    ("nc-task-independent-provider-dropped",
     "D4: the independent enumeration is dropped entirely, leaving schtasks "
     "ratifying schtasks",
     "zip",
     """    if (-not $w.Ok) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "the schtasks table cannot be corroborated: $($w.Detail)" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ State = "unmeasurable"; Detail = "INJECTED" }
    }""",
     ["task-cim-unreadable-is-not-ours"]),

    ("nc-process-set-not-corroborated",
     "D3: the independent read goes back to covering only the total-absence "
     "exception, so a Get-Process that SUCCEEDS with an incomplete set produces "
     "a snapshot missing a pre-existing daemon -- which the ownership filter "
     "then classifies as one this run started, and kills",
     "zip",
     """    $missing = @($cs.Pids | Where-Object { $gp -notcontains $_ })
    $extra = @($gp | Where-Object { $cs.Pids -notcontains $_ })
    if ($missing.Count -ne 0 -or $extra.Count -ne 0) {""",
     """    $missing = @()
    $extra = @()
    if ($false) {""",
     ["stop-daemon-partial-process-read"]),


    ("nc-process-probe-failure-is-absence",
     "finding 3: a process-table read that failed is recorded as a process that exited",
     "nsis",
     """        if (-not $isAbsence) {
            return [pscustomobject]@{ State = "unmeasurable"
                Detail = "Get-Process for $what failed without answering ($($ex.GetType().FullName), id '$fqid': $($ex.Message))" }
        }""",
     """        if (-not $isAbsence) {
            return [pscustomobject]@{ State = "gone"
                Detail = "INJECTED: any Get-Process failure read as an exited process" }
        }""",
     ["app-probe-fails", "sidecar-probe-fails"]),

    ("nc-process-table-witness-dropped",
     "finding 3's witness: an absence is believed with no witness at all",
     "nsis",
     """        $witness = Get-ProcessTableWitness -Id $Id -Name $Name
        if (-not $witness.Ok) {""",
     """        $witness = Get-ProcessTableWitness -Id $Id -Name $Name
        if ($false) {""",
     ["app-absent-but-table-is-a-fragment", "app-absent-but-long-table-lacks-pid4",
      "app-absent-but-short-table-has-pid4", "app-absent-but-table-contains-it",
      "app-absent-but-wmi-has-it", "app-absent-but-wmi-unreadable",
      "app-absent-but-wmi-table-lacks-pid4", "app-absent-but-wmi-table-is-short",
      "sidecar-absent-but-table-contains-it", "sidecar-absent-but-wmi-has-it"]),

    ("nc-witness-pid4-check-dropped",
     "B4: the pid-4 witness alone is reverted, and only a LONG table missing "
     "pid 4 can notice",
     "nsis",
     """    if (-not @($all | Where-Object { $_.Id -eq 4 }).Count) {""",
     """    if ($false) {""",
     ["app-absent-but-long-table-lacks-pid4"]),

    ("nc-witness-row-floor-dropped",
     "B4: the row floor alone is reverted, and only a SHORT table containing "
     "pid 4 can notice",
     "nsis",
     """    if ($all.Count -lt 10) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the process table has only $($all.Count) entries; a running Windows session has far more" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ Ok = $false; Detail = "INJECTED" }
    }""",
     ["app-absent-but-short-table-has-pid4"]),

    ("nc-witness-not-covariant",
     "B3: the witness stops checking whether the table contains the process "
     "whose absence it is ratifying",
     "nsis",
     """    if ($present.Count -ne 0) {
        return [pscustomobject]@{ Ok = $false
            Detail = ("the targeted read said there is no $what, but this table of $($all.Count) processes CONTAINS it (pid " +
                      (($present | ForEach-Object { $_.Id }) -join ", ") + "); the two reads contradict each other, so neither is a measurement") }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ Ok = $false; Detail = "INJECTED" }
    }""",
     ["app-absent-but-table-contains-it", "sidecar-absent-but-table-contains-it"]),

    ("nc-witness-independent-read-dropped",
     "C5: the second PROVIDER is dropped, leaving one provider read twice -- "
     "which establishes that two Get-Process reads agree, not that the absence "
     "was independently witnessed",
     "nsis",
     """    $cim = Get-CimProcessWitness -ProcessId $Id -Name $Name
    if (-not $cim.Ok) {""",
     """    $cim = Get-CimProcessWitness -ProcessId $Id -Name $Name
    if ($false) {""",
     ["app-absent-but-wmi-has-it", "app-absent-but-wmi-unreadable",
      "app-absent-but-wmi-table-lacks-pid4", "app-absent-but-wmi-table-is-short",
      "sidecar-absent-but-wmi-has-it"]),

    ("nc-cim-witness-not-covariant",
     "the independent read stops being asked about the process whose absence "
     "it ratifies, so it is back to answering 'is this a process table'",
     "lib",
     """    if ($present.Count -ne 0) {
        return [pscustomobject]@{ Ok = $false
            Detail = ("Get-Process reported no $what, but WMI's Win32_Process table of $($all.Count) rows CONTAINS it (pid " +
                      (($present | ForEach-Object { $_.ProcessId }) -join ", ") + "); two INDEPENDENT providers contradict each other, so neither is a measurement") }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ Ok = $false; Detail = "INJECTED" }
    }""",
     # The teardown sweep's absence rests on this same co-variance rule -- it
     # refuses to believe Get-Process's "no wenlan-server" when the independent
     # Win32_Process table names one -- so it belongs on this control's roster
     # rather than being an unpinned case.
     ["app-absent-but-wmi-has-it", "sidecar-absent-but-wmi-has-it",
      "nsis-sweep-absence-not-ratified"]),

    ("nc-cim-witness-shape-not-checked",
     "the independent table stops being checked for completeness, so a "
     "believable fragment ratifies the absence it was never asked about",
     "lib",
     """    if (-not @($all | Where-Object { $_.ProcessId -eq 4 }).Count) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the Win32_Process table has $($all.Count) rows but no ProcessId 4 (System); it is not the whole table" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ Ok = $false; Detail = "INJECTED: shape not checked" }
    }""",
     ["app-absent-but-wmi-table-lacks-pid4"]),

    # ---- ROUND 5 ---------------------------------------------------------
    ("nc-sweep-failure-is-absence",
     "ROUND 5: the teardown sweep goes back to collapsing every failed read "
     "into an empty result, so access denied and a broken provider produce what "
     "'there is no such process' produces -- and the directory deletes below it "
     "run on the strength of a measurement that never happened",
     "nsis",
     """        $ex = $_.Exception
        $fqid = "$($_.FullyQualifiedErrorId)"
        $typeName = if ($null -ne $ex) { $ex.GetType().FullName } else { "" }
        $isAbsence = ($typeName -eq "Microsoft.PowerShell.Commands.ProcessCommandException") -and
                     ($fqid -like "NoProcessFoundForGivenName,*")
        if (-not $isAbsence) {
            return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
                Detail = "the '$Name' process table could not be read ($typeName, id '$fqid': $($ex.Message)); nothing was killed and nothing may be deleted on the strength of this" }
        }
        $w = Get-CimProcessWitness -Name $Name
        if (-not $w.Ok) {
            return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
                Detail = "Get-Process reported no '$Name', but that absence is not ratified: $($w.Detail)" }
        }
        return [pscustomobject]@{ State = "measured"; Processes = @()
            Detail = "no '$Name' process is running; $($w.Detail)" }""",
     """        return [pscustomobject]@{ State = "measured"; Processes = @()
            Detail = "no '$Name' process is running (INJECTED: every failure of the read is an absence)" }""",
     ["nsis-sweep-read-fails", "nsis-sweep-absence-not-ratified"]),

    ("nc-sweep-silence-is-absence",
     "ROUND 5: a targeted read that SUCCEEDS with nothing in it goes back to "
     "meaning no such process, although a query that matched returns objects "
     "and one that did not throws",
     "nsis",
     """    if ($found.Count -eq 0) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "Get-Process for '$Name' returned no error and no process; silence is not absence" }
    }""",
     """    if ($false) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @(); Detail = "INJECTED" }
    }""",
     ["nsis-sweep-silent-read"]),

    ("nc-sweep-unreadable-image-is-not-ours",
     "ROUND 5: a wenlan-server whose image cannot be read goes back to being "
     "quietly spared and the sweep called complete, although an identity this "
     "run cannot prove may be its own daemon holding the tree about to be "
     "deleted",
     "nsis",
     """        if (-not $path) {
            return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
                Detail = "pid $($p.Id) is a '$Name' whose image cannot be read, so this run can show neither that it is its own nor that it is not; nothing killed and nothing deleted" }
        }""",
     """        if (-not $path) { $spared.Add("pid $($p.Id) (INJECTED: image unreadable, so unidentifiable)"); continue }""",
     ["nsis-sweep-image-unreadable"]),

    ("nc-nsis-leftover-dirs-install-tree-not-examined",
     "ROUND 5, C4, and the hole exactly as found: the row goes back to reading "
     "only the data dir, so an install dir this run could not remove -- or was "
     "never entitled to remove -- leaves the row green",
     "nsis",
     """        $bad = @(); $unproven = @()
        foreach ($d in $CleanupDirs) {""",
     """        $bad = @(); $unproven = @()
        foreach ($d in @($CleanupDirs[1])) {""",
     ["nsis-dirs-install-delete-failed", "nsis-dirs-install-licence-refused"]),

    ("nc-nsis-leftover-dirs-removal-report-ignored",
     "ROUND 5, C4's other half: the delete's OWN report stops being required, "
     "so ten exhausted retries and a clean removal are the same silence again",
     "nsis",
     """            if ($r.State -ne "removed") { $bad += "$($d.Name) $($d.Path) was not removed ($($r.State)): $($r.Detail)"; continue }""",
     """            if ($false) { $bad += "INJECTED: the delete's own report is not required"; continue }""",
     ["nsis-dirs-install-delete-failed"]),

    ("nc-nsis-leftover-dirs-licence-not-required",
     "ROUND 5, C1/C2: a tree this run was refused permission to delete stops "
     "making the row UNPROVEN, so a run that deleted nothing certifies that it "
     "left the machine as it found it",
     "nsis",
     """            if ($lic.State -eq "refused") { $unproven += $lic.Detail; continue }""",
     """            if ($false) { $unproven += "INJECTED: a refused licence no longer makes this row unproven" }""",
     ["nsis-dirs-install-licence-refused", "nsis-dirs-data-licence-refused"]),

    # The defect this row replaced: the observation read the data ROOT, which
    # this run keeps undeletable with its own DeleteOnClose handle, so it was
    # true whatever the uninstaller did to the files under it.
    ("nc-nsis-user-data-files-not-compared",
     "the row stops comparing the files and rests on the data root, which this "
     "run's own open handle guarantees -- so an uninstaller that erased every "
     "user file still passes",
     "nsis",
     """        foreach ($rel in $preDataSnapshot.Files.Keys) {""",
     """        foreach ($rel in @()) {""",
     ["nsis-user-data-file-erased", "nsis-user-data-file-rewritten"]),

    ("nc-nsis-user-data-post-read-failure-is-survival",
     "a post-uninstall snapshot that could not be taken stops making the row "
     "unproven, so a failed read becomes a measured loss or a measured survival "
     "depending only on what the pre-read happened to hold",
     "nsis",
     """        if ($postDataSnapshot.State -ne "taken") {
            throw "could not measure whether the user's data survived: $($postDataSnapshot.Detail); recorded as unproven, not as survived"
        }""",
     """        if ($false) { throw "INJECTED: the post-uninstall snapshot no longer has to have been taken" }""",
     ["nsis-user-data-post-read-failed"]),

    ("nc-nsis-user-data-pre-read-failure-is-survival",
     "the pre-uninstall snapshot stops having to have been taken, so the row "
     "compares against nothing",
     "nsis",
     """        if ($null -eq $preDataSnapshot -or $preDataSnapshot.State -ne "taken") {""",
     """        if ($false) {""",
     ["nsis-user-data-pre-read-failed"]),

    ("nc-nsis-user-data-nothing-to-lose-is-survival",
     "an empty pre-uninstall snapshot stops making the row unproven, so a run "
     "that had no user data certifies that the uninstaller left it alone",
     "nsis",
     """        if ($preDataSnapshot.Files.Count -lt 1) {""",
     """        if ($false) {""",
     ["nsis-user-data-empty-before"]),

    ("nc-zip-leftover-dirs-install-tree-not-examined",
     "ROUND 5, C4 in the other channel: the row reads only the data dir",
     "zip",
     """        $bad = @(); $unproven = @()
        foreach ($d in $CleanupDirs) {""",
     """        $bad = @(); $unproven = @()
        foreach ($d in @($CleanupDirs[1])) {""",
     ["zip-dirs-install-delete-failed", "zip-dirs-install-licence-refused"]),

    ("nc-zip-leftover-dirs-removal-report-ignored",
     "ROUND 5, C4's other half in the zip channel: the delete's own report "
     "stops being required",
     "zip",
     """            if ($r.State -ne "removed") { $bad += "$($d.Name) $($d.Path) was not removed ($($r.State)): $($r.Detail)"; continue }""",
     """            if ($false) { $bad += "INJECTED: the delete's own report is not required"; continue }""",
     ["zip-dirs-install-delete-failed"]),

    ("nc-zip-leftover-dirs-licence-not-required",
     "ROUND 5, C1/C2 in the zip channel: a refused licence stops making the row "
     "unproven",
     "zip",
     """            if ($lic.State -eq "refused") { $unproven += $lic.Detail; continue }""",
     """            if ($false) { $unproven += "INJECTED: a refused licence no longer makes this row unproven" }""",
     ["zip-dirs-install-licence-refused", "zip-dirs-data-licence-refused"]),
]


# --------------------------------------------------------------------------
# Every name that can reach the machine, listed rather than derived. The PREAMBLE
# defines all of them, so a driver in which one is not a local function is a
# driver that lost a stub, and it dies before the block under test runs.
#
# Start-Sleep is here because the DERIVED check below put it here: it was
# stubbed in the PREAMBLE and missing from this list, so a driver that lost the
# stub would have run the shipped twenty-attempt poll at its real wall clock
# and no guard would have said so. The hand-written list was wrong on its first
# contact with the derived one, which is the argument for having both.
MUST_BE_STUBBED = ["Get-NetTCPConnection", "Get-Process", "Invoke-WebRequest",
                   "Get-CimInstance", "Stop-Process", "schtasks.exe",
                   "netstat.exe", "Stop-OwnedServerProcess", "Stop-ProcessByImage",
                   "Start-Sleep", "Get-ScheduledTask", "wenlan.exe", "Get-Item",
                   "Get-TreeFileDigests"]

# Cmdlets that must never appear in EXTRACTED text. The channel scripts delete
# %LOCALAPPDATA%\wenlan, run installers, unpack archives and KILL PROCESSES;
# this harness is safe only because the regions it lifts out contain none of
# that. That is a property of where the braces happen to fall today, so it is
# asserted on every build rather than assumed -- if a future edit moves one of
# these inside a probe or a Check block, the harness refuses to write the driver.
#
# GetProcessById and .Kill() are the newest entries and the most important: the
# two shipped killers are stubbed rather than extracted, and this is what makes
# that a guarantee rather than a habit.
FORBIDDEN_IN_EXTRACT = [
    "Remove-Item", "Remove-ItemProperty", "Set-ItemProperty", "New-Item",
    "Start-Process", "Expand-Archive", "Copy-Item", "Move-Item",
    "Set-Content", "Add-Content", "Out-File", "Invoke-Expression",
    "Restart-Computer", "Stop-Computer", "Register-ScheduledTask",
    "GetProcessById", ".Kill()",
]

# THE ALIAS BYPASS. The list above is matched as a SUBSTRING, and `ri` is not a
# substring of `Remove-Item`. Every entry above has a short alias that reaches
# exactly the same cmdlet and reads straight past it, so the aliases are refused
# too -- at COMMAND POSITION only, because `ri`, `sc`, `mi` and `sp` are all
# common inside ordinary words and a bare substring test would refuse the whole
# file.
#
# NOTE, MEASURED, on why the stubbed names need no equivalent: an alias resolves
# to its target NAME and the name is then looked up again, so a script-scope
# FUNCTION wins. On this host, with `function Invoke-WebRequest` defined,
# `iwr`, `curl`, `gps` and `ps` all reached the stub. Aliases cannot escape a
# stub; they can only escape a substring ban, which is what this closes.
FORBIDDEN_ALIASES = [
    "ri", "rm", "rmdir", "rd", "del", "erase",           # Remove-Item
    "ni", "md", "mkdir",                                 # New-Item
    "sp", "rp",                                          # Set/Remove-ItemProperty
    "saps", "start",                                     # Start-Process
    "spps", "kill",                                      # Stop-Process
    "cpi", "copy", "cp",                                 # Copy-Item
    "mi", "move", "mv",                                  # Move-Item
    "sc", "ac",                                          # Set/Add-Content
    "iex",                                               # Invoke-Expression
]
# `(?![-\w.])` and not `\b`: `\b` matches between the `t` of `start` and the
# hyphen of `Start-Sleep`, so every alias here would fire on the cmdlet it is
# the abbreviation of. An alias is a WHOLE command name -- if a hyphen follows,
# this is a Verb-Noun and the scans below cover it instead.
ALIAS_AT_COMMAND_POSITION = re.compile(
    r"(?:^|[|;{(&]|\|\|)[ \t]*(" + "|".join(FORBIDDEN_ALIASES) + r")(?![-\w.])",
    re.M | re.I)

# Commands that cannot change anything: pure language and formatting. Anything
# in an extracted region that is NOT here, NOT stubbed, and NOT one of the
# shipped functions the driver defines for itself is a name that could reach the
# real machine, and the pre-check refuses to build a driver containing it.
#
# This is the answer to "does the stub-escape guard cover every cmdlet the
# edited regions can now reach" -- DERIVED on every run rather than asserted
# once. MUST_BE_STUBBED is a hand-written list, and a hand-written list is
# exactly the thing that goes stale the day someone adds a call.
#
# ROUND 6: `Add-Type` came OFF this list. It compiles and loads arbitrary C#,
# which is a way to reach the machine that no name-based ban can read -- the
# same family as the call-operator and .NET-method holes closed below. Measured
# on this revision: zero occurrences in any extracted function or Check block
# (the PREAMBLE's own New-HttpErrorException uses it, and the PREAMBLE is not
# scanned because it is this harness's text, not the channel's). So removing it
# costs nothing today and refuses the day a channel edit moves one inside a
# probe -- which is exactly when someone would want to be told.
INERT_COMMANDS = {
    "ForEach-Object", "Where-Object", "New-Object", "Write-Host", "Write-Output",
    "Write-Error", "Select-Object", "Sort-Object", "Measure-Object", "Out-String",
    "Join-Path", "Split-Path", "Test-Path", "Select-String",
    "Get-Date", "Get-Command", "Compare-Object", "Group-Object",
}

# `Invoke-Native` takes the executable as a PARAMETER, which is the one place
# in the extracted text where a command name is data rather than syntax -- and
# therefore the one place the checks above cannot see. Every call site must
# name a double-quoted literal from this set, both of which are stubbed.
NATIVE_ALLOWED = {"netstat.exe", "schtasks.exe"}
# The single dynamic invocation the extracted text is allowed to contain: the
# one inside Invoke-Native, whose argument the rule above pins down.
DYNAMIC_INVOKE_ALLOWED = "& $File @Arguments"

# --------------------------------------------------------------------------
# ROUND 6 (Codex Sol), RANKED #1: THE EXTRACTION GUARD WAS A DENY-LIST WITH
# THREE DOORS LEFT OPEN, and this guard is the only thing between the driver
# and %LOCALAPPDATA%\wenlan.
#
# The dynamic-invocation ban recognised exactly `& $var` (optionally `& $var
# @splat`). Every one of these read straight past it, past the substring ban,
# past the alias scan and past the reachability scan, and every one of them
# deletes the developer's real memorydb:
#
#     & ("Remove-" + "Item") -LiteralPath "$env:LOCALAPPDATA\wenlan" -Recurse -Force
#     [System.IO.Directory]::Delete("$env:LOCALAPPDATA\wenlan", $true)
#     (Get-Item $p).Delete($true)
#
# The first is a command name built at run time -- there is no `Remove-Item`
# token anywhere in it. The second and third never name a command at all.
#
# A deny-list cannot close this: the set of spellings that resolve to a
# destructive call is not enumerable (string concatenation alone makes it
# infinite). An ALLOW-LIST can, because the extracted surface is small and
# measured. Three constructs are the only ways extracted PowerShell can invoke
# anything, and each now has its own allow-list, DERIVED FROM THE SHIPPED TEXT
# ON THIS REVISION and refusing everything else:
#
#   1. the call operator `&`      -> CALL_TARGET_ALLOWED
#   2. a static .NET member `::`  -> STATIC_MEMBER_ALLOWED
#   3. an instance method `.X()`  -> INSTANCE_METHOD_ALLOWED
#
# WHAT THIS STILL DOES NOT COVER, stated because the comment it replaces
# claimed a coverage it did not have. These scans are LEXICAL and run over
# non-comment lines; they are not a PowerShell parse. A property ASSIGNMENT
# with a side effect (`$x.Enabled = $true` on a COM object) invokes nothing
# syntactically and is not seen. Neither is a destructive cmdlet whose name
# arrives as a `-Name` parameter to something already allow-listed. What is
# closed is every route from extracted text to a CALL: no `&` except the pinned
# Invoke-Native body and stubbed executables, no Invoke-Expression, no
# `.Invoke(`, no `::Create`, no Add-Type, and Invoke-Native's own argument
# pinned to two quoted literals. There is no evaluator left inside the
# allow-lists, which is the property the extraction depends on.

# `&` may name a command LITERALLY only if that command is a function the
# driver defines. Derived from MUST_BE_STUBBED rather than hand-listed, for the
# reason the reachability scan already gives: a hand list is the thing that
# goes stale the day someone adds a call. Today this is
# {schtasks.exe, netstat.exe, wenlan.exe}; the shipped text uses only
# `& wenlan.exe background on|off`, and a bare `& schtasks.exe /end|/delete` is
# refused a second time by FORBIDDEN_IN_SOURCE.
CALL_TARGET_ALLOWED = {n.lower() for n in MUST_BE_STUBBED if n.lower().endswith(".exe")}

# Every `[Type]::Member` the shipped functions and Check blocks name on this
# revision, measured, lower-cased because PowerShell resolves type and member
# names case-insensitively. All six are pure: a parse, a comparison, a
# comparer, two comparison enums and a WebExceptionStatus constant. Nothing
# here touches the filesystem, a process or the registry.
#
# The point of pinning the SET rather than banning known-bad members: both
# channels already contain [System.IO.File]::WriteAllText, [Guid]::NewGuid and
# [System.Diagnostics.Process]::GetProcessById OUTSIDE the extracted regions.
# The day a brace moves one of those inside a probe, this refuses -- which is
# a deny-list's best case and an allow-list's default one.
STATIC_MEMBER_ALLOWED = {
    "[int]::tryparse",
    "[string]::equals",
    "[system.net.webexceptionstatus]::connectfailure",
    "[system.stringcomparer]::ordinalignorecase",
    "[system.stringcomparison]::ordinal",
    "[system.stringcomparison]::ordinalignorecase",
}

# Every instance method the shipped functions and Check blocks CALL on this
# revision, measured, lower-cased for the same reason. List/HashSet mutation
# and string inspection; none of them can reach outside the process.
# `.Kill()` and `.Delete()` are absent, which is the whole point -- they are
# refused by not being here rather than by being named somewhere.
INSTANCE_METHOD_ALLOWED = {
    "add", "contains", "containskey", "gettype", "indexof", "startswith",
    "substring", "toarray", "tostring", "trim",
}

# `&` followed by what it invokes. The alternation order matters: the pinned
# Invoke-Native form must win over the bare-word branch, which would otherwise
# capture only `$File`.
CALL_OPERATOR_TARGET = re.compile(
    r"""&[ \t]*(?P<target>\$\w+(?:[ \t]+@\w+)?|"[^"\n]*"|'[^'\n]*'|[^\s(){}\[\];|&<>'"]+)""")

# A `[Type]::Member` reference, and -- separately -- any `::` that is being
# used as a member access at all. The second is what makes this an allow-list:
# `$type::Delete($p)` and `[System.IO.Directory]::("Del" + "ete")` never match
# the first pattern, so a rule written only in terms of it would pass them.
#
# `::` followed by `]` is NOT a member access, and that exception is load
# bearing rather than defensive: the shipped netstat parse compares against the
# IPv6 wildcard `"[::]:0"`, so a bare `::` scan refuses real shipped code.
STATIC_MEMBER = re.compile(r"\[[\w\.\[\]]+\]::[A-Za-z_]\w*")
MEMBER_ACCESS = re.compile(r"::[ \t]*[A-Za-z_$(\"']")

# `.Name(` -- an instance method being CALLED. `.Name` without parentheses is a
# property read and invokes nothing, so it is not matched.
INSTANCE_METHOD_CALL = re.compile(r"\.([A-Za-z_]\w*)[ \t]*\(")


def _call_operator_offences(text):
    """Every `&` in non-comment text that invokes something off the allow-list.

    Returns (line number within the chunk, the spelling) per offence. The two
    `&` spellings that are NOT the call operator are skipped by shape rather
    than by a list: a redirection (`2>&1`, `1>&2`) has `>` immediately before
    it, and the pipeline-chain operator is a doubled `&`.
    """
    out = []
    for i, line in code_lines(text):
        for m in re.finditer(r"&", line):
            at = m.start()
            if at and line[at - 1] == ">":
                continue
            if line[at:at + 2] == "&&" or (at and line[at - 1] == "&"):
                continue
            hit = CALL_OPERATOR_TARGET.match(line[at:])
            if hit is None:
                # Nothing name-shaped follows: a parenthesised expression, a
                # script block, a sub-expression. This is the round-6 bypass.
                out.append((i, line[at:at + 60].strip()))
                continue
            target = hit.group("target")
            if "& " + target == DYNAMIC_INVOKE_ALLOWED:
                continue
            if target.strip("\"'").lower() in CALL_TARGET_ALLOWED:
                continue
            out.append((i, line[at:at + 60].strip()))
    return out


def _static_member_offences(text):
    """Every `::` member access in non-comment text that is not allow-listed."""
    out = []
    for i, line in code_lines(text):
        ok = [m.span() for m in STATIC_MEMBER.finditer(line)
              if m.group(0).lower() in STATIC_MEMBER_ALLOWED]
        for m in MEMBER_ACCESS.finditer(line):
            if not any(a <= m.start() and m.start() + 2 <= b for a, b in ok):
                out.append((i, line[max(0, m.start() - 40):m.start() + 30].strip()))
    return out


def _instance_method_offences(text):
    """Every `.Name(` call in non-comment text that is not allow-listed."""
    out = []
    for i, line in code_lines(text):
        for m in INSTANCE_METHOD_CALL.finditer(line):
            if m.group(1).lower() not in INSTANCE_METHOD_ALLOWED:
                out.append((i, m.group(1), line.strip()[:70]))
    return out

# Patterns that must not appear on any NON-COMMENT line of either channel, ever.
# These are not behavioural -- no case can drive a line that no longer exists --
# but they are the regression guard for the finding that motivated all of it: a
# kill selected by NAME cannot be made safe by any amount of tri-state around
# it, so the shape is banned outright rather than tested for.
FORBIDDEN_IN_SOURCE = [
    (r"\bStop-Process\b",
     "a kill selected by name or by an unverified pid; use the identity-checked "
     "killer (Stop-OwnedServerProcess / Stop-ProcessByImage)"),
    (r"\bschtasks\.exe\s+/(delete|end)\b",
     "an un-guarded schtasks /end or /delete; both must go through Invoke-Native "
     "inside an ownership check"),
    (r"2>&1\s*\|\s*Out-Null",
     "a native call whose stderr is turned into error records and then "
     "discarded; use Invoke-Native"),
]


def stubbed_names(text):
    """Every function this driver defines for itself -- what the guard checks."""
    return sorted(set(re.findall(r"^function ([A-Za-z][\w\-\.]*)", text, re.M)))


def assert_extract_is_inert(text, where):
    # CASE-FOLDED. PowerShell resolves command names case-insensitively --
    # `stop-process`, `Stop-Process` and `STOP-PROCESS` are one command -- so a
    # case-sensitive ban is a ban with a shift key for a bypass. Round-4 review
    # found every regex and every substring test in this file missing that, and
    # it was true of all of them.
    low = text.lower()
    hits = [c for c in FORBIDDEN_IN_EXTRACT if c.lower() in low]
    if hits:
        raise ValueError("%s now contains %s; extracting it would make this "
                         "harness change the machine" % (where, ", ".join(hits)))
    aliases = sorted(set(a.lower() for a in ALIAS_AT_COMMAND_POSITION.findall(text)))
    if aliases:
        raise ValueError("%s uses %s at command position; those are aliases for "
                         "cmdlets this harness refuses by name, and an alias reads "
                         "straight past a substring ban"
                         % (where, ", ".join(aliases)))
    # ROUND 6. An ALLOW-LIST per invocation construct; see the block above
    # CALL_TARGET_ALLOWED for why a deny-list cannot close this.
    for line_no, spelling in _call_operator_offences(text):
        raise ValueError(
            "%s invokes through the call operator at line %d of the extracted "
            "region (%r). The extracted text may use `&` only as %r, or on a "
            "literal name from %s -- every one of which the driver replaces "
            "with a function of its own. Anything else is a command name that "
            "is DATA: `& (\"Remove-\" + \"Item\")` contains no banned token, "
            "resolves at run time, and deletes %%LOCALAPPDATA%%\\wenlan"
            % (where, line_no, spelling, DYNAMIC_INVOKE_ALLOWED,
               sorted(CALL_TARGET_ALLOWED)))
    for line_no, spelling in _static_member_offences(text):
        raise ValueError(
            "%s reaches a .NET member at line %d of the extracted region (%r), "
            "and it is not one of the %d pure members this harness has measured "
            "in the shipped text (%s). A static call names no command, so no "
            "stub and no command ban can see it: "
            "[System.IO.Directory]::Delete($p, $true) is a complete deletion of "
            "the developer's data directory. If the shipped code now needs a new "
            "member, add it to STATIC_MEMBER_ALLOWED and say why it is inert"
            % (where, line_no, spelling, len(STATIC_MEMBER_ALLOWED),
               ", ".join(sorted(STATIC_MEMBER_ALLOWED))))
    for line_no, method, spelling in _instance_method_offences(text):
        raise ValueError(
            "%s calls the instance method .%s() at line %d of the extracted "
            "region (%r), which is not among the %d inert methods measured in "
            "the shipped text (%s). `(Get-Item $p).Delete($true)` and "
            "`$sb.Invoke()` are both invocations that name no command. If this "
            "one is inert, add it to INSTANCE_METHOD_ALLOWED"
            % (where, method, line_no, spelling, len(INSTANCE_METHOD_ALLOWED),
               ", ".join(sorted(INSTANCE_METHOD_ALLOWED))))
    for m in re.finditer(r"Invoke-Native\s+(\S+)", text):
        arg = m.group(1)
        if arg == "{":          # the function's own definition line
            continue
        if not (arg.startswith('"') and arg.endswith('"')
                and arg.strip('"').lower() in {n.lower() for n in NATIVE_ALLOWED}):
            raise ValueError("%s calls Invoke-Native with %s; it must be a quoted "
                             "literal from %s, all of which are stubbed"
                             % (where, arg, sorted(NATIVE_ALLOWED)))


def assert_every_reachable_command_is_accounted_for(chunks, driver_defines):
    """Nothing an extracted region can name may be able to reach this machine.

    Over-approximating on purpose: every Verb-Noun token and every *.exe token
    ANYWHERE in the text, string contents included. A false positive costs one
    line in INERT_COMMANDS; a false negative is a driver that reaches the
    developer's real scheduler, listener table or process list while reporting
    on a fixture.
    """
    # WHAT THIS IS, stated so it is not mistaken for more: a case-insensitive
    # LEXICAL scan of the extracted text. It is not a PowerShell parse and not a
    # command-resolution analysis -- it cannot see a name built at run time, a
    # name arriving through a parameter, or a call made by a .NET method.
    #
    # ROUND 6 (Codex Sol) refuted what this comment used to say next. It claimed
    # those three were "covered by the bans above", and they were not: the ban
    # recognised `& $var` only, so `& ("Remove-" + "Item") -LiteralPath
    # "$env:LOCALAPPDATA\wenlan" -Recurse -Force` and
    # `[System.IO.Directory]::Delete(...)` passed every check in this file. The
    # claim is now true because the bans changed, not because the sentence did:
    # assert_extract_is_inert holds three ALLOW-LISTS -- the call operator, the
    # `::` member access and the `.X()` instance call -- each derived from the
    # shipped text and refusing everything else, so a name built at run time has
    # no construct left to be invoked through. What remains uncovered is stated
    # where those lists are defined, not here.
    #
    # Case-folded because PowerShell is: `stop-process` and `Stop-Process` are
    # the same command, and the earlier `[A-Z][a-z]...` pattern saw only one.
    lowered_ok = {c.lower() for c in INERT_COMMANDS} | {d.lower() for d in driver_defines}
    # Two passes, because one regex cannot be both broad enough to catch a
    # lowercase spelling and narrow enough to ignore ordinary hyphenated prose.
    #   PASS 1, conventional spelling: Verb-Noun with both parts capitalised.
    #       This is what a cmdlet call actually looks like, and it does not match
    #       `tri-state`, `well-formed` or `wenlan-server`.
    #   PASS 2, any casing, but only for names already KNOWN to matter. This is
    #       the case-insensitivity fix: `stop-process` is caught here even though
    #       pass 1 cannot see it, without dragging prose in with it.
    known_any_case = {c.lower() for c in list(MUST_BE_STUBBED) + FORBIDDEN_IN_EXTRACT}
    unaccounted = {}
    for where, text in chunks:
        toks = (set(re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]*)*-[A-Z][A-Za-z0-9]*\b", text))
                | set(re.findall(r"\b[A-Za-z0-9_]+\.exe\b", text)))
        for m in re.finditer(r"\b[A-Za-z]+-[A-Za-z][A-Za-z0-9]*\b", text):
            if m.group(0).lower() in known_any_case:
                toks.add(m.group(0))
        for t in sorted(t for t in toks if t.lower() not in lowered_ok):
            unaccounted.setdefault(t, where)
    if unaccounted:
        raise ValueError(
            "these names appear in extracted text but are neither stubbed, nor "
            "defined by the driver, nor known-inert: "
            + ", ".join("%s (in %s)" % (t, w) for t, w in sorted(unaccounted.items())))


def code_lines(text):
    """Lines that are not whole-line comments. Crude on purpose: every hit this
    is used for lives on a line of its own in both channels, and a cleverer
    stripper that tried to find `#` inside strings would be the thing that
    silently stopped matching."""
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        yield i, line


# --------------------------------------------------------------------------
# THE GUARDS, CHECKED AGAINST THEMSELVES.
#
# Everything above is a refusal, and a refusal is a claim that something WOULD
# be caught. That claim is worth exactly as much as the last time the refusal
# was seen to fire. A guard nothing has ever tripped is indistinguishable from
# a guard that cannot trip -- which is this file's own thesis, turned on this
# file: the pre-check printing `ok` for every function is the same green
# whether the bans work or whether every regex silently stopped matching.
#
# So each guard is fed something it MUST refuse, and something it must NOT (one
# that refuses everything measures nothing either), and the run stops if any of
# them is silent. Round 4 is why: every ban here was case-SENSITIVE while
# PowerShell resolves command names case-INSENSITIVELY, so `stop-process`,
# `SCHTASKS.EXE /DELETE` and `Kill` all read straight past guards this file
# claimed were closed. Nothing detected that, because nothing ever asked a
# guard to fire.
# --------------------------------------------------------------------------
MUST_REFUSE_INERT = [
    ("banned cmdlet, as written",
     'Remove-Item -Recurse -Force "$env:LOCALAPPDATA\\wenlan"'),
    # ROUND 4, DEFECT 8. Each of the next four is the SAME command as the line
    # above as far as PowerShell is concerned.
    ("banned cmdlet, lowercase", "    remove-item -Recurse -Force C:\\x"),
    ("banned cmdlet, UPPERCASE", "    REMOVE-ITEM C:\\x"),
    ("banned cmdlet, MiXeD", "    ReMoVe-ItEm C:\\x"),
    ("alias ri", "    ri -Recurse C:\\x"),
    ("alias RI, uppercase", "    RI -Recurse C:\\x"),
    ("alias spps in a pipe", "Get-Process -Name x | spps -Force"),
    ("alias kill in a pipe", "Get-Process -Name x | kill -Force"),
    ("alias iex", "    iex $payload"),
    ("alias IEX, uppercase", "    IEX $payload"),
    ("the real killer, .Kill()", "$p.Kill()"),
    ("the real killer, lowercase", "$p.kill()"),
    ("GetProcessById", "[System.Diagnostics.Process]::GetProcessById(4)"),
    ("getprocessbyid, lowercase", "[System.Diagnostics.Process]::getprocessbyid(4)"),
    ("a command name held in a VARIABLE", "& $cmd /delete /tn WenlanServer /f"),
    # ROUND 6 (Codex Sol), RANKED #1. THE BYPASS, VERBATIM. Every one of these
    # reached %LOCALAPPDATA%\wenlan through a guard that recognised `& $var`
    # and nothing else. The first is the finding as filed: no banned token
    # appears in it anywhere, and PowerShell still resolves and runs Remove-Item.
    ("a command name BUILT AT RUN TIME",
     '& ("Remove-" + "Item") -LiteralPath "$env:LOCALAPPDATA\\wenlan" -Recurse -Force'),
    ("a command name built at run time, lowercase",
     '& ("remove-" + "item") -LiteralPath C:\\x -Recurse -Force'),
    ("a command name in a QUOTED literal off the allow-list",
     '& "taskkill.exe" /F /IM wenlan-server.exe'),
    # No alias and no banned token inside either of these, on purpose: the
    # refusal has to come from the call-operator rule and from nothing else,
    # or the fixture proves the alias scan works instead.
    ("a script block invoked with the call operator", "& { $env:PATH }"),
    ("a sub-expression invoked with the call operator",
     "& (Get-Command $verb) -Path C:\\x"),
    # The second bypass: a .NET static call names no command at all.
    ("a .NET static method that deletes a tree",
     '[System.IO.Directory]::Delete("$env:LOCALAPPDATA\\wenlan", $true)'),
    ("a .NET static method, lowercase", "[system.io.file]::delete($p)"),
    ("a .NET static method reached through a VARIABLE type", "$t::Delete($p)"),
    ("a .NET static member built from a string", '[System.IO.Directory]::("Del" + "ete")'),
    # The third: an instance method on an object the extracted text already has.
    ("a .NET instance method that deletes a tree", "(Get-Item $p).Delete($true)"),
    ("a script block invoked through .Invoke()", "$sb.Invoke()"),
    ("Invoke-Native off the allow-list", 'Invoke-Native "taskkill.exe" @("/F")'),
    ("Invoke-Native, UPPERCASE, off the allow-list", 'Invoke-Native "TASKKILL.EXE" @("/F")'),
    ("Invoke-Native with a non-literal", 'Invoke-Native $tool @("/query")'),
]

# These go through the reachability scan instead, which answers a different
# question: not "is this text dangerous" but "could this text name something
# the driver does not replace". Driver-defines is deliberately only the shipped
# functions -- a driver that stubbed nothing.
MUST_REFUSE_REACHABLE = [
    ("a cmdlet nothing stubs", "$x = Get-WmiObject -Class Win32_Service"),
    ("stop-process, lowercase, reaching the machine",
     "gps wenlan-server | stop-process -Force"),
    ("an .exe nothing stubs", "& taskkill.exe /F /IM wenlan-server.exe"),
    # ROUND 6. Add-Type came off INERT_COMMANDS, and this is the fixture that
    # says so: it compiles and loads arbitrary C#, which is a route to the
    # machine that names no cmdlet and no executable.
    ("Add-Type, which compiles and loads arbitrary C#",
     "    Add-Type -TypeDefinition $csharp"),
]

# A guard that fires on ordinary prose gets switched off within a week, so the
# other half of the measurement is what must still pass.
MUST_ACCEPT = [
    ("the shipped Invoke-Native body",
     '$out = @(& $File @Arguments 2>&1 | ForEach-Object { "$_" })'),
    ("an allow-listed native call", '$r = Invoke-Native "netstat.exe" @("-ano")'),
    # The allow-list is case-folded for the same reason everything else here is:
    # `& "NETSTAT.EXE"` resolves to the `function netstat.exe` stub, so refusing
    # this spelling would be a false alarm about a call that IS stubbed. This is
    # the case that makes that fold a measurement -- an UPPERCASE name that is
    # NOT on the list is refused either way, so it isolates nothing.
    ("an allow-listed native call in a different case",
     '$r = Invoke-Native "NETSTAT.EXE" @("-ano")'),
    ("prose containing the letters ri / sc / mi / start",
     'Write-Host "the description is scripted, mismatched, arbitrary, started"'),
    ("Start-Sleep, which begins with the alias `start`", "    Start-Sleep -Milliseconds 500"),
    # ROUND 6. The other half of each new allow-list, and every one of these is
    # a line the SHIPPED channels actually contain -- an allow-list that
    # refused them would stop the harness from running at all, which is the
    # failure mode that gets a guard deleted rather than fixed.
    ("the shipped CLI call, a bare name the driver stubs",
     "        & wenlan.exe background on"),
    ("the IPv6 wildcard in the shipped netstat parse, whose `::` is not a "
     "member access at all",
     '        if ($f[2] -ne "0.0.0.0:0" -and $f[2] -ne "[::]:0") { continue }'),
    ("the allow-listed .NET members the shipped probes use",
     '    if (-not [int]::TryParse($raw, [ref]$n)) { return $null }'),
    ("an allow-listed comparer passed to a constructor",
     "    $set = New-Object System.Collections.Generic.HashSet[string] "
     "([System.StringComparer]::OrdinalIgnoreCase)"),
    ("the allow-listed instance methods the shipped probes call",
     '    $null = $names.Add($row.Substring(1, $row.IndexOf(\'","\') - 1).Trim())'),
    ("a COMMENT naming a .NET method, which invokes nothing",
     "    # the array subexpression THROWS System.ArgumentException (\"Argument "
     "types do not match\") here"),
]

# Spellings of a banned SOURCE shape, re-injected into a copy of the shipped
# channel. FORBIDDEN_IN_SOURCE must catch all of them.
MUST_CATCH_IN_SOURCE = [
    ("Stop-Process, as written", "Get-Process -Name wenlan-server | Stop-Process -Force"),
    ("stop-process, lowercase", "get-process -Name wenlan-server | stop-process -Force"),
    ("STOP-PROCESS, uppercase", "GET-PROCESS -Name wenlan-server | STOP-PROCESS -Force"),
    ("SCHTASKS.EXE /DELETE", "SCHTASKS.EXE /DELETE /tn WenlanServer /f"),
    ("schtasks.exe /end", "schtasks.exe /end /tn WenlanServer"),
    ("2>&1 | out-null, lowercase", "& schtasks.exe /end /tn X 2>&1 | out-null"),
]

SELF_CHECK_ANCHOR = "    $ours = @($inv.Processes | Where-Object {"


def source_ban_hits(text):
    """Every (pattern, why, [line numbers]) where a banned shape appears in code.

    The pre-check runs this and the self-check below makes it fire, on purpose:
    a self-check that re-implemented the scan would be checking its own copy,
    and the copy would stay green on the day the real one stopped matching.

    re.I because PowerShell resolves command names case-insensitively --
    `stop-process` and `SCHTASKS.EXE /DELETE` are the same commands the banned
    spellings name, and a case-sensitive scan sees neither.
    """
    out = []
    for pattern, why in FORBIDDEN_IN_SOURCE:
        ns = [n for n, line in code_lines(text) if re.search(pattern, line, re.I)]
        if ns:
            out.append((pattern, why, ns))
    return out


# Each entry reverts ONE case-insensitivity fix in a COPY of THIS file and
# requires self_check to go red. Same rule as every control below, turned on the
# self-check: a check that cannot fail is not evidence that anything works, and
# the self-check is the newest and least-tested thing here.
#
# Cheap enough to run every time -- five module loads, no subprocesses -- which
# is the point. The proof that these guards can fire was a scratchpad script
# once, run once, and a scratchpad script run once is precisely the "green that
# cannot go red" this whole file exists to refuse.
# EVERY ANCHOR HERE IS SPLIT ACROSS TWO ADJACENT STRING LITERALS, and that is
# not a style choice. A mutation table that quotes the code it mutates contains
# a second copy of that code -- itself -- so a single-literal anchor matches
# twice, `count(old) != 1` fires, and the mutation silently reverts nothing.
# Two of these were written the obvious way first and did exactly that: the
# phase reported STALE for both, which is the failure mode working, but the
# fix is to make the table's own text differ from the code's. Adjacent literals
# concatenate to the same value while spelling it differently on disk.
SELF_CHECK_MUTATIONS = [
    ("the alias regex loses re.I",
     "    re.M" " | re.I)",
     "    re.M)"),
    ("the extract ban stops case-folding",
     "    low = text.lower()\n"
     "    hits = [c for c in FORBIDDEN_IN_EXTRACT if c.lower() in low]",
     "    hits = [c for c in FORBIDDEN_IN_EXTRACT if c in text]"),
    ("the reachability scan drops its known-any-case pass",
     '        for m in re.finditer(r"\\b[A-Za-z]+-[A-Za-z][A-Za-z0-9]*\\b", text):\n'
     "            if m.group(0).lower() in known_any_case:\n"
     "                toks.add(m.group(0))",
     "        pass"),
    ("the source ban loses re.I",
     "        ns = [n for n, line in code_lines(text)" " if re.search(pattern, line, re.I)]",
     "        ns = [n for n, line in code_lines(text) if re.search(pattern, line)]"),
    ("the Invoke-Native allow-list compares case-sensitively",
     "                and arg.strip('\"').lower() in {n.lower() for n in NATIVE_ALLOWED}):",
     "                and arg.strip('\"') in NATIVE_ALLOWED):"),
    # ROUND 6. The three invocation allow-lists, each deleted in turn. These
    # are the rules that stand between an extracted region and
    # %LOCALAPPDATA%\wenlan, so each one has to be watched refusing and watched
    # NOT refusing once its rule is gone -- otherwise the fixtures above are
    # only evidence that SOME rule fired.
    ("the call-operator allow-list is dropped",
     "    for line_no, spelling in _call_operator_off" "ences(text):",
     "    for line_no, spelling in []:"),
    ("the .NET static-member allow-list is dropped",
     "    for line_no, spelling in _static_member_off" "ences(text):",
     "    for line_no, spelling in []:"),
    ("the .NET instance-method allow-list is dropped",
     "    for line_no, method, spelling in _instance_method_off" "ences(text):",
     "    for line_no, method, spelling in []:"),
    # ROUND 6 (shell/release lane). The two halves of the replica-vs-lib.ps1
    # comparison, each deleted in turn. Both are silent on a healthy tree by
    # construction -- that is what "no drift" looks like -- and a guard that is
    # silent every run and has never been watched refusing is the one shape this
    # file will not accept anywhere else, so it does not accept it here either.
    ("the Check parameter-set comparison is dropped",
     "    for missing in sorted(set(shipped) - set(rep" "lica)):",
     "    for missing in []:"),
    ("the Check rule-anchor comparison is dropped",
     "    for what, anchor in SHIPPED_CHECK_" "RULES:",
     "    for what, anchor in []:"),
    ("Add-Type goes back on the inert list",
     '    "Join-Path", "Split-Path", "Test-Path", "Select-' 'String",',
     '    "Join-Path", "Split-Path", "Test-Path", "Add-Type", "Select-String",'),
]


def self_check_can_fail(work, zip_src, lib_src):
    """Make the SELF-CHECK fail. Returns the number of mutations it missed."""
    print("self-check-can-fail: reverting each case-fold in a copy of this file")
    me = Path(__file__).read_text(encoding="utf-8")
    bad = 0
    for i, (label, old, new) in enumerate(SELF_CHECK_MUTATIONS):
        if me.count(old) != 1:
            print("  STALE  %s: anchor matched %d times, so nothing was reverted"
                  % (label, me.count(old)))
            bad += 1
            continue
        p = work / ("selfcheck_mutant_%d.py" % i)
        p.write_bytes(me.replace(old, new, 1).encode("utf-8"))
        spec = importlib.util.spec_from_file_location("nc_selfcheck_mutant_%d" % i, str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            missed = mod.self_check(zip_src, lib_src)
        if missed:
            first = next((l.strip() for l in buf.getvalue().splitlines()
                          if "SILENT" in l or "OVEREAGER" in l), "")
            print("  ok     %s -> self-check went red (%d): %s" % (label, missed, first))
        else:
            print("  SILENT %s -> self-check stayed GREEN; it does not measure that fix"
                  % label)
            bad += 1

    # The EOL guard, made to fire. On a healthy tree crlf_offenders is silent
    # every single run, which is the one property that makes a guard impossible
    # to trust: nothing distinguishes "no CRLF here" from "no longer looking".
    # So it is shown a fully-CRLF copy of the shipped channel and required to
    # see it -- and the SAME copy is compared through read_text, which must call
    # it identical to the LF original. That second line is the point: it is the
    # blindness the guard exists to route around, demonstrated rather than
    # asserted, on the same bytes, in the same run.
    fixture = work / "crlf-fixture.ps1"
    fixture.write_bytes(zip_src.encode("utf-8").replace(b"\n", b"\r\n"))
    hits = crlf_offenders((("crlf-fixture.ps1", fixture),))
    if hits:
        print("  ok     the EOL guard sees a fully-CRLF copy: %s" % (hits,))
    else:
        print("  SILENT the EOL guard did not see a fully-CRLF file; it cannot fire")
        bad += 1
    if fixture.read_text(encoding="utf-8") == zip_src:
        print("  ok     and read_text calls that same copy identical to the LF "
              "original, which is why the guard reads bytes")
    else:
        print("  STALE  read_text distinguished CRLF from LF here, so the stated "
              "reason for reading bytes does not hold on this host")
        bad += 1

    # parse_ok's third value, made to fire. Its two failure verdicts have to be
    # told apart by something other than "not zero", and the way to show that is
    # to hand it a checker that answers the way a broken host does. Each probe
    # gets its own directory because parse_ok reads `parsecheck.ps1` from the
    # one it is given.
    #
    # ROUND 5 (Codex Sol), FINDING 2 added the fifth probe. Four probes
    # demonstrated the branch TABLE while the syntax criterion was still
    # "exit 1 and any output at all", so nothing in this file ever showed that
    # criterion refusing anything: a checker that printed an infrastructure
    # complaint and exited 1 was scored `syntax`, and every control in the run
    # would have been failed as "the mutant does not parse". The fifth probe is
    # that checker. It is here so the criterion is demonstrated firing rather
    # than asserted in a docstring.
    probes = (
        ("a checker that never ran", "exit 9", "unmeasured"),
        ("a checker that failed silently", "exit 1", "unmeasured"),
        ("a checker that failed NOISILY without parsing anything",
         'Write-Host "PowerShell host initialization failed"' + chr(10) + "exit 1",
         "unmeasured"),
        ("a checker reporting a real syntax error",
         'Write-Host "line 1: unexpected token"' + chr(10) + "exit 1", "syntax"),
        ("a checker reporting a clean parse", "exit 0", "parses"),
    )
    for i, (label, body, want) in enumerate(probes):
        d = work / ("parseprobe_%d" % i)
        d.mkdir(exist_ok=True)
        (d / "parsecheck.ps1").write_bytes(
            ("param([string]$Target)" + chr(10) + body + chr(10)).encode("utf-8"))
        got, detail = parse_ok(d, "$x = 1" + chr(10), "probe")
        if got == want:
            print("  ok     parse_ok reads %s as %r" % (label, got))
        else:
            print("  SILENT parse_ok read %s as %r, wanted %r (%s); an unmeasured "
                  "parse would be scored as a defect in the mutant"
                  % (label, got, want, detail.strip()[:120]))
            bad += 1

    # THE DRIVER-OUTCOME BRANCHES, made to fire, with REAL drivers.
    #
    # ROUND 5 follow-up. `classify_driver_output` is where a dead driver is told
    # apart from a failed case, and the split now turns on a marker that a
    # PowerShell script has to emit and this file has to parse -- a contract
    # across an escaping boundary (`SETUP-THREW` + a literal tab, written with a
    # backtick escape), which is exactly the kind of thing that silently stops
    # working. So these run through the same subprocess path the cases use,
    # rather than being handed synthetic strings: if the tab ever arrives as the
    # letter t, or Write-Host stops reaching stdout, probe 1 goes red here
    # instead of every setup throw quietly reverting to "unmeasured".
    #
    # ROUND 6 grew this table from four probes to eleven, because four were not
    # enough to distinguish the rule from its absence. B1: the marker branch was
    # taken on the PREFIX alone, so a driver that printed a marker AND answered
    # the row AND exited 0 was scored `fail` -- a control credited with a defect
    # it never caught. B2: once one matching row existed the classifier returned
    # ("pass", "") without reading `returncode`, so a driver that answered and
    # then died at exit 7 was clean. Neither shape was covered here, which is why
    # neither was noticed. Each probe below is one of those shapes.
    #
    # `witness` is the (control, case) declaration the real run looks up; None
    # is what every case gets against the shipped source and under every control
    # that has not claimed a setup death.
    tab = chr(9)
    nl = chr(10)
    good_marker = ("RemoteException: ERROR: The system cannot find the file "
                   "specified. -- raised at driver line 3: $stopped = Stop-Daemon")
    witness = "RemoteException: ERROR: The system cannot find the file specified."
    driver_probes = (
        ("a driver whose SETUP threw with the exception its control names",
         'Write-Host ("SETUP-THREW`t" + "' + good_marker + '")' + nl + "exit 3" + nl,
         witness, "fail"),
        # B1, the finding as filed: the same marker, with nobody claiming it.
        ("the same setup throw, with no control claiming a setup death",
         'Write-Host ("SETUP-THREW`t" + "' + good_marker + '")' + nl + "exit 3" + nl,
         None, "unmeasured"),
        # B1's failing input, verbatim: a marker, a green row, and exit 0.
        ("a driver that printed a SETUP-THREW marker AND answered the row",
         'Write-Host ("SETUP-THREW`t" + "RuntimeException: unrelated harness '
         'failure -- raised at driver line 3: $x = 1")' + nl +
         'Write-Host ("ROW`tprobe-row`tPASS`tthe detail this case wants")' + nl,
         witness, "unmeasured"),
        ("a setup throw whose driver did not exit 3",
         'Write-Host ("SETUP-THREW`t" + "' + good_marker + '")' + nl,
         witness, "unmeasured"),
        ("two setup-throw markers, which the wrapper cannot produce",
         'Write-Host ("SETUP-THREW`t" + "' + good_marker + '")' + nl +
         'Write-Host ("SETUP-THREW`t" + "' + good_marker + '")' + nl + "exit 3" + nl,
         witness, "unmeasured"),
        ("a setup throw naming a DIFFERENT exception from the one claimed",
         'Write-Host ("SETUP-THREW`t" + "RuntimeException: the task scheduler '
         'service is not available -- raised at driver line 3: $p = '
         'Get-TaskPresence")' + nl + "exit 3" + nl,
         witness, "unmeasured"),
        ("a setup throw that attributes itself to no source line",
         'Write-Host ("SETUP-THREW`t" + "RemoteException: ERROR: The system '
         'cannot find the file specified.")' + nl + "exit 3" + nl,
         witness, "unmeasured"),
        ("a driver that died saying nothing attributable",
         'throw "the host fell over"' + nl,
         None, "unmeasured"),
        ("a driver that answered the row it was asked",
         'Write-Host ("ROW`tprobe-row`tPASS`tthe detail this case wants")' + nl,
         None, "pass"),
        # B2: the answer is right and the process is not.
        ("a driver that answered the row and then died",
         'Write-Host ("ROW`tprobe-row`tPASS`tthe detail this case wants")' + nl +
         "exit 7" + nl,
         None, "unmeasured"),
        ("a driver that answered with the wrong status",
         'Write-Host ("ROW`tprobe-row`tFAIL`tthe detail this case wants")' + nl,
         None, "fail"),
    )
    for i, (label, body, probe_witness, want) in enumerate(driver_probes):
        try:
            out, rc = run_driver(work, "probe_%d" % i, body)
        except (subprocess.TimeoutExpired, OSError) as exc:
            print("  STALE  the driver probe %r could not be run at all (%r), so the "
                  "outcome branches were not exercised" % (label, exc))
            bad += 1
            continue
        got, why = classify_driver_output(
            "probe-row", out, rc, "PASS", "the detail this case wants", None,
            "see logs/probe", probe_witness)
        if got == want:
            print("  ok     a case is %r for %s" % (got, label))
        else:
            print("  SILENT a case is %r for %s, wanted %r (%s); the setup-throw, "
                  "answered-then-died and driver-died outcomes are not being told "
                  "apart" % (got, label, want, why[:160]))
            bad += 1
    # ...and the marker has to survive PowerShell, not just Python. Probe 1 above
    # would still pass if the tab were any other character, because it is the
    # PREFIX that selects the branch; this is the field split the reason text
    # depends on.
    marker = [ln for ln in run_driver(work, "probe_0", driver_probes[0][1])[0].splitlines()
              if ln.startswith("SETUP-THREW")]
    if marker and marker[0].startswith("SETUP-THREW" + tab) and "Stop-Daemon" in marker[0]:
        print("  ok     ...and the marker crosses PowerShell tab-separated, with the "
              "raising line intact")
    else:
        print("  SILENT the SETUP-THREW marker did not cross PowerShell in the shape "
              "this file parses (%r); the reason would be lost" % (marker[:1],))
        bad += 1

    # THE CONTROL SCORER'S THIRD BUCKET, made to fire. ROUND 5 (Codex Sol),
    # FINDING 1: `run_all` used to hand `main` two lists, and a case whose
    # DRIVER DIED landed in `failed`, where `want in mfailed` credited it as a
    # defect caught. That is a failed measurement scored as a negative one --
    # the exact thing every probe in this file is built to refuse -- living in
    # the scorer. Driving `score_control` with synthetic buckets shows the
    # refusal in the transcript instead of asserting it in a comment, and costs
    # no processes: the buckets are the whole input.
    scored = score_control("probe", ["c1"], mfailed=[], munmeasured=["c1"])
    if any(kind == "ok" for kind, _ in scored):
        print("  SILENT score_control credited an UNMEASURED case as caught; a "
              "mutant that killed the driver would read as a control that fired")
        bad += 1
    else:
        print("  ok     score_control refuses an unmeasured must_fail case "
              "(%d finding(s), none of them 'ok')" % len(scored))
    scored = score_control("probe", ["c1"], mfailed=["c1"], munmeasured=[])
    if [kind for kind, _ in scored] == ["ok"]:
        print("  ok     ...and still credits a case that genuinely went red")
    else:
        print("  OVEREAGER score_control refused a case that genuinely failed: %r"
              % (scored,))
        bad += 1
    scored = score_control("probe", ["c1"], mfailed=["c1"], munmeasured=["c2"])
    if any(kind == "FAIL" and "c2" in msg for kind, msg in scored):
        print("  ok     ...and an unmeasured case OUTSIDE must_fail is reported "
              "rather than dropped with the second bucket")
    else:
        print("  SILENT an unmeasured case outside must_fail vanished from the "
              "score; the 'not pinned to the fix' check would never see it")
        bad += 1

    # THE Check REPLICA, driven against the contract it claims to model, and
    # then made to get every one of those outcomes wrong. This is where the
    # falsifiability for the replica lives: a CONTROL mutates a CHANNEL script,
    # and the rule being defended here is in THIS file, so the mutation has to
    # be of this file -- exactly like SELF_CHECK_MUTATIONS above, but scored on
    # behaviour in a real powershell process rather than on a static guard.
    bad += replica_contract_probes(work)
    bad += replica_contract_can_fail(work)
    return bad


# THE LINE ENDINGS, IN BYTES, because nothing else in this file can see them.
# Both channels are LF in the index and every other .ps1 in the repo is LF, and
# there is no *.ps1 rule in .gitattributes to normalise them back. A text-mode
# writer -- Python's write_text, PowerShell's Set-Content/Out-File/`>` -- rewrites
# every line as CRLF on Windows, which turns a twenty-line edit into a
# whole-file diff.
#
# It has to be read_bytes. read_text applies universal newlines, so a CRLF file
# decodes to the same string as its LF original: every comparison in this file,
# INCLUDING the "did the source change during the run" check below, would
# compare equal across a rewrite of every byte in the file. That is this
# review's own defect class -- a measurement that cannot distinguish "unchanged"
# from "changed in the one way the reader is blind to" -- sitting in the
# verifier rather than in the thing verified.
def crlf_offenders(pairs=None):
    out = []
    for label, path in (pairs or (("windows-zip.ps1", ZIP), ("windows-nsis.ps1", NSIS),
                                  ("lib.ps1", LIB))):
        raw = path.read_bytes()
        n = raw.count(b"\r\n")
        if n:
            out.append((label, n, raw.count(b"\n")))
    return out


# --------------------------------------------------------------------------
# THE REPLICA IS A REPLICA, AND lib.ps1 IS THE AUTHORITY.
#
# ROUND 6, found by the shell/release lane. The `Check` in PREAMBLE is a model
# of `scripts/first-run/lib.ps1`'s `Check`, and for a whole review cycle it
# modelled a DIFFERENT contract: it declared `-ExpectFail` and never read it.
# Nothing here noticed, because nothing here compared the two. A reviewer did.
#
# That is the wrong place for the check to live. A replica silently diverging
# from the thing it replicates is exactly the failure this directory exists to
# catch -- a measurement that has stopped measuring while still reporting -- so
# it gets measured, on every run, in two ways that fail differently:
#
#   * the PARAMETER SET. `Check`'s parameters are its interface; a parameter
#     added upstream that the replica does not declare means shipped call sites
#     this harness would drive through a rule it has never seen, and a parameter
#     the replica declares that upstream does not have means cases written
#     against a rule the gauntlet does not run.
#   * the RULES. A parameter set can match while the body's rules do not, so
#     each construct the replica claims to model has an anchor that must be
#     present in BOTH bodies. Present upstream and missing here is drift; present
#     here and missing upstream is a model of something that no longer exists.
#
# Neither is a substitute for the cases; both are cheap, and both fail loudly at
# the pre-check rather than quietly at review time. Both are made to fire in
# `self_check` below, against fixtures, because a comparison that is silent on a
# healthy tree every single run is a comparison nobody has ever watched refuse.
#
# lib.ps1 is read here and NOWHERE mutated: no control touches it, and this file
# never writes it.
SHIPPED_CHECK_RULES = [
    ("the ExpectFail branch", "if ($ExpectFail) {"),
    ("the reach witness", "$script:CheckReached"),
    ("the single-pipeline AST fallback", "Test-SingleStatementBlock $Script"),
    ("the unmeasured third state, riding in the rc column", "Record-Row FAIL $Name 2"),
    ("the capture-phase fault split", '$faultedIn -eq "capture"'),
]

PS_PARAM_VAR = re.compile(r"\$([A-Za-z_]\w*)")


def ps_function_params(text, name):
    """The parameter NAMES declared by `function <name>`'s param(...) block.

    Attribute and type literals -- `[Parameter(Mandatory)]`, `[string]` -- carry
    no `$`, so the variables are the only sigil-bearing tokens inside the block.
    """
    body = extract_function(text, name)
    m = re.search(r"(?im)^\s*param\s*\(", body)
    if not m:
        raise ValueError("function %s declares no param() block" % name)
    j = body.index("(", m.start())
    depth = 0
    for k in range(j, len(body)):
        if body[k] == "(":
            depth += 1
        elif body[k] == ")":
            depth -= 1
            if depth == 0:
                return sorted(set(PS_PARAM_VAR.findall(body[j + 1:k])))
    raise ValueError("function %s: param( is unbalanced" % name)


def check_param_drift(shipped, replica):
    """Problems with the replica's parameter set, as sentences."""
    problems = []
    for missing in sorted(set(shipped) - set(replica)):
        problems.append(
            "lib.ps1's Check takes -%s and the replica does not declare it, so any "
            "shipped row using it is driven here through a rule this harness has "
            "never modelled" % missing)
    for extra in sorted(set(replica) - set(shipped)):
        problems.append(
            "the replica declares -%s and lib.ps1's Check does not, so a case could "
            "be written against a rule the gauntlet does not have" % extra)
    return problems


def check_rule_drift(shipped_body, replica_body):
    """Problems with the replica's RULES, as sentences."""
    problems = []
    for what, anchor in SHIPPED_CHECK_RULES:
        in_shipped = anchor in shipped_body
        in_replica = anchor in replica_body
        if in_shipped and not in_replica:
            problems.append("lib.ps1's Check still has %s (%r) and the replica no "
                            "longer models it" % (what, anchor))
        elif in_replica and not in_shipped:
            problems.append("the replica models %s (%r) and lib.ps1's Check no "
                            "longer has it" % (what, anchor))
    return problems


# Fixtures for the two comparisons: (label, shipped, replica, how many problems).
# The last of each pair is the converse -- agreement must produce NO problems, or
# the comparison would be a permanent red that gets deleted rather than fixed.
CHECK_PARAM_DRIFT_FIXTURES = [
    ("a parameter added upstream that the replica never modelled",
     ["Name", "Script", "Expect", "ExpectFail", "Because"],
     ["Name", "Script", "Expect", "ExpectFail"], 1),
    ("a parameter the replica declares and the gauntlet does not have",
     ["Name", "Script", "Expect"],
     ["Name", "Script", "Expect", "ExpectFail"], 1),
    ("drift in both directions at once",
     ["Name", "Script", "Expect", "Because"],
     ["Name", "Script", "Expect", "ExpectFail"], 2),
    ("the two agreeing",
     ["Name", "Script", "Expect", "ExpectFail"],
     ["Name", "Script", "Expect", "ExpectFail"], 0),
]

CHECK_RULE_DRIFT_FIXTURES = [
    ("the exact defect: the replica takes -ExpectFail and has no branch for it",
     "if ($ExpectFail) { throw }", "param([string]$ExpectFail)", 1),
    ("a rule the replica models that upstream has dropped",
     "nothing here", "if ($ExpectFail) { throw }", 1),
    ("both bodies carrying the rule",
     "if ($ExpectFail) { throw }", "if ($ExpectFail) { throw }", 0),
]

# --------------------------------------------------------------------------
# THE MODELLED OUTCOMES, DRIVEN.
#
# The comparisons above are static: they see that the rules are spelled in both
# files, not that the replica's copy BEHAVES like the original. These probes are
# the behavioural half. Each is one of lib.ps1's outcomes, driven through the
# replica in a real powershell process, and each asserts the rc as well as the
# status -- because rc is where the third state lives, and a run that could not
# tell rc=2 from rc=1 would have collapsed "unmeasured" into "failed" inside the
# very model built to keep them apart.
#
# The single shipped `-ExpectFail` call site is a single pipeline, so the reach
# witness is satisfied by the AST fallback and NO shipped row can exercise the
# rc=2 arm. That is why these blocks are hand-written rather than extracted:
# stated plainly, the unmeasured arm of `ExpectFail` is modelled and demonstrated
# here, and is not reachable from either channel's text today. If a multi-
# statement `-ExpectFail` block is ever shipped, it becomes reachable, and the
# case for it belongs in CASES.
REPLICA_CONTRACT_PROBES = (
    ("a single-pipeline -ExpectFail block that failed with the expected text",
     '$script:CliSearchMode = "refuses"\n'
     'Check -Name "probe" -ExpectFail "daemon stopped by" -Script { & wenlan.exe search x }\n',
     "PASS", 1, "daemon stopped by"),
    ("an -ExpectFail block that SUCCEEDED",
     'Check -Name "probe" -ExpectFail "daemon stopped by" -Script { & wenlan.exe search x }\n',
     "FAIL", 0, "expected nonzero exit with substring: daemon stopped by"),
    ("an -ExpectFail block that failed with the WRONG text",
     '$script:CliSearchMode = "fails-otherwise"\n'
     'Check -Name "probe" -ExpectFail "daemon stopped by" -Script { & wenlan.exe search x }\n',
     "FAIL", 1, "expected nonzero exit with substring: daemon stopped by"),
    ("a MULTI-statement -ExpectFail block with nothing witnessing the reach",
     '$script:CliSearchMode = "refuses"\n'
     'Check -Name "probe" -ExpectFail "daemon stopped by" -Script '
     '{ $q = "x"; & wenlan.exe search $q }\n',
     "FAIL", 2, "nothing witnesses that execution reached the construct under test"),
    ("...the same block, with Reached called immediately before the construct",
     '$script:CliSearchMode = "refuses"\n'
     'Check -Name "probe" -ExpectFail "daemon stopped by" -Script '
     '{ $q = "x"; Reached "the CLI call"; & wenlan.exe search $q }\n',
     "PASS", 1, "daemon stopped by"),
    ("a fault raised by the harness\'s OWN output capture, not by the block",
     "$ErrorActionPreference = 'Stop'\n"
     '$obj = New-Object PSObject\n'
     '$obj | Add-Member -MemberType ScriptMethod -Name ToString -Value '
     '{ throw "formatting exploded" } -Force\n'
     'Check -Name "probe" -Script { $obj }\n',
     "FAIL", 2, "the fault came from this harness's own output capture"),
    ("an -Expect block that produced the expected text",
     'Check -Name "probe" -Expect "Background registration kept" '
     '-Script { & wenlan.exe background off }\n',
     "PASS", 0, "Background registration kept"),
    ("an -Expect block that did not",
     'Check -Name "probe" -Expect "Background registration kept" '
     '-Script { & wenlan.exe background on }\n',
     "FAIL", 0, "expected substring: Background registration kept"),
    ("Reached called outside any Check block, where it witnesses nothing",
     'try { Reached "nothing"\n'
     '      Write-Host ("ROW`tprobe`tFAIL`tRC[0] Reached returned quietly") }\n'
     'catch { Write-Host ("ROW`tprobe`tPASS`tRC[0] " + $_.ToString()) }\n',
     "PASS", 0, "Reached was called outside a Check block"),
)


def replica_contract_probes(work):
    """Drive lib.ps1's Check outcomes through the replica. Returns misses."""
    print("replica-contract: the shipped Check's outcomes, driven through the replica")

    def one(indexed):
        i, (label, body, want_status, want_rc, want_detail) = indexed
        try:
            out, rc = run_driver(work, "replica_%d" % i, PREAMBLE + "\n" + body)
        except (subprocess.TimeoutExpired, OSError) as exc:
            return (label, None, "the driver could not be run at all (%r)" % (exc,))
        rows = [ln for ln in out.splitlines() if ln.startswith("ROW\t")]
        if len(rows) != 1:
            return (label, None, "the driver printed %d ROW lines, exit %d: %r"
                    % (len(rows), rc, out[-220:]))
        _, _row, got_status, got_detail = (rows[0].split("\t", 3) + [""])[:4]
        if got_status != want_status:
            return (label, None, "status %s, wanted %s -- %s"
                    % (got_status, want_status, got_detail[:200]))
        if ("RC[%d]" % want_rc) not in got_detail:
            return (label, None, "the rc is not %d, so the third state was not told "
                    "apart -- %s" % (want_rc, got_detail[:200]))
        if want_detail not in got_detail:
            return (label, None, "detail %r does not contain %r"
                    % (got_detail[:220], want_detail))
        return (label, "%s / RC[%d]" % (got_status, want_rc), None)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, enumerate(REPLICA_CONTRACT_PROBES)))
    bad = 0
    for label, got, why in results:
        if why is None:
            print("  ok     %s -> %s" % (label, got))
        else:
            print("  MISS   %s: %s" % (label, why))
            bad += 1
    return bad


# Each entry reverts ONE rule of the replica in a copy of this file. Every one
# of them must make at least one probe above go red; a rule whose removal
# changes nothing is a rule the probes do not measure, and the probes would then
# be decoration. This is the control the shell/release lane asked for, in the
# only place it can live: a CONTROL mutates a CHANNEL script, and the defect
# being defended against is in THIS file.
REPLICA_MUTATIONS = [
    ("the ExpectFail branch is dropped, which is the defect as filed",
     "\n    if ($ExpectFail) {\n",
     "\n    if ($false) {  # REVERTED: -ExpectFail declared and never read\n"),
    ("the reach witness gate is dropped",
     "        if ($rc -ne 0 -and -not $reached -and -not "
     "(Test-SingleStatementBlock $Script)) {",
     "        if ($false) {  # REVERTED: nothing has to witness the reach"),
    ("the single-pipeline AST fallback is dropped",
     "-not (Test-SingleStatementBlock $Scr" "ipt)) {",
     "$true) {  # REVERTED: a one-pipeline block no longer witnesses itself"),
    ("the unmeasured ExpectFail row loses its own rc",
     'Record-Row FAIL $Name 2 ("unmeasured: the bl' 'ock failed',
     'Record-Row FAIL $Name 1 ("unmeasured: the block failed'),
    ("the capture-phase fault split is dropped",
     '    if ($faultedIn -eq "capture"' ') {',
     "    if ($false) {  # REVERTED: a capture fault reads as the block failing"),
    ("Reached stops recording a witness",
     '    $script:CheckReached = if ($What) { $What } else ' '{ "yes" }',
     "    $null = $What  # REVERTED: Reached witnesses nothing"),
]


def replica_contract_can_fail(work):
    """Revert each modelled rule; the probes must go red. Returns misses."""
    print("replica-contract-can-fail: reverting each rule the replica models")
    me = Path(__file__).read_text(encoding="utf-8")
    bad = 0
    for i, (label, old, new) in enumerate(REPLICA_MUTATIONS):
        if me.count(old) != 1:
            print("  STALE  %s: anchor matched %d times, so nothing was reverted"
                  % (label, me.count(old)))
            bad += 1
            continue
        p = work / ("replica_mutant_%d.py" % i)
        p.write_bytes(me.replace(old, new, 1).encode("utf-8"))
        spec = importlib.util.spec_from_file_location("nc_replica_mutant_%d" % i, str(p))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except SyntaxError as exc:
            print("  STALE  %s: the mutated copy does not import (%r)" % (label, exc))
            bad += 1
            continue
        d = work / ("replica_mutant_%d_run" % i)
        d.mkdir(exist_ok=True)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                missed = mod.replica_contract_probes(d)
        except Exception as exc:                       # noqa: BLE001 -- reported
            print("  STALE  %s: the mutated replica could not be driven (%r)"
                  % (label, exc))
            bad += 1
            continue
        if missed:
            first = next((l.strip() for l in buf.getvalue().splitlines()
                          if l.strip().startswith("MISS")), "")
            print("  ok     %s -> %d probe(s) went red: %s" % (label, missed, first[:180]))
        else:
            print("  SILENT %s -> every probe stayed GREEN, so the replica does not "
                  "measure that rule" % label)
            bad += 1
    return bad


def self_check(zip_src, lib_src):
    """Make every refusal in this file fire. Returns the number that did not.

    Both sources are passed IN, never read from ZIP/LIB here. A mutated copy of
    this file lives in a temp directory and resolves `ROOT` from its own path,
    so a copy that read the shipped files itself would fail to find them and go
    red for a reason that has nothing to do with the mutation -- crediting the
    mutation with a defect nobody measured.
    """
    print("self-check: this harness's own guards, made to fire")
    bad = 0
    shipped_funcs = set(ZIP_FUNCS) | set(NSIS_FUNCS) | set(LIB_FUNCS)
    for label, text in MUST_REFUSE_INERT:
        try:
            assert_extract_is_inert(text, "self-check")
        except ValueError:
            continue
        print("  SILENT the inert-extract guard ACCEPTED %s (%r); it cannot fire"
              % (label, text))
        bad += 1
    for label, text in MUST_REFUSE_REACHABLE:
        try:
            assert_every_reachable_command_is_accounted_for(
                [("self-check", text)], shipped_funcs)
        except ValueError:
            continue
        print("  SILENT the reachability guard ACCEPTED %s (%r); it cannot fire"
              % (label, text))
        bad += 1
    for label, text in MUST_ACCEPT:
        try:
            assert_extract_is_inert(text, "self-check")
        except ValueError as e:
            print("  OVEREAGER the inert-extract guard REFUSED %s -- %s" % (label, e))
            bad += 1
    if SELF_CHECK_ANCHOR not in zip_src:
        print("  STALE  the self-check injection anchor is gone from windows-zip.ps1; "
              "the source bans were not exercised at all")
        bad += 1
    else:
        for label, injected in MUST_CATCH_IN_SOURCE:
            mutated = zip_src.replace(
                SELF_CHECK_ANCHOR, "    " + injected + "\n" + SELF_CHECK_ANCHOR, 1)
            if not source_ban_hits(mutated):
                print("  SILENT FORBIDDEN_IN_SOURCE did not catch re-injected %s" % label)
                bad += 1
        # ...and the converse: a COMMENT naming a banned cmdlet is prose about
        # the ban, not a call. If this fired, the ban would be unusable in the
        # comments that explain it, and it would be deleted rather than fixed.
        commented = zip_src.replace(
            "# Ported from kill_by_image_path",
            "# Stop-Process is what this replaced. Ported from kill_by_image_path", 1)
        if source_ban_hits(commented) != source_ban_hits(zip_src):
            print("  OVEREAGER a COMMENT naming Stop-Process was read as code")
            bad += 1

    # THE REPLICA'S CONTRACT, compared against the authority. See the block
    # above SHIPPED_CHECK_RULES for why this is measured rather than reviewed.
    # lib.ps1 is READ here; nothing in this file writes it, and no control
    # mutates it.
    try:
        lib_check = extract_function(lib_src, "Check")
        shipped_params = ps_function_params(lib_src, "Check")
        replica_check = extract_function(PREAMBLE, "Check")
        replica_params = ps_function_params(PREAMBLE, "Check")
    except ValueError as e:
        print("  STALE  lib.ps1's Check could not be read (%s), so the replica's "
              "contract was compared against nothing" % e)
        bad += 1
    else:
        for problem in (check_param_drift(shipped_params, replica_params)
                        + check_rule_drift(lib_check, replica_check)):
            print("  DRIFT  %s" % problem)
            bad += 1
        # ...and both comparisons, made to fire. A comparison whose only input is
        # a tree where it has nothing to say is a comparison nobody has watched
        # say anything.
        for label, shipped, replica, want in CHECK_PARAM_DRIFT_FIXTURES:
            got = len(check_param_drift(shipped, replica))
            if got != want:
                print("  %s the Check parameter-drift comparison reported %d "
                      "problem(s) for %s, wanted %d"
                      % ("SILENT" if got < want else "OVEREAGER", got, label, want))
                bad += 1
        for label, shipped, replica, want in CHECK_RULE_DRIFT_FIXTURES:
            got = len(check_rule_drift(shipped, replica))
            if got != want:
                print("  %s the Check rule-drift comparison reported %d "
                      "problem(s) for %s, wanted %d"
                      % ("SILENT" if got < want else "OVEREAGER", got, label, want))
                bad += 1
    if bad == 0:
        print("  ok     %d dangerous spellings refused, %d legitimate constructs "
              "accepted, %d re-injected source shapes caught, comments still prose"
              % (len(MUST_REFUSE_INERT) + len(MUST_REFUSE_REACHABLE),
                 len(MUST_ACCEPT), len(MUST_CATCH_IN_SOURCE)))
    return bad


def build_driver(texts, subject, stub_keys, row):
    subject_text = texts[subject]
    funcs = [(subject_text, f) for f in (ZIP_FUNCS if subject == "zip" else NSIS_FUNCS)]
    funcs += [(texts["lib"], f) for f in LIB_FUNCS]
    stub_text = "".join(STUBS[k] for k in stub_keys)
    parts = [PREAMBLE, stub_text]
    for src, fn in funcs:
        body = extract_function(src, fn)
        assert_extract_is_inert(body, "function %s" % fn)
        parts.append(body)
        parts.append("\n")
    names = sorted(set(stubbed_names(PREAMBLE + stub_text)) | set(MUST_BE_STUBBED))
    parts.append(GUARD.replace("$STUBBED$", ", ".join('"%s"' % n for n in names)))
    if (subject, row) in SETUP:
        setup = SETUP[(subject, row)][1]
        # A callable is handed the SUBJECT TEXT, so setup lines that mirror a
        # shipped decision are EXTRACTED from it rather than copied. A copy
        # would keep working when a control reverted the original -- the
        # control would then mutate the source, the driver would run the old
        # copy, and the case would stay green over a defect that was put back.
        #
        # Wrapped so that a setup that THROWS says so instead of vanishing: see
        # SETUP_OPEN/SETUP_CLOSE above.
        parts.append(SETUP_OPEN)
        parts.append(setup(subject_text) if callable(setup) else setup)
        parts.append(SETUP_CLOSE)
    block = extract_check_block(subject_text, row)
    assert_extract_is_inert(block, "Check block %s" % row)
    parts.append(block)
    parts.append("\n")
    return "\n".join(parts)


def case_log_name(case_name, label=None):
    """Where one case's transcript goes.

    ROUND 5 follow-up. Every control re-runs its channel's whole case list, and
    all of them used to write `case-<name>.log`, so the file a control failure
    pointed at held the LAST run of that case -- usually a later control's, and
    usually a green one. A failure message pointing at evidence that contradicts
    it is worse than no pointer: the reader trusts the file, sees a PASS, and
    concludes the message is wrong. Under a control the transcript is keyed by
    the control that produced it.
    """
    return ("%s--case-%s.log" % (label, case_name)) if label else ("case-%s.log" % case_name)


def run_driver(work, label, text, env=None):
    """Run one driver script; return (stdout+stderr, returncode).

    Split out of `run_case` so the self-check can push hand-written drivers
    through the SAME subprocess path the cases use. The marker contract below
    -- a tab-separated `SETUP-THREW` line reaching stdout intact -- is a
    property of PowerShell's output handling and of this file's escaping, not
    something either can be assumed to get right, so it is measured.
    """
    p = work / ("driver-%s.ps1" % label)
    # BYTES: write_text would rewrite every \n to \r\n on Windows, so the script
    # measured would not be the script written.
    p.write_bytes(text.encode("utf-8"))
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(p)],
        capture_output=True, text=True, timeout=300, env=env)
    return (proc.stdout or "") + (proc.stderr or ""), proc.returncode


SETUP_THROW_EXIT = 3          # what SETUP_CLOSE exits with, and nothing else does
SETUP_THROW_RAISER = " -- raised at driver line "   # SETUP_CLOSE's attribution


def _classify_setup_throw(row, threw, rows, returncode, setup_witness, log_hint):
    """A `SETUP-THREW` marker as ("fail" | "unmeasured", why).

    ROUND 6, B1 and RANKED #2. Six things have to hold before a marker is an
    attributed failure of the construct a control reverted, and the version
    this replaces checked none of them. Each `unmeasured` below is a marker
    that establishes only that SOMETHING outside `Check` declined.
    """
    detail = threw[0].split("\t", 1)[1]
    if setup_witness is None:
        return ("unmeasured",
                "the setup for row %s threw, and no control claims a setup death "
                "as its witness here. The marker says something outside Check "
                "declined; it does not say the reverted construct did, and "
                "SETUP_CLOSE wraps the whole transitive setup, so an unrelated "
                "regression produces the identical line: %s (%s)"
                % (row, detail[:300], log_hint))
    if len(threw) != 1:
        return ("unmeasured",
                "%d SETUP-THREW markers for row %s; the wrapper raises once and "
                "exits, so more than one means the transcript is not the one this "
                "driver produced: %s (%s)" % (len(threw), row, detail[:200], log_hint))
    if rows:
        # The B1 counterexample, exactly: a marker AND an answered row.
        return ("unmeasured",
                "row %s produced a SETUP-THREW marker AND %d ROW line(s). The "
                "driver claims both that the setup died before Check could run and "
                "that Check ran; the two cannot both be true, so neither is a "
                "measurement: %s (%s)" % (row, len(rows), detail[:200], log_hint))
    if returncode != SETUP_THROW_EXIT:
        return ("unmeasured",
                "row %s produced a SETUP-THREW marker and the driver exited %d, "
                "not %d. Only SETUP_CLOSE prints that marker and it exits %d on "
                "the same path, so the marker and the status contradict each "
                "other: %s (%s)"
                % (row, returncode, SETUP_THROW_EXIT, SETUP_THROW_EXIT,
                   detail[:200], log_hint))
    if SETUP_THROW_RAISER not in detail:
        return ("unmeasured",
                "row %s threw in setup with no raising line in the record. "
                "SETUP_CLOSE appends %r from InvocationInfo, so a record without "
                "it names no source and attributes the failure to nothing: %s (%s)"
                % (row, SETUP_THROW_RAISER, detail[:200], log_hint))
    if setup_witness not in detail:
        return ("unmeasured",
                "row %s threw in setup, but not with the exception this control's "
                "rule emits. Wanted %r in the record; got %s. A control whose pass "
                "condition is an exception must check WHICH exception, or any "
                "setup regression satisfies it (%s)"
                % (row, setup_witness, detail[:250], log_hint))
    return ("fail",
            "the setup that drives row %s threw before the Check block could run, "
            "with the exception this control's rule emits (%r), so the shipped "
            "statement outside Check is where this failed: %s (%s)"
            % (row, setup_witness, detail[:300], log_hint))


def classify_driver_output(row, out, returncode, want_status, want_detail,
                           want_absent, log_hint, setup_witness=None):
    """One driver's output as ("pass" | "fail" | "unmeasured", why).

    The three-way split, and where the two failure kinds part company:

      SETUP-THREW  -- the setup statements ran, one of them raised, and the
                      driver said which line and which exception before it
                      exited. That is a case that FAILED only where a control
                      declared that exception as its witness, and only when the
                      whole record holds together: see _classify_setup_throw.
                      Undeclared, contradicted or unattributed, it is a marker
                      about something, and something is not a measurement.
      no row at all -- the driver stopped without saying anything a reader
                      could attribute: powershell never started, a stub escaped
                      (exit 9), the host died. Nothing was measured, and that
                      is NOT a negative result about the code under test.
      a row, then a nonzero exit -- ROUND 6, B2. The driver answered and then
                      died. `run-all.sh` refuses exactly this shape one level
                      up, where a child's marker says failures=0 beside a
                      non-zero status, and calls it CONTRADICTORY; the
                      per-case classifier used to return ("pass", "") the
                      moment one matching row existed, without ever reading
                      `returncode`. MEASURED on this host, so that the rule
                      costs nothing legitimate: $LASTEXITCODE set by a stub, a
                      native command exiting 4, and a non-terminating
                      Write-Error all leave `powershell -File` at 0. The only
                      nonzero exits a driver has are the stub-escape guard's 9
                      and SETUP_CLOSE's 3, and neither can have printed a row.

    Keeping those apart is the whole point. Collapsing them in either direction
    reproduces this directory's subject inside its own scorer: one way a broken
    driver is credited as a caught defect, the other way a real defect is
    written off as an unmeasurable.
    """
    lines = out.splitlines()
    threw = [ln for ln in lines if ln.startswith("SETUP-THREW\t")]
    rows = [ln for ln in lines if ln.startswith("ROW\t")]
    if threw:
        return _classify_setup_throw(row, threw, rows, returncode, setup_witness,
                                     log_hint)
    if len(rows) != 1:
        # A driver that DIED also produces no row, and would otherwise read as a
        # case that failed -- which, inside a control, reads as a control that
        # fired. Name it instead, and keep the name all the way to the scorer.
        return ("unmeasured",
                "driver produced %d ROW lines and no setup diagnosis (died rather "
                "than answering), exit %d; %s" % (len(rows), returncode, log_hint))
    if returncode != 0:
        return ("unmeasured",
                "row %s was answered and the driver then exited %d. A process that "
                "printed its answer and died has not made a statement anything can "
                "be scored against -- believing the row over the status, or the "
                "status over the row, is a guess: %r (%s)"
                % (row, returncode, rows[0][:200], log_hint))
    _, got_row, got_status, got_detail = (rows[0].split("\t", 3) + [""])[:4]
    if got_row != row:
        return ("fail", "row %r, wanted %r" % (got_row, row))
    if got_status != want_status:
        return ("fail", "%s, wanted %s -- detail: %s" % (got_status, want_status, got_detail[:200]))
    if want_detail not in got_detail:
        return ("fail", "detail %r does not contain %r" % (got_detail[:220], want_detail))
    if want_absent is not None and want_absent in got_detail:
        return ("fail", "detail %r contains %r, which this case forbids"
                % (got_detail[:220], want_absent))
    return ("pass", "")


def run_case(work, texts, case, label=None):
    """Run one case. Returns ("pass" | "fail" | "unmeasured", why).

    ROUND 5 (Codex Sol), FINDING 1. The verdict used to be a BOOLEAN with the
    driver-died reason folded into the False arm, and every caller downstream
    read False as "the case failed" -- which, inside a control, reads as "the
    control fired". The three words are the fix: a driver that never printed a
    row measured nothing, and nothing is not a negative result.
    """
    name, subject, stub_keys, row, want_status, want_detail = case[:6]
    want_absent = case[6] if len(case) > 6 else None
    env = dict(os.environ)
    env.pop("GAUNTLET_HEALTH_TIMEOUT_SEC", None)
    env.update(CASE_ENV.get(name, {}))
    log_base = case_log_name(name, label)
    try:
        out, rc = run_driver(work, ("%s--%s" % (label, name)) if label else name,
                             build_driver(texts, subject, stub_keys, row), env)
    except (subprocess.TimeoutExpired, OSError) as exc:
        # The driver never returned, and there is no transcript to point at.
        return ("unmeasured", "the driver did not return: %r" % (exc,))
    # ROUND 6, RANKED #2. The setup-death witness belongs to the (control, case)
    # pair, not to the case: against the shipped source nothing may claim one at
    # all, and under a DIFFERENT control the same case must not inherit this
    # one's licence to be reddened by a generic marker.
    witness = SETUP_THROW_WITNESS.get((label, name)) if label is not None else None
    verdict, why = classify_driver_output(row, out, rc, want_status, want_detail,
                                          want_absent, "see logs/" + log_base,
                                          witness)
    # Every case of the shipped-source sweep keeps its transcript; under a
    # CONTROL only the ones a reader would go looking for are kept, because 58
    # controls times a channel's case list is thousands of files and the green
    # ones are the copies nobody opens.
    if label is None or verdict != "pass":
        (LOGS / log_base).write_text(out, encoding="utf-8")
    return (verdict, why)


def run_all(work, texts, subject_filter=None, label=None):
    """Returns three lists: passed, failed, unmeasured.

    ROUND 5 (Codex Sol), FINDING 1. This boundary used to be where the runtime
    tri-state died: `(passed if ok else failed).append(...)` folded "the driver
    died rather than answering" into the same bucket as "the case failed", and
    the control scorer below credits membership in that bucket as a defect
    CAUGHT. A mutant that broke the driver was therefore scored as a control
    that fired -- this directory's own defect class, in the scorer that is
    supposed to be looking for it. Three buckets out, three branches in every
    caller.
    """
    # A SET of channels, not one: a control on lib.ps1 can name cases in more
    # than one channel, because lib.ps1 is shared by all of them.
    todo = [c for c in CASES if not subject_filter or c[1] in subject_filter]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(lambda c: run_case(work, texts, c, label), todo))
    buckets = {"pass": [], "fail": [], "unmeasured": []}
    for case, (verdict, why) in zip(todo, results):
        buckets[verdict].append(case[0])
        if verdict == "fail":
            print("    (%s: %s)" % (case[0], why))
        elif verdict == "unmeasured":
            print("    (%s: COULD NOT MEASURE -- %s)" % (case[0], why))
    return buckets["pass"], buckets["fail"], buckets["unmeasured"]


def score_control(name, must_fail, mfailed, munmeasured):
    """A control's verdict as a list of (kind, message); kind is "ok" or "FAIL".

    Split out of `main` so the scoring can be DRIVEN with synthetic buckets in
    the self-check. Round 5's finding was a scorer that credited a dead driver
    as a caught defect, and a scorer only ever exercised by a real sweep is a
    scorer nobody has watched refuse.
    """
    out = []
    # UNMEASURED FIRST, and it is never an "ok". A case that did not run says
    # nothing about the reverted fix, whether or not the control was hoping to
    # see it go red.
    for got in munmeasured:
        if got in must_fail:
            out.append(("FAIL",
                        "%s: COULD NOT MEASURE %s -- the driver died rather than "
                        "answering, so nothing was measured about the reverted fix. "
                        "A case that did not run is not a control that fired; see "
                        "logs/%s" % (name, got, case_log_name(got, name))))
        else:
            out.append(("FAIL",
                        "%s: COULD NOT MEASURE %s, a case this control does not "
                        "claim to redden -- the mutant broke the harness rather "
                        "than the fix, and an unpinned case that did not run is "
                        "not an unpinned case that passed; see logs/%s"
                        % (name, got, case_log_name(got, name))))
    for want in must_fail:
        if want in mfailed:
            out.append(("ok", "caught:   %s" % want))
        elif want in munmeasured:
            continue  # already reported above, as what it actually is
        else:
            out.append(("FAIL", "%s: survived: %s -- the case does not defend this fix"
                        % (name, want)))
    # And ONLY those. A mutation that reddens every case of its subject has not
    # localised anything.
    for got in mfailed:
        if got not in must_fail:
            out.append(("FAIL",
                        "%s: case %r also failed; the control is not pinned to the fix"
                        % (name, got)))
    return out


# The shape PARSECHECK (defined just below) prints for EVERY syntax error it
# finds: `line <n>: <message>`. This is the parser's own record, and nothing
# else in that script writes to stdout, so its presence is the difference
# between "the parser diagnosed the file" and "the process exited 1".
PARSE_DIAGNOSIS = re.compile(r"(?m)^\s*line \d+:\s*\S")


def parse_ok(work, text, label):
    """The PowerShell equivalent of `bash -n`: a mutant that does not parse
    reverts nothing, and its red would read as a caught defect.

    Returns one of "parses" / "syntax" / "unmeasured", plus detail.

    The third value is the point. `returncode == 0, else it did not parse` reads
    a verdict off an exit status alone, so a powershell.exe that never got far
    enough to open the file -- process-table pressure, a transient CreateProcess
    failure, a timeout -- arrives as `syntax error` and every control in the run
    is scored as a failure. That is not hypothetical: a full sweep of this
    directory produced `CONTROL FAILURES: 113`, every line reading `the mutant
    does not parse, so it reverted nothing:` with an EMPTY reason, minutes
    before the identical revision ran standalone at `CONTROL FAILURES: 0`. The
    empty reason was the tell -- a real parse error prints its line and message.

    So a syntax verdict now needs BOTH the exit status the checker reserves for
    it AND the diagnosis it always prints alongside. Anything else is a
    measurement that did not happen, and says so.

    ROUND 5 (Codex Sol), FINDING 2. "The diagnosis it prints alongside" used to
    be spelled `out.strip()` -- any output at all. That is three-valued
    syntactically and binary semantically: a checker that printed "PowerShell
    host initialization failed" and exited 1 without ever looking at the file
    was called a syntax measurement. Nonempty output is not proof the PARSER
    diagnosed anything. PARSECHECK's contract is one `line <n>: <message>`
    record per syntax error, so that shape -- PARSE_DIAGNOSIS below -- is what
    a syntax verdict now requires. A noisy exit 1 without it is unmeasured.
    """
    p = work / ("parse-%s.ps1" % label)
    # BYTES: `write_text` on Windows rewrites every \n to \r\n, so the thing
    # parsed would not be the thing the controls then run.
    try:
        p.write_bytes(text.encode("utf-8"))
    except OSError as exc:
        # Round 5: this used to escape as an exception. A mutant whose text was
        # never written to disk was not a mutant that failed to parse.
        return "unmeasured", "the mutant could not be written for parsing: %r" % (exc,)
    checker = work / "parsecheck.ps1"
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(checker),
             "-Target", str(p)],
            capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return "unmeasured", "the parse checker did not return: %r" % (exc,)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return "parses", out
    if proc.returncode == 1 and PARSE_DIAGNOSIS.search(out):
        return "syntax", out
    if proc.returncode == 1:
        return "unmeasured", (
            "the parse checker exited 1 but printed no parser diagnosis. It prints "
            "'line <n>: <message>' for every syntax error it finds, so a bare or "
            "off-contract exit 1 -- a host that failed to start, a profile that "
            "threw, an execution-policy refusal -- is an exit status and not a "
            "verdict about the mutant: %r" % (out.strip()[:300] or "no output"))
    return "unmeasured", (
        "the parse checker exited %d with %s output; it reserves 0 for parses and "
        "1-with-a-diagnosis for a syntax error, so this is neither answer: %r"
        % (proc.returncode, "no" if not out.strip() else "unexpected", out.strip()[:300]))


PARSECHECK = r"""param([string]$Target)
$tokens = $null; $errors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile($Target, [ref]$tokens, [ref]$errors)
if ($errors -and $errors.Count -gt 0) {
    foreach ($e in $errors) { Write-Host ("line " + $e.Extent.StartLineNumber + ": " + $e.Message) }
    exit 1
}
exit 0
"""


def free_variables(block):
    """Variables a Check block reads without assigning them itself.

    A block that closes over a variable no driver defines is not a case that
    fails; it is a case that evaluates $null.State, falls into the
    could-not-measure arm, and passes for a reason that has nothing to do with
    the code under test. The pre-check treats that as staleness.
    """
    assigned = set(re.findall(r"\$(\w+)\s*=", block))
    assigned |= set(re.findall(r"for\s*\(\s*\$(\w+)", block))
    assigned |= set(re.findall(r"foreach\s*\(\s*\$(\w+)", block))
    builtin = {"_", "true", "false", "null", "LASTEXITCODE", "PSItem", "args"}
    used = set(re.findall(r"\$(\w+)", block))
    return sorted(used - assigned - builtin)


def main():
    global failures
    LOGS.mkdir(parents=True, exist_ok=True)
    zip_bytes_before = ZIP.read_bytes()
    nsis_bytes_before = NSIS.read_bytes()
    # lib.ps1 too, now that the shared helpers and the ownership acts are
    # there: it is extracted from and mutated exactly as the channels are, so
    # an edit landing mid-run would have the first controls scoring one
    # lib.ps1 and the rest scoring another.
    try:
        lib_bytes_before = LIB.read_bytes()
    except OSError:
        lib_bytes_before = None
    zip_before = ZIP.read_text(encoding="utf-8")
    nsis_before = NSIS.read_text(encoding="utf-8")
    # lib.ps1 is a driven subject: `Check`'s shipped contract is compared
    # against the replica in PREAMBLE, AND the web-exception classifiers, the
    # CIM witnesses and Get-DirPresence -- everything more than one channel
    # uses -- are extracted from it and mutated by controls.
    try:
        lib_before = LIB.read_text(encoding="utf-8")
    except OSError as e:
        print("  STALE  %s could not be read (%s); the replica's Check could not "
              "be compared against the contract it models" % (LIB, e))
        lib_before = ""
    texts = {"zip": zip_before, "nsis": nsis_before, "lib": lib_before}

    print("windows-probes-negative-controls")

    with tempfile.TemporaryDirectory(prefix="windows-probes-nc-") as tmp:
        work = Path(tmp)
        (work / "parsecheck.ps1").write_bytes(PARSECHECK.encode("utf-8"))

        # --- PRE-CHECK ---------------------------------------------------
        # Every anchor is validated against the REAL source before anything
        # runs. A stale anchor would mutate nothing and the control would
        # report a green suite as a caught defect -- a harness measuring
        # nothing while saying it measured. Hard error, not a warning.
        hard = self_check(zip_before, lib_before)
        hard += self_check_can_fail(work, zip_before, lib_before)
        for label, n, total in crlf_offenders():
            print("  CRLF   %s has %d CRLF line endings of %d; the index blob is LF, "
                  "so a text-mode writer rewrote the whole file and every diff of it "
                  "is now whole-file churn" % (label, n, total))
            hard += 1
        print("pre-check: anchors and extractions against the shipped source")
        for label, text in (("windows-zip.ps1", zip_before), ("windows-nsis.ps1", nsis_before),
                            ("lib.ps1", lib_before)):
            for pattern, why, ns in source_ban_hits(text):
                print("  BANNED %s matches %r at %s -- %s"
                      % (label, pattern, ", ".join("%s:%d" % (label, n) for n in ns), why))
                hard += 1
        reach_chunks = []
        for fn_list, key in ((ZIP_FUNCS, "zip"), (NSIS_FUNCS, "nsis"),
                             (LIB_FUNCS, "lib")):
            for fn in fn_list:
                try:
                    body = extract_function(texts[key], fn)
                except ValueError as e:
                    print("  STALE  function %s: %s" % (fn, e))
                    hard += 1
                    continue
                if not body.rstrip().endswith("}") or body.count("{") != body.count("}"):
                    print("  STALE  function %s extracted unbalanced" % fn)
                    hard += 1
                    continue
                try:
                    assert_extract_is_inert(body, "function %s" % fn)
                except ValueError as e:
                    print("  UNSAFE %s" % e)
                    hard += 1
                    continue
                reach_chunks.append(("function %s" % fn, body))
                print("  ok     function %s (%d chars)" % (fn, len(body)))
        # Which channel owns a row is DERIVED from the cases rather than listed
        # here: a hard-coded list is one more thing that goes stale silently
        # when a row is added, and it would send the extractor at the wrong file.
        #
        # A (channel, row) PAIR, not a row. Both channels record
        # `no-leftover-dirs` now -- about different trees, from different code
        # -- and the row-keyed version of this had to call that a collision and
        # refuse, which would have left one of the two rows undriven.
        subject_rows = []
        for case in CASES:
            name, subject, _stub, row = case[:4]
            if subject not in texts:
                print("  STALE  case %s names unknown channel %r" % (name, subject))
                hard += 1
                continue
            if (subject, row) not in subject_rows:
                subject_rows.append((subject, row))
        for case in CASES:
            for k in case[2]:
                if k not in STUBS:
                    print("  STALE  case %s names unknown stub %r" % (case[0], k))
                    hard += 1
        known_vars = set(PREAMBLE_VARS) | {v for d, _s in SETUP.values()
                                           for v in _setup_vars(d)}
        for subject, row in sorted(subject_rows):
            label = "%s row %s" % (subject, row)
            try:
                body = extract_check_block(texts[subject], row)
            except ValueError as e:
                print("  STALE  Check %s: %s" % (label, e))
                hard += 1
                continue
            if body.count("{") != body.count("}"):
                print("  STALE  Check %s extracted unbalanced" % label)
                hard += 1
                continue
            try:
                assert_extract_is_inert(body, "Check block %s" % label)
            except ValueError as e:
                print("  UNSAFE %s" % e)
                hard += 1
                continue
            loose = [v for v in free_variables(body) if v not in known_vars]
            if loose:
                print("  STALE  Check %s closes over %s, which no driver defines "
                      "(it would evaluate to $null and the row would pass for the "
                      "wrong reason)" % (label, ", ".join("$" + v for v in loose)))
                hard += 1
                continue
            unused = [v for v in _setup_vars(SETUP.get((subject, row), ("",))[0])
                      if v and v not in free_variables(body)]
            if unused:
                print("  STALE  Check %s no longer uses %s, so its setup line "
                      "drives nothing"
                      % (label, ", ".join("$" + v for v in unused)))
                hard += 1
                continue
            reach_chunks.append(("Check %s" % label, body))
            print("  ok     Check %s (%d chars)" % (label, len(body)))
        # The stub-escape guard's coverage, derived rather than trusted.
        try:
            assert_every_reachable_command_is_accounted_for(
                reach_chunks,
                set(MUST_BE_STUBBED) | set(ZIP_FUNCS) | set(NSIS_FUNCS)
                | set(LIB_FUNCS))
            print("  ok     every command reachable from extracted text is "
                  "stubbed, driver-defined or inert")
        except ValueError as e:
            print("  UNSAFE %s" % e)
            hard += 1
        for name, _why, subject, old, _new, _mf in CONTROLS:
            n = texts[subject].count(old)
            if n != 1:
                print("  STALE  control %s anchor matched %d times, wanted 1" % (name, n))
                hard += 1
            else:
                print("  ok     control %s anchor" % name)
        known = {c[0] for c in CASES}
        for name, _why, _s, _o, _n, must_fail in CONTROLS:
            for mf in must_fail:
                if mf not in known:
                    print("  STALE  control %s names unknown case %r" % (name, mf))
                    hard += 1
        for cname in CASE_ENV:
            if cname not in known:
                print("  STALE  CASE_ENV names unknown case %r" % cname)
                hard += 1
        # ROUND 6. A setup-death witness that names a control or a case that no
        # longer exists, or a case the control does not even claim to redden,
        # would sit here licensing nothing and be indistinguishable from one
        # doing its job. The must_fail membership test is the one that matters:
        # a witness on a case OUTSIDE must_fail would let a mutation redden a
        # case the control never claimed, through the exception path.
        control_must_fail = {c[0]: set(c[5]) for c in CONTROLS}
        for (cn, casename), text in sorted(SETUP_THROW_WITNESS.items()):
            if cn not in control_must_fail:
                print("  STALE  SETUP_THROW_WITNESS names unknown control %r" % cn)
                hard += 1
            elif casename not in known:
                print("  STALE  SETUP_THROW_WITNESS names unknown case %r" % casename)
                hard += 1
            elif casename not in control_must_fail[cn]:
                print("  STALE  SETUP_THROW_WITNESS lets %s redden %s through a setup "
                      "throw, but that control does not list the case in must_fail"
                      % (cn, casename))
                hard += 1
            elif not text.strip():
                print("  STALE  SETUP_THROW_WITNESS for (%s, %s) is empty, which "
                      "matches every diagnosis" % (cn, casename))
                hard += 1
            else:
                print("  ok     setup-throw witness for %s: %r" % (casename, text))
        for key in SETUP:
            if key not in subject_rows:
                print("  STALE  SETUP names %r, which no case drives" % (key,))
                hard += 1
        if hard:
            print("FATAL: %d silent guard(s), stale anchor(s), banned pattern(s) "
                  "or extraction(s); the controls below would measure nothing" % hard)
            return 1

        # --- CASES against the shipped source ----------------------------
        print("cases against the shipped source:")
        passed, failed, unmeasured = run_all(work, texts)
        for n in passed:
            print("  ok   %s" % n)
        for n in failed:
            print("  FAIL %s" % n)
            failures += 1
        for n in unmeasured:
            print("  COULD NOT MEASURE %s -- the driver died rather than answering, "
                  "so this case has no verdict either way" % n)
            failures += 1
        if failed or unmeasured:
            print("  (the shipped probes do not pass their own cases; the controls "
                  "below would measure nothing)")

        # --- CONTROLS ----------------------------------------------------
        print("controls:")
        mutants_applied = 0
        red_outcomes_required = 0
        red_outcomes_observed = 0
        for name, why, subject, old, new, must_fail in CONTROLS:
            print("  %s  (%s)" % (name, why))
            mutated = texts[subject].replace(old, new)
            if mutated == texts[subject]:
                fail("%s: the anchor did not apply; this control tested nothing" % name)
                continue
            verdict, detail = parse_ok(work, mutated, name)
            if verdict == "syntax":
                fail("%s: the mutant does not parse, so it reverted nothing: %s"
                     % (name, detail.strip()[:300]))
                continue
            if verdict == "unmeasured":
                # Still a failure -- the control did not run and the sweep must
                # not be green -- but named as what it is. Scored as a syntax
                # error it would read as a defect in the mutation, and the next
                # person would go looking for one in a mutant that is fine.
                fail("%s: COULD NOT MEASURE whether the mutant parses, so this "
                     "control did not run: %s" % (name, detail.strip()[:300]))
                continue
            mtexts = dict(texts)
            mtexts[subject] = mutated
            mpassed, mfailed, munmeasured = run_all(
                work, mtexts,
                subject_filter=control_case_channels(subject, must_fail),
                label=name)
            (LOGS / ("%s.log" % name)).write_text(
                "passed: %s\nfailed: %s\nunmeasured: %s\n"
                % (mpassed, mfailed, munmeasured), encoding="utf-8")
            mutants_applied += 1
            red_outcomes_required += len(must_fail)
            # ROUND 5, FINDING 1: the scoring lives in score_control so the
            # self-check above can drive it with synthetic buckets and show it
            # refusing an unmeasured case. Only "ok" lines are credited; a
            # third bucket that did not run is a control failure in both
            # directions -- pinned and unpinned alike.
            for kind, msg in score_control(name, must_fail, mfailed, munmeasured):
                if kind == "ok":
                    red_outcomes_observed += 1
                    print("    ok   %s" % msg)
                else:
                    fail(msg)

    # --- the harness's own aim --------------------------------------------
    # BYTES, not text: see crlf_offenders above. A text-mode read decodes CRLF
    # back to \n, so this comparison would pass over a file whose every line
    # ending had been rewritten -- certifying as untouched the one change this
    # harness is most likely to make.
    if ZIP.read_bytes() != zip_bytes_before:
        print("FATAL: scripts/first-run/windows-zip.ps1 changed during the run")
        return 1
    if NSIS.read_bytes() != nsis_bytes_before:
        print("FATAL: scripts/first-run/windows-nsis.ps1 changed during the run")
        return 1
    if lib_bytes_before is not None and LIB.read_bytes() != lib_bytes_before:
        print("FATAL: scripts/first-run/lib.ps1 changed during the run")
        return 1

    # SAY WHAT THE NUMBERS ARE. Round-4 review corrected a claim of "89
    # mutations observed": there is one mutant per control, and 89 was the sum
    # of REQUIRED CASE TRANSITIONS across them, which is a different thing and a
    # larger-sounding one. The count is printed in the shape of the accepted
    # claim so it cannot be restated wrongly from the output.
    print("CASES AGAINST THE SHIPPED SOURCE: %d (every one must pass; a red case "
          "here is a defect in the shipped script, not in a control -- and a case "
          "that COULD NOT be measured is counted here as red, never as absent)"
          % (len(passed) + len(failed) + len(unmeasured)))
    print("MUTANTS APPLIED: %d (one source mutation per control)" % mutants_applied)
    # ROUND 5: REQUIRED and OBSERVED are now two numbers. Printing one number
    # under the label "REQUIRED AND OBSERVED" asserted the equality it was
    # supposed to report, and it stayed true-looking on a run where a required
    # outcome was never measured at all.
    print("RED CASE OUTCOMES REQUIRED: %d (sum across those %d mutants, not a "
          "mutation count; any unlisted red outcome is a failure)"
          % (red_outcomes_required, mutants_applied))
    print("RED CASE OUTCOMES OBSERVED:  %d (a case that could not be measured is "
          "not counted here; it is a control failure below)" % red_outcomes_observed)
    print("CONTROL FAILURES: %d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
