# First-run gauntlet helpers (PowerShell). Dot-source this file.
#
# Same contract and TSV shape as lib.sh: channel, name, status (PASS|FAIL|INFO),
# rc, detail. Checks never throw; Evaluate returns $false when any FAIL exists.
#
#   Check -Name n -Script { ... }                    PASS when the block does not throw and $LASTEXITCODE is 0/unset
#   Check -Name n -Expect "substr" -Script { ... }   PASS additionally requires the block's output to contain substr
#   Check -Name n -ExpectFail "substr" -Script {...} PASS when the block throws or exits nonzero AND output contains
#                                                    substr AND something witnesses that execution reached the
#                                                    construct under test -- see "did execution reach the construct"
#   Reached                                          call it inside a multi-statement -ExpectFail block, immediately
#                                                    before the construct; a single-pipeline block needs no call
#   Info -Name n -Value v
#   Wait-Health -Url u -Seconds s                    polls for up to s WALL-CLOCK seconds; records seconds-to-health;
#                                                    returns $true/$false
#   Assert-Version -Url u -Expected v
#   Collect path...                                  copy into $GAUNTLET_OUT/logs
#   Expect-Rows -Names @("a","b")                    declare the PASS/FAIL rows this process contracts to record
#   Record-CarriedRow -Name n                        judge a row the workflow recorded BEFORE this process started
#                                                    (it is above the mark, so Evaluate never judges it) and record
#                                                    the verdict as `n-carried` in this run's window
#   Check-Helper -Name n -Interpreter i -Path p      run a helper script through an interpreter AS a recorded check
#     -MustDeclare "^mcp-"                         ... and FAIL unless the helper declared a row matching it
#   Evaluate                                         print table; return $true when no FAIL row and no declared row is missing
#
# Counting FAIL rows cannot tell ten passing checks from ten that never ran,
# so every channel declares the rows it owes before it runs them and Evaluate
# fails on any that never arrived. A row recorded but not declared is printed
# as drift rather than failed: an extra row can never hide a skipped block,
# and several rows here are legitimately conditional.

$script:GauntletOut = if ($env:GAUNTLET_OUT) { $env:GAUNTLET_OUT } else { Join-Path (Get-Location) "gauntlet-out" }
$script:GauntletChannel = if ($env:GAUNTLET_CHANNEL) { $env:GAUNTLET_CHANNEL } else { [IO.Path]::GetFileNameWithoutExtension($MyInvocation.PSCommandPath) }
if (-not $script:GauntletChannel) { $script:GauntletChannel = "windows" }
$script:GauntletTsv = Join-Path $script:GauntletOut "findings.tsv"
# Declared separately from findings.tsv: that file's shape is asserted by
# first-run-gauntlet.yml and read by summary.py, and a contract is not a finding.
$script:GauntletExpectedFile = Join-Path $script:GauntletOut "expected.tsv"
New-Item -ItemType Directory -Force -Path (Join-Path $script:GauntletOut "checks") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $script:GauntletOut "logs") | Out-Null

# --- the run window ----------------------------------------------------------
#
# Both ledgers are append-only and GAUNTLET_OUT is REUSED: first-run-gauntlet.yml's
# own header tells a human to keep one `$PWD/gauntlet-out` and run channel after
# channel into it, and a re-run of a leg writes into the same artifact directory
# again. So what is on disk is the union of every run that ever wrote there, and
# Evaluate judged all of it. One failed manual run left a FAIL row behind, and
# every later run -- including a completely clean one -- stayed red on it. A
# verdict that cannot return to green is not a verdict; it is a stuck light, and
# the first thing anyone does with a stuck light is stop reading it.
#
# The remedy is NOT to forget the old rows. A row that simply vanishes is the
# absence-reads-as-success defect this file exists to stop, one turn further out.
# It is to judge this run on THIS run's rows, and to say out loud how many rows
# belong to earlier ones and how many of those failed.
#
# The window is the number of lines each ledger ALREADY held when this process
# dot-sourced the library. Everything after that line was written by this run --
# including by a helper in its own process (cli-roundtrip.ps1, mcp-roundtrip.py,
# port-precheck.sh), which starts after this point and appends below the mark.
#
# A count that could not be taken is not zero. Zero would place every historical
# row inside the window and bring the stale FAIL straight back, so the failure is
# recorded and Evaluate refuses on it, exactly as it refuses on a broken write.
$script:GauntletWindowBroken = ""
$script:GauntletFindingsBase = 0
$script:GauntletExpectedBase = 0
$script:GauntletFindingsDigest = ""
$script:GauntletExpectedDigest = ""
# The IDENTITY of the carried rows, not merely how many there were.
#
# A COUNT is not a boundary. Delete one carried row and let this run append two,
# and the file is longer than it started: no shrink, nothing refuses, and the
# first $base lines -- the ones Evaluate reports as "written by an EARLIER run"
# and never judges -- now include a row THIS run wrote. A FAIL of this run's own
# is then attributed to a corpse and not judged, which is the absence-reads-as-
# success defect this window was added to close, one turn further in. Rewriting
# expected.tsv the same way lets a declaration and its finding disappear
# together and still compare equal.
#
# So the mark is the digest of the exact lines that were there. Same lines, same
# order, or the mark means nothing and Evaluate refuses.
function Get-LinesDigest([string[]]$lines) {
    if ($null -eq $lines -or $lines.Count -eq 0) { return "empty" }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes(($lines -join "`n"))
        return [System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace("-", "")
    } finally { $sha.Dispose() }
}
# ROUND 5. THE COUNT AND THE DIGEST ARE ONE MEASUREMENT, so they come from ONE
# READ. They used to be two: `Measure-LedgerLines` opened the file and counted,
# then `Measure-LedgerPrefix` opened it AGAIN and hashed the first $count lines.
# GAUNTLET_OUT is shared -- helpers in their own processes (cli-roundtrip.ps1,
# mcp-roundtrip.py, port-precheck.sh) append to these same ledgers -- so between
# those two opens the rows can change. Count two rows `A,B`, let a writer replace
# them with `C,D`, hash `C,D`: the baseline is `count=2, digest=H(C,D)`, perfectly
# self-consistent, and Evaluate then accepts `C,D` as "the rows that were there
# when this run started" and can return GREEN over evidence that was swapped
# underneath it. A digest taken to prove the identity of the carried rows must be
# taken of the SAME bytes the count came from, or it certifies a snapshot nobody
# observed.
#
# The residual, stated: `Get-Content` reads the file sequentially, so a writer
# can still interleave DURING this one read. What cannot happen any more is the
# two halves of the mark describing different contents -- whatever this read
# returned is both counted and hashed.
#
# ROUND 6. `Test-Path` USED TO BE THE DISCRIMINATOR HERE, AND IT ANSWERS $false
# TO TWO DIFFERENT QUESTIONS. The file is not there, and the file's ancestor
# cannot be searched -- under the default non-terminating error behaviour, in
# silence. A failed measurement therefore reached the "there was nothing here"
# branch and became a baseline of ZERO, which places every row already on disk
# inside this run's window: exactly the stale-evidence hole the window was added
# to close, entering through the guard's own absence test.
#
# The read is attempted instead, and its EXCEPTION is classified. Only
# ItemNotFoundException -- the provider saying the path does not exist -- is an
# absence. Everything else propagates, and the caller records
# GauntletWindowBroken, which Evaluate refuses on. The type is compared by NAME
# because `-is [T]` throws outright when the assembly holding T is not loaded,
# and this runs before anything else in the file.
function Measure-LedgerBaseline([string]$path) {
    # -ErrorAction Stop for the same reason Read-Ledger has it: Get-Content is
    # non-terminating, and a partial read here would under-count the rows already
    # on disk and pull an earlier run's tail into this run's window.
    $lines = $null
    try {
        $lines = @(Get-Content -LiteralPath $path -ErrorAction Stop)
    } catch {
        if ($_.Exception -and
            $_.Exception.GetType().FullName -eq "System.Management.Automation.ItemNotFoundException") {
            return [pscustomobject]@{ Count = 0; Digest = "empty" }
        }
        throw
    }
    return [pscustomobject]@{ Count = $lines.Count; Digest = (Get-LinesDigest $lines) }
}
try {
    $findingsMark = Measure-LedgerBaseline $script:GauntletTsv
    $expectedMark = Measure-LedgerBaseline $script:GauntletExpectedFile
    $script:GauntletFindingsBase = $findingsMark.Count
    $script:GauntletFindingsDigest = $findingsMark.Digest
    $script:GauntletExpectedBase = $expectedMark.Count
    $script:GauntletExpectedDigest = $expectedMark.Digest
} catch {
    $script:GauntletWindowBroken = "cannot measure how many rows the ledgers already held: $($_.Exception.Message)"
}

function Escape-Detail([string]$text) {
    $one = ($text -replace "`t", " " -replace "`r", "" -replace "`n", "|")
    if ($one.Length -gt 2000) { $one = $one.Substring(0, 2000) }
    return $one
}

function Assert-RowName([string]$name) {
    # findings.tsv and expected.tsv are tab-separated with no quoting, so a name
    # carrying a tab or newline does not merely look odd: "x`tPASS" writes its
    # own status column and every later field slides right, which can turn a
    # FAIL row into something summary.py reads as a pass. A name is a name.
    if ([string]::IsNullOrWhiteSpace($name)) { throw "row name is empty" }
    if ($name -match "[`t`r`n]") { throw "row name contains a tab or newline: '$name'" }
}

# Set when a ledger read or write failed. Both ledgers are append-only and
# GAUNTLET_OUT is reused, so a run whose writes all failed leaves the PREVIOUS
# run's balanced ledgers on disk -- and Evaluate, reading only the files, would
# certify this run with last run's evidence. `Add-Content` and `Get-Content`
# are non-terminating by default, so that happens in silence: read-only files,
# a full disk, a lock. Evaluate refuses on this flag; a run that could not write
# down what it saw did not measure anything.
$script:GauntletLedgerBroken = ""

# --- who wrote the rows below the mark ---------------------------------------
#
# ROUND 6. THE WINDOW IS A POSITION; IT IS NOT AN ATTRIBUTION. Everything above
# establishes where this run's rows START -- the count of what was carried in
# and the digest that proves those carried rows are still the same rows. None of
# it says that what comes AFTER the mark was written by this run. Both ledgers
# are append-only in a directory first-run-gauntlet.yml's header tells a human to
# reuse, so a second process running the same channel into it appends below this
# run's mark too:
#
#   A dot-sources this library and baselines both ledgers as EMPTY.
#   B -- same channel, same directory -- declares a row and records it PASSING.
#   A runs no checks at all and calls Evaluate.
#
# A's window is B's rows. Declared and recorded in balance, no FAIL, no GONE:
# GREEN, over a run that measured nothing. That is absent-then-present read as
# causation, and no count or digest of the CARRIED rows can see it, because both
# are statements about what came BEFORE the mark.
#
# So the rows below the mark are attributed. This process counts every row it
# appends, and Check-Helper measures each helper process's rows across the
# invocation that produced them, which gives the window a size it must equal.
# More rows than that and something else wrote into this run's window; fewer and
# rows this run wrote are no longer in it. Neither is a window this run can be
# judged on, and Evaluate refuses on both.
#
# THE RESIDUAL, stated: this is attribution by ARITHMETIC, not by a per-row
# stamp. It detects that the window holds rows this run cannot account for; it
# cannot say WHICH rows those are, and it cannot tell a foreign row from one
# written by a channel script that appends to the ledgers directly instead of
# through Record-Row / Expect-Rows. No shipped channel does -- the row shape is
# five unquoted TSV columns asserted by first-run-gauntlet.yml and read by
# summary.py, so a sixth run-id column is not available to stamp rows with.
$script:GauntletFindingsMine = 0
$script:GauntletExpectedMine = 0
$script:GauntletFindingsHelper = 0
$script:GauntletExpectedHelper = 0

# How many rows in $path name THIS channel. Absence is zero and ONLY absence:
# every other failure throws, because a read that failed must not be reported as
# a count, which is the same rule Count-Declared and Measure-LedgerBaseline hold.
function Measure-ChannelRows([string]$path) {
    $lines = $null
    try {
        $lines = @(Get-Content -LiteralPath $path -ErrorAction Stop)
    } catch {
        if ($_.Exception -and
            $_.Exception.GetType().FullName -eq "System.Management.Automation.ItemNotFoundException") {
            return 0
        }
        throw
    }
    $n = 0
    foreach ($one in $lines) {
        $cols = $one -split "`t"
        if ($cols.Count -ge 2 -and $cols[0] -eq $script:GauntletChannel) { $n++ }
    }
    return $n
}

function Write-Ledger([string]$path, [string]$line) {
    try {
        Add-Content -Path $path -Value $line -Encoding utf8 -ErrorAction Stop
        # Counted only on the path where the append SUCCEEDED. A row this
        # process failed to write is not in the window, and counting it here
        # would make the arithmetic below report a shortfall that is really a
        # write failure -- which GauntletLedgerBroken already reports, better.
        if ($path -eq $script:GauntletTsv) { $script:GauntletFindingsMine++ }
        elseif ($path -eq $script:GauntletExpectedFile) { $script:GauntletExpectedMine++ }
    } catch {
        if (-not $script:GauntletLedgerBroken) {
            $script:GauntletLedgerBroken = "cannot write ${path}: $($_.Exception.Message)"
        }
        Write-Host "[LEDGER] cannot write ${path}: $($_.Exception.Message)"
    }
}

# Every read of either ledger goes through here.
#
# `Get-Content` is NON-TERMINATING by default: on an I/O error partway through
# it hands back the lines it managed to read and carries on. Evaluate then
# compares two multisets, and a short read of expected.tsv drops declarations
# while a short read of findings.tsv drops the rows that would have been
# reported missing -- the two shortfalls cancel to zero FAILs, zero missing,
# zero surplus, and a green channel over files nobody finished reading. Same
# shape as the write path (round 13b, finding 2); round 13c found it on the
# read side. -ErrorAction Stop makes a partial read throw; the caller refuses.
function Read-Ledger([string]$path) {
    return @(Get-Content -LiteralPath $path -ErrorAction Stop)
}

function Count-Declared([string]$pattern) {
    # How many rows THIS channel has declared that match $pattern. Separated
    # from Check-Helper so the same count is taken the same way before and
    # after; two slightly different filters would be a measurement of nothing.
    #
    # -ErrorAction Stop: a read that FAILED must not return 0. The before-count
    # is taken outside Check, so a silent 0 there plus a working after-read is a
    # positive delta over rows this run never declared.
    #
    # Which is why `Test-Path` is not the absence test here either, for the
    # reason Measure-LedgerBaseline gives: it answers $false to a failed lookup
    # as well as to a missing file, and a 0 from the first is the silent 0 the
    # paragraph above is about. Only ItemNotFoundException is an absence.
    $lines = $null
    try {
        $lines = Read-Ledger $script:GauntletExpectedFile
    } catch {
        if ($_.Exception -and
            $_.Exception.GetType().FullName -eq "System.Management.Automation.ItemNotFoundException") {
            return 0
        }
        throw
    }
    return @($lines | Where-Object {
        $c = $_ -split "`t"
        $c.Count -ge 2 -and $c[0] -eq $script:GauntletChannel -and $c[1] -cmatch $pattern
    }).Count
}

function Record-Row([string]$status, [string]$name, [int]$rc, [string]$detail) {
    Assert-RowName $name
    $line = "{0}`t{1}`t{2}`t{3}`t{4}" -f $script:GauntletChannel, $name, $status, $rc, (Escape-Detail $detail)
    Write-Ledger $script:GauntletTsv $line
    $short = if ($detail) { " — " + (Escape-Detail $detail).Substring(0, [Math]::Min(200, (Escape-Detail $detail).Length)) } else { "" }
    Write-Host "[$status] $name (rc=$rc)$short"
}

# --- did execution reach the construct under test? ---------------------------
#
# ROUND 6, and it is the highest-leverage finding in this file because every
# channel row goes through Check. `-ExpectFail` read ANY nonzero outcome whose
# text contains the expected substring as a PASS -- and Check catches the whole
# script block AND the pipeline that captures its output. So this passed:
#
#   Check -Name target-rejects -ExpectFail "does not exist" -Script {
#       Get-Item C:\missing-fixture -ErrorAction Stop
#       & $actualTarget
#   }
#
# The SETUP line throws "does not exist", `$actualTarget` never runs, and the
# row records that the construct rejected something it never saw. A capture or
# formatting exception inside Check itself satisfies the same condition. That is
# scripts/AGENTS.md's own rule -- "a witness reached only from the exception path
# ratifies total absence, never the specific case" -- broken inside the harness
# that every control depends on.
#
# An ExpectFail PASS now needs a witness that execution REACHED the construct,
# and there are exactly two ways to hold one:
#
#   * the block is a single simple pipeline, in which case nothing else in it
#     could have thrown. That is a structural fact read off the block's own AST,
#     not a convention, and it is why the one shipped ExpectFail call site
#     (windows-zip.ps1's `stopped-marker-error`) needs no change;
#   * the block called `Reached` immediately before the construct. Every
#     multi-statement block must, because from in here one statement is
#     indistinguishable from another.
#
# Three outcomes, all spelled, none folded into a neighbour:
#
#   PASS         reached the construct, it failed, and it failed with the named
#                text.
#   FAIL, rc as  reached it and it did NOT fail that way -- the measured
#   the block's  negative.
#   FAIL, rc=2   COULD NOT MEASURE: never reached the construct, or the fault
#                came from this function's own capture. Unchecked is not a pass,
#                so it is still a FAIL row; the detail begins "unmeasured:" and
#                the rc distinguishes it in the ledger and in summary.py. The row
#                STATUS vocabulary is PASS|FAIL|INFO, asserted by
#                first-run-gauntlet.yml and read by summary.py, so the third
#                state is carried in the rc column rather than by inventing a
#                fourth status here.
$script:CheckReached = $null

function Reached {
    # Called from inside a Check block, immediately before the construct the
    # check is about. Outside one it throws rather than passing quietly: a
    # witness that can be set from anywhere witnesses nothing.
    param([string]$What = "")
    if ($null -eq $script:CheckReached) {
        throw "Reached was called outside a Check block, where it witnesses nothing"
    }
    $script:CheckReached = if ($What) { $What } else { "yes" }
}

# Is $Script one simple pipeline -- a single statement that is not a compound?
# $true only when that can be ESTABLISHED: an AST this cannot read answers
# $false, which demands the explicit witness rather than assuming one.
function Test-SingleStatementBlock([scriptblock]$Script) {
    try {
        $ast = $Script.Ast
        if ($null -eq $ast) { return $false }
        # A param block, or a begin/process block, is not the one-liner this
        # rule is about.
        if ($null -ne $ast.ParamBlock) { return $false }
        if ($null -ne $ast.BeginBlock -or $null -ne $ast.ProcessBlock) { return $false }
        $end = $ast.EndBlock
        if ($null -eq $end) { return $false }
        $statements = $end.Statements
        if ($null -eq $statements) { return $false }
        if ($statements.Count -ne 1) { return $false }
        # Type NAME, not `-is`: `-is [T]` throws when the assembly holding T is
        # not loaded. A PipelineAst is one command or one chain of them; an if,
        # a try, a loop or a block is some other Ast type and hides statements
        # this rule cannot see.
        return ($statements[0].GetType().Name -eq "PipelineAst")
    } catch {
        return $false
    }
}

function Check {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Script,
        [string]$Expect,
        [string]$ExpectFail
    )
    $log = Join-Path $script:GauntletOut ("checks\" + $Name + ".log")
    $global:LASTEXITCODE = 0
    $rc = 0
    # WHERE a fault came from. The block and the capture pipeline run
    # interleaved and land in the same catch, so the phase is recorded as the
    # pipeline moves between them. A hashtable and not a plain variable because
    # a ForEach-Object body is a child scope, where an assignment would make a
    # new local instead of updating this one.
    $phase = @{ Value = "block" }
    $faultedIn = ""
    # Cleared before the block, restored after: `Reached` uses $null to mean
    # "no Check is running", and a nested Check must not inherit the outer one's
    # witness.
    $outerReached = $script:CheckReached
    $script:CheckReached = ""
    # Stream the block's output line by line so a throw mid-block keeps
    # everything printed before it (a single Out-String assignment would not).
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
    Set-Content -Path $log -Value $out -Encoding utf8
    # A fault raised by this function's own output capture is not evidence about
    # the block at all, whichever rule is being applied to it.
    if ($faultedIn -eq "capture") {
        Record-Row FAIL $Name 2 ("unmeasured: the fault came from this harness's own output capture " +
            "rather than from the block, so nothing was learned about the check; got: " + $out)
        return
    }
    if ($ExpectFail) {
        # The witness comes first, because "it failed with the right text" is a
        # claim about the construct and is worth nothing until something says
        # the construct ran. Only the FAILING case needs it: a block that
        # completed ran every statement in it, so the construct was reached and
        # the answer is the measured negative below.
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

function Expect-Rows {
    # Declare the PASS/FAIL rows this process contracts to record. Written to a
    # file rather than a variable so a helper running in its own process
    # (cli-roundtrip.ps1, mcp-roundtrip.py) declares its own contract from its
    # own inputs, instead of the channel pasting a copy that silently rots.
    param([Parameter(Mandatory)][string[]]$Names)
    foreach ($n in $Names) {
        Assert-RowName $n
        Write-Ledger $script:GauntletExpectedFile ("{0}`t{1}" -f $script:GauntletChannel, $n)
    }
}

function Record-CarriedRow {
    # A row the WORKFLOW recorded in its own step before this process started,
    # so it sits above the mark and Evaluate reports it as carried in, never
    # judges it -- and a declaration of it could therefore never be balanced.
    # This restates its verdict as one row, `<Name>-carried`, inside this run's
    # window, which is the row a channel declares instead.
    #
    #   PASS  exactly one carried row of this channel is named $Name and PASSed.
    #   FAIL  it FAILed, or it is absent (the workflow step did not record it),
    #         or it appears more than once and they cannot be told apart.
    #   no row at all when findings.tsv could not be READ: nothing was measured,
    #         so this sets GauntletLedgerBroken and Evaluate refuses, exactly as
    #         it does for a ledger that could not be written.
    param([Parameter(Mandatory)][string]$Name)
    Assert-RowName $Name
    $rowName = "$Name-carried"
    # An unusable mark makes "the carried region" meaningless, and Evaluate is
    # already refusing on it; a row recorded here would only add a verdict to a
    # run that has none.
    if ($script:GauntletWindowBroken) { return }
    $lines = @()
    try {
        $lines = Read-Ledger $script:GauntletTsv
    } catch {
        if ($_.Exception -and
            $_.Exception.GetType().FullName -eq "System.Management.Automation.ItemNotFoundException") {
            $lines = @()
        } else {
            if (-not $script:GauntletLedgerBroken) {
                $script:GauntletLedgerBroken = "cannot read $script:GauntletTsv to look for the carried '$Name' row: $($_.Exception.Message)"
            }
            Write-Host "[LEDGER] cannot read $script:GauntletTsv to look for the carried '$Name' row: $($_.Exception.Message)"
            return
        }
    }
    $carried = @()
    if ($script:GauntletFindingsBase -gt 0) {
        $carried = @($lines | Select-Object -First $script:GauntletFindingsBase)
    }
    # -ceq, because Evaluate matches declared against recorded names with an
    # ordinal comparer; a name that differs by case is a different row there.
    $hits = @()
    foreach ($line in $carried) {
        $cols = $line -split "`t"
        if ($cols.Count -ge 3 -and $cols[0] -eq $script:GauntletChannel -and $cols[1] -ceq $Name) {
            $hits += , $cols
        }
    }
    if ($hits.Count -eq 0) {
        Record-Row FAIL $rowName 1 ("no '$Name' row for $script:GauntletChannel was carried into this run: " +
            "the workflow step that records it before this script starts did not record it")
        return
    }
    if ($hits.Count -gt 1) {
        Record-Row FAIL $rowName 1 ("$($hits.Count) '$Name' rows for $script:GauntletChannel were carried into " +
            "this run, so which one this run started behind cannot be told")
        return
    }
    $cols = $hits[0]
    $detail = if ($cols.Count -ge 5) { $cols[4] } else { "" }
    if ($cols[2] -eq "PASS") { Record-Row PASS $rowName 0 $detail }
    else { Record-Row FAIL $rowName 1 ("the carried '$Name' row is $($cols[2]): " + $detail) }
}

function Check-Helper {
    # Run a helper script through an interpreter and record one row for the run
    # itself. Calling the interpreter bare is not equivalent: when `pwsh` is not
    # installed, `& pwsh ...` raises CommandNotFoundException, which terminates
    # the enclosing try block, skips every statement after it, still runs the
    # finally that deletes the data root, and lets the script exit 0. Inside
    # Check the same exception is a FAIL row.
    #
    # -MustDeclare is the caller's half of the helper's contract. The helper
    # declares its own rows, from its own inputs, because only it knows which
    # optional ones apply -- but that makes the producer the sole authority that
    # the producer exists. A helper that exits 0 before reaching its own
    # Expect-Rows leaves nothing declared and nothing owed, and the channel
    # stays green with the whole round-trip missing. So the caller states the
    # PREFIX it must see declared, which is knowable here and survives the
    # helper's list changing shape.
    #
    # It is counted BEFORE and AFTER, and the count must go up. expected.tsv is
    # append-only and GAUNTLET_OUT is reused across runs, so "is there a row
    # matching ^mcp- in this file" is answered `yes` by the PREVIOUS run's rows.
    # A helper that starts exiting early would then keep passing on the strength
    # of the last run that worked -- a stale measurement read as a fresh one,
    # which is the same defect one level further out. Only the delta is this
    # run's evidence.
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Interpreter,
        [Parameter(Mandatory)][string]$Path,
        [string[]]$InterpreterArgs = @(),
        [string]$MustDeclare = ""
    )
    $helperLog = Join-Path $script:GauntletOut ("logs\" + $Name + ".log")
    # The before-count is taken OUTSIDE Check, and Count-Declared throws by
    # design (-ErrorAction Stop, so a failed read cannot return 0 and make the
    # after-count a positive delta over rows this run never declared). A throw
    # here would escape Check-Helper entirely: no row recorded, the channel's
    # remaining statements skipped, and -- exactly as with the bare `& pwsh`
    # this function exists to wrap -- the script still exits 0. Carry the error
    # into the block instead, where it becomes the FAIL row it is.
    $before = 0
    $beforeError = ""
    if ($MustDeclare) {
        try { $before = Count-Declared $MustDeclare }
        catch { $beforeError = $_.Exception.Message }
    }
    Check -Name $Name -Script {
        if ($beforeError) {
            throw ("cannot count the rows already declared matching '$MustDeclare' in " +
                   "$script:GauntletExpectedFile ($beforeError); without a before-count " +
                   "the after-count is not a delta, and this run's declarations cannot " +
                   "be told from the previous run's in a reused GAUNTLET_OUT")
        }
        # The helper appends to the same two ledgers from its OWN process, so
        # its rows are this run's evidence and land inside this run's window
        # without passing through Write-Ledger. Measured across the invocation
        # that produces them, which is the only span in which they can be
        # attributed to it. A measurement that fails does not become zero: it
        # sets GauntletWindowBroken, and Evaluate refuses, because an
        # unattributable window is exactly what the arithmetic exists to catch.
        $spanFindings = 0
        $spanExpected = 0
        $spanOk = $true
        try {
            $spanFindings = Measure-ChannelRows $script:GauntletTsv
            $spanExpected = Measure-ChannelRows $script:GauntletExpectedFile
        } catch {
            $spanOk = $false
            if (-not $script:GauntletWindowBroken) {
                $script:GauntletWindowBroken = "cannot count the ledger rows around $Path : $($_.Exception.Message)"
            }
        }
        $out = (& $Interpreter @InterpreterArgs $Path 2>&1 | Out-String)
        $rc = $LASTEXITCODE
        # Before the nonzero-exit throw below, so a helper that wrote rows and
        # then failed still has them attributed to it.
        if ($spanOk) {
            try {
                $script:GauntletFindingsHelper += ((Measure-ChannelRows $script:GauntletTsv) - $spanFindings)
                $script:GauntletExpectedHelper += ((Measure-ChannelRows $script:GauntletExpectedFile) - $spanExpected)
            } catch {
                if (-not $script:GauntletWindowBroken) {
                    $script:GauntletWindowBroken = "cannot count the ledger rows $Path added: $($_.Exception.Message)"
                }
            }
        }
        Set-Content -Path $helperLog -Value $out -Encoding utf8
        if ($rc -is [int] -and $rc -ne 0) { throw "$Interpreter $Path exited $rc; output in $helperLog" }
        if ($MustDeclare) {
            $after = Count-Declared $MustDeclare
            $added = $after - $before
            if ($added -le 0) {
                # Verdict first, paths after. Record-Row truncates the
                # console line at 200 characters, and both $Path and
                # $Interpreter are absolute; with the verdict last, whether it
                # is visible depends on where the interpreter happens to be
                # installed. Measured: pwsh from the MSIX package pushes it off
                # the end, pwsh from the MSI does not.
                throw ("declared no NEW row matching '$MustDeclare' " +
                       "(before=$before after=$after): $Path exited 0, so it " +
                       "cannot have reached its own Expect-Rows, and nothing " +
                       "it was supposed to check is owed. Output in $helperLog")
            }
            # Verdict first, for the reason given on the throw above.
            Write-Output "declared $added new '$MustDeclare' row(s); $Interpreter $Path ran; output in $helperLog"
            return
        }
        Write-Output "$Interpreter $Path ran; output in $helperLog"
    }
}

function Info([string]$Name, [string]$Value) {
    Record-Row INFO $Name 0 $Value
}

# --- the health probes -------------------------------------------------------
#
# ROUND 6. `catch { }` stood in both probes below and a FAIL row was recorded
# underneath it, so a DNS failure, a TLS failure, a proxy, a runner with no
# networking and a daemon that really is not listening all produced the same
# answer. The standing rule is that a failed measurement must never be
# indistinguishable from a negative one, and this was that collapse in the two
# checks a channel leans on hardest.
#
# THE EXCEPTION TYPE IS EDITION-SPECIFIC, and the edition that runs these
# scripts is not the edition the first table was measured on. first-run-
# gauntlet.yml runs every channel with `shell: pwsh` -- PowerShell 7, whose
# Invoke-WebRequest is built on HttpClient and never raises WebException. A
# classifier that tests for WebException therefore falls to its default on
# every failure under CI, and the default is "unmeasured". Both probes below
# lean on the NEGATIVE answer -- "the peer refused, nothing is listening" -- and
# that answer was unreachable on the only host that runs them.
#
# RE-MEASURED, both editions, same machine, loopback, isolated ports, and for
# the reset case a listener that accepts and closes with SO_LINGER(1,0):
#
#   pwsh 7.6.5 (Core)                    Windows PowerShell 5.1.26100.9278
#   ---------------------------------    ---------------------------------
#   HTTP 500 from a live server
#     HttpResponseException                WebException Status=ProtocolError
#     Response NON-NULL                    Response NON-NULL
#   connection refused
#     HttpRequestException                 WebException Status=ConnectFailure
#       -> SocketException                   -> SocketException
#          ConnectionRefused                    ConnectionRefused
#   unresolvable host
#     HttpRequestException                 WebException
#       -> SocketException HostNotFound     Status=NameResolutionFailure
#   live listener that RSTs
#     HttpRequestException                 WebException Status=ReceiveFailure
#       -> IOException                       -> IOException
#          -> SocketException                    -> SocketException
#             ConnectionReset                       ConnectionReset
#   timeout
#     TaskCanceledException                WebException Status=Timeout
#       -> TimeoutException
#
# WHAT THAT COSTS THE DESIGN. Under 5.1 a refusal and a reset were told apart by
# `.Status` (ConnectFailure vs ReceiveFailure). Under pwsh 7 they arrive with the
# SAME outer type and there is no Status property at all, so the only thing left
# is the socket error and HOW DEEP IT SITS: a refusal puts the SocketException
# directly inside the outer exception, a reset hides it one layer down under an
# IOException. That is the same line 5.1 drew, read off different facts, and it
# matters as much here as it did there -- A RESET IS A LIVE PEER. Classifying it
# as "down" certifies a daemon gone while it is still on the port.
#
# Only two things are NEGATIVES: a connection the peer REFUSED (nothing is
# listening on that port, which is the answer this probe exists to be able to
# give) and a response that completed with a non-200 status. Everything else --
# a timeout above all, which is a request that did not finish -- is unmeasured.
# Types are compared by NAME because `-is [T]` throws when the assembly holding
# T is not loaded.

# The two facts both classifiers branch on, read once. `SocketDepth` is 1 when
# the SocketException is the outer exception's direct inner, and deeper when it
# is buried -- which is the refusal/reset discriminator under pwsh 7.
function Get-WebExceptionShape($Exception) {
    $shape = [pscustomobject]@{
        Type        = ""
        Status      = ""
        HasResponse = $false
        SocketError = ""
        SocketDepth = 0
    }
    if ($null -eq $Exception) { return $shape }
    $shape.Type = $Exception.GetType().FullName
    $names = $Exception.PSObject.Properties.Name
    if ($names -contains "Status") { $shape.Status = "" + $Exception.Status }
    if ($names -contains "Response") { $shape.HasResponse = ($null -ne $Exception.Response) }
    $depth = 0
    $cur = $Exception.InnerException
    while ($cur) {
        $depth++
        if ($cur.GetType().FullName -eq "System.Net.Sockets.SocketException") {
            $shape.SocketError = "" + $cur.SocketErrorCode
            $shape.SocketDepth = $depth
            break
        }
        $cur = $cur.InnerException
    }
    return $shape
}

# Did the peer REFUSE the connection -- the one shape that means "nothing is
# listening on that port"? Not a reset, which is a live peer; not HostNotFound,
# which is not an answer about the port at all.
function Test-ConnectionRefused($Shape) {
    if ($null -eq $Shape) { return $false }
    if ($Shape.SocketError -ne "ConnectionRefused") { return $false }
    # Directly inside, not buried under an IOException: see the table above.
    if ($Shape.SocketDepth -ne 1) { return $false }
    if ($Shape.Type -eq "System.Net.Http.HttpRequestException") { return $true }
    if ($Shape.Type -eq "System.Net.WebException" -and $Shape.Status -eq "ConnectFailure") { return $true }
    return $false
}

# Did a response COMPLETE with a non-200 status? Something answered in HTTP,
# which is a measured negative for a version check and proof of reachability
# for a health check.
function Test-HttpErrorResponse($Shape) {
    if ($null -eq $Shape) { return $false }
    if ($Shape.Type -eq "Microsoft.PowerShell.Commands.HttpResponseException") { return $true }
    if ($Shape.Type -eq "System.Net.WebException" -and $Shape.Status -eq "ProtocolError") { return $true }
    return $false
}

function Get-WebFailureKind($ErrorRecord) {
    $ex = $null
    if ($ErrorRecord) { $ex = $ErrorRecord.Exception }
    if ($null -eq $ex) { return "unmeasured" }
    $shape = Get-WebExceptionShape $ex
    if (Test-HttpErrorResponse $shape) { return "negative" }
    if (Test-ConnectionRefused $shape) { return "negative" }
    return "unmeasured"
}

function Wait-Health([string]$Url, [int]$Seconds = 120) {
    # -TimeoutSec 5, not 2, and the constant is a measurement rather than a
    # taste. scripts/AGENTS.md records it: a refused loopback connect takes
    # ~2.05s on Windows because the SYN is retried, so a 2s budget turned every
    # genuine refusal into a Timeout -- and a Timeout is a request that did not
    # finish, which is unmeasured. Under the old two-state code that made no
    # visible difference because both answers were the same FAIL row; now it
    # would make this probe's NEGATIVE unreachable in principle, which is the
    # defect the AGENTS.md rule names and not a tuning choice. 5s is the floor
    # that rule states.
    #
    # $Seconds is a DEADLINE in wall-clock seconds, which is what its name says.
    # It used to bound the number of ATTEMPTS, each costing a 2s timeout plus a
    # 1s sleep, so `-Seconds 240` waited up to about 720s and the recorded
    # `seconds-to-health` was an attempt index rather than a time.
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $attempts = 0
    $unmeasured = $false
    $lastDetail = "no attempt completed"
    while ($true) {
        $attempts++
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) {
                Info "seconds-to-health" ([string][int][Math]::Ceiling($sw.Elapsed.TotalSeconds))
                return $true
            }
            # A response that completed and is not 200 is a measured negative.
            $lastDetail = "HTTP " + $r.StatusCode
        } catch {
            $lastDetail = $_.Exception.Message
            if ((Get-WebFailureKind $_) -eq "unmeasured") { $unmeasured = $true }
        }
        if ($sw.Elapsed.TotalSeconds -ge $Seconds) { break }
        Start-Sleep -Seconds 1
    }
    if ($unmeasured) {
        # ANY unmeasurable attempt spoils the window, because the negative being
        # claimed is about the WHOLE of it: "nothing answered for ${Seconds}s" is
        # not established for an instant nobody managed to ask about.
        Record-Row FAIL "health-timeout" 2 ("unmeasured: no 200 from $Url within ${Seconds}s, and at least " +
            "one of the $attempts attempt(s) never completed, so nothing was learned for that part of the " +
            "window. Last: " + $lastDetail)
        return $false
    }
    Record-Row FAIL "health-timeout" 1 ("no 200 from $Url within ${Seconds}s over $attempts attempt(s); " +
        "last: " + $lastDetail)
    return $false
}

function Assert-Version([string]$Url, [string]$Expected) {
    $want = $Expected.TrimStart("v")
    $body = ""
    $fetchFailed = ""
    try { $body = (Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5).Content }
    catch { $fetchFailed = $_.Exception.Message }
    # A fetch that failed says nothing about the VERSION even when the failure
    # itself was a measured negative: this row's question is "which version is
    # running", and a refused connection did not answer it. Both catches used to
    # be empty, so a body nobody could read and a body that read `1.2.2` when
    # `1.2.3` was wanted produced the same FAIL row with the same rc.
    if ($fetchFailed) {
        Record-Row FAIL "health-version" 2 ("unmeasured: the health body could not be read (" + $fetchFailed +
            "), so there is nothing to compare with " + $want)
        return
    }
    $got = ""
    $parsed = $false
    try { $got = "" + ($body | ConvertFrom-Json).version; $parsed = $true } catch { }
    if (-not $parsed -or -not $got) {
        Record-Row FAIL "health-version" 2 ("unmeasured: the health body did not parse as JSON carrying a " +
            "version, so there is nothing to compare with " + $want + "; health body: " + $body)
        return
    }
    # Published builds report `X.Y.Z+g<sha>`; compare the release part only.
    if (($got -split '\+')[0] -eq $want) { Record-Row PASS "health-version" 0 $got }
    else { Record-Row FAIL "health-version" 1 ("expected " + $want + "; health body: " + $body) }
}

function Collect {
    foreach ($p in $args) {
        if (Test-Path $p) { Copy-Item -Recurse -Force $p (Join-Path $script:GauntletOut "logs") -ErrorAction SilentlyContinue }
    }
}

function Evaluate {
    Write-Host ""
    # FIRST, before reading either file. Both ledgers are append-only in a
    # reused directory, so if this run's writes failed the files still hold the
    # PREVIOUS run's rows -- declared and recorded in balance, no FAIL, no GONE.
    # Reading them would certify this run with last run's evidence, and every
    # check that "failed" would have failed to say so. Nothing on disk can
    # distinguish the two; only this flag can.
    if ($script:GauntletLedgerBroken) {
        Write-Host "==> the ledger could not be written ($script:GauntletLedgerBroken):"
        Write-Host "    this run recorded nothing, and $script:GauntletTsv holds whatever"
        Write-Host "    the last run left. Unchecked, not a pass."
        return $false
    }
    # And the same refusal for the window itself. Without a trustworthy count of
    # what was already on disk, this run's rows cannot be told from an earlier
    # run's -- which is the whole question below.
    if ($script:GauntletWindowBroken) {
        Write-Host "==> $script:GauntletWindowBroken"
        Write-Host "    so the rows this run wrote cannot be told from the rows an earlier"
        Write-Host "    run left in this reused directory. Unchecked, not a pass."
        return $false
    }
    # A run that recorded nothing is unchecked, never a pass.
    #
    # Read ONCE, here, and reuse the lines further down. `-ErrorAction
    # SilentlyContinue` used to make a read that FAILED indistinguishable from a
    # file that was EMPTY -- benign at this line, since both return $false, but
    # the second read at the bottom of this function had no such guard. Reading
    # the same file twice also lets the two reads disagree: the second one
    # returning fewer rows than the first is a run judged against a contract
    # that was measured against different evidence.
    if (-not (Test-Path $script:GauntletTsv)) {
        Write-Host "==> no findings recorded ($script:GauntletTsv missing): unchecked, not a pass"
        return $false
    }
    try {
        $recorded = Read-Ledger $script:GauntletTsv
    } catch {
        Write-Host "==> cannot read $script:GauntletTsv ($($_.Exception.Message)):"
        Write-Host "    the rows this run recorded cannot be counted, so a FAIL row that is"
        Write-Host "    there cannot be seen. Unchecked, not a pass."
        return $false
    }
    # Split at the mark. `$carriedRows` is reported, never judged; `$recorded` is
    # narrowed to what this run wrote.
    if ($recorded.Count -lt $script:GauntletFindingsBase) {
        Write-Host "==> $script:GauntletTsv now holds $($recorded.Count) row(s) but held"
        Write-Host "    $script:GauntletFindingsBase when this run started. An append-only ledger"
        Write-Host "    cannot shrink, so the mark separating this run's rows from an earlier"
        Write-Host "    run's is meaningless. Unchecked, not a pass."
        return $false
    }
    $carriedRows = @()
    if ($script:GauntletFindingsBase -gt 0) {
        $carriedRows = @($recorded | Select-Object -First $script:GauntletFindingsBase)
    }
    # Identity, not length. A file that lost a carried row and gained one of
    # this run's is exactly as long as it was, so the count above is satisfied
    # while the mark now cuts through the middle of this run's own rows.
    if ((Get-LinesDigest $carriedRows) -ne $script:GauntletFindingsDigest) {
        # The path goes on its own line: interpolated into the sentence it wraps
        # at whatever width the temp path happens to have, and a caller matching
        # on the sentence then matches nothing.
        Write-Host "==> $script:GauntletTsv :"
        Write-Host "    its first $script:GauntletFindingsBase row(s) are not the rows that were there when this run started."
        Write-Host "    An append-only ledger"
        Write-Host "    only grows at the end, so the mark separating this run's rows from an"
        Write-Host "    earlier run's no longer points anywhere. Rows of this run could be"
        Write-Host "    sitting above it, reported as an earlier run's and never judged."
        Write-Host "    Unchecked, not a pass."
        return $false
    }
    $recorded = @($recorded | Select-Object -Skip $script:GauntletFindingsBase)
    if (-not $recorded) {
        Write-Host "==> no findings recorded by THIS run in $script:GauntletTsv : unchecked, not a pass"
        return $false
    }
    # Attribution, not position. See the note above $script:GauntletFindingsMine:
    # the digest proves the rows ABOVE the mark are still the rows that were
    # there, and says nothing about who wrote the rows below it. `$row` and not
    # `$line` deliberately -- `foreach ($line in $recorded) {` is the anchor a
    # negative control cuts on, and a second copy of that line would make the
    # control refuse to apply rather than measure anything.
    $windowMine = 0
    foreach ($row in $recorded) {
        $cols = $row -split "`t"
        if ($cols.Count -ge 2 -and $cols[0] -eq $script:GauntletChannel) { $windowMine++ }
    }
    $windowOwed = $script:GauntletFindingsMine + $script:GauntletFindingsHelper
    if ($windowMine -ne $windowOwed) {
        Write-Host "==> $script:GauntletTsv :"
        Write-Host "    this run's window holds $windowMine row(s) for $script:GauntletChannel, but this process"
        Write-Host "    wrote $script:GauntletFindingsMine and the helpers it ran wrote $script:GauntletFindingsHelper."
        Write-Host "    Rows below this run's mark came from somewhere this run cannot account"
        Write-Host "    for -- another process writing this channel into the same reused"
        Write-Host "    GAUNTLET_OUT -- or rows this run wrote are no longer in it. Either way"
        Write-Host "    the rows judged below would not all be this run's, and a run judged on"
        Write-Host "    another's evidence is not judged. Unchecked, not a pass."
        return $false
    }
    # A run that declared no contract cannot be told apart from a run whose
    # checks all vanished, so an absent declaration is unchecked too.
    #
    # Built and tested PER CHANNEL, before anything else looks at it. The file
    # is shared: GAUNTLET_OUT is one directory, and the header of
    # first-run-gauntlet.yml tells a human to reuse `$PWD/gauntlet-out` while
    # running channel after channel by hand. Testing the FILE for content
    # instead of this channel's rows would let a channel that declared nothing
    # ride on a previous channel's declarations -- expected empty, missing 0,
    # green -- which is the unchecked-reads-as-pass defect this guard exists to
    # stop, one level further out.
    # ORDINAL, not the default. `@{}` is a Hashtable with a case-INSENSITIVE
    # comparer, so a row declared as "Publisher-Identity" and recorded as
    # "publisher-identity" matched and the run went green on an identifier that
    # never arrived. Measured on this host: with the ordinal comparer,
    # ContainsKey('ABC') is False against a stored 'abc'. An entry named Keys
    # still shadows the member, so PSBase below stays load-bearing.
    $expected = New-Object System.Collections.Hashtable([System.StringComparer]::Ordinal)
    if (Test-Path $script:GauntletExpectedFile) {
        try {
            $declared = Read-Ledger $script:GauntletExpectedFile
        } catch {
            # The declarations ARE the "did it run" half of this contract. A
            # short read here drops rows nobody will then report missing, and a
            # short read of findings.tsv drops the rows that would have been
            # missing -- the two shortfalls cancel, and a channel whose checks
            # never ran comes out green over two files nobody finished reading.
            Write-Host "==> cannot read $script:GauntletExpectedFile ($($_.Exception.Message)):"
            Write-Host "    the contract this run declared cannot be read, so a declared row"
            Write-Host "    that never arrived cannot be reported. Unchecked, not a pass."
            return $false
        }
        # The same mark, for the same reason. Windowing findings.tsv alone would
        # leave an earlier run's DECLARATIONS owed by this one, so a clean run
        # would report every historical row GONE; windowing expected.tsv alone
        # would let an earlier run's recorded row satisfy this run's contract,
        # which is worse -- a check that never ran, certified by a corpse.
        if ($declared.Count -lt $script:GauntletExpectedBase) {
            Write-Host "==> $script:GauntletExpectedFile shrank from $script:GauntletExpectedBase row(s)"
            Write-Host "    to $($declared.Count). An append-only ledger cannot shrink, so this"
            Write-Host "    run's declarations cannot be separated from an earlier run's."
            Write-Host "    Unchecked, not a pass."
            return $false
        }
        $declaredCarried = @()
        if ($script:GauntletExpectedBase -gt 0) {
            $declaredCarried = @($declared | Select-Object -First $script:GauntletExpectedBase)
        }
        if ((Get-LinesDigest $declaredCarried) -ne $script:GauntletExpectedDigest) {
            Write-Host "==> the first $script:GauntletExpectedBase row(s) of $script:GauntletExpectedFile"
            Write-Host "    are not the rows that were there when this run started. A declaration"
            Write-Host "    this run made could be sitting above the mark, unread, and its finding"
            Write-Host "    dropped with it -- the two sides then agree about a check that never"
            Write-Host "    ran. Unchecked, not a pass."
            return $false
        }
        $declared = @($declared | Select-Object -Skip $script:GauntletExpectedBase)
        # The same attribution on the declaration side, and it is the half that
        # matters more: a foreign DECLARATION satisfied by a foreign finding is
        # a contract this run never made, reported as met.
        $declaredMine = 0
        foreach ($row in $declared) {
            $cols = $row -split "`t"
            if ($cols.Count -ge 2 -and $cols[0] -eq $script:GauntletChannel) { $declaredMine++ }
        }
        $declaredOwed = $script:GauntletExpectedMine + $script:GauntletExpectedHelper
        if ($declaredMine -ne $declaredOwed) {
            Write-Host "==> $script:GauntletExpectedFile :"
            Write-Host "    this run's window holds $declaredMine declaration(s) for $script:GauntletChannel, but this"
            Write-Host "    process declared $script:GauntletExpectedMine and the helpers it ran declared $script:GauntletExpectedHelper."
            Write-Host "    The contract judged below would not all be this run's. Unchecked, not a pass."
            return $false
        }
        foreach ($line in $declared) {
            $cols = $line -split "`t"
            if ($cols.Count -lt 2 -or $cols[0] -ne $script:GauntletChannel) { continue }
            $expected[$cols[1]] = 1 + [int]$expected[$cols[1]]
        }
    }
    # PSBase.Count for the same reason as PSBase.Keys below: a row named Count
    # would shadow the member and report its own tally as the row total.
    if ($expected.PSBase.Count -eq 0) {
        Write-Host "==> no contracted rows declared for $script:GauntletChannel in $script:GauntletExpectedFile : unchecked, not a pass"
        return $false
    }

    Write-Host "==> findings for $script:GauntletChannel"
    $fails = 0
    $foreign = 0
    $actual = New-Object System.Collections.Hashtable([System.StringComparer]::Ordinal)
    foreach ($line in $recorded) {
        $cols = $line -split "`t"
        if ($cols.Count -lt 4) { continue }
        # Scoped to this channel, like the declarations above. A shared ledger
        # otherwise reports a neighbour's FAIL against this channel, with no
        # column in the printout to say whose it was.
        if ($cols[0] -ne $script:GauntletChannel) { $foreign++; continue }
        Write-Host ("  {0,-4} {1,-40} rc={2}" -f $cols[2], $cols[1], $cols[3])
        if ($cols[2] -eq "FAIL") { $fails++ }
        if ($cols[2] -eq "PASS" -or $cols[2] -eq "FAIL") {
            $actual[$cols[1]] = 1 + [int]$actual[$cols[1]]
        }
    }
    if ($foreign -gt 0) {
        Write-Host ("  ---- {0} row(s) in this shared ledger belong to other channels and are not judged here" -f $foreign)
    }

    # Carried-over rows are REPORTED, not judged and not dropped in silence.
    # Judging them is what made one historical failure permanent; deleting them,
    # or saying nothing about them, would be the same defect from the other side
    # -- evidence that disappeared, with a green run over the gap. So they stay
    # in findings.tsv, summary.py still renders them, and this line says how many
    # there are and how many failed, by name, for the channel being judged.
    $carriedMine = 0
    $carriedFails = @()
    foreach ($line in $carriedRows) {
        $cols = $line -split "`t"
        if ($cols.Count -lt 4 -or $cols[0] -ne $script:GauntletChannel) { continue }
        $carriedMine++
        if ($cols[2] -eq "FAIL") { $carriedFails += $cols[1] }
    }
    if ($carriedMine -gt 0) {
        Write-Host ("  ---- {0} row(s) for {1} were written by an EARLIER run into this reused" -f $carriedMine, $script:GauntletChannel)
        Write-Host  "       GAUNTLET_OUT. They are still in the ledger and still in the summary,"
        Write-Host  "       but this run is judged on its own rows."
        if ($carriedFails.Count -gt 0) {
            Write-Host ("  ---- {0} of them FAILED: {1}" -f $carriedFails.Count, ($carriedFails -join ", "))
            Write-Host  "       A later clean run does not undo an earlier failure; it also does"
            Write-Host  "       not inherit it. Read both."
        }
    }

    # Missing is the failure: a declared check that recorded nothing did not run,
    # which is exactly what counting FAIL rows alone reports as green.
    # PSBase throughout. A hashtable ENTRY named Keys, Count or Values shadows
    # the member of the same name: with $h['Keys'] = 7, `$h.Keys` returns 7, not
    # the key list, so this loop would iterate over a number and check nothing --
    # a row contract that silently verifies nothing, which is the exact defect
    # class this whole change exists to close. Measured on this host, not assumed.
    $missing = 0
    foreach ($name in ($expected.PSBase.Keys | Sort-Object)) {
        $want = [int]$expected[$name]
        $got = [int]$actual[$name]
        if ($got -lt $want) {
            Write-Host ("  GONE {0,-40} declared {1}x, recorded {2}x - the check never ran" -f $name, $want, $got)
            $missing += ($want - $got)
        }
    }
    # Surplus is a failure too, so this is multiset EQUALITY rather than
    # containment. Reporting an undeclared row and passing anyway left a hole
    # exactly the size of the row: nothing declared it, so its later
    # disappearance is invisible -- a check could be added, quietly deleted, and
    # every run stay green in between. The same arithmetic catches a row
    # recorded more times than it was declared, which means a loop ran twice or
    # two checks share a name and one of them can vanish behind the other.
    $surplus = 0
    foreach ($name in ($actual.PSBase.Keys | Sort-Object)) {
        $want = [int]$expected[$name]
        $got = [int]$actual[$name]
        if ($got -gt $want) {
            Write-Host ("  DRIFT {0,-40} recorded {1}x, declared {2}x - undeclared checks can vanish unnoticed" -f $name, $got, $want)
            $surplus += ($got - $want)
        }
    }

    Write-Host "==> $fails FAIL row(s), $missing declared row(s) never recorded, $surplus undeclared row(s)"
    return (($fails -eq 0) -and ($missing -eq 0) -and ($surplus -eq 0))
}
