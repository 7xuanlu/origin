# Tests for scripts/first-run/lib.ps1. Run: pwsh -NoProfile -File scripts/first-run/lib.test.ps1
# (Windows PowerShell 5.1 works too: powershell -NoProfile -File ...)
#
# These cover the two properties that make a green channel run mean something:
#
#   1. An interpreter invoked through Check-Helper that is missing or fails is a
#      FAIL row, and the caller keeps going. Called bare, `& pwsh ...` raises a
#      CommandNotFoundException that terminates the enclosing try block, skips
#      every check after it, still runs the finally that deletes the data root,
#      and lets the script exit 0.
#   2. Evaluate fails when a declared row never arrived. Counting FAIL rows alone
#      cannot tell ten passing checks from ten that never ran, which is what a
#      skipped block looks like in the ledger.
#
# Each case runs in its own process against its own GAUNTLET_OUT, because lib.ps1
# keeps script-scoped state and appends to files. Only reaching the last line
# counts as a pass.
param([string]$Case)

# Deliberately NOT "Stop", for two reasons. The channels under test never set it
# globally, so a subject that did would not be measuring their behaviour. And in
# the harness it turns a child process writing to stderr into a terminating
# NativeCommandError, which aborts the whole run instead of recording one failed
# case - the negative controls in
# scripts/negative-controls/lib-ps1-negative-controls.sh caught exactly that.
$ErrorActionPreference = "Continue"

# --------------------------------------------------------------------------
# Subject: one scenario per -Case, exiting with Evaluate's verdict.
# --------------------------------------------------------------------------
if ($Case) {
    . (Join-Path $PSScriptRoot "lib.ps1")
    switch ($Case) {
        "all-declared-recorded" {
            Expect-Rows -Names @("a", "b")
            Check -Name "a" -Script { Write-Output "ok" }
            Check -Name "b" -Script { Write-Output "ok" }
        }
        "one-missing" {
            Expect-Rows -Names @("a", "b", "skipped-by-a-dead-block")
            Check -Name "a" -Script { Write-Output "ok" }
            Check -Name "b" -Script { Write-Output "ok" }
        }
        "only-info-rows" {
            # AC1's exact shape: the ledger is non-empty and holds no FAIL row,
            # so the old Evaluate returned $true while every contracted check
            # had been skipped.
            Expect-Rows -Names @("a", "b")
            Info "install-dir" "C:\\nowhere"
        }
        "no-declaration" {
            Check -Name "a" -Script { Write-Output "ok" }
        }
        "declared-by-another-channel-only" {
            # The shared-directory shape first-run-gauntlet.yml's own header
            # documents: one GAUNTLET_OUT, channels run one after another by
            # hand. This channel declares nothing, but expected.tsv already
            # holds another channel's rows, so a guard that only asks whether
            # the FILE has content passes -- and then finds nothing owed here.
            # Green with nothing contracted, which is the whole defect.
            Add-Content -Path $script:GauntletExpectedFile -Value "some-other-channel`ta" -Encoding utf8
            Check -Name "a" -Script { Write-Output "ok" }
        }
        "empty-ledger" {
            Expect-Rows -Names @("a")
        }
        "helper-interpreter-missing" {
            Expect-Rows -Names @("driver", "after-the-driver")
            Check-Helper -Name "driver" -Interpreter "wenlan-no-such-interpreter-8231" -Path "irrelevant.ps1"
            # Recorded only if the missing interpreter did NOT abort the block.
            Check -Name "after-the-driver" -Script { Write-Output "execution continued" }
        }
        "helper-nonzero-exit" {
            Expect-Rows -Names @("driver", "after-the-driver")
            $script = Join-Path $script:GauntletOut "exit7.ps1"
            Set-Content -Path $script -Value 'exit 7' -Encoding utf8
            Check-Helper -Name "driver" -Interpreter (Get-Process -Id $PID).Path -InterpreterArgs @("-NoProfile", "-File") -Path $script
            Check -Name "after-the-driver" -Script { Write-Output "execution continued" }
        }
        "helper-ok" {
            Expect-Rows -Names @("driver")
            $script = Join-Path $script:GauntletOut "ok.ps1"
            Set-Content -Path $script -Value 'Write-Output "helper ran"; exit 0' -Encoding utf8
            Check-Helper -Name "driver" -Interpreter (Get-Process -Id $PID).Path -InterpreterArgs @("-NoProfile", "-File") -Path $script
        }
        "helper-exits-0-declaring-nothing" {
            # The helper declares its own rows, from its own inputs, so the
            # producer is the only authority asserting the producer exists. A
            # regression that returns before its Expect-Rows leaves nothing
            # declared and nothing owed: the driver row is a PASS, no cli-* row
            # is missing because none was ever promised, and the channel stays
            # green with the entire round trip gone. -MustDeclare is the
            # caller's half of that contract.
            Expect-Rows -Names @("driver")
            $script = Join-Path $script:GauntletOut "silent.ps1"
            Set-Content -Path $script -Value 'Write-Output "returned before Expect-Rows"; exit 0' -Encoding utf8
            Check-Helper -Name "driver" -Interpreter (Get-Process -Id $PID).Path -InterpreterArgs @("-NoProfile", "-File") -Path $script -MustDeclare "^cli-"
        }
        "health-refused-is-a-negative" {
            # A REAL refusal, from the shell running this suite. Everything else
            # in this file is a stub, and stubs are exactly what hid the defect
            # this covers: they raised System.Net.WebException, which Windows
            # PowerShell 5.1 raises and pwsh 7 -- the edition
            # first-run-gauntlet.yml runs every channel with -- never does. So
            # `down` and `negative`, the two answers the health probes exist to
            # be able to give, were unreachable on the only host that runs them,
            # and no test in the tree could see it.
            Expect-Rows -Names @("refusal-classified")
            Check -Name "refusal-classified" -Script {
                # Bind port 0, read what the OS gave out, release it. The port
                # was free one instruction ago, which is the closest thing to a
                # guaranteed-closed port that does not hard-code a number.
                $probe = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback), 0
                $probe.Start()
                $port = $probe.LocalEndpoint.Port
                $probe.Stop()
                $kind = "no exception was raised at all"
                try {
                    $null = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" `
                        -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
                } catch { $kind = Get-WebFailureKind $_ }
                if ($kind -ne "negative") {
                    throw ("a refused loopback connection classified as '$kind' under " +
                           "PowerShell $($PSVersionTable.PSVersion) ($($PSVersionTable.PSEdition)): " +
                           "the probe cannot report a daemon that is not there")
                }
                Write-Output "refused -> negative"
            }
        }
        "health-reset-is-not-a-negative" {
            # A reset is a LIVE peer slamming the door -- a proxy, a firewall, a
            # service falling over mid-response. Reporting it as "the daemon is
            # gone" certifies a shutdown while the daemon is still on the port.
            # Under 5.1 a refusal and a reset were told apart by WebException's
            # .Status; under pwsh 7 they share an outer type and there is no
            # .Status, so the only thing left is how deep the SocketException
            # sits: directly inside for a refusal, under an IOException for a
            # reset. This case is what holds that rule.
            Expect-Rows -Names @("reset-classified")
            Check -Name "reset-classified" -Script {
                # Windows PowerShell 5.1 does not load System.Net.Http unless
                # asked, and without it this case dies constructing its own
                # fixture -- measured: "Cannot find type
                # [System.Net.Http.HttpRequestException]". That is the same fact
                # lib.ps1 compares type names for rather than using `-is [T]`.
                Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue
                $sock = New-Object System.Net.Sockets.SocketException 10054
                $io = New-Object System.IO.IOException -ArgumentList @("the connection was reset", $sock)
                $ex = New-Object System.Net.Http.HttpRequestException -ArgumentList @("boom", $io)
                $rec = New-Object System.Management.Automation.ErrorRecord -ArgumentList @($ex, "stub", "NotSpecified", $null)
                $kind = Get-WebFailureKind $rec
                if ($kind -ne "unmeasured") {
                    throw ("a reset from a live listener classified as '$kind'; " +
                           "something was there to send it, which is evidence " +
                           "AGAINST a shutdown, not for one")
                }
                Write-Output "reset -> unmeasured"
            }
        }
        "health-refused-stub-is-a-negative" {
            # The positive control for the case above. Same constructed shape,
            # same outer type, SocketException moved to depth 1 and the errno
            # changed to a refusal. Without this, a Get-WebFailureKind that
            # returned "unmeasured" for everything -- which is precisely the bug
            # being fixed -- would pass `health-reset-is-not-a-negative` and look
            # like a working rule.
            Expect-Rows -Names @("refusal-stub-classified")
            Check -Name "refusal-stub-classified" -Script {
                Add-Type -AssemblyName System.Net.Http -ErrorAction SilentlyContinue
                $sock = New-Object System.Net.Sockets.SocketException 10061
                $ex = New-Object System.Net.Http.HttpRequestException -ArgumentList @("boom", $sock)
                $rec = New-Object System.Management.Automation.ErrorRecord -ArgumentList @($ex, "stub", "NotSpecified", $null)
                $kind = Get-WebFailureKind $rec
                if ($kind -ne "negative") {
                    throw ("a refusal whose SocketException is directly inside " +
                           "classified as '$kind'; depth 1 is the refusal shape")
                }
                Write-Output "refused-stub -> negative"
            }
        }
        "helper-declares-required-prefix" {
            # The positive half of the same property: a helper that does reach
            # its Expect-Rows passes. Without this case, an assertion that
            # threw unconditionally would look identical to a working one.
            Expect-Rows -Names @("driver")
            $script = Join-Path $script:GauntletOut "declaring.ps1"
            Set-Content -Path $script -Encoding utf8 -Value @(
                ('. "' + (Join-Path $PSScriptRoot "lib.ps1") + '"'),
                'Expect-Rows -Names @("cli-roundtrip")',
                'Check -Name "cli-roundtrip" -Script { Write-Output "ok" }',
                'exit 0'
            )
            Check-Helper -Name "driver" -Interpreter (Get-Process -Id $PID).Path -InterpreterArgs @("-NoProfile", "-File") -Path $script -MustDeclare "^cli-"
        }
        "helper-reuses-stale-declaration" {
            # GAUNTLET_OUT is reused (first-run-gauntlet.yml's own header
            # documents one directory, channels run one after another) and
            # expected.tsv is append-only. So the run BEFORE this one left a
            # real cli-* row behind, under this very channel name. A guard that
            # asks "does the file hold a matching row" is answered yes by that
            # corpse, and a helper that has started returning before its own
            # Expect-Rows passes on the strength of the last run that worked.
            # Only rows this invocation added are this invocation's evidence.
            #
            # BOTH ledgers carry the previous run, because both are append-only.
            # That matters: with the old row present in findings.tsv too, the
            # declared/recorded multisets still balance and Evaluate sees no
            # GONE. The before/after delta is then the ONLY thing standing
            # between this run and a green certificate for a round trip that did
            # not happen -- which is why the assertion below rejects "GONE".
            Add-Content -Path $script:GauntletExpectedFile `
                -Value ("{0}`tcli-roundtrip" -f $script:GauntletChannel) -Encoding utf8
            Add-Content -Path $script:GauntletTsv `
                -Value ("{0}`tcli-roundtrip`tPASS`t0`tleft behind by the previous run" -f $script:GauntletChannel) -Encoding utf8
            Expect-Rows -Names @("driver")
            $script = Join-Path $script:GauntletOut "silent.ps1"
            Set-Content -Path $script -Value 'Write-Output "returned before Expect-Rows"; exit 0' -Encoding utf8
            Check-Helper -Name "driver" -Interpreter (Get-Process -Id $PID).Path -InterpreterArgs @("-NoProfile", "-File") -Path $script -MustDeclare "^cli-"
        }
        "ledger-unwritable" {
            # The reused directory again, one turn further. Both ledgers hold a
            # BALANCED previous run and are then made read-only. Add-Content is
            # non-terminating, so every row this run tries to write -- including
            # the FAIL -- vanishes with a warning, and what stays on disk is the
            # last run's clean result. Evaluate reading only the files would
            # certify this run with last run's evidence.
            Add-Content -Path $script:GauntletExpectedFile `
                -Value ("{0}`ta" -f $script:GauntletChannel) -Encoding utf8
            Add-Content -Path $script:GauntletTsv `
                -Value ("{0}`ta`tPASS`t0`tleft behind by the previous run" -f $script:GauntletChannel) -Encoding utf8
            Set-ItemProperty -Path $script:GauntletExpectedFile -Name IsReadOnly -Value $true
            Set-ItemProperty -Path $script:GauntletTsv -Name IsReadOnly -Value $true
            Expect-Rows -Names @("b")
            Check -Name "b" -Script { throw "this check failed" }
        }
        "ledger-read-truncated" {
            # The read side of the same defect. `Get-Content` is NON-TERMINATING:
            # on an I/O error partway through a file it emits the lines it did
            # read and then writes an error that the default preference prints
            # and steps over. Both ledgers are append-only in a reused
            # GAUNTLET_OUT, so the PREVIOUS run's balanced pair sits at the top
            # of both files and this run's rows come after it -- and a read that
            # stops at the same point in both leaves a prefix that balances
            # perfectly. Zero FAIL, zero GONE, zero DRIFT, over two files
            # nobody finished reading, with this run's real FAIL below the cut.
            #
            # The stub is that behaviour, not an invention: it emits the first
            # line and then Write-Error, which in an advanced function honours
            # the CALLER's -ErrorAction exactly as the cmdlet does. Under
            # Read-Ledger's -ErrorAction Stop it throws and Evaluate refuses;
            # against a bare `Get-Content` it hands back the balanced prefix.
            Add-Content -Path $script:GauntletExpectedFile `
                -Value ("{0}`ta" -f $script:GauntletChannel) -Encoding utf8
            Add-Content -Path $script:GauntletTsv `
                -Value ("{0}`ta`tPASS`t0`tleft behind by the previous run" -f $script:GauntletChannel) -Encoding utf8
            Expect-Rows -Names @("b")
            Check -Name "b" -Script { throw "this check failed" }
            function Get-Content {
                [CmdletBinding()]
                param(
                    [Parameter(Position = 0)][string]$Path,
                    [string]$LiteralPath,
                    [switch]$Raw,
                    [string]$Encoding
                )
                $target = if ($LiteralPath) { $LiteralPath } else { $Path }
                $all = @(Microsoft.PowerShell.Management\Get-Content -LiteralPath $target)
                if ($all.Count -gt 0) { Write-Output $all[0] }
                Write-Error "simulated I/O error partway through $target"
            }
        }
        "helper-declaration-count-unreadable" {
            # Count-Declared reads expected.tsv with -ErrorAction Stop by
            # design: a read that FAILED must not return 0, or the after-count
            # becomes a positive delta over rows this run never declared. But
            # the BEFORE count is taken outside Check, so that throw escaped
            # Check-Helper entirely -- no row recorded, every later statement
            # skipped, and the script still exits 0. Precisely the bare-`& pwsh`
            # shape Check-Helper exists to wrap, reintroduced by its own guard.
            #
            # Only expected.tsv is locked, so the FAIL row still reaches
            # findings.tsv; the failure being modelled is a read that cannot
            # happen, not a ledger that cannot be written.
            Expect-Rows -Names @("driver", "after-the-driver")
            $script = Join-Path $script:GauntletOut "ok.ps1"
            Set-Content -Path $script -Value 'Write-Output "helper ran"; exit 0' -Encoding utf8
            $lock = [System.IO.File]::Open($script:GauntletExpectedFile,
                [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::None)
            try {
                Check-Helper -Name "driver" -Interpreter (Get-Process -Id $PID).Path `
                    -InterpreterArgs @("-NoProfile", "-File") -Path $script -MustDeclare "^cli-"
            } finally { $lock.Dispose() }
            # Recorded only if the unreadable declaration file did NOT abort the
            # caller, exactly as for helper-interpreter-missing.
            Check -Name "after-the-driver" -Script { Write-Output "execution continued" }
        }
        "row-name-with-a-tab" {
            # Both ledgers are tab-separated with no quoting. A name carrying a
            # tab writes its own status column and slides every later field
            # right, so "gone`tPASS" can be read back as a passing row. Names
            # come from Check -Name, which is free text.
            Expect-Rows -Names @("a")
            Check -Name "a" -Script { Write-Output "ok" }
            Check -Name "smuggled`tPASS" -Script { throw "this check failed" }
        }
        "shadowing-row-names" {
            # A hashtable entry named Keys/Count/Values shadows the member of
            # the same name, so `$h.Keys` returns that entry's value instead of
            # the key list and the missing-row loop iterates over a number,
            # checking nothing. Row names come from Check -Name, which is free
            # text, so this is reachable.
            Expect-Rows -Names @("Keys", "Count", "Values", "gone-behind-the-shadow")
            Check -Name "Keys" -Script { Write-Output "ok" }
            Check -Name "Count" -Script { Write-Output "ok" }
            Check -Name "Values" -Script { Write-Output "ok" }
        }
        "drift-row-only" {
            # An undeclared row leaves a hole exactly its own size: nothing
            # declared it, so its later disappearance is invisible. Reported
            # AND fatal, so the declaration cannot fall behind the file.
            Expect-Rows -Names @("a")
            Check -Name "a" -Script { Write-Output "ok" }
            Check -Name "never-declared" -Script { Write-Output "ok" }
        }
        "case-differs-from-declaration" {
            # `@{}` is case-INSENSITIVE, so this pair matched and the run went
            # green on an identifier that never arrived. With the ordinal
            # comparer the declared row is GONE and the recorded one is DRIFT.
            Expect-Rows -Names @("Publisher-Identity")
            Check -Name "publisher-identity" -Script { Write-Output "ok" }
        }
        "recorded-more-often-than-declared" {
            # Two checks sharing a name: declared once, recorded twice. Either
            # one can be deleted and the other keeps the contract satisfied.
            Expect-Rows -Names @("a")
            Check -Name "a" -Script { Write-Output "ok" }
            Check -Name "a" -Script { Write-Output "ok" }
        }
        "clean-run-after-a-historical-failure" {
            # The harness seeded BOTH ledgers before this process started, so the
            # rows are already on disk when lib.ps1 is dot-sourced -- which is
            # exactly what a previous run into a reused GAUNTLET_OUT leaves. This
            # run declares "a" and records it PASSING.
            #
            # Evaluate used to judge every historical row for the channel, so the
            # seeded FAIL made this run red for a check that passed. One bad
            # manual run poisoned every later one, and the only way back was to
            # delete the directory.
            Expect-Rows -Names @("a")
            Check -Name "a" -Script { Write-Output "ok" }
        }
        "historical-pass-does-not-cover-a-check-that-did-not-run" {
            # The other direction, and the one that matters more. The seeded rows
            # are a PASSING pair from an earlier run. This run declares "a" and
            # never records it -- the block died. If the window were applied to
            # expected.tsv alone, the seeded PASS would satisfy this run's
            # declaration and the channel would go green over a check that never
            # ran: a corpse certifying a live contract.
            Expect-Rows -Names @("a")
        }
        "carried-prefix-rewritten" {
            # The window used to be a COUNT, and a count is not a boundary.
            #
            # Seeded: two rows from an earlier run, the second of them a PASS
            # for "a". This run declares "a" and records it FAILING, then swaps
            # the last two lines -- the file is exactly as long as it was, so
            # nothing shrank and nothing refused. The mark now cuts BELOW this
            # run's FAIL row: Evaluate reports it as an earlier run's and never
            # judges it, and the earlier run's PASS drops into this run's window
            # in its place. Zero FAIL rows, every declared row recorded, green
            # -- over a check that this run measured as failing.
            Expect-Rows -Names @("a")
            Check -Name "a" -Script { throw "the check failed" }
            $lines = @(Get-Content -LiteralPath $script:GauntletTsv)
            $swapped = @($lines[0], $lines[2], $lines[1])
            Set-Content -LiteralPath $script:GauntletTsv -Value $swapped -Encoding utf8
        }
        "declared-prefix-rewritten" {
            # The same mark on the other ledger. This run declares "a" and
            # records it, then moves its own DECLARATION above the mark and an
            # earlier run's below it. Without a digest the count is untouched
            # and Evaluate reads a contract that is not this run's.
            Expect-Rows -Names @("a")
            Check -Name "a" -Script { Write-Output "ok" }
            $lines = @(Get-Content -LiteralPath $script:GauntletExpectedFile)
            $swapped = @($lines[1], $lines[0])
            Set-Content -LiteralPath $script:GauntletExpectedFile -Value $swapped -Encoding utf8
        }
        default { Write-Host "unknown case: $Case"; exit 99 }
    }
    if (Evaluate) { exit 0 } else { exit 1 }
}

# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
$self = $MyInvocation.MyCommand.Path
$shell = (Get-Process -Id $PID).Path
$root = Join-Path ([IO.Path]::GetTempPath()) ("lib-ps1-test-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $root | Out-Null
$failures = 0
$reachedEnd = $false

function Assert-Case {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][int]$ExpectExit,
        [string[]]$ExpectText = @(),
        [string[]]$RejectText = @(),
        # Rows written into the ledgers BEFORE the subject process starts, which
        # is what a previous run into a reused GAUNTLET_OUT actually leaves. The
        # existing stale-row cases seed from inside the subject, after lib.ps1 is
        # loaded, so their rows belong to the subject's own run; a historical row
        # has to predate the load or the case is not the case.
        [string[]]$SeedFindings = @(),
        [string[]]$SeedExpected = @()
    )
    $out = Join-Path $root $Name
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    if ($SeedFindings.Count -gt 0) {
        Set-Content -Path (Join-Path $out "findings.tsv") -Value $SeedFindings -Encoding utf8
    }
    if ($SeedExpected.Count -gt 0) {
        Set-Content -Path (Join-Path $out "expected.tsv") -Value $SeedExpected -Encoding utf8
    }
    $env:GAUNTLET_OUT = $out
    $env:GAUNTLET_CHANNEL = "libtest"
    # A case that dies is a failed case, never a dead harness.
    $text = ""
    $rc = -1
    try { $text = (& $shell -NoProfile -File $self -Case $Name 2>&1 | Out-String); $rc = $LASTEXITCODE }
    catch { $text = $_.ToString(); $rc = -1 }
    $bad = @()
    if ($rc -ne $ExpectExit) { $bad += "exit $rc, expected $ExpectExit" }
    foreach ($t in $ExpectText) { if (-not $text.Contains($t)) { $bad += "missing text: $t" } }
    foreach ($t in $RejectText) { if ($text.Contains($t)) { $bad += "unexpected text: $t" } }
    if ($bad.Count -eq 0) {
        Write-Host "  ok   $Name"
    } else {
        Write-Host "  FAIL $Name"
        foreach ($b in $bad) { Write-Host "         $b" }
        Write-Host ($text -replace '(?m)^', '         | ')
        $script:failures++
    }
}

Write-Host "lib.test.ps1"

# --- Evaluate's row contract ---------------------------------------------
Assert-Case -Name "all-declared-recorded" -ExpectExit 0 `
    -ExpectText @("0 FAIL row(s), 0 declared row(s) never recorded") -RejectText @("GONE")

Assert-Case -Name "one-missing" -ExpectExit 1 `
    -ExpectText @("GONE", "skipped-by-a-dead-block", "1 declared row(s) never recorded")

# The case that makes the whole change worth making: no FAIL row anywhere, a
# non-empty ledger, and every contracted check absent.
Assert-Case -Name "only-info-rows" -ExpectExit 1 `
    -ExpectText @("GONE", "0 FAIL row(s), 2 declared row(s) never recorded")

Assert-Case -Name "no-declaration" -ExpectExit 1 -ExpectText @("no contracted rows declared")

# The declaration must be this channel's own. Riding on a neighbour's rows in a
# shared GAUNTLET_OUT is the same unchecked-reads-as-pass hole one level out.
Assert-Case -Name "declared-by-another-channel-only" -ExpectExit 1 `
    -ExpectText @("no contracted rows declared for libtest") -RejectText @("GONE")

Assert-Case -Name "empty-ledger" -ExpectExit 1 -ExpectText @("no findings recorded")

# Multiset EQUALITY, not containment: a surplus row is as fatal as a missing
# one, because nothing declared it and so nothing will notice it leaving.
Assert-Case -Name "drift-row-only" -ExpectExit 1 `
    -ExpectText @("DRIFT", "never-declared", "1 undeclared row(s)")

Assert-Case -Name "recorded-more-often-than-declared" -ExpectExit 1 `
    -ExpectText @("DRIFT", "recorded 2x, declared 1x")

# The declared name must be the recorded name, byte for byte.
Assert-Case -Name "case-differs-from-declaration" -ExpectExit 1 `
    -ExpectText @("GONE", "Publisher-Identity", "DRIFT", "publisher-identity")

# Row names are free text, so a row called Keys/Count/Values must not disable
# the contract it is part of.
Assert-Case -Name "shadowing-row-names" -ExpectExit 1 `
    -ExpectText @("GONE", "gone-behind-the-shadow", "1 declared row(s) never recorded")

# --- the run window -------------------------------------------------------
# Both ledgers are append-only and GAUNTLET_OUT is reused, so every run judged
# every run that came before it. One failed manual run left a FAIL row and made
# every later run red -- including a completely clean one -- for a check that
# passed. A verdict that cannot come back to green is a stuck light.
#
# The seeded rows here predate the subject process, which is what makes them
# HISTORICAL rather than this run's own. This case catches all three ways of
# getting the window wrong: no window at all (the seeded FAIL is counted, exit
# 1), a window on findings.tsv alone (this run's declaration is owed twice and
# recorded once, GONE, exit 1), and a window on expected.tsv alone (the seeded
# row is surplus, DRIFT, exit 1).
Assert-Case -Name "clean-run-after-a-historical-failure" -ExpectExit 0 `
    -SeedExpected @("libtest`ta") `
    -SeedFindings @("libtest`ta`tFAIL`t1`tthis failed in an earlier run") `
    -ExpectText @("0 FAIL row(s), 0 declared row(s) never recorded, 0 undeclared",
                  "written by an EARLIER run", "1 of them FAILED: a") `
    -RejectText @("GONE", "DRIFT")

# And the direction that matters more, because it is the one where a wrong
# window would go GREEN. The seeded pair is a PASSING previous run; this run
# declares "a" and its block dies before recording anything. Windowing only
# expected.tsv would let that corpse satisfy this run's contract -- a check that
# never ran, certified by an earlier run's row.
# A count-only window cannot see a ledger that lost a carried row and gained
# one of this run's: same length, different rows, and the mark now points into
# the middle of this run's own output. The first of these is green without the
# digest -- the strongest form of the hole, since the run that FAILED reports
# success -- and the second moves the declaration instead of the finding.
Assert-Case -Name "carried-prefix-rewritten" -ExpectExit 1 `
    -SeedFindings @("libtest`tzz`tINFO`t0`tan earlier run said something",
                    "libtest`ta`tPASS`t0`tthis passed in an earlier run") `
    -ExpectText @("are not the rows that were there when this run started",
                  "Unchecked, not a pass") `
    -RejectText @("0 FAIL row(s), 0 declared row(s) never recorded")

Assert-Case -Name "declared-prefix-rewritten" -ExpectExit 1 `
    -SeedExpected @("libtest`tzz") `
    -ExpectText @("are not the rows that were there when this run started",
                  "Unchecked, not a pass") `
    -RejectText @("GONE")

Assert-Case -Name "historical-pass-does-not-cover-a-check-that-did-not-run" -ExpectExit 1 `
    -SeedExpected @("libtest`ta") `
    -SeedFindings @("libtest`ta`tPASS`t0`tthis passed in an earlier run") `
    -ExpectText @("no findings recorded by THIS run")

# --- Check-Helper ---------------------------------------------------------
# A missing interpreter must be a FAIL row AND must not abort the caller.
Assert-Case -Name "helper-interpreter-missing" -ExpectExit 1 `
    -ExpectText @("[FAIL] driver", "[PASS] after-the-driver") -RejectText @("GONE")

Assert-Case -Name "helper-nonzero-exit" -ExpectExit 1 `
    -ExpectText @("[FAIL] driver", "exited 7", "[PASS] after-the-driver")

Assert-Case -Name "helper-ok" -ExpectExit 0 -ExpectText @("[PASS] driver")

# A helper that exits 0 without ever declaring its own rows cannot have reached
# its Expect-Rows, so nothing it was supposed to check is owed. The driver row
# must FAIL rather than certify a round trip that never happened.
Assert-Case -Name "helper-exits-0-declaring-nothing" -ExpectExit 1 `
    -ExpectText @("[FAIL] driver", "declared no NEW row matching '^cli-'") -RejectText @("GONE")

# The health classifier, against a REAL exception from the shell running this
# suite. Every other case here is a stub, and stubs raising WebException are
# what let a classifier that cannot answer under pwsh 7 look correct.
Assert-Case -Name "health-refused-is-a-negative" -ExpectExit 0 `
    -ExpectText @("[PASS] refusal-classified", "refused -> negative") `
    -RejectText @("GONE", "unmeasured")

# A reset is a live peer. The pair below is a rule and its control: remove the
# depth test and the first fails; break the classifier entirely and the second
# fails. One without the other proves nothing.
Assert-Case -Name "health-reset-is-not-a-negative" -ExpectExit 0 `
    -ExpectText @("[PASS] reset-classified", "reset -> unmeasured") `
    -RejectText @("GONE")

Assert-Case -Name "health-refused-stub-is-a-negative" -ExpectExit 0 `
    -ExpectText @("[PASS] refusal-stub-classified", "refused-stub -> negative") `
    -RejectText @("GONE")

Assert-Case -Name "helper-declares-required-prefix" -ExpectExit 0 `
    -ExpectText @("[PASS] driver", "PASS cli-roundtrip", "declared 1 new '^cli-' row(s)") `
    -RejectText @("GONE", "DRIFT")

# expected.tsv is append-only and its directory is reused, so "a matching row
# exists" is satisfied by the PREVIOUS run's rows. Counting before and after is
# what makes the declaration this run's evidence rather than last run's.
Assert-Case -Name "helper-reuses-stale-declaration" -ExpectExit 1 `
    -ExpectText @("[FAIL] driver", "declared no NEW row matching '^cli-'", "before=1 after=1") `
    -RejectText @("GONE")

# A tab in a row name rewrites the ledger's columns. Loud abort, never a row.
Assert-Case -Name "row-name-with-a-tab" -ExpectExit 1 `
    -ExpectText @("contains a tab or newline")

# A run whose ledger writes all failed leaves the previous run's balanced files
# on disk. Nothing readable can tell the two apart, so Evaluate must refuse on
# what it knows in memory rather than on what it can read.
Assert-Case -Name "ledger-unwritable" -ExpectExit 1 `
    -ExpectText @("the ledger could not be written", "Unchecked, not a pass") `
    -RejectText @("0 FAIL row(s), 0 declared row(s) never recorded")

# And the read side. Get-Content is non-terminating, so a file that failed
# partway through comes back as the lines before the failure -- which, in two
# append-only ledgers holding the previous run first, is a balanced prefix.
# The FAIL row this run recorded sits below the cut and is never counted.
Assert-Case -Name "ledger-read-truncated" -ExpectExit 1 `
    -ExpectText @("cannot read", "Unchecked, not a pass") `
    -RejectText @("0 FAIL row(s), 0 declared row(s) never recorded")

# Count-Declared throws rather than returning a wrong 0; the before-count is
# taken outside Check, so that throw must not escape the way a bare interpreter
# call does.
Assert-Case -Name "helper-declaration-count-unreadable" -ExpectExit 1 `
    -ExpectText @("[FAIL] driver", "cannot count the rows already declared",
                  "[PASS] after-the-driver") `
    -RejectText @("GONE")

# --------------------------------------------------------------------------
Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue
if ($failures -gt 0) {
    Write-Host "FAIL: lib.test.ps1 ($failures case(s))"
    exit 1
}
$reachedEnd = $true
if (-not $reachedEnd) { exit 1 }
Write-Host "PASS: lib.test.ps1"
exit 0
