# First-run gauntlet: Windows zip channel. Follows the documented flow verbatim
# (install.sh header: "download wenlan-windows-x64.zip from the GitHub release
# page"): extract, add to PATH, `wenlan.exe setup --basic`, `wenlan.exe background on`,
# `wenlan.exe status`. Everything comes from env (TAG, VERSION, GAUNTLET_OUT,
# GAUNTLET_CHANNEL, REPO_ROOT). Never sets $ErrorActionPreference = "Stop" globally.
$ProgressPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot "lib.ps1")

$Tag = $env:TAG
$Version = $env:VERSION
$Health = "http://127.0.0.1:7878/api/health"
$Helpers = Join-Path $env:REPO_ROOT "scripts\first-run"
$Work = Join-Path $script:GauntletOut "work-zip"
$Extract = Join-Path $Work "extract"
$InstallDir = Join-Path $env:LOCALAPPDATA "Programs\wenlan"
$DataDir = Join-Path $env:LOCALAPPDATA "wenlan"
$Wenlan = Join-Path $InstallDir "wenlan.exe"
# What this run may kill and delete. See "WHAT THIS RUN OWNS" below; both start
# as NOT OURS on purpose, and only a measurement taken before anything is
# started can move them.
$TaskName = "WenlanServer"
$OwnedServerImage = Join-Path $InstallDir "wenlan-server.exe"
$PreexistingServerPids = $null
# TWO gates, not one, because creating and destroying need different evidence.
#
#   $MayDriveTask -- the name was FREE before this run. This is what licenses
#       `wenlan background on` to register at that name at all. It gates the
#       CLI verbs and not only `schtasks /end` and `/delete`: `background on` /
#       `background off` reach exactly the same task through the CLI instead of
#       the scheduler binary, and an ownership refusal with a second door in it
#       is not a refusal.
#   $TaskOwned -- the name was free before AND a task was MEASURED at it after
#       the registration ran. Only this licenses ending, deleting, or switching
#       off. "It was absent before" alone infers ownership from a pre-state and
#       never asks what the registration actually did, so a `background on`
#       that failed while something else took the name reads as ours to delete.
$MayDriveTask = $false
$TaskOwned = $false
# The delete's own result, so `no-leftover-task` can require it rather than
# read it off a log line. Declared HERE, at file scope, because `finally` runs
# even when `try` aborts on its first line and every name it reads must exist.
$TaskDeleteResult = $null
$preTask = $null
$postTask = $null
# The same two-fact licence, for the two directories this run installs into.
# $DataDir is %LOCALAPPDATA%\wenlan -- on a developer machine that is the real
# memorydb, config and logs, not a scratch tree, and unlike a scheduled-task
# registration it cannot be put back. An unconditional `Remove-Retry` in the
# cleanup deletes it after a run that aborted before installing anything, and
# on a machine where Wenlan was already installed.
#
# Absent before AND THIS RUN PERFORMED THE ACT THAT CLAIMED IT. Absent-before
# alone fails here for the same reason it fails for the task: it infers
# ownership from a pre-state and never asks what this run actually did. The
# second half is an ACT rather than a second read -- see WHAT BINDS A TREE TO
# THIS RUN in lib.ps1 -- so there is no `...Owned` boolean; the claim's result
# is the mark's own tri-state.
$preInstallDir = $null
$preDataDir = $null
# The run-scoped ownership marks. Declared here for the reason everything else
# in this block is: `finally` reads them and `try` may never reach them.
$InstallDirMark = $null
$DataDirMark = $null
$ExpectedMembers = @("wenlan.exe", "wenlan-server.exe", "wenlan-mcp.exe", "onnxruntime.dll", "vulkan-1.dll", "VulkanRT-License.txt")
if (-not $env:GAUNTLET_CHANNEL) { $env:GAUNTLET_CHANNEL = $script:GauntletChannel }
# Recovery is the behaviour under test; a harness-wide opt-out would hide it.
Remove-Item Env:\WENLAN_NO_AUTOSTART -ErrorAction SilentlyContinue

function Get-Asset([string]$AssetName, [string]$Dest) {
    $url = "https://github.com/7xuanlu/wenlan/releases/download/$Tag/$AssetName"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try { Invoke-WebRequest -Uri $url -OutFile $Dest -UseBasicParsing -ErrorAction Stop; return $true }
        catch { Write-Host "download attempt $attempt failed: $($_.Exception.Message)"; Start-Sleep -Seconds (5 * $attempt) }
    }
    return $false
}
# Run a native executable and MEASURE it: did it run, what did it exit with,
# what did it say. Returns { Ran; ExitCode; Output }.
#
# The form this replaces, `& schtasks.exe /end /tn WenlanServer 2>&1 | Out-Null`,
# has two outcomes and neither is a measurement.
#
# MEASURED ON THIS HOST (Windows PowerShell 5.1.26100.9278), with a native
# command that writes one benign line to stderr and exits 0:
#
#   $ErrorActionPreference = 'Continue'
#       & cmd.exe /c "echo benign 1>&2 & exit /b 0" 2>&1 | Out-Null
#       -> $? = False, $Error.Count = 1, LASTEXITCODE = 0, text discarded
#   $ErrorActionPreference = 'Stop', the same call inside a function
#       -> THREW System.Management.Automation.RemoteException, and the function
#          NEVER REACHED ITS `return`
#
# `2>&1` on a NATIVE command turns each stderr LINE into a NativeCommandError
# record in the PowerShell error stream. `schtasks /end` writes a benign line
# there when the task is not running -- the common case, since this runs at
# cleanup -- so under an inherited `Stop` preference `Stop-Daemon` throws ON ITS
# FIRST STATEMENT and the tri-state its two callers branch on is never
# returned. THE TRI-STATE DIES INSIDE THE CALLEE. This script never sets `Stop`
# itself, but $ErrorActionPreference is DYNAMICALLY SCOPED: a wrapper, a
# dot-source or a CI shell that sets it owns every frame below it, and a probe
# that is only correct because of what its caller happens to prefer is not a
# probe. And when it does NOT throw, the other outcome stands: `| Out-Null`
# destroys the text and `$LASTEXITCODE = 0` afterwards destroys the status.
#
# So: the preference is pinned HERE, where dynamic scoping makes it cover the
# call whatever the caller set; the output is turned into plain strings before
# it leaves; and the exit code is read explicitly and handed back as data.
# Nothing about this call is allowed to become an exception in the caller.
function Invoke-Native {
    param([Parameter(Mandatory)][string]$File, [string[]]$Arguments = @())
    $ErrorActionPreference = 'Continue'
    $global:LASTEXITCODE = 0
    try {
        # `2>&1` is what keeps a native command's stderr out of the caller's
        # error stream under an inherited `Stop` -- that is C2, and it stays.
        # What changed is WHEN the two streams stop being distinguishable.
        # This used to stringify inside the same pipeline (`| ForEach-Object {
        # "$_" }`), which flattened an ErrorRecord into a line that reads
        # exactly like a line of stdout. A caller whose rule is "any stderr at
        # all means this read is not a measurement" then had nothing to read.
        #
        # So the capture stays UNFLATTENED and the split is made here: an
        # ErrorRecord is still an ErrorRecord until something asks for its
        # text. `Output` is unchanged for every caller that only wants the
        # combined transcript; `ErrorCount`/`ErrorText` are additive, and
        # Get-PortListenerWitness refuses on them, because its preamble rule
        # must tolerate two unmatchable localised lines and therefore cannot
        # also be the thing that catches a diagnostic sitting among them.
        $raw = @(& $File @Arguments 2>&1)
    } catch {
        # A missing executable, a bad path: the command did not run at all,
        # which is not the same as running and failing.
        return [pscustomobject]@{ Ran = $false; ExitCode = $null
            Output = "$File could not be run ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }
    $rc = $LASTEXITCODE
    # The stale rc must not leak into the next Check block; the returned
    # ExitCode above is the copy that is meant to be read.
    $global:LASTEXITCODE = 0
    $errs = @($raw | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] })
    $out = @($raw | ForEach-Object { "$_" })
    return [pscustomobject]@{ Ran = $true; ExitCode = $rc; Output = ($out -join "`n")
        ErrorCount = $errs.Count
        ErrorText = (@($errs | ForEach-Object { "$_" }) -join "`n") }
}
# The health probe's client timeout, in seconds, and the floor under it.
#
# MEASURED ON THIS HOST (Windows PowerShell 5.1.26100.9278), same process, same
# closed loopback port, back to back:
#
#   -TimeoutSec 5 -> ConnectFailure / ConnectionRefused after 2090 ms
#   -TimeoutSec 2 -> Timeout                            after 2011 ms
#
# A refused connection takes about two seconds to come back because Windows
# retries the SYN, so a client timeout shorter than that WINS THE RACE and the
# refusal arrives as Status=Timeout. Timeout is classified unmeasurable below,
# and rightly -- a daemon that is up but wedged times out too -- so under the
# old `-TimeoutSec 2` the genuine negative was unreachable IN PRINCIPLE and
# `health-unreachable-after-off` could never have passed honestly.
#
# 5 IS EVIDENCE ABOUT ONE HOST, NOT A GUARANTEE. On a loaded CI runner a
# refusal can take longer, and when it does this probe returns `unmeasurable`
# and the row FAILS. That is the safe direction -- no shutdown is certified
# that nobody measured -- but it is flaky, and it is stated here rather than
# discovered. GAUNTLET_HEALTH_TIMEOUT_SEC is the documented way to raise the
# value on such a runner.
#
# It cannot LOWER it. A value under the floor is refused, reported on stdout,
# and the floor used instead, because the failure mode of a too-small timeout
# is not a slower test but a probe whose negative cannot be reached at all --
# which is this file's entire defect class, reintroduced through a knob.
#
# IT IS BOUNDED ABOVE TOO, and the ceiling is not symmetry for its own sake.
# This probe is not run once: `Stop-Daemon`'s wait runs it up to twenty times,
# so the knob multiplies. GAUNTLET_HEALTH_TIMEOUT_SEC=86400 -- a value nobody
# types on purpose, but exactly the kind of thing a CI environment or a
# developer's exported shell variable carries in -- makes one probe wait a day
# and the stop-wait loop effectively non-terminating. A gauntlet that hangs
# records no row at all, and no row is the one outcome this ledger cannot tell
# from a run that never started. Out of range at EITHER end is a clamp that
# says so on stdout, never a silent one.
function Get-HealthTimeoutSec {
    $floor = 5
    $ceiling = 60
    $raw = "$env:GAUNTLET_HEALTH_TIMEOUT_SEC"
    if (-not $raw) { return $floor }
    $n = 0
    if (-not [int]::TryParse($raw, [ref]$n)) {
        Write-Host "GAUNTLET_HEALTH_TIMEOUT_SEC='$raw' is not an integer; using the ${floor}s floor"
        return $floor
    }
    if ($n -lt $floor) {
        Write-Host ("GAUNTLET_HEALTH_TIMEOUT_SEC=$n is below the ${floor}s floor: a refused loopback " +
                    "connect measured ~2.1 s here, and a timeout shorter than the refusal turns every " +
                    "genuine refusal into an unmeasurable Timeout. Using $floor.")
        return $floor
    }
    if ($n -gt $ceiling) {
        Write-Host ("GAUNTLET_HEALTH_TIMEOUT_SEC=$n is above the ${ceiling}s ceiling: the stop-wait loop " +
                    "runs this probe twenty times, so an unbounded value does not make the run slow, it " +
                    "makes it non-terminating, and a run that never finishes records no row at all. " +
                    "Using $ceiling.")
        return $ceiling
    }
    return $n
}
# Tri-state health probe: reachable / down / unmeasurable.
#
#   A FAILED MEASUREMENT MUST NEVER BE INDISTINGUISHABLE FROM A NEGATIVE ONE.
#
# The two-state form this replaces was:
#
#     catch { return ($null -ne $_.Exception.Response) }
#
# Every exception carrying no Response returned $false -- and $false was also
# what a refused connection returned. A DNS failure, a TLS failure, a timeout, a
# malformed URI, a probe that never left this process: all of them spelled
# "the daemon is down". `health-unreachable-after-off` therefore PASSED when the
# HTTP probe itself broke, certifying a shutdown nobody measured. Same defect as
# the `-ErrorAction SilentlyContinue` port probe below, in a different tool.
#
# ONLY A REFUSED CONNECTION IS `down`. That is the daemon answering the question
# by not being there; everything else is the question not being asked.
#
# THE TABLE BELOW WAS MEASURED ON THE WRONG EDITION. It is correct for Windows
# PowerShell 5.1 and this script is run by first-run-gauntlet.yml with
# `shell: pwsh`, which is PowerShell 7, where Invoke-WebRequest is built on
# HttpClient and raises none of these types. The verdict logic now reads the
# exception SHAPE through Get-WebExceptionShape / Test-ConnectionRefused /
# Test-HttpErrorResponse in lib.ps1, whose header carries the re-measured table
# for both editions. This one is kept because the reasoning under it -- why a
# reset is not a refusal, why ConnectFailure alone is not enough -- is what
# those helpers implement, and it is edition-independent.
#
# MEASURED ON THIS HOST (Windows PowerShell 5.1.26100.9278), because a
# classification built on guessed exception classes measures nothing:
#
#   closed loopback port  -> WebException Status=ConnectFailure (2),
#                            InnerException SocketException/ConnectionRefused
#                            (10061), Response null
#   unresolvable host     -> WebException Status=NameResolutionFailure (1)
#   live server, HTTP 500 -> WebException Status=ProtocolError, Response NON-NULL
#   "notaurl:::"          -> NotSupportedException -- not a WebException at all
#   ""                    -> UriFormatException
#
# A RESET IS NOT A REFUSAL, and the earlier form treated the two as one
# negative. A reset is what a LIVE peer does: a proxy, a firewall, a load
# balancer, or an HTTP service that is up and falling over mid-response. It
# says something was there to slam the door, which is the opposite of the claim
# `health-unreachable-after-off` makes. Measured here, on a TcpListener that
# accepts and closes with LingerState(true, 0) -- a listener that is
# indisputably UP:
#
#   live listener that RSTs -> WebException Status=ReceiveFailure (3),
#                              InnerException System.IO.IOException,
#                              and only INSIDE that, SocketException/
#                              ConnectionReset (10054)
#
# Two things follow. First, `ConnectionReset` never belonged in the negative:
# with it there, a live listener resetting the gauntlet's probe certified that
# the daemon was gone while it was still on the port. Second, the socket error
# is not always $we.InnerException -- here it sits one layer deeper -- so the
# chain is walked. That is for the DETAIL; the verdict still requires
# ConnectFailure, whose inner SocketException is direct.
#
# ConnectFailure alone is not enough either: it also carries NetworkUnreachable
# (10051) and HostUnreachable (10065), which say the path is broken, not that
# the daemon is gone. Only ConnectionRefused (10061) is the door being answered
# with a slam. Everything else -- including a ConnectFailure that carries a
# reset, which a proxy can produce -- is unmeasurable.
#
# The catch is deliberately untyped and classifies on $_.Exception itself: a
# typed `catch [System.Net.WebException]` would not cover the UriFormatException
# and NotSupportedException arms above, and an arm that is not covered here
# falls to the default -- which is unmeasurable, the safe direction.
function Get-HealthReachability([string]$Url) {
    $timeoutSec = Get-HealthTimeoutSec
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $timeoutSec -ErrorAction Stop
        return [pscustomobject]@{ State = "reachable"; Detail = "HTTP $($r.StatusCode) from $Url" }
    } catch {
        # Shape, not type. `$ex -as [System.Net.WebException]` stood here, and
        # under pwsh 7 -- the edition first-run-gauntlet.yml actually runs this
        # script with -- it is $null for EVERY failure, so this probe could
        # never once return "down". See the re-measured table over
        # Get-WebExceptionShape in lib.ps1.
        $ex = $_.Exception
        $shape = Get-WebExceptionShape $ex
        # THE ONLY NEGATIVE: the peer refused. Not a reset, which is a live peer
        # slamming the door and if anything evidence AGAINST a shutdown; not
        # HostNotFound; not a timeout, because a wedged daemon times out too.
        if (Test-ConnectionRefused $shape) {
            return [pscustomobject]@{ State = "down"; Detail = "$($shape.SocketError) from $Url" }
        }
        # An HTTP error still proves something is listening and speaking HTTP,
        # which is the property this probe is for. Kept from the old form, and
        # asked AFTER the refusal so the two cannot overlap.
        if (Test-HttpErrorResponse $shape) {
            return [pscustomobject]@{ State = "reachable"
                Detail = "HTTP error from ${Url} -- an HTTP error still proves reachable ($($shape.Type))" }
        }
        # Nothing was asked of the network at all: a bad URI, a missing
        # assembly, a scriptblock error.
        if ($shape.Type -eq "System.NotSupportedException" -or
            $shape.Type -eq "System.UriFormatException" -or
            $shape.Type -eq "System.ArgumentException") {
            return [pscustomobject]@{ State = "unmeasurable"
                Detail = "the health probe did not reach the network ($($shape.Type): $($ex.Message))" }
        }
        # Everything else, so the reader can see it is a decision and not an
        # oversight: a timeout (TaskCanceledException under pwsh 7, Status=
        # Timeout under 5.1 -- a wedged daemon times out too), a reset or a
        # ConnectionClosed (the measured shape of a LIVE peer), HostNotFound and
        # NameResolutionFailure, TLS and proxy failures, and any refusal whose
        # SocketException was buried rather than direct.
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "the health probe failed without reaching a verdict ($($shape.Type)$(if ($shape.Status) { ", status $($shape.Status)" })$(if ($shape.SocketError) { ", socket $($shape.SocketError) at depth $($shape.SocketDepth)" }): $($ex.Message))" }
    }
}
# Tri-state listener probe: found / none / unmeasurable.
#
# The form this replaces was, in the `port-7878-closed` cleanup check:
#
#     $open = @(Get-NetTCPConnection -LocalPort 7878 -State Listen `
#                 -ErrorAction SilentlyContinue)
#     if ($open.Count -ne 0) { throw ... }
#
# `-ErrorAction SilentlyContinue` turns EVERY failure of the TCP provider into
# an empty array, and an empty array is what a genuinely closed port also
# produces. A CIM server that is down, an unloaded NetTCPIP module, a WMI
# repository hiccup: `$open.Count -eq 0`, no throw, and the row PASSES. The
# gauntlet's last act is to certify that it left no daemon behind on the
# developer's shared production port, and it certified that by failing to look.
# This is verbatim the defect scripts/first-run/port-precheck.sh was written to
# close at the START of a run -- the same cmdlet, the same flag, quoted in that
# script's header -- reintroduced at the END of the same run.
#
# THE TABLE IS THE PRIMARY READ, like the netstat parse in
# scripts/lib/host-process.sh: the whole listener table is fetched and then
# asked about the port, rather than the provider being asked about the port
# directly. That is what makes it self-validating -- a table that came back is
# evidence the provider answered, so its shape can be checked before it is
# believed, and a per-port query has no shape to check.
#
# MEASURED ON THIS HOST: `Get-NetTCPConnection -State Listen` returns 39 rows
# (29 distinct ports, 135 and 445 among them) and 217 rows unfiltered. Asking
# instead about a single free port throws CimJobException with
# FullyQualifiedErrorId `CmdletizationQuery_NotFound,Get-NetTCPConnection` --
# so on the per-port form even the negative arrives as an exception, and telling
# it from a provider failure means parsing an error id. That is why the per-port
# form is the WITNESS below and not the primary read: alone, "an error came
# back" cannot be told from "the provider broke", which is the whole defect.
# Together the two are a measurement neither is on its own.
#
# THE RESIDUAL THAT USED TO BE STATED HERE -- a provider that returns a
# well-formed but INCOMPLETE table, since there is no socket that must appear in
# every listener table the way pid 4 must appear in every process table -- is
# put to the witness below. It has to be: the shape of the table says the
# PROVIDER answered, but 38 rows about other people's sockets are not evidence
# about ours, so a table-only negative is ratified by a second read.
#
# The other stated cost stands: a host with NO listening TCP socket at all
# answers "unmeasurable" rather than "free", because the empty-table case is
# indistinguishable from the provider failing. That is the safe direction, and
# on Windows it is not reachable in practice -- the RPC endpoint mapper and
# svchost listen before a login shell exists.
#
# THE WITNESS, AND WHY IT IS NOT Get-NetTCPConnection ANY MORE.
#
# The previous witness asked `Get-NetTCPConnection -LocalPort $Port` -- the same
# cmdlet, the same NetTCPIP cmdletization, the same CIM session, the same
# MSFT_NetTCPConnection provider as the table read it was ratifying. Round-3
# review, verbatim: "the port and process witnesses query THE SAME PROVIDER
# whose failure they purport to detect." That is exactly right, and it is worth
# being precise about what it could and could not do. It DID catch a targeted
# read that disagreed with the table (a per-connection race, a filtered view)
# and a targeted read that failed with an unexpected error id. It could NOT
# catch the case it was written for: one provider-level fault -- a stale
# cmdletization cache, a CIM session that has lost half the instance set, a WMI
# repository serving a plausible subset -- omits our socket from the table AND
# answers CmdletizationQuery_NotFound to the targeted query. Both reads agree,
# both are wrong, and `port-7878-closed` passes over a live daemon.
#
# So the witness is now `netstat -ano`, which is a DIFFERENT PROVIDER.
#
# HOW MUCH OF THAT IS MEASURED, since the distinction matters: what this run
# established is that netstat is a separate PROCESS whose output does not come
# through the cmdletization/CIM path at all, and that the two disagree in shape
# (one prints text rows, the other returns instances). The next sentence's
# account of netstat's internals -- iphlpapi's GetExtendedTcpTable -- is
# documented behaviour taken on trust, NOT something measured here; nothing in
# this repo inspects netstat's imports. The independence claim rests on the
# separate process and the separate transport, which are observable; the API
# name is offered as explanation, not as evidence.
#
# a native executable calling GetExtendedTcpTable in iphlpapi.dll, with no CIM session,
# no WMI repository and no cmdletization layer anywhere in it. A fault in any
# of those breaks one read and not the other, which is what makes disagreement
# between them information.
#
# WHAT IT STILL CANNOT WITNESS AGAINST, stated rather than implied: both
# providers ultimately enumerate the SAME kernel TCP table. A kernel- or
# driver-level omission -- a filter driver hiding a socket, a rootkit -- hides
# it from both. This witnesses against PROVIDER failure, not against the kernel
# lying, and no read available from user mode would do better.
#
# The parse is the one scripts/lib/host-process.sh already argues for, ported:
#
#   EVERY non-blank line must BE a row -- TCP with five fields, UDP with four,
#       a numeric pid last -- except netstat's own preamble before the first
#       row (the banner and the column header, both localised, so they can only
#       be counted). A status-0 `WARNING: partial results` merged beside real
#       rows begins with neither protocol name, so a parse that only inspected
#       lines already starting TCP/UDP never looked at it and reported a
#       truncated remainder as a measured negative.
#   ANY STDERR IS A REFUSAL, and the two streams are kept apart to make that
#       readable. Invoke-Native no longer flattens an ErrorRecord into a line
#       of output, so `$r.ErrorCount` is a fact about which stream a line came
#       from. This is not a refinement of the rule below, it is what makes the
#       rule below possible: a preamble that must tolerate two unmatchable
#       localised lines cannot also be the filter that catches a diagnostic
#       standing among them.
#   THE PREAMBLE IS A SHAPE, NOT A BUDGET. `netstat -ano` prints blank, banner,
#       blank, header, rows. The header is therefore the LAST non-blank line
#       before the first row -- the line whose columns this parse is about --
#       and at most ONE other non-blank line may precede it, the banner, which
#       must be SEPARATED FROM THE HEADER BY A BLANK LINE. A bare count of two
#       was the defect: `WARNING: provider returned partial results` directly
#       on top of the header is two non-blank lines too, fits the budget
#       exactly, and a reviewer used it to get a port reported measured closed.
#       Neither line can be matched by text; the blank line between them is not
#       localised. A preamble with no non-blank line at all is also refused --
#       nothing there carries the columns.
#       THE RESIDUAL OF THIS RULE, stated: a stdout warning that mimics the
#       banner's exact shape -- one non-blank line, a blank line, then the
#       header -- is indistinguishable from the banner and is admitted. In that
#       position the banner IS a line of localised prose, so nothing in the
#       stream separates them. What closed the reviewer's case is that a
#       diagnostic must now be POSITIONED like a banner to survive at all, and
#       that netstat's diagnostics go to stderr, which is refused outright.
#   LISTENING IS NOT A KEY. netstat's State column is localised -- German
#       Windows prints ABHOEREN -- so the structural rule is used instead: a
#       listening socket is the TCP row with a WILDCARD foreign address, because
#       a connected one names its peer.
#   A UDP ROW IS THE END WITNESS. `netstat -ano` prints the whole TCP table and
#       then the whole UDP table, so a UDP row is evidence the stream got past
#       every TCP row there was; without one the table may have stopped early,
#       and every remaining row still validates. The ordering that inference
#       rests on is checked rather than assumed: a TCP row after a UDP row is a
#       stream this argument does not apply to.
#   At least one listening row must exist somewhere in the table, so a netstat
#       whose columns have moved is unmeasurable rather than empty. Same stated
#       cost as above, same reason it is unreachable on Windows.
#   THE RESIDUAL, stated: a hole in the MIDDLE of the TCP section is invisible
#       here -- the UDP rows after it still arrive -- and so is a short TCP
#       section, because netstat has no row that must appear in every table.
function Get-PortListenerWitness([int]$Port) {
    $r = Invoke-Native "netstat.exe" @("-ano")
    if (-not $r.Ran) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the independent netstat read could not be run ($($r.Output))" }
    }
    if ($r.ExitCode -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "netstat -ano exited $($r.ExitCode): $($r.Output -replace "`r?`n", ' ')" }
    }
    # Any stderr at all, whatever the exit status said. A zero exit alongside a
    # complaint is exactly the shape that made this branch report a port free.
    if ($r.ErrorCount -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "netstat -ano exited 0 but wrote $($r.ErrorCount) line(s) to stderr, so this read is not a measurement: $($r.ErrorText -replace "`r?`n", ' ')" }
    }
    $listening = 0
    $rows = 0
    $preamble = 0
    $notARow = 0
    $udp = 0
    $tcpAfterUdp = 0
    # Where a blank line falls is evidence, so it is recorded rather than
    # skipped: `blankSeen` says a blank line has arrived since the last
    # non-blank pre-row line, and `preambleSep` freezes that answer for the
    # gap between the banner and the header. After the first row it is
    # irrelevant -- the real table has blank lines among its rows.
    $blankSeen = 0
    $preambleSep = 0
    $hits = New-Object System.Collections.Generic.List[string]
    foreach ($line in @($r.Output -split "`r?`n")) {
        $f = @(($line -split '\s+') | Where-Object { $_ })
        if ($f.Count -eq 0) {
            if ($rows -eq 0) { $blankSeen = 1 }
            continue
        }
        $wellFormed = (($f[0] -eq "TCP" -and $f.Count -eq 5 -and $f[4] -match '^\d+$') -or
                       ($f[0] -eq "UDP" -and $f.Count -eq 4 -and $f[3] -match '^\d+$'))
        if (-not $wellFormed) {
            if ($rows -ne 0) { $notARow++ }
            else {
                $preamble++
                if ($preamble -eq 2) { $preambleSep = $blankSeen }
                $blankSeen = 0
            }
            continue
        }
        $rows++
        if ($f[0] -eq "UDP") { $udp++ } elseif ($udp -ne 0) { $tcpAfterUdp++ }
        if ($f[0] -ne "TCP") { continue }
        if ($f[2] -ne "0.0.0.0:0" -and $f[2] -ne "[::]:0" -and $f[2] -ne "*:*") { continue }
        $listening++
        # Anchored, so :7878 does not match :78780 or a foreign address.
        if ($f[1] -match (':' + $Port + '$')) { $hits.Add($f[4]) }
    }
    if ($notARow -ne 0 -or $preamble -gt 2) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "$($preamble + $notARow) of netstat's lines are not rows this parse understands ($preamble before the first row, where netstat's own preamble is two, and $notARow after it); a table it cannot read cannot witness for $Port" }
    }
    # The SHAPE rule, kept separate from the count above so each says which
    # property failed. Two non-blank pre-row lines with no blank line between
    # them is a diagnostic standing where the banner stands, not a banner; zero
    # of them is a table with no header, so nothing there carries the columns.
    if ($preamble -lt 1 -or ($preamble -eq 2 -and $preambleSep -eq 0)) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "netstat's preamble is not the shape netstat prints ($preamble non-blank line(s) before the first row, blank line between them: $preambleSep; expected blank, banner, blank, header); the last one before the rows must be the header and only a blank-separated banner may precede it, so this table cannot witness for $Port" }
    }
    if ($tcpAfterUdp -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "netstat printed $tcpAfterUdp TCP rows after a UDP row, so the sections are interleaved and a UDP row no longer witnesses that the TCP section ended; the table cannot witness for $Port" }
    }
    if ($udp -lt 1) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "netstat printed no UDP row, so nothing witnesses that the TCP section ENDED rather than stopped early; a table that may be truncated cannot witness for $Port" }
    }
    if ($listening -lt 1) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "netstat returned no listening TCP row at all; a Windows host always has some, so this is a failed read, not an idle machine" }
    }
    if ($hits.Count -ne 0) {
        return [pscustomobject]@{ State = "found"; OwningProcess = $hits[0]
            Detail = "netstat, an independent provider, reports pid $($hits[0]) listening on $Port" }
    }
    return [pscustomobject]@{ State = "none"; OwningProcess = $null
        Detail = "netstat, an independent provider, read $listening listening TCP sockets and none on $Port" }
}
function Get-PortListenerState([int]$Port) {
    try {
        $table = @(Get-NetTCPConnection -State Listen -ErrorAction Stop)
    } catch {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the TCP listener table could not be read ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }
    # A table with nothing in it is not a measurement of an idle machine; it is
    # a provider that answered without answering.
    if ($table.Count -lt 1) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the TCP listener table came back empty; a Windows host always has listening sockets, so this is a failed read, not an idle machine" }
    }
    # Every row's port is parsed HERE, inside a handler, and the parse is
    # TryParse rather than a cast.
    #
    # The form this replaces was `Where-Object { $null -eq $_.LocalPort -or
    # [int]$_.LocalPort -le 0 }`, and its casts sat OUTSIDE the try that read
    # the table. It coped with a row whose LocalPort was NULL and not with one
    # whose LocalPort was UNPARSEABLE -- "unknown", an empty string, an object
    # -- and `[int]"unknown"` does not return anything, it THROWS. The throw
    # left this function without returning any of its three states, so the one
    # value the caller branches on never arrived. A probe that can die instead
    # of answering has not left the defect class; it has moved it one frame up
    # the stack. An unparseable row is `unmeasurable`, exactly like a null one.
    $parsed = New-Object System.Collections.Generic.List[object]
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
    }
    if ($unusable -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "$unusable of $($table.Count) listener rows carry no usable LocalPort; the table cannot be asked about $Port" }
    }
    $hit = @($parsed | Where-Object { $_.Port -eq $Port })
    if ($hit.Count -ne 0) {
        return [pscustomobject]@{ State = "found"; OwningProcess = $hit[0].OwningProcess
            Detail = "pid $($hit[0].OwningProcess) is listening on $Port" }
    }
    # The table has no row for $Port. Every row it DOES have is about someone
    # else's socket, so on its own that is a claim the evidence does not cover.
    # Put it to the independent provider.
    $witness = Get-PortListenerWitness $Port
    if ($witness.State -eq "found") {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $witness.OwningProcess
            Detail = "the listener table ($($table.Count) rows) has no row for $Port, but $($witness.Detail); the two reads contradict each other, so neither is a measurement" }
    }
    if ($witness.State -ne "none") {
        return [pscustomobject]@{ State = "unmeasurable"; OwningProcess = $null
            Detail = "the listener table ($($table.Count) rows) has no row for $Port, but that absence could not be corroborated: $($witness.Detail)" }
    }
    return [pscustomobject]@{ State = "none"; OwningProcess = $null
        Detail = "measured closed: $($table.Count) listening sockets read, none on $Port, and $($witness.Detail)" }
}
# --- WHAT THIS RUN OWNS ----------------------------------------------------
#
# The form this replaces was:
#
#     & schtasks.exe /end /tn WenlanServer 2>&1 | Out-Null
#     Get-Process -Name wenlan-server -ErrorAction SilentlyContinue |
#         Stop-Process -Force -ErrorAction SilentlyContinue | Out-Null
#
# `-Name wenlan-server` SELECTS EVERY PROCESS WITH THAT NAME and the pipeline
# force-killed all of them. A developer's production daemon, another worktree's
# dev daemon, a hand-started wenlan-server: this run started none of them.
# `/tn WenlanServer` is the same defect in the task namespace, and the cleanup's
# `schtasks /delete /tn WenlanServer /f` then removed the developer's
# registration for good. That is a defect in the SHIPPING gauntlet, not in a
# harness around it, and it is the reason an unstubbed control run could stop a
# real daemon.
#
# Ownership is measured now, before anything is started, and it is two facts:
#
#   THE IMAGE. A wenlan-server running from anywhere but this run's own
#       $InstallDir\wenlan-server.exe belongs to somebody else, whatever it is
#       called. Same reasoning as kill_by_image_path in
#       scripts/lib/host-process.sh: every wenlan-server on the machine shares
#       one image NAME, so the name cannot tell them apart and the PATH can.
#
#   AND THE PID WAS NOT ALREADY RUNNING. The image test alone is not enough in
#       THIS channel, and that is the trap worth naming. The zip flow installs
#       to the DOCUMENTED per-user location, %LOCALAPPDATA%\Programs\wenlan --
#       which is exactly where a developer's production install lives, so their
#       daemon's image path is identical to ours. The one thing that separates
#       them is that theirs was already running when this script started.
#
#       ROUND 6 NAMES WHAT THAT PAIR IS AND IS NOT. "Our image, and a pid that
#       was not in the startup snapshot" is a CORRELATION with "this run
#       started it", not the thing itself, and it is wrong in both directions:
#
#         Not ours, and killed. Anything that starts a server from that same
#         path AFTER the snapshot gets a pid the snapshot cannot contain. The
#         developer's own daemon restarting mid-run does it -- their scheduled
#         task firing, `wenlan` autostarting on a CLI call in another terminal,
#         or them simply restarting it -- and so does a second gauntlet run.
#         All of them look exactly like this run's own child.
#
#         Ours, and spared. Windows reuses pids, and the snapshot is a list of
#         NUMBERS with no birth time in it. A pre-existing server that exits
#         during the run frees its pid; if this run's own server is later given
#         that number, it matches $PreexistingServerPids and is left running --
#         which is the direction that makes `daemon-stopped-at-cleanup` and
#         `port-7878-closed` disagree with each other.
#
#       The safe direction is the second one -- sparing something is
#       recoverable, killing the developer's daemon is not -- and that is the
#       direction the unmeasurables already resolve to. Closing the first would
#       take a fact the pid pair does not carry: the process's own start time
#       compared against this run's, or a parent chain back to this script.
#       Neither is read here, so this is a residual, not a solved problem.
#
#   THE TASK. `WenlanServer` is named by the PRODUCT -- `wenlan background on`
#       registers that spelling -- so this run cannot scope the name to itself.
#       What it can establish is that the name was FREE before it started, which
#       is the only available evidence that the registration now present is the
#       one this run made. A task that already existed, or a presence that could
#       not be measured, is NOT OURS: not ended, not deleted, and said out loud
#       rather than done quietly.
#
# Every one of these is tri-state, and every unmeasurable resolves to NOT OURS.
# The cost of that direction is a FAIL row. The cost of the other direction is
# the developer's daemon.

# Every wenlan-server this user can see, with the image each is running from.
# Tri-state: read / none / unmeasurable.
function Get-ServerProcessInventory {
    $name = "wenlan-server"
    try {
        $found = @(Get-Process -Name $name -ErrorAction Stop)
    } catch {
        $fqid = "$($_.FullyQualifiedErrorId)"
        $typeName = if ($null -ne $_.Exception) { $_.Exception.GetType().FullName } else { "" }
        # Type AND error id, compared as strings. A type LITERAL is resolved at
        # run time and throws when its assembly is not loaded -- inside the very
        # catch whose job is to make a failure legible.
        if ($typeName -ne "Microsoft.PowerShell.Commands.ProcessCommandException" -or
            $fqid -notlike "NoProcessFoundForGivenName,*") {
            return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
                Detail = "the $name process read failed without answering ($typeName, id '$fqid': $($_.Exception.Message))" }
        }
        # "There is no such process" is a NEGATIVE, and a negative is not taken
        # from the provider whose failure would produce the same answer.
        $w = Get-CimProcessWitness -Name $name
        if (-not $w.Ok) {
            return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
                Detail = "Get-Process reports no $name, but that absence is not ratified: $($w.Detail)" }
        }
        return [pscustomobject]@{ State = "none"; Processes = @()
            Detail = "no $name process ($fqid; $($w.Detail))" }
    }
    if ($found.Count -eq 0) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "Get-Process -Name $name returned no error and no process; here a name that matches nothing throws, so silence is not absence" }
    }
    # A SUCCESSFUL read is corroborated too, not just a failed one. The pids
    # this returns become the snapshot that decides, later, which daemons this
    # run is allowed to kill -- so a set that is merely SHORT is the dangerous
    # answer, not the loud one. Set equality, in both directions: a pid WMI has
    # and Get-Process does not is the omission that gets a stranger's daemon
    # killed; a pid Get-Process has and WMI does not is the same disagreement
    # with the providers swapped, and neither is a measurement.
    $cs = Get-CimProcessSet $name
    if (-not $cs.Ok) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "Get-Process found $($found.Count) $name process(es), but that set is not corroborated: $($cs.Detail)" }
    }
    $gp = @(@($found | ForEach-Object { [int]$_.Id }) | Sort-Object)
    $missing = @($cs.Pids | Where-Object { $gp -notcontains $_ })
    $extra = @($gp | Where-Object { $cs.Pids -notcontains $_ })
    if ($missing.Count -ne 0 -or $extra.Count -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = ("the two providers do not agree on which $name processes exist -- Get-Process has [" + ($gp -join ",") +
                      "], WMI has [" + (@($cs.Pids) -join ",") + "]" +
                      $(if ($missing.Count) { "; WMI sees pid(s) " + ($missing -join ",") + " that Get-Process omitted, and a pid missing from this snapshot would later be mistaken for one this run started" } else { "" }) +
                      $(if ($extra.Count) { "; Get-Process sees pid(s) " + ($extra -join ",") + " that WMI does not" } else { "" })) }
    }
    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($p in $found) {
        $path = ""
        $why = ""
        try { $path = "$($p.Path)" } catch { $why = "$($_.Exception.GetType().FullName): $($_.Exception.Message)" }
        if (-not $path -and -not $why) { $why = "the process exposes no image path" }
        $rows.Add([pscustomobject]@{ Id = $p.Id; Path = $path; Why = $why })
    }
    # .ToArray(), NOT @($rows). MEASURED on Windows PowerShell 5.1.26100.9278:
    #
    #   $l = New-Object System.Collections.Generic.List[object]; $l.Add(1)
    #   @($l)   ->  THROWS System.ArgumentException: Argument types do not match
    #
    # and a List[string] does not. So the array subexpression operator -- the
    # idiom this whole file uses to force a scalar into an array -- is the one
    # thing that must not be applied to the accumulator, and it fails by
    # THROWING out of the probe rather than by returning a wrong state. That is
    # the round-3 defect exactly: a tri-state that dies inside the callee before
    # any caller can branch on it, reintroduced by the fix for it. The
    # negative-control harness caught this; nothing else would have, because
    # this arm only runs when a wenlan-server actually exists.
    return [pscustomobject]@{ State = "read"; Processes = $rows.ToArray()
        Detail = ("[corroborated: $($cs.Detail)] $($found.Count) $name process(es): " +
                  (($rows | ForEach-Object { if ($_.Path) { "pid $($_.Id) -> $($_.Path)" } else { "pid $($_.Id) -> IMAGE UNREADABLE ($($_.Why))" } }) -join "; ")) }
}

# The subset of the above that belongs to THIS run.
# Tri-state: owned / none / unmeasurable. `none` means "none of ours is
# running", which is the claim `daemon-stopped-*` needs; it says nothing about
# anyone else's daemon and is not allowed to.
function Get-OwnedServerProcesses {
    if ($null -eq $script:PreexistingServerPids) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "the wenlan-server processes already running when this script started were never measured, so no process can be shown to belong to this run and none may be killed" }
    }
    $inv = Get-ServerProcessInventory
    if ($inv.State -eq "unmeasurable") {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @(); Detail = $inv.Detail }
    }
    if ($inv.State -eq "none") {
        return [pscustomobject]@{ State = "none"; Processes = @()
            Detail = "none of this run's wenlan-server processes is running: $($inv.Detail)" }
    }
    $unreadable = @($inv.Processes | Where-Object { -not $_.Path })
    if ($unreadable.Count -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "$($unreadable.Count) of $($inv.Processes.Count) wenlan-server processes have an unreadable image, and a process this run cannot identify is neither proven its own nor proven someone else's: $($inv.Detail)" }
    }
    $ours = @($inv.Processes | Where-Object {
        [string]::Equals($_.Path, $script:OwnedServerImage, [System.StringComparison]::OrdinalIgnoreCase) -and
        ($script:PreexistingServerPids -notcontains $_.Id) })
    $preList = if (@($script:PreexistingServerPids).Count) { @($script:PreexistingServerPids) -join "," } else { "none" }
    if ($ours.Count -eq 0) {
        return [pscustomobject]@{ State = "none"; Processes = @()
            Detail = "none of this run's wenlan-server processes is running (owned image '$($script:OwnedServerImage)'; pids already running at start: $preList); $($inv.Detail)" }
    }
    return [pscustomobject]@{ State = "owned"; Processes = @($ours)
        Detail = ("this run owns " + (($ours | ForEach-Object { "pid $($_.Id)" }) -join ", ") +
                  " (image '$($script:OwnedServerImage)'; pids already running at start: $preList); $($inv.Detail)") }
}

# Kill ONE process, by pid, and only while it is still the image the caller
# identified. Ported from kill_by_image_path in scripts/lib/host-process.sh,
# including the reason the handle comes first:
#
# checking the path and then killing by pid leaves a reuse window open -- the
# daemon can exit and Windows can hand its pid to one of the neighbours this
# whole exercise exists to protect, between the check and the kill. Windows
# keeps a pid reserved while any handle to it is open, so once $p.Handle
# materialises, the path read below and the process killed after it are the
# same one. A path that cannot be read is not a match, so an identity this
# cannot prove kills nothing.
#
# Returns killed / gone / refused / unmeasurable, and NEVER throws: the verdict
# on a stop is the liveness poll afterwards, never this receipt, but a refusal
# has to be distinguishable from a success in the diagnostics.
function Stop-OwnedServerProcess {
    param([Parameter(Mandatory)][int]$ProcessId, [Parameter(Mandatory)][string]$ImagePath)
    $want = ($ImagePath -replace "/", "\")
    try { $p = [System.Diagnostics.Process]::GetProcessById($ProcessId) }
    catch { return [pscustomobject]@{ State = "gone"; Detail = "pid $ProcessId is not a running process ($($_.Exception.Message))" } }
    try { $null = $p.Handle }
    catch { return [pscustomobject]@{ State = "unmeasurable"; Detail = "pid $ProcessId could not be opened, so its pid cannot be pinned and its identity cannot be trusted ($($_.Exception.Message))" } }
    $got = ""
    try { $got = "$($p.MainModule.FileName)" }
    catch { return [pscustomobject]@{ State = "unmeasurable"; Detail = "pid $ProcessId has an unreadable image, so this run cannot show it is its own ($($_.Exception.Message))" } }
    if (-not [string]::Equals($got, $want, [System.StringComparison]::OrdinalIgnoreCase)) {
        return [pscustomobject]@{ State = "refused"; Detail = "pid $ProcessId is running '$got', not '$want'; nothing killed" }
    }
    try { $p.Kill() }
    catch { return [pscustomobject]@{ State = "unmeasurable"; Detail = "the kill of pid $ProcessId threw ($($_.Exception.Message))" } }
    return [pscustomobject]@{ State = "killed"; Detail = "pid $ProcessId ('$got') was killed" }
}

# Is a scheduled task registered? Tri-state: present / absent / unmeasurable.
#
# `schtasks /query /tn <name>` cannot answer this. It exits 1 both when the task
# does not exist and when the query itself failed, and the only thing separating
# the two is an ERROR line that is LOCALISED. Reading that status as "absent" is
# precisely how this run would come to believe it had registered a task that was
# already there -- and then delete the developer's.
#
# So the WHOLE task table is the read, for the same reason the whole listener
# table is: a table that came back is evidence the service answered, and it can
# be checked for shape before it is believed. Every Windows install carries
# tasks under \Microsoft\Windows\ from setup onwards; that is this table's pid 4,
# and the folder path is a literal, not a localised string.
# The INDEPENDENT read of the same question, and the reason the row floor below
# is no longer asked to do a job it cannot do.
#
# PROVIDER INDEPENDENCE: `schtasks.exe` is a native binary driving the Task
# Scheduler COM interface; `Get-ScheduledTask` is a CIM-cmdletized function over
# the `MSFT_ScheduledTask` class in root/Microsoft/Windows/TaskScheduler,
# serviced by winmgmt. Different process, different transport, different failure
# modes. MEASURED on this host: 286 schtasks CSV rows against 207
# Get-ScheduledTask objects -- they do not even agree on cardinality, because
# schtasks prints one row per TRIGGER and CIM returns one object per TASK.
#
# WHAT IT CANNOT WITNESS AGAINST, stated rather than implied: both ultimately
# consult the same Task Scheduler service. A service-level omission hides a task
# from both. This witnesses against one PROVIDER failing or truncating, which is
# the round-4 finding, not against the scheduler itself lying.
#
# NOTE the auto-import hazard: `Get-ScheduledTask` is CommandType=Function from
# module ScheduledTasks (measured), not a Cmdlet -- so a test double for it is a
# function that a module auto-import can silently REPLACE. The harness guard
# that requires each stub to be a Function with an EMPTY ModuleName is what
# catches that; it is the same NetTCPIP hazard, and it applies here too.
function Get-ScheduledTaskWitness([string]$Name) {
    try {
        $all = @(Get-ScheduledTask -ErrorAction Stop)
    } catch {
        return [pscustomobject]@{ Ok = $false; Present = $false; Count = 0; Names = @()
            Detail = "the independent MSFT_ScheduledTask enumeration could not be read ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }
    if ($all.Count -lt 10) {
        return [pscustomobject]@{ Ok = $false; Present = $false; Count = $all.Count; Names = @()
            Detail = "the independent task enumeration returned only $($all.Count) tasks; a Windows install has far more, so it is a failed read" }
    }
    if (-not @($all | Where-Object { "$($_.TaskPath)" -like '\Microsoft\Windows\*' }).Count) {
        return [pscustomobject]@{ Ok = $false; Present = $false; Count = $all.Count; Names = @()
            Detail = "the independent task enumeration ($($all.Count) tasks) contains no \Microsoft\Windows\ task; it is not a whole table" }
    }
    # WHAT THIS WITNESS STILL CANNOT WITNESS AGAINST, stated rather than left
    # to be discovered: its NAMES are what make the caller's completeness test
    # possible, and nothing here proves the CIM enumeration is itself complete.
    # A truncated MSFT_ScheduledTask read that still carries ten tasks and a
    # \Microsoft\Windows\ path would report a short name list, and a short list
    # makes the caller's "every one of these appears over there" test easier to
    # pass, not harder. What narrows it is that the two providers must also
    # AGREE about $Name: for a truncated schtasks prefix to be accepted, the CIM
    # read would have to be cut in the same place, losing the same task. That is
    # a real residual and it is smaller than the one it replaced -- a row floor
    # could be satisfied by any prefix at all -- but it is not zero.
    #
    # TaskPath already carries its trailing separator, so a root task is
    # '\' + 'WenlanServer'. Measured: TaskPath='\', TaskName='Adobe Uninstaller'.
    $names = @($all | ForEach-Object { "$($_.TaskPath)$($_.TaskName)" })
    $hit = @($all | Where-Object { "$($_.TaskPath)$($_.TaskName)" -eq ('\' + $Name) })
    return [pscustomobject]@{ Ok = $true; Present = ($hit.Count -ne 0); Count = $all.Count; Names = $names
        Detail = "the independent MSFT_ScheduledTask enumeration read $($all.Count) tasks and $(if ($hit.Count) { "CONTAINS $Name" } else { "does not contain $Name" })" }
}

function Get-TaskPresence([string]$Name) {
    $r = Invoke-Native "schtasks.exe" @("/query", "/fo", "CSV", "/nh")
    if (-not $r.Ran) {
        return [pscustomobject]@{ State = "unmeasurable"; Detail = "the scheduled-task table could not be read ($($r.Output))" }
    }
    if ($r.ExitCode -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "schtasks /query exited $($r.ExitCode), so the task table was not read: $($r.Output -replace "`r?`n", ' ')" }
    }
    $rows = @(($r.Output -split "`r?`n") | Where-Object { $_.Trim() })
    # EVERY ROW, not a sample. `/fo CSV /nh` emits exactly three quoted fields
    # per row and the first begins with the path separator; measured, all 286
    # rows on this host match. A table cut off mid-record leaves a final row
    # that fails this, which is the shape a truncated prefix actually has --
    # and a prefix that reported WenlanServer absent would otherwise authorize
    # ending and deleting a task that was merely below the cut.
    $shape = '^"\\[^"]*","[^"]*","[^"]*"\s*$'
    $malformed = @($rows | Where-Object { $_ -notmatch $shape })
    if ($malformed.Count -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "$($malformed.Count) of $($rows.Count) scheduled-task rows are not well-formed CSV records (first: '$($malformed[0])'); a table this parse cannot read whole cannot answer about $Name" }
    }
    # PROVENANCE, and labelled as such. A \Microsoft\Windows\ row proves the
    # output came from a real task table. It does NOT prove the table is
    # complete -- a truncated prefix has plenty of them (measured: 246 of 286
    # rows here). Completeness is the independent count below, because that is
    # the only thing a prefix cannot fake.
    if (-not @($rows | Where-Object { $_ -like '*\Microsoft\Windows\*' }).Count) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "the scheduled-task table ($($rows.Count) rows) has no \Microsoft\Windows\ task in it; it did not come from a real task table" }
    }
    $w = Get-ScheduledTaskWitness $Name
    if (-not $w.Ok) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "the schtasks table cannot be corroborated: $($w.Detail)" }
    }
    # THE COMPLETENESS TEST, and the row FLOOR it replaces was not one.
    #
    # Round-5 review is right: `$rows.Count -lt $w.Count` is a NECESSARY
    # condition and nothing more. schtasks prints one row per TRIGGER and CIM
    # one object per TASK, so the two counts are not comparable in the first
    # place -- with CIM at 207 and a whole table of 286 trigger rows, a
    # well-formed 220-row PREFIX satisfies `220 -ge 207` while dropping 66 rows,
    # and a name below the cut reads as absent from a table nobody established
    # was whole. Two truncated reads that agree on an absence then authorise
    # registering over a task that is really there.
    #
    # So the comparison is between SETS OF NAMES, which is what completeness
    # actually means: every task the independent enumeration lists must appear
    # among the names schtasks printed. A prefix cannot satisfy that without
    # containing everything the other provider saw.
    #
    # MEASURED ON THIS HOST: schtasks /query /fo CSV /nh printed 286 rows naming
    # 207 distinct tasks; Get-ScheduledTask returned 207; the two name sets are
    # equal, with nothing on either side the other lacks.
    #
    # THE RESIDUAL, and it is not small, so it is stated rather than left to be
    # inferred from the word "completeness". This test is INCLUSION, and it is
    # exactly as complete as the witness it includes AGAINST. Every task CIM
    # lists must appear among schtasks' names -- so a schtasks table missing a
    # task CIM saw is caught, which is the truncated prefix this replaced a row
    # floor for. What it cannot see is a task NEITHER provider lists. Both are
    # asking the Task Scheduler service, so a folder this user may not
    # enumerate, or a scheduler that quietly drops a subtree, is absent from
    # both name sets, the sets agree, and the table passes as whole. The
    # measurement above says these two providers agree ON THIS HOST; it does
    # not say either of them saw every registration on it. That is the same
    # shape as the netstat residual in scripts/AGENTS.md -- no row must appear
    # in every table, so no table can be proven complete from inside itself --
    # and the honest form of the claim is: schtasks' table is not a truncated
    # prefix OF WHAT GET-SCHEDULEDTASK CAN SEE.
    #
    # $Name ITSELF IS EXCLUDED from this comparison, deliberately. Whether the
    # target is present is the question this function exists to answer, and it
    # is adjudicated two statements below by the two providers having to agree
    # about it -- which produces a message about the disagreement rather than
    # about the table. Including it here would answer the same question in a
    # worse place; excluding it loses nothing, because a target CIM sees and
    # schtasks does not is a disagreement either way.
    $rowNames = New-Object System.Collections.Generic.HashSet[string] ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($row in $rows) {
        # The shape test above already guaranteed `"\...","...","..."`, so the
        # first field ends at the first `","` and contains no quote of its own.
        # Ordinal, because a culture-sensitive IndexOf is a comparison nobody
        # here asked for.
        $end = $row.IndexOf('","', [System.StringComparison]::Ordinal)
        if ($end -gt 1) { $null = $rowNames.Add($row.Substring(1, $end - 1)) }
    }
    $target = '\' + $Name
    $missing = @($w.Names | Where-Object {
        -not [string]::Equals($_, $target, [System.StringComparison]::OrdinalIgnoreCase) -and
        -not $rowNames.Contains($_) })
    if ($missing.Count -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "schtasks printed $($rows.Count) rows naming $($rowNames.Count) tasks, but $($missing.Count) of the $($w.Count) tasks the independent enumeration lists are not among them (first: '$($missing[0])'); the table is a truncated prefix, not the whole table" }
    }
    $want = '"\' + $Name + '"'
    # OrdinalIgnoreCase, because every other name test in this file is folded
    # and PowerShell resolves task names case-insensitively: a row beginning
    # `"\WENLANSERVER",` is the task this run is asking about. The bare
    # .StartsWith was the one unfolded comparison left, and it made that row
    # invisible on the schtasks side.
    $mine = @($rows | Where-Object { $_.StartsWith($want + ",", [System.StringComparison]::OrdinalIgnoreCase) -or $_.Trim() -ieq $want })
    $present = ($mine.Count -ne 0)
    # Two independent providers, one question. Disagreement is not a casting
    # vote for whichever answer is convenient; it is the absence of a
    # measurement.
    if ($present -ne $w.Present) {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "schtasks says $Name is $(if ($present) { 'present' } else { 'absent' }) and the independent enumeration says $(if ($w.Present) { 'present' } else { 'absent' }); two providers contradict each other, so neither is a measurement" }
    }
    if ($present) {
        return [pscustomobject]@{ State = "present"
            Detail = "$Name is already registered ($($rows.Count) rows read): $($mine[0]); $($w.Detail)" }
    }
    return [pscustomobject]@{ State = "absent"
        Detail = "$Name is not registered ($($rows.Count) rows read, none of them it); $($w.Detail)" }
}

# RETURNS a tri-state about the PROCESS -- stopped / alive / unmeasurable.
#
# It used to return the HEALTH probe's tri-state, and round-3 review named that
# exactly right: "the returned type is a reachability tri-state, not a
# process-stop tri-state". `daemon-stopped-before-recovery` and
# `daemon-stopped-at-cleanup` are claims about a PROCESS, and a refused connect
# is not one. It proves that THIS connection to THIS endpoint was refused. It
# does not prove the process exited, does not identify a process, does not
# exclude a daemon that survived and merely unbound the port, and does not
# exclude a local firewall rejecting the connect. Under the old form a failed
# Stop-Process, or a daemon that closed 7878 and stayed up, produced
# ConnectionRefused on the next probe and both rows PASSED -- after which
# cleanup deleted the task, the installation and the data directory out from
# under a live process.
#
# So the verdict is now the OWNED-PROCESS poll. Reachability is still measured
# and still reported, because it is genuinely useful evidence, but it is
# reported as its own separate fact and it decides nothing here.
# `health-unreachable-after-off` is the row that actually asks about the port,
# and it still asks Get-HealthReachability directly.
function Stop-Daemon {
    $notes = New-Object System.Collections.Generic.List[string]
    if ($script:TaskOwned) {
        $r = Invoke-Native "schtasks.exe" @("/end", "/tn", $script:TaskName)
        $notes.Add("schtasks /end /tn $($script:TaskName): ran=$($r.Ran) exit=$($r.ExitCode) $($r.Output -replace "`r?`n", ' ')")
    } else {
        $notes.Add("schtasks /end SKIPPED: this run did not register $($script:TaskName), so ending it would stop a task it does not own")
    }
    $before = Get-OwnedServerProcesses
    $notes.Add("before the kill: $($before.Detail)")
    if ($before.State -eq "owned") {
        foreach ($p in $before.Processes) {
            $k = Stop-OwnedServerProcess -ProcessId $p.Id -ImagePath $p.Path
            $notes.Add("kill pid $($p.Id): $($k.State) -- $($k.Detail)")
        }
    }
    # Break only on a MEASURED absence of this run's own processes.
    $last = $null
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $last = Get-OwnedServerProcesses
        if ($last.State -eq "none") { break }
        Start-Sleep -Milliseconds 500
    }
    # A loop that never assigned is not a stopped daemon either.
    if ($null -eq $last) {
        $last = [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "the stop-wait loop ended without running a single process read" }
    }
    $reach = Get-HealthReachability $Health
    $notes.Add("health probe, REPORTED AND NOT THE VERDICT: $($reach.State) -- $($reach.Detail)")
    $state = if ($last.State -eq "none") { "stopped" }
             elseif ($last.State -eq "owned") { "alive" }
             else { "unmeasurable" }
    $detail = "$($last.Detail) || " + ($notes -join " || ")
    Write-Host "Stop-Daemon: this run's daemon is '$state' -- $detail"
    # A stale rc from any native call above must not leak into the next Check
    # block. The STATE is the verdict; this only keeps an exit code from being
    # mistaken for one.
    $global:LASTEXITCODE = 0
    return [pscustomobject]@{ State = $state; Detail = $detail; Health = $reach.State }
}

# The rows this channel owes. Declared before the run, not derived from it: a
# list generated from the checks that executed would shrink silently with them.
# The helpers declare their own rows (cli-*, mcp-*) from their own inputs.
Expect-Rows -Names @(
    # The workflow's precheck step records `port-7878-precheck` before this
    # script starts, so that row is carried in; Record-CarriedRow below restates
    # its verdict as a row of this run's.
    "port-7878-precheck-carried",
    "download-zip",
    "zip-members",
    "wenlan-on-path",
    "setup-basic (wenlan.exe setup --basic)",
    "background-on (wenlan.exe background on)",
    "schtasks-registered",
    "health-version",
    "status (wenlan.exe status)",
    "cli-roundtrip-driver",
    "mcp-roundtrip-driver",
    "doctor (wenlan.exe doctor)",
    "dll-identity",
    "daemon-stopped-before-recovery",
    "autostart-recovery (wenlan.exe memories --limit 1)",
    "healthy-after-recovery",
    "background-off (wenlan.exe background off)",
    "task-kept-after-off",
    "health-unreachable-after-off",
    "autostart-marker",
    "stopped-marker-error (wenlan.exe search x)",
    "background-on-again",
    "health-version",
    "daemon-stopped-at-cleanup",
    "no-leftover-task",
    "no-leftover-dirs",
    "port-7878-closed"
)
Record-CarriedRow -Name "port-7878-precheck"

try {
    # OWNERSHIP FIRST, before a single byte is installed or started, because
    # everything this run later kills or deletes is licensed by these two reads
    # and by nothing else. Both are recorded as INFO rows so a run that took no
    # ownership is visible in the ledger rather than only on the console.
    $preTask = Get-TaskPresence $TaskName
    Info "task-registered-before-run" "$($preTask.State) -- $($preTask.Detail)"
    # THE DIRECTORY PRE-READS ARE NOT HERE ANY MORE, and where they are is the
    # whole of round-6's A2. They used to sit on these lines, above
    # `Get-ServerProcessInventory`, the release download, the extract, the
    # member diff, the recursive copy, the PATH mutation and `setup --basic` --
    # so the window in which a foreign creator gets credited to this run was
    # tens of seconds wide, not one statement. A developer who opened the real
    # Wenlan app during the download created %LOCALAPPDATA%\wenlan, and this run
    # would then have claimed and deleted it. Each pre-read now sits IMMEDIATELY
    # above the statement that creates its tree. The task read stays here
    # because `background on` is the creating statement for the task and it runs
    # further down; a task-name pre-read taken later would be taken after this
    # run had already installed the binary that registers it.
    # Decided HERE, before anything is installed and long before anything is
    # run, because `wenlan background on` is itself one of the operations that
    # needs the licence. Nothing below may reach $TaskName -- through schtasks
    # or through the CLI -- unless this is true.
    $MayDriveTask = ($preTask.State -eq "absent")
    $preServers = Get-ServerProcessInventory
    Info "server-processes-before-run" "$($preServers.State) -- $($preServers.Detail)"
    if ($preServers.State -eq "none") { $PreexistingServerPids = @() }
    elseif ($preServers.State -eq "read") { $PreexistingServerPids = @($preServers.Processes | ForEach-Object { $_.Id }) }
    else { $PreexistingServerPids = $null }

    # $Extract is under GAUNTLET_OUT and is nobody's data; it is created on its
    # own line so that the ONLY statement between the install dir's pre-read and
    # its claim is the one that creates the install dir.
    New-Item -ItemType Directory -Force -Path $Extract | Out-Null
    $preInstallDir = Get-DirPresence $InstallDir
    Info "install-dir-before-run" "$($preInstallDir.State) -- $($preInstallDir.Detail)"
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    # THE CLAIM, and it is one act rather than a second read plus a write. See
    # WHAT BINDS A TREE TO THIS RUN. A `New-Item` that did not create the tree
    # leaves this `not-owned` and the tree therefore undeleted.
    $InstallDirMark = New-OwnerMark $InstallDir $preInstallDir
    Info "install-dir-claimed" "$($InstallDirMark.State) -- $($InstallDirMark.Detail)"
    $Zip = Join-Path $Work "wenlan-windows-x64.zip"
    Info "documented-flow" "download wenlan-windows-x64.zip; extract; add to PATH; wenlan.exe setup --basic; wenlan.exe background on; wenlan.exe status"
    Check -Name "download-zip" -Script { if (-not (Get-Asset "wenlan-windows-x64.zip" $Zip)) { throw "download failed after 3 attempts" }; Write-Output ("bytes=" + (Get-Item $Zip).Length) }
    Check -Name "zip-members" -Script {
        Expand-Archive -Path $Zip -DestinationPath $Extract -Force -ErrorAction Stop
        $got = @(Get-ChildItem -Path $Extract -Recurse -File | ForEach-Object { $_.FullName.Substring($Extract.Length).TrimStart('\') } | Sort-Object)
        $diff = @(Compare-Object -ReferenceObject @($ExpectedMembers | Sort-Object) -DifferenceObject $got)
        if ($diff.Count -ne 0) { throw ("member mismatch (<= expected only, => zip only): " + (($diff | ForEach-Object { "$($_.SideIndicator) $($_.InputObject)" }) -join ", ")) }
        Write-Output ("members: " + ($got -join ", "))
    }
    Copy-Item -Path (Join-Path $Extract "*") -Destination $InstallDir -Recurse -Force
    $env:PATH = "$InstallDir;$env:PATH"
    Info "install-dir" $InstallDir
    Check -Name "wenlan-on-path" -Script { $cmd = Get-Command wenlan.exe -ErrorAction Stop; Write-Output $cmd.Source; if ($cmd.Source -ne $Wenlan) { throw "resolved to $($cmd.Source)" } }
    # `setup --basic` is what creates $DataDir, so the pre-read goes IMMEDIATELY
    # above it and the claim IMMEDIATELY below it: the window in which someone
    # else's %LOCALAPPDATA%\wenlan could be credited to this run is that one
    # Check and nothing else. Nothing above this line creates the data root --
    # the download, the extract and the copy all land under GAUNTLET_OUT or the
    # install dir -- so moving the read down loses no coverage.
    $preDataDir = Get-DirPresence $DataDir
    Info "data-dir-before-run" "$($preDataDir.State) -- $($preDataDir.Detail)"
    Check -Name "setup-basic (wenlan.exe setup --basic)" -Expect "Wenlan is set up for local memory." -Script { & wenlan.exe setup --basic }
    # A setup that failed leaves this `not-owned` -- CreateNew into a directory
    # that is not there raises DirectoryNotFoundException -- and an unowned tree
    # is never deleted.
    $DataDirMark = New-OwnerMark $DataDir $preDataDir
    Info "data-dir-claimed" "$($DataDirMark.State) -- $($DataDirMark.Detail)"
    # `background on` REGISTERS AND STARTS a Windows scheduled task at a fixed,
    # product-chosen name. If that name is already taken, this command does not
    # create the developer's task -- it takes it over: rewrites its definition
    # and starts it. Gating only `schtasks /end` and `/delete`, as the previous
    # version did, left this door wide open, and it is the SAME door.
    #
    # A refusal here is recorded as a FAIL, never skipped. A skipped row is a
    # row that is simply not there, and this file's entire argument is that a
    # measurement which did not happen must not look like one that did.
    Check -Name "background-on (wenlan.exe background on)" -Expect "Installed and started Windows scheduled task" -Script {
        if (-not $MayDriveTask) {
            throw ("$TaskName was '$($preTask.State)' before this run, so registering at that name would take over a task this run did not create. " +
                   "Refusing to run 'wenlan background on'; the documented flow is UNTESTED here, which is not the same as broken.")
        }
        & wenlan.exe background on
    }
    Check -Name "schtasks-registered" -Script {
        if (-not $MayDriveTask) { throw "not attempted: this run declined to register $TaskName (it was '$($preTask.State)' before the run)" }
        & schtasks.exe /query /tn $TaskName /fo LIST
    }
    # NO ANALOGOUS BINDING EXISTS FOR THE REGISTRATION, and this is where that
    # is said rather than left for the next reader to assume one is there.
    #
    # The two directory trees get a run-scoped GUID marker: this run writes a
    # fresh GUID into the tree and refuses to delete anything it cannot read
    # that GUID back out of (see WHAT BINDS A TREE TO THIS RUN, beside
    # Get-DirPresence). Round-5 review asked for the same for the scheduled
    # task -- the definition XML or the registration date, captured here and
    # re-verified before `/end` and `/delete`. There is no cheap one, and the
    # reason is MEASURED ON THIS HOST rather than assumed:
    #
    #   Export-ScheduledTask returns the definition, and its
    #   RegistrationInfo/Date is AUTHOR-SUPPLIED metadata carried INSIDE that
    #   definition -- not a receipt the scheduler stamps at registration.
    #   Four of the six root-path (`\`) tasks on this machine have no Date
    #   element at all; the one that has it carries whole-second resolution
    #   ('2026-08-31T03:54:10'); and three sibling \Microsoft\Windows\ tasks
    #   share ONE identical Date to the tick, because it was written once by
    #   whoever authored them.
    #
    # So the XML is identical across two registrations of the same definition
    # -- which is exactly the replacement such a check would have to detect:
    # the developer, or a second gauntlet run, executing `wenlan background on`
    # again between this read and the teardown. It would refuse nothing it
    # ought to refuse, while reading like a guarantee.
    #
    # Nor can this run WRITE a mark into the task the way it writes one into a
    # directory. It does not author the definition -- `wenlan background on`
    # does -- and editing the registration afterwards would both mutate the
    # very object under test and write to the developer's task scheduler, which
    # is the thing the safe default here forbids.
    #
    # What the registration has instead is the two-provider agreement below,
    # and that is a weaker fact, not the same one. THE RESIDUAL, stated: between
    # this post-read and the teardown, a registration this run did not make
    # could be standing at this name, and this run would end and delete it.
    # Deleting a registration is RECOVERABLE -- `wenlan background on` puts it
    # back -- and destroying %LOCALAPPDATA%\wenlan is not. That asymmetry is
    # why the trees got a marker and this did not; it is not a claim that the
    # race is closed.
    #
    # ROUND 6, A6 narrowed WHEN the licence is spent without closing what it is
    # a licence FOR. `$TaskOwned` is now re-measured immediately before the two
    # teardown acts that destroy state on the developer's machine -- the cleanup
    # `wenlan background off` and `schtasks /delete` -- so those two rest on a
    # reading taken at the moment of the act rather than one taken before a
    # dozen intervening CLI checks. THE TWO PLACES IT IS NOT RE-READ, and why:
    #
    #   `schtasks /end` inside Stop-Daemon. Ending a run of a task is the
    #   least destructive of the three acts -- it stops an execution, it does
    #   not unregister anything, and the next trigger starts it again -- and
    #   Stop-Daemon is called from two places on a path where an extra table
    #   read costs a full `schtasks /query /fo CSV` plus a Get-ScheduledTask
    #   enumeration each time.
    #
    #   The `background-off` CLI row above. It runs `wenlan background off`,
    #   which is the same act as the cleanup one, and it has the same staleness.
    #
    # Both are the same residual as the paragraph above: a registration this
    # run did not make, standing at this name, ended or switched off by this
    # run. The re-reads below do not fix that either -- a re-read establishes
    # that SOMETHING is registered here right now, never that it is ours. What
    # the re-reads do close is narrower and worth having on its own: acting on
    # a name that has since gone, and acting on a name nobody can read.
    # OWNERSHIP, measured after the fact rather than inferred from the pre-state.
    # Round-4 review: "the result of the registration operation does not flow
    # into that decision". It does now -- the name must have been free before
    # AND a task must be measurably there after. A `background on` that failed
    # while something else claimed the name leaves this false, and a post-read
    # that could not be taken leaves it false too, which is the safe direction:
    # the run declines to touch a task rather than guessing it is its own.
    $postTask = Get-TaskPresence $TaskName
    Info "task-registered-after-background-on" "$($postTask.State) -- $($postTask.Detail)"
    $TaskOwned = ($MayDriveTask -and $postTask.State -eq "present")
    Info "task-owned-by-this-run" ("$TaskOwned (free before: $MayDriveTask, it was '$($preTask.State)'; present after registering: " +
                                  "'$($postTask.State)'). If this is False, this run will neither end, nor switch off, " +
                                  "nor delete $TaskName.")
    if (Wait-Health -Url $Health -Seconds 240) { Assert-Version -Url $Health -Expected $Version }
    Check -Name "status (wenlan.exe status)" -Script { & wenlan.exe status }

    $env:WENLAN_BIN = $Wenlan
    Check-Helper -Name "cli-roundtrip-driver" -Interpreter "pwsh" -InterpreterArgs @("-NoProfile", "-File") -Path (Join-Path $Helpers "cli-roundtrip.ps1") -MustDeclare "^cli-"
    $env:MCP_BIN = Join-Path $InstallDir "wenlan-mcp.exe"
    $env:MCP_ARGS = "[]"
    $env:EXPECT_TOOL_COUNT = "29"
    $env:MCP_TOOLS = "capture,recall,brief"
    Check-Helper -Name "mcp-roundtrip-driver" -Interpreter "python" -Path (Join-Path $Helpers "mcp-roundtrip.py") -MustDeclare "^mcp-"

    Check -Name "doctor (wenlan.exe doctor)" -Expect "Daemon: running on" -Script { & wenlan.exe doctor }
    Check -Name "dll-identity" -Script {
        # THIS RUN'S daemon, not the first process that happens to share the
        # name. `Get-Process -Name wenlan-server | Select -First 1` would happily
        # measure a developer's production daemon -- which, installed to the same
        # documented location, loads the same DLLs and would pass this row on
        # behalf of a build nobody here downloaded.
        $owned = Get-OwnedServerProcesses
        if ($owned.State -ne "owned") { throw "no wenlan-server process could be shown to belong to this run: $($owned.Detail)" }
        $srv = $owned.Processes[0]
        Write-Output "wenlan-server pid=$($srv.Id) path=$($srv.Path)"
        foreach ($dll in @("onnxruntime.dll", "vulkan-1.dll")) {
            $loaded = @(Get-Process -Id $srv.Id -Module -ErrorAction Stop | Where-Object { $_.ModuleName -ieq $dll } | ForEach-Object { $_.FileName })
            $want = Join-Path $InstallDir $dll
            if ($loaded.Count -ne 1 -or -not [string]::Equals($loaded[0], $want, [System.StringComparison]::OrdinalIgnoreCase)) { throw "$dll loaded from [$($loaded -join ', ')], expected exactly $want" }
            Write-Output "$dll -> $($loaded[0])"
        }
    }

    # Recovery: kill the daemon behind the task's back; a read command must restart it.
    #
    # The row exists because the next check is only a recovery test if the
    # daemon was actually stopped first. `autostart-recovery` expects "daemon
    # not reachable" from the CLI -- against a daemon that never went down that
    # string cannot appear, so the row would fail; but against a daemon whose
    # state NOBODY MEASURED, a pass certifies a recovery from an unknown
    # starting point. All three states, recorded.
    #
    # The state branched on here is a PROCESS state. `down` used to mean "the
    # health socket refused once", which is not this row's claim; see the header
    # of Stop-Daemon.
    $stopped = Stop-Daemon
    Check -Name "daemon-stopped-before-recovery" -Script {
        if ($stopped.State -eq "stopped") { Write-Output "daemon stopped before the recovery check: $($stopped.Detail)"; return }
        if ($stopped.State -eq "alive") { throw "the daemon this run started is still running after Stop-Daemon: $($stopped.Detail); the recovery check below would be testing nothing" }
        throw "could not measure whether the daemon stopped before the recovery check: $($stopped.Detail); recorded as unproven, not as stopped"
    }
    Check -Name "autostart-recovery (wenlan.exe memories --limit 1)" -Expect "daemon not reachable" -Script { & wenlan.exe memories --limit 1 }
    # All three states, here and at every other caller of this probe. A row that
    # says "recovered" must rest on a reachable daemon, never on a probe whose
    # own failure happened to look like one.
    Check -Name "healthy-after-recovery" -Script {
        $h = Get-HealthReachability $Health
        if ($h.State -eq "reachable") { Write-Output "recovered: $($h.Detail)"; return }
        if ($h.State -eq "down") { throw "health unreachable after recovery: $($h.Detail)" }
        throw "could not measure whether the daemon recovered: $($h.Detail); recorded as unproven, not as recovered"
    }

    # `background off` STOPS the task and changes its autostart state. Same
    # door as `background on`, same licence required, and here the stricter one:
    # switching off is a destructive edit to a live registration, so it needs
    # the ownership that was MEASURED after registering, not merely a free name.
    Check -Name "background-off (wenlan.exe background off)" -Expect "Background registration kept" -Script {
        if (-not $TaskOwned) {
            throw ("this run does not own $TaskName (free before: $MayDriveTask, present after registering: '$($postTask.State)'), " +
                   "so switching it off would change the autostart state of a task belonging to someone else. Refusing.")
        }
        & wenlan.exe background off
    }
    Check -Name "task-kept-after-off" -Script { & schtasks.exe /query /tn $TaskName }
    # The row this defect was found in. "Unreachable" is a claim about the
    # DAEMON, so only a measured refusal may make it; a probe that fell over on
    # its way to the socket proves nothing about what is on the other end.
    Check -Name "health-unreachable-after-off" -Script {
        $h = Get-HealthReachability $Health
        if ($h.State -eq "down") { Write-Output "background off left $Health unreachable: $($h.Detail)"; return }
        if ($h.State -eq "reachable") { throw "background off left $Health reachable: $($h.Detail)" }
        throw "could not measure whether $Health is reachable after background off: $($h.Detail); recorded as unproven, not as a clean shutdown"
    }
    Check -Name "autostart-marker" -Script { $marker = Join-Path $DataDir "autostart.off"; if (-not (Test-Path $marker)) { throw "missing $marker" }; Write-Output $marker }
    Check -Name "stopped-marker-error (wenlan.exe search x)" -ExpectFail "daemon stopped by" -Script { & wenlan.exe search x }
    Check -Name "background-on-again" -Expect "Installed and started Windows scheduled task" -Script {
        if (-not $TaskOwned) { throw "not attempted: this run does not own $TaskName, so it may not re-register at that name" }
        & wenlan.exe background on
    }
    if (Wait-Health -Url $Health -Seconds 120) { Assert-Version -Url $Health -Expected $Version }
} finally {
    # The cleanup copy of the same door. This used to run unconditionally, so a
    # run that had correctly refused to END or DELETE a stranger's task would
    # then reach it here anyway and switch it off -- the ownership refusal
    # undone by its own teardown, at the point where nobody is watching.
    #
    # ROUND 6, A6. The licence is RE-MEASURED here rather than spent on a
    # reading taken before the run's own CLI checks. $TaskOwned was decided at
    # `task-registered-after-background-on`, and everything between that point
    # and this one is time in which the registration could have changed hands.
    # What the re-read establishes is narrow and is worth stating exactly: that
    # a registration is standing at $TaskName AT THE MOMENT of the destructive
    # call, and that the question could be answered at all. It does NOT
    # establish that the registration is the one this run made -- the task
    # carries no identity this run can bind, for the reasons MEASURED beside
    # `schtasks-registered`. It closes "switched off a task that had already
    # gone" and "switched off a task nobody could read"; it does not close the
    # replacement, and nothing available here does.
    $offTask = if ($TaskOwned) { Get-TaskPresence $TaskName } else { $null }
    if ($TaskOwned -and $null -ne $offTask -and $offTask.State -eq "present" -and (Test-Path $Wenlan)) {
        $off = Invoke-Native $Wenlan @("background", "off")
        Write-Host "cleanup: wenlan background off ran=$($off.Ran) exit=$($off.ExitCode) $($off.Output -replace "`r?`n", ' ')"
    } elseif (-not $TaskOwned) {
        Write-Host ("cleanup: wenlan background off SKIPPED -- this run does not own $TaskName " +
                    "(free before: $MayDriveTask, present after registering: '$($postTask.State)'), " +
                    "so its autostart state is not this run's to change")
    } elseif ($null -eq $offTask -or $offTask.State -ne "present") {
        $why = if ($null -eq $offTask) { "never re-read" } else { "$($offTask.State): $($offTask.Detail)" }
        Write-Host ("cleanup: wenlan background off SKIPPED -- immediately before the call $TaskName reads " +
                    "'$why', and a registration that is not measurably standing right now is not one this " +
                    "run may switch off")
    } else {
        Write-Host "cleanup: wenlan background off SKIPPED -- there is no $Wenlan to run it with"
    }
    # The SECOND caller boundary, and the one that matters most: this is the
    # teardown that is supposed to leave the developer's machine as it found it.
    # `port-7878-closed` below reports the port; this reports whether the daemon
    # this run started was seen to stop. An unmeasurable stop here is exactly
    # the state that used to be printed and dropped.
    $stoppedAtCleanup = Stop-Daemon
    Check -Name "daemon-stopped-at-cleanup" -Script {
        if ($stoppedAtCleanup.State -eq "stopped") { Write-Output "daemon stopped during cleanup: $($stoppedAtCleanup.Detail)"; return }
        if ($stoppedAtCleanup.State -eq "alive") { throw "the daemon this run started is still running at the end of the run: $($stoppedAtCleanup.Detail)" }
        throw "could not measure whether the daemon stopped during cleanup: $($stoppedAtCleanup.Detail); recorded as unproven, not as stopped"
    }
    # Only the registration this run made, and only while a registration is
    # measurably standing at that name RIGHT NOW. Deleting a task this run did
    # not create is not cleanup, it is removing the developer's background
    # service; deleting one on the strength of a reading taken several minutes
    # and a dozen CLI calls ago is the same act with a staler excuse. This is
    # the SHAPE `scripts/attest.sh` uses to break a stale lock -- re-read
    # immediately before destroying, and refuse when the re-read fails -- but
    # it is deliberately the weaker half of it. There, the re-read is compared
    # against the exact owner value whose age licensed the break, so it detects
    # a REPLACEMENT. Here there is no value to compare against, for the reasons
    # measured beside `schtasks-registered`, so the re-read establishes only
    # that a registration is standing at this name at the moment of the delete.
    $delTask = if ($TaskOwned) { Get-TaskPresence $TaskName } else { $null }
    if ($TaskOwned -and $null -ne $delTask -and $delTask.State -eq "present") {
        $del = Invoke-Native "schtasks.exe" @("/delete", "/tn", $TaskName, "/f")
        $TaskDeleteResult = $del
        Write-Host "cleanup: schtasks /delete ran=$($del.Ran) exit=$($del.ExitCode) $($del.Output -replace "`r?`n", ' ')"
    } elseif (-not $TaskOwned) {
        Write-Host "cleanup: schtasks /delete SKIPPED -- this run did not register $TaskName, so the registration is not its to remove"
    } else {
        # $TaskDeleteResult stays $null deliberately, which makes the
        # `no-leftover-task` row below UNPROVEN rather than green: this run
        # owned the name, deleted nothing, and is in no position to claim the
        # machine was left as it was found. An `absent` re-read lands here too
        # -- something took the registration away between the ownership read
        # and this line, and that is not a clean teardown either.
        $why = if ($null -eq $delTask) { "never re-read" } else { "$($delTask.State): $($delTask.Detail)" }
        Write-Host ("cleanup: schtasks /delete SKIPPED -- immediately before the delete $TaskName reads '$why'")
    }
    Collect (Join-Path $DataDir "logs")
    # Only the trees this run created. Unowned means LEFT IN PLACE and said
    # so out loud: a silent skip and a silent delete look identical in a log,
    # and only one of them is recoverable.
    #
    # ROUND 6 removed the `$InstallDirOwned` / `$DataDirOwned` booleans this
    # loop used to read. They were a SECOND READ of the directory taken one
    # statement after the creating one, and a boolean carries no evidence: by
    # the time it reached here it said "yes" and could not say what it had been
    # a yes ABOUT. The mark that replaced them is the claim itself -- a handle
    # this run has held open inside the tree since the moment it created it --
    # so `$d.Mark.State -eq "owned"` is not a remembered verdict, it is a live
    # one, and `Test-OwnerMark` below re-establishes it against the path two
    # statements before the delete.
    $CleanupDirs = @(
        @{ Name = "install dir"; Path = $InstallDir; Mark = $InstallDirMark; Pre = $preInstallDir
           Licence = $null; Release = $null; Removal = $null },
        @{ Name = "data dir";    Path = $DataDir;    Mark = $DataDirMark;    Pre = $preDataDir
           Licence = $null; Release = $null; Removal = $null }
    )
    $ReleaseNotes = New-Object System.Collections.Generic.List[string]
    foreach ($d in $CleanupDirs) {
        $preState = if ($null -eq $d.Pre) { "never measured -- the run aborted before the pre-read" } else { $d.Pre.State }
        $markState = if ($null -eq $d.Mark) { "never attempted -- the run aborted before the claim" } else { $d.Mark.State }
        $markWhy = if ($null -eq $d.Mark) { "" } else { ": $($d.Mark.Detail)" }
        if (-not $d.Path) {
            $d.Licence = [pscustomobject]@{ State = "refused"; Detail = "no $($d.Name) path was ever discovered by this run" }
        } elseif ($null -eq $d.Mark -or $d.Mark.State -ne "owned") {
            # All three of the claim's states that are not "owned" land here,
            # and they are NOT the same fact: `not-owned` means something was
            # already at that path, `unmeasurable` means the claim could not be
            # attempted or read back. Both refuse, but the detail says which,
            # because "we left it alone because it was someone else's" and "we
            # left it alone because we could not tell" are different reports to
            # the person reading this log afterwards.
            $d.Licence = [pscustomobject]@{ State = "refused"
                Detail = ("this run has no claim on the $($d.Name) $($d.Path) (before this run: $preState; " +
                          "the claim is '$markState'$markWhy), so it is not this run's to delete") }
        } elseif ($stoppedAtCleanup.State -ne "stopped") {
            # The daemon this run started was not SEEN to stop, so something may
            # still be writing into the tree about to be deleted. Round-3's C3
            # is this same hazard one level up; the row above records it, and
            # this is the delete declining to run ahead of that row.
            $d.Licence = [pscustomobject]@{ State = "refused"
                Detail = "the daemon was not measured stopped ($($stoppedAtCleanup.State)): $($stoppedAtCleanup.Detail); nothing is deleted while a daemon may still be in it" }
        } else {
            # RE-VERIFIED HERE, two statements before the delete, rather than
            # once at the top of the teardown. The gap between an ownership
            # decision and the destruction it authorises is the whole of the
            # hazard: everything above this line -- the daemon stop, the task
            # delete, the log collection -- is time in which the tree could
            # have been swapped. The marker's own handle is what makes that
            # swap impossible rather than merely detectable (a held
            # CreateNew/DeleteOnClose file with Delete withheld from its share
            # mode makes `Remove-Item -Recurse` and `Rename-Item` on the tree
            # fail; that was MEASURED, see the block beside New-OwnerMark), but
            # "impossible" is a claim about a mechanism, and this is the read
            # that checks the mechanism actually held.
            $bound = Test-OwnerMark $d.Mark $d.Path
            if ($bound.State -eq "verified") {
                $d.Licence = [pscustomobject]@{ State = "granted"; Detail = "$($d.Name) $($d.Path): $($bound.Detail)" }
            } else {
                $d.Licence = [pscustomobject]@{ State = "refused"
                    Detail = "the $($d.Name) $($d.Path) is not bound to this run ($($bound.State)): $($bound.Detail)" }
            }
        }
        # The claim is given up whatever the verdict was, and the result is
        # RECORDED. Two reasons it is unconditional. The tree cannot be deleted
        # while this run holds a file inside it, so a release that did not
        # happen is a delete that cannot happen -- saying so here is clearer
        # than ten sharing violations out of Remove-Retry. And a marker left
        # behind under a tree this run declines to delete makes the NEXT run's
        # CreateNew fail on a directory it created itself, which is the marker
        # residue review found in round 6's predecessor. DeleteOnClose means the
        # kernel removes the file when the handle goes, including on a kill --
        # but a release that could not be CONFIRMED is still unproven, and is
        # reported as such rather than assumed.
        $d.Release = Close-OwnerMark $d.Mark
        $ReleaseNotes.Add("$($d.Name): $($d.Release.State) -- $($d.Release.Detail)")
        Write-Host "cleanup: $($d.Name) marker release -- $($d.Release.State): $($d.Release.Detail)"
        if ($d.Licence.State -eq "granted" -and $d.Release.State -ne "released") {
            $d.Licence = [pscustomobject]@{ State = "refused"
                Detail = ("the $($d.Name) $($d.Path) is bound to this run, but the claim on it could not be " +
                          "released ($($d.Release.State)): $($d.Release.Detail); a tree this run still holds " +
                          "a handle inside cannot be removed, so nothing is attempted") }
        }
        if ($d.Licence.State -eq "granted") {
            $d.Removal = Remove-Retry $d.Path
            Write-Host "cleanup: $($d.Name) $($d.Path) delete -- $($d.Removal.State): $($d.Removal.Detail)"
        } else {
            Write-Host "cleanup: $($d.Name) $($d.Path) LEFT IN PLACE -- $($d.Licence.State): $($d.Licence.Detail)"
        }
    }
    # Not a Check, because a marker this run failed to release is a residue on
    # the developer's disk rather than a claim this run got wrong -- but it goes
    # in the ledger rather than only on the console, because it is the one piece
    # of state this run can leave behind that will make the NEXT run refuse to
    # claim a tree it created itself.
    Info "owner-marker-release" ($ReleaseNotes -join " || ")
    # An unowned task cannot make this row honestly: the run neither created it
    # nor deleted it, so "no task is left behind by this run" is not something
    # it is in a position to claim. Unproven, not a pass.
    # This row asserts the machine was left as it was found, so it may not be
    # built on a query whose FAILURE looks like the answer it wants. The
    # previous version accepted any non-zero exit from `schtasks /query /tn` as
    # "not there" -- access denied, a scheduler fault and a genuine absence all
    # exit non-zero and all read as success -- and it only LOGGED the delete, so
    # a failed delete followed by a failed query passed. That is this file's own
    # defect class, in the one place it exists to disprove.
    #
    # Three things are now required, in order: the delete ran and succeeded, and
    # the tri-state read says ABSENT. `unmeasurable` is unproven, never removed.
    # The directory half of "the machine was left as it was found". It is a
    # row rather than a log line for the same reason no-leftover-task is: a
    # teardown nobody grades is a teardown that quietly stops happening. An
    # unowned tree makes this row UNPROVEN, not green -- the run neither
    # created it nor removed it, so it is in no position to claim anything
    # about it either way.
    Check -Name "no-leftover-dirs" -Script {
        $bad = @(); $unproven = @()
        foreach ($d in $CleanupDirs) {
            $lic = $d.Licence
            if ($null -eq $lic) { $unproven += "$($d.Name) $($d.Path): the teardown recorded no decision about it"; continue }
            if ($lic.State -eq "refused") { $unproven += $lic.Detail; continue }
            if ($lic.State -eq "nothing-to-delete") { continue }
            $r = $d.Removal
            if ($null -eq $r) { $bad += "$($d.Name) $($d.Path) was licensed for deletion and no delete was attempted"; continue }
            # The delete's OWN report, required rather than assumed. Ten failed
            # attempts used to return exactly what a success returned.
            if ($r.State -ne "removed") { $bad += "$($d.Name) $($d.Path) was not removed ($($r.State)): $($r.Detail)"; continue }
            $after = Get-DirPresence $d.Path
            if ($after.State -eq "absent") { continue }
            if ($after.State -eq "present") { $bad += "$($d.Name) $($d.Path) is still there after this run deleted it" }
            else { $bad += "could not measure whether $($d.Name) $($d.Path) is gone -- $($after.Detail)" }
        }
        if ($bad.Count -gt 0) { throw ($bad -join '; ') }
        if ($unproven.Count -gt 0) {
            throw ("this run cannot claim the machine was left as it was found: " + ($unproven -join '; ') + "; recorded as unproven")
        }
        Write-Output "every tree this run created is gone"
    }
    Check -Name "no-leftover-task" -Script {
        # The REASON, not just the verdict. "present after registering:
        # 'unmeasurable'" tells a reader nothing about which of half a dozen
        # refusals fired -- a truncated table, a torn row, two providers
        # disagreeing, an independent read that could not be taken -- and a
        # refusal nobody can diagnose is one somebody will eventually delete.
        if (-not $TaskOwned) { throw "this run does not own $TaskName (free before: $MayDriveTask -- $($preTask.State): $($preTask.Detail)) (present after registering: $($postTask.State): $($postTask.Detail)), so it did not delete it and cannot claim the machine was left as it was found; recorded as unproven" }
        if ($null -eq $TaskDeleteResult) { throw "this run owned $TaskName but no delete was attempted, so nothing establishes the registration is gone; recorded as unproven" }
        if (-not $TaskDeleteResult.Ran) { throw "the $TaskName delete could not be run ($($TaskDeleteResult.Output)), so the registration this run made may still be there; recorded as unproven" }
        if ($TaskDeleteResult.ExitCode -ne 0) { throw "schtasks /delete /tn $TaskName exited $($TaskDeleteResult.ExitCode) ($($TaskDeleteResult.Output -replace "`r?`n", ' ')); the delete FAILED, so this run left its own registration behind" }
        $t = Get-TaskPresence $TaskName
        if ($t.State -eq "absent") { Write-Output "the registration this run made is gone: $($t.Detail)"; return }
        if ($t.State -eq "present") { throw "$TaskName is still registered after this run deleted it: $($t.Detail)" }
        throw "could not measure whether $TaskName is still registered: $($t.Detail); recorded as unproven, not as removed"
    }
    Check -Name "port-7878-closed" -Script {
        $p = Get-PortListenerState 7878
        if ($p.State -eq "none") { Write-Output "port 7878 $($p.Detail)"; return }
        if ($p.State -eq "found") { throw "port 7878 still listening (pid $($p.OwningProcess))" }
        throw "could not measure whether port 7878 is closed: $($p.Detail); recorded as unproven, not as closed"
    }
    # `(-not (Test-Path $DataDir))` was the last collapse left in this file: a
    # directory that exists but cannot be read returns $false from Test-Path
    # exactly as a missing one does, so an ACL that hid the developer's data dir
    # was reported here as "True -- removed". The tri-state read has an answer
    # for that case and this row prints it verbatim.
    $dataAfter = Get-DirPresence $DataDir
    Info "data-dir-removed" "$($dataAfter.State) -- $($dataAfter.Detail)"
    if (-not (Evaluate)) { exit 1 }
}
