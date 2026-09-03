# First-run gauntlet: Windows desktop installer channel. README: "On Windows x64,
# run the -setup.exe from the same Releases page." Installs silently (NSIS /S,
# per-user), launches Wenlan.exe, proves the sidecar daemon, CLI, and MCP, then
# uninstalls silently and checks what is left behind. Everything comes from env
# (TAG, VERSION, GAUNTLET_OUT, GAUNTLET_CHANNEL, REPO_ROOT). No global Stop.
$ProgressPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot "lib.ps1")

$Tag = $env:TAG
$Version = $env:VERSION
$Health = "http://127.0.0.1:7878/api/health"
$Helpers = Join-Path $env:REPO_ROOT "scripts\first-run"
$Work = Join-Path $script:GauntletOut "work-nsis"
$DataDir = Join-Path $env:LOCALAPPDATA "wenlan"
$SetupName = "Wenlan_${Version}_x64-setup.exe"
$Setup = Join-Path $Work $SetupName
$Bundled = @("wenlan.exe", "wenlan-server.exe", "wenlan-mcp.exe", "onnxruntime.dll", "vulkan-1.dll")
$UninstallRoots = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
    "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall",
    "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall")
$AppExe = $null
$Install = $null
$App = $null
# What this run may kill at teardown. Both start as NOT OURS -- an empty image
# and an unmeasured pid set -- and only a measurement taken before the app is
# launched moves them. `finally` runs even when the try block aborts on its
# first line, so they have to be defined here rather than where they are set.
$OwnedServerImage = ""
$PreexistingServerPids = $null
# Why the snapshot above is $null, when it is. Declared here for the same
# reason the others are: `finally` reads it and `try` may never reach it.
$SnapshotDoubt = "not recorded"
# What this run may DELETE at teardown, on the same two-fact licence the
# process sweep above uses. `Remove-Retry $DataDir` ran unconditionally, and
# $DataDir is %LOCALAPPDATA%\wenlan -- the developer's real memorydb, config
# and logs. Sixteen lines above that call this file records
# `user-data-survives-uninstall`, so the invariant was already written down
# here; the teardown simply broke it afterwards, where nobody was reading.
#
# Absent before AND present after this run installed. Absent-before alone
# licenses creating and nothing else; a pre-existing tree is never this
# run's to remove, and unlike the install dir it cannot be reinstalled.
#
# ROUND 6 CHANGED WHAT THE SECOND HALF IS. It used to be a second READ, taken
# a statement after the first, with a marker written a statement after that;
# it is now a single atomic act that IS the claim. The `$DataDirOwned` and
# `$installOwned` booleans that carried the old verdict are gone -- a boolean
# arrives at the teardown saying "yes" and unable to say what it was a yes
# ABOUT. See WHAT BINDS A TREE TO THIS RUN.
$preDataDir = $null
$preInstall = $null
# The user's files under $DataDir as they were on the statement before the
# uninstaller ran. `finally` does not read it, but the Check that does runs
# late enough that an abort before the snapshot must leave it recognisably
# unset rather than undefined.
$preDataSnapshot = $null
# The install dir's post-read is KEPT for the log; nothing decides on it.
$postInstall = $null
$DataDirMark = $null
$InstallMark = $null
$InstallMarkRelease = $null
# The DOCUMENTED per-user install location. The install dir's pre-state has
# to be read against a path known before the installer runs -- reading it
# after discovery would read a tree the installer has just created, and every
# run would call it present. A run whose installer landed somewhere else
# (the depth-3 fallback search below finds older layouts) owns nothing here.
$InstallCandidate = Join-Path $env:LOCALAPPDATA "Programs\Wenlan"
if (-not $env:GAUNTLET_CHANNEL) { $env:GAUNTLET_CHANNEL = $script:GauntletChannel }

function Get-Asset([string]$AssetName, [string]$Dest) {
    $url = "https://github.com/7xuanlu/wenlan/releases/download/$Tag/$AssetName"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try { Invoke-WebRequest -Uri $url -OutFile $Dest -UseBasicParsing -ErrorAction Stop; return $true }
        catch { Write-Host "download attempt $attempt failed: $($_.Exception.Message)"; Start-Sleep -Seconds (5 * $attempt) }
    }
    return $false
}
# Tri-state, and the reason is the same one that runs through this whole file:
# `Get-ItemProperty ... -ErrorAction SilentlyContinue` returns NOTHING both when
# a key is not there and when it could not be read, and the second call site
# below is `uninstall-removes-registry-key` -- a row that certifies the
# uninstaller cleaned up after itself. A registry hive this account may not
# enumerate produced exactly what a clean uninstall produces.
#
# The roots are handled separately from the subkeys under them, because they
# are different facts: HKLM\...\WOW6432Node genuinely does not exist on some
# machines, and a root that is simply absent contributes nothing and is not a
# failure. A root that is THERE and unreadable is, and so is a subkey under a
# readable root -- which is why -ErrorVariable is read rather than discarded:
# `Get-ItemProperty root\*` walks every subkey, and SilentlyContinue drops the
# ones it could not open while still returning the ones it could. A partial
# enumeration that finds no Wenlan is not a measurement that there is none.
function Get-UninstallEntry {
    $found = New-Object System.Collections.Generic.List[object]
    $trouble = New-Object System.Collections.Generic.List[string]
    $searched = New-Object System.Collections.Generic.List[string]
    foreach ($root in $UninstallRoots) {
        try { $null = Get-Item -LiteralPath $root -ErrorAction Stop }
        catch [System.Management.Automation.ItemNotFoundException] { continue }
        catch {
            $trouble.Add("$root exists as far as this run can tell but would not open ($($_.Exception.GetType().FullName): $($_.Exception.Message))")
            continue
        }
        $searched.Add($root)
        $subErrors = $null
        $props = @(Get-ItemProperty -Path (Join-Path $root "*") -ErrorAction SilentlyContinue -ErrorVariable subErrors)
        if ($null -ne $subErrors -and @($subErrors).Count -gt 0) {
            $trouble.Add("$(@($subErrors).Count) subkey(s) under $root could not be read (first: $(@($subErrors)[0].Exception.Message))")
        }
        foreach ($p in $props) {
            # -eq, not .Equals: PowerShell's -eq on strings is case-insensitive
            # and the display name is a product string, not an identifier.
            if ("$($p.DisplayName)" -eq "Wenlan") { $found.Add($p) }
        }
    }
    if ($found.Count -gt 0) {
        return [pscustomobject]@{ State = "present"; Entry = $found[0]
            Detail = "$($found.Count) uninstall entry/entries named 'Wenlan'; first at $($found[0].PSPath)" }
    }
    if ($trouble.Count -gt 0) {
        return [pscustomobject]@{ State = "unmeasurable"; Entry = $null
            Detail = "no uninstall entry named 'Wenlan' was found, but the search was incomplete, so its absence is not a measurement: $($trouble -join '; ')" }
    }
    return [pscustomobject]@{ State = "absent"; Entry = $null
        Detail = "no uninstall entry named 'Wenlan' under $($searched.Count) readable root(s): $($searched -join ', ')" }
}
# Is the process table believable, AND does it agree that this particular
# process is absent? The positive witness the liveness probe below demands
# before it will accept a NEGATIVE.
#
# It is passed the target on purpose. A witness that does not co-vary with the
# claim it ratifies is not a witness: the first form of this function checked
# only that the table had pid 4 in it and at least ten rows, so it answered the
# question "is this a process table" and was then used to answer the question
# "is wenlan-server not in it". Those come apart exactly where it matters --
# `Get-Process -Name wenlan-server` can throw its absence error while the
# whole-table read taken one line later CONTAINS wenlan-server (a race with a
# restart, a filtered or per-session view on the targeted call, a name lookup
# that failed for its own reasons). The old witness said Ok, and the function
# reported `gone` about a process it had just seen listed.
#
# Three checks on the whole-table read, and the third is the one that covers
# the claim:
#
#   pid 4 is present -- the System process exists on every Windows NT kernel
#       from boot to shutdown. A "process table" without it is not one.
#   at least 10 entries -- the weakest of the two, and it exists only to reject
#       a fragment that happens to contain pid 4. A Windows session with a login
#       shell in it cannot have ten processes: smss, csrss, wininit, services,
#       lsass and several svchosts are running before anything user-visible.
#   the target is NOT in the table -- the whole-table read is asked the same
#       question the targeted read was just asked, and the two must agree.
#       Disagreement is `unmeasurable`: two reads of the same fact that
#       contradict each other are not a measurement of it, and the one that
#       says "gone" does not win by default.
#
# The first two are independent of each other by construction, so a control can
# revert either alone; scripts/negative-controls/windows-probes-negative-controls.py
# isolates them with a long table missing pid 4 and a short table containing it.
#
# MEASURED ON THIS HOST: `Get-Process` returns 288 entries; Id 4 is `System` and
# Id 0 is `Idle`, both present.
#
# AND A FOURTH CHECK, WHICH IS THE ONLY ONE THAT IS INDEPENDENT.
#
# Round-3 review adjudicated the three above precisely, and the adjudication is
# correct: they CAN fail -- Ok=$false when the whole-table call throws, when
# pid 4 is absent, when fewer than ten rows appear, or when the table
# contradicts the targeted read -- so they are not tautologically true. But all
# three read ONE provider, the Win32 process snapshot behind
# System.Diagnostics.Process, twice. A provider that is filtered, per-session,
# access-limited or simply incomplete omits the target from the targeted call
# AND from the whole-table call, both reads agree, and the absence is ratified
# by its own cause. What those three establish, exactly, is "two Get-Process
# reads do not contradict each other" -- not "this absence was independently
# witnessed".
#
# So a second PROVIDER is asked: WMI's Win32_Process, served by the winmgmt
# service, which shares no code with Get-Process below the kernel. A stopped or
# corrupted WMI repository breaks that read and not this one; a filtered process
# snapshot breaks this one and not that one. See Get-CimProcessWitness.
#
# WHAT IS STILL NOT WITNESSED, said plainly rather than left to be discovered:
# both providers enumerate the SAME kernel process list, so an omission made by
# the kernel or by something below it hides the target from both. This witnesses
# against PROVIDER failure. Nothing available from user mode witnesses against
# the kernel.
#
# A witness with nothing to cover cannot cover anything, so being called with
# neither an Id nor a Name is itself a refusal rather than a pass.
function Get-CimProcessWitness {
    param([int]$ProcessId = 0, [string]$Name = "")
    $what = if ($Name) { "name '$Name'" } else { "pid $ProcessId" }
    if (-not $Name -and $ProcessId -le 0) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the independent witness was given no process to cover" }
    }
    try {
        $all = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)
    } catch {
        return [pscustomobject]@{ Ok = $false
            Detail = "the independent Win32_Process table could not be read ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }
    if (-not @($all | Where-Object { $_.ProcessId -eq 4 }).Count) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the Win32_Process table has $($all.Count) rows but no ProcessId 4 (System); it is not the whole table" }
    }
    if ($all.Count -lt 10) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the Win32_Process table has only $($all.Count) rows; a running Windows session has far more" }
    }
    # Win32_Process spells the name with its extension; Get-Process does not.
    $present = if ($Name) { @($all | Where-Object { $_.Name -ieq $Name -or $_.Name -ieq ($Name + ".exe") }) }
               else { @($all | Where-Object { $_.ProcessId -eq $ProcessId }) }
    if ($present.Count -ne 0) {
        return [pscustomobject]@{ Ok = $false
            Detail = ("Get-Process reported no $what, but WMI's Win32_Process table of $($all.Count) rows CONTAINS it (pid " +
                      (($present | ForEach-Object { $_.ProcessId }) -join ", ") + "); two INDEPENDENT providers contradict each other, so neither is a measurement") }
    }
    return [pscustomobject]@{ Ok = $true
        Detail = "WMI's Win32_Process table ($($all.Count) rows, ProcessId 4 present) independently has no $what either" }
}
# The witness above only ever answers "and it is absent there too", so it can
# only corroborate a TOTAL absence. Round-4 review found the gap that leaves:
# if `Get-Process -Name wenlan-server` SUCCEEDS but returns an incomplete
# non-empty set, the pre-run snapshot silently omits a pid -- and a
# pre-existing, same-image daemon that is missing from that snapshot is later
# classified as one this run created, and killed. A partial success was not
# corroborated at all, which is this file's whole defect class arriving through
# the one door still open to it.
#
# So the positive path gets its own independent read: the SET of pids, from
# WMI, to compare against. Same provider independence as above, same stated
# limit -- both enumerate the same kernel table.
function Get-CimProcessSet([string]$Name) {
    try {
        $all = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)
    } catch {
        return [pscustomobject]@{ Ok = $false; Pids = @()
            Detail = "the independent Win32_Process table could not be read ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }
    if (-not @($all | Where-Object { $_.ProcessId -eq 4 }).Count) {
        return [pscustomobject]@{ Ok = $false; Pids = @()
            Detail = "the Win32_Process table has $($all.Count) rows but no ProcessId 4 (System); it is not the whole table" }
    }
    if ($all.Count -lt 10) {
        return [pscustomobject]@{ Ok = $false; Pids = @()
            Detail = "the Win32_Process table has only $($all.Count) rows; a running Windows session has far more" }
    }
    $hit = @($all | Where-Object { $_.Name -ieq $Name -or $_.Name -ieq ($Name + ".exe") })
    return [pscustomobject]@{ Ok = $true
        Pids = @(@($hit | ForEach-Object { [int]$_.ProcessId }) | Sort-Object)
        Detail = "WMI's Win32_Process table ($($all.Count) rows, ProcessId 4 present) independently reports $($hit.Count) '$Name' process(es)" }
}

function Get-ProcessTableWitness {
    param([int]$Id = 0, [string]$Name = "")
    $what = if ($Name) { "name '$Name'" } else { "pid $Id" }
    if (-not $Name -and $Id -le 0) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the witness was given no process to cover, so it can only report on the table's shape, not on the absence being claimed" }
    }
    try {
        $all = @(Get-Process -ErrorAction Stop)
    } catch {
        return [pscustomobject]@{ Ok = $false
            Detail = "the process table could not be read ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }
    if (-not @($all | Where-Object { $_.Id -eq 4 }).Count) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the process table has $($all.Count) entries but no pid 4 (System); it is not the whole table" }
    }
    if ($all.Count -lt 10) {
        return [pscustomobject]@{ Ok = $false
            Detail = "the process table has only $($all.Count) entries; a running Windows session has far more" }
    }
    # The claim being ratified is "$what is not running". Ask the table.
    $present = if ($Name) { @($all | Where-Object { $_.ProcessName -ieq $Name }) }
               else { @($all | Where-Object { $_.Id -eq $Id }) }
    if ($present.Count -ne 0) {
        return [pscustomobject]@{ Ok = $false
            Detail = ("the targeted read said there is no $what, but this table of $($all.Count) processes CONTAINS it (pid " +
                      (($present | ForEach-Object { $_.Id }) -join ", ") + "); the two reads contradict each other, so neither is a measurement") }
    }
    # Everything above is one provider read twice. The claim is only
    # independently witnessed if a DIFFERENT provider says the same thing.
    $cim = Get-CimProcessWitness -ProcessId $Id -Name $Name
    if (-not $cim.Ok) {
        return [pscustomobject]@{ Ok = $false
            Detail = "$($all.Count) Get-Process entries, pid 4 present, and no $what among them -- but that is one provider read twice, and the independent read does not ratify it: $($cim.Detail)" }
    }
    return [pscustomobject]@{ Ok = $true
        Detail = "$($all.Count) processes, pid 4 present, and no $what among them; $($cim.Detail)" }
}

# Tri-state liveness probe: alive / gone / unmeasurable.
#
# The form this replaces was `Get-Process ... -ErrorAction SilentlyContinue`,
# used to decide both `app-exited-after-kill` and `sidecar-exits-after-app`:
#
#     if (-not (Get-Process -Id $App.Id -ErrorAction SilentlyContinue)) { ...exited... }
#     $left = @(Get-Process -Name wenlan-server -ErrorAction SilentlyContinue)
#
# A process-table read that FAILED produced exactly what a dead process
# produces -- nothing -- so one broken read passed BOTH rows at once: the app
# was certified as having exited and the sidecar as having followed it, with
# neither process observed. The two rows are supposed to be independent
# evidence; swallowing the error made them the same non-observation twice.
#
# MEASURED ON THIS HOST (Windows PowerShell 5.1.26100.9278) rather than assumed:
#
#   Get-Process -Name <absent> -ErrorAction Stop
#       -> Microsoft.PowerShell.Commands.ProcessCommandException
#          FullyQualifiedErrorId NoProcessFoundForGivenName,...GetProcessCommand
#          CategoryInfo ObjectNotFound
#   Get-Process -Id 999999 -ErrorAction Stop
#       -> the same exception type, FullyQualifiedErrorId
#          NoProcessFoundForGivenId,...GetProcessCommand
#   Get-Process -Name <present> -ErrorAction Stop  -> returns, does not throw
#
# The error id is checked as well as the type, deliberately. ProcessCommandException
# is not raised only for absence -- it also carries "the process has exited"
# races and access failures -- so the TYPE alone would fold a different failure
# back into the negative this function exists to isolate. Only these two ids
# mean "asked, and there is no such process".
function Get-ProcessLiveness {
    param([int]$Id = 0, [string]$Name = "")
    $what = if ($Name) { "name '$Name'" } else { "pid $Id" }
    try {
        $found = if ($Name) { @(Get-Process -Name $Name -ErrorAction Stop) }
                 else { @(Get-Process -Id $Id -ErrorAction Stop) }
        # Success with nothing in hand is state three, not an absence: a query
        # that matched returns objects and one that did not throws, so silence
        # here means this is not the cmdlet we think it is.
        if ($found.Count -eq 0) {
            return [pscustomobject]@{ State = "unmeasurable"
                Detail = "Get-Process for $what returned no error and no process; silence is not absence" }
        }
        return [pscustomobject]@{ State = "alive"
            Detail = "$what is running (pid " + (($found | ForEach-Object { $_.Id }) -join ", ") + ")" }
    } catch {
        $ex = $_.Exception
        $fqid = "$($_.FullyQualifiedErrorId)"
        # The type is compared BY NAME, not with a `-is [Type]` literal. A type
        # literal is resolved at run time and THROWS when its assembly is not
        # loaded -- `Unable to find type [Microsoft.PowerShell.Commands.ProcessCommandException]`
        # -- and that throw happens inside this catch, i.e. inside the very
        # classifier whose job is to make a failure legible. The negative
        # controls caught exactly that: the classifier died where it was
        # supposed to report. A probe that can fail while deciding whether a
        # probe failed has not left the defect class. String comparison cannot.
        $typeName = if ($null -ne $ex) { $ex.GetType().FullName } else { "" }
        $isAbsence = ($typeName -eq "Microsoft.PowerShell.Commands.ProcessCommandException") -and
                     ($fqid -like "NoProcessFoundForGivenName,*" -or $fqid -like "NoProcessFoundForGivenId,*")
        if (-not $isAbsence) {
            return [pscustomobject]@{ State = "unmeasurable"
                Detail = "Get-Process for $what failed without answering ($($ex.GetType().FullName), id '$fqid': $($ex.Message))" }
        }
        # The cmdlet says there is no such process. Believe it only if the table
        # it read is a table AND that table says the same thing about THIS
        # process: "not in it" is a measurement only when "it" is the whole
        # thing and the question asked of it is the question being answered.
        $witness = Get-ProcessTableWitness -Id $Id -Name $Name
        if (-not $witness.Ok) {
            return [pscustomobject]@{ State = "unmeasurable"
                Detail = "Get-Process reported no $what, but that absence is not ratified: $($witness.Detail)" }
        }
        return [pscustomobject]@{ State = "gone"; Detail = "no process for $what ($($witness.Detail))" }
    }
}

# --- WHAT THIS RUN MAY KILL ------------------------------------------------
#
# The form this replaces, in the teardown below, was:
#
#     Get-Process -Name Wenlan, wenlan-server -ErrorAction SilentlyContinue |
#         Stop-Process -Force -ErrorAction SilentlyContinue
#
# It force-killed EVERY process with either name: a developer's production
# daemon, another worktree's, a hand-started one. This run started none of them.
# `-Name Wenlan` is worse than it looks -- the GUI binary is `wenlan-app.exe`,
# so that name never matched the app this script launched; what it did match is
# `wenlan.exe`, the CLI, so a developer running a `wenlan` command in another
# window had it killed by a gauntlet that was not even aiming at it.
#
# Two facts make a process this run's own, and it needs both: the IMAGE is the
# one this run installed, AND the pid was not already running when this run
# started -- because a production install of the same version has the same image
# path. Anything unmeasurable resolves to NOT OURS and is left alone.
#
# ROUND 6 NAMES WHAT THAT PAIR IS AND IS NOT. "Our image, and a pid that was not
# in the startup snapshot" is a CORRELATION with "this run started it", not the
# thing itself, and it is wrong in both directions:
#
#   Not ours, and killed. Anything that starts a server from that same path
#   AFTER the snapshot gets a pid the snapshot cannot contain. This channel
#   installs to the DOCUMENTED per-user location, so the developer's own daemon
#   restarting mid-run -- their scheduled task firing, a `wenlan` command in
#   another terminal autostarting it, or them restarting it by hand -- looks
#   exactly like this run's own sidecar.
#
#   Ours, and spared. Windows reuses pids, and the snapshot is a list of NUMBERS
#   with no birth time in it. A pre-existing server that exits during the run
#   frees its pid; if this run's own sidecar is later given that number it
#   matches $PreexistingServerPids and is left running.
#
# The safe direction is the second -- sparing is recoverable, killing the
# developer's daemon is not -- and it is the direction every unmeasurable
# already resolves to. Closing the first would need a fact the pid pair does not
# carry: the process's own start time against this run's, or a parent chain back
# to this script. Neither is read here, so this is a residual, not a solved
# problem, and `teardown-sidecar-gone` grades the outcome without claiming
# otherwise.
function Get-OwnedProcessesByImage {
    param([Parameter(Mandatory)][string]$Name, [string]$ImagePath = "", $ExcludePids)
    if (-not $ImagePath) {
        return [pscustomobject]@{ State = "no-licence"; Processes = @()
            Detail = "this run never established an owned image path for '$Name', so no such process can be shown to be its own; nothing killed" }
    }
    if ($null -eq $ExcludePids) {
        return [pscustomobject]@{ State = "no-licence"; Processes = @()
            Detail = "the '$Name' processes already running when this run started were never measured, so none can be shown to be its own; nothing killed" }
    }
    $skip = @($ExcludePids)
    $found = @()
    try { $found = @(Get-Process -Name $Name -ErrorAction Stop) }
    catch {
        # THE HOLE ROUND-5 REVIEW FOUND HERE: this used to return an empty
        # Processes list for EVERY failure, so access denied and a broken
        # provider produced the same answer as "there is no such process" --
        # and the teardown then went on to delete directories on the strength
        # of a measurement that never happened. The two are separated now, by
        # the same error id Get-ProcessLiveness keys on, and the absence is
        # ratified by the INDEPENDENT provider before it is believed.
        $ex = $_.Exception
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
            Detail = "no '$Name' process is running; $($w.Detail)" }
    }
    # Success with nothing in hand is state three, not an absence: a query that
    # matched returns objects and one that did not throws.
    if ($found.Count -eq 0) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "Get-Process for '$Name' returned no error and no process; silence is not absence" }
    }
    # ROUND 6, A4. A SUCCESSFUL non-empty read is corroborated too, not just a
    # failed one. `Get-CimProcessWitness` above is reachable only from the
    # absence arm, so everything it establishes is about a TOTAL absence -- and
    # the claim this function actually makes on the path below is a claim about
    # a SET: these pids are this run's, those are not. A merely SHORT read is
    # the dangerous shape here, not the loud one, and it is dangerous in the
    # direction that matters: a pre-existing same-image daemon missing from
    # $ExcludePids is classified as one this run started, and killed. The zip
    # channel's Get-ServerProcessInventory has had this since round 4; this
    # function was the door still open.
    #
    # Set equality, in both directions. A pid WMI has and Get-Process does not
    # is the omission that gets a stranger's daemon killed; a pid Get-Process
    # has and WMI does not is the same disagreement with the providers swapped,
    # and neither is a measurement. Same provider independence as the witness
    # above, and the same stated limit: both enumerate one kernel table.
    $cs = Get-CimProcessSet $Name
    if (-not $cs.Ok) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = "Get-Process found $($found.Count) '$Name' process(es), but that set is not corroborated: $($cs.Detail); nothing killed and nothing may be deleted on the strength of it" }
    }
    $gp = @(@($found | ForEach-Object { [int]$_.Id }) | Sort-Object)
    $missing = @($cs.Pids | Where-Object { $gp -notcontains $_ })
    $extra = @($gp | Where-Object { $cs.Pids -notcontains $_ })
    if ($missing.Count -ne 0 -or $extra.Count -ne 0) {
        return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
            Detail = ("the two providers do not agree on which '$Name' processes exist -- Get-Process has [" + ($gp -join ",") +
                      "], WMI has [" + (@($cs.Pids) -join ",") + "]" +
                      $(if ($missing.Count) { "; WMI sees pid(s) " + ($missing -join ",") + " that Get-Process omitted, and a pid missing from a snapshot like this is later mistaken for one this run started" } else { "" }) +
                      $(if ($extra.Count) { "; Get-Process sees pid(s) " + ($extra -join ",") + " that WMI does not" } else { "" }) +
                      "; nothing killed and nothing may be deleted on the strength of it") }
    }
    $rows = New-Object System.Collections.Generic.List[object]
    $spared = New-Object System.Collections.Generic.List[string]
    foreach ($p in $found) {
        $path = ""
        try { $path = "$($p.Path)" } catch { $path = "" }
        # An image that cannot be read is an identity that cannot be proved,
        # and a '$Name' this run cannot identify may be its own -- holding the
        # very tree the teardown is about to delete. Sparing it and carrying on
        # was the same non-measurement in a quieter place; it is unmeasurable.
        if (-not $path) {
            return [pscustomobject]@{ State = "unmeasurable"; Processes = @()
                Detail = "pid $($p.Id) is a '$Name' whose image cannot be read, so this run can show neither that it is its own nor that it is not; nothing killed and nothing deleted" }
        }
        if ($skip -contains $p.Id) { $spared.Add("pid $($p.Id) (was already running before this run)"); continue }
        if (-not [string]::Equals($path, $ImagePath, [System.StringComparison]::OrdinalIgnoreCase)) {
            $spared.Add("pid $($p.Id) ($path is not $ImagePath)"); continue
        }
        $rows.Add([pscustomobject]@{ Id = $p.Id; Path = $path })
    }
    $ownedText = if ($rows.Count) { (($rows | ForEach-Object { "pid $($_.Id)" }) -join ", ") } else { "none" }
    $sparedText = if ($spared.Count) { ($spared -join ", ") } else { "none" }
    # .ToArray(), NOT @($rows) -- on Windows PowerShell 5.1 the array
    # subexpression operator THROWS System.ArgumentException ("Argument types do
    # not match") on a Generic.List[object], and only on that. See the same note
    # in windows-zip.ps1's Get-ServerProcessInventory.
    return [pscustomobject]@{ State = "measured"; Processes = $rows.ToArray()
        Detail = "owned: $ownedText; left alone: $sparedText" }
}

# Kill ONE process, by pid, and only while it is still the image the caller
# identified. Ported from kill_by_image_path in scripts/lib/host-process.sh:
# checking the path and then killing by pid leaves a reuse window open -- the
# target can exit and Windows can hand its pid to one of the neighbours this
# exists to protect. Opening a handle first pins the pid, so the path read here
# and the process killed after it are the same one. An identity this cannot
# prove kills nothing, and nothing here throws: the receipt is diagnostics, and
# the verdict on an exit is the liveness poll.
function Stop-ProcessByImage {
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

# Tri-state: `Test-Path` answers $false for BOTH "not there" and "there, but
# I was not allowed to look", and a delete licensed by the second reading is
# a delete of a tree this run never made.
#
# The Path travels WITH the answer. This channel reads the documented install
# path before the installer runs and then DISCOVERS the real one by searching,
# so a pre-read and the tree it is later spent on are two different variables
# of the same shape; carrying the path lets New-OwnerMark refuse a pre-read
# that is about somewhere else instead of silently honouring it.
function Get-DirPresence([string]$Path) {
    try {
        $item = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        return [pscustomobject]@{ State = "present"; CreatedUtc = $item.CreationTimeUtc; Path = $Path
            Detail = "$Path exists, created $($item.CreationTimeUtc.ToString('o'))" }
    } catch [System.Management.Automation.ItemNotFoundException] {
        return [pscustomobject]@{ State = "absent"; CreatedUtc = $null; Path = $Path; Detail = "$Path does not exist" }
    } catch {
        return [pscustomobject]@{ State = "unmeasurable"; CreatedUtc = $null; Path = $Path
            Detail = "could not read $Path -- $($_.Exception.Message)" }
    }
}

# Every file under a tree, by relative path and SHA-256. Two states: taken, or
# unmeasurable. A PARTIAL snapshot is unmeasurable rather than short, because
# the files it failed to read are exactly the ones a later comparison would
# report as untouched.
#
# The ownership marker is left out: it is this run's file, not the user's, and
# it is DeleteOnClose, so it is present in the pre-read and absent from any
# read taken after the handle goes.
function Get-TreeFileDigests([string]$Root) {
    $files = @{}
    try {
        $items = @(Get-ChildItem -LiteralPath $Root -Recurse -File -Force -ErrorAction Stop)
    } catch {
        return [pscustomobject]@{ State = "unmeasurable"; Files = $files
            Detail = "$Root could not be enumerated ($($_.Exception.GetType().FullName): $($_.Exception.Message))" }
    }
    $prefix = "$Root".TrimEnd('\') + '\'
    foreach ($f in $items) {
        $full = "$($f.FullName)"
        $rel = if ($full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            $full.Substring($prefix.Length)
        } else { $full }
        if ($rel -eq $script:OwnerMarkerName) { continue }
        try { $h = Get-FileHash -LiteralPath $full -Algorithm SHA256 -ErrorAction Stop }
        catch {
            return [pscustomobject]@{ State = "unmeasurable"; Files = $files
                Detail = "$full could not be hashed ($($_.Exception.GetType().FullName): $($_.Exception.Message)); a snapshot missing a file cannot say that file survived" }
        }
        $files[$rel] = "$($h.Hash)"
    }
    return [pscustomobject]@{ State = "taken"; Files = $files
        Detail = "$($files.Count) files under $Root, SHA-256 each, the ownership marker excluded" }
}

# --- WHAT BINDS A TREE TO THIS RUN -----------------------------------------
#
# "Absent before AND present after" is a SEQUENCE OF STATES, and round-5 review
# named exactly what it is not: causation. Two different races fit inside it.
#
#   REPLACEMENT -- this run creates the tree; something removes and recreates
#       it before the teardown; the teardown deletes the replacement. Both
#       reads were true and both were about a different directory.
#   CREATION -- another process, most plausibly the developer launching Wenlan
#       while the gauntlet runs, creates %LOCALAPPDATA%\wenlan between the
#       pre-read and the post-read, and the post-read sees a tree this run did
#       not make.
#
# ROUND 6 CHANGED WHAT THE SECOND HALF IS, in both channels. It used to be a
# second READ -- `$pre.State -eq "absent" -and (Get-DirPresence $X).State -eq
# "present"` -- and then, on the NEXT statement, a marker was written into
# whatever was there. Two operations, and review found the schedule that fits
# between them: the post-read sees this run's tree, the tree is replaced, the
# marker is written into the REPLACEMENT, and the teardown verifies its own
# marker and deletes a stranger's directory. The marker was supposed to close
# the replacement race and it was written on the far side of it.
#
# THE CLAIM IS NOW ONE ACT. `New-OwnerMark` opens the marker file with
# FileMode.CreateNew, which is the filesystem's own atomic
# create-if-and-only-if-absent -- there is no window between deciding the file
# is not there and making it. That single call is BOTH halves of the licence:
#
#   It is the post-read. CreateNew under a path that does not exist raises
#       DirectoryNotFoundException, so a statement that was supposed to create
#       the tree and did not is reported as "there is nothing here to own"
#       rather than being papered over by a separate Get-DirPresence.
#   It is the mark. A marker that already exists raises IOException with
#       ERROR_FILE_EXISTS, which means another run -- concurrent, or one that
#       died holding the tree -- claimed this path first. This run does not
#       claim it as well.
#
# AND THE HANDLE IS HELD FOR THE LIFE OF THE RUN. The stream stays open, with
# FileOptions.DeleteOnClose and with FileShare.Delete deliberately WITHHELD.
# MEASURED ON THIS HOST (Windows PowerShell 5.1.26100.9278), against a scratch
# tree, with the marker held exactly as this code holds it:
#
#   Remove-Item -Recurse -Force <tree>  -> System.IO.IOException, the tree
#                                          survived, the marker survived
#   Rename-Item <tree>                  -> System.IO.IOException, refused
#   a second CreateNew on the marker    -> IOException 0x80070050 (exists)
#   reading the marker THROUGH ITS PATH -> the GUID, byte for byte
#   Dispose()                           -> the marker was gone afterwards
#
# and, run again with FileShare.Delete GRANTED on the writer, the same
# `Remove-Item -Recurse -Force` DELETED THE WHOLE TREE while the handle was
# open. That is the control on the choice: the withheld Delete is what makes
# the replacement impossible rather than merely detectable.
#
# DeleteOnClose is also what closes the marker LEAK. The previous version wrote
# an ordinary file and, on any failure after the write, returned `Ok = $false`
# without removing it -- so a tree this run declined to own kept a marker for
# ever, and every later run refused to claim a directory it had created itself.
# Here the kernel removes the file when the handle goes, including when this
# process is killed. `Close-OwnerMark` still CHECKS that it went, because
# "the kernel does this" is a claim about a mechanism.
#
# THE INSTALL DIR IS MARKED TOO, AND RELEASED ON PURPOSE. Round 5 gave it a
# creation-timestamp check instead, on the reasoning that a marker file left
# inside %LOCALAPPDATA%\Programs\Wenlan would make the uninstaller fail and
# `uninstall-removes-dir` -- the row this whole channel exists to produce --
# would grade this script rather than the product. The reasoning about the
# uninstaller was right and the remedy was not: NTFS FILE-SYSTEM TUNNELING
# gives a directory recreated under the same name within about fifteen seconds
# the creation timestamp of the one it replaced, so the check answered
# "verified" for exactly the replacement it existed to catch. It is gone. The
# install dir carries a real marker, and the marker is RELEASED immediately
# before `uninstall-silent`, which removes the file before the uninstaller ever
# sees it. After that release the mark can establish one thing about the path,
# and it is the one the teardown backstop needs: whether anything is still
# standing there.
#
# WHAT IS STILL NOT CLOSED, stated here rather than left to be discovered:
#
#   CREATION IS NARROWED TO THE CREATING STATEMENT, NOT CLOSED. The claim is
#       taken on the statement after the one that creates the tree, so the
#       window is that statement alone. For the data dir that is the silent
#       installer's own run; for the install dir it is the same. If something
#       else creates %LOCALAPPDATA%\wenlan inside the installer's runtime and
#       the installer then finds it already there, this run marks and later
#       deletes a tree it did not make. The filesystem does not record which
#       process created a directory, so nothing available here makes that
#       window zero. It is a real residual and it is smaller than the whole-run
#       window it replaced.
#
#   THE RELEASE-TO-DELETE WINDOW. The handle has to be closed before the tree
#       can be removed, so between `Close-OwnerMark` and `Remove-Retry` the
#       tree is unprotected. It is a few statements wide and it is not zero.
#
#   THE INSTALL DIR AFTER THE RELEASE. From `uninstall-silent` onward nothing
#       binds that path to this run at all -- deliberately, so the uninstaller
#       is graded on its own. The teardown backstop therefore deletes NOTHING
#       there once the release has happened; it can only report the tree gone
#       or refuse.
#
#   IDENTITY, NOT CONTENTS. A verified marker says the tree at this path is the
#       one this run created. It says nothing about what else is inside it.
$OwnerMarkerName = ".wenlan-first-run-owner"
# ERROR_FILE_EXISTS as .NET reports it. MEASURED on this host: a CreateNew over
# an existing file raises System.IO.IOException with HResult -2147024816
# (0x80070050). The HResult is the only usable key -- the same exception TYPE
# also carries the sharing violation (0x80070020) and the access denial
# (0x80070005), and the MESSAGE that separates them is localised.
$OwnerMarkExistsHResult = -2147024816

# Read the marker THROUGH ITS PATH. That is the whole point of this function:
# this run holds a handle to the marker, and reading through that handle would
# only establish that the handle is still open. What has to be established is
# that the file AT THAT PATH is still the one this run created.
#
# The share flags are a measurement, not a guess. DeleteOnClose makes .NET open
# the marker with DELETE access on top of ReadWrite, so a reader that does not
# itself grant FileShare.Delete is refused by the sharing check -- MEASURED
# here, both `[System.IO.File]::ReadAllText` and a FileStream opened with
# FileShare.ReadWrite failed with 0x80070020 against this run's OWN marker.
# Granting Delete in the READER changes nothing about who may delete the file;
# only the writer's share flags decide that, and those withhold it.
#
# Tri-state: read / gone / unmeasurable. `gone` is the marker or its tree not
# being there, which is a fact about the tree; anything else is a failure to
# look, which is a fact about this probe.
function Get-OwnerMarkText([string]$File) {
    $share = ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete)
    $stream = $null
    $reader = $null
    try {
        $stream = New-Object System.IO.FileStream ($File, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, $share)
        $reader = New-Object System.IO.StreamReader ($stream, [System.Text.Encoding]::UTF8)
        return [pscustomobject]@{ State = "read"; Text = "$($reader.ReadToEnd())".Trim()
            Detail = "$File was read back through its own path" }
    } catch {
        # MEASURED ON THIS HOST: PowerShell wraps a .NET constructor's exception
        # in System.Management.Automation.MethodInvocationException exactly as it
        # wraps a method's, so the type that says WHICH failure this is sits one
        # level in -- MethodInvocationException -> System.IO.FileNotFoundException
        # for a missing marker, -> System.IO.DirectoryNotFoundException for a
        # missing tree. Reading only the outer type puts both in the unmeasurable
        # arm, which is safe but mute, and the teardown log is where anyone finds
        # out which one happened.
        $ex = $_.Exception
        if ($ex.GetType().FullName -eq "System.Management.Automation.MethodInvocationException" -and
            $null -ne $ex.InnerException) { $ex = $ex.InnerException }
        $t = $ex.GetType().FullName
        if ($t -eq "System.IO.FileNotFoundException") {
            return [pscustomobject]@{ State = "gone"; Text = ""
                Detail = "$File is not there ($t): $($ex.Message)" }
        }
        if ($t -eq "System.IO.DirectoryNotFoundException") {
            return [pscustomobject]@{ State = "gone"; Text = ""
                Detail = "the tree holding $File is not there ($t): $($ex.Message)" }
        }
        return [pscustomobject]@{ State = "unmeasurable"; Text = ""
            Detail = "$File could not be read ($t, hresult $($ex.HResult)): $($ex.Message)" }
    } finally {
        # Disposing the reader disposes the stream under it; if the reader was
        # never built, the stream still has to go.
        if ($null -ne $reader) { try { $reader.Dispose() } catch { Write-Host "teardown: the marker read handle on $File would not close: $($_.Exception.Message)" } }
        elseif ($null -ne $stream) { try { $stream.Dispose() } catch { Write-Host "teardown: the marker read handle on $File would not close: $($_.Exception.Message)" } }
    }
}

# CLAIM a tree for this run, in one act. Tri-state -- owned / not-owned /
# unmeasurable -- and every caller branches on all three.
#
# $Pre is the pre-read of THIS path, and it is a parameter rather than a
# file-scope lookup so that the pair cannot drift apart. THIS CHANNEL IS WHERE
# THAT MATTERS: it reads the DOCUMENTED install path before the installer runs
# and then DISCOVERS the real one afterwards by searching, so "the pre-read"
# and "the tree being claimed" are two different variables that look
# interchangeable. The mark refuses a pre-read taken against a different path.
function New-OwnerMark([string]$Path, $Pre) {
    $file = Join-Path $Path $script:OwnerMarkerName
    if ($null -eq $Pre) {
        return [pscustomobject]@{ State = "not-owned"; Guid = ""; File = $file; Path = $Path; Stream = $null
            Detail = "$Path was never read before this run created anything, so nothing can show the tree there now is this run's" }
    }
    if (-not [string]::Equals("$($Pre.Path)".TrimEnd('\'), "$Path".TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase)) {
        return [pscustomobject]@{ State = "not-owned"; Guid = ""; File = $file; Path = $Path; Stream = $null
            Detail = "the pre-read offered for $Path is about '$($Pre.Path)'; a licence taken out on one path is not a licence over another" }
    }
    if ($Pre.State -ne "absent") {
        return [pscustomobject]@{ State = "not-owned"; Guid = ""; File = $file; Path = $Path; Stream = $null
            Detail = "$Path was '$($Pre.State)' before this run ($($Pre.Detail)), so this run did not create it and may not delete it" }
    }
    $guid = [guid]::NewGuid().ToString()
    $stream = $null
    try {
        $stream = New-Object System.IO.FileStream ($file,
            [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::ReadWrite, 4096, [System.IO.FileOptions]::DeleteOnClose)
    } catch {
        $ex = $_.Exception
        if ($ex.GetType().FullName -eq "System.Management.Automation.MethodInvocationException" -and
            $null -ne $ex.InnerException) { $ex = $ex.InnerException }
        $t = $ex.GetType().FullName
        if ($t -eq "System.IO.DirectoryNotFoundException") {
            return [pscustomobject]@{ State = "not-owned"; Guid = ""; File = $file; Path = $Path; Stream = $null
                Detail = "$Path is not there, so the step that was supposed to create it did not; there is nothing here for this run to own" }
        }
        if ($t -eq "System.IO.IOException" -and $ex.HResult -eq $script:OwnerMarkExistsHResult) {
            return [pscustomobject]@{ State = "not-owned"; Guid = ""; File = $file; Path = $Path; Stream = $null
                Detail = "$file already exists, so the tree at $Path is already claimed -- by a concurrent run, or by one that died holding it; this run will not claim it as well" }
        }
        return [pscustomobject]@{ State = "unmeasurable"; Guid = ""; File = $file; Path = $Path; Stream = $null
            Detail = "the ownership marker $file could not be created ($t, hresult $($ex.HResult): $($ex.Message)); this run will not delete a tree it could not claim" }
    }
    $failure = ""
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($guid)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush()
    } catch {
        $failure = "the ownership marker $file was created but could not be written ($($_.Exception.GetType().FullName): $($_.Exception.Message))"
    }
    if (-not $failure) {
        $back = Get-OwnerMarkText $file
        if ($back.State -ne "read") {
            $failure = "the ownership marker $file did not read back through its own path: $($back.Detail)"
        } elseif ($back.Text -ne $guid) {
            $failure = "the ownership marker $file reads back as '$($back.Text)', not the '$guid' just written"
        }
    }
    if ($failure) {
        # A marker written and then NOT owned must not be left behind: that is
        # the marker residue, and it is a leak that makes every later run read
        # this tree as pre-existing. Dispose IS the removal here, because of
        # DeleteOnClose -- and a Dispose that failed is reported rather than
        # swallowed, which is the other half of the same finding.
        $rid = "the half-made marker went with its handle"
        try { $stream.Dispose() }
        catch { $rid = "AND THE HANDLE WOULD NOT CLOSE ($($_.Exception.Message)), so $file MAY STILL BE THERE" }
        return [pscustomobject]@{ State = "unmeasurable"; Guid = ""; File = $file; Path = $Path; Stream = $null
            Detail = "$failure; $rid" }
    }
    return [pscustomobject]@{ State = "owned"; Guid = $guid; File = $file; Path = $Path; Stream = $stream
        Detail = "$file was created exclusively by this run, carries $guid, and is HELD OPEN, so $Path cannot be removed or renamed while this run holds it" }
}

# Is the tree at $Path still the tree this run claimed? Asked IMMEDIATELY
# before the release-and-delete, not once at the top of the teardown.
#
# Every state but `verified` and `gone` means DO NOT DELETE. The exception
# types inside Get-OwnerMarkText are compared BY NAME for the reason
# Get-ProcessLiveness gives at length: a type literal is resolved at run time
# and throws when its assembly is not loaded, and a throw inside the classifier
# is the classifier failing at the one job it has.
function Test-OwnerMark($Mark, [string]$Path) {
    if ($null -eq $Mark) {
        return [pscustomobject]@{ State = "unmarked"
            Detail = "no ownership marker was ever attempted for this tree, so nothing binds it to this run" }
    }
    if ($Mark.State -eq "released") {
        # The install dir's claim is given up ON PURPOSE immediately before the
        # uninstaller runs, so that `uninstall-removes-dir` grades the product
        # and not a file this script is holding open. After that there is
        # exactly one thing left to establish about the path, and it happens to
        # be the one the teardown backstop needs.
        $now = Get-DirPresence $Mark.Path
        if ($now.State -eq "absent") {
            return [pscustomobject]@{ State = "gone"
                Detail = "$($Mark.Path) is gone, which is what this run released its claim on it for: the uninstaller was left free to remove it" }
        }
        if ($now.State -eq "present") {
            return [pscustomobject]@{ State = "released"
                Detail = "this run gave up its claim on $($Mark.Path) before the uninstaller ran, and something is standing there now; nothing binds that tree to this run any more, so it is not this run's to delete" }
        }
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "this run released its claim on $($Mark.Path) and cannot now read whether anything is there -- $($now.Detail)" }
    }
    if ($Mark.State -ne "owned" -or $null -eq $Mark.Stream) {
        return [pscustomobject]@{ State = "unmarked"
            Detail = "this run never claimed this tree ($($Mark.State)): $($Mark.Detail)" }
    }
    if (-not [string]::Equals("$($Mark.Path)".TrimEnd('\'), "$Path".TrimEnd('\'), [System.StringComparison]::OrdinalIgnoreCase)) {
        return [pscustomobject]@{ State = "not-this-run"
            Detail = "this run's mark is on '$($Mark.Path)' and the tree about to be deleted is '$Path'; a claim on one path is not a claim on another" }
    }
    $got = Get-OwnerMarkText $Mark.File
    if ($got.State -eq "gone") {
        return [pscustomobject]@{ State = "not-this-run"
            Detail = "the marker this run created is no longer at $($Mark.File) ($($got.Detail)); whatever is at $Path now is not the tree this run created" }
    }
    if ($got.State -ne "read") {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "the marker $($Mark.File) could not be read: $($got.Detail)" }
    }
    if ([string]::Equals($got.Text, $Mark.Guid, [System.StringComparison]::OrdinalIgnoreCase)) {
        return [pscustomobject]@{ State = "verified"
            Detail = "$($Mark.File) still carries this run's marker $($Mark.Guid), read back through its own path" }
    }
    return [pscustomobject]@{ State = "not-this-run"
        Detail = "$($Mark.File) carries '$($got.Text)', not this run's marker $($Mark.Guid)" }
}

# Give up the claim, and MEASURE that it was given up. Four states:
# released / residue / unmeasurable / nothing-held.
#
# Two callers, for two different reasons. `uninstall-silent` releases the
# install dir so the uninstaller can remove a tree this script is not holding
# open. The teardown releases whatever is left, because a tree cannot be
# deleted while a handle is open inside it -- the lock working as designed. A
# failure to release is REPORTED, never swallowed, because the marker residue
# IS a removal that did not happen and did not say so.
function Close-OwnerMark($Mark) {
    if ($null -eq $Mark -or $Mark.State -ne "owned" -or $null -eq $Mark.Stream) {
        return [pscustomobject]@{ State = "nothing-held"
            Detail = "this run holds no marker to release$(if ($null -ne $Mark) { " (the mark is '$($Mark.State)': $($Mark.Detail))" })" }
    }
    $file = $Mark.File
    try { $Mark.Stream.Dispose() }
    catch {
        return [pscustomobject]@{ State = "unmeasurable"
            Detail = "the marker handle on $file would not close ($($_.Exception.GetType().FullName): $($_.Exception.Message)); the marker may still be there and the tree may still be locked against deletion" }
    }
    $Mark.Stream = $null
    $Mark.State = "released"
    # DeleteOnClose is a kernel flag, not a promise this script keeps, so it is
    # CHECKED rather than assumed.
    $after = Get-OwnerMarkText $file
    if ($after.State -eq "gone") {
        return [pscustomobject]@{ State = "released"; Detail = "$file went with its handle, as DeleteOnClose requires" }
    }
    if ($after.State -eq "read") {
        return [pscustomobject]@{ State = "residue"
            Detail = "$file is STILL THERE after its handle was closed; this run has left a marker behind, the uninstaller may now fail to remove the tree because of it, and the next run will refuse to claim that tree" }
    }
    return [pscustomobject]@{ State = "unmeasurable"
        Detail = "whether $file went with its handle could not be read: $($after.Detail)" }
}

# A delete that failed ten times was indistinguishable from one that succeeded:
# every error was swallowed, nothing was returned, and the caller's next line --
# and the row that grades the teardown -- read a silent exhaustion as a removal.
# Round-5 review's C4 turns on exactly that. It reports now, in three states,
# and both callers require the report.
function Remove-Retry([string]$Target) {
    $lastError = ""
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try { Remove-Item -Recurse -Force -LiteralPath $Target -ErrorAction Stop }
        catch { $lastError = "$($_.Exception.GetType().FullName): $($_.Exception.Message)" }
        # Test-Path answers $false for "there, but I was not allowed to look",
        # which is how the old loop could return on a tree it never touched.
        $after = Get-DirPresence $Target
        if ($after.State -eq "absent") {
            return [pscustomobject]@{ State = "removed"; Attempts = $attempt
                Detail = "$Target was measured gone after attempt $attempt$(if ($lastError) { " (last error before it went: $lastError)" })" }
        }
        if ($after.State -eq "unmeasurable") {
            return [pscustomobject]@{ State = "unmeasurable"; Attempts = $attempt
                Detail = "after attempt $attempt, whether $Target is gone could not be read -- $($after.Detail)$(if ($lastError) { "; last delete error: $lastError" })" }
        }
        if ($attempt -lt 10) { Start-Sleep -Milliseconds 500 }
    }
    return [pscustomobject]@{ State = "failed"; Attempts = 10
        Detail = "$Target is still there after 10 delete attempts over about 5s; last error: $(if ($lastError) { $lastError } else { "none was reported" })" }
}

# The rows this channel owes, declared before the run rather than derived from
# it. The helpers declare their own (cli-*, mcp-*) from their own inputs.
Expect-Rows -Names @(
    # The workflow's precheck step records `port-7878-precheck` before this
    # script starts, so that row is carried in; Record-CarriedRow below restates
    # its verdict as a row of this run's.
    "port-7878-precheck-carried",
    "download-setup",
    "nsis-silent-install",
    "app-exe-found",
    "install-dir-outside-data-root",
    "uninstall-key-present",
    "uninstall-display-version",
    "bundled-binaries",
    "app-alive-30s",
    "health-version",
    "sidecar-parent-is-app",
    "cli-roundtrip-driver",
    "mcp-roundtrip-driver",
    "doctor (bundled wenlan.exe doctor)",
    "app-exited-after-kill",
    "sidecar-exits-after-app",
    "uninstall-silent",
    "uninstall-removes-dir",
    "uninstall-removes-registry-key",
    "user-data-survives-uninstall",
    # Recorded from the `finally`, so they exist on every run including one that
    # aborted on its first statement. `sidecar-sweep-measured` grades the READ
    # -- can this run account for its own daemons -- and the two `teardown-*`
    # rows grade the OUTCOME, which used to live only in console lines.
    "sidecar-sweep-measured",
    "teardown-app-gone",
    "teardown-sidecar-gone",
    "no-leftover-dirs"
)
Record-CarriedRow -Name "port-7878-precheck"

try {
    New-Item -ItemType Directory -Force -Path $Work | Out-Null
    Info "documented-flow" "run $SetupName from the Releases page (gauntlet runs it as: $SetupName /S)"
    Check -Name "download-setup" -Script { if (-not (Get-Asset $SetupName $Setup)) { throw "download failed after 3 attempts" }; Write-Output ("bytes=" + (Get-Item $Setup).Length) }
    # THE PRE-READS SIT HERE, on the statement before the installer, and that
    # placement is the whole of round-6's A2 in this channel. They used to be
    # the first two lines of the try block, which put the ENTIRE download --
    # three attempts, a multi-megabyte transfer, minutes of wall clock -- inside
    # the window during which somebody else could create one of these trees and
    # have this run claim it. The window is now the installer's own run, which
    # is the narrowest it goes here: the installer is the thing that creates
    # both trees, so the claim cannot be taken any earlier than after it.
    $preDataDir = Get-DirPresence $DataDir
    Info "data-dir-before-run" "$($preDataDir.State) -- $($preDataDir.Detail)"
    $preInstall = Get-DirPresence $InstallCandidate
    Info "install-dir-before-run" "$($preInstall.State) -- $($preInstall.Detail)"
    Check -Name "nsis-silent-install" -Script {
        $proc = Start-Process -FilePath $Setup -ArgumentList '/S' -Wait -PassThru
        Write-Output "installer exit=$($proc.ExitCode)"
        if ($proc.ExitCode -ne 0) { throw "installer exit $($proc.ExitCode)" }
    }
    # BOTH CLAIMS, on the statement after the installer, before anything else
    # this run does. The install dir's claim in particular used to be taken --
    # as a creation-timestamp reading -- only AFTER the depth-3 recursive search
    # for wenlan-app.exe under three roots, which is the widest window in either
    # file. It is now the installer alone, and the tree is claimed at the
    # DOCUMENTED path rather than at whatever the search finds: a run whose
    # installer landed somewhere else has claimed nothing and deletes nothing.
    #
    # A run whose installer failed owns neither tree. What this does NOT
    # establish is that the INSTALLER is what created them; see WHAT BINDS A
    # TREE TO THIS RUN for the residual that leaves.
    $DataDirMark = New-OwnerMark $DataDir $preDataDir
    Info "data-dir-claimed" "$($DataDirMark.State) -- $($DataDirMark.Detail)"
    $InstallMark = New-OwnerMark $InstallCandidate $preInstall
    Info "install-dir-claimed" "$($InstallMark.State) -- $($InstallMark.Detail)"

    # The GUI executable is wenlan-app.exe (Cargo package name; Tauri names the
    # main binary after the crate's default-run, not the productName). The
    # display name Wenlan.exe resolves case-insensitively to the CLI sidecar
    # wenlan.exe — launching that was this script's own bug in the first runs.
    # The installer hook moves the per-user default off the data root
    # (%LOCALAPPDATA%\Wenlan is %LOCALAPPDATA%\wenlan on a case-insensitive
    # filesystem); the search below still finds an older layout.
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Wenlan\wenlan-app.exe"
    if (Test-Path $candidate) { $AppExe = $candidate }
    else {
        foreach ($root in @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
            if ($AppExe -or -not $root -or -not (Test-Path $root)) { continue }
            $hit = Get-ChildItem -Path $root -Recurse -Depth 3 -Filter "wenlan-app.exe" -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { $AppExe = $hit.FullName }
        }
    }
    Check -Name "app-exe-found" -Script { if (-not $AppExe) { throw "wenlan-app.exe not found under LOCALAPPDATA or Program Files" }; Write-Output $AppExe }
    if ($AppExe) { $Install = Split-Path -Parent $AppExe }
    # The install dir's post-read, taken as soon as the path is known. It is an
    # OBSERVATION now, not a licence: the claim on this tree was taken on the
    # documented path immediately after the installer, and this read happens
    # after the depth-3 search. Nothing below decides on it, and that is the
    # point -- a reading taken this late is exactly the wide window round-6's A2
    # is about, so it goes in the log and nowhere else.
    $postInstall = if ($Install) { Get-DirPresence $Install } else { $null }
    Info "install-dir" "$Install"
    Info "install-dir-after-install" $(if ($null -eq $postInstall) { "no install dir was discovered" } else { "$($postInstall.State) -- $($postInstall.Detail)" })
    Check -Name "install-dir-outside-data-root" -Script {
        if (-not $Install) { throw "no install dir was discovered" }
        if ($Install.TrimEnd('\') -ieq $DataDir.TrimEnd('\')) { throw "app installed into the CLI data root $DataDir (finding F6)" }
        Write-Output "$Install is not $DataDir"
    }
    if ($Install) {
        Info "install-dir-exes" ((Get-ChildItem -Path $Install -Filter *.exe -File -ErrorAction SilentlyContinue | ForEach-Object { "$($_.Name) $($_.Length)" }) -join "; ")
    }

    $entry = Get-UninstallEntry
    Check -Name "uninstall-key-present" -Script {
        if ($entry.State -eq "present") { Write-Output $entry.Entry.PSPath; return }
        if ($entry.State -eq "absent") { throw "no uninstall entry with DisplayName 'Wenlan' under HKCU/HKLM: $($entry.Detail)" }
        throw "could not measure whether an uninstall entry exists: $($entry.Detail); recorded as unproven, not as missing"
    }
    Check -Name "uninstall-display-version" -Script {
        if ($entry.State -ne "present") { throw "there is no uninstall entry to read a DisplayVersion from ($($entry.State)): $($entry.Detail)" }
        if ("$($entry.Entry.DisplayVersion)" -ne $Version) { throw "DisplayVersion '$($entry.Entry.DisplayVersion)' != $Version" }
        Write-Output $entry.Entry.DisplayVersion
    }
    Info "uninstall-string" $(if ($entry.State -eq "present") { "$($entry.Entry.UninstallString)" } else { "$($entry.State) -- $($entry.Detail)" })
    Check -Name "bundled-binaries" -Script {
        if (-not $Install) { throw "no install dir" }
        $missing = @($Bundled | Where-Object { -not (Test-Path (Join-Path $Install $_)) })
        if ($missing.Count -ne 0) { throw ("missing beside wenlan-app.exe: " + ($missing -join ", ")) }
        Write-Output ("present: " + ($Bundled -join ", "))
    }

    # WebView2 is the app's renderer; a missing runtime is the classic reason a
    # Tauri app exits at once on a fresh Windows box.
    $wv2 = @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
        "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    ) | ForEach-Object { (Get-ItemProperty -Path $_ -ErrorAction SilentlyContinue).pv } | Where-Object { $_ }
    Info "webview2-runtime" $(if ($wv2) { "pv=" + ($wv2 -join ",") } else { "not registered" })

    $AppOut = Join-Path $script:GauntletOut "logs\app-stdout.log"
    $AppErr = Join-Path $script:GauntletOut "logs\app-stderr.log"
    $Launched = Get-Date
    # Measured BEFORE the app is launched, because at teardown the only thing
    # separating this run's sidecar from a daemon that was already running is
    # that the other one was already there. An absence taken from Get-Process
    # alone would be one provider ratifying itself, so it goes through the
    # independent Win32_Process read; unratified, the set stays $null and the
    # teardown kills nothing.
    $OwnedServerImage = if ($Install) { Join-Path $Install "wenlan-server.exe" } else { "" }
    # Corroborated in BOTH directions, not just on the throw. Round-4 review:
    # the witness was reached only by the total-absence exception, so a
    # `Get-Process` that succeeded with an INCOMPLETE non-empty set produced a
    # snapshot missing a pid -- and a pre-existing, same-image daemon absent
    # from this snapshot is exactly what the teardown sweep below then treats
    # as one this run started, and kills. $null here means "not measured", and
    # not measured means the sweep kills nothing.
    $PreexistingServerPids = $null
    try {
        $seen = @(@(Get-Process -Name wenlan-server -ErrorAction Stop) | ForEach-Object { [int]$_.Id }) | Sort-Object
        $cs = Get-CimProcessSet "wenlan-server"
        $disagree = @(@($cs.Pids | Where-Object { @($seen) -notcontains $_ }) +
                      @(@($seen) | Where-Object { $cs.Pids -notcontains $_ }))
        if ($cs.Ok -and $disagree.Count -eq 0) { $PreexistingServerPids = @($seen) }
        else { $script:SnapshotDoubt = "Get-Process saw [" + (@($seen) -join ",") + "] and the independent read " + $(if ($cs.Ok) { "saw [" + (@($cs.Pids) -join ",") + "]" } else { "failed: $($cs.Detail)" }) }
    }
    catch {
        $w = Get-CimProcessWitness -Name "wenlan-server"
        if ("$($_.FullyQualifiedErrorId)" -like "NoProcessFoundForGivenName,*" -and $w.Ok) { $PreexistingServerPids = @() }
        else { $script:SnapshotDoubt = "the process read failed and was not ratified: $($_.Exception.Message); $($w.Detail)" }
    }
    Info "server-processes-before-launch" $(
        if ($null -eq $PreexistingServerPids) { "COULD NOT MEASURE -- no wenlan-server process may be killed at teardown ($script:SnapshotDoubt)" }
        elseif ($PreexistingServerPids.Count) { "already running: " + ($PreexistingServerPids -join ",") + " (none of these may be killed at teardown)" }
        else { "none were running; owned image $OwnedServerImage" })
    if ($AppExe) {
        $App = Start-Process -FilePath $AppExe -WorkingDirectory $Install -PassThru `
            -RedirectStandardOutput $AppOut -RedirectStandardError $AppErr
    }
    Check -Name "app-alive-30s" -Script {
        if (-not $App) { throw "app was not launched" }
        Start-Sleep -Seconds 30
        if ($App.HasExited) { throw "wenlan-app.exe exited within 30s (exit $($App.ExitCode))" }
        Write-Output "pid=$($App.Id) alive after 30s"
    }
    if ($App -and $App.HasExited) {
        # Why it died: its own streams, its log file if it got that far, and
        # the Application event log (crash reports land there as event 1000).
        $streams = @()
        foreach ($f in @($AppOut, $AppErr)) {
            if (Test-Path $f) { $streams += ("[" + (Split-Path -Leaf $f) + "] " + ((Get-Content $f -Raw -ErrorAction SilentlyContinue) | Out-String).Trim()) }
        }
        Info "app-exit-streams" $(if ($streams) { $streams -join " | " } else { "both streams empty" })
        $events = @()
        try {
            $events = @(Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $Launched.AddSeconds(-5) } -ErrorAction Stop |
                Where-Object { $_.Message -match 'Wenlan|wenlan' } |
                ForEach-Object { "id=$($_.Id) $($_.ProviderName): " + ($_.Message -replace '\s+', ' ').Substring(0, [Math]::Min(600, ($_.Message -replace '\s+', ' ').Length)) })
        } catch { $events = @("event log query failed: $($_.Exception.Message)") }
        Info "app-exit-events" $(if ($events) { $events -join " || " } else { "no Application events mention Wenlan" })
    }
    if (Wait-Health -Url $Health -Seconds 240) { Assert-Version -Url $Health -Expected $Version }
    Check -Name "sidecar-parent-is-app" -Script {
        $procs = @(Get-CimInstance Win32_Process -Filter "Name='wenlan-server.exe'")
        $desc = ($procs | ForEach-Object { "pid=$($_.ProcessId) ppid=$($_.ParentProcessId) path=$($_.ExecutablePath)" }) -join "; "
        Write-Output "app pid=$($App.Id); daemons: $desc"
        if ($procs.Count -eq 0) { throw "no wenlan-server.exe process" }
        $children = @($procs | Where-Object { $_.ParentProcessId -eq $App.Id })
        if ($children.Count -eq 0) { throw "no wenlan-server.exe has the app as parent: $desc" }
    }

    $env:WENLAN_BIN = Join-Path $Install "wenlan.exe"
    Check-Helper -Name "cli-roundtrip-driver" -Interpreter "pwsh" -InterpreterArgs @("-NoProfile", "-File") -Path (Join-Path $Helpers "cli-roundtrip.ps1") -MustDeclare "^cli-"
    $env:MCP_BIN = Join-Path $Install "wenlan-mcp.exe"
    $env:MCP_ARGS = "[]"
    $env:EXPECT_TOOL_COUNT = "29"
    $env:MCP_TOOLS = "capture,recall,brief"
    Check-Helper -Name "mcp-roundtrip-driver" -Interpreter "python" -Path (Join-Path $Helpers "mcp-roundtrip.py") -MustDeclare "^mcp-"
    Check -Name "doctor (bundled wenlan.exe doctor)" -Expect "Daemon: running on" -Script { & $env:WENLAN_BIN doctor }

    $Png = Join-Path $script:GauntletOut "logs\windows-nsis.png"
    try {
        Add-Type -AssemblyName System.Drawing, System.Windows.Forms
        $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
        $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
        $gfx = [System.Drawing.Graphics]::FromImage($bmp)
        $gfx.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        $bmp.Save($Png, [System.Drawing.Imaging.ImageFormat]::Png)
        $gfx.Dispose(); $bmp.Dispose()
        Info "screenshot" $Png
    } catch { Info "screenshot" "skipped: $($_.Exception.Message)" }

    if ($App) {
        $appKill = Stop-ProcessByImage -ProcessId $App.Id -ImagePath $AppExe
        Info "app-kill" "pid $($App.Id): $($appKill.State) -- $($appKill.Detail)"
    }
    # A surviving daemon is only an orphan if the app really died: prove the
    # kill landed before blaming the sidecar (adversarial review, F13).
    Check -Name "app-exited-after-kill" -Script {
        if (-not $App) { throw "no app process handle to kill" }
        $last = $null
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            $last = Get-ProcessLiveness -Id $App.Id
            if ($last.State -eq "gone") { Write-Output "app pid $($App.Id) exited — $($last.Detail)"; return }
            Start-Sleep -Milliseconds 500
        }
        # Three outcomes, three sentences. The old form had two, and the missing
        # one was the one that mattered: a table it could not read said "exited".
        if ($last.State -eq "unmeasurable") {
            throw "could not measure whether app pid $($App.Id) exited: $($last.Detail); recorded as unproven, not as exited"
        }
        throw "app pid $($App.Id) still alive 10s after the identity-checked kill — $($last.Detail)"
    }
    Check -Name "sidecar-exits-after-app" -Script {
        $last = $null
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            $last = Get-ProcessLiveness -Name wenlan-server
            if ($last.State -eq "gone") { break }
            Start-Sleep -Milliseconds 500
        }
        if ($last.State -eq "gone") {
            Write-Output "no wenlan-server process within 10s of app exit — $($last.Detail)"
            return
        }
        # This row and app-exited-after-kill used to share one swallowed read,
        # so a single failed look at the process table passed them both.
        if ($last.State -eq "unmeasurable") {
            throw "could not measure whether wenlan-server exited with the app: $($last.Detail); recorded as unproven, not as exited"
        }
        throw "orphan wenlan-server after the app was killed: $($last.Detail)"
    }
    Info "app-log-dir" (Join-Path $DataDir "logs")
    Collect (Join-Path $DataDir "logs") (Join-Path $env:TEMP "wenlan\logs")
    $AppLog = Join-Path $DataDir "logs\wenlan.log"
    Info "app-log-tail" $(if (Test-Path $AppLog) { ((Get-Content $AppLog -Tail 25) -join "`n") } else { "absent: $AppLog" })

    $Uninstaller = if ($Install) { Join-Path $Install "uninstall.exe" } else { "" }
    Info "uninstall-command" "$Uninstaller /S"
    # THE CLAIM ON THE INSTALL DIR IS GIVEN UP HERE, deliberately, before the
    # uninstaller is asked to remove that tree. The marker is held open with
    # FileShare.Delete withheld, which is exactly what makes the tree
    # undeletable -- and `uninstall-removes-dir` below is the row this whole
    # channel exists to produce. A marker still held would make that row fail on
    # a file this script put there, which is grading the harness and calling it
    # the product. The release is MEASURED and recorded: a release that did not
    # happen means the row below is about to blame the uninstaller for a lock
    # this run is holding, and the reader needs to be told which it was.
    $InstallMarkRelease = Close-OwnerMark $InstallMark
    Info "install-dir-claim-released" "$($InstallMarkRelease.State) -- $($InstallMarkRelease.Detail)"
    # The claim on the DATA dir is deliberately still held here, which is why
    # the row below cannot be a read of the root: this run holds a
    # DeleteOnClose handle inside that tree with FileShare.Delete withheld, and
    # that alone is what keeps the directory undeletable (measured, see WHAT
    # BINDS A TREE TO THIS RUN). The files are what the uninstaller can still
    # take, so the files are what gets snapshotted.
    $preDataSnapshot = Get-TreeFileDigests $DataDir
    Info "user-data-before-uninstall" "$($preDataSnapshot.State) -- $($preDataSnapshot.Detail)"
    Check -Name "uninstall-silent" -Script {
        if (-not $Uninstaller -or -not (Test-Path $Uninstaller)) { throw "no uninstaller at '$Uninstaller'" }
        $proc = Start-Process -FilePath $Uninstaller -ArgumentList '/S' -Wait -PassThru
        Write-Output "uninstaller exit=$($proc.ExitCode)"
        if ($proc.ExitCode -ne 0) { throw "uninstaller exit $($proc.ExitCode)" }
    }
    Check -Name "uninstall-removes-dir" -Script {
        if (-not $Install) { throw "no install dir was discovered" }
        # `Test-Path` answers $false for BOTH "not there" and "there, but I was
        # not allowed to look", and this is the row that grades the product's
        # uninstaller -- so an unreadable path used to be graded as a clean
        # removal. It was worse inside the wait than in the verdict: the loop
        # broke out on the first iteration and the row passed immediately.
        $last = Get-DirPresence $Install
        for ($attempt = 0; $attempt -lt 60 -and $last.State -eq "present"; $attempt++) {
            Start-Sleep -Seconds 1
            $last = Get-DirPresence $Install
        }
        # Whether this run was still holding its own marker inside the tree is
        # PART of this verdict either way. A handle this script left open makes
        # the tree undeletable, which would fail the uninstaller for something
        # that is not the uninstaller's doing -- exactly the confusion between
        # grading the product and grading the harness that the timestamp check
        # this replaced was introduced to avoid.
        $claim = if ($null -eq $InstallMarkRelease) { "this run never recorded giving up its claim on the tree" }
                 else { "this run's claim was given up with '$($InstallMarkRelease.State)': $($InstallMarkRelease.Detail)" }
        if ($last.State -eq "absent") {
            if ($null -ne $InstallMarkRelease -and $InstallMarkRelease.State -eq "residue") {
                Write-Output "$Install removed, and it went despite $claim"
            } else {
                Write-Output "$Install removed"
            }
            return
        }
        if ($last.State -eq "present") {
            throw ("install dir still present after 60s: " +
                   ((Get-ChildItem $Install -Name -ErrorAction SilentlyContinue) -join ", ") + "; $claim")
        }
        throw "could not measure whether $Install was removed: $($last.Detail); recorded as unproven, not as removed"
    }
    Check -Name "uninstall-removes-registry-key" -Script {
        $after = Get-UninstallEntry
        if ($after.State -eq "absent") { Write-Output "the uninstall entry is gone: $($after.Detail)"; return }
        if ($after.State -eq "present") { throw "uninstall entry still present: $($after.Entry.PSPath)" }
        throw "could not measure whether the uninstall entry is gone: $($after.Detail); recorded as unproven, not as removed"
    }
    # THE ROOT'S SURVIVAL IS NOT EVIDENCE, so it is logged and decides nothing.
    # This run is still holding the DeleteOnClose marker inside $DataDir with
    # FileShare.Delete withheld, which makes the directory undeletable by
    # anyone -- so "present" here is a fact about this script's own handle. An
    # uninstaller that erased every file underneath would produce it too.
    $dataAfterUninstall = Get-DirPresence $DataDir
    Info "data-dir-root-after-uninstall" ("$($dataAfterUninstall.State) -- $($dataAfterUninstall.Detail) " +
                                          "(this run holds a handle inside it, so its presence witnesses nothing)")
    # The claim itself, file by file. Three outcomes: every pre-existing file
    # still there with the same SHA-256 (PASS), one missing or rewritten
    # (FAIL), either snapshot not taken (unproven, never survived).
    #
    # It attributes a difference to the uninstaller because the app and the
    # sidecar are measured gone by the two rows above, so nothing of this run's
    # is still writing under $DataDir. A process this run does not own -- the
    # developer's own daemon -- would make this row fail for someone else's
    # write, and that is not distinguishable from here.
    $postDataSnapshot = Get-TreeFileDigests $DataDir
    Info "user-data-after-uninstall" "$($postDataSnapshot.State) -- $($postDataSnapshot.Detail)"
    Check -Name "user-data-survives-uninstall" -Script {
        if ($null -eq $preDataSnapshot -or $preDataSnapshot.State -ne "taken") {
            $pre = if ($null -eq $preDataSnapshot) { "never taken" } else { "$($preDataSnapshot.State) -- $($preDataSnapshot.Detail)" }
            throw "could not measure whether the user's data survived: the pre-uninstall snapshot of $DataDir was $pre; recorded as unproven, not as survived"
        }
        if ($postDataSnapshot.State -ne "taken") {
            throw "could not measure whether the user's data survived: $($postDataSnapshot.Detail); recorded as unproven, not as survived"
        }
        # Nothing to lose is not proof that nothing was lost.
        if ($preDataSnapshot.Files.Count -lt 1) {
            throw "there were no files under $DataDir before the uninstall, so this run cannot show the uninstaller left any; recorded as unproven, not as survived"
        }
        $missing = @()
        $changed = @()
        foreach ($rel in $preDataSnapshot.Files.Keys) {
            if (-not $postDataSnapshot.Files.ContainsKey($rel)) { $missing += $rel; continue }
            if ($postDataSnapshot.Files[$rel] -ne $preDataSnapshot.Files[$rel]) { $changed += $rel }
        }
        if ($missing.Count -ne 0 -or $changed.Count -ne 0) {
            $first = @(@($missing) + @($changed) | Select-Object -First 5)
            throw ("the uninstaller took $($missing.Count) and rewrote $($changed.Count) of the " +
                   "$($preDataSnapshot.Files.Count) files under ${DataDir}: " + ($first -join ", ") +
                   "; user data is not the uninstaller's to remove")
        }
        Write-Output "all $($preDataSnapshot.Files.Count) files under $DataDir are byte-for-byte what they were before the uninstall"
    }
} finally {
    # THE GUI KILL RECEIPT IS KEPT. `Stop-ProcessByImage` is tri-state --
    # killed / gone / refused / unmeasurable -- and every one of those went to
    # the console and nowhere else, so a kill that was REFUSED (the pid is
    # running a different image now) and a kill that could not even be
    # attempted (the process would not open) left the same trace as a kill that
    # worked: one line in a log nobody grades. The row below reads it.
    $AppKillReceipt = $null
    if ($App -and $AppExe) {
        $AppKillReceipt = Stop-ProcessByImage -ProcessId $App.Id -ImagePath $AppExe
        Write-Host "teardown: app pid $($App.Id): $($AppKillReceipt.State) -- $($AppKillReceipt.Detail)"
    }
    # Orphan sweep, and only over processes this run can show are its own. See
    # "WHAT THIS RUN MAY KILL" above for what the bare-name form did instead.
    #
    # ROUND 6 SPLIT THIS IN TWO, and the reason is what `measured` was being
    # read to mean. There was ONE sweep, taken BEFORE the kills, and the row
    # below graded it -- so `sidecar-sweep-measured` passing said the
    # enumeration succeeded, which is a fact about the process table and not
    # about the teardown. A run that enumerated three of its own daemons and
    # failed to kill any of them recorded exactly what a clean run recorded.
    # The kills now have receipts, and the sweep that the row grades is taken
    # AFTER them, so `$ownedLeft` means what its name says: what is still
    # running, of this run's own, at the end.
    $preKillSweep = Get-OwnedProcessesByImage -Name "wenlan-server" -ImagePath $OwnedServerImage -ExcludePids $PreexistingServerPids
    Write-Host "teardown: wenlan-server pre-kill sweep -- $($preKillSweep.State): $($preKillSweep.Detail)"
    $KillReceipts = New-Object System.Collections.Generic.List[object]
    foreach ($srv in $preKillSweep.Processes) {
        $k = Stop-ProcessByImage -ProcessId $srv.Id -ImagePath $srv.Path
        $KillReceipts.Add([pscustomobject]@{ Id = $srv.Id; State = $k.State; Detail = $k.Detail })
        Write-Host "teardown: wenlan-server pid $($srv.Id): $($k.State) -- $($k.Detail)"
    }
    $ownedLeft = Get-OwnedProcessesByImage -Name "wenlan-server" -ImagePath $OwnedServerImage -ExcludePids $PreexistingServerPids
    Write-Host "teardown: wenlan-server post-kill sweep -- $($ownedLeft.State): $($ownedLeft.Detail)"
    # A ROW, not a console line. The sweep is a measurement, its failure used to
    # look exactly like its success, and everything below it -- including the
    # delete of %LOCALAPPDATA%\wenlan -- runs afterwards. A teardown nobody
    # grades is one that quietly stops happening.
    #
    # THIS ROW GRADES THE READ, NOT THE OUTCOME, and that division is on
    # purpose: the licence to delete a directory turns on whether this run can
    # ACCOUNT for its daemons, and a sweep that measured "one of ours is still
    # up" has accounted for it. `teardown-sidecar-gone` below is the row that
    # grades whether anything is left.
    Check -Name "sidecar-sweep-measured" -Script {
        if ($ownedLeft.State -eq "measured") {
            Write-Output "the teardown sweep read the wenlan-server processes: $($ownedLeft.Detail)"
            return
        }
        if ($ownedLeft.State -eq "no-licence") {
            throw "this run cannot say which wenlan-server processes are its own, so it swept none: $($ownedLeft.Detail); recorded as unproven"
        }
        throw "the teardown sweep could not read the wenlan-server processes: $($ownedLeft.Detail); recorded as unproven, not as none left running"
    }
    # THE OUTCOME, which is what the row above was silently being read as. Three
    # facts, and each one used to be a console line: the GUI process this run
    # launched was seen to go, every kill this run attempted reported success or
    # an already-gone target, and the post-kill sweep found none of this run's
    # own daemons still up.
    Check -Name "teardown-app-gone" -Script {
        if ($null -eq $AppKillReceipt) {
            if ($App) { throw "this run launched wenlan-app pid $($App.Id) but never attempted to stop it, because it could not name the image it launched; recorded as unproven" }
            Write-Output "this run launched no GUI process, so it left none behind"
            return
        }
        if ($AppKillReceipt.State -eq "killed" -or $AppKillReceipt.State -eq "gone") {
            Write-Output "the GUI process this run launched is gone: $($AppKillReceipt.Detail)"
            return
        }
        if ($AppKillReceipt.State -eq "refused") {
            # Not a failure of the teardown: the identity check declining is the
            # protection working. It is still not a clean exit, and the run may
            # not claim one.
            throw "the GUI process this run launched was not stopped, because pid $($App.Id) is no longer the image this run started: $($AppKillReceipt.Detail); recorded as unproven"
        }
        throw "could not measure whether the GUI process this run launched was stopped: $($AppKillReceipt.Detail); recorded as unproven, not as exited"
    }
    Check -Name "teardown-sidecar-gone" -Script {
        $bad = @($KillReceipts | Where-Object { $_.State -ne "killed" -and $_.State -ne "gone" })
        if ($bad.Count -gt 0) {
            throw ("this run could not stop " + $bad.Count + " of the " + $KillReceipts.Count +
                   " wenlan-server process(es) it identified as its own: " +
                   (($bad | ForEach-Object { "pid $($_.Id) ($($_.State)): $($_.Detail)" }) -join "; ") + "; recorded as unproven")
        }
        if ($ownedLeft.State -ne "measured") {
            throw "whether any of this run's own wenlan-server processes are still running could not be measured ($($ownedLeft.State)): $($ownedLeft.Detail); recorded as unproven, not as none left"
        }
        if (@($ownedLeft.Processes).Count -ne 0) {
            throw ("after " + $KillReceipts.Count + " kill(s), this run's own wenlan-server is still running: " + $($ownedLeft.Detail))
        }
        Write-Output ("none of this run's own wenlan-server processes is still running after " +
                      $KillReceipts.Count + " kill(s): " + $($ownedLeft.Detail))
    }
    # The install dir is the uninstaller's job and `uninstall-removes-dir`
    # already grades it; this is the backstop for a run that never got that
    # far, and after `uninstall-silent` it can no longer delete anything at all
    # -- the claim on that tree is given up before the uninstaller runs, on
    # purpose, and a released claim licenses nothing.
    #
    # ROUND 6 REPLACED `$installOwned`. It was three remembered facts -- absent
    # before, present after, and the discovery landed on the documented path --
    # and the middle one was a creation TIMESTAMP, which NTFS file-system
    # tunneling hands to a directory recreated under the same name within about
    # fifteen seconds. It answered "verified" for exactly the replacement it
    # existed to catch. The tree carries a real marker now; what survives from
    # the old test is the path check, because a marker taken out on the
    # documented path is not a licence over whatever the depth-3 fallback found
    # somewhere else, and that is a different question from identity.
    $installIsTheClaimedPath = ($Install -and
                                $Install.TrimEnd('\') -ieq $InstallCandidate.TrimEnd('\'))
    # BOTH TREES, IN ONE LIST, and the row below reads the same list. It read
    # only $DataDir before, which is the false green round-5 review constructed:
    # the uninstaller removes $Install so `uninstall-removes-dir` passes, another
    # installer recreates the documented path before this block with a file
    # locked open, the install-dir delete exhausts its retries in silence, the
    # data-dir delete succeeds -- and `no-leftover-dirs` passed while $Install
    # was still standing, because nothing in it ever looked there.
    $CleanupDirs = @(
        @{ Name = "install dir"; Path = $Install; Mark = $InstallMark; Pre = $preInstall
           OnClaimedPath = $installIsTheClaimedPath; Licence = $null; Release = $null; Removal = $null },
        @{ Name = "data dir"; Path = $DataDir; Mark = $DataDirMark; Pre = $preDataDir
           OnClaimedPath = $true; Licence = $null; Release = $null; Removal = $null }
    )
    $ReleaseNotes = New-Object System.Collections.Generic.List[string]
    foreach ($d in $CleanupDirs) {
        $pre = if ($null -eq $d.Pre) { "never measured -- the run aborted before the pre-read" } else { $d.Pre.State }
        $markState = if ($null -eq $d.Mark) { "never attempted -- the run aborted before the claim" } else { $d.Mark.State }
        $markWhy = if ($null -eq $d.Mark) { "" } else { ": $($d.Mark.Detail)" }
        if (-not $d.Path) {
            $d.Licence = [pscustomobject]@{ State = "refused"; Detail = "no $($d.Name) path was ever discovered by this run" }
        } elseif (-not $d.OnClaimedPath) {
            $d.Licence = [pscustomobject]@{ State = "refused"
                Detail = ("the $($d.Name) this run found is $($d.Path), which is not the documented path $InstallCandidate " +
                          "this run took its claim out on; a licence over one path is not a licence over another") }
        } elseif ($null -eq $d.Mark -or ($d.Mark.State -ne "owned" -and $d.Mark.State -ne "released")) {
            # `not-owned` and `unmeasurable` are NOT the same fact -- one means
            # something was already at that path, the other means the claim
            # could not be attempted or read back -- so the detail says which.
            # "We left it alone because it was someone else's" and "we left it
            # alone because we could not tell" read very differently to whoever
            # finds this log afterwards.
            $d.Licence = [pscustomobject]@{ State = "refused"
                Detail = ("this run has no claim on the $($d.Name) $($d.Path) (before this run: $pre; " +
                          "the claim is '$markState'$markWhy), so it is not this run's to delete") }
        } elseif ($ownedLeft.State -ne "measured") {
            # A daemon this run cannot account for may be holding the very tree
            # about to be deleted. Round-3's C3 is the same hazard one level up.
            $d.Licence = [pscustomobject]@{ State = "refused"
                Detail = "the wenlan-server sweep did not measure ($($ownedLeft.State)): $($ownedLeft.Detail); nothing is deleted while a daemon may still be in it" }
        } else {
            # RE-VERIFIED HERE, two statements before the delete, rather than
            # once at the top of the teardown. Everything above this line -- the
            # kills, the sweep, the rows -- is time in which the tree could have
            # been swapped. The held marker is what makes that swap impossible
            # rather than merely detectable (Remove-Item and Rename-Item on the
            # tree both fail while it is held; MEASURED, see the block beside
            # New-OwnerMark), but "impossible" is a claim about a mechanism, and
            # this is the read that checks the mechanism held. For the install
            # dir the claim was deliberately released before `uninstall-silent`,
            # so the only verdict available there is `gone` or a refusal.
            $bound = Test-OwnerMark $d.Mark $d.Path
            if ($bound.State -eq "verified") {
                $d.Licence = [pscustomobject]@{ State = "granted"; Detail = "$($d.Name) $($d.Path): $($bound.Detail)" }
            } elseif ($bound.State -eq "gone") {
                $d.Licence = [pscustomobject]@{ State = "nothing-to-delete"; Detail = "$($d.Name) $($d.Path): $($bound.Detail)" }
            } else {
                $d.Licence = [pscustomobject]@{ State = "refused"
                    Detail = "the $($d.Name) $($d.Path) is not bound to this run ($($bound.State)): $($bound.Detail)" }
            }
        }
        # The claim is given up whatever the verdict was, and the result is
        # RECORDED. Two reasons it is unconditional. A tree cannot be deleted
        # while this run holds a file inside it, so a release that did not
        # happen is a delete that cannot happen -- saying so here is clearer
        # than ten sharing violations out of Remove-Retry. And a marker left
        # behind under a tree this run declines to delete makes the NEXT run's
        # CreateNew fail on a directory it created itself. On the install dir
        # this is normally a no-op: the claim was already released before the
        # uninstaller ran, and `nothing-held` says so.
        $d.Release = Close-OwnerMark $d.Mark
        $ReleaseNotes.Add("$($d.Name): $($d.Release.State) -- $($d.Release.Detail)")
        Write-Host "teardown: $($d.Name) marker release -- $($d.Release.State): $($d.Release.Detail)"
        if ($d.Licence.State -eq "granted" -and $d.Release.State -ne "released" -and $d.Release.State -ne "nothing-held") {
            $d.Licence = [pscustomobject]@{ State = "refused"
                Detail = ("the $($d.Name) $($d.Path) is bound to this run, but the claim on it could not be " +
                          "released ($($d.Release.State)): $($d.Release.Detail); a tree this run still holds a " +
                          "handle inside cannot be removed, so nothing is attempted") }
        }
        if ($d.Licence.State -eq "granted") {
            $d.Removal = Remove-Retry $d.Path
            Write-Host "teardown: $($d.Name) $($d.Path) delete -- $($d.Removal.State): $($d.Removal.Detail)"
        } else {
            Write-Host "teardown: $($d.Name) $($d.Path) LEFT IN PLACE -- $($d.Licence.State): $($d.Licence.Detail)"
        }
    }
    # Not a Check, because a marker this run failed to release is a residue on
    # the developer's disk rather than a claim this run got wrong -- but it goes
    # in the ledger rather than only on the console, because it is the one piece
    # of state this run can leave behind that will make the NEXT run refuse to
    # claim a tree it created itself.
    Info "owner-marker-release" ($ReleaseNotes -join " || ")
    # A row, not a log line: a teardown nobody grades is one that quietly
    # stops happening. Unlicensed is UNPROVEN, never green -- this run neither
    # created the tree nor removed it, so it can claim nothing about it.
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
    $global:LASTEXITCODE = 0
    if (-not (Evaluate)) { exit 1 }
}
