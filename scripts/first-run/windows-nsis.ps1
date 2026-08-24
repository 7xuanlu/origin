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
if (-not $env:GAUNTLET_CHANNEL) { $env:GAUNTLET_CHANNEL = $script:GauntletChannel }

function Get-Asset([string]$AssetName, [string]$Dest) {
    $url = "https://github.com/7xuanlu/wenlan/releases/download/$Tag/$AssetName"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try { Invoke-WebRequest -Uri $url -OutFile $Dest -UseBasicParsing -ErrorAction Stop; return $true }
        catch { Write-Host "download attempt $attempt failed: $($_.Exception.Message)"; Start-Sleep -Seconds (5 * $attempt) }
    }
    return $false
}
function Get-UninstallEntry {
    $entries = foreach ($root in $UninstallRoots) {
        Get-ItemProperty -Path (Join-Path $root "*") -ErrorAction SilentlyContinue | Where-Object { $_.DisplayName -eq "Wenlan" }
    }
    return @($entries) | Select-Object -First 1
}
function Remove-Retry([string]$Target) {
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Remove-Item -Recurse -Force $Target -ErrorAction SilentlyContinue
        if (-not (Test-Path $Target)) { return }
        Start-Sleep -Milliseconds 500
    }
}

try {
    New-Item -ItemType Directory -Force -Path $Work | Out-Null
    Info "documented-flow" "run $SetupName from the Releases page (gauntlet runs it as: $SetupName /S)"
    Check -Name "download-setup" -Script { if (-not (Get-Asset $SetupName $Setup)) { throw "download failed after 3 attempts" }; Write-Output ("bytes=" + (Get-Item $Setup).Length) }
    Check -Name "nsis-silent-install" -Script {
        $proc = Start-Process -FilePath $Setup -ArgumentList '/S' -Wait -PassThru
        Write-Output "installer exit=$($proc.ExitCode)"
        if ($proc.ExitCode -ne 0) { throw "installer exit $($proc.ExitCode)" }
    }

    $candidate = Join-Path $env:LOCALAPPDATA "Wenlan\Wenlan.exe"
    if (Test-Path $candidate) { $AppExe = $candidate }
    else {
        foreach ($root in @($env:LOCALAPPDATA, $env:ProgramFiles, ${env:ProgramFiles(x86)})) {
            if ($AppExe -or -not $root -or -not (Test-Path $root)) { continue }
            $hit = Get-ChildItem -Path $root -Recurse -Depth 3 -Filter "Wenlan.exe" -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($hit) { $AppExe = $hit.FullName }
        }
    }
    Check -Name "app-exe-found" -Script { if (-not $AppExe) { throw "Wenlan.exe not found under LOCALAPPDATA or Program Files" }; Write-Output $AppExe }
    if ($AppExe) { $Install = Split-Path -Parent $AppExe }
    Info "install-dir" "$Install"

    $entry = Get-UninstallEntry
    Check -Name "uninstall-key-present" -Script { if (-not $entry) { throw "no uninstall entry with DisplayName 'Wenlan' under HKCU/HKLM" }; Write-Output $entry.PSPath }
    Check -Name "uninstall-display-version" -Script { if ("$($entry.DisplayVersion)" -ne $Version) { throw "DisplayVersion '$($entry.DisplayVersion)' != $Version" }; Write-Output $entry.DisplayVersion }
    Info "uninstall-string" "$($entry.UninstallString)"
    Check -Name "bundled-binaries" -Script {
        if (-not $Install) { throw "no install dir" }
        $missing = @($Bundled | Where-Object { -not (Test-Path (Join-Path $Install $_)) })
        if ($missing.Count -ne 0) { throw ("missing beside Wenlan.exe: " + ($missing -join ", ")) }
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
    if ($AppExe) {
        $App = Start-Process -FilePath $AppExe -WorkingDirectory $Install -PassThru `
            -RedirectStandardOutput $AppOut -RedirectStandardError $AppErr
    }
    Check -Name "app-alive-30s" -Script {
        if (-not $App) { throw "app was not launched" }
        Start-Sleep -Seconds 30
        if ($App.HasExited) { throw "Wenlan.exe exited within 30s (exit $($App.ExitCode))" }
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
    & pwsh -NoProfile -File (Join-Path $Helpers "cli-roundtrip.ps1")
    $env:MCP_BIN = Join-Path $Install "wenlan-mcp.exe"
    $env:MCP_ARGS = "[]"
    $env:EXPECT_TOOL_COUNT = "29"
    $env:MCP_TOOLS = "capture,recall,brief"
    & python (Join-Path $Helpers "mcp-roundtrip.py")
    $global:LASTEXITCODE = 0
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

    if ($App) { Stop-Process -Id $App.Id -Force -ErrorAction SilentlyContinue }
    Check -Name "sidecar-exits-after-app" -Script {
        $left = $null
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            $left = @(Get-Process -Name wenlan-server -ErrorAction SilentlyContinue)
            if ($left.Count -eq 0) { break }
            Start-Sleep -Milliseconds 500
        }
        if ($left.Count -ne 0) { throw ("orphan wenlan-server after app Stop-Process: pid " + (($left | ForEach-Object { $_.Id }) -join ", ")) }
        Write-Output "no wenlan-server process within 10s of app exit"
    }
    Info "app-log-dir" (Join-Path $DataDir "logs")
    Collect (Join-Path $DataDir "logs") (Join-Path $env:TEMP "wenlan\logs")
    $AppLog = Join-Path $DataDir "logs\wenlan.log"
    Info "app-log-tail" $(if (Test-Path $AppLog) { ((Get-Content $AppLog -Tail 25) -join "`n") } else { "absent: $AppLog" })

    $Uninstaller = if ($Install) { Join-Path $Install "uninstall.exe" } else { "" }
    Info "uninstall-command" "$Uninstaller /S"
    Check -Name "uninstall-silent" -Script {
        if (-not $Uninstaller -or -not (Test-Path $Uninstaller)) { throw "no uninstaller at '$Uninstaller'" }
        $proc = Start-Process -FilePath $Uninstaller -ArgumentList '/S' -Wait -PassThru
        Write-Output "uninstaller exit=$($proc.ExitCode)"
        if ($proc.ExitCode -ne 0) { throw "uninstaller exit $($proc.ExitCode)" }
    }
    Check -Name "uninstall-removes-dir" -Script {
        if (-not $Install) { throw "no install dir was discovered" }
        for ($attempt = 0; $attempt -lt 60; $attempt++) { if (-not (Test-Path $Install)) { break }; Start-Sleep -Seconds 1 }
        if (Test-Path $Install) { throw ("install dir still present after 60s: " + ((Get-ChildItem $Install -Name) -join ", ")) }
        Write-Output "$Install removed"
    }
    Check -Name "uninstall-removes-registry-key" -Script { $after = Get-UninstallEntry; if ($after) { throw "uninstall entry still present: $($after.PSPath)" } }
    Info "data-dir-survives-uninstall" ((Test-Path $DataDir).ToString() + " ($DataDir; expected true — user data is not removed by the uninstaller)")
} finally {
    if ($App) { Stop-Process -Id $App.Id -Force -ErrorAction SilentlyContinue }
    Get-Process -Name Wenlan, wenlan-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    if ($Install -and (Test-Path $Install)) { Remove-Retry $Install }
    Remove-Retry $DataDir
    $global:LASTEXITCODE = 0
    if (-not (Evaluate)) { exit 1 }
}
