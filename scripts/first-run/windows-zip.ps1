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
function Test-HealthReachable {
    try { $null = Invoke-WebRequest -Uri $Health -UseBasicParsing -TimeoutSec 2; return $true }
    catch { return ($null -ne $_.Exception.Response) }   # an HTTP error still proves reachable
}
function Remove-Retry([string]$Target) {
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Remove-Item -Recurse -Force $Target -ErrorAction SilentlyContinue
        if (-not (Test-Path $Target)) { return }
        Start-Sleep -Milliseconds 500
    }
}
function Stop-Daemon {
    & schtasks.exe /end /tn WenlanServer 2>&1 | Out-Null
    Get-Process -Name wenlan-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($attempt = 0; $attempt -lt 20; $attempt++) { if (-not (Test-HealthReachable)) { break }; Start-Sleep -Milliseconds 500 }
    $global:LASTEXITCODE = 0
}

try {
    New-Item -ItemType Directory -Force -Path $Extract, $InstallDir | Out-Null
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
    Check -Name "setup-basic (wenlan.exe setup --basic)" -Expect "Wenlan is set up for local memory." -Script { & wenlan.exe setup --basic }
    Check -Name "background-on (wenlan.exe background on)" -Expect "Installed and started Windows scheduled task" -Script { & wenlan.exe background on }
    Check -Name "schtasks-registered" -Script { & schtasks.exe /query /tn WenlanServer /fo LIST }
    if (Wait-Health -Url $Health -Seconds 240) { Assert-Version -Url $Health -Expected $Version }
    Check -Name "status (wenlan.exe status)" -Script { & wenlan.exe status }

    $env:WENLAN_BIN = $Wenlan
    & pwsh -NoProfile -File (Join-Path $Helpers "cli-roundtrip.ps1")
    $env:MCP_BIN = Join-Path $InstallDir "wenlan-mcp.exe"
    $env:MCP_ARGS = "[]"
    $env:EXPECT_TOOL_COUNT = "29"
    $env:MCP_TOOLS = "capture,recall,brief"
    & python (Join-Path $Helpers "mcp-roundtrip.py")
    $global:LASTEXITCODE = 0

    Check -Name "doctor (wenlan.exe doctor)" -Expect "Daemon: running on" -Script { & wenlan.exe doctor }
    Check -Name "dll-identity" -Script {
        $srv = Get-Process -Name wenlan-server -ErrorAction Stop | Select-Object -First 1
        Write-Output "wenlan-server pid=$($srv.Id) path=$($srv.Path)"
        foreach ($dll in @("onnxruntime.dll", "vulkan-1.dll")) {
            $loaded = @(Get-Process -Id $srv.Id -Module -ErrorAction Stop | Where-Object { $_.ModuleName -ieq $dll } | ForEach-Object { $_.FileName })
            $want = Join-Path $InstallDir $dll
            if ($loaded.Count -ne 1 -or -not [string]::Equals($loaded[0], $want, [System.StringComparison]::OrdinalIgnoreCase)) { throw "$dll loaded from [$($loaded -join ', ')], expected exactly $want" }
            Write-Output "$dll -> $($loaded[0])"
        }
    }

    # Recovery: kill the daemon behind the task's back; a read command must restart it.
    Stop-Daemon
    Check -Name "autostart-recovery (wenlan.exe memories --limit 1)" -Expect "daemon not reachable" -Script { & wenlan.exe memories --limit 1 }
    Check -Name "healthy-after-recovery" -Script { if (-not (Test-HealthReachable)) { throw "health unreachable after recovery" } }

    Check -Name "background-off (wenlan.exe background off)" -Expect "Background registration kept" -Script { & wenlan.exe background off }
    Check -Name "task-kept-after-off" -Script { & schtasks.exe /query /tn WenlanServer }
    Check -Name "health-unreachable-after-off" -Script { if (Test-HealthReachable) { throw "background off left $Health reachable" } }
    Check -Name "autostart-marker" -Script { $marker = Join-Path $DataDir "autostart.off"; if (-not (Test-Path $marker)) { throw "missing $marker" }; Write-Output $marker }
    Check -Name "stopped-marker-error (wenlan.exe search x)" -ExpectFail "daemon stopped by" -Script { & wenlan.exe search x }
    Check -Name "background-on-again" -Expect "Installed and started Windows scheduled task" -Script { & wenlan.exe background on }
    if (Wait-Health -Url $Health -Seconds 120) { Assert-Version -Url $Health -Expected $Version }
} finally {
    if (Test-Path $Wenlan) { & $Wenlan background off 2>&1 | Out-Null }
    Stop-Daemon
    $deleteOut = & schtasks.exe /delete /tn WenlanServer /f 2>&1
    Write-Host "cleanup: schtasks /delete exit=$LASTEXITCODE $deleteOut"
    Collect (Join-Path $DataDir "logs")
    Remove-Retry $InstallDir
    Remove-Retry $DataDir
    Check -Name "no-leftover-task" -Script { $q = & schtasks.exe /query /tn WenlanServer 2>&1 | Out-String; if ($LASTEXITCODE -eq 0) { throw "WenlanServer task still registered: $q" }; $global:LASTEXITCODE = 0; $q }
    Check -Name "port-7878-closed" -Script { $open = @(Get-NetTCPConnection -LocalPort 7878 -State Listen -ErrorAction SilentlyContinue); if ($open.Count -ne 0) { throw "port 7878 still listening (pid $($open[0].OwningProcess))" } }
    Info "data-dir-removed" (-not (Test-Path $DataDir)).ToString()
    if (-not (Evaluate)) { exit 1 }
}
