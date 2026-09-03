# First-run gauntlet: npx MCP channel on Windows. Proves the release zip's daemon
# with the existing smoke (scripts/smoke-windows.ps1), then a plain daemon on
# port 17882, then `npx -y wenlan-mcp[@VERSION]` with a cold npm cache so the
# postinstall really downloads the binary, then the MCP round trip through npx.
# Everything comes from env (TAG, VERSION, IS_LATEST, GAUNTLET_OUT,
# GAUNTLET_CHANNEL, REPO_ROOT). No global Stop.
$ProgressPreference = 'SilentlyContinue'
. (Join-Path $PSScriptRoot "lib.ps1")

$Tag = $env:TAG
$Version = $env:VERSION
$Port = 17882
$Origin = "http://127.0.0.1:$Port"
$Health = "$Origin/api/health"
$Helpers = Join-Path $env:REPO_ROOT "scripts\first-run"
$Smoke = Join-Path $env:REPO_ROOT "scripts\smoke-windows.ps1"
$Work = Join-Path $script:GauntletOut "work-npx"
$Bin = Join-Path $Work "bin"
$DaemonData = Join-Path $Work "data"
$Server = Join-Path $Bin "wenlan-server.exe"
$OutLog = Join-Path $Work "wenlan-server.stdout.log"
$ErrLog = Join-Path $Work "wenlan-server.stderr.log"
$Spec = if ($env:IS_LATEST -eq "true") { "wenlan-mcp" } else { "wenlan-mcp@$Version" }
$Daemon = $null
if (-not $env:GAUNTLET_CHANNEL) { $env:GAUNTLET_CHANNEL = $script:GauntletChannel }

function Get-Asset([string]$AssetName, [string]$Dest) {
    $url = "https://github.com/7xuanlu/wenlan/releases/download/$Tag/$AssetName"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try { Invoke-WebRequest -Uri $url -OutFile $Dest -UseBasicParsing -ErrorAction Stop; return $true }
        catch { Write-Host "download attempt $attempt failed: $($_.Exception.Message)"; Start-Sleep -Seconds (5 * $attempt) }
    }
    return $false
}

# The rows this channel owes. $Spec is fixed above, so the npx row names are
# known before the run; the MCP helper declares its own mcp-* rows.
Expect-Rows -Names @(
    # The workflow's precheck step records `port-7878-precheck` before this
    # script starts, so that row is carried in; Record-CarriedRow below restates
    # its verdict as a row of this run's.
    "port-7878-precheck-carried",
    "download-zip",
    "extract-zip",
    "smoke-windows",
    "health-version",
    "npx-postinstall (npx -y $Spec --version)",
    "npx-version-matches",
    "mcp-roundtrip-driver"
)
Record-CarriedRow -Name "port-7878-precheck"

try {
    New-Item -ItemType Directory -Force -Path $Bin, $DaemonData | Out-Null
    $Zip = Join-Path $Work "wenlan-windows-x64.zip"
    Check -Name "download-zip" -Script { if (-not (Get-Asset "wenlan-windows-x64.zip" $Zip)) { throw "download failed after 3 attempts" }; Write-Output ("bytes=" + (Get-Item $Zip).Length) }
    Check -Name "extract-zip" -Script { Expand-Archive -Path $Zip -DestinationPath $Bin -Force -ErrorAction Stop; if (-not (Test-Path $Server)) { throw "missing $Server" }; Write-Output ((Get-ChildItem $Bin -Name) -join ", ") }

    Info "smoke-command" "pwsh -File scripts/smoke-windows.ps1 -ExePath $Server -HealthTimeoutSeconds 240"
    Check -Name "smoke-windows" -Script { & pwsh -NoProfile -File $Smoke -ExePath $Server -HealthTimeoutSeconds 240 }

    # Plain daemon on a private port and data dir; env is scoped to the spawn only.
    $env:WENLAN_PORT = "$Port"
    $env:WENLAN_BIND_ADDR = "127.0.0.1:$Port"
    $env:WENLAN_DATA_DIR = $DaemonData
    $Daemon = Start-Process -FilePath $Server -PassThru -WindowStyle Hidden -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
    Remove-Item Env:\WENLAN_PORT, Env:\WENLAN_BIND_ADDR, Env:\WENLAN_DATA_DIR -ErrorAction SilentlyContinue
    Info "daemon-command" "wenlan-server.exe (WENLAN_BIND_ADDR=127.0.0.1:$Port WENLAN_DATA_DIR=$DaemonData) pid=$($Daemon.Id)"
    if (Wait-Health -Url $Health -Seconds 180) { Assert-Version -Url $Health -Expected $Version }

    # Cold cache so `npx -y wenlan-mcp` really runs install.js (the postinstall that
    # downloads the binary); foreground-scripts makes npm show its stdout.
    $env:npm_config_cache = Join-Path $script:GauntletOut "npm-cache"
    $env:npm_config_foreground_scripts = "true"
    New-Item -ItemType Directory -Force -Path $env:npm_config_cache | Out-Null
    Info "pinned-mode" $(if ($env:IS_LATEST -eq "true") { "unpinned: npx -y wenlan-mcp (IS_LATEST=true)" } else { "pinned: npx -y $Spec (IS_LATEST=$($env:IS_LATEST))" })
    $latest = (& npm.cmd view wenlan-mcp version 2>&1 | Out-String).Trim()
    Info "npm-wenlan-mcp-latest" $latest
    $global:LASTEXITCODE = 0
    Check -Name "npx-postinstall (npx -y $Spec --version)" -Expect "wenlan-mcp installed successfully" -Script { & npx.cmd -y $Spec --version }
    Check -Name "npx-version-matches" -Expect $Version -Script { & npx.cmd -y $Spec --version }

    $env:MCP_BIN = "npx.cmd"
    $env:MCP_ARGS = (@("-y", $Spec, "--origin-url", $Origin) | ConvertTo-Json -Compress)
    $env:EXPECT_TOOL_COUNT = "29"
    $env:MCP_TOOLS = "capture,recall,brief"
    Info "mcp-command" "npx.cmd $($env:MCP_ARGS)"
    Check-Helper -Name "mcp-roundtrip-driver" -Interpreter "python" -Path (Join-Path $Helpers "mcp-roundtrip.py") -MustDeclare "^mcp-"
} finally {
    if ($Daemon) { Stop-Process -Id $Daemon.Id -Force -ErrorAction SilentlyContinue }
    Get-Process -Name wenlan-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$Work*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    Collect $OutLog $ErrLog
    # Reported rather than dropped: this channel's Remove-Retry used to swallow
    # every error and return nothing, so ten failed deletes read exactly like a
    # success. It has no ledger row here -- this channel makes no claim about
    # leftover trees -- but the console must not be silent about a work
    # directory that is still on disk.
    $removedWork = Remove-Retry $Work
    Write-Host "cleanup: $Work delete -- $($removedWork.State): $($removedWork.Detail)"
    $global:LASTEXITCODE = 0
    if (-not (Evaluate)) { exit 1 }
}
