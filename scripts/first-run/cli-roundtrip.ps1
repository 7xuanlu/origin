# First-run gauntlet: the wenlan CLI round-trip on Windows. Mirrors
# cli-roundtrip.sh step for step: status, capture a sentinel, memories lists
# it, search finds it (polled up to 60s). Every step is recorded through
# lib.ps1 and the script never exits early. Always exits 0 — the channel
# script calls Evaluate.
#
# Env:
#   WENLAN_BIN    path to wenlan.exe (required)
#   WENLAN_HOST   daemon URL; leave unset to exercise the CLI's own default
#   GAUNTLET_OUT / GAUNTLET_CHANNEL   see lib.ps1
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "lib.ps1")

# This helper owes four rows on every path, including the WENLAN_BIN guard
# below, which records cli-status and stops: the other three then read as
# never run, which is what they are.
Expect-Rows -Names @("cli-status", "cli-capture", "cli-memories", "cli-search")

# Never let a connect failure start a registered daemon; the gauntlet boots its own.
if (-not $env:WENLAN_NO_AUTOSTART) { $env:WENLAN_NO_AUTOSTART = "1" }
# An empty WENLAN_HOST would reach the CLI as a real (invalid) value.
if ($null -ne $env:WENLAN_HOST -and $env:WENLAN_HOST -eq "") { Remove-Item Env:\WENLAN_HOST -ErrorAction SilentlyContinue }

if (-not $env:WENLAN_BIN) {
    Check -Name "cli-status" -Script { throw "WENLAN_BIN (path to wenlan.exe) is required" }
    exit 0
}

$bin = $env:WENLAN_BIN
$sentinel = "kumquat-lighthouse-8231"

Write-Host "==> wenlan status"
# `status --format json` now exits non-zero when the daemon is unreachable,
# but check the health payload's version field too, not just rc=0.
Check -Name "cli-status" -Expect '"version"' -Script { & $bin --format json status }

Write-Host "==> wenlan capture (sentinel)"
Check -Name "cli-capture" -Script {
    & $bin --format json capture "The $sentinel sentinel sentence lives in the CLI smoke." --type fact
}

Write-Host "==> wenlan memories contains the sentinel"
Check -Name "cli-memories" -Expect $sentinel -Script { & $bin --format json memories --limit 20 }

Write-Host "==> wenlan search finds the sentinel"
# Poll first (embedding/indexing is async), then record one final search so
# the check log holds the output that actually matched — or the last miss.
$hit = $false
for ($i = 1; $i -le 30; $i++) {
    $out = ""
    try { $out = (& $bin --format json search "kumquat lighthouse sentinel sentence" --limit 5 2>&1 | Out-String) } catch { break }
    if ($LASTEXITCODE -is [int] -and $LASTEXITCODE -ne 0) { break }
    if ($out.Contains($sentinel)) {
        Write-Host "    hit after $i poll(s)"
        $hit = $true
        break
    }
    Start-Sleep -Seconds 2
}
if (-not $hit) { Write-Host "    sentinel not retrievable via wenlan search within 60s" }
Check -Name "cli-search" -Expect $sentinel -Script {
    & $bin --format json search "kumquat lighthouse sentinel sentence" --limit 5
}

exit 0
