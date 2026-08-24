# First-run gauntlet helpers (PowerShell). Dot-source this file.
#
# Same contract and TSV shape as lib.sh: channel, name, status (PASS|FAIL|INFO),
# rc, detail. Checks never throw; Evaluate returns $false when any FAIL exists.
#
#   Check -Name n -Script { ... }                    PASS when the block does not throw and $LASTEXITCODE is 0/unset
#   Check -Name n -Expect "substr" -Script { ... }   PASS additionally requires the block's output to contain substr
#   Check -Name n -ExpectFail "substr" -Script {...} PASS when the block throws or exits nonzero AND output contains substr
#   Info -Name n -Value v
#   Wait-Health -Url u -Seconds s                    records seconds-to-health; returns $true/$false
#   Assert-Version -Url u -Expected v
#   Collect path...                                  copy into $GAUNTLET_OUT/logs
#   Evaluate                                         print table; return $true when no FAIL rows

$script:GauntletOut = if ($env:GAUNTLET_OUT) { $env:GAUNTLET_OUT } else { Join-Path (Get-Location) "gauntlet-out" }
$script:GauntletChannel = if ($env:GAUNTLET_CHANNEL) { $env:GAUNTLET_CHANNEL } else { [IO.Path]::GetFileNameWithoutExtension($MyInvocation.PSCommandPath) }
if (-not $script:GauntletChannel) { $script:GauntletChannel = "windows" }
$script:GauntletTsv = Join-Path $script:GauntletOut "findings.tsv"
New-Item -ItemType Directory -Force -Path (Join-Path $script:GauntletOut "checks") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $script:GauntletOut "logs") | Out-Null

function Escape-Detail([string]$text) {
    $one = ($text -replace "`t", " " -replace "`r", "" -replace "`n", "|")
    if ($one.Length -gt 2000) { $one = $one.Substring(0, 2000) }
    return $one
}

function Record-Row([string]$status, [string]$name, [int]$rc, [string]$detail) {
    $line = "{0}`t{1}`t{2}`t{3}`t{4}" -f $script:GauntletChannel, $name, $status, $rc, (Escape-Detail $detail)
    Add-Content -Path $script:GauntletTsv -Value $line -Encoding utf8
    $short = if ($detail) { " — " + (Escape-Detail $detail).Substring(0, [Math]::Min(200, (Escape-Detail $detail).Length)) } else { "" }
    Write-Host "[$status] $name (rc=$rc)$short"
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
    # Stream the block's output line by line so a throw mid-block keeps
    # everything printed before it (a single Out-String assignment would not).
    $lines = New-Object System.Collections.Generic.List[string]
    try {
        & $Script 2>&1 | ForEach-Object { $lines.Add(($_ | Out-String).TrimEnd()) }
        if ($LASTEXITCODE -is [int] -and $LASTEXITCODE -ne 0) { $rc = $LASTEXITCODE }
    } catch {
        $lines.Add($_.ToString())
        $rc = 1
    }
    $out = ($lines -join "`n")
    Set-Content -Path $log -Value $out -Encoding utf8
    if ($ExpectFail) {
        if ($rc -ne 0 -and $out.Contains($ExpectFail)) { Record-Row PASS $Name $rc $out }
        else { Record-Row FAIL $Name $rc ("expected nonzero exit with substring: " + $ExpectFail + "; got: " + $out) }
        return
    }
    if ($rc -eq 0 -and (-not $Expect -or $out.Contains($Expect))) { Record-Row PASS $Name $rc $out }
    elseif ($Expect) { Record-Row FAIL $Name $rc ("expected substring: " + $Expect + "; got: " + $out) }
    else { Record-Row FAIL $Name $rc $out }
}

function Info([string]$Name, [string]$Value) {
    Record-Row INFO $Name 0 $Value
}

function Wait-Health([string]$Url, [int]$Seconds = 120) {
    for ($i = 1; $i -le $Seconds; $i++) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { Info "seconds-to-health" "$i"; return $true }
        } catch { }
        Start-Sleep -Seconds 1
    }
    Record-Row FAIL "health-timeout" 1 "no 200 from $Url within ${Seconds}s"
    return $false
}

function Assert-Version([string]$Url, [string]$Expected) {
    $want = $Expected.TrimStart("v")
    $body = ""
    try { $body = (Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5).Content } catch { }
    $got = ""
    try { $got = ($body | ConvertFrom-Json).version } catch { }
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
    # A run that recorded nothing is unchecked, never a pass.
    if (-not (Test-Path $script:GauntletTsv) -or -not (Get-Content $script:GauntletTsv -ErrorAction SilentlyContinue)) {
        Write-Host "==> no findings recorded ($script:GauntletTsv missing or empty): unchecked, not a pass"
        return $false
    }
    Write-Host "==> findings for $script:GauntletChannel"
    $fails = 0
    if (Test-Path $script:GauntletTsv) {
        foreach ($line in Get-Content $script:GauntletTsv) {
            $cols = $line -split "`t"
            if ($cols.Count -ge 4) {
                Write-Host ("  {0,-4} {1,-40} rc={2}" -f $cols[2], $cols[1], $cols[3])
                if ($cols[2] -eq "FAIL") { $fails++ }
            }
        }
    }
    Write-Host "==> $fails FAIL row(s)"
    return ($fails -eq 0)
}
