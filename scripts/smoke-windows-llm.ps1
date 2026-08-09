param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$ProbePath = "target\release\model_probe.exe",

    [ValidatePattern("^(auto|cpu|\d+)$")]
    [string]$Device = "auto",

    [ValidateSet("cpu", "vulkan")]
    [string]$ExpectedBackend = "cpu",

    [string]$ExpectedDevicePattern,

    [string]$ExpectedFallbackPattern,

    [switch]$SkipHardwareInventory
)

$ErrorActionPreference = "Stop"
# The harness checks native exit codes explicitly after capturing stdout/stderr.
# Keep PowerShell 7.3+'s optional native-error promotion from pre-empting that path.
$PSNativeCommandUseErrorActionPreference = $false

if (-not (Test-Path $ProbePath -PathType Leaf)) {
    throw "expected model probe at $ProbePath; build it with cargo build --release -p wenlan-core --bin model_probe"
}
if (-not (Test-Path $ModelPath -PathType Leaf)) {
    throw "expected GGUF model at $ModelPath"
}

$ProbePath = (Resolve-Path $ProbePath).Path
$ModelPath = (Resolve-Path $ModelPath).Path
$previousRustLog = $env:RUST_LOG
$previousLlmDevice = $env:WENLAN_LLM_DEVICE
$env:RUST_LOG = "wenlan_core=info"
$env:WENLAN_LLM_DEVICE = $Device

try {
    if (-not $SkipHardwareInventory) {
        Write-Host "==> Windows video controllers"
        Get-CimInstance Win32_VideoController |
            Select-Object Name, DriverVersion, AdapterRAM |
            Format-Table -AutoSize
    }

    Write-Host "==> Running Qwen model probe"
    # Windows PowerShell 5 turns native stderr records into terminating errors
    # when ErrorActionPreference is Stop. llama.cpp logs normally on stderr, so
    # capture under Continue and check LASTEXITCODE ourselves. PowerShell 7
    # follows the same explicit path because PSNativeCommandUseErrorActionPreference
    # is disabled above.
    $savedErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & $ProbePath $ModelPath 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $savedErrorActionPreference
    }
    $text = ($output | Out-String)
    Write-Host $text

    if ($exitCode -ne 0) {
        throw "model probe exited with code $exitCode"
    }

    $backendMarker = "--- Inference backend: $ExpectedBackend ---"
    if (-not $text.Contains($backendMarker)) {
        throw "expected backend marker '$backendMarker'"
    }
    if (
        $ExpectedBackend -eq "cpu" -and
        $text -match "(?im)\bVulkan\d+\s+(?:KV|compute|output)\s+buffer\s+size\s*="
    ) {
        throw "CPU smoke observed GPU runtime allocation: $($Matches[0])"
    }
    if ($ExpectedDevicePattern -and $text -notmatch $ExpectedDevicePattern) {
        throw "expected device matching '$ExpectedDevicePattern'"
    }
    if ($ExpectedFallbackPattern -and $text -notmatch $ExpectedFallbackPattern) {
        throw "expected fallback reason matching '$ExpectedFallbackPattern'"
    }
    $classificationMarker = "--- Verified classification: preference ---"
    if (-not $text.Contains($classificationMarker)) {
        throw "model probe did not produce the expected classification"
    }

    Write-Host "==> PASS: Qwen produced a valid classification on $ExpectedBackend"
}
finally {
    if ($null -eq $previousRustLog) {
        Remove-Item Env:RUST_LOG -ErrorAction SilentlyContinue
    } else {
        $env:RUST_LOG = $previousRustLog
    }
    if ($null -eq $previousLlmDevice) {
        Remove-Item Env:WENLAN_LLM_DEVICE -ErrorAction SilentlyContinue
    } else {
        $env:WENLAN_LLM_DEVICE = $previousLlmDevice
    }
}
