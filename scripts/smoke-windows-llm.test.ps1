$ErrorActionPreference = "Stop"

$smokeScript = Join-Path $PSScriptRoot "smoke-windows-llm.ps1"
$tempRoot = if ($env:RUNNER_TEMP) {
    $env:RUNNER_TEMP
}
else {
    [System.IO.Path]::GetTempPath()
}
$testDirectory = Join-Path $tempRoot "wenlan-windows-llm-smoke-test-$PID"
$probePath = Join-Path $testDirectory "model-probe-stub.cmd"
$modelPath = Join-Path $testDirectory "model.gguf"
$previousScenario = $env:WENLAN_LLM_SMOKE_STUB_SCENARIO

function Assert-SmokeFails {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Scenario,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedMessage
    )

    $env:WENLAN_LLM_SMOKE_STUB_SCENARIO = $Scenario
    $failed = $false
    try {
        & $smokeScript `
            -ModelPath $modelPath `
            -ProbePath $probePath `
            -SkipHardwareInventory
    }
    catch {
        $failed = $true
        if (-not $_.Exception.Message.Contains($ExpectedMessage)) {
            throw "scenario '$Scenario' failed for the wrong reason: $($_.Exception.Message)"
        }
    }

    if (-not $failed) {
        throw "scenario '$Scenario' unexpectedly passed"
    }
}

try {
    New-Item -ItemType Directory -Path $testDirectory | Out-Null
    New-Item -ItemType File -Path $modelPath | Out-Null
    @'
@echo off
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="exit" exit /b 17
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="vulkan" echo --- Inference backend: vulkan ---
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="vulkan" echo --- Inference device: NVIDIA GeForce RTX 3060 Laptop GPU ---
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="missing-device" echo --- Inference backend: vulkan ---
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="fallback" echo --- Inference backend: cpu ---
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="fallback" echo --- Inference fallback: requested GPU device index 99 is unavailable ---
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="pass" echo --- Inference backend: cpu ---
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="cpu-gpu-leak" echo --- Inference backend: cpu ---
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="cpu-gpu-leak" echo sched_reserve: Vulkan1 compute buffer size = 630.52 MiB
if /I "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="missing-classification" echo --- Inference backend: cpu ---
if /I not "%WENLAN_LLM_SMOKE_STUB_SCENARIO%"=="missing-classification" echo --- Verified classification: preference ---
exit /b 0
'@ | Set-Content -Path $probePath -Encoding ascii

    $env:WENLAN_LLM_SMOKE_STUB_SCENARIO = "pass"
    & $smokeScript `
        -ModelPath $modelPath `
        -ProbePath $probePath `
        -SkipHardwareInventory

    $env:WENLAN_LLM_SMOKE_STUB_SCENARIO = "vulkan"
    & $smokeScript `
        -ModelPath $modelPath `
        -ProbePath $probePath `
        -ExpectedBackend vulkan `
        -ExpectedDevicePattern "NVIDIA.*RTX 3060" `
        -SkipHardwareInventory

    $env:WENLAN_LLM_SMOKE_STUB_SCENARIO = "fallback"
    & $smokeScript `
        -ModelPath $modelPath `
        -ProbePath $probePath `
        -Device 99 `
        -ExpectedBackend cpu `
        -ExpectedFallbackPattern "requested GPU device index 99 is unavailable" `
        -SkipHardwareInventory

    Assert-SmokeFails `
        -Scenario "exit" `
        -ExpectedMessage "model probe exited with code 17"
    Assert-SmokeFails `
        -Scenario "missing-backend" `
        -ExpectedMessage "expected backend marker"
    Assert-SmokeFails `
        -Scenario "missing-classification" `
        -ExpectedMessage "expected classification"
    Assert-SmokeFails `
        -Scenario "cpu-gpu-leak" `
        -ExpectedMessage "CPU smoke observed GPU runtime allocation"
    $env:WENLAN_LLM_SMOKE_STUB_SCENARIO = "missing-device"
    $failed = $false
    try {
        & $smokeScript `
            -ModelPath $modelPath `
            -ProbePath $probePath `
            -ExpectedBackend vulkan `
            -ExpectedDevicePattern "NVIDIA.*RTX 3060" `
            -SkipHardwareInventory
    }
    catch {
        $failed = $true
        if (-not $_.Exception.Message.Contains("expected device matching")) {
            throw "missing-device failed for the wrong reason: $($_.Exception.Message)"
        }
    }
    if (-not $failed) {
        throw "missing-device unexpectedly passed"
    }

    Write-Host "PASS: Windows LLM smoke harness behavior"
}
finally {
    if ($null -eq $previousScenario) {
        Remove-Item Env:WENLAN_LLM_SMOKE_STUB_SCENARIO -ErrorAction SilentlyContinue
    }
    else {
        $env:WENLAN_LLM_SMOKE_STUB_SCENARIO = $previousScenario
    }
    Remove-Item -Recurse -Force $testDirectory -ErrorAction SilentlyContinue
}
