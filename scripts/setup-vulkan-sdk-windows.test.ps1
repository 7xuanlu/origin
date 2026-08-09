$ErrorActionPreference = "Stop"

$setupScript = Join-Path $PSScriptRoot "setup-vulkan-sdk-windows.ps1"
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) "wenlan-vulkan-sdk-test-$PID"
$validSdk = Join-Path $testRoot "valid"
$invalidSdk = Join-Path $testRoot "invalid"
$githubEnv = Join-Path $testRoot "github-env.txt"
$githubPath = Join-Path $testRoot "github-path.txt"
$previousGithubEnv = $env:GITHUB_ENV
$previousGithubPath = $env:GITHUB_PATH
$previousVulkanSdk = $env:VULKAN_SDK

try {
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $validSdk "Bin"), `
        (Join-Path $validSdk "Lib"), `
        (Join-Path $validSdk "Include\vulkan"), `
        $invalidSdk | Out-Null
    New-Item -ItemType File -Force -Path `
        (Join-Path $validSdk "Bin\glslc.exe"), `
        (Join-Path $validSdk "Lib\vulkan-1.lib"), `
        (Join-Path $validSdk "Include\vulkan\vulkan.h") | Out-Null
    New-Item -ItemType File -Force -Path $githubEnv, $githubPath | Out-Null

    $env:GITHUB_ENV = $githubEnv
    $env:GITHUB_PATH = $githubPath
    & $setupScript -InstallRoot $validSdk -ValidateOnly

    if ($env:VULKAN_SDK -ne [System.IO.Path]::GetFullPath($validSdk)) {
        throw "setup script did not export VULKAN_SDK"
    }
    if (-not (Get-Content $githubEnv -Raw).Contains("VULKAN_SDK=")) {
        throw "setup script did not write GITHUB_ENV"
    }
    if (-not (Get-Content $githubPath -Raw).Contains((Join-Path $validSdk "Bin"))) {
        throw "setup script did not write GITHUB_PATH"
    }

    $failed = $false
    try {
        & $setupScript -InstallRoot $invalidSdk -ValidateOnly
    }
    catch {
        $failed = $true
        if (-not $_.Exception.Message.Contains("is incomplete")) {
            throw "invalid SDK failed for the wrong reason: $($_.Exception.Message)"
        }
    }
    if (-not $failed) {
        throw "incomplete Vulkan SDK unexpectedly passed validation"
    }

    $source = Get-Content $setupScript -Raw
    if (-not $source.Contains("copy_only=1")) {
        throw "setup must stay non-admin and registry-free"
    }
    if (-not $source.Contains("Get-FileHash -Algorithm SHA256")) {
        throw "setup must verify the pinned SDK checksum"
    }

    Write-Host "PASS: Vulkan SDK setup script contract"
}
finally {
    if ($null -eq $previousGithubEnv) {
        Remove-Item Env:GITHUB_ENV -ErrorAction SilentlyContinue
    } else {
        $env:GITHUB_ENV = $previousGithubEnv
    }
    if ($null -eq $previousGithubPath) {
        Remove-Item Env:GITHUB_PATH -ErrorAction SilentlyContinue
    } else {
        $env:GITHUB_PATH = $previousGithubPath
    }
    if ($null -eq $previousVulkanSdk) {
        Remove-Item Env:VULKAN_SDK -ErrorAction SilentlyContinue
    } else {
        $env:VULKAN_SDK = $previousVulkanSdk
    }
    Remove-Item -Recurse -Force $testRoot -ErrorAction SilentlyContinue
}
