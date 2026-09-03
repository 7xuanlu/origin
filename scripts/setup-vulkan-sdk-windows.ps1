# Installs the pinned LunarG Vulkan SDK for BUILDING wenlan on Windows.
#
# Scope, stated precisely because it is easy to overstate: the SDK is a
# build-time prerequisite and that is established -- llama-cpp-sys-2's build
# script panics without VULKAN_SDK, so no daemon crate compiles without it.
# Nothing this script installs is redistributed to end users; the release
# archive ships only the separately verified loader (vulkan-1.dll), staged by
# scripts\stage-vulkan-loader-windows.ps1.
#
# What is NOT established, and must not be written anywhere as if it were:
# that an end user with the loader but no vendor ICD is unaffected. The shipped
# ZIP and NSIS channels are measured starting and answering /api/health on a
# driverless hosted runner (.github/workflows/first-run-gauntlet.yml), but no
# gate has ever loaded a GGUF there. The CPU plan in
# crates/wenlan-core/src/engine.rs is built from a live device list reached only
# after LlamaBackend::init() succeeds (engine.rs:33), so a backend-init failure
# on a driverless machine never reaches it. The gate that would settle it: a
# clean Windows VM or Sandbox with no vendor ICD, the real signed installer or
# release ZIP, one real GGUF inference, asserting backend=cpu and gpu_layers=0.
# See docs/windows-vulkan.md.

param(
    [string]$Version = "1.4.350.0",

    [string]$ExpectedSha256 = "855b27ba05d2d8119c5114c5d4ff870ca38f2c632b11e1bb9923b9b7e6ecfe7b",

    [string]$InstallRoot,

    [string]$InstallerPath,

    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

if (-not $InstallRoot) {
    $base = if ($env:RUNNER_TEMP) {
        $env:RUNNER_TEMP
    }
    else {
        Join-Path $env:LOCALAPPDATA "wenlan-build"
    }
    $InstallRoot = Join-Path $base "VulkanSDK\$Version"
}
$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot)

function Assert-VulkanSdk {
    param([Parameter(Mandatory = $true)][string]$Root)

    $required = @(
        (Join-Path $Root "Bin\glslc.exe"),
        (Join-Path $Root "Lib\vulkan-1.lib"),
        (Join-Path $Root "Include\vulkan\vulkan.h")
    )
    $missing = @($required | Where-Object { -not (Test-Path $_ -PathType Leaf) })
    if ($missing.Count -gt 0) {
        throw "Vulkan SDK at '$Root' is incomplete; missing: $($missing -join ', ')"
    }
}

if (-not $ValidateOnly -and -not (Test-Path (Join-Path $InstallRoot "Bin\glslc.exe"))) {
    if (-not $InstallerPath) {
        $downloadRoot = Join-Path ([System.IO.Path]::GetTempPath()) "wenlan-vulkan-sdk-$Version"
        New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
        $InstallerPath = Join-Path $downloadRoot "vulkansdk-windows-X64-$Version.exe"
        if (-not (Test-Path $InstallerPath -PathType Leaf)) {
            $uri = "https://sdk.lunarg.com/sdk/download/$Version/windows/vulkansdk-windows-X64-$Version.exe"
            Write-Host "Downloading Vulkan SDK $Version from LunarG..."
            Invoke-WebRequest -UseBasicParsing -Uri $uri -OutFile $InstallerPath
        }
    }

    $InstallerPath = (Resolve-Path $InstallerPath).Path
    $actualSha256 = (Get-FileHash -Algorithm SHA256 -Path $InstallerPath).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "Vulkan SDK checksum mismatch: expected $ExpectedSha256, got $actualSha256"
    }

    New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
    Write-Host "Installing Vulkan SDK $Version into $InstallRoot (copy-only)..."
    & $InstallerPath `
        --root $InstallRoot `
        --accept-licenses `
        --default-answer `
        --confirm-command `
        install `
        copy_only=1
    if ($LASTEXITCODE -ne 0) {
        throw "Vulkan SDK installer exited with code $LASTEXITCODE"
    }
}

Assert-VulkanSdk -Root $InstallRoot

$env:VULKAN_SDK = $InstallRoot
$sdkBin = Join-Path $InstallRoot "Bin"
$env:PATH = "$sdkBin;$env:PATH"

if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "VULKAN_SDK=$InstallRoot"
}
if ($env:GITHUB_PATH) {
    Add-Content -Path $env:GITHUB_PATH -Value $sdkBin
}

Write-Host "VULKAN_SDK=$InstallRoot"
Write-Host "PASS: Vulkan SDK $Version is ready"
