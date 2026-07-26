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
