param(
    [string]$VisualStudioInstallPath
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Get-UniqueSemicolonList {
    param([string[]]$Values)

    $seen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    $items = foreach ($value in $Values) {
        foreach ($item in ($value -split ";")) {
            $trimmed = $item.Trim()
            if ($trimmed -and $seen.Add($trimmed)) {
                $trimmed
            }
        }
    }
    return ($items -join ";")
}

if (-not $VisualStudioInstallPath) {
    $vsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vsWhere -PathType Leaf)) {
        $vsWhereCommand = Get-Command vswhere.exe -ErrorAction SilentlyContinue
        if ($vsWhereCommand) {
            $vsWhere = $vsWhereCommand.Source
        }
    }
    if (-not (Test-Path -LiteralPath $vsWhere -PathType Leaf)) {
        throw "vswhere.exe is required to locate the MSVC C++ build tools"
    }

    $VisualStudioInstallPath = (
        & $vsWhere `
            -latest `
            -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
    ) | Select-Object -First 1
}

if (-not $VisualStudioInstallPath) {
    throw "Visual Studio with the MSVC x64 build tools was not found"
}
$VisualStudioInstallPath = [System.IO.Path]::GetFullPath($VisualStudioInstallPath.Trim())

$devCmd = Join-Path $VisualStudioInstallPath "Common7\Tools\VsDevCmd.bat"
if (-not (Test-Path -LiteralPath $devCmd -PathType Leaf)) {
    throw "Visual Studio developer command file is missing at '$devCmd'"
}

$originalPath = $env:PATH
$originalLib = $env:LIB
$devCommand = "call `"$devCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
$devEnvironment = & $env:ComSpec /d /s /c $devCommand
if ($LASTEXITCODE -ne 0) {
    throw "VsDevCmd.bat failed with exit code $LASTEXITCODE"
}

foreach ($line in $devEnvironment) {
    if ($line -match "^([^=][^=]*)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# The SQLite vcpkg path is installed before this script in CI. VsDevCmd sets
# LIB from scratch, so preserve the caller-owned prefix while importing MSVC.
$env:LIB = Get-UniqueSemicolonList @($originalLib, $env:LIB)

$ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
if (-not $ninja) {
    $bundledNinja = Join-Path `
        $VisualStudioInstallPath `
        "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
    if (Test-Path -LiteralPath $bundledNinja -PathType Leaf) {
        $ninjaDirectory = Split-Path -Parent $bundledNinja
        $env:PATH = Get-UniqueSemicolonList @($ninjaDirectory, $env:PATH)
        $ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
    }
}
if (-not $ninja) {
    throw "ninja.exe was not found in PATH or the Visual Studio CMake tools"
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "VsDevCmd.bat did not expose the x64 MSVC compiler"
}

$env:CMAKE_GENERATOR = "Ninja"
$env:CMAKE_BUILD_PARALLEL_LEVEL = "1"

if ($env:GITHUB_ENV) {
    foreach ($name in @(
        "INCLUDE",
        "LIB",
        "LIBPATH",
        "VCINSTALLDIR",
        "VCToolsInstallDir",
        "VSINSTALLDIR",
        "VisualStudioVersion",
        "WindowsSdkDir",
        "WindowsSDKVersion",
        "UniversalCRTSdkDir",
        "UCRTVersion",
        "CMAKE_GENERATOR",
        "CMAKE_BUILD_PARALLEL_LEVEL"
    )) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        if ($null -ne $value) {
            Add-Content -LiteralPath $env:GITHUB_ENV -Value "$name=$value"
        }
    }
}

if ($env:GITHUB_PATH) {
    $originalPathEntries = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    foreach ($entry in ($originalPath -split ";")) {
        if ($entry.Trim()) {
            [void]$originalPathEntries.Add($entry.Trim())
        }
    }
    foreach ($entry in ($env:PATH -split ";")) {
        $trimmed = $entry.Trim()
        if ($trimmed -and -not $originalPathEntries.Contains($trimmed)) {
            Add-Content -LiteralPath $env:GITHUB_PATH -Value $trimmed
        }
    }
}

Write-Host "MSVC=$((Get-Command cl.exe).Source)"
Write-Host "Ninja=$($ninja.Source)"
Write-Host "CMAKE_GENERATOR=$env:CMAKE_GENERATOR"
Write-Host "CMAKE_BUILD_PARALLEL_LEVEL=$env:CMAKE_BUILD_PARALLEL_LEVEL"
Write-Host "PASS: x64 MSVC Ninja environment is ready"
