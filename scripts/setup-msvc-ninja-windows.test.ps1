$ErrorActionPreference = "Stop"

$setupScript = Join-Path $PSScriptRoot "setup-msvc-ninja-windows.ps1"
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) "wenlan-msvc-ninja-test-$PID"
$vsRoot = Join-Path $testRoot "Visual Studio"
$compilerDir = Join-Path $vsRoot "VC\Tools\MSVC\14.44.0\bin\Hostx64\x64"
$ninjaDir = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
$devShellDir = Join-Path $vsRoot "Common7\Tools"
$githubEnv = Join-Path $testRoot "github-env.txt"
$githubPath = Join-Path $testRoot "github-path.txt"

$trackedEnvironment = @(
    "PATH",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "VCINSTALLDIR",
    "VCToolsInstallDir",
    "VSINSTALLDIR",
    "VisualStudioVersion",
    "CMAKE_GENERATOR",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "GITHUB_ENV",
    "GITHUB_PATH"
)
$previousEnvironment = @{}
foreach ($name in $trackedEnvironment) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

try {
    New-Item -ItemType Directory -Force -Path $compilerDir, $ninjaDir, $devShellDir | Out-Null
    Copy-Item -LiteralPath $env:ComSpec -Destination (Join-Path $compilerDir "cl.exe")
    Copy-Item -LiteralPath $env:ComSpec -Destination (Join-Path $ninjaDir "ninja.exe")
    New-Item -ItemType File -Force -Path $githubEnv, $githubPath | Out-Null

    $devCmd = Join-Path $devShellDir "VsDevCmd.bat"
    @"
@echo off
set "PATH=$compilerDir;%PATH%"
set "INCLUDE=$testRoot\include"
set "LIB=$testRoot\msvc-lib"
set "LIBPATH=$testRoot\libpath"
set "VCINSTALLDIR=$vsRoot\VC\"
set "VCToolsInstallDir=$vsRoot\VC\Tools\MSVC\14.44.0\"
set "VSINSTALLDIR=$vsRoot\"
set "VisualStudioVersion=17.0"
"@ | Set-Content -LiteralPath $devCmd -Encoding ascii

    $env:GITHUB_ENV = $githubEnv
    $env:GITHUB_PATH = $githubPath
    # Keep the fixture independent of runner-image tools such as Chocolatey's
    # preinstalled Ninja. This forces the Visual Studio fallback path.
    $env:PATH = "$env:SystemRoot\System32"
    $env:LIB = "$testRoot\vcpkg-lib"

    & $setupScript -VisualStudioInstallPath $vsRoot

    if ($env:CMAKE_GENERATOR -ne "Ninja") {
        throw "setup did not select the Ninja CMake generator"
    }
    if ($env:CMAKE_BUILD_PARALLEL_LEVEL -ne "1") {
        throw "setup did not serialize nested CMake builds"
    }
    if (-not $env:LIB.Contains("$testRoot\vcpkg-lib")) {
        throw "setup discarded the existing vcpkg LIB path"
    }
    if (-not $env:LIB.Contains("$testRoot\msvc-lib")) {
        throw "setup did not import the MSVC LIB path"
    }
    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "setup did not expose the x64 MSVC compiler"
    }
    if (-not (Get-Command ninja.exe -ErrorAction SilentlyContinue)) {
        throw "setup did not expose Ninja"
    }

    $persistedEnvironment = Get-Content -LiteralPath $githubEnv -Raw
    foreach ($expected in @(
        "CMAKE_GENERATOR=Ninja",
        "CMAKE_BUILD_PARALLEL_LEVEL=1",
        "INCLUDE=$testRoot\include",
        "LIBPATH=$testRoot\libpath"
    )) {
        if (-not $persistedEnvironment.Contains($expected)) {
            throw "GITHUB_ENV is missing '$expected'"
        }
    }

    $persistedPath = Get-Content -LiteralPath $githubPath -Raw
    foreach ($expected in @($compilerDir, $ninjaDir)) {
        if (-not $persistedPath.Contains($expected)) {
            throw "GITHUB_PATH is missing '$expected'"
        }
    }

    Write-Host "PASS: MSVC Ninja setup script contract"
}
finally {
    foreach ($name in $trackedEnvironment) {
        $previous = $previousEnvironment[$name]
        if ($null -eq $previous) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $previous, "Process")
        }
    }
    Remove-Item -Recurse -Force $testRoot -ErrorAction SilentlyContinue
}
