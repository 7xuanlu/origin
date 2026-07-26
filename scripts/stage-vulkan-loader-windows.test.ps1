$ErrorActionPreference = "Stop"

$stageScript = Join-Path $PSScriptRoot "stage-vulkan-loader-windows.ps1"
$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) "wenlan-vulkan-loader-test-$PID"
$componentRoot = Join-Path `
    $testRoot `
    "archive\VulkanRT-X64-1.4.350.0-Components\x64"
$runtimeRoot = Split-Path -Parent $componentRoot
$archivePath = Join-Path $testRoot "vulkan-runtime-components.zip"
$destination = Join-Path $testRoot "staged"
$wrongDestination = Join-Path $testRoot "wrong-hash"
$githubPath = Join-Path $testRoot "github-path.txt"
$previousGithubPath = $env:GITHUB_PATH
$previousPath = $env:PATH

try {
    New-Item -ItemType Directory -Force -Path $componentRoot | Out-Null
    $fixtureLoader = Join-Path $componentRoot "vulkan-1.dll"
    $fixtureLicense = Join-Path $runtimeRoot "VulkanRT-License.txt"
    Set-Content -LiteralPath $fixtureLoader -Value "fixture Vulkan loader" -NoNewline
    Set-Content -LiteralPath $fixtureLicense -Value "fixture Vulkan runtime license" -NoNewline
    Compress-Archive `
        -Path (Join-Path $testRoot "archive\*") `
        -DestinationPath $archivePath
    New-Item -ItemType File -Force -Path $githubPath | Out-Null

    $archiveSha256 = (
        Get-FileHash -LiteralPath $archivePath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    $loaderSha256 = (
        Get-FileHash -LiteralPath $fixtureLoader -Algorithm SHA256
    ).Hash.ToLowerInvariant()

    $env:GITHUB_PATH = $githubPath
    & $stageScript `
        -DestinationDirectory $destination `
        -ArchivePath $archivePath `
        -ExpectedArchiveSha256 $archiveSha256 `
        -ExpectedLoaderSha256 $loaderSha256 `
        -SkipAuthenticodeValidationForFixture

    $stagedLoader = Join-Path $destination "vulkan-1.dll"
    if (-not (Test-Path -LiteralPath $stagedLoader -PathType Leaf)) {
        throw "stage script did not copy the x64 Vulkan loader"
    }
    $stagedLicense = Join-Path $destination "VulkanRT-License.txt"
    if (-not (Test-Path -LiteralPath $stagedLicense -PathType Leaf)) {
        throw "stage script did not copy the Vulkan runtime license"
    }
    if (
        (Get-FileHash -LiteralPath $stagedLicense -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $fixtureLicense -Algorithm SHA256).Hash
    ) {
        throw "staged Vulkan runtime license changed"
    }
    if (
        (Get-FileHash -LiteralPath $stagedLoader -Algorithm SHA256).Hash `
            -ne $loaderSha256
    ) {
        throw "staged Vulkan loader hash changed"
    }
    if (-not (Get-Content -LiteralPath $githubPath -Raw).Contains($destination)) {
        throw "stage script did not write GITHUB_PATH"
    }
    if (-not $env:PATH.StartsWith($destination, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "stage script did not update the current process PATH"
    }

    $failed = $false
    try {
        & $stageScript `
            -DestinationDirectory $wrongDestination `
            -ArchivePath $archivePath `
            -ExpectedArchiveSha256 ("0" * 64) `
            -ExpectedLoaderSha256 $loaderSha256 `
            -SkipAuthenticodeValidationForFixture
    }
    catch {
        $failed = $true
        if (-not $_.Exception.Message.Contains("archive checksum mismatch")) {
            throw "bad archive hash failed for the wrong reason: $($_.Exception.Message)"
        }
    }
    if (-not $failed) {
        throw "bad Vulkan runtime archive hash unexpectedly passed"
    }

    $source = Get-Content -LiteralPath $stageScript -Raw
    if (-not $source.Contains("sdk.lunarg.com")) {
        throw "stage script must download the pinned official LunarG runtime"
    }
    if (
        -not $source.Contains("Get-AuthenticodeSignature") -or
        -not $source.Contains('"LunarG, Inc."')
    ) {
        throw "stage script must authenticate the LunarG-signed x64 loader"
    }
    if ($source.Contains("System32")) {
        throw "stage script must not mutate the machine-wide Vulkan runtime"
    }

    Write-Host "PASS: Vulkan loader staging script contract"
}
finally {
    if ($null -eq $previousGithubPath) {
        Remove-Item Env:GITHUB_PATH -ErrorAction SilentlyContinue
    }
    else {
        $env:GITHUB_PATH = $previousGithubPath
    }
    $env:PATH = $previousPath
    Remove-Item -Recurse -Force $testRoot -ErrorAction SilentlyContinue
}
