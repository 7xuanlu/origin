param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationDirectory,

    [string]$Version = "1.4.350.0",

    [string]$ExpectedArchiveSha256 = "23ce69f32cef3e2799617e2b1776cd0c71030d23a91f8375821cc40d76b185b9",

    [string]$ExpectedLoaderSha256 = "0419974f00e82a3d619077ba414da265a774f8db9d45ad93bc1843f44b2c2c1f",

    [string]$ArchivePath,

    [switch]$SkipAuthenticodeValidationForFixture
)

$ErrorActionPreference = "Stop"

$tempBase = if ($env:RUNNER_TEMP) {
    $env:RUNNER_TEMP
}
else {
    [System.IO.Path]::GetTempPath()
}
$tempBase = [System.IO.Path]::GetFullPath($tempBase)
$tempRoot = Join-Path $tempBase "wenlan-vulkan-loader-$([System.Guid]::NewGuid())"
$tempRoot = [System.IO.Path]::GetFullPath($tempRoot)
$safePrefix = $tempBase.TrimEnd(
    [System.IO.Path]::DirectorySeparatorChar,
    [System.IO.Path]::AltDirectorySeparatorChar
) + [System.IO.Path]::DirectorySeparatorChar
if (-not $tempRoot.StartsWith($safePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "unsafe Vulkan loader staging directory '$tempRoot'"
}

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

try {
    if ($ArchivePath) {
        $resolvedArchive = (Resolve-Path -LiteralPath $ArchivePath).Path
    }
    else {
        $resolvedArchive = Join-Path $tempRoot "vulkan-runtime-components.zip"
        $archiveUrl = (
            "https://sdk.lunarg.com/sdk/download/" +
            "$Version/windows/vulkan-runtime-components.zip"
        )
        Write-Host "Downloading the official LunarG Vulkan runtime components $Version..."
        # Bounded on purpose: the default is no timeout at all, so a stalled
        # connection to sdk.lunarg.com would hang the build silently.
        Invoke-WebRequest `
            -UseBasicParsing `
            -Uri $archiveUrl `
            -OutFile $resolvedArchive `
            -TimeoutSec 120
    }

    $actualArchiveSha256 = (
        Get-FileHash -LiteralPath $resolvedArchive -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($actualArchiveSha256 -ne $ExpectedArchiveSha256.ToLowerInvariant()) {
        throw (
            "Vulkan runtime archive checksum mismatch: expected " +
            "$ExpectedArchiveSha256, got $actualArchiveSha256"
        )
    }

    $extractRoot = Join-Path $tempRoot "extracted"
    Expand-Archive -LiteralPath $resolvedArchive -DestinationPath $extractRoot
    $loaders = @(
        Get-ChildItem `
            -LiteralPath $extractRoot `
            -Recurse `
            -File `
            -Filter "vulkan-1.dll" |
            Where-Object { $_.Directory.Name -ieq "x64" }
    )
    if ($loaders.Count -ne 1) {
        throw "expected exactly one x64 vulkan-1.dll, found $($loaders.Count)"
    }
    $licenses = @(
        Get-ChildItem `
            -LiteralPath $extractRoot `
            -Recurse `
            -File `
            -Filter "VulkanRT-License.txt"
    )
    if ($licenses.Count -ne 1) {
        throw "expected exactly one VulkanRT-License.txt, found $($licenses.Count)"
    }

    $actualLoaderSha256 = (
        Get-FileHash -LiteralPath $loaders[0].FullName -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($actualLoaderSha256 -ne $ExpectedLoaderSha256.ToLowerInvariant()) {
        throw (
            "Vulkan loader checksum mismatch: expected " +
            "$ExpectedLoaderSha256, got $actualLoaderSha256"
        )
    }
    if (-not $SkipAuthenticodeValidationForFixture) {
        $signature = Get-AuthenticodeSignature -LiteralPath $loaders[0].FullName
        $signerName = if ($signature.SignerCertificate) {
            $signature.SignerCertificate.GetNameInfo(
                [System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName,
                $false
            )
        }
        else {
            ""
        }
        if ($signature.Status -ne "Valid" -or $signerName -ne "LunarG, Inc.") {
            throw (
                "Vulkan loader Authenticode verification failed: " +
                "status=$($signature.Status), signer='$signerName'"
            )
        }
    }

    $destination = [System.IO.Path]::GetFullPath($DestinationDirectory)
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    $stagedLoader = Join-Path $destination "vulkan-1.dll"
    $stagedLicense = Join-Path $destination "VulkanRT-License.txt"
    Copy-Item -LiteralPath $loaders[0].FullName -Destination $stagedLoader -Force
    Copy-Item -LiteralPath $licenses[0].FullName -Destination $stagedLicense -Force

    $pathEntries = @($env:PATH -split ";" | Where-Object { $_.Trim() })
    if (-not ($pathEntries | Where-Object { $_ -ieq $destination })) {
        $env:PATH = "$destination;$env:PATH"
    }
    if ($env:GITHUB_PATH) {
        Add-Content -LiteralPath $env:GITHUB_PATH -Value $destination
    }

    Write-Host "VULKAN_LOADER=$stagedLoader"
    Write-Host "VULKAN_RUNTIME_LICENSE=$stagedLicense"
    Write-Host "PASS: official LunarG Vulkan loader $Version is staged"
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
