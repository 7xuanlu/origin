param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationDirectory
)

$ErrorActionPreference = "Stop"

# Verified against ort 2.0.0-rc.11 (ort commit
# 2de34065983a5c034f5afcc072b23b99479f465b):
# ort-sys/build/download/dist.txt pins Windows x64 CPU to ms@1.23.2.
$OrtVersion = "1.23.2"
$ExpectedZipSha256 = "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd"
$ZipUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-$OrtVersion.zip"
$TempBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
$TempRoot = Join-Path $TempBase "wenlan-ort-$([System.Guid]::NewGuid())"
$ZipPath = Join-Path $TempRoot "onnxruntime.zip"
$ExtractPath = Join-Path $TempRoot "extracted"

New-Item -ItemType Directory -Path $TempRoot | Out-Null

try {
    Write-Host "Downloading $ZipUrl"
    Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath

    $ActualZipSha256 = (Get-FileHash -Path $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualZipSha256 -ne $ExpectedZipSha256) {
        throw "onnxruntime archive SHA-256 mismatch: expected $ExpectedZipSha256, got $ActualZipSha256"
    }

    Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath
    $Dll = Get-ChildItem -Path $ExtractPath -Recurse -Filter onnxruntime.dll |
        Select-Object -First 1
    if (-not $Dll) {
        throw "onnxruntime.dll not found in verified archive"
    }

    New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    $Destination = Join-Path $DestinationDirectory "onnxruntime.dll"
    Copy-Item -Path $Dll.FullName -Destination $Destination -Force
    Write-Host "Staged verified ONNX Runtime $OrtVersion at $Destination"
}
finally {
    Remove-Item -Recurse -Force $TempRoot -ErrorAction SilentlyContinue
}
