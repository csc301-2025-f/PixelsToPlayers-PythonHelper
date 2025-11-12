<#
    Builds the PixelsToPlayers MSI with WiX v6.
    - Ensures the WiX CLI and UI extension are available
    - Targets the shared project root while keeping Windows artifacts local
    - Emits the MSI to platforms/windows/dist (override via -OutputPath)
#>
param(
    [string]$OutputPath = "platforms/windows/dist/PixelsToPlayers.msi"
)

set-psdebug -strict
$ErrorActionPreference = "Stop"

function Ensure-Command {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing command '$Name'. $InstallHint"
    }
}

Ensure-Command -Name "dotnet" -InstallHint "Install .NET 8+ SDK to continue."

if (-not (Get-Command "wix" -ErrorAction SilentlyContinue)) {
    Write-Host "WiX CLI not detected. Installing globally via dotnet tool..."
    dotnet tool install --global wix --version "6.*"
}

Write-Host "Adding WixToolset.UI extension (idempotent)..."
wix extension add WixToolset.UI.wixext | Out-Null

$windowsRoot = $PSScriptRoot
$projectRoot = (Resolve-Path (Join-Path $windowsRoot "..\\..")).Path
$wxsPath = Join-Path $windowsRoot "wix/PixelsToPlayers.wxs"
$licensePath = Join-Path $windowsRoot "wix/License.rtf"
$appExe = Join-Path $windowsRoot "dist/PixelsToPlayers/PixelsToPlayers.exe"

foreach ($path in @($wxsPath, $licensePath, $appExe)) {
    if (-not (Test-Path $path)) {
        throw "Required file not found: $path"
    }
}

$outputFullPath = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath
} else {
    Join-Path $projectRoot $OutputPath
}

New-Item -ItemType Directory -Path (Split-Path $outputFullPath) -Force | Out-Null

Write-Host "Building installer..."
wix build $wxsPath `
    -ext WixToolset.UI.wixext `
    -d ProjectRoot="$projectRoot" `
    -o "$outputFullPath"

if ($LASTEXITCODE -ne 0) {
    throw "WiX build failed with exit code $LASTEXITCODE"
}

Write-Host "MSI created at $outputFullPath"
