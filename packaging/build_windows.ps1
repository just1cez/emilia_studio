param(
    [string]$Python = "python",
    [long]$SourceDateEpoch = 1767225600
)

$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT") {
    throw "The Windows executable must be built on Windows."
}
$Root = Split-Path -Parent $PSScriptRoot
$PreviousEpoch = $env:SOURCE_DATE_EPOCH
$env:SOURCE_DATE_EPOCH = [string]$SourceDateEpoch
Push-Location $Root
try {
    # Install build tools first: python -m pip install ".[build]"
    & $Python -m build --wheel --outdir dist .
    if ($LASTEXITCODE -ne 0) { throw "Wheel build failed ($LASTEXITCODE)." }
    & $Python -m PyInstaller --clean --noconfirm --distpath dist --workpath build/pyinstaller packaging/emilia.spec
    if ($LASTEXITCODE -ne 0) { throw "Windows executable build failed ($LASTEXITCODE)." }
} finally {
    Pop-Location
    $env:SOURCE_DATE_EPOCH = $PreviousEpoch
}
