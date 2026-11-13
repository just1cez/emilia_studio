param(
    [UInt64]$Seed = 20261027,
    [ValidateSet('quality', 'ultra')]
    [string]$Profile = 'quality',
    [switch]$Pag,
    [switch]$AutoGate,
    [switch]$Fast
)

# Thin wrapper over `emilia render`; every option maps 1:1 onto the CLI.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\EmiliaPython.ps1')
$Python = Resolve-EmiliaPython -Root $PSScriptRoot

$Arguments = @('-m', 'emilia_studio.cli', 'render', '--seed', $Seed, '--profile', $Profile)
if ($Pag) {
    $Arguments += '--pag'
}
if ($AutoGate) {
    $Arguments += '--auto-gate'
}
if ($Fast) {
    $Arguments += '--fast'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python @Arguments
} finally {
    Pop-Location
}
exit $LASTEXITCODE
