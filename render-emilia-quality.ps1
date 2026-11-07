param(
    [UInt64]$Seed = 20261027,
    [ValidateSet('quality', 'ultra')]
    [string]$Profile = 'quality',
    [switch]$Pag,
    [switch]$AutoGate
)

$ErrorActionPreference = 'Stop'
$Candidates = @(
    $env:EMILIA_CONDA_PYTHON,
    'D:\conda-data\envs\comfyui\python.exe',
    (Join-Path $PSScriptRoot 'env\python.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }

if (-not $Candidates -or $Candidates.Count -eq 0) {
    throw "ComfyUI Python environment was not found."
}
$Python = $Candidates[0]
$Script = Join-Path $PSScriptRoot 'app\script_examples\render_emilia_quality.py'

$Arguments = @($Script, '--seed', $Seed, '--profile', $Profile)
if ($Pag) {
    $Arguments += '--pag'
}
if ($AutoGate) {
    $Arguments += '--auto-gate'
}
& $Python @Arguments
exit $LASTEXITCODE
