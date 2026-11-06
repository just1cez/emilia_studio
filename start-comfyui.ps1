$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$App = Join-Path $Root 'app'
$Candidates = @(
    $env:EMILIA_CONDA_PYTHON,
    'D:\conda-data\envs\comfyui\python.exe',
    (Join-Path $Root 'env\python.exe')
) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }

if (-not $Candidates -or $Candidates.Count -eq 0) {
    throw "ComfyUI Python environment was not found."
}
$Python = $Candidates[0]

if (-not (Test-Path -LiteralPath (Join-Path $App 'main.py') -PathType Leaf)) {
    throw "ComfyUI application was not found: $App"
}

# Keep caches and temporary files inside this project for this process only.
$env:PIP_CACHE_DIR = Join-Path $Root 'cache\pip'
$env:TEMP = Join-Path $Root 'temp'
$env:TMP = $env:TEMP

Set-Location -LiteralPath $App
& $Python (Join-Path $App 'main.py') --listen 127.0.0.1 --port 8188 --reserve-vram 0.8 --fp32-vae --disable-pinned-memory --preview-method latent2rgb --disable-api-nodes
exit $LASTEXITCODE
