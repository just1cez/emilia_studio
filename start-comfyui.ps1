param(
    [string]$Listen = '127.0.0.1',
    [int]$Port = 8188
)

$ErrorActionPreference = 'Stop'

$Root = $PSScriptRoot
$App = Join-Path $Root 'app'
. (Join-Path $Root 'scripts\EmiliaPython.ps1')
$Python = Resolve-EmiliaPython -Root $Root

if (-not (Test-Path -LiteralPath (Join-Path $App 'main.py') -PathType Leaf)) {
    throw "ComfyUI application was not found: $App"
}

# Keep caches and temporary files inside this project for this process only.
$env:PIP_CACHE_DIR = Join-Path $Root 'cache\pip'
$env:TEMP = Join-Path $Root 'temp'
$env:TMP = $env:TEMP

Set-Location -LiteralPath $App
& $Python (Join-Path $App 'main.py') --listen $Listen --port $Port --reserve-vram 0.8 --fp32-vae --disable-pinned-memory --mmap-torch-files --preview-method latent2rgb --disable-api-nodes
exit $LASTEXITCODE
