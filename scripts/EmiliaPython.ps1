# Shared by the root launchers. Resolution order mirrors emilia_studio.config.load_config:
# EMILIA_CONDA_PYTHON, then conda_python from the user config, then from the checkout
# config.toml, then a local .\env interpreter.
function Resolve-EmiliaPython {
    param([Parameter(Mandatory)][string]$Root)

    $Candidates = @($env:EMILIA_CONDA_PYTHON)
    $Configs = @()
    if ($env:APPDATA) { $Configs += Join-Path $env:APPDATA 'EmiliaStudio\config.toml' }
    $Configs += Join-Path $Root 'config.toml'
    foreach ($Config in $Configs) {
        if (Test-Path -LiteralPath $Config -PathType Leaf) {
            $Match = Select-String -LiteralPath $Config -Pattern '^\s*conda_python\s*=\s*"([^"]+)"' |
                Select-Object -First 1
            if ($Match) { $Candidates += $Match.Matches[0].Groups[1].Value }
        }
    }
    $Candidates += Join-Path $Root 'env\python.exe'
    $Found = $Candidates |
        Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        Select-Object -First 1
    if (-not $Found) {
        throw 'Inference Python was not found. Set EMILIA_CONDA_PYTHON or conda_python in config.toml.'
    }
    return $Found
}
