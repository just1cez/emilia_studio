# Shared by the root launchers. Resolution order mirrors emilia_studio.config.load_config:
# EMILIA_CONDA_PYTHON, then conda_python from EMILIA_CONFIG_TOML, then user config
# (EMILIA_CONFIG_DIR or %APPDATA%\EmiliaStudio\config.toml), then checkout config.toml,
# then a local .\env interpreter. Relative paths in TOML resolve beside that TOML file.
function Resolve-EmiliaPython {
    param([Parameter(Mandatory)][string]$Root)

    $Candidates = @()
    if ($env:EMILIA_CONDA_PYTHON) {
        $Candidates += $env:EMILIA_CONDA_PYTHON
    }
    $Configs = @()
    if ($env:EMILIA_CONFIG_TOML -and $env:EMILIA_CONFIG_TOML.Trim()) {
        $Configs += $env:EMILIA_CONFIG_TOML.Trim()
    } else {
        if ($env:EMILIA_CONFIG_DIR -and $env:EMILIA_CONFIG_DIR.Trim()) {
            $Configs += Join-Path $env:EMILIA_CONFIG_DIR.Trim() 'config.toml'
        } elseif ($env:APPDATA) {
            $Configs += Join-Path $env:APPDATA 'EmiliaStudio\config.toml'
        }
        $Configs += Join-Path $Root 'config.toml'
    }
    foreach ($Config in $Configs) {
        if (Test-Path -LiteralPath $Config -PathType Leaf) {
            $Match = Select-String -LiteralPath $Config -Pattern "^\s*conda_python\s*=\s*(?:`"((?:[^`"\\]|\\.)*)`"|'([^']*)')" |
                Select-Object -First 1
            if ($Match) {
                $Raw = if ($Match.Matches[0].Groups[1].Success) {
                    $Match.Matches[0].Groups[1].Value -replace '\\\\', '\' -replace '\\"', '"'
                } else {
                    $Match.Matches[0].Groups[2].Value
                }
                if ($Raw) {
                    if (-not [IO.Path]::IsPathRooted($Raw)) {
                        $ConfigDir = Split-Path -Parent ([IO.Path]::GetFullPath($Config))
                        $Raw = [IO.Path]::GetFullPath((Join-Path $ConfigDir $Raw))
                    }
                    $Candidates += $Raw
                }
            }
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
