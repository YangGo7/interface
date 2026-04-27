$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$BackendDir = Join-Path $Root "gpts"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
$DotEnvPath = Join-Path $BackendDir ".env"
$LogsDir = Join-Path $Root "logs"

function Initialize-LogFile {
    param([string]$Prefix)

    New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
    $cutoff = (Get-Date).AddDays(-7)
    Get-ChildItem -Path $LogsDir -File -Filter "*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    return Join-Path $LogsDir ("{0}-{1}.log" -f $Prefix, (Get-Date -Format "yyyyMMdd-HHmmss"))
}

function Get-DotEnvValue {
    param(
        [string]$Path,
        [string]$Name
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    foreach ($line in Get-Content -Path $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }
        if ($parts[0].Trim() -ne $Name) {
            continue
        }
        return $parts[1].Trim()
    }

    return $null
}

function Resolve-EnvSetting {
    param(
        [string]$CurrentValue,
        [string]$DotEnvName,
        [string]$DefaultValue
    )

    if (-not [string]::IsNullOrWhiteSpace($CurrentValue)) {
        return $CurrentValue
    }

    $dotEnvValue = Get-DotEnvValue -Path $DotEnvPath -Name $DotEnvName
    if (-not [string]::IsNullOrWhiteSpace($dotEnvValue)) {
        return $dotEnvValue
    }

    return $DefaultValue
}

$env:FLASK_ENV = Resolve-EnvSetting -CurrentValue $env:FLASK_ENV -DotEnvName "FLASK_ENV" -DefaultValue "production"
$env:PANO_DEVICE = Resolve-EnvSetting -CurrentValue $env:PANO_DEVICE -DotEnvName "PANO_DEVICE" -DefaultValue "cpu"
$env:ENABLE_OPTIONAL_LLM_ROUTES = Resolve-EnvSetting -CurrentValue $env:ENABLE_OPTIONAL_LLM_ROUTES -DotEnvName "ENABLE_OPTIONAL_LLM_ROUTES" -DefaultValue "0"

$LogPath = Initialize-LogFile -Prefix "server"
Write-Host "Server log: $LogPath"
try {
    Start-Transcript -Path $LogPath -Append | Out-Null

    if (Test-Path $VenvPython) {
        & $VenvPython (Join-Path $BackendDir "app.py")
        if ($LASTEXITCODE -ne 0) {
            throw "Server exited with code $LASTEXITCODE"
        }
    }
    else {
        throw "Installation is not complete. Run .\0-install-and-run.bat or .\1-install.bat first."
    }
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
}
