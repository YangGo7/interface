$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$BackendDir = Join-Path $Root "gpts"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"

$env:FLASK_ENV = if ($env:FLASK_ENV) { $env:FLASK_ENV } else { "production" }
$env:PANO_DEVICE = if ($env:PANO_DEVICE) { $env:PANO_DEVICE } else { "cpu" }

if (Test-Path $VenvPython) {
    & $VenvPython (Join-Path $BackendDir "app.py")
}
else {
    throw "Installation is not complete. Run .\0-install-and-run.bat or .\1-install.bat first."
}
