$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$env:PANO_DEVICE = "gpu"
& (Join-Path $PSScriptRoot "start_server.ps1")
