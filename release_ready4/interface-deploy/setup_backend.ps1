$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$BackendDir = Join-Path $Root "gpts"
$VenvDir = Join-Path $BackendDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

function Get-PythonCommand {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        & $pyCmd.Source -3.10 -c "import sys; print(sys.executable)" | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{
                Exe = $pyCmd.Source
                Args = @("-3.10")
            }
        }
    }

    $preferredPython = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"),
        "C:\Program Files\Python310\python.exe",
        "C:\Python310\python.exe"
    )

    foreach ($candidate in $preferredPython) {
        if (Test-Path $candidate) {
            return @{
                Exe = $candidate
                Args = @()
            }
        }
    }

    $wingetCmd = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetCmd) {
        Write-Host "Python not found. Installing Python 3.10 with winget..."
        winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.10 installation failed with exit code $LASTEXITCODE"
        }

        $pyCmd = Get-Command py -ErrorAction SilentlyContinue
        if ($pyCmd) {
            & $pyCmd.Source -3.10 -c "import sys; print(sys.executable)" | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return @{
                    Exe = $pyCmd.Source
                    Args = @("-3.10")
                }
            }
        }

        foreach ($candidate in $preferredPython) {
            if (Test-Path $candidate) {
                return @{
                    Exe = $candidate
                    Args = @()
                }
            }
        }
    }

    throw "Python 3.10 not found. Install Python 3.10 and rerun this file."
}

function Test-IsPython310 {
    param([string]$PythonPath)

    if (-not (Test-Path $PythonPath)) {
        return $false
    }

    & $PythonPath -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" | Out-Null
    return ($LASTEXITCODE -eq 0)
}

$needsCreateVenv = $true
if (Test-Path $PythonExe) {
    if (Test-IsPython310 -PythonPath $PythonExe) {
        $needsCreateVenv = $false
    } else {
        Write-Host "Existing virtual environment is not Python 3.10. Recreating..."
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
}

if ($needsCreateVenv) {
    $pythonCmd = Get-PythonCommand
    $createArgs = @()
    $createArgs += $pythonCmd["Args"]
    $createArgs += "-m", "venv", $VenvDir
    & $pythonCmd["Exe"] @createArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment with exit code $LASTEXITCODE"
    }
}

& $PythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE"
}

& $PythonExe -m pip install -r (Join-Path $BackendDir "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "requirements installation failed with exit code $LASTEXITCODE"
}

& $PythonExe -c "import cv2, flask, ultralytics, torch; print('OK:', cv2.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "post-install import verification failed with exit code $LASTEXITCODE"
}

Write-Host "Backend virtual environment is ready."
