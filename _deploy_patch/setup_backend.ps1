$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$BackendDir = Join-Path $Root "gpts"
$VenvDir = Join-Path $BackendDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
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

function Normalize-PathEntry {
    param([string]$Entry)

    if ([string]::IsNullOrWhiteSpace($Entry)) {
        return $null
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($Entry).Trim()
    try {
        return [System.IO.Path]::GetFullPath($expanded).TrimEnd('\')
    }
    catch {
        return $expanded.TrimEnd('\')
    }
}

function Add-PathEntry {
    param(
        [string]$ExistingPath,
        [string]$Entry
    )

    if ([string]::IsNullOrWhiteSpace($Entry) -or -not (Test-Path $Entry)) {
        return $ExistingPath
    }

    $normalizedEntry = Normalize-PathEntry -Entry $Entry
    $parts = @()

    if (-not [string]::IsNullOrWhiteSpace($ExistingPath)) {
        $parts = @($ExistingPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    }

    foreach ($part in $parts) {
        if ((Normalize-PathEntry -Entry $part) -eq $normalizedEntry) {
            return ($parts -join ';')
        }
    }

    $parts += $Entry
    return ($parts -join ';')
}

function Ensure-PythonOnPath {
    param([string]$PythonPath)

    if ([string]::IsNullOrWhiteSpace($PythonPath) -or -not (Test-Path $PythonPath)) {
        return
    }

    $pythonDir = Split-Path -Parent $PythonPath
    $scriptsDir = Join-Path $pythonDir "Scripts"

    $processPath = $env:Path
    $updatedProcessPath = Add-PathEntry -ExistingPath $processPath -Entry $pythonDir
    $updatedProcessPath = Add-PathEntry -ExistingPath $updatedProcessPath -Entry $scriptsDir
    if ($updatedProcessPath -ne $processPath) {
        $env:Path = $updatedProcessPath
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $updatedUserPath = Add-PathEntry -ExistingPath $userPath -Entry $pythonDir
    $updatedUserPath = Add-PathEntry -ExistingPath $updatedUserPath -Entry $scriptsDir
    if ($updatedUserPath -ne $userPath) {
        [Environment]::SetEnvironmentVariable("Path", $updatedUserPath, "User")
        Write-Host "Added Python 3.10 to user PATH."
    }
}

function Test-NvidiaDriver {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Write-Warning "nvidia-smi not found. NVIDIA driver may be missing, not on PATH, or this PC may not have an NVIDIA GPU."
        return $false
    }

    Write-Host "Checking NVIDIA driver with nvidia-smi..."
    & $nvidiaSmi.Source --query-gpu=name,driver_version --format=csv,noheader
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "nvidia-smi failed. GPU driver is not ready for CUDA execution."
        return $false
    }

    return $true
}

function Get-NvidiaCudaVersion {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $null
    }

    $raw = & $nvidiaSmi.Source 2>$null | Out-String
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) {
        return $null
    }

    $match = [regex]::Match($raw, 'CUDA Version:\s*([0-9]+\.[0-9]+)')
    if (-not $match.Success) {
        return $null
    }

    try {
        return [version]$match.Groups[1].Value
    }
    catch {
        return $null
    }
}

function Get-TorchInstallTarget {
    param([version]$CudaVersion)

    $targets = @(
        @{ MinimumCuda = [version]'13.0'; Label = 'cu130'; IndexUrl = 'https://download.pytorch.org/whl/cu130'; Torch = '2.10.0'; TorchVision = '0.25.0' },
        @{ MinimumCuda = [version]'12.8'; Label = 'cu128'; IndexUrl = 'https://download.pytorch.org/whl/cu128'; Torch = '2.10.0'; TorchVision = '0.25.0' },
        @{ MinimumCuda = [version]'12.6'; Label = 'cu126'; IndexUrl = 'https://download.pytorch.org/whl/cu126'; Torch = '2.10.0'; TorchVision = '0.25.0' },
        @{ MinimumCuda = [version]'12.1'; Label = 'cu121'; IndexUrl = 'https://download.pytorch.org/whl/cu121'; Torch = '2.5.1'; TorchVision = '0.20.1' },
        @{ MinimumCuda = [version]'11.8'; Label = 'cu118'; IndexUrl = 'https://download.pytorch.org/whl/cu118'; Torch = '2.7.1'; TorchVision = '0.22.1' }
    )

    if ($CudaVersion) {
        foreach ($target in $targets) {
            if ($CudaVersion -ge $target.MinimumCuda) {
                return $target
            }
        }
    }

    return @{
        MinimumCuda = $null
        Label = 'cpu'
        IndexUrl = 'https://download.pytorch.org/whl/cpu'
        Torch = '2.10.0'
        TorchVision = '0.25.0'
    }
}

function Install-TorchStack {
    param([string]$PythonPath)

    $cudaVersion = $null
    if (Test-NvidiaDriver) {
        $cudaVersion = Get-NvidiaCudaVersion
    }

    $target = Get-TorchInstallTarget -CudaVersion $cudaVersion
    if ($cudaVersion) {
        Write-Host ("Detected NVIDIA CUDA {0}. Installing PyTorch wheel target {1}." -f $cudaVersion, $target.Label)
    }
    else {
        Write-Host ("Installing PyTorch wheel target {0}." -f $target.Label)
    }

    & $PythonPath -m pip install --upgrade --force-reinstall --index-url $target.IndexUrl ("torch==" + $target.Torch) ("torchvision==" + $target.TorchVision)
    if ($LASTEXITCODE -ne 0) {
        throw ("PyTorch install failed for target {0} with exit code {1}" -f $target.Label, $LASTEXITCODE)
    }

    # Torch wheel installation can upgrade numpy to 2.x, which breaks our pinned OpenCV/Ultralytics stack.
    & $PythonPath -m pip install --upgrade --force-reinstall "numpy==1.26.4" "opencv-python==4.8.1.78" "Pillow==10.1.0"
    if ($LASTEXITCODE -ne 0) {
        throw ("Failed to restore compatible numpy/opencv/pillow versions with exit code {0}" -f $LASTEXITCODE)
    }
}

function Get-PythonCommand {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        $py310 = (& $pyCmd.Source -3.10 -c "import sys; print(sys.executable)").Trim()
        if ($LASTEXITCODE -eq 0 -and $py310) {
            return @{
                Exe = $pyCmd.Source
                Args = @("-3.10")
                PythonPath = $py310
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
                PythonPath = $candidate
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
            $py310 = (& $pyCmd.Source -3.10 -c "import sys; print(sys.executable)").Trim()
            if ($LASTEXITCODE -eq 0 -and $py310) {
                return @{
                    Exe = $pyCmd.Source
                    Args = @("-3.10")
                    PythonPath = $py310
                }
            }
        }

        foreach ($candidate in $preferredPython) {
            if (Test-Path $candidate) {
                return @{
                    Exe = $candidate
                    Args = @()
                    PythonPath = $candidate
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

$LogPath = Initialize-LogFile -Prefix "setup"
Write-Host "Setup log: $LogPath"
try {
    Start-Transcript -Path $LogPath -Append | Out-Null

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
        Ensure-PythonOnPath -PythonPath $pythonCmd["PythonPath"]
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

    Install-TorchStack -PythonPath $PythonExe

    & $PythonExe -c "import cv2, flask, ultralytics, torch; print('OK:', cv2.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "post-install import verification failed with exit code $LASTEXITCODE"
    }

    & $PythonExe -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())"
    if ($LASTEXITCODE -ne 0) {
        throw "torch CUDA verification failed with exit code $LASTEXITCODE"
    }

    & $PythonExe -c "import torch, sys; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "PyTorch CUDA is not available. Use 2-run.bat for CPU mode, or install/update the NVIDIA driver and rerun 1-install.bat."
    }

    Write-Host "Backend virtual environment is ready."
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
}
