param(
    [string]$OutputRoot = "release",
    [string]$Version = "",
    [switch]$SkipFrontendBuild,
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-FullPathSafe {
    param([string]$PathValue)

    $combined = if ([System.IO.Path]::IsPathRooted($PathValue)) {
        $PathValue
    } else {
        Join-Path $RepoRoot $PathValue
    }

    return [System.IO.Path]::GetFullPath($combined)
}

function Assert-ChildPath {
    param(
        [string]$ParentPath,
        [string]$ChildPath
    )

    $normalizedParent = [System.IO.Path]::GetFullPath($ParentPath).TrimEnd('\', '/')
    $normalizedChild = [System.IO.Path]::GetFullPath($ChildPath)

    if (-not $normalizedChild.StartsWith($normalizedParent, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside target root: $normalizedChild"
    }
}

function Assert-ReleaseVersion {
    param([string]$VersionValue)

    if ([string]::IsNullOrWhiteSpace($VersionValue)) {
        return
    }

    if ($VersionValue -notmatch '^\d+\.\d+\.\d+([\-+][0-9A-Za-z.-]+)?$') {
        throw "Version must look like semver, for example 1.2.3 or 1.2.3-beta.1."
    }
}

function Get-PackageVersion {
    param([string]$PackageJsonPath)

    if (-not (Test-Path $PackageJsonPath)) {
        return "0.0.0"
    }

    $packageJson = Get-Content -Path $PackageJsonPath -Raw | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace($packageJson.version)) {
        return "0.0.0"
    }

    return [string]$packageJson.version
}

function Set-PackageVersion {
    param(
        [string]$PackageJsonPath,
        [string]$PackageLockPath,
        [string]$VersionValue
    )

    $packageJson = Get-Content -Path $PackageJsonPath -Raw | ConvertFrom-Json
    $packageJson.version = $VersionValue
    $packageJson | ConvertTo-Json -Depth 100 | Set-Content -Path $PackageJsonPath -Encoding ASCII

    if (Test-Path $PackageLockPath) {
        $packageLock = Get-Content -Path $PackageLockPath -Raw | ConvertFrom-Json
        $packageLock.version = $VersionValue
        $rootPackage = $packageLock.packages.PSObject.Properties[""]
        if ($rootPackage) {
            $rootPackage.Value.version = $VersionValue
        }
        $packageLock | ConvertTo-Json -Depth 100 | Set-Content -Path $PackageLockPath -Encoding ASCII
    }
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $RepoRoot "frontend"
$BackendDir = Join-Path $RepoRoot "gpts"
$FrontendPackageJson = Join-Path $FrontendDir "package.json"
$FrontendPackageLock = Join-Path $FrontendDir "package-lock.json"
$OutputRootFull = Get-FullPathSafe -PathValue $OutputRoot
$RequestedVersion = $Version.Trim()
Assert-ReleaseVersion -VersionValue $RequestedVersion

if (-not [string]::IsNullOrWhiteSpace($RequestedVersion)) {
    Write-Host "Updating frontend package version to $RequestedVersion..."
    Set-PackageVersion -PackageJsonPath $FrontendPackageJson -PackageLockPath $FrontendPackageLock -VersionValue $RequestedVersion
}

$ReleaseVersion = Get-PackageVersion -PackageJsonPath $FrontendPackageJson
$PackageName = if ([string]::IsNullOrWhiteSpace($RequestedVersion)) { "interface-deploy" } else { "interface-deploy-$ReleaseVersion" }
$PackageDir = Join-Path $OutputRootFull $PackageName
$ZipPath = Join-Path $OutputRootFull "$PackageName.zip"

Write-Host "Repo Root: $RepoRoot"
Write-Host "Package Dir: $PackageDir"
Write-Host "Release Version: $ReleaseVersion"

New-Item -ItemType Directory -Force -Path $OutputRootFull | Out-Null

if (-not $SkipFrontendBuild) {
    Write-Host "Building frontend..."
    Push-Location $FrontendDir
    try {
        if (-not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
            npm.cmd ci
            if ($LASTEXITCODE -ne 0) {
                throw "npm ci failed with exit code $LASTEXITCODE"
            }
        }
        npm.cmd run build
        if ($LASTEXITCODE -ne 0) {
            throw "npm run build failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$FrontendDistDir = Join-Path $FrontendDir "dist"
if (-not (Test-Path (Join-Path $FrontendDistDir "index.html"))) {
    throw "frontend/dist/index.html not found. Run frontend build first."
}

if (Test-Path $PackageDir) {
    Assert-ChildPath -ParentPath $OutputRootFull -ChildPath $PackageDir
    Remove-Item -LiteralPath $PackageDir -Recurse -Force
}

if ($Zip -and (Test-Path $ZipPath)) {
    Assert-ChildPath -ParentPath $OutputRootFull -ChildPath $ZipPath
    Remove-Item -LiteralPath $ZipPath -Force
}

New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir "frontend") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir "gpts") | Out-Null

$releaseManifest = @"
name=$PackageName
version=$ReleaseVersion
created_at=$(Get-Date -Format "yyyy-MM-ddTHH:mm:ssK")
"@
Set-Content -Path (Join-Path $PackageDir "VERSION.txt") -Value $releaseManifest -Encoding ASCII

Write-Host "Copying frontend dist contents..."
Copy-Item -Path (Join-Path $FrontendDistDir "*") -Destination (Join-Path $PackageDir "frontend") -Recurse -Force

$PackagedFrontendIndex = Join-Path $PackageDir "frontend\index.html"
if (-not (Test-Path $PackagedFrontendIndex)) {
    throw "Packaged frontend/index.html not found after copy."
}

$backendDirs = @(
    "api",
    "imgs",
    "models",
    "services",
    "templates",
    "test",
    "utils",
    "weights"
)

$backendFiles = @(
    ".env.example",
    "app.py",
    "config.py",
    "prompts.yaml",
    "requirements.txt",
    "requirements_api.txt"
)

Write-Host "Copying backend files..."
foreach ($dir in $backendDirs) {
    Copy-Item -Path (Join-Path $BackendDir $dir) -Destination (Join-Path $PackageDir "gpts") -Recurse -Force
}

foreach ($file in $backendFiles) {
    $sourceFile = Join-Path $BackendDir $file
    if (Test-Path $sourceFile) {
        Copy-Item -Path $sourceFile -Destination (Join-Path $PackageDir "gpts") -Force
    }
    else {
        Write-Warning "Skipping missing backend file: $sourceFile"
    }
}

foreach ($dir in @("data", "reports", "temp", "runs")) {
    New-Item -ItemType Directory -Force -Path (Join-Path (Join-Path $PackageDir "gpts") $dir) | Out-Null
}

New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir "logs") | Out-Null

Copy-Item -Path (Join-Path $RepoRoot "docs\BUILD_OTHER_PC.md") -Destination (Join-Path $PackageDir "BUILD_OTHER_PC.md") -Force

$releaseEnv = @"
FLASK_ENV=production
SECRET_KEY=change-me
PANO_DEVICE=cuda
ENABLE_OPTIONAL_LLM_ROUTES=0
# GEMINI_API_KEY=
"@
Set-Content -Path (Join-Path $PackageDir "gpts\.env") -Value $releaseEnv -Encoding ASCII

$setupScript = @'
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
        @{ MinimumCuda = [version]'12.1'; Label = 'cu121'; IndexUrl = 'https://download.pytorch.org/whl/cu121'; Torch = '2.5.1'; TorchVision = '0.20.1' },
        @{ MinimumCuda = [version]'11.8'; Label = 'cu118'; IndexUrl = 'https://download.pytorch.org/whl/cu118'; Torch = '2.5.1'; TorchVision = '0.20.1' }
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
        Torch = '2.5.1'
        TorchVision = '0.20.1'
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

    & $PythonPath -m pip install --upgrade --force-reinstall --no-deps --index-url $target.IndexUrl ("torch==" + $target.Torch) ("torchvision==" + $target.TorchVision)
    if ($LASTEXITCODE -ne 0) {
        throw ("PyTorch install failed for target {0} with exit code {1}" -f $target.Label, $LASTEXITCODE)
    }

    & $PythonPath -m pip install --upgrade --force-reinstall --no-deps "numpy==1.26.4" "opencv-python==4.8.1.78" "Pillow==10.1.0" "pydicom==3.0.1" "ultralytics==8.4.6"
    if ($LASTEXITCODE -ne 0) {
        throw ("Failed to restore compatible vision package versions with exit code {0}" -f $LASTEXITCODE)
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

    & $PythonExe -c "import cv2, flask, pydicom, ultralytics, torch; print('OK:', cv2.__version__, pydicom.__version__)"
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
'@
Set-Content -Path (Join-Path $PackageDir "setup_backend.ps1") -Value $setupScript -Encoding ASCII

$startCpuScript = @'
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
'@
Set-Content -Path (Join-Path $PackageDir "start_server.ps1") -Value $startCpuScript -Encoding ASCII

$startGpuScript = @'
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$env:PANO_DEVICE = "gpu"
& (Join-Path $PSScriptRoot "start_server.ps1")
'@
Set-Content -Path (Join-Path $PackageDir "start_server_gpu.ps1") -Value $startGpuScript -Encoding ASCII

$installBat = @'
@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0setup_backend.ps1"
if errorlevel 1 (
  echo.
  echo Install failed.
  pause
  exit /b 1
)
echo.
echo Install completed.
pause
'@
Set-Content -Path (Join-Path $PackageDir "1-install.bat") -Value $installBat -Encoding ASCII

$runBat = @'
@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start_server.ps1"
echo.
echo Server stopped.
pause
'@
Set-Content -Path (Join-Path $PackageDir "2-run.bat") -Value $runBat -Encoding ASCII

$runGpuBat = @'
@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start_server_gpu.ps1"
echo.
echo Server stopped.
pause
'@
Set-Content -Path (Join-Path $PackageDir "3-run-gpu.bat") -Value $runGpuBat -Encoding ASCII

$installAndRunBat = @'
@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0setup_backend.ps1"
if errorlevel 1 (
  echo.
  echo Install failed.
  pause
  exit /b 1
)
echo.
echo Starting server...
powershell -ExecutionPolicy Bypass -File "%~dp0start_server.ps1"
echo.
echo Server stopped.
pause
'@
Set-Content -Path (Join-Path $PackageDir "0-install-and-run.bat") -Value $installAndRunBat -Encoding ASCII

$releaseReadme = @"
Release package usage

Version: $ReleaseVersion

For non-technical users

1. Double-click 0-install-and-run.bat
2. Wait until the install finishes and the server starts
3. Open http://localhost:5000

Manual mode

1. Run .\setup_backend.ps1
2. Edit gpts\.env if you need GEMINI_API_KEY or GPU mode
3. Run .\start_server.ps1
4. Open http://localhost:5000

GPU mode

- Run .\start_server_gpu.ps1 or double-click 3-run-gpu.bat
- If CUDA is not available, the server falls back to CPU
"@
Set-Content -Path (Join-Path $PackageDir "README_RELEASE.txt") -Value $releaseReadme -Encoding ASCII

$installGuideKr = @'
치과 AI 프로그램 설치 방법

이 폴더에는 단일 exe 설치파일 대신 더블클릭해서 실행하는 설치 파일이 들어 있습니다.

설치 파일

- 0-install-and-run.bat : 설치와 실행을 한 번에 진행
- 1-install.bat : 설치만 진행
- 2-run.bat : 설치 후 CPU 모드로 실행
- 3-run-gpu.bat : 설치 후 GPU 모드로 실행

가장 쉬운 방법

1. 0-install-and-run.bat 를 더블클릭합니다.
2. 검은 창이 열리면 닫지 말고 기다립니다.
3. 설치가 끝나면 서버가 시작됩니다.
4. 인터넷 브라우저에서 http://localhost:5000 을 엽니다.

처음 설치할 때 필요한 것

- 인터넷 연결
- Windows에서 Python 설치 허용
- 보안 경고가 나오면 실행 허용

주의 사항

- 처음 설치는 시간이 걸릴 수 있습니다.
- 설치 중에는 창을 닫으면 안 됩니다.
- Gemini 기능을 쓰려면 gpts\.env 에 API 키를 넣어야 합니다.
- GPU 실행이 안 되면 2-run.bat 로 CPU 실행을 사용하면 됩니다.

문제가 있을 때

- Python 관련 오류가 나면 1-install.bat 를 다시 실행합니다.
- 실행만 다시 하려면 2-run.bat 를 사용합니다.
- GPU가 안 잡히면 3-run-gpu.bat 대신 2-run.bat 를 사용합니다.
'@
$installGuideKr = @'
치과 AI 프로그램 설치 방법

이 폴더에는 별도 exe 설치 파일 대신 바로 실행할 수 있는 설치 스크립트가 들어 있다.

설치 파일

- 0-install-and-run.bat : 설치와 실행을 한 번에 진행
- 1-install.bat : 설치만 진행
- 2-run.bat : 설치 후 CPU 모드로 실행
- 3-run-gpu.bat : 설치 후 GPU 모드로 실행

가장 쉬운 방법

1. 0-install-and-run.bat 를 더블클릭한다.
2. 검은 창이 열리면 닫지 말고 기다린다.
3. 설치가 끝나면 서버가 시작된다.
4. 인터넷 브라우저에서 http://localhost:5000 으로 접속한다.

처음 설치할 때 필요한 것
- 인터넷 연결
- Windows에서 Python 설치 허용
- 보안 경고가 나오면 실행 허용

주의 사항

- 처음 설치에는 시간이 조금 걸릴 수 있다.
- 설치 중에는 창을 닫으면 안 된다.
- Gemini 기능을 쓰려면 gpts\.env 에 API 키를 넣어야 한다.
- GPU 실행이 안 되면 2-run.bat 으로 CPU 실행을 사용하면 된다.

문제가 있을 때
- Python 관련 오류가 나면 1-install.bat 를 다시 실행한다.
- 실행만 다시 하려면 2-run.bat 를 사용한다.
- GPU가 제대로 안 되면 3-run-gpu.bat 대신 2-run.bat 를 사용한다.
'@
$installGuideKr | Out-File -FilePath (Join-Path $PackageDir "install-guide-ko.txt") -Encoding utf8

if ($Zip) {
    Write-Host "Creating zip archive..."
    Compress-Archive -Path (Join-Path $PackageDir "*") -DestinationPath $ZipPath
}

Write-Host ""
Write-Host "Release package created:"
Write-Host $PackageDir
if ($Zip) {
    Write-Host $ZipPath
}
