param(
    [string]$OutputRoot = "release",
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

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $RepoRoot "frontend"
$BackendDir = Join-Path $RepoRoot "gpts"
$OutputRootFull = Get-FullPathSafe -PathValue $OutputRoot
$PackageName = "interface-deploy"
$PackageDir = Join-Path $OutputRootFull $PackageName
$ZipPath = Join-Path $OutputRootFull "$PackageName.zip"

Write-Host "Repo Root: $RepoRoot"
Write-Host "Package Dir: $PackageDir"

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
    Copy-Item -Path (Join-Path $BackendDir $file) -Destination (Join-Path $PackageDir "gpts") -Force
}

foreach ($dir in @("data", "reports", "temp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path (Join-Path $PackageDir "gpts") $dir) | Out-Null
}

Copy-Item -Path (Join-Path $RepoRoot "docs\BUILD_OTHER_PC.md") -Destination (Join-Path $PackageDir "BUILD_OTHER_PC.md") -Force

$releaseEnv = @"
FLASK_ENV=production
SECRET_KEY=change-me
PANO_DEVICE=cpu
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
'@
Set-Content -Path (Join-Path $PackageDir "setup_backend.ps1") -Value $setupScript -Encoding ASCII

$startCpuScript = @'
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

$releaseReadme = @'
Release package usage

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
'@
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
