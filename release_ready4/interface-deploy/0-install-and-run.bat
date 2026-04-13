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
