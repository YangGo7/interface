@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start_server.ps1"
echo.
echo Server stopped.
pause
