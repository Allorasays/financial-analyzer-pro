@echo off
echo ==================================================
echo    Financial Analyzer Pro - Windows Startup
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Financial Analyzer Pro...
echo.

REM Start the startup script
python start_app.py

pause

