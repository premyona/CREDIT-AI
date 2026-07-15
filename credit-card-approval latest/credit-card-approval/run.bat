@echo off
echo ============================================================
echo   Credit Card Approval Prediction - Starting Server
echo ============================================================
echo.

REM Use Python 3.13 (Microsoft Store version with all packages)
SET PYTHON=C:\Users\mohan\AppData\Local\Microsoft\WindowsApps\python3.13.exe

REM Check if models exist, train if not
IF NOT EXIST "models\best_model.pkl" (
    echo [1/2] Models not found - running training pipeline...
    echo       Generating dataset...
    %PYTHON% data\generate_dataset.py
    echo       Training 4 ML models (this takes ~2 minutes)...
    %PYTHON% model_training.py
    echo.
)

echo [OK] Models ready!
echo.
echo Starting Flask application...
echo.
echo Host address: http://0.0.0.0:5000
echo Local browser: http://127.0.0.1:5000
set "LOCALIP="
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr "IPv4 Address" ^| findstr /V "127.0.0.1"') do @set "LOCALIP=%%A"
if defined LOCALIP (
    setlocal enabledelayedexpansion
    set "LOCALIP=!LOCALIP: =!"
    echo Or access from another device at: http://!LOCALIP!:5000
    endlocal
)
echo Press CTRL+C to stop the server.
echo.
%PYTHON% app.py
