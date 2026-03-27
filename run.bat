@echo off
title Sigma-Captioner Runner
setlocal

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [!] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b
)

echo [🚀] Activating Environment...
call venv\Scripts\activate

echo [🧠] Starting Sigma-Captioner (main.py)...
echo.

:: Runs the main script. 
:: 'python' now refers to the venv version.
python main.py

:: Keeps the window open if the script crashes or finishes
if %errorlevel% neq 0 (
    echo.
    echo [!] Script exited with an error (Code: %errorlevel%)
)
echo.
echo Press any key to exit...
pause