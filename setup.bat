@echo off
title Sigma-Captioner Environment Setup
setlocal enabledelayedexpansion

:: Windows Confirmation
echo ========================================================
echo   SIGMA-CAPTIONER: WINDOWS ENVIRONMENT SETUP
echo ========================================================
echo This script will create a Python Virtual Environment (venv)
echo and install dependencies for Windows systems.
echo.
pause
echo.

:: Create Virtual Environment
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Venv already exists. Skipping creation.
)

:: Activate Environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch 2.7.1
echo [2/4] Installing PyTorch 2.7.1 (CUDA 12.8)...
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

:: Manual Wheel Instruction
echo.
echo --------------------------------------------------------
echo ATTENTION: ACTION REQUIRED
echo Please visit: https://github.com/wildminder/AI-windows-whl
echo Download and install any specific wheels (like Triton) 
echo required for your specific hardware/GPU.
echo --------------------------------------------------------
echo.
pause

:: Force Install requirements.txt
if exist "requirements.txt" (
    echo [3/4] Force installing requirements.txt...
    :: --force-reinstall ensures it overwrites existing versions
    :: --no-cache-dir ensures it fetches fresh copies if needed
    pip install --force-reinstall --no-cache-dir -r requirements.txt
) else (
    echo [!] requirements.txt not found. Skipping.
)

echo.
echo [4/4] Setup Complete! 
echo To start your environment in the future, run: venv\Scripts\activate
pause