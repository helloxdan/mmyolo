@echo off
REM YOLOv8 Cat Dataset GPU Training Script

echo ========================================
echo YOLOv8 Cat Dataset Training
echo ========================================
echo.

REM Check Python environment
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check GPU
echo Checking GPU...
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
if errorlevel 1 (
    echo Warning: PyTorch is not installed, please install dependencies first
    pause
    exit /b 1
)

echo.
echo Starting training...
echo.

REM Execute training
python train_cat.py

echo.
echo Training script completed
pause
