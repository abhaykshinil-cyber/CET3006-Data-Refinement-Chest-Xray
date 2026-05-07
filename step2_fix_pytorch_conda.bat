@echo off
REM ============================================================
REM STEP 2 (CONDA variant) — Fix PyTorch in a conda environment
REM Use this IF your project runs inside a conda environment
REM ============================================================

echo.
echo ============================================================
echo   PyTorch CUDA Fix (Conda) for NVIDIA RTX 3090
echo ============================================================
echo.

REM Show current conda env
conda info --envs
echo.
echo Active env: %CONDA_DEFAULT_ENV%
echo.

REM Remove CPU-only torch
echo [1] Removing old CPU-only PyTorch...
conda remove pytorch torchvision torchaudio cpuonly -y 2>nul
pip uninstall torch torchvision torchaudio -y 2>nul

REM Install CUDA-enabled via pip (most reliable for exact CUDA version)
echo.
echo [2] Installing PyTorch 2.x + CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3] Verifying...
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
pause
