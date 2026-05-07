@echo off
REM ============================================================
REM STEP 2 (PyCharm venv variant)
REM Use this if your project runs inside a PyCharm virtual env
REM ============================================================

echo.
echo ============================================================
echo   PyTorch CUDA Fix (PyCharm venv) for RTX 3090
echo ============================================================

REM Try to find PyCharm venv in common locations
set VENV_PY=
if exist "%USERPROFILE%\.virtualenvs" (
    echo Found virtualenvs folder. Searching...
    dir "%USERPROFILE%\.virtualenvs" /b
)

REM Check if there's a venv in this project folder
if exist "venv\Scripts\python.exe" (
    echo Found venv in project folder.
    set VENV_PY=venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    echo Found .venv in project folder.
    set VENV_PY=.venv\Scripts\python.exe
)

if defined VENV_PY (
    echo Using: %VENV_PY%
    echo.
    echo Removing CPU-only PyTorch...
    "%VENV_PY%" -m pip uninstall torch torchvision torchaudio -y
    echo.
    echo Installing CUDA 12.1 PyTorch...
    "%VENV_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo Verifying...
    "%VENV_PY%" -c "import torch; print('GPU:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
) else (
    echo No local venv found.
    echo Please activate your environment manually, then run:
    echo.
    echo   pip uninstall torch torchvision torchaudio -y
    echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo.
pause
