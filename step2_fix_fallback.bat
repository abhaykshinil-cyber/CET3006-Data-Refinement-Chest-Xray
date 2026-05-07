@echo off
REM ============================================================
REM STEP 2 FALLBACK — Try cu124, then cu121, then nightly
REM Use this ONLY if step2_fix_pytorch.bat failed
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   PyTorch CUDA Fix FALLBACK - Python 3.14 / RTX 3050
echo ============================================================
echo.

echo Removing any partial install...
C:\Python314\python.exe -m pip uninstall torch torchvision torchaudio -y 2>nul

echo.
echo Trying CUDA 12.4 build...
C:\Python314\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo Verifying cu124 install...
C:\Python314\python.exe -c "import torch; print('GPU:', torch.cuda.is_available())"

C:\Python314\python.exe -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS with cu124 build.
    goto :verify
)

echo.
echo cu124 failed. Trying PyTorch nightly with CUDA 12.8...
C:\Python314\python.exe -m pip uninstall torch torchvision torchaudio -y 2>nul
C:\Python314\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

:verify
echo.
echo ============================================================
echo   Final verification:
C:\Python314\python.exe -c "import torch; ok=torch.cuda.is_available(); print('  torch :', torch.__version__); print('  CUDA  :', torch.version.cuda); print('  GPU   :', torch.cuda.get_device_name(0) if ok else 'NONE'); print('  READY :', ok)"
echo ============================================================
pause
