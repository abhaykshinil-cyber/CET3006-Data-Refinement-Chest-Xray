@echo off
REM ============================================================
REM STEP 2 — Fix PyTorch: Install CUDA-enabled build
REM Tailored for: Python 3.14, RTX 3050, Driver CUDA 13.0
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   PyTorch CUDA Fix - RTX 3050 / CUDA 13.0 / Python 3.14
echo ============================================================
echo   Python : C:\Python314\python.exe
echo   GPU    : NVIDIA GeForce RTX 3050 ^(6GB^)
echo   Driver : CUDA 13.0 ^(backwards-compatible with cu124/cu126^)
echo ============================================================
echo.

echo [1] Removing CPU-only PyTorch ^(torch 2.11.0+cpu^)...
C:\Python314\python.exe -m pip uninstall torch torchvision torchaudio -y
echo     Done.
echo.

echo [2] Installing PyTorch with CUDA 12.6 support...
echo     ^(Downloading ~2.5 GB -- please wait, do NOT close this window^)
echo.
C:\Python314\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo.
echo [3] Verifying installation...
C:\Python314\python.exe -c "import torch; ok=torch.cuda.is_available(); print(); print('  torch version :', torch.__version__); print('  CUDA version  :', torch.version.cuda); print('  GPU available :', ok); print('  GPU name      :', torch.cuda.get_device_name(0) if ok else 'NOT DETECTED'); print('  GPU memory GB :', round(torch.cuda.get_device_properties(0).total_memory/1e9,2) if ok else 'N/A'); print(); print('  >>> STATUS: GPU READY - run step3_verify_and_run.bat' if ok else '  >>> STATUS: FAILED - run step2_fix_fallback.bat')"

echo.
echo ============================================================
echo   If the above shows GPU READY, run: step3_verify_and_run.bat
echo   If it failed, run: step2_fix_fallback.bat
echo ============================================================
pause
