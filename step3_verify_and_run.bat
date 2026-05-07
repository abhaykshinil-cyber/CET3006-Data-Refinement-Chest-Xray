@echo off
REM ============================================================
REM STEP 3 — Verify GPU is working, then launch the full pipeline
REM Uses: C:\Python314\python.exe (confirmed from diagnosis)
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   Step 3: GPU Verification + Pipeline Launch
echo ============================================================

echo.
echo [1] Full GPU verification:
C:\Python314\python.exe -c "import torch; ok=torch.cuda.is_available(); print(); print('  torch version  :', torch.__version__); print('  CUDA version   :', torch.version.cuda); print('  GPU available  :', ok); print('  GPU name       :', torch.cuda.get_device_name(0) if ok else 'NONE'); print('  GPU memory GB  :', round(torch.cuda.get_device_properties(0).total_memory/1e9,2) if ok else 'N/A'); t=torch.randn(1000,1000).cuda() if ok else None; print('  Tensor on GPU  :', str(t.device) if t is not None else 'FAILED -- aborting'); print(); import sys; sys.exit(0 if ok else 1)"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   GPU is NOT available. Run step2_fix_pytorch.bat first.
    pause
    exit /b 1
)

echo.
echo   GPU confirmed. Launching 10-stage pipeline...
echo ============================================================
echo.
C:\Python314\python.exe run_pipeline.py

echo.
echo ============================================================
echo   Pipeline complete.
echo   Results  : results\artifacts\metrics\final_metrics.json
echo   Plots    : plots\
echo   Histories: results\artifacts\histories\
echo ============================================================
pause
