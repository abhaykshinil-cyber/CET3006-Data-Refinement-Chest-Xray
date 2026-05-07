"""
STEP 1 — GPU Diagnosis Script
Run this first to understand what environment you have.
Usage: python step1_diagnose.py
"""
import sys
import subprocess

print("=" * 65)
print("  GPU & PyTorch Diagnosis")
print("=" * 65)

# 1. Python info
print(f"\n[1] Python executable : {sys.executable}")
print(f"    Python version     : {sys.version.split()[0]}")

# 2. nvidia-smi
print("\n[2] NVIDIA System Info (nvidia-smi):")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        for line in result.stdout.splitlines()[:15]:
            print("    " + line)
    else:
        print("    ERROR:", result.stderr.strip())
except FileNotFoundError:
    print("    nvidia-smi NOT FOUND — driver may not be installed or not in PATH")
except Exception as e:
    print(f"    ERROR: {e}")

# 3. PyTorch check
print("\n[3] PyTorch:")
try:
    import torch
    print(f"    torch version          : {torch.__version__}")
    print(f"    torch.version.cuda     : {torch.version.cuda}")
    print(f"    torch.cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU name               : {torch.cuda.get_device_name(0)}")
        print(f"    GPU count              : {torch.cuda.device_count()}")
        print(f"\n  STATUS: ✅ GPU is READY — no fix needed!")
    else:
        print(f"\n  STATUS: ❌ GPU NOT available")
        if torch.version.cuda is None:
            print("  CAUSE:  PyTorch was installed WITHOUT CUDA support (CPU-only build)")
            print("  FIX:    Run step2_fix_pytorch.bat")
        else:
            print("  CAUSE:  CUDA build present but GPU not detected")
            print("  FIX:    Check that your NVIDIA driver is installed (nvidia-smi above)")
except ImportError:
    print("    torch NOT installed")
    print("  FIX: Run step2_fix_pytorch.bat")

# 4. pip list for torch packages
print("\n[4] Installed torch packages (pip list):")
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True, text=True, timeout=30
    )
    for line in result.stdout.splitlines():
        if "torch" in line.lower():
            print("    " + line)
except Exception as e:
    print(f"    ERROR: {e}")

print("\n" + "=" * 65)
print("  Done. If GPU is NOT available, run:  step2_fix_pytorch.bat")
print("=" * 65)
