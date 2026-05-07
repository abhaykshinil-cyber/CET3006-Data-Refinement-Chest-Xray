"""
verify_env.py
=============
Pre-flight check before running the full pipeline.
Confirms: Python version, required packages, GPU, and dataset structure.

Usage:   python verify_env.py
"""

import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
PROJECT_DIR = ROOT / "chest_xray_project"
DATASET_ROOT = ROOT / "chest_xray"

sys.path.insert(0, str(PROJECT_DIR))

REQUIRED_PACKAGES = [
    "torch", "torchvision", "sklearn", "matplotlib",
    "numpy", "PIL",
]

print("=" * 60)
print("  Environment Pre-flight Check")
print("=" * 60)

# 1. Python
print(f"\n[1] Python : {sys.version.split()[0]}")
assert sys.version_info >= (3, 8), "Python 3.8+ required"

# 2. Packages
print("\n[2] Package availability")
missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
        print(f"    {'OK':>4}  {pkg}")
    except ImportError:
        print(f"    MISS  {pkg}  <-- INSTALL NEEDED")
        missing.append(pkg)

if missing:
    print(f"\n  Install missing packages:")
    print("  pip install torch torchvision scikit-learn matplotlib pillow numpy")
    sys.exit(1)

# 3. GPU
import torch
print("\n[3] GPU / CUDA")
print(f"    torch version           : {torch.__version__}")
print(f"    torch.cuda.is_available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU name                : {torch.cuda.get_device_name(0)}")
    print(f"    GPU count               : {torch.cuda.device_count()}")
    device = torch.device("cuda")
else:
    print("    No CUDA GPU found — will run on CPU (slower, but correct)")
    device = torch.device("cpu")

# 4. Dataset
print("\n[4] Dataset structure")
splits = ("train", "val", "test")
classes = ("NORMAL", "PNEUMONIA")
total = 0
for split in splits:
    for cls in classes:
        d = DATASET_ROOT / split / cls
        n = len(list(d.glob("*.*"))) if d.is_dir() else 0
        flag = "" if n > 0 else "  <-- MISSING"
        print(f"    {split:<6}/{cls:<10}: {n:>5} images{flag}")
        total += n
print(f"    {'TOTAL':<18}: {total:>5} images")

# 5. Project files
print("\n[5] Project files")
required_files = [
    "config.py", "data_loader.py", "detection.py",
    "evaluation.py", "main.py", "model.py",
    "train.py", "uncertainty.py",
]
for fname in required_files:
    p = PROJECT_DIR / fname
    status = "OK" if p.exists() else "MISSING"
    print(f"    {status:>7}  {fname}")

print("\n" + "=" * 60)
print("  Pre-flight complete.  Run: python run_pipeline.py")
print("=" * 60)
