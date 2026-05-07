"""
run_pipeline.py
===============
One-click launcher for the Chest X-Ray Data-Refinement Research pipeline.

Usage
-----
    python run_pipeline.py

Requirements (install once)
----------------------------
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install scikit-learn matplotlib pillow numpy

    # If you do NOT have a CUDA GPU, use the CPU-only build instead:
    pip install torch torchvision

GPU check
---------
    The pipeline auto-detects CUDA.  If a GPU is present it will be used;
    otherwise the code falls back to CPU automatically.
"""

import sys
import pathlib

# Add the project sub-package to sys.path so its imports resolve correctly.
PROJECT_DIR = pathlib.Path(__file__).parent / "chest_xray_project"
sys.path.insert(0, str(PROJECT_DIR))

if __name__ == "__main__":
    # Late import so sys.path is patched first
    from main import main          # noqa: E402
    main()
