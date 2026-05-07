"""
config.py — Central configuration for the Chest X-Ray project.
All paths, hyperparameters, and thresholds are declared here.
"""

import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = pathlib.Path(r"C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH")
DATASET_ROOT = BASE_DIR / "chest_xray"

CKPT_CLEAN   = BASE_DIR / "best_model_clean.pth"     # trained on original data
CKPT_NOISY   = BASE_DIR / "best_model.pth"            # trained on noisy data
CKPT_CLEANED = BASE_DIR / "best_model_cleaned.pth"    # retrained after cleaning

PLOTS_DIR    = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
LABEL_MAP        = {"NORMAL": 0, "PNEUMONIA": 1}
CLASS_NAMES      = ["NORMAL", "PNEUMONIA"]
IMG_SIZE         = (224, 224)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

# ── DataLoader ────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
NUM_WORKERS  = 0        # increase for multi-core loading

# ── Model ─────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 2
DROPOUT_P    = 0.5

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 15
EARLY_STOP_PAT = 5
EARLY_STOP_DELTA = 1e-5

# ── Noise injection ───────────────────────────────────────────────────────────
NOISE_RATE = 0.20

# ── MC Dropout ────────────────────────────────────────────────────────────────
MC_PASSES = 30
EPS       = 1e-8        # numerical stability for entropy

# ── Error detection ───────────────────────────────────────────────────────────
UNCERTAINTY_THRESH = 0.02
CONFIDENCE_THRESH  = 0.70

# ── Threshold-based cleaning experiments ─────────────────────────────────────
# Each value is the fraction of the FLAGGED pool to remove (sorted by
# uncertainty descending).  0.10 = remove only the top-10% most uncertain
# flagged samples; 1.00 = remove every flagged sample (original behaviour).
CLEANING_LEVELS = [0.10, 0.20, 0.30]

# ── Research improvements (Phase 6) ──────────────────────────────────────────
BACKBONE      = "resnet18"   # "resnet18" | "efficientnet_b0"
WEIGHT_FLOOR  = 0.2          # minimum loss weight assigned to most uncertain samples
ENTROPY_ALPHA = 0.5          # combined score: (1-alpha)*norm_var + alpha*norm_entropy
