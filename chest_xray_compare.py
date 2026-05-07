"""
Model Comparison — Chest X-Ray Pneumonia Classifier
====================================================
Evaluates and compares three model variants on the same held-out test set:

  Model 1 — CLEAN     : trained on original data  (no noise injected)
  Model 2 — NOISY     : trained on 20% label-noisy data
  Model 3 — CLEANED   : retrained after MC Dropout noise filtering

For each model the following are computed on the test set:
  • Accuracy
  • Precision  (macro)
  • Recall     (macro)
  • F1-score   (macro)
  + per-class breakdown

If a checkpoint file is missing, that model is trained from scratch
automatically before evaluation.

Outputs
-------
  • Formatted side-by-side comparison table
  • Per-class detail table
  • Delta columns showing improvement of CLEANED over NOISY
"""

import pathlib
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ── Our modules ───────────────────────────────────────────────────────────────
from chest_xray_pipeline import (
    DATASET_ROOT,
    BATCH_SIZE,
    collect_all_samples,
    stratified_split,
    build_dataloaders,
)
from chest_xray_model import build_model, ChestXRayClassifier
from chest_xray_noise import build_noisy_dataloaders, NOISE_RATE
from chest_xray_train import (
    train_one_epoch,
    validate,
    EpochLogger,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOP_PAT,
)
from chest_xray_cleaning import run_dataset_cleaning
from chest_xray_error_detection import MC_PASSES

# ─────────────────────────────────────────────
# Checkpoint paths for each model
# ─────────────────────────────────────────────
BASE_DIR = pathlib.Path(
    r"C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH"
)
CKPT = {
    "CLEAN"  : BASE_DIR / "best_model_clean.pth",    # trained on original data
    "NOISY"  : BASE_DIR / "best_model.pth",           # trained on noisy data
    "CLEANED": BASE_DIR / "best_model_cleaned.pth",   # retrained after cleaning
}

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# 1.  Shared early-stopping used during any on-the-fly training
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int, save_path: pathlib.Path, delta: float = 1e-5):
        self.patience    = patience
        self.save_path   = save_path
        self.delta       = delta
        self.best_loss   = float("inf")
        self.best_state  = None
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_state, self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────
# 2.  Training helpers
# ─────────────────────────────────────────────
def _run_training_loop(
    model:         nn.Module,
    train_loader,
    val_loader,
    device:        torch.device,
    save_path:     pathlib.Path,
    label:         str,
    num_epochs:    int   = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> EpochLogger:
    """Generic training loop shared by all three model variants."""

    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(EARLY_STOP_PAT, save_path)
    logger     = EpochLogger()

    print(f"\n  Training {label} model  (max {num_epochs} epochs) …")
    logger.print_header()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_acc = validate(model, val_loader, device)
        logger.log(epoch, tr_loss, tr_acc, vl_loss, vl_acc, time.time() - t0)

        early_stop.step(vl_loss, model)
        if early_stop.should_stop:
            print(f"\n  ⚡ Early stopping at epoch {epoch}.")
            break

    logger.print_footer()
    early_stop.restore_best(model)
    model.eval()
    return logger


def _load_weights(model: nn.Module, path: pathlib.Path, device: torch.device):
    """Load saved checkpoint into model."""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"    ✓ Loaded  {path.name}")


# ─────────────────────────────────────────────
# 3.  Obtain each of the three models
# ─────────────────────────────────────────────
def obtain_clean_model(
    train_loader, val_loader, device: torch.device
) -> nn.Module:
    """
    Model 1 — trained on the ORIGINAL clean dataset (no noise).
    Loads from checkpoint if available; otherwise trains from scratch.
    """
    model, _ = build_model(pretrained=True)
    model.to(device)

    if CKPT["CLEAN"].exists():
        _load_weights(model, CKPT["CLEAN"], device)
    else:
        print("  [CLEAN] Checkpoint not found — training now …")
        _run_training_loop(
            model, train_loader, val_loader, device,
            save_path=CKPT["CLEAN"], label="CLEAN",
        )
    return model


def obtain_noisy_model(
    train_paths, train_labels,
    val_paths,   val_labels,
    test_paths,  test_labels,
    device: torch.device,
) -> nn.Module:
    """
    Model 2 — trained on the 20% LABEL-NOISY dataset.
    Loads from checkpoint if available; otherwise trains from scratch.
    """
    model, _ = build_model(pretrained=True)
    model.to(device)

    if CKPT["NOISY"].exists():
        _load_weights(model, CKPT["NOISY"], device)
    else:
        print("  [NOISY] Checkpoint not found — training now …")
        noisy_loader, val_loader, _, _, _ = build_noisy_dataloaders(
            train_paths, train_labels,
            val_paths,   val_labels,
            test_paths,  test_labels,
            batch_size=BATCH_SIZE,
        )
        _run_training_loop(
            model, noisy_loader, val_loader, device,
            save_path=CKPT["NOISY"], label="NOISY",
        )
    return model


def obtain_cleaned_model(device: torch.device):
    """
    Model 3 — retrained on the MC-Dropout CLEANED dataset.
    Loads from checkpoint if available; otherwise runs full cleaning + retraining.

    Returns
    -------
    model       : trained model
    test_loader : the test DataLoader (same split)
    """
    model, _ = build_model(pretrained=True)
    model.to(device)

    if CKPT["CLEANED"].exists():
        _load_weights(model, CKPT["CLEANED"], device)
        # Still need test_loader → rebuild splits
        all_paths, all_labels = collect_all_samples(DATASET_ROOT)
        (tr_p, tr_l, vl_p, vl_l, ts_p, ts_l) = stratified_split(all_paths, all_labels)
        _, _, test_loader = build_dataloaders(
            tr_p, tr_l, vl_p, vl_l, ts_p, ts_l, batch_size=BATCH_SIZE
        )
    else:
        print("  [CLEANED] Checkpoint not found — running cleaning + retraining …")
        clean_loader, val_loader, test_loader, _ = run_dataset_cleaning(
            noise_rate=NOISE_RATE, n_passes=MC_PASSES, batch_size=BATCH_SIZE
        )
        _run_training_loop(
            model, clean_loader, val_loader, device,
            save_path=CKPT["CLEANED"], label="CLEANED",
        )

    return model, test_loader


# ─────────────────────────────────────────────
# 4.  Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model:  nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, true_labels) arrays for the full loader."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        preds  = model(images).argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


# ─────────────────────────────────────────────
# 5.  Metrics
# ─────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return dict(
        accuracy  = accuracy_score (y_true, y_pred),
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall    = recall_score   (y_true, y_pred, average="macro", zero_division=0),
        f1        = f1_score       (y_true, y_pred, average="macro", zero_division=0),
        # per-class
        precision_per = precision_score(y_true, y_pred, average=None, zero_division=0),
        recall_per    = recall_score   (y_true, y_pred, average=None, zero_division=0),
        f1_per        = f1_score       (y_true, y_pred, average=None, zero_division=0),
    )


# ─────────────────────────────────────────────
# 6.  Printing
# ─────────────────────────────────────────────
def _bar(value: float, width: int = 20) -> str:
    """Simple ASCII progress bar for metric visualisation."""
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def print_comparison_table(results: dict[str, dict]):
    """
    Render a side-by-side comparison table for all three models.

    Parameters
    ----------
    results : { model_name → metrics_dict }
    """
    models  = list(results.keys())          # ["CLEAN", "NOISY", "CLEANED"]
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = {
        "accuracy" : "Accuracy ",
        "precision": "Precision",
        "recall"   : "Recall   ",
        "f1"       : "F1-Score ",
    }

    COL = 13    # column width per model

    # ── Header ────────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print("║        Three-Model Comparison — Test Set Metrics              ║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Metric':<12}" +
          "".join(f"  {m:>{COL}}" for m in models) +
          f"  {'Δ (Clean→Cleaned)':>17}  ║")
    print("╠" + "─" * 68 + "╣")

    improvements = {}
    for metric in metrics:
        vals = {m: results[m][metric] * 100 for m in models}

        # Δ = CLEANED − CLEAN  (how much the cleaning recovery gained back)
        delta = vals.get("CLEANED", 0) - vals.get("CLEAN", 0)
        improvements[metric] = delta

        delta_str  = f"{delta:+.2f}pp"
        best_model = max(vals, key=vals.get)

        row = f"║  {labels[metric]:<12}"
        for m in models:
            v     = vals[m]
            star  = "★" if m == best_model else " "
            row  += f"  {v:>10.2f}%{star}"
        row += f"  {delta_str:>17}  ║"
        print(row)

    print("╠" + "═" * 68 + "╣")
    print("║  ★ = best model for that metric                               ║")
    print("║  Δ = CLEANED minus CLEAN  (positive = closer to clean)        ║")
    print("╚" + "═" * 68 + "╝")

    # ── Visual bar chart ──────────────────────────────────────────────────
    BAR_W = 30
    print()
    print("  Visual comparison  (each bar = % score)")
    print("  " + "─" * 62)
    for metric in metrics:
        print(f"\n  {labels[metric]}")
        for m in models:
            v   = results[m][metric]
            bar = _bar(v, BAR_W)
            print(f"    {m:<8} │{bar}│ {v*100:>6.2f}%")

    # ── Per-class breakdown ───────────────────────────────────────────────
    print()
    print("  Per-Class Breakdown  (macro rows above; per-class below)")
    print()

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        print(f"  Class: {cls_name}")
        print(f"  {'Metric':<12}" +
              "".join(f"  {m:>{COL}}" for m in models))
        print("  " + "─" * (12 + (COL + 2) * len(models)))
        for metric, key in [
            ("Precision", "precision_per"),
            ("Recall",    "recall_per"),
            ("F1",        "f1_per"),
        ]:
            row = f"  {metric:<12}"
            vals_cls = {m: results[m][key][cls_idx] * 100 for m in models}
            best_m   = max(vals_cls, key=vals_cls.get)
            for m in models:
                star  = "★" if m == best_m else " "
                row  += f"  {vals_cls[m]:>10.2f}%{star}"
            print(row)
        print()

    # ── sklearn detailed reports ──────────────────────────────────────────
    return improvements


def print_sklearn_reports(results_raw: dict[str, tuple]):
    """
    Print sklearn classification_report for each model.

    Parameters
    ----------
    results_raw : { model_name → (y_true, y_pred) }
    """
    for model_name, (y_true, y_pred) in results_raw.items():
        print(f"\n  ── {model_name} — Full Classification Report ──")
        print("  " + "─" * 54)
        report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
        for line in report.splitlines():
            print(f"  {line}")


# ─────────────────────────────────────────────
# 7.  Master pipeline
# ─────────────────────────────────────────────
def run_comparison():
    """
    Full three-model comparison pipeline.

    Steps
    -----
    1.  Prepare the shared test DataLoader (same split for all models)
    2.  Obtain / train each model
    3.  Run inference on the shared test set
    4.  Compute metrics for each model
    5.  Print comparison table + visual bars + per-class breakdown
    """
    print("=" * 70)
    print("  Three-Model Comparison — Chest X-Ray Pneumonia Classifier")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ── Shared data splits ────────────────────────────────────────────────
    print("\n[1/5] Preparing shared dataset splits …")
    all_paths, all_labels = collect_all_samples(DATASET_ROOT)
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(all_paths, all_labels)

    clean_train_loader, val_loader, test_loader = build_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        batch_size=BATCH_SIZE,
    )
    print(f"  Shared test set : {len(test_paths):,} images  "
          f"(same for all three models)")

    # ── Obtain models ─────────────────────────────────────────────────────
    print("\n[2/5] Obtaining models …")

    print("\n  → Model 1 : CLEAN  (no noise, original data)")
    model_clean = obtain_clean_model(clean_train_loader, val_loader, device)

    print("\n  → Model 2 : NOISY  (20% label noise injected)")
    model_noisy = obtain_noisy_model(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        device,
    )

    print("\n  → Model 3 : CLEANED  (retrained after MC Dropout filtering)")
    model_cleaned, test_loader_cleaned = obtain_cleaned_model(device)
    # Use the same test_loader (same seed → same test images)

    # ── Inference ─────────────────────────────────────────────────────────
    print("\n[3/5] Running inference on test set for each model …")
    preds_clean,   labels_true = run_inference(model_clean,   test_loader, device)
    preds_noisy,   _           = run_inference(model_noisy,   test_loader, device)
    preds_cleaned, _           = run_inference(model_cleaned, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────
    print("\n[4/5] Computing metrics …")
    results = {
        "CLEAN"  : compute_metrics(labels_true, preds_clean),
        "NOISY"  : compute_metrics(labels_true, preds_noisy),
        "CLEANED": compute_metrics(labels_true, preds_cleaned),
    }
    results_raw = {
        "CLEAN"  : (labels_true, preds_clean),
        "NOISY"  : (labels_true, preds_noisy),
        "CLEANED": (labels_true, preds_cleaned),
    }

    # ── Print comparison ──────────────────────────────────────────────────
    print("\n[5/5] Results")
    improvements = print_comparison_table(results)
    print_sklearn_reports(results_raw)

    # ── Final summary sentence ────────────────────────────────────────────
    print()
    print("  Summary")
    print("  " + "─" * 54)
    for metric, delta in improvements.items():
        direction = "recovered" if delta >= 0 else "degraded"
        print(f"  {metric.capitalize():<10}: {direction} by {abs(delta):.2f} pp "
              f"after cleaning  (CLEANED vs CLEAN)")
    print()
    print(f"  Checkpoint paths:")
    for name, path in CKPT.items():
        status = "✓ exists" if path.exists() else "✗ missing"
        print(f"    {name:<8} {status}  →  {path.name}")
    print()

    return results


# ─────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    final_results = run_comparison()
