"""
Chest X-Ray Pneumonia Classifier — Full Train + Evaluation Pipeline
====================================================================
Runs end-to-end:
  1. Train model (with per-epoch train/val logging)
  2. Load best saved weights automatically
  3. Evaluate on held-out test set

Metrics reported (test set)
----------------------------
  • Accuracy
  • Precision  (macro & per-class)
  • Recall     (macro & per-class)
  • F1-score   (macro & per-class)
  • Confusion matrix

Dependencies
------------
  pip install torch torchvision scikit-learn pillow
"""

import pathlib
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
from chest_xray_train import (
    train,
    SAVE_PATH,
    NUM_EPOCHS,
    LEARNING_RATE,
)

# ─────────────────────────────────────────────
# Label names (index → human-readable)
# ─────────────────────────────────────────────
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]   # index 0 → NORMAL, 1 → PNEUMONIA


# ─────────────────────────────────────────────
# 1.  Inference — collect ALL predictions at once
# ─────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model:  ChestXRayClassifier,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pass the full dataloader through the model.

    Returns
    -------
    all_preds  : (N,) int numpy array — argmax class indices
    all_probs  : (N, C) float numpy array — softmax probabilities
    all_labels : (N,) int numpy array — ground-truth labels
    """
    model.eval()

    all_preds  = []
    all_probs  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        logits = model(images)                          # (B, 2)
        probs  = torch.softmax(logits, dim=1)           # (B, 2) in [0,1]
        preds  = logits.argmax(dim=1)                   # (B,)

        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_probs, all_labels


# ─────────────────────────────────────────────
# 2.  Metrics computation
# ─────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1 (macro + per-class)
    and the raw confusion matrix.

    Parameters
    ----------
    y_true : ground-truth integer labels, shape (N,)
    y_pred : predicted integer labels,    shape (N,)

    Returns
    -------
    dict with all metric values
    """
    acc = accuracy_score(y_true, y_pred)

    # Macro averages (unweighted mean across classes)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(   y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(       y_true, y_pred, average="macro", zero_division=0)

    # Per-class values (one score per class)
    prec_per   = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per    = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1_per     = f1_score(       y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    return dict(
        accuracy       = acc,
        precision_macro= prec_macro,
        recall_macro   = rec_macro,
        f1_macro       = f1_macro,
        precision_per  = prec_per,   # shape (num_classes,)
        recall_per     = rec_per,
        f1_per         = f1_per,
        confusion_matrix = cm,
    )


# ─────────────────────────────────────────────
# 3.  Pretty-print metrics
# ─────────────────────────────────────────────
def print_metrics(metrics: dict, split_name: str = "TEST"):
    """Render a clean, structured metrics report."""

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()           # binary classification

    print()
    print("╔" + "═" * 52 + "╗")
    print(f"║  {split_name} SET — Evaluation Results" + " " * (22 - len(split_name)) + "║")
    print("╠" + "═" * 52 + "╣")

    # ── Overall ───────────────────────────────────────────────────────────
    print(f"║  {'Accuracy':<22} : {metrics['accuracy']*100:>7.2f}%               ║")
    print("╠" + "═" * 52 + "╣")

    # ── Macro averages ────────────────────────────────────────────────────
    print(f"║  {'Metric':<22}   {'Macro Avg':>10}               ║")
    print("╠" + "─" * 52 + "╣")
    print(f"║  {'Precision':<22} : {metrics['precision_macro']*100:>7.2f}%               ║")
    print(f"║  {'Recall':<22} : {metrics['recall_macro']*100:>7.2f}%               ║")
    print(f"║  {'F1-Score':<22} : {metrics['f1_macro']*100:>7.2f}%               ║")
    print("╠" + "═" * 52 + "╣")

    # ── Per-class ─────────────────────────────────────────────────────────
    print(f"║  {'Class':<12} {'Precision':>9} {'Recall':>8} {'F1':>8}       ║")
    print("╠" + "─" * 52 + "╣")
    for i, cls in enumerate(CLASS_NAMES):
        p = metrics["precision_per"][i] * 100
        r = metrics["recall_per"][i]    * 100
        f = metrics["f1_per"][i]        * 100
        print(f"║  {cls:<12} {p:>8.2f}% {r:>7.2f}% {f:>7.2f}%       ║")
    print("╠" + "═" * 52 + "╣")

    # ── Confusion matrix ──────────────────────────────────────────────────
    print("║  Confusion Matrix:                                 ║")
    print("║                                                    ║")
    print("║         Pred: NORMAL   Pred: PNEUMONIA             ║")
    print(f"║  True: NORMAL    {tn:>6}           {fp:>6}             ║")
    print(f"║  True: PNEUMONIA {fn:>6}           {tp:>6}             ║")
    print("║                                                    ║")
    print(f"║  TN={tn}  FP={fp}  FN={fn}  TP={tp}" +
          " " * (33 - len(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")) + "║")
    print("╚" + "═" * 52 + "╝")

    # ── sklearn detailed report ───────────────────────────────────────────
    print()
    print("  Full Classification Report (sklearn)")
    print("  " + "─" * 48)


def print_sklearn_report(y_true, y_pred):
    """Print sklearn's classification_report for completeness."""
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    for line in report.splitlines():
        print(f"  {line}")


# ─────────────────────────────────────────────
# 4.  Load best weights helper
# ─────────────────────────────────────────────
def load_best_model(
    save_path: pathlib.Path,
    device:    torch.device,
) -> ChestXRayClassifier:
    """
    Instantiate a fresh model and load the best saved weights from disk.

    Parameters
    ----------
    save_path : path to best_model.pth
    device    : torch device to load onto

    Returns
    -------
    model with best weights, in eval mode
    """
    model, _ = build_model(pretrained=False)   # architecture only
    state    = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"  ✓ Best weights loaded from: {save_path}")
    return model


# ─────────────────────────────────────────────
# 5.  Master routine
# ─────────────────────────────────────────────
def run_full_pipeline():
    """
    Complete pipeline:
      Step 1 → Train  (calls chest_xray_train.train())
      Step 2 → best_model.pth written by EarlyStopping during training
      Step 3 → Load best weights from disk
      Step 4 → Evaluate on test set with all metrics
    """

    # ── Step 1 & 2: Train (validation happens inside each epoch) ─────────
    print("\n" + "=" * 58)
    print("  PHASE 1 — Training")
    print("=" * 58)

    trained_model, epoch_logger = train(
        num_epochs    = NUM_EPOCHS,
        learning_rate = LEARNING_RATE,
        batch_size    = BATCH_SIZE,
    )

    # ── Re-build data splits (same seed → same test set) ─────────────────
    print("\n" + "=" * 58)
    print("  PHASE 2 — Loading Best Weights + Test Evaluation")
    print("=" * 58)

    print("\n[1/3] Reconstructing dataset splits …")
    all_paths, all_labels = collect_all_samples(DATASET_ROOT)
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(all_paths, all_labels)

    _, _, test_loader = build_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        batch_size=BATCH_SIZE,
    )
    print(f"  Test set size : {len(test_paths):,} images")

    # ── Step 3: Load best model ───────────────────────────────────────────
    print("\n[2/3] Loading best model weights …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = load_best_model(SAVE_PATH, device)

    # ── Step 4: Run inference on test set ────────────────────────────────
    print("\n[3/3] Running inference on test set …")
    preds, probs, true_labels = run_inference(best_model, test_loader, device)

    # ── Compute & display metrics ─────────────────────────────────────────
    metrics = compute_metrics(true_labels, preds)
    print_metrics(metrics, split_name="TEST")
    print_sklearn_report(true_labels, preds)

    # ── Training history summary ──────────────────────────────────────────
    best = epoch_logger.best_epoch()
    print()
    print("  Training Summary")
    print("  " + "─" * 40)
    print(f"  Best epoch      : {best['epoch']}")
    print(f"  Best val loss   : {best['val_loss']:.4f}")
    print(f"  Best val acc    : {best['val_acc']:.2f}%")
    print(f"  Epochs run      : {len(epoch_logger.history)}")
    print()

    return metrics, preds, true_labels


# ─────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    metrics, predictions, ground_truth = run_full_pipeline()
