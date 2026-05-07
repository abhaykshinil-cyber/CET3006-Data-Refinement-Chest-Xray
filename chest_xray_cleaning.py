"""
Dataset Cleaning — Remove Flagged Suspicious Samples
=====================================================
Takes the flagged indices produced by chest_xray_error_detection.py
and filters them out of the training set, producing a clean dataset.

Pipeline
--------
  1. Collect all images → stratified 70/15/15 split
  2. Inject 20% label noise (reproduces exact noisy training set)
  3. Run MC Dropout → detect suspicious samples
  4. Remove flagged indices from train paths & labels
  5. Build a fresh cleaned DataLoader (val & test untouched)

Outputs
-------
  • Cleaned DataLoader  (ready to plug into chest_xray_train.py)
  • Printed comparison: Original vs Cleaned dataset size
  • Per-class breakdown before and after cleaning
  • Overlap report: how many true noise samples were removed
"""

import pathlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Our modules ───────────────────────────────────────────────────────────────
from chest_xray_pipeline import (
    DATASET_ROOT,
    BATCH_SIZE,
    collect_all_samples,
    stratified_split,
    build_dataloaders,
    ChestXRayDataset,
    get_transform,
)
from chest_xray_noise import (
    build_noisy_dataloaders,
    inject_label_noise,
    NOISE_RATE,
)
from chest_xray_mc_dropout import mc_dropout_predict
from chest_xray_error_detection import (
    detect_errors,
    load_best_model,
    FlaggedSample,
    UNCERTAINTY_THRESH,
    CONFIDENCE_THRESH,
    MC_PASSES,
    SAVE_PATH,
    CLASS_NAMES,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# 1.  Core cleaning function
# ─────────────────────────────────────────────
def clean_training_set(
    train_paths:  list,
    train_labels: list[int],
    flagged:      list[FlaggedSample],
) -> tuple[list, list[int], list[int]]:
    """
    Remove flagged sample indices from the training set.

    Parameters
    ----------
    train_paths   : list of pathlib.Path — all training image paths
    train_labels  : list of int          — corresponding labels (may be noisy)
    flagged       : list of FlaggedSample from detect_errors()

    Returns
    -------
    clean_paths   : filtered image paths  (flagged indices removed)
    clean_labels  : filtered labels       (flagged indices removed)
    removed_idx   : sorted list of removed indices for audit trail
    """
    # Collect the indices to drop as a set for O(1) lookup
    flagged_set = {s.index for s in flagged}
    removed_idx = sorted(flagged_set)

    clean_paths  = []
    clean_labels = []

    for i, (path, label) in enumerate(zip(train_paths, train_labels)):
        if i not in flagged_set:
            clean_paths.append(path)
            clean_labels.append(label)

    return clean_paths, clean_labels, removed_idx


# ─────────────────────────────────────────────
# 2.  Build cleaned DataLoader
# ─────────────────────────────────────────────
def build_cleaned_dataloaders(
    clean_paths:  list,
    clean_labels: list[int],
    val_paths:    list,
    val_labels:   list[int],
    test_paths:   list,
    test_labels:  list[int],
    batch_size:   int = BATCH_SIZE,
    num_workers:  int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap cleaned training data plus unchanged val/test into DataLoaders.

    Parameters
    ----------
    clean_paths / clean_labels : output of clean_training_set()
    val_paths / val_labels     : original validation split (UNCHANGED)
    test_paths / test_labels   : original test split       (UNCHANGED)

    Returns
    -------
    clean_train_loader : DataLoader  — shuffled, no suspicious samples
    val_loader         : DataLoader  — unchanged
    test_loader        : DataLoader  — unchanged
    """
    train_transform = get_transform(augment=True)
    eval_transform  = get_transform(augment=False)

    clean_train_dataset = ChestXRayDataset(clean_paths,  clean_labels,  train_transform)
    val_dataset         = ChestXRayDataset(val_paths,    val_labels,    eval_transform)
    test_dataset        = ChestXRayDataset(test_paths,   test_labels,   eval_transform)

    pin = torch.cuda.is_available()

    clean_train_loader = DataLoader(
        clean_train_dataset,
        batch_size  = batch_size,
        shuffle     = True,           # shuffle only training set
        num_workers = num_workers,
        pin_memory  = pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin,
    )

    return clean_train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 3.  Reporting
# ─────────────────────────────────────────────
def _class_distribution(labels: list[int]) -> dict[str, int]:
    dist = {cls: 0 for cls in CLASS_NAMES}
    for lbl in labels:
        dist[CLASS_NAMES[lbl]] += 1
    return dist


def print_cleaning_report(
    original_paths:      list,
    original_labels:     list[int],
    clean_paths:         list,
    clean_labels:        list[int],
    removed_idx:         list[int],
    flagged:             list[FlaggedSample],
    known_noisy_indices: set | None = None,
):
    """
    Print a detailed before/after comparison for the dataset cleaning step.
    """
    orig_total    = len(original_paths)
    clean_total   = len(clean_paths)
    n_removed     = orig_total - clean_total

    orig_dist  = _class_distribution(original_labels)
    clean_dist = _class_distribution(clean_labels)

    # ── Summary box ───────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 60 + "╗")
    print("║   Dataset Cleaning Report                               ║")
    print("╠" + "═" * 60 + "╣")

    # Size comparison
    print(f"║  {'Metric':<30} {'Before':>9} {'After':>9}     ║")
    print("╠" + "─" * 60 + "╣")
    print(f"║  {'Total training samples':<30} "
          f"{orig_total:>9,} {clean_total:>9,}     ║")
    for cls in CLASS_NAMES:
        print(f"║  {'  ' + cls:<30} "
              f"{orig_dist[cls]:>9,} {clean_dist[cls]:>9,}     ║")
    print("╠" + "─" * 60 + "╣")
    print(f"║  Samples removed          : {n_removed:>7,}  "
          f"({n_removed/orig_total*100:.2f}% of original)   ║")

    # Criterion breakdown for removed samples
    crit_a = sum(1 for s in flagged if s.criterion_a)
    crit_b = sum(1 for s in flagged if s.criterion_b)
    both   = sum(1 for s in flagged if s.criterion_a and s.criterion_b)
    print(f"║    ├─ Criterion A only (uncertainty)  : {crit_a - both:>6,}              ║")
    print(f"║    ├─ Criterion B only (conf mismatch): {crit_b - both:>6,}              ║")
    print(f"║    └─ Both criteria                   : {both:>6,}              ║")
    print("╠" + "─" * 60 + "╣")

    # Class balance after cleaning
    clean_pct_normal    = clean_dist["NORMAL"]    / clean_total * 100 if clean_total else 0
    clean_pct_pneumonia = clean_dist["PNEUMONIA"] / clean_total * 100 if clean_total else 0
    print(f"║  Class balance after cleaning:                          ║")
    print(f"║    NORMAL    : {clean_dist['NORMAL']:>7,}  ({clean_pct_normal:>5.1f}%)                ║")
    print(f"║    PNEUMONIA : {clean_dist['PNEUMONIA']:>7,}  ({clean_pct_pneumonia:>5.1f}%)                ║")
    print("╠" + "═" * 60 + "╣")

    # Noise recovery (if ground-truth noisy indices known)
    if known_noisy_indices:
        removed_set    = set(removed_idx)
        true_removed   = removed_set & known_noisy_indices    # noise correctly removed
        false_removed  = removed_set - known_noisy_indices    # clean samples wrongly removed
        missed_noise   = known_noisy_indices - removed_set    # noise NOT caught
        precision = len(true_removed) / len(removed_set) * 100 if removed_set else 0
        recall    = len(true_removed) / len(known_noisy_indices) * 100

        print("║   Noise Recovery Analysis                               ║")
        print("╠" + "─" * 60 + "╣")
        print(f"║  Total injected noise    : {len(known_noisy_indices):>7,}                    ║")
        print(f"║  Noise correctly removed : {len(true_removed):>7,}  (True  Positives)    ║")
        print(f"║  Clean samples removed   : {len(false_removed):>7,}  (False Positives)    ║")
        print(f"║  Noise still in dataset  : {len(missed_noise):>7,}  (False Negatives)    ║")
        print(f"║  Cleaning Precision      : {precision:>7.2f}%                    ║")
        print(f"║  Cleaning Recall         : {recall:>7.2f}%                    ║")
        print("╚" + "═" * 60 + "╝")
    else:
        print("╚" + "═" * 60 + "╝")

    # ── DataLoader summary ────────────────────────────────────────────────
    print()
    print("  DataLoaders ready:")
    print(f"  clean_train_loader → {clean_total:,} samples  "
          f"({-(-clean_total // BATCH_SIZE)} batches, batch={BATCH_SIZE}, shuffle=True)")
    print("  val_loader         → unchanged")
    print("  test_loader        → unchanged")
    print()


# ─────────────────────────────────────────────
# 4.  Full cleaning pipeline
# ─────────────────────────────────────────────
def run_dataset_cleaning(
    uncertainty_thresh : float = UNCERTAINTY_THRESH,
    confidence_thresh  : float = CONFIDENCE_THRESH,
    noise_rate         : float = NOISE_RATE,
    n_passes           : int   = MC_PASSES,
    batch_size         : int   = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader, list[int]]:
    """
    End-to-end dataset cleaning pipeline.

    Steps
    -----
    1. Load dataset → 70/15/15 stratified split
    2. Inject 20% label noise into training set
    3. Load trained model checkpoint
    4. Run MC Dropout on noisy training set → compute per-sample metrics
    5. Detect suspicious samples via flagging criteria
    6. Remove flagged samples → produce clean training paths + labels
    7. Build cleaned DataLoaders
    8. Print before/after comparison

    Returns
    -------
    clean_train_loader : DataLoader  — cleaned training set
    val_loader         : DataLoader  — unchanged validation set
    test_loader        : DataLoader  — unchanged test set
    removed_indices    : list[int]   — sorted list of removed sample indices
    """
    print("=" * 62)
    print("  Dataset Cleaning Pipeline — Chest X-Ray")
    print("=" * 62)

    # ── Step 1: Load & split ──────────────────────────────────────────────
    print("\n[1/6] Loading dataset and splitting 70/15/15 …")
    all_paths, all_labels = collect_all_samples(DATASET_ROOT)
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(all_paths, all_labels)

    print(f"  Train : {len(train_paths):,}  |  "
          f"Val : {len(val_paths):,}  |  "
          f"Test : {len(test_paths):,}")

    # ── Step 2: Inject noise ──────────────────────────────────────────────
    print(f"\n[2/6] Injecting {noise_rate*100:.0f}% label noise into training set …")
    noisy_labels, noisy_indices = inject_label_noise(
        train_labels, noise_rate=noise_rate, seed=RANDOM_SEED
    )
    known_noisy_set = set(noisy_indices)
    print(f"  Noise injected into {len(noisy_indices):,} samples.")

    # Build noisy DataLoader for MC Dropout pass
    (noisy_train_loader, val_loader, test_loader, _, _) = build_noisy_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        noise_rate  = noise_rate,
        batch_size  = batch_size,
        seed        = RANDOM_SEED,
    )

    # ── Step 3: Load model ────────────────────────────────────────────────
    print("\n[3/6] Loading model checkpoint …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    model = load_best_model(SAVE_PATH, device)

    # ── Step 4: MC Dropout on noisy training set ──────────────────────────
    print(f"\n[4/6] Running{n_passes} MC Dropout passes on training set …")
    mc_result = mc_dropout_predict(
        model    = model,
        loader   = noisy_train_loader,
        device   = device,
        n_passes = n_passes,
    )
    print(f"  Inference complete — {len(train_paths):,} samples processed.")

    # ── Step 5: Detect suspicious samples ────────────────────────────────
    print("\n[5/6] Detecting suspicious samples …")
    flagged = detect_errors(
        mc_result            = mc_result,
        uncertainty_thresh   = uncertainty_thresh,
        confidence_thresh    = confidence_thresh,
        known_noisy_indices  = known_noisy_set,
    )
    print(f"  Flagged : {len(flagged):,} suspicious samples out of {len(train_paths):,}.")

    # ── Step 6: Clean the training set ───────────────────────────────────
    print("\n[6/6] Removing flagged samples from training set …")

    # Use noisy_labels (post-injection) as the "original" so the report
    # reflects what the model actually sees before cleaning
    noisy_labels_list = noisy_labels if isinstance(noisy_labels, list) else noisy_labels.tolist()

    clean_paths, clean_labels, removed_idx = clean_training_set(
        train_paths   = train_paths,
        train_labels  = noisy_labels_list,
        flagged       = flagged,
    )

    # ── Build cleaned DataLoaders ─────────────────────────────────────────
    clean_train_loader, val_loader, test_loader = build_cleaned_dataloaders(
        clean_paths   = clean_paths,
        clean_labels  = clean_labels,
        val_paths     = val_paths,
        val_labels    = val_labels,
        test_paths    = test_paths,
        test_labels   = test_labels,
        batch_size    = batch_size,
    )

    # ── Print comparison report ───────────────────────────────────────────
    print_cleaning_report(
        original_paths      = train_paths,
        original_labels     = noisy_labels_list,
        clean_paths         = clean_paths,
        clean_labels        = clean_labels,
        removed_idx         = removed_idx,
        flagged             = flagged,
        known_noisy_indices = known_noisy_set,
    )

    return clean_train_loader, val_loader, test_loader, removed_idx


# ─────────────────────────────────────────────
# 5.  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    clean_train_loader, val_loader, test_loader, removed_indices = run_dataset_cleaning()

    # ── Verify cleaned batch ──────────────────────────────────────────────
    print("  Sanity check — first batch from clean_train_loader:")
    images, labels = next(iter(clean_train_loader))
    print(f"    Image tensor shape : {tuple(images.shape)}")
    print(f"    Labels             : {labels.tolist()[:16]} …")
    print(f"    Pixel range        : [{images.min():.3f}, {images.max():.3f}]")

    print(f"\n  Removed indices (first 20): {removed_indices[:20]}")
    print(f"  Total removed            : {len(removed_indices):,}")
