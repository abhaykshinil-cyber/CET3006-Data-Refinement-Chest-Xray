"""
Label Noise Injection — Chest X-Ray Training Pipeline
======================================================
Injects symmetric label noise into the TRAINING set only.

Behaviour
---------
  • Randomly selects exactly 20% of training indices (stratified across
    classes so noise is proportional, not skewed to one class).
  • Flips labels:  0 (NORMAL) → 1 (PNEUMONIA)  and  1 → 0
  • Validation and test sets are NEVER touched.
  • A reproducible random seed is used so experiments are repeatable.
  • Detailed per-class noise statistics are printed after injection.

Integration
-----------
  Import inject_label_noise() and wrap it around the training labels
  returned by stratified_split() in chest_xray_pipeline.py, then pass
  the noisy labels straight into build_dataloaders() via ChestXRayDataset.
"""

import pathlib
import random
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Our existing modules ──────────────────────────────────────────────────────
from chest_xray_pipeline import (
    DATASET_ROOT,
    BATCH_SIZE,
    collect_all_samples,
    stratified_split,
    build_dataloaders,
    ChestXRayDataset,
    get_transform,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
NOISE_RATE  = 0.20      # fraction of training labels to flip
RANDOM_SEED = 42        # reproducibility

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


# ─────────────────────────────────────────────
# 1.  Core noise injection function
# ─────────────────────────────────────────────
def inject_label_noise(
    labels:      list[int],
    noise_rate:  float = NOISE_RATE,
    seed:        int   = RANDOM_SEED,
) -> tuple[list[int], list[int]]:
    """
    Randomly flip a `noise_rate` fraction of labels in-place (binary flip).

    Strategy: stratified selection
    ──────────────────────────────
    Rather than sampling uniformly from the full list (which could
    accidentally corrupt a larger share of the minority class), we sample
    `noise_rate * class_count` indices independently from each class.
    This keeps the corruption rate identical across NORMAL and PNEUMONIA.

    Parameters
    ----------
    labels     : original integer labels [0 or 1], length N
    noise_rate : fraction to flip  (default 0.20 → 20 %)
    seed       : numpy random seed for reproducibility

    Returns
    -------
    noisy_labels    : new list with flipped entries
    noisy_indices   : sorted list of indices that were flipped
    """
    if not (0.0 < noise_rate < 1.0):
        raise ValueError(f"noise_rate must be in (0, 1), got {noise_rate}")

    rng          = np.random.default_rng(seed)
    labels_array = np.array(labels, dtype=np.int64)
    noisy_labels = labels_array.copy()

    noisy_indices_all = []

    # Stratified selection: process each class separately
    unique_classes = np.unique(labels_array)
    for cls in unique_classes:
        cls_indices  = np.where(labels_array == cls)[0]          # all indices for this class
        n_flip       = max(1, int(round(len(cls_indices) * noise_rate)))
        flip_indices = rng.choice(cls_indices, size=n_flip, replace=False)
        noisy_labels[flip_indices] = 1 - noisy_labels[flip_indices]  # binary flip
        noisy_indices_all.extend(flip_indices.tolist())

    noisy_indices_all = sorted(noisy_indices_all)
    return noisy_labels.tolist(), noisy_indices_all


# ─────────────────────────────────────────────
# 2.  Reporting
# ─────────────────────────────────────────────
def print_noise_report(
    original_labels: list[int],
    noisy_labels:    list[int],
    noisy_indices:   list[int],
):
    """
    Print detailed before/after statistics for the injected noise.
    """
    orig  = np.array(original_labels)
    noisy = np.array(noisy_labels)

    total_train   = len(orig)
    total_flipped = len(noisy_indices)

    print()
    print("╔" + "═" * 54 + "╗")
    print("║   Label Noise Injection Report                       ║")
    print("╠" + "═" * 54 + "╣")
    print(f"║  Noise rate          : {NOISE_RATE*100:.0f}%                           ║")
    print(f"║  Total training imgs : {total_train:>6,}                       ║")
    print(f"║  Labels flipped      : {total_flipped:>6,}  "
          f"({total_flipped/total_train*100:.2f}% of train)      ║")
    print("╠" + "═" * 54 + "╣")

    # Per-class breakdown
    print(f"║  {'Class':<12} {'Original':>9} {'After Noise':>12} {'Flipped':>8}  ║")
    print("╠" + "─" * 54 + "╣")

    for cls_id, cls_name in CLASS_NAMES.items():
        orig_count     = int((orig  == cls_id).sum())
        noisy_count    = int((noisy == cls_id).sum())
        # How many from the original class were flipped away?
        flipped_away   = int(((orig == cls_id) & (noisy != cls_id)).sum())
        print(
            f"║  {cls_name:<12} {orig_count:>9,} {noisy_count:>12,} {flipped_away:>8,}  ║"
        )

    print("╠" + "═" * 54 + "╣")

    # Label distribution shift
    orig_normal_pct  = (orig  == 0).sum() / total_train * 100
    noisy_normal_pct = (noisy == 0).sum() / total_train * 100
    print(f"║  NORMAL %  before noise : {orig_normal_pct:>5.1f}%                   ║")
    print(f"║  NORMAL %  after  noise : {noisy_normal_pct:>5.1f}%                   ║")
    print("╚" + "═" * 54 + "╝")

    # Flip direction summary
    print()
    print("  Flip direction breakdown:")
    print("  " + "─" * 36)
    flip_0_to_1 = int(((orig[noisy_indices] == 0) & (noisy[noisy_indices] == 1)).sum())
    flip_1_to_0 = int(((orig[noisy_indices] == 1) & (noisy[noisy_indices] == 0)).sum())
    print(f"  NORMAL → PNEUMONIA (0→1) : {flip_0_to_1:>5,}")
    print(f"  PNEUMONIA → NORMAL (1→0) : {flip_1_to_0:>5,}")
    print(f"  Total                    : {flip_0_to_1 + flip_1_to_0:>5,}")
    print()


# ─────────────────────────────────────────────
# 3.  Build noisy DataLoaders
# ─────────────────────────────────────────────
def build_noisy_dataloaders(
    train_paths,  train_labels,
    val_paths,    val_labels,
    test_paths,   test_labels,
    noise_rate:  float = NOISE_RATE,
    batch_size:  int   = BATCH_SIZE,
    seed:        int   = RANDOM_SEED,
    num_workers: int   = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, list[int], list[int]]:
    """
    Inject label noise into train_labels, then return three DataLoaders.

    Val and test loaders use the ORIGINAL, clean labels.

    Returns
    -------
    train_loader_noisy : DataLoader  — noisy training set
    val_loader         : DataLoader  — clean validation set  (unchanged)
    test_loader        : DataLoader  — clean test set        (unchanged)
    noisy_labels       : list[int]   — final (post-flip) training labels
    noisy_indices      : list[int]   — which indices were flipped
    """
    # ── Inject noise into training labels only ────────────────────────────
    noisy_labels, noisy_indices = inject_label_noise(
        train_labels, noise_rate=noise_rate, seed=seed
    )

    # ── Print report ──────────────────────────────────────────────────────
    print_noise_report(train_labels, noisy_labels, noisy_indices)

    # ── Build datasets ────────────────────────────────────────────────────
    train_transform = get_transform(augment=True)
    eval_transform  = get_transform(augment=False)

    noisy_train_dataset = ChestXRayDataset(train_paths, noisy_labels, train_transform)
    val_dataset         = ChestXRayDataset(val_paths,   val_labels,   eval_transform)
    test_dataset        = ChestXRayDataset(test_paths,  test_labels,  eval_transform)

    pin = torch.cuda.is_available()

    train_loader_noisy = DataLoader(
        noisy_train_dataset,
        batch_size=batch_size,
        shuffle=True,               # shuffle only training set
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    return train_loader_noisy, val_loader, test_loader, noisy_labels, noisy_indices


# ─────────────────────────────────────────────
# 4.  Standalone demo / entry point
# ─────────────────────────────────────────────
def main():
    print("=" * 56)
    print("  Label Noise Injection — Chest X-Ray Pipeline")
    print("=" * 56)

    # ── Step 1: Load all data and split ──────────────────────────────────
    print("\n[1/3] Loading dataset and applying 70/15/15 split …")
    all_paths, all_labels = collect_all_samples(DATASET_ROOT)

    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(all_paths, all_labels)

    print(f"  Clean train set : {len(train_paths):,} images")
    print(f"  Val set         : {len(val_paths):,} images  ← unchanged")
    print(f"  Test set        : {len(test_paths):,} images  ← unchanged")

    # ── Step 2: Inject noise ─────────────────────────────────────────────
    print(f"\n[2/3] Injecting {NOISE_RATE*100:.0f}% label noise into training set …")

    (train_loader_noisy,
     val_loader,
     test_loader,
     noisy_labels,
     noisy_indices) = build_noisy_dataloaders(
        train_paths,  train_labels,
        val_paths,    val_labels,
        test_paths,   test_labels,
        noise_rate=NOISE_RATE,
        batch_size=BATCH_SIZE,
    )

    # ── Step 3: Verify DataLoader batch ──────────────────────────────────
    print("[3/3] Verifying noisy DataLoader …")
    img_batch, lbl_batch = next(iter(train_loader_noisy))
    print(f"  Batch image shape : {tuple(img_batch.shape)}")
    print(f"  Batch labels      : {lbl_batch.tolist()}")
    print(f"  Unique values     : {sorted(lbl_batch.unique().tolist())}")

    print("\n✓ Noise injection complete.")
    print(f"  train_loader_noisy → {len(train_loader_noisy)} batches of {BATCH_SIZE}")
    print(f"  val_loader         → {len(val_loader)} batches  (CLEAN)")
    print(f"  test_loader        → {len(test_loader)} batches  (CLEAN)")

    return train_loader_noisy, val_loader, test_loader, noisy_labels, noisy_indices


if __name__ == "__main__":
    (train_loader_noisy,
     val_loader,
     test_loader,
     noisy_labels,
     noisy_indices) = main()
