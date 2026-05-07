"""
Data Error Detection System — Chest X-Ray Training Set
========================================================
Uses Monte Carlo Dropout inference to surface suspicious training samples.

Detection criteria (either condition triggers a flag)
------------------------------------------------------
  Criterion A — High Uncertainty:
      variance > 0.02  (model is uncertain regardless of label)

  Criterion B — Confident Mismatch:
      prediction ≠ true label  AND  confidence > 0.70
      (model is confidently wrong → strong signal of a label error)

Per-sample metrics computed
---------------------------
  predicted_label : argmax(mean_probs across T passes)
  confidence      : max(mean_probs)   — probability of predicted class
  uncertainty     : max(variance)     — worst-case variance across classes

Pipeline
--------
  1. Run MC Dropout on the full noisy training set (T=30 passes)
  2. Apply flagging criteria
  3. Report flagged indices with full metric details
  4. Return structured FlaggedSample list for downstream use

Integration with noise injection
---------------------------------
  Designed to work with chest_xray_noise.py — the noisy training
  DataLoader is passed in so that injected label errors appear in the
  flagged set, validating that the detector catches them.
"""

import pathlib
from dataclasses import dataclass
from typing import Optional

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
)
from chest_xray_model      import build_model, ChestXRayClassifier
from chest_xray_noise      import build_noisy_dataloaders, inject_label_noise
from chest_xray_mc_dropout import mc_dropout_predict, enable_mc_dropout, MCDropoutResult

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MC_PASSES          = 30
UNCERTAINTY_THRESH = 0.02    # Criterion A: flag if max variance > this
CONFIDENCE_THRESH  = 0.70    # Criterion B: flag if confidence > this AND wrong
NOISE_RATE         = 0.20
RANDOM_SEED        = 42
SAVE_PATH          = pathlib.Path(
    r"C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH\best_model.pth"
)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ─────────────────────────────────────────────
# 1.  Flagged sample record
# ─────────────────────────────────────────────
@dataclass
class FlaggedSample:
    """
    Stores all diagnostic information for one suspicious training sample.

    Attributes
    ----------
    index         : position in the training set (0-indexed)
    true_label    : label stored in the dataset (may be noisy)
    pred_label    : model's predicted class (from mean MC probs)
    confidence    : max(mean_probs) — how sure the model is
    uncertainty   : max(variance)   — worst-case variance across classes
    mean_probs    : full mean probability vector, shape (C,)
    variance      : full variance vector, shape (C,)
    criterion_a   : True if flagged by HIGH UNCERTAINTY rule
    criterion_b   : True if flagged by CONFIDENT MISMATCH rule
    is_noisy      : True if this sample is a known injected noise index
                    (only set when ground-truth noise indices are provided)
    """
    index       : int
    true_label  : int
    pred_label  : int
    confidence  : float
    uncertainty : float
    mean_probs  : np.ndarray
    variance    : np.ndarray
    criterion_a : bool         = False   # uncertainty > threshold
    criterion_b : bool         = False   # confident wrong prediction
    is_noisy    : Optional[bool] = None  # known noise? (if ground truth available)

    @property
    def trigger(self) -> str:
        """Human-readable string of which criteria fired."""
        parts = []
        if self.criterion_a:
            parts.append("HIGH_UNCERTAINTY")
        if self.criterion_b:
            parts.append("CONFIDENT_MISMATCH")
        return " + ".join(parts)

    @property
    def correct(self) -> bool:
        return self.pred_label == self.true_label


# ─────────────────────────────────────────────
# 2.  Core detection function
# ─────────────────────────────────────────────
def detect_errors(
    mc_result:           MCDropoutResult,
    uncertainty_thresh:  float              = UNCERTAINTY_THRESH,
    confidence_thresh:   float              = CONFIDENCE_THRESH,
    known_noisy_indices: Optional[set]      = None,
) -> list[FlaggedSample]:
    """
    Apply flagging criteria to MC Dropout results and return suspicious samples.

    Parameters
    ----------
    mc_result            : output of mc_dropout_predict() on the training set
    uncertainty_thresh   : Criterion A threshold on max per-class variance
    confidence_thresh    : Criterion B threshold on confidence for wrong preds
    known_noisy_indices  : set of indices injected as noise (for recall analysis)

    Returns
    -------
    flagged : list of FlaggedSample, one per suspicious training sample
    """
    N = len(mc_result.pred_classes)

    # Scalar uncertainty = max variance across both classes
    uncertainty = mc_result.variance.max(axis=1)     # shape (N,)
    confidence  = mc_result.confidence                # shape (N,) = max(mean_probs)
    preds       = mc_result.pred_classes              # shape (N,)
    labels      = mc_result.true_labels               # shape (N,)

    flagged: list[FlaggedSample] = []

    for i in range(N):
        crit_a = bool(uncertainty[i] > uncertainty_thresh)
        crit_b = bool(
            preds[i] != labels[i] and confidence[i] > confidence_thresh
        )

        if crit_a or crit_b:
            is_noisy = (i in known_noisy_indices) if known_noisy_indices is not None else None

            flagged.append(FlaggedSample(
                index       = i,
                true_label  = int(labels[i]),
                pred_label  = int(preds[i]),
                confidence  = float(confidence[i]),
                uncertainty = float(uncertainty[i]),
                mean_probs  = mc_result.mean_probs[i],
                variance    = mc_result.variance[i],
                criterion_a = crit_a,
                criterion_b = crit_b,
                is_noisy    = is_noisy,
            ))

    return flagged


# ─────────────────────────────────────────────
# 3.  Reporting
# ─────────────────────────────────────────────
def print_detection_report(
    flagged:             list[FlaggedSample],
    total_train:         int,
    known_noisy_indices: Optional[set] = None,
    show_samples:        int           = 20,
):
    """
    Print a structured detection report.

    Parameters
    ----------
    flagged              : list of FlaggedSample from detect_errors()
    total_train          : total number of training samples evaluated
    known_noisy_indices  : ground-truth noise set for precision/recall
    show_samples         : number of sample rows to print in the detail table
    """
    n_flagged    = len(flagged)
    crit_a_count = sum(1 for s in flagged if s.criterion_a)
    crit_b_count = sum(1 for s in flagged if s.criterion_b)
    both_count   = sum(1 for s in flagged if s.criterion_a and s.criterion_b)
    wrong_count  = sum(1 for s in flagged if not s.correct)

    # ── Summary box ───────────────────────────────────────────────────────
    print()
    print("╔" + "═" * 62 + "╗")
    print("║   Data Error Detection Report                             ║")
    print("╠" + "═" * 62 + "╣")
    print(f"║  Total training samples   : {total_train:>7,}                       ║")
    print(f"║  Flagged as suspicious    : {n_flagged:>7,}  "
          f"({n_flagged/total_train*100:>5.1f}% of train)        ║")
    print("╠" + "─" * 62 + "╣")
    print(f"║  Criterion A  (uncertainty > {UNCERTAINTY_THRESH}) : {crit_a_count:>7,} samples              ║")
    print(f"║  Criterion B  (conf > {CONFIDENCE_THRESH} & wrong) : {crit_b_count:>7,} samples              ║")
    print(f"║  Both criteria triggered  : {both_count:>7,} samples              ║")
    print("╠" + "─" * 62 + "╣")

    # ── Per-class breakdown of flagged ────────────────────────────────────
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cls_flags = sum(1 for s in flagged if s.true_label == cls_id)
        print(f"║  Flagged from {cls_name:<11}: {cls_flags:>7,}                              ║")

    print("╠" + "─" * 62 + "╣")

    # ── Uncertainty & confidence stats for flagged samples ────────────────
    if flagged:
        unc_vals  = [s.uncertainty for s in flagged]
        conf_vals = [s.confidence  for s in flagged]
        print(f"║  Avg uncertainty (flagged): {np.mean(unc_vals):>9.5f}                    ║")
        print(f"║  Avg confidence  (flagged): {np.mean(conf_vals)*100:>8.2f}%                   ║")

    print("╠" + "=" * 62 + "╣")

    # ── Precision / Recall vs known noise (if available) ─────────────────
    if known_noisy_indices is not None and len(known_noisy_indices) > 0:
        flagged_set   = {s.index for s in flagged}
        true_positive = flagged_set & known_noisy_indices
        precision     = len(true_positive) / len(flagged_set) * 100 if flagged_set else 0
        recall        = len(true_positive) / len(known_noisy_indices) * 100

        print("║   Noise Detection Performance (vs injected noise)         ║")
        print("╠" + "─" * 62 + "╣")
        print(f"║  Known noisy samples      : {len(known_noisy_indices):>7,}                       ║")
        print(f"║  True positives detected  : {len(true_positive):>7,}                       ║")
        print(f"║  Precision                : {precision:>7.2f}%                       ║")
        print(f"║  Recall                   : {recall:>7.2f}%                       ║")
        print("╚" + "═" * 62 + "╝")
    else:
        print("╚" + "═" * 62 + "╝")

    # ── Detail table (top N by uncertainty) ──────────────────────────────
    top_n    = sorted(flagged, key=lambda s: s.uncertainty, reverse=True)[:show_samples]
    col_w    = 70

    print(f"\n  Top-{show_samples} Flagged Samples  (sorted by uncertainty ↓)")
    print("  " + "─" * col_w)
    hdr = (
        f"  {'Idx':>6}  {'TrueLabel':>10}  {'PredLabel':>10}  "
        f"{'Confidence':>10}  {'Uncertainty':>11}  {'Trigger':<22}  {'Noisy?':>6}"
    )
    print(hdr)
    print("  " + "─" * col_w)

    for s in top_n:
        true_name = CLASS_NAMES[s.true_label]
        pred_name = CLASS_NAMES[s.pred_label]
        noisy_str = "YES" if s.is_noisy else ("NO" if s.is_noisy is False else "N/A")
        match_sym = "✓" if s.correct else "✗"
        print(
            f"  {s.index:>6}  {true_name:>10}  {pred_name:>10} {match_sym} "
            f"{s.confidence*100:>9.2f}%  {s.uncertainty:>11.6f}  "
            f"{s.trigger:<22}  {noisy_str:>6}"
        )
    print()


def print_flagged_indices(flagged: list[FlaggedSample]):
    """Print the raw list of all flagged indices."""
    indices = sorted(s.index for s in flagged)
    print(f"\n  Flagged sample indices ({len(indices)} total):")
    print("  " + "─" * 60)
    # Print in rows of 15
    for row_start in range(0, len(indices), 15):
        chunk = indices[row_start:row_start + 15]
        print("  " + "  ".join(f"{i:>5}" for i in chunk))
    print()


# ─────────────────────────────────────────────
# 4.  Load checkpoint helper
# ─────────────────────────────────────────────
def load_best_model(save_path: pathlib.Path, device: torch.device) -> ChestXRayClassifier:
    model, _ = build_model(pretrained=False)
    state    = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    print(f"  ✓ Weights loaded from: {save_path}")
    return model


# ─────────────────────────────────────────────
# 5.  Main pipeline
# ─────────────────────────────────────────────
def run_error_detection(
    uncertainty_thresh : float = UNCERTAINTY_THRESH,
    confidence_thresh  : float = CONFIDENCE_THRESH,
    noise_rate         : float = NOISE_RATE,
    n_passes           : int   = MC_PASSES,
) -> tuple[list[FlaggedSample], MCDropoutResult]:
    """
    End-to-end error detection pipeline.

    Steps
    -----
    1. Load dataset → 70/15/15 split
    2. Inject 20% label noise into training set
    3. Load trained model from checkpoint
    4. Run MC Dropout on the NOISY training set
    5. Apply detection criteria → collect flagged samples
    6. Report results

    Returns
    -------
    flagged   : list of FlaggedSample
    mc_result : full MCDropoutResult (mean_probs, variance, entropy, …)
    """
    print("=" * 64)
    print("  Data Error Detection — Chest X-Ray Training Set")
    print("=" * 64)
    print(f"\n  MC passes          : {n_passes}")
    print(f"  Uncertainty thresh : > {uncertainty_thresh}  (Criterion A)")
    print(f"  Confidence thresh  : > {confidence_thresh}  (Criterion B, if also wrong)")
    print(f"  Noise rate         : {noise_rate*100:.0f}% of training labels flipped")

    # ── Step 1: Dataset ───────────────────────────────────────────────────
    print("\n[1/5] Loading dataset …")
    all_paths, all_labels = collect_all_samples(DATASET_ROOT)
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(all_paths, all_labels)
    print(f"  Train : {len(train_paths):,}  |  Val : {len(val_paths):,}  |  Test : {len(test_paths):,}")

    # ── Step 2: Inject noise ──────────────────────────────────────────────
    print(f"\n[2/5] Injecting {noise_rate*100:.0f}% label noise into training set …")
    noisy_labels, noisy_indices = inject_label_noise(
        train_labels, noise_rate=noise_rate, seed=RANDOM_SEED
    )
    known_noisy_set = set(noisy_indices)
    print(f"  Injected noise into {len(noisy_indices):,} training samples.")

    # Build DataLoaders with noisy train labels
    (train_loader_noisy,
     val_loader,
     test_loader,
     _,
     _) = build_noisy_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        noise_rate=noise_rate,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED,
    )

    # ── Step 3: Load model ────────────────────────────────────────────────
    print("\n[3/5] Loading best model checkpoint …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    model  = load_best_model(SAVE_PATH, device)

    # ── Step 4: MC Dropout on the noisy TRAINING set ─────────────────────
    print(f"\n[4/5] Running MC Dropout ({n_passes} passes) on training set …")
    print(f"  Total forward passes : {n_passes * len(train_paths):,}")
    mc_result = mc_dropout_predict(
        model     = model,
        loader    = train_loader_noisy,
        device    = device,
        n_passes  = n_passes,
    )
    print(f"  mean_probs shape : {mc_result.mean_probs.shape}")
    print(f"  variance   shape : {mc_result.variance.shape}")

    # ── Step 5: Detect errors ─────────────────────────────────────────────
    print("\n[5/5] Applying detection criteria …")
    flagged = detect_errors(
        mc_result            = mc_result,
        uncertainty_thresh   = uncertainty_thresh,
        confidence_thresh    = confidence_thresh,
        known_noisy_indices  = known_noisy_set,
    )

    # ── Report ────────────────────────────────────────────────────────────
    print_detection_report(
        flagged              = flagged,
        total_train          = len(train_paths),
        known_noisy_indices  = known_noisy_set,
        show_samples         = 20,
    )
    print_flagged_indices(flagged)

    return flagged, mc_result


# ─────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    flagged_samples, mc_result = run_error_detection(
        uncertainty_thresh = UNCERTAINTY_THRESH,
        confidence_thresh  = CONFIDENCE_THRESH,
        noise_rate         = NOISE_RATE,
        n_passes           = MC_PASSES,
    )

    # Export: raw list of all flagged indices
    flagged_indices = sorted(s.index for s in flagged_samples)
    print(f"  Total flagged indices : {len(flagged_indices):,}")
    print(f"  First 10              : {flagged_indices[:10]}")
