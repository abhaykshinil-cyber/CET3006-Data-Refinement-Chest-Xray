"""
Monte Carlo Dropout — Uncertainty Estimation
============================================
Implements MC Dropout inference for the ChestXRayClassifier.

Theory
------
Standard dropout is disabled at test time (model.eval()).
MC Dropout deliberately KEEPS dropout active during inference and runs
T stochastic forward passes per sample.  Each pass samples a different
random sub-network, producing a slightly different prediction.

Across T passes for sample x:
  p_t = softmax(f_θ_t(x))         ← probability vector for pass t

Aggregate:
  mean_prob  = (1/T) Σ p_t         ← mean predicted probability  (shape: C,)
  variance   = (1/T) Σ (p_t - mean_prob)²   ← uncertainty score  (shape: C,)
  pred_class = argmax(mean_prob)
  confidence = max(mean_prob)
  entropy    = -Σ mean_prob * log(mean_prob) ← alternative scalar uncertainty

High variance / high entropy → model is UNCERTAIN about this sample.
Low variance / low entropy  → model is CONFIDENT.

Configuration
-------------
  T (forward passes) : 30
  Device             : CUDA if available, else CPU

Integration
-----------
  from chest_xray_mc_dropout import mc_dropout_predict, MCDropoutPredictor
"""

import pathlib
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Our modules ───────────────────────────────────────────────────────────────
from chest_xray_pipeline import (
    DATASET_ROOT,
    BATCH_SIZE,
    collect_all_samples,
    stratified_split,
    build_dataloaders,
)
from chest_xray_model import build_model, ChestXRayClassifier

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MC_PASSES   = 30        # number of stochastic forward passes  T
SAVE_PATH   = pathlib.Path(
    r"C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH\best_model.pth"
)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
EPS         = 1e-8      # numerical stability for entropy


# ─────────────────────────────────────────────
# 1.  Activate dropout at inference time
# ─────────────────────────────────────────────
def enable_mc_dropout(model: nn.Module) -> None:
    """
    Switch the model into MC Dropout inference mode:
      • model.eval()  — freezes BatchNorm running statistics (desired)
      • Then re-enables every Dropout layer by setting training=True

    This is the standard MC Dropout recipe:
      BatchNorm behaves deterministically (eval mode)
      Dropout   behaves stochastically   (train mode)

    Parameters
    ----------
    model : any nn.Module containing Dropout layers
    """
    model.eval()                          # freeze BN stats globally

    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()               # re-enable stochastic masking


# ─────────────────────────────────────────────
# 2.  Result dataclass
# ─────────────────────────────────────────────
@dataclass
class MCDropoutResult:
    """
    Container for Monte Carlo Dropout results for a batch / dataset.

    Attributes
    ----------
    mean_probs   : np.ndarray  shape (N, C)
        Mean softmax probability across T passes.
        High value for class c → model leans toward class c.

    variance     : np.ndarray  shape (N, C)
        Variance of softmax probabilities across T passes.
        High value → model is uncertain about that class.

    entropy      : np.ndarray  shape (N,)
        Predictive entropy: H = -Σ p * log(p)
        Single scalar uncertainty per sample.
        Range [0, log(C)].  Higher = more uncertain.

    pred_classes : np.ndarray  shape (N,)
        argmax(mean_probs) — final predicted class label.

    confidence   : np.ndarray  shape (N,)
        max(mean_probs) — probability assigned to the predicted class.
        High confidence + low variance = reliable prediction.

    true_labels  : np.ndarray  shape (N,) or None
        Ground-truth labels if available.

    all_passes   : np.ndarray  shape (T, N, C)
        Raw softmax probabilities from every forward pass.
        Useful for custom downstream analysis.
    """
    mean_probs   : np.ndarray
    variance     : np.ndarray
    entropy      : np.ndarray
    pred_classes : np.ndarray
    confidence   : np.ndarray
    true_labels  : np.ndarray | None = None
    all_passes   : np.ndarray | None = None


# ─────────────────────────────────────────────
# 3.  Core MC Dropout inference function
# ─────────────────────────────────────────────
@torch.no_grad()
def mc_dropout_predict(
    model:       ChestXRayClassifier,
    loader:      DataLoader,
    device:      torch.device,
    n_passes:    int  = MC_PASSES,
    store_passes: bool = False,
) -> MCDropoutResult:
    """
    Run Monte Carlo Dropout inference over a DataLoader.

    For every batch:
        For t in 1..T:
            logits_t = model(images)     # stochastic (dropout active)
            probs_t  = softmax(logits_t)
        mean_prob = mean over T passes
        variance  = var  over T passes
        entropy   = -sum(mean_prob * log(mean_prob))

    Parameters
    ----------
    model        : trained ChestXRayClassifier (with best weights loaded)
    loader       : DataLoader (any split — typically test set)
    device       : torch device
    n_passes     : number of stochastic forward passes  (default 30)
    store_passes : if True, store all T raw probability matrices
                   (memory-intensive for large datasets)

    Returns
    -------
    MCDropoutResult
    """
    enable_mc_dropout(model)   # BN frozen, Dropout active

    all_mean_probs   = []
    all_variances    = []
    all_entropies    = []
    all_preds        = []
    all_confidences  = []
    all_true_labels  = []
    all_pass_storage = []   # only filled when store_passes=True

    for images, labels in loader:
        images = images.to(device, non_blocking=True)   # (B, 3, 224, 224)
        B      = images.size(0)

        # ── T stochastic forward passes ───────────────────────────────────
        pass_probs = torch.zeros(n_passes, B, len(CLASS_NAMES), device=device)

        for t in range(n_passes):
            logits          = model(images)              # (B, C) — stochastic
            probs           = F.softmax(logits, dim=1)  # (B, C) in [0,1]
            pass_probs[t]   = probs                     # store pass t

        # ── Aggregate across passes  shape: (B, C) ────────────────────────
        mean_probs = pass_probs.mean(dim=0)             # E[p]
        variance   = pass_probs.var(dim=0)              # Var[p]  (unbiased)

        # ── Predictive entropy  (scalar per sample) ───────────────────────
        entropy = -(mean_probs * (mean_probs + EPS).log()).sum(dim=1)  # (B,)

        # ── Predicted class and confidence ────────────────────────────────
        pred_classes = mean_probs.argmax(dim=1)         # (B,)
        confidence   = mean_probs.max(dim=1).values     # (B,)

        # ── Move to CPU and collect ───────────────────────────────────────
        all_mean_probs.append(mean_probs.cpu().numpy())
        all_variances.append(variance.cpu().numpy())
        all_entropies.append(entropy.cpu().numpy())
        all_preds.append(pred_classes.cpu().numpy())
        all_confidences.append(confidence.cpu().numpy())
        all_true_labels.append(labels.numpy())

        if store_passes:
            all_pass_storage.append(pass_probs.cpu().numpy())  # (T, B, C)

    # ── Concatenate all batches ───────────────────────────────────────────
    mean_probs_out  = np.concatenate(all_mean_probs,  axis=0)  # (N, C)
    variance_out    = np.concatenate(all_variances,   axis=0)  # (N, C)
    entropy_out     = np.concatenate(all_entropies,   axis=0)  # (N,)
    preds_out       = np.concatenate(all_preds,       axis=0)  # (N,)
    confidence_out  = np.concatenate(all_confidences, axis=0)  # (N,)
    true_labels_out = np.concatenate(all_true_labels, axis=0)  # (N,)

    passes_out = None
    if store_passes:
        # Stack: list of (T, B, C) → concatenate along sample axis → (T, N, C)
        passes_out = np.concatenate(all_pass_storage, axis=1)

    return MCDropoutResult(
        mean_probs   = mean_probs_out,
        variance     = variance_out,
        entropy      = entropy_out,
        pred_classes = preds_out,
        confidence   = confidence_out,
        true_labels  = true_labels_out,
        all_passes   = passes_out,
    )


# ─────────────────────────────────────────────
# 4.  Reporting helpers
# ─────────────────────────────────────────────
def print_uncertainty_report(result: MCDropoutResult, n_samples: int = 10):
    """
    Print a summary of the MC Dropout uncertainty estimation results.

    Parameters
    ----------
    result    : MCDropoutResult from mc_dropout_predict()
    n_samples : number of example samples to display in detail table
    """
    N = len(result.pred_classes)

    # ── Overall statistics ─────────────────────────────────────────────
    print()
    print("╔" + "═" * 60 + "╗")
    print("║   Monte Carlo Dropout — Uncertainty Report            ║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Samples evaluated : {N:>6,}                               ║")
    print(f"║  Forward passes (T): {MC_PASSES:>6}                               ║")
    print("╠" + "═" * 60 + "╣")

    # Mean probability stats
    print(f"║  Mean Probability  (avg across samples & classes)        ║")
    print(f"║    Mean  : {result.mean_probs.mean():.4f}                                    ║")
    print(f"║    Std   : {result.mean_probs.std():.4f}                                    ║")
    print("╠" + "─" * 60 + "╣")

    # Variance (uncertainty) stats
    print(f"║  Variance / Uncertainty (avg across samples & classes)   ║")
    print(f"║    Mean  : {result.variance.mean():.6f}                                  ║")
    print(f"║    Max   : {result.variance.max():.6f}                                  ║")
    print(f"║    Min   : {result.variance.min():.6f}                                  ║")
    print("╠" + "─" * 60 + "╣")

    # Entropy stats
    print(f"║  Predictive Entropy (per sample)                         ║")
    print(f"║    Mean  : {result.entropy.mean():.4f}                                    ║")
    print(f"║    Max   : {result.entropy.max():.4f}  ← most uncertain sample        ║")
    print(f"║    Min   : {result.entropy.min():.4f}  ← most confident sample        ║")
    print("╠" + "═" * 60 + "╣")

    # Confidence stats
    avg_conf  = result.confidence.mean() * 100
    high_conf = (result.confidence >= 0.90).sum()
    low_conf  = (result.confidence <  0.60).sum()
    print(f"║  Confidence (max mean prob per sample)                   ║")
    print(f"║    Avg confidence      : {avg_conf:>6.2f}%                         ║")
    print(f"║    High conf (≥90%)    : {high_conf:>6,} samples                   ║")
    print(f"║    Low  conf (< 60%)   : {low_conf:>6,} samples  ← uncertain       ║")
    print("╚" + "═" * 60 + "╝")

    # ── Per-sample detail table (top n_samples most uncertain) ─────────────
    print(f"\n  Top-{n_samples} most uncertain samples (highest entropy):")
    print("  " + "─" * 70)
    print(
        f"  {'Idx':>6}  {'True':>10}  {'Pred':>10}  "
        f"{'MeanProb[P]':>11}  {'Variance[P]':>11}  {'Entropy':>8}"
    )
    print("  " + "─" * 70)

    top_k_idx = np.argsort(result.entropy)[::-1][:n_samples]
    for idx in top_k_idx:
        true_cls = CLASS_NAMES[result.true_labels[idx]] if result.true_labels is not None else "N/A"
        pred_cls = CLASS_NAMES[result.pred_classes[idx]]
        mp       = result.mean_probs[idx, 1]    # P(PNEUMONIA)
        vr       = result.variance[idx, 1]      # Var for PNEUMONIA
        ent      = result.entropy[idx]
        match    = "✓" if result.true_labels is not None and result.true_labels[idx] == result.pred_classes[idx] else "✗"
        print(
            f"  {idx:>6}  {true_cls:>10}  {pred_cls:>10}  "
            f"{mp:>11.4f}  {vr:>11.6f}  {ent:>8.4f}  {match}"
        )
    print()


def load_best_model(save_path: pathlib.Path, device: torch.device) -> ChestXRayClassifier:
    """Load best weights from disk into a fresh model instance."""
    model, _ = build_model(pretrained=False)
    state    = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    print(f"  ✓ Loaded weights from: {save_path}")
    return model


# ─────────────────────────────────────────────
# 5.  MC Dropout Predictor class (convenience wrapper)
# ─────────────────────────────────────────────
class MCDropoutPredictor:
    """
    High-level wrapper around mc_dropout_predict().

    Usage
    -----
        predictor = MCDropoutPredictor.from_checkpoint(SAVE_PATH)
        result    = predictor.predict(test_loader)
        predictor.report(result)
    """

    def __init__(self, model: ChestXRayClassifier, device: torch.device, n_passes: int = MC_PASSES):
        self.model    = model
        self.device   = device
        self.n_passes = n_passes

    @classmethod
    def from_checkpoint(
        cls,
        save_path: pathlib.Path = SAVE_PATH,
        n_passes:  int          = MC_PASSES,
    ) -> "MCDropoutPredictor":
        """Build predictor directly from a saved checkpoint file."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = load_best_model(save_path, device)
        return cls(model, device, n_passes)

    def predict(
        self,
        loader:       DataLoader,
        store_passes: bool = False,
    ) -> MCDropoutResult:
        """Run MC Dropout inference. Returns MCDropoutResult."""
        return mc_dropout_predict(
            model        = self.model,
            loader       = loader,
            device       = self.device,
            n_passes     = self.n_passes,
            store_passes = store_passes,
        )

    def report(self, result: MCDropoutResult, n_samples: int = 10):
        """Pretty-print uncertainty report."""
        print_uncertainty_report(result, n_samples=n_samples)


# ─────────────────────────────────────────────
# 6.  Entry point / standalone demo
# ─────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Monte Carlo Dropout — Inference & Uncertainty Estimation")
    print("=" * 62)

    # ── Load data (test split only needed for evaluation) ────────────────
    print("\n[1/4] Loading dataset splits …")
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

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[2/4] Loading best model from checkpoint …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")
    model  = load_best_model(SAVE_PATH, device)

    # ── Run MC Dropout inference ──────────────────────────────────────────
    print(f"\n[3/4] Running MC Dropout  ({MC_PASSES} passes × {len(test_paths):,} samples) …")
    result = mc_dropout_predict(
        model        = model,
        loader       = test_loader,
        device       = device,
        n_passes     = MC_PASSES,
        store_passes = True,    # keep all passes for demo
    )
    print(f"  mean_probs  shape : {result.mean_probs.shape}")
    print(f"  variance    shape : {result.variance.shape}")
    print(f"  entropy     shape : {result.entropy.shape}")
    print(f"  all_passes  shape : {result.all_passes.shape}")

    # ── Print report ──────────────────────────────────────────────────────
    print("\n[4/4] Uncertainty Report")
    print_uncertainty_report(result, n_samples=10)

    # ── Quick accuracy check ──────────────────────────────────────────────
    if result.true_labels is not None:
        acc = (result.pred_classes == result.true_labels).mean() * 100
        print(f"  MC Dropout Test Accuracy : {acc:.2f}%")
        print(f"  (Using argmax of mean probability across {MC_PASSES} passes)")

    return result


if __name__ == "__main__":
    result = main()
