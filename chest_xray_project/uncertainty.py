"""
uncertainty.py - Monte Carlo Dropout uncertainty estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CLASS_NAMES, EPS, MC_PASSES


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@dataclass
class MCDropoutResult:
    mean_probs: np.ndarray
    variance: np.ndarray
    entropy: np.ndarray
    pred_classes: np.ndarray
    confidence: np.ndarray
    true_labels: np.ndarray
    sample_ids: np.ndarray
    file_paths: list[str]
    all_passes: Optional[np.ndarray] = None


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_passes: int = MC_PASSES,
    store_passes: bool = False,
) -> MCDropoutResult:
    enable_mc_dropout(model)

    all_mean, all_var, all_ent = [], [], []
    all_pred, all_conf, all_labels = [], [], []
    all_sample_ids: list[np.ndarray] = []
    all_file_paths: list[str] = []
    all_raw: list[np.ndarray] = []
    n_classes = len(CLASS_NAMES)

    for images, labels, sample_ids, file_paths in loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.size(0)
        passes = torch.zeros(n_passes, batch_size, n_classes, device=device)

        for pass_index in range(n_passes):
            passes[pass_index] = F.softmax(model(images), dim=1)

        mean = passes.mean(0)
        variance = passes.var(0)
        entropy = -(mean * (mean + EPS).log()).sum(1)
        preds = mean.argmax(1)
        confidence = mean.max(1).values

        all_mean.append(mean.cpu().numpy())
        all_var.append(variance.cpu().numpy())
        all_ent.append(entropy.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_conf.append(confidence.cpu().numpy())
        all_labels.append(labels.numpy())
        all_sample_ids.append(sample_ids.numpy())
        all_file_paths.extend(list(file_paths))
        if store_passes:
            all_raw.append(passes.cpu().numpy())

    return MCDropoutResult(
        mean_probs=np.concatenate(all_mean),
        variance=np.concatenate(all_var),
        entropy=np.concatenate(all_ent),
        pred_classes=np.concatenate(all_pred),
        confidence=np.concatenate(all_conf),
        true_labels=np.concatenate(all_labels),
        sample_ids=np.concatenate(all_sample_ids),
        file_paths=all_file_paths,
        all_passes=np.concatenate(all_raw, axis=1) if store_passes else None,
    )


def compute_combined_score(
    mc_result: MCDropoutResult,
    entropy_alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend normalised predictive variance and predictive entropy into a single
    uncertainty score per sample.

    Why this matters
    ----------------
    Variance (epistemic proxy) and entropy (total uncertainty) capture
    complementary failure modes: variance flags model disagreement across MC
    passes while entropy flags low-confidence predictions even when passes
    agree.  Blending both produces a more discriminative signal for detecting
    mislabelled samples than either measure alone.

    Parameters
    ----------
    mc_result     : output of mc_dropout_predict()
    entropy_alpha : weight given to entropy (1-alpha → variance).
                    0.0 = pure variance, 1.0 = pure entropy, 0.5 = equal blend.

    Returns
    -------
    combined : (N,) float array, values in [0, 1]
    """
    def _minmax(x: np.ndarray) -> np.ndarray:
        rng = x.max() - x.min()
        return (x - x.min()) / (rng + 1e-8)

    var_max  = mc_result.variance.max(axis=1)           # (N,) max variance per sample
    ent      = mc_result.entropy                         # (N,) predictive entropy

    norm_var = _minmax(var_max)
    norm_ent = _minmax(ent)

    return (1.0 - entropy_alpha) * norm_var + entropy_alpha * norm_ent
