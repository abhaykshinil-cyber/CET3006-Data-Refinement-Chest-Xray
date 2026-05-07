"""
detection.py - Data error detection and dataset cleaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CLASS_NAMES, CONFIDENCE_THRESH, NUM_WORKERS, UNCERTAINTY_THRESH, WEIGHT_FLOOR
from data_loader import ChestXRayDataset, SampleRecord, get_transform
from uncertainty import MCDropoutResult


@dataclass
class FlaggedSample:
    sample_id: int
    file_path: str
    true_label: int
    pred_label: int
    confidence: float
    uncertainty: float
    criterion_a: bool = False
    criterion_b: bool = False
    is_noisy: Optional[bool] = None

    @property
    def trigger(self) -> str:
        parts = []
        if self.criterion_a:
            parts.append("HIGH_UNCERTAINTY")
        if self.criterion_b:
            parts.append("CONFIDENT_MISMATCH")
        return " + ".join(parts)


def detect_errors(
    mc_result: MCDropoutResult,
    uncertainty_thresh: float = UNCERTAINTY_THRESH,
    confidence_thresh: float = CONFIDENCE_THRESH,
    known_noisy_ids: Optional[set[int]] = None,
) -> list[FlaggedSample]:
    uncertainty = mc_result.variance.max(axis=1)
    flagged: list[FlaggedSample] = []

    for idx, sample_id in enumerate(mc_result.sample_ids):
        pred_label = int(mc_result.pred_classes[idx])
        true_label = int(mc_result.true_labels[idx])
        confidence = float(mc_result.confidence[idx])
        score = float(uncertainty[idx])
        criterion_a = score > uncertainty_thresh
        criterion_b = pred_label != true_label and confidence > confidence_thresh

        if criterion_a or criterion_b:
            flagged.append(
                FlaggedSample(
                    sample_id=int(sample_id),
                    file_path=mc_result.file_paths[idx],
                    true_label=true_label,
                    pred_label=pred_label,
                    confidence=confidence,
                    uncertainty=score,
                    criterion_a=criterion_a,
                    criterion_b=criterion_b,
                    is_noisy=(int(sample_id) in known_noisy_ids) if known_noisy_ids is not None else None,
                )
            )

    return flagged


def clean_training_set(
    train_samples: list[SampleRecord],
    flagged: list[FlaggedSample],
) -> tuple[list[SampleRecord], list[int]]:
    """Remove ALL flagged samples (original full-cleaning behaviour)."""
    removed_sample_ids = {sample.sample_id for sample in flagged}
    clean_samples = [sample for sample in train_samples if sample.sample_id not in removed_sample_ids]
    return clean_samples, sorted(removed_sample_ids)


def clean_training_set_partial(
    train_samples: list[SampleRecord],
    flagged: list[FlaggedSample],
    fraction: float,
) -> tuple[list[SampleRecord], list[int]]:
    """Remove only the top fraction of the flagged pool ranked by uncertainty."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ranked = sorted(flagged, key=lambda s: s.uncertainty, reverse=True)
    n_remove = max(1, int(round(len(ranked) * fraction)))
    removed_ids = {s.sample_id for s in ranked[:n_remove]}
    clean_samples = [s for s in train_samples if s.sample_id not in removed_ids]
    return clean_samples, sorted(removed_ids)


def build_cleaned_loader(
    clean_samples: list[SampleRecord],
    val_samples: list[SampleRecord],
    test_samples: list[SampleRecord],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()

    def make(samples: list[SampleRecord], augment: bool, shuffle: bool) -> DataLoader:
        return DataLoader(
            ChestXRayDataset(samples, get_transform(augment)),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin,
        )

    return (
        make(clean_samples, True, True),
        make(clean_samples, False, False),
        make(val_samples, False, False),
        make(test_samples, False, False),
    )


def assign_sample_weights(
    noisy_samples: list[SampleRecord],
    flagged: list[FlaggedSample],
    weight_floor: float = WEIGHT_FLOOR,
) -> dict[int, float]:
    """
    Assign per-sample loss weights.
    Flagged samples ranked by uncertainty: rank 0 -> weight_floor, rank N-1 -> ~1.0.
    Unflagged samples all receive weight 1.0.
    """
    weight_map: dict[int, float] = {s.sample_id: 1.0 for s in noisy_samples}
    if not flagged:
        return weight_map
    ranked = sorted(flagged, key=lambda s: s.uncertainty, reverse=True)
    n = len(ranked)
    for rank, sample in enumerate(ranked):
        w = weight_floor + (1.0 - weight_floor) * (rank / max(n - 1, 1))
        weight_map[sample.sample_id] = w
    return weight_map


class WeightedChestXRayDataset(torch.utils.data.Dataset):
    """Wraps ChestXRayDataset, returns 5-tuple: (image, label, sample_id, path, weight)."""

    def __init__(self, samples, transform, weight_map: dict[int, float]):
        self._base = ChestXRayDataset(samples, transform)
        self._weight_map = weight_map

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        image, label, sample_id, path = self._base[idx]
        weight = self._weight_map.get(int(sample_id), 1.0)
        return image, label, sample_id, path, torch.tensor(weight, dtype=torch.float32)


def build_weighted_training_loader(
    train_samples: list[SampleRecord],
    weight_map: dict[int, float],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """DataLoader over WeightedChestXRayDataset (augmented, shuffle=True)."""
    pin = torch.cuda.is_available()
    dataset = WeightedChestXRayDataset(train_samples, get_transform(True), weight_map)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )


def print_detection_report(
    flagged: list[FlaggedSample],
    total_train: int,
    known_noisy_ids: Optional[set[int]] = None,
    show_samples: int = 15,
):
    n_flagged = len(flagged)
    criterion_a_count = sum(1 for s in flagged if s.criterion_a)
    criterion_b_count = sum(1 for s in flagged if s.criterion_b)

    print()
    print("+" + "-" * 70 + "+")
    print("| Data Error Detection Report                                      |")
    print("+" + "-" * 70 + "+")
    print(f"| Training set size      : {total_train:>8,}                                 |")
    print(f"| Flagged suspicious     : {n_flagged:>8,} ({(n_flagged / total_train * 100):>5.1f}%)                    |")
    print(f"| Criterion A count      : {criterion_a_count:>8,}                                 |")
    print(f"| Criterion B count      : {criterion_b_count:>8,}                                 |")

    if known_noisy_ids is not None:
        flagged_ids = {s.sample_id for s in flagged}
        true_positives = flagged_ids & known_noisy_ids
        precision = len(true_positives) / len(flagged_ids) * 100 if flagged_ids else 0.0
        recall = len(true_positives) / len(known_noisy_ids) * 100 if known_noisy_ids else 0.0
        print("+" + "-" * 70 + "+")
        print(f"| Detection precision    : {precision:>8.2f}%                               |")
        print(f"| Detection recall       : {recall:>8.2f}%                               |")

    print("+" + "-" * 70 + "+")

    top_flagged = sorted(flagged, key=lambda s: s.uncertainty, reverse=True)[:show_samples]
    print(f"\n  Top-{show_samples} flagged samples")
    print(f"  {'ID':>6}  {'True':>10}  {'Pred':>10}  {'Conf':>7}  {'Unc':>9}  Path")
    print("  " + "-" * 110)
    for s in top_flagged:
        print(
            f"  {s.sample_id:>6}  {CLASS_NAMES[s.true_label]:>10}  "
            f"{CLASS_NAMES[s.pred_label]:>10}  {s.confidence * 100:>6.1f}%  "
            f"{s.uncertainty:>9.5f}  {s.file_path}"
        )
