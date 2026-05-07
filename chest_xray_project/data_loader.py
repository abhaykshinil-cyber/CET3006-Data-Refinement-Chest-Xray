"""
data_loader.py - Dataset loading, audit, splitting, and noise injection.
"""

from __future__ import annotations

import pathlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    DATASET_ROOT,
    IMG_SIZE,
    LABEL_MAP,
    NOISE_RATE,
    NUM_WORKERS,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    VALID_EXTENSIONS,
)


@dataclass(frozen=True)
class SampleRecord:
    sample_id: int
    path: pathlib.Path
    label: int


def _iter_class_files(split_dir: pathlib.Path, class_dir: pathlib.Path):
    if class_dir.name.startswith(".") or class_dir.name == "__MACOSX":
        return
    if not class_dir.is_dir():
        return

    class_name = class_dir.name.upper()
    if class_name not in LABEL_MAP:
        return

    label = LABEL_MAP[class_name]
    for child in sorted(class_dir.iterdir()):
        if child.name.startswith(".") or not child.is_file():
            continue
        if child.suffix.lower() not in VALID_EXTENSIONS:
            continue
        yield child.resolve(), label, split_dir.name, child.name


def collect_all_samples(root: pathlib.Path = DATASET_ROOT) -> list[SampleRecord]:
    """
    Load samples from a single canonical dataset root only.

    Only the immediate train/val/test folders under `root` are considered.
    Nested duplicate datasets are ignored.
    """
    root = root.resolve()
    split_names = ("train", "val", "test")

    seen_paths: set[pathlib.Path] = set()
    seen_filenames: set[str] = set()
    duplicate_paths: list[str] = []
    duplicate_filenames: list[str] = []
    unique_items: list[tuple[pathlib.Path, int]] = []

    for split_name in split_names:
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            for file_path, label, _, filename in _iter_class_files(split_dir, class_dir) or []:
                if file_path in seen_paths:
                    duplicate_paths.append(str(file_path))
                    continue
                if filename in seen_filenames:
                    duplicate_filenames.append(filename)
                    continue

                seen_paths.add(file_path)
                seen_filenames.add(filename)
                unique_items.append((file_path, label))

    if duplicate_paths:
        print(f"  Warning: skipped {len(duplicate_paths)} duplicate file paths.")
    if duplicate_filenames:
        print(f"  Warning: skipped {len(duplicate_filenames)} duplicate filenames.")

    samples = [
        SampleRecord(sample_id=index, path=path, label=label)
        for index, (path, label) in enumerate(unique_items)
    ]

    counts = Counter(sample.label for sample in samples)
    print("\n  Verified dataset summary")
    print(f"  Total samples : {len(samples):,}")
    for class_name, class_index in LABEL_MAP.items():
        print(f"  {class_name:<10}: {counts.get(class_index, 0):,}")

    return samples


def stratified_split(
    samples: list[SampleRecord],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    labels = [sample.label for sample in samples]
    val_test_ratio = 1.0 - train_ratio

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )
    temp_labels = [sample.label for sample in temp_samples]
    val_fraction = val_ratio / val_test_ratio
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=1.0 - val_fraction,
        stratify=temp_labels,
        random_state=seed,
    )
    return train_samples, val_samples, test_samples


def get_transform(augment: bool = False) -> transforms.Compose:
    base = [
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
    if augment:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
            + base
        )
    return transforms.Compose(base)


class ChestXRayDataset(Dataset):
    def __init__(
        self,
        samples: list[SampleRecord],
        transform: transforms.Compose,
    ):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label, sample.sample_id, str(sample.path)


def _make_loader(dataset: Dataset, shuffle: bool, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_dataloaders(
    train_samples: list[SampleRecord],
    val_samples: list[SampleRecord],
    test_samples: list[SampleRecord],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    train_dataset = ChestXRayDataset(train_samples, get_transform(augment=True))
    analysis_dataset = ChestXRayDataset(train_samples, get_transform(augment=False))
    val_dataset = ChestXRayDataset(val_samples, get_transform(augment=False))
    test_dataset = ChestXRayDataset(test_samples, get_transform(augment=False))
    return (
        _make_loader(train_dataset, True, batch_size, num_workers),
        _make_loader(analysis_dataset, False, batch_size, num_workers),
        _make_loader(val_dataset, False, batch_size, num_workers),
        _make_loader(test_dataset, False, batch_size, num_workers),
    )


def inject_label_noise(
    samples: list[SampleRecord],
    noise_rate: float = NOISE_RATE,
    seed: int = RANDOM_SEED,
) -> tuple[list[SampleRecord], list[int]]:
    if not 0.0 < noise_rate < 1.0:
        raise ValueError(f"noise_rate must be in (0,1), got {noise_rate}")

    rng = np.random.default_rng(seed)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    noisy_labels = labels.copy()
    noisy_sample_ids: list[int] = []

    for cls in np.unique(labels):
        cls_positions = np.where(labels == cls)[0]
        n_flip = max(1, int(round(len(cls_positions) * noise_rate)))
        chosen_positions = rng.choice(cls_positions, size=n_flip, replace=False)
        noisy_labels[chosen_positions] = 1 - noisy_labels[chosen_positions]
        noisy_sample_ids.extend(samples[position].sample_id for position in chosen_positions)

    noisy_samples = [
        SampleRecord(sample_id=sample.sample_id, path=sample.path, label=int(noisy_labels[idx]))
        for idx, sample in enumerate(samples)
    ]
    return noisy_samples, sorted(noisy_sample_ids)


def build_noisy_dataloaders(
    train_samples: list[SampleRecord],
    val_samples: list[SampleRecord],
    test_samples: list[SampleRecord],
    noise_rate: float = NOISE_RATE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[SampleRecord], list[int]]:
    noisy_train_samples, noisy_sample_ids = inject_label_noise(train_samples, noise_rate, seed)
    train_dataset = ChestXRayDataset(noisy_train_samples, get_transform(augment=True))
    analysis_dataset = ChestXRayDataset(noisy_train_samples, get_transform(augment=False))
    val_dataset = ChestXRayDataset(val_samples, get_transform(augment=False))
    test_dataset = ChestXRayDataset(test_samples, get_transform(augment=False))
    return (
        _make_loader(train_dataset, True, batch_size, num_workers),
        _make_loader(analysis_dataset, False, batch_size, num_workers),
        _make_loader(val_dataset, False, batch_size, num_workers),
        _make_loader(test_dataset, False, batch_size, num_workers),
        noisy_train_samples,
        noisy_sample_ids,
    )
