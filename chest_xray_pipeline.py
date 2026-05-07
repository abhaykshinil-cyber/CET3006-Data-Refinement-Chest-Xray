"""
Chest X-Ray Pneumonia Classification — Dataset Pipeline
========================================================
Steps performed
---------------
1. Crawl all images from the original directory structure (train/val/test).
2. Pool them into a single corpus, then re-split:  70% train | 15% val | 15% test
   using stratified sampling so class balance is preserved across every split.
3. Apply transforms: resize → 224×224, convert to float, normalise to [0, 1].
4. Wrap splits in a custom PyTorch Dataset and DataLoader.
   - Batch size  : 32
   - Shuffle     : True only for the training set
5. Print dataset and dataloader statistics for each split.

Label mapping
-------------
  NORMAL    → 0
  PNEUMONIA → 1
"""

import os
import pathlib
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────
# Root directory that contains train/, val/, test/ sub-folders.
DATASET_ROOT = pathlib.Path(
    r"C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH\chest_xray"
)

LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}
IMG_SIZE  = (224, 224)
BATCH_SIZE = 32

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15        # implicitly 1 - TRAIN_RATIO - VAL_RATIO

RANDOM_SEED = 42          # for reproducibility

# ─────────────────────────────────────────────
# 1.  Crawl all images from the directory tree
# ─────────────────────────────────────────────
def collect_all_samples(root: pathlib.Path) -> Tuple[List[pathlib.Path], List[int]]:
    """
    Walk every sub-folder under `root` that matches a known class name
    (NORMAL / PNEUMONIA) and collect (image_path, label) pairs.

    The original dataset ships with three top-level splits (train, val, test).
    We pool them all here so we can perform our own stratified re-split.
    """
    image_paths: List[pathlib.Path] = []
    labels:      List[int]          = []

    # Each immediate child of root is a split folder (train/val/test)
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        if split_dir.name.startswith("__"):   # skip __MACOSX etc.
            continue

        # Each child of the split is a class folder (NORMAL / PNEUMONIA)
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.upper()
            if class_name not in LABEL_MAP:
                continue
            label = LABEL_MAP[class_name]

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in VALID_EXTENSIONS:
                    image_paths.append(img_file)
                    labels.append(label)

    return image_paths, labels


# ─────────────────────────────────────────────
# 2.  Stratified 70 / 15 / 15 split
# ─────────────────────────────────────────────
def stratified_split(
    paths:  List[pathlib.Path],
    labels: List[int],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed: int = RANDOM_SEED,
):
    """
    Returns six lists:
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels
    """
    # First cut: separate train from (val + test)
    val_test_ratio = 1.0 - train_ratio

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second cut: split (val + test) into val and test in equal halves
    # val_ratio / val_test_ratio gives the fraction of the temp set that becomes val
    val_fraction = val_ratio / val_test_ratio

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=1.0 - val_fraction,
        stratify=temp_labels,
        random_state=seed,
    )

    return (
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
    )


# ─────────────────────────────────────────────
# 3.  Transforms  —  resize + normalise to [0,1]
# ─────────────────────────────────────────────
#
# ToTensor() converts a PIL image in the range [0, 255] to a float32 tensor
# in the range [0.0, 1.0] — satisfying the normalisation requirement directly.
# We deliberately skip mean/std standardisation so the pixels stay in [0, 1].

def get_transform(augment: bool = False) -> transforms.Compose:
    """
    Returns the appropriate transform pipeline.
    • augment=True  → used for the training set (add flips / minor jitter)
    • augment=False → used for val and test sets (deterministic)
    """
    base = [
        transforms.Resize(IMG_SIZE),          # 224 × 224
        transforms.Grayscale(num_output_channels=3),  # ensure 3-channel RGB
        transforms.ToTensor(),                 # [0,1] float32
    ]
    if augment:
        augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
        return transforms.Compose(augmentation + base)
    return transforms.Compose(base)


# ─────────────────────────────────────────────
# 4.  PyTorch Dataset
# ─────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    """
    Custom PyTorch Dataset for the Chest X-Ray Pneumonia dataset.

    Parameters
    ----------
    image_paths : list of pathlib.Path
    labels      : list of int  (0 = NORMAL, 1 = PNEUMONIA)
    transform   : torchvision transform pipeline
    """

    def __init__(
        self,
        image_paths: List[pathlib.Path],
        labels:      List[int],
        transform:   transforms.Compose,
    ):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        # Open as RGB so the Grayscale → RGB step is always consistent
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


# ─────────────────────────────────────────────
# 5.  Build DataLoaders
# ─────────────────────────────────────────────
def build_dataloaders(
    train_paths, train_labels,
    val_paths,   val_labels,
    test_paths,  test_labels,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,       # set >0 for faster loading on multi-core systems
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_transform = get_transform(augment=True)
    eval_transform  = get_transform(augment=False)

    train_dataset = ChestXRayDataset(train_paths, train_labels, train_transform)
    val_dataset   = ChestXRayDataset(val_paths,   val_labels,   eval_transform)
    test_dataset  = ChestXRayDataset(test_paths,  test_labels,  eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,            # ← shuffle only the training set
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 6.  Reporting helpers
# ─────────────────────────────────────────────
def class_counts(labels: List[int]) -> dict:
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    counts  = {inv_map[k]: 0 for k in inv_map}
    for lbl in labels:
        counts[inv_map[lbl]] += 1
    return counts


def print_split_info(
    split_name: str,
    paths: List[pathlib.Path],
    labels: List[int],
    loader: DataLoader,
):
    total   = len(paths)
    counts  = class_counts(labels)
    batches = len(loader)

    print(f"\n{'─'*46}")
    print(f"  Split      : {split_name}")
    print(f"  Total imgs : {total:>6,}")
    for cls_name, cnt in counts.items():
        pct = cnt / total * 100
        print(f"  {cls_name:<11}: {cnt:>6,}  ({pct:.1f}%)")
    print(f"  Batches    : {batches:>6,}  (batch_size={BATCH_SIZE})")
    print(f"{'─'*46}")


# ─────────────────────────────────────────────
# 7.  Main entry point
# ─────────────────────────────────────────────
def main():
    print("=" * 46)
    print("  Chest X-Ray Pneumonia — Dataset Pipeline")
    print("=" * 46)

    # ── Step 1: collect all images ────────────
    print(f"\n[1/4] Scanning dataset root:\n      {DATASET_ROOT}\n")
    all_paths, all_labels = collect_all_samples(DATASET_ROOT)

    total = len(all_paths)
    if total == 0:
        raise FileNotFoundError(
            f"No images found under {DATASET_ROOT}. "
            "Check that DATASET_ROOT points to the folder containing "
            "train/, val/, test/ sub-directories."
        )

    counts = class_counts(all_labels)
    print(f"  Total images found : {total:,}")
    for cls_name, cnt in counts.items():
        print(f"    {cls_name:<11}: {cnt:,}  ({cnt/total*100:.1f}%)")

    # ── Step 2: stratified split ──────────────
    print(f"\n[2/4] Splitting  →  {TRAIN_RATIO*100:.0f}% / "
          f"{VAL_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}% (stratified)")

    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(all_paths, all_labels)

    # ── Step 3 & 4: build DataLoaders ─────────
    print("\n[3/4] Building Datasets & DataLoaders …")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
    )
    print("      Done.")

    # ── Step 5: report ────────────────────────
    print("\n[4/4] Dataset Statistics")

    print_split_info("TRAIN",      train_paths, train_labels, train_loader)
    print_split_info("VALIDATION", val_paths,   val_labels,   val_loader)
    print_split_info("TEST",       test_paths,  test_labels,  test_loader)

    # ── Sanity-check one batch ────────────────
    print("\n[Sanity Check] Fetching one batch from train_loader …")
    images, labels_batch = next(iter(train_loader))
    print(f"  images tensor shape : {tuple(images.shape)}")
    print(f"  images dtype        : {images.dtype}")
    print(f"  pixel value range   : [{images.min():.4f}, {images.max():.4f}]")
    print(f"  labels tensor shape : {tuple(labels_batch.shape)}")
    print(f"  unique labels       : {labels_batch.unique().tolist()}")
    print("\n✓ Pipeline ready. DataLoaders are accessible as:")
    print("    train_loader  — shuffled, batch=32")
    print("    val_loader    — ordered,  batch=32")
    print("    test_loader   — ordered,  batch=32")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = main()
