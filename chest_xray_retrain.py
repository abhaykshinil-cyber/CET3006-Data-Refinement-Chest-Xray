"""
Retrain on Cleaned Dataset — Chest X-Ray Pneumonia Classifier
==============================================================
Retrains ResNet18 from scratch on the cleaned training set produced
by chest_xray_cleaning.py, using identical hyperparameters to the
original training run.

Architecture   : ResNet18 (ImageNet pretrained) + Dropout(0.5) + Linear(512→2)
Optimizer      : Adam  (lr = 0.0001)
Loss           : CrossEntropyLoss
Batch size     : 32
Max epochs     : 15
Early stopping : patience = 5  (on validation loss)
Best weights   : saved to  best_model_cleaned.pth

Outputs
-------
  • Per-epoch training log (train loss, train acc, val loss, val acc, time)
  • best_model_cleaned.pth  — best weights from the cleaned-data run
  • Summary comparison:  original run (noisy) vs retrained run (clean)
"""

import time
import copy
import pathlib

import torch
import torch.nn as nn

# ── Our modules ───────────────────────────────────────────────────────────────
from chest_xray_model    import build_model
from chest_xray_cleaning import run_dataset_cleaning
from chest_xray_train    import (
    train_one_epoch,
    validate,
    EpochLogger,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOP_PAT,
    SAVE_PATH         as ORIGINAL_SAVE_PATH,
)
from chest_xray_error_detection import MC_PASSES
from chest_xray_noise           import NOISE_RATE

# ─────────────────────────────────────────────
# Configuration  (mirrors original training)
# ─────────────────────────────────────────────
BATCH_SIZE       = 32
CLEANED_SAVE_PATH = pathlib.Path(
    r"C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH\best_model_cleaned.pth"
)


# ─────────────────────────────────────────────
# Early stopping (same as original)
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    Identical to the one in chest_xray_train.py but saves to a
    configurable path so we don't overwrite the original best_model.pth.
    """

    def __init__(
        self,
        patience:  int           = EARLY_STOP_PAT,
        save_path: pathlib.Path  = CLEANED_SAVE_PATH,
        delta:     float         = 1e-5,
    ):
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
# Retrain function
# ─────────────────────────────────────────────
def retrain_on_cleaned_data(
    num_epochs:    int   = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size:    int   = BATCH_SIZE,
    noise_rate:    float = NOISE_RATE,
    n_mc_passes:   int   = MC_PASSES,
) -> tuple[nn.Module, EpochLogger]:
    """
    Full retrain pipeline on the cleaned dataset.

    Steps
    -----
    1.  Run dataset cleaning  (noise injection → MC Dropout → filtering)
    2.  Build fresh ResNet18 classifier with the same architecture
    3.  Train for up to `num_epochs` with early stopping
    4.  Save best weights to  best_model_cleaned.pth
    5.  Return trained model + epoch history

    Parameters
    ----------
    num_epochs    : max training epochs   (default 15)
    learning_rate : Adam learning rate    (default 0.0001)
    batch_size    : DataLoader batch size (default 32)
    noise_rate    : training  noise ratio used in cleaning step
    n_mc_passes   : MC Dropout passes during error detection

    Returns
    -------
    model  : best-weight-restored model, in eval mode
    logger : EpochLogger with full per-epoch history
    """

    print("=" * 64)
    print("  Retraining on Cleaned Dataset — Chest X-Ray Classifier")
    print("=" * 64)
    print(f"\n  Architecture   : ResNet18  (ImageNet pretrained)")
    print(f"  Optimizer      : Adam")
    print(f"  Learning rate  : {learning_rate}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Max epochs     : {num_epochs}")
    print(f"  Early stopping : patience = {EARLY_STOP_PAT} epochs (val loss)")
    print(f"  Save path      : {CLEANED_SAVE_PATH}")

    # ── Step 1: Dataset cleaning ──────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  STEP 1 / 3  — Dataset Cleaning")
    print("─" * 64)

    clean_train_loader, val_loader, test_loader, removed_indices = run_dataset_cleaning(
        noise_rate = noise_rate,
        n_passes   = n_mc_passes,
        batch_size = batch_size,
    )

    n_clean  = len(clean_train_loader.dataset)
    n_val    = len(val_loader.dataset)
    n_test   = len(test_loader.dataset)

    print(f"\n  Cleaned train set : {n_clean:,} samples  "
          f"({len(removed_indices):,} removed)")
    print(f"  Val set           : {n_val:,} samples  (unchanged)")
    print(f"  Test set          : {n_test:,} samples  (unchanged)")

    # ── Step 2: Build model ───────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  STEP 2 / 3  — Model Initialisation")
    print("─" * 64)

    model, device = build_model(pretrained=True)
    print(f"\n  Device : {device}")
    print(f"  Parameters : "
          f"{sum(p.numel() for p in model.parameters()):,}  total  |  "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}  trainable")

    optimizer  = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stop = EarlyStopping(patience=EARLY_STOP_PAT, save_path=CLEANED_SAVE_PATH)

    # ── Step 3: Training loop ─────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  STEP 3 / 3  — Training Loop")
    print("─" * 64)

    logger      = EpochLogger()
    total_start = time.time()

    logger.print_header()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, clean_train_loader, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, device)

        elapsed = time.time() - t0
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, elapsed)

        early_stop.step(val_loss, model)
        if early_stop.should_stop:
            print(f"\n  ⚡ Early stopping at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PAT} epochs).")
            break

    logger.print_footer()

    total_elapsed = time.time() - total_start
    print(f"\n  Total training time : {total_elapsed:.1f}s  "
          f"({total_elapsed/60:.1f} min)")

    # ── Restore best weights ──────────────────────────────────────────────
    early_stop.restore_best(model)
    model.eval()

    best = logger.best_epoch()
    print(f"\n  Best epoch     : {best['epoch']}")
    print(f"  Best val loss  : {best['val_loss']:.4f}")
    print(f"  Best val acc   : {best['val_acc']:.2f}%")
    print(f"\n  ✓ Best weights saved → {CLEANED_SAVE_PATH}")

    return model, logger, test_loader


# ─────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────
def print_comparison(
    noisy_logger:  EpochLogger | None,
    clean_logger:  EpochLogger,
    test_loader,
    clean_model:   nn.Module,
    device:        torch.device,
):
    """
    Side-by-side comparison of key metrics:
      Original model (trained on noisy data) vs
      Retrained model (trained on cleaned data)

    The original model metrics come from the EpochLogger returned
    by chest_xray_train.train() if it was run in the same session.
    If not available, only the cleaned model metrics are shown.
    """
    test_loss, test_acc = validate(clean_model, test_loader, device)

    print()
    print("═" * 64)
    print("  FINAL RESULTS — Comparison Summary")
    print("═" * 64)

    # Header
    print(f"\n  {'Metric':<26} {'Noisy Model':>14} {'Cleaned Model':>14}")
    print("  " + "─" * 56)

    clean_best = clean_logger.best_epoch()

    if noisy_logger is not None:
        noisy_best = noisy_logger.best_epoch()
        rows = [
            ("Best epoch",       f"{noisy_best['epoch']}",
                                 f"{clean_best['epoch']}"),
            ("Best val loss",    f"{noisy_best['val_loss']:.4f}",
                                 f"{clean_best['val_loss']:.4f}"),
            ("Best val acc",     f"{noisy_best['val_acc']:.2f}%",
                                 f"{clean_best['val_acc']:.2f}%"),
            ("Epochs run",       f"{len(noisy_logger.history)}",
                                 f"{len(clean_logger.history)}"),
        ]
        for label, noisy_val, clean_val in rows:
            print(f"  {label:<26} {noisy_val:>14} {clean_val:>14}")
    else:
        rows = [
            ("Best epoch",   f"{clean_best['epoch']}"),
            ("Best val loss",f"{clean_best['val_loss']:.4f}"),
            ("Best val acc", f"{clean_best['val_acc']:.2f}%"),
            ("Epochs run",   f"{len(clean_logger.history)}"),
        ]
        for label, val in rows:
            print(f"  {label:<26} {'N/A':>14} {val:>14}")

    print("  " + "─" * 56)
    print(f"\n  Test set evaluation (cleaned model):")
    print(f"  ┌────────────────────────────────┐")
    print(f"  │  Test Loss : {test_loss:>8.4f}           │")
    print(f"  │  Test Acc  : {test_acc:>7.2f}%           │")
    print(f"  └────────────────────────────────┘")
    print()
    print(f"  Model saved to : {CLEANED_SAVE_PATH}")
    print()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Retrain
    clean_model, clean_logger, test_loader = retrain_on_cleaned_data(
        num_epochs    = NUM_EPOCHS,
        learning_rate = LEARNING_RATE,
        batch_size    = BATCH_SIZE,
        noise_rate    = NOISE_RATE,
        n_mc_passes   = MC_PASSES,
    )

    # Comparison report (no noisy logger since we're not rerunning original)
    device = next(clean_model.parameters()).device
    print_comparison(
        noisy_logger = None,
        clean_logger = clean_logger,
        test_loader  = test_loader,
        clean_model  = clean_model,
        device       = device,
    )
