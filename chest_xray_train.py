"""
Chest X-Ray Pneumonia Classifier — Training Script
===================================================
Integrates:
  • chest_xray_pipeline.py  → DataLoaders
  • chest_xray_model.py     → ChestXRayClassifier

Training configuration
----------------------
  Optimizer      : Adam (lr = 0.0001)
  Loss           : CrossEntropyLoss  (inside model)
  Batch size     : 32
  Epochs         : 15
  Early stopping : patience = 5 epochs (on val loss)
  Best weights   : saved to  best_model.pth
"""

import os
import time
import copy
import pathlib

import torch
import torch.nn as nn

# ── Import our modules ────────────────────────────────────────────────────────
from chest_xray_pipeline import (
    DATASET_ROOT,
    BATCH_SIZE,
    collect_all_samples,
    stratified_split,
    build_dataloaders,
)
from chest_xray_model import build_model

# ─────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────
LEARNING_RATE    = 1e-4        # 0.0001
NUM_EPOCHS       = 15
EARLY_STOP_PAT   = 5           # stop if val loss doesn't improve for 5 epochs
SAVE_PATH        = pathlib.Path(
    r"C:\Users\abhay\OneDrive\Documents\DATA REFINEMENT RESEARCH\best_model.pth"
)


# ─────────────────────────────────────────────
# 1.  Logging helpers
# ─────────────────────────────────────────────
class EpochLogger:
    """
    Keeps a running record of per-epoch metrics and pretty-prints them.
    """

    HEADER = (
        f"{'Epoch':>6} │ "
        f"{'Train Loss':>10} │ {'Train Acc':>9} │ "
        f"{'Val Loss':>8} │ {'Val Acc':>8} │ "
        f"{'Time(s)':>7}"
    )
    SEP = "─" * len(HEADER)

    def __init__(self):
        self.history = []          # list of dicts, one per epoch

    def print_header(self):
        print("\n" + self.SEP)
        print(self.HEADER)
        print(self.SEP)

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, elapsed):
        self.history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                elapsed=elapsed,
            )
        )
        marker = " ✓" if val_loss == min(h["val_loss"] for h in self.history) else ""
        print(
            f"{epoch:>6} │ "
            f"{train_loss:>10.4f} │ {train_acc:>8.2f}% │ "
            f"{val_loss:>8.4f} │ {val_acc:>7.2f}% │ "
            f"{elapsed:>6.1f}s"
            f"{marker}"
        )

    def print_footer(self):
        print(self.SEP)

    def best_epoch(self):
        return min(self.history, key=lambda h: h["val_loss"])


# ─────────────────────────────────────────────
# 2.  One training epoch
# ─────────────────────────────────────────────
def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns
    -------
    avg_loss : float  — mean CrossEntropyLoss over all batches
    accuracy : float  — percentage of correct predictions
    """
    model.train()
    running_loss  = 0.0
    correct       = 0
    total         = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)                      # forward
        loss   = model.compute_loss(logits, labels) # CrossEntropyLoss
        loss.backward()                             # backward
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total * 100.0
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# 3.  One validation epoch
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(
    model:  nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation set without gradient tracking.

    Returns
    -------
    avg_loss : float
    accuracy : float  — percentage
    """
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = model.compute_loss(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total * 100.0
    return avg_loss, accuracy


# ─────────────────────────────────────────────
# 4.  Early-stopping tracker
# ─────────────────────────────────────────────
class EarlyStopping:
    """
    Stops training when validation loss has not improved for `patience` epochs.
    Also saves the best model weights to disk.

    Parameters
    ----------
    patience  : int   — number of epochs to wait after last improvement
    save_path : Path  — where to write best_model.pth
    delta     : float — minimum improvement to count as 'better'
    """

    def __init__(
        self,
        patience:  int            = EARLY_STOP_PAT,
        save_path: pathlib.Path   = SAVE_PATH,
        delta:     float          = 1e-5,
    ):
        self.patience  = patience
        self.save_path = save_path
        self.delta     = delta

        self.best_loss   = float("inf")
        self.best_state  = None
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module):
        """Call at the end of every epoch."""
        if val_loss < self.best_loss - self.delta:
            # Improvement — save weights and reset counter
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self._save()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def _save(self):
        torch.save(self.best_state, self.save_path)

    def restore_best(self, model: nn.Module):
        """Load the best weights back into the model after training."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────
# 5.  Main training routine
# ─────────────────────────────────────────────
def train(
    num_epochs:    int   = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size:    int   = BATCH_SIZE,
):
    """
    End-to-end training procedure.

    Flow
    ----
    1. Load & split dataset
    2. Build model + move to device
    3. Adam optimizer
    4. Epoch loop → train → validate → log → early-stop check
    5. Restore best weights after training
    """

    print("=" * 58)
    print("  Chest X-Ray Classifier — Training")
    print("=" * 58)
    print(f"\n  Optimizer      : Adam")
    print(f"  Learning rate  : {learning_rate}")
    print(f"  Batch size     : {batch_size}")
    print(f"  Max epochs     : {num_epochs}")
    print(f"  Early stop     : patience = {EARLY_STOP_PAT} epochs")
    print(f"  Save path      : {SAVE_PATH}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    all_paths, all_labels = collect_all_samples(DATASET_ROOT)

    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = stratified_split(all_paths, all_labels)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels,
        batch_size=batch_size,
    )
    print(f"  Train : {len(train_paths):,}  |  "
          f"Val : {len(val_paths):,}  |  "
          f"Test : {len(test_paths):,}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building model …")
    model, device = build_model(pretrained=True)
    print(f"  Device         : {device}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── Early stopping ────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=EARLY_STOP_PAT, save_path=SAVE_PATH)

    # ── Epoch loop ────────────────────────────────────────────────────────
    print("\n[3/4] Training …")
    logger = EpochLogger()
    logger.print_header()

    total_start = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, device)

        elapsed = time.time() - t0
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, elapsed)

        # Early-stop check (also saves best weights internally)
        early_stop.step(val_loss, model)
        if early_stop.should_stop:
            print(f"\n  ⚡ Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PAT} epochs).")
            break

    logger.print_footer()

    total_time = time.time() - total_start
    print(f"\n  Total training time : {total_time:.1f}s")

    # ── Restore best weights ──────────────────────────────────────────────
    early_stop.restore_best(model)
    best = logger.best_epoch()
    print(f"\n  Best epoch  : {best['epoch']}")
    print(f"  Best val loss : {best['val_loss']:.4f}")
    print(f"  Best val acc  : {best['val_acc']:.2f}%")
    print(f"\n  Weights saved to : {SAVE_PATH}")

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n[4/4] Evaluating on held-out test set …")
    test_loss, test_acc = validate(model, test_loader, device)
    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  Test Loss : {test_loss:>8.4f}        │")
    print(f"  │  Test Acc  : {test_acc:>7.2f}%        │")
    print(f"  └─────────────────────────────┘")

    return model, logger


# ─────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    trained_model, history = train(
        num_epochs    = NUM_EPOCHS,
        learning_rate = LEARNING_RATE,
        batch_size    = BATCH_SIZE,
    )
