"""
train.py - Training loop, validation, early stopping, and epoch logging.
"""

from __future__ import annotations

import copy
import json
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import EARLY_STOP_DELTA, EARLY_STOP_PAT, LEARNING_RATE, NUM_EPOCHS


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = model.compute_loss(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total * 100.0


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = model.compute_loss(logits, labels)
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, correct / total * 100.0


class EarlyStopping:
    def __init__(self, patience, save_path, delta=EARLY_STOP_DELTA):
        self.patience = patience
        self.save_path = save_path
        self.delta = delta
        self.best_loss = float("inf")
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_state, self.save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class EpochLogger:
    HEADER = (
        f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>8} | {'Time(s)':>7}"
    )
    SEP = "-" * len(HEADER)

    def __init__(self):
        self.history = []

    def print_header(self):
        print("\n" + self.SEP)
        print(self.HEADER)
        print(self.SEP)

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, elapsed):
        self.history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "elapsed": elapsed,
        })
        best_marker = " *" if val_loss == min(e["val_loss"] for e in self.history) else ""
        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | "
            f"{val_loss:>8.4f} | {val_acc:>7.2f}% | {elapsed:>6.1f}s{best_marker}"
        )

    def print_footer(self):
        print(self.SEP)

    def best_epoch(self):
        return min(self.history, key=lambda e: e["val_loss"])

    def save(self, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)


def train_one_epoch_weighted(model, loader, optimizer, device):
    """
    Weighted training loop. Loader yields 5-tuples: (images, labels, ids, paths, weights).
    Each sample loss is multiplied by its pre-computed weight before averaging.
    This keeps all samples in training while downweighting likely mislabelled ones.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels, _ids, _paths, weights in loader:
        images  = images.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss   = model.compute_loss_weighted(logits, labels, weights)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += images.size(0)
    return running_loss / total, correct / total * 100.0


def _run_loop(model, train_fn, train_loader, val_loader, device,
              save_path, history_path, num_epochs, learning_rate, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    stopper   = EarlyStopping(patience=patience, save_path=save_path)
    logger    = EpochLogger()
    t_start   = time.time()
    logger.print_header()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_fn(model, train_loader, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, device)
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, time.time() - t0)
        stopper.step(val_loss, model)
        if stopper.should_stop:
            print(f"\n  Early stopping at epoch {epoch}.")
            break
    logger.print_footer()
    stopper.restore_best(model)
    model.eval()
    logger.save(history_path)
    elapsed = time.time() - t_start
    best = logger.best_epoch()
    print(f"\n  Training complete ({elapsed:.0f}s)")
    print(f"  Best epoch : {best['epoch']}  val_loss={best['val_loss']:.4f}  val_acc={best['val_acc']:.2f}%")
    print(f"  Saved weights  -> {save_path}")
    print(f"  Saved history  -> {history_path}")
    return logger


def run_training(model, train_loader, val_loader, device, save_path, history_path,
                 num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, patience=EARLY_STOP_PAT):
    return _run_loop(model, train_one_epoch, train_loader, val_loader, device,
                     save_path, history_path, num_epochs, learning_rate, patience)


def run_training_weighted(model, train_loader, val_loader, device, save_path, history_path,
                          num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, patience=EARLY_STOP_PAT):
    """Same as run_training but uses train_one_epoch_weighted. Val loader yields standard 4-tuples."""
    return _run_loop(model, train_one_epoch_weighted, train_loader, val_loader, device,
                     save_path, history_path, num_epochs, learning_rate, patience)
