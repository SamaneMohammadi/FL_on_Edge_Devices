"""
Small helpers for evaluation and for writing the per-round CSV logs that the
plotting scripts read.
"""

import os
import csv
import torch


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (loss, accuracy) on a loader using cross-entropy."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


class CSVLogger:
    """Append-only per-client CSV logger."""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
