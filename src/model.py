"""
Lightweight 1D CNN for speech emotion recognition.

This is the architecture from the paper (Section III-A): two 1D conv blocks
(64 then 128 filters, kernel 5) each with normalization + ReLU + max-pool +
dropout, followed by a dense layer and the classifier.

Note on normalization: the paper uses Group Normalization, NOT batch norm.
That is deliberate - DP-SGD needs per-sample gradients, and batch norm couples
samples within a batch, so Opacus rejects it. GroupNorm keeps the model
DP-compatible while behaving similarly.
"""

import torch
import torch.nn as nn


class SERNet(nn.Module):
    def __init__(self, in_channels=128, num_classes=4, n_groups=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.GroupNorm(n_groups, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(n_groups, 128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),
        )

        # the flattened size depends on the input length, so we work it out once
        # with a dummy forward pass instead of hard-coding it.
        self.classifier = None
        self._num_classes = num_classes

    def _build_classifier(self, flat_dim):
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self._num_classes),
        )

    def forward(self, x):
        # x: (batch, mel_bins=128, time_frames)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        if self.classifier is None:
            self._build_classifier(x.shape[1])
            self.classifier = self.classifier.to(x.device)
        return self.classifier(x)


def build_model(in_channels=128, num_classes=4, sample_length=108, device="cpu"):
    """Create the model and force the lazy classifier to materialize."""
    model = SERNet(in_channels=in_channels, num_classes=num_classes).to(device)
    dummy = torch.zeros(1, in_channels, sample_length, device=device)
    model(dummy)  # triggers _build_classifier
    return model
