import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SERDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float()      # (N, 128, 108)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _load(out_dir, client_id, split):
    x = np.load(os.path.join(out_dir, f"client_{client_id}_x_{split}.npy"))
    y = np.load(os.path.join(out_dir, f"client_{client_id}_y_{split}.npy"))
    return x, y


def get_loaders(client_id, data_dir="client_data", batch_size=128):
    """Return (train_loader, test_loader) for one client."""
    x_tr, y_tr = _load(data_dir, client_id, "train")
    x_te, y_te = _load(data_dir, client_id, "test")

    train_loader = DataLoader(
        SERDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        SERDataset(x_te, y_te), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, len(x_tr)
