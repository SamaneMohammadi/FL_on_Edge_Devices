"""
Base client shared by FedAvg and FedAsync.

It owns everything that does not depend on the aggregation mode:
  - load this client's data and build the model
  - run E local epochs of DP-SGD (clip to C, add N(0, sigma^2 C^2 I))
  - update the per-client Moments Accountant once per round and report epsilon
  - measure CPU time / RAM so we can reproduce Table II
  - read/write parameters in the numpy format Flower expects

The FedAvg and FedAsync clients subclass this and only change what extra
information they report to the server (FedAsync also reports its local round so
the server can compute staleness).
"""

from collections import OrderedDict

import numpy as np
import torch
import flwr as fl

from model import build_model
from dataset import get_loaders
from utils.privacy.moments_accountant import MomentsAccountant
from utils.resource_monitor import ResourceMonitor
from utils.metrics import evaluate, CSVLogger
import config as cfg


# Per-client accountants are kept here so cumulative epsilon survives across
# rounds. On the real testbed each client is its own long-lived process, so this
# just holds a single entry. In the in-process simulation Flower recreates the
# client object every round, and this registry is what lets privacy loss keep
# accumulating instead of resetting each round.
_ACCOUNTANTS = {}


def _get_accountant(client_id, sigma, delta):
    key = (client_id, sigma)
    if key not in _ACCOUNTANTS:
        _ACCOUNTANTS[key] = MomentsAccountant(sigma=sigma, delta=delta)
    return _ACCOUNTANTS[key]


def get_parameters(model):
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, parameters):
    keys = list(model.state_dict().keys())
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state, strict=True)


class BaseSERClient(fl.client.NumPyClient):
    def __init__(self, client_id, data_dir, sigma, device="cpu", log_dir="logs"):
        self.client_id = client_id
        self.sigma = sigma
        self.device = device

        self.train_loader, self.test_loader, self.n_train = get_loaders(
            client_id, data_dir=data_dir, batch_size=cfg.BATCH_SIZE
        )
        self.model = build_model(
            in_channels=128, num_classes=cfg.NUM_CLASSES, device=device
        )

        # sampling rate q = B / |D_k|, used by the accountant
        self.q = cfg.BATCH_SIZE / self.n_train
        self.accountant = _get_accountant(client_id, sigma, cfg.DELTA)

        self.monitor = ResourceMonitor()
        self.logger = CSVLogger(
            f"{log_dir}/client_{client_id}.csv",
            fieldnames=[
                "round", "dp_steps", "wall_time_s", "cpu_user_s", "cpu_system_s",
                "ram_rss_mb", "ram_percent", "train_acc", "test_acc",
                "test_loss", "epsilon",
            ],
        )

    # --- Flower API ----------------------------------------------------------
    def get_parameters(self, config):
        return get_parameters(self.model)

    def _train_one_round(self):
        """Run E local epochs of DP-SGD and return train accuracy."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.LEARNING_RATE)
        criterion = torch.nn.CrossEntropyLoss()

        model, opt, loader = self.model, optimizer, self.train_loader
        engine = None

        if self.sigma > 0:
            # attach Opacus only when we actually want privacy
            from utils.privacy.dp_engine import make_private
            model, opt, loader, engine = make_private(
                self.model, optimizer, self.train_loader,
                sigma=self.sigma, clip_norm=cfg.CLIP_NORM,
            )

        model.train()
        correct, total, n_steps = 0, 0, 0
        for _ in range(cfg.LOCAL_EPOCHS):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                n_steps += 1                       # one DP-SGD (sub-sampled Gaussian) step
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)

        # if we wrapped with Opacus, copy the trained weights back into self.model
        if engine is not None:
            self.model.load_state_dict(model._module.state_dict())

        return correct / max(total, 1), n_steps

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        rnd = config.get("server_round", 0)

        self.monitor.snapshot_start()
        train_acc, n_steps = self._train_one_round()
        res = self.monitor.snapshot_end()

        # Per Algorithm 1 the round's log-moment composes EVERY DP-SGD step taken
        # this round, each a sub-sampled Gaussian query at rate q = B/|D_k|.
        # With E=1 local epoch and Poisson sampling that is ~|D_k|/B steps, not 1.
        # Accounting only one step here would under-state the true privacy loss.
        # (PER_ROUND_ACCOUNTING=True reverts to one query/round to match the
        # paper's published table - see config/settings.py.)
        accounted_steps = 1 if cfg.PER_ROUND_ACCOUNTING else n_steps
        self.accountant.step(q=self.q, num_steps=accounted_steps)
        epsilon = self.accountant.get_epsilon()

        test_loss, test_acc = evaluate(self.model, self.test_loader, self.device)

        self.logger.log({
            "round": rnd,
            "dp_steps": n_steps,
            "wall_time_s": round(res["wall_time_s"], 3),
            "cpu_user_s": round(res["cpu_user_s"], 3),
            "cpu_system_s": round(res["cpu_system_s"], 3),
            "ram_rss_mb": round(res["ram_rss_mb"], 1),
            "ram_percent": round(res["ram_percent"], 1),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "test_loss": round(test_loss, 4),
            "epsilon": round(epsilon, 4),
        })

        metrics = {
            "client_id": self.client_id,
            "accuracy": float(test_acc),
            "epsilon": float(epsilon),
            "wall_time_s": float(res["wall_time_s"]),
        }
        return get_parameters(self.model), self.n_train, metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(acc)}
