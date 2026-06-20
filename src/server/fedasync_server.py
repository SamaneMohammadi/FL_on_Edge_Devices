"""
Asynchronous FedAsync server with staleness-aware aggregation.

For every update the server applies, as it arrives, the staleness-weighted rule
from Xie et al. / the paper (Section III, Algorithm 1):

    tau_k    = t - t_k                 (how many rounds stale this update is)
    alpha_k  = alpha / (1 + tau_k)     (older updates count less)
    W_G      <- (1 - alpha_k) * W_G + alpha_k * W_k

`alpha` is the base decay factor, swept over {0.2, 0.4, 0.6} in the paper. The
update rule above is implemented exactly; the only thing the surrounding harness
has to get right is producing real staleness `tau_k > 0`.

HOW STALENESS ARISES (important):
Stock Flower is round-synchronous: by default the server waits for every
selected client each round, so every client trains on the current global model,
reports base_round == server_round, and tau is always 0 - even on heterogeneous
hardware. That defeats the point. To recover true asynchrony we run the server
straggler-tolerant: it advances a round as soon as `min_fit_clients` updates are
in (or `round_timeout` elapses), without blocking on the slow devices. Fast
devices (HW T4/T5) then complete many more rounds than slow ones (HW T1/T2);
when a slow device finally returns, the global round has moved on, so its
reported base_round lags and tau_k = t - t_k > 0. This reproduces both the
participation imbalance and the staleness profile (tau ~ 4, 6, 7 for the slower
tiers) reported in the paper.

Each client reports the global round it trained on (`base_round`); the server
computes tau_k = server_round - base_round from that.

    python -m server.fedasync_server --rounds 150 --alpha 0.2 \
        --min_fit_clients 1 --round_timeout 30
"""

import os
import json
import time
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import flwr as fl
from flwr.common import (
    Metrics, Parameters, Scalar, FitRes,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import config as cfg


def weighted_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    return {"accuracy": sum(n * m["accuracy"] for n, m in metrics) / total}


class FedAsyncServer(FedAvg):
    def __init__(self, alpha=0.2, out_dir="metrics_fedasync", **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.out_dir = out_dir
        self.global_ndarrays = None      # current global model as a list of arrays
        self._round_start = None
        os.makedirs(out_dir, exist_ok=True)

    def initialize_parameters(self, client_manager):
        params = super().initialize_parameters(client_manager)
        if params is not None:
            self.global_ndarrays = parameters_to_ndarrays(params)
        return params

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.time()
        if self.global_ndarrays is None:
            self.global_ndarrays = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        latency = time.time() - self._round_start

        # process updates oldest-first so staleness ordering is sensible
        def staleness(res):
            base = res.metrics.get("base_round", server_round)
            return server_round - int(base)

        ordered = sorted((r for _, r in results), key=staleness, reverse=True)

        per_client = []
        for res in ordered:
            tau = max(0, staleness(res))
            alpha_k = self.alpha / (1.0 + tau)
            client_ndarrays = parameters_to_ndarrays(res.parameters)

            # W_G <- (1 - alpha_k) W_G + alpha_k W_k, layer by layer
            self.global_ndarrays = [
                (1 - alpha_k) * g + alpha_k * c
                for g, c in zip(self.global_ndarrays, client_ndarrays)
            ]

            m = dict(res.metrics)
            m["staleness"] = tau
            m["alpha_k"] = alpha_k
            m["num_examples"] = res.num_examples
            per_client.append(m)

        record = {
            "round": server_round,
            "alpha": self.alpha,
            "round_latency_s": latency,
            "clients": per_client,
        }
        with open(os.path.join(self.out_dir, f"round_{server_round}.json"), "w") as f:
            json.dump(record, f, indent=2)

        new_params = ndarrays_to_parameters(self.global_ndarrays)
        return new_params, {"alpha": self.alpha}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS_FEDASYNC)
    parser.add_argument("--alpha", type=float, default=cfg.DEFAULT_ALPHA,
                        help="base decay factor (paper sweeps 0.2, 0.4, 0.6)")
    parser.add_argument("--min_fit_clients", type=int, default=1,
                        help="advance the round once this many updates arrive; "
                             "keep < NUM_CLIENTS so slow devices fall behind and "
                             "report staleness")
    parser.add_argument("--round_timeout", type=float, default=None,
                        help="seconds to wait for updates before advancing")
    parser.add_argument("--address", default=cfg.SERVER_ADDRESS)
    args = parser.parse_args()

    strategy = FedAsyncServer(
        alpha=args.alpha,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_fit_clients,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
        on_fit_config_fn=lambda r: {"server_round": r},
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(
            num_rounds=args.rounds, round_timeout=args.round_timeout
        ),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
