"""
Synchronous FedAvg server.

Standard dataset-size-weighted aggregation

    p_k = N_k / sum_j N_j ,    W_G = sum_k p_k W_k

(Eq. fedavg in the paper). We wrap Flower's FedAvg only to (a) time each round
and (b) dump per-client accuracy / epsilon to JSON so the analysis scripts have
everything in one place.

    python -m server.fedavg_server --rounds 60
"""

import os
import json
import time
import argparse
from typing import List, Tuple, Optional, Dict

import flwr as fl
from flwr.common import Metrics, Parameters, Scalar, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import config as cfg


def weighted_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    acc = sum(n * m["accuracy"] for n, m in metrics) / total
    return {"accuracy": acc}


class FedAvgServer(FedAvg):
    def __init__(self, out_dir="metrics_fedavg", **kwargs):
        super().__init__(**kwargs)
        self.out_dir = out_dir
        self._round_start = None
        os.makedirs(out_dir, exist_ok=True)

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        latency = time.time() - self._round_start
        params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        per_client = []
        for _, res in results:
            m = dict(res.metrics)
            m["num_examples"] = res.num_examples
            per_client.append(m)

        record = {
            "round": server_round,
            "round_latency_s": latency,
            "clients": per_client,
        }
        with open(os.path.join(self.out_dir, f"round_{server_round}.json"), "w") as f:
            json.dump(record, f, indent=2)

        return params, agg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS_FEDAVG)
    parser.add_argument("--address", default=cfg.SERVER_ADDRESS)
    args = parser.parse_args()

    strategy = FedAvgServer(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg.NUM_CLIENTS,
        min_available_clients=cfg.NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
        on_fit_config_fn=lambda r: {"server_round": r},
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
