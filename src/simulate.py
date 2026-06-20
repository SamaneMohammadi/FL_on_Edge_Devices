"""
Run the whole thing on a single machine with Flower's simulation engine.

This is the easiest way to reproduce the experiments without the physical
testbed: it spins up all five clients in-process and runs either FedAvg or
FedAsync. On real hardware you would instead start server/ and client/ on
separate devices (see the README), which is what produces the heterogeneity
and staleness effects the paper studies.

    python simulate.py --mode fedavg   --sigma 1.0 --rounds 60
    python simulate.py --mode fedasync --sigma 1.0 --rounds 150 --alpha 0.2
"""

import argparse

import flwr as fl

from client.fedavg_client import FedAvgClient
from client.fedasync_client import FedAsyncClient
from server.fedavg_server import FedAvgServer, weighted_accuracy
from server.fedasync_server import FedAsyncServer
import config as cfg
from utils.seed import set_seed


def make_client_fn(mode, sigma, data_dir):
    Client = FedAvgClient if mode == "fedavg" else FedAsyncClient

    def client_fn(cid):
        # Flower passes a string cid in [0, NUM_CLIENTS); our files are 1-indexed
        client_id = int(cid) + 1
        return Client(
            client_id=client_id, data_dir=data_dir, sigma=sigma, device="cpu"
        ).to_client()

    return client_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fedavg", "fedasync"], default="fedavg")
    parser.add_argument("--sigma", type=float, default=cfg.DEFAULT_SIGMA)
    parser.add_argument("--alpha", type=float, default=cfg.DEFAULT_ALPHA)
    parser.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS_FEDAVG)
    parser.add_argument("--data_dir", default="client_data")
    parser.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg.NUM_CLIENTS,
        min_available_clients=cfg.NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
        on_fit_config_fn=lambda r: {"server_round": r},
    )

    if args.mode == "fedavg":
        strategy = FedAvgServer(**common)
    else:
        strategy = FedAsyncServer(alpha=args.alpha, **common)

    fl.simulation.start_simulation(
        client_fn=make_client_fn(args.mode, args.sigma, args.data_dir),
        num_clients=cfg.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
