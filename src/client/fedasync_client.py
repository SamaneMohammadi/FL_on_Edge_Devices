"""
Asynchronous (FedAsync) client.

The only difference from the FedAvg client is that this one reports the server
round it *started* training on. The server uses that to compute staleness
tau_k = t - t_k for the staleness-aware update

    alpha_k = alpha / (1 + tau_k)
    W_G <- (1 - alpha_k) W_G + alpha_k W_k

On the physical testbed the asynchrony comes for free: slow devices (HW T1/T2)
finish several rounds later than fast ones (HW T4/T5), so their updates arrive
stale. We surface the round they trained on so the server can weight them.

Run one per device:
    python -m client.fedasync_client --client_id 1 --sigma 1.0 --server 192.168.1.18:8080
"""

import argparse
import flwr as fl

from client.base_client import BaseSERClient, get_parameters


class FedAsyncClient(BaseSERClient):
    def fit(self, parameters, config):
        # remember which global round this update is based on
        base_round = config.get("server_round", 0)
        new_params, n, metrics = super().fit(parameters, config)
        metrics["base_round"] = int(base_round)  # server reads this for staleness
        return new_params, n, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="DP noise multiplier; 0 disables DP")
    parser.add_argument("--data_dir", default="client_data")
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    client = FedAsyncClient(
        client_id=args.client_id,
        data_dir=args.data_dir,
        sigma=args.sigma,
        device=args.device,
    )
    fl.client.start_client(server_address=args.server, client=client.to_client())
