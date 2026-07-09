import argparse
import flwr as fl

from client.base_client import BaseSERClient


class FedAvgClient(BaseSERClient):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="DP noise multiplier; 0 disables DP")
    parser.add_argument("--data_dir", default="client_data")
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    client = FedAvgClient(
        client_id=args.client_id,
        data_dir=args.data_dir,
        sigma=args.sigma,
        device=args.device,
    )
    fl.client.start_client(server_address=args.server, client=client.to_client())
