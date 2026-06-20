#!/usr/bin/env bash
# Start a FedAvg run. Launch the server on the machine that has SERVER_IP,
# then run one client per device with the matching --client_id.
#
# Example (server machine):
#   ./scripts/run_fedavg.sh server
# Example (device i):
#   ./scripts/run_fedavg.sh client 1 1.0 192.168.1.18:8080

set -e
cd "$(dirname "$0")/../src"
ROLE=$1

if [ "$ROLE" = "server" ]; then
    ROUNDS=${2:-60}
    python -m server.fedavg_server --rounds "$ROUNDS"
elif [ "$ROLE" = "client" ]; then
    CID=$2
    SIGMA=${3:-1.0}
    SERVER=${4:-127.0.0.1:8080}
    python -m client.fedavg_client --client_id "$CID" --sigma "$SIGMA" --server "$SERVER"
else
    echo "usage: $0 [server ROUNDS | client CLIENT_ID SIGMA SERVER_ADDR]"
    exit 1
fi
