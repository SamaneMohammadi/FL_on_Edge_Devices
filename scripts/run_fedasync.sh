#!/usr/bin/env bash
# Start a FedAsync run. Same idea as run_fedavg.sh but the server takes an
# --alpha base decay factor (paper sweeps 0.2, 0.4, 0.6).
#
# Example (server machine):
#   ./scripts/run_fedasync.sh server 150 0.2
# Example (device i):
#   ./scripts/run_fedasync.sh client 1 1.0 192.168.1.18:8080

set -e
cd "$(dirname "$0")/../src"
ROLE=$1

if [ "$ROLE" = "server" ]; then
    ROUNDS=${2:-150}
    ALPHA=${3:-0.2}
    python -m server.fedasync_server --rounds "$ROUNDS" --alpha "$ALPHA"
elif [ "$ROLE" = "client" ]; then
    CID=$2
    SIGMA=${3:-1.0}
    SERVER=${4:-127.0.0.1:8080}
    python -m client.fedasync_client --client_id "$CID" --sigma "$SIGMA" --server "$SERVER"
else
    echo "usage: $0 [server ROUNDS ALPHA | client CLIENT_ID SIGMA SERVER_ADDR]"
    exit 1
fi
