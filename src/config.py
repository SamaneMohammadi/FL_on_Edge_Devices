"""
All the experiment settings live here so the client, server and the privacy
accountant read the exact same numbers. Values follow the paper
(Section IV - Experimental Design, "Parameter Settings").
"""

# --- Federated setup ---------------------------------------------------------
NUM_CLIENTS = 5                 # HW T1..T5, one client per physical device
LOCAL_EPOCHS = 1                # E = 1 local epoch per round
BATCH_SIZE = 128                # B = 128
LEARNING_RATE = 1e-3            # eta, Adam
NUM_ROUNDS_FEDAVG = 60          # FedAvg converges in ~60 rounds
NUM_ROUNDS_FEDASYNC = 150       # async keeps streaming updates; we cap it

# --- Model / task ------------------------------------------------------------
NUM_CLASSES = 4                 # Neutral, Happy, Angry, Sad
EMOTIONS = ["ANG", "HAP", "NEU", "SAD"]   # kept sorted so the label encoding is stable

# --- Differential privacy (DP-SGD, Section III) ------------------------------
CLIP_NORM = 1.0                 # C = 1, per-sample gradient clipping
DELTA = 1e-5                    # failure probability delta
# noise multipliers swept in the paper; pick one per run with --sigma
NOISE_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]
DEFAULT_SIGMA = 1.0

# Privacy-accounting granularity.
#   False (default): account EVERY DP-SGD mini-batch step taken in a round.
#       This is the correct composition for E>=1 local epochs - one local epoch
#       over |D_k|=941 with B=128 is ~8 sub-sampled Gaussian steps, so a round
#       composes ~8 queries, not 1. Cumulative epsilon is larger and truthful.
#   True: account exactly ONE query per round at rate q. This reproduces the
#       coarser per-round convention behind the paper's Table values, but it
#       under-states the privacy loss of multi-batch local training. Use it only
#       to match the published numbers, not for a real privacy guarantee.
PER_ROUND_ACCOUNTING = False

# --- FedAsync staleness (Section III, eq. fedasync) --------------------------
# alpha_k = alpha / (1 + tau_k);  alpha swept over {0.2, 0.4, 0.6}
FEDASYNC_ALPHAS = [0.2, 0.4, 0.6]
DEFAULT_ALPHA = 0.2

# --- Reproducibility ---------------------------------------------------------
# paper averages results over 10 random seeds
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DEFAULT_SEED = 42

# --- Networking --------------------------------------------------------------
SERVER_ADDRESS = "0.0.0.0:8080"
