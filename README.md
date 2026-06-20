# Async-FL Heterogeneity

Implementation of **Empirical Analysis of Asynchronous Federated Learning on
Heterogeneous Devices: Efficiency, Fairness, and Privacy Trade-offs** (IJCNN 2025).

📄 Paper: [arXiv:2505.07041](https://arxiv.org/abs/2505.07041)

This study examines the efficiency–fairness–privacy trade-off of synchronous
versus asynchronous federated learning under realistic device heterogeneity. It
compares synchronous `FedAvg` with staleness-aware asynchronous `FedAsync` across
a testbed of five heterogeneous edge devices, applying Local Differential Privacy
(DP-SGD) on each client and tracking per-client cumulative privacy loss with the
Moments Accountant, using Speech Emotion Recognition (SER) as a privacy-critical
benchmark. The analysis shows that `FedAsync` converges much faster but amplifies
fairness and privacy disparities: high-end devices contribute far more updates
and incur higher privacy loss, while low-end devices suffer accuracy degradation
from infrequent, stale, and noise-perturbed updates.

## Methods

- **`FedAvg`** (synchronous) — dataset-size-weighted aggregation
  `p_k = N_k / Σ_j N_j`, `W_G = Σ_k p_k W_k`.
- **`FedAsync`** (asynchronous, staleness-aware) — for each arriving update,
  `τ_k = t − t_k`, `α_k = α / (1 + τ_k)`, `W_G ← (1 − α_k) W_G + α_k W_k`.
  Straggler-tolerant rounds make fast devices outpace slow ones, producing the
  real staleness and participation imbalance the paper studies.
- **Local Differential Privacy** — per-client DP-SGD (per-sample gradient
  clipping + Gaussian noise via Opacus), with the **Moments Accountant**
  (`ε = min_λ (μ(λ) − log δ)/λ`, moments composing additively) tracking each
  client's cumulative `ε`.
- **Model** — a lightweight 1D CNN for SER: two conv blocks (64, 128 filters,
  kernel 5) with GroupNorm + ReLU + max-pool + dropout, then a 128-unit dense
  layer and classifier, on mel-spectrogram features. GroupNorm (not BatchNorm)
  keeps the model DP-SGD compatible.

Orchestration uses [Flower](https://flower.ai).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src

# 1) prepare per-client SER data (mel-spectrograms)
python prepare_data.py --data_path ./CREMA-D --out_dir ./client_data

# 2a) single-process simulation
python simulate.py --mode fedavg   --rounds 60  --sigma 1.0
python simulate.py --mode fedasync --rounds 150 --alpha 0.2 --sigma 1.0

# 2b) physical testbed (separate processes / devices)
python -m server.fedasync_server --rounds 150 --alpha 0.2 --min_fit_clients 1
python -m client.fedasync_client --client_id 0 --sigma 1.0 --server <ip:8080>
```

`--alpha` is the FedAsync base decay (paper sweeps 0.2 / 0.4 / 0.6); `--sigma` is
the DP noise scale.

## Citation

```bibtex
@inproceedings{mohammadi2025async,
  title={Empirical Analysis of Asynchronous Federated Learning on Heterogeneous Devices: Efficiency, Fairness, and Privacy Trade-offs},
  author={Mohammadi, Samaneh and Symeonidis, Iraklis and Balador, Ali and Flammini, Francesco},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2025},
  note={arXiv:2505.07041}
}
```

## License

MIT.
