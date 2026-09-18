"""Non-blockchain FL baselines + LGP-lite SOTA comparison (guideline 5.6).

All functions operate on lists of per-client weight lists (same structure as
Keras model.get_weights()) and return the aggregated weight list.
Numpy-only (testable without TF). Reimplemented from the published
equations/descriptions — not copied code:
- FedAvg (McMahan et al. 2017), FedProx (Li et al. 2020, mu),
  q-FedAvg/q-FFL (Li et al. 2020, q).
- lgp_lite: simplified layer-wise gradient purification inspired by
  Issa et al., IEEE TDSC 2026 (good-faith reimplementation from the
  published description; state as such in the paper).
"""
import numpy as np


def fedavg_round(client_weights, sample_counts=None):
    n = len(client_weights)
    if sample_counts is None:
        sample_counts = [1.0] * n
    total = float(sum(sample_counts))
    agg = [np.zeros_like(w) for w in client_weights[0]]
    for cw, cnt in zip(client_weights, sample_counts):
        for j, w in enumerate(cw):
            agg[j] += np.asarray(w) * (cnt / total)
    return agg


def fedprox_round(client_weights, global_weights, mu=0.01, sample_counts=None):
    """FedProx-style: pull each client delta toward global by proximal term,
    then average. Aggregation-level equivalent of the proximal regularizer."""
    proxed = []
    for cw in client_weights:
        proxed.append([np.asarray(w) - mu * (np.asarray(w) - np.asarray(g))
                       for w, g in zip(cw, global_weights)])
    return fedavg_round(proxed, sample_counts)


def qfedavg_round(client_weights, client_losses, q=1.0):
    """q-FFL weighting: weight ~ loss^q (upweights worst performers)."""
    losses = np.array(client_losses, dtype=float)
    w = np.power(np.maximum(losses, 1e-12), q)
    w /= w.sum()
    agg = [np.zeros_like(np.asarray(x)) for x in client_weights[0]]
    for cw, wi in zip(client_weights, w):
        for j, x in enumerate(cw):
            agg[j] += np.asarray(x) * wi
    return agg


def lgp_lite_round(client_weight_updates, k_layers=2.0):
    """Per-layer: median/MAD over client update norms; drop layers whose norm
    deviates > k_layers*MAD from the median, average survivors."""
    n_layers = len(client_weight_updates[0])
    out = []
    for l in range(n_layers):
        norms = np.array([np.linalg.norm(np.asarray(upd[l]))
                          for upd in client_weight_updates])
        median = float(np.median(norms))
        mad = float(np.median(np.abs(norms - median))) * 1.4826 or 1e-6
        keep = [upd[l] for upd, nm in zip(client_weight_updates, norms)
                if abs(nm - median) <= k_layers * mad]
        pooled = keep if keep else [upd[l] for upd in client_weight_updates]
        out.append(np.mean(np.stack([np.asarray(x) for x in pooled]), axis=0))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    base = [rng.normal(size=(3, 3)), rng.normal(size=(3,))]
    honest = [[w + rng.normal(scale=0.05, size=w.shape) for w in base] for _ in range(5)]
    poison = [[w * 10.0 for w in base]]
    clients = honest + poison
    ref = fedavg_round(honest)
    for name, fn in [("fedavg", lambda: fedavg_round(clients)),
                     ("fedprox", lambda: fedprox_round(clients, base)),
                     ("qfedavg", lambda: qfedavg_round(clients, [0.5] * 5 + [5.0])),
                     ("lgp_lite", lambda: lgp_lite_round(clients))]:
        agg = fn()
        d = sum(float(np.linalg.norm(np.asarray(a) - np.asarray(r)))
                for a, r in zip(agg, ref))
        print(f"{name}: dist-to-honest-avg = {d:.4f}")
    print("BASELINES SELF-TEST PASSED")
