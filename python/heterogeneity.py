"""HT-AS heterogeneity signal + trust (guideline 5.3, proposal Eq. 2-4).

Numpy-only (testable without TF). Integrates into main.py's
get_trust_model / get_committee_consensus_model as an additive term on the
real accuracy-based score (guideline 2.2).
"""
import numpy as np


def flatten_weights(weights):
    return np.concatenate([np.asarray(w).flatten() for w in weights])


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def heterogeneity_scores(client_updates: dict, reference_update):
    """Eq. 2: {client_id: S_i^t} cosine vs reference update direction."""
    ref = flatten_weights(reference_update) if isinstance(reference_update, list) else reference_update
    return {cid: cosine_similarity(flatten_weights(upd) if isinstance(upd, list) else upd, ref)
            for cid, upd in client_updates.items()}


def robust_stats(values: dict):
    """Median + normal-consistent MAD over a round's S_i values."""
    arr = np.array(list(values.values()), dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) * 1.4826
    return median, mad if mad > 0 else 1e-6


def u_data(s_i, mu, sigma, k=2.0):
    """Eq. 3: two-tier tolerance band. 1 at center -> 0 at k*sigma edge, 0 outside."""
    dev = abs(float(s_i) - float(mu))
    if dev <= k * float(sigma):
        return 1.0 - dev / (k * float(sigma))
    return 0.0


def het_aware_trust(accuracy_score, u_data_value, w_data=0.3):
    """Eq. 4 adapted to the code's 0..100000 integer accuracy scale."""
    return accuracy_score * (1 - w_data) + (u_data_value * 100000) * w_data


def static_threshold_trust(accuracy_score, s_i, cutoff=0.5, w_data=0.3):
    """Ablation: same S_i signal, FIXED cutoff instead of dynamic k*sigma band.
    1 if s_i >= cutoff else 0, blended with the same weight."""
    u = 1.0 if float(s_i) >= cutoff else 0.0
    return accuracy_score * (1 - w_data) + (u * 100000) * w_data


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ref = [rng.normal(size=(4, 4)), rng.normal(size=(4,))]
    # honest clients: small noise around ref; adversary: flipped direction
    updates = {i: [w + rng.normal(scale=0.05, size=w.shape) for w in ref] for i in range(8)}
    updates[8] = [-w * 2.0 for w in ref]
    S = heterogeneity_scores(updates, ref)
    mu, sigma = robust_stats(S)
    print("S honest:", round(float(np.mean([S[i] for i in range(8)])), 4),
          "S adv:", round(S[8], 4), "mu:", round(mu, 4), "sigma:", round(sigma, 4))
    uh = [u_data(S[i], mu, sigma, k=2.0) for i in range(8)]
    ua = u_data(S[8], mu, sigma, k=2.0)
    assert min(uh) > ua, "adversary must score lower than every honest client"
    assert all(0.0 <= v <= 1.0 for v in uh + [ua])
    t = het_aware_trust(85000, uh[0])
    assert 0 < t <= 100000
    print("HONEST u_data range:", round(min(uh), 3), "..", round(max(uh), 3),
          "ADV u_data:", ua, "sample trust:", round(t))
    print("HETEROGENEITY SELF-TEST PASSED")
