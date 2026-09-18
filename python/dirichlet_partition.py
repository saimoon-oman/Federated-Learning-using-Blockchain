"""Dirichlet label-skew partitioning (guideline 5.1).

Turns any (X, y) pair into N non-IID client shards controlled by
concentration parameter alpha. Small alpha -> severe label skew;
large alpha (e.g. 100) -> near-IID. Numpy-only (testable without TF).
"""
import numpy as np


def dirichlet_partition(y, n_clients, alpha, seed=42):
    """Return a list of length n_clients, each a list of sample indices,
    partitioned by a Dirichlet(alpha) distribution over classes."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=np.repeat(float(alpha), n_clients))
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        shards = np.split(idx_c, split_points)
        for cid, shard in enumerate(shards):
            client_indices[cid].extend(shard.tolist())
    for cid in range(n_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def quantity_skew_sizes(n_samples, n_clients, seed=42, s=1.5):
    """Zipf/power-law client dataset sizes (guideline 6.3 quantity-skew add-on).
    Returns integer sample counts per client summing to n_samples.
    Larger s -> more skewed (few clients hold most data)."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_clients + 1)
    weights = 1.0 / np.power(ranks, s)
    weights /= weights.sum()
    counts = (weights * n_samples).astype(int)
    # fix rounding remainder on the largest client
    counts[0] += n_samples - counts.sum()
    perm = rng.permutation(n_clients)
    return counts[perm].tolist()


if __name__ == "__main__":
    # self-test (no TF needed)
    y = np.array([0] * 1000 + [1] * 1000 + [2] * 500)
    for a in (0.1, 0.5, 1.0, 100):
        parts = dirichlet_partition(y, 9, a, seed=0)
        assert sum(len(p) for p in parts) == len(y), "index loss!"
        assert len(set(i for p in parts for i in p)) == len(y), "overlap/dup!"
        frac0 = [sum(1 for i in p if y[i] == 0) / max(len(p), 1) for p in parts]
        print(f"alpha={a}: sizes={[len(p) for p in parts]}, class0-frac range="
              f"{min(frac0):.2f}..{max(frac0):.2f}")
    print("quantity skew:", quantity_skew_sizes(2500, 9))
    print("DIRICHLET SELF-TEST PASSED")
