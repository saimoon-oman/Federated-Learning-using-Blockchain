"""N-BaIoT per-device loader (guideline 2.1, 6.3).

Layout: python/model/nbaiot/<device>/...csv  (any nesting accepted).
Label comes from the FILENAME ('benign' -> 0, anything else -> 1),
because N-BaIoT traffic CSVs carry no label column.
Defensive about headers: tries header='infer', falls back to no-header.
Subsamples per device for CPU feasibility (guideline: 20-50k rows/device).

Provides:
  NBAIOT_N_FEATURES = 115
  load_nbaiot(data_root, per_device=30000, seed=42)
    -> {device_name: (X_device, y_device)}  with natural device split
"""
import glob
import os

import numpy as np
import pandas as pd

NBAIOT_N_FEATURES = 115


def _read_csv_smart(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)
    # drop non-numeric / id-like columns defensively
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"no numeric columns in {path}")
    if num.shape[1] > NBAIOT_N_FEATURES:
        num = num.iloc[:, -NBAIOT_N_FEATURES:]
    return num.to_numpy(dtype=np.float32)


def load_nbaiot(data_root, per_device=30000, seed=42):
    rng = np.random.default_rng(seed)
    devices = {}
    subdirs = sorted(d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d))
    # also accept flat layout: files directly under data_root with device prefix "N.benign.csv"
    if not subdirs:
        subdirs = [data_root]
    for d in subdirs:
        name = os.path.basename(d.rstrip(os.sep))
        files = sorted(glob.glob(os.path.join(d, "*.csv")))
        if not files:
            continue
        Xs, ys = [], []
        for f in files:
            base = os.path.basename(f).lower()
            label = 0 if "benign" in base else 1
            try:
                X = _read_csv_smart(f)
            except Exception as e:
                print(f"skip {f}: {e}")
                continue
            Xs.append(X)
            ys.append(np.full(X.shape[0], label, dtype=np.int64))
        if not Xs:
            continue
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        # stratified-ish subsample: keep benign:attack ratio, cap rows
        if X.shape[0] > per_device:
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]
            r = len(idx1) / max(len(idx0) + len(idx1), 1)
            n1 = min(len(idx1), max(1, int(per_device * r)))
            n0 = min(len(idx0), per_device - n1)
            sel = np.concatenate([rng.choice(idx0, n0, replace=False),
                                  rng.choice(idx1, n1, replace=False)])
            rng.shuffle(sel)
            X, y = X[sel], y[sel]
        devices[name] = (X, y)
        print(f"device {name}: X={X.shape}, fraud-rate={y.mean():.4f}", flush=True)
    if not devices:
        raise FileNotFoundError(f"no device CSVs found under {data_root}")
    return devices
