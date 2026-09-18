"""Phase A baseline reproduction (runs on Kaggle notebook, GPU).

Replicates Glass.js flow in-process (no React, no blockchain — the chain only
stores hash+scores and does not affect model math; on-chain overhead is
measured separately by onchain_smoke_test.py locally).

Matrix: dataset in {minst, credit card} x method in {committee consensus, trust based}
Clients=9, committee=3, threshold=5, iterations=3 (Glass.js defaults).

Usage on Kaggle:
  1. Attach/clone this repo branch (het-aware-trust), cd into
     Federated-Learning-using-Blockchain/
  2. !pip install -r python/requirements.txt   (if TF conflicts arise, fall back
     to the notebook's preinstalled TF + only `pip install tensorflow-privacy==0.9.0`)
  3. Download credit-card-2023 CSV via Kaggle API into python/model/creditcard_2023.csv
     (needs KAGGLE_API_TOKEN env var — never paste/commit it)
  4. !python kaggle/phase_a_baseline.py
Output: kaggle/phase_a_results_<ts>.json + per-round console lines.
Compare curves against thesis Fig 4.1-4.4 / Ch.4 claims (comparable acc,
trust-based faster than committee consensus).
"""
import json
import os
import sys
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import main as fl


def evaluate_current():
    if fl.dataset == "minst":
        probs = fl.server.model.predict(fl.X_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        y_true = np.argmax(fl.y_test, axis=1)
    else:
        probs = fl.server.model.predict(fl.X_test, verbose=0)
        y_pred = np.round(probs).astype(int).flatten()
        y_true = np.array(fl.y_test).flatten()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def run_config(dataset, method, n_clients=9, committee=3, threshold=5, iterations=3):
    fl.clients_datalist_X.clear()
    fl.clients_datalist_y.clear()
    fl.current_timestamp = 0
    fl.dataset = dataset
    fl.method = method
    fl.server = fl.Server(dataset, method)
    fl.split_dataset_between_clients(n_clients)

    t0 = time.time()
    trust_score, aggregated_model = fl.train_threshold_times(n_clients, committee, threshold)
    trust_score = [int(s * 100000) for s in trust_score]
    server_hash = fl.update_server_model(aggregated_model)
    init_s = time.time() - t0

    rounds = [{"round": 0, "metrics": evaluate_current(), "trust_score": list(trust_score),
               "seconds": round(init_s, 2)}]
    for it in range(1, iterations + 1):
        t0 = time.time()
        res = {"client": n_clients, "committee": committee, "threshold": threshold,
               "hash": server_hash, "trust_score": trust_score}
        fl.copy_server(res["hash"])
        if method != "committee consensus":
            server_hash, trust_score = fl.get_trust_model(res)
        else:
            server_hash, trust_score = fl.get_committee_consensus_model(res)
        fl.current_timestamp += 1
        fl.copy_server(server_hash)
        m = evaluate_current()
        dt = time.time() - t0
        rounds.append({"round": it, "metrics": m, "trust_score": list(trust_score),
                       "seconds": round(dt, 2)})
        print(f"[{dataset} | {method}] round {it}: {m} ({dt:.1f}s)", flush=True)
    return rounds


def main():
    if not os.path.exists("model/creditcard_2023.csv"):
        print("WARNING: model/creditcard_2023.csv missing — credit-card configs will fail. "
              "Download via: kaggle datasets download -d "
              "nelgiriyewithana/credit-card-fraud-detection-dataset-2023 "
              "-p python/model/ --unzip")
    all_results = {}
    for dataset in ["minst", "credit card"]:
        for method in ["committee consensus", "trust based"]:
            print(f"=== {dataset} / {method} ===", flush=True)
            t0 = time.time()
            try:
                rounds = run_config(dataset, method)
                all_results[f"{dataset}||{method}"] = {"ok": True, "rounds": rounds,
                                                       "total_s": round(time.time() - t0, 1)}
            except Exception as e:
                all_results[f"{dataset}||{method}"] = {"ok": False, "error": repr(e)}
                print(f"FAILED {dataset}/{method}: {e!r}", flush=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPO_ROOT, "kaggle", f"phase_a_results_{ts}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
