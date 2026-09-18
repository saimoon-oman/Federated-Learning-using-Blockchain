"""Structured JSONL results logging (guideline 5.7).

One JSON line per round; consumed by the analysis/figures stage.
Replaces the plain accuracy.txt / precision.txt append files.
"""
import json
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def new_run_path(dataset, method, alpha, k, w_data):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_ds = str(dataset).replace(" ", "")
    fname = f"{safe_ds}_{method}_alpha{alpha}_k{k}_w{w_data}_{ts}.jsonl"
    return os.path.join(RESULTS_DIR, fname)


class RunLogger:
    def __init__(self, dataset, method, alpha=None, k=None, w_data=None,
                 n_clients=None, n_committee=None, seed=None, extra=None):
        self.path = new_run_path(dataset, method, alpha, k, w_data)
        self.base = {"dataset": dataset, "method": method, "alpha": alpha,
                     "k": k, "w_data": w_data, "n_clients": n_clients,
                     "n_committee": n_committee, "seed": seed}
        if extra:
            self.base.update(extra)
        self.fh = open(self.path, "a")

    def log_round(self, round_no, wall_clock_seconds, metrics,
                  trust_scores=None, s_i_values=None,
                  honest_excluded_ids=None, on_chain_gas_used=None):
        rec = dict(self.base)
        rec.update({"round": round_no, "wall_clock_seconds": wall_clock_seconds,
                    "on_chain_gas_used": on_chain_gas_used,
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"), "f1": metrics.get("f1"),
                    "trust_scores": trust_scores, "s_i_values": s_i_values,
                    "honest_excluded_ids": honest_excluded_ids})
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()
