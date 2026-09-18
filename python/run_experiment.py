"""Full experiment driver (guideline 5.5) — Glass.js ported 1:1 to Python.

In-process FL (imports main.py directly) + optional real on-chain calls
(--with_chain, needs local Ganache; off on Kaggle) + structured JSONL
logging (results_logger) + all methods:

  --method {committee_consensus, trust_based, het_aware, static_threshold,
            lgp_lite, fedavg, fedprox, qfedavg}
  --dataset {minst, credit_card, nbaiot}  (credit_card maps to main's 'credit card')
  --alpha {0.1,0.5,1.0,100}  (Dirichlet label-skew; minst/credit_card/nbaiot-pooled)
  --quantity_skew  (MNIST only add-on: Zipf client sizes at fixed alpha)
  --k, --w_data, --static_cutoff, --n_clients, --n_committee, --threshold,
  --rounds, --seed, --adversarial_frac, --attack {labelflip,scaling},
  --dp (use method='differential privacy' path), --with_chain

het_aware / static_threshold wrap main.get_trust_model's accuracy scores:
two-pass (collect S_i for all clients -> robust mu/sigma -> blend).
Non-blockchain baselines bypass committee/trust (plain rounds on Client models).

Label-flip attack: applied once to clients_datalist_y of adv ids.
Scaling attack: via main.ATTACK hooks (applied post-fit in main.py).
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import numpy as np

import main as fl
from dirichlet_partition import dirichlet_partition, quantity_skew_sizes
from heterogeneity import (heterogeneity_scores, robust_stats, u_data,
                           het_aware_trust, flatten_weights)
from heterogeneity import static_threshold_trust
from baselines import fedavg_round, fedprox_round, qfedavg_round, lgp_lite_round
from results_logger import RunLogger
from model.client import Client

DATASET_MAP = {"minst": "minst", "credit_card": "credit card", "nbaiot": "nbaiot"}


def evaluate_current():
    if fl.dataset == "minst":
        probs = fl.server.model.predict(fl.X_test, verbose=0)
        y_pred = np.argmax(probs, axis=1)
        y_true = np.argmax(fl.y_test, axis=1)
    else:
        probs = fl.server.model.predict(fl.X_test, verbose=0)
        y_pred = np.round(probs).astype(int).flatten()
        y_true = np.array(fl.y_test).flatten()
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def apply_label_flip(adv_ids):
    for cid in adv_ids:
        for t in range(fl.max_timestamp):
            y = np.asarray(fl.clients_datalist_y[cid][t])
            if fl.dataset == "minst":
                lab = np.argmax(y, axis=1)
                lab = (lab + 1) % 10
                flipped = np.zeros_like(y)
                flipped[np.arange(len(y)), lab] = 1
            else:
                flipped = 1 - y
            fl.clients_datalist_y[cid][t] = flipped


def pick_adversaries(n_clients, frac, seed):
    n_adv = int(round(n_clients * frac))
    rng = np.random.default_rng(seed + 999)
    return sorted(rng.choice(n_clients, size=n_adv, replace=False).tolist())


def chain_calls(server_hash, trust_scores):
    """Optional real on-chain writes (local Ganache only). Returns total gas."""
    from web3 import Web3
    from blockchain_config import GANACHE_URL, CONTRACT_ADDRESS
    import json as _json
    with open(os.path.join(REPO_ROOT, "build", "contracts", "FedLearning.json")) as f:
        abi = _json.load(f)["abi"]
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    accts = w3.eth.accounts
    c = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
    gas = 0
    gas += w3.eth.wait_for_transaction_receipt(
        c.functions.setServer(server_hash).transact({"from": accts[0], "gas": 3000000}))["gasUsed"]
    for i, s in enumerate(trust_scores):
        gas += w3.eth.wait_for_transaction_receipt(
            c.functions.setScore(accts[(i % (len(accts) - 1)) + 1], int(s)
                                 ).transact({"from": accts[(i % (len(accts) - 1)) + 1],
                                             "gas": 3000000}))["gasUsed"]
    return gas


def rebuild_dirichlet_split(n_clients, alpha, seed, quantity_skew):
    """Replace main's default round-robin split with Dirichlet control.

    Reuses main's per-class bucketing by re-running the loader then
    re-indexing: simplest faithful approach is to load full (X, y) once,
    partition indices, then rebuild the timestamp dicts in main's format.
    Only for minst/credit_card here (nbaiot handled inside main.nbaiot_dataset)."""
    if fl.dataset == "minst":
        from keras.datasets import mnist
        from keras.utils import to_categorical
        (Xtr, ytr), (Xte, yte) = mnist.load_data()
        Xtr = Xtr.astype("float32") / 255
        fl.X_test = Xte.astype("float32") / 255
        ytr_o = ytr
        y_onehot = to_categorical(ytr)
        fl.y_test = to_categorical(yte)
        X_all, y_all, y_idx = Xtr, y_onehot, ytr_o
    elif fl.dataset == "credit card":
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        data = pd.read_csv("model/creditcard_2023.csv")
        X = np.array(data.drop(["id", "Class"], axis=1))
        y = np.array(data["Class"])
        Xs = StandardScaler().fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.15, random_state=seed)
        fl.X_test, fl.y_test = Xte, yte
        X_all, y_all, y_idx = Xtr, ytr, ytr
    else:
        raise ValueError("dirichlet rebuild only for minst/credit card")

    if quantity_skew:
        counts = quantity_skew_sizes(len(X_all), n_clients, seed=seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X_all))
        parts, s = [], 0
        for c in counts:
            parts.append(perm[s:s + c].tolist())
            s += c
    else:
        parts = dirichlet_partition(y_idx, n_clients, alpha, seed=seed)

    fl.clients_datalist_X.clear()
    fl.clients_datalist_y.clear()
    rng = np.random.default_rng(seed)
    for i, p in enumerate(parts):
        Xc = X_all[np.array(p)] if len(p) else X_all[:0]
        yc = y_all[np.array(p)] if len(p) else y_all[:0]
        if len(Xc) == 0:  # empty shard guard (see log 2026-09-18)
            for c in np.unique(y_idx):
                j = np.where(np.asarray(y_idx) == c)[0][0]
                Xc = np.concatenate([Xc, X_all[j:j + 1]], axis=0)
                yc = np.concatenate([yc, y_all[j:j + 1]], axis=0)
        Xd, yd = fl.get_dummy(Xc, yc, 10 if fl.dataset == "minst" else 2)
        idx = np.arange(len(Xc))
        rng.shuffle(idx)
        Xc, yc = Xc[idx], yc[idx]
        cX = np.array_split(Xc, fl.max_timestamp)
        cy = np.array_split(yc, fl.max_timestamp)
        dX, dy = {}, {}
        for j in range(fl.max_timestamp):
            dX[j] = np.concatenate([np.asarray(cX[j]), Xd], axis=0)
            dy[j] = np.concatenate([np.asarray(cy[j]), yd], axis=0)
        fl.clients_datalist_X.append(dX)
        fl.clients_datalist_y.append(dy)


def run_round_blockchain(args, trust_scores, server_hash):
    """One trust/committee round via main.py (het_aware/static blended in)."""
    res = {"client": args.n_clients, "committee": args.n_committee,
           "threshold": args.threshold, "hash": server_hash, "trust_score": trust_scores}
    fl.copy_server(res["hash"])
    if args.method in ("committee_consensus", "trust_based"):
        if fl.method != "committee consensus":
            server_hash, trust_scores = fl.get_trust_model(res)
        else:
            server_hash, trust_scores = fl.get_committee_consensus_model(res)
        s_vals = None
    else:
        server_hash, trust_scores, s_vals = run_round_het(args, res)
    fl.current_timestamp += 1
    return server_hash, trust_scores, s_vals


def _client_list(res):
    lst = []
    for i in range(res["client"]):
        t = Client(i, fl.dataset, fl.method)
        t.model.set_weights(fl.server.model.get_weights())
        lst.append(t)
    return lst


def run_round_het(args, res):
    """Trust round + heterogeneity blending (two-pass over S_i)."""
    import main as _fl
    client_list = _client_list(res)
    if _fl.method == "committee consensus":
        committee = sorted(range(len(res["trust_score"])),
                           key=lambda i: res["trust_score"][i], reverse=True)[:res["committee"]]
    else:
        committee = sorted(range(len(res["trust_score"])),
                           key=lambda i: res["trust_score"][i], reverse=True)[:3]
    model_list, trained_ids = [], []
    for i in range(res["client"]):
        if i in committee:
            continue
        client_list[i].model.fit(
            _fl.clients_datalist_X[i][_fl.current_timestamp],
            _fl.clients_datalist_y[i][_fl.current_timestamp],
            epochs=_fl.epochs, batch_size=_fl.batch_size, validation_split=0.2)
        _fl._maybe_apply_scaling_attack(client_list[i].model, i)
        model_list.append(client_list[i].model.get_weights())
        trained_ids.append(i)
    aggregated = _fl.model_aggregation(model_list)
    # pass 1: raw accuracy scores + S_i
    raw, S = {}, {}
    ref_flat = flatten_weights(aggregated.get_weights())
    for i, idx in enumerate(trained_ids):
        acc = 0.0
        for k in committee:
            sc = client_list[idx].model.evaluate(
                x=_fl.clients_datalist_X[k][_fl.current_timestamp],
                y=_fl.clients_datalist_y[k][_fl.current_timestamp], verbose=0)
            acc += sc[1]
        raw[idx] = int(acc / res["committee"] * 100000)
        upd = [w - s for w, s in zip(client_list[idx].model.get_weights(),
                                     _fl.server.model.get_weights())]
        from heterogeneity import cosine_similarity
        S[idx] = cosine_similarity(flatten_weights(upd), ref_flat)
    for i in range(res["client"]):
        if i not in S:
            S[i] = 1.0
            raw[i] = int(res["trust_score"][i])
    # pass 2: robust stats once, then blend
    mu, sigma = robust_stats(S)
    updated = [int(s) for s in res["trust_score"]]
    for i in committee:
        updated[i] = max(int(res["trust_score"][i]) - 5000, 0)
    non_comm = [i for i in range(res["client"]) if i not in committee]
    if _fl.method == "committee consensus":
        for i in non_comm:
            base = raw[i]
            if args.method == "het_aware":
                updated[i] = int(het_aware_trust(base, u_data(S[i], mu, sigma, args.k), args.w_data))
            else:
                updated[i] = int(static_threshold_trust(base, S[i], args.static_cutoff, args.w_data))
        server_hash = _fl.update_server_model(aggregated)
    else:
        top = sorted(non_comm, key=lambda i: res["trust_score"][i])[:len(non_comm) // 2]
        for i in non_comm:
            if i in top:
                updated[i] = max(int(res["trust_score"][i]) - 5000, 0)
            else:
                if args.method == "het_aware":
                    updated[i] = int(het_aware_trust(raw[i], u_data(S[i], mu, sigma, args.k), args.w_data))
                else:
                    updated[i] = int(static_threshold_trust(raw[i], S[i], args.static_cutoff, args.w_data))
        server_hash = _fl.update_server_model(aggregated)
    return server_hash, updated, S


def run_round_plain(args, clients, server_model, sample_counts, prev_losses=None):
    """Non-blockchain baseline round: local fit then aggregate."""
    updates, losses = [], []
    for i, c in enumerate(clients):
        c.model.set_weights(server_model.get_weights())
        h = c.model.fit(fl.clients_datalist_X[i][fl.current_timestamp],
                        fl.clients_datalist_y[i][fl.current_timestamp],
                        epochs=fl.epochs, batch_size=fl.batch_size,
                        validation_split=0.2, verbose=0)
        fl._maybe_apply_scaling_attack(c.model, i)
        updates.append(c.model.get_weights())
        losses.append(float(h.history["loss"][-1]))
    if args.method == "fedavg":
        agg = fedavg_round(updates, sample_counts)
    elif args.method == "fedprox":
        agg = fedprox_round(updates, server_model.get_weights(), mu=0.01,
                            sample_counts=sample_counts)
    elif args.method == "qfedavg":
        agg = qfedavg_round(updates, losses, q=1.0)
    elif args.method == "lgp_lite":
        agg = lgp_lite_round(updates)
    server_model.set_weights(agg)
    fl.current_timestamp += 1
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="minst", choices=list(DATASET_MAP))
    ap.add_argument("--alpha", type=float, default=100)
    ap.add_argument("--quantity_skew", action="store_true")
    ap.add_argument("--method", default="trust_based",
                    choices=["committee_consensus", "trust_based", "het_aware",
                             "static_threshold", "lgp_lite", "fedavg", "fedprox", "qfedavg"])
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--w_data", type=float, default=0.3)
    ap.add_argument("--static_cutoff", type=float, default=0.5)
    ap.add_argument("--n_clients", type=int, default=9)
    ap.add_argument("--n_committee", type=int, default=3)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adversarial_frac", type=float, default=0.0)
    ap.add_argument("--attack", default="labelflip", choices=["labelflip", "scaling"])
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--with_chain", action="store_true")
    ap.add_argument("--nbaiot_root", default="model/nbaiot")
    ap.add_argument("--nbaiot_per_device", type=int, default=30000)
    args = ap.parse_args()

    np.random.seed(args.seed)
    import random as _r
    _r.seed(args.seed)

    fl.clients_datalist_X.clear()
    fl.clients_datalist_y.clear()
    fl.current_timestamp = 0
    fl.dataset = DATASET_MAP[args.dataset]
    if args.method == "committee_consensus":
        fl.method = "committee consensus"
    elif args.dp:
        fl.method = "differential privacy"
    else:
        fl.method = "trust based"
    fl.server = fl.Server(fl.dataset, fl.method)

    if fl.dataset == "nbaiot":
        fl.split_dataset_between_clients(args.n_clients)  # natural or pooled inside
        # NOTE: nbaiot alpha-sweep (>9 clients) routes through main.nbaiot_dataset's
        # pooled Dirichlet path; pass alpha via env override below if needed.
    else:
        if args.alpha == 100 and not args.quantity_skew:
            fl.split_dataset_between_clients(args.n_clients)  # original split (Phase A)
        else:
            fl.split_dataset_between_clients(args.n_clients)
            rebuild_dirichlet_split(args.n_clients, args.alpha, args.seed, args.quantity_skew)

    adv_ids = pick_adversaries(args.n_clients, args.adversarial_frac, args.seed)
    fl.ADVERSARIAL_IDS = set(adv_ids)
    fl.ATTACK = "scaling" if (adv_ids and args.attack == "scaling") else None
    if adv_ids and args.attack == "labelflip":
        apply_label_flip(adv_ids)
    print("adversaries:", adv_ids, "attack:", args.attack if adv_ids else None, flush=True)

    logger = RunLogger(args.dataset, args.method, alpha=args.alpha, k=args.k,
                       w_data=args.w_data, n_clients=args.n_clients,
                       n_committee=args.n_committee, seed=args.seed,
                       extra={"adversarial_frac": args.adversarial_frac,
                              "attack": args.attack, "dp": args.dp,
                              "quantity_skew": args.quantity_skew})
    blockchain = args.method in ("committee_consensus", "trust_based",
                                 "het_aware", "static_threshold")
    gas_total = 0
    t0 = time.time()
    if blockchain:
        trust, agg = fl.train_threshold_times(args.n_clients, args.n_committee, args.threshold)
        trust = [int(s * 100000) for s in trust]
        server_hash = fl.update_server_model(agg)
        if args.with_chain:
            gas_total += chain_calls(server_hash, trust)
        logger.log_round(0, round(time.time() - t0, 2), evaluate_current(),
                         trust_scores=list(trust), on_chain_gas_used=gas_total)
        for r in range(1, args.rounds + 1):
            t0 = time.time()
            server_hash, trust, s_vals = run_round_blockchain(args, trust, server_hash)
            if args.with_chain:
                gas_total += chain_calls(server_hash, trust)
            honest_excl = [i for i in range(args.n_clients)
                           if i not in adv_ids and trust[i] < 0.5 * max(trust)] if max(trust) > 0 else []
            logger.log_round(r, round(time.time() - t0, 2), evaluate_current(),
                             trust_scores=list(trust),
                             s_i_values={str(k): v for k, v in (s_vals or {}).items()},
                             honest_excluded_ids=honest_excl,
                             on_chain_gas_used=gas_total)
            print(f"round {r}: {logger and ''}trust={trust}", flush=True)
    else:
        clients = [Client(i, fl.dataset, fl.method) for i in range(args.n_clients)]
        counts = [len(fl.clients_datalist_X[i][0]) for i in range(args.n_clients)]
        for r in range(args.rounds + 1):
            t0 = time.time()
            if r > 0:
                run_round_plain(args, clients, fl.server.model, counts)
            m = evaluate_current()
            logger.log_round(r, round(time.time() - t0, 2), m)
            print(f"round {r}: {m}", flush=True)
    logger.close()
    print("wrote", logger.path)


if __name__ == "__main__":
    main()
