"""On-chain smoke test (no TF needed): exercises the real FedLearning contract
on the local Ganache chain the same way run_experiment.py will —
setServer / getServer / setScore / getScore — and measures per-tx time + gas.
"""
import json
import time

from web3 import Web3

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "python"))
from blockchain_config import GANACHE_URL, CONTRACT_ADDRESS

with open("build/contracts/FedLearning.json") as f:
    abi = json.load(f)["abi"]

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
assert w3.is_connected(), f"cannot reach Ganache at {GANACHE_URL}"
accts = w3.eth.accounts
print("chain connected, accounts:", len(accts))

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
out = {"tx": []}

t0 = time.time()
tx = contract.functions.setServer("testhash123").transact({"from": accts[0], "gas": 3000000})
rcpt = w3.eth.wait_for_transaction_receipt(tx)
dt = time.time() - t0
out["tx"].append({"op": "setServer", "seconds": round(dt, 4), "gas": rcpt["gasUsed"]})
print("setServer:", round(dt, 4), "s, gas:", rcpt["gasUsed"])

t0 = time.time()
tx = contract.functions.setScore(accts[1], 85000).transact({"from": accts[1], "gas": 3000000})
rcpt = w3.eth.wait_for_transaction_receipt(tx)
dt = time.time() - t0
out["tx"].append({"op": "setScore", "seconds": round(dt, 4), "gas": rcpt["gasUsed"]})
print("setScore:", round(dt, 4), "s, gas:", rcpt["gasUsed"])

print("getServer:", contract.functions.getServer().call())
print("getScore:", contract.functions.getScore(accts[1]).call())
assert contract.functions.getServer().call() == "testhash123"
assert contract.functions.getScore(accts[1]).call() == 85000

with open("python/onchain_smoke_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("SMOKE TEST PASSED")
