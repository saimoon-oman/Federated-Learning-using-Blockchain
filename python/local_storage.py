"""Local content-addressed weight store.

Replaces the defunct Infura IPFS backend (see EXECUTION_GUIDELINE.md 5.2).
Same type/shape contract as before: save_weights -> hex str hash,
load_weights(hash) -> original object. The hash is what goes on-chain
via setServer, unchanged.
"""
import hashlib
import os
import pickle

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_store")
os.makedirs(STORE_DIR, exist_ok=True)


def save_weights(weights) -> str:
    blob = pickle.dumps(weights)
    content_hash = hashlib.sha256(blob).hexdigest()
    with open(os.path.join(STORE_DIR, content_hash), "wb") as f:
        f.write(blob)
    return content_hash


def load_weights(content_hash: str):
    with open(os.path.join(STORE_DIR, content_hash), "rb") as f:
        return pickle.loads(f.read())
