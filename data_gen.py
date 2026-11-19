"""Task 1.2 - Synthetic data generation.

Messages and keys are random 16-bit vectors encoded as -1/+1 (so they match the
tanh output range of the networks). Saved as .npy files for reuse.

Usage:  python data_gen.py --out data
"""
import argparse
import os

import numpy as np

N_BITS = 16


def gen_bits(n, n_bits=N_BITS, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    return (rng.integers(0, 2, size=(n, n_bits)) * 2 - 1).astype(np.float32)


def generate_and_save(out_dir="data", n_train=50_000, n_val=2_000, n_test=1_000,
                      n_bits=N_BITS, seed=0):
    """Train / validation / test splits use different seeds, so test is unseen."""
    os.makedirs(out_dir, exist_ok=True)
    for name, n, s in [("train", n_train, seed), ("val", n_val, seed + 1),
                       ("test", n_test, seed + 2)]:
        rng = np.random.default_rng(s)
        np.save(os.path.join(out_dir, f"{name}_messages.npy"), gen_bits(n, n_bits, rng))
        np.save(os.path.join(out_dir, f"{name}_keys.npy"), gen_bits(n, n_bits, rng))
    print(f"Saved train/val/test data to {out_dir}/")


def load_split(split, out_dir="data"):
    m = np.load(os.path.join(out_dir, f"{split}_messages.npy"))
    k = np.load(os.path.join(out_dir, f"{split}_keys.npy"))
    return m, k


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate_and_save(a.out, n_train=a.n_train, seed=a.seed)

