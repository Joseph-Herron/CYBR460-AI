"""Phase 2/3 - Forward pass check, adversarial training loop, metric logging.

Each iteration:
  1. Train Eve (Alice and Bob frozen) to recover the plaintext from the ciphertext.
  2. Freeze Eve; train Alice & Bob so Bob decodes well AND Eve is pushed to chance.

Loss (as in Abadi & Andersen 2016), with errors counted in bits:
  eve_loss = L1 bit error of Eve
  ab_loss  = bob_err + w * (N/2 - eve_err)^2 / (N/2)^2   (w = --eve-weight)

Usage:  python train.py --epochs 30 --lr 0.001 --batch 64 --out runs/baseline
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import data_gen
from models import build_alice, build_bob, build_eve

N = data_gen.N_BITS


def bit_err(pred, true):
    """Mean (over batch) number of wrong bits, using the L1 distance on -1/+1 values."""
    return tf.reduce_mean(tf.reduce_sum(tf.abs(pred - true), axis=1)) / 2.0


def bit_acc(pred, true):
    return tf.reduce_mean(tf.cast(tf.sign(pred) == tf.sign(true), tf.float32))


def forward_pass_check(alice, bob, n_bits=N):
    """Task 2.1 - untrained networks should run end to end (Bob will be poor)."""
    m = tf.constant(data_gen.gen_bits(4, n_bits))
    k = tf.constant(data_gen.gen_bits(4, n_bits))
    c = alice([m, k], training=False)
    p = bob([c, k], training=False)
    print("Forward pass OK. cipher shape:", c.shape, "| untrained Bob acc:",
          float(bit_acc(p, m)))


def run(cfg):
    """cfg keys: out, epochs, batch, lr, hidden_ab (tuple), hidden_eve (tuple),
    eve_steps, eve_noise, eve_weight, data_dir, seed, n_train"""
    cfg = {**dict(out="runs/baseline", epochs=30, batch=64, lr=1e-3, hidden_ab=(64,),
                  hidden_eve=(64, 64), eve_steps=2, eve_noise=0.0, eve_weight=1.0,
                  data_dir="data", seed=0, n_train=50_000), **cfg}
    os.makedirs(cfg["out"], exist_ok=True)
    tf.keras.utils.set_random_seed(cfg["seed"])

    if not os.path.exists(os.path.join(cfg["data_dir"], "train_messages.npy")):
        data_gen.generate_and_save(cfg["data_dir"], n_train=cfg["n_train"])
    tr_m, tr_k = data_gen.load_split("train", cfg["data_dir"])
    va_m, va_k = data_gen.load_split("val", cfg["data_dir"])

    alice = build_alice(N, cfg["hidden_ab"])
    bob = build_bob(N, cfg["hidden_ab"])
    eve = build_eve(N, cfg["hidden_eve"], cfg["eve_noise"])
    forward_pass_check(alice, bob)

    opt_ab = tf.keras.optimizers.Adam(cfg["lr"])
    opt_eve = tf.keras.optimizers.Adam(cfg["lr"])
    ab_vars = alice.trainable_variables + bob.trainable_variables

    @tf.function
    def eve_step(m, k):
        c = alice([m, k], training=False)          # Alice frozen (no gradient taken)
        with tf.GradientTape() as tape:
            loss = bit_err(eve(c, training=True), m)
        opt_eve.apply_gradients(zip(tape.gradient(loss, eve.trainable_variables),
                                    eve.trainable_variables))
        return loss

    @tf.function
    def ab_step(m, k):
        with tf.GradientTape() as tape:            # only ab_vars get updated -> Eve frozen
            c = alice([m, k], training=True)
            b_err = bit_err(bob([c, k], training=True), m)
            e_err = bit_err(eve(c, training=False), m)
            loss = b_err + cfg["eve_weight"] * tf.square(N / 2.0 - e_err) / (N / 2.0) ** 2
        opt_ab.apply_gradients(zip(tape.gradient(loss, ab_vars), ab_vars))
        return loss, b_err

    def evaluate(m, k):
        m, k = tf.constant(m), tf.constant(k)
        c = alice([m, k], training=False)
        pb, pe = bob([c, k], training=False), eve(c, training=False)
        return dict(bob_loss=float(bit_err(pb, m)), eve_loss=float(bit_err(pe, m)),
                    bob_acc=float(bit_acc(pb, m)), eve_acc=float(bit_acc(pe, m)))

    steps = len(tr_m) // cfg["batch"]
    hist = []
    rng = np.random.default_rng(cfg["seed"])
    for ep in range(1, cfg["epochs"] + 1):
        idx = rng.permutation(len(tr_m))
        for s in range(steps):
            b = idx[s * cfg["batch"]:(s + 1) * cfg["batch"]]
            m, k = tf.constant(tr_m[b]), tf.constant(tr_k[b])
            for _ in range(cfg["eve_steps"]):
                eve_step(m, k)
            ab_step(m, k)
        r = {"epoch": ep, **evaluate(va_m, va_k)}     # Task 3.2 / 2.3: validation metrics
        hist.append(r)
        print(f"ep {ep:3d} | Bob acc {r['bob_acc']:.3f} loss {r['bob_loss']:.2f} "
              f"| Eve acc {r['eve_acc']:.3f} loss {r['eve_loss']:.2f}")

    with open(os.path.join(cfg["out"], "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    plot_history(hist, cfg["out"])
    for name, mdl in [("alice", alice), ("bob", bob), ("eve", eve)]:
        mdl.save(os.path.join(cfg["out"], f"{name}.keras"))
    with open(os.path.join(cfg["out"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return hist[-1]


def plot_history(hist, out):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["bob_acc"] for h in hist], label="Bob")
    ax[0].plot(ep, [h["eve_acc"] for h in hist], label="Eve")
    ax[0].axhline(0.5, ls="--", c="gray", lw=1)
    ax[0].axhline(0.95, ls=":", c="gray", lw=1)
    ax[0].set(title="Bit accuracy (validation)", xlabel="epoch", ylabel="accuracy")
    ax[0].legend()
    ax[1].plot(ep, [h["bob_loss"] for h in hist], label="Bob")
    ax[1].plot(ep, [h["eve_loss"] for h in hist], label="Eve")
    ax[1].set(title="Bit error (validation)", xlabel="epoch", ylabel="wrong bits / 16")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-ab", type=int, nargs="+", default=[64])
    ap.add_argument("--hidden-eve", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--eve-steps", type=int, default=2)
    ap.add_argument("--eve-noise", type=float, default=0.0)
    ap.add_argument("--eve-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(dict(out=a.out, epochs=a.epochs, batch=a.batch, lr=a.lr,
             hidden_ab=tuple(a.hidden_ab), hidden_eve=tuple(a.hidden_eve),
             eve_steps=a.eve_steps, eve_noise=a.eve_noise,
             eve_weight=a.eve_weight, seed=a.seed))