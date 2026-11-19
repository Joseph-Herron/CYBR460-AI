"""Task 4.2 - Demo: Alice encrypts, Bob decrypts, Eve fails.

Usage:  python demo.py --run runs/baseline
"""
import argparse
import os

import numpy as np
import tensorflow as tf

import data_gen

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/baseline")
ap.add_argument("--trials", type=int, default=3)
a = ap.parse_args()

alice, bob, eve = (tf.keras.models.load_model(os.path.join(a.run, f"{n}.keras"))
                   for n in ("alice", "bob", "eve"))
bits = lambda v: "".join("1" if x > 0 else "0" for x in np.ravel(v))

for t in range(a.trials):
    m, k = data_gen.gen_bits(1), data_gen.gen_bits(1)
    c = alice.predict([m, k], verbose=0)
    pb = bob.predict([c, k], verbose=0)
    pe = eve.predict(c, verbose=0)
    print(f"\n--- Trial {t + 1} ---")
    print("Message    :", bits(m))
    print("Key        :", bits(k))
    print("Ciphertext :", bits(c), " (sign of Alice's output)")
    print("Bob decodes:", bits(pb), f" -> {np.mean(np.sign(pb) == m):.0%} correct")
    print("Eve guesses:", bits(pe), f" -> {np.mean(np.sign(pe) == m):.0%} correct")
