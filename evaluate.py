"""Task 4.1 - Final evaluation on 1,000 unseen messages/keys.

Usage:  python evaluate.py --run runs/baseline
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf

import data_gen

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/baseline")
ap.add_argument("--data", default="data")
a = ap.parse_args()

alice, bob, eve = (tf.keras.models.load_model(os.path.join(a.run, f"{n}.keras"))
                   for n in ("alice", "bob", "eve"))
m, k = data_gen.load_split("test", a.data)        # 1,000 samples never used in training
c = alice.predict([m, k], verbose=0)
bob_acc = float(np.mean(np.sign(bob.predict([c, k], verbose=0)) == m))
eve_acc = float(np.mean(np.sign(eve.predict(c, verbose=0)) == m))

# Extra sanity check: Bob with the WRONG key should also fail (he is not magic).
wrong_k = np.roll(k, 1, axis=0)
bob_wrong = float(np.mean(np.sign(bob.predict([c, wrong_k], verbose=0)) == m))

res = dict(n_test=len(m), bob_bit_accuracy=bob_acc, eve_bit_accuracy=eve_acc,
           bob_with_wrong_key=bob_wrong,
           bob_meets_target=bob_acc > 0.95, eve_near_chance=abs(eve_acc - 0.5) < 0.05)
print(json.dumps(res, indent=2))
with open(os.path.join(a.run, "final_eval.json"), "w") as f:
    json.dump(res, f, indent=2)
