"""Task 3.1 - Hyperparameter sweep: learning rate, batch size, network depth.
Writes runs/tuning/summary.csv (paste into your report).

Usage:  python tune.py --epochs 20
"""
import argparse
import csv
import itertools
import os

import train

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
a = ap.parse_args()

grid = itertools.product([1e-3, 1e-4],          # learning rate
                         [32, 64],              # batch size
                         [(64,), (64, 64)])     # Alice/Bob depth (1 vs 2 hidden layers)
rows = []
for lr, batch, depth in grid:
    name = f"lr{lr}_b{batch}_d{len(depth)}"
    print(f"\n=== {name} ===")
    last = train.run(dict(out=os.path.join("runs", "tuning", name), epochs=a.epochs,
                          lr=lr, batch=batch, hidden_ab=depth))
    rows.append(dict(lr=lr, batch=batch, ab_hidden_layers=len(depth),
                     bob_acc=round(last["bob_acc"], 4), eve_acc=round(last["eve_acc"], 4)))

os.makedirs("runs/tuning", exist_ok=True)
with open("runs/tuning/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\nBest by Bob accuracy:", max(rows, key=lambda r: r["bob_acc"]))
