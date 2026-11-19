# CYBR460-AI
Adversarial Neural Cryptography - Secure Communication with AI
# Adversarial Neural Cryptography (Alice, Bob, Eve)

## Run order (matches the assignment phases)
```
pip install -r requirements.txt        # Task 1.1
python data_gen.py --out data          # Task 1.2  (.npy files)
python train.py --out runs/baseline    # Tasks 1.3, 2.1, 2.2, 2.3 (writes curves + models)
python tune.py --epochs 20             # Task 3.1  -> runs/tuning/summary.csv
python evaluate.py --run runs/baseline # Task 4.1  (1,000 unseen messages/keys)
python demo.py --run runs/baseline     # Task 4.2
```
Task 1.1 also asks for a shared GitHub repo: `git init`, add these files, push.

## File map
| File | Tasks |
|---|---|
| data_gen.py | 1.2 |
| models.py | 1.3 |
| train.py | 2.1 forward-pass check, 2.2 adversarial loop, 2.3 metrics/plots |
| tune.py | 3.1 (lr 1e-3 vs 1e-4, batch 32 vs 64, 1 vs 2 hidden layers) |
| evaluate.py | 3.2 / 4.1 |
| demo.py | 4.2 |

## Debugging tips (Task 3.3)
- Eve too good: `--hidden-ab 128 128` (more Alice/Bob capacity) or `--eve-noise 0.1`.
- Bob not learning: `--hidden-ab 32` (simpler), `--lr 0.0001`, or more `--epochs`.
- Training unstable / curves oscillate: lower `--lr`, lower `--eve-steps`.

## Report outline (Task 4.3, submit as PDF)
1. Setup: 16-bit messages and keys encoded as -1/+1; Alice = Dense(relu) -> tanh(16), Bob takes (cipher, key), Eve takes cipher only.
2. Training: Eve step then Alice/Bob step; AB loss = Bob error + (N/2 - Eve error)^2 / (N/2)^2.
3. Results: `training_curves.png`, `tuning/summary.csv`, `final_eval.json` (Bob > 95 %, Eve ~ 50 %).
4. Dynamics: describe the early phase (Eve and Bob both near 50 %), when Bob pulls ahead, and whether Eve recovers.
5. Limitations: no formal security proof, Eve is one fixed architecture, ciphertext is real-valued, key/message sizes are tiny, results depend on seeds and hyperparameters, and a stronger Eve (or known-plaintext attack) may break it.