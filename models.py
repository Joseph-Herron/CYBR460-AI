"""Task 1.3 - Alice, Bob and Eve (Keras Functional API).

"Mix & transform" architecture from Abadi & Andersen (2016):
  fully-connected layer (lets every output bit depend on every input bit)
  -> stack of 1-D convolutions (learn XOR-like bit mixing)
  -> tanh output of N values.

`hidden` is kept for compatibility with train.py/tune.py: its LENGTH sets depth.
  hidden=(64,)    -> base conv stack
  hidden=(64,64)  -> base conv stack + 1 extra conv block
"""
from tensorflow import keras
from tensorflow.keras import layers

N_BITS = 16


def _mix_and_transform(x, n_bits, depth):
    x = layers.Dense(2 * n_bits, activation="sigmoid")(x)      # FC "mix" layer
    x = layers.Reshape((2 * n_bits, 1))(x)
    x = layers.Conv1D(2, 4, padding="same", activation="sigmoid")(x)             # (2N, 2)
    x = layers.Conv1D(4, 2, strides=2, padding="same", activation="sigmoid")(x)  # (N, 4)
    for _ in range(max(depth - 1, 0)):                          # optional extra blocks
        x = layers.Conv1D(4, 1, padding="same", activation="sigmoid")(x)
    x = layers.Conv1D(1, 1, padding="same", activation="tanh")(x)                # (N, 1)
    return layers.Flatten()(x)                                  # (N,)


def build_alice(n_bits=N_BITS, hidden=(64,)):
    """(message, key) -> ciphertext"""
    msg = keras.Input((n_bits,), name="message")
    key = keras.Input((n_bits,), name="key")
    x = layers.Concatenate()([msg, key])
    return keras.Model([msg, key], _mix_and_transform(x, n_bits, len(hidden)), name="alice")


def build_bob(n_bits=N_BITS, hidden=(64,)):
    """(ciphertext, key) -> recovered message"""
    cipher = keras.Input((n_bits,), name="cipher")
    key = keras.Input((n_bits,), name="key")
    x = layers.Concatenate()([cipher, key])
    return keras.Model([cipher, key], _mix_and_transform(x, n_bits, len(hidden)), name="bob")


def build_eve(n_bits=N_BITS, hidden=(64, 64), input_noise=0.0):
    """ciphertext (no key) -> guessed message.
    input_noise > 0 adds Gaussian noise to Eve's input during training only."""
    cipher = keras.Input((n_bits,), name="cipher")
    x = cipher
    if input_noise > 0:
        x = layers.GaussianNoise(input_noise)(x)
    return keras.Model(cipher, _mix_and_transform(x, n_bits, len(hidden)), name="eve")