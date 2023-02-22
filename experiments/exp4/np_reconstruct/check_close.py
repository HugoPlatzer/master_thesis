import pickle
import numpy as np
import sys

state_a = pickle.load(open(sys.argv[1], "rb"))
state_b = pickle.load(open(sys.argv[2], "rb"))

if state_a.shape != state_b.shape:
    print(f"shape mismatch {state_a.shape} vs. {state_b.shape}, reshaping")
state_b = state_b.reshape(state_a.shape)

max_diff = np.max(np.abs(state_a - state_b))
print(f"maxdiff: {max_diff}")
is_close = np.allclose(state_a, state_b)
print(f"allclose: {is_close}")
