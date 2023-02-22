import pickle
import numpy as np
import os

before_ln = pickle.load(open("../hf_model/state/block0_start.pickle", "rb"))
after_ln = pickle.load(open("../hf_model/state/block0_ln1.pickle", "rb"))
params = pickle.load(open("../hf_model/params/params.pickle", "rb"))

w = params["block"][0]["ln1_w"]
b = params["block"][0]["ln1_b"]

x = before_ln.flatten()
y = (x - np.mean(x)) / np.sqrt(np.var(x) + 1e-5)
y = w * y + b

print(np.mean(after_ln), np.var(after_ln))
print(np.mean(y), np.var(y))
