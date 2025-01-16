from misc import *
import numpy as np
import pickle

n = 100000
n_bits = 32
sample_len = 100
samples_file = "samples.dat"

samples = gen_train_samples(n, n_bits, sample_len)
with open(samples_file, "wb") as f:
    f.write(pickle.dumps(samples))
