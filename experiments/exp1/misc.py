import numpy as np
import random

def str_to_ints(s):
    return np.array([ord(c) for c in s])

def ints_to_str(a):
    return "".join(chr(x) for x in a)


def gen_train_samples(n, n_bits=32, sample_len=100):
    samples = []
    for i in range(n):
        x1 = random.randint(0, 2**n_bits - 1)
        x2 = random.randint(0, 2**n_bits - 1)
        s = "{}+{}={}.".format(x1, x2, x1 + x2)
        if len(s) > sample_len:
            raise Exception("sample too long")
        s += "."  * (sample_len - len(s))
        s_enc = str_to_ints(s)
        samples.append(s_enc)
    return samples


def gen_test_samples(n, n_bits=32):
    samples = []
    for i in range(n):
        x1 = random.randint(0, 2**n_bits - 1)
        x2 = random.randint(0, 2**n_bits - 1)
        s_query = "{}+{}=".format(x1, x2)
        s_query_enc = str_to_ints(s_query)
        s_result = "{}.".format(x1 + x2)
        s_result_enc = str_to_ints(s_result)
        samples.append((s_query_enc, s_result_enc))
    return samples
