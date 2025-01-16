from model import GPT2
import sys
import torch
from time import time


def print_prediction_values(model_output):
    a = [(x.item(), i) for i, x in enumerate(model_output[0][-1])]
    a.sort()
    for value, i in a:
        if chr(i).isprintable():
            print(value, i, f"'{chr(i)}'")
        else:
            print(value, i)

m = GPT2(n_pos=1024, n_vocab=128, n_hidden=768, n_heads=12, n_blocks=12)
m.eval()
m.load_weights_from_dump("../hf_model/params/params.pickle")
input_s = sys.argv[1]
x = torch.tensor([ord(c) for c in input_s]).unsqueeze(0)
t1 = time()
y = m(x)
t2 = time()


print_prediction_values(y)
print("t={:.5f}s".format(t2-t1))

