import transformers
import torch
import sys
import os
from time import time

input_str = sys.argv[1]

model_path = os.getenv("HF_MODEL_PATH")
if model_path is None:
    raise Exception("model path not given")

m = transformers.GPT2LMHeadModel.from_pretrained(model_path)
x = torch.tensor([ord(c) for c in input_str])

t1 = time()
x_out = m.forward(x)
t2 = time()


a = [(tensor.item(), i) for i, tensor in enumerate(x_out.logits[-1])]
a.sort()
for value, i in a:
    if chr(i).isprintable():
        print(value, i, f"'{chr(i)}'")
    else:
        print(value, i)

print("t={:.5f}s".format(t2-t1))
