from transformers import GPT2LMHeadModel
from misc import *
import torch
import sys

num_queries = 100
query_bits = 32
model_dir = sys.argv[1]
model = GPT2LMHeadModel.from_pretrained(model_dir)

samples = gen_test_samples(num_queries, query_bits)
num_correct, num_total = 0, 0
for in_seq, corr_out_seq in samples:
    corr_out_str = ints_to_str(corr_out_seq)
    in_seq_torch = torch.tensor(np.array([in_seq]))
    model_out = model.generate(in_seq_torch, max_length=model.config.max_length)
    model_out_str = ints_to_str(model_out[0][len(in_seq):])
    print(ints_to_str(in_seq))
    print("output", model_out_str, "corr", corr_out_str)
    if corr_out_str == model_out_str:
        num_correct += 1
    num_total += 1

print("{}/{}".format(num_correct, num_total))
