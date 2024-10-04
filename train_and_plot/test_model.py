import model
import torch
import os

m = model.GPT2Model(max_prompt_len=8, max_response_len=8, n_embd=768, n_layer=12, n_head=12)
m.save_to_file("model.bin")
m2 = model.GPT2Model.load_from_file("model.bin")

print(m)
print(m2)

os.remove("model.bin")