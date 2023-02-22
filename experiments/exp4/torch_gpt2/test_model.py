from model import GPT2
import torch

model = GPT2(n_pos=1024, n_vocab=128, n_hidden=768, n_heads=12, n_blocks=12)
model.load_state_dict(torch.load("model_states/test1.bin"))

print(model(torch.tensor([[1]]))[0][0][:10].tolist())
print(model(torch.tensor([[2]]))[0][0][:10].tolist())
print(model(torch.tensor([[3]]))[0][0][:10].tolist())
