from model import GPT2
import torch
from torch.utils.data import DataLoader
import h5py
import tqdm
import numpy as np


def run_step(model, loss_fn, optimizer, input_batch):
    input_batch = input_batch.to(torch.int64)
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
    model_inputs = input_batch[:, :-1]
    target_outputs = input_batch[:, 1:]
    model_output = model(model_inputs)
    loss = loss_fn(model_output.swapaxes(1, 2), target_outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def print_status(n_step, loss):
    print(f"step {n_step} loss={loss}")



samples_file = "samples_digits.hdf5"
num_epochs = 1
batch_size = 16
logging_steps = 10

model = GPT2(n_pos=1024, n_vocab=300, n_hidden=768, n_heads=12, n_blocks=12)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


if torch.cuda.is_available():
    model = model.to("cuda")
model.train()
loss_fn = torch.nn.CrossEntropyLoss()

samples = h5py.File(samples_file)["a"]
samples = samples[:10000]
samples = torch.tensor(np.array(samples, dtype=np.int32))
loader = DataLoader(samples, batch_size=batch_size, shuffle=True, num_workers=4)

for i, samples_batch in enumerate(tqdm.tqdm(loader)):
    loss = run_step(model, loss_fn, optimizer, samples_batch)
    n_step = i + 1
    if n_step % logging_steps == 0:
        print_status(n_step, loss)

print_status(n_step, loss)
torch.save(model.state_dict(), "model_2.bin")


