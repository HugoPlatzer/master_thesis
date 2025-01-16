from model import GPT2
import torch
from torch.utils.data import DataLoader
import h5py
import tqdm
import numpy as np
import sys
import os


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


def print_status(rank, n_step, loss):
    if rank == 0:
        print(f"step {n_step} loss={loss}")


def save_model(model, epoch):
    file_name = "model_{:03}.bin".format(epoch)
    torch.save(model.state_dict(), file_name)




samples_file = "samples_digits.hdf5"
num_epochs = 100
batch_size = 16
logging_steps = 10

torch.distributed.init_process_group("nccl")
rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)

model = GPT2(n_pos=1024, n_vocab=300, n_hidden=768, n_heads=12, n_blocks=12)
model.train()
model = model.to("cuda")
dist_model = torch.nn.parallel.DistributedDataParallel(model)

optimizer = torch.optim.AdamW(dist_model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

samples = h5py.File(samples_file)["a"]
# ~ samples = samples[:20000]
samples = torch.tensor(np.array(samples, dtype=np.int32))
sampler = torch.utils.data.DistributedSampler(samples, shuffle=True)
loader = DataLoader(samples, batch_size=batch_size, sampler=sampler)

for epoch in range(1, num_epochs + 1):
    for i, samples_batch in enumerate(tqdm.tqdm(loader)):
        loss = run_step(dist_model, loss_fn, optimizer, samples_batch)
        n_step = i + 1
        if n_step % logging_steps == 0:
            print_status(rank, n_step, loss)

    torch.distributed.barrier()
    print_status(rank, n_step, loss)
    if rank == 0:
        save_model(model, epoch)


