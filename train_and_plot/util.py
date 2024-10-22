import torch

def move_tensor_to_device(tensor):
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda:0")
        return tensor.to(device=cuda_device)
    elif torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        return tensor.to(device=mps_device)
    else:
        return tensor