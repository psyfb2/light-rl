import torch

if torch.cuda.is_available():
    DEVICE = "cuda"
else:   
    DEVICE = "cpu"