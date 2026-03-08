import torch

if torch.cuda.is_available():
    print("Usando cuda:", torch.cuda.get_device_name(0))
