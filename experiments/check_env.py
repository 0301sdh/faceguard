import torch

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Tensor + 1:", x + 1)