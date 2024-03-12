import torch
from torch.nn import Module, Parameter


class DOE(Module):
    def __init__(self, shape: int):
        super().__init__()
        self.phase_params = Parameter(2 * torch.pi * torch.rand(shape, shape))

    def forward(self, x):
        return torch.exp(1j * self.phase_params) * x
