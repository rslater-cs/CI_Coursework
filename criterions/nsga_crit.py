from torch.nn import Module, Softmax, Parameter
from torch import argmax
import torch
from typing import Iterator

class Correct(Module):

    def __init__(self):
        super().__init__()

        self.softmax = Softmax(1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.softmax(x)
        x = argmax(x, dim=1)
        x = (x == labels).float().sum()
        return x

class Complexity(Module):
    
    def __init__(self, device):
        super().__init__()

        self.device = device

    def forward(self, x: Iterator[Parameter]) -> torch.Tensor:
        result = torch.empty(1).to(self.device)

        for param in x:
            result += torch.sum(torch.square(param.data))

        return result


