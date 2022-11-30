from torch.nn import Module, Softmax
from torch import argmax
from torch import Tensor

class Correct(Module):

    def __init__(self):
        super().__init__()

        self.softmax = Softmax(1)

    def forward(self, x: , labels):
        x = self.softmax(x)
        x = argmax(x, dim=1)
        x = (x == labels).float().sum()
        return x

class Complexity(Module):
    
    def __init__(self):
        super().__init__()

        self.