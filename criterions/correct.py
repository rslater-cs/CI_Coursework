from torch.nn import Module, Softmax
from torch import argmax

class Correct(Module):

    def __init__(self):
        super().__init__()

        self.softmax = Softmax(1)

    def forward(self, x, labels):
        x = self.softmax(x)
        x = argmax(x, dim=1)
        x = (x == labels).float().sum()
        return x