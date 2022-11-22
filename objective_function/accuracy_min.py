from torch.nn import Module, Softmax
from torch import Tensor
from torch import argmax


class Accuracy(Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(1)

    def forward(self, input: Tensor, target: Tensor):
        soft = self.softmax(input)
        max_idx = argmax(soft)
        correct = (max_idx == target).float().sum()

        # Higher accuracy is better
        accuracy = 100*correct

        # Convert into a minimisation problem
        return accuracy