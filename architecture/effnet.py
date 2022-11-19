from torch.nn import Sequential, Dropout, Linear, Module
from torchvision.models import efficientnet_b1

class EfficientNet(Module):

    def __init__(self):
        super().__init__()
        classifier = Sequential(
            Dropout(0.2),
            Linear(1280, 10, bias=True),
        )

        self.model = efficientnet_b1()
        self.model.classifier = classifier

    def forward(self, x):
        return self.model(x)