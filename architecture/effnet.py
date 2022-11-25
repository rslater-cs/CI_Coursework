from torch.nn import Sequential, Dropout, Linear, Module
from torchvision.models import efficientnet_b1, efficientnet_b0

class NoneLayer(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class EfficientNet(Module):

    def __init__(self):
        super().__init__()
        self.classifier = Sequential(
            Dropout(0.2),
            Linear(1280, 10, bias=True),
        )

        self.backbone = efficientnet_b1()
        self.backbone.classifier = NoneLayer()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x