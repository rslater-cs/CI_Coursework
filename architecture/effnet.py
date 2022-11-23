from torch.nn import Sequential, Dropout, Linear, Module
from torchvision.models import efficientnet_b1, efficientnet_b0

class EfficientNet(Module):

    def __init__(self):
        super().__init__()
        classifier = Sequential(
            Dropout(0.2),
            Linear(1280, 10, bias=True),
        )

        self.backbone = efficientnet_b1()
        self.backbone.classifier = classifier

    def train_backbone(self, state=True):
        self.backbone.requires_grad_(state)
        self.backbone.classifier.requires_grad_(not state)

    def forward(self, x):
        return self.backbone(x)