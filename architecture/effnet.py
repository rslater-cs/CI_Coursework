from torch.nn import Sequential, Dropout, Linear, Module
from torchvision.models import efficientnet_b1, efficientnet_b0

class EfficientNet(Module):

    def __init__(self, lock_classifier=False):
        super().__init__()
        classifier = Sequential(
            Dropout(0.2),
            Linear(1280, 10, bias=True),
        )

        if lock_classifier:
            classifier.requires_grad_(False)

        self.backbone = efficientnet_b1()
        self.backbone.classifier = classifier

    def forward(self, x):
        return self.backbone(x)