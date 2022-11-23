from train_sgd import train_sgd
from architecture.effnet import EfficientNet
from loader.cifar10 import CIFAR10_Loader
from torch.nn import Softmax, CrossEntropyLoss

PSO_EPOCHS = 10
SGD_EPOCHS = 40

BATCH_SIZE = 16

MODEL_NAME = "effnet_pso"

model = EfficientNet()

# Lock classsifier head
model.train_backbone(True)

model = train_sgd(model, SGD_EPOCHS, BATCH_SIZE, MODEL_NAME)

# Lock everything apart from classifier head
model.train_backbone(False)