from train_sgd import train_sgd
from architecture.effnet import EfficientNet

PSO_EPOCHS = 10
SGD_EPOCHS = 40

BATCH_SIZE = 16

MODEL_NAME = "effnet_pso"

model = EfficientNet(lock_classifier=True)

model = train_sgd(model, SGD_EPOCHS, BATCH_SIZE, MODEL_NAME)