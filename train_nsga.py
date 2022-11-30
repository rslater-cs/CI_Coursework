from train_sgd import train_sgd
from architecture.effnet import EfficientNet
from loader.cifar10 import CIFAR10_Loader, CIFAR10_Features
from training_logger import Train_Logger
from torch.utils.data import DataLoader
from NSGAII import NSGA
from torch.nn import Softmax, CrossEntropyLoss
from torch import argmax
import torch
from torch_pso import ParticleSwarmOptimizer

NSGA_EPOCHS = 50
SGD_EPOCHS = 50

BATCH_SIZE = 32

NUM_PARTICLES = None

MODEL_NAME = "effnet_pso"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

logger = Train_Logger("pso")

model = EfficientNet()
model.to(device)

loader = CIFAR10_Loader(BATCH_SIZE)

# Lock everything apart from classifier head
model.requires_grad_(False)

optimizer = NSGA(model.parameters(), num_induviduals=100)

for epoch in range(NSGA_EPOCHS):
    def closure(particle_id):
        tr_accuracy = 0.0
        tr_loss = 0.0
        for inputs, labels in iter(loader.train):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.classifier(inputs)

            tr_loss += criterion(outputs, labels)

            sft_outputs = softmax(outputs)
            sft_outputs = argmax(sft_outputs, dim=1)
            tr_accuracy += (sft_outputs == labels).float().sum()

        print(f"Particle {particle_id} (loss:{tr_loss/loader.train_len}, accuracy:{100*tr_accuracy/loader.train_len})")
        return tr_loss/loader.train_len, 100*tr_accuracy/loader.train_len

    bloss, baccuracy = pso_optimizer.step(closure=closure)

    print(f"Best Loss: {bloss}, Best Accuracy: {baccuracy}")

    val_accuracy = 0.0
    for inputs, labels in iter(loader.valid):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        sft_outputs = softmax(outputs)
        sft_outputs = argmax(sft_outputs, dim=1)
        val_accuracy += (sft_outputs == labels).float().sum()

    val_accuracy = 100*val_accuracy.item()/loader.val_len
    print("\t\tValid Accuracy:", val_accuracy)

    logger.put(epoch=epoch+SGD_EPOCHS, tloss=bloss.item(), taccuracy=baccuracy.item(), vaccuracy=val_accuracy)

test_accuracy = 0.0
for inputs, labels in iter(loader.test):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    sft_outputs = softmax(outputs)
    sft_outputs = argmax(sft_outputs, dim=1)
    test_accuracy += (sft_outputs == labels).float().sum()

val_accuracy = 100*test_accuracy.item()/loader.test_len
print("\t\tTest Accuracy:", test_accuracy)

logger.close()