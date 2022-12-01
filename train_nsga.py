from train_sgd import train_sgd
from architecture.effnet import EfficientNet
from loader.cifar10 import CIFAR10_Loader, CIFAR10_Features
from training_logger import NSGA_Logger
from torch.utils.data import DataLoader
from NSGAII import NSGA
from criterions.nsga_crit import Correct, Complexity
from torch.nn import Softmax, CrossEntropyLoss
from torch import argmax
import torch
from torch_pso import ParticleSwarmOptimizer

NSGA_EPOCHS = 50

BATCH_SIZE = 256

NUM_PARTICLES = 10

MODEL_NAME = "effnet_nsga"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

logger = NSGA_Logger("nsga")

model = EfficientNet()
model.to(device)

loader = CIFAR10_Loader(BATCH_SIZE)

# Lock everything apart from classifier head
model.requires_grad_(False)

criterion1 = Correct()
criterion2 = Complexity(device)

optimizer = NSGA(model.parameters(), num_induviduals=2, device=device)

for epoch in range(NSGA_EPOCHS):
    def closure():
        tr_accuracy = 0.0
        progress = 0
        for inputs, labels in iter(loader.train):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            tr_accuracy += criterion1(outputs, labels)

            if progress % 10 == 0:
                print(progress)
            progress += 1

        accuracy = 100*tr_accuracy/loader.train_len
        complexity = criterion2(model.parameters())
        return [accuracy, complexity]

    best = optimizer.step(closure=closure)

    print(f"Best Accuracy: {best[0].item()}, Best Complexity: {best[1].item()}")

    val_accuracy = 0.0
    for inputs, labels in iter(loader.valid):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        val_accuracy += criterion1(outputs, labels)

    val_accuracy = 100*val_accuracy.item()/loader.val_len
    print("\t\tValid Accuracy:", val_accuracy)

    val_comp = criterion2(model.parameters())

    logger.put(epoch=epoch, tacc=best[0].item(), tcomp=best[1].item(), vacc=val_accuracy, vcomp=val_comp.item())

test_accuracy = 0.0
for inputs, labels in iter(loader.test):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    test_accuracy += criterion1(outputs, labels)

test_accuracy = 100*test_accuracy.item()/loader.test_len
print("\t\tTest Accuracy:", test_accuracy)

logger.close()