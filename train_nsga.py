from train_sgd import train_sgd
from architecture.effnet import EfficientNet
from data_interface.cifar10 import CIFAR10_Loader, CIFAR10_Features
from data_interface.training_logger import NSGA_Logger, Train_Logger
from torch.utils.data import DataLoader
from optimisation.NSGAII import NSGA
from criterions.nsga_crit import Correct, Complexity
from torch.nn import Softmax, CrossEntropyLoss
from torch import argmax
import torch
from torch_pso import ParticleSwarmOptimizer

NSGA_EPOCHS = 50
SGD_EPOCHS = 0

BATCH_SIZE = 32

NUM_PARTICLES = 100

MODEL_NAME = "effnet_nsga"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

sgd_logger = Train_Logger("nsga_sgd")
logger = NSGA_Logger("nsga")

model = EfficientNet()
model.to(device)

loader = CIFAR10_Loader(BATCH_SIZE)

model.classifier.requires_grad_(False)

sgd_crit = CrossEntropyLoss()

model: EfficientNet = train_sgd(model=model, loader=loader, logger=sgd_logger, criterion=sgd_crit, device=device, epochs=SGD_EPOCHS, model_name=MODEL_NAME)

model.requires_grad_(False)

criterion1 = Correct()
criterion2 = Complexity(device)

optimizer = NSGA(model.classifier.parameters(), num_induviduals=NUM_PARTICLES, device=device)

print("\t\tLoading all features")
all_features = torch.empty((loader.train_len, 1280)).to(device)
all_labels = torch.empty((loader.train_len)).to(device)
batch_no = 0

for inputs, labels in loader.train:
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model.backbone(inputs)

    end = (batch_no+1)*BATCH_SIZE
    end = min(end, loader.train_len)

    all_features[batch_no*BATCH_SIZE:end] = outputs
    all_labels[batch_no*BATCH_SIZE:end] = labels

    batch_no += 1

all_labels = all_labels.type(torch.LongTensor)

features_dataset = CIFAR10_Features(all_features, all_labels)
features_loader = DataLoader(features_dataset, batch_size=loader.train_len)

print("\t\t Loading complete")

for epoch in range(NSGA_EPOCHS):
    def closure():
        tr_accuracy = 0.0
        for inputs, labels in iter(features_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.classifier(inputs)

            tr_accuracy += criterion1(outputs, labels)

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