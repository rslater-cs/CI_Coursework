from train_sgd import train_sgd
from architecture.effnet import EfficientNet
from loader.cifar10 import CIFAR10_Loader, CIFAR10_Features
from training_logger import Train_Logger
from torch.utils.data import DataLoader
from EffNetPSO import PSO
from torch.nn import Softmax, CrossEntropyLoss
from torch import argmax
import torch
from torch_pso import ParticleSwarmOptimizer

PSO_EPOCHS = 50
SGD_EPOCHS = 50

BATCH_SIZE = 32

NUM_PARTICLES = None

MODEL_NAME = "effnet_pso"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

logger = Train_Logger("pso")

model = EfficientNet()

criterion = CrossEntropyLoss()
loader = CIFAR10_Loader(BATCH_SIZE)

# Lock classsifier head
model.classifier.requires_grad_(False)

model = train_sgd(model=model, loader=loader, logger=logger, criterion=criterion, device=device, epochs=SGD_EPOCHS, model_name=MODEL_NAME)

# Lock everything apart from classifier head
model.requires_grad_(False)

pso_optimizer = PSO(num_particles=NUM_PARTICLES, net=model.classifier, device=device)

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

print("\t\tLoading done")

softmax = Softmax(1)

for epoch in range(PSO_EPOCHS):
    def closure(particle_id):
        tr_accuracy = 0.0
        tr_loss = 0.0
        for inputs, labels in iter(features_loader):
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

test_accuracy = 100*test_accuracy.item()/loader.test_len
print("\t\tTest Accuracy:", test_accuracy)

logger.close()