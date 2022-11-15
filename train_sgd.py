from torch import cuda
import torch.optim as optim
from torch.nn import CrossEntropyLoss, Softmax
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet18_Weights
from torch import argmax

from tqdm import tqdm
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

EPOCHS = 20
BATCH_SIZE = 4096

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = model.to(device)
print(model)

criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

transform = ToTensor()
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

softmax = Softmax(1)

for epoch in range(EPOCHS):
    tr_accuracy = 0.0
    progress = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            sft_outputs = softmax(outputs)
            sft_outputs = argmax(sft_outputs, dim=1)
            tr_accuracy += (sft_outputs == labels).float().sum()
            progress += 1

            tepoch.set_postfix({"loss":loss.item(), "accuracy":((tr_accuracy*100)/(progress*BATCH_SIZE)).item()})
    
    val_accuracy = 0.0
    for inputs, labels in iter(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        
        max_outputs = argmax(softmax(outputs))
        val_accuracy += (max_outputs == labels).float().sum()

    print('\t\tval_accuracy: {0}'.format((val_accuracy/len(test_loader)).item()))

