import torch
from torch import cuda
import torch.optim as optim
from torch.nn import CrossEntropyLoss, Softmax, Sequential, Dropout, Linear
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision import transforms
from torch import argmax
from torchvision.transforms import ToTensor

from tqdm import tqdm
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

EPOCHS = 100
BATCH_SIZE = 16

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

classifier = Sequential(
    Dropout(0.2),
    Linear(1280, 10, bias=True),
)

model = efficientnet_b1()
model.classifier = classifier
model = model.to(device)
model.train()
print(model)

criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

transform = transforms.Compose([
    transforms.Resize(240),
    ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_len = 5000
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_len = len(trainset)-val_len
trainset, valset = random_split(trainset, [train_len, val_len])

testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
test_len = len(testset)

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

softmax = Softmax(1)

max_val_acc = 0.0

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

            tepoch.set_postfix({"loss":loss.item(), "accuracy":(100*((tr_accuracy.item())/train_len))})
    
    val_accuracy = 0.0
    for inputs, labels in iter(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        
        max_outputs = argmax(softmax(outputs), dim=1)
        val_accuracy += (max_outputs == labels).float().sum()
    
    val_accuracy = (val_accuracy/val_len).item()*100
    print('\t\tval_accuracy: {0}'.format(val_accuracy))

    if(val_accuracy > max_val_acc):
        max_val_acc = val_accuracy
        torch.save(model.state_dict(), "./models/best_effnet_sgd.pth")
        print("\t\tSaved state")

test_accuracy = 0.0
for inputs, labels in iter(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    
    max_outputs = argmax(softmax(outputs), dim=1)
    test_accuracy += (max_outputs == labels).float().sum()
    
test_accuracy = (test_accuracy/test_len).item()*100

print("Test Accuracy: {0}".format(test_accuracy))
torch.save(model.state_dict(), "./models/final_effnet_sgd.pth")

