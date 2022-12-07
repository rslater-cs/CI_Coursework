from architecture.effnet import EfficientNet
from data_interface.cifar10 import CIFAR10_Loader
import torch
from torch import argmax
from torch.nn import Softmax

models_path = "./models/final_effnet_"
model_name = "sgd"

full_path = "./models/final_effnet_sgd.pth"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

model = EfficientNet()
model = model.to(device)
model.load_state_dict(torch.load(full_path, map_location=torch.device(device)))

loader = CIFAR10_Loader(batch_size=32)

softmax = Softmax(1)

test_accuracy = 0.0
for inputs, labels in iter(loader.test):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    
    max_outputs = argmax(softmax(outputs), dim=1)
    test_accuracy += (max_outputs == labels).float().sum()
    
test_accuracy = (test_accuracy/loader.test_len).item()*100

print("Test Accuracy: {0}".format(test_accuracy))
