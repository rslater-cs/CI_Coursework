from torch import cuda
from architecture.effnet import EfficientNet
import torch
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

EPOCHS = 100
BATCH_SIZE = 16

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

model = EfficientNet()
model = model.to(device)
model.train()

total_params = 0
trainable_params = 0
tensors = 0
for param in model.parameters():
    tensors += 1
    print(torch.tensor(param.shape))
    total_params += torch.prod(torch.tensor(param.shape))
    if param.requires_grad:
        trainable_params += torch.prod(torch.tensor(param.shape))

print("Tensors:", tensors)
print("Params:", total_params)
print("Trainable Params:", trainable_params)