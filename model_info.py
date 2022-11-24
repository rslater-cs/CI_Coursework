from torch import cuda
from architecture.effnet import EfficientNet
import torch
from torch.nn import Module
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

def get_param_count(model: Module):
    total_params = 0
    for param in model.parameters():
        total_params += torch.prod(torch.tensor(param.shape))
    return total_params

if __name__ == "__main__":

    device = "cuda:0" if cuda.is_available() else "cpu"

    print("Using", device)

    model = EfficientNet()
    model = model.to(device)
    model.train()

    print(model)
    print(get_param_count(model))