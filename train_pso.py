import torch
from torch import cuda
from torch.nn import CrossEntropyLoss, Softmax, MSELoss
from torch import argmax
from loader.cifar10 import CIFAR10_Loader
from torch_pso import ParticleSwarmOptimizer, AcceleratedPSO

from architecture.effnet import EfficientNet
from objective_function.accuracy_min import MinAccuracy

from tqdm import tqdm
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

EPOCHS = 100
BATCH_SIZE = 32

softmax = Softmax(1)

device = "cuda:0" if cuda.is_available() else "cpu"

print("Using", device)

model = EfficientNet()
model = model.to(device)
model.train()
print(model)

loader = CIFAR10_Loader(BATCH_SIZE)

optimizer = ParticleSwarmOptimizer(model.parameters(), inertial_weight=0.9, num_particles=20, max_param_value=10, min_param_value=-10)
# optimizer = AcceleratedPSO(model.parameters(), num_particles=10, max_param_value=10, min_param_value=-10)

criterion = MinAccuracy()

max_val_acc = 0.0

for epoch in range(EPOCHS):
    tr_accuracy = 0.0
    progress = 0
    with tqdm(loader.train, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for inputs, labels in tepoch:

            inputs, labels = inputs.to(device), labels.to(device)

            # outputs = model(inputs)

            # loss = criterion(outputs, labels)

            def closure():
                optimizer.zero_grad()  
                return criterion(model(inputs), labels)

            # loss.backward()
            optimizer.step(closure)

            # sft_outputs = softmax(outputs)
            # sft_outputs = argmax(sft_outputs, dim=1)
            # tr_accuracy += (sft_outputs == labels).float().sum()
            # progress += 1

            # tepoch.set_postfix({"accuracy":(100-closure().item())})
    
    val_accuracy = 0.0
    for inputs, labels in iter(loader.valid):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        
        max_outputs = argmax(softmax(outputs), dim=1)
        val_accuracy += (max_outputs == labels).float().sum()
    
    val_accuracy = (val_accuracy/loader.val_len).item()*100
    print('\t\tval_accuracy: {0}'.format(val_accuracy))

    if(val_accuracy > max_val_acc):
        max_val_acc = val_accuracy
        torch.save(model.state_dict(), "./models/best_effnet_pso.pth")
        print("\t\tSaved state")

test_accuracy = 0.0
for inputs, labels in iter(loader.test):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    
    max_outputs = argmax(softmax(outputs), dim=1)
    test_accuracy += (max_outputs == labels).float().sum()
    
test_accuracy = (test_accuracy/loader.test_len).item()*100

print("Test Accuracy: {0}".format(test_accuracy))
torch.save(model.state_dict(), "./models/final_effnet_pso.pth")

