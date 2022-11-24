from train_sgd import train_sgd
from architecture.effnet import EfficientNet
from loader.cifar10 import CIFAR10_Loader
from EffNetPSO import PSO
from torch.nn import Softmax, CrossEntropyLoss
from torch import argmax
from torch import cuda
from tqdm import tqdm

PSO_EPOCHS = 10
SGD_EPOCHS = 40

BATCH_SIZE = 16

NUM_PARTICLES = None

MODEL_NAME = "effnet_pso"

device = "cuda:0" if cuda.is_available() else "cpu"

model = EfficientNet()

criterion = CrossEntropyLoss()
loader = CIFAR10_Loader(BATCH_SIZE)

# Lock classsifier head
model.train_backbone(True)

model = train_sgd(model=model, loader=loader, criterion=criterion, device=device, epochs=PSO_EPOCHS, model_name=MODEL_NAME)

# Lock everything apart from classifier head
model.train_backbone(False)

pso_optimizer = PSO(num_particles=NUM_PARTICLES, net=model.backbone.classifier, device=device)

softmax = Softmax(1)

for epoch in range(PSO_EPOCHS):
    def closure(particle_id):
        tr_accuracy = 0.0
        tr_loss = 0.0
        with tqdm(loader.train, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}_{particle_id}")

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                tr_loss += criterion(outputs, labels)

                sft_outputs = softmax(outputs)
                sft_outputs = argmax(sft_outputs, dim=1)
                tr_accuracy += (sft_outputs == labels).float().sum()

                tepoch.set_postfix({"loss":tr_loss.item(), "accuracy":(100*((tr_accuracy.item())/loader.train_len))})

        return tr_loss/loader.train_len, tr_accuracy/loader.train_len

    pso_optimizer.step(closure=closure)

    val_accuracy = 0.0
    for inputs, labels in iter(loader.valid):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        sft_outputs = softmax(outputs)
        sft_outputs = argmax(sft_outputs, dim=1)
        val_accuracy += (sft_outputs == labels).float().sum()

    val_accuracy = 100*val_accuracy.item()/loader.val_len
    print("\t\tValid Accuracy:", val_accuracy)

test_accuracy = 0.0
for inputs, labels in iter(loader.test):
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(inputs)
    sft_outputs = softmax(outputs)
    sft_outputs = argmax(sft_outputs, dim=1)
    test_accuracy += (sft_outputs == labels).float().sum()

val_accuracy = 100*test_accuracy.item()/loader.test_len
print("\t\tTest Accuracy:", test_accuracy)




    
