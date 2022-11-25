import torch
from torch import cuda
import torch.optim as optim
from torch.nn import CrossEntropyLoss, Softmax
from torch import argmax
from loader.cifar10 import CIFAR10_Loader
from training_logger import Train_Logger

from architecture.effnet import EfficientNet

from tqdm import tqdm
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

def train_sgd(model, loader, logger, criterion, device, epochs, model_name):
    print("Using", device)

    model = model.to(device)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    softmax = Softmax(1)

    max_val_acc = 0.0

    for epoch in range(epochs):
        tr_accuracy = 0.0
        tr_loss = 0.0
        progress = 0
        with tqdm(loader.train, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                tr_loss += loss.item()

                loss.backward()
                optimizer.step()

                sft_outputs = softmax(outputs)
                sft_outputs = argmax(sft_outputs, dim=1)
                tr_accuracy += (sft_outputs == labels).float().sum().item()
                progress += 1

                tepoch.set_postfix({"loss":loss.item(), "accuracy":(100*tr_accuracy/loader.train_len)})
        
        tr_accuracy = 100*tr_accuracy/loader.train_len
        
        val_accuracy = 0.0
        for inputs, labels in iter(loader.valid):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            
            max_outputs = argmax(softmax(outputs), dim=1)
            val_accuracy += (max_outputs == labels).float().sum().item()
        
        val_accuracy = (val_accuracy/loader.val_len)*100
        print('\t\tval_accuracy: {0}'.format(val_accuracy))

        logger.put(epoch=epoch, tloss=tr_loss, taccuracy=tr_accuracy, vaccuracy=val_accuracy)

        if(val_accuracy > max_val_acc):
            max_val_acc = val_accuracy
            torch.save(model.state_dict(), f"./models/best_{model_name}.pth")
            print("\t\tSaved state")

    test_accuracy = 0.0
    for inputs, labels in iter(loader.test):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        
        max_outputs = argmax(softmax(outputs), dim=1)
        test_accuracy += (max_outputs == labels).float().sum()
        
    test_accuracy = (test_accuracy/loader.test_len).item()*100

    print("Test Accuracy: {0}".format(test_accuracy))
    torch.save(model.state_dict(), f"./models/final_{model_name}.pth")

    return model

if __name__ == "__main__":
    EPOCHS = 50
    BATCH_SIZE = 16
    MODEL = EfficientNet()
    LOADER = CIFAR10_Loader(BATCH_SIZE)
    CRITERION = CrossEntropyLoss()
    DEVICE = "cuda:0" if cuda.is_available() else "cpu"
    MODEL_NAME = "effnet_sgd"
    train_sgd(model=MODEL, loader=LOADER, criterion=CRITERION, device=DEVICE, epochs=EPOCHS, model_name=MODEL_NAME)