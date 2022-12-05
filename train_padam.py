import torch
from loader.cifar10 import CIFAR10_Loader
from architecture.effnet import EfficientNet
from training_logger import Train_Logger
from torch.nn import CrossEntropyLoss, Softmax, Module
from torch import argmax
from torch import cuda
from tqdm import tqdm
from padam import Padam


optim_params = {
    'padam': {
        'weight_decay': 0.0005,
        'lr': 0.1,
        'p': 0.125,
        'betas': (0.9, 0.999),
        'color': 'darkred',
        'linestyle':'-'
    }
}

padam_params = {
    'weight_decay': [0.0001, 0.0005, 0.001],
    'lr': [0.0001, 0.001, 0.01],
    'p': [0, 0.25, 0.5],
    'betas': [(0.9, 0.999)]
}

def train_padam(model: Module, loader, logger, criterion, device, epochs, model_name, hyperparams):
    print("Using", device)
    model = model.to(device)
    model.train()
    learning_rate= 0.1
    optimizer = Padam(model.parameters(), lr=learning_rate, partial=hyperparams['p'], betas=hyperparams['betas'])

    softmax = Softmax(1)

    max_val_acc = 0.0

    for epoch in range(epochs):
        tr_accuracy = 0.0
        tr_loss = 0.0
        progress = 0

        model.requires_grad_(True)

        with tqdm(loader.train, unit="batch") as tepoch:
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

                tepoch.set_postfix({"loss":loss.item(), "accuracy":(100*((tr_accuracy.item())/loader.train_len))})

        model.requires_grad_(False)
        val_accuracy = 0.0    
        for inputs, labels in iter(loader.valid):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            
            max_outputs = argmax(softmax(outputs), dim=1)
            val_accuracy += (max_outputs == labels).float().sum()
        
        val_accuracy = (val_accuracy/loader.val_len).item()*100
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
    EPOCHS = 100
    BATCH_SIZE = 32
    MODEL = EfficientNet()
    LOADER = CIFAR10_Loader(BATCH_SIZE)
    LOGGER = Train_Logger("padam")
    CRITERION = CrossEntropyLoss()
    DEVICE = "cuda:0" if cuda.is_available() else "cpu"
    MODEL_NAME = "effnet_padam"
    train_padam(model=MODEL, loader=LOADER, logger=LOGGER, criterion=CRITERION, device=DEVICE, epochs=EPOCHS, model_name=MODEL_NAME)