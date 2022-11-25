from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split, Dataset

class CIFAR10_Loader():

    def __init__(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize(240),
            ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.val_len = 5000
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.train_len = len(trainset)-self.val_len
        trainset, valset = random_split(trainset, [self.train_len, self.val_len])

        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_len = len(testset)

        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self.valid = DataLoader(valset, batch_size=4, shuffle=False)
        self.test = DataLoader(testset, batch_size=4, shuffle=False)

class CIFAR10_Features(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.labels[index]