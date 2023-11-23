import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
# from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *


def load_data(image_size = 32):
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root="data/cifar10",
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )
        
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=16,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )
        
    return dataset, dataloader