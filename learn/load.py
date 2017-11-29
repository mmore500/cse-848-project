# Import PyTorch Utilities
import torch
import torchvision
import torch.nn as nn

# PyTorch supports loading many commonly used datasets

def load():
    batch_size = 16

    train_loader = torch.utils.data.DataLoader(
            torch.load('../data/traindata.pt')/1000,
            batch_size=batch_size,
            shuffle=True
        )

    return train_loader
