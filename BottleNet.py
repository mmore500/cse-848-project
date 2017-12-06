# Import PyTorch Utilities
import torch
import torchvision
import torch.nn as nn


# Define the bottlenecked autoencoder architecture

class BottleNet(nn.Module):
    def __init__(self):
        super(BottleNet, self).__init__()
        self.fc1   = nn.Linear(100, 1)
        self.fc2   = nn.Linear(1, 100)

    def forward(self, x):
        out = self.fc1(x)
        return self.codeforward(out)

    def codeforward(self, x):
        out = self.fc2(x)
        return out

    def forwardverbose(self, x):
        out = self.fc1(x)
        print(out.data.numpy())
        return self.codeforward(out)

    def forwardtimes(self, x):
        out = self.forward(x) * 1000
        return out
