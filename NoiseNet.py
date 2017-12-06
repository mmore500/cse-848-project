# Import PyTorch Utilities
import torch
import torchvision
import torch.nn as nn


# Define the denoising autoencoder architecture

class NoiseNet(nn.Module):
    def __init__(self):
        super(NoiseNet, self).__init__()
        self.fc1   = nn.Linear(100, 100, bias=False)
        # self.nl1   = nn.ReLU()
        # self.fc2   = nn.Linear(100, 100)
        # self.nl2   = nn.ReLU()
        # self.fc3   = nn.Linear(100, 100)
        # self.nl3   = nn.Tanh()
        # self.fc4   = nn.Linear(100, 100)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.nl1(out)
        # out = self.fc2(out)
        # out = self.nl2(out)
        # out = self.fc3(out)
        # out = self.nl3(out)
        # out = self.fc4(out)
        return out

    def forwardverbose(self, x):
        out = self.forward(x)
        print(out.data.numpy())
        return out

    def forwardtimes(self, x):
        out = self.forward(x) * 1000
        return out
