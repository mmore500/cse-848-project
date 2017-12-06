# CNN Example on MNIST

# Import PyTorch Utilities
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from NoiseNet import NoiseNet

from learn.load import load_noise as load
from learn.train import train

from torch.nn.init import uniform

train_loader = load()

# Initialize Model

model = NoiseNet()
uniform(model.fc1.weight.data, a = 0.005, b = 0.015)
criterion = nn.MSELoss()

# setup optimization routine
learning_rate = 1e-4
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Training Function

log_interval = 2048
# Actual Loop

nepochs = 80000

data = next(enumerate(train_loader))[1]

data, target = Variable(data).float(), Variable(data).float()
# make sure gradients are reset to zero.
optimizer.zero_grad()
output = model.forwardverbose(data)
loss = criterion(output, target)
print(loss.data.numpy())

try:
    for epoch in range(nepochs):
        train(epoch, train_loader, model, optimizer, criterion, log_interval, noise=True)
except KeyboardInterrupt:
    pass


data = next(enumerate(train_loader))[1]

data, target = Variable(data).float(), Variable(data).float()

# make sure gradients are reset to zero.
optimizer.zero_grad()
output = model.forwardverbose(data + Variable(torch.randn(data.size()) * 1))

print(model.forwardtimes(data).data.numpy())

torch.save(model, "data/noise_model.pt")
