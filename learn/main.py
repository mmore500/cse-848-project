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

from Network import Network

from load import load

train_loader = load()

# Initialize Model

model = Network()
criterion = nn.MSELoss()

# setup optimization routine
learning_rate = 1e-3
momentum = 0.7
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


# Training Function

log_interval = 1024
def train(epoch):
    model.train()
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):

        data, target = Variable(data).float(), Variable(data).float()

        # make sure gradients are reset to zero.
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)

        cur_loss = loss.data[0]
        total_loss += cur_loss

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cur_loss))

# Actual Loop

nepochs = 1000

data = next(enumerate(train_loader))[1]

data, target = Variable(data).float(), Variable(data).float()
# make sure gradients are reset to zero.
optimizer.zero_grad()
output = model.forwardverbose(data)
loss = criterion(output, target)
print(loss.data.numpy())

try:
    for epoch in range(nepochs):
        train(epoch)
except KeyboardInterrupt:
    pass


data = next(enumerate(train_loader))[1]

data, target = Variable(data).float(), Variable(data).float()

# make sure gradients are reset to zero.
optimizer.zero_grad()
output = model.forwardverbose(data)

print(model.forwardtimes(data).data.numpy())

torch.save(model, "../data/model.pt")
