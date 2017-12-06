from torch.autograd import Variable

def train(epoch, train_loader, model, optimizer, criterion, log_interval, noise=False):
    model.train()
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):

        data, target = Variable(data).float(), Variable(data).float()

        # make sure gradients are reset to zero.
        optimizer.zero_grad()

        if noise:
            output =  model(data + Variable(torch.randn(data.size()) * 1))
        else:
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
