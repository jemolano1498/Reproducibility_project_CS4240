import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FCNet(nn.Module):
    """
    Simple fully connected neural network with residual connections in PyTorch.
    Layers are defined in __init__ and forward pass implemented in forward.
    """

    def __init__(self):
        super(FCNet, self).__init__()


        p = 0.4

        self.h_in = nn.Linear(25, 100)
        self.layer_1 = nn.Linear(100, 100)
        self.layer_2 = nn.Linear(100, 100)
        self.layer_3 = nn.Linear(101, 100)
        self.layer_4 = nn.Linear(100, 100)
        self.layer_5 = nn.Linear(100, 100)

        self.do1 = torch.nn.Dropout(p=p)
        self.do2 = torch.nn.Dropout(p=p)
        self.do3 = torch.nn.Dropout(p=p)
        self.do4 = torch.nn.Dropout(p=p)
        self.do5 = torch.nn.Dropout(p=p)
        self.do6 = torch.nn.Dropout(p=p)


        self.fc6 = nn.Linear(100, 100)

    def forward(self, x, t):
        h = self.do1(F.relu(self.h_in(x)))
        h = self.do2(F.relu(self.layer_1(h)))
        h = self.do3(F.relu(self.layer_2(h)))
        h = torch.cat((h,t),2)                           # Concatenating with t
        h = self.do4(F.relu(self.layer_3(h)))
        h = self.do5(F.relu(self.layer_4(h)))
        h = self.do6(F.relu(self.layer_5(h)))
        h = self.fc6(h)

        return h


def train(train_loader, net, optimizer, criterion):
    """
    Trains network for one epoch in batches.

    Args:
        train_loader: Data loader for training set.
        net: Neural network model.
        optimizer: Optimizer (e.g. SGD).
        criterion: Loss function (e.g. cross-entropy loss).
    """

    avg_loss = 0
    correct = 0
    total = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, t = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        temp_in = torch.transpose(inputs,1,2)
        temp_t = torch.transpose(t,0,1).unsqueeze(0)
        outputs = net(temp_in,temp_t)
        temp_label = torch.transpose(labels, 0, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total
