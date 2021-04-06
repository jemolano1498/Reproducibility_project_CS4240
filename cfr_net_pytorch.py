import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import *

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
        h_rep = self.do3(F.relu(self.layer_2(h)))
        h = torch.cat((h_rep,t),2)                           # Concatenating with t
        h = self.do4(F.relu(self.layer_3(h)))
        h = self.do5(F.relu(self.layer_4(h)))
        h = self.do6(F.relu(self.layer_5(h)))
        h = self.fc6(h)

        return h, h_rep


def train(train_loader, net, optimizer, criterion, p_t, flags):
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

        # Sample reweighting
        if flags.get_val('reweight_sample'):
            w_t = t / (2 * p_t)
            w_c = (1 - t) / (2 * 1 - p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        temp_in = torch.transpose(inputs,1,2)
        temp_t = torch.transpose(t,0,1).unsqueeze(0)
        outputs,h_rep = net(temp_in,temp_t)
        temp_label = torch.transpose(labels, 0, 1)

        # Normalisation of h_rep
        if flags.get_val('normalization') == 'divide':  # normalization set to none default
            h_rep_norm = h_rep / torch.sqrt((torch.sum(torch.square(h_rep), 1,keepdim=True))) # Not sure if this works.. but we don't use normalisation
        else:
            h_rep_norm = 1.0 * h_rep

        r_alpha = 0.05 # No idea what this should be

        # Imbalance error
        if flags.get_val('use_p_correction'):
            p_ipm = p_t
        else:
            p_ipm = 0.5

            # I just copy pasted this code from the orignial CFR net to remind you which imbalance functiions we should use
        if flags.get_val('imb_fun') == 'mmd2_lin':
            imb_error = 0
            #imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
            #imb_error = safe_sqrt(torch.square(r_alpha) * imb_dist)
        elif flags.get_val('imb_fun') == 'wass':
            imb_error = 0
            #imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=flags.get_val('wass_lambda'), its=flags.get_val('wass_iterations'), sq=False, backpropT=flags.get_val('wass_bpt'))
            #imb_error = r_alpha * imb_dist
        else:
            imb_error = 0

        # Backward + optimize
        loss = criterion(outputs*sample_weight, labels)         # Called the risk in the original file
        loss = loss + imb_error
        pred_error = torch.sqrt(criterion(outputs,labels))
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return avg_loss / len(train_loader), 100 * correct / total, outputs
