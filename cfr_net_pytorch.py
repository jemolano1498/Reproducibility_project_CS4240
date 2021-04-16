import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import *

split_output = False
class FCNet(nn.Module):
    """
    Simple fully connected neural network with residual connections in PyTorch.
    Layers are defined in __init__ and forward pass implemented in forward.
    """

    def __init__(self):
        super(FCNet, self).__init__()

        p = 0.3

        # Creating the linear layers
        self.h_in = nn.Linear(25, 100)
        self.layer_1 = nn.Linear(100, 100)
        self.layer_2 = nn.Linear(100, 100)
        self.layer_3 = nn.Linear(101, 100)
        self.layer_4 = nn.Linear(100, 100)
        self.layer_5 = nn.Linear(100, 100)

        # Creating the dropout layers
        self.do1 = torch.nn.Dropout(p=p)
        self.do2 = torch.nn.Dropout(p=p)
        self.do3 = torch.nn.Dropout(p=p)
        self.do4 = torch.nn.Dropout(p=p)
        self.do5 = torch.nn.Dropout(p=p)
        self.do6 = torch.nn.Dropout(p=p)

        # Creating the linear classifier layer
        self.fc6 = nn.Linear(100, 1)

    # Forward pass for the first half of the network
    def forward(self, x, t):
        h = self.do1(F.relu(self.h_in(x)))
        h = self.do2(F.relu(self.layer_1(h)))
        h_rep = self.do3(F.relu(self.layer_2(h)))
        h = self._build_output_graph( h, t)
        self.h_rep = h_rep

        return h, h_rep

    # Forward pass for the second half of the network
    def _build_output (self, h):
        h = self.do4(F.relu(self.layer_3(h)))
        h = self.do5(F.relu(self.layer_4(h)))
        h = self.do6(F.relu(self.layer_5(h)))
        h = self.fc6(h)
        return h

    # Function for splitting data OR concatenate with t
    def _build_output_graph(self, h, t):
        ''' Construct output/regression layers '''
        t = torch.round(t)
        if split_output :
            i0 = torch.where(t < 1)
            i1 = torch.where(t > 0)

            rep0 = torch.cat((torch.index_select(h, 0, i0[0]),torch.unsqueeze(i0[1],1)),1)
            rep1 = torch.cat((torch.index_select(h, 0, i1[0]),torch.unsqueeze(i1[1],1)),1)

            y0 = self._build_output(rep0)
            y1 = self._build_output(rep1)

            y = torch.unsqueeze(torch.FloatTensor(dynamic_stitch([i0[0], i1[0]], [y0, y1])),1)
        else:
            h = torch.cat((h,t),1)
            y = self._build_output(h)

        return y

# Test funtion for the Net
def test (train_loader, net, p_t, flags):
    inputs, labels, t = train_loader[0], train_loader[1], train_loader[2]

    r_alpha = flags.get_val('p_alpha')

    # Sample reweighting
    if flags.get_val('reweight_sample'):
        w_t = t / (2 * p_t)
        w_c = (1 - t) / (2 * 1 - p_t)
        sample_weight = w_t + w_c
    else:
        sample_weight = 1.0

    # Imbalance error
    if flags.get_val('use_p_correction'):
        p_ipm = p_t
    else:
        p_ipm = 0.5

    with torch.no_grad():
        # forward
        outputs, h_rep = net(inputs, t)
        if flags.get_val('normalization') == 'divide':  # normalization set to none default
            h_rep_norm = h_rep / torch.sqrt((torch.sum(torch.square(h_rep), 1,
                                                       keepdim=True)))  # Not sure if this works.. but we don't use normalisation
        else:
            h_rep_norm = 1.0 * h_rep
        imb_error = get_imbalance_error(h_rep_norm, t, p_ipm, r_alpha, flags)
        pred_loss = torch.square(torch.mean(torch.square(outputs - labels)))
        risk = torch.mean(sample_weight * torch.square(outputs - labels))

    loss = imb_error + risk

    return loss, pred_loss, imb_error

# Train funtion for the Net
def train(train_loader, net, optimizer, p_t, flags):

    # get the inputs; data is a list of [inputs, labels]
    inputs, labels, t = train_loader[0], train_loader[1] , train_loader [2]

    # zero the parameter gradients
    optimizer.zero_grad()

    # Sample reweighting
    if flags.get_val('reweight_sample'):
        w_t = t / (2 * p_t)
        w_c = (1 - t) / (2 * 1 - p_t)
        sample_weight = w_t + w_c
    else:
        sample_weight = 1.0

    # forward
    outputs,h_rep = net(inputs,t)

    # Normalisation of h_rep
    if flags.get_val('normalization') == 'divide':  # normalization set to none default
        h_rep_norm = h_rep / torch.sqrt((torch.sum(torch.square(h_rep), 1,keepdim=True))) # Not sure if this works.. but we don't use normalisation
    else:
        h_rep_norm = 1.0 * h_rep

    r_alpha = flags.get_val('p_alpha')

    # Imbalance error
    if flags.get_val('use_p_correction'):
        p_ipm = p_t
    else:
        p_ipm = 0.5

    imb_error = get_imbalance_error(h_rep_norm, t, p_ipm, r_alpha, flags)

    # Backward + optimize
    # temp_pred_loss = criterion(outputs, labels)         # Called the risk in the original file
    pred_loss = torch.square(torch.mean(torch.square(outputs - labels)))
    risk = torch.mean(sample_weight * torch.square(outputs - labels))

    loss = imb_error + risk

    loss.backward()
    optimizer.step()

    return loss, pred_loss, imb_error

# Imbalance error calculation
def get_imbalance_error (h_rep_norm, t, p_ipm, r_alpha, flags):
    # I just copy pasted this code from the orignial CFR net to remind you which imbalance functiions we should use
    if flags.get_val('imb_fun') == 'mmd2_lin':
        imb_error = 0
        imb_dist = mmd2_lin(h_rep_norm, t, p_ipm)
        imb_error = safe_sqrt(torch.square(torch.tensor(r_alpha)) * imb_dist)

    elif flags.get_val('imb_fun') == 'wass':
        imb_error = 0
        imb_dist, imb_mat = wasserstein(h_rep_norm, t, p_ipm, lam=flags.get_val('wass_lambda'),
                                        its=flags.get_val('wass_iterations'), sq=False,
                                        backpropT=flags.get_val('wass_bpt'))
        imb_error = r_alpha * imb_dist

    else:
        imb_error = 0

    return imb_error