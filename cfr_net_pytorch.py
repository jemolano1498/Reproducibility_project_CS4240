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

    def __init__(self, dim_in, n_in, dim_out, n_out, dropout_in, dropout_out):
        super(FCNet, self).__init__()

        self.in_layers=[]
        self.in_drop_layers = []
        self.out_layers = []
        self.out_drop_layers = []

        self.h_in = nn.Linear(25, dim_in)
        self.do1 = torch.nn.Dropout(p=dropout_in)

        for i in range(n_in):
            self.in_layers.append(nn.Linear(dim_in, dim_in))
            self.in_drop_layers.append(torch.nn.Dropout(p=dropout_in))

        if not split_output:
            self.out_layers.append(nn.Linear(dim_out+1, dim_out))
        else:
            self.out_layers.append(nn.Linear(dim_out, dim_out))
        self.out_drop_layers.append(torch.nn.Dropout(p=dropout_out))

        for i in range(n_out):
            self.out_layers.append(nn.Linear(dim_out, dim_out))
            self.out_drop_layers.append(torch.nn.Dropout(p=dropout_out))

        self.h_out = nn.Linear(dim_out, 1)

    def forward(self, x, t):
        h = self.do1(F.relu(self.h_in(x)))
        for i in range(len(self.in_layers)):
            h = self.in_drop_layers[i](F.relu(self.in_layers[i](h)))

        h_rep = h
        h = self._build_output_graph( h, t)
        self.h_rep = h_rep

        return h, h_rep

    def _build_output (self, h):
        for i in range(len(self.out_layers)):
            h = self.out_drop_layers[i](F.relu(self.out_layers[i](h)))
        h = self.h_out(h)
        return h

    def _build_output_graph(self, h, t):
        ''' Construct output/regression layers '''
        t = torch.round(t)
        if split_output :
            i0 = torch.where(t < 1)
            i1 = torch.where(t > 0)

            rep0 = torch.index_select(h, 0, i0[0])
            rep1 = torch.index_select(h, 0, i1[0])

            y0 = self._build_output(rep0)
            y1 = self._build_output(rep1)

            y = torch.unsqueeze(torch.FloatTensor(dynamic_stitch([i0[0], i1[0]], [y0, y1])),1)
        else:
            h = torch.cat((h,t),1)
            y = self._build_output(h)

        return y

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