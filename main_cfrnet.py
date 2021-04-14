from data_loader import *
from dictionary import *
from util import *
import random
import torch
import cfr_net_pytorch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

def init_parameters (flags):
    flags.add_val('loss', 'l2', """Which loss function to use (l1/l2/log)""")
    flags.add_val('n_in', 2, """Number of representation layers. """)
    flags.add_val('n_out', 2, """Number of regression layers. """)
    flags.add_val('p_alpha', 1e-4, """Imbalance regularization param. """)
    flags.add_val('p_lambda', 0.0, """Weight decay regularization parameter. """)
    flags.add_val('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
    flags.add_val('dropout_in', 0.9, """Input layers dropout keep rate. """)
    flags.add_val('dropout_out', 0.9, """Output layers dropout keep rate. """)
    flags.add_val('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
    flags.add_val('lrate', 0.05, """Learning rate. """)
    flags.add_val('decay', 0.5, """RMSProp decay. """)
    flags.add_val('batch_size', 100, """Batch size. """)
    flags.add_val('dim_in', 100, """Pre-representation layer dimensions. """)
    flags.add_val('dim_out', 100, """Post-representation layer dimensions. """)
    flags.add_val('batch_norm', 0, """Whether to use batch normalization. """)
    flags.add_val('normalization', 'none',
                  """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
    flags.add_val('rbf_sigma', 0.1, """RBF MMD sigma """)
    flags.add_val('experiments', 1, """Number of experiments. """)
    flags.add_val('iterations', 2000, """Number of iterations. """)
    flags.add_val('weight_init', 0.01, """Weight initialization scale. """)
    flags.add_val('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
    flags.add_val('wass_iterations', 20, """Number of iterations in Wasserstein computation. """)
    flags.add_val('wass_lambda', 1, """Wasserstein lambda. """)
    flags.add_val('wass_bpt', 0, """Backprop through T matrix? """)
    flags.add_val('varsel', 0, """Whether the first layer performs variable selection. """)
    flags.add_val('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
    flags.add_val('datadir', '../data/topic/csv/', """Data directory. """)
    flags.add_val('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
    flags.add_val('data_test', '', """Test data filename form. """)
    flags.add_val('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
    flags.add_val('seed', 1, """Seed. """)
    flags.add_val('repetitions', 1, """Repetitions with different seed.""")
    flags.add_val('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
    flags.add_val('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
    flags.add_val('imb_fun', 'mmd2_lin',
                  """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
    flags.add_val('output_csv', 0, """Whether to save a CSV file with the results""")
    flags.add_val('output_delay', 100, """Number of iterations between log/loss outputs. """)
    flags.add_val('pred_output_delay', -1,
                  """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
    flags.add_val('debug', 0, """Debug mode. """)
    flags.add_val('save_rep', 0, """Save representations after training. """)
    flags.add_val('val_part', 0.1, """Validation part. """)
    flags.add_val('split_output', 0, """Whether to split output layers between treated and control. """)
    flags.add_val('reweight_sample', 1,
                  """Whether to reweight sample for prediction loss with average treatment probability. """)

def train(net, D, I_valid, flags):
    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n);
    I_train = list(set(I) - set(I_valid))
    n_train = len(I_train)


    factual_tensor = torch.Tensor(D['x'][I_train, :]), torch.Tensor(D['yf'][I_train, :]), torch.Tensor(D['t'][I_train, :])
    cfactual_tensor = torch.Tensor(D['x'][I_valid, :]), torch.Tensor(D['yf'][I_valid, :]), torch.Tensor(D['t'][I_valid, :])
    valid_tensor = torch.Tensor(D['x'][I_train, :]), torch.Tensor(D['ycf'][I_train, :]), torch.Tensor(D['t'][I_train, :])

    # Loss Function
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=flags.get_val('lrate'))

    # Probability of being treated # Must be adapted if we use test- and trainingset
    ''' Compute treatment probability'''
    p_t = torch.mean(factual_tensor[2])

    for i in range(flags.get_val('iterations')):
        ''' Fetch sample '''
        I = random.sample(range(0, n_train), flags.get_val('batch_size'))
        x_batch = D['x'][I_train,:][I,:]
        t_batch = D['t'][I_train,:][I]
        y_batch = D['yf'][I_train,:][I]

        train_tensor = torch.Tensor(x_batch), torch.Tensor(y_batch), torch.Tensor(t_batch)

        # Create dataloaders from the training and test set for easier iteration over the data
        # train_loader = DataLoader(TensorDataset(*train_tensor), batch_size=flags.get_val('batch_size'), shuffle=True)

        obj_loss,f_error,imb_err = cfr_net_pytorch.train(train_tensor, net, optimizer, criterion, p_t, flags)

        if i % flags.get_val('output_delay') == 0 or i == flags.get_val('iterations') - 1:
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tImb: %.2g' \
                       % (obj_loss, f_error, imb_err)
            print(loss_str)


def run():

    flags = Parameters()
    init_parameters(flags)

    net = cfr_net_pytorch.FCNet()
    summary(net, [(25,), (1,)])

    D = load_data('data/ihdp_npci_1-100.train.npz')
    i_exp =1
    D_exp = {}
    D_exp['x'] = D['x'][:, :, i_exp - 1]
    D_exp['t'] = D['t'][:, i_exp - 1:i_exp]
    D_exp['yf'] = D['yf'][:, i_exp - 1:i_exp]
    if D['HAVE_TRUTH']:
        D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]
    else:
        D_exp['ycf'] = None

    I_train, I_valid = validation_split(D_exp, flags.get_val('val_part'))

    losses, preds_train, preds_test, reps, reps_test = train(net, D_exp, I_valid, flags)



if __name__ == '__main__':
    run()