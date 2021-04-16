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
from evaluate import *

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
    flags.add_val('outdir', '../results/', """Output directory. """)
    flags.add_val('datadir', '../data/topic/csv/', """Data directory. """)
    flags.add_val('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
    flags.add_val('data_test', 'data/ihdp_npci_1-100.test.npz', """Test data filename form. """)
    flags.add_val('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
    flags.add_val('seed', 1, """Seed. """)
    flags.add_val('repetitions', 1, """Repetitions with different seed.""")
    flags.add_val('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
    flags.add_val('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
    # flags.add_val('imb_fun', 'mmd2_lin',"""Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
    flags.add_val('imb_fun', 'wass',"""Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
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

def train(net, D, D_test, I_valid, flags, i_exp):

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n);
    I_train = list(set(I) - set(I_valid))
    n_train = len(I_train)



    factual_tensor = torch.Tensor(D['x'][I_train, :]), torch.Tensor(D['yf'][I_train, :]), torch.Tensor(D['t'][I_train, :])
    valid_tensor = torch.Tensor(D['x'][I_valid, :]), torch.Tensor(D['yf'][I_valid, :]), torch.Tensor(D['t'][I_valid, :])
    cfactual_tensor = torch.Tensor(D['x'][I_train, :]), torch.Tensor(D['ycf'][I_train, :]), torch.Tensor(D['t'][I_train, :])
    test_factual_tensor = torch.Tensor(D_test['x']), torch.Tensor(D_test['yf']), torch.Tensor(D_test['t'])
    test_cfactual_tensor = torch.Tensor(D_test['x']), torch.Tensor(D_test['ycf']), torch.Tensor(D_test['t'])

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=flags.get_val('lrate'))

    # Probability of being treated # Must be adapted if we use test- and trainingset
    ''' Compute treatment probability'''
    p_t = torch.mean(factual_tensor[2])

    reps = []
    reps_test = []

    for i in range(flags.get_val('iterations')):
        ''' Fetch sample '''
        I = random.sample(range(0, n_train), flags.get_val('batch_size'))
        x_batch = D['x'][I_train,:][I,:]
        t_batch = D['t'][I_train,:][I]
        y_batch = D['yf'][I_train,:][I]

        train_tensor = torch.Tensor(x_batch), torch.Tensor(y_batch), torch.Tensor(t_batch)

        cfr_net_pytorch.train(train_tensor, net, optimizer, p_t, flags)

        if i % flags.get_val('output_delay') == 0 or i == flags.get_val('iterations') - 1:
            obj_loss,f_error,imb_err = cfr_net_pytorch.test(factual_tensor, net, p_t, flags)
            _, cf_error, _ = cfr_net_pytorch.test(cfactual_tensor, net, p_t, flags)
            valid_obj, valid_f_error, valid_imb = cfr_net_pytorch.test(valid_tensor, net, p_t, flags)
            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' \
                       % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)
            print(loss_str)

        ''' Compute predictions every M iterations '''
        if (flags.get_val('output_delay')> 0 and i % flags.get_val('pred_output_delay') == 0) or i == flags.get_val('iterations') - 1:
            with torch.no_grad():
                y_pred_f, _ = net(torch.Tensor(D['x']), torch.Tensor(D['t']))
                y_pred_cf, _ = net(torch.Tensor(D['x']), torch.Tensor(1 - D['t']))

            preds_train.append(np.concatenate((y_pred_f, y_pred_cf), axis=1))

            if D_test is not None:
                with torch.no_grad():
                    y_pred_f_test, _ = net(torch.Tensor(D_test['x']), torch.Tensor(D_test['t']))
                    y_pred_cf_test, _ = net(torch.Tensor(D_test['x']), torch.Tensor(1 - D_test['t']))
                    preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test), axis=1))

            if flags.get_val('save_rep') and i_exp == 1:
                with torch.no_grad():
                    _, reps_i = net(torch.Tensor(D['x']), torch.Tensor(D['t']))
                reps.append(reps_i)

                if D_test is not None:
                    with torch.no_grad():
                        _, reps_test_i = net(torch.Tensor(D_test['x']), torch.Tensor(D_test['t']))
                    reps_test.append(reps_test_i)

    return losses, preds_train, preds_test, reps, reps_test

def run():

    flags = Parameters()
    ''' Save parameters '''
    outdir = 'results/testing/'
    init_parameters(flags)
    save_config(outdir + 'config.txt', flags)

    npzfile = outdir + 'result'
    npzfile_test = outdir + 'result.test'
    repfile = outdir + 'reps'
    repfile_test = outdir + 'reps.test'

    net = cfr_net_pytorch.FCNet()
    summary(net, [(25,), (1,)])

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []

    has_test = False
    if not flags.get_val('data_test') == '':  # if test set supplied
        has_test = True
        dataform_test = flags.get_val('data_test')

    D = load_data('data/ihdp_npci_1-100.train.npz')
    D_test = load_data(dataform_test)
    for i_exp in range(1, flags.get_val('repetitions') + 1):
        D_exp = {}
        D_exp['x'] = D['x'][:, :, i_exp - 1]
        D_exp['t'] = D['t'][:, i_exp - 1:i_exp]
        D_exp['yf'] = D['yf'][:, i_exp - 1:i_exp]
        if D['HAVE_TRUTH']:
            D_exp['ycf'] = D['ycf'][:, i_exp - 1:i_exp]
        else:
            D_exp['ycf'] = None

        if has_test:
            D_exp_test = {}
            D_exp_test['x'] = D_test['x'][:, :, i_exp - 1]
            D_exp_test['t'] = D_test['t'][:, i_exp - 1:i_exp]
            D_exp_test['yf'] = D_test['yf'][:, i_exp - 1:i_exp]
            if D_test['HAVE_TRUTH']:
                D_exp_test['ycf'] = D_test['ycf'][:, i_exp - 1:i_exp]
            else:
                D_exp_test['ycf'] = None

        I_train, I_valid = validation_split(D_exp, flags.get_val('val_part'))

        losses, preds_train, preds_test, reps, reps_test = train(net, D_exp, D_exp_test, I_valid, flags, i_exp)

        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train, 1, 3), 0, 2)
        if has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test, 1, 3), 0, 2)
        out_losses = np.swapaxes(np.swapaxes(all_losses, 0, 2), 0, 1)

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if flags.get_val('save_rep') and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)



if __name__ == '__main__':
    run()
    evaluate('configs/example_ihdp.txt')