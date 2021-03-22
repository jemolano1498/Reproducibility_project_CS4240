import tensorflow as tf
import numpy as np
import torch 
import torch.nn as nn

from util import *
from dictionary import flags

class cfr_net(object):
    """
    cfr_net implements the counterfactual regression neural network
    by F. Johansson, U. Shalit and D. Sontag: https://arxiv.org/abs/1606.03976

    This file contains the class cfr_net as well as helper functions.
    The network is implemented as a tensorflow graph. The class constructor
    creates an object containing relevant TF nodes as member variables.
    """

    def __init__(self, x, t, y_ , p_t, r_alpha, r_lambda, do_in, do_out, dims):              #we don't need the FLAGS anymore right?
        self.variables = {}
        self.wd_loss = 0

        if flags.get_val('nonlin') == 'elu':
            self.nonlin = torch.nn.ELU         # Exponential activation layer, in pytorch: torch.nn.ELU
        else:
            self.nonlin = torch.nn.ReLU        # Relu activation function layer, in pytorch: torch.nn.ReLU 

        self._build_graph(x, t, y_ , p_t, r_alpha, r_lambda, do_in, do_out, dims)

    def _add_variable(self, var, name):
        ''' Adds variables to the internal track-keeper '''
        basename = name
        i = 0
        while name in self.variables:
            name = '%s_%d' % (basename, i) #@TODO: not consistent with TF internally if changed
            i += 1

        self.variables[name] = var

    def _create_variable(self, var, name):
        ''' Create and adds variables to the internal track-keeper '''

        var = torch.tensor(var, name=name)      #Sort of tensor in pytorch
        self._add_variable(var, name)
        return var

    def _create_variable_with_weight_decay(self, initializer, name, wd):
        ''' Create and adds variables to the internal track-keeper
            and adds it to the list of weight decayed variables '''
        var = self._create_variable(initializer, name)
        self.wd_loss += wd*(torch.sum(var**2)/2)  # loss with the L2 
        return var

    def _build_graph(self, x, t, y_ , p_t, r_alpha, r_lambda, do_in, do_out, dims):
        """
        Constructs a TensorFlow subgraph for counterfactual regression.
        Sets the following member variables (to TF nodes):

        self.output         The output prediction "y"
        self.tot_loss       The total objective to minimize
        self.imb_loss       The imbalance term of the objective
        self.pred_loss      The prediction term of the objective
        self.weights_in     The input/representation layer weights
        self.weights_out    The output/post-representation layer weights
        self.weights_pred   The (linear) prediction layer weights
        self.h_rep          The layer of the penalized representation
        """

        self.x = x
        self.t = t
        self.y_ = y_
        self.p_t = p_t
        self.r_alpha = r_alpha
        self.r_lambda = r_lambda
        self.do_in = do_in                                              # dropout rate
        self.do_out = do_out

        dim_input = dims[0]                                             # Where is this dims list?
        dim_in = dims[1]
        dim_out = dims[2]

        weights_in = []; biases_in = []                                 # creating empty lists for the weights in and biases with 

        if flag.get_val('n_in') == 0 or (flag.get_val('n_in') == 1 and flag.get_val('varsel')):       # n_in is number of representation layers is set to 2
            dim_in = dim_input
        if flag.get_val('n_out') == 0:                                            # n_out is number of regression layers is set to 2
            if flag.get_val('split_output') == False:                             # split_output, wheter or not to split output layers between treated and control --> default = 0
                dim_out = dim_in+1
            else:
                dim_out = dim_in

        if flag.get_val('batch_norm'):                                            # batchnorm wheter or not to use batch normalisation --> default = 0
            bn_biases = []
            bn_scales = []

        ''' Construct input/representation layers '''
        h_in = [x]
        for i in range(0, flag.get_val(n_in):
            if i==0:
                ''' If using variable selection, first layer is just rescaling'''
                if flag.get_val('varsel'):                                         # varsel tells us if you do variable selection in the first layer, defeault it is set to 0 
                    weights_in.append(torch.tensor(1.0/dim_input*torch.ones([dim_input])))
                else:
                    weights_in.append(torch.tensor(torch.randn([dim_input, dim_in], stddev=flags.add_val('weight_init')/np.sqrt(dim_input))))      # random normal --> pytroch: torch.randn
            else:
                weights_in.append(torch.tensor(torch.randn([dim_in,dim_in], stddev=flags.add_val('weight_init')/np.sqrt(dim_in))))

            ''' If using variable selection, first layer is just rescaling'''
            if flags.add_val('varsel') and i==0:
                biases_in.append([])
                h_in.append(torch.mul(h_in[i],weights_in[i]))                  # tf.mul --> matrix mutiplication --> torch.mul
            else:
                biases_in.append(tf.Variable(torch.zeros([1,dim_in])))         # tf.zeros --> torch.zeros
                z = torch.matmul(h_in[i], weights_in[i]) + biases_in[i]        # tf.matmul --> element wise multiplication --> torch.matmul

                if flags.add_val('batch_norm'):                                        # batch_norm wheter or not to use batch normalisation, is set to 0 --> no 
                    batch_mean = torch.mean(z)
                    batch_var = torch.std(z)           # tf.nn.moments --> calculate the mean and variance; in pytorch: torch.std() + torch.mean() --> not possible to compute them together 

                    if flags.add_val('normalization) == 'bn_fixed':                   # normalise after batchnorm, set to 'none' 
                        z = torch.nn.Batchnorm1(z, batch_mean, batch_var, 0, 1, 1e-3) # tf.nn.batch_normalization, normalisation of the batch with variance and mean, torch.nn.Batchnorm1 (for 2D/3D) and Batchnomr2 (for 4D), computes directly the mean and variance so perhaps is calculation in the step above not necesarry 
                    else:
                        bn_biases.append(torch.tensor(torch.zeros([dim_in])))
                        bn_scales.append(torch.tensor(torch.ones([dim_in])))
                        z = torch.nn.Batchnorm1(z, batch_mean, batch_var, bn_biases[-1], bn_scales[-1], 1e-3)

                h_in.append(self.nonlin(z))
                h_in[i+1] = torch.nn.Dropout(h_in[i+1], do_in)                 # tf.nn.dropout creates dropout whith rate do_in --> pytorch: torch.nn.Dropout with rate, see documentation for implementation since it is a bit different from the tensorflow module
                                                                                # I hope do_in is the dropout rate :)
        h_rep = h_in[len(h_in)-1]

        if flags.add_val('normalization') == 'divide':                                 # normalization set to none default
            h_rep_norm = h_rep / safe_sqrt(torch.sum(torch.square(h_rep),1, keep_dims=True))     #tf.reduce_sum computes the sum of elements over a dimension, torch.sum gives the same but you have to give a dimension otherwise you get the summation of all elements, I think the dimension is 1 (second argument)
        else:
            h_rep_norm = 1.0*h_rep

        ''' Construct ouput layers '''
        y, weights_out, weights_pred = self._build_output_graph(h_rep_norm, t, dim_in, dim_out, do_out)

        ''' Compute sample reweighting '''
        if flags.add_val('reweight_sample'):           # wheter or not to reweight the sample with average before calculation of the loss, set to 1 
            w_t = t/(2*p_t)
            w_c = (1-t)/(2*1-p_t)
            sample_weight = w_t + w_c
        else:
            sample_weight = 1.0

        self.sample_weight = sample_weight

        ''' Construct factual loss function '''
        if flags.add_val('loss') == 'l1':                                  # loss definition
            risk = torch.mean(sample_weight*torch.abs(y_-y))       # tf.reduce_mean() computes the mean across the dimension in pytorch it is torch.mean(x,dim) 
            pred_error = -torch.mean(res)                           # maybe include a dimension to compute the mean over 
        elif FLAGS.loss == 'log':
            y = 0.995/(1.0+torch.exp(-y)) + 0.0025
            res = y_*torch.log(y) + (1.0-y_)*torch.log(1.0-y)

            risk = -torch.mean(sample_weight*res)
            pred_error = -torch.mean(res)
        else:
            risk = torch.mean(sample_weight*torch.square(y_ - y))
            pred_error = torch.sqrt(torch.mean(torch.square(y_ - y)))

        ''' Regularization '''
        if flags.add_val('p_lambda')>0 and flags.add_val('rep_weight_decay'):
            for i in range(0, flags.add_val('n_in')):
                if not (flags.add_val('varsel') and i==0): # No penalty on W in variable selection
                    self.wd_loss += torch.sum(weights_in[i]**2) / 2       #computes half the L2 loss 

        ''' Imbalance error '''
        if flags.add_val('use_p_correction'):
            p_ipm = self.p_t
        else:
            p_ipm = 0.5

        if flags.add_val('imb_fun') == 'mmd2_rbf':                 #imb.fun --> selection of an imbalance penalty
            imb_dist = mmd2_rbf(h_rep_norm,t,p_ipm,flags.add_val('rbf_sigma'))
            imb_error = r_alpha*imb_dist
        elif flags.add_val('imb_fun') == 'mmd2_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = r_alpha*mmd2_lin(h_rep_norm,t,p_ipm)
        elif flags.add_val('imb_fun') == 'mmd_rbf':
            imb_dist = torch.abs(mmd2_rbf(h_rep_norm,t,p_ipm,flags.add_val('rbf_sigma')))
            imb_error = safe_sqrt(torch.square(r_alpha)*imb_dist)
        elif flags.add_val('imb_fun') == 'mmd_lin':
            imb_dist = mmd2_lin(h_rep_norm,t,p_ipm)
            imb_error = safe_sqrt(torch.square(r_alpha)*imb_dist)
        elif flags.add_val('imb_fun') == 'wass':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=flags.add_val('wass_lambda'),its=flags.add_val('wass_iterations'),sq=False,backpropT=flags.add_val('wass_bpt'))
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        elif flags.add_val('imb_fun') == 'wass2':
            imb_dist, imb_mat = wasserstein(h_rep_norm,t,p_ipm,lam=flags.add_val('wass_lambda'),its=flags.add_val('wass_iterations'),sq=True,backpropT=flags.add_val('wass_bpt'))
            imb_error = r_alpha * imb_dist
            self.imb_mat = imb_mat # FOR DEBUG
        else:
            imb_dist = lindisc(h_rep_norm,p_ipm,t)
            imb_error = r_alpha * imb_dist

        ''' Total error '''
        tot_error = risk

        if flags.add_val('p_alpha')>0:
            tot_error = tot_error + imb_error

        if flags.add_val('p_alpha')>0:
            tot_error = tot_error + r_lambda*self.wd_loss

        if flags.add_val('varsel'):                                        #if the first layer uses variable selection
            self.w_proj = tf.tensor("float", shape=[dim_input], name='w_proj') #pytorch doesn't work with placeholders.. so I created a tensor instead
            self.projection = weights_in[0].assign(self.w_proj)

        self.output = y
        self.tot_loss = tot_error
        self.imb_loss = imb_error
        self.imb_dist = imb_dist
        self.pred_loss = pred_error
        self.weights_in = weights_in
        self.weights_out = weights_out
        self.weights_pred = weights_pred
        self.h_rep = h_rep
        self.h_rep_norm = h_rep_norm

    def _build_output(self, h_input, dim_in, dim_out, do_out):
        h_out = [h_input]
        dims = [dim_in] + ([dim_out]*flags.add_val('n_out'))

        weights_out = []; biases_out = []

        for i in range(0, flags.add_val('n_out')):
            wo = self._create_variable_with_weight_decay(
                    torch.randn([dims[i], dims[i+1]],
                        stddev=flags.add_val('weight_init')/np.sqrt(dims[i])),
                    'w_out_%d' % i, 1.0)
            weights_out.append(wo)

            biases_out.append(torch.tensor(torch.zeros([1,dim_out])))
            z = torch.matmul(h_out[i], weights_out[i]) + biases_out[i]
            # No batch norm on output because p_cf != p_f

            h_out.append(self.nonlin(z))
            h_out[i+1] = torch.nn.Dropout(h_out[i+1], do_out)

        weights_pred = self._create_variable(torch.randn([dim_out,1],
            stddev=flags.add_val('weight_init')/np.sqrt(dim_out)), 'w_pred')                   #weight initialization scale 
        bias_pred = self._create_variable(torch.zeros([1]), 'b_pred')

        if flags.add_val('varsel') or flags.add_val('n_out') == 0:
            self.wd_loss += tf.nn.l2_loss(tf.slice(weights_pred,[0,0],[dim_out-1,1])) #Don't know how to transform 
        else:
            self.wd_loss += weights_pred*(torch.sum(var**2)/2)                 #half l2 loss 

        ''' Construct linear classifier '''
        h_pred = h_out[-1]
        y = torch.matmul(h_pred, weights_pred)+bias_pred

        return y, weights_out, weights_pred

    def _build_output_graph(self, rep, t, dim_in, dim_out, do_out):
        ''' Construct output/regression layers '''

        if flag.add_val('split_output'):

            i0 = torch.int32(torch.where(t < 1)[:,0])
            i1 = torch.int32(torch.where(t > 0)[:,0])

            rep0 = torch.gather(rep, i0)               # get slices from parameter axis based on indices, torch.gather(x,dim,index)
            rep1 = torch.gather(rep, i1)

            y0, weights_out0, weights_pred0 = self._build_output(rep0, dim_in, dim_out, do_out, FLAGS)
            y1, weights_out1, weights_pred1 = self._build_output(rep1, dim_in, dim_out, do_out, FLAGS)

            y = tf.dynamic_stitch([i0, i1], [y0, y1]) #Don't know how to transform 
            weights_out = weights_out0 + weights_out1
            weights_pred = weights_pred0 + weights_pred1
        else:
            h_input = tf.concat(1,[rep, t])
            y, weights_out, weights_pred = self._build_output(h_input, dim_in+1, dim_out, do_out, FLAGS)

        return y, weights_out, weights_pred
