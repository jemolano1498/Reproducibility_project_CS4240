import torch
import numpy as np

SQRT_CONST = 1e-10

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    Y_temp = torch.transpose(Y,1,2)
    C = -2*torch.matmul(X,Y_temp)
    nx = torch.sum(torch.square(X),2,keepdim=True)
    ny = torch.sum(torch.square(Y),2,keepdim=True)
    D = (C + torch.transpose(ny,1,2)) + nx
    return D

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, np.inf))

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """
    X = X.squeeze(0)

    it = torch.where(t>0)[1]
    ic = torch.where(t<1)[1]

    Xc = torch.index_select(X,1,ic)
    Xt = torch.index_select(X,1,it)

    Kcc = torch.exp(-pdist2sq(Xc,Xc)/np.square(sig))
    Kct = torch.exp(-pdist2sq(Xc,Xt)/np.square(sig))
    Ktt = torch.exp(-pdist2sq(Xt,Xt)/np.square(sig))
    m = torch.float(torch.shape(Xc)[0])
    n = torch.float(torch.shape(Xt)[0])

    mmd = torch.square(1.0-p)/(m*(m-1.0))*(torch.sum(Kcc)-m)
    mmd = mmd + torch.square(p)/(n*(n-1.0))*(torch.sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*torch.sum(Kct)
    mmd = 4.0*mmd

    return mmd

def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = torch.where(t>0)[1] # getting the positions
    ic = torch.where(t<1)[1]

    Xt = torch.index_select(X, 1, it) # Getting the nx100 for each value
    Xc = torch.index_select(X, 1, ic)

    mean_control = torch.mean(Xc,1) # mean of 1x100
    mean_treated = torch.mean(Xt,1)

    mmd = torch.sum(torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = torch.where(t > 0)[1]  # getting the positions
    ic = torch.where(t < 1)[1]

    Xt = torch.index_select(X, 1, it)  # Getting the nx100 for each value
    Xc = torch.index_select(X, 1, ic)

    nc = Xc.shape[1]
    nt = Xt.shape[1]

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = torch.mean(M)
    M_drop = torch.nn.Dropout(10/(nc*nt))(M)
    delta = torch.max(M)
    eff_lam = lam/M_mean

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*torch.ones(torch.shape(M[0:1,:]))
    # col = tf.concat(0,[delta*tf.ones(tf.shape(M[:,0:1])),tf.zeros((1,1))])
    # Mt = tf.concat(0,[M,row])
    # Mt = tf.concat(1,[Mt,col])
    #
    # ''' Compute marginal vectors '''
    # a = tf.concat(0,[p*tf.ones(tf.shape(tf.where(t>0)[:,0:1]))/nt, (1-p)*tf.ones((1,1))])
    # b = tf.concat(0,[(1-p)*tf.ones(tf.shape(tf.where(t<1)[:,0:1]))/nc, p*tf.ones((1,1))])
    #
    # ''' Compute kernel matrix'''
    # Mlam = eff_lam*Mt
    # K = tf.exp(-Mlam) + 1e-6 # added constant to avoid nan
    # U = K*Mt
    # ainvK = K/a
    #
    # u = a
    # for i in range(0,its):
    #     u = 1.0/(tf.matmul(ainvK,(b/tf.transpose(tf.matmul(tf.transpose(u),K)))))
    # v = b/(tf.transpose(tf.matmul(tf.transpose(u),K)))
    #
    # T = u*(tf.transpose(v)*K)
    #
    # if not backpropT:
    #     T = tf.stop_gradient(T)
    #
    # E = T*Mt
    # D = 2*tf.reduce_sum(E)
    D =0
    Mlam = 0
    return D, Mlam