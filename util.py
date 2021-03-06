import torch
import numpy as np


SQRT_CONST = 1e-10

def save_config(fname, flags):
    """ Save configuration """
    k_flags = flags.get_dict()
    s = '\n'.join(['%s: %s' % (k,str(k_flags[k][0])) for k in sorted(k_flags.keys())])
    f = open(fname,'w')
    f.write(s)
    f.close()

def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp['x'].shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid

def dynamic_stitch(indices, data):
  n = sum(idx.numel() for idx in indices)
  res  = [None] * n
  for i, data_ in enumerate(data):
    idx = indices[i].view(-1)
    d = data_.view(idx.numel(), -1)
    k = 0
    for idx_ in idx: res[idx_] = d[k]; k += 1
  return res

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    Y_temp = torch.transpose(Y,0,1)
    C = -2*torch.matmul(X,Y_temp)
    nx = torch.sum(torch.square(X),1,keepdim=True)
    ny = torch.sum(torch.square(Y),1,keepdim=True)
    D = (C + torch.transpose(ny,0,1)) + nx
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

    it = torch.where(t>0)[0] # getting the positions
    ic = torch.where(t<1)[0]

    Xt = torch.index_select(X, 0, it) # Getting the nx100 for each value
    Xc = torch.index_select(X, 0, ic)

    mean_control = torch.mean(Xc,0) # mean of 1x100
    mean_treated = torch.mean(Xt,0)

    mmd = torch.sum(torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = torch.where(t > 0)[0]  # getting the positions
    ic = torch.where(t < 1)[0]

    Xt = torch.index_select(X, 0, it)  # Getting the nx100 for each value
    Xc = torch.index_select(X, 0, ic)

    nc = Xc.shape[0]
    nt = Xt.shape[0]

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
    row = delta*torch.ones(M.shape[1])
    col = torch.cat((delta*torch.ones(M.shape[0]),torch.zeros((1))),0)
    Mt = torch.cat((M, torch.unsqueeze(row, 0)), 0)
    Mt = torch.cat((Mt, torch.unsqueeze(col, 1)), 1)

    ''' Compute marginal vectors '''
    temp = torch.where(t > 0)[0].shape
    a = torch.cat((p * torch.ones((torch.where(t > 0)[0].shape[0],1)) / nt, (1 - p) * torch.ones((1,1))), 0)
    b = torch.cat(((1-p) * torch.ones((torch.where(t < 1)[0].shape[0],1)) / nc, p * torch.ones((1,1))), 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = torch.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        temp = torch.transpose(torch.matmul(torch.transpose(u,0,1),K),0,1)
        u = 1.0/(torch.matmul(ainvK,( b / temp)))
    temp = torch.transpose(torch.matmul(torch.transpose(u,0,1),K),0,1)
    v = b/(temp)

    T = u*(torch.transpose(v,0,1)*K)

    E = T*Mt
    D = 2*torch.sum(E)

    return D, Mlam