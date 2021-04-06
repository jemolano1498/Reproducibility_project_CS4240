import torch
import numpy as np



def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    Y_temp = torch.transpose(Y,0,1)
    C = -2*torch.matmul(X,Y_temp)
    nx = torch.sum(torch.square(X),1,keepdim=True)
    ny = torch.sum(torch.square(Y),1,keepdim=True)
    D = C + nx + ny
    return D

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """
    X = X.squeeze(0)

    it = torch.where(t>0)
    ic = torch.where(t<1)
    it = it[1]
    ic = ic[1]

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