import numpy as np

def load_data(fname):
    """ Load data set """
    if fname[-3:] == 'npz':
        data_in = np.load(fname)
        data = {'x': data_in['x'], 't': data_in['t'], 'yf': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    # else:
    #     if FLAGS.sparse>0:
    #         data_in = np.loadtxt(open(fname+'.y',"rb"),delimiter=",")
    #         x = load_sparse(fname+'.x')
    #     else:
    #         data_in = np.loadtxt(open(fname,"rb"),delimiter=",")
    #         x = data_in[:,5:]
    #
    #     data['x'] = x
    #     data['t'] = data_in[:,0:1]
    #     data['yf'] = data_in[:,1:2]
    #     data['ycf'] = data_in[:,2:3]

    data['HAVE_TRUTH'] = not data['ycf'] is None

    data['dim'] = data['x'].shape[1]
    data['n'] = data['x'].shape[0]

    return data