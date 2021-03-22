class Parameters():
    
    def __init__(self):
        self.parameter = {}
        
    def add_val(self, name, value, description):
        self.parameter[name] = [value, description]
        
    def get_val(self, name):
        return  self.parameter[name][0]
    
    def get_description(self, name):
        return  self.parameter[name][1]
        
flags = Parameters()
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
flags.add_val('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
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
flags.add_val('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
flags.add_val('output_csv',0,"""Whether to save a CSV file with the results""")
flags.add_val('output_delay', 100, """Number of iterations between log/loss outputs. """)
flags.add_val('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
flags.add_val('debug', 0, """Debug mode. """)
flags.add_val('save_rep', 0, """Save representations after training. """)
flags.add_val('val_part', 0, """Validation part. """)
flags.add_val('split_output', 0, """Whether to split output layers between treated and control. """)
flags.add_val('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)

print(flags.get_val('n_in'))


