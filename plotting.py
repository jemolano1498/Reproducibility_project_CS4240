import sys
import os
import numpy as np

from loader import *

LINE_WIDTH = 2
FONTSIZE_LGND = 8
FONTSIZE = 16

EARLY_STOP_SET_CONT = 'valid'
EARLY_STOP_CRITERION_CONT = 'objective'
CONFIG_CHOICE_SET_CONT = 'valid'
CONFIG_CRITERION_CONT = 'pehe_nn'
CORR_CRITERION_CONT = 'pehe'
CORR_CHOICE_SET_CONT = 'test'

EARLY_STOP_SET_BIN = 'valid'
EARLY_STOP_CRITERION_BIN = 'policy_risk'
CONFIG_CHOICE_SET_BIN = 'valid'
CONFIG_CRITERION_BIN = 'policy_risk'
CORR_CRITERION_BIN = 'policy_risk'
CORR_CHOICE_SET_BIN = 'test'

CURVE_TOP_K = 7

def fill_bounds(data, axis=0, std_error=False):
    if std_error:
        dev = np.std(data, axis)/np.sqrt(data.shape[axis])
    else:
        dev = np.std(data, axis)

    ub = np.mean(data, axis) + dev
    lb = np.mean(data, axis) - dev

    return lb, ub

def cap(s):
    t = s[0].upper() + s[1:]
    return t

def table_str_bin(result_set, row_labels, labels_long=None, binary=False):
    if binary:
        cols = ['policy_risk', 'bias_att', 'err_fact', 'objective', 'pehe_nn']
    else:
        cols = ['pehe', 'bias_ate', 'rmse_fact', 'rmse_ite', 'objective', 'pehe_nn']

    cols = [c for c in cols if c in result_set[0]]

    head = [cap(c) for c in cols]
    colw = np.max([16, np.max([len(h)+1 for h in head])])
    col1w = np.max([len(h)+1 for h in row_labels])

    def rpad(s):
        return s+' '*(colw-len(s))

    def r1pad(s):
        return s+' '*(col1w-len(s))

    head_pad = [r1pad('')]+[rpad(h) for h in head]

    head_str = '| '.join(head_pad)
    s = head_str + '\n' + '-'*len(head_str) + '\n'

    for i in range(len(result_set)):
        vals = [np.mean(np.abs(result_set[i][c])) for c in cols] # @TODO: np.abs just to make err not bias. change!
        stds = [np.std(result_set[i][c])/np.sqrt(result_set[i][c].shape[0]) for c in cols]
        val_pad = [r1pad(row_labels[i])] + [rpad('%.3f +/- %.3f ' % (vals[j], stds[j])) for j in range(len(vals))]
        val_str = '| '.join(val_pad)

        if labels_long is not None:
            s += labels_long[i] + '\n'

        s += val_str + '\n'

    return s

def evaluation_summary(result_set, row_labels, output_dir, labels_long=None, binary=False):
    s = ''
    for i in ['train', 'valid', 'test']:
        s += 'Mode: %s\n' % cap(i)
        s += table_str_bin([results[i] for results in result_set], row_labels, labels_long, binary)
        s += '\n'

    return s

def select_parameters(results, configs, stop_set, stop_criterion, choice_set, choice_criterion):

    if stop_criterion == 'objective' and 'objective' not in results[stop_set]:
        if 'err_fact' in results[stop_set]:
            stop_criterion = 'err_fact'
        else:
            stop_criterion = 'rmse_fact'

    ''' Select early stopping for each repetition '''
    n_exp = results[stop_set][stop_criterion].shape[1]
    i_sel = np.argmin(results[stop_set][stop_criterion],2)
    results_sel = {'train': {}, 'valid': {}, 'test': {}}

    for k in results['valid'].keys():
        # To reduce dimension
        results_sel['train'][k] = np.sum(results['train'][k],2)
        results_sel['valid'][k] = np.sum(results['valid'][k],2)

        if k in results['test']:
            results_sel['test'][k] = np.sum(results['test'][k],2)

        for ic in range(len(configs)):
            for ie in range(n_exp):
                results_sel['train'][k][ic,ie,] = results['train'][k][ic,ie,i_sel[ic,ie],]
                results_sel['valid'][k][ic,ie,] = results['valid'][k][ic,ie,i_sel[ic,ie],]

                if k in results['test']:
                    results_sel['test'][k][ic,ie,] = results['test'][k][ic,ie,i_sel[ic,ie],]

    print ('Early stopping:')
    print (np.mean(i_sel,1))

    ''' Select configuration '''
    results_all = [dict([(k1, dict([(k2, v[i,]) for k2,v in results_sel[k1].items()]))
                        for k1 in results_sel.keys()]) for i in range(len(configs))]

    labels = ['%d' % i for i in range(len(configs))]

    sort_key = np.argsort([np.mean(r[choice_set][choice_criterion]) for r in results_all])
    results_all = [results_all[i] for i in sort_key]
    configs_all = [configs[i] for i in sort_key]
    labels = [labels[i] for i in sort_key]

    return results_all, configs_all, labels, sort_key


def plot_evaluation_cont(results, configs, output_dir, data_train_path, data_test_path, filters=None):

    data_train = load_data(data_train_path)
    data_test = load_data(data_test_path)

    propensity = {}
    propensity['train'] = np.mean(data_train['t'])
    propensity['valid'] = np.mean(data_train['t'])
    propensity['test'] = np.mean(data_test['t'])

    ''' Select by filter '''
    filter_str = ''
    if filters is not None:
        filter_str = '.'+'.'.join(['%s.%s' % (k,filters[k]) for k in sorted(filters.keys())])

        N = len(configs)
        I = [i for i in range(N) if np.all( \
                [configs[i][k]==filters[k] for k in filters.keys()] \
            )]

        results = dict([(s,dict([(k,results[s][k][I,]) for k in results[s].keys()])) for s in ['train', 'valid', 'test']])
        configs = [configs[i] for i in I]

    ''' Do parameter selection and early stopping '''
    results_all, configs_all, labels, sort_key = select_parameters(results,
        configs, EARLY_STOP_SET_CONT, EARLY_STOP_CRITERION_CONT,
        CONFIG_CHOICE_SET_CONT, CONFIG_CRITERION_CONT)

    ''' Save sorted configurations by parameters that differ '''
    diff_opts = sorted([k for k in configs[0] if len(set([cfg[k] for cfg in configs]))>1])
    labels_long = [', '.join(['%s=%s' % (k,str(configs[i][k])) for k in diff_opts]) for i in sort_key]

    with open('%s/configs_sorted%s.txt' % (output_dir, filter_str), 'w') as f:
        f.write('\n'.join(labels_long))

    ''' Compute evaluation summary and store'''
    eval_str = evaluation_summary(results_all, labels, output_dir, binary=False)

    with open('%s/results_summary%s.txt' % (output_dir, filter_str), 'w') as f:
        f.write('Selected early stopping based on individual \'%s\' on \'%s\'\n' % (EARLY_STOP_CRITERION_CONT, EARLY_STOP_SET_CONT))
        f.write('Selected configuration based on mean \'%s\' on \'%s\'\n' % (CONFIG_CRITERION_CONT, CONFIG_CHOICE_SET_CONT))
        f.write(eval_str)


def plot_cfr_evaluation_bin(results, configs, output_dir):
    alphas = [cfg['p_alpha'] for cfg in configs]
    palphas = alphas; palphas[0] = 1e-7;

    EARLY_STOP_CRITERION_BIN = 'err_fact'
    ALPHA_CRITERION = 'policy_risk'
    EARLY_STOP_SET_BIN = 'valid'
    ALPHA_CHOICE_SET = 'valid'

    ''' Select early stopping for each repetition '''
    n_exp = results[EARLY_STOP_SET_BIN][EARLY_STOP_CRITERION_BIN].shape[1]
    i_sel = np.argmin(results[EARLY_STOP_SET_BIN][EARLY_STOP_CRITERION_BIN],2)
    results_sel = {'train': {}, 'valid': {}, 'test': {}}

    for k in results['valid'].keys():
        # To reduce dimension
        results_sel['train'][k] = np.sum(results['train'][k],2)
        results_sel['valid'][k] = np.sum(results['valid'][k],2)
        results_sel['test'][k] = np.sum(results['test'][k],2)

        for ia in range(len(alphas)):
            for ie in range(n_exp):
                results_sel['train'][k][ia,ie,] = results['train'][k][ia,ie,i_sel[ia,ie],]
                results_sel['valid'][k][ia,ie,] = results['valid'][k][ia,ie,i_sel[ia,ie],]
                results_sel['test'][k][ia,ie,] = results['test'][k][ia,ie,i_sel[ia,ie],]

    print ('Early stopping:')
    print (np.mean(i_sel,1))

    ''' Select alpha based on mean criterion'''
    i_skip=1
    A = np.mean(results_sel[ALPHA_CHOICE_SET][ALPHA_CRITERION],1)
    ia = i_skip + A[i_skip:].argmin()
    print ('Alpha selection criterion:')
    print (A)

    ''' Print evaluation results '''
    results_alphas = [dict([(k1, dict([(k2, v[i,]) for k2,v in results_sel[k1].iteritems()]))
                        for k1 in results_sel.keys()]) for i in range(len(alphas))]
    di = configs[0]['n_in']
    do = configs[0]['n_out']
    labels=['CFR-%d-%d' % (di,do)]
    for i in range(len(alphas)):
        if i==0:
            continue
        m = ''
        if i==ia:
            m = ' *'
        labels.append('CFR-%d-%d %s a=%.2g%s' % (di,do,configs[0]['imb_fun'],alphas[i],m))

    with open('%s/results_summary.txt' % output_dir, 'w') as f:
        f.write('Selected early stopping based on individual \'%s\' on \'%s\'\n' % (EARLY_STOP_CRITERION_BIN, EARLY_STOP_SET_BIN))
        f.write('Selected alpha based on mean \'%s\' on \'%s\'\n\n' % (ALPHA_CRITERION, ALPHA_CHOICE_SET))
        f.write(eval_str)

    ''' Plotting policy curves '''
    pc = np.mean(results['train']['policy_curve'][0,:,-1,:],0)
    x = np.array(range(len(pc))).astype(np.float32)/(len(pc)-1)
    plot_with_fill(x, results_sel['train']['policy_curve'][0,:,:], axis=0, std_error=True, color='b')
    plot_with_fill(x, results_sel['train']['policy_curve'][ia,:,:], axis=0, std_error=True, color='g')
    plt.plot([0,1], [pc[0], pc[-1]], '--k', linewidth=2)
    plt.xlabel(r'$\mathrm{Inclusion\/rate}$', fontsize=FONTSIZE)
    plt.ylabel(r'$\mathrm{Policy\/value}$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{Policy\/curve\/(w.\/early\/stopping)}$')
    plt.legend(['alpha=0', 'alpha=%.2g' % alphas[ia]])
    plot_format()
    plt.savefig('%s/policy_curve_train.pdf' % (output_dir))
    plt.close()

    pc = np.mean(results['test']['policy_curve'][0,:,-1,:],0)
    x = np.array(range(len(pc))).astype(np.float32)/(len(pc)-1)
    plot_with_fill(x, results_sel['test']['policy_curve'][0,:,:], axis=0, std_error=True, color='b')
    plot_with_fill(x, results_sel['test']['policy_curve'][ia,:,:], axis=0, std_error=True, color='g')
    plt.plot([0,1], [pc[0], pc[-1]], '--k', linewidth=2)
    plt.xlabel(r'$\mathrm{Inclusion\/rate}$', fontsize=FONTSIZE)
    plt.ylabel(r'$\mathrm{Policy\/value}$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{Policy\/curve\/(w.\/early\/stopping)}$')
    plt.legend(['alpha=0', 'alpha=%.2g' % alphas[ia]])
    plot_format()
    plt.savefig('%s/policy_curve_test.pdf' % (output_dir))
    plt.close()

    ''' Policy value at early stopping point '''
    plot_with_fill(palphas, results_sel['train']['policy_value'][:,:], axis=1, std_error=True, color='r')
    plot_with_fill(palphas, results_sel['valid']['policy_value'][:,:], axis=1, std_error=True, color='g')
    plot_with_fill(palphas, results_sel['test']['policy_value'][:,:], axis=1, std_error=True, color='b')

    plt.xscale('log')
    fix_log_axes(palphas)
    plt.ylabel(r'$\mathrm{Policy\/value}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Imbalance\/penalty},\/\alpha$', fontsize=FONTSIZE)
    plt.legend(['Train', 'Valid', 'Test'])
    plot_format()
    plt.savefig('%s/policy_value_sel.pdf' % (output_dir))
    plt.close()

    ''' Policy value at each stage of training '''
    for t in range(results['train']['policy_value'].shape[2]):
        plot_with_fill(palphas, results['train']['policy_value'][:,:,t], axis=1, std_error=True, color='r')
        plot_with_fill(palphas, results['valid']['policy_value'][:,:,t], axis=1, std_error=True, color='g')
        plot_with_fill(palphas, results['test']['policy_value'][:,:,t], axis=1, std_error=True, color='b')

        plt.xscale('log')
        fix_log_axes(palphas)
        plt.ylabel(r'$\mathrm{Policy\/value}$', fontsize=FONTSIZE)
        plt.xlabel(r'$\mathrm{Imbalance\/penalty},\/\alpha$', fontsize=FONTSIZE)
        plt.legend(['Train', 'Valid', 'Test'])
        plot_format()
        plt.savefig('%s/policy_value_end_t%d.pdf' % (output_dir,t))
        plt.close()

    ''' Accuracy at end of training '''
    err_train = results_sel['train']['err_fact']
    err_valid = results_sel['valid']['err_fact']
    err_test = results_sel['test']['err_fact']

    plot_with_fill(palphas, err_train, axis=1, std_error=True, color='r')
    plot_with_fill(palphas, err_valid, axis=1, std_error=True, color='g')
    plot_with_fill(palphas, err_test, axis=1, std_error=True, color='b')
    plt.xscale('log')
    fix_log_axes(palphas)

    plt.ylabel(r'$\mathrm{Factual\/error\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Imbalance\/penalty},\/\alpha$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{Factual\/error\/(w.\/early\/stopping)}$')
    plt.legend(['Train', 'Valid', 'Test'])
    plot_format()
    plt.savefig('%s/err_fact_alpha.pdf' % output_dir)
    plt.close()

    ''' Accuracy for different iterations '''
    colors = 'rgbcmyk'
    markers = '.d*ox'
    err_test = results['test']['err_fact'][:,:,:]
    ts = range(err_test.shape[2])
    for i in range(len(alphas)):
        plt.plot(ts, np.mean(err_test[i,],0), '-%s' % markers[i%len(markers)],
            color=colors[i%len(colors)], linewidth=LINE_WIDTH)
    plt.ylabel(r'$\mathrm{Factual\/error\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Iteration}$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{Test\/factual\/error}$')
    plt.legend(['Alpha=%.2g' % a for a in alphas], fontsize=(FONTSIZE_LGND-2))
    plot_format()
    plt.savefig('%s/err_fact_iterations_test.pdf' % output_dir)
    plt.close()

    ''' Policy value for different iterations '''
    colors = 'rgbcmyk'
    markers = '.d*ox'
    y_test = results['test']['policy_value'][:,:,:]
    ts = range(y_test.shape[2])
    for i in range(len(alphas)):
        plt.plot(ts, np.mean(y_test[i,],0), '-%s' % markers[i%len(markers)],
            color=colors[i%len(colors)], linewidth=LINE_WIDTH)
    plt.ylabel(r'$\mathrm{Polcy\/value\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Iteration}$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{Policy\/value}$')
    plt.legend(['Alpha=%.2g' % a for a in alphas], fontsize=(FONTSIZE_LGND-2))
    plot_format()
    plt.savefig('%s/policy_val_iterations_test.pdf' % output_dir)
    plt.close()

def plot_cfr_evaluation_cont(results, configs, output_dir):
    alphas = [cfg['p_alpha'] for cfg in configs]

    ''' Select early stopping for each experiment '''
    n_exp = results['valid']['pehe'].shape[1]
    i_sel = np.argmin(results['valid']['pehe'],2)
    results_sel = {'train': {}, 'valid': {}, 'test': {}}

    for k in results['valid'].keys():
        # To reduce dimension
        results_sel['train'][k] = np.sum(results['train'][k],2)
        results_sel['valid'][k] = np.sum(results['valid'][k],2)
        results_sel['test'][k] = np.sum(results['test'][k],2)

        for ia in range(len(alphas)):
            for ie in range(n_exp):
                results_sel['train'][k][ia,ie] = results['train'][k][ia,ie,i_sel[ia,ie]].copy()
                results_sel['valid'][k][ia,ie] = results['valid'][k][ia,ie,i_sel[ia,ie]].copy()
                results_sel['test'][k][ia,ie] = results['test'][k][ia,ie,i_sel[ia,ie]].copy()


    ''' Select alpha and early stopping based on MEAN validation pehe (@TODO: not used) '''
    i_skip=1
    j_skip=1
    A = np.mean(results['valid']['pehe'],1)
    i,j = np.unravel_index(A[i_skip:,j_skip:].argmin(), A[i_skip:,j_skip:].shape)
    ia = i+i_skip
    it = j+j_skip

    ''' Factual vs alphas '''
    err_train = results_sel['train']['rmse_fact']
    err_valid = results_sel['valid']['rmse_fact']
    err_test = results_sel['test']['rmse_fact']

    plot_with_fill(alphas, err_train, axis=1, std_error=True, color='r')
    plot_with_fill(alphas, err_valid, axis=1, std_error=True, color='g')
    plot_with_fill(alphas, err_test, axis=1, std_error=True, color='b')
    plt.xscale('log')
    fix_log_axes(palphas)

    plt.ylabel(r'$\mathrm{Factual\/error\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Imbalance\/penalty},\/\alpha$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{RMSE\/fact\/vs\/alpha}$')
    plt.legend(['Train', 'Valid', 'Test'])
    plot_format()
    plt.savefig('%s/err_fact_alpha.pdf' % output_dir)
    plt.close()

    ''' Counterfactual vs alphas '''
    err_train = results_sel['train']['rmse_cfact']
    err_valid = results_sel['valid']['rmse_cfact']
    err_test = results_sel['test']['rmse_cfact']

    plot_with_fill(alphas, err_train, axis=1, std_error=True, color='r')
    plot_with_fill(alphas, err_valid, axis=1, std_error=True, color='g')
    plot_with_fill(alphas, err_test, axis=1, std_error=True, color='b')
    plt.xscale('log')
    fix_log_axes(palphas)

    plt.ylabel(r'$\mathrm{Factual\/error\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Imbalance\/penalty},\/\alpha$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{RMSE\/cfact\/vs\/\alpha}$')
    plt.legend(['Train', 'Valid', 'Test'])
    plot_format()
    plt.savefig('%s/err_cfact_alpha.pdf' % output_dir)
    plt.close()

    ''' PEHE vs alphas '''
    err_train = results_sel['train']['pehe']
    err_valid = results_sel['valid']['pehe']
    err_test = results_sel['test']['pehe']

    plot_with_fill(alphas, err_train, axis=1, std_error=True, color='r')
    plot_with_fill(alphas, err_valid, axis=1, std_error=True, color='g')
    plot_with_fill(alphas, err_test, axis=1, std_error=True, color='b')
    plt.xscale('log')
    fix_log_axes(palphas)

    plt.ylabel(r'$\mathrm{Factual\/error\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Imbalance\/penalty},\/\alpha$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{PEHE vs alpha}$')
    plt.legend(['Train', 'Valid', 'Test'])
    plot_format()
    plt.savefig('%s/pehe_alpha.pdf' % output_dir)
    plt.close()

    ''' Accuracy for different iterations '''
    colors = 'rgbcmyk'
    markers = '.d*ox'
    err_test = results['test']['rmse_fact'][:,:,:]
    ts = range(err_test.shape[2])
    for i in range(len(alphas)):
        plt.plot(ts, np.mean(err_test[i,],0), '-%s' % markers[i%len(markers)],
            color=colors[i%len(colors)], linewidth=LINE_WIDTH)
    plt.ylabel(r'$\mathrm{Factual\/error\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Iteration}$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{Test\/factual\/error}$')
    plt.legend(['Alpha=%.2g' % a for a in alphas], fontsize=(FONTSIZE_LGND-2))
    plot_format()
    plt.savefig('%s/err_fact_iterations_test.pdf' % output_dir)
    plt.close()

    ''' PEHE for different iterations '''
    colors = 'rgbcmyk'
    markers = '.d*ox'
    y_test = results['test']['pehe'][:,:,:]
    ts = range(y_test.shape[2])
    for i in range(len(alphas)):
        plt.plot(ts, np.mean(y_test[i,],0), '-%s' % markers[i%len(markers)],
            color=colors[i%len(colors)], linewidth=LINE_WIDTH)
    plt.ylabel(r'$\mathrm{Polcy\/value\/(test)}$', fontsize=FONTSIZE)
    plt.xlabel(r'$\mathrm{Iteration}$', fontsize=FONTSIZE)
    plt.title(r'$\mathrm{PEHE\/(Test)}$')
    plt.legend(['Alpha=%.2g' % a for a in alphas], fontsize=(FONTSIZE_LGND-2))
    plot_format()
    plt.savefig('%s/pehe_iterations_test.pdf' % output_dir)
    plt.close()
