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
