from Module import cost_matrix_uncertainty as cmu
from Module import optimal_k_aggregations as oka
from Module import monotonic_regression_uncertainty as mru
from Module import tools
from Module import selection_algorithm as sa
from Module import preselection as pls

import pandas as pd
import matplotlib.pyplot as plt
import os


from sklearn.metrics import roc_auc_score


def stage0(df, nbcpus, threshold):
    reg = psl.regression_error_matrix(df, nbcpus)
    if threshold is not None:
        reg = psl.preselection_reg_err(reg, threshold)
    return reg




def stage_1(df, cls, max_k, nbcpus, strat, funct, log):
    k_error = oka.k_missclassification(df, cls, nbcpus, funct, strat, max_k, log)
    k_opt, err_k = oka.optimal_k(k_error)
    return k_opt, k_error



def stage_2(df, cls, k_opt, auc_file, nbcpus, funct, strat, logs):
    errors = list()
    probas = list()
    labels = list()

    for i in range(len(df)):
        out = df.iloc[i, :]
        df_2 = df.drop([i])
        df_2.reset_index(drop=True, inplace=True)
        f = open(logs, 'a')
        f.write('Patient {} \n'.format(i))
        f.close()

        ndf_err = cmu.error_matrix(df_2, cls, nbcpus,funct)
        cost = cmu.cost_classifiers(ndf_err)

        f = open(logs, 'a')
        f.write('Matrix computed \n')
        f.close()

        mve, pairs, algo = oka.find_k_metamodel(df_2, ndf_err, cost, k_opt, nbcpus, strat)
        pred, proba = oka.create_and_predict_metamodel(df_2, out, pairs, nbcpus, funct)

        f = open(logs, 'a')
        f.write('Pred={}, Proba={} \n'.format(pred, proba))
        f.close()

        errors.append(abs(out['diagnostic']-pred))
        probas.append(proba)
        labels.append(out['diagnostic'])


    acc = errors.count(0)/len(errors)
    auc = tools.auc_score(probas, labels, auc_file)
    auc2 = roc_auc_score(labels, probas)
    CI = tools.confidence_interval(auc, labels)

    f = open(logs, 'a')
    f.write('acc={}, auc={}, CI = {} \n'.format(acc, auc, CI))
    f.close()

    labels, probas, uncertain_pts = tools.unclassified_points(labels, probas)
    return acc, auc, CI

def stage_3(df, cls, k_opt, nbcpus, funct, strat):
    ndf_err = cmu.error_matrix(df, cls, nbcpus,funct)
    ndf_err.to_csv('cm_st3.csv')
    cost = cmu.cost_classifiers(ndf_err)
    mve, pairs, algo = oka.find_k_metamodel(df, ndf_err, cost, k_opt, nbcpus, strat)
    return pairs, mve
