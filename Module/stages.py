from Module import cost_matrix_uncertainty as cmu
from Module import optimal_k_aggregations as oka
from Module import monotonic_regression_uncertainty as mru
from Module import tools

import pandas as pd
import matplotlib.pyplot as plt
import os



def stage_1(df, max_k, nbcpus, log, funct = mru.predict_uncertainty):
    f = open(log, 'a')
    f.write('Start stage 1\n')
    f.close()
    pairs_err, k_error = oka.k_missclassification(df, nbcpus, funct, max_k)
    f = open(log, 'a')
    f.write('Pairs and errors\n')
    f.close()
    k_opt, err_k = oka.optimal_k(k_error)
    f = open(log, 'a')
    f.write('End stage 1 with k = {}\n'.format(k_opt))
    f.close()
    return k_opt



def stage_2(df, k_opt, auc_file, nbcpus, log, funct = mru.predict_uncertainty):
    preds = list()
    probas = list()
    labels = list()
    f = open(log, 'a')
    f.write('Start stage 2\n')
    f.close()

    for i in range(len(df)):
        f = open(log, 'a')
        f.write('Patient {}\n'.format(i))
        f.close()
        out = df.iloc[i, :]
        df_2 = df.drop([i])
        df_2.reset_index(drop=True, inplace=True)

        m_err, i_err = cmu.error_matrix(df_2, nbcpus, prediction=funct)
        m_err, i_err = cmu.error(m_err, i_err, df_2)
        m_err, i_err = cmu.nb_uncertainty(m_err, i_err, df_2)
        ndf_err_ = cmu.matrix_csv(m_err, i_err, df_2, sort1 = 'error')
        ndf_err = cmu.filter_uncertainty(ndf_err_, 20)

        mve, pairs, algo = oka.find_k_metamodel(df_2, ndf_err, k_opt, nbcpus)
        pred, proba = oka.create_and_predict_metamodel(df_2, out, pairs, nbcpus)

        preds.append(abs(out['diagnostic']-pred))
        probas.append(proba)
        labels.append(out['diagnostic'])

    f = open(log, 'a')
    f.write('Compute AUC\n')
    f.close()

    acc = preds.count(0)/len(preds)
    auc = tools.auc_score(probas, labels, auc_file)
    CI = tools.confidence_interval(auc, labels)

    f = open(log, 'a')
    f.write('AUC = {}, CI = {}\n'.format(auc, CI))
    f.close()

    labels, probas, uncertain_pts = tools.unclassified_points(labels, probas)
    return acc, auc

def stage_3(df, k_opt, nbcpus, log, funct = mru.predict_uncertainty):
    f = open(log, 'a')
    f.write('Start stage 3\n')
    f.close()
    m_err, i_err = cmu.error_matrix(df, nbcpus, prediction=funct)
    m_err, i_err = cmu.error(m_err, i_err, df)
    m_err, i_err = cmu.nb_uncertainty(m_err, i_err, df)
    ndf_err_ = cmu.matrix_csv(m_err, i_err, df, sort1 = 'error')
    ndf_err = cmu.filter_uncertainty(ndf_err_, 20)
    f = open(log, 'a')
    f.write('Find best metamodel\n')
    f.close()
    mve, pairs, algo = oka.find_k_metamodel(df, ndf_err, k_opt, nbcpus)
    return pairs, mve
