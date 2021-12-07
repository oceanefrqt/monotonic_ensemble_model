from Module import cost_matrix_uncertainty as cmu
from Module import optimal_k_aggregations as oka
from Module import monotonic_regression_uncertainty as mru
from Module import tools
from Module import selection_algorithm as sa

import pandas as pd
import matplotlib.pyplot as plt
import os



def stage_1(df, max_k, nbcpus, strat, funct):
    pairs_err, k_error = oka.k_missclassification(df, nbcpus, funct, strat, max_k)
    k_opt, err_k = oka.optimal_k(k_error)
    return k_opt



def stage_2(df, k_opt, auc_file, nbcpus, funct, strat):
    preds = list()
    probas = list()
    labels = list()

    for i in range(len(df)):
        out = df.iloc[i, :]
        df_2 = df.drop([i])
        df_2.reset_index(drop=True, inplace=True)

        m_err, i_err = cmu.error_matrix(df_2, nbcpus, funct)
        m_err, i_err = cmu.error(m_err, i_err, df_2)
        m_err, i_err = cmu.nb_uncertainty(m_err, i_err, df_2)
        ndf_err_ = cmu.matrix_csv(m_err, i_err, df_2, sort1 = 'error')
        ndf_err = cmu.filter_uncertainty(ndf_err_, 20)

        mve, pairs, algo = oka.find_k_metamodel(df_2, ndf_err, k_opt, nbcpus, strat)
        pred, proba = oka.create_and_predict_metamodel(df_2, out, pairs, nbcpus, funct)

        print('Point', out)

        print('pred',pred)
        print('probas', proba)

        preds.append(abs(out['diagnostic']-pred))
        probas.append(proba)
        labels.append(out['diagnostic'])


    acc = preds.count(0)/len(preds)
    auc = tools.auc_score(probas, labels, auc_file)
    CI = tools.confidence_interval(auc, labels)

    labels, probas, uncertain_pts = tools.unclassified_points(labels, probas)
    return acc, auc, CI

def stage_3(df, k_opt, nbcpus, funct, strat):
    m_err, i_err = cmu.error_matrix(df, nbcpus, funct)
    m_err, i_err = cmu.error(m_err, i_err, df)
    m_err, i_err = cmu.nb_uncertainty(m_err, i_err, df)
    ndf_err_ = cmu.matrix_csv(m_err, i_err, df, sort1 = 'error')
    ndf_err = cmu.filter_uncertainty(ndf_err_, 20)
    mve, pairs, algo = oka.find_k_metamodel(df, ndf_err, k_opt, nbcpus, strat)
    return pairs, mve
