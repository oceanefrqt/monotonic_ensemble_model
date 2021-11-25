from Module import cost_matrix_uncertainty as cmu
from Module import optimal_k_aggregations as oka
from Module import monotonic_regression_uncertainty as mru
from Module import tools

import pandas as pd
import matplotlib.pyplot as plt
import os



def stage_1(df, max_k, log, funct = mru.predict_severe):
    f = open(log, 'a')
    f.write('Start stage 1')
    f.close()
    pairs_err, k_error = oka.k_missclassification(df, max_k, funct)
    f = open(log, 'a')
    f.write('Pairs and errors')
    f.close()
    k_opt, err_k = oka.optimal_k(k_error)
    f = open(log, 'a')
    f.write('End stage 1')
    f.close()
    return k_opt



def stage_2(df, k_opt, auc_file, log, funct = mru.predict_severe):
    preds = list()
    probas = list()
    labels = list()
    f = open(log, 'a')
    f.write('Start stage 2')
    f.close()

    for i in range(len(df)):
        f = open(log, 'a')
        f.write('Patient', i)
        f.close()
        out = df.iloc[i, :]
        df_2 = df.drop([i])
        df_2.reset_index(drop=True, inplace=True)

        m_err, i_err = cmu.error_matrix(df_2, prediction=funct)
        m_err, i_err = cmu.error(m_err, i_err, df_2)
        m_err, i_err = cmu.nb_uncertainty(m_err, i_err, df_2)
        ndf_err_ = cmu.matrix_csv(m_err, i_err, df_2, sort1 = 'error')
        ndf_err = cmu.filter_uncertainty(ndf_err_, 20)

        mve, pairs, algo = oka.find_k_metamodel(df_2, ndf_err, k_opt)
        pred, proba = oka.create_and_predict_metamodel(df_2, out, pairs)

        preds.append(abs(out['diagnostic']-pred))
        probas.append(proba)
        labels.append(out['diagnostic'])

    f = open(log, 'a')
    f.write('Compute AUC')
    f.close()

    acc = preds.count(0)/len(preds)
    auc = tools.auc_score(probas, labels, auc_file)
    labels, probas, uncertain_pts = tools.unclassified_points(labels, probas)
    #print('probas stage 2:', probas)
    #print('uncertain pts stage 2:', uncertain_pts)
    return acc, auc

def stage_3(df, k_opt, log, funct = mru.predict_severe):
    f = open(log, 'a')
    f.write('Start stage 3')
    f.close()
    m_err, i_err = cmu.error_matrix(df, prediction=funct)
    m_err, i_err = cmu.error(m_err, i_err, df)
    m_err, i_err = cmu.nb_uncertainty(m_err, i_err, df)
    ndf_err_ = cmu.matrix_csv(m_err, i_err, df, sort1 = 'error')
    ndf_err = cmu.filter_uncertainty(ndf_err_, 20)
    f = open(log, 'a')
    f.write('Find best metamodel')
    f.close()
    mve, pairs, algo = oka.find_k_metamodel(df, ndf_err, k_opt)
    return pairs, mve
