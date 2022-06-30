from Module_confidence import cost_matrix_uncertainty as cmu
from Module_confidence import optimal_k_aggregations as oka
from Module_confidence import monotonic_regression_uncertainty as mru
from Module_confidence import tools
from Module_confidence import selection_algorithm as sa

import multiprocessing as mp

import pandas as pd
import matplotlib.pyplot as plt
import os


def all_configurations(df):
    transcripts = list(df.columns)
    transcripts.remove('diagnostic')

    configurations = list()
    for i in range(len(transcripts)):
        for j in range(i, len(transcripts)):
            for key in range(1,5):
                configurations.append('/'.join([transcripts[i], transcripts[j], str(key)]))
    return configurations


def single_score(cl, df):
    p1, p2, key = cl.split('/')
    key = int(key)
    rev, up = tools.equiv_key_case(key)

    diag = df['diagnostic'].to_list()
    tr1, tr2 = df[p1].to_list(), df[p2].to_list()
    data = [((tr1[i], tr2[i]), 1, diag[i]) for i in range(len(diag))]


    X, m = mru.compute_recursion(data, (rev, up, key))
    reg_err, bpr, bpb, r_p, b_p = m[key]

    return (cl, reg_err)

def regression_error_score(df, cls, nbcpus):
    pool = mp.Pool(nbcpus)

    vals = [(cl, df) for cl in cls]
    res = pool.starmap(single_score, vals, max(1, len(vals)//nbcpus))

    dico = {r[0] : r[1] for r in res}
    s = pd.Series(dico)
    s.name = 'regression error'
    return s

def regression_error_matrix(df, nbcpus):
    config = all_configurations(df)
    reg = regression_error_score(df, config, nbcpus)
    return reg


def preselection_reg_err(reg_err_mat, threshold):
    return reg_err_mat[reg_err_mat<threshold]
