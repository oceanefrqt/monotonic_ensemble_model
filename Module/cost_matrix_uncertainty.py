from Module import monotonic_regression_uncertainty as mru
from Module import tools


import numpy as np
import pandas as pd
import os
import time
from itertools import chain
import copy


import multiprocessing as mp

### Useful functions for parallele

def vals_mp(col, df_2, out, prediction):
    vals = list()
    for k in range(len(col)-1):
        for l in range(k+1, len(col)):
            vals.append((col[k], col[l], df_2, out, prediction))
            vals.append((col[l], col[k], df_2, out, prediction))
    return vals


#### ERROR MATRIX ######


def single_error(p1, p2, df_2, out, prediction):

    diag = df_2['diagnostic'].values.tolist()

    tr1, tr2 = df_2[p1].values.tolist(), df_2[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])
    X, models, r_p, b_p = mru.compute_recursion(data)
    preds = list()

    for key in models.keys():
        key = int(key)
        rev, up = tools.equiv_key_case(key)
        bpr, bpb = models[key]
        pred = prediction(out_p, bpr, bpb, rev, up)

        if pred == -1:
            preds.append(("".join([p1, '/', p2, '/', str(key)]), -1))
        else:
            preds.append(("".join([p1, '/', p2, '/', str(key)]), abs(1-int(pred == out['diagnostic']))))

    return preds


def error_matrix(df_, nbcpus, prediction):
    try:
        nbcpus = int (os.getenv('OMP_NUM_THREADS') )
    except:
        pass
    pool = mp.Pool(nbcpus)
    print('nb cpus count:', mp.cpu_count())
    print('nb cpus put:', nbcpus)
    #m = mp.Manager()
    #lock = m.Lock()

    df = copy.deepcopy(df_)

    index = list()
    col = list(df.columns)
    if 'diagnostic' in col:
        col.remove('diagnostic')
    else:
        print('diagnostic not in column, check the file')
    pairs = list()
    for k in range(len(col)-1):
        for l in range(k+1, len(col)):
            for nb in range(1,5):
                pairs.append("".join([col[k], '/', col[l], '/', str(nb)]))
                pairs.append("".join([col[l], '/', col[k], '/', str(nb)]))


    matrix = {pairs[i] : list() for i in range(len(pairs))} #the idea is to construct a kind of matriw with pairs in columns and patient in rows and each value
    #correspond to the error prediction of the patient according the model based on all the other patients. For that we use a dictionnary and for each case of pairs, we got
    #a list that'll receive the error predictions of all the patients

    for j in range(len(df)):# For each patient j
        index.append('x'+str(j+1))
        out = df.iloc[j, :]
        df_2 = df.drop([j])
        df_2.reset_index(drop=True, inplace=True)

        vals = vals_mp(col, df_2, out, prediction)

        res = pool.starmap(single_error, vals, max(1,len(vals)//nbcpus)) #res is an array (size = nb of pairs) that contains arrays (size=4) containing a
        #tuple with the case of model and the error prediction


        for i in range(len(res)): #For each pair of transcripts
            for k in range(len(res[i])): #For each case of model with that pair of transcript
                matrix[res[i][k][0]].append(res[i][k][1]) # We store the error prediction on j

    del df
    return matrix, index




#### GOING error matrix to prediction matrix
def error_to_prediction(matrix, df):
    diags = df['diagnostic'].values.tolist()

    prediction_mat = {}
    for cls in matrix.keys():
        if cls != 'phenotype':
            errors = matrix[cls]
            pred = list()
            for i in range(len(errors)):
                if errors[i] == 0:
                    pred.append(diags[i])
                elif errors[i] == 1:
                    pred.append(int(abs(1-diags[i])))
                elif errors[i] == -1:
                    pred.append(-1)
            prediction_mat[cls] = pred

    return prediction_mat


### ERRORS functions applied only on error matrices

def error(matrix, index, df):
    ## applied to an error matrix
    for k in matrix.keys():
        if len(matrix[k]) != matrix[k].count(-1):
            tot = len(matrix[k]) - matrix[k].count(-1)
            matrix[k].append(matrix[k].count(1)/tot)
        else:
            matrix[k].append(None)

    index.append('error')

    return matrix, index

def nb_misclassification(matrix, index, df):
    ## applied to an error matrix
    for k in matrix.keys():
        matrix[k].append(matrix[k].count(1))

    index.append('misclassififcation')

    return matrix, index

def nb_uncertainty(matrix, index, df):
    #can be applied on error matrices but also on prediction matrices
    index.append('uncertain')
    for k in matrix.keys():
        matrix[k].append(matrix[k].count(-1))

    return matrix, index


### SORTING functions


def matrix_csv(matx, idx, df, sort1 = None, sort2 = None):
    diags = df['diagnostic'].values.tolist()

    mat = copy.deepcopy(matx)
    indx = copy.deepcopy(idx)


    ndf = pd.DataFrame(mat, index = indx)

    if (sort1 is not None) and (sort2 is None):
        ndf.sort_values(axis = 1, by=[sort1], inplace=True)
    elif (sort1 is not None) and (sort2 is not None):
        ndf.sort_values(axis = 1, by=[sort1, sort2], inplace=True)


    if len(diags) != len(indx):
        add = [None] * (len(indx) - len(diags))
        diags = list(chain(diags,add))


    ndf.insert(0, 'phenotype', diags)

    return ndf


def cost_classifiers(ndf):
    cols = list(ndf.columns)
    cols.remove('phenotype')
    errors = ndf.loc[['error'], :]
    #print(errors.values.tolist())
    errors = errors.values.tolist()[0][1:]
    cost = {cols[i] : errors[i] for i in range(len(cols))}
    return cost


def filter_uncertainty(ndf, threshold):
    cols = list(ndf.columns)
    rem = list()
    for col in cols:
        val = ndf.at['uncertain', col]
        if val > threshold:
            rem.append(col)

    ndf.drop(rem, inplace=True, axis=1)
    return ndf
