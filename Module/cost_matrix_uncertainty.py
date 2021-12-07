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

def vals_mp(col, df_2, out, funct):
    vals = list()
    for k in range(len(col)):
        vals.append((col[k], col[k], df_2, out, funct))
        for l in range(k+1, len(col)):
            vals.append((col[k], col[l], df_2, out, funct))
            vals.append((col[l], col[k], df_2, out, funct))
    return vals


#### ERROR MATRIX ######


def single_error(p1, p2, df_2, out, funct):

    diag = df_2['diagnostic'].values.tolist()

    tr1, tr2 = df_2[p1].values.tolist(), df_2[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])
    X, models = mru.compute_recursion(data)
    errors = list() #Store the error prediction in following format ('A/B/k', err)
    #with A and B two transcripts, k the case of the function (decreasing or increasing) and err the error

    for key in models.keys():
        print('key', key)
        key = int(key)
        rev = tools.equiv_key_case(key)
        bpr, bpb, r_p, b_p = models[key]
        pred = funct(out_p, bpr, bpb, rev)
        print('pred is done', pred)

        if pred == -1:
            errors.append(("".join([p1, '/', p2, '/', str(key)]), -1)) #if uncertain, we keep it like this
        else:
            errors.append(("".join([p1, '/', p2, '/', str(key)]), abs(1-int(pred == out['diagnostic'])))) #int(True) = 1 so if the pred is equal to real label, error is equal to 0

    return errors


def error_matrix(df_, nbcpus, funct):
    try:
        nbcpus = int (os.getenv('OMP_NUM_THREADS') )
    except:
        pass
    pool = mp.Pool(nbcpus)


    df = copy.deepcopy(df_)

    index = list()
    col = list(df.columns)

    assert 'diagnostic' in col, 'diagnostic not in column, check the file'

    col.remove('diagnostic')

    pairs = list()
    for nb in range(1,3):
        for k in range(len(col)):
            pairs.append("".join([col[k], '/', col[k], '/', str(nb)]))
            for l in range(k+1, len(col)): #We test each pair of transcript in all possible ways: A/A, A/B, B/A
             #For each pair, we test both case (decreasing or increasing function)
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

        vals = vals_mp(col, df_2, out, funct)

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
            matrix[k].append(1)

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
    #Création of the DataFrame
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
