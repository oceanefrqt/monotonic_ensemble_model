from Module import monotonic_regression_uncertainty as mru
from Module import tools


import numpy as np
import pandas as pd
import os
import time
from itertools import chain
import copy


import multiprocessing as mp

from IPython.display import display

### Useful functions for parallele

def vals_mp(pairs, df_2, out, funct):
    vals = list()
    for p in pairs:
        vals.append((p, df_2, out, funct))
    return vals


#### ERROR MATRIX ######


def single_error(p, df_2, out, funct):

    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = tools.equiv_key_case(key)

    diag = df_2['diagnostic'].values.tolist()

    tr1, tr2 = df_2[p1].values.tolist(), df_2[p2].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])
    X, models = mru.compute_recursion(data, (rev, up, key))

    bpr, bpb, r_p, b_p = models[key]
    pred = funct(out_p, bpr, bpb, rev, up)

    if pred == -1:
        return (p, -1) #if uncertain, we keep it like this
    else:
        return (p, abs(1-int(pred == out['diagnostic']))) #int(True) = 1 so if the pred is equal to real label, error is equal to 0


def error_matrix(df_, pairs, nbcpus, funct):
    try:
        nbcpus = int (os.getenv('OMP_NUM_THREADS') )
    except:
        pass
    pool = mp.Pool(nbcpus)


    df = copy.deepcopy(df_)

    index = list()

    mat_err = pd.DataFrame(columns = pairs + ['phenotype']) #Dataframe with possible classifiers as columns

    for j in range(len(df)):# For each patient j, we add a line to the dataframe contaning whereas the patient was misclassified or not (1 if misclassified, 0 otherwise)
        out = df.iloc[j, :]
        df_2 = df.drop([j])#classifiers are constructed according to the set of patients without patient j
        df_2.reset_index(drop=True, inplace=True)

        vals = vals_mp(pairs, df_2, out, funct)

        res = pool.starmap(single_error, vals, max(1,len(vals)//nbcpus)) #res is an array (size = nb of pairs) that the name of the classifier and whereas the poatient was misclassified or not

        dico_err = {r[0] : r[1] for r in res}
        dico_err['phenotype'] = out['diagnostic']
        dico_err_s = pd.Series(dico_err)
        dico_err_s.name = 'P'+str(j+1)
        mat_err = pd.concat((mat_err, dico_err_s.to_frame().T), axis=0)
        del df_2




    unc = {col: mat_err[col].to_list().count(-1) for col in pairs}
    unc['phenotype'] = np.nan
    unc_s = pd.Series(unc)
    unc_s.name = 'uncertain'

    mat_err_unc = pd.concat((mat_err,unc_s.to_frame().T), axis=0)



    cols = list(mat_err_unc.columns)
    cols.remove('phenotype')
    rem = list()
    for col in cols:
        val = mat_err_unc.at['uncertain', col]
        if val > len(df)/3:
            rem.append(col)
    mat_err.drop(rem, axis=1, inplace=True)
    mat_err_unc.drop(rem, axis=1, inplace=True)

    err = {col: mat_err[col].to_list().count(1)/(mat_err[col].to_list().count(1) + mat_err[col].to_list().count(0)) for col in pairs if col not in rem}
    err['phenotype'] = np.nan
    err_s = pd.Series(err)
    err_s.name = 'error'

    mat_err_final = pd.concat((mat_err_unc,err_s.to_frame().T), axis=0)


    mat_err_final.sort_values(axis = 1, by=['error', 'uncertain'], inplace=True)

    del df
    return mat_err_final




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





### Get dict relating classifiers with error score


def cost_classifiers(ndf):
    cols = list(ndf.columns)
    cols.remove('phenotype')
    cost = {cols[i] : ndf[cols[i]].loc[['error']][0] for i in range(len(cols))}
    return cost
