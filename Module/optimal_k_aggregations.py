from Module import cost_matrix_uncertainty as cmu
from Module import monotonic_regression_uncertainty as mru
from Module import selection_algorithm as sa
from Module import tools


import time
import pandas as pd
import multiprocessing as mp
import copy



def find_k_metamodel(df, ndf, k, nbcpus):
    p1, mve1, ndf1 = sa.N_best(df, ndf, k)
    p2, mve2, ndf2 = sa.BS(df, ndf, k, nbcpus)
    p3, mve3, ndf3 = sa.FS(df, ndf, k, nbcpus)

    l = sorted([(mve1, p1, 'NB'), (mve2, p2, 'BS'), (mve3, p3,'FS')])
    return l[0]

def prediction_pairs(df, out, pair):
    p1, p2, key = pair.split('/')
    key = int(key)
    rev, up = tools.equiv_key_case(key)
    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()
    diag = df['diagnostic'].values.tolist()

    data, out_p = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))], (out[p1], out[p2])

    X, models, r_p, b_p = mru.compute_recursion(data, (rev, up, key))
    bpr, bpb = models[key]
    pred = mru.predict_uncertainty(out_p, bpr, bpb, rev, up)
    return pred


def create_and_predict_metamodel(df_, out, pairs, nbcpus):
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

    print('nb cpus', nbcpus)

    vals = [(df, out, p) for p in pairs]

    preds = pool.starmap(prediction_pairs, vals, max(1,len(vals)//nbcpus))

    del df
    return tools.pred_metamodel(preds), tools.proba_metamodel(preds)


def k_missclassification(df, nbcpus, funct, max_k=50):
    print('k misclassification : {}\n'.format(funct))

    k_mis = {k : list() for k in range(1, max_k)}

    pairs_err = {}


    for j in range(len(df)):
        out = df.iloc[j, :]
        df_2 = df.drop([j])
        df_2.reset_index(drop=True, inplace=True)

        m_err, i_err = cmu.error_matrix(df_2, nbcpus, prediction = funct )
        m_err, i_err = cmu.error(m_err, i_err, df_2)
        m_err, i_err = cmu.nb_uncertainty(m_err, i_err, df_2)
        ndf_err_ = cmu.matrix_csv(m_err, i_err, df_2, sort1 = 'error')
        ndf_err = cmu.filter_uncertainty(ndf_err_, 20)


        cost = cmu.cost_classifiers(ndf_err)
        pairs_err = keep_pairs(cost, pairs_err)

        #k = 1 : first pair
        pair = [ndf_err.columns[1]]
        pred, proba = create_and_predict_metamodel(df_2, out, pair, nbcpus)
        if proba != -1: #case of proba = -1 means that all the classifiers in the metamodel predicted the point as uncertain
            k_mis[1].append(abs(out['diagnostic']-pred))
        else:
            print('case of unknown point in oka')

        #k > 1: ensemble classifiers
        for k in range(2, max_k):
            mve, pairs, algo = find_k_metamodel(df_2, ndf_err, k, nbcpus)
            pred, proba = create_and_predict_metamodel(df_2, out, pairs, nbcpus)
            if proba != -1:
                k_mis[k].append(abs(out['diagnostic']-pred))
            else:  #case of proba = -1 means that all the classifiers in the metamodel predicted the point as uncertain
                print('case of unknown point in oka')


    k_error = {k : k_mis[k].count(1)/len(k_mis[k]) for k in range(1, max_k)}
    pairs_err = {k : sum(pairs_err[k])/len(pairs_err[k]) for k in pairs_err.keys()}


    return pairs_err, k_error

def keep_pairs(cost, pairs_err):
    for k in cost.keys():
        if k not in pairs_err.keys():
            pairs_err[k] = [cost[k]]
        else:
            pairs_err[k].append(cost[k])

    return pairs_err


def optimal_k(k_error):
    mini = min(k_error.values())
    keys = [k for k in k_error.keys() if k_error[k] == mini]
    return min(keys), mini
