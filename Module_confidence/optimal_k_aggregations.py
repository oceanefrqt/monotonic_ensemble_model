from Module_confidence import cost_matrix_uncertainty as cmu
from Module_confidence import monotonic_regression_uncertainty as mru
from Module_confidence import selection_algorithm as sa
from Module_confidence import tools
from Module_confidence import ranking_jumps as rj



import time
import pandas as pd
import multiprocessing as mp
import copy
import numpy as np
import matplotlib.pyplot as plt



def find_k_metamodel(df, ndf, cost, k, nbcpus, strat):
    l = list()
    mini = 1
    for s in strat:
        p, mve = s(df, ndf, cost, k, nbcpus)
        if mve<mini:
            mini = mve
        l.append((mve, p, s.__name__))
    potential = [L for L in l if L[0] == mini]
    potential.sort(key=lambda x:x[2], reverse = True)
    return potential[0]


#Functions that given some classifiers, construct the associated ensemble model and predict the output for a single patient

def prediction_pairs(df, out, pair, funct):
    p1, p2, key = pair.split('/')
    key = int(key)
    rev, up = tools.equiv_key_case(key)

    tr1, tr2 = df[p1].values.tolist(), df[p2].values.tolist()
    diag = df['diagnostic'].values.tolist()

    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]
    out_p = (out[p1], out[p2])

    X, models = mru.compute_recursion(data, (rev, up, key))
    reg_err, bpr, bpb, r_p, b_p = models[key]
    pred = funct(out_p, bpr, bpb, rev, up) #

    jump = rj.jump_borders(out_p, X, bpr, bpb, rev, up)

    sc = rj.score(jump, len(X))
    return pred, sc


def create_and_predict_metamodel(df_, out, pairs, nbcpus, funct):
    #Return the majority vote prediction (pred_metamodel) and the probability of the being severe (nb of severe/total nb)
    pool = mp.Pool(nbcpus)
    df = copy.deepcopy(df_)

    vals = [(df, out, p, funct) for p in pairs]

    res = pool.starmap(prediction_pairs, vals, max(1,len(vals)//nbcpus)) #Get all the predictions made by the different pairs for a single patient

    preds = [r[0] for r in res]
    scores = [r[1] for r in res]

    print('Prediction = {}, Score = {}'.format(tools.pred_metamodel(preds), sum(scores)))
    del df
    return rj.prediction_score(sum(scores)), tools.proba_metamodel(preds)



##Big loop to compute the average error for ensemble contaning k classifiers


def k_missclassification(df, cls, nbcpus, funct, strat, min_k, max_k, log):
    print('k misclassification : {}\n'.format(funct))

    k_mis = {k : list() for k in range(3, max_k)} #Store for each value of k, whereas patients were misclassified or not with an ensemble of k classifiers

    #pairs_err = {cl : list() for cl in cls} #For each classifiers we are going to store the average misclassification error (computed with LOOCV) made with each patients


    for j in range(len(df)):
        t1 = time.time()
        f = open(log, 'a')
        f.write('Patient {} \n'.format(j))
        f.close()
        out = df.iloc[j, :]
        df_2 = df.drop([j])
        df_2.reset_index(drop=True, inplace=True)

        ndf_err = cmu.error_matrix(df_2, cls, nbcpus,funct)


        f = open(log, 'a')
        f.write('cost matrix computed \n')
        f.close()


        cost = cmu.cost_classifiers(ndf_err)


        if min_k == 1:
        #k = 1 : As a first pair, we take one with the lowest error
            temp = min(cost.values())
            res = [key for key in cost.keys() if cost[key] == temp] #Many classifiers can have the lowest error
            pair = [res[0]] #We take one arbitrary


            pred, proba = create_and_predict_metamodel(df_2, out, pair, nbcpus, funct)
            if proba != -1:
                k_mis[1].append(abs(out['diagnostic']-pred))
            else:  #case of proba = -1 means that all the classifiers in the metamodel predicted the point as uncertain
                print('case of unknown point in oka')

            min_k +=1

        #k > 1: ensemble classifiers
        for k in range(min_k, max_k):
            t0 = time.time()
            mve, pairs, algo = find_k_metamodel(df_2, ndf_err, cost, k, nbcpus, strat)
            pred, proba = create_and_predict_metamodel(df_2, out, pairs, nbcpus, funct)

            if proba != -1:
                k_mis[k].append(abs(out['diagnostic']-pred))
            else:  #case of proba = -1 means that all the classifiers in the metamodel predicted the point as uncertain
                print('case of unknown point in oka')

            f = open(log, 'a')
            f.write('Creation metamodel for k = {} in {}s \n'.format(k, time.time()-t0))
            f.close()

        f = open(log, 'a')
        f.write('End patient in {}s \n'.format(time.time() - t1))
        f.close()


    k_error = {k : np.mean(k_mis[k]) for k in range(3, max_k)}

    return k_error



def optimal_k(k_error):
    mini = min(k_error.values())
    keys = [k for k in k_error.keys() if k_error[k] == mini]
    return min(keys), mini


def print_k_error(k_error, file):
    x, y = k_error.keys(), k_error.values()
    plt.figure()
    plt.plot(x,y)
    plt.title('Average error for ensembles of classifiers')
    plt.xlabel('Nb of classifiers')
    plt.ylabel('Average error')
    if file:
        plt.savefig(file)
    else:
        plt.show()
