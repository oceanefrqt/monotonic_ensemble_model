from Module_confidence import cost_matrix_uncertainty as cmu
from Module_confidence import measures as ms

import copy
import multiprocessing as mp
import time


#Filter the matrix: eliminate redundant transcripts

def filter_pairs_greedy(pairs):
    used_trans = list()
    delete_col = list()
    for i in range(len(pairs)):
        p1, p2, key = pairs[i].split("/")
        if p1 in used_trans or p2 in used_trans:
            delete_col.append(pairs[i])
        else:
            used_trans.append(p1)
            used_trans.append(p2)
    for x in delete_col:
        pairs.remove(x)
    return pairs



def filter_pairs_adapt(pairs, cl):
    delete_pairs = list()
    idx = pairs.index(cl)
    for i in range(idx, len(pairs)):
        p1, p2, key = pairs[i].split("/")
        if p1 in cl or p2 in cl:
            delete_pairs.append(pairs[i])
    for x in delete_pairs:
        pairs.remove(x)
    return pairs


#########N-best



def NB(df, ndf_, cost, k, nbcpus, mes = ms.MVE):

    ndf = copy.deepcopy(ndf_)
    ndf.drop(['uncertain', 'error'], inplace=True)

    pairs = sorted(cost.items(), key=lambda t: t[1])
    pairs = [pairs[i][0] for i in range(len(pairs))]
    pairs = filter_pairs_greedy(pairs)
    pairs = pairs[0:k]

    set = [ndf[p].values.tolist() for p in pairs]

    return pairs, ms.MVE(set)


##### Forward search


def test_candidate_FS(cand_pairs, cls, ndf, mes, i):
    cp = cand_pairs[i]

    candidate_set_pairs = [ndf[cl].values.tolist() for cl in cls]
    candidate_set_pairs.append(ndf[cp].values.tolist())
    cp_ms = mes(candidate_set_pairs)

    return (i, cp, cp_ms)



def FS(df, ndf_, cost, k, nbcpus, mes = ms.MVE, jump = 30):
    pool = mp.Pool(nbcpus)

    ndf = copy.deepcopy(ndf_)
    ndf.drop(['uncertain', 'error'], inplace=True)

    temp = min(cost.values())
    res = [key for key in cost.keys() if cost[key] == temp] #Many classifiers can have the lowest error
    cl = res[0] #We take one arbitrary
    cls = [cl] #We start with an ensemble of k=1 classifiers

    pairs = sorted(cost.items(), key=lambda t: t[1])
    pairs = [pairs[i][0] for i in range(len(pairs))]

    ind = 1
    tot_ind = ind

    while len(cls) < k: #We add classifiers until we reach k
        #Condition if we reach the end of the list of classifiers, we start again from the begining by eliminating the already used cls
        if tot_ind + jump > len(pairs):
            pairs = [pairs[p] for p in range(len(pairs)) if pairs[p] not in cls]
            ind = 1
            tot_ind = ind

        cand_pairs = pairs[ind:ind+jump]

        vals = [(cand_pairs, cls, ndf, mes, i) for i in range(len(cand_pairs))]

        res = pool.starmap(test_candidate_FS, vals, max(1,len(vals)//nbcpus))
        res.sort(key=lambda x:x[2])

        i, cp, cp_ms = res[0]
        best_cp_ms = cp_ms
        best_cand = cp
        ind = tot_ind + i +1
        tot_ind = ind


        cls.append(best_cand)

        pairs = filter_pairs_adapt(pairs, best_cand)


    set = [ndf[p].values.tolist() for p in cls]

    return cls, ms.MVE(set)


#### Backward Search


def test_candidate_BS(cand_pairs, cls, ndf, mes, i):
    cp = cand_pairs[i]

    candidate_set_pairs = [ndf[cl].values.tolist() for cl in cls if cl != cp]

    cp_ms = mes(candidate_set_pairs)

    return (i, cp, cp_ms)


def BS(df, ndf_, cost, k, nbcpus, mes = ms.F2, end = 30):

    pool = mp.Pool(nbcpus)
    ndf = copy.deepcopy(ndf_)
    ndf.drop(['uncertain', 'error'], inplace=True)


    pairs = sorted(cost.items(), key=lambda t: t[1])
    pairs = [pairs[i][0] for i in range(len(pairs))]
    pairs = filter_pairs_greedy(pairs)
    cls = pairs[:end]

    while len(cls) > k:

        cand_pairs = cls.copy()

        vals = [(cand_pairs, cls, ndf, mes, i) for i in range(len(cand_pairs))]

        res = pool.starmap(test_candidate_BS, vals, max(1,len(vals)//nbcpus))

        res.sort(key=lambda x:x[2])

        i, cp, cp_ms = res[0]
        cls.remove(cp)


    set = [ndf[p].values.tolist() for p in cls]

    return cls, ms.MVE(set)
