from Module import cost_matrix_uncertainty as cmu
from Module import measures as ms

import copy
import multiprocessing as mp
import time


#Filter the matrix: eliminate redundant transcripts

def filter_matrix_greedy(ndf):
    cols = ndf.columns.tolist()
    cols.remove('phenotype')
    used_trans = list()
    delete_col = list()
    for i in range(len(cols)):
        trans = cols[i].split("/")
        p1, p2 = trans[0], trans[1]
        if p1 in used_trans or p2 in used_trans:
            delete_col.append(cols[i])
        else:
            used_trans.append(p1)
            used_trans.append(p2)
    ndf.drop(labels=delete_col, axis = 1, inplace = True)
    return ndf

def filter_matrix_adapt(ndf, cls):
    cols = ndf.columns.tolist()
    cols.remove('phenotype')
    delete_col = list()
    for i in range(len(cols)):
        trans = cols[i].split("/")
        p1, p2 = trans[0], trans[1]
        if not cls == cols[i] and (p1 in cls or p2 in cls):
            delete_col.append(cols[i])
    ndf.drop(labels=delete_col, axis = 1, inplace = True)
    return ndf

def filter_pairs_adapt(pairs, cls):
    delete_pairs = list()
    idx = pairs.index(cls)
    for i in range(idx, len(pairs)):
        trans = pairs[i].split("/")
        p1, p2 = trans[0], trans[1]
        if p1 in cls or p2 in cls:
            delete_pairs.append(pairs[i])
    for x in delete_pairs:
        pairs.remove(x)
    return pairs


#########N-best



def NB(df, ndf_, k, nbcpus, mes = ms.MVE):

    ndf = copy.deepcopy(ndf_)
    ndf = filter_matrix_greedy(ndf)
    pairs = ndf.columns.tolist()[1:k+1]
    set = [ndf[p].values.tolist() for p in pairs]
    phen = ndf['phenotype']

    return pairs, ms.MVE(set), ndf


##### Forward search


def test_candidate_FS(cand_pairs, set_pairs, ndf, mes, i):

    cp = cand_pairs[i]

    candidate_set_pairs = copy.deepcopy(set_pairs)
    candidate_set_pairs.append(ndf[cp].values.tolist())
    cp_ms = mes(candidate_set_pairs)


    return (i, cp, cp_ms)



def FS(df, ndf_, k, nbcpus, mes = ms.MVE, start = 0):
    try:
        nbcpus = int (os.getenv('OMP_NUM_THREADS') )
    except:
        pass
    pool = mp.Pool(nbcpus)

    ndf = copy.deepcopy(ndf_)
    pairs_ = ndf.columns.tolist()[1:]
    pairs = [pairs_[start]]
    set_pairs = [ndf[pairs[0]].values.tolist()]
    phen = ndf['phenotype']
    ind = 1
    tot_ind = ind

    while len(pairs) < k:

        cand_pairs = pairs_[ind:ind+20]
        best_cp_ms = len(ndf)
        best_cand = None

        vals = [(cand_pairs, set_pairs, ndf, mes, i) for i in range(len(cand_pairs))]

        res = pool.starmap(test_candidate_FS, vals, max(1,len(vals)//nbcpus))
        res.sort(key=lambda x:x[2])

        i, cp, cp_ms = res[0]
        best_cp_ms = cp_ms
        best_cand = cp
        ind = tot_ind + i +1

        tot_ind = ind

        if best_cand != None:
            pairs.append(best_cand)
            set_pairs.append(ndf[best_cand].values.tolist())
            ndf = filter_matrix_adapt(ndf, best_cand)
            pairs_ = filter_pairs_adapt(pairs_, best_cand)
        else:
            print("Adding another classifier doesn't improve the metamodel")
            break
    return pairs, ms.MVE(set_pairs), ndf


#### Backward Search


def test_candidate_BS(cand_pairs, set_pairs, ndf, mes, i):
    #print('inside BS')
    cp = cand_pairs[i]
    candidate_set_pairs = copy.deepcopy(set_pairs)
    candidate_set_pairs.remove(ndf[cp].values.tolist())
    cp_ms = mes(candidate_set_pairs)
    #print((i, cp, cp_ms))
    return (i, cp, cp_ms)


def BS(df, ndf_, k, nbcpus, mes = ms.F2, end = 50):
    try:
        nbcpus = int (os.getenv('OMP_NUM_THREADS') )
    except:
        pass
    pool = mp.Pool(nbcpus)

    ndf = copy.deepcopy(ndf_)
    pairs_ = ndf.columns.tolist()[1:]
    pairs = [pairs_[i] for i in range(min(end, len(pairs_)))]
    set_pairs = [ndf[p].values.tolist() for p in pairs]
    phen = ndf['phenotype']

    while len(pairs) > k:

        cand_pairs = pairs.copy()

        best_cand = None

        vals = [(cand_pairs, set_pairs, ndf, mes, i) for i in range(len(cand_pairs))]

        res = pool.starmap(test_candidate_BS, vals, max(1,len(vals)//nbcpus))

        res.sort(key=lambda x:x[2])

        i, cp, cp_ms = res[0]
        best_cp_ms = cp_ms

        best_cand = cp

        if best_cand != None:
            pairs.remove(best_cand)
            set_pairs.remove(ndf[best_cand].values.tolist())
        else:
            print("Adding another classifier doesn't improve the metamodel")
            break
    return pairs, ms.MVE(set_pairs), ndf
