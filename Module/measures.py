import math


#cls = classifier
# boolean prediction : 0 if the prediction is correct, 1 if it's wrong
#pred_xi : array of the boolean prediction of xi over the M classifiers
#set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients

def m_xi(pred_xi):
    #number of classifiers producing error for the input sample xi
    #pred_xi is the boolean prediction of xi over the M classifiers
    return sum(pred_xi)

def error_rate_clj(pred_clj):
    # error rate of jth classifier
    #pred_clj is the boolean prediction of the N patients by classifier j
    return sum(pred_clj)/len(pred_clj)

def ensemble_mean_error_rate(set):
    #average error rate over the M classifiers
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    M = len(set)
    e = 0
    for i in range(M):
        e += error_rate_clj(set[i])
    return e/M



def yi_MV_1(pred_xi):
    #pred_xi is the boolean prediction of xi over the M classifiers
    #majority boolean prediction error in favor of the wrong pred 1
    M = len(pred_xi)
    if m_xi(pred_xi) >= M/2:
        #if the nb of misclassification is greater than or equal to the nb of classifiers
        #then the preditcion made with majority voting is wrong
        return 1
    else:
        return 0

def yi_MV_0(pred_xi):
    #pred_xi is the boolean prediction of xi over the M classifiers
    #majority boolean prediction error in favor of correct pred 0
    M = len(pred_xi)
    if m_xi(pred_xi) > M/2:
        #if the nb of misclassification is greater than the nb of classifiers
        #then the preditcion made with majority voting is wrong
        return 1
    else:
        return 0

def MVE(set, meth = yi_MV_1):
    #majority voting error rate
    M = len(set)
    N = len(set[0])
    mve = 0
    for i in range(N):
        #construction of pred_xi
        pred_xi = list()
        for j in range(M):
            pred_xi.append(set[j][i])

        yi_mv = meth(pred_xi)
        mve += yi_mv
    return mve/N



def D2_ij(pred1, pred2):
    #disagreement measure for 2 pairs
    #pred1 and pred2 (size N) are the output of classifiers for the N patients
    N = len(pred1)
    D2 = 0
    for i in range(N):
        if pred1[i] != pred2[i]:
            D2 += 1
    return D2/N

def D2(set):
    #average disagreement measure over all pairs of a set
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    M = len(set)
    D2 = 0
    for i in range(M):
        for j in range(i+1, M):
            if i != j:
                D2 += D2_ij(set[i], set[j])

    return (2*D2)/(M*(M-1))


def F2_ij(pred1, pred2):
    #double fault measure
    #pred1 and pred2 (size N) are the output of classifiers for the N patients
    N = len(pred1)
    N11 = 0
    for i in range(N):
        if pred1[i] == 1 and pred2[i] == 1:
            #double fault
            N11 +=1
    return N11/N

def F2(set):
    #average double fault measure over all pairs of a set
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    M = len(set)
    F2 = 0
    for i in range(M):
        for j in range(i+1, M):
            if i != j:
                F2 += F2_ij(set[i], set[j])
    return (2*F2)/(M*(M-1))

def entropy(set):
    #set (size MxN) is a list of the boolean prediction of the M classifiers over the N patients
    #The entropy measure reaches its maximum (EN 1⁄4 1) for the highest disagreement, which is the case of observing M/2 votes with identical value (0 or 1) and
    #M M/2 with the alternative value. The lowest entropy (EN 1⁄4 0) is observed if all classifier outputs are
    #identical.
    M = len(set)
    N = len(set[0])

    EN = 0
    for i in range(N):
        #construction of pred_xi
        pred_xi = list()
        for j in range(M):
            pred_xi.append(set[j][i])

        EN += min(m_xi(pred_xi), M-m_xi(pred_xi)) / (M-math.ceil(M/2))
    return EN/N
