from Module import tools

import numpy as np
from math import ceil, log, pow
import pandas as pd
import os
import multiprocessing as mp
import copy
import time



#Useful functions

def err(v, w, z): #error function defined in the paper
    return w*abs(v-z)

next_ = lambda x: int( pow(2, ceil(log(x, 2))))

def index_leaves(A, k):
    #As we use a balanced binary tree, leaves can be at at most two levels. This function constructs
    #a dictionary linking each leaf index to the index of the corresponding node in the tree
    p_lev = next_(k) - k
    lev = k - p_lev
    ind_leaves = {}
    for i in range(1, lev+1):
        ind_leaves[i] = (len(A)-lev) + i
    for i in range(1, p_lev+1):
        ind_leaves[lev+i ] = (len(A)-k) + i
    return ind_leaves

def is_leaf(ind_leaves, num):
    #confirms or not whether the node under study is a leaf
    if num in ind_leaves.values():
        return True
    else:
        return False

def find_leaves(ind_leaves, num, L):
    #return a list with all the leaves below the node num
    if not is_leaf(ind_leaves, num):
        find_leaves(ind_leaves, 2*num, L)
        find_leaves(ind_leaves, 2*num +1, L)
    else:
        L.append(num)

def Z_(H, A, ind_leaves):
    #Show Z : this one won't appear in the final programm. It's just to check the tree for the moment
    Z = list()
    for i in range(1,len(H)+1):
        v_i = ind_leaves[i]
        Z.append(int(compute_Z(A,v_i)))
    print('Z', Z)
    return Z

def is_A_balanced(A):
    flag = True
    for i in range(1, len(A), 2):
        if A[i] != 0 and A[i+1] != 0:
            flag = False
    return flag

########Initialize A
def initialization(data, rev):
    #rev enable us to test increasing and decreasing isotonicity
    #X is an array with sorted data
    #H is a soretd list of all column values
    #A is the array initialised at zero
    #ind_leaves is the dictionnary linking each leaf index to the index of the corresponding node in the tree
    X = sorted(data, reverse = rev)
    H = sorted(list(set([X[i][0][1] for i in range(len(X))])))
    A = np.zeros(2*len(H)+1)
    H.append(float('inf'))
    ind_leaves = index_leaves(A, len(H))
    return X, H, A, ind_leaves


#####STEP 1 : compute Z and Z(g,c_g)
def compute_Z(A,v):
    #compute Z by going up the tree from leaf v to the root (Z(g,h) = sum of z_v on the path to the root)
    Z = A[v-1]
    while True:
        if v == 1:
            break
        v = v//2
        Z += A[v-1]
    return Z

#These two functions help for updating the tree when Z(g,c_g) is computed (it can change a lot in the path to the root)
def degenerate_interval(A, v, ind_leaves):
    p = v //2
    mod = v%2
    while True:
        if p == 1:
            break
        if A[p-1] != 0:
            add_value(A, p, v, ind_leaves)
            A[p-1] = 0
        p = p//2
        mod = p%2



def add_value(A, p, v, ind_leaves):
    L = list()
    find_leaves(ind_leaves, p, L)
    for l in L:
        if l != v:
            A[l-1] += A[p-1]


def rebalance(A,v, ind_leaves):
    if v != 0:
        p = v//2
        mod = v%2
        w = 2*p+(1-mod)
        if A[v-1] != 0 and A[w-1] !=0:
            delta = min(A[v-1], A[w-1])
            A[p-1] += delta
            A[v-1] -= delta
            A[w-1] -= delta
    else:
        mini = float('inf')
        for i in ind_leaves.values():
            Z = compute_Z(A,i)
            if Z < mini:
                mini = Z
        A[0] = mini




def step1(ind_cg, A, nb_leaves, ind_leaves, err1, S, H):
    #t0 = time.process_time()
    mini = float('inf')
    h = 0
    for i in range(ind_cg, nb_leaves+1):
        v_i = ind_leaves[i]
        Z = compute_Z(A,v_i)

        if Z < mini:
            mini = Z
            ind = i
            c = 1
            h = H[i-1]
        elif Z == mini:
            c+=1
    if c>1:
        while compute_Z(A,ind_leaves[ind]) == mini and ind < nb_leaves:
            ind+=1
        if ind == nb_leaves:
            h = H[ind-1]  #value of the leaves that give minimum Z(g,h)
        else:
            h = H[ind-2]
    S.append(h)



    deg = compute_Z(A,ind_leaves[ind_cg]) - A[ind_leaves[ind_cg]-1] #sum of z_v on the path from c_g to the root, but minus the leaf c_g

    #if the new value of Z(g,c_g) minus deg is greater or equal to 0, then the value in A for leaf c_g is equal to the difference
    # however, in the other case, it means that we have to change the path to make it correspond


    if deg <= mini + err1:
        A[ind_leaves[ind_cg]-1] = mini + err1 - deg
    else:
        A[ind_leaves[ind_cg]-1] = mini + err1
        degenerate_interval(A, ind_leaves[ind_cg], ind_leaves)
    #print('Step1', time.process_time() - t0)






#######STEP 2 : update right and left

#Update right c_g
def v_has_right(v, A):
    #indicates whether the node v has a right child branch
    if 2*v+1 <= len(A):
        return True
    else:
        return False

def update_right(v, A, val):
    #update right child branch by adding val
    A[2*v] += val

def update_all_right(v, A, err0):
    p = v//2
    while True:
        if v_has_right(p, A) and 2*p+1 !=v:
            update_right(p, A, err0)

        if p == 1: #end when root has been updated
            break
        v = p
        p = p//2



#Update left c_g
def v_has_left(v, A):
    #indicates whether the node v has a left child branch
    if 2*v <= len(A):
        return True
    else:
        return False

def update_left(v, A, val):
    #update left child branch by adding val
    A[2*v-1] += val



def update_all_left(v, A, err1):
    p = v//2
    while True:
        if v_has_left(p, A) and 2*p != v:
            update_left(p, A, err1)

        if p == 1: #end when root has been updated
            break
        v = p
        p = p//2


def step2(A, v, err0, err1):
    #t0 = time.process_time()
    #add the error in left and right intervals
    update_all_right(v, A, err0)
    update_all_left(v, A, err1)
    #print('Step2', time.process_time() - t0)





########RECURSION

def recursion(A, H, c_g, ind_leaves, err0, err1, nb_leaves, S):
    ind_cg = H.index(c_g) + 1
    v_cg = ind_leaves[ind_cg]

    step1(ind_cg, A, nb_leaves, ind_leaves, err1, S, H)

    step2(A, v_cg, err0, err1)



#######TRACEBACK
def find_h_Z_min(A, ind_leaves):
    p = 1
    while True:
        v = 2*p
        w = 2*p +1
        if A[v-1] == 0:
            p = v
        else:
            p = w

        if is_leaf(ind_leaves,p):
            return p
    return None

def search_key(dico, val):
    l = [c for c,v in dico.items() if v==val]
    if len(l) != 0:
        return l[0]
    else:
        return -1


def traceback(A, X, H, ind_leaves, S):
    b = search_key(ind_leaves,find_h_Z_min(A, ind_leaves))
    h = H[b-1]
    breakpoint = list()
    for i in range(len(X)-1, -1, -1):
        x = X[i] #ie point en partant de la fin
        xy, w, lab = x
        cg = xy[1] #column
        if h == cg:
            h = S[i]
            breakpoint.append(xy)
    return min(A), breakpoint


def labels_point(X, bpr, rev, up):
    r_p = list()
    b_p = list()

    for x in X:
        x = x[0]
        if x in bpr:
            r_p.append(x)
        else:
            if not rev and up: #CASE 1
                flag = 0
                for br in bpr:
                    if x[0] >= br[0] and x[1] >= br[1]:
                        flag = 1
                if flag == 0:
                    b_p.append(x)
                else:
                    r_p.append(x)

            if rev and up: #CASE 2
                flag = 0 #consider as blue by default
                for br in bpr:
                    if x[0] <= br[0] and x[1] >= br[1]:
                        flag = 1
                if flag == 0:
                    b_p.append(x)
                else:
                    r_p.append(x)

            if not rev and not up: #CASE 3
                flag = 0 #consider as blue by default
                for br in bpr:
                    if x[0] <= br[0] and x[1] <= br[1]:
                        flag = 1
                if flag == 0:
                    b_p.append(x)
                else:
                    r_p.append(x)

            if rev and not up: #CASE 4
                flag = 0 #consider as blue by default
                for br in bpr:
                    if  x[0] >= br[0] and x[1] <= br[1]:
                        flag =1
                if flag == 0:
                    b_p.append(x)
                else:
                    r_p.append(x)
    return r_p, b_p




#previous functions give us the data labelled as 0 and 1. But if we want
#to predict new points, we must know the official separation. As we prefer
#false positive rather than false negative, we draw the lines closest
#to "blue" points

def breakpoint_b(X, b_p, rev, up):
    bpb = list()
    b_ps = sorted(b_p)
    if not rev and up: #CASE 1
        while len(b_ps) != 0:
            maxi = b_ps[-1]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[1] > y]
            b_ps = sorted(b_ps)


    elif rev and up: #CASE 2
        while len(b_ps) != 0:
            maxi = b_ps[0]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[1] > y]
            b_ps = sorted(b_ps)
    elif not rev and not up: #CASE 3
        while len(b_ps) != 0:
            maxi = b_ps[0]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[1] < y]
            b_ps = sorted(b_ps)

    elif rev and not up: #CASE 4
        while len(b_ps) != 0:
            maxi = b_ps[-1]
            x, y = maxi
            bpb.append(maxi)
            b_ps = [pt for pt in b_ps if pt[1] < y]
            b_ps = sorted(b_ps)
    return bpb

#####MAIN FUNCTION

def compute_recursion(data, case = None):
        #return X : sorted data
        # labels : labelisation of each points
        # bpb : points creating the delimitation of blue area
        # bpr : points creating the delimitation of red area

    models = {}
    if case is None:

        for rev in [True, False]:

            X, H, A, ind_leaves = initialization(data, rev)

            S = list()
            labs = [x[2] for x in X]
            nb_leaves = len(H)

            for i in range(len(X)): #((r,c), w, label)
                x = X[i]
                xy, w, lab = x
                cg = xy[1]
                err0, err1 = err(lab, w, 0), err(lab, w, 1)

                recursion(A, H, cg, ind_leaves, err0, err1, nb_leaves, S)

            while not is_A_balanced(A):
                for v in range(len(A), -1, -1):
                    rebalance(A,v, ind_leaves)

            error, bpr = traceback(A, X, H, ind_leaves, S)


            for up in [True, False]:

                r_p, b_p = labels_point(X, bpr, rev, up)

                bpb = breakpoint_b(X, b_p, rev, up)


                if rev and up:
                    models[2] = (bpr, bpb)
                elif not rev and up:
                    models[1] = (bpr, bpb)
                elif not rev and not up:
                    models[3] = (bpr, bpb)
                else:
                    models[4] = (bpr, bpb)

    else:
        #print('case {}'.format(case))
        rev, up = case[0], case[1]
        X, H, A, ind_leaves = initialization(data, rev)
        S = list()
        labs = [x[2] for x in X]
        nb_leaves = len(H)
        #print(X)
        for i in range(len(X)): #((r,c), w, label)
            x = X[i]
            xy, w, lab = x
            cg = xy[1]
            err0 = err(lab, w, 0)
            err1 = err(lab, w, 1)

            recursion(A, H, cg, ind_leaves, err0, err1, nb_leaves, S)

        while not is_A_balanced(A):
            for v in range(len(A), -1, -1):
                rebalance(A,v, ind_leaves)

        error, bpr = traceback(A, X, H, ind_leaves, S)

        r_p, b_p = labels_point(X, bpr, rev, up)

        bpb = breakpoint_b(X, b_p, rev, up)

        models[case[2]] = (bpr, bpb)

    return X, models, r_p, b_p


#### PREDICTION For diff case

def predict_uncertainty(p, bpr, bpb, rev, up):
    flag = -1
    if not rev and up: #CASE 1
        for b in bpr:
            if p[0] >= b[0] and p[1] >= b[1]:
                flag = 1

        for b in bpb:
            if p[0] <= b[0] and p[1] <= b[1]:
                flag = 0


    elif rev and up: #CASE 2
        for b in bpr:
            if p[0] <= b[0] and p[1] >= b[1]:
                flag = 1

        for b in bpb:
            if p[0] >= b[0] and p[1] <= b[1]:
                flag = 0

    elif not rev and not up: #CASE 3
        for b in bpr:
            if p[0] <= b[0] and p[1] <= b[1]:
                flag = 1

        for b in bpb:
            if p[0] >= b[0] and p[1] >= b[1]:
                flag = 0


    elif rev and not up: #CASE 4
        for b in bpr:
            if p[0] >= b[0] and p[1] <= b[1]:
                flag = 1

        for b in bpb:
            if p[0] <= b[0] and p[1] >= b[1]:
                flag = 0
    return flag



def predict_severe(p, bpr, bpb, rev, up):
    # points in grey area are automatically labelled in red area
    flag = 1
    if not rev and up: #CASE 1
        for b in bpb:
            if p[0] <= b[0] and p[1] <= b[1]:
                flag = 0


    elif rev and up: #CASE 2
        for b in bpb:
            if p[0] >= b[0] and p[1] <= b[1]:
                flag = 0

    elif not rev and not up: #CASE 3
        for b in bpb:
            if p[0] >= b[0] and p[1] >= b[1]:
                flag = 0


    elif rev and not up: #CASE 4
        for b in bpb:
            if p[0] <= b[0] and p[1] >= b[1]:
                flag = 0
    return flag


def predict_non_severe(p, bpr, bpb, rev, up):
    # points in grey area are automatically labelled in blue area
    flag = 0
    if not rev and up: #CASE 1
        for b in bpr:
            if p[0] >= b[0] and p[1] >= b[1]:
                flag = 1


    elif rev and up: #CASE 2
        for b in bpr:
            if p[0] <= b[0] and p[1] >= b[1]:
                flag = 1

    elif not rev and not up: #CASE 3
        for b in bpr:
            if p[0] <= b[0] and p[1] <= b[1]:
                flag = 1

    elif rev and not up: #CASE 4
        for b in bpr:
            if p[0] >= b[0] and p[1] <= b[1]:
                flag = 1

    return flag
