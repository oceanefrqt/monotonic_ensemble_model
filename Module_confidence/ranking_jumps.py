import copy
import numpy as np


def rank_h(x, D, rev):
    h = [d[0] for d in D]
    h = sorted(h, reverse=rev)
    r = h.index(x)

    if rev:
        return len(D) - r + 1
    else:
        return r+1

def rank_v(x, D, up):
    h = [d[1] for d in D]
    h.sort()
    r = h.index(x)
    #if up:
    return r+1
    #else:
    #    return len(D) - r + 1

def manhattan_distance_rank(x, y):
    rhx, rvx = x
    rhy, rvy = y
    return abs(rhx-rhy) + abs(rvx-rvy)


def compute_label(r_out, D, RD, k, labs):
    dist = {}
    for i in range(len(D)):
        d = D[i]
        rd = RD[i]
        dist[d] = manhattan_distance_rank(r_out, rd)

    ord_dist =  sorted(dist.items(), key=lambda t: t[1])[:k]
    c1 = 0
    c0 = 0
    for di in ord_dist:
        if labs[di[0]]==0:
            c0+=1
        elif labs[di[0]]==1:
            c1+=1

    if c0 <= c1:
        pred_lab = 1
    else:
        pred_lab = 0

    return pred_lab


def recompute_rank(out, rank_D):
    RD = list()
    for i in range(len(rank_D)):
        rh, rv = rank_D[i]
        if rh >= out[0]:
            rh += 1
        if rv >= out[1]:
            rv += 1
        RD.append((rh, rv))
    return RD







def jump_borders(out, X, bpr, bpb, rev, up):

    #Initialization
    labs = {x[0]:x[2] for x in X}
    D = [x[0] for x in X]
    #print('len D', len(D))
    D_hat = copy.deepcopy(D)
    D_hat.append(out)
    #print('len D_hat', len(D_hat))

    rxo = rank_h(out[0], D_hat, rev)
    ryo = rank_v(out[1], D_hat, up)

    rank_bpr = {}
    rank_bpb = {}
    for i in range(len(D)):
        rh = rank_h(D[i][0], D_hat, rev)
        rv = rank_v(D[i][1], D_hat, up)
        if D[i] in bpr:
            rank_bpr[D[i]] = (rh, rv)
        if D[i] in bpb:
            rank_bpb[D[i]] = (rh, rv)



    r_out = (rxo, ryo)
    #print('rout', r_out)

    #CASE2
    if rev and up:
        flag = -1
        mx = 0
        my = 0
        for pr in bpr:
            if rank_bpr[pr][0] >= rxo and rank_bpr[pr][1] <= ryo:
                flag = 1
                if abs(rank_bpr[pr][0] - rxo) > mx:
                    mx = abs(rank_bpr[pr][0] - rxo)
                if abs(rank_bpr[pr][1] - ryo) > my:
                    my = abs(rank_bpr[pr][1] - ryo)

        for pb in bpb:
            if rank_bpb[pb][0] <= rxo and rank_bpb[pb][1] >= ryo:
                flag = 0
                if abs(rank_bpb[pb][0] - rxo) > mx:
                    mx = abs(rank_bpb[pb][0] - rxo)
                if abs(rank_bpb[pb][1] - ryo) > my:
                    my = abs(rank_bpb[pb][1] - ryo)

    #CASE1
    elif not rev and up:
        flag = -1
        mx = 0
        my = 0
        for pr in bpr:
            if rank_bpr[pr][0] <= rxo and rank_bpr[pr][1] <= ryo:
                flag = 1
                if abs(rank_bpr[pr][0] - rxo) > mx:
                    mx = abs(rank_bpr[pr][0] - rxo)
                if abs(rank_bpr[pr][1] - ryo) > my:
                    my = abs(rank_bpr[pr][1] - ryo)

        for pb in bpb:
            if rank_bpb[pb][0] >= rxo and rank_bpb[pb][1] >= ryo:
                flag = 0
                if abs(rank_bpb[pb][0] - rxo) > mx:
                    mx = abs(rank_bpb[pb][0] - rxo)
                if abs(rank_bpb[pb][1] - ryo) > my:
                    my = abs(rank_bpb[pb][1] - ryo)
    #CASE3
    elif not rev and not up:
        flag = -1
        mx = 0
        my = 0
        for pr in bpr:
            if rank_bpr[pr][0] >= rxo and rank_bpr[pr][1] >= ryo:
                flag = 1
                if abs(rank_bpr[pr][0] - rxo) > mx:
                    mx = abs(rank_bpr[pr][0] - rxo)
                if abs(rank_bpr[pr][1] - ryo) > my:
                    my = abs(rank_bpr[pr][1] - ryo)

        for pb in bpb:
            if rank_bpb[pb][0] <= rxo and rank_bpb[pb][1] <= ryo:
                flag = 0
                if abs(rank_bpb[pb][0] - rxo) > mx:
                    mx = abs(rank_bpb[pb][0] - rxo)
                if abs(rank_bpb[pb][1] - ryo) > my:
                    my = abs(rank_bpb[pb][1] - ryo)

    #CASE4
    elif rev and not up:
        flag = -1
        mx = 0
        my = 0
        for pr in bpr:
            if rank_bpr[pr][0] <= rxo and rank_bpr[pr][1] >= ryo:
                flag = 1
                if abs(rank_bpr[pr][0] - rxo) > mx:
                    mx = abs(rank_bpr[pr][0] - rxo)
                if abs(rank_bpr[pr][1] - ryo) > my:
                    my = abs(rank_bpr[pr][1] - ryo)

        for pb in bpb:
            if rank_bpb[pb][0] >= rxo and rank_bpb[pb][1] <= ryo:
                flag = 0
                if abs(rank_bpb[pb][0] - rxo) > mx:
                    mx = abs(rank_bpb[pb][0] - rxo)
                if abs(rank_bpb[pb][1] - ryo) > my:
                    my = abs(rank_bpb[pb][1] - ryo)


    #print('Pred', flag)
    if flag == -1:
        #print('r out', r_out)


        r_bpr = [rank_bpr[pr] for pr in bpr]
        r_bpr.append(r_out)
        r_bprx = copy.deepcopy(r_bpr)
        r_bprx.sort(key= lambda x : x[0])
        #print('r_bprx', r_bprx)
        r_bpry = copy.deepcopy(r_bpr)
        r_bpry.sort(key= lambda x : x[1])
        #print('r_bpry', r_bpry)

        ix = r_bprx.index(r_out)
        #print('ix', ix)
        if ix > 0 and ix < len(r_bpr)-1:
            antx = r_bprx[ix-1]
            posx = r_bprx[ix+1]
        elif ix == 0:
            antx = r_out
            posx = r_bprx[ix+1]
        elif ix == len(r_bpr)-1:
            antx = r_bprx[ix-1]
            posx = r_out

        #print('x red', antx, posx)

        iy = r_bpry.index(r_out)
        #print('iy', iy)
        if iy > 0 and iy < len(r_bpry)-1:
            anty = r_bpry[iy-1]
            posy = r_bpry[iy+1]
        elif iy == 0:
            anty = r_out
            posy = r_bpry[iy+1]
        elif iy == len(r_bpry)-1:
            anty = r_bpry[iy-1]
            posy = r_out
        #print('y red', anty, posy)

        if not rev and up:
            myr = abs(antx[1] - r_out[1])
            mxr = abs(anty[0] - r_out[0])

        elif rev and up:
            myr = abs(posx[1] - r_out[1])
            mxr = abs(anty[0] - r_out[0])

        elif not rev and not up:
            myr = abs(posx[1] - r_out[1])
            mxr = abs(posy[0] - r_out[0])

        elif rev and not up:
            myr = abs(antx[1] - r_out[1])
            mxr = abs(posy[0] - r_out[0])




        r_bpb = [rank_bpb[pb] for pb in bpb]
        r_bpb.append(r_out)
        r_bpbx = copy.deepcopy(r_bpb)
        r_bpbx.sort(key= lambda x : x[0])
        #print('r_bpbx', r_bpbx)
        r_bpby = copy.deepcopy(r_bpb)
        r_bpby.sort(key= lambda x : x[1])
        #print('r_bpby', r_bpby)

        ix = r_bpbx.index(r_out)
        if ix > 0 and ix < len(r_bpb)-1:
            antx = r_bpbx[ix-1]
            posx = r_bpbx[ix+1]
        elif ix == 0:
            antx = r_out
            posx = r_bpbx[ix+1]
        elif ix == len(r_bpb)-1:
            antx = r_bpbx[ix-1]
            posx = r_out

        #print('x blue', antx, posx)

        iy = r_bpby.index(r_out)
        if iy > 0 and iy < len(r_bpby)-1:
            anty = r_bpby[iy-1]
            posy = r_bpby[iy+1]
        elif iy == 0:
            anty = r_out
            posy = r_bpby[iy+1]
        elif iy == len(r_bpby)-1:
            anty = r_bpby[iy-1]
            posy = r_out
        #print('y blue', anty, posy)



        if not rev and not up:
            myb = abs(antx[1] - r_out[1])
            mxb = abs(anty[0] - r_out[0])

        elif rev and not up:
            myb = abs(posx[1] - r_out[1])
            mxb = abs(anty[0] - r_out[0])

        elif not rev and up:
            myb = abs(posx[1] - r_out[1])
            mxb = abs(posy[0] - r_out[0])

        elif rev and up:
            myb = abs(antx[1] - r_out[1])
            mxb = abs(posy[0] - r_out[0])

        if mxr == 0:
            mxr = np.nan
        if myr == 0:
            myr = np.nan
        if mxb == 0:
            mxb = np.nan
        if myb == 0:
            myb = np.nan




        return [flag, (mxr, myr), (mxb, myb)]
    else:
        return [flag, (mx, my)]


def jumps_adjusted_labels(out, X, bpr, bpb, rev, up):
    label, jumps = jump_borders(out, X, bpr, bpb, rev, up)
    labs = {x[0]:x[2] for x in X}
    D = [x[0] for x in X]
    D_hat = copy.deepcopy(D)
    D_hat.append(out)
    rxo = rank_h(out[0], D_hat, rev)
    ryo = rank_v(out[1], D_hat, up)

    #CASE2
    if rev and up:
        if label == 1:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) > rxo and rank_h(x[0], D_hat, rev) <= rxo + jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) < ryo and rank_v(x[1], D_hat, up) >= ryo - jumps[1]]
        elif label == 0:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) < rxo and rank_h(x[0], D_hat, rev) >= rxo - jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) > ryo and rank_v(x[1], D_hat, up) <= ryo + jumps[1]]

    #CASE1
    elif not rev and up:
        if label == 1:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) < rxo and rank_h(x[0], D_hat, rev) >= rxo - jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) < ryo and rank_v(x[1], D_hat, up) >= ryo - jumps[1]]
        elif label == 0:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) > rxo and rank_h(x[0], D_hat, rev) <= rxo + jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) > ryo and rank_v(x[1], D_hat, up) <= ryo + jumps[1]]

    #CASE3
    elif not rev and not up:
        if label == 0:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) > rxo and rank_h(x[0], D_hat, rev) <= rxo + jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) < ryo and rank_v(x[1], D_hat, up) >= ryo - jumps[1]]
        elif label == 1:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) < rxo and rank_h(x[0], D_hat, rev) >= rxo - jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) > ryo and rank_v(x[1], D_hat, up) <= ryo + jumps[1]]

    #CASE4
    elif rev and not up:
        if label == 0:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) < rxo and rank_h(x[0], D_hat, rev) >= rxo - jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) < ryo and rank_v(x[1], D_hat, up) >= ryo - jumps[1]]
        elif label == 1:
            pts_jumped_h = [x for x in D if rank_h(x[0], D_hat, rev) > rxo and rank_h(x[0], D_hat, rev) <= rxo + jumps[0]]
            pts_jumped_v = [x for x in D if rank_v(x[1], D_hat, up) > ryo and rank_v(x[1], D_hat, up) <= ryo + jumps[1]]


    if label == 1:
        jx = len([p for p in pts_jumped_h if labs[p] == 1])
        jy = len([p for p in pts_jumped_v if labs[p] == 1])
    elif label == 0:
        jx = len([p for p in pts_jumped_h if labs[p] == 0])
        jy = len([p for p in pts_jumped_v if labs[p] == 0])

    if label in [0,1]:
        return [label, (jx, jy)]
    else:
        return [None]


def score(jump, maxi):
    if jump[0] == -1:
        return 0
    else:
        if jump[0] == 1:
            sign = 1
        elif jump[0] == 0:
            sign = -1
        return sign*sum(jump[1])/maxi


def prediction_score(score):
    if score > 0:
        return 1
    elif score < 0:
        return 0
    else:
        return -1
