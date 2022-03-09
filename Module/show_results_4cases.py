from Module import monotonic_regression_uncertainty as mru
from Module import tools

import multiprocessing as mp

import matplotlib.pyplot as plt
from matplotlib import patches
import colorsys




def cr_models(p, df):
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = tools.equiv_key_case(key)
    tr1 = df[p1].values.tolist()
    tr2 = df[p2].values.tolist()
    diag = df['diagnostic'].values.tolist()
    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

    X, models = mru.compute_recursion(data, (rev, up, key))

    return p1, p2, models, data


def print_model(data, models, p1, p2, df1, pathname = None):
    #print the monotonic space with the 3 areas

    if '.1.1' in p1:
        p1 = p1[:-2]
    if '.1.1' in p2:
        p2 = p2[:-2]

    try:
        g1 = df1[df1['Probeset_ID'] == p1]['Gene.Symbol'].values.tolist()[0]
    except:
        g1 = p1
    try:
        g2 = df1[df1['Probeset_ID'] == p2]['Gene.Symbol'].values.tolist()[0]
    except:
        g2 = p2

    for key in models.keys():
        key = int(key)
        print('Key', key)
        plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.set_facecolor("lightgray")

        x_r = list()
        y_r = list()
        x_b = list()
        y_b = list()
        for i in range(len(data)):
            xy, w, lab = data[i]
            x, y = xy
            if lab == 0: #blue
                x_r.append(x)
                y_r.append(y)
            else: #red
                x_b.append(x)
                y_b.append(y)

        bpr, bpb, r_p, b_p = models[key]

        for bp in bpb:
            x, y = bp
            if key == 1:
                ax.add_artist(patches.Rectangle((0.0, 0.0), x, y, facecolor = 'lightskyblue', zorder = 1))
            elif key == 2:
                ax.add_artist(patches.Rectangle((x, 0), 1000, y, facecolor = 'lightskyblue', zorder = 1))
            elif key == 3:
                ax.add_artist(patches.Rectangle((x, y), 1000, 1000, facecolor = 'lightskyblue', zorder = 1))
            else:
                ax.add_artist(patches.Rectangle((0, y ), x, 1000, facecolor = 'lightskyblue', zorder = 1))


        for bp in bpr:
            x, y = bp


            if key == 1:
                ax.add_artist(patches.Rectangle((x, y), 1000, 1000, facecolor ='lightcoral', zorder = 1))
            elif key == 2:
                ax.add_artist(patches.Rectangle((0, y ), x, 1000, facecolor = 'lightcoral', zorder = 1))
            elif key == 3:
                ax.add_artist(patches.Rectangle((0.0, 0.0), x, y, facecolor = 'lightcoral', zorder = 1))
            else:
                ax.add_artist(patches.Rectangle((x, 0), 1000, y, facecolor = 'lightcoral', zorder = 1))


        plt.scatter(x_r, y_r, c = 'blue', zorder = 2)
        plt.scatter(x_b, y_b, c = 'red', zorder = 2)
        plt.xlabel(g1)
        plt.ylabel(g2)

        if pathname is not None:
            plt.savefig(pathname + g1 + '_' + g2  + '.png')

            f = open(pathname + 'gene.txt', 'a')
            f.write('{} : {}\n'.format(g1, p1))
            f.write('{} : {}\n'.format(g2, p2))
            f.close()
        else:
            plt.show()


def print_model_out(data, out, models, p1, p2, df1, pathname = None):
    #print the monotonic space with the 3 areas

    if '.1.1' in p1:
        p1 = p1[:-2]
    if '.1.1' in p2:
        p2 = p2[:-2]

    try:
        g1 = df1[df1['Probeset_ID'] == p1]['Gene.Symbol'].values.tolist()[0]
    except:
        g1 = p1
    try:
        g2 = df1[df1['Probeset_ID'] == p2]['Gene.Symbol'].values.tolist()[0]
    except:
        g2 = p2

    for key in models.keys():
        key = int(key)
        print('Key', key)
        plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.set_facecolor("lightgray")

        x_r = list()
        y_r = list()
        x_b = list()
        y_b = list()
        for i in range(len(data)):
            xy, w, lab = data[i]
            x, y = xy
            if lab == 0: #blue
                x_r.append(x)
                y_r.append(y)
            else: #red
                x_b.append(x)
                y_b.append(y)

        bpr, bpb, r_p, b_p = models[key]

        for bp in bpb:
            x, y = bp
            if key == 1:
                ax.add_artist(patches.Rectangle((0.0, 0.0), x, y, facecolor = 'lightskyblue', zorder = 1))
            elif key == 2:
                ax.add_artist(patches.Rectangle((x, 0), 1000, y, facecolor = 'lightskyblue', zorder = 1))
            elif key == 3:
                ax.add_artist(patches.Rectangle((x, y), 1000, 1000, facecolor = 'lightskyblue', zorder = 1))
            else:
                ax.add_artist(patches.Rectangle((0, y ), x, 1000, facecolor = 'lightskyblue', zorder = 1))


        for bp in bpr:
            x, y = bp


            if key == 1:
                ax.add_artist(patches.Rectangle((x, y), 1000, 1000, facecolor ='lightcoral', zorder = 1))
            elif key == 2:
                ax.add_artist(patches.Rectangle((0, y ), x, 1000, facecolor = 'lightcoral', zorder = 1))
            elif key == 3:
                ax.add_artist(patches.Rectangle((0.0, 0.0), x, y, facecolor = 'lightcoral', zorder = 1))
            else:
                ax.add_artist(patches.Rectangle((x, 0), 1000, y, facecolor = 'lightcoral', zorder = 1))


        plt.scatter(x_r, y_r, c = 'blue', zorder = 2)
        plt.scatter(x_b, y_b, c = 'red', zorder = 2)
        plt.scatter(out[0], out[1], c = 'g', zorder = 2)
        plt.xlabel(g1)
        plt.ylabel(g2)

        if pathname is not None:
            plt.savefig(pathname + g1 + '_' + g2  + '.png')

            f = open(pathname + 'gene.txt', 'a')
            f.write('{} : {}\n'.format(g1, p1))
            f.write('{} : {}\n'.format(g2, p2))
            f.close()
        else:
            plt.show()

def print_severe(data, models, p1, p2, df1, pathname = None):
    #print the monotonic space with the 3 areas

    if '.1.1' in p1:
        p1 = p1[:-2]
    if '.1.1' in p2:
        p2 = p2[:-2]

    try:
        g1 = df1[df1['Probeset_ID'] == p1]['Gene.Symbol'].values.tolist()[0]
    except:
        g1 = p1
    try:
        g2 = df1[df1['Probeset_ID'] == p2]['Gene.Symbol'].values.tolist()[0]
    except:
        g2 = p2

    for key in models.keys():
        key = int(key)
        print('Key', key)
        plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.set_facecolor("lightcoral")

        x_r = list()
        y_r = list()
        x_b = list()
        y_b = list()
        for i in range(len(data)):
            xy, w, lab = data[i]
            x, y = xy
            if lab == 0: #blue
                x_r.append(x)
                y_r.append(y)
            else: #red
                x_b.append(x)
                y_b.append(y)

        bpr, bpb, r_p, b_p = models[key]

        for bp in bpb:
            x, y = bp
            if key == 1:
                ax.add_artist(patches.Rectangle((0.0, 0.0), x, y, facecolor = 'lightskyblue', zorder = 1))
            elif key == 2:
                ax.add_artist(patches.Rectangle((x, 0), 1000, y, facecolor = 'lightskyblue', zorder = 1))
            elif key == 3:
                ax.add_artist(patches.Rectangle((x, y), 1000, 1000, facecolor = 'lightskyblue', zorder = 1))
            else:
                ax.add_artist(patches.Rectangle((0, y ), x, 1000, facecolor = 'lightskyblue', zorder = 1))


        #for bp in bpr:
        #    x, y = bp


        #    if key == 1:
        #        ax.add_artist(patches.Rectangle((x, y), 1000, 1000, facecolor ='lightcoral', zorder = 1))
        #    elif key == 2:
        #        ax.add_artist(patches.Rectangle((0, y ), x, 1000, facecolor = 'lightcoral', zorder = 1))
        #    elif key == 3:
        #        ax.add_artist(patches.Rectangle((0.0, 0.0), x, y, facecolor = 'lightcoral', zorder = 1))
        #    else:
        #        ax.add_artist(patches.Rectangle((x, 0), 1000, y, facecolor = 'lightcoral', zorder = 1))


        plt.scatter(x_r, y_r, c = 'blue', zorder = 2)
        plt.scatter(x_b, y_b, c = 'red', zorder = 2)
        plt.xlabel(g1)
        plt.ylabel(g2)

        if pathname is not None:
            plt.savefig(pathname + g1 + '_' + g2  + '.png')

            f = open(pathname + 'gene.txt', 'a')
            f.write('{} : {}\n'.format(g1, p1))
            f.write('{} : {}\n'.format(g2, p2))
            f.close()
        else:
            plt.show()



def show_results(df, probs_df, pairs, nbcpus, pathname):
    pool = mp.Pool(nbcpus)
    vals = [(p, df) for p in pairs]

    res = pool.starmap(cr_models, vals, max(1,len(vals)//nbcpus) )

    for r in res:
        p1, p2, models, data = r
        print_model(data, models, p1, p2, probs_df, pathname)
