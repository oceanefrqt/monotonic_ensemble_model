from Module import monotonic_regression_uncertainty as mru
from Module import selection_algorithm as sa
from Module import stages as stg

import pandas as pd
import sys
import os





def main():
    args = sys.argv[1:]
    dataset = args[0]
    auc_file =  args[1]


    df = pd.read_csv(dataset, index_col=0)
    df.reset_index(drop=True, inplace=True)

    nbcpus = 128
    max_k = 12
    strat = [sa.NB]
    funct = mru.predict_severe

    k_opt = stg.stage_1(df, max_k, nbcpus, strat, funct)
    print('k opt', k_opt)

    acc, auc, CI = stg.stage_2(df, k_opt, auc_file, nbcpus, funct, strat)
    print('For previous k_opt, ACC = {}, AUC = {}, CI = {}'.format(acc, auc, CI))

    acc, auc, CI = stg.stage_2(df, 10, auc_file, nbcpus, funct, strat)
    print('For k_opt=10, ACC = {}, AUC = {}, CI = {}'.format(acc, auc, CI))

    pairs, mve = stg.stage_3(df, k_opt, nbcpus, funct, strat)
    print('Pairs', pairs)
    print('MVE', mve)

    

    pairs, mve = stg.stage_3(df, 10, nbcpus, funct, strat)
    print('For k_opt = 10,')
    print('Pairs', pairs)
    print('MVE', mve)

if __name__ == "__main__":
    main()
