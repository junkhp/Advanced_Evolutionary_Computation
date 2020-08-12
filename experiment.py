# -*- coding: utf-8 -*-

# 突然変異率を変化させて実験を行う時のスクリプト

import subprocess
import csv
import os
import numpy as np


def csv_mt(mutpb):
    sga = 'sga_new_' + str(mutpb) + '.csv'
    tdga = 'tdga_' + str(mutpb) + '.csv'
    tdga_new = 'tdga_new_' + str(mutpb) + '.csv'

    sga = 'sga_result.csv'
    tdga = 'tdga_result.csv'
    tdga_new = 'tdga_new_result.csv'

    all = []

    f_sga = open(sga)
    f_tdga = open(tdga)
    f_tdga_new = open(tdga_new)

    r_sga = csv.reader(f_sga)
    r_tdga = csv.reader(f_tdga)
    r_tdga_new = csv.reader(f_tdga_new)
    for s, t, tn in zip(r_sga, r_tdga, r_tdga_new):
        all.append(s+t+tn)
    f_tdga_new.close()
    f_tdga.close()
    f_sga.close()

    all_np = np.array(all, dtype=int)
    sums = np.sum(all_np, axis=0)
    aves = sums/all_np.shape[0]

    all.append(sums.tolist())
    all.append(aves.tolist())

    print(all)
    out_file_name = 'reslut_all_' + str(mutpb) + '.csv'
    with open(out_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(all)


mutpb = 0.01

cmd0 = 'python knapsack_TDGA.py --mutpb ' + str(mutpb)
cmd1 = 'python knapsack_TDGA_newcrossover.py --mutpb ' + str(mutpb)
cmd2 = 'python knapsack_sga.py --mutpb ' + str(mutpb)

cmd0 = cmd0.split(' ')
cmd1 = cmd1.split(' ')
cmd2 = cmd2.split(' ')

for i in range(30):
    print('num: ' + str(i))
    subprocess.call(cmd0)
    subprocess.call(cmd1)
    subprocess.call(cmd2)

csv_mt(mutpb)
