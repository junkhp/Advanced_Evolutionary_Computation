# -*- coding: utf-8 -*-
# subset_sizeを変えて実験を行うためのスクリプト
import subprocess

subset_size_list = [2, 4, 6, 8, 11, 16, 32]
for subset_size in subset_size_list:
    cmd1 = 'python knapsack_TDGA_newcrossover.py --part_num ' + str(subset_size) + ' --mutpb 0'
    cmd1 = cmd1.split(' ')

    for i in range(30):
        print('size' + str(subset_size) + 'num: ' + str(i))
        subprocess.call(cmd1)
