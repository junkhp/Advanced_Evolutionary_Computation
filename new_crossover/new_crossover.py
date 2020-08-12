# -*- coding: utf-8 -*-
'''
新しい演算子
講義で紹介されていた一様交叉の拡張を実装
'''
from deap import base
import numpy as np
import random

toolbox = base.Toolbox()


class NewCrossOver:
    def __init__(self, part_num=4, new_crossover_rate=0.5):
        self.part_num = part_num    # 一つの部分集合に何個の個体を含むか
        self.new_crossover_rate = new_crossover_rate        # 交叉を発生させる確率

    def new_crossover(self, individuals):
        # 1部分集合あたりpart_num個の個体を含む部分集合に分割
        subsets = self.make_subsets(individuals)

        new_individuals = []

        for subset in subsets:
            # 各列(各遺伝子座)の0と1の割合を計算
            proportions = self.calc_01_proportion(np.array(subset))

            # 各個体に処理
            for individual in subset:
                # 各遺伝子座の値を変更するための個体をコピー
                individual_copy = toolbox.clone(individual)
                # 各遺伝子座についてproportionに基づいて固定&固定されなかったら確率0.5で変異
                for p_index, p in enumerate(proportions):
                    p = max(p, 1 - p)
                    if random.random() > p:
                        if random.random() < self.new_crossover_rate:
                            del individual_copy.fitness.values
                            if individual[p_index] == 0:
                                individual_copy[p_index] = 1
                            else:
                                individual_copy[p_index] = 0
                new_individuals.append(individual_copy)

        return new_individuals

    # 1部分集合あたりpart_num個の個体を含む部分集合に分割
    def make_subsets(self, individuals):
        subsets = []
        for i, ind in enumerate(individuals):
            if i == 0:
                subset = []
                subset.append(ind)
            else:
                if i % self.part_num == 0:
                    subsets.append(subset)
                    subset = []
                    subset.append(ind)
                else:
                    subset.append(ind)
        subsets.append(subset)

        return subsets

    # 各列(各遺伝子座)の0と1の割合を計算
    def calc_01_proportion(self, subset):
        proportions = np.count_nonzero(subset == 0, axis=0)
        proportions = proportions/subset.shape[0]
        return proportions
