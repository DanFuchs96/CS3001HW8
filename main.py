#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #8: Scikit Surprise
"""
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, evaluate, print_perf
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (15, 5)


def main():
    data_file = 'restaurant_ratings.txt'
    data_s3 = build_dataset(data_file)
    data_s3.split(n_folds=3)
    task_5(data_s3)
    task_6(data_s3)
    task_7(data_s3)
    task_8(data_s3)
    task_9(data_s3)
    task_14(data_s3)
    task_15(data_s3)


def build_dataset(filename):
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    return Dataset.load_from_file(filename, reader=reader)


def mean(value_list):
    total = 0
    for value in value_list:
        total += value
    return total / len(value_list)


def task_5(data_set):
    print(']> SVD Performance:')
    performance = evaluate(SVD(), data_set, measures=['RMSE', 'MAE'], verbose=0)
    print_perf(performance)
    print('')
    return performance


def task_6(data_set):
    print(']> PMF Performance:')
    performance = evaluate(SVD(biased=False), data_set, measures=['RMSE', 'MAE'], verbose=0)
    print_perf(performance)
    print('')
    return performance


def task_7(data_set):
    print(']> NMF Performance:')
    performance = evaluate(NMF(), data_set, measures=['RMSE', 'MAE'], verbose=0)
    print_perf(performance)
    print('')
    return performance


def task_8(data_set):
    print(']> User-based CF Performance:')
    performance = evaluate(KNNBasic(sim_options={'user_based': True}, verbose=False),
                           data_set, measures=['RMSE', 'MAE'], verbose=0)
    print_perf(performance)
    print('')
    return performance


def task_9(data_set):
    print(']> Item-based CF Performance:')
    performance = evaluate(KNNBasic(sim_options={'user_based': False}, verbose=False),
                           data_set, measures=['RMSE', 'MAE'], verbose=0)
    print_perf(performance)
    print('')
    return performance


def task_14(data_set):
    ib_perfs = [evaluate(KNNBasic(sim_options={'name': 'MSD', 'user_based': False}, verbose=False),
                         data_set, measures=['RMSE', 'MAE'], verbose=0),
                evaluate(KNNBasic(sim_options={'name': 'cosine', 'user_based': False}, verbose=False),
                         data_set, measures=['RMSE', 'MAE'], verbose=0),
                evaluate(KNNBasic(sim_options={'name': 'pearson', 'user_based': False}, verbose=False),
                         data_set, measures=['RMSE', 'MAE'], verbose=0)]
    ub_perfs = [evaluate(KNNBasic(sim_options={'name': 'MSD', 'user_based': True}, verbose=False),
                         data_set, measures=['RMSE', 'MAE'], verbose=0),
                evaluate(KNNBasic(sim_options={'name': 'cosine', 'user_based': True}, verbose=False),
                         data_set, measures=['RMSE', 'MAE'], verbose=0),
                evaluate(KNNBasic(sim_options={'name': 'pearson', 'user_based': True}, verbose=False),
                         data_set, measures=['RMSE', 'MAE'], verbose=0)]
    ib_vals = []
    ub_vals = []
    for i in range(3):
        ib_vals.append(mean(ib_perfs[i]['RMSE']))
        ub_vals.append(mean(ub_perfs[i]['RMSE']))
    labels = ['msd', 'cosine', 'pearson']
    s1 = np.arange(3)
    s2 = [x + 0.25 for x in s1]
    plt.bar(s1, ib_vals, color='red', width=0.25, edgecolor='white', label='Item-based RMSE')
    plt.bar(s2, ub_vals, color='blue', width=0.25, edgecolor='white', label='User-based RMSE')
    plt.xlabel('Similarity Metric', fontweight='bold')
    plt.xticks([r + 0.125 for r in range(3)], labels)
    plt.ylim(bottom=0.7)
    plt.legend()
    plt.show()
    plt.clf()
    ib_vals = []
    ub_vals = []
    for i in range(3):
        ib_vals.append(mean(ib_perfs[i]['MAE']))
        ub_vals.append(mean(ub_perfs[i]['MAE']))
    plt.bar(s1, ib_vals, color='red', width=0.25, edgecolor='white', label='Item-based MAE')
    plt.bar(s2, ub_vals, color='blue', width=0.25, edgecolor='white', label='User-based MAE')
    plt.xlabel('Similarity Metric', fontweight='bold')
    plt.xticks([r + 0.125 for r in range(3)], labels)
    plt.ylim(bottom=0.7)
    plt.legend()
    plt.show()
    return


def task_15(data_set):
    ib_perf = lambda k: evaluate(KNNBasic(k=k, sim_options={'name': 'MSD', 'user_based': False}, verbose=False),
                                 data_set, measures=['RMSE', 'MAE'], verbose=0)
    ub_perf = lambda k: evaluate(KNNBasic(k=k, sim_options={'name': 'MSD', 'user_based': True}, verbose=False),
                                 data_set, measures=['RMSE', 'MAE'], verbose=0)
    k_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    k_values = [k + 10 for k in k_values]
    rmse_vals = []
    mae_vals = []
    for k in k_values:
        rmse_vals.append(mean(ib_perf(k)['RMSE']))
        mae_vals.append(mean(ub_perf(k)['RMSE']))
    labels = [str(x) for x in k_values]
    s1 = np.arange(len(k_values))
    s2 = [x + 0.25 for x in s1]
    plt.bar(s1, rmse_vals, color='red', width=0.25, edgecolor='white', label='Item-based RMSE')
    plt.bar(s2, mae_vals, color='blue', width=0.25, edgecolor='white', label='User-based RMSE')
    plt.xlabel('Similarity Metric', fontweight='bold')
    plt.xticks([r + 0.125 for r in range(len(k_values))], labels)
    plt.ylim(bottom=0.975, top=1.01)
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    main()
