import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

subset_names = ['pr', 'pr_su', 'pr_su_bf', 'pr_su_bf_ma', 'pr_su_bf_ma_tsfp']  #  'pr_su_bf_ma_tsfp_tsfd'

print('m')
final_frame = pd.read_csv('results/results_m.csv', index_col=0)
print(tabulate(final_frame, headers=final_frame.columns, tablefmt="fancy_grid"))

for name in subset_names:
    temp_frame = pd.read_csv('results/results_' + name + '.csv', index_col=0)
    final_frame = pd.concat([final_frame, temp_frame])
    print('\n')
    print(name)
    #print(temp_frame)

    #table = [['mse'] + mses, ['root_mse'] + rmses, ['r2_score'] + r2s, ['adj_r2_score'] + adj_r2s,
    #         ['accuracy'] + accuracies,
    #         ['bal_accuracy'] + balanced_accuracies]

    #print('dataset: ' + name)
    print(tabulate(temp_frame, headers=temp_frame.columns, tablefmt="fancy_grid"))  # plain
