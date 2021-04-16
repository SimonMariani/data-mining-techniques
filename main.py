import argparse
import yaml
import time
from train import train_svm, train_linear, train_net, train_lstm, train_bilstm, train_random_forrest, train_baseline
from tabulate import tabulate
from utils import load_object, get_folds
import numpy as np
import warnings
import pandas as pd

def get_config(config):

    if config['model'] == 'LSTM':
        return {**config['LSTM'], 'model': config['model']}
    if config['model'] == 'BiLSTM':
        return {**config['BiLSTM'], 'model': config['model']}
    if config['model'] == 'linear':
        return {**config['linear'], 'model': config['model']}
    if config['model'] == 'svm':
        return {**config['svm'], 'model': config['model']}
    if config['model'] == 'NN':
        return {**config['NN'], 'model': config['model']}
    if config['model'] == 'random_forest':
        return {**config['random_forest'], 'model': config['model']}
    if config['model'] == 'baseline':
        return {**config['baseline'], 'model': config['model']}

def run_model(config, fold, fold_base=None):

    if config['model'] == 'LSTM':
        return train_lstm(config, fold)
    if config['model'] == 'BiLSTM':
        return train_bilstm(config, fold)
    if config['model'] == 'NN':
        return train_net(config, fold)
    if config['model'] == 'linear':
        return train_linear(config, fold)
    if config['model'] == 'svm':
        return train_svm(config, fold)
    if config['model'] == 'random_forest':
        return train_random_forrest(config, fold)
    if config['model'] == 'baseline':
        return train_baseline(config, fold, fold_base)


def run(config):

    models = config['models']

    total_mse = [0 for i in range(len(models))]
    total_rmse = [0 for i in range(len(models))]
    total_r2 = [0 for i in range(len(models))]
    total_adj_r2 = [0 for i in range(len(models))]

    total_accuracy = [0 for i in range(len(models))]
    total_balanced_accuracy = [0 for i in range(len(models))]

    print("training and validating")

    for i, model in enumerate(models):

        print(model)
        temp_config = get_config({**config, 'model': model})
        temp_config['print'] = config['print']
        all_folds, all_folds_baseline = get_folds(temp_config)

        for index, (fold, fold_base) in enumerate(zip(all_folds, all_folds_baseline)):

            if model == 'baseline':
                mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = run_model(temp_config, fold, fold_base)
            else:
                mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = run_model(temp_config, fold)

            total_mse[i] += mse
            total_rmse[i] += rmse
            total_r2[i] += r2
            total_adj_r2[i] += adj_r2

            total_accuracy[i] += accuracy
            total_balanced_accuracy[i] += balanced_accuracy

    # Calculate the average over all runs
    mses = [mse / len(all_folds) for mse in total_mse]
    rmses = [rmse / len(all_folds) for rmse in total_rmse]
    r2s = [r2 / len(all_folds) for r2 in total_r2]
    adj_r2s = [adj_r2 / len(all_folds) for adj_r2 in total_adj_r2]

    accuracies = [accuracy / len(all_folds) for accuracy in total_accuracy]
    balanced_accuracies = [balanced_accuracy / len(all_folds) for balanced_accuracy in total_balanced_accuracy]

    # Print the results in a table
    table = [['mse'] + mses, ['root_mse'] + rmses, ['r2_score'] + r2s, ['adj_r2_score'] + adj_r2s,
             ['accuracy'] + accuracies, ['bal_accuracy'] + balanced_accuracies]

    print(tabulate(table, headers=['metrics'] + models, tablefmt="fancy_grid"))  # plain

    if config['test']:
        run_test(config)

def run_test(config):

    models = config['models']

    total_mse = [0 for i in range(len(models))]
    total_rmse = [0 for i in range(len(models))]
    total_r2 = [0 for i in range(len(models))]
    total_adj_r2 = [0 for i in range(len(models))]

    total_accuracy = [0 for i in range(len(models))]
    total_balanced_accuracy = [0 for i in range(len(models))]

    print("Training and testing")

    for i, model in enumerate(models):

        print(model)
        temp_config = get_config({**config, 'model': model})
        temp_config['print'] = config['print']

        data_train, labels_train = load_object(temp_config['data_path'])
        data_test, labels_test = load_object(temp_config['test_path'])

        baseline_targets_train = np.array(load_object('./data_processed/baseline_targets_train.pkl'), dtype=object)
        baseline_targets_test = np.array(load_object('./data_processed/baseline_targets_test.pkl'), dtype=object)

        fold = [data_train, labels_train, data_test, labels_test]
        fold_base = [baseline_targets_train, baseline_targets_test]

        if model == 'baseline':
            mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = run_model(temp_config, fold, fold_base)
        else:
            mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = run_model(temp_config, fold)

        total_mse[i] += mse
        total_rmse[i] += rmse
        total_r2[i] += r2
        total_adj_r2[i] += adj_r2

        total_accuracy[i] += accuracy
        total_balanced_accuracy[i] += balanced_accuracy

    # Print the results in a table
    table = [['mse'] + total_mse, ['root_mse'] + total_rmse, ['r2_score'] + total_r2, ['adj_r2_score'] + total_adj_r2,
             ['accuracy'] + total_accuracy, ['bal_accuracy'] + total_balanced_accuracy]

    pd.DataFrame(table, columns=models).to_csv("final_results.csv")
    print(tabulate(table, headers=['metrics'] + models, tablefmt="fancy_grid"))  # plain


if __name__ == '__main__':

    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)

    warnings.filterwarnings(action='ignore', category=UserWarning)

    print("Starting timer")
    start_time = time.time()
    if config['test_only']:
        run_test(config)
    else:
        run(config)
    print("--- %s seconds ---" % (time.time() - start_time))





