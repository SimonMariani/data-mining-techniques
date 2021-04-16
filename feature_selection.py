from utils import load_object, save_object, get_folds
import argparse
import yaml
import time
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
from main import get_config, run_model, run_test
from tabulate import tabulate
import warnings


def create_subsets(config):
    """
    pr: participant reported (fixed length)
    su: smartphone usage (fixed length)
    bf: basic features (fixed length)
    ma: moving averages (len(columns) * 3)
    tsfp: tsfresh participant (len(columns) * nmbr_of_features)
    tsfd: tsfresh day (nmbr_of features)
    """

    """# We need the data config to extract certain properties of the data
    data_config = {**config['dataset']}

    # The number of columns that we used to extract features
    nmbr_columns = len(data_config['columns'])

    # Some values that we need to determine until where the columns go
    nmbr_ma = nmbr_columns * 3
    nmbr_tsfp = nmbr_columns * 787  # note that this only works if you use the comprehensive tsfresh pack

    # We load the data
    data, labels = load_object(data_config['save_folder'] + '/processed_data_advanced_train.pkl')
    data_test, labels_test = load_object(data_config['save_folder'] + '/processed_data_advanced_test.pkl')
    total_columns = len(data[0][0])"""


    """for name, indices in zip(subset_names, subset_indices):
        subset = []

        for i in range(len(data)):
            subset.append(data[i][:, indices[0] : indices[1]])

        subset_test = []
        for j in range(len(data_test)):
            subset_test.append(data_test[j][:, indices[0] : indices[1]])

        save_object((subset, labels), data_config['save_folder'] + '/subdata_' + name + '.pkl')
        save_object((subset_test, labels_test), data_config['save_folder'] + '/subdata_' + name + '_test.pkl')"""

    data_config = {**config['dataset']}

    # The number of columns that we used to extract features
    nmbr_columns = len(data_config['columns'])

    # Some values that we need to determine until where the columns go
    nmbr_ma = nmbr_columns * 3
    nmbr_tsfp = nmbr_columns * 787  # note that this only works if you use the comprehensive tsfresh pack

    subset_names = ['m', 'pr', 'pr_su', 'pr_su_bf', 'pr_su_bf_ma']  # 'pr_su_bf_ma_tsfp' , 'pr_su_bf_ma_tsfp_tsfd'
    subset_indices = [(0, 1), (0, 4), (0, 20), (0, 43),
                      (0, 43 + nmbr_ma)]  # (0, 43 + nmbr_ma + nmbr_tsfp), (0, total_columns)

    run_models(config, subset_names, subset_indices)


def run_models(config, subset_names, subset_indices):

    models = config['models']
    folder = config['dataset']['save_folder']

    for name, indices in zip(subset_names, subset_indices):

        total_mse = [0 for i in range(len(models))]
        total_rmse = [0 for i in range(len(models))]
        total_r2 = [0 for i in range(len(models))]
        total_adj_r2 = [0 for i in range(len(models))]

        total_accuracy = [0 for i in range(len(models))]
        total_balanced_accuracy = [0 for i in range(len(models))]

        for i, model in enumerate(models):

            temp_config = get_config({**config, 'model': model})
            temp_config['data_path'] = folder + '/subdata_' + name + '.pkl'
            temp_config['print'] = config['print']

            if model in ['NN', 'LSTM', 'BiLSTM']:
                temp_config['in_dim'] = indices[1]

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
        table = [['mse'] + mses, ['root_mse'] + rmses, ['r2_score'] + r2s, ['adj_r2_score'] + adj_r2s, ['accuracy'] + accuracies,
                 ['bal_accuracy'] + balanced_accuracies]

        print('dataset: ' + name)
        print(tabulate(table, headers=['metrics'] + models, tablefmt="fancy_grid"))  # plain


if __name__ == "__main__":


    warnings.filterwarnings(action='ignore', category=UserWarning)

    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)

    print("Starting timer")
    start_time = time.time()
    create_subsets(config)
    print("--- %s seconds ---" % (time.time() - start_time))








