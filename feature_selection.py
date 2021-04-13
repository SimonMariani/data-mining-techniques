from utils import load_object, save_object
import argparse
import yaml
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
from main import get_config, run_model
from tabulate import tabulate

def create_subsets(config):
    """
    pr: participant reported (fixed length)
    su: smartphone usage (fixed length)
    bf: basic features (fixed length)
    ma: moving averages (len(columns) * 3)
    tsfp: tsfresh participant (len(columns) * nmbr_of_features)
    tsfd: tsfresh day (nmbr_of features)
    """

    # We need the data config to extract certain properties of the data
    data_config = {**config['dataset']}

    # The number of columns that we used to extract features
    nmbr_columns = len(data_config['columns'])

    # Some values that we need to determine until where the columns go
    nmbr_ma = nmbr_columns * 3
    nmbr_tsfp = nmbr_columns * 787  # note that this only works if you use the comprehensive tsfresh pack

    # We load the data
    data, labels = load_object(data_config['save_folder'] + '/processed_data_advanced.pkl')
    total_columns = len(data[0][0])

    #print(total_columns)
    #print(len(ComprehensiveFCParameters()))
    #print(43+nmbr_ma+nmbr_tsfp)

    subset_names = ['pr', 'pr_su', 'pr_su_bf', 'pr_su_bf_ma', 'pr_su_bf_ma_tsfp']  #  , 'pr_su_bf_ma_tsfp_tsfd'
    subset_indices = [(0,4), (0,20), (0,43), (0,43+nmbr_ma), (0, 43+nmbr_ma+nmbr_tsfp)]  # , (43+nmbr_ma+nmbr_tsfp, total_columns)

    for name, indices in zip(subset_names, subset_indices):
        subset = []
        for i in range(len(data)):
            subset.append(data[i][:, indices[0] : indices[1]])

        save_object((subset, labels), data_config['save_folder'] + '/subdata_' + name + '.pkl')

    run_models(config, subset_names, subset_indices)

def run_models(config, subset_names, subset_indices):

    models = config['models']
    folder = config['dataset']['save_folder']

    for name, indices in zip(subset_names, subset_indices):

        total_accuracy = [0 for i in range(len(models))]
        total_mse = [0 for i in range(len(models))]

        for seed in range(config['nmbr_of_runs']):
            for i, model in enumerate(models):
                temp_config = get_config({**config, 'model': model})
                temp_config['seed'] = seed
                temp_config['data_path'] = folder + '/subdata_' + name + '.pkl'
                temp_config['print'] = config['print']

                if model in ['NN', 'LSTM']:
                    temp_config['in_dim'] = indices[1]

                mse, accuracy = run_model(temp_config)
                total_mse[i] += mse
                total_accuracy[i] += accuracy

        # Calculate the average over all runs
        accuracies = [accuracy / config['nmbr_of_runs'] for accuracy in total_accuracy]
        mses = [mse / config['nmbr_of_runs'] for mse in total_mse]

        # Print the results in a table
        table = [['accuracy'] + accuracies, ['mean_sqrd_errors'] + mses]
        print('\n')
        print("subset: " + name)
        print(tabulate(table, headers=['metrics'] + models, tablefmt="fancy_grid"))  # plain


if __name__ == "__main__":

    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)

    create_subsets(config)

    #import pandas as pd
    #temp = pd.read_csv('data_processed/processed_data_pandas.csv')
    #print(temp.isnull().sum().sum())








