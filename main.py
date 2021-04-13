import argparse
import yaml
from train import train_svm, train_linear, train_net, train_lstm, train_bilstm, train_random_forrest, train_baseline
from tabulate import tabulate

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

def run_model(config):

    if config['model'] == 'LSTM':
        return train_lstm(config)
    if config['model'] == 'BiLSTM':
        return train_bilstm(config)
    if config['model'] == 'NN':
        return train_net(config)
    if config['model'] == 'linear':
        return train_linear(config)
    if config['model'] == 'svm':
        return train_svm(config)
    if config['model'] == 'random_forest':
        return train_random_forrest(config)
    if config['model'] == 'baseline':
        return train_baseline(config)

def run(config):

    models = config['models']

    total_accuracy = [0 for i in range(len(models))]
    total_mse = [0 for i in range(len(models))]

    for seed in range(config['nmbr_of_runs']):
        for i, model in enumerate(models):
            temp_config = get_config({**config, 'model': model})
            temp_config['seed'] = seed
            temp_config['print'] = config['print']

            mse, accuracy = run_model(temp_config)
            total_mse[i] += mse
            total_accuracy[i] += accuracy

    # Calculate the averag over all runs
    accuracies = [accuracy / config['nmbr_of_runs'] for accuracy in total_accuracy]
    mses = [mse / config['nmbr_of_runs'] for mse in total_mse]

    # Print the results in a table
    table = [['accuracy'] + accuracies, ['mean_sqrd_errors'] + mses]
    print('\n')
    print(tabulate(table, headers=['metrics'] + models, tablefmt="fancy_grid"))  # plain


if __name__ == '__main__':
    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)

    #config = get_config(config)

    run(config)





