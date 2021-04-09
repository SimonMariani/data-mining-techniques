import argparse
import yaml
from train import train_svm, train_logistic, train_net, train_lstm
from tabulate import tabulate

def run(config):

    if config['model'] == 'LSTM':
        return train_lstm(config)
    if config['model'] == 'NN':
        return train_net(config)
    if config['model'] == 'logistic':
        return train_logistic(config)
    if config['model'] == 'svm':
        return train_svm(config)
    if config['model'] == 'all':
        run_all(config)


def run_all(config):

    models = ['logistic', 'svm', 'NN', 'LSTM']
    accuracies = [0 for i in range(len(models))]

    for seed in range(config['nmbr_of_runs']):
        for i, model in enumerate(models):
            temp_config = get_config({**config, 'model': model})
            temp_config['seed'] = seed
            accuracies[i] += run(temp_config)

    accuracies = [accuracy / config['nmbr_of_runs'] for accuracy in accuracies]

    print(tabulate([models, accuracies]))


def get_config(config):

    if config['model'] == 'LSTM':
        return {**config['LSTM'], 'model': config['model']}
    if config['model'] == 'logistic':
        return {**config['logistic'], 'model': config['model']}
    if config['model'] == 'svm':
        return {**config['svm'], 'model': config['model']}
    if config['model'] == 'NN':
        return {**config['NN'], 'model': config['model']}
    if config['model'] == 'all':
        return config


if __name__ == '__main__':
    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)

    config = get_config(config)

    run(config)





