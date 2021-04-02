from dataset import MOOD_loader
from dataset import load_object
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from models import Basic_LSTM, Basic_Net
import argparse
import yaml
from sklearn import linear_model, svm, metrics
from dataset import load_object


def train_pytorch(config):
    if config['model'] == 'LSTM':
        train_lstm(config)
    elif config['model'] == 'NN':
        train_net(config)


def train_net(config):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    # device = torch.device(config.device)

    # Initialize the dataset and data loader
    dataset = MOOD_loader(root=config['data_path'])
    data_loader = DataLoader(dataset, config['batch_size'])  # batch size=9 means exactly 3 batches per epoch

    # Initialize the model that we are going to use
    model = Basic_Net(config['batch_size'], config['hidden'], config['layers'], config['in_dim'],
                      config['out_dim'], device=config['device'])
    print(model)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    count = 0
    for epoch in range(config['epochs']):
        for inputs, targets in data_loader:
            model.zero_grad()

            # send intputs to the device
            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)

            # Forward pass
            log_probs = model(inputs.float())
            loss = criterion(log_probs, targets.long())
            loss.backward()

            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config['max_norm'])
            count += 1

        predictions = torch.argmax(log_probs, dim=1)
        targets = torch.round(targets)

        correct = (predictions == targets).sum().item()
        accuracy = correct / (log_probs.size(0) * log_probs.size(1))

        print(f'epoch: {epoch}')
        print(f'loss: {loss}')
        print(f'accuracy: {accuracy}')
        print('\n')


def train_lstm(config):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    #device = torch.device(config.device)

    # Initialize the dataset and data loader
    dataset = MOOD_loader(root=config['data_path'])
    data_loader = DataLoader(dataset, config['batch_size'])  # batch size=9 means exactly 3 batches per epoch

    # Initialize the model that we are going to use
    model = Basic_LSTM(config['batch_size'], config['hidden'], config['layers'], config['in_dim'],
                           config['out_dim'], device=config['device'])

    print(model)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    count = 0
    for epoch in range(config['epochs']):
        for inputs, targets in data_loader:
            model.zero_grad()

            # send intputs to the device
            inputs = torch.stack(inputs)
            targets = torch.stack(targets)

            # Forward pass
            log_probs = model(inputs.float())
            targets = targets.permute(1, 0)
            log_probs = log_probs.permute(1, 2, 0)

            loss = criterion(log_probs, targets.long())
            loss.backward()

            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config['max_norm'])
            count += 1

        targets = targets.permute(1, 0)
        log_probs = log_probs.permute(2, 0, 1)

        predictions = torch.argmax(log_probs, dim=2)
        targets = torch.round(targets)

        correct = (predictions == targets).sum().item()
        accuracy = correct / (log_probs.size(0) * log_probs.size(1))

        print(f'epoch: {epoch}')
        print(f'loss: {loss}')
        print(f'accuracy: {accuracy}')
        print('\n')


def train_features(config, split=0.8):
    data, labels = load_object(config['data_path'])
    split = int(0.8 * len(data))
    labels = [round(label, 0) for label in labels]
    data_train, labels_train = data[:split], labels[:split]
    data_test, labels_test = data[split:], labels[split:]

    if config['model'] == 'logistic':
        model = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)
    elif config['model'] == 'svm':
        model = svm.LinearSVC()

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    correct = (predictions == labels_test).sum().item()
    accuracy = correct / len(labels_test)

    print(f'test accuracy: {accuracy}')


def get_config(config):
    if config['model'] == 'LSTM':
        return {**config['LSTM'], 'model': config['model']}
    if config['model'] == 'logistic':
        return {**config['logistic'], 'model': config['model']}
    if config['model'] == 'svm':
        return {**config['svm'], 'model': config['model']}
    if config['model'] == 'NN':
        return {**config['NN'], 'model': config['model']}


if __name__ == '__main__':
    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)

    config = get_config(config)

    if config['model'] in ['LSTM', 'NN']:
        train_pytorch(config)
    elif config['model'] in ['logistic', 'svm']:
        train_features(config, 0.8)






