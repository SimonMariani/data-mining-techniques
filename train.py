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
import random


def train_lstm(config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    #device = torch.device(config.device)

    # Initialize the dataset and data loader
    dataset_train = MOOD_loader(root=config['data_path'], train=True, shuffle=True)
    dataset_test = MOOD_loader(root=config['data_path'], train=False)
    train_loader = DataLoader(dataset_train, config['batch_size'])
    test_loader = DataLoader(dataset_test, config['batch_size'])

    # Initialize the model that we are going to use
    model = Basic_LSTM(config['batch_size'], config['hidden'], config['layers'], config['in_dim'],
                           config['out_dim'], device=config['device'])

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        for inputs, targets in train_loader:
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

        accuracy = eval_LSTM(model, test_loader)

        if config['print']:
            print(f'epoch: {epoch}')
            print(f'loss: {loss}')
            print(f'accuracy: {accuracy}')
            print('\n')

    return accuracy


def eval_LSTM(model, dataloader):

    correct = 0
    total = 0
    for inputs, targets in dataloader:

        # send intputs to the device
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        # Forward pass
        log_probs = model(inputs.float())

        predictions = torch.argmax(log_probs, dim=2)
        targets = torch.round(targets)

        correct += (predictions == targets).sum().item()
        total += log_probs.size(0) * log_probs.size(1)

    accuracy = correct / total

    return accuracy


def train_net(config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    # device = torch.device(config.device)

    # Initialize the dataset and data loader
    dataset_train = MOOD_loader(root=config['data_path'], train=True, shuffle=True)
    dataset_test = MOOD_loader(root=config['data_path'], train=False)
    data_train = DataLoader(dataset_train, config['batch_size'])  # batch size=9 means exactly 3 batches per epoch
    data_test = DataLoader(dataset_test, config['batch_size'])

    # Initialize the model that we are going to use
    model = Basic_Net(config['batch_size'], config['hidden'], config['layers'], config['in_dim'],
                      config['out_dim'], device=config['device'])

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config['epochs']):
        for inputs, targets in data_train:
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

        accuracy = eval_NN(model, data_test)

        if config['print']:
            print(f'epoch: {epoch}')
            print(f'loss: {loss}')
            print(f'accuracy: {accuracy}')
            print('\n')

        return accuracy


def eval_NN(model, dataloader):

    correct = 0
    total = 0
    for inputs, targets in dataloader:
        model.zero_grad()

        # send intputs to the device
        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)

        # Forward pass
        log_probs = model(inputs.float())

        predictions = torch.argmax(log_probs, dim=1)
        targets = torch.round(targets)

        correct += (predictions == targets).sum().item()
        total += log_probs.size(0)

    accuracy = correct / total

    return accuracy


def train_logistic(config):
    random.seed(config['seed'])
    data, labels = load_object(config['data_path'])
    data_train, labels_train, data_test, labels_test = split_data(data, labels, split=0.8)

    model = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    correct = (predictions == labels_test).sum().item()
    accuracy = correct / len(labels_test)

    if config['print']:
        print(f'test accuracy: {accuracy}')

    return accuracy


def train_svm(config):
    random.seed(config['seed'])
    data, labels = load_object(config['data_path'])
    data_train, labels_train, data_test, labels_test = split_data(data, labels, split=0.8)

    model = svm.LinearSVC()

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    correct = (predictions == labels_test).sum().item()
    accuracy = correct / len(labels_test)

    if config['print']:
        print(f'test accuracy: {accuracy}')

    return accuracy


def split_data(data, targets, split=0.8, shuffle=True):

    if shuffle:
        temp = list(zip(data, targets))
        random.shuffle(temp)
        data, labels = zip(*temp)

    split = int(split * len(data))
    targets = [round(label, 0) for label in targets]
    data_train, labels_train = data[:split], targets[:split]
    data_test, labels_test = data[split:], targets[split:]

    return data_train, labels_train, data_test, labels_test





