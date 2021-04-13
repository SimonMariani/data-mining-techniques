from dataset import MOOD_loader
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import Basic_LSTM, Basic_Net
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from dataset import load_object
import random


def train_lstm(config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    device = torch.device(config['device'])

    # Initialize the dataset and data loader
    dataset_train = MOOD_loader(root=config['data_path'], train=True, shuffle=True, temporal=True)
    dataset_test = MOOD_loader(root=config['data_path'], train=False, temporal=True)
    train_loader = DataLoader(dataset_train, config['batch_size'])
    test_loader = DataLoader(dataset_test, config['batch_size'])

    # Initialize the model that we are going to use
    model = Basic_LSTM(lstm_num_hidden=config['hidden'], lstm_num_layers=config['layers'], input_dim=config['in_dim'],
                        output_dim=config['out_dim'])
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    for epoch in range(config['epochs']):
        for inputs, targets in train_loader:
            model.zero_grad()

            inputs, targets = inputs.float().to(device), targets.float().to(device)

            # Forward pass
            log_probs = model(inputs)  # shape -> (bs, seq_len, out_dim)
            log_probs = log_probs.view(-1, log_probs.shape[1]*log_probs.shape[2])  # shape -> (bs, seq_len*out_dim)

            loss = criterion(log_probs, targets)
            loss.backward()

            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config['max_norm'])

        mse, accuracy = eval_LSTM(model, test_loader, device)

        if epoch % config['print_every'] == 0:
            print(f'epoch: {epoch}')
            print(f'loss: {loss}')
            print(f'test mse: {mse}')
            print(f'accuracy: {accuracy}')
            print('\n')

    print(f'loss: {loss}')
    print(f'test mse: {mse}')
    print(f'accuracy: {accuracy}')

    return mse, accuracy


def eval_LSTM(model, dataloader, device):

    all_targets = []
    all_predictions = []
    for inputs, targets in dataloader:
        model.zero_grad()

        inputs, targets = inputs.float().to(device), targets.float().to(device)

        # Forward pass
        log_probs = model(inputs)
        log_probs = log_probs.view(-1, log_probs.shape[1] * log_probs.shape[2])

        all_targets.append(targets.flatten())
        all_predictions.append(log_probs.flatten())

    all_targets, all_predictions = torch.cat(all_targets, dim=0), torch.cat(all_predictions, dim=0)
    mse, accuracy = get_metrics(all_targets.cpu().detach().numpy(), all_predictions.cpu().detach().numpy())

    return mse, accuracy


def train_net(config):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    device = torch.device(config['device'])

    # Initialize the dataset and data loader
    dataset_train = MOOD_loader(root=config['data_path'], train=True, shuffle=True)
    dataset_test = MOOD_loader(root=config['data_path'], train=False)
    data_train = DataLoader(dataset_train, config['batch_size'])  # batch size=9 means exactly 3 batches per epoch
    data_test = DataLoader(dataset_test, config['batch_size'])

    # Initialize the model that we are going to use
    model = Basic_Net(config['in_dim'], config['out_dim'])
    model = model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    for epoch in range(config['epochs']):
        for inputs, targets in data_train:
            model.zero_grad()

            # send intputs to the device
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)


            # Forward pass
            log_probs = model(inputs)

            loss = criterion(log_probs.flatten(), targets)

            loss.backward()

            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config['max_norm'])

        mse, accuracy = eval_NN(model, data_test, device)

        if epoch % config['print_every'] == 0:
            print(f'epoch: {epoch}')
            print(f'loss: {loss}')
            print(f'test mse: {mse}')
            print(f'accuracy: {accuracy}')
            print('\n')

    print(f'loss: {loss}')
    print(f'test mse: {mse}')
    print(f'accuracy: {accuracy}')

    return mse, accuracy


def eval_NN(model, dataloader, device):

    all_targets = []
    all_predictions = []
    for inputs, targets in dataloader:
        model.zero_grad()

        # send intputs to the device
        inputs = torch.as_tensor(inputs).to(device)
        targets = torch.as_tensor(targets).float().to(device)

        # Forward pass
        log_probs = model(inputs.float())

        all_targets.append(targets)
        all_predictions.append(log_probs.flatten())


    all_targets, all_predictions = torch.cat(all_targets, dim=0), torch.cat(all_predictions, dim=0)
    mse, accuracy = get_metrics(all_targets.cpu().detach().numpy(), all_predictions.cpu().detach().numpy())

    return mse, accuracy


def train_linear(config):
    random.seed(config['seed'])
    data, labels = load_object(config['data_path'])
    data_train, labels_train, data_test, labels_test = get_data(data, labels, split=0.8)

    model = linear_model.LinearRegression()

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    mse, accuracy = get_metrics(labels_test, predictions)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, accuracy


def train_svm(config):
    random.seed(config['seed'])
    data, labels = load_object(config['data_path'])
    data_train, labels_train, data_test, labels_test = get_data(data, labels, split=0.8)

    model = SVR(max_iter=10000)

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    mse, accuracy = get_metrics(labels_test, predictions)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, accuracy


def train_random_forrest(config):
    random.seed(config['seed'])
    data, labels = load_object(config['data_path'])
    data_train, labels_train, data_test, labels_test = get_data(data, labels, split=0.8)

    model = RandomForestRegressor()

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    mse, accuracy = get_metrics(labels_test, predictions)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, accuracy


def train_baseline(config):
    random.seed(config['seed'])
    data, labels = load_object(config['data_path'])
    data_train, labels_train, data_test, labels_test = get_data(data, labels, split=0.8)

    # We need to set the seed again to make sure that the behavior is the same
    random.seed(config['seed'])

    # We need to take the baseline targets and obtain the same split as the test data
    baseline_targets = load_object(config['targets_path'])
    random.shuffle(baseline_targets)
    split = int(0.8 * len(baseline_targets))
    baseline_targets_test = baseline_targets[split:]


    # Now we convert it to a full set of targets
    predictions = np.concatenate(baseline_targets_test, axis=0)

    mse, accuracy = get_metrics(labels_test, predictions)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, accuracy


def get_metrics(y_real, y_pred):

    mse = mean_squared_error(y_real, y_pred)
    accuracy = accuracy_score(np.around(y_real), np.around(y_pred))

    return mse, accuracy


def get_data(data, targets, split=0.8, shuffle=True):

    if shuffle:
        temp = list(zip(data, targets))
        random.shuffle(temp)
        data, targets = zip(*temp)

    split = int(split * len(data))

    data_train, targets_train = data[:split], targets[:split]
    data_test, targets_test = data[split:], targets[split:]

    data_train, targets_train = np.concatenate(data_train, axis=0), np.concatenate(targets_train, axis=0)
    data_test, targets_test = np.concatenate(data_test, axis=0), np.concatenate(targets_test, axis=0)

    return data_train, targets_train, data_test, targets_test





