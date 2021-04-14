from dataset import MOOD_loader
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import Basic_LSTM, Basic_BiLSTM, Basic_Net
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score, balanced_accuracy_score, r2_score
from sklearn.ensemble import RandomForestRegressor
from dataset import load_object
import random


def train_lstm(config, fold):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    device = torch.device(config['device'])

    # Extract the data
    data_train, labels_train, data_test, labels_test = fold
    total_num_cols = data_train[0].shape[1]

    # Initialize the dataset and data loader
    dataset_train = MOOD_loader(data_train, labels_train, temporal=True)
    dataset_test = MOOD_loader(data_test, labels_test, temporal=True)
    data_train = DataLoader(dataset_train, config['batch_size'])  # batch size=9 means exactly 3 batches per epoch
    data_test = DataLoader(dataset_test, config['batch_size'])

    # Initialize the model that we are going to use
    model = Basic_LSTM(lstm_num_hidden=config['hidden'], lstm_num_layers=config['layers'], input_dim=config['in_dim'],
                        output_dim=config['out_dim'])
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    # Some eraly stopping parameters for if early stopping is enabled
    val_len = config['val_len']
    val_history = [0 for i in range(val_len)]

    for epoch in range(config['epochs']):
        for inputs, targets in data_train:
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

        mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = eval_LSTM(model, data_test, device, total_num_cols)

        val_history.pop(0)
        val_history.append(mse)

        if config['early_stopping'] and epoch > val_len:
            half = int(val_len / 2)
            if (sum(val_history[:half]) / val_len) < (sum(val_history[half:]) / val_len):
                break

        if config['print']:
            if epoch % config['print_every'] == 0:
                print(f'epoch: {epoch}')
                print(f'loss: {loss}')
                print(f'test mse: {mse}')
                print(f'accuracy: {accuracy}')
                print('\n')

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def train_bilstm(config, fold):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    device = torch.device(config['device'])

    # Extract the data
    data_train, labels_train, data_test, labels_test = fold
    total_num_cols = data_train[0].shape[1]

    # Initialize the dataset and data loader
    dataset_train = MOOD_loader(data_train, labels_train, temporal=True)
    dataset_test = MOOD_loader(data_test, labels_test, temporal=True)
    data_train = DataLoader(dataset_train, config['batch_size'])  # batch size=9 means exactly 3 batches per epoch
    data_test = DataLoader(dataset_test, config['batch_size'])

    # Initialize the model that we are going to use
    model = Basic_BiLSTM(lstm_num_hidden=config['hidden'], lstm_num_layers=config['layers'], input_dim=config['in_dim'],
                        output_dim=config['out_dim'])
    model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    # Some eraly stopping parameters for if early stopping is enabled
    val_len = config['val_len']
    val_history = [0 for i in range(val_len)]

    for epoch in range(config['epochs']):
        for inputs, targets in data_train:
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

        mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = eval_LSTM(model, data_test, device, total_num_cols)

        val_history.pop(0)
        val_history.append(mse)

        if config['early_stopping'] and epoch > val_len:
            half = int(val_len / 2)
            if (sum(val_history[:half]) / val_len) < (sum(val_history[half:]) / val_len):
                break

        if config['print']:
            if epoch % config['print_every'] == 0:
                print(f'epoch: {epoch}')
                print(f'loss: {loss}')
                print(f'test mse: {mse}')
                print(f'accuracy: {accuracy}')
                print('\n')

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def eval_LSTM(model, dataloader, device, total_num_cols):

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
    mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = get_metrics(all_targets.cpu().detach().numpy(), all_predictions.cpu().detach().numpy(),total_num_cols)

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def train_net(config, fold):
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # Initialize the device which to run the model on
    device = torch.device(config['device'])

    # Extract the data
    data_train, labels_train, data_test, labels_test = fold
    total_num_cols = data_train[0].shape[1]

    # Initialize the dataset and data loader
    dataset_train = MOOD_loader(data_train, labels_train, temporal=False)
    dataset_test = MOOD_loader(data_test, labels_test, temporal=False)
    data_train = DataLoader(dataset_train, config['batch_size'])  # batch size=9 means exactly 3 batches per epoch
    data_test = DataLoader(dataset_test, config['batch_size'])

    # Initialize the model that we are going to use
    model = Basic_Net(config['in_dim'], config['out_dim'])
    model = model.to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    # Some eraly stopping parameters for if early stopping is enabled
    val_len = config['val_len']
    val_history = [0 for i in range(val_len)]

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

        mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = eval_NN(model, data_test, device, total_num_cols)

        val_history.pop(0)
        val_history.append(mse)

        if config['early_stopping'] and epoch > val_len:
            half = int(val_len/2)
            if (sum(val_history[:half]) / val_len) < (sum(val_history[half:]) / val_len):
                break

        if config['print']:
            if epoch % config['print_every'] == 0:
                print(f'epoch: {epoch}')
                print(f'loss: {loss}')
                print(f'test mse: {mse}')
                print(f'accuracy: {accuracy}')
                print('\n')

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy

def eval_NN(model, dataloader, device, total_num_cols):

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
    mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = get_metrics(all_targets.cpu().detach().numpy(), all_predictions.cpu().detach().numpy(),total_num_cols)

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def train_linear(config, fold):

    data_train, labels_train, data_test, labels_test = fold
    total_num_cols = data_train[0].shape[1]
    data_train, labels_train = np.concatenate(data_train, axis=0), np.concatenate(labels_train, axis=0)
    data_test, labels_test = np.concatenate(data_test, axis=0), np.concatenate(labels_test, axis=0)

    model = linear_model.LinearRegression()

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = get_metrics(labels_test, predictions, total_num_cols)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def train_svm(config, fold):

    data_train, labels_train, data_test, labels_test = fold
    total_num_cols = data_train[0].shape[1]
    data_train, labels_train = np.concatenate(data_train, axis=0), np.concatenate(labels_train, axis=0)
    data_test, labels_test = np.concatenate(data_test, axis=0), np.concatenate(labels_test, axis=0)

    model = SVR(max_iter=10000)

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = get_metrics(labels_test, predictions, total_num_cols)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def train_random_forrest(config, fold):

    data_train, labels_train, data_test, labels_test = fold
    total_num_cols = data_train[0].shape[1]
    data_train, labels_train = np.concatenate(data_train, axis=0), np.concatenate(labels_train, axis=0)
    data_test, labels_test = np.concatenate(data_test, axis=0), np.concatenate(labels_test, axis=0)

    model = RandomForestRegressor()

    model.fit(data_train, labels_train)
    predictions = model.predict(data_test)

    mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = get_metrics(labels_test, predictions, total_num_cols)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def train_baseline(config, fold, folds_baseline):

    data_train, _, _, labels_test = fold
    total_num_cols = data_train[0].shape[1]
    labels_test = np.concatenate(labels_test, axis=0)

    _, baseline_test = folds_baseline
    baseline_test = np.concatenate(baseline_test, axis=0)

    mse, rmse, r2, adj_r2, accuracy, balanced_accuracy = get_metrics(labels_test, baseline_test, total_num_cols)

    if config['print']:
        print(f'test mse: {mse}')
        print(f'test accuracy: {accuracy}')

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy


def get_metrics(y_real, y_pred, total_num_cols):

    # Rgression metrics
    mse = mean_squared_error(y_real.astype('float32'), y_pred.astype('float32'))
    rmse = mean_squared_error(y_real.astype('float32'), y_pred.astype('float32'), squared=False)
    r2 = r2_score(y_real.astype('float32'), y_pred.astype('float32'))
    n = y_real.shape[0]
    k = total_num_cols
    adj_r2 = 1-(((1-r2)*(n-1))/(n-k-1))

    # Classificationm metrics for interpretability
    accuracy = accuracy_score(np.around(y_real).astype('int'), np.around(y_pred).astype('int'))
    balanced_accuracy = balanced_accuracy_score(np.around(y_real).astype('int'), np.around(y_pred).astype('int'))

    return mse, rmse, r2, adj_r2, accuracy, balanced_accuracy






