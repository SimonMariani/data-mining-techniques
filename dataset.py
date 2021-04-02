import pandas as pd
from datetime import datetime
import numpy as np
import time
import pickle
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import argparse
import yaml
from sklearn import preprocessing

def participant(row):
    person = int(row.id.split('.')[1])
    return person

def day(row):
    day = row.time.day
    return day

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        object_file = pickle.load(input)

    return object_file


def initial_processing(data):
    data['time'] = pd.to_datetime(data['time'], infer_datetime_format=True)
    data['person'] = data.apply(lambda row: participant(row), axis=1, result_type='expand')
    data['time_seconds'] = data.time.values.astype(np.int64) // 10 ** 9
    data['day'] = data.apply(lambda row: day(row), axis=1, result_type='expand')

    return data

def mid_processing(data):

    all_persons = np.sort(data['person'].unique())
    all_data = []

    for person in all_persons:

        data_person = data.loc[data.person == person]
        all_days = np.sort(data_person['day'].unique())
        days = []

        for day in all_days:
            data_day = data_person.loc[data.day == day]
            data_day.sort_values('time_seconds')

            days.append(data_day)

        all_data.append(days)

    return all_data

def final_processing(data, window=5, day_value='mean', temporal=False):
    """returns the data which is still sorted per person and per day but now every person is a pandas dataframe
    with the number of days as index and the columns as features"""

    all_types = data[0][0]['variable'].unique()
    days = np.arange(len(data[0]))
    new_data = [pd.DataFrame(index=days, columns=all_types) for person in data]

    for i in range(len(data)):  # every person
        for j in range(len(data[i])):  # every day for that person
            for k, type in enumerate(all_types):  # every type for that day

                # We take a subframe of all the values
                all_vals = data[i][j].loc[data[i][j].variable == type]['value']

                #  We can chose not to do this and impute the missing data points with for example knn
                if (len(all_vals) == 0 and j != 0) or (all_vals.isnull().values.any() and j!=0):
                    new_data[i][type].iloc[j] = 0
                    continue
                elif len(all_vals) == 0 or all_vals.isnull().values.any():
                    new_data[i][type].iloc[j] = 0
                    continue

                # Decide what value we take from the current day
                if day_value == 'mean':
                    val = all_vals.mean()
                elif day_value == 'max':
                    val = all_vals.max()
                elif day_value == 'min':
                    val = all_vals.min()
                else:
                    val = all_vals.mean()

                # Save the new value
                new_data[i][type].iloc[j] = val

        # Normalize the data w.r.t every person
        new_data[i] = normalize(new_data[i], new_data[i].columns.drop(['mood']))

    # For the non temporal dataset we add the moving averages
    if not temporal:
        new_data = moving_averages(new_data, window)
    else:
        return new_data

    return new_data

def moving_averages(data, window):

    # We first identify all types of variables
    all_types = data[0].columns

    # We loop over the players and find the moving averages of every type
    for i in range(len(data)):
        for type in all_types:
            moving_average, cumulative_average, exponential_average = get_moving_averages(data[i], window, type)

            # initialize the columns of the dataframe
            data[i][type + '_moving'] = moving_average
            data[i][type + '_cumulative'] = cumulative_average
            data[i][type + '_exponential'] = exponential_average

    return data

def get_moving_averages(days, window, type):

    # Variables to keep track of the moving averages
    prev_vals = [0 for i in range(window)]
    moving_average = cumulative_average = exponential_average = 0
    all_moving_average = []
    all_cumulative_average = []
    all_exponential_average = []

    # We loop over all the days and save the moving averages for each day
    for j in range(len(days)):

        # get the value belonging to the current day
        val = days[type].iloc[j]

        # Append the new val and calculate the new averages
        prev_vals.append(val)

        if j == 0:
            exponential_average = val
            cumulative_average = val
            moving_average = val

        elif j < window:
            moving_average = (val + (j * moving_average)) / (j + 1)
            cumulative_average = (val + (j * cumulative_average)) / (j + 1)
            exponential_average = (1 - 2 / (j + 1)) * exponential_average + 2 / (j + 1) * val

        elif j >= window:
            moving_average += (val - prev_vals[0]) / window
            cumulative_average = (val + (j * cumulative_average)) / (j + 1)
            exponential_average = (1 - 2 / (window + 1)) * exponential_average + 2 / (window + 1) * val

        # Remove the last value
        prev_vals.pop(0)

        # Save all the new changes
        all_moving_average.append(moving_average)
        all_cumulative_average.append(cumulative_average)
        all_exponential_average.append(exponential_average)

    return all_moving_average, all_cumulative_average, all_exponential_average

def normalize(data, columns):
    x = data[columns].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data[columns] = pd.DataFrame(x_scaled)
    return data

def save_temporal(data, filename='processed_data_temporal.pkl'):

    targets = []
    new_data = []
    for i in range(len(data)):
        target_days = []
        data_days = []

        for j in range(len(data[i]) - 1):
            data_days.append(torch.tensor(data[i].iloc[j]))
            target_days.append(torch.tensor(data[i].iloc[j + 1]['mood']))

        new_data.append(data_days)
        targets.append(target_days)

    save_object((new_data, targets), filename)

def save_features(data, filename='processed_data_features.pkl'):

    targets = []
    new_data = []
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            new_data.append(data[i].iloc[j].to_numpy().astype(float))
            targets.append(float(data[i].iloc[j + 1]['mood']))

    save_object((new_data, targets), filename)

class MOOD_loader(Dataset):

    def __init__(self, root='processed_data_temporal.pkl'):

        data, targets = self.load_data(root)

        self.data = data
        self.targets = targets
        self.root = root

    def load_data(self, root):
        data, labels = load_object(root)
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        sequence = self.data[item]
        targets = self.targets[item]

        return sequence, targets


def get_data(path='./dataset_mood_smartphone.csv', window=5, day_value='mean', temporal=True):
    """Returns the completely processed data given a path"""
    data = pd.read_csv(path, index_col=0)
    data = initial_processing(data)
    data = mid_processing(data)  # this drastically changes the shape of the data
    data = final_processing(data, window=window, day_value=day_value, temporal=temporal)  # also changes the shape of the data

    if temporal:
        save_temporal(data)
        data, labels = load_object('processed_data_temporal.pkl')
    else:
        save_features(data)
        data, labels = load_object('processed_data_features.pkl')

    print(data[0:10])
    print(len(data))
    print(len(data[0]))
    print(labels)


if __name__ == "__main__":
    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)
    config = {**config['dataset']}

    get_data(config['path'], config['window'], config['day_value'], config['temporal'])