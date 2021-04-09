import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import argparse
import yaml
from sklearn import preprocessing
from utils import save_object, load_object, participant, day, moving_averages
import random
from utils import add_tsfresh_participant, add_tsfresh_day, add_weather_data


def add_columns(data):
    """Adds columns to the data that are necesary for other tasks"""

    data['time'] = pd.to_datetime(data['time'], infer_datetime_format=True)
    data['participant'] = data.apply(lambda row: participant(row), axis=1, result_type='expand')
    data['time_seconds'] = data.time.values.astype(np.int64) // 10 ** 9
    data['day'] = data.apply(lambda row: day(row), axis=1, result_type='expand')

    return data


def split_participant_day(data):
    """Split the dataframe on participants and days to create multiple subframes.
    The output shape is then (participants, days, dataframe)"""

    all_persons = np.sort(data['participant'].unique())
    all_data = []

    for person in all_persons:

        data_person = data.loc[data.participant == person]
        all_days = np.sort(data_person['day'].unique())
        days = []

        for day in all_days:
            data_day = data_person.loc[data.day == day]
            data_day.sort_values('time_seconds')

            # only append if it has a mood variable. So this basically takes out all days without a mood
            if 'mood' in data_day['variable'].unique():
                days.append(data_day)


        all_data.append(days)

    return all_data


def extract_values(data):
    """Turns all the dataframes into other dataframes where the columns are the values and the rows are the days.
    It does this for every participant, so the output is of the shape (participants, days, values) where the last
    two elements are in a data frame"""

    mean_vars = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    sum_vars = ['screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment',
                'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social',
                'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
    all_types = mean_vars + sum_vars

    new_data = [pd.DataFrame(index=np.arange(len(person)), columns=all_types) for person in data]

    for i in range(len(data)):  # every person
        for j in range(len(data[i])):  # every day for that person

            # First we take all the variables over which we want the mean
            for var in all_types:

                # Take all the current values for that variable
                all_vals = data[i][j].loc[data[i][j].variable == var]['value']

                # If there are no values or there is a nan value we simply make it a nan value, else
                # we check if we need to take the mean or the sum
                if (len(all_vals) == 0) or all_vals.isnull().values.any():
                    new_data[i][var].iloc[j] = float('nan')
                elif var in mean_vars:
                    new_data[i][var].iloc[j] = all_vals.mean()
                elif var in sum_vars:
                    new_data[i][var].iloc[j] = all_vals.sum()

    return new_data


def impute(data):
    """Now that we have a dataframe for every person we still need to impute the missing values, we do this
    per participant"""

    for i in range(len(data)):  # every person
        data[i] = data[i].fillna(0)

    return data


def add_features(new_data, data, config):
    """This method returns the data object with additional features such as the moving averages"""

    #new_data = moving_averages(new_data, window=window)
    #new_data = add_tsfresh_participant(new_data, config['tsfresh_features'], columns=config['columns'], k=config['window'])
    #new_data = add_tsfresh_day(new_data, data, config['tsfresh_features'], columns=config['columns'])
    new_data = add_weather_data(new_data, data)

    return new_data

def normalize(data):
    for person in range(len(data)):
        columns = np.delete(data[person].columns, 0)  # we delete the first (mood) column

        x = data[person][columns].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        data[person][columns] = pd.DataFrame(x_scaled)
    return data


def save_temporal(data, filename='processed_data_temporal.pkl'):

    new_data = []
    targets = []

    for i in range(len(data)):  # all participants
        target_days = []
        data_days = []

        for j in range(len(data[i]) - 1):  # all days
            data_days.append(torch.tensor(data[i].iloc[j]))
            target_days.append(torch.tensor(data[i].iloc[j + 1]['mood']))

        new_data.append(data_days)
        targets.append(target_days)

    save_object((new_data, targets), filename)


def save_features(data, filename='processed_data_features.pkl'):

    new_data = []
    targets = []
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            new_data.append(data[i].iloc[j].to_numpy().astype(float))
            targets.append(float(data[i].iloc[j + 1]['mood']))

    save_object((new_data, targets), filename)


class MOOD_loader(Dataset):

    def __init__(self, root='processed_data_temporal.pkl', train=True, split=0.8, shuffle=True):

        self.load(root, train, split, shuffle)
        self.root = root

    def load(self, root, train, split, shuffle):
        data, targets = load_object(root)

        if shuffle:
            temp = list(zip(data, targets))
            random.shuffle(temp)
            data, targets = zip(*temp)

        split = int(split * len(data))

        if train:
            data = data[:split]
            targets = targets[:split]
        else:
            data = data[split:]
            targets = targets[split:]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        sequence = self.data[item]
        targets = self.targets[item]

        return sequence, targets


def get_data(path='./dataset_mood_smartphone.csv', window=5, temporal=True):
    """Returns the completely processed data given a path"""
    data = pd.read_csv(path, index_col=0)

    data = add_columns(data)
    data = split_participant_day(data)  # up until this point there is no data loss

    new_data = extract_values(data)
    new_data = impute(new_data)
    new_data = add_features(new_data, data, config)
    new_data = normalize(new_data)

    # Saves the data to a pandas file before saving it as a pickle object in a different format
    if config['save_panda']:
        pd.concat(new_data).to_csv('all_data.csv')

    # For the temporal model we only need these values but for the feature model we still do some feature engineering
    if temporal:
        save_temporal(new_data)
        data, labels = load_object('processed_data_temporal.pkl')  # just an example on how to load the data
    else:
        save_features(new_data)
        data, labels = load_object('processed_data_features.pkl')  # just an example on how to load the data

if __name__ == "__main__":

    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)
    config = {**config['dataset']}

    get_data(config['path'], config['window'], config['temporal'])