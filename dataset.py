import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import argparse
import yaml
from utils import save_object, load_object, participant, day
import random
from utils import moving_averages, add_tsfresh_participant, add_tsfresh_day, add_weather_data, add_day_specifics
import copy
import os

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


def add_basic_features(new_data, data, config):
    """This method returns the data object with additional features"""

    new_data = add_weather_data(new_data, data)
    new_data = add_day_specifics(new_data, data)

    return new_data


def add_advanced_features(new_data, data, config):
    """This method returns the data object with additional features such as the moving averages"""

    # Second we add complex statistics features
    new_data = moving_averages(new_data, window=5, variables=config['columns'])
    #new_data = add_tsfresh_participant(new_data, config['tsfresh_features'], columns=config['columns'], k=config['window'])
    #new_data = add_tsfresh_day(new_data, data, 'minimal', columns=config['columns'])

    return new_data


def get_labels(data):

    new_data = []
    targets = []
    baseline_targets = []
    for i in range(len(data)):
        # Extract the labels
        labels_person = pd.DataFrame(data[i]['mood'][1:].reset_index(drop=True))
        labels_person.columns = ['labels']

        # Extract the baseline predictions
        labels_person_baseline = pd.DataFrame(data[i]['mood'][:-1].reset_index(drop=True))

        # Remove the last datapoint as we do not have a label for that datapoint
        data_person = data[i].iloc[:-1].reset_index(drop=True)

        new_data.append(data_person)
        targets.append(labels_person)
        baseline_targets.append(labels_person_baseline)

    return new_data, targets, baseline_targets


def normalize(data, standard=True, exclude=[]):
    for person in range(len(data)):

        # We only take take the columns that we want to normalize
        sub_frame = data[person][data[person].columns.difference(exclude)]

        # If standard than we perform standarization standardization
        if standard:
            normalized_df = (sub_frame - sub_frame.mean()) / sub_frame.std()
        else:
            normalized_df = (sub_frame - sub_frame.min()) / (sub_frame.max() - sub_frame.min())

        normalized_df = normalized_df.fillna(0)

        data[person][data[person].columns.difference(exclude)] = normalized_df

    return data


def save_data(data, targets, filename='processed_data_temporal.pkl'):

    new_data = [person.to_numpy() for person in data]
    targets = [target.to_numpy().squeeze(axis=1) for target in targets]

    save_object((new_data, targets), filename)


def save_pandas(data, targets, filename='processed_data_pandas.csv'):

    new_data = pd.concat(data, axis=0)
    targets = pd.concat(targets, axis=0)
    data_full = pd.concat([targets, new_data], axis=1)

    data_full.to_csv(filename)


def split_test(data, targets, seed, split=0.8):

    random.seed(seed)
    split = int(split * len(targets))

    # If we just want to shuffle baseline targets
    if data is None:
        random.shuffle(targets)
        targets_train = targets[:split]
        targets_test = targets[split:]

        return targets_train, targets_test

    else:
        temp = list(zip(data, targets))
        random.shuffle(temp)
        data, targets = zip(*temp)

        data_train = data[:split]
        targets_train = targets[:split]

        data_test = data[split:]
        targets_test = targets[split:]

        return data_train, targets_train, data_test, targets_test


def save_subset(data_train, targets_train, data_test, targets_test, baseline_targets, config):

    # Now we want to save all the data
    if not os.path.exists(config['save_folder']):
        os.makedirs(config['save_folder'])

    # Now we inmediately divide the data into subsets
    nmbr_columns = len(config['columns'])
    total_columns = len(data_train[0].iloc[0])

    # Some values that we need to determine until where the columns go
    nmbr_ma = nmbr_columns * 3
    nmbr_tsfp = nmbr_columns * 787  # note that this only works if you use the comprehensive tsfresh pack
    subset_names = ['m', 'pr', 'pr_su', 'pr_su_bf', 'pr_su_bf_ma']  # 'pr_su_bf_ma_tsfp' , 'pr_su_bf_ma_tsfp_tsfd'
    subset_indices = [(0, 1), (0, 4), (0, 20), (0, 43),
                      (0, 43 + nmbr_ma)]  # (0, 43 + nmbr_ma + nmbr_tsfp), (0, total_columns)

    for name, indices in zip(subset_names, subset_indices):
        subset = []

        for i in range(len(data_train)):
            subset.append(data_train[i].iloc[:, indices[0]: indices[1]])

        subset_test = []
        for j in range(len(data_test)):
            subset_test.append(data_test[j].iloc[:, indices[0]: indices[1]])

        save_data(subset, targets_train, config['save_folder'] + '/subdata_' + name + '.pkl')
        save_data(subset_test, targets_test, config['save_folder'] + '/subdata_' + name + '_test.pkl')
        save_pandas(subset, targets_train, config['save_folder'] + '/subdata_' + name + '.csv')


    save_data(data_train, targets_train, filename=config['save_folder'] + '/processed_data_basic_train.pkl')
    save_data(data_test, targets_test, filename=config['save_folder'] + '/processed_data_basic_test.pkl')

    # Saves the data to a pandas file before saving it as a pickle object in a different format
    if config['save_panda']:
        save_pandas(data_train, targets_train, filename=config['save_folder'] + '/processed_data_pandas.csv')

    # Now we do the same for the baseline targets
    targets_train, targets_test = split_test(None, baseline_targets, seed=config['seed'], split=config['test_split'])
    save_object(targets_train, filename=config['save_folder'] + '/baseline_targets_train.pkl')
    save_object(targets_test, filename=config['save_folder'] + '/baseline_targets_test.pkl')


class MOOD_loader(Dataset):

    def __init__(self, data, labels, temporal=False):

        self.load(data, labels, temporal)

    def load(self, data, labels, temporal):

        # If the data is temporal we want to maintain our original list structure
        if temporal:
            self.data = [torch.from_numpy(person) for person in data]
            self.labels = [torch.from_numpy(label) for label in labels]
        else:
            self.data = torch.from_numpy(np.concatenate(data, axis=0))
            self.labels = torch.from_numpy(np.concatenate(labels, axis=0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        sequence = self.data[item]
        targets = self.labels[item]

        return sequence, targets


def get_data(config, path='./data_raw/dataset_mood_smartphone.csv'):
    """Returns the completely processed data given a path"""
    data = pd.read_csv(path, index_col=0)

    data = add_columns(data)
    data = split_participant_day(data)  # up until this point there is no data loss

    new_data = extract_values(data)
    new_data = impute(new_data)

    # For the temporal and feature model we add different features and therefore we also need to
    # take care of the imputation and normalization in a different manner
    new_data = add_basic_features(new_data, data, config)
    new_data = add_advanced_features(new_data, data, config)

    new_data = impute(new_data)

    # This can be done before only ones but because we want to reset the index and such it is easier to do it here
    new_data, targets, baseline_targets = get_labels(new_data)
    baseline_targets = [target.to_numpy().squeeze(axis=1) for target in baseline_targets]

    if config['normalize']:
        new_data = normalize(new_data, standard=config['standard'], exclude=config['exclude_norm'])

    data_train, targets_train, data_test, targets_test = split_test(new_data, targets, seed=config['seed'],
                                                                    split=config['test_split'])

    save_subset(data_train, targets_train, data_test, targets_test, baseline_targets, config)


if __name__ == "__main__":

    # The arguments are determined by command line arguments and a .yml file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configuration.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config, "r"), yaml.SafeLoader)
    config = {**config['dataset']}

    get_data(config, config['path'])