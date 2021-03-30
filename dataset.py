import pandas as pd
from datetime import datetime
import numpy as np
import time
import pickle

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


def final_processing(data, type='mood', k=5, day_value='mean'):
    """returns the data which is still sorted per person and per day but every day is now a data point instead
    of a pandas dataframe."""

    for i in range(len(data)):
        prev_vals = [0 for i in range(k)]
        moving_average = 0
        cumulative_average = 0
        exponential_average = 0

        for j in range(len(data[i])):
            all_vals = data[i][j].loc[data[i][j].variable == type]['value']

            # We skip this iteration and treat it as if it never happened when there are no records
            if len(all_vals) == 0:
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

            # Append the new val and calculate the new average
            prev_vals.append(val)

            if j == 0:
                exponential_average = val
                cumulative_average = val
                moving_average = val

            elif j < k:
                moving_average = (val + (j*moving_average)) / (j+1)
                cumulative_average = (val + (j*cumulative_average)) / (j+1)
                exponential_average = (1 - 2 / (j+1)) * exponential_average + 2 / (j+1) * val

            elif j >= k:
                moving_average += (val - prev_vals[0]) / k
                cumulative_average = (val + (j*cumulative_average)) / (j+1)
                exponential_average = (1 - 2 / (k+1)) * exponential_average + 2 / (k+1) * val

            # Remove the last value
            prev_vals.pop(0)

            # Save all the new changes
            data[i][j] = [val, round(moving_average, 1), round(cumulative_average,1), round(exponential_average, 1)]

    return data

def get_data(path='./dataset_mood_smartphone.csv'):
    """Returns the completely processed data given a path"""
    data = pd.read_csv(path, index_col=0)
    data = initial_processing(data)
    data = mid_processing(data)  # this drastically changes the shape of the data
    data = final_processing(data, k=5, type='mood', day_value='mean')  # also changes the shape of the data


    save_object(data, 'processed_data.pkl')

    data = load_object('processed_data.pkl')

    print(data[0][0:10])


if __name__ == "__main__":
    get_data()