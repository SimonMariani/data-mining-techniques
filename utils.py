import pickle
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.distribution import MultiprocessingDistributor
import pandas as pd


def add_day_specifics(new_data, data):

    # Go over all the participants
    for participant in range(len(data)):

        # Add weekdays as columns in the new data
        weekdays, months, nmbr_moods = get_day_specifics(data[participant])

        new_data[participant]['weekday'] = weekdays
        new_data[participant]['months'] = months
        new_data[participant]['nmbr_moods'] = nmbr_moods

        new_data[participant] = pd.concat([new_data[participant], pd.get_dummies(new_data[participant]['weekday'])], axis=1)

        # For the months we might not have all the months but we need to represent all of them
        new_data[participant]['months'] = pd.Categorical(new_data[participant]['months'],
                                                         categories=['April','February', 'March', 'May', 'June'])

        new_data[participant] = pd.concat([new_data[participant], pd.get_dummies(new_data[participant]['months'])], axis=1)

        del new_data[participant]['weekday']
        del new_data[participant]['months']


    return new_data


def get_day_specifics(days):

    weekdays = []
    months = []
    nmbr_moods = []

    for day in days:
        weekdays.append(day['time'].iloc[0].day_name())
        months.append(day['time'].iloc[0].month_name())
        nmbr_moods.append(len(day.loc[day.variable == 'mood']))

    return weekdays, months, nmbr_moods


def add_weather_data(new_data, data):

    # We load the weather data
    weather_data = pd.read_csv('./data_raw/weerdata.txt')

    # We add the data about the dates that we need
    weather_data['year'] = [int(str(date)[:4]) for date in weather_data['YYYYMMDD']]
    weather_data['month'] = [int(str(date)[4:6]) for date in weather_data['YYYYMMDD']]
    weather_data['day'] = [int(str(date)[6:]) for date in weather_data['YYYYMMDD']]

    # We remove all the spaces from the weather data
    weather_data.columns = [column.split(' '[0])[-1] for column in weather_data.columns]

    # We remove all the information that we do not need (from 2014 and between 8 and 22)
    weather_data_2014 = weather_data.loc[(weather_data['year'] == 2014) & (weather_data['HH'] > 7) & (weather_data['HH'] < 23)]

    weather_data_2014_sub = weather_data_2014[['FH', 'T', 'SQ', 'Q', 'DR', 'RH', 'N', 'R', 'S', 'O', 'Y', 'day', 'month']]

    # We go over all the participants and days for that participant and add the weather data
    for i, participant in enumerate(data):
        all_days = []
        for day in participant:

            timestamp = day['time'].iloc[0]  # the first time measurement of the day
            date_day, date_month = timestamp.day, timestamp.month

            # We take the mean for all the values
            weather_day = weather_data_2014_sub.loc[
                                (weather_data_2014_sub['day'] == date_day) &
                                (weather_data_2014_sub['month'] == date_month)]

            # Take the mean over all measurements
            weather_day = weather_day.astype('int64').mean(axis=0, numeric_only=False).to_frame().T
            del weather_day['day']
            del weather_day['month']

            # Add the datafame to your list of days
            all_days.append(weather_day)

        # Add all the days together
        all_days_frame = pd.concat(all_days, axis=0).reset_index(drop=True)
        new_data[i] = pd.concat([new_data[i], all_days_frame], axis=1)

    return new_data


def add_tsfresh_participant(data, tsfresh_features, columns, k):

    # The dictionary containing the features that we want to extract and the setting for those features
    if tsfresh_features == 'minimal':
        settings = MinimalFCParameters()
    elif tsfresh_features == 'efficient':
        settings = EfficientFCParameters()
    elif tsfresh_features == 'comprehensive':
        settings = ComprehensiveFCParameters()
    else:
        settings = MinimalFCParameters()

    for participant in range(len(data)):

        # First we add the necesary columns
        data[participant]['id'] = 0
        data[participant]['index'] = data[participant].index

        # We create the rolled time series which also creates new ids, also note that putting max_timeshift to none
        # means that it takes the maximal possible lengths
        rolled_series = roll_time_series(data[participant], column_id='id', column_sort='index', max_timeshift=k)

        all_features = []
        for column in columns:
            # We extract the features for every element of the time series which return a dataframe with the same number
            # of rows as the original dataframe but a different number of columns
            extracted = extract_features(rolled_series, default_fc_parameters=settings, column_id='id',
                                         column_sort='index', column_value=column)

            # We need to reset the indexes as they have been changed and add them to our list of features
            all_features.append(extracted.reset_index(drop=True))

        # Add all the features together
        extracted = pd.concat(all_features, axis=1)

        # We drop the columns that we previously created because we do no want them in the data
        del data[participant]['id']  # note that you can also use df.drop here
        del data[participant]['index']

        data[participant] = pd.concat([data[participant], extracted], axis=1)

    return data


def add_tsfresh_day(new_data, data, tsfresh_features, columns):

    # The dictionary containing the features that we want to extract and the setting for those features
    if tsfresh_features == 'minimal':
        settings = MinimalFCParameters()
    elif tsfresh_features == 'efficient':
        settings = EfficientFCParameters()
    elif tsfresh_features == 'comprehensive':
        settings = ComprehensiveFCParameters()
    else:
        settings = MinimalFCParameters()

    for participant in range(len(data)):

        all_days = []
        for day in range(len(data[participant])):

            # We only take the columns that we are interested in
            sub_data = data[participant][day].loc[data[participant][day]['variable'].isin(columns)]

            # Drop all nan values
            sub_data = sub_data.dropna(axis=0)

            # If a columns is missing we add a row with that column and a 0.
            # If a column contains nan values we do the same
            for col in columns:
                if col not in sub_data['variable']:
                    new_row = sub_data.iloc[0].copy(deep=True)
                    new_row['variable'] = col
                    new_row['value'] = 0
                    sub_data.append(new_row)

            from tsfresh.utilities.dataframe_functions import impute_dataframe_zero
            # Extract features for every variable still left in the dataframe
            extracted = extract_features(sub_data, default_fc_parameters=settings, column_id='variable',
                                         column_sort='time_seconds', column_value='value')

            # We do not want multiple rows therefore in the case of multiple variables therefore we need to change it
            # We also change the column names so that we know what kind if features they are
            extracted = extracted.stack()
            extracted.index = extracted.index.map('{0[1]}_{0[0]}_day'.format)
            extracted = extracted.to_frame().T

            # Add the extracted features to a list
            all_days.append(extracted)

        # Concat the days to make a new dataframe and reset the index to prevent conflicts
        all_days = pd.concat(all_days, axis=0).reset_index(drop=True)

        # Add the new features to the data
        new_data[participant] = pd.concat([new_data[participant], all_days], axis=1)

    return new_data


def moving_averages(data, window, variables):

    # We loop over the players and find the moving averages of every type
    for i in range(len(data)):
        for variable in variables:
            moving_average, cumulative_average, exponential_average = get_moving_averages(data[i], window, variable)

            # initialize the columns of the dataframe
            data[i][variable + '_moving'] = moving_average
            data[i][variable + '_cumulative'] = cumulative_average
            data[i][variable + '_exponential'] = exponential_average

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


def participant(row):
    person = int(row.id.split('.')[1])
    return person


def day(row):
    day = row.time.dayofyear  # because there is only one year this gives us distinct days
    return day


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        object_file = pickle.load(input)

    return object_file
