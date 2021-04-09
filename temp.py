from utils import load_object
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
from tsfresh import select_features, extract_relevant_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import roll_time_series

if __name__ == '__main__':
    # We load the weather data
    weather_data = pd.read_csv('weerdata.txt')

    # We add the data about the dates that we need
    weather_data['year'] = [int(str(date)[:4]) for date in weather_data['YYYYMMDD']]
    weather_data['month'] = [int(str(date)[4:6]) for date in weather_data['YYYYMMDD']]
    weather_data['day'] = [int(str(date)[6:]) for date in weather_data['YYYYMMDD']]

    # We remove all the spaces from the weather data
    weather_data.columns = [column.split(' '[0])[-1] for column in weather_data.columns]

    # We remove all the information that we do not need
    weather_data_2014 = weather_data.loc[weather_data['year'] == 2014]

    weather_data_2014_sub = weather_data_2014[
        ['FH', 'T', 'SQ', 'Q', 'DR', 'RH', 'N', 'R', 'S', 'O', 'Y', 'day', 'month']]

    print(weather_data_2014_sub)

    weather_day = weather_data_2014_sub.loc[
                                (weather_data_2014_sub['day'] == 7) &
                                (weather_data_2014_sub['month'] == 4)]

    print(weather_day.dtypes)
    print(weather_day.astype('int64').dtypes)
    print(weather_day.mean(axis=0))
    weather_day = weather_day.astype('int64').mean(axis=0).to_frame().T

    print(weather_day)

    print(weather_day.columns)



"""data, labels = load_object('processed_data_features.pkl')

other_data = pd.read_csv('example_3.csv', index_col=0)
#print(other_data.columns)

data = pd.DataFrame(data)
labels = pd.Series(labels)

data.columns = other_data.columns

print(data)
print(labels)

import scipy

correlation = []
significance = []
for col in data.columns:

    print(col)

    stats = scipy.stats.pearsonr(data[col], labels)
    correlation.append(stats[0])
    significance.append(stats[1])

    print(stats)

list1, list2, list3 = zip(*sorted(zip(correlation, significance, data.columns)))
merge = [(a, b, c) for a, b, c in zip(list1, list2, list3)]"""