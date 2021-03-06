from utils import load_object
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, MinimalFCParameters
from tsfresh import select_features, extract_relevant_features
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import roll_time_series
import random
import numpy as np

if __name__ == '__main__':

    data, labels = load_object('./data_processed/processed_data_advanced_train.pkl')

    print(len(data))
    print(len(labels))

    print(data[0])
    print(labels[0])

"""data, targets = load_object('processed_data_features.pkl')

other_data = pd.read_csv('example_3.csv', index_col=0)
#print(other_data.columns)

data = pd.DataFrame(data)
targets = pd.Series(targets)

data.columns = other_data.columns

print(data)
print(targets)

import scipy

correlation = []
significance = []
for col in data.columns:

    print(col)

    stats = scipy.stats.pearsonr(data[col], targets)
    correlation.append(stats[0])
    significance.append(stats[1])

    print(stats)

list1, list2, list3 = zip(*sorted(zip(correlation, significance, data.columns)))
merge = [(a, b, c) for a, b, c in zip(list1, list2, list3)]"""