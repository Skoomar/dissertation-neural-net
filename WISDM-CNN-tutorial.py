# http://aqibsaeed.github.io/2016-11-04-human-activity-recognition-cnn/ #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

plt.style.use('ggplot')


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

    data = pd.read_csv(file_path,
                     header=None,
                     names=column_names)

    # remove ";" character from end of line
    data['z-axis'].replace(regex=True,
                         inplace=True,
                         to_replace=r';',
                         value=r'')
    # convert z-axis values back to floats after having turned them into strings to remove the ;
    data['z-axis'] = data['z-axis'].apply(convert_to_float)
    # remove missing values
    data.dropna(axis=0, how='any', inplace=True)

    return data


def feature_normalise(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


dataset = read_data('wisdm-dataset/raw/watch/accel/data_1600_accel_watch.txt')
dataset['x-axis'] = feature_normalise(dataset['x-axis'])
dataset['y-axis'] = feature_normalise(dataset['y-axis'])
dataset['z-axis'] = feature_normalise(dataset['z-axis'])

# plot the sensor data
# for activity in np.unique(dataset['activity']):
#     subset = dataset[dataset['activity'] == activity][:180]
#     plot_activity(activity, subset)


def windows(data, size):
    start = 0
