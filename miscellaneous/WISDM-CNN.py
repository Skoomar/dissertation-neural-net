# https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_data(file_path):
    column_names = ['user-id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)

    # remove ";" character from end of line
    df['z-axis'].replace(regex=True,
                         inplace=True,
                         to_replace=r';',
                         value=r'')
    # remove missing values
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def dataframe_summary(dataframe):
    print(f'No of columns in dataframe: {dataframe.shape[1]}')
    print(f'No of rows in dataframe: {dataframe.shape[0]}')


df = read_data('wisdm-dataset/raw/watch/accel/data_1600_accel_watch.txt')
dataframe_summary(df)
# print first 20 records of the data
print(df.head(20))

# show how many training examples exist for each activity
df['activity'].value_counts().plot(kind='bar', title='No. of Examples of Each Activity Type')
plt.show()


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
                                        figsize=(15, 10),
                                        sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'X-Axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'Y-Axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'r')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


for activity in np.unique(df['activity']):
    subset = df[df['activity'] == activity][:180]
    plot_activity(activity, subset)

# model = models.Sequential(12)
#
# model.add(layers.Conv1D(2), activation='relu', input_shape=(5))
# model.add(layers.BatchNormalization())
