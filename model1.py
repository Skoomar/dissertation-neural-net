import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras import models, layers
from keras import callbacks
from keras.utils import np_utils
import basic_model


pd.options.display.float_format = '{:.1f}'.format
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


# split dataset so there is an appropriate amount of records for EACH activity in the training and test sets
# avoids problem of e.g. some activities having 5 records in training and only 1 in test while others have 2 in training
# and 4 in test
def split_training_test(data, labels):
    # activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
    encoded_activity_ids = np.arange(18)
    train_data = np.empty((0, data.shape[1], 3))
    train_labels = np.empty((0))
    test_data = np.empty((0, data.shape[1], 3))
    test_labels = np.empty((0))

    # TODO: change it to work based on the segments not individual things
    # find segments corresponding to each activity label and split them 70:30
    for a_id in encoded_activity_ids:
        matching_indexes = np.where(labels == a_id)[0]
        current_data = data[matching_indexes]
        current_labels = labels[matching_indexes]

        # calculate a new random split for every activity label
        # TODO: try seeing what putting the same split for every label does to results
        split = np.random.rand(len(matching_indexes)) < 0.70
        # axis set to 0 to stop it flattening the data when it appends
        train_data = np.append(train_data, current_data[split], axis=0)
        train_labels = np.append(train_labels, current_labels[split])
        test_data = np.append(test_data, current_data[~split], axis=0)
        test_labels = np.append(test_labels, current_labels[~split])

    return train_data, train_labels, test_data, test_labels


def create_segments_and_labels(data, time_steps, step, label_name):
    # features are signals on the x, y, and z axes
    N_FEATURES = 3

    segments = []
    labels = []
    for i in range(0, len(data) - time_steps, step):
        xs = data['x-axis'].values[i: i + time_steps]
        ys = data['y-axis'].values[i: i + time_steps]
        zs = data['z-axis'].values[i: i + time_steps]

        # define this segment with the activity that occurs most in this segment
        label = stats.mode(data[label_name][i: i + time_steps])[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # bring segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


dataset = read_data('wisdm-dataset/raw/watch/accel/data_1627_accel_watch.txt')
dataset['x-axis'] = feature_normalise(dataset['x-axis'])
dataset['y-axis'] = feature_normalise(dataset['y-axis'])
dataset['z-axis'] = feature_normalise(dataset['z-axis'])

# encode the labels into numerical representations so the neural network can work with them
ENCODED_LABEL = 'ActivityEncoded'
# use LabelEncoder to convert from String to Integer
le = preprocessing.LabelEncoder()
# add the new encoded labels as a column in the dataset
dataset[ENCODED_LABEL] = le.fit_transform(dataset['activity'].values.ravel())

# number of steps within one time segment
TIME_PERIODS = 80
# steps to take from one segment to next - if same as TIME_PERIODS, then no overlap occurs between segments
STEP_DISTANCE = 40

segments, labels = create_segments_and_labels(dataset, TIME_PERIODS, STEP_DISTANCE, ENCODED_LABEL)

# split = np.random.rand(len(segments)) < 0.70
# train_x = segments[split]
# train_y = labels[split]
# test_x = segments[~split]
# test_y = labels[~split]
train_x, train_y, test_x, test_y = split_training_test(segments, labels)

# a=np.arange(18)
# for i in a:
#     tr=len(np.where(train_y == i)[0])
#     te=len(np.where(test_y == i)[0])
#     print("train for label", i, ":", tr)
#     print("test for label", i, ":", te)
#     print("%:", (tr/(tr+te)) * 100)
#     print("%:", (te/(tr+te)) * 100)
#     print("sum:", tr+te, "\n")

# store the following variables to use for constructing the neural network
# no of time periods within in one record (we've set it to 80 because each data point has an interval of 4 seconds)
# and there are 3 sensors in this dataset
num_time_periods, num_sensors = train_x.shape[1], train_x.shape[2]
# the number of different activities we have - will be used to define the number of output nodes in our network
num_classes = le.classes_.size

# flatten the data so the network so we can input it into the network
# TODO: pretty sure you can just have a 'Flatten' input layer in the network instead of doing it manually here
input_shape = (num_time_periods * num_sensors)
train_x = train_x.reshape(train_x.shape[0], input_shape)
test_x = test_x.reshape(test_x.shape[0], input_shape)

# convert all data to float32 so TF can read it
train_x = train_x.astype('float32')
train_y = train_y.astype('float32')
test_x = test_x.astype('float32')
test_y = test_y.astype('float32')

# perform one-hot encoding on the labels
# TODO: try doing this the way the other tutorial does so i don't have to import keras as well
train_y_hot = np_utils.to_categorical(train_y, num_classes)
test_y_hot = np_utils.to_categorical(test_y, num_classes)

model = basic_model.create_model(input_shape, num_classes)

# trained_model = basic_model.train_model(model, train_x, train_y_hot, verbose=0)
# print(basic_model.evaluate_model(trained_model, test_x, test_y_hot, verbose=0))
#
# trained_model = basic_model.train_model(model, train_x, train_y_hot, verbose=0)
# print(basic_model.evaluate_model(trained_model, test_x, test_y_hot, verbose=0))
#
# trained_model = basic_model.train_model(model, train_x, train_y_hot, verbose=0)
# print(basic_model.evaluate_model(trained_model, test_x, test_y_hot, verbose=0))
