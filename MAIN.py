import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import np_utils
from scipy import stats, fftpack
from sklearn import preprocessing
from sklearn.metrics import classification_report
import models_spec
import math
import myDeepSense

pd.options.display.float_format = '{:.1f}'.format
plt.style.use('ggplot')


def read_data(file_path):
    # Only use these if putting the original (non-merged) data
    # column_names = ['user-id', 'activity', 'timestamp',
    #                 'x-axis_accel_phone', 'y-axis_accel_phone', 'z-axis_accel_phone',
    #                 'x-axis_gyro_phone', 'y-axis_gyro_phone', 'z-axis_gyro_phone',
    #                 'x-axis_accel_watch', 'y-axis_accel_watch', 'z-axis_accel_watch',
    #                 'x-axis_gyro_watch', 'y-axis_gyro_watch', 'z-axis_gyro_watch']

    # no need for header & column_names unless you're using the original data
    data = pd.read_csv(file_path
                       # header=None,
                       # names=column_names
                       )

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
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11) = plt.subplots(nrows=12, figsize=(15, 10),
                                                                                       sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis_accel_phone'], 'xap')
    plot_axis(ax1, data['timestamp'], data['y-axis_accel_phone'], 'yap')
    plot_axis(ax2, data['timestamp'], data['z-axis_accel_phone'], 'zap')
    plot_axis(ax3, data['timestamp'], data['x-axis_gyro_phone'], 'xgp')
    plot_axis(ax4, data['timestamp'], data['y-axis_gyro_phone'], 'ygp')
    plot_axis(ax5, data['timestamp'], data['z-axis_gyro_phone'], 'zgp')
    plot_axis(ax6, data['timestamp'], data['x-axis_accel_watch'], 'xaw')
    plot_axis(ax7, data['timestamp'], data['y-axis_accel_watch'], 'yaw')
    plot_axis(ax8, data['timestamp'], data['z-axis_accel_watch'], 'zaw')
    plot_axis(ax9, data['timestamp'], data['x-axis_gyro_watch'], 'xgw')
    plot_axis(ax10, data['timestamp'], data['y-axis_gyro_watch'], 'ygw')
    plot_axis(ax11, data['timestamp'], data['z-axis_gyro_watch'], 'zgw')

    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


# split dataset so there is an appropriate amount of records for EACH activity in the training and test sets
# avoids problem of e.g. some activities having 5 records in training and only 1 in test while others have 2 in training
# and 4 in test
def split_train_test(data, labels, validation_split=0.7, random=True):
    encoded_activity_ids = np.arange(18)
    # train_data = np.empty((0, data.shape[1], data.shape[2]))
    train_data = np.empty((0, *data.shape[1:]))
    train_labels = np.empty((0))
    test_data = np.empty((0, data.shape[1], data.shape[2]))
    test_labels = np.empty((0))

    # TODO: change it to work based on the segments not individual things
    # TODO: i think i've already done the above todo but check anyway at point
    # find segments corresponding to each activity label and split them 70:30
    for a_id in encoded_activity_ids:
        matching_indexes = np.where(labels == a_id)[0]
        current_data = data[matching_indexes]
        current_labels = labels[matching_indexes]

        if random:
            # calculate a new random split for every activity label
            # TODO: try seeing what putting the same split for every label does to results
            split = np.random.rand(len(matching_indexes)) < validation_split
            # axis set to 0 to stop it flattening the data when it appends
            train_data = np.append(train_data, current_data[split], axis=0)
            train_labels = np.append(train_labels, current_labels[split])
            test_data = np.append(test_data, current_data[~split], axis=0)
            test_labels = np.append(test_labels, current_labels[~split])
        else:
            split = math.floor(len(matching_indexes) * validation_split)
            train_data = np.append(train_data, current_data[:split], axis=0)
            train_labels = np.append(train_labels, current_labels[:split])
            test_data = np.append(test_data, current_data[split:], axis=0)
            test_labels = np.append(test_labels, current_labels[split:])

    return train_data, train_labels, test_data, test_labels


def split_sensors_train_test(accel_phone, gyro_phone, accel_watch, gyro_watch, labels, validation_split=0.7,
                             random=True):
    train_ap = np.empty((0, *accel_phone.shape[1:]))
    train_gp = np.empty((0, *gyro_phone.shape[1:]))
    train_aw = np.empty((0, *accel_watch.shape[1:]))
    train_gw = np.empty((0, *gyro_watch.shape[1:]))
    train_labels = np.empty((0))
    test_ap = np.empty((0, *accel_phone.shape[1:]))
    test_gp = np.empty((0, *gyro_phone.shape[1:]))
    test_aw = np.empty((0, *accel_watch.shape[1:]))
    test_gw = np.empty((0, *gyro_watch.shape[1:]))
    test_labels = np.empty((0))

    encoded_activity_ids = np.arange(18)
    for activity in encoded_activity_ids:
        matching_indexes = np.where(labels == activity)[0]
        current_ap = accel_phone[matching_indexes]
        current_gp = gyro_phone[matching_indexes]
        current_aw = accel_watch[matching_indexes]
        current_gw = gyro_watch[matching_indexes]
        current_labels = labels[matching_indexes]

        if random:
            # calculate a new random split for every activity label
            # TODO: try seeing what putting the same split for every label does to results
            split = np.random.rand(len(matching_indexes)) < validation_split
            # axis set to 0 to stop it flattening the data when it appends
            train_ap = np.append(train_ap, current_ap[split], axis=0)
            train_gp = np.append(train_gp, current_gp[split], axis=0)
            train_aw = np.append(train_aw, current_aw[split], axis=0)
            train_gw = np.append(train_gw, current_gw[split], axis=0)
            train_labels = np.append(train_labels, current_labels[split])

            test_ap = np.append(test_ap, current_ap[~split], axis=0)
            test_gp = np.append(test_gp, current_gp[~split], axis=0)
            test_aw = np.append(test_aw, current_aw[~split], axis=0)
            test_gw = np.append(test_gw, current_gw[~split], axis=0)
            test_labels = np.append(test_labels, current_labels[~split])
        else:
            split = math.floor(len(matching_indexes) * validation_split)

            train_ap = np.append(train_ap, current_ap[:split], axis=0)
            train_gp = np.append(train_gp, current_gp[:split], axis=0)
            train_aw = np.append(train_aw, current_aw[:split], axis=0)
            train_gw = np.append(train_gw, current_gw[:split], axis=0)
            train_labels = np.append(train_labels, current_labels[:split])

            test_ap = np.append(test_ap, current_ap[split:], axis=0)
            test_gp = np.append(test_gp, current_gp[split:], axis=0)
            test_aw = np.append(test_aw, current_aw[split:], axis=0)
            test_gw = np.append(test_gw, current_gw[split:], axis=0)
            test_labels = np.append(test_labels, current_labels[split:])
    return train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_aw, test_gw, test_labels


def create_windows_and_labels(data, time_steps, step):
    # features are signals on the x, y, and z axes for each accel and gyro of both phone and watch
    N_FEATURES = 12

    windows = []
    labels = []
    for i in range(0, len(data) - time_steps, step):
        xap = data['x-axis_accel_phone'].values[i: i + time_steps]
        yap = data['y-axis_accel_phone'].values[i: i + time_steps]
        zap = data['z-axis_accel_phone'].values[i: i + time_steps]
        xgp = data['x-axis_gyro_phone'].values[i: i + time_steps]
        ygp = data['y-axis_gyro_phone'].values[i: i + time_steps]
        zgp = data['z-axis_gyro_phone'].values[i: i + time_steps]
        xaw = data['x-axis_accel_watch'].values[i: i + time_steps]
        yaw = data['y-axis_accel_watch'].values[i: i + time_steps]
        zaw = data['z-axis_accel_watch'].values[i: i + time_steps]
        xgw = data['x-axis_gyro_watch'].values[i: i + time_steps]
        ygw = data['y-axis_gyro_watch'].values[i: i + time_steps]
        zgw = data['z-axis_gyro_watch'].values[i: i + time_steps]

        # define this segment with the activity that occurs most in this segment
        label = stats.mode(data['ActivityEncoded'][i: i + time_steps])[0][0]
        windows.append([xap, yap, zap, xgp, ygp, zgp, xaw, yaw, zaw, xgw, ygw, zgw])
        labels.append(label)

    # bring windows into a better shape
    reshaped_segments = np.asarray(windows, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def create_sensor_window(data, time_steps, step, sensor_name, device_name):
    N_FEATURES = 3
    windows = []
    for i in range(0, len(data) - time_steps, step):
        x = data['x-axis_' + sensor_name + '_' + device_name].values[i: i + time_steps]
        y = data['y-axis_' + sensor_name + '_' + device_name].values[i: i + time_steps]
        z = data['z-axis_' + sensor_name + '_' + device_name].values[i: i + time_steps]
        windows.append([x, y, z])
    return np.asarray(windows, dtype=np.float32).reshape(-1, time_steps, N_FEATURES)


def create_windows_by_sensor(data, time_steps, step):
    # create windows for accelerometer and gyroscope of both phone and watch
    accel_phone = create_sensor_window(data, time_steps, step, 'accel', 'phone')
    gyro_phone = create_sensor_window(data, time_steps, step, 'gyro', 'phone')
    accel_watch = create_sensor_window(data, time_steps, step, 'accel', 'watch')
    gyro_watch = create_sensor_window(data, time_steps, step, 'gyro', 'watch')

    # create corresponding labels for the above windows
    labels = []
    for i in range(0, len(data) - time_steps, step):
        label = stats.mode(data['ActivityEncoded'][i: i + time_steps])[0][0]
        labels.append(label)
    labels = np.asarray(labels)

    return accel_phone, gyro_phone, accel_watch, gyro_watch, labels


def dct_windows(data, window_length=10):
    """Applies Discrete Cosine Transform to each axis every 1 second (by default) of the given data"""
    # features are signals on the x, y, and z axes for each accel and gyro of both phone and watch
    N_FEATURES = 12
    dct_data = []
    for window in data:
        dct_window = []
        for axis in range(N_FEATURES):
            dct_axis = np.array([])
            for i in range(0, window.shape[1], window_length):
                split_axis = window[axis][i:i + window_length]
                dct_split_axis = np.abs(fftpack.dct(split_axis, norm='ortho'))
                dct_axis = np.append(dct_axis, dct_split_axis)
            dct_window.append(dct_axis)
        dct_data.append(dct_window)
    return np.asarray(dct_data)


def prepare_subject_data(data_path, window_size=80, window_overlap=40, validation_split=0.7, random_split=True,
                         perform_dct=False):
    """Get all data for the given subject, process it and split it into training and testing sets for the model"""
    dataset = read_data(data_path)

    # TODO: think putting a BatchNormalisation layer in the model might be better than this
    # TODO: DO THE NORMALISATION AFTER IT'S SPLIT INTO TRAINING/TESTING - and need to normalise the testing set with whatever scale is made for the training
    dataset['x-axis_accel_phone'] = feature_normalise(dataset['x-axis_accel_phone'])
    dataset['y-axis_accel_phone'] = feature_normalise(dataset['y-axis_accel_phone'])
    dataset['z-axis_accel_phone'] = feature_normalise(dataset['z-axis_accel_phone'])
    dataset['x-axis_gyro_phone'] = feature_normalise(dataset['x-axis_gyro_phone'])
    dataset['y-axis_gyro_phone'] = feature_normalise(dataset['y-axis_gyro_phone'])
    dataset['z-axis_gyro_phone'] = feature_normalise(dataset['z-axis_gyro_phone'])
    dataset['x-axis_accel_watch'] = feature_normalise(dataset['x-axis_accel_watch'])
    dataset['y-axis_accel_watch'] = feature_normalise(dataset['y-axis_accel_watch'])
    dataset['z-axis_accel_watch'] = feature_normalise(dataset['z-axis_accel_watch'])
    dataset['x-axis_gyro_watch'] = feature_normalise(dataset['x-axis_gyro_watch'])
    dataset['y-axis_gyro_watch'] = feature_normalise(dataset['y-axis_gyro_watch'])
    dataset['z-axis_gyro_watch'] = feature_normalise(dataset['z-axis_gyro_watch'])

    # encode the labels into numerical representations so the neural network can work with them
    ENCODED_LABEL = 'ActivityEncoded'
    # use LabelEncoder to convert from String to Integer
    le = preprocessing.LabelEncoder()
    # add the new encoded labels as a column in the dataset
    dataset[ENCODED_LABEL] = le.fit_transform(dataset['activity'].values.ravel())

    # number of steps within one time segment
    # using 8 second window size
    # TIME_PERIODS = 80
    # steps to take from one segment to next - if same as TIME_PERIODS, then no overlap occurs between windows
    # STEP_DISTANCE = 40
    # use window_size and overlap instead of TIME_PERIODS and STEP_DISTANCE so we can change it when calling the function
    windows, labels = create_windows_and_labels(dataset, window_size, window_overlap)

    # if perform_dct:
    # windows_dcted = dct_windows(windows).reshape(-1, window_size, 12)

    windows = windows.reshape(-1, window_size, 12)
    train_x, train_y, test_x, test_y = split_train_test(windows, labels, validation_split, random_split)
    # dct_train_x, dct_train_y, dct_test_x, dct_test_y = split_training_test(windows_dcted, labels, validation_split, random_split)

    # store the following variables to use for constructing the neural network
    # no of time periods within in one record (we've set it to 80 because each data point has an interval of 4 seconds)
    # and there are 3 sensors in this dataset
    num_time_periods, num_sensors = train_x.shape[1], train_x.shape[2]
    # the number of different activities we have - will be used to define the number of output nodes in our network
    num_classes = le.classes_.size

    # convert all data to float32 so TF can read it
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('float32')
    # dct_train_x = dct_train_x.astype('float32')
    # dct_train_y = dct_train_y.astype('float32')
    # dct_test_x = dct_test_x.astype('float32')
    # dct_test_y = dct_test_y.astype('float32')

    # perform one-hot encoding on the labels
    # TODO: try doing this one-hot encoding the way the other tutorial does so i don't have to import keras as well
    train_y_hot = np_utils.to_categorical(train_y, num_classes)
    test_y_hot = np_utils.to_categorical(test_y, num_classes)
    # dct_train_y_hot = np_utils.to_categorical(dct_train_y, num_classes)
    # dct_test_y_hot = np_utils.to_categorical(dct_test_y, num_classes)

    return train_x, train_y_hot, test_x, test_y_hot  # , dct_train_x, dct_train_y, dct_test_x, dct_test_y


def prepare_subject_data_by_sensor(data_path, window_size=80, window_overlap=40, validation_split=0.7,
                                   random_split=True,
                                   perform_dct=False):
    """Get all data for the given subject, process it and split it into training and testing sets for the model"""
    dataset = read_data(data_path)

    # TODO: think putting a BatchNormalisation layer in the model might be better than this
    # TODO: DO THE NORMALISATION AFTER IT'S SPLIT INTO TRAINING/TESTING - and need to normalise the testing set with whatever scale is made for the training
    dataset['x-axis_accel_phone'] = feature_normalise(dataset['x-axis_accel_phone'])
    dataset['y-axis_accel_phone'] = feature_normalise(dataset['y-axis_accel_phone'])
    dataset['z-axis_accel_phone'] = feature_normalise(dataset['z-axis_accel_phone'])
    dataset['x-axis_gyro_phone'] = feature_normalise(dataset['x-axis_gyro_phone'])
    dataset['y-axis_gyro_phone'] = feature_normalise(dataset['y-axis_gyro_phone'])
    dataset['z-axis_gyro_phone'] = feature_normalise(dataset['z-axis_gyro_phone'])
    dataset['x-axis_accel_watch'] = feature_normalise(dataset['x-axis_accel_watch'])
    dataset['y-axis_accel_watch'] = feature_normalise(dataset['y-axis_accel_watch'])
    dataset['z-axis_accel_watch'] = feature_normalise(dataset['z-axis_accel_watch'])
    dataset['x-axis_gyro_watch'] = feature_normalise(dataset['x-axis_gyro_watch'])
    dataset['y-axis_gyro_watch'] = feature_normalise(dataset['y-axis_gyro_watch'])
    dataset['z-axis_gyro_watch'] = feature_normalise(dataset['z-axis_gyro_watch'])

    # encode the labels into numerical representations so the neural network can work with them
    ENCODED_LABEL = 'ActivityEncoded'
    # use LabelEncoder to convert from String to Integer
    le = preprocessing.LabelEncoder()
    # add the new encoded labels as a column in the dataset
    dataset[ENCODED_LABEL] = le.fit_transform(dataset['activity'].values.ravel())

    ##### original method testing ####
    # windows, labels = create_windows_and_labels(dataset, window_size, window_overlap)
    # train_x, train_y, test_x, test_y = split_train_test(windows, labels, validation_split, random_split)
    ##################################

    ap, gp, aw, gw, labels = create_windows_by_sensor(dataset, window_size, window_overlap)
    # print(ap.shape, gp.shape, aw.shape, gw.shape, labels.shape)
    train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_aw, test_gw, test_labels = split_sensors_train_test(
        ap, gp, aw, gw, labels, random=False)

    # store the following variables to use for constructing the neural network
    # no of time periods within in one record (we've set it to 80 because each data point has an interval of 4 seconds)
    # and there are 3 sensors in this dataset
    num_time_periods, num_sensors = train_ap.shape[1:]
    # the number of different activities we have - will be used to define the number of output nodes in our network
    num_classes = le.classes_.size

    # convert all data to float32 so TF can read it
    train_ap = train_ap.astype('float32')
    train_gp = train_gp.astype('float32')
    train_aw = train_aw.astype('float32')
    train_gw = train_gw.astype('float32')
    train_labels = train_labels.astype('float32')
    test_ap = test_ap.astype('float32')
    test_gp = test_gp.astype('float32')
    test_aw = test_aw.astype('float32')
    test_gw = test_gw.astype('float32')
    test_labels = test_labels.astype('float32')

    # perform one-hot encoding on the labels
    # TODO: try doing this one-hot encoding the way the other tutorial does so i don't have to import keras as well
    train_y_hot = np_utils.to_categorical(train_labels, num_classes)
    test_y_hot = np_utils.to_categorical(test_labels, num_classes)

    return train_ap, train_gp, train_aw, train_gw, train_y_hot, test_ap, test_gp, test_aw, test_gw, test_y_hot


def test_data_prep():
    train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_accel_watch, test_gw, test_labels = prepare_subject_data_by_sensor(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', validation_split=0.7,
        random_split=False)
    print("bysensor:",train_ap.shape, train_gp.shape, train_aw.shape, train_gw.shape, train_labels.shape, test_ap.shape, test_gp.shape, test_accel_watch.shape, test_gw.shape, test_labels.shape)

    train_x, train_y, test_x, test_y = prepare_subject_data(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', validation_split=0.7,
        random_split=False)
    print("original:",train_x.shape, train_y.shape, test_x.shape, test_y.shape)


def run_by_subject(iterations_per_subject=1):
    # make different models for different number of features
    # some subjects don't have data for certain classes so need to use different number of features for their model
    cnn_18_classes = models_spec.basic_mlp()
    cnn_17_classes = models_spec.basic_mlp(num_classes=17)
    cnn_16_classes = models_spec.basic_mlp(num_classes=16)
    # TODO: SUBJECT 07 and 09 have no records for activity J, so need to change the model to have 17 features
    #  when training on their data
    # output = ""
    for i in range(51):
        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id

        print("Subject " + subject_id)
        # output += "\n\nSubject" + subject_id
        train_x, train_y, test_x, test_y = prepare_subject_data(
            'wisdm-merged/subject_full_merge/16' + subject_id + '_merged_data.txt', random_split=False)

        if train_y.shape[1] == 18:
            model = cnn_18_classes
        elif train_y.shape[1] == 17:
            model = cnn_17_classes
            # output += "\nSubject has 17 classes"
            print("Subject has 17 classes")
        elif train_y.shape[1] == 16:
            model = cnn_16_classes
            # output += "\nSubject has 16 classes"
            print("Subject has 16 classes")
        else:
            raise ValueError("Subject is missing more features than expected, only has:", train_y.shape[1])

        for run_no in range(1, iterations_per_subject + 1):
            print("Run", run_no)
            # output += "\n\nRun: " + str(run_no)
            # epochs and batch size [1] B. Oluwalade, S. Neela, J. Wawira, T. Adejumo, and S. Purkayastha, “Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data,” pp. 645–650, 2021, doi: 10.5220/0010325906450650.
            trained_model = models_spec.train_model(model, train_x, train_y, batch_size=32, epochs=148, verbose=0)
            accuracy = models_spec.evaluate_model(trained_model, test_x, test_y)
            print("Subject ID " + subject_id + ":", accuracy, "\n")
            # output += "\nSubject ID " + subject_id + " accuracy: " + str(accuracy)

    # f = open("outputs/run" + str(execution_id) + ".txt", "w")
    # f.write(output)
    # f.close()


def run_all_data(iterations=1):
    train_x, train_y, test_x, test_y = prepare_subject_data('wisdm-merged/complete_merge.txt')
    model = models_spec.basic_mlp()
    for run_no in range(1, iterations + 1):
        trained_model = models_spec.train_model(model, train_x, train_y, verbose=1)
        accuracy = models_spec.evaluate_model(trained_model, test_x, test_y)
        print("Accuracy for run #" + str(run_no) + ":", accuracy)


def simple_run():
    # model = models.wijekoon_wiratunga()
    model = models_spec.paper_cnn()
    # model = models_spec.basic_lstm()
    for i in range(51):
        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id

        print("Subject " + subject_id)
        # output += "\n\nSubject" + subject_id
        train_x, train_y, test_x, test_y = prepare_subject_data(
            'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/16' + subject_id + '_merged_data.txt',
            validation_split=0.7,
            random_split=True)

        if train_y.shape[1] != 18:
            print("Subject only has:", train_y.shape[1], "features")
            continue

        # output += "\n\nRun: " + str(run_no)
        # epochs and batch size [1] B. Oluwalade, S. Neela, J. Wawira, T. Adejumo, and S. Purkayastha, “Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data,” pp. 645–650, 2021, doi: 10.5220/0010325906450650.
        trained_model = models_spec.train_model(model, train_x, train_y, test_x, test_y, batch_size=32, epochs=25, verbose=1)
        accuracy = models_spec.evaluate_model(trained_model, test_x, test_y)
        print("Subject ID " + subject_id + ":", accuracy, "\n")


def parallel_run():
    train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_accel_watch, test_gw, test_labels = prepare_subject_data_by_sensor(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', validation_split=0.7,
        random_split=False)

    model = models_spec.parallel_test2(train_gp.shape[1:], 18)
    trained_model = models_spec.train_model_by_sensor(model, train_ap, train_gp, train_aw, train_gw, train_labels, batch_size=32, epochs=25, verbose=1)
    accuracy = models_spec.evaluate_model_by_sensor(model, test_ap, test_gp, test_accel_watch, test_gw, test_labels)
    print(accuracy)
    # train_x, train_y, test_x, test_y = prepare_subject_data(
    #     'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', validation_split=0.7)


def deepSenseRun():
    train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_accel_watch, test_gw, test_labels = prepare_subject_data_by_sensor(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', validation_split=0.7,
        random_split=False)

    model = myDeepSense.uDeepSense(train_gp.shape[1:], 18)
    trained_model = myDeepSense.train(model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                                      batch_size=32, epochs=25, verbose=1)
    accuracy = myDeepSense.evaluate(model, test_ap, test_gp, test_accel_watch, test_gw, test_labels)
    print(accuracy)
    # train_x, train_y, test_x, test_y = prepare_subject_data(
    #     'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', validation_split=0.7)



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# np.set_printoptions(threshold=np.inf)


# run_by_subject()
# run_all_data()
# simple_run()
# parallel_run()
# test_data_prep()
deepSenseRun()
