import numpy as np
from keras.utils import np_utils
from scipy import stats
from sklearn import preprocessing
import math
import pandas as pd


def read_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    return data


def feature_normalise(dataset):
    """Normalise given dataset by setting mean to 0 and setting a unit standard deviation"""
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def create_windows_and_labels(data, window_size, step):
    """Segment dataset into windows with corresponding target labels for each window
        :parameter data: the dataset to segment
        :parameter window_size: how many data records each window should contain
        :parameter step: define how much overlap there should be between windows. If step==window_size, then no overlap
    """
    # features are signals on the x, y, and z axes for each accel and gyro of both phone and watch
    N_FEATURES = 12

    windows = []
    labels = []
    for i in range(0, len(data) - window_size, step):
        xap = data['x-axis_accel_phone'].values[i: i + window_size]
        yap = data['y-axis_accel_phone'].values[i: i + window_size]
        zap = data['z-axis_accel_phone'].values[i: i + window_size]
        xgp = data['x-axis_gyro_phone'].values[i: i + window_size]
        ygp = data['y-axis_gyro_phone'].values[i: i + window_size]
        zgp = data['z-axis_gyro_phone'].values[i: i + window_size]
        xaw = data['x-axis_accel_watch'].values[i: i + window_size]
        yaw = data['y-axis_accel_watch'].values[i: i + window_size]
        zaw = data['z-axis_accel_watch'].values[i: i + window_size]
        xgw = data['x-axis_gyro_watch'].values[i: i + window_size]
        ygw = data['y-axis_gyro_watch'].values[i: i + window_size]
        zgw = data['z-axis_gyro_watch'].values[i: i + window_size]

        # define this segment with the activity that occurs most in this segment
        label = stats.mode(data['ActivityEncoded'][i: i + window_size])[0][0]
        windows.append([xap, yap, zap, xgp, ygp, zgp, xaw, yaw, zaw, xgw, ygw, zgw])
        labels.append(label)

    # bring windows into a better shape
    reshaped_segments = np.asarray(windows, dtype=np.float32).reshape(-1, window_size, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def create_sensor_window(data, window_size, step, sensor_name, device_name):
    """Create window for the given sensor"""
    N_FEATURES = 3
    windows = []
    for i in range(0, len(data) - window_size, step):
        x = data['x-axis_' + sensor_name + '_' + device_name].values[i: i + window_size]
        y = data['y-axis_' + sensor_name + '_' + device_name].values[i: i + window_size]
        z = data['z-axis_' + sensor_name + '_' + device_name].values[i: i + window_size]
        windows.append([x, y, z])
    return np.asarray(windows, dtype=np.float32).reshape(-1, window_size, N_FEATURES)


def create_windows_by_sensor(data, window_size, step):
    """Create windows and labels for each accelerometer and gyroscope of both the phone and watch"""
    accel_phone = create_sensor_window(data, window_size, step, 'accel', 'phone')
    gyro_phone = create_sensor_window(data, window_size, step, 'gyro', 'phone')
    accel_watch = create_sensor_window(data, window_size, step, 'accel', 'watch')
    gyro_watch = create_sensor_window(data, window_size, step, 'gyro', 'watch')

    # create corresponding labels for the above windows
    labels = []
    for i in range(0, len(data) - window_size, step):
        label = stats.mode(data['ActivityEncoded'][i: i + window_size])[0][0]
        labels.append(label)
    labels = np.asarray(labels)

    return accel_phone, gyro_phone, accel_watch, gyro_watch, labels


# split dataset so there is an appropriate amount of records for EACH activity in the training and test sets
# avoids problem of e.g. some activities having 5 records in training and only 1 in test while others have 2 in training
# and 4 in test
def split_train_test(data, labels, split_ratio, random):
    """Split the dataset into training and testing data"""
    encoded_activity_ids = np.arange(18)
    # train_data = np.empty((0, data.shape[1], data.shape[2]))
    train_data = np.empty((0, *data.shape[1:]))
    train_labels = np.empty((0))
    test_data = np.empty((0, *data.shape[1:]))
    test_labels = np.empty((0))

    # find segments corresponding to each activity label and split them according to the given ratio
    for a_id in encoded_activity_ids:
        matching_indexes = np.where(labels == a_id)[0]
        current_data = data[matching_indexes]
        current_labels = labels[matching_indexes]

        if random:
            # calculate a new random split for every activity label
            split = np.random.rand(len(matching_indexes)) < split_ratio
            # axis set to 0 to stop it flattening the data when it appends
            train_data = np.append(train_data, current_data[split], axis=0)
            train_labels = np.append(train_labels, current_labels[split])
            test_data = np.append(test_data, current_data[~split], axis=0)
            test_labels = np.append(test_labels, current_labels[~split])
        else:
            split = math.floor(len(matching_indexes) * split_ratio)
            train_data = np.append(train_data, current_data[:split], axis=0)
            train_labels = np.append(train_labels, current_labels[:split])
            test_data = np.append(test_data, current_data[split:], axis=0)
            test_labels = np.append(test_labels, current_labels[split:])

    return train_data, train_labels, test_data, test_labels


def split_train_test_val(data, labels, split_ratio, random):
    train_data = np.empty((0, *data.shape[1:]))
    train_labels = np.empty((0))
    test_data = np.empty((0, *data.shape[1:]))
    test_labels = np.empty((0))
    val_data = np.empty((0, *data.shape[1:]))
    val_labels = np.empty((0))

    encoded_activity_ids = np.arange(18)
    # find segments corresponding to each activity label and split them according to the given ratio
    for a_id in encoded_activity_ids:
        matching_indexes = np.where(labels == a_id)[0]
        current_data = data[matching_indexes]
        current_labels = labels[matching_indexes]

        # if random:
        #     # randomly select windows rather than in order
        #     # calculate a new random split for every activity label
        #     split = np.random.rand(len(matching_indexes)) < split_ratio
        #     # axis set to 0 to stop it flattening the data when it appends
        #     train_data = np.append(train_data, current_data[split], axis=0)
        #     train_labels = np.append(train_labels, current_labels[split])
        #     test_data = np.append(test_data, current_data[~split], axis=0)
        #     test_labels = np.append(test_labels, current_labels[~split])
        # else:
        train_split = math.floor(len(matching_indexes) * split_ratio[0])
        test_split = train_split + math.floor(len(matching_indexes) * split_ratio[1])

        train_data = np.append(train_data, current_data[:train_split], axis=0)
        train_labels = np.append(train_labels, current_labels[:train_split])
        test_data = np.append(test_data, current_data[train_split:test_split], axis=0)
        test_labels = np.append(test_labels, current_labels[train_split:test_split])
        val_data = np.append(val_data, current_data[test_split:], axis=0)
        val_labels = np.append(val_labels, current_labels[test_split:])

    return train_data, train_labels, test_data, test_labels, val_data, val_labels


def split_by_sensor_train_test(accel_phone, gyro_phone, accel_watch, gyro_watch, labels, split_ratio,
                               random):
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
            split = np.random.rand(len(matching_indexes)) < split_ratio
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
            split = math.floor(len(matching_indexes) * split_ratio)

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


def normalise_and_encode_activities(data):
    data['x-axis_accel_phone'] = feature_normalise(data['x-axis_accel_phone'])
    data['y-axis_accel_phone'] = feature_normalise(data['y-axis_accel_phone'])
    data['z-axis_accel_phone'] = feature_normalise(data['z-axis_accel_phone'])
    data['x-axis_gyro_phone'] = feature_normalise(data['x-axis_gyro_phone'])
    data['y-axis_gyro_phone'] = feature_normalise(data['y-axis_gyro_phone'])
    data['z-axis_gyro_phone'] = feature_normalise(data['z-axis_gyro_phone'])
    data['x-axis_accel_watch'] = feature_normalise(data['x-axis_accel_watch'])
    data['y-axis_accel_watch'] = feature_normalise(data['y-axis_accel_watch'])
    data['z-axis_accel_watch'] = feature_normalise(data['z-axis_accel_watch'])
    data['x-axis_gyro_watch'] = feature_normalise(data['x-axis_gyro_watch'])
    data['y-axis_gyro_watch'] = feature_normalise(data['y-axis_gyro_watch'])
    data['z-axis_gyro_watch'] = feature_normalise(data['z-axis_gyro_watch'])

    # encode the labels into numerical representations so the neural network can work with them
    ENCODED_LABEL = 'ActivityEncoded'
    # use LabelEncoder to convert from String to Integer
    le = preprocessing.LabelEncoder()
    # add the new encoded labels as a column in the dataset
    data[ENCODED_LABEL] = le.fit_transform(data['activity'].values.ravel())
    return data, le


def preprocess_subject_data(data_path, window_size=80, window_overlap=40):
    """Get all data for the given subject, process it and split it into training and testing sets for the model"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset)

    # number of steps within one time segment
    # using 8 second window size
    # TIME_PERIODS = 80
    # steps to take from one segment to next - if same as TIME_PERIODS, then no overlap occurs between windows
    # STEP_DISTANCE = 40
    windows, labels = create_windows_and_labels(dataset, window_size, window_overlap)

    # convert all data to float32 so TF can read it
    x = windows.astype('float32')
    y = labels.astype('float32')
    # dct_train_x = dct_train_x.astype('float32')
    # dct_train_y = dct_train_y.astype('float32')
    # dct_test_x = dct_test_x.astype('float32')
    # dct_test_y = dct_test_y.astype('float32')
    num_classes = label_encoder.classes_.size
    # perform one-hot encoding on the labels
    y_hot = np_utils.to_categorical(y, num_classes)

    return x, y_hot, label_encoder


def preprocess_subject_data_train_test(data_path, window_size=80, window_overlap=40, split=0.7,
                                       random_split=True):
    """Get all data for the given subject, process it and split it into training and testing sets for the model"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset)

    windows, labels = create_windows_and_labels(dataset, window_size, window_overlap)

    train_x, train_y, test_x, test_y = split_train_test(windows, labels, split, random_split)

    # convert all data to float32 so TF can read it
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('float32')

    num_classes = label_encoder.classes_.size
    # perform one-hot encoding on the labels
    train_y_hot = np_utils.to_categorical(train_y, num_classes)
    test_y_hot = np_utils.to_categorical(test_y, num_classes)

    return train_x, train_y_hot, test_x, test_y_hot, label_encoder


def preprocess_subject_data_train_test_val(data_path, window_size=80, window_overlap=40, split=(0.5, 0.3, 0.2),
                                           random_split=True):
    """Get all data for the given subject, process it and split it into training, testing, and validation
        sets for the model"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset)

    windows, labels = create_windows_and_labels(dataset, window_size, window_overlap)

    windows = windows.reshape(-1, window_size, 12)
    train_x, train_y, test_x, test_y, val_x, val_y = split_train_test_val(windows, labels, split, random_split)

    # convert all data to float32 so TF can read it
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')
    test_x = test_x.astype('float32')
    test_y = test_y.astype('float32')
    val_x = val_x.astype('float32')
    val_y = val_y.astype('float32')

    num_classes = label_encoder.classes_.size
    # perform one-hot encoding on the labels
    train_y_hot = np_utils.to_categorical(train_y, num_classes)
    test_y_hot = np_utils.to_categorical(test_y, num_classes)
    val_y_hot = np_utils.to_categorical(val_y, num_classes)

    return train_x, train_y_hot, test_x, test_y_hot, val_x, val_y_hot, label_encoder


def preprocess_subject_data_by_sensor(data_path, window_size=80, window_overlap=40):
    """Same as prepare_subject_data but splits the features from each sensor, giving the x,y,z axes of each
    accelerometer and gyroscope from both phone and watch in separate arrays"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset)

    ap, gp, aw, gw, labels = create_windows_by_sensor(dataset, window_size, window_overlap)

    # convert all data to float32 so TF can read it
    ap = ap.astype('float32')
    gp = gp.astype('float32')
    aw = aw.astype('float32')
    gw = gw.astype('float32')
    labels = labels.astype('float32')

    num_classes = label_encoder.classes_.size
    # perform one-hot encoding on the labels
    y_hot = np_utils.to_categorical(labels, num_classes)

    return ap, gp, aw, gw, y_hot, label_encoder


def preprocess_subject_data_by_sensor_train_test(data_path, window_size=80, window_overlap=40, split_ratio=0.7,
                                                 random_split=True):
    """Same as prepare_subject_data but splits the features from each sensor, giving the x,y,z axes of each
    accelerometer and gyroscope from both phone and watch in separate arrays"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset)

    ap, gp, aw, gw, labels = create_windows_by_sensor(dataset, window_size, window_overlap)
    train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_aw, test_gw, test_labels = split_by_sensor_train_test(
        ap, gp, aw, gw, labels, split_ratio, random_split)

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

    num_classes = label_encoder.classes_.size
    # perform one-hot encoding on the labels
    train_y_hot = np_utils.to_categorical(train_labels, num_classes)
    test_y_hot = np_utils.to_categorical(test_labels, num_classes)

    # put the data into dictionaries to make everything neater
    train_data = {'ap': train_ap, 'gp': train_gp, 'aw': train_aw, 'gw': train_gw, 'labels': train_y_hot}
    test_data = {'ap': test_ap, 'gp': test_gp, 'aw': test_aw, 'gw': test_gw, 'labels': test_y_hot}
    return train_data, test_data, label_encoder
    # return train_ap, train_gp, train_aw, train_gw, train_y_hot, test_ap, test_gp, test_aw, test_gw, test_y_hot, label_encoder


def leave_one_out_cv_by_sensor(data_path, left_out, window_size=80, window_overlap=40, split_ratio=0.7,
                                                 random_split=True):
    """Preprocess data from all subjects except one
        Parameters
            data_path (str): path to the directory where all subject data is stored
            left_out (int): the ID of the subject to be left out
            window_size (int): how many records to be in each window
            window_overlap (int): how much overlap between each window. window_overlap==window_size means no overlap
            split_ratio (float): the ratio of training to testing data
            random_split (bool): whether the train:test data should be split randomly or in order that they come in time

        Returns:

    """
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for i in range(51):
        # skip the subject chosen to be left out
        if i == left_out:
            continue

        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id
        file_path = data_path + '/16' + subject_id + '_merged_data.txt'
        subject_data = preprocess_subject_data_by_sensor_train_test(file_path, window_size, window_overlap, split_ratio,
                                                                    random_split)

