import numpy as np
from keras.utils import np_utils
from scipy import stats
from sklearn import preprocessing
import math
import pandas as pd


def read_data(file_path):
    """Read dataset from the path given, remove NA values, and return it as a Pandas DataFrame"""
    data = pd.read_csv(file_path)
    data.dropna(axis=0, how='any', inplace=True)
    return data


def feature_normalise(data_column):
    """Normalise given column from a dataset by setting mean to 0 and setting a unit standard deviation"""
    mu = np.mean(data_column, axis=0)
    sigma = np.std(data_column, axis=0)
    return (data_column - mu) / sigma


def create_windows_and_labels(data, window_size, step):
    """Segment dataset into windows with corresponding target labels for each window

        Parameters:
            data (pandas DataFrame): the dataset to segment
            window_size (int): how many data records each window should contain
            step (int): define how much overlap there should be between windows. If step==window_size, then no overlap

        Returns:
            reshaped_segments (NumPy array): an array containing each window created from the dataset
            labels (NumPy array): array of labels corresponding to each window
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


def create_sensor_windows_and_labels(data, window_size, step, sensor_name, device_name):
    """Create windows and labels for the given sensor"""
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
    accel_phone = create_sensor_windows_and_labels(data, window_size, step, 'accel', 'phone')
    gyro_phone = create_sensor_windows_and_labels(data, window_size, step, 'gyro', 'phone')
    accel_watch = create_sensor_windows_and_labels(data, window_size, step, 'accel', 'watch')
    gyro_watch = create_sensor_windows_and_labels(data, window_size, step, 'gyro', 'watch')

    # create corresponding labels for the above windows
    labels = []
    for i in range(0, len(data) - window_size, step):
        label = stats.mode(data['ActivityEncoded'][i: i + window_size])[0][0]
        labels.append(label)
    labels = np.asarray(labels)

    return accel_phone, gyro_phone, accel_watch, gyro_watch, labels


def split_train_test(data, labels, split_ratio, random):
    """Split the dataset into training and testing data. Splits the dataset so there is an appropriate amount of records
        for EACH activity in the training and test sets - thus avoiding problem of different activities being present in
        disproportionate amounts relative to each other in training and testing set

        Parameters:
            data (NumPy array): the windowed version of the dataset
            labels (NumPy array): labels corresponding to the windows
            split_ratio (float): the fraction of data that should be set as training
            random (bool): whether the activities should be randomly selected for each set or done in order
    """
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
    """Split the dataset into training, testing, AND validation data"""
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


def split_train_test_by_sensor(accel_phone, gyro_phone, accel_watch, gyro_watch, labels, split_ratio,
                               random):
    """Same as the other split_train_test but does it separately for each sensor rather than all as one"""
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


def split_train_test_val_by_sensor(accel_phone, gyro_phone, accel_watch, gyro_watch, labels, split_ratio, random):
    """Same as split_train_test_val but does it for each sensor separately"""
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
    val_ap = np.empty((0, *accel_phone.shape[1:]))
    val_gp = np.empty((0, *gyro_phone.shape[1:]))
    val_aw = np.empty((0, *accel_watch.shape[1:]))
    val_gw = np.empty((0, *gyro_watch.shape[1:]))
    val_labels = np.empty((0))

    encoded_activity_ids = np.arange(18)
    # find segments corresponding to each activity label and split them according to the given ratio
    for a_id in encoded_activity_ids:
        matching_indexes = np.where(labels == a_id)[0]
        current_ap = accel_phone[matching_indexes]
        current_gp = gyro_phone[matching_indexes]
        current_aw = accel_watch[matching_indexes]
        current_gw = gyro_watch[matching_indexes]
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

        train_ap = np.append(train_ap, current_ap[:train_split], axis=0)
        train_gp = np.append(train_gp, current_gp[:train_split], axis=0)
        train_aw = np.append(train_aw, current_aw[:train_split], axis=0)
        train_gw = np.append(train_gw, current_gw[:train_split], axis=0)
        train_labels = np.append(train_labels, current_labels[:train_split])

        test_ap = np.append(test_ap, current_ap[train_split:test_split], axis=0)
        test_gp = np.append(test_gp, current_gp[train_split:test_split], axis=0)
        test_aw = np.append(test_aw, current_aw[train_split:test_split], axis=0)
        test_gw = np.append(test_gw, current_gw[train_split:test_split], axis=0)
        test_labels = np.append(test_labels, current_labels[train_split:test_split])

        val_ap = np.append(val_ap, current_ap[test_split:], axis=0)
        val_gp = np.append(val_gp, current_gp[test_split:], axis=0)
        val_aw = np.append(val_aw, current_aw[test_split:], axis=0)
        val_gw = np.append(val_gw, current_gw[test_split:], axis=0)
        val_labels = np.append(val_labels, current_labels[test_split:])

    # put data in dicts so it's neater
    train_data = {'ap': train_ap, 'gp': train_gp, 'aw': train_aw, 'gw': train_gw, 'labels': train_labels}
    test_data = {'ap': test_ap, 'gp': test_gp, 'aw': test_aw, 'gw': test_gw, 'labels': test_labels}
    val_data = {'ap': val_ap, 'gp': val_gp, 'aw': val_aw, 'gw': val_gw, 'labels': val_labels}
    return train_data, test_data, val_data


def normalise_and_encode_activities(data, label_encoder=None):
    """Normalise each column of sensor data and encode the activities with numerical labels

        Parameter:
            data (Pandas DataFrame): the dataset to normalise
            label_encoder (scikit-learn LabelEncoder): pass in a pre-fit LabelEncoder to transform this data's labels

        Returns:
            data (Pandas DataFrame): the normalised dataset
            label_encoder: if no LabelEncoder is initially passed in, returns the new one fit on this data, else
                            returns the same one passed in. Is later used to convert activity labels back
                            when evaluating performance
    """
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
    if label_encoder is None:
        label_encoder = preprocessing.LabelEncoder()
        # add the new encoded labels as a column in the dataset
        data[ENCODED_LABEL] = label_encoder.fit_transform(data['activity'].values.ravel())
    else:
        data[ENCODED_LABEL] = label_encoder.transform(data['activity'].values.ravel())
    return data, label_encoder


def preprocess_subject(data_path, window_size=80, window_step=80, label_encoder=None):
    """Get all data for the given subject, process it (without splitting into training and testing)
            
        Parameters:
            data_path (str): path to the file containing the subject's data
            window_size (int): the size of the windows to split the data into
            window_step (int): defines how much overlap should be in the windows,
                                    window_step==window_size means no overlap

        Returns:
            x (NumPy array): the preprocessed sensor data
            y_hot (NumPy array): one-hot encoded target labels for x
            label_encoder (scikit-learn LabelEncoder): the encoder used to transform this data's activity labels
    """
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset, label_encoder)

    # number of steps within one time segment
    # using 8 second window size
    # TIME_PERIODS = 80
    # steps to take from one segment to next - if same as TIME_PERIODS, then no overlap occurs between windows
    # STEP_DISTANCE = 40
    windows, labels = create_windows_and_labels(dataset, window_size, window_step)

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


def preprocess_subject_train_test(data_path, window_size=80, window_step=80, split=0.7,
                                  random_split=True):
    """Get all data for the given subject, process it and split it into training and testing sets for the model

        Parameters:
            split (float): the fraction of data to be split into training/testing data
            random_split (bool): whether train/test split should be random or not
    """
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset)

    windows, labels = create_windows_and_labels(dataset, window_size, window_step)

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


def preprocess_subject_train_test_val(data_path, window_size=80, window_step=80, split=(0.5, 0.3, 0.2),
                                      random_split=True):
    """Get all data for the given subject, process it and split it into training, testing, and validation
        sets for the model"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset)

    windows, labels = create_windows_and_labels(dataset, window_size, window_step)

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


def preprocess_subject_by_sensor(data_path, window_size=80, window_step=80, label_encoder=None):
    """Same as prepare_subject_data but splits the features from each sensor, giving the x,y,z axes of each
    accelerometer and gyroscope from both phone and watch in separate arrays"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset, label_encoder)

    ap, gp, aw, gw, labels = create_windows_by_sensor(dataset, window_size, window_step)

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


def preprocess_subject_by_sensor_train_test(data_path, window_size=80, window_step=80, split_ratio=0.7,
                                            random_split=True, label_encoder=None):
    """Same as preprocess_subject_by_sensor but splits the data into training and testing sets"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset, label_encoder)

    ap, gp, aw, gw, labels = create_windows_by_sensor(dataset, window_size, window_step)
    train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_aw, test_gw, test_labels = split_train_test_by_sensor(
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


def preprocess_subject_by_sensor_train_test_val(data_path, window_size=80, window_step=80, split_ratio=(0.7, 0.2, 0.1),
                                                random_split=True, label_encoder=None):
    """Same as prepare_subject_data_by_sensor but splits data into training, testing, and validation"""
    dataset = read_data(data_path)

    dataset, label_encoder = normalise_and_encode_activities(dataset, label_encoder)

    ap, gp, aw, gw, labels = create_windows_by_sensor(dataset, window_size, window_step)
    train_data, test_data, val_data = split_train_test_val_by_sensor(ap, gp, aw, gw, labels, split_ratio, random_split)

    # convert all data to float32 so TF can read it
    train_ap = train_data['ap'].astype('float32')
    train_gp = train_data['gp'].astype('float32')
    train_aw = train_data['aw'].astype('float32')
    train_gw = train_data['gw'].astype('float32')
    train_labels = train_data['labels'].astype('float32')
    test_ap = test_data['ap'].astype('float32')
    test_gp = test_data['gp'].astype('float32')
    test_aw = test_data['aw'].astype('float32')
    test_gw = test_data['gw'].astype('float32')
    test_labels = test_data['labels'].astype('float32')
    val_ap = val_data['ap'].astype('float32')
    val_gp = val_data['gp'].astype('float32')
    val_aw = val_data['aw'].astype('float32')
    val_gw = val_data['gw'].astype('float32')
    val_labels = val_data['labels'].astype('float32')

    num_classes = label_encoder.classes_.size
    # perform one-hot encoding on the labels
    train_y_hot = np_utils.to_categorical(train_labels, num_classes)
    test_y_hot = np_utils.to_categorical(test_labels, num_classes)
    val_y_hot = np_utils.to_categorical(val_labels, num_classes)

    # put the data into dictionaries to make everything neater
    train_data = {'ap': train_ap, 'gp': train_gp, 'aw': train_aw, 'gw': train_gw, 'labels': train_y_hot}
    test_data = {'ap': test_ap, 'gp': test_gp, 'aw': test_aw, 'gw': test_gw, 'labels': test_y_hot}
    val_data = {'ap': val_ap, 'gp': val_gp, 'aw': val_aw, 'gw': val_gw, 'labels': val_y_hot}
    return train_data, test_data, val_data, label_encoder


def leave_one_out_cv(data_path, left_out, window_size=80, window_step=80):
    """Preprocess data from all subjects except one for Leave-One-Out Cross-Validation

        Parameters
            data_path (str): path to the directory where all subject data is stored
            left_out (str): the ID of the subject to be left out
            window_size (int): how many records to be in each window
            window_step (int): how much overlap between each window. window_step==window_size means no overlap

        Returns:
            x (numpy array): data segmented into windows to be put in the model
            y (numpy array): labels corresponding to each window
            label_encoder (scikit-learn LabelEncoder): the object used to encode the activity labels
    """
    # Get data for the subject to be left out
    loocv_file_path = data_path + '/' + left_out + '_merged_data.txt'
    loocv_train_x, loocv_train_y, loocv_test_x, loocv_test_y, label_encoder = preprocess_subject_train_test(
        loocv_file_path,
        window_size,
        window_step)

    x = np.empty((0, window_size, 12))
    y = np.empty((0, 18))

    for i in range(51):
        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id
        subject_id = '16' + subject_id

        # skip the subject chosen to be left out
        if subject_id == left_out:
            continue

        print('Preprocess Subject ' + subject_id)

        file_path = data_path + '/' + subject_id + '_merged_data.txt'
        subject_x, subject_y, subject_le = preprocess_subject(file_path, window_size, window_step, label_encoder)

        print(subject_x.shape, subject_y.shape)
        x = np.append(x, subject_x, axis=0)
        y = np.append(y, subject_y, axis=0)

        print(x.shape, y.shape)
    return x, y, label_encoder


def leave_one_out_cv_train_test(data_path, left_out, window_size=80, window_step=80, split_ratio=0.7,
                                random_split=True):
    """Like leave_one_out_cv but splits data into training and testing

        Parameters
            data_path (str): path to the directory where all subject data is stored
            left_out (str): the ID of the subject to be left out
            window_size (int): how many records to be in each window
            window_step (int): how much overlap between each window. window_step==window_size means no overlap
            split_ratio (float): the ratio of training to testing data
            random_split (bool): whether the train:test data should be split randomly or in order that they come in time

        Returns:
            train_x (numpy array): data segmented into windows for training of the model
            train_y (numpy array): labels corresponding to each window for training of the model
            test_x (numpy array): data segmented into windows for evaluation of the model
            test_y (numpy array): labels corresponding to each window for evaluation of the model
            le (scikit-learn LabelEncoder): the object used to encode the activity labels
    """
    # Get data for the subject to be left out
    loocv_file_path = data_path + '/' + left_out + '_merged_data.txt'
    loocv_train_x, loocv_train_y, loocv_test_x, loocv_test_y, le = preprocess_subject_train_test(loocv_file_path,
                                                                                                 window_size,
                                                                                                 window_step,
                                                                                                 split_ratio,
                                                                                                 random_split)

    train_x = np.empty((0, window_size, 12))
    train_y = np.empty((0, 18))
    test_x = np.empty((0, window_size, 12))
    test_y = np.empty((0, 18))

    for i in range(51):

        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id
        subject_id = '16' + subject_id
        # skip the subject chosen to be left out
        if subject_id == left_out:
            continue

        print('Preprocess Subject ' + subject_id)

        file_path = data_path + '/' + subject_id + '_merged_data.txt'
        subject_train_x, subject_train_y, subject_test_x, subject_test_y, le = preprocess_subject_train_test(file_path,
                                                                                                             window_size,
                                                                                                             window_step,
                                                                                                             split_ratio,
                                                                                                             random_split,
                                                                                                             le)

        print(subject_train_x.shape, subject_train_y.shape, subject_test_x.shape, subject_test_y.shape)
        train_x = np.append(train_x, subject_train_x, axis=0)
        train_y = np.append(train_y, subject_train_y, axis=0)
        test_x = np.append(test_x, subject_test_x, axis=0)
        test_y = np.append(test_y, subject_test_y, axis=0)
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y, le


def leave_one_out_cv_by_sensor(data_path, left_out, window_size=80, window_step=80):
    """Preprocess data from all subjects except one for Leave-One-Out Cross-Validation. Uses what we have named
        the 'by sensor' format for input into a model with multiple inputs using TensorFlow's Functional API

        Parameters
            data_path (str): path to the directory where all subject data is stored
            left_out (str): the ID of the subject to be left out
            window_size (int): how many records to be in each window
            window_step (int): how much overlap between each window. window_step==window_size means no overlap
            split_ratio (float): the ratio of training to testing data
            random_split (bool): whether the train:test data should be split randomly or in order that they come in time

        Returns:
            ap (NumPy array): the preprocessed accelerometer training data from the phone
            gp (NumPy array): the preprocessed gyroscope training data from the phone
            aw (NumPy array): the preprocessed accelerometer training data from the watch
            aw (NumPy array): the preprocessed gyroscope training data from the watch
            labels (NumPy array): the ground truth labels corresponding to the windows of the above data
            label_encoder (scikit-learn LabelEncoder): the object used to encode the activity labels

    """
    # Get data for the subject to be left out
    loocv_file_path = data_path + '/' + left_out + '_merged_data.txt'
    loocv_ap, loocv_gp, loocv_aw, loocv_gw, loocv_labels, label_encoder = preprocess_subject_by_sensor(loocv_file_path,
                                                                                                       window_size,
                                                                                                       window_step)

    ap = np.empty((0, window_size, 3))
    gp = np.empty((0, window_size, 3))
    aw = np.empty((0, window_size, 3))
    gw = np.empty((0, window_size, 3))
    labels = np.empty((0, 18))

    for i in range(51):

        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id
        subject_id = '16' + subject_id

        # skip the subject chosen to be left out
        if subject_id == left_out:
            print('Skipping left-out subject ' + subject_id)
            continue

        print('Preprocess Subject ' + subject_id)

        file_path = data_path + '/' + subject_id + '_merged_data.txt'
        subject_ap, subject_gp, subject_aw, subject_gw, subject_labels, subject_le = preprocess_subject_by_sensor(
            file_path,
            window_size,
            window_step,
            label_encoder)

        ap = np.append(ap, subject_ap, axis=0)
        gp = np.append(gp, subject_gp, axis=0)
        aw = np.append(aw, subject_aw, axis=0)
        gw = np.append(gw, subject_gw, axis=0)
        labels = np.append(labels, subject_labels, axis=0)

    return ap, gp, aw, gw, labels, label_encoder


def leave_one_out_cv_by_sensor_train_test(data_path, left_out, window_size=80, window_step=80, split_ratio=0.7,
                                          random_split=True):
    """same as leave_one_out_cv_by_sensor but splits data into training and testing

        Parameters
            data_path (str): path to the directory where all subject data is stored
            left_out (str): the ID of the subject to be left out
            window_size (int): how many records to be in each window
            window_step (int): how much overlap between each window. window_step==window_size means no overlap
            split_ratio (float): the ratio of training to testing data
            random_split (bool): whether the train:test data should be split randomly or in order that they come in time

        Returns:
            train_data (dict): a dictionary containing data for training with the following indices:
                                    'ap' the preprocessed accelerometer training data from the phone
                                    'gp' the preprocessed gyroscope training data from the phone
                                    'aw' the preprocessed accelerometer training data from the watch
                                    'aw' the preprocessed gyroscope training data from the watch
                                    'labels' the ground truth labels corresponding to the windows of the above data
            test_data (dict): a dictionary containing the data to be used for evaluation of the model. Has same format
                                as train_data
            le (scikit-learn LabelEncoder): the object used to encode the activity labels

    """
    # Get data for the subject to be left out
    loocv_file_path = data_path + '/' + left_out + '_merged_data.txt'
    loocv_train_data, loocv_test_data, le = preprocess_subject_by_sensor_train_test(loocv_file_path,
                                                                                    window_size,
                                                                                    window_step,
                                                                                    split_ratio,
                                                                                    random_split)

    train_ap = np.empty((0, window_size, 3))
    train_gp = np.empty((0, window_size, 3))
    train_aw = np.empty((0, window_size, 3))
    train_gw = np.empty((0, window_size, 3))
    train_labels = np.empty((0, 18))
    test_ap = np.empty((0, window_size, 3))
    test_gp = np.empty((0, window_size, 3))
    test_aw = np.empty((0, window_size, 3))
    test_gw = np.empty((0, window_size, 3))
    test_labels = np.empty((0, 18))

    for i in range(51):
        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id
        subject_id = '16' + subject_id

        # skip the subject chosen to be left out
        if subject_id == left_out:
            print('Skipping left-out subject ' + subject_id)
            continue

        print('Preprocess Subject ' + subject_id)

        file_path = data_path + '/' + subject_id + '_merged_data.txt'
        subject_train, subject_test, le = preprocess_subject_by_sensor_train_test(file_path, window_size,
                                                                                  window_step, split_ratio,
                                                                                  random_split, le)

        train_ap = np.append(train_ap, subject_train['ap'], axis=0)
        train_gp = np.append(train_gp, subject_train['gp'], axis=0)
        train_aw = np.append(train_aw, subject_train['aw'], axis=0)
        train_gw = np.append(train_gw, subject_train['gw'], axis=0)
        train_labels = np.append(train_labels, subject_train['labels'], axis=0)
        test_ap = np.append(test_ap, subject_test['ap'], axis=0)
        test_gp = np.append(test_gp, subject_test['gp'], axis=0)
        test_aw = np.append(test_aw, subject_test['aw'], axis=0)
        test_gw = np.append(test_gw, subject_test['gw'], axis=0)
        test_labels = np.append(test_labels, subject_test['labels'], axis=0)

    train_data = {'ap': train_ap, 'gp': train_gp, 'aw': train_aw, 'gw': train_gw, 'labels': train_labels}
    test_data = {'ap': test_ap, 'gp': test_gp, 'aw': test_aw, 'gw': test_gw, 'labels': test_labels}
    return train_data, test_data, le
