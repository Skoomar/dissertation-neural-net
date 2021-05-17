import pandas as pd
import numpy as np

# set pandas to display every column whenever a DataFrame is printed out
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def convert_to_float(x):
    try:
        return np.float(x)
    except:
        return np.nan


def read_data(file_path):
    """Reads in the CSV data from the given file, removes any missing values and returns a Pandas DataFrame"""
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


def manual_merge_accel_gyro_data(accel_subset, gyro_subset, device):
    """Merge data by directly matching up the rows of accel and gyro data in order
        Only used when none of the timestamps of accel and gyro data matches up
    """
    accel_subset.rename(columns={'x-axis': 'x-axis_accel_' + device,
                                 'y-axis': 'y-axis_accel_' + device,
                                 'z-axis': 'z-axis_accel_' + device
                                 },
                        inplace=True)
    accel_subset['x-axis_gyro_' + device] = gyro_subset['x-axis']
    accel_subset['y-axis_gyro_' + device] = gyro_subset['y-axis']
    accel_subset['z-axis_gyro_' + device] = gyro_subset['z-axis']

    accel_subset = accel_subset.dropna(axis=0, how='any').reset_index(drop=True)
    return accel_subset


def interpolate_subset(merged_data, accel_subset, gyro_subset, device, target_size):
    """If timestamps of accel_row and gyro data are only partially
        matched up then use this to fill in remaining values"""
    gyro_index = 0
    for accel_index, accel_row in accel_subset.iterrows():
        # stop when data has been filled
        if len(merged_data) == target_size:
            break
        if accel_row['timestamp'] not in merged_data['timestamp'].values:
            while gyro_index < len(gyro_subset):
                gyro_row = gyro_subset.loc[gyro_index]
                if gyro_row['timestamp'] not in merged_data['timestamp'].values:
                    new_row = pd.DataFrame(data={'user-id': [accel_row['user-id']],
                                                 'activity': [accel_row['activity']],
                                                 'timestamp': [accel_row['timestamp']],
                                                 'x-axis_accel_' + device: [accel_row['x-axis']],
                                                 'y-axis_accel_' + device: [accel_row['y-axis']],
                                                 'z-axis_accel_' + device: [accel_row['z-axis']],
                                                 'x-axis_gyro_' + device: [gyro_row['x-axis']],
                                                 'y-axis_gyro_' + device: [gyro_row['y-axis']],
                                                 'z-axis_gyro_' + device: [gyro_row['z-axis']],
                                                 })
                    merged_data = merged_data.append(new_row, ignore_index=True)
                    gyro_index += 1
                    break
                gyro_index += 1
    return merged_data.sort_values(by='timestamp').reset_index(drop=True)


def merge_accel_gyro_data(accel, gyro, device):
    """Merge accelerometer and gyroscope data from the given device into the same dataframe
        Parameters:
            accel (Pandas DataFrame)
            gyro (Pandas DataFrame)
            device (str): string containing name of the device; either 'phone' or 'watch'

        Returns
            merged_data (Pandas DataFrame)
    """
    merged_data = pd.DataFrame(
        columns=['user-id', 'activity', 'timestamp',
                 'x-axis_accel_' + device, 'y-axis_accel_' + device, 'z-axis_accel_' + device,
                 'x-axis_gyro_' + device, 'y-axis_gyro_' + device, 'z-axis_gyro_' + device])

    activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
    for a_id in activity_ids:
        accel_subset = accel.loc[accel['activity'] == a_id].reset_index(drop=True)
        gyro_subset = gyro.loc[gyro['activity'] == a_id].reset_index(drop=True)
        merged_subset = pd.merge(accel_subset, gyro_subset, on=['user-id', 'activity', 'timestamp'])
        merged_subset.rename(columns={'x-axis_x': 'x-axis_accel_' + device,
                                      'y-axis_x': 'y-axis_accel_' + device,
                                      'z-axis_x': 'z-axis_accel_' + device,
                                      'x-axis_y': 'x-axis_gyro_' + device,
                                      'y-axis_y': 'y-axis_gyro_' + device,
                                      'z-axis_y': 'z-axis_gyro_' + device
                                      },
                             inplace=True)
        merged_subset.dropna(axis=0, how='any', inplace=True)

        # some subjects have different timestamps for accel and gyro data so match them by row instead of by timestamp
        # if none of the timestamps match at all, then line the data up by activity
        if len(merged_subset) == 0:
            merged_subset = manual_merge_accel_gyro_data(accel_subset, gyro_subset, device)
        # if the timestamps partially match up, then interpolate the missing timestamps
        elif len(merged_subset) < min(len(accel_subset), len(gyro_subset)):
            merged_subset = interpolate_subset(merged_subset, accel_subset, gyro_subset, device,
                                               min(len(accel_subset), len(gyro_subset)))
        merged_data = merged_data.append(merged_subset, ignore_index=True)
    merged_data = merged_data.dropna(axis=0, how='any').reset_index(drop=True)
    return merged_data


def merge_phone_watch_data(phone, watch):
    """Merge phone and watch sensor data measured for the given subject
        Parameters:
            phone (Pandas DataFrame): merged accelerometer and gyroscope data from the phone
            watch (Pandas DataFrame): merged accelerometer and gyroscope data from the watch

        Returns:
            merged_data (Pandas DataFrame)
    """
    merged_data = pd.DataFrame(
        columns=['user-id', 'activity', 'timestamp', 'x-axis_accel_phone', 'y-axis_accel_phone', 'z-axis_accel_phone',
                 'x-axis_gyro_phone', 'y-axis_gyro_phone', 'z-axis_gyro_phone', 'x-axis_accel_watch',
                 'y-axis_accel_watch', 'z-axis_accel_watch', 'x-axis_gyro_watch', 'y-axis_gyro_watch',
                 'z-axis_gyro_watch'])

    activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
    for a_id in activity_ids:
        # reset indexes to start from 0 so data from both dataframes is properly lined up
        phone_subset = phone.loc[phone['activity'] == a_id].reset_index(drop=True)
        watch_subset = watch.loc[watch['activity'] == a_id].reset_index(drop=True)
        if (not phone_subset.empty) and (not watch_subset.empty):
            phone_subset['x-axis_accel_watch'] = watch_subset['x-axis_accel_watch']
            phone_subset['y-axis_accel_watch'] = watch_subset['y-axis_accel_watch']
            phone_subset['z-axis_accel_watch'] = watch_subset['z-axis_accel_watch']
            phone_subset['x-axis_gyro_watch'] = watch_subset['x-axis_gyro_watch']
            phone_subset['y-axis_gyro_watch'] = watch_subset['y-axis_gyro_watch']
            phone_subset['z-axis_gyro_watch'] = watch_subset['z-axis_gyro_watch']
            merged_data = merged_data.append(phone_subset, ignore_index=True)
    merged_data = merged_data.dropna(axis=0, how='any').reset_index(drop=True)
    return merged_data


def merge_subject_data(subject_id):
    """Merge accelerometer and gyroscope from both phone and watch for the given subject"""
    phone_accel = read_data('wisdm-dataset/raw/phone/accel/data_16' + subject_id + '_accel_phone.txt')
    phone_gyro = read_data('wisdm-dataset/raw/phone/gyro/data_16' + subject_id + '_gyro_phone.txt')
    watch_accel = read_data('wisdm-dataset/raw/watch/accel/data_16' + subject_id + '_accel_watch.txt')
    watch_gyro = read_data('wisdm-dataset/raw/watch/gyro/data_16' + subject_id + '_gyro_watch.txt')

    phone_merge = merge_accel_gyro_data(phone_accel, phone_gyro, 'phone')
    watch_merge = merge_accel_gyro_data(watch_accel, watch_gyro, 'watch')
    complete_data = merge_phone_watch_data(phone_merge, watch_merge)

    return complete_data


def merge_single_device(device, write_to_csv=False):
    """Merges the accelerometer and gyroscope data of ONLY one device for each subject
        Parameter:
            device (str): the device whose data to merge; either 'phone' or 'watch'
    """
    merged_data = pd.DataFrame(columns=['user-id', 'activity', 'timestamp',
                                        'x-axis_accel_' + device, 'y-axis_accel_' + device, 'z-axis_accel_' + device,
                                        'x-axis_gyro_' + device, 'y-axis_gyro_' + device, 'z-axis_gyro_' + device])
    for i in range(51):
        str_id = str(i)
        if i < 10:
            str_id = '0' + str_id
        print("Processing subject " + str_id)
        device_accel = read_data(
            'wisdm-dataset/raw/' + device + '/accel/data_16' + str_id + '_accel_' + device + '.txt')
        device_gyro = read_data('wisdm-dataset/raw/' + device + '/gyro/data_16' + str_id + '_gyro_' + device + '.txt')
        device_data = merge_accel_gyro_data(device_accel, device_gyro, device)
        merged_data = merged_data.append(device_data, ignore_index=True)
        # write the particular subject's data to a separate CSV
        if write_to_csv:
            print("Writing to CSV...")
            device_data.to_csv('wisdm-merged/subject_' + device + '_merge/16' + str_id + '_' + device + '_merge.txt')
    merged_data.dropna(axis=0, how='any').reset_index(drop=True)
    # write all data for this device merged from every subject to one CSV file
    if write_to_csv:
        print("Writing to CSV...")
        merged_data.to_csv('wisdm-merged/' + device + '_merge.txt')


def merge_all_wisdm_by_subject(write_to_csv=False):
    """Create separate CSV files for each subject containing a merge of the data from the
        accels and gyros of both their phone and watch"""
    for i in range(51):
        str_id = str(i)
        if i < 10:
            str_id = '0' + str_id
        print("Processing subject " + str_id)
        subject_data = merge_subject_data(str_id)
        if write_to_csv:
            print("Writing to CSV...")
            subject_data.to_csv('wisdm-merged/16' + str_id + '_merged_data.txt')


def merge_all_wisdm_combined(write_to_csv=False):
    """Create one CSV file containing a merging of all data from all subjects"""
    complete_merge = pd.DataFrame(columns=['user-id', 'activity', 'timestamp',
                                           'x-axis_accel_phone', 'y-axis_accel_phone', 'z-axis_accel_phone',
                                           'x-axis_gyro_phone', 'y-axis_gyro_phone', 'z-axis_gyro_phone',
                                           'x-axis_accel_watch', 'y-axis_accel_watch', 'z-axis_accel_watch',
                                           'x-axis_gyro_watch', 'y-axis_gyro_watch', 'z-axis_gyro_watch'])
    for i in range(51):
        str_id = str(i)
        if i < 10:
            str_id = '0' + str_id
        print("Processing subject " + str_id)
        subject_data = merge_subject_data(str_id)
        complete_merge = complete_merge.append(subject_data, ignore_index=True)
    complete_merge.dropna(axis=0, how='any').reset_index(drop=True)
    if write_to_csv:
        print("Writing to CSV...")
        complete_merge.to_csv('wisdm-merged/complete_merge.txt')
    return complete_merge


def main():
    merge_all_wisdm_by_subject()
    # merge_all_wisdm_combined()
