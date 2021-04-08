import pandas as pd
import numpy as np


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


def match_accel_and_gyro_data(accel, gyro, device):
    merged_data = pd.merge(accel, gyro, how='inner', on=['user-id', 'activity', 'timestamp'])
    merged_data.rename(columns={'x-axis_x': 'x-axis_accel_' + device,
                                'y-axis_x': 'y-axis_accel_' + device,
                                'z-axis_x': 'z-axis_accel_' + device,
                                'x-axis_y': 'x-axis_gyro_' + device,
                                'y-axis_y': 'y-axis_gyro_' + device,
                                'z-axis_y': 'z-axis_gyro_' + device
                                },
                       inplace=True)
    return merged_data


def merge_phone_watch_data(phone, watch):
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
    merged_data.dropna(axis=0, how='any', inplace=True)
    return merged_data


def merge_subject_data(subject_id):
    phone_accel = read_data('wisdm-dataset/raw/phone/accel/data_16' + subject_id + '_accel_phone.txt')
    phone_gyro = read_data('wisdm-dataset/raw/phone/gyro/data_16' + subject_id + '_gyro_phone.txt')
    watch_accel = read_data('wisdm-dataset/raw/watch/accel/data_16' + subject_id + '_accel_watch.txt')
    watch_gyro = read_data('wisdm-dataset/raw/watch/gyro/data_16' + subject_id + '_gyro_watch.txt')

    phone_merge = match_accel_and_gyro_data(phone_accel, phone_gyro, 'phone')
    watch_merge = match_accel_and_gyro_data(watch_accel, watch_gyro, 'watch')
    complete_data = merge_phone_watch_data(phone_merge, watch_merge)
    return complete_data


def merge_all_wisdm():
    for i in range(51):
        str_id = str(i)
        if i < 10:
            str_id = '0' + str_id
        subject_data = merge_subject_data(str_id)
        subject_data.to_csv('wisdm-merged/16' + str_id + '_merged_data.txt')


merge_all_wisdm()
# TODO: fix the indexes on the big merge, think I just need to reset_index at the end of merge_phone_watch_data

# activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
# for i in activity_ids:
#     print("\nphone activity", i, len(phone_merge.loc[phone_merge['activity'] == i]))
#     print("watch activity", i, len(watch_merge.loc[watch_merge['activity'] == i]))
#     print("complete activity", i, len(complete_data.loc[complete_data['activity'] == i]))
#
# print(len(phone_merge))
# print(len(watch_merge))
# print(len(complete_data))
# complete_data.to_csv('1600_merged_data.txt')
