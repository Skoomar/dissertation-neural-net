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


def manual_merge_accel_gyro_data(accel_subset, gyro_subset, device):
    """Merge data by lining up the rows of accel and gyro data"""
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


def fill_subset(accel, gyro, device, target):
    """If timestamps of accel and gyro data are only partially matched up then use this to fill in remaining values"""
    for index, row in gyro.iterrows():
        print(index)


def merge_accel_gyro_data(accel, gyro, device):
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
        # some subjects have different timestamps for accel and gyro data so match them by row instead of by timestamp
        if len(merged_subset) == 0:
            merged_subset = manual_merge_accel_gyro_data(accel_subset, gyro_subset, device)
        elif len(merged_subset) < min(len(accel_subset), len(gyro_subset)):
            fill_subset(accel, gyro, device, min(len(accel_subset), len(gyro_subset)))
        merged_data = merged_data.append(merged_subset, ignore_index=True)
    merged_data = merged_data.dropna(axis=0, how='any').reset_index(drop=True)
    return merged_data


# def merge_accel_and_gyro_data(accel, gyro, device):
#
#     merged_data = pd.merge(accel, gyro, how='inner', on=['user-id', 'activity', 'timestamp'])
#     merged_data.rename(columns={'x-axis_x': 'x-axis_accel_' + device,
#                                 'y-axis_x': 'y-axis_accel_' + device,
#                                 'z-axis_x': 'z-axis_accel_' + device,
#                                 'x-axis_y': 'x-axis_gyro_' + device,
#                                 'y-axis_y': 'y-axis_gyro_' + device,
#                                 'z-axis_y': 'z-axis_gyro_' + device
#                                 },
#                        inplace=True)
#
#     activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
#     for i in activity_ids:
#         print("\nDevice:", device)
#         print("accel activity", i, len(accel.loc[accel['activity'] == i]))
#         print("gyro activity", i, len(gyro.loc[gyro['activity'] == i]))
#         print("merged activity", i, len(merged_data.loc[merged_data['activity'] == i]))
#     return merged_data


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
    merged_data = merged_data.dropna(axis=0, how='any').reset_index(drop=True)
    return merged_data


def merge_subject_data(subject_id):
    phone_accel = read_data('wisdm-dataset/raw/phone/accel/data_16' + subject_id + '_accel_phone.txt')
    phone_gyro = read_data('wisdm-dataset/raw/phone/gyro/data_16' + subject_id + '_gyro_phone.txt')
    watch_accel = read_data('wisdm-dataset/raw/watch/accel/data_16' + subject_id + '_accel_watch.txt')
    watch_gyro = read_data('wisdm-dataset/raw/watch/gyro/data_16' + subject_id + '_gyro_watch.txt')

    phone_merge = merge_accel_gyro_data(phone_accel, phone_gyro, 'phone')
    watch_merge = merge_accel_gyro_data(watch_accel, watch_gyro, 'watch')
    complete_data = merge_phone_watch_data(phone_merge, watch_merge)

    # activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
    # for i in activity_ids:
    #     print("\n" + subject_id + "\nphone activity", i, len(phone_merge.loc[phone_merge['activity'] == i]))
    #     print("watch activity", i, len(watch_merge.loc[watch_merge['activity'] == i]))
    #     print("complete activity", i, len(complete_data.loc[complete_data['activity'] == i]))

    print(len(phone_merge))
    print(len(watch_merge))
    print(len(complete_data))
    return complete_data


def merge_all_wisdm(write_to_csv=False):
    # for i in range(51):
    if True:
        i = 2
        str_id = str(i)
        if i < 10:
            str_id = '0' + str_id
        subject_data = merge_subject_data(str_id)
        if write_to_csv:
            subject_data.to_csv('wisdm-merged/16' + str_id + '_merged_data.txt')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
merge_all_wisdm()
# TODO: fix the indexes on the big merge, think I just need to reset_index at the end of merge_phone_watch_data
# TODO: 1650's data doesn't look complete, only 27000 values

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
