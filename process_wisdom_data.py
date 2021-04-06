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


# def match_records(match_label, arr1, arr2):
#     for i in arr1.:
#         print(i[match_label])
#         # if arr1[match_label] in arr2[match_label]:
#         #     arr1[match_label]
#         # elif arr1[match_label] not in arr2[match_label]:

#
# def match_data_by_timestamp(accel, gyro):
#     print(type(accel))
#     if len(accel) > len(gyro):
#         return match_records('timestamp', accel, gyro)
#     else:
#         return match_records('timestamp', gyro, accel)

def match_accel_and_gyro_data(accel, gyro):
    merged_data = pd.merge(accel, gyro, how='inner', on=['user-id', 'activity', 'timestamp'])
    merged_data.rename(columns={'x-axis_x': 'x-axis_accel',
                                'y-axis_x': 'y-axis_accel',
                                'z-axis_x': 'z-axis_accel',
                                'x-axis_y': 'x-axis_gyro',
                                'y-axis_y': 'y-axis_gyro',
                                'z-axis_y': 'z-axis_gyro'},
                       inplace=True)
    return merged_data


phone_accel = read_data('wisdm-dataset/raw/phone/accel/data_1600_accel_phone.txt')
phone_gyro = read_data('wisdm-dataset/raw/phone/gyro/data_1600_gyro_phone.txt')
watch_accel = read_data('wisdm-dataset/raw/watch/accel/data_1600_accel_watch.txt')
watch_gyro = read_data('wisdm-dataset/raw/watch/gyro/data_1600_gyro_watch.txt')

# phone_merge = pd.merge(phone_accel, phone_gyro, how='inner', on=['user-id', 'activity', 'timestamp'])
# phone_merge.rename(columns={'x-axis_x': 'x-axis_accel',
#                             'y-axis_x': 'y-axis_accel',
#                             'z-axis_x': 'z-axis_accel',
#                             'x-axis_y': 'x-axis_gyro',
#                             'y-axis_y': 'y-axis_gyro',
#                             'z-axis_y': 'z-axis_gyro'},
#                    inplace=True)
phone_merge = match_accel_and_gyro_data(phone_accel, phone_gyro)
watch_merge = match_accel_and_gyro_data(watch_accel, watch_gyro)

print(phone_merge)
print(watch_merge)

phone_merge['timestamp'].to_csv('phone_merge_test.csv')
watch_merge['timestamp'].to_csv('watch_merge_test.csv')