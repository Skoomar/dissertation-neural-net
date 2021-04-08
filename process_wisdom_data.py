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


def manual_merge(d1, d2):
    pass


def merge_phone_watch_data(phone, watch):
    merged_data = pd.DataFrame(
        columns=['user-id', 'activity', 'timestamp', 'x-axis_accel_phone', 'y-axis_accel_phone', 'z-axis_accel_phone',
                 'x-axis_gyro_phone', 'y-axis_gyro_phone', 'z-axis_gyro_phone', 'x-axis_accel_watch',
                 'y-axis_accel_watch', 'z-axis_accel_watch', 'x-axis_gyro_watch', 'y-axis_gyro_watch',
                 'z-axis_gyro_watch'])

    activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
    for a_id in activity_ids:
        phone_subset = phone.loc[phone['activity'] == a_id]
        watch_subset = watch.loc[watch['activity'] == a_id]
        if (not phone_subset.empty) and (not watch_subset.empty):
            phone_subset[['x-axis_accel_watch', 'y-axis_accel_watch', 'z-axis_accel_watch',
                          'x-axis_gyro_watch', 'y-axis_gyro_watch', 'z-axis_gyro_watch']
            ] = watch_subset[['x-axis_accel_watch', 'y-axis_accel_watch', 'z-axis_accel_watch',
                              'x-axis_gyro_watch', 'y-axis_gyro_watch', 'z-axis_gyro_watch']]
            merged_data = merged_data.append(phone_subset, ignore_index=True)
    merged_data.dropna(axis=0, how='any', inplace=True)

    return merged_data


phone_accel = read_data('wisdm-dataset/raw/phone/accel/data_1600_accel_phone.txt')
phone_gyro = read_data('wisdm-dataset/raw/phone/gyro/data_1600_gyro_phone.txt')
watch_accel = read_data('wisdm-dataset/raw/watch/accel/data_1600_accel_watch.txt')
watch_gyro = read_data('wisdm-dataset/raw/watch/gyro/data_1600_gyro_watch.txt')

phone_merge = match_accel_and_gyro_data(phone_accel, phone_gyro, 'phone')
watch_merge = match_accel_and_gyro_data(watch_accel, watch_gyro, 'watch')
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', -1)
complete_data = merge_phone_watch_data(phone_merge, watch_merge)
# print(complete_data)

activity_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'O', 'P', 'Q', 'R', 'S']
for i in activity_ids:
    print("\nphone activity", i, len(phone_merge.loc[phone_merge['activity'] == i]))
    print("watch activity", i, len(watch_merge.loc[watch_merge['activity'] == i]))
    print("complete activity", i, len(complete_data.loc[complete_data['activity'] == i]))

print(len(phone_merge))
print(len(watch_merge))
print(len(complete_data))

# phone_watch_merge = merge_phone_watch_data(phone_merge, watch_merge)
# phone_watch_merge.to_csv('phone_watch_merge.txt')
# print(phone_watch_merge)
