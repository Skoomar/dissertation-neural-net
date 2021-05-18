import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

import preprocess

plt.style.use('ggplot')


def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


def plot_activity(title, data):
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11) = plt.subplots(nrows=12, figsize=(15, 10),
                                                                                       sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis_accel_phone'], 'x-axis Phone Accelerometer')
    plot_axis(ax1, data['timestamp'], data['y-axis_accel_phone'], 'y-axis Phone Accelerometer')
    plot_axis(ax2, data['timestamp'], data['z-axis_accel_phone'], 'z-axis Phone Accelerometer')
    plot_axis(ax3, data['timestamp'], data['x-axis_gyro_phone'], 'x-axis Phone Gyroscope')
    plot_axis(ax4, data['timestamp'], data['y-axis_gyro_phone'], 'y-axis Phone Gyroscope')
    plot_axis(ax5, data['timestamp'], data['z-axis_gyro_phone'], 'z-axis Phone Gyroscope')
    plot_axis(ax6, data['timestamp'], data['x-axis_accel_watch'], 'x-axis Watch Accelerometer')
    plot_axis(ax7, data['timestamp'], data['y-axis_accel_watch'], 'y-axis Watch Accelerometer')
    plot_axis(ax8, data['timestamp'], data['z-axis_accel_watch'], 'z-axis Watch Accelerometer')
    plot_axis(ax9, data['timestamp'], data['x-axis_gyro_watch'], 'x-axis Watch Gyroscope')
    plot_axis(ax10, data['timestamp'], data['y-axis_gyro_watch'], 'y-axis Watch Gyroscope')
    plot_axis(ax11, data['timestamp'], data['z-axis_gyro_watch'], 'z-axis Watch Gyroscope')

    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(title)
    plt.subplots_adjust(top=0.90)
    plt.show()


def pearson_correlation(data1, data2):
    if len(data1) > len(data2):
        data1 = data1.head(len(data2))
    elif len(data2) > len(data1):
        data2 = data2.head(len(data1))

    pearson_apx = pearsonr(data1['x-axis_accel_phone'], data2['x-axis_accel_phone'])[0]
    pearson_apy = pearsonr(data1['y-axis_accel_phone'], data2['y-axis_accel_phone'])[0]
    pearson_apz = pearsonr(data1['z-axis_accel_phone'], data2['z-axis_accel_phone'])[0]
    pearson_gpx = pearsonr(data1['x-axis_gyro_phone'], data2['x-axis_gyro_phone'])[0]
    pearson_gpy = pearsonr(data1['y-axis_gyro_phone'], data2['y-axis_gyro_phone'])[0]
    pearson_gpz = pearsonr(data1['z-axis_gyro_phone'], data2['z-axis_gyro_phone'])[0]
    pearson_awx = pearsonr(data1['x-axis_accel_watch'], data2['x-axis_accel_watch'])[0]
    pearson_awy = pearsonr(data1['y-axis_accel_watch'], data2['y-axis_accel_watch'])[0]
    pearson_awz = pearsonr(data1['z-axis_accel_watch'], data2['z-axis_accel_watch'])[0]
    pearson_gwx = pearsonr(data1['x-axis_gyro_watch'], data2['x-axis_gyro_watch'])[0]
    pearson_gwy = pearsonr(data1['y-axis_gyro_watch'], data2['y-axis_gyro_watch'])[0]
    pearson_gwz = pearsonr(data1['z-axis_gyro_watch'], data2['z-axis_gyro_watch'])[0]
    print("pearson_apx:", pearson_apx)
    print("pearson_apy:", pearson_apy)
    print("pearson_apz:", pearson_apz)
    print("pearson_gpx:", pearson_gpx)
    print("pearson_gpy:", pearson_gpy)
    print("pearson_gpz:", pearson_gpz)
    print("pearson_awx:", pearson_awx)
    print("pearson_awy:", pearson_awy)
    print("pearson_awz:", pearson_awz)
    print("pearson_gwx:", pearson_gwx)
    print("pearson_gwy:", pearson_gwy)
    print("pearson_gwz:", pearson_gwz)

    mean_pearsonr = (
                                pearson_apy + pearson_apz + pearson_gpx + pearson_gpy + pearson_gpz + pearson_awx + pearson_awy + pearson_awz + pearson_gwx + pearson_gwy + pearson_gwz) / 12
    print("mean:", mean_pearsonr)


def main():
    subject1 = '1600'
    subject2 = '1601'
    dataset1 = preprocess.read_data(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/' + subject1 + '_merged_data.txt')
    dataset2 = preprocess.read_data(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/' + subject2 + '_merged_data.txt')
    activity = 'C'
    activity_subset1 = dataset1[dataset1['activity'] == activity]
    activity_subset2 = dataset2[dataset2['activity'] == activity]
    plot_activity('Subject ' + subject1 + ' doing Activity ' + activity, activity_subset1)
    plot_activity('Subject ' + subject2 + ' doing Activity ' + activity, activity_subset2)

    print(activity_subset1.loc[:1784])
    # pearson_correlation(activity_subset1, activity_subset2)
    activity_subset3 = activity_subset1.loc[:len(activity_subset1) // 2]
    activity_subset4 = activity_subset1.loc[len(activity_subset1) // 2:]
    pearson_correlation(activity_subset3, activity_subset4)


main()
