import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, layers
import tensorflow_addons as tfa
from keras.utils import np_utils
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import f1_score
import os
import math

import preprocess
import models_spec
import myDeepSense
import evaluation

# stops cudnn giving an error when it tries to use too much memory on the GPU while training RNN layers
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

pd.options.display.float_format = '{:.1f}'.format
plt.style.use('ggplot')







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






def undo_one_hot_encoding(encoded):
    non_hot = np.array([], dtype=np.int)
    for i in range(len(encoded)):
        non_hot = np.append(non_hot, int(np.where(encoded[i] == 1)[0]))
    return non_hot


def decode_labels(encoded_y, label_encoder):
    """Reverse the one-hot encoding and numerical encoding of the original alphabetical identifiers for each activity
        class

    Parameters:
        encoded_y (numpy array): array of labels in shape (x, 18) as we have 18 classes
        label_encoder (scikit-learn LabelEncoder): the object that was used to encode this particular set of classes
                                                    during preprocessing

    Returns:
            decoded_y (numpy array): array of values
    """

    # if it's one-hot encoded then reverse it to get numerical identifiers for the target label
    if len(encoded_y.shape) == 2:
        non_hot_y = undo_one_hot_encoding(encoded_y)
    elif len(encoded_y.shape > 2):
        raise ValueError('Array of labels should only have up to 2 dimensions')

    # decode the numerical identifiers to the original alphabetical identifiers
    decoded_y = label_encoder.inverse_transform(non_hot_y)
    return decoded_y



def simple_run(make_confusion_matrix=False):
    # model = models.wijekoon_wiratunga()
    model = models_spec.paper_cnn()
    # model = models_spec.basic_lstm()
    for i in range(51):
        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id

        print("Subject " + subject_id)
        # output += "\n\nSubject" + subject_id
        train_x, train_y, test_x, test_y, val_x, val_y, label_encoder = preprocess.preprocess_subject_data_train_test_val(
            'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/16' + subject_id + '_merged_data.txt',
            split=[0.6, 0.2, 0.2],
            random_split=False)

        if train_y.shape[1] != 18:
            print("Subject only has:", train_y.shape[1], "features")
            continue

        trained_model = models_spec.train_model(model, train_x, train_y, batch_size=32, epochs=25, verbose=1)
        evaluated_model, loss, accuracy = models_spec.evaluate_model(trained_model, test_x, test_y)
        print("Subject ID " + subject_id + ":", accuracy, "\n")

        pred_y = evaluated_model.predict_classes(val_x)
        decoded_pred_y = label_encoder.inverse_transform(pred_y)
        decoded_val_y = decode_labels(val_y, label_encoder)
        non_hot_val_y = undo_one_hot_encoding(val_y)
        correct_predictions = 0
        for i in range(len(pred_y)):
            if pred_y[i] == non_hot_val_y[i]:
                correct_predictions += 1
        print("validation accuracy:", correct_predictions / len(val_y))

        if make_confusion_matrix:
            evaluation.plot_confusion_matrix(subject_id, non_hot_val_y, pred_y, label_encoder)


def train_one_test_two():
    x, y, le = preprocess.preprocess_subject_data('C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt')
    x2, y2, le = preprocess.preprocess_subject_data(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1601_merged_data.txt')

    model = models_spec.paper_cnn()
    print("Training")
    trained_model = models_spec.train_model(model, x, y, batch_size=32, epochs=25, verbose=1)
    # print(trained_model(include_top=False))
    accuracy = models_spec.evaluate_model(model, x2, y2)
    print("Accuracy:", accuracy)


def deep_sense_run():
    # train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_aw, test_gw, test_labels, label_encoder = preprocess_subject_data_by_sensor_train_test(
    #     'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', window_overlap=40,
    #     split_ratio=0.7,
    #     random_split=False)
    train_data, test_data, label_encoder = preprocess.preprocess_subject_data_by_sensor_train_test(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', window_overlap=80,
        split_ratio=0.7,
        random_split=False)
    train_ap = train_data['ap']
    train_gp = train_data['gp']
    train_aw = train_data['aw']
    train_gw = train_data['gw']
    train_labels = train_data['labels']
    test_ap = test_data['ap']
    test_gp = test_data['gp']
    test_aw = test_data['aw']
    test_gw = test_data['gw']
    test_labels = test_data['labels']

    # print("1:", train_ap.shape, train_gp.shape, train_aw.shape, train_gw.shape, train_labels.shape, test_ap.shape,
    #       test_gp.shape, test_aw.shape, test_gw.shape, test_labels.shape)
    # ap2, gp2, aw2, gw2, labels2, label_encoder = preprocess_subject_data_by_sensor(
    #     'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1601_merged_data.txt')
    # print("2:", ap2.shape, gp2.shape, aw2.shape, gw2.shape, labels2.shape)
    batch_size = 32
    model = myDeepSense.u_deep_sense(train_gp.shape[1:], 18, batch_size=batch_size)
    print("Training")
    trained_model = myDeepSense.train(model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                      batch_size=batch_size, epochs=25, verbose=1)

    file_path = 'saved_models/uDeepSense'
    saved = False
    i = 1
    while not saved:
        if not os.path.exists(file_path + str(i)):
            trained_model.save('saved_models/uDeepSense' + str(i))
            saved = True
        i += 1

    accuracy = myDeepSense.evaluate(model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("Accuracy:", accuracy)


# deep_sense_run_all_but_one()



def transfer_learn():
    # train_ap, train_gp, train_aw, train_gw, train_labels, test_ap, test_gp, test_aw, test_gw, test_labels, label_encoder = preprocess_subject_data_by_sensor_train_test(
    #     'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1601_merged_data.txt')

    train_data, test_data, label_encoder = preprocess.preprocess_subject_data_by_sensor_train_test(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', window_overlap=80,
        split_ratio=0.7,
        random_split=False)
    train_ap = train_data['ap']
    train_gp = train_data['gp']
    train_aw = train_data['aw']
    train_gw = train_data['gw']
    train_labels = train_data['labels']
    test_ap = test_data['ap']
    test_gp = test_data['gp']
    test_aw = test_data['aw']
    test_gw = test_data['gw']
    test_labels = test_data['labels']

    original_model = tf.keras.models.load_model('saved_models/uDeepSense3')

    for i in range(len(original_model.layers) - 2):
        original_model.layers[i].trainable = False

    # TODO: see if the transfer learning can be be just as accurate with a low proportion of train:test data
    # transfer_layer1 = original_model.layers[-2].output
    # transfer_dense_layers = layers.Dense(36)(transfer_layer1)
    # transfer_dense_layers = layers.Dense(72)(transfer_dense_layers)
    # transfer_out_layer = layers.Dense(18, activation='softmax')(transfer_dense_layers)
    # new_model = models.Model(inputs=original_model.input, outputs=transfer_out_layer)
    # # new_model.add_metric(tfa.metrics.f_scores.F1Score()(transfer_out_layer), name='f1_score')
    # new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(new_model.summary)
    transfer_learn_model = myDeepSense.transfer_learn_model(original_model)

    trained_model = myDeepSense.train(transfer_learn_model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                      batch_size=32, epochs=25, verbose=1)
    print("Accuracy:", myDeepSense.evaluate(trained_model, test_ap, test_gp, test_aw, test_gw, test_labels))


# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# np.set_printoptions(threshold=np.inf)


# run_by_subject()
# run_all_data()
# simple_run()
# test_data_prep()
# deep_sense_run()
# train_one_test_two()
transfer_learn()
