import os

import numpy as np
import pandas as pd
import tensorflow as tf

import evaluation
import extra_models
import preprocess
import uModel

# stops tensorflow-gpu giving an error when it tries to use too much memory on the GPU while training RNN layers
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

pd.options.display.float_format = '{:.1f}'.format


def simple_cnn_run(make_confusion_matrix=False):
    model = extra_models.paper_cnn()
    for i in range(51):
        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id

        print("Subject " + subject_id)
        # output += "\n\nSubject" + subject_id
        train_x, train_y, test_x, test_y, val_x, val_y, label_encoder = preprocess.preprocess_subject_train_test_val(
            'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/16' + subject_id + '_merged_data.txt',
            window_size=80,
            window_step=40,
            split=[0.7, 0.2, 0.1],
            random_split=False)
        print(train_x.shape)

        if train_y.shape[1] != 18:
            print("Subject only has:", train_y.shape[1], "features")
            continue

        trained_model = extra_models.train_model(model, train_x, train_y, batch_size=32, epochs=25, verbose=1)
        loss, accuracy = extra_models.evaluate_model(trained_model, test_x, test_y)
        print("Subject ID " + subject_id + ":", accuracy, "\n")

        pred_y = trained_model.predict_classes(val_x)
        decoded_pred_y = label_encoder.inverse_transform(pred_y)
        decoded_val_y = evaluation.decode_labels(val_y, label_encoder)
        non_hot_val_y = evaluation.undo_one_hot_encoding(val_y)
        correct_predictions = 0
        for i in range(len(pred_y)):
            if pred_y[i] == non_hot_val_y[i]:
                correct_predictions += 1
        print("validation accuracy:", correct_predictions / len(val_y))

        if make_confusion_matrix:
            evaluation.plot_confusion_matrix(subject_id, non_hot_val_y, pred_y, label_encoder)


def train_one_test_two():
    x, y, le = preprocess.preprocess_subject(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt')
    x2, y2, le = preprocess.preprocess_subject(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1601_merged_data.txt')

    model = extra_models.paper_cnn()
    print("Training")
    trained_model = extra_models.train_model(model, x, y, batch_size=32, epochs=25, verbose=1)
    # print(trained_model(include_top=False))
    accuracy = extra_models.evaluate_model(model, x2, y2)
    print("Accuracy:", accuracy)


def deep_sense_run_train_test():
    # train_data, test_data, label_encoder = preprocess.preprocess_subject_by_sensor_train_test(
    #     'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', window_step=80,
    #     split_ratio=0.7,
    #     random_split=False)
    train_data, test_data, label_encoder = preprocess.leave_one_out_cv_by_sensor_train_test(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge', '1600', window_step=80, split_ratio=0.7,
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
    batch_size = 64
    model = uModel.uModel(train_gp.shape[1:], 18, batch_size=batch_size)
    print("Training")
    trained_model = uModel.train(model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                 batch_size=batch_size, epochs=25, verbose=1)
    print('Evaluating')
    loss, accuracy = uModel.evaluate(trained_model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("Accuracy:", accuracy)

    print('Saving model')
    file_path = 'saved_models/uDeepSense'
    saved = False
    i = 1
    while not saved:
        if not os.path.exists(file_path + str(i)):
            trained_model.save('saved_models/uDeepSense' + str(i))
            saved = True
        i += 1
    print('Saved')


def cnn_full_data():
    x, y, label_encoder = preprocess.leave_one_out_cv(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge', '1600', window_step=80)

    batch_size = 64
    model = extra_models.paper_cnn(x.shape[1:], 18)
    print("Training")
    trained_model = extra_models.train_model(model, x, y,
                                             batch_size=batch_size, epochs=25, verbose=1)
    print('Saving model')
    file_path = 'saved_models/CNN'
    saved = False
    i = 1
    while not saved:
        if not os.path.exists(file_path + str(i)):
            model_path = file_path + str(i)
            trained_model.save(model_path)
            saved = True
        i += 1
    print('Saved as ' + model_path)


def evaluate_cnn():
    x, y, label_encoder = preprocess.preprocess_subject(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', window_step=80)
    model = tf.keras.models.load_model('saved_models/benchmarkCNN')
    print('Evaluating')
    loss, accuracy = extra_models.evaluate_model(model, x, y)
    print('CNN Accuracy:', accuracy)


def deep_sense_test_run():
    train_data, test_data, label_encoder = preprocess.preprocess_subject_by_sensor_train_test(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/1600_merged_data.txt', window_step=80,
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
    batch_size = 64
    model = uModel.uModel(train_ap.shape[1:], 18)
    print("Training")
    trained_model = uModel.train(model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                 batch_size=batch_size, epochs=25, verbose=1)
    loss, accuracy = uModel.evaluate(trained_model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("Accuracy:", accuracy)


def deep_sense_run():
    left_out = '1648'
    ap, gp, aw, gw, labels, label_encoder = preprocess.leave_one_out_cv_by_sensor(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge', left_out, window_step=80)

    print("1:", ap.shape, gp.shape, aw.shape, gw.shape, labels.shape)

    batch_size = 64
    model = uModel.uModel(ap.shape[1:], 18)
    print("Training")
    trained_model = uModel.train(model, ap, gp, aw, gw, labels,
                                 batch_size=batch_size, epochs=25, verbose=1)
    print('Saving model')
    model_path = 'saved_models/uDeepSense_loocv' + left_out
    saved = False
    i = 1
    while not saved:
        if os.path.exists(model_path):
            print("Model already exists here... Overwrite?")
            overwrite = input("Overwrite?")
            if overwrite != 'y':
                break
        trained_model.save(model_path)
        saved = True
    print('Saved as ' + model_path)


def transfer_learn():
    left_out = '1619'
    # around 0.25/0.3 split_ratio of train:test data seems to be the sweet spot where accuracy is not lost too much
    train_data, test_data, label_encoder = preprocess.preprocess_subject_by_sensor_train_test(
        'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/' + left_out + '_merged_data.txt', window_step=80,
        split_ratio=0.3,
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

    original_model = tf.keras.models.load_model('saved_models/uDeepSense_loocv' + left_out)

    # evaluate the model on new data before re-training
    print('Evaluating')
    loss, accuracy = uModel.evaluate(original_model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("original_model Accuracy:", accuracy)

    original_pred_labels = original_model.predict([test_ap, test_gp, test_aw, test_gw])
    original_pred_labels = np.argmax(original_pred_labels, axis=1)
    true_y = evaluation.undo_one_hot_encoding(test_labels)

    evaluation.plot_confusion_matrix('Pre-transfer-learning LOOCV on Subject ' + left_out, true_y,
                                     original_pred_labels, label_encoder)

    transfer_learn_model = uModel.transfer_learn_model(original_model)

    trained_model = uModel.train(transfer_learn_model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                 batch_size=32, epochs=25, verbose=1)
    print('Evaluating')
    loss, accuracy = uModel.evaluate(trained_model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("Accuracy:", accuracy)

    pred_y = trained_model.predict([test_ap, test_gp, test_aw, test_gw])
    pred_y = np.argmax(pred_y, axis=1)
    true_y = evaluation.undo_one_hot_encoding(test_labels)

    evaluation.plot_confusion_matrix('Transfer-Learn LOOCV on Subject ' + left_out, true_y, pred_y,
                                     label_encoder)
    print("F1-Score: ", evaluation.calculate_f1_score(true_y, pred_y, label_encoder))


# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# np.set_printoptions(threshold=np.inf)


# simple_run()
# deep_sense_test_run()
# deep_sense_run()
transfer_learn()
# cnn_full_data()
# evaluate_cnn()
