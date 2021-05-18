import os

import numpy as np
import pandas as pd
import tensorflow as tf

import evaluation
import extra_models
import preprocess
import uTransferL

# stops tensorflow-gpu giving an error when it tries to use too much memory on the GPU while training RNN layers
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

pd.options.display.float_format = '{:.1f}'.format

# path to the directory containing the merged WISDM data
DATA_PATH = 'C:/Users/umar_/prbx-data/wisdm-merged/subject_full_merge/'


def uTransferL_pretrain(left_out):
    """Pre-train the uTransferL model on all but the left-out subject's data and save it to file
        in the saved_models directory

        Input:
            left_out (str): String ID of a user between 1600-1650
    """

    ap, gp, aw, gw, labels, label_encoder = preprocess.leave_one_out_cv_by_sensor(
        DATA_PATH, left_out)

    print("1:", ap.shape, gp.shape, aw.shape, gw.shape, labels.shape)

    batch_size = 64
    model = uTransferL.uTransferL(ap.shape[1:], 18)
    print("Training")
    trained_model = uTransferL.train(model, ap, gp, aw, gw, labels,
                                     batch_size=batch_size, epochs=25, verbose=1)
    print('Saving model')
    model_path = 'saved_models/uTransferL_loocv' + left_out
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


def transfer_learn(left_out):
    """Carry out the transfer learning step on uTransferL using left-out subject's data
        and evaluate results

       Input:
            left_out (str): String ID of a user between 1600-1650
    """

    # around 0.3 split_ratio of train:test data seems to be the sweet spot where accuracy is not lost too much
    train_data, test_data, label_encoder = preprocess.preprocess_subject_by_sensor_train_test(
        DATA_PATH + left_out + '_merged_data.txt',
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

    original_model = tf.keras.models.load_model('saved_models/uTransferL_loocv' + left_out)

    # evaluate the model on new data before re-training
    print('Evaluating')
    loss, accuracy = uTransferL.evaluate(original_model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("original_model Accuracy:", accuracy)

    original_pred_labels = original_model.predict([test_ap, test_gp, test_aw, test_gw])
    original_pred_labels = np.argmax(original_pred_labels, axis=1)
    true_y = evaluation.undo_one_hot_encoding(test_labels)

    evaluation.plot_confusion_matrix('Pre-transfer-learning LOOCV on Subject ' + left_out, true_y,
                                     original_pred_labels, label_encoder)

    # do transfer learning and evaluate
    transfer_learn_model = uTransferL.transfer_learn_model(original_model)

    trained_model = uTransferL.train(transfer_learn_model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                     batch_size=32, epochs=25, verbose=1)
    print('Evaluating')
    loss, accuracy = uTransferL.evaluate(trained_model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("Accuracy:", accuracy)

    pred_y = trained_model.predict([test_ap, test_gp, test_aw, test_gw])
    pred_y = np.argmax(pred_y, axis=1)
    true_y = evaluation.undo_one_hot_encoding(test_labels)

    evaluation.plot_confusion_matrix('Transfer-Learn LOOCV on Subject ' + left_out, true_y, pred_y,
                                     label_encoder)
    print("F1-Score: ", evaluation.calculate_f1_score(true_y, pred_y, label_encoder))


def benchmark_cnn_pretrain(left_out):
    """Pre-train the benchmark CNN on all but the left-out subject's data and save it to file
        in the saved_models directory
        Input:
            left_out (str): String ID of the user between 1600-1650
    """
    x, y, label_encoder = preprocess.leave_one_out_cv(
        DATA_PATH, left_out)

    model = extra_models.benchmark_cnn(x.shape[1:], 18)
    print("Training")
    trained_model = extra_models.train_model(model, x, y,
                                             batch_size=64, epochs=25, verbose=1)
    print('Saving model')
    model_path = 'saved_models/benchmarkCNN_loocv' + left_out
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


def evaluate_benchmark_cnn(left_out):
    """Evaluate the benchmark CNN on the left-out subject's data
        Input:
            left_out (str): String ID of the user between 1600-1650
    """
    train_x, train_y, test_x, test_y, label_encoder = preprocess.preprocess_subject_train_test(
        DATA_PATH + left_out + '_merged_data.txt',
        split_ratio=0.3,
        random_split=False)

    cnn = tf.keras.models.load_model('saved_models/benchmarkCNN_loocv' + left_out)

    print('Evaluating')
    loss, accuracy = extra_models.evaluate_model(cnn, test_x, test_y)
    print("CNN Accuracy:", accuracy)

    pred_y = cnn.predict(test_x)
    pred_y = np.argmax(pred_y, axis=1)
    true_y = evaluation.undo_one_hot_encoding(test_y)

    evaluation.plot_confusion_matrix('Benchmark CNN LOOCV on Subject ' + left_out, true_y, pred_y,
                                     label_encoder)
    print("F1-Score: ", evaluation.calculate_f1_score(true_y, pred_y, label_encoder))


def uTransferL_test_run():
    """Small test on a single subject's data without transfer learning step"""
    train_data, test_data, label_encoder = preprocess.preprocess_subject_by_sensor_train_test(
        DATA_PATH + '1600_merged_data.txt',
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
    model = uTransferL.uTransferL(train_ap.shape[1:], 18)
    print("Training")
    trained_model = uTransferL.train(model, train_ap, train_gp, train_aw, train_gw, train_labels,
                                     batch_size=batch_size, epochs=25, verbose=1)
    loss, accuracy = uTransferL.evaluate(trained_model, test_ap, test_gp, test_aw, test_gw, test_labels)
    print("Accuracy:", accuracy)


def simple_cnn_run(make_confusion_matrix=False):
    """Evaluates the benchmark CNN separately on each user's individual data"""
    model = extra_models.benchmark_cnn()
    for i in range(51):
        subject_id = str(i)
        if i < 10:
            subject_id = '0' + subject_id
        subject_id = '16' + subject_id
        print("Subject " + subject_id)

        train_x, train_y, test_x, test_y, val_x, val_y, label_encoder = preprocess.preprocess_subject_train_test_val(
            DATA_PATH + subject_id + '_merged_data.txt',
            window_size=80,
            window_step=80,
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
            evaluation.plot_confusion_matrix('Confusion Matrix Example', non_hot_val_y, pred_y, label_encoder)


if __name__ == '__main__':
    # the subject to be left out for leave-one-out cross-validation
    left_out = '1648'

    # uncomment method you want to run
    # uTransferL_pretrain(left_out)
    # benchmark_cnn_pretrain(left_out)
    # transfer_learn(left_out)
    # evaluate_benchmark_cnn(left_out)
    # simple_cnn_run()
