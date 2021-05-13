import tensorflow as tf
from tensorflow.keras import models, layers


def individual_conv_layers(input_layer):
    num_filters = 64
    kernel_size1 = 18
    conv1 = layers.Conv1D(num_filters, kernel_size1)(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.2)(conv1)

    kernel_size2 = 3
    conv2 = layers.Conv1D(num_filters, kernel_size2)(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.2)(conv2)

    conv3 = layers.Conv1D(num_filters, kernel_size2)(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)

    conv3_shape = conv3.get_shape()
    out = tf.reshape(conv3, [-1, conv3_shape[1], 1, conv3_shape[2]])
    return out


# TODO: work out the right sizes for filter size, kernel size, stride etc from what Davide defined his values based on
def u_deep_sense(input_shape=(80, 3), num_classes=18, batch_size=64):
    ap = layers.Input(shape=input_shape)
    gp = layers.Input(shape=input_shape)
    aw = layers.Input(shape=input_shape)
    gw = layers.Input(shape=input_shape)

    accel_phone_conv = individual_conv_layers(ap)
    gyro_phone_conv = individual_conv_layers(gp)
    accel_watch_conv = individual_conv_layers(aw)
    gyro_watch_conv = individual_conv_layers(gw)

    individual_output = layers.concatenate([accel_phone_conv, gyro_phone_conv, accel_watch_conv, gyro_watch_conv], 2)
    individual_output = layers.Dropout(0.2)(individual_output)

    merge_conv1 = layers.Conv2D(64, [2, 16], padding='same')(individual_output)
    merge_conv1 = layers.BatchNormalization()(merge_conv1)
    merge_conv1 = layers.Activation('relu')(merge_conv1)
    merge_conv1 = layers.Dropout(0.2)(merge_conv1)

    merge_conv2 = layers.Conv2D(64, [2, 12], padding='same')(merge_conv1)
    merge_conv2 = layers.BatchNormalization()(merge_conv2)
    merge_conv2 = layers.Activation('relu')(merge_conv2)
    merge_conv2 = layers.Dropout(0.2)(merge_conv2)

    merge_conv3 = layers.Conv2D(64, [2, 8], padding='same')(merge_conv2)
    merge_conv3 = layers.BatchNormalization()(merge_conv3)
    merge_conv3 = layers.Activation('relu')(merge_conv3)

    merge_conv3_shape = merge_conv3.get_shape()
    merge_output = tf.reshape(merge_conv3, [-1, merge_conv3_shape[1], merge_conv3_shape[2] * merge_conv3_shape[3]])
    print(merge_conv3_shape)
    print("merge_output:", merge_output.get_shape())

    # length = input_shape[0]
    # num_cells = 18
    # gru_cell1 = layers.GRUCell(num_cells)
    # gru_cell1 = tf.nn.RNNCellDropoutWrapper(gru_cell1, 0.5)
    #
    # gru_cell2 = layers.GRUCell(num_cells)
    # gru_cell2 = tf.nn.RNNCellDropoutWrapper(gru_cell2, 0.5)
    #
    # rnn_cell = layers.StackedRNNCells([gru_cell1, gru_cell2])
    # init_state = rnn_cell.get_initial_state(merge_output, batch_size, tf.float32)
    # print("init_state:", init_state)
    # rnn = layers.RNN(rnn_cell, return_state=True)
    # rnn_output, final_state1, final_state2 = rnn(merge_output, initial_state=init_state)
    # print("rnn_out:", rnn_output)
    # sum_rnn_output = tf.reduce_sum(rnn_output, axis=0, keepdims=False)
    # print("sum_rnn:",sum_rnn_output)
    # # avg_rnn_output = sum_rnn_output / tf.tile(length, [1, num_cells])

    # gru = layers.GRU(18, batch_input_shape=(batch_size, input_shape[0], 18), dropout=0.5)(merge_output)
    gru = layers.GRU(36, dropout=0.5)(merge_output)

    output = layers.Dense(num_classes, activation='softmax')(gru)

    model = models.Model(inputs=[ap, gp, aw, gw], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def transfer_learn_model(original_model):
    for i in range(len(original_model.layers) - 2):
        original_model.layers[i].trainable = False

    # TODO: see if the transfer learning can be be just as accurate with a low proportion of train:test data
    transfer_layer1 = original_model.layers[-2].output
    # transfer_dense_layers = layers.Dense(36)(transfer_layer1)
    # transfer_dense_layers = layers.Dense(72)(transfer_dense_layers)
    transfer_out_layer = layers.Dense(18, activation='softmax')(transfer_layer1)
    new_model = models.Model(inputs=original_model.input, outputs=transfer_out_layer)
    # new_model.add_metric(tfa.metrics.f_scores.F1Score()(transfer_out_layer), name='f1_score')
    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(new_model.summary())
    return new_model


def train(model, train_ap, train_gp, train_aw, train_gw, train_labels, batch_size, epochs, verbose=1):
    model.fit([train_ap, train_gp, train_aw, train_gw], train_labels,
              batch_size=batch_size,
              epochs=epochs,
              # callbacks=callbacks_list,
              # validation_data=(test_x, testy_y_hot),
              verbose=verbose)
    return model


def evaluate(model, test_ap, test_gp, test_aw, test_gw, test_labels, verbose=2):
    test_loss, test_accuracy = model.evaluate([test_ap, test_gp, test_aw, test_gw], test_labels, verbose=verbose)
    return test_loss, test_accuracy
