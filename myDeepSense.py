import tensorflow as tf
from tensorflow.keras import models, layers


def individual_conv_layers(input_layer):
    num_filters = 64
    kernel_size1 = 18
    conv1 = layers.Conv1D(num_filters, kernel_size1)(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    # accel_conv = layers.MaxPooling1D(pool_size=2)(accel_conv)
    # conv1 = layers.Flatten()(conv1)

    kernel_size2 = 3
    conv2 = layers.Conv1D(num_filters, kernel_size2)(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    # conv2 = layers.Flatten()(conv2)

    conv3 = layers.Conv1D(num_filters, kernel_size2)(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)

    # out = layers.Flatten()(conv3)
    conv3_shape = conv3.get_shape()
    out = tf.reshape(conv3, [-1, conv3_shape[1], 1, conv3_shape[2]])
    return out


def uDeepSense(input_shape=(80, 3), num_classes=18):

    ap = layers.Input(shape=input_shape)
    gp = layers.Input(shape=input_shape)
    aw = layers.Input(shape=input_shape)
    gw = layers.Input(shape=input_shape)

    # accel_phone_conv = layers.Conv1D(filter_size, kernel_size)(ap)
    # accel_phone_conv = layers.BatchNormalization()(accel_phone_conv)
    # accel_phone_conv = layers.Activation('relu')(accel_phone_conv)
    # accel_phone_conv = layers.Dropout(0.2)(accel_phone_conv)
    # # accel_phone_conv = layers.MaxPooling1D(pool_size=2)(accel_phone_conv)
    # accel_phone_conv = layers.Flatten()(accel_phone_conv)
    #
    # gyro_phone_conv = layers.Conv1D(filter_size, kernel_size)(gp)
    # gyro_phone_conv = layers.BatchNormalization()(gyro_phone_conv)
    # gyro_phone_conv = layers.Activation('relu')(gyro_phone_conv)
    # gyro_phone_conv = layers.Dropout(0.2)(gyro_phone_conv)
    # # gyro_phone_conv = layers.MaxPooling1D(pool_size=2)(gyro_phone_conv)
    # gyro_phone_conv = layers.Flatten()(gyro_phone_conv)
    #
    accel_phone_conv = individual_conv_layers(ap)
    gyro_phone_conv = individual_conv_layers(gp)
    accel_watch_conv = individual_conv_layers(aw)
    gyro_watch_conv = individual_conv_layers(gw)

    individual_output = layers.concatenate([accel_phone_conv, gyro_phone_conv, accel_watch_conv, gyro_watch_conv], 2)
    individual_output = layers.Dropout(0.2)(individual_output)
    print(individual_output.get_shape())

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
    merge_output = tf.reshape(merge_conv3, [-1, merge_conv3_shape[1], merge_conv3_shape[2]*merge_conv3_shape[3]])
    print("merge_output:", merge_output.get_shape())

    output = layers.Dense(num_classes, activation='softmax')(merge_conv3)

    model = models.Model(inputs=[ap, gp, aw, gw], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


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
    return test_accuracy