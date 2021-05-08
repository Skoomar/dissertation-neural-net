from tensorflow.keras import models, layers


def individual_conv_layers(input_layer, filter_size, kernel_size):
    conv = layers.Conv1D(filter_size, kernel_size)(input_layer)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)
    conv = layers.Dropout(0.2)(conv)
    # accel_conv = layers.MaxPooling1D(pool_size=2)(accel_conv)
    conv = layers.Flatten()(conv)
    return conv


def uDeepSense(input_shape=(80, 3), num_classes=18):
    filter_size = 64
    kernel_size = 12

    ap = layers.Input(shape=input_shape)
    gp = layers.Input(shape=input_shape)
    aw = layers.Input(shape=input_shape)
    gw = layers.Input(shape=input_shape)

    accel_conv = layers.Conv1D(filter_size, kernel_size)(ap)
    accel_conv = layers.BatchNormalization()(accel_conv)
    accel_conv = layers.Activation('relu')(accel_conv)
    accel_conv = layers.Dropout(0.2)(accel_conv)
    # accel_conv = layers.MaxPooling1D(pool_size=2)(accel_conv)
    accel_conv = layers.Flatten()(accel_conv)

    gyro_conv = layers.Conv1D(filter_size, kernel_size)(gp)
    gyro_conv = layers.BatchNormalization()(gyro_conv)
    gyro_conv = layers.Activation('relu')(gyro_conv)
    gyro_conv = layers.Dropout(0.2)(gyro_conv)
    # gyro_conv = layers.MaxPooling1D(pool_size=2)(gyro_conv)
    gyro_conv = layers.Flatten()(gyro_conv)
    # accel_conv = individual_conv_layers(ap, filter_size, kernel_size)
    # gyro_conv = individual_conv_layers(gp, filter_size, kernel_size)

    merge_layer = layers.concatenate([accel_conv, gyro_conv])
    out = layers.Dense(num_classes, activation='softmax')(merge_layer)

    model = models.Model(inputs=[ap, gp], outputs=[out])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train(model, train_ap, train_gp, train_aw, train_gw, train_labels, batch_size, epochs, verbose=1):
    model.fit([train_ap, train_gp], train_labels,
              batch_size=batch_size,
              epochs=epochs,
              # callbacks=callbacks_list,
              # validation_data=(test_x, testy_y_hot),
              verbose=verbose)
    return model


def evaluate(model, test_ap, test_gp, test_aw, test_gw, test_labels, verbose=2):
    test_loss, test_accuracy = model.evaluate([test_ap, test_gp], test_labels, verbose=verbose)
    return test_accuracy