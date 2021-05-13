from tensorflow.keras import models, layers


def basic_mlp(input_shape=(960,), num_classes=18):
    model = models.Sequential()
    model.add(layers.Dense(24, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def basic_cnn(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()
    # using filter size 128 and kernel size 10 from https://arxiv.org/ftp/arxiv/papers/2103/2103.03836.pdf
    # model.add(layers.Conv1D(128, 10, input_shape=(80, 12)))
    # model architecture from:
    #    https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
    model.add(layers.Conv1D(128, 10, activation='relu', input_shape=input_shape))
    model.add(layers.Conv1D(128, 10, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# model from [1] B. Oluwalade
#   “Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data,”
def paper_cnn(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()
    model.add(layers.Conv1D(128, 10, activation='relu', input_shape=input_shape))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv1D(128, 10, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# From A. Wijekoon and N. Wiratunga, “LEARNING-TO-LEARN PERSONALISED HUMAN ACTIVITY RECOGNITION MODELS.”
def maml(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Dense(num_classes * 2), input_shape=input_shape))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())
    model.add(layers.TimeDistributed(layers.Dense(num_classes * 4)))
    model.add(layers.Conv1D(64, 5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())
    model.add(layers.TimeDistributed(layers.Dense(num_classes * 2)))
    # model.add(layers.Flatten())
    model.add(layers.LSTM(1200))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(600, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def wijekoon_wiratunga(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Dense(num_classes * 2), input_shape=input_shape))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.BatchNormalization())
    model.add(layers.TimeDistributed(layers.Dense(num_classes * 2, activation='relu')))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def basic_lstm(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()
    model.add(layers.LSTM(100, input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def parallel_test(input_shape=(80, 3)):
    """Test using parallel layers with TensorFlow"""
    filter_size = 64
    kernel_size = 12
    num_classes = 18

    input_1 = layers.Input(shape=input_shape, name='input_1')
    input_2 = layers.Input(shape=input_shape, name='input_2')

    c1 = layers.Conv1D(filter_size, kernel_size)(input_1)
    p1 = layers.MaxPooling1D(pool_size=2)(c1)
    f1 = layers.Flatten()(p1)
    c2 = layers.Conv1D(filter_size, kernel_size)(input_2)
    p2 = layers.MaxPooling1D(pool_size=2)(c2)
    f2 = layers.Flatten()(p2)

    x = layers.concatenate([f1, f2])
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[input_1, input_2], outputs=[x])
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])


def parallel_test2(input_shape=(80, 3), num_classes=18):
    filter_size = 64
    kernel_size = 12

    ap = layers.Input(shape=input_shape)
    gp = layers.Input(shape=input_shape)

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

    merge_layer = layers.concatenate([accel_conv, gyro_conv])
    out = layers.Dense(num_classes, activation='softmax')(merge_layer)

    model = models.Model(inputs=[ap, gp], outputs=[out])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, train_x, train_y, batch_size, epochs, verbose=1):
    history = model.fit(train_x,
                        train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        # callbacks=callbacks_list,
                        # validation_data=(test_x, test_y),
                        verbose=verbose)
    return model


def train_model_by_sensor(model, train_ap, train_gp, train_aw, train_gw, train_labels, batch_size, epochs, verbose=1):
    model.fit([train_ap, train_gp], train_labels,
              batch_size=batch_size,
              epochs=epochs,
              # callbacks=callbacks_list,
              # validation_data=(test_x, testy_y_hot),
              verbose=verbose)
    return model


def evaluate_model(model, test_x, test_y, verbose=2):
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=verbose)
    return test_loss, test_accuracy


def evaluate_model_by_sensor(model, test_ap, test_gp, test_aw, test_gw, test_labels, verbose=2):
    test_loss, test_accuracy = model.evaluate([test_ap, test_gp], test_labels, verbose=verbose)
    return test_accuracy
