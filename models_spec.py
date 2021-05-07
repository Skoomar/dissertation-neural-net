import tensorflow as tf
from tensorflow.keras import models, layers


# print(tf.config.list_physical_devices())

def basic_mlp(input_shape=(960,), num_classes=18):
    model = models.Sequential()
    model.add(layers.Dense(24, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# From wikipedia
# Number of filters
#
# Since feature map size decreases with depth, layers near the input layer tend to have fewer filters while higher layers can have more. To equalize computation at each layer, the product of feature values va with pixel position is kept roughly constant across layers. Preserving more information about the input would require keeping the total number of activations (number of feature maps times number of pixel positions) non-decreasing from one layer to the next.
#
# The number of feature maps directly controls the capacity and depends on the number of available examples and task complexity.
# Filter size
#
# Common filter sizes found in the literature vary greatly, and are usually chosen based on the data set.
#
# The challenge is to find the right level of granularity so as to create abstractions at the proper scale, given a particular data set, and without overfitting.
# Pooling type and size
#
# In modern CNNs, max pooling is typically used, and often of size 2×2, with a stride of 2. This implies that the input is drastically downsampled, further improving the computational efficiency.
#
# Very large input volumes may warrant 4×4 pooling in the lower layers.[72] However, choosing larger shapes will dramatically reduce the dimension of the signal, and may result in excess information loss. Often, non-overlapping pooling windows perform best.[65]

def basic_cnn(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()
    # using filter size 128 and kernel size 10 for this Conv1D layer because: https://arxiv.org/ftp/arxiv/papers/2103/2103.03836.pdf
    # model.add(layers.Conv1D(128, 10, input_shape=(80, 12)))
    # model architecture used from here: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
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


# model from [1] B. Oluwalade, S. Neela, J. Wawira, T. Adejumo, and S. Purkayastha, “Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data,” pp. 645–650, 2021, doi: 10.5220/0010325906450650.
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


def transfer_cnn(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()
    model.add(layers.experimental.preprocessing.Discretization(12))
    # model.add(layers.Embedding())
    model.compile(loss='categorical_')
    print(model.summary())


# 226 epochs with a batch size of 32 - from Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data by Oluwalade, Bolu & Neela, Sunil
# def rnn(input_shape=(80, 12), num_classses=18):


def personalised_cnn(input_shape, num_classes=18):
    model = models.Sequential()
    model.add(layers)


# From A. Wijekoon and N. Wiratunga, “LEARNING-TO-LEARN PERSONALISED HUMAN AC-TIVITY RECOGNITION MODELS.”
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


def parallel_test(input_shape=(80, 3), num_classes=18):
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

    merge_layer = layers.concatenate([accel_conv, gyro_conv])
    out = layers.Dense(num_classes, activation='softmax')(merge_layer)

    model = models.Model(inputs=[ap, gp], outputs=[out])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def deep_sense(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()

    # accel individual conv layers
    model.add(layers.Conv1D(64, 18, activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.BatchNormalization())

    # gyroscope individual conv layers
    model.add(layers.Conv1D(64, 18, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.2))

    # Merge convolutional layers
    model.add(layers.Conv1D(64, 16, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(64, 12, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv1D(64, 8, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def buffelli(input_shape=(80, 12), num_classes=18):
    model = models.Sequential()


def train_model(model, train_x, train_y, batch_size, epochs, verbose=1):
    # our Hyperparameters - reasons to use certain hyperparameters: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
    # BATCH_SIZE = 400
    BATCH_SIZE = 32
    EPOCHS = 100
    # TODO: try messing around with/researching different values for batch_size and epochs
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
    return test_accuracy


def evaluate_model_by_sensor(model, test_ap, test_gp, test_aw, test_gw, test_labels, verbose=2):
    test_loss, test_accuracy = model.evaluate([test_ap, test_gp], test_labels, verbose=verbose)
    return test_accuracy
