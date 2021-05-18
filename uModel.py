import tensorflow as tf
from tensorflow.keras import models, layers


def individual_conv_layers(input_layer, num_filters):
    """Create the individual convolutional layers"""
    kernel_size1 = 3 * 3
    conv1 = layers.Conv1D(num_filters, kernel_size1, strides=3)(input_layer)
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


def uModel(input_shape=(80, 3), num_classes=18):
    """My model based on the DeepSense framework and D. Buffelli's adaptation of it
        Uses the TensorFlow Functional API to use parallel inputs and parallel layers
    """
    ap = layers.Input(shape=input_shape)
    gp = layers.Input(shape=input_shape)
    aw = layers.Input(shape=input_shape)
    gw = layers.Input(shape=input_shape)

    num_filters = 64
    accel_phone_conv = individual_conv_layers(ap, num_filters)
    gyro_phone_conv = individual_conv_layers(gp, num_filters)
    accel_watch_conv = individual_conv_layers(aw, num_filters)
    gyro_watch_conv = individual_conv_layers(gw, num_filters)

    individual_output = layers.concatenate([accel_phone_conv, gyro_phone_conv, accel_watch_conv, gyro_watch_conv], 2)
    individual_output = layers.Dropout(0.2)(individual_output)

    merge_conv1 = layers.Conv2D(num_filters, [4, 8], padding='same')(individual_output)
    merge_conv1 = layers.BatchNormalization()(merge_conv1)
    merge_conv1 = layers.Activation('relu')(merge_conv1)
    merge_conv1 = layers.Dropout(0.2)(merge_conv1)

    merge_conv2 = layers.Conv2D(num_filters, [4, 6], padding='same')(merge_conv1)
    merge_conv2 = layers.BatchNormalization()(merge_conv2)
    merge_conv2 = layers.Activation('relu')(merge_conv2)
    merge_conv2 = layers.Dropout(0.2)(merge_conv2)

    merge_conv3 = layers.Conv2D(num_filters, [4, 4], padding='same')(merge_conv2)
    merge_conv3 = layers.BatchNormalization()(merge_conv3)
    merge_conv3 = layers.Activation('relu')(merge_conv3)

    merge_conv3_shape = merge_conv3.get_shape()
    merge_output = tf.reshape(merge_conv3, [-1, merge_conv3_shape[1], merge_conv3_shape[2] * merge_conv3_shape[3]])

    gru = layers.GRU(80, dropout=0.5)(merge_output)

    output = layers.Dense(num_classes, activation='softmax')(gru)

    model = models.Model(inputs=[ap, gp, aw, gw], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def transfer_learn_model(original_model):
    """Carries out the transfer learning step"""
    for i in range(len(original_model.layers) - 2):
        original_model.layers[i].trainable = False

    # get the output from the final convolutional layer
    transfer_layer1 = original_model.layers[-2].output
    # create the new output layer
    transfer_out_layer = layers.Dense(18, activation='softmax')(transfer_layer1)
    # compile the newly re-trained model
    new_model = models.Model(inputs=original_model.input, outputs=transfer_out_layer)
    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(new_model.summary())
    return new_model


def train(model, train_ap, train_gp, train_aw, train_gw, train_labels, batch_size, epochs, verbose=1):
    model.fit([train_ap, train_gp, train_aw, train_gw], train_labels,
              batch_size=batch_size,
              epochs=epochs,
              # validation_split=0.1,
              verbose=verbose)
    return model


def evaluate(model, test_ap, test_gp, test_aw, test_gw, test_labels, verbose=2):
    test_loss, test_accuracy = model.evaluate([test_ap, test_gp, test_aw, test_gw], test_labels, verbose=verbose)
    return test_loss, test_accuracy
