from tensorflow.keras import models, layers


def benchmark_cnn(input_shape=(80, 12), num_classes=18):
    """CNN Model from B. Oluwalade's et al., “Human Activity Recognition using Deep Learning Models on Smartphones
        and Smartwatches Sensor Data”"""
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
