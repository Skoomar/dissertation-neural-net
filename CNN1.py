# made by following https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0

from tensorflow.keras import models, layers


def create_model(input_shape=(80, 12), num_classes=18):
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


def train_model(model, train_x, train_y, batch_size=400, epochs=50, verbose=1):

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
                        # validation_data=(test_x, testy_y_hot),
                        verbose=verbose)
    return model


def evaluate_model(model, test_x, test_y, verbose=2):
    test_loss, test_accuracy = model.evaluate(test_x, test_y, verbose=verbose)
    return test_accuracy

# create_model()
