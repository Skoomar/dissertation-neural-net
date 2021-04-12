# made by following https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0

from tensorflow.keras import models, layers


def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Dense(100, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))
    print(model.summary())
    return model


# create an early callback monitor for training accuracy - if accuracy doesn't improve for 2 epochs, then stop training
# TODO: don't think i really need this
# callbacks_list = [
#     callbacks.ModelCheckpoint(
#         filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
#     callbacks.EarlyStopping(monitor='acc', patience=1)
# ]

def train_model(model, train_x, train_y, batch_size=400, epochs=50, verbose=1):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # our Hyperparameters - reasons to use certain hyperparameters: https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9
    # BATCH_SIZE = 400
    BATCH_SIZE = 32
    EPOCHS = 100

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
