import tensorflow as tf
from tensorflow.keras import models, layers

x=80
y=12
filter_size = 64
kernel_size = 12
num_classes = 18

input_1= layers.Input(shape=(x,y), name='input_1')
input_2= layers.Input(shape=(x,y), name='input_2')

c1 = layers.Conv1D(filter_size, kernel_size)(input_1)
p1 = layers.MaxPooling1D(pool_size=2)(c1)
f1 = layers.Flatten()(p1)
c2 = layers.Conv1D(filter_size, kernel_size)(input_2)
p2 = layers.MaxPooling1D(pool_size=2)(c2)
f2 = layers.Flatten()(p2)

x = layers.concatenate([f1, f2])
x = layers.Dense(num_classes, activation='sigmoid')(x)

model = models.Model(inputs=[input_1, input_2], outputs=[x])
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

batch_size = 32
epochs = 100
# TODO: try messing around with/researching different values for batch_size and epochs
history = model.fit(,
                    train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    # callbacks=callbacks_list,
                    # validation_data=(test_x, testy_y_hot),
                    verbose=verbose)
return model