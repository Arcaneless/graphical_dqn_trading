import tensorflow as tf
from tensorflow.keras import layers, models

"""
Input: A grayscale image
Output: binary
"""
class GDQNModel:
    def __init__(self, height, width):
        model = models.Sequential()
        model.add(layers.Conv2D(120, (32, 32), activation='relu', input_shape=(height, width, 1)))
        model.add(layers.MaxPooling2D((32, 32)))
        model.add(layers.Conv2D(16, (4, 4), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(2, activation='linear'))
        self.model = model

    def compile_n_run(self, Xtrain, Xtest, Ytrain, Ytest, epochs):
        self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
        self.model.fit(Xtrain, Ytrain, epochs=epochs, validation_data=(Xtest, Ytest), verbose=2)

    def evaluate(self):
        pass