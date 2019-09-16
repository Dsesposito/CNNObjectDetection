import os
from os import listdir
from os.path import isfile, join
import keras
from keras import models, layers

class Classifier(object):

    def __init__(self, data_set):
        self.data_set = data_set
        self.amt_classes = data_set.amt_classes
        self.model = None
        self.history = None

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation=keras.activations.relu,
                                input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=keras.activations.relu))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation=keras.activations.relu))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation=keras.activations.relu))
        model.add(layers.Dense(256, activation=keras.activations.relu))
        model.add(layers.Dense(self.amt_classes, activation=keras.activations.softmax))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])

        self.model = model

    def train_model(self):
        history = self.model.fit(
            x=self.data_set.train.data,
            y=self.data_set.train.labels,
            validation_data=(self.data_set.val.data, self.data_set.val.labels),
            batch_size=64,
            epochs=50,
            verbose=1
        )
        self.history = history

    def save_model(self):
        path = join(os.getcwd(), 'saved_models')
        model_json = self.model.to_json()
        with open(os.path.join(path, "classifier.json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(path, "classifier.h5"))

    def load_model(self):
        path = join(os.getcwd(), 'saved_models')
        json_file = open(os.path.join(path, 'classifier.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(path, "classifier.h5"))
        self.model = loaded_model