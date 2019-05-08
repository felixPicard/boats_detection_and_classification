from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from ImageMaker import ImageMaker
import cv2 as cv
import random
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_network():
    model = Sequential()

    model.add(Convolution2D(filters=64, kernel_size=[3, 3], input_shape=(75, 150, 3), padding='same', activation='relu'))

    model.add(Convolution2D(filters=64, kernel_size=[3, 3], activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(units = 128, activation= 'relu'))
    model.add(Dense(units=2, activation = 'softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save("../files/neural_networks/jooj")

    return model


def get_all_paths(pos_path, neg_path):

    pos_file = open("../files/data_samples/" + pos_path + ".txt", 'r')
    neg_file = open("../files/data_samples/" + neg_path + ".txt", 'r')

    paths = []

    for line in pos_file:
        paths.append("../files/data_samples/" + line[2:-1])

    for line in neg_file:
        paths.append("../files/data_samples/" + line[2:-1])

    return paths


def get_labels(nb_of_positives, nb_of_negatives):

    labels = [[1,0]] * nb_of_positives

    for i in range(nb_of_negatives):
        labels.append([0,1])

    return np.asarray(labels)


# user parameters
new_network = False

if new_network:
    model = build_network()
else:
    model = load_model("../files/neural_networks/jooj")

print("Loading dataset...")
paths = get_all_paths("positives", "negatives")
random.shuffle(paths)
imageMaker = ImageMaker(paths, batch_size=20)

nb_of_epochs = int(np.ceil(len(paths)/200))
print("number of epochs : ", nb_of_epochs)

model.fit_generator(imageMaker, steps_per_epoch=10, epochs=nb_of_epochs, verbose=1)

model.save("../files/neural_networks/jeej")

print("Network trained.")
