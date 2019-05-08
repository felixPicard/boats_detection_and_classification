from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2 as cv
import random
import numpy as np


def build_network():
    model = Sequential()

    model.add(Convolution2D(filters=64, kernel_size=[3, 3], input_shape=(75, 150, 3), padding='same', activation='relu'))

    model.add(Convolution2D(filters=64, kernel_size=[3, 3], activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(units = 128, activation= 'relu'))
    model.add(Dense(units=1, activation = 'softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.save("../files/neural_networks/jooj")

    return model


def get_images(nb_of_pos, pos_path, nb_of_neg, neg_path):

    pos_file = open("../files/data_samples/" + pos_path + ".txt", 'r')
    neg_file = open("../files/data_samples/" + neg_path + ".txt", 'r')

    pos_maximum = 0
    neg_maximum = 0

    pos_paths = []
    neg_paths = []

    for line in pos_file:
        pos_maximum += 1
        pos_paths.append("../files/data_samples/" + line[2:-1])
        if pos_maximum > nb_of_pos:
            break

    for line in neg_file:
        neg_maximum += 1
        neg_paths.append("../files/data_samples/" + line[2:-1])
        if neg_maximum > nb_of_neg:
            break

    if nb_of_pos > pos_maximum or nb_of_neg > neg_maximum:
        exit(-1)

    nb_of_images_to_read = nb_of_neg + nb_of_pos

    out_images = np.zeros((nb_of_images_to_read, 75, 150, 3))

    for i in range(nb_of_pos):
        img_temp = cv.imread(pos_paths[i])
        out_images[i] = cv.resize(img_temp, (150, 75))

    for i in range(nb_of_neg):
        img_temp = cv.imread(neg_paths[i])
        out_images[i+nb_of_pos] = cv.resize(img_temp, (150, 75))


    return out_images


def get_labels(nb_of_positives, nb_of_negatives):

    labels = [1] * nb_of_positives

    for i in range(nb_of_negatives):
        labels.append(0)

    return labels


model = build_network()

nb_of_positives = 200
nb_of_negatives = 200
nb_of_samples = nb_of_negatives + nb_of_positives

print("Loading dataset...")

samples = get_images(200, "positives", 200, "negatives")

print(nb_of_positives, "positives, ", nb_of_negatives, "negatives, ", nb_of_samples, "total")

labels = get_labels(nb_of_positives, nb_of_negatives)

#model = load_model("../files/neural_networks/jooj")

model.fit(samples,labels, shuffle=True)

print("Network trained.")
