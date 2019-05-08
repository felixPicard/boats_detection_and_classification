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

    model.add(Convolution2D(filters=128, kernel_size=[3, 3], input_shape=(150, 300, 3), padding='same', activation='relu'))

    model.add(Convolution2D(filters=64, kernel_size=[3, 3], activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(units = 128, activation= 'relu'))
    model.add(Dense(units=2, activation = 'softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save("../files/neural_networks/jooj")

    return model


# returns all paths to all images in a single file
def get_all_paths(pos_path, neg_path):

    pos_file = open("../files/data_samples/" + pos_path + ".txt", 'r')
    neg_file = open("../files/data_samples/" + neg_path + ".txt", 'r')

    paths = []

    for line in pos_file:
        paths.append("../files/data_samples/" + line[2:-1])

    for line in neg_file:
        paths.append("../files/data_samples/" + line[2:-1])

    return paths

def test_network(model, pos_path, neg_path, nb_of_samples):
    pos_file = open("../files/data_samples/" + pos_path + ".txt", 'r')
    neg_file = open("../files/data_samples/" + neg_path + ".txt", 'r')

    paths = []

    samples = np.zeros((2*nb_of_samples, 150, 300, 3))
    labels = np.zeros((2*nb_of_samples, 2))

    for i in range(nb_of_samples):
        im = cv.imread("../files/data_samples/" + pos_file.readline()[2:-1])
        samples[i] = cv.resize(im, (150, 75))
        labels[i] = [1, 0]

    for i in range(nb_of_samples):
        im = cv.imread("../files/data_samples/" + neg_file.readline()[2:-1])
        samples[i+nb_of_samples] = cv.resize(im, (150, 75))

    results = model.predict(samples)

    error = sum(abs(results[i][0] - labels[i][0]) for i in range(nb_of_samples*2))
    print(results)
    print("Error : ", float(error) / float(nb_of_samples*2) * 100, "%")


# user parameters
new_network = True
load_trained_network = False
learning_mode = True
batch_size = 20
steps_per_epoch = 2


#main code
if new_network:
    model = build_network()
elif load_trained_network:
    model = load_model("../files/neural_networks/jeej")
else:
    model = load_model("../files/neural_networks/jooj")


if learning_mode:
    print("Loading dataset...")
    paths = get_all_paths("positives", "negatives")
    paths = paths[:int(np.floor(len(paths)/(batch_size*steps_per_epoch))*(batch_size*steps_per_epoch) )]

    nb_of_epochs = int(np.ceil(len(paths)/(batch_size*steps_per_epoch)))
    print("number of epochs : ", nb_of_epochs)

    print("Using : ", len(paths), " samples.")
    imageMaker = ImageMaker(paths, batch_size=batch_size)

    model.fit_generator(imageMaker, steps_per_epoch=steps_per_epoch, epochs=nb_of_epochs, verbose=1, shuffle=True)

    model.save("../files/neural_networks/jeej")

    print("Network trained.")


else:
    test_network(model, "positives", "negatives", 1)