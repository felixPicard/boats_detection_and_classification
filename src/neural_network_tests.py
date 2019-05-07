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

    model.add(Convolution2D(16, 3, 3, input_shape=(150, 300, 3), activation='relu'))

    model.add(Convolution2D(64, (3, 3), activation='relu'))

    model.add(Flatten())

    #model.add(Dense(output_dim = 128, activation= 'relu'))
    model.add(Dense(output_dim= 1, activation = 'softmax'))

    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    model.save("../files/neural_networks/jooj")


def get_images(value):

    file = open("../files/data_samples/" + value + ".txt", 'r')

    paths = []
    for line in file:
        paths.append(line)

    out_images = np.zeros((len(paths), 150, 300, 3))
    for i in range(len(paths)):
        out_images[i] = cv.imread(paths[i])

    return out_images


build_network()

print("\nLoading database...\n")
print("positive images...")
samples = get_images("positives")
print("positive images done !")
nb_of_positives = len(samples)
print("negatives images...")
samples = np.concatenate([samples, get_images("negatives")])
print("negative images done !")

print("\n")

nb_of_samples = len(samples)
nb_of_negatives = nb_of_samples - nb_of_positives

print(nb_of_positives, "positives, ", nb_of_negatives, "negatives, ", nb_of_samples, "total")

labels = [1] * nb_of_positives
for i in range(nb_of_negatives):
    labels.append(0)

model = load_model("../files/neural_networks/jooj")

positives_to_train = int(nb_of_positives * 0.9)
negatives_to_train = int(nb_of_negatives * 0.9)

# TRAINING
print("Network will be trained on ", positives_to_train + negatives_to_train, "samples.")

training_samples = np.concatenate([samples[:positives_to_train], samples[nb_of_positives:nb_of_positives+negatives_to_train]])
training_labels = np.concatenate([labels[:positives_to_train], labels[nb_of_positives:nb_of_positives+negatives_to_train]])

testing_samples = np.concatenate([samples[positives_to_train:nb_of_positives], samples[nb_of_positives+negatives_to_train:]])
testing_labels = np.concatenate([labels[positives_to_train:nb_of_positives], labels[nb_of_positives+negatives_to_train:]])

model.fit(training_samples,training_labels)

print(training_labels[0])
print("Network trained.")

answer = model.predict(testing_samples)

print(labels)
#print(answer)
#print("we wnated")
#print(testing_labels)
"""
error = sum( abs(max(answer[i])-testing_labels[i]) for i in range(len(answer)) )


print("Erreur de : ", float(error)/float(nb_of_samples-(positives_to_train + negatives_to_train))*100, "%")
"""
