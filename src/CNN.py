from keras.applications import
from keras.applications.vgg16 import  preprocess_input, decode_predictions
import cv2 as cv


model = VGG16()
image = cv.imread("../files/images/jeej.jpg")
image = cv.resize(image, (224, 224))



image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

prediction = model.predict(image)

label = decode_predictions(prediction)[0][0]

print(label[1], " : ", label[2]*100, "%")
