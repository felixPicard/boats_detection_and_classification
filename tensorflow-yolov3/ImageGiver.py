import cv2 as cv
import numpy as np


class ImageGiver():

    def __init__(self, directory_path):
        self.directory = open(directory_path)
        self.names = [line[:-1] for line in self.directory]
        self.length = len(self.names)
        self.index = 0


    def get_content(self):
        return self.names


    def get_length(self):
        return self.length

    def next(self):

        if self.index < self.length:
            self.index += 1

        else:
            self.index = 1

        return cv.imread(self.names[self.index - 1])