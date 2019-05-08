import cv2 as cv
import numpy as np
from keras.utils import Sequence


class ImageMaker(Sequence):

    def __init__(self, paths, batch_size=0):
        self.paths = paths
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_paths = self.paths[idx * self.batch_size : (idx+1) * self.batch_size]

        out_images = np.zeros((self.batch_size, 75, 150, 3))
        out_labels = np.zeros((self.batch_size, 2))

        for i in range(self.batch_size):
            img_temp = cv.imread(batch_paths[i])
            out_images[i] = cv.resize(img_temp, (150, 75))
            if batch_paths[i][22] == 'p':
                out_labels[i] = np.array([1,0])
            else:
                out_labels[i] = np.array([0,1])

        return out_images, out_labels
