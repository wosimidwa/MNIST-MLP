# data/dataloader.py
import struct
import numpy as np
from array import array
from os.path import join

class Dataloader(object):
    def __init__(self, train_img_filepath, train_labels_filepath, test_img_filepath, test_labels_filepath):
        self.train_img_filepath = train_img_filepath
        self.train_labels_filepath = train_labels_filepath
        self.test_img_filepath = test_img_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, "rb") as f:
            magic, size = struct.unpack(">II", f.read(8))
            assert magic == 2049, f"Expected magic 2049, got {magic}"
            labels = array("B", f.read())

        with open(images_filepath, "rb") as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            assert magic == 2051, f"Expected magic 2051, got {magic}"
            image_data = array("B", f.read())

        images = np.zeros((size, rows, cols), dtype=np.uint8)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            images[i] = img.reshape(rows, cols)

        return images, np.array(labels)

    def load_data(self):
        x_train, y_train = self.read_labels(self.train_img_filepath, self.train_labels_filepath)
        x_test, y_test = self.read_labels(self.test_img_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
