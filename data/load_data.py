#%% Read MNIST Dataset / kütüphane kullanmadan dosyaları okuyup hazır hale getirmek amaçlı sınıf yapısı
import struct
import numpy as np #linear algebra
from array import array
from os.path import join #dosya yollarını birleştirmeyi sağlar

class Dataloader(object):
    def __init__(self, train_img_filepath, train_lables_filepath, test_img_filepath, test_labels_filepath):
        self.train_img_filepath = train_img_filepath
        self.train_labels_filepath = train_lables_filepath
        self.test_img_filepath = test_img_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_labels (self, images_filepath, labels_filepath):
        labels = []

        with open(labels_filepath, "rb") as file:
            magic,size = struct.unpack(">II", file.read(8))

            if magic != 2049:
                raise ValueError('Magic number mismatch. Expected 2049, got {}'.format(magic))
            labels = array("B", file.read()) # B burda 8-bit unsigned integer oluyor diye anladım

        with open(images_filepath, "rb") as file: #görüntü dosyamız için magic number 2051 miş
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Magic number mismatch. Expected 2051, got {}".format(magic))
            image_data = array("B", file.read())
        # image Magic Number: 2051 | label Magic Number: 2049

        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28,28) #satırları görsel haline getiriyor
            images[i][:] = img #görsel liste içine vektörleştirilmiş şekilde yükleniyor diye anladım
        
        return images, labels
    
    def load_data(self):
       x_train, y_train = self.read_labels(self.train_img_filepath, self.train_labels_filepath)
       x_test, y_test = self.read_labels(self.test_img_filepath, self.test_labels_filepath)
       return (x_train, y_train), (x_test, y_test)
