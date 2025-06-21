# main.py
from data.load_data import Dataloader
from data.preprocess_data import preprocess_data
from model.train_model import build_model, train_with_fit, test_model


(x_train, y_train), (x_test, y_test) = get_dataset()
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

def get_dataset():
    input_path = 'PATH/TO/YOUR/MNIST_ARCHIVE/'  
    train_images = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    train_labels = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    loader = Dataloader(train_images, train_labels, test_images, test_labels)
    return loader.load_data()

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_dataset()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    model = build_model()
    model = train_with_fit()
    test_model()
