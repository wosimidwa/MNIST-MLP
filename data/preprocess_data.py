# %% şimdi vektörleştirip normalleştirmemiz lazım
def preprocess_data(images, labels):
    x = np.array([np.array(img).reshape(-1)/255.0 for img in images]) #img.reshape(-1) vektörleştirmiş oluyoruz
    y = np.array(labels)
    return x, y

x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

print("Feature matrix (x_train, x_test):", x_train.shape, y_train.shape)
print("Feature matrix(y_train, y_test):", y_train.shape, y_test.shape)

