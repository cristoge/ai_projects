import idx2numpy

X_test = idx2numpy.convert_from_file("../t10k-images-idx3-ubyte")
Y_test = idx2numpy.convert_from_file("../t10k-labels-idx1-ubyte")
X_train = idx2numpy.convert_from_file("../train-images-idx3-ubyte")
Y_train = idx2numpy.convert_from_file("../train-labels-idx1-ubyte")


def transform(x):
    return x / 255


X_test = transform(X_test)
Y_test = transform(Y_test)
X_train = transform(X_train)
Y_train = transform(Y_train)
