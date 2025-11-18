import numpy as np
import idx2numpy

X_test = idx2numpy.convert_from_file("../t10k-images-idx3-ubyte")
Y_test = idx2numpy.convert_from_file("../t10k-labels-idx1-ubyte")
X_train = idx2numpy.convert_from_file("../train-images-idx3-ubyte")
Y_train = idx2numpy.convert_from_file("../train-labels-idx1-ubyte")


X_test = X_test / 255
X_train = X_train / 255


def relu(x):
    if x > 0:
        return input
    else:
        return 0


# filas 784 y el batch size
def inicializar(entrada, salida):
    w = np.random.randn(salida, entrada)
    b = np.random.randn(1, salida)
    print(w, b)
    return w, b


w, b = inicializar(784, 10)  # el 100 es el batch size
