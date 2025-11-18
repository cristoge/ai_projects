import numpy as np
import idx2numpy

X_test = idx2numpy.convert_from_file("../t10k-images-idx3-ubyte")
Y_test = idx2numpy.convert_from_file("../t10k-labels-idx1-ubyte")
X_train = idx2numpy.convert_from_file("../train-images-idx3-ubyte")
Y_train = idx2numpy.convert_from_file("../train-labels-idx1-ubyte")


X_test = X_test / 255
X_train = X_train / 255

X_test_flat = X_test.reshape(X_test.shape[0], -1)


# hace falta aplanar la x para poder hacer la multiplicacion en el forward
def relu(x):
    """
    La funcion de Activacion
    """
    if x > 0:
        return x
    else:
        return 0


# filas 784 y el numero de neuronas que es la salida
def inicializar(entrada, salida):
    w = np.random.randn(salida, entrada)
    b = np.random.randn(1, salida)
    print(w, b)
    return w, b


def forward(x, w, b):
    return x @ w + b  #


w, b = inicializar(784, 100)  # el 100 es la capa oculta hidden layer
print(forward(X_test, w, b))
