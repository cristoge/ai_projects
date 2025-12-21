import numpy as np


# Funciones de activación
def relu(z):
    return np.maximum(0, z)


def drelu(z):
    return (z > 0).astype(float)


# numpy reshape
class CapaOculta:
    def __init__(self, n_entradas, n_salidas, alpha=0.01):
        self.W = np.random.rand(n_entradas, n_salidas)
        self.b = np.random.rand(n_salidas)
        self.alpha = alpha

    def forward(self, X):
        self.X = X  # para luego el back
        self.Z = X @ self.W + self.b
        self.A = relu(self.Z)
        return self.A

    def backward(self, dA):
        # derivada de la función de activación
        dZ = dA * drelu(self.Z)
        # derivadas de los pesos y bias
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0)
        # gradiente para la capa anterior
        dX = dZ @ self.W.T
        # actualizar parámetros
        self.update()
        return dX

    # Actualización de pesos y bias
    def update(self):
        self.W -= self.alpha * self.dW
        self.b -= self.alpha * self.db
