import numpy as np

from capa import CapaOculta


class Modelo:
    def __init__(self, w_shapes: list, b_shapes: list):
        self.capas = []
        for w_shape, b in zip(w_shapes, b_shapes):
            n_entradas, n_salidas = w_shape  # desempaquetamos la tupla
            self.capas.append(
                CapaOculta(n_entradas, n_salidas)
            )  # Forward por todas las capas

    def forward(self, X):
        salida = X
        for capa in self.capas:
            salida = capa.forward(salida)
        return salida

    # Backward propagando el gradiente de la pérdida
    def backward(self, grad_salida):
        grad = grad_salida
        # Recorremos las capas en orden inverso
        for capa in reversed(self.capas):
            grad = capa.backward(grad)

    # Pérdida
    def calcular_loss(self, y_pred, y_true):
        error = y_pred - y_true
        loss_val = np.sum(error**2)  # valor de la perdida
        grad_loss = 2 * error / y_true.shape[0]  # gradiente promedio
        return loss_val, grad_loss
