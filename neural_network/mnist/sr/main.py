import numpy as np
import idx2numpy
from model import Modelo

X_train = idx2numpy.convert_from_file("../train-images-idx3-ubyte") / 255
Y_train = idx2numpy.convert_from_file("../train-labels-idx1-ubyte")
X_test = idx2numpy.convert_from_file("../t10k-images-idx3-ubyte") / 255
Y_test = idx2numpy.convert_from_file("../t10k-labels-idx1-ubyte")

# Aplanar imágenes
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Capas
w_shapes = [
    (784, 256),
    (256, 128),
    (128, 64),
    (64, 32),
    (32, 16),
    (16, 10),
]

# Bias para cada capa
b_shapes = [256, 128, 64, 32, 16, 10]

# creamos el modelo
modelo = Modelo(w_shapes, b_shapes)

# definims el numero de epocas y el tamano del lote
num_epochs = 10
batch_size = 500

# Convertir etiquetas a onehot es decir solo la posicion correcta es 1 lo demas 0
Y_train_onehot = np.eye(10)[Y_train]  # vector de 10 por cada numero

for epoca in range(num_epochs):
    total_loss = 0
    num_batches = 0

    for i in range(0, X_train_flat.shape[0], batch_size):
        x_batch = X_train_flat[i : i + batch_size]
        y_batch = Y_train_onehot[i : i + batch_size]

        # Forward
        y_pred = modelo.forward(x_batch)

        # Calcular perdida y gradiente
        error = y_pred - y_batch
        loss_val = np.sum(error**2)
        grad_loss = 2 * error / y_batch.shape[0]

        # Backward
        modelo.backward(grad_loss)

        total_loss += loss_val
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Epoca {epoca + 1}/{num_epochs} — Loss promedio: {avg_loss:.4f}")

# forward con el lote del test
y_pred_test = modelo.forward(X_test_flat)

# la prediccion
y_pred_labels = np.argmax(y_pred_test, axis=1)

# comparacion y calcula el promedio
accuracy = np.mean(y_pred_labels == Y_test)
print(f"Precisión en test: {accuracy * 100:.2f}%")
