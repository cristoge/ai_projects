import torch
from torch import nn

import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# Misma red neuronal que hemos creado en el jupyter
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
# Crear y cargar el modelo, en este caso los pesos del modelo
model = NeuralNetwork().to(device)
model.load_state_dict(
    torch.load("model_weights.pth", map_location=device, weights_only=True)
)
model.eval()

# pongamos que esto da el usuario
data = [34, 0, 1, 118, 210, 0, 1, 192, 0, 0.7, 2, 0, 2]

# convertimos los datos en un tensor para poder enviarlos al modelo
x = torch.tensor([data], dtype=torch.float32).to(device)

# no_grad() para quitar el calculo de gradientes porque estamos precidiendo
with torch.no_grad():
    output = model(x)
    prediction = output.argmax(1).item() # obtiene el index de la mayor probabilidad y lo convierte a numero

print("Datos:", data)
print("La prediccion es que es ", prediction)

if prediction == 0:
    print("Resultado: Clase 0")
else:
    print("Resultado: Clase 1")
