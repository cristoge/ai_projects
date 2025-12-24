import torch
from torch import nn
from dataset import get_data

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Usando:", device)

batch_size = 50
train_loader, test_loader = get_data(batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"Accuracy: {correct / total:.2f}")


epochs = 20
for epoch in range(epochs):
    train_loop(train_loader, model, loss_fn, optimizer)
    print(f"Epoch {epoch + 1}/{epochs} completada")
    test_loop(test_loader, model)
