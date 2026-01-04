from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

dataset = datasets.ImageFolder(root="./archive/data/", transform=transform)

train_dataset = datasets.ImageFolder(root="./archive/data/train", transform=transform)
test_dataset  = datasets.ImageFolder(root="./archive/data/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.stack(x)


model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    correct = 0
    total = 0
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)  # acumula la loss

    avg_loss = total_loss / total  # loss promedio de toda la época
    accuracy = correct / total
    print(f"Train Accuracy epoch: {accuracy:.2f}, Loss: {avg_loss:.4f}")
    return avg_loss, accuracy


def test_loop(dataloader, model, loss_fn=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

            if loss_fn is not None:
                total_loss += loss_fn(pred, y).item() * y.size(0)

    accuracy = correct / total
    if loss_fn is not None:
        avg_loss = total_loss / total
        print(f"Test Accuracy: {accuracy:.2f}, Loss: {avg_loss:.4f}")
        return avg_loss, accuracy
    else:
        print(f"Test Accuracy: {accuracy:.2f}")
        return accuracy


epochs = 5

for epoch in range(epochs):
    train_loop(train_loader, model, loss_fn, optimizer)
    print(f"Epoch {epoch + 1}/{epochs} completada")
    test_loop(test_loader, model)
torch.save(model.state_dict(), "skin_model.pth")
