import torch
import torch.nn as nn

from data import get_data
from tokenizer import sp as SentencePiece


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=384, hidden_dim=384, num_layers=1):
        super(SentimentLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.long()

        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)

        hidden = hidden[-1]

        logits = self.linear(hidden)

        return logits


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device).long()

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Train Loss: {total_loss / len(dataloader):.4f}")


def test_loop(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).long()

            pred = model(X)
            preds = pred.argmax(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Accuracy: {acc:.4f}")
    return acc


def train_model(file_path, tokenizer, vocab_size, device, epochs=10):

    train_loader, test_loader = get_data(file_path, tokenizer)

    model = SentimentLSTM(vocab_size).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loop(train_loader, model, loss_fn, optimizer, device)
        acc = test_loop(test_loader, model, device)

        if acc > best_acc:
            best_acc = acc
            print("Best Acc:", best_acc)

    print(f"\nMejor accuracy: {best_acc:.4f}")

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = train_model(
    file_path="./archive/IMDB Dataset SPANISH.csv",
    tokenizer=SentencePiece,
    vocab_size=25000,
    device=device,
    epochs=10,
)
# Mejor accuracy en espanol: 0.8299
# Mejor accuracy en ingles: 0.8559
