import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import get_data  # tu función para cargar el dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", device)

MODEL_NAME = "google-bert/bert-base-uncased"
batch_size = 32
max_length = 250
epochs = 3
learning_rate = 2e-5
num_labels = 2

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader, test_loader = get_data(
    "./IMDB Dataset SPANISH.csv",
    tokenizer,
    batch_size=batch_size,
    max_length=max_length,
)


class BertClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


model = BertClassifier(MODEL_NAME, num_labels).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss train: {total_loss / len(dataloader):.4f}")


def test_loop(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct / total:.4f}")


for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model)
