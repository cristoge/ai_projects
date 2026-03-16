import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import get_data  # tu función para cargar el dataset

# 🔹 Configuración de dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", device)

# 🔹 Modelo y tokenizer
MODEL_NAME = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

# 🔹 Configuración rápida para entrenar menos tiempo
batch_size = 32  # más ejemplos por batch
max_length = 128  # menos tokens por review
epochs = 1  # solo 1 pasada sobre todo el dataset
learning_rate = 1e-3  # lr más alto para acelerar el aprendizaje

# 🔹 DataLoaders
train_loader, test_loader = get_data(
    "./IMDB Dataset SPANISH.csv",
    tokenizer,
    batch_size=batch_size,
    max_length=max_length,
)

# 🔹 Loss y optimizador
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 🔹 Función de entrenamiento
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss train: {total_loss / len(dataloader):.4f}")


# 🔹 Función de evaluación
def test_loop(dataloader, model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = outputs.logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct / total:.4f}")


# 🔹 Loop principal de entrenamiento
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model)
