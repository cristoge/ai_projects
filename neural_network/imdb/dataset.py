import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

"https://www.kaggle.com/datasets/luisdiegofv97/imdb-dataset-of-50k-movie-reviews-spanish"


class ImdbDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=250):
        self.df = pd.read_csv("./archive/IMDB Dataset SPANISH.csv")
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.textos = self.df["review_es"].tolist()
        self.etiquetas = (
            self.df["sentimiento"].map({"negativo": 0, "positivo": 1}).tolist()
        )

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = self.textos[idx]
        tokens = self.tokenizer.EncodeAsIds(texto)
        tokens = [self.tokenizer.bos_id()] + tokens + [self.tokenizer.eos_id()]

        # Padding o truncado
        if len(tokens) < self.max_length:
            tokens += [self.tokenizer.pad_id()] * (self.max_length - len(tokens))
        else:
            tokens = tokens[: self.max_length]

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(self.etiquetas[idx], dtype=torch.long),
        }


def get_data(file_path, tokenizer, batch_size=16, max_length=250, split=0.7):
    dataset = ImdbDataset(file_path, tokenizer, max_length)

    train_size = int(len(dataset) * split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
