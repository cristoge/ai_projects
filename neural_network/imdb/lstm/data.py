import pandas as pd
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, random_split

from torch.nn.utils.rnn import pad_sequence


class ImdbDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=250):
        self.df = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # columnas correctas del dataset español
        self.textos = self.df["review_es"].tolist()
        self.etiquetas = (
            self.df["sentiment"].map({"negative": 0, "positive": 1}).tolist()
        )

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        text = self.textos[idx]
        label = self.etiquetas[idx]

        ids = self.tokenizer.EncodeAsIds(text)

        ids = ids[: self.max_length]

        return torch.tensor(ids), torch.tensor(label)


def collate_fn(batch):
    """
    Convertir a la misma longitud usando un padding
    """
    texts, labels = zip(*batch)

    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    return texts, labels


def get_data(file_path, tokenizer, batch_size=16, max_length=250, split=0.7):
    dataset = ImdbDataset(file_path, tokenizer, max_length)

    train_size = int(len(dataset) * split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader
