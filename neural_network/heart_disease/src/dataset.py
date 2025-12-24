from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


class HeartData(Dataset):
    def __init__(self, file_path):
        raw_data = pd.read_csv(file_path)
        x = raw_data.values[:, :-1]
        y = raw_data.values[:, -1].astype(int)
        min_max_scaler = MinMaxScaler()
        self.x = min_max_scaler.fit_transform(x)
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.long
        )


def get_data(batch_s):
    dataset = HeartData("heart.csv")
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_s, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_s, shuffle=True)
    return train_loader, test_loader
