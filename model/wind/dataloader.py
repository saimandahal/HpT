import torch

from torch.utils.data import DataLoader, Dataset

# Train data
class TimeSeriesDataset(Dataset):

    def __init__(self, data, seq_length):

        self.data = data

        self.seq_length = seq_length

    def __len__(self):

        return len(self.data) - self.seq_length

    def __getitem__(self, index):

        x = self.data[index:index+self.seq_length]

        y = self.data[index+self.seq_length]


        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Test data
class TimeSeriesTestDataset(Dataset):

    def __init__(self, data, seq_length):

        self.data = data

        self.seq_length = seq_length

    def __len__(self):

        return len(self.data) - self.seq_length

    def __getitem__(self, index):

        x = self.data[index:index+self.seq_length]

        y = self.data[index+self.seq_length]


        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

