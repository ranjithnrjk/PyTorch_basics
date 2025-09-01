import torch
from torch.utils.data import Dataset

class MyTrainDataset(Dataset):
    """
    Let's create each data point with 20 features
    """
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]