from torch.utils.data import Dataset
from load_data import get_dataset
import torch
import numpy as np


class PumpAndDumpDataset(Dataset):
    def __init__(self, window_size, train=True, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.window_size = window_size
        X, Y = get_dataset(window_size)
        X = X[:, :, 1:]
        X[:, :, 0] = np.vectorize(
            lambda x: 0 if x == "sell" else 1)(X[:, :, 0])
        # Xnorm = np.linalg.norm(X)
        X = X / np.linalg.norm(X)
        X = X.astype('float64')
        Y = Y.astype('int')

        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        n, _, _ = self.X.shape
        return n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X[idx]
        y = self.Y[idx]
        return x, y
