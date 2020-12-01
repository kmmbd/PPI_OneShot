import torch
import torch.utils.data as data
import numpy as np


class Dataset(data.Dataset):
    # characterizes a dataset for PyTorch
    def __init__(self, list_IDs, pairs, labels, embeddings):
        self.list_IDs = list_IDs
        self.labels = labels
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # generates one sample of data
        # select sample
        ID = self.list_IDs[index]
        # data = np.load(r"H:\embeddings\embeddings.npz")
        x1 = self.pairs[ID][0]
        x1_t = torch.Tensor(self.embeddings[x1])
        x2 = self.pairs[ID][1]
        x2_t = torch.Tensor(self.embeddings[x2])
        # load data and get label
        y = self.labels[ID]

        return x1_t, x2_t, torch.from_numpy(np.array([y], dtype=np.float32))
