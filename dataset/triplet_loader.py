import torch
import torch.utils.data as data
import numpy as np


class Dataset(data.Dataset):
    # characterizes a dataset for PyTorch
    def __init__(self, list_IDs, triplets, embeddings):
        self.list_IDs = list_IDs
        self.triplets = triplets
        self.embeddings = embeddings

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # generates one sample of data
        # select sample
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ID = self.list_IDs[index]
        # get embedding of anchor protein
        x1 = self.triplets[ID][0]
        x1_t = torch.Tensor(self.embeddings[x1])
        x1_t = x1_t.to(device)
        # get embedding of positive protein
        x2 = self.triplets[ID][1]
        x2_t = torch.Tensor(self.embeddings[x2])
        x2_t = x2_t.to(device)
        # get embedding of negative protein
        x3 = self.triplets[ID][2]
        x3_t = torch.Tensor(self.embeddings[x3])
        x3_t = x3_t.to(device)
        # print(x1_t.is_cuda)

        return x1_t, x2_t, x3_t
