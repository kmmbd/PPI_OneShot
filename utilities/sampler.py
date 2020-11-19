import torch
import json
from dataset.loader import Dataset
import pickle


class Sampler():
    def __init__(self, batch_size_train, batch_size_val, batch_size_test):
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test

    # Training Data Loader
    # Takes in a dataset and a training sampler for loading
    # num_workers deals with system memory and threads
    # this is the default data loader
    def get_datasets(self, train_data, train_l, val_data, val_l, embedding_path):

        with open(train_data, 'r') as fp:
            train_pairs = json.load(fp)
        with open(train_l, 'r') as fp:
            train_labels = json.load(fp)
        with open(val_data, 'r') as fp:
            val_pairs = json.load(fp)
        with open(val_l, 'r') as fp:
            val_labels = json.load(fp)

        list_IDs_train = list(train_pairs.keys())
        list_IDs_val = list(val_pairs.keys())

        with open(embedding_path, 'rb') as fp:
            embeddings = pickle.load(fp)

        # Generators
        training_set = Dataset(list_IDs_train, train_pairs, train_labels, embeddings)
        train_loader = torch.utils.data.DataLoader(training_set, shuffle=True, num_workers=0)

        validation_set = Dataset(list_IDs_val, val_pairs, val_labels, embeddings)
        val_loader = torch.utils.data.DataLoader(validation_set, shuffle=True, num_workers=0)

        return train_loader, val_loader
