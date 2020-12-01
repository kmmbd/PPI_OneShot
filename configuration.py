import pathlib
from datetime import datetime


class Configuration():
    """ Initialize all defalut values """
    time = datetime.now().strftime('%Y-%m-%d %H_%M_%S')

    def __init__(self):
        # define all the file paths
        self.home_dir = pathlib.Path.home().joinpath('PycharmProjects', 'one_shot_ppi')
        self.embedding_path = pathlib.Path.joinpath(self.home_dir, 'dataset', 'embeddings', 'per_prot_emb.p')
        self.train_data = pathlib.Path.joinpath(self.home_dir, 'dataset', 'train', 'full_train_set_balanced.json')
        self.train_l = pathlib.Path.joinpath(self.home_dir, 'dataset', 'train', 'full_train_label_balanced.json')
        self.val_data = pathlib.Path.joinpath(self.home_dir, 'dataset', 'val', 'full_val_set.json')
        self.val_l = pathlib.Path.joinpath(self.home_dir, 'dataset', 'val', 'full_val_label.json')
        self.path_to_saved_model = pathlib.Path.joinpath(self.home_dir, 'trained_models', 'siameseNet_96_0.8_balanced_neg.pt')
        self.path_to_overfit_model = pathlib.Path.joinpath(self.home_dir, 'trained_models', 'overfit_96_0.8_balanced_neg.pt')

        # set learning parameters
        self.batch_size_train = 96
        self.batch_size_test = 16
        self.batch_size_validation = 96
        self.learning_rate = 0.0006
        self.number_of_epochs = 40
        self.positive_threshold = 0.65