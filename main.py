# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from configuration import Configuration
import argparse
import torch
import time
from utilities.utils import Utilities
from utilities.sampler import Sampler
from model.fcn import Net


class ModelTrainer():
    def __init__(self):
        pass

    def count_parameters(self, model):
        temp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'The model architecture:\n\n', model)
        print(f'\nThe model has {temp:,} trainable parameters')

    def pipeline(self):
        print("pipeline class")
        # start a timer that keeps track of total time needed
        start_time = time.time()
        # load all the necessary parameters from configuration file
        config = Configuration()
        # define the default device
        # cuda:0 if a compatible CUDA GPU has been found, CPU otherwise
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        utils = Utilities(device)
        sampler = Sampler(batch_size_train=config.batch_size_train,
                          batch_size_val=config.batch_size_validation,
                          batch_size_test=config.batch_size_test)
        training_loader, validation_loader = sampler.get_datasets(config.train_data,
                                                                  config.train_l,
                                                                  config.val_data,
                                                                  config.val_l,
                                                                  config.embedding_path)

        net = Net().to(device)
        print(net)
        self.count_parameters(net)
        # start training the model
        utils.train(net, config.learning_rate, training_loader, validation_loader,
                    config.number_of_epochs, config.path_to_saved_model, config.path_to_overfit_model)

        end_time = time.time()
        print("Total time elapsed: " + str(end_time - start_time) + " seconds")
        # clear GPU Memory buffer after all tasks have been completed
        torch.cuda.empty_cache()


# create the main function and execute the pipeline function
def main():
    """ Main pipeline execution block """
    model_trainer = ModelTrainer()
    model_trainer.pipeline()


if __name__ == '__main__':
    main()
