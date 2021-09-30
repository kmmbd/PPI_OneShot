"""
Optimize and Build Siamese network
"""

import torch
import torch.backends.cudnn
import torch.optim as optim
import torch.nn as nn
from losses.contrastiveLoss import ContrastiveLoss
import time


class Utilities():

    def __init__(self, device):
        # super(Utilities, self).__init__()
        self.device = device
        # optimize CUDA tasks using cuDNN
        self.cudnn()

    # define the loss and opitmizer functions
    # we use the CrossEntropyLoss and ADAM as Optimizer
    def cudnn(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def createLossAndOptimizer(self, net, learning_rate):
        # Loss function
        # loss = ContrastiveLoss()
        loss = ContrastiveLoss(margin=0.8)

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        return loss, optimizer

    def save_checkpoint(self, save_path, model, optimizer, val_loss):
        if save_path is None:
            return
        save_path = save_path
        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'val_loss': val_loss}

        torch.save(state_dict, save_path)

        print(f'Model saved to ==> {save_path}')

    def load_checkpoint(model, optimizer, model_path):
        load_path = model_path
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        val_loss = state_dict['val_loss']
        print(f'Model loaded from <== {load_path}')

        return val_loss

    # training and validation after every epoch
    def train(self, model, learning_rate, train_loader, val_loader, num_epochs, save_path, overfit_path):
        loss, optimizer = self.createLossAndOptimizer(model, learning_rate)
        best_val_loss = float("Inf")
        train_losses = []
        val_losses = []
        train_length = len(train_loader.dataset)
        print("Length of training set:", train_length)

        for epoch in range(num_epochs):
            i = 0
            j = 0
            running_loss = 0.0
            model.train()
            print("Starting epoch " + str(epoch + 1))
            for prot1, prot2, labels in train_loader:
                # Forward
                prot1 = prot1.to(self.device)
                prot2 = prot2.to(self.device)
                labels = labels.to(self.device)
                output1, output2 = model(prot1, prot2)
                # print("here3")
                current_loss = loss(output1, output2, labels)
                #print(current_loss)
                # Backward and optimize
                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                running_loss += current_loss.item()
                #print(running_loss)
                i+=1
                if (i%100==0):
                    j+=1
                    print((0.5*j),"% complete")

            avg_train_loss = running_loss / len(train_loader)
            # wandb.log({'epoch': epoch, 'train_loss': avg_train_loss})
            train_losses.append(avg_train_loss)

            val_running_loss = 0.0
            with torch.no_grad():
                model.eval()
                for prot1, prot2, labels in val_loader:
                    prot1 = prot1.to(self.device)
                    prot2 = prot2.to(self.device)
                    labels = labels.to(self.device)
                    output1, output2 = model(prot1, prot2)
                    val_loss = loss(output1, output2, labels)
                    val_running_loss += val_loss.item()
            avg_val_loss = val_running_loss / len(val_loader)
            # wandb.log({'epoch': epoch, 'validation_loss': loss})
            val_losses.append(avg_val_loss)
            self.save_checkpoint(overfit_path, model, optimizer, avg_val_loss)
            print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
                  .format(epoch + 1, num_epochs, avg_train_loss, avg_val_loss))
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint(save_path, model, optimizer, best_val_loss)

        print("Finished Training")
        return train_losses, val_losses
