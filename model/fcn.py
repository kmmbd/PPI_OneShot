import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import abs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d(input_channels, output_channels, kernel_size)
        self.dropout1 = nn.Dropout(0.3)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1 * 1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fcOut = nn.Linear(32, 1)
        # define batchnorm
        #self.bn1 = nn.BatchNorm1d(num_features=256)
        # self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()

    def forward_once(self, input):
        #print(input)
        #print(input.shape)
        x = input
        # x = self.dropout1(F.relu(self.fc1(x)))
        # x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = self.dropout1(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        # x = self.fc3(x)
        # x = abs(x1 - x2)
        # x = F.tanh(self.fcOut(x))
        x = self.fcOut(x)
        return x

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2
