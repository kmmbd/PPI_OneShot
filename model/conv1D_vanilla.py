# import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import abs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=9, padding=0)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=6, stride=1, padding=0)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=4, stride=4, padding=0)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 9, 1024)
        self.fcOut = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def convs(self, x):
        # out_dim = in_dim - kernel_size + 1
        # 1, 105, 105
        x = F.relu(self.bn1(self.conv1(x)))
        # 64, 96, 96
        # x = F.max_pool2d(x, (2, 2))
        # 64, 48, 48
        x = F.relu(self.bn2(self.conv2(x)))
        # 128, 42, 42
        # x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, (3, 3))

        return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)
        x1 = x1.view(-1, 256 * 9)
        x1 = self.sigmoid(self.fc1(x1))
        x2 = self.convs(x2)
        x2 = x2.view(-1, 256 * 9)
        x2 = self.sigmoid(self.fc1(x2))
        x = abs(x1 - x2)
        x = self.fcOut(x)
        return x
