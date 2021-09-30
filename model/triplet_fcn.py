import torch.nn.functional as F
import torch.nn as nn
import torch
from torch import abs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1 * 1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fcOut = nn.Linear(32, 1)
        # define layernorm
        self.ln1 = nn.LayerNorm(1024)
        #self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.ln4 = nn.LayerNorm(32)
        # self.sigmoid = nn.Sigmoid()
        #self.tanh = nn.Tanh()

    def forward_once(self, input):
        #print(input)
        #print(input.shape)
        x = self.ln1(input)
        x = F.relu(self.fc1(x))
        # x = self.ln2(x)
        x = self.dropout1(F.relu(self.fc2(x)))
        #x = F.relu(self.fc2(x))
        x = self.ln3(x)
        x = F.relu(self.fc3(x))
        x = self.ln4(x)
        # x = F.tanh(self.fcOut(x))
        x = self.fcOut(x)
        return x

    def forward(self, input1, input2, input3):
        # output1 = anchor, output2 = pos, output3 = neg

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3
