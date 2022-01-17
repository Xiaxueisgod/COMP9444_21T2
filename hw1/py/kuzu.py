# kuzu.py


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        output = F.log_softmax(self.linear1(x))
        return output # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28, 1000)
        self.linear2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.flatten(x)
        hide_node = torch.tanh(self.linear1(x))
        output = F.log_softmax(self.linear2(hide_node))
        return output # CHANGE CODE HERE



class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=100,kernel_size=(5,5),padding=2,stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=128, kernel_size=(5,5), padding=2, stride=(2,2))
        self.pooling = nn.MaxPool2d(kernel_size=5,padding=2,stride=2)
        # conv1:1+(28+2*2-5)/2=14, conv2:1+(28+2*2-5)/2=7, pooling:1+(28+2*2-5)/2=4
        # linear1:4*4*128(out_channels)=2048
        self.linear1 = nn.Linear(2048, 100)
        self.linear2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        convo1 = self.conv1(x)
        convo2 = self.conv2(convo1)
        pooling1 = self.pooling(convo2)
        re = self.relu(pooling1)
        x = self.flatten(re)
        hide_node = self.relu(self.linear1(x))
        output = F.log_softmax(self.linear2(hide_node))
        return output # CHANGE CODE HERE


