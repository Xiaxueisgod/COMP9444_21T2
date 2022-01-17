# rect.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Network(torch.nn.Module):
    def __init__(self, layer, hid):
        super(Network, self).__init__()
        self.layer = layer
        # INSERT CODE HERE
        self.hid = hid
        self.linear1 = nn.Linear(2, hid)
        self.linear2 = nn.Linear(hid, hid)
        self.linear3 = nn.Linear(hid,1)

    def forward(self, input):
         # CHANGE CODE HERE
        if self.layer == 1:
            self.hid1 = torch.tanh(self.linear1(input))
            output = torch.sigmoid(self.linear3(self.hid1))
        else:
            self.hid1 = torch.tanh(self.linear1(input))
            self.hid2 = torch.tanh(self.linear2(self.hid1))
            output = torch.sigmoid(self.linear3(self.hid2))
        return output

def graph_hidden(net, layer, node):
    # INSERT CODE HERE
    xrange = torch.arange(start=-8, end=8.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-8, end=8.1, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)


    with torch.no_grad():  # suppress updating of gradients
        net.eval()  # toggle batch norm, dropout
        output = net(grid)
        net.train()  # toggle batch norm, dropout back again
        if 1 == layer:
            pred = (net.hid1[:,node] >= 0.5).float()
        else:
            pred = (net.hid2[:,node] >= 0.5).float()
        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]),
                       cmap='Wistia', shading='auto')
