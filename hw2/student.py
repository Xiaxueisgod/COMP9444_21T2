#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

a. choice of architecture, algorithms and enhancements (if any)
Answer:
a. The architecture I used is resnet18. First of all, I create a convolutional 
neural network, then use BatchNorm2d, and use activation function 'relu()'. 
Then, maxpooling it. I create two functions which are 'ResidualBlock_same()' and
'ResidualBlock_diff()' to implement the two different basic blocks that exist in
the resnet18 network architecture. 
'ResidualBlock_same()' implements a residual structure operation under the condition 
that the number of channels is unchanged which is f(x) + x. In my function 
'ResidualBlock_same()', f(x) means that x needs to go through two layers of network with 
BatchNorm2d and relu(). Then, the residual structure is calculated (plus x).
'ResidualBlock_diff()' implement a residual structure operation under a jump 
connection structure, the number of channels has changed, which is f(x) + g(x),
In my function 'ResidualBlock_diff()', F(x) is the same as the function 'ResidualBlock_same()'.
g(x) is that x goes through another network with BatchNorm2d and relu(), make the same number
of channels, and then carry out residual structure operation (f(x) + g(x)).
There are 8 residual operations in the whole neural network. Among them, the 1st, 
2nd, 4th, 6th, 8th times are residual structure operations under the condition that 
the number of channals is unchanged, 'ResidualBlock_same()' is used. The 3rd, 5th and 
7th times are residual structure operations under the condition that the number of 
channels changes, using the 'ResidualBlock_diff()' function.
Finally, use 'nn.AdaptiveAvgPool2d()' downsampling the x of 512*7*7 to 512*1*1 and add
a fully connected layer (nn.Linear(512,14)) and output the results.

b. choice of loss function and optimiser
Answer:
b. Loss function I used is CrossEntropyLoss() besides, I use Adam(Adaptive Moment 
Estimation) optimization algorithm the reason is that Adam has a small memory requirement
and is also suitable for large data sets and high-dimensional spaces and Adam is better at 
converging the loss.

c. choice of image transformations
Answer:
c. As for the image transformations, since I need to use the resnet18 network structure,
I resize the given image to 224 size and crop it in the middle of the image. In order to
increase the training data set, I also stipulated that the given image should be flipped 
at a probability level of 0.5, and Normalized an tensor image with mean 'mean=[0.485,0.456,0.406]'
and standard deviation 'std=[0.229,0.224,0.225]'. Finally, convert a image to Tensor(C*H*W) 
in the range [0,255] to a torch. Tensor(C*H*W) in the range [0.0,1.0].

d. tuning of metaparameters
Answer:
d. 
train_val_split = 0.8
batch_size = 128
epochs = 30
lr = 0.0001

e. use of validation set, and any other steps taken to improve generalization and avoid overfitting
Answer:
e. 
1.Data augmentation: Rotate the original image by a small angle (transforms.RandomHorizontalFlip(0.5)).
                      Crop a portion of the original image (transforms.CenterCrop(224)).
2.Adjust parameters, such as batchsize, epochs and learning rate.
3.Use batch normalization (BatchNorm2d()).
"""


############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.Compose([
            # Resize the input image to the same size
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            # Normalization processing --> transform to standard normal distribution to make the model easier to converge
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    elif mode == 'test':
        return transforms.Compose([
            # Resize the input image to the same size
            transforms.Resize(224),
            transforms.ToTensor(),
            # Normalization processing --> transform to standard normal distribution to make the model easier to converge
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# solid lines note jump connections part (F(x)+x)
class ResidualBlock_same(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock_same, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.layer1(x)
        out += x
        out = F.relu(out)
        return out


# dotted lines note jump connecting part (F(x)+G(x))
class ResidualBlock_diff(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResidualBlock_diff, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        out = self.left(x)
        out += self.right(x)
        out = F.relu(out)
        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bnm2d = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet1 = ResidualBlock_same(64, 64)
        self.resnet2 = ResidualBlock_same(64, 64)
        self.resnet3 = ResidualBlock_diff(64, 128)
        self.resnet4 = ResidualBlock_same(128, 128)
        self.resnet5 = ResidualBlock_diff(128, 256)
        self.resnet6 = ResidualBlock_same(256, 256)
        self.resnet7 = ResidualBlock_diff(256, 512)
        self.resnet8 = ResidualBlock_same(512, 512)
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(512, 14)

    def forward(self, t):
        t = self.conv1(t)
        t = self.bnm2d(t)
        t = self.relu(t)
        t = self.maxpooling(t)
        t = self.resnet1(t)
        t = self.resnet2(t)
        t = self.resnet3(t)
        t = self.resnet4(t)
        t = self.resnet5(t)
        t = self.resnet6(t)
        t = self.resnet7(t)
        t = self.resnet8(t)
        t = self.avgpooling(t)
        t = t.view(t.size(0), -1)
        t = self.linear1(t)

        return t


class loss(nn.Module):
    """
    Class for creating a custom loss function, if desired.
    If you instead specify a standard loss function,
    you can remove or comment out this class.
      """

    def __init__(self):
        super(loss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, output, target):
        loss = self.loss_func(output, target)
        return loss


net = Network()
lossFunc = loss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 128
epochs = 30
optimiser = optim.Adam(net.parameters(), lr=0.0001)