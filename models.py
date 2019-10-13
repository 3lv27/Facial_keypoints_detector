## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## 3. Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, 4) #inChannels, outChannels, kernelSize
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # MaxPool layer
        self.pool = nn.MaxPool2d(2, 2) #kernelSize, stride
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.1) #probability
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.6)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(43264, 1000) #@todo calculate input layer 
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        

        
    def forward(self, x):
        ## Feedforward behavior of NaimishNet model
        
        # Convolutiuonal layers
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout1(x)
        #print("First: ", x.shape)
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout2(x)
        #print("Second: ", x.shape)
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout3(x)
        #print("third: ", x.shape)
        x = self.pool(F.elu(self.conv4(x)))
        x = self.dropout4(x)
        #print("Fourth: ", x.shape)
        
        # Flatten layer
        x = x.view(x.size(0), -1)
        #print("Flattened: ", x.shape)
        
        # Fully Connected layers
        x = F.elu(self.fc1(x))
        x = self.dropout5(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x
