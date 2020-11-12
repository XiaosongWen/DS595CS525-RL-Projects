#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import empty
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    
    This is just a hint. You can build your own structure.
    """
    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # modified based on the official DQN guide
        # I heard that initializer matters. here is he initializer
        nn.init.kaiming_uniform_(empty(3, 5), a=0, mode='fan_in', nonlinearity='leaky_relu')

        # this structure is mentioned in this vanilla paper, quote as below:
        # The exact architecture, shown schematically in Fig. 1, is as follows.
        # The input tothe neural network consists of an 84x84x4 image produced by the preprocess-ing mapw.
        # The first hidden layer convolves 32 filters of 8x8 
        # with stride 4 with the input image and applies a rectifier nonlinearity[31,32].
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        # The second hidden layer con-volves 64 filters of 434 with stride 2, 
        # again followed by a rectifier nonlinearity.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        # This is followed by a third convolutional layer that convolves 64 filters of 3x3 
        # with stride 1 followed by a rectifier. 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        # The final hidden layer is fully-connected and con-sists of 512 rectifier units. 
        ## here it is (9-3+1)^2*64=7*7*64=3136
        self.fc4 = nn.Linear(3136, 512)
        # The output layer is a fully-connected linear layer with a single output for each valid action. 
        # The number of valid actions varied between 4 and 18 on the games we considered.
        self.fc5 = nn.Linear(512, 4)
    
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # modified based on the official DQN guide
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        x = self.fc5(x)
        ###########################
        return x
