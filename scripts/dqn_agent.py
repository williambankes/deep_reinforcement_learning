# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:02:58 2020

Understand how certain hyperparameters affect the performance of the DQN

- The neural net used
    - batchNorm included and excluded
    
- Number of iterations
    - Does the DQN stabilise and then unstabilise
    
- Apply hyperparameter tuning to the algorithms hyperparameters



@author: William Bankes
"""

#%%
from drlAgents.policy import Policy
from drlAgents.agent import DQN_Agent
from drlAgents.logger import Logger

import gym
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F


#%%

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        a = 32
        
        self.in_layer = nn.Linear(4,a)
        self.h1 = nn.Linear(a,a)
        self.bn1 = nn.BatchNorm1d(a)
        self.h2 = nn.Linear(a,a)
        self.bn2 = nn.BatchNorm1d(a)
        self.h3 = nn.Linear(a,a)
        self.bn3 = nn.BatchNorm1d(a)
        self.out_layer = nn.Linear(a,2)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.bn1(self.h1(x)))
        x = F.relu(self.bn2(self.h2(x)))
        x = F.relu(self.bn3(self.h3(x)))
        return self.out_layer(x)
    
#%%
        
env = gym.make('CartPole-v0')
p = Policy(Net, discount=0.999)
d = DQN_Agent(env, p)
d.train(1000)

d.play()


