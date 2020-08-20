# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:02:58 2020

@author: William Bankes
"""

#%%
from drlAgents import *

import gym

import torch
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
d = DQN_Agent(env, p, 10_000, 20_000)
d.train(112)

d.play()