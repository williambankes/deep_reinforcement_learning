# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:20:57 2020

@author: William
"""

#%%
#imports

from drlAgents.dqnAgent import DQNAgent
import matplotlib.pyplot as plt
import gym
import torch.nn as nn
import torch.nn.functional as F

#%%
#Create a pytorch neural network:

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        a = 20
        
        self.in_layer = nn.Linear(4,a)
        self.h1 = nn.Linear(a,a)
        self.bn1 = nn.BatchNorm1d(a)
        self.h2 = nn.Linear(a,a)
        self.bn2 = nn.BatchNorm1d(a)
        self.out_layer = nn.Linear(a,2)
    
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.bn1(self.h1(x)))
        x = F.relu(self.bn2(self.h2(x)))
        return self.out_layer(x)
    
#%%
#create the environment:

env = gym.make('CartPole-v0')
agent = DQNAgent(env, Net)   
agent.train(epochs=300)

#%%
print(agent.agent.logger.get_names())
plt.plot(agent.agent.logger.get('ep_reward'))

#%%
agent.play(length=100)


    