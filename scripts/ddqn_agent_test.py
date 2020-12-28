# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 11:47:22 2020

@author: William
"""

#%%
#imports

from drlAgents import DDQNAgent
import matplotlib.pyplot as plt
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
#Set a seed to compare results

seed = 42

torch.manual_seed(seed)
random.seed(seed)


#%%
#Create a pytorch neural network:

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        a = 20
        self.out_features = 2
        
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

#Set the environment's seed:
env.seed(seed)
env.action_space.seed(seed)

#Create and train the model:
agent = DDQNAgent(env, Net)   
agent.train(epochs=100)

#%%
plt.plot(agent.agent.logger.get('ep_reward'))

#%%
agent.play(length=100)