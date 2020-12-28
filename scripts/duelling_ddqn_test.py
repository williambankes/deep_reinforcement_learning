# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 22:05:05 2020

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

#%%
#Set a seed to compare results

seed = 42

torch.manual_seed(seed)
random.seed(seed)


#%%
#Create a pytorch neural network:

class DuellingNet(nn.Module):
    ##Taken from: https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
    ##by Chris Yoon

    def __init__(self):
        
        super(DuellingNet, self).__init__()
        
        self.out_features = 2
        a = 16
        
        self.feature_layers = nn.Sequential(
            nn.Linear(4, a),
            nn.BatchNorm1d(a),
            nn.ReLU(),
            nn.Linear(a, a),
            nn.BatchNorm1d(a),
            nn.ReLU()
            )
        
        self.value_function = nn.Sequential(
            nn.Linear(a,a),
            nn.ReLU(),
            nn.BatchNorm1d(a),
            nn.Linear(a,1)
            )
        
        self.advantage_function = nn.Sequential(
            nn.Linear(a,a),
            nn.ReLU(),
            nn.BatchNorm1d(a),
            nn.Linear(a, self.out_features)
            )
        
    def forward(self, x):
        
        features = self.feature_layers(x)
        val      = self.value_function(features)
        adv      = self.advantage_function(features)
        action_values = val + (adv - adv.mean())
        
        return action_values
            
#%%
#create the environment:

env = gym.make('CartPole-v0')

#Set the environment's seed:
env.seed(seed)
env.action_space.seed(seed)

#Create and train the model:
agent = DDQNAgent(env, DuellingNet)   
agent.train(epochs=200)

#%%
plt.plot(agent.agent.logger.get('ep_reward'))

#%%
agent.play(length=300)